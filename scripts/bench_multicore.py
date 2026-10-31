#!/usr/bin/env python
"""GEMM and DF-MP2 pair energy split across the logical NeuronCores of one chip (#24).

Every mode prints ``RESULT {json}`` lines; times are warm medians in ms of
``--reps`` runs after one warm-up, synchronised on every core used. Inputs are
built on the host (device ``randn`` is shape-sensitive) and copied over.

GEMM modes (``gemm --shape M,K,N``), C = A @ B in fp32:

    cpu       ``torch.matmul`` on host tensors, ``--threads`` intra-op threads.
    cpupar    rows of A split over ``--threads`` Python threads, one intra-op
              thread each (the same threads as `cpu`, all of them busy).
    dev       one core, A and B resident on it: ``trnblas.gemm`` (aten) and,
              with ``--nki``, the NKI kernel (``TRNBLAS_NATIVE_GEMM=nki``).
    split     ``--cores k`` cores, ``--split row`` (A rows split, B replicated)
              or ``col`` (B columns split, A replicated), one launch per core
              from one thread (launches are asynchronous). Three timings:
                resident  the shards are already on their cores (compute only)
                host      host A, B -> shards -> compute -> one host C
                staged    A, B resident on neuron:0, shards staged through the
                          host to the other cores, C gathered back on neuron:0
                          (what a transparent multi-core `gemm` would do)
    threads   as `split` resident, one host thread per core.
    proc      as `split` resident and host, one process per core
              (``NEURON_RT_VISIBLE_CORES=i``); a rep's time is from the first
              start to the last finish across processes. Run it on its own
              (``--modes proc``): the workers cannot open their cores once
              the parent process has initialised the runtime.

Transfer probe (``bw --mb N``): host -> core and core -> host copies on
1, 2 and 4 cores at once, from one thread.

Pair energy (``energy --mp2 nocc,nvir,naux``): `nki_batched_pair_energy` on B
resident on neuron:0, with `trnblas.set_cores(k)`; ``--route sym|aten``.
``energy-proc`` runs the same row split with one process per core from a host
B, to separate the single-thread launch cost from the device time.

Example (whole chip, exclusive lock):
    NEURON_RT_VISIBLE_CORES=0-3 taskset -c 4,5 python scripts/bench_multicore.py \\
        gemm --shape 4096,4096,4096 --modes dev,split,threads,proc --cores 1,2,4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time


def _emit(**rec) -> None:
    print("RESULT " + json.dumps(rec), flush=True)


def _sync(idx) -> None:
    import torch_neuronx

    for k in idx:
        torch_neuronx.synchronize(k)


def _median_ms(fn, reps: int, sync) -> float:
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


def _host_ab(M: int, K: int, N: int, seed: int = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, K, generator=g), torch.randn(K, N, generator=g)


def _ranges(n: int, parts: int) -> list[tuple[int, int]]:
    from trnblas.nki.multicore import occ_chunks

    return occ_chunks(n, parts)


def _shards(A, B, split: str, parts: int):
    """[(a_part, b_part, (r0, r1), (c0, c1))] host views for each core."""
    M, N = A.shape[0], B.shape[1]
    out = []
    for s, e in _ranges(M if split == "row" else N, parts):
        if split == "row":
            out.append((A[s:e], B, (s, e), (0, N)))
        else:
            out.append((A, B[:, s:e], (0, M), (s, e)))
    return out


# ── GEMM ────────────────────────────────────────────────────────────────────


def gemm_cpu(args, M, K, N) -> None:
    import torch

    A, B = _host_ab(M, K, N)
    torch.set_num_threads(args.threads)
    ms = _median_ms(lambda: torch.matmul(A, B), args.reps, lambda: None)
    _emit(op="gemm", mode="cpu", threads=args.threads, M=M, K=K, N=N, ms=round(ms, 3))

    from concurrent.futures import ThreadPoolExecutor

    torch.set_num_threads(1)
    pool = ThreadPoolExecutor(args.threads)
    C = torch.empty(M, N)
    parts = _ranges(M, args.threads)

    def one(se):
        s, e = se
        torch.set_num_threads(1)
        torch.matmul(A[s:e], B, out=C[s:e])

    ms = _median_ms(lambda: list(pool.map(one, parts)), args.reps, lambda: None)
    _emit(op="gemm", mode="cpupar", threads=args.threads, M=M, K=K, N=N, ms=round(ms, 3))


def gemm_dev(args, M, K, N) -> None:
    import torch

    import trnblas

    trnblas.set_backend("nki")
    A, B = _host_ab(M, K, N)
    Ad, Bd = A.to("neuron:0"), B.to("neuron:0")
    _sync([0])
    ref = None
    for kernel in ["aten"] + (["nki"] if args.nki else []):
        os.environ["TRNBLAS_NATIVE_GEMM"] = "auto" if kernel == "aten" else "nki"
        ms = _median_ms(lambda: trnblas.gemm(1.0, Ad, Bd), args.reps, lambda: _sync([0]))
        out = trnblas.gemm(1.0, Ad, Bd).cpu()
        ref = A.double() @ B.double() if ref is None else ref
        err = float((out.double() - ref).abs().max() / ref.abs().max())
        _emit(
            op="gemm",
            mode="dev",
            kernel=kernel,
            cores=1,
            M=M,
            K=K,
            N=N,
            ms=round(ms, 3),
            rel_err=err,
        )
    os.environ["TRNBLAS_NATIVE_GEMM"] = "auto"
    # Host input, one core: copy over, aten, copy back.
    ms = _median_ms(
        lambda: (A.to("neuron:0") @ B.to("neuron:0")).cpu(), args.reps, lambda: _sync([0])
    )
    _emit(op="gemm", mode="host", split="none", cores=1, M=M, K=K, N=N, ms=round(ms, 3))


def gemm_split(args, M, K, N, k: int, split: str) -> None:
    import torch

    A, B = _host_ab(M, K, N)
    idx = list(range(k))
    devs = [torch.device("neuron", i) for i in idx]
    sh = _shards(A, B, split, k)
    sync = lambda: _sync(idx)  # noqa: E731
    base = dict(op="gemm", split=split, cores=k, M=M, K=K, N=N)

    # resident: shards already on their cores
    res = [
        (a.contiguous().to(d), b.contiguous().to(d))
        for (a, b, _, _), d in zip(sh, devs, strict=True)
    ]
    sync()
    ms = _median_ms(lambda: [a @ b for a, b in res], args.reps, sync)
    C = torch.empty(M, N)
    for (a, b), (_, _, (r0, r1), (c0, c1)) in zip(res, sh, strict=True):
        C[r0:r1, c0:c1] = (a @ b).cpu()
    ref = A.double() @ B.double()
    err = float((C.double() - ref).abs().max() / ref.abs().max())
    _emit(mode="resident", ms=round(ms, 3), rel_err=err, **base)

    if args.threads_mode:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(k)
        ms = _median_ms(lambda: list(pool.map(lambda ab: ab[0] @ ab[1], res)), args.reps, sync)
        _emit(mode="threads", ms=round(ms, 3), **base)
    res = None  # free the shards before the next case

    # host: host A, B -> shards -> compute -> host C
    def host_run():
        outs = []
        for (a, b, rr, cc), d in zip(sh, devs, strict=True):
            outs.append(((a.contiguous().to(d) @ b.contiguous().to(d)), rr, cc))
        for c, (r0, r1), (c0, c1) in outs:
            C[r0:r1, c0:c1] = c.cpu()

    ms = _median_ms(host_run, args.reps, sync)
    _emit(mode="host", ms=round(ms, 3), **base)

    # staged: A, B on neuron:0, other shards via the host, C back on neuron:0
    A0, B0 = A.to(devs[0]), B.to(devs[0])
    sync()

    def staged_run():
        a_sh = _shards(A0, B0, split, k)
        hosted = [None] + [(a.cpu(), b.cpu()) for a, b, _, _ in a_sh[1:]]
        outs = []
        for i, ((a, b, rr, cc), d) in enumerate(zip(a_sh, devs, strict=True)):
            if i == 0:
                outs.append((a @ b, rr, cc))
            else:
                ha, hb = hosted[i]
                outs.append((ha.to(d) @ hb.to(d), rr, cc))
        C0 = torch.empty(M, N, device=devs[0]) if k > 1 else None
        if k == 1:
            return outs[0][0]
        for i, (c, (r0, r1), (c0, c1)) in enumerate(outs):
            part = c if i == 0 else c.cpu().to(devs[0])
            C0[r0:r1, c0:c1] = part
        return C0

    ms = _median_ms(staged_run, args.reps, sync)
    _emit(mode="staged", ms=round(ms, 3), **base)


def _proc_gemm_worker(core, M, K, N, split, k, reps, barrier, q):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    import torch

    A, B = _host_ab(M, K, N)
    a, b, _, _ = _shards(A, B, split, k)[core]
    a, b = a.contiguous(), b.contiguous()
    ad, bd = a.to("neuron:0"), b.to("neuron:0")
    _sync([0])
    out = {}
    for label, fn in (
        ("resident", lambda: ad @ bd),
        ("host", lambda: (a.to("neuron:0") @ b.to("neuron:0")).cpu()),
    ):
        fn()
        _sync([0])
        ts = []
        for _ in range(reps):
            barrier.wait()
            t0 = time.time()
            fn()
            _sync([0])
            ts.append((t0, time.time()))
        out[label] = ts
    q.put((core, out))


def gemm_proc(args, M, K, N, k: int, split: str) -> None:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(k), ctx.Queue()
    ps = [
        ctx.Process(target=_proc_gemm_worker, args=(i, M, K, N, split, k, args.reps, barrier, q))
        for i in range(k)
    ]
    for p in ps:
        p.start()
    got = dict(q.get(timeout=600) for _ in ps)
    for p in ps:
        p.join()
    for label in ("resident", "host"):
        walls = [
            max(got[c][label][r][1] for c in got) - min(got[c][label][r][0] for c in got)
            for r in range(args.reps)
        ]
        _emit(
            op="gemm",
            mode=f"proc-{label}",
            split=split,
            cores=k,
            M=M,
            K=K,
            N=N,
            ms=round(1e3 * statistics.median(walls), 3),
        )


def run_gemm(args) -> None:
    M, K, N = (int(x) for x in args.shape.split(","))
    modes = args.modes.split(",")
    cores = [int(c) for c in args.cores.split(",")]
    splits = args.split.split(",")
    if "cpu" in modes:
        gemm_cpu(args, M, K, N)
    if "dev" in modes:
        gemm_dev(args, M, K, N)
    for split in splits:
        for k in cores:
            if "split" in modes:
                gemm_split(args, M, K, N, k, split)
            if "proc" in modes:
                gemm_proc(args, M, K, N, k, split)


# ── transfers ───────────────────────────────────────────────────────────────


def run_bw(args) -> None:
    import torch

    n = args.mb * (1 << 20) // 4
    x = torch.randn(n)
    for k in (1, 2, 4):
        idx = list(range(k))
        ms = _median_ms(
            lambda idx=idx: [x.to(f"neuron:{i}") for i in idx],
            args.reps,
            lambda idx=idx: _sync(idx),
        )
        _emit(
            op="bw",
            dir="h2d",
            mb=args.mb,
            cores=k,
            ms=round(ms, 3),
            gbps_total=round(k * args.mb / 1024 / (ms / 1e3), 2),
        )
        ds = [x.to(f"neuron:{i}") for i in idx]
        _sync(idx)
        ms = _median_ms(lambda ds=ds: [d.cpu() for d in ds], args.reps, lambda: None)
        _emit(
            op="bw",
            dir="d2h",
            mb=args.mb,
            cores=k,
            ms=round(ms, 3),
            gbps_total=round(k * args.mb / 1024 / (ms / 1e3), 2),
        )
        ds = None


# ── pair energy ─────────────────────────────────────────────────────────────


def _mp2_inputs(nocc: int, nvir: int, naux: int, seed: int = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    B = torch.randn(nocc, nvir, naux, generator=g) * 0.05
    eo = -torch.sort(torch.rand(nocc, generator=g))[0] - 0.5
    ev = torch.sort(torch.rand(nvir, generator=g))[0] + 0.1
    return B, eo, ev


def run_energy(args) -> None:
    import torch

    import trnblas
    from trnblas.nki import nki_batched_pair_energy

    nocc, nvir, naux = (int(x) for x in args.mp2.split(","))
    os.environ["TRNBLAS_PAIR_ENERGY"] = args.route
    B, eo, ev = _mp2_inputs(nocc, nvir, naux)
    core_list = [int(c) for c in args.cores.split(",") if c]
    if core_list:
        trnblas.set_backend("nki")
        Bd, eod, evd = B.to("neuron:0"), eo.to("neuron:0"), ev.to("neuron:0")
        _sync([0])
    e1 = None
    for k in core_list:
        trnblas.set_cores(k)
        idx = list(range(k))
        # The first call per core count compiles (sequentially, one thread).
        t = time.perf_counter()
        e = float(nki_batched_pair_energy(Bd, eod, evd))
        first = time.perf_counter() - t
        ms = _median_ms(
            lambda: float(nki_batched_pair_energy(Bd, eod, evd)),
            args.reps,
            lambda idx=idx: _sync(idx),
        )
        e1 = e if k == 1 else e1
        _emit(
            op="energy",
            route=args.route,
            cores=k,
            nocc=nocc,
            nvir=nvir,
            naux=naux,
            ms=round(ms, 3),
            first_s=round(first, 2),
            e=e,
            rel_vs_1core=None if e1 is None else abs(e - e1) / abs(e1),
        )
    trnblas.set_cores(None)
    if args.cpu:
        torch.set_num_threads(args.threads)
        from trnblas.nki.dispatch import _torch_chunked_pair_energy

        ms = _median_ms(
            lambda: float(_torch_chunked_pair_energy(B, eo, ev)),
            max(1, args.reps // 2),
            lambda: None,
        )
        _emit(
            op="energy",
            route="cpu",
            threads=args.threads,
            nocc=nocc,
            nvir=nvir,
            naux=naux,
            ms=round(ms, 3),
        )
        # Fair CPU split: occupied rows over the same threads, one intra-op thread each.
        from concurrent.futures import ThreadPoolExecutor

        from trnblas.nki.multicore import aten_rows, occ_chunks

        pool = ThreadPoolExecutor(args.threads)
        cpu = torch.device("cpu")

        def one(rows):
            torch.set_num_threads(1)
            return float(aten_rows(B, eo, ev, rows, cpu))

        def cpupar():
            return sum(pool.map(one, occ_chunks(nocc, args.threads)))

        ms = _median_ms(cpupar, max(1, args.reps // 2), lambda: None)
        _emit(
            op="energy",
            route="cpupar",
            threads=args.threads,
            nocc=nocc,
            nvir=nvir,
            naux=naux,
            ms=round(ms, 3),
        )


def _proc_energy_worker(core, k, nocc, nvir, naux, route, reps, barrier, q):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    os.environ["TRNBLAS_PAIR_ENERGY"] = route
    import torch

    import trnblas
    from trnblas.nki import dispatch as D
    from trnblas.nki import multicore as MC

    trnblas.set_backend("nki")
    D._get_native_ops()
    B, eo, ev = _mp2_inputs(nocc, nvir, naux)
    rows = MC.occ_chunks(nocc, k)[core]
    dev = torch.device("neuron", 0)
    fn = MC.sym_rows if route == "sym" else MC.aten_rows

    def run():
        # Each process owns one row range and starts from host B.
        return float(fn(B, eo, ev, rows, dev))

    e = run()
    ts = []
    for _ in range(reps):
        barrier.wait()
        t0 = time.time()
        run()
        ts.append((t0, time.time()))
    q.put((core, e, ts))


def run_energy_proc(args) -> None:
    import multiprocessing as mp

    nocc, nvir, naux = (int(x) for x in args.mp2.split(","))
    ctx = mp.get_context("spawn")
    for k in [int(c) for c in args.cores.split(",")]:
        barrier, q = ctx.Barrier(k), ctx.Queue()
        ps = [
            ctx.Process(
                target=_proc_energy_worker,
                args=(i, k, nocc, nvir, naux, args.route, args.reps, barrier, q),
            )
            for i in range(k)
        ]
        for p in ps:
            p.start()
        got = [q.get(timeout=600) for _ in ps]
        for p in ps:
            p.join()
        walls = [
            max(g[2][r][1] for g in got) - min(g[2][r][0] for g in got) for r in range(args.reps)
        ]
        _emit(
            op="energy",
            mode="proc",
            route=args.route,
            cores=k,
            nocc=nocc,
            nvir=nvir,
            naux=naux,
            ms=round(1e3 * statistics.median(walls), 3),
            e=sum(g[1] for g in got),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gemm")
    g.add_argument("--shape", required=True)
    g.add_argument("--modes", default="cpu,dev,split")
    g.add_argument("--cores", default="1,2,4")
    g.add_argument("--split", default="row,col")
    g.add_argument("--nki", action="store_true")
    g.add_argument("--threads-mode", action="store_true")
    b = sub.add_parser("bw")
    b.add_argument("--mb", type=int, default=256)
    for name in ("energy", "energy-proc"):
        e = sub.add_parser(name)
        e.add_argument("--mp2", required=True, help="nocc,nvir,naux")
        e.add_argument("--route", default="sym", choices=["sym", "aten"])
        e.add_argument("--cores", default="1,2,4")
        e.add_argument("--cpu", action="store_true")
    for p in sub.choices.values():
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    {"gemm": run_gemm, "bw": run_bw, "energy": run_energy, "energy-proc": run_energy_proc}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
