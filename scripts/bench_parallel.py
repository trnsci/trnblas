#!/usr/bin/env python
"""SPMD DF-MP2 and tensor-parallel GEMM, one process per NeuronCore (#24).

Run under torchrun with the TorchNeuron ``neuron`` backend; rank 0 prints
``RESULT {json}`` lines with warm medians (ms) over ``--reps`` after one
warm-up rep. Every rep starts at a barrier, and its time runs until the
slowest rank finishes.

    torchrun --nproc_per_node=4 scripts/bench_parallel.py mp2 --shape 768,96,2304
    torchrun --nproc_per_node=4 scripts/bench_parallel.py gemm --shape 8192,8192,8192 --split row

``mp2``   `trnblas.parallel.df_mp2_energy` from host inputs (synthetic
          `examples/df_mp2._make_inputs`), with the per-stage split.
``gemm``  C = A @ B with the shards already resident on their cores:
          ``row`` (A rows split), ``col`` (B columns split), ``k`` (the
          contraction split, all-reduced). ``compute`` is the local GEMM
          alone; ``gathered`` adds the collective that gives every rank the
          full C.

``baseline`` (no torchrun) times the single-core library path for ``mp2``:
`examples/df_mp2.df_mp2_energy` with ``use_batched_pair=True`` on inputs
resident on ``neuron:0``. ``ms`` adds the one host-to-device upload of the
inputs to the warm compute time, so it compares with the SPMD runs, which
start from host inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))


def _emit(**rec) -> None:
    print("RESULT " + json.dumps(rec), flush=True)


def _sync(dev) -> None:
    import torch_neuronx

    torch_neuronx.synchronize(dev.index)


def _init():
    import torch.distributed as dist
    from torch_neuronx.distributed.backend import _register_neuron_backend

    if "neuron" not in dist.Backend.backend_type_map:
        _register_neuron_backend()
    dist.init_process_group(backend="neuron")
    return dist


def _inputs(args):
    """Synthetic inputs; with ``--sym`` (mu nu|P) is symmetrised like real integrals."""
    from df_mp2 import _make_inputs

    nbasis, nocc, naux = (int(x) for x in args.shape.split(","))
    inputs = list(_make_inputs(nbasis, nocc, naux))
    if args.sym or args.packed:
        eri = inputs[2]
        inputs[2] = 0.5 * (eri + eri.transpose(0, 1))
    return inputs


def run_mp2(args) -> None:
    import trnblas
    import trnblas.parallel as tp

    dist = _init()
    r, w = dist.get_rank(), dist.get_world_size()
    trnblas.set_backend("nki")
    nbasis, nocc, naux = (int(x) for x in args.shape.split(","))
    inputs = _inputs(args)
    if args.packed:
        inputs[2] = tp.pack_tril(inputs[2])  # what PySCF's _cderi.T already is
    dev = tp.device()
    walls, stages, e = [], [], None
    for rep in range(args.reps + 1):
        t = {}
        dist.barrier()
        t0 = time.perf_counter()
        sym = args.sym or args.packed
        e = tp.df_mp2_energy(*inputs, timings=t if args.stages else None, eri_symmetric=sym)
        _sync(dev)
        dist.barrier()
        wall = time.perf_counter() - t0
        if rep == 0:
            first = wall
            continue
        walls.append(wall)
        stages.append(t)
    if r == 0:
        rec = dict(
            op="mp2",
            mode="spmd-packed" if args.packed else "spmd-sym" if args.sym else "spmd",
            ranks=w,
            nbasis=nbasis,
            nocc=nocc,
            naux=naux,
            ms=round(1e3 * statistics.median(walls), 2),
            first_s=round(first, 2),
            e=e,
        )
        if args.stages:
            rec["stages_ms"] = {
                k: round(1e3 * statistics.median(s[k] for s in stages), 2) for k in stages[0]
            }
        _emit(**rec)
    dist.destroy_process_group()


def run_baseline(args) -> None:
    from df_mp2 import df_mp2_energy

    import trnblas

    trnblas.set_backend("nki")
    nbasis, nocc, naux = (int(x) for x in args.shape.split(","))
    host = _inputs(args)
    dev = torch.device("neuron", 0)
    torch.ones(1 << 18).to(dev)  # runtime init, so it is not counted as upload
    _sync(dev)
    t0 = time.perf_counter()
    inputs = [x.to(dev) for x in host]
    _sync(dev)
    h2d = time.perf_counter() - t0
    del host
    walls, stages, e = [], [], None
    for rep in range(args.reps + 1):
        t = {}
        t0 = time.perf_counter()
        e = df_mp2_energy(*inputs, timings=t, use_batched_pair=True)
        wall = time.perf_counter() - t0
        if rep == 0:
            first = wall
            continue
        walls.append(wall)
        stages.append(t)
    _emit(
        op="mp2",
        mode="single-core",
        ranks=1,
        nbasis=nbasis,
        nocc=nocc,
        naux=naux,
        ms=round(1e3 * (statistics.median(walls) + h2d), 2),
        compute_ms=round(1e3 * statistics.median(walls), 2),
        first_s=round(first, 2),
        e=e,
        stages_ms={
            "h2d": round(1e3 * h2d, 2),
            **{k: round(1e3 * statistics.median(s[k] for s in stages), 2) for k in stages[0]},
        },
    )


def run_gemm(args) -> None:
    import trnblas.parallel as tp

    dist = _init()
    r, w = dist.get_rank(), dist.get_world_size()
    M, K, N = (int(x) for x in args.shape.split(","))
    g = torch.Generator().manual_seed(0)
    A = torch.randn(M, K, generator=g)
    B = torch.randn(K, N, generator=g)
    dev = tp.device()
    if args.split == "row":
        m = M // w
        a, b = A[r * m : (r + 1) * m].to(dev), B.to(dev)
        local = lambda: a @ b  # noqa: E731
        full = lambda: tp.gemm_row_parallel(a, b, gather=True)  # noqa: E731
    elif args.split == "col":
        n = N // w
        a, b = A.to(dev), B[:, r * n : (r + 1) * n].contiguous().to(dev)
        local = lambda: a @ b  # noqa: E731
        full = lambda: tp.gemm_col_parallel(a, b, gather=True)  # noqa: E731
    else:
        k = K // w
        a, b = A[:, r * k : (r + 1) * k].contiguous().to(dev), B[r * k : (r + 1) * k].to(dev)
        local = lambda: a @ b  # noqa: E731
        full = lambda: tp.gemm_k_parallel(a, b)  # noqa: E731
    _sync(dev)
    out = {}
    for label, fn in (("compute", local), ("gathered", full)):
        fn()
        _sync(dev)
        ts = []
        for _ in range(args.reps):
            dist.barrier()
            t0 = time.perf_counter()
            fn()
            _sync(dev)
            dist.barrier()
            ts.append(time.perf_counter() - t0)
        out[label] = round(1e3 * statistics.median(ts), 3)
    Cf = full().cpu()
    if r == 0:
        ref = A.double() @ B.double()
        err = float((Cf.double() - ref).abs().max() / ref.abs().max())
        _emit(op="gemm", mode="spmd", split=args.split, ranks=w, M=M, K=K, N=N, rel_err=err, **out)
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["mp2", "baseline", "gemm"])
    ap.add_argument("--shape", required=True)
    ap.add_argument("--split", default="row", choices=["row", "col", "k"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stages", action="store_true", help="per-stage timings (adds syncs)")
    ap.add_argument("--sym", action="store_true", help="symmetric ERI; spmd reads half of it")
    ap.add_argument("--packed", action="store_true", help="--sym, ERI passed as pack_tril")
    args = ap.parse_args()
    {"mp2": run_mp2, "baseline": run_baseline, "gemm": run_gemm}[args.cmd](args)


if __name__ == "__main__":
    main()
