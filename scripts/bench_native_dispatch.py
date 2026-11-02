#!/usr/bin/env python
"""Native-path re-benchmarks for #50: batched-pair routing and NKI GEMM vs aten.

Run on a TorchNeuron (Trn2/Trn3) host:

    NKI_ENABLE_TRACE_CACHE=0 python scripts/bench_native_dispatch.py pair
    NKI_ENABLE_TRACE_CACHE=0 python scripts/bench_native_dispatch.py gemm

``pair``  For each (nocc, nvir, naux), times the full-batch kernel
          (``_batched_pair_kernel``, one launch) against the chunked kernel
          (``_j_batched_kernel``, one launch per i-row). The native default
          (``_BATCHED_PAIR_CHUNK_THRESHOLD_NATIVE = 3072``) came from this
          sweep on trn2 (#52).
``gemm``  ``trnblas.gemm`` (NKI) vs ``A @ B`` (aten) on device-resident inputs,
          aligned and ragged shapes.
``call``  Host cost of one kernel call on device-resident inputs: the
          ``trnblas::`` custom op against the cached eager handle
          (``TRNBLAS_NKI_EAGER_HANDLE``), with aten for scale. ``issue`` is the
          host time per call of ``n`` back-to-back calls, ``e2e`` the median of
          one call plus a synchronize; ms.

Each line: ``RESULT <mode> <case> first=<s> warm_median=<ms> [extra]``.
Inputs are created on the host and copied to the device before timing:
on-device ``torch.randn`` is shape-sensitive (aws-neuron/aws-neuron-sdk#1439).
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import trnblas
from trnblas.nki import dispatch as D

WARM = 10


def _sync():
    D._device_synchronize()


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    _sync()
    first = time.perf_counter() - t0
    warm = []
    for _ in range(WARM):
        t0 = time.perf_counter()
        fn()
        _sync()
        warm.append(time.perf_counter() - t0)
    return out, first, statistics.median(warm) * 1e3


def bench_pair(shapes):
    trnblas.set_backend("nki")
    for nocc, nvir, naux in shapes:
        g = torch.Generator().manual_seed(0)
        B = 0.1 * torch.randn(nocc, nvir, naux, generator=g)
        eps_occ = -1.0 - torch.rand(nocc, generator=g)
        eps_vir = 0.5 + torch.rand(nvir, generator=g)
        ref = float(D._torch_batched_pair_energy(B.double(), eps_occ.double(), eps_vir.double()))
        for label, threshold in (("full", 1 << 62), ("chunked", 0)):
            case = f"nocc={nocc},nvir={nvir},naux={naux},{label}"
            D.os.environ["TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD"] = str(threshold)
            try:
                e, first, warm = _time(
                    lambda B=B, eo=eps_occ, ev=eps_vir: float(
                        D._nki_batched_pair_energy_impl(B, eo, ev)
                    )
                )
                print(
                    f"RESULT pair {case} first={first:.2f} warm_median={warm:.3f} "
                    f"rel_err={abs(e - ref) / abs(ref):.2e}",
                    flush=True,
                )
            except Exception as exc:
                print(f"RESULT pair {case} ERROR {str(exc).splitlines()[0][:160]}", flush=True)
        D.os.environ.pop("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", None)


def bench_gemm(sizes):
    trnblas.set_backend("nki")
    dev = "neuron"
    for n in sizes:
        g = torch.Generator().manual_seed(n)
        A = torch.randn(n, n, generator=g).to(dev)
        B = torch.randn(n, n, generator=g).to(dev)
        _sync()
        _, f_nki, w_nki = _time(lambda A=A, B=B: trnblas.gemm(1.0, A, B))
        _, f_aten, w_aten = _time(lambda A=A, B=B: A @ B)
        print(
            f"RESULT gemm N={n} nki_first={f_nki:.2f} nki_warm_median={w_nki:.3f} "
            f"aten_first={f_aten:.2f} aten_warm_median={w_aten:.3f} "
            f"aten_speedup={w_nki / w_aten:.2f}",
            flush=True,
        )


def _issue_e2e(fn, n=200):
    for _ in range(10):
        fn()
    _sync()
    issue = []
    for _ in range(5):
        t0 = time.perf_counter()
        rs = [fn() for _ in range(n)]
        issue.append((time.perf_counter() - t0) / n * 1e3)
        _sync()
        del rs
    e2e = []
    for _ in range(100):
        t0 = time.perf_counter()
        fn()
        _sync()
        e2e.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(issue), statistics.median(e2e)


def bench_call():
    trnblas.set_backend("nki")
    D._REQUIRE_NKI = True
    g = torch.Generator().manual_seed(0)
    cases = []
    for m, k, n in ((128, 128, 512), (1024, 1024, 1024)):
        A, B = torch.randn(m, k, generator=g), torch.randn(k, n, generator=g)
        a, b = A.to("neuron"), B.to("neuron")
        tile = D._gemm_kernel_registry[(D._TILE_M, D._TILE_K, D._TILE_N)]
        op = D._gemm_op_name(D._TILE_M, D._TILE_K, D._TILE_N)
        ref = A.double() @ B.double()
        cases += [
            (f"aten a@b {m}x{k}x{n}", lambda a=a, b=b: a @ b, None),
            (
                f"kernel {op} {m}x{k}x{n}",
                lambda a=a, b=b, t=tile, o=op: D._call_kernel(t, o, a, b),
                ref,
            ),
            (f"trnblas.gemm {m}x{k}x{n}", lambda a=a, b=b: trnblas.gemm(1.0, a, b), ref),
        ]
    S = torch.randn(256, 128, generator=g)
    s_dev = S.to("neuron")
    cases.append(
        ("kernel syrk 256x128", lambda: D._call_kernel(D._syrk_kernel, "syrk", s_dev), None)
    )
    print(f"{'case':36s} {'path':7s} {'issue':>8s} {'e2e':>8s}  rel_err", flush=True)
    for name, fn, ref in cases:
        paths = ("aten",) if ref is None and name.startswith("aten") else ("op", "handle")
        for path in paths:
            D._EAGER_HANDLE_ENABLED = path != "op"
            out = fn()
            err = ""
            if ref is not None:
                err = f"{((out.double().cpu() - ref).norm() / ref.norm()).item():.1e}"
            del out
            issue, e2e = _issue_e2e(fn)
            print(f"{name:36s} {path:7s} {issue:8.4f} {e2e:8.4f}  {err}", flush=True)
    D._EAGER_HANDLE_ENABLED = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pair", "gemm", "call"])
    args = ap.parse_args()
    path = D._get_dispatch_path()
    print(f"INFO dispatch_path={path}", flush=True)
    if path != "native":
        raise SystemExit("needs the native TorchNeuron path")
    if args.mode == "call":
        bench_call()
    elif args.mode == "pair":
        # All at or under the XLA threshold (est_iters <= 4096), so the full-batch
        # kernel stays within the trace size known to compile.
        bench_pair([(4, 128, 128), (16, 128, 384), (32, 128, 384), (64, 128, 128), (16, 256, 512)])
    else:
        bench_gemm([128, 256, 512, 1000, 1024, 2047, 2048, 4095, 4096])


if __name__ == "__main__":
    main()
