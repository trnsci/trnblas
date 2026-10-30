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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pair", "gemm"])
    args = ap.parse_args()
    path = D._get_dispatch_path()
    print(f"INFO dispatch_path={path}", flush=True)
    if path != "native":
        raise SystemExit("needs the native TorchNeuron path")
    if args.mode == "pair":
        # All at or under the XLA threshold (est_iters <= 4096), so the full-batch
        # kernel stays within the trace size known to compile.
        bench_pair([(4, 128, 128), (16, 128, 384), (32, 128, 384), (64, 128, 128), (16, 256, 512)])
    else:
        bench_gemm([128, 256, 512, 1000, 1024, 2047, 2048, 4095, 4096])


if __name__ == "__main__":
    main()
