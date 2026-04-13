"""Minimal NKI health probe — is the kernel actually dispatching?

Forces TRNBLAS_REQUIRE_NKI=1 (re-raise on kernel errors instead of
silently falling back to torch.matmul), forces the `nki` backend, and
calls trnblas.gemm once on a tiny shape. If NKI is healthy we get a
measurable cold/warm split (cold includes NEFF compile on a fresh
instance; warm is cache-hit fast). If NKI has been silently falling
back, the underlying `_to_xla` / kernel-dispatch error surfaces as an
uncaught traceback.

Run via SSM on trn1:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh  # make sure instance up
    # then inline SSM sending:
    # TRNBLAS_REQUIRE_NKI=1 \\
    # /opt/aws_neuronx_venv_pytorch_*/bin/python \\
    # /home/ubuntu/trnblas/scripts/probe_nki.py

Exits 0 on success, non-zero on failure. The failure path's traceback
is the diagnostic signal.
"""

from __future__ import annotations

import os
import sys
import time

# Force the re-raise path before importing trnblas — _REQUIRE_NKI is
# read at module import time in trnblas.nki.dispatch.
os.environ["TRNBLAS_REQUIRE_NKI"] = "1"

import torch

import trnblas
from trnblas.nki import HAS_NKI


def main() -> int:
    print(f"HAS_NKI = {HAS_NKI}")
    print(f"TRNBLAS_REQUIRE_NKI = {os.environ.get('TRNBLAS_REQUIRE_NKI')}")
    if not HAS_NKI:
        print("HAS_NKI is False — neuronxcc not importable. Exiting.")
        return 0

    trnblas.set_backend("nki")
    print(f"backend = {trnblas.get_backend()}")

    torch.manual_seed(0)
    A = torch.randn(128, 128)
    B = torch.randn(128, 128)
    ref = A @ B

    # Cold: first call. If NKI is healthy, this includes NEFF compile
    # on a cold instance (seconds). If it's silently falling back, we
    # would see torch.matmul speed (~ms) — but with REQUIRE_NKI=1,
    # the exception surfaces as a traceback.
    t0 = time.perf_counter()
    out = trnblas.gemm(1.0, A, B)
    cold = time.perf_counter() - t0
    print(f"cold call: {cold*1000:.2f} ms")

    # Warm average.
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = trnblas.gemm(1.0, A, B)
        times.append(time.perf_counter() - t0)
    warm = sum(times) / len(times)
    print(f"warm mean: {warm*1000:.3f} ms")

    # Correctness.
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-4)
    print(f"matches torch @ atol=1e-3: {ok}")

    # NKI signature test: if cold is ≥ 10× warm, NEFF compile happened
    # → real NKI. If cold ≈ warm, suspicious (likely CPU).
    ratio = cold / warm if warm > 0 else 0.0
    print(f"cold/warm ratio: {ratio:.1f}x  "
          f"({'NKI signature' if ratio > 10 else 'CPU-like (suspect!)'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
