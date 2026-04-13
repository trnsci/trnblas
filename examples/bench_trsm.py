"""Per-op timing for trnblas.trsm (blocked NKI on trn1, cuBLAS on A10G).

Usage:
    python examples/bench_trsm.py                 # CPU
    python examples/bench_trsm.py --device cuda   # on a CUDA instance
    # on a trn1 instance: runs via the neuronxcc venv, uses NKI automatically.

Matches the bench_syrk.py pattern: reports cold and warm-mean per-call
timings + effective TFLOPS across a fixed set of (M, N) shapes. The DF-MP2
metric-inversion call shape is `trsm(1.0, L, I, uplo='lower', trans=True)`
where L is (naux, naux); we cover that plus a few more M, N combos.
"""

import argparse
import time

import torch

import trnblas


# (M, N) — M is the triangular side, N is the RHS width.
SHAPES = [
    (512, 512),
    (1024, 512),
    (1024, 1024),
    (2048, 512),
]

WARMS = 5


def _sync(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(M: int, N: int, device: str) -> None:
    torch.manual_seed(0)
    # Build a well-conditioned lower-triangular L via Cholesky of an SPD
    # matrix. Keeps the triangular solve numerically stable in fp32.
    A = torch.randn(M, M)
    SPD = A @ A.T + M * torch.eye(M)
    L = torch.linalg.cholesky(SPD)
    B = torch.randn(M, N)
    if device != "cpu":
        L = L.to(device)
        B = B.to(device)

    # cold
    t0 = time.perf_counter()
    trnblas.trsm(1.0, L, B, uplo="lower", trans=True)
    _sync(device)
    cold = time.perf_counter() - t0

    # warm average
    times = []
    for _ in range(WARMS):
        t0 = time.perf_counter()
        trnblas.trsm(1.0, L, B, uplo="lower", trans=True)
        _sync(device)
        times.append(time.perf_counter() - t0)
    warm = sum(times) / len(times)

    # TRSM flop count: M² N (each of N RHS columns does an O(M²) solve).
    flops = M * M * N
    tflops_cold = flops / cold / 1e12
    tflops_warm = flops / warm / 1e12
    print(
        f"  M={M:<5d} N={N:<5d}  "
        f"cold={cold*1000:7.2f}ms ({tflops_cold:5.2f} TFLOPS)  "
        f"warm={warm*1000:7.2f}ms ({tflops_warm:5.2f} TFLOPS)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"[trsm bench  device={args.device}  backend={trnblas.get_backend()}]")
    for M, N in SHAPES:
        bench(M, N, args.device)


if __name__ == "__main__":
    main()
