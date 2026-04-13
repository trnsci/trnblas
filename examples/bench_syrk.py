"""Per-op timing for trnblas.syrk (NKI on trn1, cuBLAS on A10G via torch).

Usage:
    python examples/bench_syrk.py                 # CPU
    python examples/bench_syrk.py --device cuda   # on a CUDA instance
    # on a trn1 instance: runs via the neuronxcc venv, uses NKI automatically.

Reports cold / warm_mean per-call timings and effective TFLOPS across a
fixed set of (M, K) shapes. Output format mirrors docs/benchmarks.md rows.
"""

import argparse
import time

import torch

import trnblas


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


def bench(M: int, K: int, device: str) -> None:
    torch.manual_seed(0)
    A = torch.randn(M, K)
    if device != "cpu":
        A = A.to(device)

    # cold
    t0 = time.perf_counter()
    trnblas.syrk(1.0, A)
    _sync(device)
    cold = time.perf_counter() - t0

    # warm average
    times = []
    for _ in range(WARMS):
        t0 = time.perf_counter()
        trnblas.syrk(1.0, A)
        _sync(device)
        times.append(time.perf_counter() - t0)
    warm = sum(times) / len(times)

    flops = 2 * M * M * K  # syrk: M*M output, each a K-length dot product
    tflops_cold = flops / cold / 1e12
    tflops_warm = flops / warm / 1e12
    print(
        f"  M={M:<5d} K={K:<5d}  "
        f"cold={cold*1000:7.2f}ms ({tflops_cold:5.2f} TFLOPS)  "
        f"warm={warm*1000:7.2f}ms ({tflops_warm:5.2f} TFLOPS)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"[syrk bench  device={args.device}  backend={trnblas.get_backend()}]")
    for M, K in SHAPES:
        bench(M, K, args.device)


if __name__ == "__main__":
    main()
