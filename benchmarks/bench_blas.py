"""BLAS benchmarks.

Run with:

    pytest benchmarks/ --benchmark-only

Baseline is PyTorch. When validated, these should be re-run on trn1/trn2
with `set_backend("nki")` for the comparison vs cuBLAS write-up.
"""

import torch

import trnblas


def test_gemm(benchmark, square_matrices):
    A, B = square_matrices
    benchmark(lambda: trnblas.gemm(1.0, A, B))


def test_syrk(benchmark, square_matrices):
    A, _ = square_matrices
    benchmark(lambda: trnblas.syrk(1.0, A, trans=True))


def test_trsm(benchmark, square_size):
    n = square_size
    torch.manual_seed(1)
    A = torch.randn(n, n)
    L = torch.linalg.cholesky(A @ A.T + n * torch.eye(n))
    B = torch.randn(n, n)
    benchmark(lambda: trnblas.trsm(1.0, L, B, uplo="lower"))


def test_batched_gemm(benchmark):
    torch.manual_seed(2)
    A = torch.randn(32, 256, 128)
    B = torch.randn(32, 128, 64)
    benchmark(lambda: trnblas.batched_gemm(1.0, A, B))
