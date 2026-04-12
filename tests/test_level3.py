"""Test BLAS Level 3 operations."""

import pytest
import torch
import numpy as np
import trnblas


class TestGemm:
    def test_identity(self):
        A = torch.eye(4)
        B = torch.randn(4, 4)
        result = trnblas.gemm(1.0, A, B)
        np.testing.assert_allclose(result.numpy(), B.numpy(), atol=1e-6)

    def test_vs_torch(self, random_matrix):
        A = random_matrix(32, 16)
        B = random_matrix(16, 24)
        result = trnblas.gemm(2.0, A, B)
        expected = 2.0 * torch.matmul(A, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_transA(self, random_matrix):
        A = random_matrix(16, 32)  # A^T is 32×16
        B = random_matrix(16, 24)  # Need K=16 to match A^T rows
        result = trnblas.gemm(1.0, A, B, transA=True)
        expected = torch.matmul(A.T, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_transB(self, random_matrix):
        A = random_matrix(32, 16)
        B = random_matrix(24, 16)  # Will be transposed to 16×24
        result = trnblas.gemm(1.0, A, B, transB=True)
        expected = torch.matmul(A, B.T)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_beta(self, random_matrix):
        A = random_matrix(16, 16)
        B = random_matrix(16, 16)
        C = random_matrix(16, 16)
        result = trnblas.gemm(1.0, A, B, beta=2.0, C=C)
        expected = torch.matmul(A, B) + 2.0 * C
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_scaled(self, matrix_size, random_matrix):
        n = matrix_size
        A = random_matrix(n, n)
        B = random_matrix(n, n)
        alpha = 0.5
        result = trnblas.gemm(alpha, A, B)
        expected = alpha * torch.matmul(A, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-3, rtol=1e-3)


class TestBatchedGemm:
    def test_basic(self, random_matrix):
        batch = 8
        A = torch.randn(batch, 16, 32)
        B = torch.randn(batch, 32, 24)
        result = trnblas.batched_gemm(1.0, A, B)
        expected = torch.bmm(A, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_matches_loop(self):
        batch = 4
        A = torch.randn(batch, 8, 8)
        B = torch.randn(batch, 8, 8)
        batched = trnblas.batched_gemm(2.0, A, B)
        for i in range(batch):
            single = trnblas.gemm(2.0, A[i], B[i])
            np.testing.assert_allclose(batched[i].numpy(), single.numpy(), atol=1e-5)


class TestSymm:
    def test_vs_full(self, random_matrix):
        n = 16
        A = random_matrix(n, n)
        A = A + A.T  # Symmetric
        B = random_matrix(n, n)
        result = trnblas.symm(1.0, A, B, side="left", uplo="upper")
        expected = torch.matmul(A, B)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_right_side(self, random_matrix):
        n = 16
        A = random_matrix(n, n)
        A = A + A.T
        B = random_matrix(n, n)
        result = trnblas.symm(1.0, A, B, side="right")
        expected = torch.matmul(B, A)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)


class TestSyrk:
    def test_symmetric_result(self, random_matrix):
        A = random_matrix(16, 8)
        result = trnblas.syrk(1.0, A)
        # Result should be symmetric
        np.testing.assert_allclose(result.numpy(), result.T.numpy(), atol=1e-6)

    def test_vs_explicit(self, random_matrix):
        A = random_matrix(16, 8)
        result = trnblas.syrk(2.0, A)
        expected = 2.0 * torch.matmul(A, A.T)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

    def test_transpose(self, random_matrix):
        A = random_matrix(8, 16)
        result = trnblas.syrk(1.0, A, trans=True)
        expected = torch.matmul(A.T, A)
        # Symmetrized
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)


class TestTrsm:
    def test_identity(self):
        A = torch.eye(4)
        B = torch.randn(4, 3)
        result = trnblas.trsm(1.0, A, B, uplo="upper")
        np.testing.assert_allclose(result.numpy(), B.numpy(), atol=1e-6)

    def test_upper_triangular(self):
        A = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
        B = torch.tensor([[5.0], [9.0]])
        # Solve A @ X = B → X = A^{-1} @ B
        result = trnblas.trsm(1.0, A, B, uplo="upper")
        # Verify: A @ result ≈ B
        check = torch.matmul(A, result)
        np.testing.assert_allclose(check.numpy(), B.numpy(), atol=1e-6)

    def test_cholesky_solve(self, spd_matrix, random_matrix):
        """Simulate DF-MP2 Cholesky path: L @ L^T = J, solve L @ X = B."""
        n = 16
        J = spd_matrix(n)
        L = torch.linalg.cholesky(J)
        B = random_matrix(n, 8)

        # Solve L @ X = B
        X = trnblas.trsm(1.0, L, B, uplo="lower")
        # Verify
        check = torch.matmul(L, X)
        np.testing.assert_allclose(check.numpy(), B.numpy(), atol=1e-4)


class TestTrmm:
    def test_basic(self):
        A = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
        B = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = trnblas.trmm(1.0, A, B, uplo="upper")
        np.testing.assert_allclose(result.numpy(), A.numpy(), atol=1e-6)
