"""Test BLAS Level 2 operations."""

import numpy as np
import pytest
import torch

import trnblas


class TestGemv:
    def test_identity(self):
        A = torch.eye(3)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = trnblas.gemv(1.0, A, x)
        np.testing.assert_allclose(result.numpy(), x.numpy())

    def test_vs_torch(self, random_matrix, random_vector):
        A = random_matrix(32, 32)
        x = random_vector(32)
        result = trnblas.gemv(2.0, A, x)
        expected = 2.0 * torch.mv(A, x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_transpose(self, random_matrix, random_vector):
        A = random_matrix(16, 32)
        x = random_vector(16)
        result = trnblas.gemv(1.0, A, x, trans=True)
        expected = torch.mv(A.T, x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_beta(self, random_matrix, random_vector):
        A = random_matrix(16, 16)
        x = random_vector(16)
        y = random_vector(16)
        result = trnblas.gemv(1.0, A, x, beta=2.0, y=y)
        expected = torch.mv(A, x) + 2.0 * y
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)


class TestSymv:
    def test_symmetric(self, random_matrix, random_vector):
        n = 16
        A = random_matrix(n, n)
        A = A + A.T  # Make symmetric
        x = random_vector(n)
        result = trnblas.symv(1.0, A, x, uplo="upper")
        expected = torch.mv(A, x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)


class TestTrmv:
    def test_upper(self):
        A = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
        x = torch.tensor([1.0, 1.0])
        result = trnblas.trmv(A, x, uplo="upper")
        np.testing.assert_allclose(result.numpy(), [3.0, 3.0])

    def test_unit_diag(self):
        A = torch.tensor([[99.0, 2.0], [0.0, 99.0]])
        x = torch.tensor([1.0, 1.0])
        result = trnblas.trmv(A, x, uplo="upper", diag="unit")
        np.testing.assert_allclose(result.numpy(), [3.0, 1.0])


class TestGer:
    def test_basic(self):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        result = trnblas.ger(1.0, x, y)
        expected = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(result.numpy(), expected.numpy())

    def test_accumulate(self):
        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([0.0, 1.0])
        A = torch.eye(2)
        result = trnblas.ger(1.0, x, y, A=A)
        expected = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(result.numpy(), expected.numpy())
