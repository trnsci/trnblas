"""Test BLAS Level 1 operations."""

import numpy as np
import pytest
import torch

import trnblas


class TestAxpy:
    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        result = trnblas.axpy(2.0, x, y)
        np.testing.assert_allclose(result.numpy(), [6.0, 9.0, 12.0])

    def test_zero_alpha(self):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        result = trnblas.axpy(0.0, x, y)
        np.testing.assert_allclose(result.numpy(), y.numpy())


class TestDot:
    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        assert trnblas.dot(x, y).item() == pytest.approx(32.0)

    def test_orthogonal(self):
        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([0.0, 1.0])
        assert trnblas.dot(x, y).item() == pytest.approx(0.0)


class TestNrm2:
    def test_unit(self):
        x = torch.tensor([3.0, 4.0])
        assert trnblas.nrm2(x).item() == pytest.approx(5.0)

    def test_random(self, random_vector):
        x = random_vector(100)
        expected = torch.linalg.norm(x)
        np.testing.assert_allclose(trnblas.nrm2(x).item(), expected.item(), rtol=1e-5)


class TestScal:
    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = trnblas.scal(3.0, x)
        np.testing.assert_allclose(result.numpy(), [3.0, 6.0, 9.0])


class TestAsum:
    def test_basic(self):
        x = torch.tensor([-1.0, 2.0, -3.0])
        assert trnblas.asum(x).item() == pytest.approx(6.0)


class TestIamax:
    def test_basic(self):
        x = torch.tensor([1.0, -5.0, 3.0, -2.0])
        assert trnblas.iamax(x) == 1
