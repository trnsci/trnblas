"""Test configuration and fixtures."""

import numpy as np
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
    config.addinivalue_line("markers", "pyscf: requires PySCF (optional dep)")
    config.addinivalue_line(
        "markers",
        "nki_simulator: runs NKI kernels via nki.simulate on CPU (requires "
        "TRNBLAS_USE_SIMULATOR=1 + nki>=0.3.0)",
    )


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture(params=[16, 32, 64, 128, 256])
def matrix_size(request):
    return request.param


@pytest.fixture
def random_matrix(rng):
    def _make(m, n, dtype=torch.float32):
        return torch.randn(m, n, generator=rng, dtype=dtype)

    return _make


@pytest.fixture
def random_vector(rng):
    def _make(n, dtype=torch.float32):
        return torch.randn(n, generator=rng, dtype=dtype)

    return _make


@pytest.fixture
def spd_matrix(rng):
    """Symmetric positive definite matrix (for Cholesky tests)."""

    def _make(n, dtype=torch.float32):
        A = torch.randn(n, n, generator=rng, dtype=dtype)
        return A @ A.T + n * torch.eye(n, dtype=dtype)

    return _make


@pytest.fixture
def nki_backend():
    """Force the NKI backend for the duration of a test."""
    from trnblas import get_backend, set_backend

    old = get_backend()
    set_backend("nki")
    yield
    set_backend(old)
