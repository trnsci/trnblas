"""Test configuration and fixtures."""

import atexit
import os
import shutil
import tempfile

# NKI trace cache (aws-neuron/aws-neuron-sdk#1454, trnsci/trnsci#35).
# `import torch_neuronx` turns the cache on (NKI_ENABLE_TRACE_CACHE=1), so
# that is how users run these kernels, and the neuron tests run the same way.
# The cache lives on disk and is versioned by wheel .dist-info, which an
# editable install or source checkout does not change, so a shared cache could
# serve a trace of an older kernel. Each session therefore gets a fresh,
# empty cache directory (NKI_TRACE_CACHE_URL), removed at exit.
# tests/test_trace_cache_safety.py checks that no kernel depends on a value
# the cache key misses (closure or global). Overrides: NKI_ENABLE_TRACE_CACHE=0
# runs with the cache off; an explicit NKI_TRACE_CACHE_URL is kept as is.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "1")
if os.environ["NKI_ENABLE_TRACE_CACHE"] != "0" and "NKI_TRACE_CACHE_URL" not in os.environ:
    _trace_cache_dir = tempfile.mkdtemp(prefix="trnblas-nki-trace-cache-")
    os.environ["NKI_TRACE_CACHE_URL"] = _trace_cache_dir
    atexit.register(shutil.rmtree, _trace_cache_dir, ignore_errors=True)

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
