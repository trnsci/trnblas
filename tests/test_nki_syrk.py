"""On-hardware validation for the NKI SYRK kernel (#18).

CPU path (torch fallback) is always exercised; the NKI path is marked
`@pytest.mark.neuron` and runs only on Trainium hardware. Mirrors the
structure of `tests/test_nki_gemm.py`.
"""

import pytest
import torch

from trnblas import syrk
from trnblas.nki import nki_syrk


ATOL = 1e-3
RTOL = 1e-4


@pytest.fixture
def aligned_shapes():
    """Square + rectangular (M, K) shapes where every dim is a multiple of 128."""
    return [
        (128, 128),
        (256, 256),
        (512, 512),
        (256, 128),
        (1024, 256),
    ]


@pytest.fixture
def edge_shapes():
    """Non-128-aligned shapes — exercise HBM padding."""
    return [
        (100, 100),
        (200, 128),
        (128, 200),
        (300, 175),
    ]


def _check_kernel(A):
    out = nki_syrk(A)
    torch.testing.assert_close(out, A @ A.T, atol=ATOL, rtol=RTOL)


class TestTorchFallbackCPU:
    """CPU path — torch.matmul(A, A.T) under the nki_syrk wrapper."""

    def test_basic(self):
        torch.manual_seed(0)
        A = torch.randn(128, 64)
        out = nki_syrk(A)
        torch.testing.assert_close(out, A @ A.T, atol=ATOL, rtol=RTOL)

    def test_syrk_public_api(self):
        torch.manual_seed(0)
        A = torch.randn(64, 32)
        out = syrk(1.0, A)
        torch.testing.assert_close(out, A @ A.T, atol=ATOL, rtol=RTOL)

    def test_syrk_trans(self):
        torch.manual_seed(0)
        A = torch.randn(32, 64)
        out = syrk(1.0, A, trans=True)
        torch.testing.assert_close(out, A.T @ A, atol=ATOL, rtol=RTOL)


class TestNkiSyrkKernel:
    """Direct `nki_syrk` kernel correctness on Trainium."""

    pytestmark = pytest.mark.neuron

    def test_aligned_shapes(self, nki_backend, aligned_shapes, rng):
        for M, K in aligned_shapes:
            A = torch.randn(M, K, generator=rng)
            _check_kernel(A)

    def test_edge_shapes(self, nki_backend, edge_shapes, rng):
        for M, K in edge_shapes:
            A = torch.randn(M, K, generator=rng)
            _check_kernel(A)

    def test_identity(self, nki_backend):
        I = torch.eye(256)
        out = nki_syrk(I)
        torch.testing.assert_close(out, torch.eye(256), atol=ATOL, rtol=RTOL)

    def test_zero(self, nki_backend):
        A = torch.zeros(128, 128)
        out = nki_syrk(A)
        torch.testing.assert_close(out, torch.zeros(128, 128), atol=0, rtol=0)


class TestSyrkDispatch:
    """Top-level `syrk()` BLAS call routes through the NKI kernel."""

    pytestmark = pytest.mark.neuron

    def test_basic(self, nki_backend, rng):
        A = torch.randn(256, 128, generator=rng)
        out = syrk(1.0, A)
        torch.testing.assert_close(out, A @ A.T, atol=ATOL, rtol=RTOL)

    def test_trans(self, nki_backend, rng):
        A = torch.randn(128, 256, generator=rng)
        out = syrk(1.0, A, trans=True)
        torch.testing.assert_close(out, A.T @ A, atol=ATOL, rtol=RTOL)

    def test_alpha_beta(self, nki_backend, rng):
        A = torch.randn(128, 64, generator=rng)
        C0 = torch.randn(128, 128, generator=rng)
        # Symmetrise C0 because syrk returns symmetric and adds beta*C.
        C0 = 0.5 * (C0 + C0.T)
        out = syrk(2.0, A, beta=0.5, C=C0.clone())
        ref = 2.0 * (A @ A.T) + 0.5 * C0
        ref = 0.5 * (ref + ref.T)  # match the symmetrisation inside syrk
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
