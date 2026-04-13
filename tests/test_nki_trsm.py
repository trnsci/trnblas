"""On-hardware validation for the NKI TRSM (blocked panel solve, #19).

CPU path (torch fallback) is always exercised; the NKI path is marked
`@pytest.mark.neuron` and runs only on Trainium hardware. Mirrors the
structure of `tests/test_nki_syrk.py`.

Blocked TRSM accumulates fp32 rounding across the trailing-GEMM updates,
so the tolerance is slightly looser than SYRK's (atol=1e-3, rtol=1e-3).
"""

import pytest
import torch

from trnblas import trsm
from trnblas.nki import nki_trsm


ATOL = 1e-3
RTOL = 1e-3


def _spd(n, rng, dtype=torch.float32):
    """Well-conditioned SPD matrix for stable Cholesky factors."""
    A = torch.randn(n, n, generator=rng, dtype=dtype)
    return A @ A.T + n * torch.eye(n, dtype=dtype)


@pytest.fixture
def tri_shapes():
    """(M, N) shapes covering aligned + non-aligned triangular sides."""
    return [
        (128, 128),
        (256, 128),
        (512, 256),
        (200, 150),   # both unaligned
    ]


class TestTorchFallbackCPU:
    """CPU path — confirm rewiring preserves behaviour."""

    def test_lower_nontrans(self):
        torch.manual_seed(0)
        A = _spd(64, torch.Generator().manual_seed(0))
        L = torch.linalg.cholesky(A)
        B = torch.randn(64, 32)
        X = trsm(1.0, L, B, uplo="lower")
        torch.testing.assert_close(L @ X, B, atol=ATOL, rtol=RTOL)

    def test_lower_trans(self):
        torch.manual_seed(0)
        A = _spd(64, torch.Generator().manual_seed(0))
        L = torch.linalg.cholesky(A)
        B = torch.randn(64, 32)
        X = trsm(1.0, L, B, uplo="lower", trans=True)
        torch.testing.assert_close(L.T @ X, B, atol=ATOL, rtol=RTOL)

    def test_upper_nontrans(self):
        torch.manual_seed(0)
        A = _spd(64, torch.Generator().manual_seed(0))
        U = torch.linalg.cholesky(A).T
        B = torch.randn(64, 32)
        X = trsm(1.0, U, B, uplo="upper")
        torch.testing.assert_close(U @ X, B, atol=ATOL, rtol=RTOL)

    def test_unit_diag(self):
        torch.manual_seed(0)
        # Unit lower-triangular with small off-diagonal values so the
        # triangular solve stays well-conditioned in fp32.
        L = torch.eye(64) + 0.05 * torch.tril(torch.randn(64, 64), diagonal=-1)
        B = torch.randn(64, 32)
        X = trsm(1.0, L, B, uplo="lower", diag="unit")
        torch.testing.assert_close(L @ X, B, atol=ATOL, rtol=RTOL)

    def test_right_side(self):
        """side='right' falls back to torch regardless of NKI."""
        torch.manual_seed(0)
        L = torch.linalg.cholesky(_spd(64, torch.Generator().manual_seed(0)))
        B = torch.randn(32, 64)
        X = trsm(1.0, L, B, side="right", uplo="lower")
        torch.testing.assert_close(X @ L, B, atol=ATOL, rtol=RTOL)


class TestNkiTrsmBlocked:
    """Direct `nki_trsm` correctness on Trainium (blocked path)."""

    pytestmark = pytest.mark.neuron

    def test_lower_nontrans(self, nki_backend, tri_shapes, rng):
        for M, N in tri_shapes:
            A = _spd(M, rng)
            L = torch.linalg.cholesky(A)
            B = torch.randn(M, N, generator=rng)
            X = nki_trsm(L, B, uplo="lower")
            torch.testing.assert_close(L @ X, B, atol=ATOL, rtol=RTOL)

    def test_lower_trans(self, nki_backend, tri_shapes, rng):
        for M, N in tri_shapes:
            A = _spd(M, rng)
            L = torch.linalg.cholesky(A)
            B = torch.randn(M, N, generator=rng)
            X = nki_trsm(L, B, uplo="lower", trans=True)
            torch.testing.assert_close(L.T @ X, B, atol=ATOL, rtol=RTOL)

    def test_upper_nontrans(self, nki_backend, tri_shapes, rng):
        for M, N in tri_shapes:
            A = _spd(M, rng)
            U = torch.linalg.cholesky(A).T
            B = torch.randn(M, N, generator=rng)
            X = nki_trsm(U, B, uplo="upper")
            torch.testing.assert_close(U @ X, B, atol=ATOL, rtol=RTOL)

    def test_upper_trans(self, nki_backend, rng):
        A = _spd(256, rng)
        U = torch.linalg.cholesky(A).T
        B = torch.randn(256, 128, generator=rng)
        X = nki_trsm(U, B, uplo="upper", trans=True)
        torch.testing.assert_close(U.T @ X, B, atol=ATOL, rtol=RTOL)

    def test_unit_diag(self, nki_backend, rng):
        L = torch.eye(256) + 0.05 * torch.tril(
            torch.randn(256, 256, generator=rng), diagonal=-1
        )
        B = torch.randn(256, 128, generator=rng)
        X = nki_trsm(L, B, uplo="lower", diag="unit")
        torch.testing.assert_close(L @ X, B, atol=ATOL, rtol=RTOL)

    def test_alpha(self, nki_backend, rng):
        A = _spd(256, rng)
        L = torch.linalg.cholesky(A)
        B = torch.randn(256, 128, generator=rng)
        X = nki_trsm(L, B, uplo="lower", alpha=2.5)
        torch.testing.assert_close(L @ X, 2.5 * B, atol=ATOL, rtol=RTOL)


class TestTrsmDispatch:
    """Top-level `trsm()` BLAS call routes through nki_trsm on Trainium."""

    pytestmark = pytest.mark.neuron

    def test_df_mp2_shape(self, nki_backend, rng):
        """The exact call pattern from examples/df_mp2.py step 1."""
        naux = 384
        J = _spd(naux, rng)
        L = torch.linalg.cholesky(J)
        I = torch.eye(naux)
        X = trsm(1.0, L, I, uplo="lower", trans=True)
        # X should be L^{-T}, so L.T @ X = I.
        torch.testing.assert_close(L.T @ X, I, atol=ATOL, rtol=RTOL)
