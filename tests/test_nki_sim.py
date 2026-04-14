"""Simulator-backed kernel correctness tests (NKI 0.3.0 Stable).

Run with `TRNBLAS_USE_SIMULATOR=1` on any x86_64 Linux host that has
`nki>=0.3.0` installed. Bypasses torch_xla + NEFF compile; routes
kernel dispatch through `nki.simulate(kernel)(np_args)`.

Intentionally curated to small shapes — the CPU simulator is slow at
1024³ and above. Correctness parity with hardware at these scales is
what we're verifying, not perf.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.nki_simulator


@pytest.fixture(autouse=True)
def _simulator_enabled():
    """Skip the whole module if TRNBLAS_USE_SIMULATOR isn't set.

    The marker alone isn't sufficient — users may `pytest -m
    nki_simulator` on a host where nki isn't importable or the env
    var hasn't been set. Fail loudly vs silently falling back.
    """
    if os.environ.get("TRNBLAS_USE_SIMULATOR", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("TRNBLAS_USE_SIMULATOR=1 required")

    from trnblas.nki import HAS_NKI

    if not HAS_NKI:
        pytest.skip("nki package not importable on this host")


class TestGemmSimulator:
    def test_aligned_128(self):
        from trnblas import gemm

        torch.manual_seed(0)
        A = torch.randn(128, 128)
        B = torch.randn(128, 128)
        out = gemm(1.0, A, B)
        torch.testing.assert_close(out, A @ B, atol=1e-3, rtol=1e-4)

    def test_rectangular(self):
        from trnblas import gemm

        torch.manual_seed(0)
        A = torch.randn(256, 128)
        B = torch.randn(128, 256)
        out = gemm(1.0, A, B)
        torch.testing.assert_close(out, A @ B, atol=1e-3, rtol=1e-4)


class TestSyrkSimulator:
    def test_small_square(self):
        from trnblas import syrk

        torch.manual_seed(0)
        A = torch.randn(128, 64)
        out = syrk(1.0, A)
        torch.testing.assert_close(out, A @ A.T, atol=1e-3, rtol=1e-4)


class TestTrsmSimulator:
    """Blocked TRSM uses nki_gemm internally — simulator plumbs through."""

    def test_lower_solve(self):
        from trnblas import trsm

        torch.manual_seed(0)
        n = 128
        A = torch.randn(n, n)
        SPD = A @ A.T + n * torch.eye(n)
        L = torch.linalg.cholesky(SPD)
        B = torch.randn(n, 32)
        X = trsm(1.0, L, B, uplo="lower")
        torch.testing.assert_close(L @ X, B, atol=1e-3, rtol=1e-3)
