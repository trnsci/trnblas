"""On-hardware validation for the NKI GEMM kernel.

These tests are skipped on CPU (require `neuronxcc` + a Trainium device).
Run on a trn1/trn2 instance via:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh

Each test forces the `nki` backend and compares against the PyTorch
reference. The kernel uses stationary tile reuse with TILE=128 — the shape
sweep deliberately covers both 128-aligned and unaligned dimensions to
exercise edge-tile handling.
"""

import pytest
import torch

from trnblas import gemm
from trnblas.nki import nki_gemm


pytestmark = pytest.mark.neuron


# Tolerance: FP32 matmul accumulates O(K) rounding errors. Use 1e-3
# absolute (matches trnfft) and a generous relative for very small values.
ATOL = 1e-3
RTOL = 1e-4


@pytest.fixture
def aligned_shapes():
    """Square + rectangular shapes where every dim is a multiple of 128."""
    return [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (256, 128, 512),
        (1024, 256, 128),
    ]


@pytest.fixture
def edge_shapes():
    """Shapes with at least one non-128-aligned dim — exercises edge tiles."""
    return [
        (200, 137, 400),     # all unaligned
        (256, 200, 128),     # N unaligned
        (137, 128, 256),     # M unaligned
        (128, 128, 200),     # K unaligned
        (300, 250, 175),     # all unaligned, prime-ish
    ]


def _check(A, B, ref):
    out = nki_gemm(A, B)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


class TestNkiGemmKernel:
    """Direct kernel correctness — bypasses the BLAS-shaped wrapper."""

    def test_aligned_shapes(self, nki_backend, aligned_shapes, rng):
        for M, K, N in aligned_shapes:
            A = torch.randn(M, K, generator=rng)
            B = torch.randn(K, N, generator=rng)
            _check(A, B, A @ B)

    def test_edge_shapes(self, nki_backend, edge_shapes, rng):
        for M, K, N in edge_shapes:
            A = torch.randn(M, K, generator=rng)
            B = torch.randn(K, N, generator=rng)
            _check(A, B, A @ B)

    def test_identity(self, nki_backend, rng):
        I = torch.eye(256)
        B = torch.randn(256, 128, generator=rng)
        _check(I, B, B)

    def test_zero(self, nki_backend):
        A = torch.zeros(128, 128)
        B = torch.randn(128, 128)
        out = nki_gemm(A, B)
        torch.testing.assert_close(out, torch.zeros(128, 128), atol=0, rtol=0)


class TestGemmDispatch:
    """Top-level `gemm()` BLAS call routes through the NKI kernel."""

    def test_basic(self, nki_backend, rng):
        A = torch.randn(256, 128, generator=rng)
        B = torch.randn(128, 256, generator=rng)
        out = gemm(1.0, A, B)
        torch.testing.assert_close(out, A @ B, atol=ATOL, rtol=RTOL)

    def test_transA(self, nki_backend, rng):
        A = torch.randn(128, 256, generator=rng)
        B = torch.randn(128, 256, generator=rng)
        out = gemm(1.0, A, B, transA=True)
        torch.testing.assert_close(out, A.T @ B, atol=ATOL, rtol=RTOL)

    def test_transB(self, nki_backend, rng):
        A = torch.randn(256, 128, generator=rng)
        B = torch.randn(256, 128, generator=rng)
        out = gemm(1.0, A, B, transB=True)
        torch.testing.assert_close(out, A @ B.T, atol=ATOL, rtol=RTOL)

    def test_alpha_beta(self, nki_backend, rng):
        A = torch.randn(128, 128, generator=rng)
        B = torch.randn(128, 128, generator=rng)
        C0 = torch.randn(128, 128, generator=rng)
        out = gemm(2.0, A, B, beta=0.5, C=C0.clone())
        torch.testing.assert_close(out, 2.0 * (A @ B) + 0.5 * C0,
                                   atol=ATOL, rtol=RTOL)


class TestStationaryTileReuse:
    """The kernel's value prop: A loaded once, reused across many B tiles.

    Run the same A against several B's back-to-back. Numerics must match
    independent calls — verifies the stationary-tile path doesn't smear
    state between dispatches.
    """

    def test_shared_A_across_calls(self, nki_backend, rng):
        A = torch.randn(256, 256, generator=rng)
        Bs = [torch.randn(256, 128, generator=rng) for _ in range(4)]
        outs = [nki_gemm(A, B) for B in Bs]
        for B, out in zip(Bs, outs):
            torch.testing.assert_close(out, A @ B, atol=ATOL, rtol=RTOL)
