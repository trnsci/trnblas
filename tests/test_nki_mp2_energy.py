"""Tests for the fused MP2 energy-reduction kernel (#15).

CPU path (torch fallback) is always exercised; the NKI path is
marked `@pytest.mark.neuron` and runs only on Trainium hardware.
"""

import pytest
import torch

from trnblas.nki import nki_mp2_energy
from trnblas.nki.dispatch import _torch_mp2_energy


ATOL = 1e-4
RTOL = 1e-4


def _make_inputs(nocc, nvir, naux, ic=None, seed=0):
    """Synthetic B, eps_occ, eps_vir and the associated T_flat chunk."""
    if ic is None:
        ic = nocc
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(nocc, nvir, naux, generator=g)
    eps_occ = torch.linspace(-1.0, -0.2, nocc)
    eps_vir = torch.linspace(0.2, 1.0, nvir)
    B_flat = B.reshape(nocc * nvir, naux)
    T_flat = B_flat[: ic * nvir] @ B_flat.T
    return T_flat, eps_occ[:ic], eps_occ, eps_vir


def _reference(T_flat, eps_occ_chunk, eps_occ_full, eps_vir):
    """Independent reference using direct einsum-style expression."""
    ic = eps_occ_chunk.shape[0]
    nocc = eps_occ_full.shape[0]
    nvir = eps_vir.shape[0]
    T = T_flat.reshape(ic, nvir, nocc, nvir).permute(0, 2, 1, 3)
    denom = (
        eps_occ_chunk.view(ic, 1, 1, 1)
        + eps_occ_full.view(1, nocc, 1, 1)
        - eps_vir.view(1, 1, nvir, 1)
        - eps_vir.view(1, 1, 1, nvir)
    )
    return (T * (2.0 * T - T.transpose(-2, -1)) / denom).sum()


class TestTorchFallback:
    """Exercised on CPU — validates the refactor and the torch path."""

    def test_small(self):
        T_flat, eo_c, eo_f, ev = _make_inputs(nocc=4, nvir=8, naux=16)
        got = _torch_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _reference(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)

    def test_chunked(self):
        T_flat, eo_c, eo_f, ev = _make_inputs(nocc=8, nvir=8, naux=16, ic=3)
        got = _torch_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _reference(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)

    def test_dispatch_wrapper_cpu(self):
        """nki_mp2_energy with pytorch backend forced — must match reference."""
        from trnblas import set_backend, get_backend
        old = get_backend()
        set_backend("pytorch")
        try:
            T_flat, eo_c, eo_f, ev = _make_inputs(nocc=4, nvir=8, naux=16)
            got = nki_mp2_energy(T_flat, eo_c, eo_f, ev)
            ref = _reference(T_flat, eo_c, eo_f, ev)
            torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)
        finally:
            set_backend(old)


class TestNkiKernel:
    """On-hardware validation of the fused NKI kernel."""

    pytestmark = pytest.mark.neuron

    def test_small(self, nki_backend):
        T_flat, eo_c, eo_f, ev = _make_inputs(nocc=4, nvir=8, naux=16)
        got = nki_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _reference(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

    def test_aligned(self, nki_backend):
        # nvir = 64 (≤128, single-strip path)
        T_flat, eo_c, eo_f, ev = _make_inputs(nocc=8, nvir=64, naux=32)
        got = nki_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _reference(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

    def test_chunked(self, nki_backend):
        T_flat, eo_c, eo_f, ev = _make_inputs(nocc=8, nvir=16, naux=32, ic=3)
        got = nki_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _reference(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

    def test_subtiled_multiple_of_128(self, nki_backend):
        """nvir=256 exercises NSTRIP=2 (P_TILE=128)."""
        T_flat, eo_c, eo_f, ev = _make_inputs(nocc=2, nvir=256, naux=8)
        got = nki_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _reference(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)

    def test_subtiled_bench_shape(self, nki_backend):
        """nvir=448 (medium bench): P_TILE=112, NSTRIP=4."""
        T_flat, eo_c, eo_f, ev = _make_inputs(nocc=2, nvir=448, naux=8)
        got = nki_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _reference(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-3)
