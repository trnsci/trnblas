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


class TestMp2EnergySimulator:
    """M2 correctness fix — previously re-skipped for partition-broadcast.

    Mirrors the 5 hardware cases in tests/test_nki_mp2_energy.py::TestNkiKernel
    across the `nvir ∈ {8, 16, 64, 256, 448}` sweep that represents the
    DF-MP2 bench shapes. Simulator catches the Python-trace-level class of
    errors (bad kwargs, shape mismatches) but NOT MLIR partition-dim
    broadcast verification — that gate is still hardware-only.
    """

    @staticmethod
    def _make_inputs(nocc, nvir, naux, ic=None, seed=0):
        if ic is None:
            ic = nocc
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(nocc, nvir, naux, generator=g)
        eps_occ = torch.linspace(-1.0, -0.2, nocc)
        eps_vir = torch.linspace(0.2, 1.0, nvir)
        B_flat = B.reshape(nocc * nvir, naux)
        T_flat = B_flat[: ic * nvir] @ B_flat.T
        return T_flat, eps_occ[:ic], eps_occ, eps_vir

    def _run(self, nocc, nvir, naux, ic=None, atol=1e-3, rtol=1e-3):
        from trnblas.nki import nki_mp2_energy
        from trnblas.nki.dispatch import _torch_mp2_energy

        T_flat, eo_c, eo_f, ev = self._make_inputs(nocc, nvir, naux, ic=ic)
        got = nki_mp2_energy(T_flat, eo_c, eo_f, ev)
        ref = _torch_mp2_energy(T_flat, eo_c, eo_f, ev)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

    def test_small_nvir8(self):
        self._run(nocc=4, nvir=8, naux=16)

    def test_chunked_nvir16(self):
        self._run(nocc=8, nvir=16, naux=32, ic=3)

    def test_nvir64_single_strip(self):
        """NVIR=64 ≤128: P_TILE=64, NSTRIP=1 — single-strip path."""
        self._run(nocc=8, nvir=64, naux=32)

    def test_nvir256_two_strips(self):
        """NVIR=256: P_TILE=128, NSTRIP=2 — exercises multi-strip loop."""
        self._run(nocc=2, nvir=256, naux=8)

    def test_nvir448_bench_shape(self):
        """NVIR=448 (medium DF-MP2): P_TILE=112, NSTRIP=4."""
        self._run(nocc=2, nvir=448, naux=8, atol=1e-2, rtol=1e-3)


class TestFusedGemmEnergySimulator:
    """Simulator correctness tests for _fused_gemm_energy_kernel (#38, v0.5.1).

    Small shapes only — the simulator is slow at large NVIR/NAUX.  Shapes
    are restricted to multiples of TILE=128 here (the kernel requires this
    post-padding); the padding logic is tested on hardware via
    TestFusedGemmEnergy in test_nki_gemm.py.
    """

    def _ref(self, b_i, b_j, eps_occ_i, eps_occ_j, eps_vir):
        T = b_i @ b_j.T
        denom = eps_occ_i + eps_occ_j - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)
        return (T * (2.0 * T - T.T) / denom).sum()

    def _run(self, nvir, naux, atol=1e-2, rtol=1e-3):
        import torch

        from trnblas.nki import nki_fused_gemm_energy

        torch.manual_seed(42)
        b_i = torch.randn(nvir, naux)
        b_j = torch.randn(nvir, naux)
        eps_occ_i = 1.5
        eps_occ_j = 1.2
        eps_vir = torch.rand(nvir) * 0.5 + 0.1  # keep denom > 0
        got = nki_fused_gemm_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
        ref = self._ref(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

    def test_single_tile(self):
        """NVIR=128, NAUX=128: one (a, b) tile — simplest case."""
        self._run(nvir=128, naux=128)

    def test_two_a_tiles(self):
        """NVIR=256: two a-strips, exercises acc_b batching."""
        self._run(nvir=256, naux=128)

    def test_two_k_tiles(self):
        """NAUX=256: two TILE_K strips in the GEMM k-loop."""
        self._run(nvir=128, naux=256)
