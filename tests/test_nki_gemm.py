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

import trnblas
from trnblas import batched_gemm, gemm
from trnblas.nki import nki_batched_gemm, nki_batched_pair_energy, nki_fused_gemm_energy, nki_gemm

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
        (200, 137, 400),  # all unaligned
        (256, 200, 128),  # N unaligned
        (137, 128, 256),  # M unaligned
        (128, 128, 200),  # K unaligned
        (300, 250, 175),  # all unaligned, prime-ish
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
        torch.testing.assert_close(out, 2.0 * (A @ B) + 0.5 * C0, atol=ATOL, rtol=RTOL)


class TestNkiBatchedGemm:
    """`batched_gemm` dispatches per-slice through the cached 2D kernel."""

    @pytest.mark.parametrize("batch", [4, 16, 32])
    def test_vs_torch(self, nki_backend, batch, rng):
        A = torch.randn(batch, 256, 128, generator=rng)
        B = torch.randn(batch, 128, 256, generator=rng)
        out = batched_gemm(1.0, A, B)
        torch.testing.assert_close(out, torch.bmm(A, B), atol=ATOL, rtol=RTOL)

    def test_edge_shapes(self, nki_backend, rng):
        A = torch.randn(8, 200, 137, generator=rng)
        B = torch.randn(8, 137, 400, generator=rng)
        out = batched_gemm(1.0, A, B)
        torch.testing.assert_close(out, torch.bmm(A, B), atol=ATOL, rtol=RTOL)

    def test_alpha_beta(self, nki_backend, rng):
        A = torch.randn(4, 256, 128, generator=rng)
        B = torch.randn(4, 128, 256, generator=rng)
        C0 = torch.randn(4, 256, 256, generator=rng)
        out = batched_gemm(2.0, A, B, beta=0.5, C=C0.clone())
        ref = 2.0 * torch.bmm(A, B) + 0.5 * C0
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


class TestPerformance:
    """Cold-cache (compile) vs warm-cache (NEFF reuse) timing.

    Run with `pytest -v -s` so the perf prints surface. The instance-wide
    NEFF cache at /var/tmp/neuron-compile-cache/ persists across pytest
    processes, so the second invocation of the script (via --warm) sees
    even the first call hit the cache.
    """

    @pytest.mark.parametrize("MKN", [(512, 512, 512), (1024, 1024, 1024)])
    def test_compile_vs_cache_timing(self, nki_backend, MKN, capsys):
        import time

        M, K, N = MKN
        A = torch.randn(M, K)
        B = torch.randn(K, N)

        # First call: may compile (cold cache).
        t0 = time.perf_counter()
        nki_gemm(A, B)
        cold = time.perf_counter() - t0

        # Subsequent calls: should hit cached NEFF + warm XLA graph.
        warm_times = []
        for _ in range(5):
            t = time.perf_counter()
            nki_gemm(A, B)
            warm_times.append(time.perf_counter() - t)
        warm_min = min(warm_times)
        warm_avg = sum(warm_times) / len(warm_times)

        with capsys.disabled():
            print(
                f"\n[perf {M}x{K}x{N}] "
                f"cold={cold * 1000:7.1f}ms  "
                f"warm_min={warm_min * 1000:7.1f}ms  "
                f"warm_avg={warm_avg * 1000:7.1f}ms  "
                f"speedup={cold / warm_min:5.1f}x"
            )

    @pytest.mark.parametrize("BMKN", [(32, 256, 128, 256)])
    def test_batched_compile_vs_cache(self, nki_backend, BMKN, capsys):
        import time

        B, M, K, N = BMKN
        A = torch.randn(B, M, K)
        Bm = torch.randn(B, K, N)
        t0 = time.perf_counter()
        batched_gemm(1.0, A, Bm)
        cold = time.perf_counter() - t0
        warm_times = []
        for _ in range(3):
            t = time.perf_counter()
            batched_gemm(1.0, A, Bm)
            warm_times.append(time.perf_counter() - t)
        warm_min = min(warm_times)
        with capsys.disabled():
            print(
                f"\n[perf batch={B} {M}x{K}x{N}] "
                f"cold={cold * 1000:7.1f}ms  "
                f"warm_min={warm_min * 1000:7.1f}ms  "
                f"per_slice={warm_min * 1000 / B:6.2f}ms"
            )


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
        for B, out in zip(Bs, outs, strict=True):
            torch.testing.assert_close(out, A @ B, atol=ATOL, rtol=RTOL)


class TestAutotuner:
    """GEMM tile-shape autotuner (#26, milestone v0.5.0).

    CPU-side tests (cache_hit, escape_hatch, persistent_cache) use monkeypatching
    to exercise the autotuner logic without hardware dispatch overhead.
    test_correctness_per_config dispatches all six tile configs on hardware.
    test_hardware_sweep verifies the full end-to-end sweep path on trn1.
    """

    def test_escape_hatch(self, nki_backend, rng, monkeypatch):
        """TRNBLAS_AUTOTUNE=0 bypasses sweep and uses default tile config."""
        import trnblas.nki.dispatch as D

        monkeypatch.setattr(D, "_AUTOTUNE_ENABLED", False)
        A = torch.randn(256, 128, generator=rng)
        B = torch.randn(128, 256, generator=rng)
        out = nki_gemm(A, B)
        torch.testing.assert_close(out, A @ B, atol=ATOL, rtol=RTOL)

    def test_cache_hit(self, nki_backend, rng, monkeypatch):
        """Same shape bucket does not trigger a second sweep."""
        import trnblas.nki.dispatch as D

        sweep_calls: list[tuple[int, int, int]] = []
        original_sweep = D._sweep_on_default_pad

        def counting_sweep(M, K, N, A, B):
            sweep_calls.append((M, K, N))
            return original_sweep(M, K, N, A, B)

        monkeypatch.setattr(D, "_autotune_mem", {})
        monkeypatch.setattr(D, "_autotune_loaded", True)  # skip disk load
        monkeypatch.setattr(D, "_sweep_on_default_pad", counting_sweep)

        A = torch.randn(256, 128, generator=rng)
        B = torch.randn(128, 256, generator=rng)
        nki_gemm(A, B)  # first call: sweep fires
        assert len(sweep_calls) == 1
        nki_gemm(A, B)  # second call: bucket in mem, no re-sweep
        assert len(sweep_calls) == 1

    def test_persistent_cache(self, nki_backend, rng, tmp_path, monkeypatch):
        """Winning tile config survives a process restart via JSON cache."""
        import trnblas.nki.dispatch as D

        cache_file = tmp_path / "autotune_cache.json"
        monkeypatch.setattr(D, "_AUTOTUNE_CACHE_FILE", cache_file)
        monkeypatch.setattr(D, "_autotune_mem", {})
        monkeypatch.setattr(D, "_autotune_loaded", False)

        A = torch.randn(256, 128, generator=rng)
        B = torch.randn(128, 256, generator=rng)
        nki_gemm(A, B)  # triggers sweep + write

        assert cache_file.exists(), "cache file was not written"
        # Simulate new process: clear in-process state, reload from disk.
        monkeypatch.setattr(D, "_autotune_mem", {})
        monkeypatch.setattr(D, "_autotune_loaded", False)
        D._load_autotune_cache()
        bucket = D._autotune_bucket(256, 128, 256)
        assert bucket in D._autotune_mem, "bucket not restored from cache"
        tile_m, tile_k, tile_n = D._autotune_mem[bucket]
        assert (tile_m, tile_k, tile_n) in D._TILE_CANDIDATES

    def test_correctness_per_config(self, nki_backend, rng):
        """Every tile candidate that divides (128,128,512) produces correct output."""
        from trnblas.nki.dispatch import _TILE_CANDIDATES, _get_gemm_kernel, _to_xla

        M, K, N = 128, 128, 512
        A = torch.randn(M, K, generator=rng)
        B = torch.randn(K, N, generator=rng)
        ref = A @ B
        for tm, tk, tn in _TILE_CANDIDATES:
            if M % tm or K % tk or N % tn:
                continue
            kernel = _get_gemm_kernel(tm, tk, tn)
            (a, b), orig = _to_xla(A.contiguous(), B.contiguous())
            result = kernel(a, b).to(orig)
            torch.testing.assert_close(
                result,
                ref,
                atol=ATOL,
                rtol=RTOL,
                msg=f"tile config ({tm},{tk},{tn}) mismatch",
            )

    def test_hardware_sweep(self, nki_backend, rng, tmp_path, capsys, monkeypatch):
        """Full sweep on hardware: result correct; winner is a valid candidate."""
        import trnblas.nki.dispatch as D

        monkeypatch.setattr(D, "_autotune_mem", {})
        monkeypatch.setattr(D, "_autotune_loaded", True)
        monkeypatch.setattr(D, "_AUTOTUNE_CACHE_FILE", tmp_path / "hw_sweep.json")

        M, K, N = 512, 128, 512
        A = torch.randn(M, K, generator=rng)
        B = torch.randn(K, N, generator=rng)
        ref = A @ B

        out = nki_gemm(A, B)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

        bucket = D._autotune_bucket(M, K, N)
        assert bucket in D._autotune_mem, "winner not stored in _autotune_mem"
        winner = D._autotune_mem[bucket]
        assert winner in D._TILE_CANDIDATES, f"winner {winner} not in candidates"
        with capsys.disabled():
            print(f"\n[autotuner] shape ({M},{K},{N}) → bucket {bucket} → winner {winner}")


class TestFusedGemmEnergy:
    """Fused GEMM+energy kernel (#38, v0.5.1).

    These tests exercise `nki_fused_gemm_energy` — one DF-MP2 pair's
    fused (B_i @ B_j.T) + energy expression in a single @nki.jit.

    Structure mirrors TestNkiKernel: correctness tests compare against
    _torch_fused_pair_energy; performance tests measure cold vs warm
    NEFF cache timing.
    """

    # Loose tolerance: FP32 GEMM accumulation + reduction chain.
    ATOL = 1e-2
    RTOL = 1e-3

    def _ref(self, b_i, b_j, eps_occ_i, eps_occ_j, eps_vir):
        """PyTorch reference: T = B_i @ B_j.T, energy = sum T*(2T-T.T)/denom."""
        T = b_i @ b_j.T
        denom = eps_occ_i + eps_occ_j - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)
        return (T * (2.0 * T - T.T) / denom).sum()

    @pytest.mark.parametrize(
        "nvir,naux",
        [
            (128, 128),  # single tile
            (256, 256),  # 2×2 tile grid
            (512, 512),  # 4×4 tile grid
            (256, 128),  # naux = one TILE_K
        ],
    )
    def test_correctness_aligned(self, nki_backend, nvir, naux, rng):
        """Aligned shapes: fused result matches torch reference."""
        b_i = torch.randn(nvir, naux, generator=rng)
        b_j = torch.randn(nvir, naux, generator=rng)
        eps_occ_i = float(torch.rand(1, generator=rng) + 1.0)
        eps_occ_j = float(torch.rand(1, generator=rng) + 1.0)
        eps_vir = torch.rand(nvir, generator=rng) * 0.5  # keep denom > 0
        ref = self._ref(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
        out = nki_fused_gemm_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
        torch.testing.assert_close(out, ref, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "nvir,naux",
        [
            (200, 137),  # all unaligned
            (256, 200),  # naux unaligned
            (144, 128),  # nvir unaligned
        ],
    )
    def test_correctness_unaligned(self, nki_backend, nvir, naux, rng):
        """Unaligned shapes: padding logic keeps result correct."""
        b_i = torch.randn(nvir, naux, generator=rng)
        b_j = torch.randn(nvir, naux, generator=rng)
        eps_occ_i = float(torch.rand(1, generator=rng) + 1.0)
        eps_occ_j = float(torch.rand(1, generator=rng) + 1.0)
        eps_vir = torch.rand(nvir, generator=rng) * 0.5
        ref = self._ref(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
        out = nki_fused_gemm_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
        torch.testing.assert_close(out, ref, atol=self.ATOL, rtol=self.RTOL)

    def test_symmetry(self, nki_backend, rng):
        """Pair energy is symmetric: E(i,j) == E(j,i)."""
        nvir, naux = 256, 256
        b_i = torch.randn(nvir, naux, generator=rng)
        b_j = torch.randn(nvir, naux, generator=rng)
        eps_occ_i = 1.2
        eps_occ_j = 0.8
        eps_vir = torch.rand(nvir, generator=rng) * 0.5
        e_ij = nki_fused_gemm_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
        e_ji = nki_fused_gemm_energy(b_j, b_i, eps_occ_j, eps_occ_i, eps_vir)
        torch.testing.assert_close(e_ij, e_ji, atol=self.ATOL, rtol=self.RTOL)

    def test_zero_b_i(self, nki_backend):
        """B_i = 0 → T = 0 → energy = 0."""
        nvir, naux = 128, 128
        b_i = torch.zeros(nvir, naux)
        b_j = torch.randn(nvir, naux)
        eps_vir = torch.rand(nvir) * 0.5
        out = nki_fused_gemm_energy(b_i, b_j, 1.0, 1.0, eps_vir)
        torch.testing.assert_close(out, torch.tensor(0.0), atol=0, rtol=0)

    def test_neff_cache_reuse(self, nki_backend, rng, capsys):
        """Second pair invocation with same shape hits NEFF cache (warm << cold)."""
        import time

        nvir, naux = 256, 256
        b_i = torch.randn(nvir, naux, generator=rng)
        b_j = torch.randn(nvir, naux, generator=rng)
        eps_vir = torch.rand(nvir, generator=rng) * 0.5

        t0 = time.perf_counter()
        nki_fused_gemm_energy(b_i, b_j, 1.0, 0.9, eps_vir)
        cold = time.perf_counter() - t0

        warm_times = []
        for _ in range(4):
            t = time.perf_counter()
            nki_fused_gemm_energy(b_i, b_j, 1.0, 0.9, eps_vir)
            warm_times.append(time.perf_counter() - t)
        warm_min = min(warm_times)

        with capsys.disabled():
            print(
                f"\n[fused_gemm_energy {nvir}×{naux}] "
                f"cold={cold * 1000:7.1f}ms  "
                f"warm_min={warm_min * 1000:7.1f}ms  "
                f"speedup={cold / warm_min:.1f}x"
            )


class TestBatchedPairEnergy:
    """Batched DF-MP2 pair energy kernel (#43, milestone v0.5.2).

    `nki_batched_pair_energy(B, eps_occ, eps_vir)` computes the full DF-MP2
    pair energy for all NOCC² orbital pairs in a single @nki.jit dispatch,
    eliminating the ~100ms × nocc² per-dispatch overhead of the per-pair loop.

    3D NKI indexing (B[i, a_off:, k_off:] with affine_range i) was validated
    on trn1 via the Spike A / Spike B scripts on 2026-04-17.
    """

    # Loose FP32 tolerance — the kernel accumulates across nocc² pairs.
    ATOL = 1e-2
    RTOL = 1e-3

    def _ref(self, B, eps_occ, eps_vir):
        """PyTorch reference: explicit nocc × nocc loop."""
        nocc = B.shape[0]
        e = torch.zeros((), dtype=B.dtype)
        for i in range(nocc):
            for j in range(nocc):
                T = B[i] @ B[j].T
                denom = (
                    float(eps_occ[i])
                    + float(eps_occ[j])
                    - eps_vir.unsqueeze(1)
                    - eps_vir.unsqueeze(0)
                )
                e = e + (T * (2.0 * T - T.T) / denom).sum()
        return e

    @pytest.mark.parametrize(
        "nocc,nvir,naux",
        [
            (4, 128, 128),  # minimal aligned
            (4, 256, 256),  # 2× tile grid
            (8, 128, 128),  # larger nocc, single tile
            (4, 256, 128),  # naux = one TILE_K
        ],
    )
    def test_correctness_aligned(self, nki_backend, nocc, nvir, naux, rng):
        """Aligned shapes: batched result matches PyTorch reference."""
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5
        ref = self._ref(B, eps_occ, eps_vir)
        out = nki_batched_pair_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(out, ref, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "nocc,nvir,naux",
        [
            (4, 200, 137),  # all unaligned
            (4, 256, 200),  # naux unaligned
            (4, 144, 128),  # nvir unaligned
        ],
    )
    def test_correctness_unaligned(self, nki_backend, nocc, nvir, naux, rng):
        """Unaligned shapes: padding logic keeps result correct."""
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5
        ref = self._ref(B, eps_occ, eps_vir)
        out = nki_batched_pair_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(out, ref, atol=self.ATOL, rtol=self.RTOL)

    def test_matches_fused_gemm_energy(self, nki_backend, rng):
        """Batched result matches per-pair nki_fused_gemm_energy sum."""
        nocc, nvir, naux = 4, 256, 256
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5

        # per-pair reference
        e_ref = torch.zeros(())
        for i in range(nocc):
            for j in range(nocc):
                e_ref = e_ref + nki_fused_gemm_energy(
                    B[i], B[j], float(eps_occ[i]), float(eps_occ[j]), eps_vir
                )

        out = nki_batched_pair_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(out, e_ref, atol=self.ATOL, rtol=self.RTOL)

    def test_zero_B(self, nki_backend):
        """B = 0 → all T = 0 → energy = 0."""
        nocc, nvir, naux = 4, 128, 128
        B = torch.zeros(nocc, nvir, naux)
        eps_occ = torch.ones(nocc) * 2.0
        eps_vir = torch.ones(nvir) * 0.5
        out = nki_batched_pair_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(out, torch.tensor(0.0), atol=0, rtol=0)

    def test_dispatch_overhead(self, nki_backend, rng, capsys):
        """Single dispatch covers all nocc² pairs; warm time << per-pair baseline.

        Reports cold/warm times for the batched kernel and the per-pair loop
        so the overhead reduction from #43 can be observed in CI output.
        """
        import time

        nocc, nvir, naux = 4, 256, 256
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5

        # batched: single dispatch
        t0 = time.perf_counter()
        nki_batched_pair_energy(B, eps_occ, eps_vir)
        cold_batched = time.perf_counter() - t0

        warm_times = []
        for _ in range(4):
            t = time.perf_counter()
            nki_batched_pair_energy(B, eps_occ, eps_vir)
            warm_times.append(time.perf_counter() - t)
        warm_batched = min(warm_times)

        # per-pair: nocc² dispatches (warm — NEFF cached from above run)
        t0 = time.perf_counter()
        for i in range(nocc):
            for j in range(nocc):
                nki_fused_gemm_energy(B[i], B[j], float(eps_occ[i]), float(eps_occ[j]), eps_vir)
        t_per_pair = time.perf_counter() - t0

        with capsys.disabled():
            print(
                f"\n[batched_pair nocc={nocc} nvir={nvir} naux={naux}] "
                f"cold={cold_batched * 1000:7.1f}ms  "
                f"warm={warm_batched * 1000:7.1f}ms  "
                f"per-pair-loop={t_per_pair * 1000:7.1f}ms  "
                f"speedup={t_per_pair / warm_batched:.1f}x"
            )

    # --- Chunked dispatch tests (#46) ---

    @pytest.mark.parametrize(
        "nocc,nvir,naux",
        [
            # est_iters = nocc² × ceil(nvir/128)² × ceil(naux/128)
            (16, 384, 512),  # 256 × 3 × 3 × 4 = 9216  > 4096 → chunked
            (16, 256, 640),  # 256 × 2 × 2 × 5 = 5120  > 4096 → chunked
            (16, 256, 384),  # 256 × 2 × 2 × 3 = 3072  ≤ 4096 → full-batch
        ],
    )
    def test_correctness_chunked_path(self, nki_backend, nocc, nvir, naux, rng):
        """Chunked dispatch matches torch reference for shapes above/below threshold."""
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5
        ref = self._ref(B, eps_occ, eps_vir)
        out = nki_batched_pair_energy(B, eps_occ, eps_vir)
        torch.testing.assert_close(out, ref, atol=self.ATOL, rtol=self.RTOL)

    def test_chunked_and_full_batch_agree(self, nki_backend, rng):
        """Forcing chunked path on a small shape agrees with full-batch path."""
        import trnblas.nki.dispatch as D

        nocc, nvir, naux = 4, 256, 256  # est_iters = 16 × 2 × 2 × 2 = 128 (full-batch)
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5

        # Default routing → full-batch kernel
        out_full = nki_batched_pair_energy(B, eps_occ, eps_vir)

        # Force chunked by setting threshold to 0
        orig_thresh = D._BATCHED_PAIR_CHUNK_THRESHOLD
        try:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = 0
            out_chunked = nki_batched_pair_energy(B, eps_occ, eps_vir)
        finally:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = orig_thresh

        torch.testing.assert_close(out_chunked, out_full, atol=self.ATOL, rtol=self.RTOL)

    def test_chunked_dispatch_overhead(self, nki_backend, rng, capsys):
        """Chunked path timing: NOCC dispatches vs per-pair loop baseline.

        Uses a shape that routes to the chunked path (est_iters > 4096).
        Reports cold/warm times and compares against the per-pair loop.
        """
        import time

        import trnblas.nki.dispatch as D

        nocc, nvir, naux = 16, 384, 512  # est_iters = 9216 → chunked
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5

        # Confirm routing: must use chunked impl
        _N_A = D._round_up(nvir, 128) // 128
        _N_K = D._round_up(naux, 128) // 128
        est = (nocc * nocc) * _N_A * _N_A * _N_K
        assert est > D._BATCHED_PAIR_CHUNK_THRESHOLD, (
            f"Shape should route to chunked (est_iters={est})"
        )

        t0 = time.perf_counter()
        nki_batched_pair_energy(B, eps_occ, eps_vir)
        cold_chunked = time.perf_counter() - t0

        warm_times = []
        for _ in range(3):
            t = time.perf_counter()
            nki_batched_pair_energy(B, eps_occ, eps_vir)
            warm_times.append(time.perf_counter() - t)
        warm_chunked = min(warm_times)

        # per-pair loop baseline (NEFF warm from above)
        t0 = time.perf_counter()
        for i in range(nocc):
            for j in range(nocc):
                nki_fused_gemm_energy(B[i], B[j], float(eps_occ[i]), float(eps_occ[j]), eps_vir)
        t_per_pair = time.perf_counter() - t0

        with capsys.disabled():
            print(
                f"\n[chunked nocc={nocc} nvir={nvir} naux={naux} "
                f"dispatches={nocc}] "
                f"cold={cold_chunked * 1000:7.1f}ms  "
                f"warm={warm_chunked * 1000:7.1f}ms  "
                f"per-pair-loop={t_per_pair * 1000:7.1f}ms  "
                f"speedup={t_per_pair / warm_chunked:.1f}x"
            )

    def test_j_sub_chunked_agrees_with_full_j(self, nki_backend, rng):
        """J-sub-chunking produces the same result as the full-j path.

        Forces j-sub-chunking by setting _J_CHUNK_MAX_B_BYTES = 0 (any B triggers
        the sub-chunk path).  Verifies the two paths agree to FP32 noise.
        """
        import trnblas.nki.dispatch as D

        nocc, nvir, naux = 8, 256, 256
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5

        # Must be on the chunked i-path for j-sub-chunking to engage.
        # Force it by zeroing _BATCHED_PAIR_CHUNK_THRESHOLD as well.
        orig_i_thresh = D._BATCHED_PAIR_CHUNK_THRESHOLD
        orig_j_thresh = D._J_CHUNK_MAX_B_BYTES
        try:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = 0  # route to chunked impl
            D._J_CHUNK_MAX_B_BYTES = 0  # force j-sub-chunking with j_chunk=1
            out_j_chunked = nki_batched_pair_energy(B, eps_occ, eps_vir)
        finally:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = orig_i_thresh
            D._J_CHUNK_MAX_B_BYTES = orig_j_thresh

        # Reference: full-j chunked path (normal thresholds, but still on chunked impl)
        orig_i_thresh2 = D._BATCHED_PAIR_CHUNK_THRESHOLD
        try:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = 0  # keep on chunked impl, j full
            out_full_j = nki_batched_pair_energy(B, eps_occ, eps_vir)
        finally:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = orig_i_thresh2

        torch.testing.assert_close(out_j_chunked, out_full_j, atol=self.ATOL, rtol=self.RTOL)

    def test_j_sub_chunked_matches_torch_ref(self, nki_backend, rng):
        """J-sub-chunking with nocc padding matches the PyTorch reference.

        nocc=7 with j_chunk_size=2 → nocc_padded=8 (1 zero-pad row),
        exercising the nocc_padded > nocc branch.  Padded row has B=0,
        so energy contribution is 0 and the result must match the 7-pair ref.
        """
        import trnblas.nki.dispatch as D

        nocc, nvir, naux = 7, 128, 128
        B = torch.randn(nocc, nvir, naux, generator=rng) * 0.1
        eps_occ = torch.rand(nocc, generator=rng) * 0.5 + 2.0
        eps_vir = torch.rand(nvir, generator=rng) * 0.5
        ref = self._ref(B, eps_occ, eps_vir)

        orig_i_thresh = D._BATCHED_PAIR_CHUNK_THRESHOLD
        orig_j_thresh = D._J_CHUNK_MAX_B_BYTES
        try:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = 0
            # threshold = 2 rows → j_chunk_size=2; nocc=7 → nocc_padded=8 (1 padding row)
            D._J_CHUNK_MAX_B_BYTES = nvir * naux * 4 * 2
            out = nki_batched_pair_energy(B, eps_occ, eps_vir)
        finally:
            D._BATCHED_PAIR_CHUNK_THRESHOLD = orig_i_thresh
            D._J_CHUNK_MAX_B_BYTES = orig_j_thresh

        torch.testing.assert_close(out, ref, atol=self.ATOL, rtol=self.RTOL)
