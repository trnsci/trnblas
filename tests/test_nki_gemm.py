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
from trnblas.nki import nki_batched_gemm, nki_gemm

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
