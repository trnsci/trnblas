"""
Host results through the caching host allocator, and in-place epilogues (#61).

CPU-only: the pinned allocation is replaced by a recording stand-in (a CPU-only
host has no accelerator to pin for), and the dispatch path is forced to
"native" where a test needs the cached-allocator route.
"""

import pytest
import torch

import trnblas
from trnblas import level3
from trnblas.nki import dispatch as D

MiB = 1 << 20


@pytest.fixture
def native_host_cache(monkeypatch):
    """Pretend we are on the native path; record pinned allocations."""
    monkeypatch.delenv("TRNBLAS_PINNED_OUTPUT", raising=False)
    monkeypatch.setattr(D, "_get_dispatch_path", lambda: "native")
    monkeypatch.setattr(D, "_use_simulator", lambda: False)
    calls = []
    real_empty = torch.empty

    def fake_empty(*args, pin_memory=False, **kw):
        if pin_memory:
            calls.append(args[0])
        return real_empty(*args, **kw)

    monkeypatch.setattr(D.torch, "empty", fake_empty)
    return calls


class TestUseHostCache:
    def test_off_the_native_path(self, monkeypatch):
        monkeypatch.setattr(D, "_get_dispatch_path", lambda: "cpu")
        assert not D._use_host_cache("cpu", 64 * MiB)

    def test_native_large_cpu_destination(self, native_host_cache):
        assert D._use_host_cache("cpu", MiB)
        assert D._use_host_cache(torch.device("cpu"), 64 * MiB)

    def test_small_results_skip(self, native_host_cache):
        assert not D._use_host_cache("cpu", MiB - 1)

    def test_device_destination_skips(self, native_host_cache):
        assert not D._use_host_cache("meta", 64 * MiB)

    @pytest.mark.parametrize("value", ["0", "false", "off", "no"])
    def test_env_disables(self, native_host_cache, monkeypatch, value):
        monkeypatch.setenv("TRNBLAS_PINNED_OUTPUT", value)
        assert not D._use_host_cache("cpu", 64 * MiB)

    def test_simulator_skips(self, native_host_cache, monkeypatch):
        monkeypatch.setattr(D, "_use_simulator", lambda: True)
        assert not D._use_host_cache("cpu", 64 * MiB)


class TestToHost:
    def test_large_result_uses_cache_and_is_fresh(self, native_host_cache):
        src = torch.randn(512, 1024)  # 2 MiB, standing in for a device result
        out = D._host_empty(src.shape, src.dtype, "cpu").copy_(src)
        assert native_host_cache == [(512, 1024)]
        assert torch.equal(out, src)
        assert out.data_ptr() != src.data_ptr()

    def test_cpu_tensor_passes_through(self, native_host_cache):
        t = torch.randn(512, 1024)
        assert D._to_host(t, "cpu") is t
        assert native_host_cache == []

    def test_host_empty_small_is_plain(self, native_host_cache):
        t = D._host_empty((4, 4), torch.float32, "cpu")
        assert t.shape == (4, 4) and native_host_cache == []

    def test_host_empty_like(self, native_host_cache):
        t = D._host_empty_like(torch.empty(1024, 1024, dtype=torch.float64))
        assert t.dtype == torch.float64 and t.is_contiguous()
        assert native_host_cache == [(1024, 1024)]


class TestScaleAdd:
    @pytest.mark.parametrize("alpha", [1.0, 0.5, -2.0])
    @pytest.mark.parametrize("beta", [0.0, 1.0, 0.25])
    def test_matches_out_of_place(self, alpha, beta):
        g = torch.Generator().manual_seed(0)
        r = torch.randn(17, 23, generator=g)
        C = torch.randn(17, 23, generator=g)
        ref = alpha * r + beta * C if beta != 0.0 else alpha * r
        out = level3._scale_add(r.clone(), alpha, beta, C)
        torch.testing.assert_close(out, ref)

    def test_in_place_on_owned_result(self):
        r = torch.ones(8, 8)
        out = level3._scale_add(r, 2.0, 1.0, torch.ones(8, 8))
        assert out.data_ptr() == r.data_ptr()
        assert torch.equal(out, torch.full((8, 8), 3.0))

    def test_broadcast_c(self):
        r, C = torch.ones(4, 5), torch.arange(5.0)
        out = level3._scale_add(r.clone(), 1.0, 2.0, C)
        torch.testing.assert_close(out, 1.0 + 2.0 * C.expand(4, 5))

    def test_c_that_grows_the_shape_stays_out_of_place(self):
        r, C = torch.ones(1, 5), torch.ones(3, 5)
        out = level3._scale_add(r, 1.0, 1.0, C)
        assert out.shape == (3, 5)

    def test_dtype_promotion_stays_out_of_place(self):
        r, C = torch.ones(3, 3), torch.ones(3, 3, dtype=torch.float64)
        out = level3._scale_add(r, 1.0, 1.0, C)
        assert out.dtype == torch.float64

    def test_non_contiguous_result_comes_back_dense(self):
        r = torch.ones(8, 16)[:, :8]
        out = level3._scale_add(r, 2.0, 0.0, None)
        assert out.is_contiguous() and torch.equal(out, torch.full((8, 8), 2.0))

    def test_autograd_stays_out_of_place(self):
        a = torch.randn(4, 4, requires_grad=True)
        r = a @ a
        out = level3._scale_add(r, 2.0, 0.0, None)
        assert out is not r
        out.sum().backward()
        assert a.grad is not None


class TestWrappersThroughCache:
    """Public wrappers on host inputs, with the NKI kernels replaced by torch."""

    @pytest.fixture
    def fake_kernels(self, native_host_cache, monkeypatch):
        old = trnblas.get_backend()
        monkeypatch.setattr(D, "HAS_NKI", True)
        D._backend = "nki"
        monkeypatch.setattr(D, "_nki_gemm_impl", lambda A, B: torch.matmul(A, B))

        def fake_syrk(A, symmetrize=False):
            r = D._host_empty((A.shape[0], A.shape[0]), A.dtype, A.device)
            torch.matmul(A, A.T, out=r)
            return D._symmetrize(r) if symmetrize else r

        monkeypatch.setattr(D, "_nki_syrk_impl", fake_syrk)
        yield native_host_cache
        D._backend = old

    def test_gemm_alpha_beta(self, fake_kernels):
        g = torch.Generator().manual_seed(1)
        A, B, C = (torch.randn(64, 64, generator=g) for _ in range(3))
        out = trnblas.gemm(0.5, A, B, beta=2.0, C=C)
        torch.testing.assert_close(out, 0.5 * (A @ B) + 2.0 * C)

    def test_gemm_results_do_not_alias(self, fake_kernels):
        A, B = torch.randn(32, 32), torch.randn(32, 32)
        c1 = trnblas.gemm(1.0, A, B)
        c2 = trnblas.gemm(2.0, A, B)
        assert c1.data_ptr() != c2.data_ptr()
        torch.testing.assert_close(c2, 2.0 * c1)

    def test_batched_gemm_preallocated_result(self, fake_kernels):
        g = torch.Generator().manual_seed(2)
        A, B = torch.randn(4, 256, 128, generator=g), torch.randn(4, 128, 512, generator=g)
        out = trnblas.batched_gemm(1.5, A, B)
        torch.testing.assert_close(out, 1.5 * torch.matmul(A, B))
        assert (4, 256, 512) in fake_kernels  # 2 MiB result from the cache

    def test_batched_gemm_empty_batch(self, fake_kernels):
        out = trnblas.batched_gemm(1.0, torch.randn(0, 3, 4), torch.randn(0, 4, 5))
        assert out.shape == (0, 3, 5)

    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((1, 4, 5), (3, 5, 6)),  # broadcast A's batch (was silently (1, 4, 6))
            ((3, 4, 5), (1, 5, 6)),
            ((3, 4, 5), (5, 6)),  # 2-D B (raised)
            ((4, 5), (3, 5, 6)),
            ((2, 3, 4, 5), (2, 3, 5, 6)),  # two batch dims (raised)
            ((2, 1, 4, 5), (3, 5, 6)),
        ],
    )
    @pytest.mark.parametrize("transA, transB", [(False, False), (True, True)])
    def test_batched_gemm_broadcast_matches_matmul(
        self, fake_kernels, a_shape, b_shape, transA, transB
    ):
        """The NKI route follows torch.matmul batch broadcasting (#84)."""
        g = torch.Generator().manual_seed(5)
        A, B = torch.randn(a_shape, generator=g), torch.randn(b_shape, generator=g)
        a = A.transpose(-2, -1).contiguous() if transA else A
        b = B.transpose(-2, -1).contiguous() if transB else B
        out = trnblas.batched_gemm(1.5, a, b, transA=transA, transB=transB)
        want = 1.5 * torch.matmul(A, B)
        assert out.shape == want.shape
        torch.testing.assert_close(out, want)

    @pytest.mark.parametrize("alpha", [1.0, 0.3])
    def test_syrk_symmetric(self, fake_kernels, alpha):
        A = torch.randn(1024, 64)
        out = trnblas.syrk(alpha, A)
        torch.testing.assert_close(out, alpha * (A @ A.T), rtol=1e-5, atol=1e-4)
        assert torch.equal(out, out.T)
        assert (1024, 1024) in fake_kernels

    def test_syrk_with_c_symmetrises_the_sum(self, fake_kernels):
        g = torch.Generator().manual_seed(4)
        A, C = torch.randn(600, 64, generator=g), torch.randn(600, 600, generator=g)
        out = trnblas.syrk(1.0, A, beta=1.0, C=C)
        ref = A @ A.T + C
        torch.testing.assert_close(out, 0.5 * (ref + ref.T), rtol=1e-5, atol=1e-4)
        assert torch.equal(out, out.T)

    @pytest.mark.parametrize("uplo", ["lower", "upper"])
    def test_trsm_blocked_in_place(self, fake_kernels, uplo):
        g = torch.Generator().manual_seed(3)
        n = 300
        L = torch.randn(n, n, generator=g, dtype=torch.float64) * 0.01 + 3 * torch.eye(
            n, dtype=torch.float64
        )
        L = torch.tril(L) if uplo == "lower" else torch.triu(L)
        B = torch.randn(n, 1000, generator=g, dtype=torch.float64)
        B0 = B.clone()
        X = trnblas.trsm(0.5, L, B, uplo=uplo)
        torch.testing.assert_close(L @ X, 0.5 * B0)
        assert torch.equal(B, B0)  # input untouched


class TestSymmetrize:
    @pytest.mark.parametrize("n", [1, 7, 512, 513, 1100])
    def test_bitwise_equal_to_formula(self, n):
        r = torch.randn(n, n)
        ref = 0.5 * (r + r.T)
        out = D._symmetrize(r.clone())
        assert torch.equal(out, ref)

    def test_in_place_on_owned_cpu(self):
        r = torch.randn(600, 600)
        assert D._symmetrize(r).data_ptr() == r.data_ptr()

    def test_non_contiguous_and_autograd_out_of_place(self):
        r = torch.randn(64, 128)[:, :64]
        torch.testing.assert_close(D._symmetrize(r), 0.5 * (r + r.T))
        a = torch.randn(8, 8, requires_grad=True)
        s = D._symmetrize(a * 2)
        s.sum().backward()
        assert a.grad is not None

    def test_nki_syrk_pytorch_backend_symmetrize(self):
        A = torch.randn(100, 30)
        out = D.nki_syrk(A, symmetrize=True)
        assert torch.equal(out, out.T)
        torch.testing.assert_close(out, A @ A.T, rtol=1e-5, atol=1e-5)
