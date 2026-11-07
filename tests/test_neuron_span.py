"""Row-block matmul for the >= 4 GiB partition-span bug on the neuron device (#112).

CPU tests: the thresholds are lowered so that small host tensors stand in for
the 32 MiB-row operands, and `_is_neuron` is patched so the host tensors take
the device logic. A spy on `torch.matmul` checks that no product the blocking
issues still has an unsafe operand or result, and every result is compared with
the plain product. The on-device regression tests at the real sizes are in
test_neuron_span_device.py.
"""

import pytest
import torch

import trnblas
from trnblas.nki import dispatch as D

SPAN = 1 << 14  # stands in for 2**32: 128 rows x 128 B (32 fp32)
BLOCK = 1 << 13  # stands in for 2**31


@pytest.fixture
def lowered(monkeypatch):
    """The device rules at 1/2**18 scale, with host tensors treated as device ones."""
    monkeypatch.setattr(D, "_NEURON_SPAN_BYTES", SPAN)
    monkeypatch.setattr(D, "_NEURON_BLOCK_BYTES", BLOCK)
    monkeypatch.setattr(D, "_is_neuron", lambda t: True)
    monkeypatch.setattr(D, "_on_neuron_device", lambda *t: True)
    calls = []
    real = torch.matmul

    def spy(a, b, *args, **kwargs):
        calls.append((tuple(a.shape), tuple(b.shape)))
        if a.dim() == 2 and b.dim() == 2:
            assert not D._span_block(a), f"unsafe lhs {tuple(a.shape)} {a.stride()}"
            assert not D._span_block(b), f"unsafe rhs {tuple(b.shape)} {b.stride()}"
            assert not D._out_block(a.shape[0], b.shape[1], a.element_size()), "unsafe result"
        return real(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", spy)
    return calls


def _r(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64).to(dtype)


def _ref(a, b):
    return (a.double() @ b.double()).to(a.dtype)


class TestRule:
    def test_threshold_edges(self, lowered):
        # B (K, N) fp32: unsafe from min(K, 128) * 4N >= SPAN, i.e. N >= 32 at K >= 128.
        assert not D._span_block(_r(128, 31))
        assert D._span_block(_r(128, 32)) == BLOCK // 128
        assert not D._span_block(_r(64, 32))  # 64 partitions: half the span
        assert D._span_block(_r(64, 64)) == BLOCK // 256
        # bf16: the same bytes at twice the columns.
        assert not D._span_block(_r(128, 63, dtype=torch.bfloat16))
        assert D._span_block(_r(128, 64, dtype=torch.bfloat16))

    def test_transposed_view_counts_its_memory_rows(self, lowered):
        x = _r(200, 16)  # short rows (64 B): safe as a transposed view
        assert not D._span_block(x.T)
        y = _r(128, 64)  # long rows
        assert D._span_block(y.T) and D._row_layout(y.T)[0] == 1

    def test_slice_counts_parent_row_stride(self, lowered):
        wide = _r(128, 64)
        assert D._span_block(wide[:, :8])  # 8 columns, but 256 B row stride

    def test_real_threshold_values(self):
        assert D._NEURON_SPAN_BYTES == 1 << 32
        assert D._NEURON_BLOCK_BYTES == (1 << 31) + (1 << 20)
        # The #112 shapes: (128, 8,389,632) fp32 is unsafe, (128, 8,388,480) is not.
        assert D._out_block(128, 8_389_632, 4) == 64
        assert D._out_block(128, 8_388_480, 4) == 0
        assert D._out_block(64, 8_389_632, 4) == 0
        assert D._out_block(128, 16_777_216, 2)  # bf16 at 2**32 bytes


class TestBlockedMatmul:
    @pytest.mark.parametrize(
        "M,K,N",
        [
            (12, 128, 40),  # the #112 shape: long B rows, K >= 128 -> split K
            (12, 300, 33),  # ragged K blocks
            (12, 64, 80),  # 64 partitions of 320 B: unsafe, 25-row blocks
            (128, 40, 6),  # long A rows -> split M
            (130, 128, 40),  # long rows in A, B and the result
        ],
    )
    def test_matches_plain_product(self, lowered, M, K, N):
        a, b = _r(M, K), _r(K, N, seed=1)
        assert D._neuron_mm_unsafe(a, b)
        out = D._neuron_matmul(a, b)
        torch.testing.assert_close(out, _ref(a, b), rtol=1e-5, atol=1e-4)
        assert len(lowered) > 1  # really blocked

    def test_transposed_views(self, lowered):
        x = _r(48, 128)  # A = x.T: (128, 48) view of long x rows -> split K
        b = _r(48, 8, seed=1)
        torch.testing.assert_close(D._neuron_matmul(x.T, b), _ref(x.T, b), rtol=1e-5, atol=1e-4)
        y = _r(130, 64, seed=2)  # B = y.T: (64, 130) view of long y rows -> split N
        a = _r(5, 64, seed=3)
        torch.testing.assert_close(D._neuron_matmul(a, y.T), _ref(a, y.T), rtol=1e-5, atol=1e-4)

    def test_batched(self, lowered):
        a, b = _r(3, 12, 128), _r(3, 128, 40, seed=1)
        torch.testing.assert_close(D._neuron_matmul(a, b), _ref(a, b), rtol=1e-5, atol=1e-4)
        b2 = _r(128, 40, seed=2)  # broadcast 2-D rhs
        torch.testing.assert_close(D._neuron_matmul(a, b2), _ref(a, b2), rtol=1e-5, atol=1e-4)

    def test_unsafe_result_built_from_row_blocks(self, lowered, monkeypatch):
        cats = []
        real_cat = torch.cat
        monkeypatch.setattr(torch, "cat", lambda ts, dim=0: cats.append(dim) or real_cat(ts, dim))
        a, b = _r(130, 16), _r(16, 40, seed=1)  # result (130, 40): 160 B rows
        torch.testing.assert_close(D._neuron_matmul(a, b), _ref(a, b), rtol=1e-5, atol=1e-4)
        assert cats == [0]

    def test_kernel_blocks_never_unsafe(self, lowered):
        seen = []

        def mm(a, b):
            assert not D._neuron_mm_unsafe(a, b)
            seen.append(a.shape)
            return a @ b

        a, b = _r(12, 128), _r(128, 40, seed=1)
        torch.testing.assert_close(D._neuron_mm2d(a, b, mm=mm), _ref(a, b), rtol=1e-5, atol=1e-4)
        assert len(seen) == 128 // (BLOCK // 160) + 1

    def test_mv(self, lowered):
        a, x = _r(128, 40), _r(40, seed=1)
        out = D._neuron_mv(a, x)
        torch.testing.assert_close(out, (a.double() @ x.double()).float(), rtol=1e-5, atol=1e-4)


class TestUntouchedWhenSafe:
    """Below the rule (and on the host) the products are the same aten call as before."""

    def test_safe_device_product_is_one_plain_call(self, monkeypatch):
        monkeypatch.setattr(D, "_is_neuron", lambda t: True)
        calls = []
        real = torch.matmul
        monkeypatch.setattr(torch, "matmul", lambda a, b: calls.append((a, b)) or real(a, b))
        a, b = _r(96, 128), _r(128, 4096, seed=1)
        out = D._neuron_matmul(a, b)
        assert len(calls) == 1 and calls[0][0] is a and calls[0][1] is b
        assert torch.equal(out, real(a, b))

    def test_host_tensors_never_blocked(self, monkeypatch):
        monkeypatch.setattr(D, "_NEURON_SPAN_BYTES", SPAN)
        a, b = _r(12, 128), _r(128, 40, seed=1)
        assert not D._neuron_mm_unsafe(a, b)
        assert torch.equal(D._neuron_matmul(a, b), torch.matmul(a, b))

    def test_1d_operands_untouched(self, lowered):
        a, v = _r(12, 128), _r(128)
        assert not D._neuron_mm_unsafe(a, v)


class TestEntryPoints:
    """Every device matmul entry point goes through the blocking."""

    @pytest.fixture(params=["auto", "pytorch"])
    def backend(self, request):
        old = trnblas.get_backend()
        trnblas.set_backend(request.param)
        yield request.param
        trnblas.set_backend(old)

    def test_gemm(self, lowered, backend):
        a, b = _r(12, 128), _r(128, 40, seed=1)
        out = trnblas.gemm(2.0, a, b)
        torch.testing.assert_close(out, 2.0 * _ref(a, b), rtol=1e-5, atol=1e-4)
        assert len(lowered) > 1
        lowered.clear()
        out = trnblas.gemm(1.0, b, a, transA=True, transB=True)
        torch.testing.assert_close(out, _ref(b.T, a.T), rtol=1e-5, atol=1e-4)

    def test_batched_gemm(self, lowered, backend):
        a, b = _r(2, 12, 128), _r(2, 128, 40, seed=1)
        out = trnblas.batched_gemm(1.0, a, b)
        torch.testing.assert_close(out, _ref(a, b), rtol=1e-5, atol=1e-4)
        assert len(lowered) > 1

    def test_symm(self, lowered, backend):
        s, b = _r(128, 128), _r(128, 40, seed=1)
        sym = torch.triu(s) + torch.triu(s, 1).T
        out = trnblas.symm(1.0, s, b)
        torch.testing.assert_close(out, _ref(sym, b), rtol=1e-5, atol=1e-4)
        assert len(lowered) > 1

    def test_syrk(self, lowered, backend):
        a = _r(128, 40)  # long rows: A @ A.T
        torch.testing.assert_close(trnblas.syrk(1.0, a), _ref(a, a.T), rtol=1e-5, atol=1e-4)
        assert len(lowered) > 1
        lowered.clear()
        t = _r(40, 128, seed=1)  # trans: A^T A, the transposed copy would be long-row
        torch.testing.assert_close(
            trnblas.syrk(1.0, t, trans=True), _ref(t.T, t), rtol=1e-5, atol=1e-4
        )

    def test_trmm_and_gemv(self, lowered, backend):
        t, b = _r(128, 128), _r(128, 40, seed=1)
        out = trnblas.trmm(1.0, t, b)
        torch.testing.assert_close(out, _ref(torch.triu(t), b), rtol=1e-5, atol=1e-4)
        a, x = _r(128, 40), _r(40, seed=2)
        out = trnblas.gemv(1.0, a, x)
        torch.testing.assert_close(out, (a.double() @ x.double()).float(), rtol=1e-5, atol=1e-4)

    def test_trsm_updates(self, lowered):
        L = torch.tril(_r(256, 256)) + 256 * torch.eye(256)
        B = _r(256, 40, seed=1)
        X = D._nki_trsm_left(L, B, "lower", False, "nonunit", 1.0, mm=D._neuron_matmul)
        ref = torch.linalg.solve_triangular(L.double(), B.double(), upper=False)
        torch.testing.assert_close(X.double(), ref, rtol=1e-5, atol=1e-5)

    def test_nki_gemm_impl_takes_blocked_aten(self, lowered, monkeypatch):
        monkeypatch.setattr(D, "HAS_NKI", True)
        monkeypatch.setattr(D, "_launch", lambda *a: pytest.fail("NKI kernel on an unsafe operand"))
        a, b = _r(12, 128), _r(128, 40, seed=1)
        torch.testing.assert_close(D._nki_gemm_impl(a, b), _ref(a, b), rtol=1e-5, atol=1e-4)
        assert len(lowered) > 1

    def test_lnc_route_blocks(self, lowered, monkeypatch):
        monkeypatch.setattr(
            D, "_call_kernel", lambda *a: pytest.fail("LNC kernel on an unsafe operand")
        )
        a, b = _r(128, 12).T, _r(128, 40, seed=1)
        torch.testing.assert_close(D._nki_gemm_lnc(a, b), _ref(a, b), rtol=1e-5, atol=1e-4)
