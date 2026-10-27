"""CPU tests for the GEMM pad-to-tile / slice-back wrapper (#50).

The NKI GEMM kernels have no remainder handling: M and K must be multiples
of the tile size, and N either fits in one N-tile or is a multiple of it.
`_run_padded_gemm` zero-pads to those multiples, launches, and slices the
result back. These tests drive it with a stand-in kernel (torch.matmul plus
alignment assertions) so they run without Neuron hardware.
"""

import pytest
import torch

import trnblas
from trnblas import batched_gemm, gemm
from trnblas.nki import dispatch

TM, TK, TN = dispatch._TILE_M, dispatch._TILE_K, dispatch._TILE_N


def _assert_kernel_aligned(a: torch.Tensor, b: torch.Tensor, tm=TM, tk=TK, tn=TN):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % tm == 0, f"M={M} not a multiple of {tm}"
    assert K % tk == 0, f"K={K} not a multiple of {tk}"
    assert tn >= N or N % tn == 0, f"N={N} neither <= {tn} nor a multiple of it"
    assert a.is_contiguous() and b.is_contiguous()


def _make_stand_in(calls: list, tm=TM, tk=TK, tn=TN, poison_pad: tuple | None = None):
    """Stand-in kernel: checks alignment, records shapes, returns a @ b.

    If ``poison_pad=(M, N)`` is given, everything outside [:M, :N] is set to NaN
    so a missing slice shows up as a NaN in the result.
    """

    def kernel(a, b):
        _assert_kernel_aligned(a, b, tm, tk, tn)
        calls.append((tuple(a.shape), tuple(b.shape), a, b))
        out = a @ b
        if poison_pad is not None:
            M, N = poison_pad
            out = out.clone()
            out[M:, :] = float("nan")
            out[:, N:] = float("nan")
        return out

    return kernel


@pytest.mark.parametrize(
    "M,K,N",
    [
        (1, 1, 1),
        (127, 129, 130),
        (1000, 1000, 1000),
        (300, 513, 700),
        (128, 128, 513),
        (129, 256, 512),
        (64, 200, 1025),
    ],
)
def test_padded_shape_and_result(M, K, N):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float64)
    B = torch.randn(K, N, dtype=torch.float64)
    calls: list = []
    out = dispatch._run_padded_gemm(A, B, _make_stand_in(calls, poison_pad=(M, N)))
    assert out.shape == (M, N)
    torch.testing.assert_close(out, A @ B)
    (a_shape, b_shape, a_p, b_p) = calls[0]
    exp = dispatch._gemm_padded_shape(M, K, N, TM, TK, TN)
    assert a_shape == (exp[0], exp[1]) and b_shape == (exp[1], exp[2])
    # Padding region must be exactly zero (K padding is what keeps it exact).
    assert torch.count_nonzero(a_p[M:, :]) == 0 and torch.count_nonzero(a_p[:, K:]) == 0
    assert torch.count_nonzero(b_p[K:, :]) == 0 and torch.count_nonzero(b_p[:, N:]) == 0


def test_padded_shape_values():
    assert dispatch._gemm_padded_shape(8191, 8191, 8191, 128, 128, 512) == (8192, 8192, 8192)
    assert dispatch._gemm_padded_shape(2048, 4096, 8191, 128, 128, 512) == (2048, 4096, 8192)
    assert dispatch._gemm_padded_shape(1000, 1000, 1000, 128, 128, 512) == (1024, 1024, 1024)
    # N <= tile_n: kernel clamps TILE_N = N, so N is left unpadded.
    assert dispatch._gemm_padded_shape(100, 100, 300, 128, 128, 512) == (128, 128, 300)
    # Other autotune candidates.
    assert dispatch._gemm_padded_shape(100, 100, 300, 64, 128, 256) == (128, 128, 512)


@pytest.mark.parametrize("tiles", dispatch._TILE_CANDIDATES)
def test_all_tile_candidates(tiles):
    tm, tk, tn = tiles
    M, K, N = 190, 333, 777
    A = torch.randn(M, K, dtype=torch.float64)
    B = torch.randn(K, N, dtype=torch.float64)
    calls: list = []
    out = dispatch._run_padded_gemm(
        A, B, _make_stand_in(calls, tm, tk, tn, poison_pad=(M, N)), tm, tk, tn
    )
    torch.testing.assert_close(out, A @ B)


@pytest.mark.parametrize("M,K,N", [(128, 128, 128), (256, 384, 512), (128, 256, 1024)])
def test_aligned_no_copy(M, K, N):
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    calls: list = []
    sentinel = torch.empty(M, N)

    def kernel(a, b):
        calls.append((a, b))
        return sentinel

    out = dispatch._run_padded_gemm(A, B, kernel)
    a_fed, b_fed = calls[0]
    assert a_fed is A and b_fed is B  # no padded copy, no contiguous copy
    assert out is sentinel  # no slice view


def test_transposed_view_inputs():
    # Non-contiguous transposed views: padded and fed densely.
    M, K, N = 300, 200, 600
    At = torch.randn(K, M, dtype=torch.float64)
    Bt = torch.randn(N, K, dtype=torch.float64)
    calls: list = []
    out = dispatch._run_padded_gemm(At.T, Bt.T, _make_stand_in(calls, poison_pad=(M, N)))
    torch.testing.assert_close(out, At.T @ Bt.T)


def test_inner_dim_mismatch():
    with pytest.raises(ValueError):
        dispatch._pad_gemm_operands(torch.randn(4, 5), torch.randn(6, 7), TM, TK, TN)


@pytest.fixture
def nki_stand_in(monkeypatch):
    """Route trnblas.gemm / batched_gemm through the nki dispatch path with
    `_nki_gemm_impl` replaced by the pad/slice wrapper + stand-in kernel."""
    calls: list = []

    def fake_impl(A, B):
        M, N = A.shape[0], B.shape[1]
        return dispatch._run_padded_gemm(A, B, _make_stand_in(calls, poison_pad=(M, N)))

    prev = dispatch.get_backend()
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setattr(dispatch, "_nki_gemm_impl", fake_impl)
    yield calls
    monkeypatch.setattr(dispatch, "_backend", prev)


@pytest.mark.parametrize(
    "transA,transB", [(False, False), (True, False), (False, True), (True, True)]
)
def test_gemm_alpha_beta_trans_odd(nki_stand_in, transA, transB):
    torch.manual_seed(1)
    M, K, N = 131, 257, 515
    A = torch.randn(K, M, dtype=torch.float64) if transA else torch.randn(M, K, dtype=torch.float64)
    B = torch.randn(N, K, dtype=torch.float64) if transB else torch.randn(K, N, dtype=torch.float64)
    C = torch.randn(M, N, dtype=torch.float64)
    alpha, beta = 1.5, -0.75
    out = gemm(alpha, A, B, beta=beta, C=C, transA=transA, transB=transB)
    opA = A.T if transA else A
    opB = B.T if transB else B
    torch.testing.assert_close(out, alpha * (opA @ opB) + beta * C)
    assert len(nki_stand_in) == 1  # really went through the padded path


def test_batched_gemm_odd(nki_stand_in):
    torch.manual_seed(2)
    batch, M, K, N = 3, 100, 130, 600
    A = torch.randn(batch, M, K, dtype=torch.float64)
    B = torch.randn(batch, N, K, dtype=torch.float64)
    C = torch.randn(batch, M, N, dtype=torch.float64)
    out = batched_gemm(2.0, A, B, beta=0.5, C=C, transB=True)
    torch.testing.assert_close(out, 2.0 * torch.bmm(A, B.transpose(-2, -1)) + 0.5 * C)
    assert len(nki_stand_in) == batch


def test_pytorch_backend_untouched():
    trnblas.set_backend("pytorch")
    try:
        A = torch.randn(1000, 1000)
        B = torch.randn(1000, 1000)
        torch.testing.assert_close(gemm(1.0, A, B), A @ B)
    finally:
        trnblas.set_backend("auto")
