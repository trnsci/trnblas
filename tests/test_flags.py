"""BLAS flag spellings select the documented branch or raise (#73)."""

import pytest
import torch

import trnblas

G = torch.Generator().manual_seed(73)
N, M = 5, 4


def _r(*shape):
    return torch.randn(*shape, generator=G, dtype=torch.float64)


A, B, X = _r(N, N), _r(N, M), _r(N)
W = _r(N, N) * 0.1 + 3 * torch.eye(N, dtype=torch.float64)


def _close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "letter, word",
    [("U", "upper"), ("u", "upper"), ("L", "lower"), ("l", "lower"), ("UPPER", "upper")],
)
def test_uplo_letters(letter, word):
    _close(trnblas.symv(1.5, A, X, uplo=letter), trnblas.symv(1.5, A, X, uplo=word))
    _close(trnblas.symm(1.5, A, B, uplo=letter), trnblas.symm(1.5, A, B, uplo=word))
    _close(trnblas.trmv(A, X, uplo=letter), trnblas.trmv(A, X, uplo=word))
    _close(trnblas.trmm(1.5, A, B, uplo=letter), trnblas.trmm(1.5, A, B, uplo=word))
    _close(trnblas.trsm(1.5, W, B, uplo=letter), trnblas.trsm(1.5, W, B, uplo=word))


@pytest.mark.parametrize(
    "letter, word", [("U", "unit"), ("u", "unit"), ("N", "nonunit"), ("n", "nonunit")]
)
def test_diag_letters(letter, word):
    _close(trnblas.trmv(A, X, diag=letter), trnblas.trmv(A, X, diag=word))
    _close(trnblas.trmm(1.5, A, B, diag=letter), trnblas.trmm(1.5, A, B, diag=word))
    _close(trnblas.trsm(1.5, W, B, diag=letter), trnblas.trsm(1.5, W, B, diag=word))


@pytest.mark.parametrize(
    "letter, word", [("L", "left"), ("l", "left"), ("R", "right"), ("r", "right")]
)
def test_side_letters(letter, word):
    Bs = B if word == "left" else B.T.contiguous()
    _close(trnblas.symm(1.5, A, Bs, side=letter), trnblas.symm(1.5, A, Bs, side=word))
    _close(trnblas.trmm(1.5, A, Bs, side=letter), trnblas.trmm(1.5, A, Bs, side=word))
    _close(trnblas.trsm(1.5, W, Bs, side=letter), trnblas.trsm(1.5, W, Bs, side=word))


@pytest.mark.parametrize("letter, value", [("N", False), ("n", False), ("T", True), ("t", True)])
def test_trans_letters(letter, value):
    _close(trnblas.gemv(1.5, A, X, trans=letter), trnblas.gemv(1.5, A, X, trans=value))
    _close(trnblas.gemm(1.5, A, B, transA=letter), trnblas.gemm(1.5, A, B, transA=value))
    _close(
        trnblas.gemm(1.5, B, B, transA=not value, transB=letter),
        trnblas.gemm(1.5, B, B, transA=not value, transB=value),
    )
    A3, B3 = _r(2, N, N), _r(2, N, M)
    _close(
        trnblas.batched_gemm(1.5, A3, B3, transA=letter),
        trnblas.batched_gemm(1.5, A3, B3, transA=value),
    )
    _close(trnblas.trmv(A, X, trans=letter), trnblas.trmv(A, X, trans=value))
    _close(trnblas.trmm(1.5, A, B, trans=letter), trnblas.trmm(1.5, A, B, trans=value))
    _close(trnblas.trsm(1.5, W, B, trans=letter), trnblas.trsm(1.5, W, B, trans=value))
    _close(trnblas.syrk(1.5, B, trans=letter), trnblas.syrk(1.5, B, trans=value))


def test_letters_match_the_independent_reference():
    """`uplo="U"` is the upper triangle, checked without trnblas on either side."""
    S = torch.triu(A) + torch.triu(A, 1).T
    _close(trnblas.symv(1.0, A, X, uplo="U"), S @ X)
    T = torch.triu(A) - torch.diag(torch.diag(A)) + torch.eye(N, dtype=A.dtype)
    _close(trnblas.trmv(A, X, diag="U"), T @ X)
    _close(trnblas.gemv(1.0, A, X, trans="N"), A @ X)


@pytest.mark.parametrize(
    "call",
    [
        lambda: trnblas.symv(1.0, A, X, uplo="up"),
        lambda: trnblas.symm(1.0, A, B, side="top"),
        lambda: trnblas.trmv(A, X, diag="yes"),
        lambda: trnblas.trsm(1.0, W, B, uplo=None),
        lambda: trnblas.trmm(1.0, A, B, trans="C"),
        lambda: trnblas.gemm(1.0, A, B, transA="yes"),
        lambda: trnblas.syrk(1.0, B, uplo="x"),
    ],
)
def test_unknown_values_raise(call):
    with pytest.raises(ValueError):
        call()


def test_bool_and_int_trans_unchanged():
    _close(trnblas.gemm(1.0, A, B, transA=1), A.T @ B)
    _close(trnblas.gemm(1.0, A, B, transA=0), A @ B)
