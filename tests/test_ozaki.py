"""CPU tests for the fp64-class Ozaki GEMM (#22): host split, schedule, plan,
the bit-exact host reference of the kernel arithmetic, and `gemm_fp64`.

Accuracy is checked against an independent near-correctly-rounded reference
(`_ref_exact`: fp64 Ozaki with slices narrow enough that every host fp64 GEMM
is exact, Neumaier-summed), never against the routine itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _ozaki_ref import DEVICE_SPLIT_KERNELS, device_split_ops
from _ozaki_ref import inputs as _inputs
from _ozaki_ref import max_cw as _max_cw

import trnblas
from trnblas import ozaki
from trnblas.nki import dispatch

# ── split ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("per", ["row", "col"])
@pytest.mark.parametrize("k", [1, 3, 8, 12])
def test_split_digits_exact_and_bounded(per, k):
    torch.manual_seed(0)
    X = torch.randn(150, 300, dtype=torch.float64) * torch.logspace(-5, 5, 300, dtype=torch.float64)
    d, e = ozaki.split(X, k, per=per, pad=(128, 64))
    assert d.dtype == torch.bfloat16 and d.shape == (k, 256, 320)
    D = d.double().numpy()
    assert np.array_equal(D, np.rint(D)) and np.abs(D).max() <= 64
    assert not D[:, 150:].any() and not D[:, :, 300:].any()  # zero padding
    sc = np.ldexp(1.0, e - 6)
    sc = sc[:, None] if per == "row" else sc[None, :]
    rec = sum(D[s, :150, :300] * 2.0 ** (-7 * s) for s in range(k)) * sc
    bound = np.ldexp(1.0, e - 7 * k)
    bound = bound[:, None] if per == "row" else bound[None, :]
    assert (np.abs(rec - X.numpy()) <= bound).all()


def test_split_extreme_and_zero_rows():
    X = torch.zeros(4, 300, dtype=torch.float64)
    X[0] = torch.linspace(-1e300, 1e300, 300, dtype=torch.float64)
    X[1] = torch.linspace(-1e-300, 1e-300, 300, dtype=torch.float64)
    X[2, 5] = 5e-324  # subnormal
    d, e = ozaki.split(X, 8, per="row", pad=(1, 1))
    D = d.double().numpy()
    rec = sum(D[s] * 2.0 ** (-7 * s) for s in range(8))
    for r in range(4):
        got = np.ldexp(rec[r], e[r] - 6)
        assert (np.abs(got - X[r].numpy()) <= np.ldexp(1.0, e[r] - 56)).all(), r
    assert not D[:, 3].any() and e[3] == 0


def _bit_patterns(rng, shape):
    """Finite fp64 from random bit patterns: every exponent, sign and
    mantissa, with a third of the columns subnormal or zero."""
    bits = rng.integers(0, 2**63, shape, dtype=np.uint64)
    bits[:, ::3] &= ~np.uint64(0x7FF0000000000000)
    X = bits.view(np.float64).copy()
    X[~np.isfinite(X)] = 1.0
    X[::2] *= -1
    return X


def _ties(rng, k):
    """Values on the digit grid of a row maximum just under 1, many of them
    exactly half-way: every round-half-even branch of the digit chain."""
    T = rng.integers(-(2**20), 2**20, (40, 300)).astype(np.float64)
    T += rng.choice([0.0, 0.5, -0.5, 0.25], T.shape)
    T *= 2.0 ** (-7 * (k - 1) - 14)
    T[:, 0] = 1.0 - 2.0**-52
    return T


@pytest.mark.parametrize("k", range(1, 13))
def test_split_from_bits_equals_split(k):
    """The device split's arithmetic (#123), op for op on the host: the same
    digits and exponents as `split`, from the raw bits, for every k."""
    rng = np.random.default_rng(k)
    cases = [
        rng.standard_normal((50, 300)),
        rng.standard_normal((30, 129)) * 10.0 ** rng.uniform(-40, 40, (30, 129)),
        rng.standard_normal((8, 70)) * 1e300,
        rng.standard_normal((8, 70)) * 1e-300,
        _bit_patterns(rng, (40, 200)),
        _ties(rng, k),
        np.array([[0.0, -0.0, 0.0], [-0.0, 0.0, 0.0]]),
    ]
    for X in cases:
        d, e = ozaki.split(torch.from_numpy(X), k, per="row", pad=(1, 1))
        D, e2 = ozaki.split_from_bits(X, k)
        assert np.array_equal(D, d.float().numpy()) and np.array_equal(e2, e)


def test_split_from_bits_subnormal_row_maximum_differs():
    """A row whose maximum is subnormal: the exponent field is 0, so the
    device route leaves it to the host split (see `_oz_device_split`)."""
    X = np.full((1, 4), 1e-310)
    assert ozaki.split_from_bits(X, 4)[1][0] == 0
    assert ozaki.split(torch.from_numpy(X), 4, per="row")[1][0] != 0


@pytest.mark.parametrize("bad", [0, 13, 2.0, True, "8"])
def test_slices_validated(bad):
    with pytest.raises((ValueError, TypeError)):
        ozaki.check_slices(bad)


# ── schedule and plan ────────────────────────────────────────────────────────


@pytest.mark.parametrize("kt", [1, 7, 8, 31, 32, 33, 66, 100])
@pytest.mark.parametrize("d", [0, 1, 4, 7])
def test_diag_groups_cover_each_tile_once(kt, d):
    groups = ozaki.diag_groups(kt, d)
    flat = [(s, t, k0 + i) for g in groups for s, t, k0, n in g for i in range(n)]
    assert flat == [(s, d - s, j) for s in range(d + 1) for j in range(kt)]
    assert all(0 < sum(n for *_, n in g) <= ozaki.GROUP_TILES for g in groups)
    if kt >= ozaki.GROUP_TILES:  # one pair per group, full groups then the remainder
        assert all(len(g) == 1 for g in groups)
        assert len(groups) == (d + 1) * -(-kt // ozaki.GROUP_TILES)
    else:  # whole pairs, as many as fit
        per = ozaki.GROUP_TILES // kt
        assert all(len(g) == per for g in groups[:-1]) and all(
            n == kt for g in groups for *_, n in g
        )


def test_group_bound_is_exact_in_fp32():
    # Every partial sum of a group is an integer of at most 24 bits.
    assert ozaki.GROUP_TERMS * (2 ** (ozaki.SLICE_BITS - 1)) ** 2 <= 2**24
    assert ozaki.GROUP_TILES * ozaki.TILE_K == ozaki.GROUP_TERMS


def test_schedule_order_and_scales():
    sch = ozaki.schedule(4, 3)
    assert [dd for dd, _, _ in sch] == [3, 2, 1, 0]  # smallest first
    assert [sc for _, sc, _ in sch] == [1.0, 128.0, 16384.0, 2097152.0]
    assert ozaki.n_products(8) == 36 and ozaki.exponent_offset(8) == -61


@pytest.mark.parametrize(
    "Mp,Kp,N,k,limit",
    [
        (4096, 4096, 4096, 8, ozaki.BLOCK_BYTES),
        (16384, 16384, 241, 8, ozaki.BLOCK_BYTES),
        (3328, 8448, 241, 8, ozaki.BLOCK_BYTES),
        (1024, 2048, 3000, 8, 1 << 22),
        (256, 1 << 20, 64, 10, 1 << 26),
    ],
)
def test_plan_respects_launch_limit(Mp, Kp, N, k, limit):
    mbs, kbs, nbs = ozaki.plan(Mp, Kp, N, k, limit)
    for blocks, D in ((mbs, Mp), (kbs, Kp), (nbs, N)):
        assert blocks[0][0] == 0 and sum(sz for _, sz in blocks) == D
        assert all(a + sa == b for (a, sa), (b, _) in zip(blocks, blocks[1:], strict=False))
    assert all(sz % 128 == 0 for _, sz in mbs + kbs)
    Mb, Kb, Nb = mbs[0][1], kbs[0][1], nbs[0][1]
    assert 2 * k * Kb * Mb <= limit and 2 * k * Kb * Nb <= limit and 8 * Mb * Nb <= limit
    assert Mb <= ozaki.LAUNCH_M and Nb <= ozaki.LAUNCH_N
    if limit == ozaki.BLOCK_BYTES and Mp * Kp * 2 * k <= limit:
        assert len(kbs) == 1
    assert all(sz == Mb for _, sz in mbs[:-1]) and all(sz == Nb for _, sz in nbs[:-1])


def test_launch_caps_are_bitwise_invariant():
    A, B = _inputs(700, 900, 600, seed=7)
    C = ozaki.reference_gemm(A, B, 4)
    assert len(ozaki.plan(768, 1024, 600, 4, max_m=256, max_n=200)[0]) == 3
    assert torch.equal(ozaki.reference_gemm(A, B, 4, max_m=256, max_n=200), C)


# ── the kernel arithmetic on the host ───────────────────────────────────────


@pytest.mark.parametrize(
    "shape", [(64, 64, 64), (37, 300, 53), (130, 4100, 7), (200, 9000, 31), (1, 1, 1)]
)
@pytest.mark.parametrize("k,tol", [(6, 2e-11), (7, 2e-13), (8, 1.5e-15)])
def test_reference_accuracy(shape, k, tol):
    A, B = _inputs(*shape)
    C = ozaki.reference_gemm(A, B, k)
    assert C.dtype == torch.float64 and C.shape == (shape[0], shape[2])
    assert _max_cw(C, A, B) <= tol


def test_reference_accuracy_wide_range():
    A, B = _inputs(64, 2000, 48, dist="logu4")
    host = _max_cw(A @ B, A, B)
    assert _max_cw(ozaki.reference_gemm(A, B, 10), A, B) <= max(4 * host, 4e-15)
    assert _max_cw(ozaki.reference_gemm(A, B, 8), A, B) <= 1e-12


def test_reference_k1_is_exact_integer_product():
    rng = np.random.default_rng(3)
    A = torch.from_numpy(rng.integers(-63, 64, (40, 5000)).astype(np.float64) * 2.0**-6)
    B = torch.from_numpy(rng.integers(-63, 64, (5000, 30)).astype(np.float64))
    A[:, 0] = 63 * 2.0**-6  # same row exponent everywhere: digits are A * 2^6 exactly
    B[0] = 63
    assert torch.equal(ozaki.reference_gemm(A, B, 1), A @ B)


def test_reference_independent_of_layout():
    A, B = _inputs(150, 700, 90, seed=1)
    C = ozaki.reference_gemm(A, B, 8)
    At = A.T.contiguous().T  # transposed views split in their own layout
    Bt = B.T.contiguous().T
    assert torch.equal(ozaki.reference_gemm(At, B, 8), C)
    assert torch.equal(ozaki.reference_gemm(A, Bt, 8), C)
    assert torch.equal(ozaki.reference_gemm(At, Bt, 8), C)


def test_reference_mn_blocking_is_bitwise_invariant():
    A, B = _inputs(2300, 500, 1500, seed=2)
    C = ozaki.reference_gemm(A, B, 3)
    limit = 2 * 3 * 512 * 1024  # K whole, M and N in blocks
    mbs, kbs, nbs = ozaki.plan(2304, 512, 1500, 3, limit)
    assert len(mbs) > 1 and len(nbs) > 1 and len(kbs) == 1
    assert torch.equal(ozaki.reference_gemm(A, B, 3, limit=limit), C)


def test_reference_k_blocking_stays_accurate():
    A, B = _inputs(40, 3000, 20, seed=4)
    limit = 2 * 8 * 128 * 1024  # K blocks of 128 rows
    assert len(ozaki.plan(128, 3072, 20, 8, limit)[1]) > 1
    assert _max_cw(ozaki.reference_gemm(A, B, 8, limit=limit), A, B) <= 1.5e-15


def test_reference_from_digits_rejects_unpadded_k():
    with pytest.raises(ValueError):
        ozaki.reference_from_digits(
            torch.zeros(2, 100, 128, dtype=torch.bfloat16),
            torch.zeros(2, 100, 4, dtype=torch.bfloat16),
        )


# ── gemm_fp64 ────────────────────────────────────────────────────────────────


@pytest.fixture
def host_route(monkeypatch):
    monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)


@pytest.mark.parametrize("backend", ["auto", "pytorch"])
def test_gemm_fp64_off_device_is_host_fp64(host_route, backend):
    prev = trnblas.get_backend()
    trnblas.set_backend(backend)
    try:
        A, B = _inputs(33, 70, 21)
        assert torch.equal(trnblas.gemm_fp64(A, B), A @ B)
        assert torch.equal(
            trnblas.gemm_fp64(A.float(), B.float()), A.float().double() @ B.float().double()
        )
        # op(X) is a transposed view, and a BLAS may sum a transposed operand
        # in another order (MKL on Linux does; Accelerate happens not to), so
        # the bitwise reference is the same host matmul on the same views.
        At, Bt = A.T.contiguous(), B.T.contiguous()
        out = trnblas.gemm_fp64(At, B, transA=True)
        assert torch.equal(out, At.mT @ B)
        torch.testing.assert_close(out, A @ B, rtol=1e-12, atol=1e-12)
        out = trnblas.gemm_fp64(A, Bt, transB="T")
        assert torch.equal(out, A @ Bt.mT)
        torch.testing.assert_close(out, A @ B, rtol=1e-12, atol=1e-12)
    finally:
        trnblas.set_backend(prev)


def test_gemm_fp64_off_device_require_nki_raises(host_route, monkeypatch):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    A, B = _inputs(8, 8, 8)
    with pytest.raises(RuntimeError, match="native"):
        trnblas.gemm_fp64(A, B)


@pytest.mark.parametrize(
    "A,B,exc",
    [
        (torch.ones(3), torch.ones(3, 2), ValueError),
        (torch.ones(3, 4), torch.ones(5, 2), ValueError),
        (torch.ones(2, 2, 2), torch.ones(2, 2), ValueError),
        (torch.ones(3, 4, dtype=torch.int64), torch.ones(4, 2), TypeError),
        (torch.ones(3, 4, dtype=torch.complex128), torch.ones(4, 2), TypeError),
    ],
)
def test_gemm_fp64_rejects_bad_operands(A, B, exc):
    with pytest.raises(exc):
        trnblas.gemm_fp64(A, B)


def test_gemm_fp64_rejects_bad_slices():
    A = torch.ones(2, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        trnblas.gemm_fp64(A, A, slices=0)
    with pytest.raises(TypeError):
        trnblas.gemm_fp64(A, A, slices=8.0)


@pytest.fixture
def native_reference(monkeypatch):
    """Native path on CPU: the `ozaki_gemm` op is the host reference of the
    kernel and the device-split ops are their CPU twins, so gemm_fp64 runs
    its whole device route (raw-bit uploads, device split, on-device
    transposes, blocks, combine) and must match reference_gemm bit for bit.
    The list records the `ozaki_gemm` launches; `.split` the split ops."""

    class Calls(list):
        split: list

    calls = Calls()

    def op(at, b):
        calls.append((tuple(at.shape), tuple(b.shape), at.is_contiguous(), b.is_contiguous()))
        return torch.from_numpy(ozaki.reference_from_digits(at, b))

    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    split_calls: list = []
    ops = {"ozaki_gemm": op}
    for name, fn in device_split_ops().items():

        def rec(*a, _n=name, _f=fn):
            split_calls.append(_n)
            return _f(*a)

        ops[name] = rec
    monkeypatch.setattr(dispatch, "_native_ops", ops)
    for name in ("_ozaki_gemm_kernel", *DEVICE_SPLIT_KERNELS):
        monkeypatch.setattr(dispatch, name, object(), raising=False)
    monkeypatch.setattr(dispatch, "_OZ_DEVICE_SPLIT", True)
    calls.split = split_calls
    return calls


@pytest.mark.parametrize("transA,transB", [(False, False), (True, False), (False, True)])
def test_gemm_fp64_native_route_matches_reference(native_reference, transA, transB):
    A, B = _inputs(200, 333, 77, seed=5)
    a = A.T.contiguous() if transA else A
    b = B.T.contiguous() if transB else B
    C = trnblas.gemm_fp64(a, b, slices=7, transA=transA, transB=transB)
    assert torch.equal(C, ozaki.reference_gemm(A, B, 7))
    ((sa, sb, ca, cb),) = native_reference
    # N = 77 < SWAP_N: one launch of C^T = B^T A^T, B's slices first
    assert sa == (7, 384, 128) and sb == (7, 384, 200) and ca and cb


def test_gemm_fp64_native_route_wide_n_not_swapped(native_reference):
    A, B = _inputs(130, 200, 600, seed=8)
    assert torch.equal(trnblas.gemm_fp64(A, B, slices=3), ozaki.reference_gemm(A, B, 3))
    ((sa, sb, _, _),) = native_reference
    assert sa == (3, 256, 256) and sb == (3, 256, 600)


def test_gemm_fp64_native_route_nonfinite_takes_host(native_reference):
    A, B = _inputs(10, 20, 5)
    A[3, 4] = float("nan")
    B[0, 0] = float("inf")
    C = trnblas.gemm_fp64(A, B)
    assert torch.equal(torch.isnan(C), torch.isnan(A @ B))
    assert not native_reference


def test_gemm_fp64_native_route_empty(native_reference):
    assert trnblas.gemm_fp64(torch.ones(0, 3), torch.ones(3, 4)).shape == (0, 4)
    assert torch.equal(
        trnblas.gemm_fp64(torch.ones(2, 0), torch.ones(0, 4)),
        torch.zeros(2, 4, dtype=torch.float64),
    )
    assert not native_reference


# ── device split route (#123) ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "transA,transB", [(False, False), (True, False), (False, True), (True, True)]
)
def test_device_split_route_splits_in_place(native_reference, transA, transB):
    """Each operand is split on the device in its own layout: per row for a
    row-major A or B^T, per column for A^T or a row-major B (the layouts of
    `split_a` / `split_b`, so the launches see the same K-major slices)."""
    A, B = _inputs(150, 260, 140, seed=11)
    a = A.T.contiguous() if transA else A
    b = B.T.contiguous() if transB else B
    C = trnblas.gemm_fp64(a, b, slices=5, transA=transA, transB=transB)
    assert torch.equal(C, ozaki.reference_gemm(A, B, 5))
    want_a = ["ozaki_emax_col", "ozaki_split_col"] if transA else ["ozaki_emax", "ozaki_split"]
    want_b = ["ozaki_emax", "ozaki_split"] if transB else ["ozaki_emax_col", "ozaki_split_col"]
    assert native_reference.split == want_a + want_b


def test_device_split_route_wide_range_and_ties(native_reference):
    rng = np.random.default_rng(12)
    A = torch.from_numpy(_ties(rng, 8)[:, :256])
    B = torch.from_numpy(rng.standard_normal((256, 90)) * 10.0 ** rng.uniform(-8, 8, (256, 90)))
    assert torch.equal(trnblas.gemm_fp64(A, B), ozaki.reference_gemm(A, B, 8))


def test_device_split_subnormal_maximum_takes_host_split(native_reference):
    A, B = _inputs(40, 60, 30, seed=13)
    A[7] = 1e-310  # a row whose maximum is subnormal
    C = trnblas.gemm_fp64(A, B, slices=6)
    assert torch.equal(C, ozaki.reference_gemm(A, B, 6))
    # A: exponent pass only (host split); B: the full device split
    assert native_reference.split == ["ozaki_emax", "ozaki_emax_col", "ozaki_split_col"]


def test_device_split_nonfinite_found_on_device(native_reference):
    A, B = _inputs(40, 60, 30, seed=14)
    B[5, 6] = float("-inf")
    C = trnblas.gemm_fp64(A, B)
    assert torch.equal(torch.isnan(C), torch.isnan(A @ B)) and torch.equal(
        C.isinf(), (A @ B).isinf()
    )
    assert not native_reference and native_reference.split[-1] == "ozaki_emax_col"


def test_device_split_off_uses_host_split(native_reference, monkeypatch):
    monkeypatch.setattr(dispatch, "_OZ_DEVICE_SPLIT", False)
    A, B = _inputs(40, 60, 30, seed=15)
    assert torch.equal(trnblas.gemm_fp64(A, B, slices=4), ozaki.reference_gemm(A, B, 4))
    assert not native_reference.split and native_reference


def test_device_split_oversize_operand_takes_host_split(native_reference, monkeypatch):
    monkeypatch.setattr(dispatch, "_OZ_RESIDENT_BYTES", 8 * 40 * 60 + 1)
    A, B = _inputs(40, 60, 30, seed=16)
    assert torch.equal(trnblas.gemm_fp64(A, B, slices=3), ozaki.reference_gemm(A, B, 3))
    assert "ozaki_split" not in native_reference.split
