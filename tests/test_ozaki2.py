"""CPU tests for the Ozaki scheme II prototype (trnblas#132): moduli and CRT
constants, scaling and residues, the kernels' arithmetic (residue GEMM with
group-close reductions, device CRT) against exact references, the host
reference GEMM's accuracy and determinism, edge cases and the #112 blocking.

Accuracy is checked against the independent near-correctly-rounded
`_ozaki_ref.ref_exact`, never against the routine itself.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
from _ozaki_ref import inputs as _inputs
from _ozaki_ref import max_cw as _max_cw

import trnblas
from trnblas import ozaki, ozaki2

KERNEL_FAMILIES = ("b7", "b8", "b9")


# ── moduli and CRT constants ─────────────────────────────────────────────────


@pytest.mark.parametrize("fam", list(ozaki2.FAMILIES))
def test_moduli_are_pairwise_coprime_and_bounded(fam):
    f = ozaki2.FAMILIES[fam]
    mods = f.moduli
    assert len(mods) == f.max_moduli and len(set(mods)) == len(mods)
    for i, a in enumerate(mods):
        for b in mods[i + 1 :]:
            assert math.gcd(a, b) == 1
        assert a // 2 <= f.rmax
    # every PSUM partial sum of a group is an exact fp32 integer, with room
    # for the accumulator and q m below 2^24
    assert f.group_terms * (f.product_bound or f.rmax**2) + 2 * f.mmax < 2**24
    if f.kernel:
        assert f.rmax <= 256 and all(m % 2 for m in mods)


@pytest.mark.parametrize("fam,n", [("b7", 2), ("b7", 16), ("b7", 28), ("b8", 14), ("b9", 32)])
def test_crt_constants(fam, n):
    c = ozaki2.crt(fam, n)
    f = ozaki2.FAMILIES[fam]
    assert math.prod(c.moduli) == c.M
    for l in range(n):
        for j, mj in enumerate(c.moduli):
            assert c.w[l] % mj == (1 if j == l else 0)
    # limbs reconstruct the weights and M, and are fp32 / fp64 exact
    for l in range(n):
        assert sum(int(x) << (c.beta * j) for j, x in enumerate(c.W[l])) == c.w[l]
        assert sum(int(x) << (c.hbeta * j) for j, x in enumerate(c.hW[l])) == c.w[l]
    assert sum(int(x) << (c.beta * j) for j, x in enumerate(c.Ml)) == c.M
    rn = n * f.rmax
    assert 2 * rn * (2**c.beta) <= 2**24
    assert 2 ** (2 * c.beta - 1) + 2 ** (c.beta - 1) <= 2**24  # packed digit pairs
    assert 2 * rn * (2**c.hbeta) <= 2**53
    assert 1 <= c.keep <= c.npk and c.j0 >= 0


def test_check_moduli():
    assert ozaki2.check_moduli(16, "b7") == 16
    for bad in (1, 29):
        with pytest.raises(ValueError):
            ozaki2.check_moduli(bad, "b7")
    with pytest.raises(TypeError):
        ozaki2.check_moduli(16.0, "b7")
    with pytest.raises(ValueError):
        ozaki2.family("b6")


# ── scaling and residues ─────────────────────────────────────────────────────


@pytest.mark.parametrize("fam,n", [("b7", 8), ("b7", 16), ("b8", 14), ("b9", 12), ("f8", 12)])
def test_split_residues_are_canonical_and_exact(fam, n):
    rng = np.random.default_rng(n)
    X = torch.from_numpy(rng.standard_normal((300, 150)) * 10.0 ** rng.uniform(-8, 8, (300, 150)))
    f = ozaki2.FAMILIES[fam]
    r = ozaki2.split(X, n, fam, per="col")
    assert r.d.shape == (n, 384, 256) and r.d.dtype == f.residue_dtype
    D = r.d.double().numpy()
    assert not D[:, 300:].any() and not D[:, :, 150:].any()
    Xbar = np.rint(np.ldexp(X.numpy(), r.e[None, :]))
    T = ozaki2.bound(n, fam)
    assert (np.sqrt((Xbar**2).sum(axis=0)) <= T).all()
    for l, m in enumerate(f.moduli[:n]):
        R = D[l, :300, :150]
        assert np.abs(R).max() <= m // 2
        big = [int(x) for x in Xbar[:5].ravel()]  # exact, Python ints
        assert all((b - int(v)) % m == 0 for b, v in zip(big, R[:5].ravel(), strict=True))


def test_split_scaling_uses_the_bound():
    """The scale is the largest power of two within the Cauchy-Schwarz budget."""
    X = torch.randn(64, 40, dtype=torch.float64)
    r = ozaki2.split(X, 12, "b7", per="col")
    Xn = X.numpy()
    T = ozaki2.bound(12, "b7") - 0.5 * math.sqrt(64)
    nrm = np.sqrt((Xn**2).sum(axis=0))
    assert (np.ldexp(nrm, r.e) <= T).all()
    assert (np.ldexp(nrm, r.e + 1) > T * (1 - 1e-9)).all()


def test_split_zero_and_extreme():
    X = torch.zeros(4, 300, dtype=torch.float64)
    X[0] = torch.linspace(-1e300, 1e300, 300, dtype=torch.float64)
    X[1] = torch.linspace(-1e-300, 1e-300, 300, dtype=torch.float64)
    X[2, 5] = 5e-324
    r = ozaki2.split(X, 16, "b7", per="row")
    assert r.e[3] == 0 and not r.d[:, :, 3].float().any()
    assert np.isfinite(np.ldexp(1.0, -r.e[:3])).all()
    D = r.d.double().numpy()
    assert D[:, 5, 2].any()  # the lone subnormal keeps its digits


def test_residues_of_large_integers():
    rng = np.random.default_rng(3)
    x = np.ldexp(np.rint(rng.uniform(-(2**52), 2**52, 200)), rng.integers(0, 300, 200))
    for m, r in zip((127, 125, 511), ozaki2.residues_of(x, (127, 125, 511)), strict=True):
        assert all((int(v) - int(rr)) % m == 0 for v, rr in zip(x, r, strict=True))
        assert np.abs(r).max() <= m // 2


# ── residue GEMM: the kernel arithmetic ──────────────────────────────────────


@pytest.mark.parametrize("fam", KERNEL_FAMILIES)
@pytest.mark.parametrize("K", [100, 4096, 4097, 9000])
def test_group_close_arithmetic_is_exact(fam, K):
    """The kernel's op sequence (exact group sums, fp32 reductions at each
    group close, canonical form) equals the exact residues of the product,
    including operands at the residue bound and K not a multiple of a group."""
    f = ozaki2.FAMILIES[fam]
    rng = np.random.default_rng(K)
    A, B = (
        torch.from_numpy(rng.standard_normal((200, K))),
        torch.from_numpy(rng.standard_normal((K, 130))),
    )
    ra, rb = ozaki2.split_ab(A, B, 3, fam)
    at, bt = ra.d.clone(), rb.d.clone()
    at[:, :K, :128] = float(f.rmax)  # worst case: every product at the bound
    bt[:, :K, :2] = float(f.rmax)
    bt[:, :K, 2:4] = -float(f.rmax)
    ops = ozaki2.residue_gemm_ops(at, bt, fam)
    ref = ozaki2.reference_residue_gemm(at, bt, fam)
    assert np.array_equal(ops, ref)
    m0 = f.moduli[0]
    assert ref[0, 0, 0] == ozaki2._canon(np.mod(K * f.rmax**2, m0), m0)


def test_fp8_digits_and_products():
    f = ozaki2.FAMILIES["f8"]
    rng = np.random.default_rng(0)
    for p in f.moduli:
        h = p // 2
        ra = rng.integers(-h, h + 1, (6, 400)).astype(np.float64)
        rb = rng.integers(-h, h + 1, (400, 5)).astype(np.float64)
        ra[0], rb[:, 0] = -h, h
        d1, d2, s = ozaki2.fp8_split(ra, p)
        assert np.abs(d1).max() <= 16 and np.abs(d2).max() <= 16
        assert np.array_equal(s * d1 + d2, ra)
        if s * s != p:
            assert np.abs(d1 + d2).max() <= 16
        assert np.array_equal(
            ozaki2.fp8_product_mod(ra, rb, p), ozaki2._canon(np.mod(ra @ rb, p), p)
        )


# ── CRT ──────────────────────────────────────────────────────────────────────


def _crt_exact(res, c):
    """Symmetric CRT with Python ints, elementwise."""
    R, C = res.shape[1:]
    out = np.empty((R, C), dtype=object)
    for i in range(R):
        for j in range(C):
            x = sum(c.w[l] * int(res[l, i, j]) for l in range(c.n)) % c.M
            out[i, j] = x - c.M if x > c.M // 2 else x
    return out


@pytest.mark.parametrize("fam,n", [("b7", 4), ("b7", 16), ("b7", 28), ("b8", 14), ("b9", 12)])
def test_device_and_host_crt(fam, n):
    """Both CRTs recover X = Abar Bbar for |X| <= M / 4: the host one to fp64
    rounding, the device one exactly in its kept digits (dropped part below
    2^(2 beta j0 - 1)), bit for bit the kernel's op sequence."""
    c = ozaki2.crt(fam, n)
    rnd = random.Random(n)
    Mq = c.M // 4
    X = [rnd.randint(-Mq, Mq) >> rnd.randrange(0, 40) for _ in range(40 * 7)]
    X[:4] = [Mq, -Mq, 0, 1]
    X = np.array(X, dtype=object).reshape(40, 7)
    res = np.stack(
        [np.vectorize(lambda v, m=m: ozaki2._canon(np.int64(v % m), m))(X) for m in c.moduli]
    ).astype(np.float32)
    assert (_crt_exact(res, c) == X).all()
    host = ozaki2.crt_host(res, c)
    Xf = np.vectorize(float)(X)
    assert (np.abs(host - Xf) <= np.abs(Xf) * 2.0**-51).all()
    packed = ozaki2.crt_device(res, c)
    assert packed.shape == (c.keep, 40, 7)
    assert (np.abs(packed) < 2**24).all() and np.array_equal(packed, np.rint(packed))
    kept = np.zeros((40, 7), dtype=object)
    for j in range(c.keep - 1, -1, -1):
        kept = kept * 2 ** (2 * c.beta) + packed[j].astype(np.int64).astype(object)
    kept = kept * 2 ** (2 * c.beta * c.j0)
    drop = (X - kept).ravel()
    assert max(abs(int(d)) for d in drop) <= 2 ** (2 * c.beta * c.j0)
    dev = np.ldexp(ozaki2.unpack_device(packed, c), 2 * c.beta * c.j0)
    assert (np.abs(dev - Xf) <= np.abs(Xf) * 2.0**-51 + 2.0 ** (2 * c.beta * c.j0)).all()


# ── the host reference GEMM ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fam,n,tol", [("b7", 10, 2e-9), ("b7", 14, 2e-13), ("b7", 17, 8e-16), ("b8", 15, 8e-16)]
)
def test_accuracy_per_moduli(fam, n, tol):
    A, B = _inputs(200, 1500, 130, seed=1)
    assert _max_cw(ozaki2.reference_gemm(A, B, n, fam), A, B) <= tol


@pytest.mark.parametrize("crt_mode", ["device", "host"])
def test_fp64_accurate_at_default(crt_mode):
    A, B = _inputs(300, 5000, 241, seed=2)  # narrow: the C^T route, K over two groups
    host = _max_cw(A @ B, A, B)
    err = _max_cw(ozaki2.reference_gemm(A, B, crt_mode=crt_mode), A, B)
    assert err <= max(2 * host, 1e-15), (err, host)


def test_wide_range_data():
    A, B = _inputs(128, 3000, 150, dist="logu4", seed=3)
    host = _max_cw(A @ B, A, B)
    assert _max_cw(ozaki2.reference_gemm(A, B, 20), A, B) <= max(2 * host, 2e-15)


def test_huge_exponent_spread():
    """Rows and columns scaled over 1e+-150, entries over 1e+-8: per-row /
    per-column scaling absorbs the first, the moduli count the second."""
    A, B = _inputs(130, 700, 140, dist="normal", seed=4)
    A = A * torch.logspace(-150, 150, 130, dtype=torch.float64)[:, None]
    B = B * torch.logspace(150, -150, 140, dtype=torch.float64)[None, :]
    rng = np.random.default_rng(4)
    A = A * torch.from_numpy(10.0 ** rng.uniform(-8, 8, A.shape))
    C = ozaki2.reference_gemm(A, B, 24)
    assert torch.isfinite(C).all()
    assert _max_cw(C, A, B) <= 4 * _max_cw(A @ B, A, B) + 1e-15


def test_zeros_and_degenerate_shapes():
    A, B = _inputs(70, 200, 60, seed=5)
    A[3] = 0
    B[:, 7] = 0
    C = ozaki2.reference_gemm(A, B, 12)
    assert not C[3].any() and not C[:, 7].any()
    assert torch.equal(
        ozaki2.reference_gemm(torch.zeros(5, 9, dtype=torch.float64), B[:9], 12),
        torch.zeros(5, 60, dtype=torch.float64),
    )
    one = ozaki2.reference_gemm(
        torch.tensor([[3.0]], dtype=torch.float64), torch.tensor([[-0.5]], dtype=torch.float64), 4
    )
    assert one.item() == -1.5
    assert ozaki2.reference_gemm(A[:0], B, 8).shape == (0, 60)
    assert ozaki2.reference_gemm(A[:, :0], B[:0], 8).abs().sum() == 0


def test_bitwise_deterministic_and_tiling_invariant():
    """Every element is a function of its row's and column's integers only:
    repeated calls and any M / N blocking give the same bits."""
    A, B = _inputs(400, 2000, 600, seed=6)
    C1 = ozaki2.reference_gemm(A, B, 16)
    C2 = ozaki2.reference_gemm(A, B, 16)
    C3 = ozaki2.reference_gemm(A, B, 16, max_m=128, max_n=200)
    assert torch.equal(C1, C2) and torch.equal(C1, C3)
    # a sub-block computed alone is the same bits (per-row / per-column scaling)
    assert torch.equal(ozaki2.reference_gemm(A[100:228], B[:, 50:250], 16), C1[100:228, 50:250])


def test_host_and_device_crt_agree():
    A, B = _inputs(150, 800, 120, seed=7)
    d = ozaki2.reference_gemm(A, B, 16, crt_mode="device")
    h = ozaki2.reference_gemm(A, B, 16, crt_mode="host")
    den = A.abs() @ B.abs()
    assert ((d - h).abs() / den).max() <= 2e-16


def test_k_blocks_are_summed():
    A, B = _inputs(130, 1000, 70, seed=8)
    one = ozaki2.reference_gemm(A, B, 16)
    blk = ozaki2.reference_gemm(A, B, 16, limit=2 * 16 * 1024 * 256)  # 256-row K blocks
    assert ((one - blk).abs() / (A.abs() @ B.abs())).max() <= 4e-16


# ── #112 blocking ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "Mp,Kp,N,n", [(4096, 4096, 4096, 16), (128, 2**21, 241, 24), (8192, 3584, 241, 16)]
)
def test_plan_respects_byte_limits(Mp, Kp, N, n):
    mbs, kbs, nbs = ozaki2.plan(Mp, Kp, N, n)
    lim = ozaki.BLOCK_BYTES
    for _, mb in mbs:
        for _, kb in kbs:
            assert 2 * n * kb * mb <= lim
            for _, nb in nbs:
                assert 2 * n * kb * nb <= lim and 4 * n * mb * nb <= lim
                assert 4 * nb < 32 << 20  # residue rows far below 32 MiB
    assert sum(b for _, b in kbs) == Kp and sum(b for _, b in mbs) == Mp


# ── gemm_fp64 API ────────────────────────────────────────────────────────────


def test_gemm_fp64_scheme_option():
    A, B = _inputs(60, 90, 40, seed=9)
    assert torch.equal(trnblas.gemm_fp64(A, B, scheme="ozaki2"), A @ B)  # CPU: host fp64
    assert torch.equal(trnblas.gemm_fp64(A.T.contiguous(), B, transA=True, scheme="ozaki2"), A @ B)
    with pytest.raises(ValueError, match="scheme"):
        trnblas.gemm_fp64(A, B, scheme="ozaki3")
    with pytest.raises(ValueError, match="moduli"):
        trnblas.gemm_fp64(A, B, moduli=12)
    with pytest.raises(ValueError):
        trnblas.gemm_fp64(A, B, scheme="ozaki2", moduli=40)


def test_cost_model_counts():
    assert ozaki2.n_groups(4096, "b7") == 1 and ozaki2.n_groups(4097, "b7") == 2
    assert ozaki2.n_groups(4096, "b8") == 4 and ozaki2.n_groups(4096, "b9") == 16
    v = ozaki2.vector_ops(8400, 16, "b7")
    assert v["reduce"] == 16 * (4 * 3 + 3) and v["crt"] > 0


# ── the device split's arithmetic (#132) ─────────────────────────────────────


def _kinds(shape, seed):
    """normal, wide-range, raw bit patterns (every exponent, subnormal and
    zero elements, a zero row), and a sparse matrix."""
    rng = np.random.default_rng(seed)
    out = {"normal": rng.standard_normal(shape)}
    out["wide"] = rng.standard_normal(shape) * 10.0 ** rng.uniform(-30, 30, shape)
    bits = rng.integers(0, 2**63, shape, dtype=np.uint64)
    bits[:, 1::3] &= ~np.uint64(0x7FF0000000000000)
    X = bits.view(np.float64).copy()
    X[~np.isfinite(X)] = 1.0
    X[::2] *= -1
    X[3] = 0.0
    out["bits"] = X
    out["sparse"] = rng.standard_normal(shape) * (rng.random(shape) < 0.02)
    return out


@pytest.mark.parametrize("kind", ["normal", "wide", "bits", "sparse"])
def test_norm_bound_is_exact_and_rigorous(kind):
    """S = sum tau: every element's term bounds its square from above, the
    sum is the same for the matrix and its transpose (exact integers in any
    order), and zero elements add nothing."""
    X = _kinds((70, 333), 1)[kind]
    e, nz = ozaki2.exponents_of(X, 1)
    tau = ozaki2.norm_terms(X, e, 1)
    assert np.array_equal(tau, np.floor(tau)) and (tau >= 0).all() and tau.max() < 2**35
    assert (tau[X == 0] == 0).all()
    S = ozaki2.norm_sums(X, e, 1)
    assert np.array_equal(S, ozaki2.norm_sums(np.ascontiguousarray(X.T), e, 0))
    assert np.array_equal(S, tau.sum(axis=1).astype(np.int64))
    # sum_k x_k^2 2^(2 (20 - e)) <= 2^6 (S + K), checked exactly
    from fractions import Fraction

    for i in np.nonzero(nz)[0][:12]:
        lhs = sum(Fraction(float(v)) ** 2 for v in X[i]) * Fraction(2) ** (2 * (20 - int(e[i])))
        assert lhs <= 64 * (int(S[i]) + X.shape[1])


def _scale_fp64_rule(X, axis, T):
    """The fp64-norm scaling the exact bound replaced (#132 prototype)."""
    K = X.shape[axis]
    amax = np.abs(X).max(axis=axis)
    e = np.frexp(amax)[1].astype(np.int64)
    Y = np.ldexp(X, -e[:, None] if axis == 1 else -e[None, :])
    nrm = np.sqrt((Y * Y).sum(axis=axis)) * (1 + 2.0**-30)
    with np.errstate(divide="ignore"):
        ex = np.frexp((T - 0.5 * math.sqrt(K)) / np.where(nrm > 0, nrm, 1.0))[1]
    return np.where(amax > 0, ex.astype(np.int64) - 1 - e, 0)


@pytest.mark.parametrize("kind", ["normal", "wide", "sparse"])
def test_scale_matches_the_fp64_norm_rule(kind):
    """The exact bound is within ~2^-10 of the fp64 norm's square, so the
    exponents equal the fp64-norm rule's but within that of a power-of-two
    boundary, never above it; and the scaled rows fit: ||rint(x 2^mu)|| <= T."""
    X = _kinds((2000, 1024), 2)[kind]
    T = ozaki2.bound(17)
    mu = ozaki2.scale_exponents(X, 1, T)
    old = _scale_fp64_rule(X, 1, T)
    assert (mu <= old).all() and (mu != old).mean() < 2e-3
    Xb = np.rint(np.ldexp(X, mu[:, None]))
    assert (np.sqrt((Xb * Xb).sum(axis=1)) <= T).all()


@pytest.mark.parametrize("fam,n", [("b7", 13), ("b7", 17), ("b7", 24), ("b8", 14), ("b9", 13)])
@pytest.mark.parametrize("kind", ["normal", "wide", "bits", "sparse"])
@pytest.mark.parametrize("per", ["row", "col"])
def test_split_from_bits_is_the_host_split(fam, n, kind, per):
    """The device split's arithmetic (exponents, exact norm sums, Ozaki I
    digits on the scaled grid, residues from digits) gives the residues and
    exponents of the fp64 host split, bit for bit."""
    X = _kinds((150, 300), n)[kind]
    got, mu = ozaki2.split_from_bits(X, n, fam, per)
    r = ozaki2.split(torch.from_numpy(X), n, fam, per=per)
    ref = r.d.float().numpy()
    assert np.array_equal(got, ref.transpose(0, 2, 1) if per == "row" else ref)
    assert np.array_equal(mu, r.e)


def test_scaled_digits_round_half_even():
    """Ozaki I digits with row exponents e' = 6 + 7 (k - 1) - mu are
    RNE(x 2^mu) on the integer grid, ties included."""
    rng = np.random.default_rng(4)
    k, R = 9, 20
    mu = rng.integers(-40, 40, R)
    J = rng.integers(-(2**40), 2**40, (R, 64)).astype(np.float64)
    X = np.ldexp(J + rng.choice([0.0, 0.5, -0.5, 0.25, 0.75], J.shape), -mu[:, None])
    X[:, 0] = np.ldexp(2.0**57 - 1, -mu)  # a full-width row maximum
    e = ozaki2.exponents_of(X, 1)[0]
    ep = 6 + 7 * (k - 1) - mu
    assert (ep >= e).all()
    D, _ = ozaki.split_from_bits(X, k, e=ep)
    val = sum(D[s].astype(np.float64) * 2.0 ** (7 * (k - 1 - s)) for s in range(k))
    assert np.array_equal(val, np.rint(np.ldexp(X, mu[:, None])))


@pytest.mark.parametrize("fam", KERNEL_FAMILIES)
@pytest.mark.parametrize("k", range(1, 13))
def test_residue_terms_stay_exact(fam, k):
    """The digit dot product's partial sums stay below PAIR_LIMIT (< 2^22), so
    one rint of t fl(1/m) gives the balanced residue."""
    f = ozaki2.FAMILIES[fam]
    pairs, terms = ozaki2.residue_terms(k, max(f.moduli))
    assert sorted(lo for _, lo, _ in terms) == sorted([lo for _, lo, _ in terms if lo is not None])
    covered = sorted(s for hi, lo, _ in terms for s in (hi, lo) if s is not None)
    assert covered == list(range(k))
    per = (64 * 128 + 64) if pairs else 64
    assert len(terms) * per * (max(f.moduli) // 2) < ozaki2.PAIR_LIMIT < 2**22


def test_residues_from_digits_at_the_bounds():
    """Every digit at +-64 (the largest balanced digits): residues equal the
    integer's, for the largest moduli of every family."""
    for fam in KERNEL_FAMILIES:
        mods = ozaki2.FAMILIES[fam].moduli[:6]
        for k in (1, 8, 9, 12):
            D = np.array([[64.0], [-64.0], [64.0]] * 4, dtype=np.float32)[:k]
            D = np.concatenate([D, -D], axis=1)
            got = ozaki2.residues_from_digits(D, mods)
            for col in range(2):
                X = sum(int(D[s, col]) << (7 * (k - 1 - s)) for s in range(k))
                assert [int(v) for v in got[:, col]] == [ozaki2._canon(X % m, m) for m in mods]


def test_digits_for_covers_the_rows():
    mu = np.array([10, -5, 0])
    e = np.array([40, 61, 0])
    k = ozaki2.digits_for(mu, e, np.array([True, True, False]))
    assert 6 + 7 * (k - 1) >= 56 > 6 + 7 * (k - 2)


# ── the combine: RN64 of the packed CRT output (#132) ────────────────────────


def _packed_of(X: int, K: int) -> np.ndarray:
    """Balanced 24-bit packed values of the integer X (|X| < 2^(24 K - 1))."""
    out = []
    for _ in range(K - 1):
        lo = X & ((1 << 24) - 1)
        if lo >= 1 << 23:
            lo -= 1 << 24
        out.append(lo)
        X = (X - lo) >> 24
    out.append(X)
    return np.array(out, dtype=np.float32)


def _rn64(X: int, E: int):
    """RN64(X 2^E), or None outside the normal float64 range."""
    from fractions import Fraction

    v = Fraction(X) * Fraction(2) ** E
    if X and not Fraction(2) ** -1022 <= abs(v) < Fraction(2) ** 1024:
        return None
    try:
        return float(v)
    except OverflowError:  # rounds up to 2^1024
        return None


def _special_integers(K):
    rnd = random.Random(K)
    top = 24 * K - 2
    xs = [0, 1, -1, (1 << 53) + 1, (1 << 53) + 3, (1 << 54) - 1, -((1 << 54) - 1)]
    for b in range(54, top, 7):  # ties, ties broken by a low bit, carries into the exponent
        xs += [(1 << b) + (1 << (b - 53)), (1 << b) + (1 << (b - 53)) + 1, (1 << (b + 1)) - 1]
        xs += [-((1 << b) + (3 << (b - 53))), (1 << b) - (1 << (b - 54))]
    xs += [rnd.randrange(-(1 << top), 1 << top) >> rnd.randrange(0, top) for _ in range(400)]
    return xs


@pytest.mark.parametrize("K", [3, 4, 5, 6])
def test_packed_to_fp64_bits_is_rn64(K):
    """The device combine's arithmetic is the correctly rounded float64 of
    the packed integer times 2^E: ties to even, sticky bits far below,
    carries into the exponent, negative values, zero; out-of-range results
    are flagged."""
    xs = _special_integers(K)
    P = np.stack([_packed_of(x, K) for x in xs], axis=1)
    rnd = random.Random(K + 1)
    E = [rnd.randrange(-400, 400) for _ in xs]
    bits, flag = ozaki2.packed_to_fp64_bits(P, np.array(E, np.float32) + ozaki2.combine_offset(K))
    got = bits.view(np.float64)
    for i, (x, e) in enumerate(zip(xs, E, strict=True)):
        ref = _rn64(x, e)
        if flag[i]:
            assert ref is None or abs(ref) >= 2.0**1023, (x, e)  # range, or a carry to 2^1024
            continue
        assert ref is not None and got[i] == ref and np.signbit(got[i]) == (x < 0), (x, e)
    big = ozaki2.packed_to_fp64_bits(P[:, 7:9], np.float32(1000 + ozaki2.combine_offset(K)))[1]
    assert big.all()


@pytest.mark.parametrize("fam,n", [("b7", 13), ("b7", 17), ("b7", 28), ("b8", 20), ("b9", 13)])
def test_combine_packed_is_rn64_of_the_crt_integer(fam, n):
    """`combine_packed` (the host reference of the combine, any packed
    width) is RN64(X') 2^(shift - eA - eB), X' the kept CRT integer: a
    subnormal or overflowing result is ldexp of RN64(X')."""
    c = ozaki2.crt(fam, n)
    rng = np.random.default_rng(n)
    R, N = 16, 9
    res = np.stack([rng.integers(-(m // 2), m // 2 + 1, (R, N)) for m in c.moduli])
    res[:, :4] = np.stack([ozaki2._canon(rng.integers(-99, 99, (4, N)) % m, m) for m in c.moduli])
    res = res.astype(np.float32)
    P = ozaki2.crt_device(res, c)
    eA, eB = rng.integers(-200, 200, R), rng.integers(-200, 200, N)
    eA[5], eA[6] = -900, 1200  # overflow; gradual underflow
    shift = 2 * c.beta * c.j0
    with np.errstate(all="ignore"):
        got = ozaki2.combine_packed(P, eA, eB, c, R, N, shift)
    g = 2 * c.beta
    for i in range(R):
        for j in range(N):
            X = sum(int(P[p, i, j]) << (g * p) for p in range(c.keep))
            E = shift - int(eA[i]) - int(eB[j])
            with np.errstate(all="ignore"):
                ref = np.ldexp(_rn64(X, 0), E)  # exact in range; the host's rule outside
            assert got[i, j] == ref or (np.isnan(ref) and np.isnan(got[i, j])), (i, j)
            if _rn64(X, E) is not None:
                assert got[i, j] == _rn64(X, E)


def test_combine_kernel_reference_layout():
    c = ozaki2.crt("b7", 17)
    rng = np.random.default_rng(5)
    res = np.stack([rng.integers(-(m // 2), m // 2 + 1, (256, 30)) for m in c.moduli])
    P = ozaki2.crt_device(res.astype(np.float32), c)
    ea = np.full((256, 1), ozaki2.combine_offset(c.keep), np.float32)
    ea[130] += 5000  # out of range: row 130 is flagged (row tile 1, partition 2)
    eb = np.zeros((1, 30), np.float32)
    out = ozaki2.combine_kernel_reference(P, ea, eb)
    assert out.shape == (384, 60)
    flags = out[256:, :2].T.reshape(-1)
    assert flags[130] == 1 and flags.sum() == 1 and not out[256:, 2:].any()
    bits, _ = ozaki2.packed_to_fp64_bits(P[:, :1], ea[:1] + eb)
    assert np.array_equal(out[0].view(np.uint64), bits[0])


# ── the all-device route on the host (#132) ──────────────────────────────────


@pytest.fixture
def native2(monkeypatch):
    """Native path on CPU with every Ozaki II op a CPU stand-in of its
    kernel (`_ozaki_ref.ozaki2_kernel_refs`, the residue GEMM / CRT host
    references, the Ozaki I exponent passes): gemm_fp64(scheme="ozaki2")
    runs its whole device route. The list records the op names."""
    from _ozaki_ref import DEVICE_SPLIT_KERNELS, device_split_ops, ozaki2_kernel_refs

    from trnblas.nki import dispatch

    calls: list = []

    def rec(name, fn):
        def op(*a):
            calls.append(name)
            return fn(*a)

        return op

    ops = {k: rec(k, v) for k, v in {**device_split_ops(), **ozaki2_kernel_refs()}.items()}
    for fam in KERNEL_FAMILIES:
        ops[f"ozaki2_gemm_{fam}"] = rec(
            f"ozaki2_gemm_{fam}",
            lambda at, b, f=fam: torch.from_numpy(ozaki2.reference_residue_gemm(at, b, f)),
        )
        ops[f"ozaki2_crt_{fam}"] = rec(
            f"ozaki2_crt_{fam}",
            lambda r, f=fam: torch.from_numpy(
                ozaki2.crt_device(r.numpy(), ozaki2.crt(f, r.shape[0]))
            ),
        )
        ops[f"ozaki2_fused_{fam}"] = rec(
            f"ozaki2_fused_{fam}",
            lambda at, b, f=fam: torch.from_numpy(
                ozaki2.crt_device(
                    ozaki2.reference_residue_gemm(at, b, f), ozaki2.crt(f, at.shape[0])
                )
            ),
        )
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(dispatch, "_native_ops", ops)
    monkeypatch.setattr(dispatch, "_OZ_DEVICE_COMBINE", True)
    for name in DEVICE_SPLIT_KERNELS:
        monkeypatch.setattr(dispatch, name, object(), raising=False)
    return calls


@pytest.mark.parametrize(
    "transA,transB", [(False, False), (True, False), (False, True), (True, True)]
)
def test_device_route_layouts(native2, transA, transB):
    """Each operand is split on the device in its own layout (per row for a
    row-major A or B^T, per column for A^T or a row-major B), the residue
    GEMM and CRT run fused, the combine on the device: bitwise the host
    reference."""
    A, B = _inputs(150, 260, 140, seed=11)
    a = A.T.contiguous() if transA else A
    b = B.T.contiguous() if transB else B
    C = trnblas.gemm_fp64(a, b, transA=transA, transB=transB, scheme="ozaki2")
    assert torch.equal(C, ozaki2.reference_gemm(A, B))
    per_row = ["ozaki_emax", "ozaki2_norm", "ozaki2_split_b7"]
    per_col = ["ozaki_emax_col", "ozaki2_norm_col", "ozaki2_split_col_b7"]
    want_a = per_col if transA else per_row
    want_b = per_row if transB else per_col
    assert native2[:6] == want_a + want_b
    assert native2[6:] == ["ozaki2_fused_b7", "ozaki2_combine"]


@pytest.mark.parametrize("shape", [(300, 200, 700), (700, 130, 90), (1, 5, 1)])
def test_device_route_bands(native2, monkeypatch, shape):
    monkeypatch.setattr(ozaki, "LAUNCH_M", 128)
    monkeypatch.setattr(ozaki, "LAUNCH_N", 256)
    A, B = _inputs(*shape, seed=sum(shape))
    C = trnblas.gemm_fp64(A, B, scheme="ozaki2", moduli=15)
    assert torch.equal(C, ozaki2.reference_gemm(A, B, 15, max_m=128, max_n=256))
    M, _, N = shape
    rows = N if (N < ozaki.SWAP_N and M > N) else M
    assert native2.count("ozaki2_combine") == -(-ozaki._round_up(rows, 128) // 128)


def test_device_route_fallbacks(native2):
    from trnblas.nki import ozaki2_kernels as K2

    A, B = _inputs(40, 60, 30, seed=12)
    A[7] = 1e-310  # a subnormal row maximum: A takes the host split
    assert torch.equal(trnblas.gemm_fp64(A, B, scheme="ozaki2"), ozaki2.reference_gemm(A, B))
    assert native2[:2] == ["ozaki_emax", "ozaki_emax_col"]
    native2.clear()
    A, B = _inputs(40, 60, 30, seed=13)
    B[5, 6] = float("inf")  # found by the exponent pass: host fp64
    C = trnblas.gemm_fp64(A, B, scheme="ozaki2")
    assert torch.equal(C.isnan(), (A @ B).isnan()) and "ozaki2_fused_b7" not in native2
    native2.clear()
    A, B = _inputs(40, 60, 30, seed=14)  # b8 at 32 moduli: 18 digits, the host split
    assert torch.equal(K2.gemm(A, B, 32, "b8"), ozaki2.reference_gemm(A, B, 32, "b8"))
    assert "ozaki2_split_b8" not in native2 and "ozaki2_norm" in native2
    native2.clear()
    for kw, gone in (({"device_combine": False}, "ozaki2_combine"), ({}, None)):
        C = K2.gemm(A, B, 13, "b8", **kw)
        assert torch.equal(C, ozaki2.reference_gemm(A, B, 13, "b8"))
        assert gone is None or gone not in native2
    native2.clear()
    C = K2.gemm(A, B, 20, "b8")  # 2 beta = 22: the combine on the host
    assert torch.equal(C, ozaki2.reference_gemm(A, B, 20, "b8"))
    assert "ozaki2_combine" not in native2 and "ozaki2_split_b8" in native2
    native2.clear()
    C = K2.gemm(A, B, 13, "b7", "host")
    assert torch.equal(C, ozaki2.reference_gemm(A, B, 13, "b7", "host"))
    assert "ozaki2_combine" not in native2 and "ozaki2_gemm_b7" in native2


def test_device_split_off(native2, monkeypatch):
    from trnblas.nki import ozaki2_kernels as K2

    monkeypatch.setattr(K2, "_OZ2_DEVICE_SPLIT", False)
    A, B = _inputs(40, 60, 30, seed=15)
    assert torch.equal(trnblas.gemm_fp64(A, B, scheme="ozaki2"), ozaki2.reference_gemm(A, B))
    assert native2 == ["ozaki2_fused_b7", "ozaki2_combine"]
