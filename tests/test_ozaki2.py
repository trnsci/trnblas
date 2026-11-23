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
