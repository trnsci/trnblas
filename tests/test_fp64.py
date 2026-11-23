"""CPU tests of `trnblas.fp64` (trnsci/trnsci#169): float-float arithmetic,
the fixed-order reductions, the integer-emulated exact path, the combine
step, the probe logic, and the edge-case policies."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from trnblas import fp64 as F
from trnblas import ozaki
from trnblas.nki import fp64_kernels as K
from trnblas.nki import fp64_probe as Q

RNG = np.random.default_rng(1234)


def _wide(n, lo=-30, hi=30, rng=RNG):
    m = rng.uniform(1.0, 2.0, n) * rng.choice([-1.0, 1.0], n)
    return np.ldexp(m, rng.integers(lo, hi, n))


def _f32(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _exact_dot(x, y) -> float:
    return float(sum((Fraction(a) * Fraction(b) for a, b in zip(x, y, strict=True)), Fraction(0)))


def _exact_sum(x) -> float:
    return math.fsum(np.asarray(x, dtype=np.float64).tolist())


# ── conversion ───────────────────────────────────────────────────────────────


def test_split_bits_is_the_canonical_double_word():
    x = np.concatenate([_wide(20000, -125, 127), _wide(2000, -100, -80), [1.0, -2.0, 3.0]])
    h, l = F.split_bits(x)
    h2 = x.astype(np.float32)
    l2 = (x - h2.astype(np.float64)).astype(np.float32)
    assert np.array_equal(h.view(np.uint32), h2.view(np.uint32))
    assert np.array_equal(l.view(np.uint32), l2.view(np.uint32))
    big = np.abs(x) >= F.FF_MIN_FULL  # below it lo is subnormal: documented loss
    err = np.abs(x - h.astype(np.float64) - l.astype(np.float64)) / np.abs(x)
    assert err[big].max() <= F.U_FF
    assert np.all(np.abs(l[big]) <= np.spacing(np.abs(h[big])) / 2)


def test_split_bits_round_half_even_ties():
    # x = 1 + 2^-24 (tie, even -> 1), 1 + 3 2^-24 (tie, -> 1 + 2^-22)
    x = np.array([1 + 2.0**-24, 1 + 3 * 2.0**-24, -(1 + 2.0**-24), 2 - 2.0**-25])
    h, l = F.split_bits(x)
    assert h.tolist() == [1.0, 1 + 2.0**-22, -1.0, 2.0]
    assert (h.astype(np.float64) + l).tolist() == x.tolist()


def test_split_bits_specials_and_range():
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 2.0**-130, 1e-45, 5e-324])
    h, l = F.split_bits(x)
    assert np.signbit(h[1]) and h[0] == 0 and l[0] == 0
    assert np.isposinf(h[2]) and np.isneginf(h[3]) and np.isnan(h[4])
    assert (l[2:5] == 0).all()
    assert h[5] == np.float32(2.0**-130)  # subnormal hi, gradual
    with pytest.raises(OverflowError):
        F.split_bits(np.array([1.0, 2.0**128]))
    with pytest.raises(OverflowError):
        F.split_bits(np.array([float(np.finfo(np.float32).max) * (1 + 2**-24)]))
    F.split_bits(np.array([float(np.finfo(np.float32).max)]))  # in range
    with pytest.raises(ValueError):
        F.split_bits(np.array([1.0, 1e-35]), underflow="raise")


def test_from_to_fp64_roundtrip():
    x = torch.from_numpy(_wide(5000))
    f = F.from_fp64(x)
    assert f.hi.dtype == torch.float32 and f.lo.dtype == torch.float32
    back = F.to_fp64(f)
    assert float(((back - x).abs() / x.abs()).max()) <= F.U_FF
    with pytest.raises(TypeError):
        F.from_fp64(x.float())
    e = np.arange(5000) % 7 - 3
    assert torch.equal(F.to_fp64(f, exp=e), torch.from_numpy(np.ldexp(back.numpy(), e)))


# ── error-free transformations ───────────────────────────────────────────────


def test_two_sum_and_two_prod_are_exact():
    a = _f32(_wide(200000, -50, 50))
    b = _f32(_wide(200000, -50, 50))
    s, e = F.two_sum(a, b)
    assert torch.equal(s.double() + e.double(), a.double() + b.double())
    p, f = F.two_prod(a, b)
    assert torch.equal(p.double() + f.double(), a.double() * b.double())
    p2, f2 = F.two_prod_fma(a, b)
    assert torch.equal(p, p2) and torch.equal(f, f2)
    h, l = F.split(a)
    assert torch.equal(h + l, a)
    # 12-bit halves: products of halves are exact in fp32
    assert torch.equal((h * h).double(), h.double() * h.double())


def test_two_prod_domain_edges():
    a = _f32(np.ldexp(RNG.uniform(1, 2, 1000), 114))
    b = _f32(np.ldexp(RNG.uniform(1, 2, 1000), -110))
    p, f = F.two_prod(a, b)
    assert torch.equal(p.double() + f.double(), a.double() * b.double())
    # beyond SPLIT_MAX the split overflows: documented, not exact
    a = _f32(np.ldexp(RNG.uniform(1, 2, 100), 120))
    p, f = F.two_prod(a, _f32(np.full(100, 0.5)))
    assert not torch.isfinite(f).all()


def test_double_word_ops_error_bounds():
    x = F.from_fp64(torch.from_numpy(_wide(50000, -20, 20)))
    y = F.from_fp64(torch.from_numpy(_wide(50000, -20, 20)))
    xv = F.to_fp64(x).numpy()
    yv = F.to_fp64(y).numpy()
    u2 = 2.0**-48  # u^2 for fp32's u = 2^-24; bounds of Joldes et al. (2017)

    def rel(got, ref):
        return np.abs(F.to_fp64(got).numpy() - ref) / np.abs(ref)

    exact_add = [
        float(Fraction(a) + Fraction(b)) for a, b in zip(xv[:2000], yv[:2000], strict=True)
    ]
    sub = F.FF(x.hi[:2000], x.lo[:2000]), F.FF(y.hi[:2000], y.lo[:2000])
    slack = 2.0**-53  # the Fraction reference is rounded to float64
    assert rel(F.ff_add(*sub), np.array(exact_add)).max() <= 3 * u2 + slack
    exact_mul = [
        float(Fraction(a) * Fraction(b)) for a, b in zip(xv[:2000], yv[:2000], strict=True)
    ]
    assert rel(F.ff_mul(*sub), np.array(exact_mul)).max() <= 7 * u2 + slack
    b = y.hi[:2000]
    exact_mfp = [
        float(Fraction(a) * Fraction(float(c))) for a, c in zip(xv[:2000], b.tolist(), strict=True)
    ]
    assert rel(F.ff_mul_fp(sub[0], b), np.array(exact_mfp)).max() <= 3 * u2 + slack
    exact_afp = [
        float(Fraction(a) + Fraction(float(c))) for a, c in zip(xv[:2000], b.tolist(), strict=True)
    ]
    assert rel(F.ff_add_fp(sub[0], b), np.array(exact_afp)).max() <= 2 * u2 + slack


# ── layout ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [0, 1, 127, 128, 1000, 100003, 262144, 262145, 10**6, 10**7])
def test_plan(n):
    pl = F.plan(n)
    assert F.NCH * pl.tpc * pl.W == pl.F and pl.W % F.W_ALIGN == 0
    assert pl.W <= F.W_MAX and n <= F.P * pl.F
    assert F.plan_from_cols(pl.F) == (pl.W, pl.tpc)
    assert K._fv_plan(pl.F) == (pl.W, pl.tpc)
    if n >= 10**5:
        assert 1.07 * n >= F.P * pl.F  # padding stays small
    for m in range(max(1, n - 3), n + 4):  # plan_from_cols inverts plan
        q = F.plan(m)
        assert F.plan_from_cols(q.F) == (q.W, q.tpc)


def test_kernel_constants_match():
    assert (K._NCH, K._W_MAX, K._SPLITTER) == (F.NCH, F.W_MAX, F.SPLITTER)


# ── reductions ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 1000, 100003, 600001])
def test_dot_arms_accuracy(n):
    x = _f32(RNG.standard_normal(n))
    y = _f32(RNG.standard_normal(n))
    p = x.double() * y.double()  # exact products
    ref = math.fsum(p.tolist())
    absum = float(p.abs().sum())
    bounds = {"plain": 2e-6, "comp": 1e-12, "sloppy": 1e-13, "accurate": 1e-13}
    for arm, bound in bounds.items():
        got = float(F.dot(x, y, accumulate=arm))
        assert abs(got - ref) <= bound * absum, arm


@pytest.mark.parametrize("n", [1, 999, 100003, 600001])
def test_sum_and_nrm2(n):
    x = _f32(RNG.standard_normal(n) * 10.0 ** RNG.integers(-5, 5, n))
    ref = math.fsum(x.double().tolist())
    absum = float(x.double().abs().sum())
    for arm in F.ARMS:
        got = float(F.sum(x, accumulate=arm))
        assert abs(got - ref) <= (1e-6 if arm == "plain" else 1e-13) * absum, arm
    nr = math.sqrt(math.fsum((x.double() ** 2).tolist()))
    assert abs(float(F.nrm2(x)) / nr - 1) < 1e-14


def test_nrm2_extreme_range():
    big = _f32(np.ldexp(RNG.uniform(1, 2, 5000), 120))  # squares overflow fp32
    tiny = _f32(np.ldexp(RNG.uniform(1, 2, 5000), -140))  # subnormal
    for x in (big, tiny, torch.cat([big, tiny])):
        ref = math.sqrt(math.fsum([float(Fraction(v) ** 2) * 1.0 for v in []]) or 0.0)
        xd = x.double().numpy()
        s = np.abs(xd).max()
        ref = s * math.sqrt(math.fsum(((xd / s) ** 2).tolist()))
        assert abs(float(F.nrm2(x)) / ref - 1) < 1e-14
    assert float(F.nrm2(torch.zeros(1000))) == 0.0


def test_dot_of_float64_and_ff_operands():
    x = torch.from_numpy(_wide(20000, -10, 10))
    y = torch.from_numpy(_wide(20000, -10, 10))
    ref = _exact_dot(x.tolist(), y.tolist())
    absum = float((x * y).abs().sum())
    for arm in ("accurate", "sloppy"):
        got = float(F.dot(x, y, accumulate=arm))
        assert abs(got - ref) <= 4 * F.U_FF * absum
    assert float(F.dot(F.from_fp64(x), F.from_fp64(y))) == float(F.dot(x, y))
    s = float(F.sum(x))
    assert abs(s - _exact_sum(x.numpy())) <= 2 * F.U_FF * float(x.abs().sum())
    assert abs(float(F.nrm2(x)) / float(torch.linalg.vector_norm(x)) - 1) < 1e-14


def test_reduction_is_deterministic_and_chunk_local():
    """The LNC stand-in: chunk c's partials depend only on chunk c's data,
    which is what the programs split at LNC=2, so any split gives the same
    bits; and repeated runs are bitwise equal."""
    n = 700001
    x = _f32(RNG.standard_normal(n))
    y = _f32(RNG.standard_normal(n))
    full = F.reference_partials("dot32", [x, y])
    assert torch.equal(full, F.reference_partials("dot32", [x, y]))
    pl = F.plan(n)
    X = F.layout(x, pl).view(F.P, F.NCH, -1)
    Y = F.layout(y, pl).view(F.P, F.NCH, -1)
    for c in range(F.NCH):
        Xc = torch.zeros_like(X)
        Xc[:, c] = X[:, c]
        part = F.reference_partials("dot32", [Xc.reshape(-1), Y.reshape(-1)])
        assert torch.equal(part[:, 2 * c : 2 * c + 2], full[:, 2 * c : 2 * c + 2])


def test_finishes_agree():
    x = _f32(RNG.standard_normal(300000))
    h = float(F.sum(x))
    f = F.sum(x, finish="ff")
    assert isinstance(f, F.FF)
    assert abs(float(F.to_fp64(f)) - h) <= 1e-13 * float(x.abs().sum())
    parts = F.sum(x, finish="partials")
    assert parts.shape == (F.P, 2 * F.NCH)
    with pytest.raises(ValueError):
        F.nrm2(x, accumulate="nope")
    with pytest.raises(ValueError):
        F.sum(x, finish="nope")


def test_nonfinite_policy():
    x = torch.ones(1000)
    for bad in (math.inf, -math.inf, math.nan):
        x2 = x.clone()
        x2[17] = bad
        for arm in F.ARMS:
            assert math.isnan(float(F.sum(x2, accumulate=arm)))
        assert math.isnan(float(F.dot(x2, x)))
        assert math.isnan(float(F.nrm2(x2)))
    # an fp32-range overflow of the running sum is non-finite too
    big = torch.full((4096,), float(np.finfo(np.float32).max))
    assert math.isnan(float(F.sum(big)))


def test_zero_and_tiny_vectors():
    assert float(F.sum(torch.zeros(0))) == 0.0
    assert float(F.dot(torch.zeros(5), torch.zeros(5))) == 0.0
    t = _f32([2.0**-140, 2.0**-149, -(2.0**-145)])
    assert float(F.sum(t)) == 2.0**-140 + 2.0**-149 - 2.0**-145


def test_input_validation():
    with pytest.raises(TypeError):
        F.dot(torch.ones(3, dtype=torch.int32), torch.ones(3, dtype=torch.int32))
    with pytest.raises(ValueError):
        F.dot(torch.ones(3), torch.ones(4))
    with pytest.raises(ValueError):
        F.dot(torch.ones(3), torch.ones(3), backend="cuda")


# ── axpy, combine ────────────────────────────────────────────────────────────


def test_axpy():
    x = torch.from_numpy(_wide(3000, -5, 5))
    y = torch.from_numpy(_wide(3000, -5, 5))
    alpha = 1 / 3
    z = F.to_fp64(F.axpy(alpha, x, y)).numpy()
    ref = np.array(
        [
            float(Fraction(alpha) * Fraction(a) + Fraction(b))
            for a, b in zip(x.tolist(), y.tolist(), strict=True)
        ]
    )
    scale = np.abs(alpha * x.numpy()) + np.abs(y.numpy())
    assert (np.abs(z - ref) / scale).max() <= 4 * F.U_FF
    zf = F.axpy(alpha, x.float(), y.float())
    ref32 = alpha * x.float().double() + y.float().double()
    assert (F.to_fp64(zf) - ref32).abs().max() <= 4 * F.U_FF * float((ref32.abs() + 1).max())


def _ozaki_parts(M=64, K=1024, N=48, kb=4, slices=8):
    rng = np.random.default_rng(5)
    A = torch.from_numpy(rng.standard_normal((M, K)))
    B = torch.from_numpy(rng.standard_normal((K, N)))
    da, db = ozaki.split_a(A, slices), ozaki.split_b(B, slices)
    at = ozaki.kmajor(da.d, da.kmajor)
    bt = ozaki.kmajor(db.d, db.kmajor)
    step = at.shape[1] // kb
    parts = [
        torch.from_numpy(ozaki.reference_from_digits(at[:, i : i + step], bt[:, i : i + step]))
        for i in range(0, at.shape[1], step)
    ]
    off = ozaki.exponent_offset(slices)
    exp = da.e[:M, None] + db.e[None, :N] + off
    return A, B, parts, exp, da, db, slices


def test_combine_single_part_equals_ozaki_combine():
    A, B, parts, exp, da, db, k = _ozaki_parts(kb=1)
    M, N = A.shape[0], B.shape[1]
    ours = F.combine([F.stacked(parts[0][:, :M, :N])], exp=exp)
    theirs = ozaki.combine(parts[0], da.e, db.e, k, M, N)
    assert torch.equal(ours, theirs)


def test_combine_multi_part_and_exact():
    A, B, parts, exp, *_ = _ozaki_parts(kb=4)
    M, N = A.shape[0], B.shape[1]
    ff = [F.stacked(p[:, :M, :N]) for p in parts]
    got = F.combine(ff, exp=exp)
    ex = F.combine_exact(ff, exp=exp)
    host = sum(F.to_fp64(f, exp) for f in ff)  # today's host fp64 accumulation
    ref = A @ B
    scale = (A.abs() @ B.abs()).numpy()
    for r in (got, ex, host):
        assert (np.abs((r - ref).numpy()) / scale).max() < 1e-15
    assert (np.abs((got - ex).numpy()) / scale).max() < 1e-14
    assert F.combine([torch.ones(3)], out="ff").hi.tolist() == [1.0, 1.0, 1.0]


# ── integer-emulated exact path ──────────────────────────────────────────────


def test_exact_sum_is_exactly_rounded():
    for x in (
        _wide(3000, -300, 300),
        np.concatenate([_wide(1000), -_wide(1000)]),
        np.array([1e300, 1.0, -1e300, 2.0**-1074]),
        np.array([2.0**-1074] * 7),
        np.zeros(10),
    ):
        got = float(F.exact_sum(torch.from_numpy(x)))
        assert got == _exact_sum(x), x[:3]
    with pytest.raises(ValueError):
        F.exact_sum(torch.tensor([1.0, math.inf], dtype=torch.float64))


def test_exact_sum_truncated_window_bound():
    x = _wide(5000, -60, 0)
    b = F._bits(x)
    L = 3
    got = float(F.exact_sum(torch.from_numpy(x), limbs=L))
    assert abs(got - _exact_sum(x)) <= len(x) * 2.0 ** (b.top - 21 * L) + abs(got) * 2**-53


def test_exact_sum_long_vector():
    x = RNG.standard_normal(10**6) * 10.0 ** RNG.integers(-8, 8, 10**6)
    assert float(F.exact_sum(torch.from_numpy(x))) == _exact_sum(x)


def test_exact_dot_is_exactly_rounded():
    x = _wide(400, -40, 40)
    y = _wide(400, -40, 40)
    assert float(F.exact_dot(torch.from_numpy(x), torch.from_numpy(y))) == _exact_dot(x, y)
    # catastrophic cancellation: x . y = 1 exactly (cond ~ 1e32)
    x = np.array([1e16, 1.0, -1e16, 3.0])
    y = np.array([1e16, 1.0, 1e16, 0.0])
    assert float(F.exact_dot(torch.from_numpy(x), torch.from_numpy(y))) == 1.0


def test_exact_limbs_stay_in_int32():
    # worst case magnitudes: every digit at its maximum, the longest lanes
    x = np.full(600000, np.nextafter(2.0, 0.0))
    assert float(F.exact_sum(torch.from_numpy(x))) == _exact_sum(x)
    assert float(F.exact_dot(torch.from_numpy(x[:300000]), torch.from_numpy(x[:300000]))) == (
        _exact_dot(x[:2], x[:2]) * 150000
    )


# ── probe logic ──────────────────────────────────────────────────────────────


def test_e0_probe_logic_on_an_ideal_device():
    rep = Q.run_e0(Q.ideal_run)
    assert rep["gate_vector_rn_and_eft"]
    for eng, ident, fam, res, _d in rep["table"]:
        if "fused" in ident:
            assert res == "two roundings (no FMA)", (eng, fam)
        elif "accumulation" in ident:
            assert res == "fp32 RN-even per add"
        elif res != "see detail":
            assert res == "PASS", (eng, ident, fam)
    assert "| engine / form |" in Q.format_table(rep)


def test_e0_probe_detects_a_bad_engine():
    def truncating(name, *arrays):
        out = Q.ideal_run(name, *arrays)
        if name == "_fp64_probe_alu_kernel":
            a, b = arrays
            with np.errstate(all="ignore"):
                ex = a.astype(np.float64) + b.astype(np.float64)
            t = ex.astype(np.float32)
            t = np.where(np.abs(t.astype(np.float64)) > np.abs(ex), np.nextafter(t, 0), t)
            out[0] = t  # VectorE add truncates
        if name == "_fp64_probe_fma_kernel":
            a, b, c = arrays
            out[2] = Q._fma_expected(a, b, c)[0][2]  # scalar_tensor_tensor fused
        return out

    rep = Q.run_e0(truncating)
    assert not rep["gate_vector_rn_and_eft"]
    res = {(r[0], r[1], r[2]): r[3] for r in rep["table"]}
    assert res[("vector", "add is IEEE RN", "ties_add")] == "FAIL"
    assert res[("vector", "subtract is IEEE RN", "ties_add")] == "PASS"
    row = "vector scalar_tensor_tensor (mul (P,1), add tensor)"
    assert res[(row, "fused multiply-add?", "cancel")].startswith("fused")


# ── fp64 bits on the device (combine) ────────────────────────────────────────


def _adversarial_hilo(n, rng):
    h = (rng.standard_normal(n) * np.exp2(rng.integers(-60, 60, n))).astype(np.float32)
    l = h.astype(np.float64) * rng.standard_normal(n) * np.exp2(rng.integers(-60, 0, n))
    l = l.astype(np.float32)
    k = n // 10
    # powers of two with lo of either sign (the binade drop), at and near ties
    h[:k] = np.exp2(rng.integers(-30, 30, k)).astype(np.float32) * rng.choice([-1, 1], k)
    f = np.array([1, 1.5, 0.5, 3, 0.75])[rng.integers(0, 5, k)]
    l[:k] = (
        -h[:k].astype(np.float64) * rng.choice([-1, 1], k) * np.exp2(-rng.integers(24, 60, k)) * f
    ).astype(np.float32)
    f = np.array([1, 1.5, 0.5, 3])[rng.integers(0, 4, k)]
    l[k : 2 * k] = (
        h[k : 2 * k].astype(np.float64)
        * np.exp2(-rng.integers(24, 32, k))
        * rng.choice([-1, 1], k)
        * f
    ).astype(np.float32)
    h[2 * k : 2 * k + 100] = 0
    l[2 * k : 2 * k + 50] = 0
    h[2 * k + 100 : 2 * k + 200] = -0.0
    l[2 * k + 100 : 2 * k + 150] = -0.0
    return h, l


def test_ff_to_fp64_bits_is_rn64_of_the_sum():
    rng = np.random.default_rng(11)
    h, l = _adversarial_hilo(1_000_000, rng)
    E = rng.integers(-900, 900, h.size)
    bits, bad = F.ff_to_fp64_bits(h, l, E)
    with np.errstate(all="ignore"):
        ref = np.ldexp(h.astype(np.float64) + l.astype(np.float64), E)
    assert np.array_equal(bits[~bad], ref[~bad].view(np.uint64))
    # flagged: only |hi + lo| < 2^-75 or results that leave the normal float64 range
    s = np.abs(h[bad].astype(np.float64) + l[bad])
    out = ~np.isfinite(ref[bad]) | (np.abs(ref[bad]) < 2.0**-1021)
    assert ((s < 2.0**-75) & (s > 0) | out).all()


def test_ff_to_fp64_bits_flags():
    h = np.array([np.inf, np.nan, 2.0**-80, 1.0, 1.0, 0.0], np.float32)
    l = np.zeros(6, np.float32)
    E = np.array([0, 0, 0, 1100, -1100, 5000])
    _bits, bad = F.ff_to_fp64_bits(h, l, E)
    assert bad.tolist() == [True, True, True, True, True, False]


def test_combine_bits_equals_ozaki_combine():
    A, B, parts, exp, da, db, k = _ozaki_parts(M=200, K=640, N=130, kb=1)
    M, N = A.shape[0], B.shape[1]
    got = F.combine_bits(parts[0], da.e, db.e, k, M, N)
    ref = ozaki.combine(parts[0], da.e, db.e, k, M, N)
    assert torch.equal(got.view(torch.int64), ref.view(torch.int64))
