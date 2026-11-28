"""CPU tests of the column-batched reductions `fp64.dot_cols` / `nrm2_cols`
(trnblas#142): every column bitwise equal to its own `dot` / `nrm2` call
(results and partials), for every arm, operand kind and finish, the
edge-case policies, and the kernels' host layout (`layout_groups`), which
must put each column's lanes where its single-column layout does."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from trnblas import fp64 as F
from trnblas.nki import fp64_kernels as K

NS = [1, 127, 128, 100000]
RS = [1, 2, 33, 128, 129]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int64) if t.dtype == torch.float64 else t.view(torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _data(n, R, seed=0):
    rng = np.random.default_rng(seed + 7 * n + R)
    X = rng.standard_normal((n, R)) * 10.0 ** rng.integers(-6, 6, (n, R))
    Y = rng.standard_normal((n, R))
    return X, Y


def _f32(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _per_col_dot(X, Y, **kw):
    return torch.stack(
        [F.dot(X[:, j], Y[:, j], backend="pytorch", **kw) for j in range(X.shape[1])]
    )


def _per_col_nrm(X, **kw):
    return torch.stack([F.nrm2(X[:, j], backend="pytorch", **kw) for j in range(X.shape[1])])


@pytest.mark.parametrize("R", RS)
@pytest.mark.parametrize("n", NS, ids=lambda n: f"n{n}")
def test_cols_equal_per_column_calls_bitwise(n, R):
    X, Y = _data(n, R)
    X, Y = _f32(X), _f32(Y)
    got = F.dot_cols(X, Y, backend="pytorch")
    assert got.shape == (R,) and got.dtype == torch.float64
    assert _same(got, _per_col_dot(X, Y))
    assert _same(F.nrm2_cols(X, backend="pytorch"), _per_col_nrm(X))


@pytest.mark.parametrize("n", [1, 5000, 300007], ids=lambda n: f"n{n}")
def test_partials_equal_per_column_partials(n):
    """The reference partials themselves, column by column, for every arm
    and kind (several tiles per chunk at n = 300007)."""
    R = 5
    X, Y = _data(n, R, seed=1)
    fx, fy = F.from_fp64(torch.from_numpy(X)), F.from_fp64(torch.from_numpy(Y))
    x32, y32 = _f32(X), _f32(Y)
    ops = {
        "dot32": [x32, y32],
        "nrm32": [x32],
        "dotff": [fx.hi, fx.lo, fy.hi, fy.lo],
        "nrmff": [fx.hi, fx.lo],
    }
    for kind, arm in sorted(K.COLS_KERNELS):
        xs = ops[kind]
        if arm == "exact":
            got = F.reference_exact_partials_cols(kind, xs)
        else:
            got = F.reference_partials_cols(kind, xs, arm)
        for j in range(R):
            col = [t[:, j] for t in xs]
            if arm == "exact":
                ref = F.reference_exact_partials(kind, col)
            else:
                ref = F.reference_partials(kind, col, arm)
            assert _same(got[:, :, j].contiguous(), ref), (kind, arm, j)


def test_every_arm_operand_kind_and_finish():
    n, R = 5000, 33
    X, Y = _data(n, R, seed=2)
    x32, y32 = _f32(X), _f32(Y)
    X64, Y64 = torch.from_numpy(X), torch.from_numpy(Y)
    for arm in F.ARMS:
        assert _same(
            F.dot_cols(x32, y32, accumulate=arm, backend="pytorch"),
            _per_col_dot(x32, y32, accumulate=arm),
        )
    for arm in ("accurate", "sloppy"):
        assert _same(
            F.dot_cols(X64, Y64, accumulate=arm, backend="pytorch"),
            _per_col_dot(X64, Y64, accumulate=arm),
        )
    fx = F.from_fp64(X64)
    assert _same(F.dot_cols(fx, Y64, backend="pytorch"), _per_col_dot(X64, Y64))
    assert _same(F.nrm2_cols(X64, backend="pytorch"), _per_col_nrm(X64))
    assert _same(F.nrm2_cols(fx, backend="pytorch"), _per_col_nrm(X64))
    # finish="ff" and "partials"
    ff = F.dot_cols(x32, y32, finish="ff", backend="pytorch")
    assert ff.hi.shape == (R,)
    for j in range(R):
        f1 = F.dot(x32[:, j], y32[:, j], finish="ff", backend="pytorch")
        assert _same(ff.hi[j], f1.hi) and _same(ff.lo[j], f1.lo)
    parts = F.dot_cols(x32, y32, finish="partials", backend="pytorch")
    assert parts.shape == (F.P, 2 * F.NCH, R)
    assert _same(parts[:, :, 3].contiguous(), F.dot(x32[:, 3], y32[:, 3], finish="partials"))


@pytest.mark.parametrize("n", [1, 1000, 100003], ids=lambda n: f"n{n}")
def test_exact_arm_cols(n):
    R = 9
    X, Y = _data(n, R, seed=3)
    x32, y32 = _f32(X), _f32(Y)
    # column 4 spreads past the 126-bit window: sticky, host fallback
    x32[0, 4], x32[-1, 4] = 2.0**100, 2.0**-100
    got = F.dot_cols(x32, y32, accumulate="exact", backend="pytorch")
    assert _same(got, _per_col_dot(x32, y32, accumulate="exact"))
    for j in range(R):
        assert float(got[j]) == float(F.exact_dot(x32[:, j].double(), y32[:, j].double()))
    parts = F.dot_cols(x32, y32, accumulate="exact", finish="partials", backend="pytorch")
    assert parts.dtype == torch.int32 and parts.shape == (F.P, F.NCH * F.EXACT_COLS, R)
    if n > 1:
        assert F.finish_exact(parts[:, :, 4].contiguous()) is None
    with pytest.raises(ValueError):
        F.dot_cols(torch.from_numpy(X), torch.from_numpy(Y), accumulate="exact")  # float64
    with pytest.raises(ValueError):
        F.nrm2_cols(x32, accumulate="exact")


def test_zeros_denormals_and_signed_zeros():
    n, R = 3000, 6
    X = torch.zeros((n, R))
    Y = torch.ones((n, R))
    X[:, 1] = -0.0
    X[::7, 2] = 2.0**-140  # subnormal fp32
    X[::5, 3] = -(2.0**-149)
    X[:, 4] = torch.from_numpy(np.ldexp(1.0, np.arange(n) % 40 - 149).astype(np.float32))
    X[:, 5] = torch.linspace(-1, 1, n)
    Y[:, 5] = 2.0**-130
    for arm in (*F.ARMS, "exact"):
        assert _same(
            F.dot_cols(X, Y, accumulate=arm, backend="pytorch"),
            _per_col_dot(X, Y, accumulate=arm),
        ), arm
    assert _same(F.nrm2_cols(X, backend="pytorch"), _per_col_nrm(X))
    assert float(F.dot_cols(X, Y)[0]) == 0.0
    assert float(F.nrm2_cols(X)[1]) == 0.0


def test_nonfinite_policy_is_per_column():
    n, R = 2000, 5
    X = torch.ones((n, R))
    Y = torch.ones((n, R))
    X[17, 1] = math.inf
    X[3, 2] = -math.inf
    X[999, 3] = math.nan
    d = F.dot_cols(X, Y)
    r = F.nrm2_cols(X)
    assert _same(d, _per_col_dot(X, Y)) and _same(r, _per_col_nrm(X))
    assert [math.isnan(v) for v in d.tolist()] == [False, True, True, True, False]
    assert [math.isnan(v) for v in r.tolist()] == [False, True, True, True, False]
    assert d[0] == n and d[4] == n
    # the exact arm raises for inf / nan input, with dot's exception per column
    for j in (1, 2, 3):
        Xj = torch.ones((n, 2))
        Xj[:, 1] = X[:, j]
        with np.errstate(invalid="ignore", over="ignore"):
            with pytest.raises(Exception) as single:
                F.dot(X[:, j], Y[:, j], accumulate="exact")
            with pytest.raises(single.type):
                F.dot_cols(Xj, Y[:, :2], accumulate="exact")
    with pytest.raises(OverflowError):  # float64 beyond the fp32 range, as dot
        F.dot_cols(torch.full((4, 2), 1e300, dtype=torch.float64), torch.ones((4, 2)))


def test_empty_and_validation():
    assert F.dot_cols(torch.zeros((0, 3)), torch.zeros((0, 3))).tolist() == [0.0, 0.0, 0.0]
    assert F.nrm2_cols(torch.zeros((10, 0))).shape == (0,)
    assert F.dot_cols(torch.zeros((10, 0)), torch.zeros((10, 0)), accumulate="exact").shape == (0,)
    with pytest.raises(ValueError):
        F.dot_cols(torch.ones(3), torch.ones(3))  # 1-D: use dot
    with pytest.raises(ValueError):
        F.dot_cols(torch.ones((3, 2)), torch.ones((3, 3)))
    with pytest.raises(TypeError):
        F.dot_cols(torch.ones((3, 2), dtype=torch.int32), torch.ones((3, 2), dtype=torch.int32))
    with pytest.raises(ValueError):
        F.dot_cols(torch.ones((3, 2)), torch.ones((3, 2)), finish="nope")
    with pytest.raises(ValueError):
        F.dot_cols(torch.ones((3, 2)), torch.ones((3, 2)), accumulate="nope")


def test_non_contiguous_operands():
    """A transposed (column-major) operand gives the same bits."""
    X, Y = _data(4000, 7, seed=4)
    x32, y32 = _f32(X), _f32(Y)
    xt = x32.T.contiguous().T  # same values, column-major strides
    assert not xt.is_contiguous()
    assert _same(F.dot_cols(xt, y32), F.dot_cols(x32, y32))
    assert _same(F.nrm2_cols(xt), F.nrm2_cols(x32))


# ── the kernels' host layout ─────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 127, 128, 5000, 100000, 300007], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("R", [1, 2, 33, 129])
@pytest.mark.parametrize("free", [None, 1, 4096])
def test_layout_groups_puts_each_column_where_layout_does(n, R, free):
    if n * R > 2 * 10**7:
        pytest.skip("large operand; the layout is covered by the smaller cases")
    pl = F.plan(n)
    G, rg = K.cols_groups(pl.W, R, free)
    assert G * rg >= R and G * rg - R < G  # balanced: fewer than G pad columns
    budget = K.COLS_FREE if free is None else free
    assert rg == 1 or rg * pl.W <= budget
    X = torch.arange(n * R, dtype=torch.float32).reshape(n, R) + 1  # no zeros
    op = K.layout_groups(F.layout_cols(X, pl), pl.W, G, rg)
    assert op.shape == (F.P, F.NCH * pl.tpc, G, pl.W, rg) and op.is_contiguous()
    flat = op.permute(0, 1, 3, 2, 4).reshape(F.P, pl.F, G * rg)  # undo the grouping
    for j in sorted({0, R // 2, R - 1}):
        assert torch.equal(flat[:, :, j], F.layout(X[:, j], pl))
    assert not flat[:, :, R:].any()  # pad columns are zero


def test_cols_kernel_table():
    assert set(K.COLS_KERNELS) == {k for k in K.REDUCE_KERNELS if k[0][:3] in ("dot", "nrm")}
    for name in K.COLS_KERNELS.values():
        assert K.KERNEL_DTYPES[name] == frozenset({torch.float32})


def test_cols_bench_dry_run():
    """The window-9 bench (`scripts/probe_fp64.py cols`) on the CPU reference."""
    from trnblas.nki import fp64_probe as Q

    rep = Q.cols_bench([(2048, 3), (5000, 1)], device="cpu", reps=1)
    assert set(rep["shapes"]) == {"n=2048,R=3", "n=5000,R=1"}
    for r in rep["shapes"].values():
        assert all(v for k, v in r.items() if k.startswith("bitwise"))
        assert {"cols_dot", "loop_dot", "pairwise_dot", "ff_iteration"} <= set(r["ms"])
        assert set(r["free_sweep"]) == {str(f) for f in Q.COLS_FREES}
    assert "| n=2048,R=3 |" in Q.format_cols(rep)
