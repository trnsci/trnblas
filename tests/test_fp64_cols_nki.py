"""The column-batched `trnblas.fp64` kernels (trnblas#142) on CPU: MLIR
verification at trn2 LNC=1 and LNC=2, and `nki.simulate` runs whose
partials equal the CPU reference bit for bit, and whose `dot_cols` /
`nrm2_cols` results equal R separate `dot` / `nrm2` calls bit for bit. Needs
the `nki` wheel (Linux x86_64); no Neuron device, no neuronx-cc."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nki_simulator

from trnblas import fp64 as F  # noqa: E402
from trnblas.nki import dispatch as D  # noqa: E402
from trnblas.nki import fp64_kernels as K  # noqa: E402

try:
    import nki
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    nki = source_to_mlir = None


@pytest.fixture(autouse=True)
def _nki_available():
    if not K.HAS_NKI or source_to_mlir is None:
        pytest.skip("nki (with nki.compiler.frontend) not importable on this host")


@pytest.fixture
def sim(monkeypatch):
    monkeypatch.setattr(D, "_USE_SIMULATOR", True)


_ARITY = {"nrm32": 1, "dot32": 2, "nrmff": 2, "dotff": 4}


def _bits(t):
    return t.view(torch.int64) if t.dtype == torch.float64 else t.view(torch.int32)


def _same(a, b):
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _f32(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _data(n, R, seed=0):
    rng = np.random.default_rng(seed + 7 * n + R)
    X = rng.standard_normal((n, R)) * 10.0 ** rng.integers(-6, 6, (n, R))
    return X, rng.standard_normal((n, R))


# (n, R) -> layouts: a tiny tile; one group of 33; two groups with a pad
# column (W = 16); 17 groups of 2 (the n = 1e5, R = 33 target shape); two
# tiles per chunk (W = 304) with a pad column
LAYOUTS = [(1, 1), (5000, 33), (5000, 47), (100000, 33), (300007, 3)]


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("nr", LAYOUTS, ids=lambda t: f"n{t[0]}-R{t[1]}")
@pytest.mark.parametrize("key", sorted(K.COLS_KERNELS), ids=lambda k: "-".join(k))
def test_cols_kernel_mlir_verifies(key, nr, lnc):
    n, R = nr
    pl = F.plan(n)
    G, rg = K.cols_groups(pl.W, R)
    op = np.zeros((F.P, F.NCH * pl.tpc, G, pl.W, rg), np.float32)
    source_to_mlir(
        K.kernel(K.COLS_KERNELS[key]),
        *[op] * _ARITY[key[0]],
        target="trn2",
        lnc=lnc,
        frontend=FrontendKind.TRACER,
    ).verify()


def _operands(kind, n, R, seed=0):
    X, Y = _data(n, R, seed)
    if kind == "nrm32":
        return [_f32(X)]
    if kind == "dot32":
        return [_f32(X), _f32(Y)]
    fx = F.from_fp64(torch.from_numpy(X))
    if kind == "nrmff":
        return [fx.hi, fx.lo]
    fy = F.from_fp64(torch.from_numpy(Y))
    return [fx.hi, fx.lo, fy.hi, fy.lo]


@pytest.mark.parametrize("lnc", ["1", "2"])
@pytest.mark.parametrize("nr", [(1, 1), (5000, 47), (300007, 3)], ids=lambda t: f"n{t[0]}-R{t[1]}")
@pytest.mark.parametrize("key", sorted(K.COLS_KERNELS), ids=lambda k: "-".join(k))
def test_cols_kernel_equals_reference_bitwise(key, nr, lnc, monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    kind, arm = key
    xs = _operands(kind, *nr)
    got = K.run_partials_cols(kind, xs, arm, simulate=True)
    if arm == "exact":
        ref = F.reference_exact_partials_cols(kind, xs)
    else:
        ref = F.reference_partials_cols(kind, xs, arm)
    assert _same(got, ref)


@pytest.mark.parametrize("R", [1, 2, 33, 128, 129])
@pytest.mark.parametrize("n", [1, 127, 128, 100000], ids=lambda n: f"n{n}")
def test_cols_equal_per_column_calls_under_the_simulator(n, R, sim):
    """dot_cols / nrm2_cols through the kernels equal R separate dot / nrm2
    calls bit for bit (the per-column calls on the CPU reference, which the
    single-column kernels equal bit for bit: test_fp64_nki.py)."""
    X, Y = _data(n, R)
    X, Y = _f32(X), _f32(Y)
    d = F.dot_cols(X, Y)
    r = F.nrm2_cols(X)
    assert _same(d, torch.stack([F.dot(X[:, j], Y[:, j], backend="pytorch") for j in range(R)]))
    assert _same(r, torch.stack([F.nrm2(X[:, j], backend="pytorch") for j in range(R)]))


def test_cols_equal_single_column_kernels(sim):
    """Against the single-column kernels themselves, both in the simulator."""
    n, R = 30000, 9
    X, Y = _data(n, R, seed=5)
    X, Y = _f32(X), _f32(Y)
    d = F.dot_cols(X, Y)
    r = F.nrm2_cols(X)
    for j in range(R):
        assert _same(d[j], F.dot(X[:, j], Y[:, j]))
        assert _same(r[j], F.nrm2(X[:, j]))
    e = F.dot_cols(X, Y, accumulate="exact")
    for j in range(R):
        assert _same(e[j], F.dot(X[:, j], Y[:, j], accumulate="exact"))


@pytest.mark.parametrize("free", [1, 64, 4096])
def test_group_width_does_not_change_the_bits(free):
    n, R = 20000, 13
    xs = _operands("dot32", n, R, seed=6)
    ref = F.reference_partials_cols("dot32", xs)
    assert _same(K.run_partials_cols("dot32", xs, "accurate", simulate=True, free=free), ref)
    xs = _operands("nrm32", n, R, seed=6)
    ref = F.reference_partials_cols("nrm32", xs)
    assert _same(K.run_partials_cols("nrm32", xs, "accurate", simulate=True, free=free), ref)


def test_edge_cases_under_the_simulator(sim):
    """Zeros (and -0), subnormals, inf / nan: per column as dot / nrm2."""
    n, R = 3000, 9
    X = torch.zeros((n, R))
    Y = torch.ones((n, R))
    X[:, 1] = -0.0
    X[::7, 2] = 2.0**-140
    X[::5, 3] = -(2.0**-149)
    X[:, 4] = torch.from_numpy(np.ldexp(1.0, np.arange(n) % 40 - 149).astype(np.float32))
    X[:, 5] = torch.linspace(-1, 1, n)
    Y[:, 5] = 2.0**-130
    X[:, 6:] = 1.0
    X[17, 6] = math.inf
    X[3, 7] = -math.inf
    X[999, 8] = math.nan
    for arm in F.ARMS:
        got = F.dot_cols(X, Y, accumulate=arm)
        ref = torch.stack(
            [F.dot(X[:, j], Y[:, j], accumulate=arm, backend="pytorch") for j in range(R)]
        )
        assert _same(got, ref), arm
    got = F.nrm2_cols(X)
    assert _same(got, torch.stack([F.nrm2(X[:, j], backend="pytorch") for j in range(R)]))
    assert all(math.isnan(v) for v in got[6:].tolist())
    fin = X[:, :6].contiguous(), Y[:, :6].contiguous()
    got = F.dot_cols(*fin, accumulate="exact")
    ref = torch.stack(
        [F.dot(fin[0][:, j], fin[1][:, j], accumulate="exact", backend="pytorch") for j in range(6)]
    )
    assert _same(got, ref)
