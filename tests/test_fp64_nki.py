"""`trnblas.fp64` kernels on CPU (trnsci/trnsci#169): MLIR verification at
trn2 LNC=1 and LNC=2 (`source_to_mlir(...).verify()`, which `nki.simulate`
does not run), and `nki.simulate` runs that must equal the CPU reference
`fp64.reference_partials` bit for bit at LNC=1 and LNC=2. Needs the `nki`
wheel (Linux x86_64); no Neuron device, no neuronx-cc."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nki_simulator

from trnblas import fp64 as F  # noqa: E402
from trnblas.nki import fp64_kernels as K  # noqa: E402
from trnblas.nki import fp64_probe as Q  # noqa: E402

try:
    import nki
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    nki = source_to_mlir = None


@pytest.fixture(autouse=True)
def _nki_available():
    if not K.HAS_NKI or source_to_mlir is None:
        pytest.skip("nki (with nki.compiler.frontend) not importable on this host")


_ARITY = {"sum32": 1, "nrm32": 1, "dot32": 2, "sumff": 2, "nrmff": 2, "dotff": 4}
# n -> layouts: one tiny tile, one tile per chunk, several tiles per chunk
# with a non-power-of-two W, and the n = 1e7 layout of the device probes.
SIZES = [1, 5000, 300007, 10**7]


def _rows(kind, n):
    pl = F.plan(n)
    return [np.zeros((F.P, pl.F), np.float32)] * _ARITY[kind]


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("key", sorted(K.REDUCE_KERNELS), ids=lambda k: "-".join(k))
def test_reduce_kernel_mlir_verifies(key, n, lnc):
    name = K.REDUCE_KERNELS[key]
    source_to_mlir(
        K.kernel(name), *_rows(key[0], n), target="trn2", lnc=lnc, frontend=FrontendKind.TRACER
    ).verify()


_PROBE_ARGS = {
    "_fp64_probe_alu_kernel": lambda: [np.ones((128, Q.W), np.float32)] * 2,
    "_fp64_probe_fma_kernel": lambda: [np.ones((128, Q.W), np.float32)] * 3,
    "_fp64_probe_eft_kernel": lambda: [np.ones((128, Q.W), np.float32)] * 2,
    "_fp64_probe_pe_kernel": lambda: [
        np.ones((Q.PE_K, 128), np.float32),
        np.ones((Q.PE_K, 512), np.float32),
    ],
    # fp64-bits combine: column tiles plus a tail, an odd row-tile count
    "_fp64_ff_bits_kernel": lambda: [
        np.ones((2, 384, 1100), np.float32),
        np.zeros((384, 1), np.float32),
        np.zeros((1, 1100), np.float32),
    ],
}


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("name", sorted(_PROBE_ARGS))
def test_probe_kernel_mlir_verifies(name, lnc):
    source_to_mlir(
        K.kernel(name), *_PROBE_ARGS[name](), target="trn2", lnc=lnc, frontend=FrontendKind.TRACER
    ).verify()


def test_every_fp64_kernel_is_covered():
    kernels = {
        name
        for name, obj in vars(K).items()
        if name.endswith("_kernel") and type(obj).__module__.startswith("nki")
    }
    covered = set(K.REDUCE_KERNELS.values()) | set(_PROBE_ARGS)  # incl. the ff_bits kernel
    assert kernels == covered
    assert set(K.KERNEL_DTYPES) == covered


def _operands(kind, n, rng):
    x = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 6, n)
    y = rng.standard_normal(n)
    if kind in ("sum32", "nrm32"):
        return [torch.from_numpy(x.astype(np.float32))]
    if kind == "dot32":
        return [torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))]
    fx = F.from_fp64(torch.from_numpy(x))
    if kind in ("sumff", "nrmff"):
        return [fx.hi, fx.lo]
    fy = F.from_fp64(torch.from_numpy(y))
    return [fx.hi, fx.lo, fy.hi, fy.lo]


@pytest.mark.parametrize("lnc", ["1", "2"])
@pytest.mark.parametrize("n", [1, 5000, 300007])
@pytest.mark.parametrize("key", sorted(K.REDUCE_KERNELS), ids=lambda k: "-".join(k))
def test_kernel_equals_reference_bitwise(key, n, lnc, monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    kind, arm = key
    xs = _operands(kind, n, np.random.default_rng(n))
    got = K.run_partials(kind, xs, arm, simulate=True)
    if arm == "exact":
        ref = F.reference_exact_partials(kind, xs)
    else:
        ref = F.reference_partials(kind, xs, arm)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize("kind", ["sum32", "dot32"])
def test_exact_kernel_window_and_sticky(kind, monkeypatch):
    """Spreads beyond the 126-bit window set the sticky flag identically."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    rng = np.random.default_rng(7)
    n = 40000
    x = (rng.standard_normal(n) * np.exp2(rng.integers(-60, 60, n))).astype(np.float32)
    x[::997] = 2.0**-140  # subnormal-scale terms far below the window
    y = rng.standard_normal(n).astype(np.float32)
    xs = [torch.from_numpy(x)] + ([torch.from_numpy(y)] if kind == "dot32" else [])
    got = K.run_partials(kind, xs, "exact", simulate=True)
    ref = F.reference_exact_partials(kind, xs)
    assert torch.equal(got, ref)
    assert F.finish_exact(ref) is None  # sticky: the host recomputes


def test_public_api_under_the_simulator(monkeypatch):
    from trnblas.nki import dispatch as D

    monkeypatch.setattr(D, "_USE_SIMULATOR", True)
    x = torch.from_numpy(np.random.default_rng(0).standard_normal(70001).astype(np.float32))
    y = x.flip(0).contiguous()
    for arm in F.ARMS:
        assert torch.equal(
            F.dot(x, y, accumulate=arm), F.dot(x, y, accumulate=arm, backend="pytorch")
        )
    assert torch.equal(F.nrm2(x), F.nrm2(x, backend="pytorch"))
    xd = x.double() / 3
    assert torch.equal(F.sum(xd), F.sum(xd, backend="pytorch"))
    with pytest.raises(NotImplementedError):
        F.nrm2(x, accumulate="comp")


def test_e0_probe_runs_in_the_simulator(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")

    def run(name, *arrays):
        return np.asarray(nki.simulate(K.kernel(name))(*arrays))

    rep = Q.run_e0(run)
    assert rep["gate_vector_rn_and_eft"]
    res = {(r[0], r[1], r[2]): r[3] for r in rep["table"]}
    # The simulator evaluates numpy fp32 op by op: no fusion anywhere.
    for row in Q.FMA_ROWS:
        assert res[(row, "fused multiply-add?", "cancel")] == "two roundings (no FMA)"


def _ozaki_like(R, N, rng):
    h = (rng.standard_normal((R, N)) * np.exp2(rng.integers(0, 40, (R, N)))).astype(np.float32)
    l = h.astype(np.float64) * rng.standard_normal((R, N)) * np.exp2(-rng.integers(20, 50, (R, N)))
    l = l.astype(np.float32)
    h[:20] = np.exp2(rng.integers(0, 30, (20, N))).astype(np.float32)  # powers of two,
    l[:20] = (-h[:20] * np.exp2(-rng.integers(24, 40, (20, N)))).astype(np.float32)  # below
    h[30, :2] = 0
    l[30, :1] = 0
    h[31, 0] = np.inf
    return torch.from_numpy(np.stack([h, l]))


@pytest.mark.parametrize("lnc", ["1", "2"])
@pytest.mark.parametrize("shape", [(256, 700, 250), (384, 241, 300), (128, 3, 128)])
def test_ff_bits_kernel_equals_ozaki_combine(shape, lnc, monkeypatch):
    from trnblas import ozaki

    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    R, N, M = shape
    rng = np.random.default_rng(R + N)
    hilo = _ozaki_like(R, N, rng)
    eA = rng.integers(-300, 300, R)
    eB = rng.integers(-300, 300, N)
    eA[5], eA[6] = 1000, -1000  # rows leaving the float64 range: flagged, host
    with np.errstate(all="ignore"):
        got = K.combine_bits_device(hilo, eA, eB, 8, M, N, simulate=True)
        ref = ozaki.combine(hilo, eA, eB, 8, M, N)
    assert torch.equal(got.view(torch.int64), ref.view(torch.int64))


@pytest.mark.parametrize("lnc", ["1", "2"])
@pytest.mark.parametrize("shape", [(256, 700), (384, 241), (128, 3)])
def test_ff_bits_kernel_reference_equals_kernel_output(shape, lnc, monkeypatch):
    """`fp64.ff_bits_kernel_reference` (the CPU stand-in that the CPU tests of
    the gemm_fp64 device combine, #125, run on) is the kernel's raw output
    bit for bit: the flag rows, and the float64 words of every row not
    flagged (a flagged row is recombined on the host, so its words are
    unspecified: they differ at a non-finite element)."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    R, N = shape
    rng = np.random.default_rng(R * N)
    hilo = _ozaki_like(R, N, rng).numpy()
    ea = rng.integers(-300, 300, (R, 1)).astype(np.float32)
    eb = rng.integers(-300, 300, (1, N)).astype(np.float32)
    ea[5], ea[6] = 1000, -1000
    with np.errstate(all="ignore"):
        got = np.asarray(nki.simulate(K.kernel("_fp64_ff_bits_kernel"))(hilo, ea, eb))
        ref = F.ff_bits_kernel_reference(hilo, ea, eb)
    assert np.array_equal(got[R:], ref[R:])
    ok = ref[R:, : R // 128].T.reshape(-1) == 0
    assert not ok[31] and not ok[6]  # an inf; results below the normal range
    assert np.array_equal(got[:R][ok], ref[:R][ok])
