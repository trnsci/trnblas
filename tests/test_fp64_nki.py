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
    covered = set(K.REDUCE_KERNELS.values()) | set(_PROBE_ARGS)
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
    ref = F.reference_partials(kind, xs, arm)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


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
