"""CPU compile check: every NKI kernel lowers to MLIR that passes verification.

`nki.simulate` does not run the MLIR verifier, so a kernel can pass every
simulator test and still fail to compile for hardware (trnsci/trnrand#56 was
such a case). This module runs the NKI frontend (trace -> MLIR -> verify) on
CPU for every trnblas kernel, for trn1 and trn2. It needs `nki` but no
Neuron device and no `neuronx-cc`; it does not run the backend compiler, so
it cannot catch errors raised after MLIR generation.

The same kernels serve the XLA and native TorchNeuron dispatch paths, so one
check covers both.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.nki_simulator

from trnblas.nki.dispatch import _TILE_CANDIDATES, HAS_NKI  # noqa: E402

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None


def _f32(*shape):
    return np.zeros(shape, dtype=np.float32)


def _eps_vir(nvir):
    # Non-zero so the reciprocal in the energy kernels is traced on real values.
    return np.full((nvir, 1), -1.0, np.float32), np.full((1, nvir), -1.0, np.float32)


def _mp2_inputs():
    ic, nocc, nvir = 2, 2, 128
    ev_col, ev_row = _eps_vir(nvir)
    return (
        _f32(ic * nvir, nocc * nvir),
        np.full((1, ic), -0.5, np.float32),
        np.full((1, nocc), -0.5, np.float32),
        ev_col,
        ev_row,
    )


def _fused_inputs():
    ev_col, ev_row = _eps_vir(128)
    return (_f32(128, 128), _f32(128, 128), np.full((1, 1), -1.0, np.float32), ev_col, ev_row)


def _batched_pair_inputs():
    nocc = 2
    ev_col, ev_row = _eps_vir(128)
    return (_f32(nocc, 128, 128), np.full((1, nocc), -0.5, np.float32), ev_col, ev_row)


def _j_batched_inputs():
    nocc = 2
    ev_col, ev_row = _eps_vir(128)
    return (
        _f32(128, 128),
        _f32(nocc, 128, 128),
        np.full((1, 1), -0.5, np.float32),
        np.full((1, nocc), -0.5, np.float32),
        ev_col,
        ev_row,
    )


def _pair_sym_inputs():
    nd = 4
    ev_col, ev_row = _eps_vir(256)
    return (
        _f32(256, 256),
        _f32(nd, 256, 256),
        np.full((1, 1), -0.5, np.float32),
        np.full((1, nd), -0.5, np.float32),
        ev_col,
        ev_row,
    )


# kernel id -> (kernel lookup on dispatch, example inputs matching the host wrappers)
_KERNELS = {
    **{
        f"gemm_{tm}_{tk}_{tn}": (
            lambda d, t=(tm, tk, tn): d._get_gemm_kernel(*t),
            lambda: (_f32(256, 256), _f32(256, 1024)),
        )
        for tm, tk, tn in _TILE_CANDIDATES
    },
    # Every remainder branch (M, K and N edge tiles plus the corner).
    "gemm_ragged": (lambda d: d._gemm_kernel_ragged, lambda: (_f32(300, 200), _f32(200, 600))),
    "gemm_ragged_small": (lambda d: d._gemm_kernel_ragged, lambda: (_f32(1, 1), _f32(1, 1))),
    "syrk": (lambda d: d._syrk_kernel, lambda: (_f32(256, 128),)),
    "mp2_energy": (lambda d: d._mp2_energy_kernel, _mp2_inputs),
    "fused_gemm_energy": (lambda d: d._fused_gemm_energy_kernel, _fused_inputs),
    "batched_pair": (lambda d: d._batched_pair_kernel, _batched_pair_inputs),
    "j_batched": (lambda d: d._j_batched_kernel, _j_batched_inputs),
    "pair_energy_sym": (lambda d: d._pair_energy_sym_kernel, _pair_sym_inputs),
}

# Kernels launched as an LNC=2 grid (one program per physical core): verified
# with lnc=2 on trn2 only, which is the only target they are dispatched on.
_LNC2 = {"pair_energy_sym"}


@pytest.fixture(autouse=True)
def _frontend_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")


@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("kernel_id", sorted(_KERNELS))
def test_kernel_mlir_verifies(kernel_id, target):
    from trnblas.nki import dispatch

    get_kernel, make_inputs = _KERNELS[kernel_id]
    lnc = 2 if kernel_id in _LNC2 else 1
    if lnc == 2 and target != "trn2":
        pytest.skip("LNC=2 kernels are native-path (Trn2) only")
    result = source_to_mlir(
        get_kernel(dispatch),
        *make_inputs(),
        target=target,
        lnc=lnc,
        frontend=FrontendKind.TRACER,
    )
    result.verify()


def test_every_kernel_is_covered():
    """A new @nki.jit kernel in dispatch.py must get an entry above."""
    from trnblas.nki import dispatch

    covered = {id(get(dispatch)) for get, _ in _KERNELS.values()}
    kernels = {
        name
        for name, obj in vars(dispatch).items()
        if name.endswith("_kernel") or name.startswith("_gemm_kernel_")
        if callable(obj) and type(obj).__module__.startswith("nki")
    }
    missing = {k for k in kernels if id(getattr(dispatch, k)) not in covered}
    assert not missing, missing
