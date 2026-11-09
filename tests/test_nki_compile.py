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


def _i32(*shape):
    return np.zeros(shape, dtype=np.int32)


def _bf16(*shape):
    import ml_dtypes

    return np.zeros(shape, dtype=ml_dtypes.bfloat16)


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
    # LNC-split GEMM (#79): M, N and K remainders, the row split (M >= 256)
    # and the column split (small M), from row-major A and from A^T.
    "gemm_lnc": (lambda d: d._gemm_lnc_kernel, lambda: (_f32(301, 200), _f32(200, 613))),
    "gemm_lnc_small_m": (lambda d: d._gemm_lnc_kernel, lambda: (_f32(17, 130), _f32(130, 2051))),
    "gemm_lnc_tn": (lambda d: d._gemm_lnc_tn_kernel, lambda: (_f32(200, 301), _f32(200, 613))),
    "gemm_lnc_tn_small_m": (
        lambda d: d._gemm_lnc_tn_kernel,
        lambda: (_f32(130, 17), _f32(130, 2051)),
    ),
    "gemm_lnc_tiny": (lambda d: d._gemm_lnc_kernel, lambda: (_f32(1, 1), _f32(1, 1))),
    "syrk": (lambda d: d._syrk_kernel, lambda: (_f32(256, 128),)),
    "mp2_energy": (lambda d: d._mp2_energy_kernel, _mp2_inputs),
    "fused_gemm_energy": (lambda d: d._fused_gemm_energy_kernel, _fused_inputs),
    "batched_pair": (lambda d: d._batched_pair_kernel, _batched_pair_inputs),
    "j_batched": (lambda d: d._j_batched_kernel, _j_batched_inputs),
    "pair_energy_sym": (lambda d: d._pair_energy_sym_kernel, _pair_sym_inputs),
    # Ozaki fp64-class GEMM (#22): an odd M-tile count (the LNC halves share
    # the middle tile), N super-tiles plus a tail, K groups that span pairs
    # (KT = 33), k = 8; a single tile; and K across several 4096-term groups.
    "ozaki_gemm": (
        lambda d: d._ozaki_gemm_kernel,
        lambda: (_bf16(8, 33 * 128, 384), _bf16(8, 33 * 128, 1300)),
    ),
    "ozaki_gemm_tiny": (
        lambda d: d._ozaki_gemm_kernel,
        lambda: (_bf16(1, 128, 128), _bf16(1, 128, 1)),
    ),
    # Two M tiles: at LNC=2 the programs split the columns (narrow-N C^T launches).
    "ozaki_gemm_col_split": (
        lambda d: d._ozaki_gemm_kernel,
        lambda: (_bf16(8, 66 * 128, 256), _bf16(8, 66 * 128, 1023)),
    ),
    "ozaki_gemm_long_k": (
        lambda d: d._ozaki_gemm_kernel,
        lambda: (_bf16(3, 9000 // 128 * 128 + 128, 256), _bf16(3, 9000 // 128 * 128 + 128, 241)),
    ),
    # Device split (#123): full and partial row tiles (odd count: the LNC
    # halves share the middle one) and column tiles, k = 8 (3 chunks, no
    # rest field) and k = 3, 12 (rest fields); a single partial tile.
    "ozaki_emax": (lambda d: d._ozaki_emax_kernel, lambda: (_i32(3 * 128 + 5, 2 * 2100),)),
    "ozaki_emax_col": (lambda d: d._ozaki_emax_col_kernel, lambda: (_i32(300, 2 * 1100),)),
    "ozaki_split": (
        lambda d: d._ozaki_split_kernel,
        lambda: (_i32(3 * 128 + 5, 2 * 1100), _f32(4 * 128, 8)),
    ),
    "ozaki_split_k3": (lambda d: d._ozaki_split_kernel, lambda: (_i32(130, 2 * 600), _f32(256, 3))),
    "ozaki_split_tiny": (lambda d: d._ozaki_split_kernel, lambda: (_i32(1, 2), _f32(128, 1))),
    "ozaki_split_col": (
        lambda d: d._ozaki_split_col_kernel,
        lambda: (_i32(300, 2 * 1100), _f32(12, 1152)),
    ),
    "ozaki_split_col_narrow": (
        lambda d: d._ozaki_split_col_kernel,
        lambda: (_i32(8400, 2 * 241), _f32(8, 256)),
    ),
}

# Native-only kernels that split work across the cores of an LNC=2 logical
# core by program_id (and degrade to one program at LNC=1): verified on trn2
# at both lnc=1 and lnc=2, the only target they are dispatched on.
_LNC2 = {
    "pair_energy_sym",
    "gemm_lnc",
    "gemm_lnc_small_m",
    "gemm_lnc_tn",
    "gemm_lnc_tn_small_m",
    "gemm_lnc_tiny",
    "ozaki_gemm",
    "ozaki_gemm_tiny",
    "ozaki_gemm_long_k",
    "ozaki_gemm_col_split",
    "ozaki_emax",
    "ozaki_emax_col",
    "ozaki_split",
    "ozaki_split_k3",
    "ozaki_split_tiny",
    "ozaki_split_col",
    "ozaki_split_col_narrow",
}


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


@pytest.mark.parametrize("kernel_id", sorted(_LNC2))
def test_lnc_split_kernel_verifies_at_lnc1(kernel_id):
    """The program_id split degrades to one program at LNC=1 (trnblas#50)."""
    from trnblas.nki import dispatch

    get_kernel, make_inputs = _KERNELS[kernel_id]
    source_to_mlir(
        get_kernel(dispatch), *make_inputs(), target="trn2", lnc=1, frontend=FrontendKind.TRACER
    ).verify()


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
