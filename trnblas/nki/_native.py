"""
Native TorchNeuron registration of the trnblas NKI kernels.

On the TorchNeuron lineage (Neuron SDK 2.33+, torch-neuronx 2.14,
PyTorch 2.14, Trn2/Trn3), Trainium is a native PyTorch `neuron` device
and an `@nki.jit` kernel is exposed to PyTorch as a custom operator via
`torch_neuronx.nki_op`. Each wrapper below calls an existing kernel from
`dispatch.py` unchanged; the op schema is inferred from the annotations.

Importing this module registers the ops, so `dispatch.py` imports it only
after selecting the native path. CPU-only installs and the Trn1 / XLA
lineage (whose `torch_neuronx` has no `nki_op`) never load it.

This module deliberately does not use `from __future__ import annotations`:
`nki_op` builds the op schema from the live annotation objects.

The kernels are looked up on `dispatch` at call time rather than imported
by name, so registration itself does not trace or compile anything; the
first call per input shape does.
"""

import torch
from torch_neuronx import nki_op

from . import dispatch as _d

# Fake (meta) implementations: output shape/dtype only, no kernel trace.
# Used by torch.compile tracing and for meta-device inputs. Each mirrors the
# `nl.ndarray(..., buffer=nl.shared_hbm)` the kernel returns.

_PAIR_TILE = 128  # output partition dim of the pair-energy kernels


def _fake_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]))


def _mp2_p_tile(nvir: int) -> int:
    # Largest divisor of NVIR under the 128-partition limit (as in the kernel).
    p = min(nvir, 128)
    while nvir % p != 0:
        p -= 1
    return p


# ── GEMM: one op per tile-config kernel, plus the ragged-edge kernel ─────────


def _register_gemm(tile: tuple) -> object:
    name = _d._gemm_op_name(*tile)

    def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _d._gemm_kernel_registry[tile](a, b)

    gemm.__name__ = name
    return nki_op(f"trnblas::{name}", gemm, mutates_args={}, fake_fn=_fake_gemm)


@nki_op("trnblas::gemm_ragged", mutates_args={}, fake_fn=_fake_gemm)
def gemm_ragged(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _d._gemm_kernel_ragged(a, b)


# ── SYRK ─────────────────────────────────────────────────────────────────────


@nki_op(
    "trnblas::syrk",
    mutates_args={},
    fake_fn=lambda a: a.new_empty((a.shape[0], a.shape[0])),
)
def syrk(a: torch.Tensor) -> torch.Tensor:
    return _d._syrk_kernel(a)


# ── DF-MP2 energy kernels ────────────────────────────────────────────────────


@nki_op(
    "trnblas::mp2_energy",
    mutates_args={},
    fake_fn=lambda t_flat, eps_occ_chunk, eps_occ_full, eps_vir_col, eps_vir_row: t_flat.new_empty(
        (
            _mp2_p_tile(eps_vir_row.shape[1]),
            eps_occ_chunk.shape[1] * eps_occ_full.shape[1],
        ),
        dtype=torch.float32,
    ),
)
def mp2_energy(
    t_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir_col: torch.Tensor,
    eps_vir_row: torch.Tensor,
) -> torch.Tensor:
    return _d._mp2_energy_kernel(t_flat, eps_occ_chunk, eps_occ_full, eps_vir_col, eps_vir_row)


@nki_op(
    "trnblas::fused_gemm_energy",
    mutates_args={},
    fake_fn=lambda b_i, b_j, eps_occ_sum, eps_vir_col, eps_vir_row: b_i.new_empty(
        (b_i.shape[0], 1), dtype=torch.float32
    ),
)
def fused_gemm_energy(
    b_i: torch.Tensor,
    b_j: torch.Tensor,
    eps_occ_sum: torch.Tensor,
    eps_vir_col: torch.Tensor,
    eps_vir_row: torch.Tensor,
) -> torch.Tensor:
    return _d._fused_gemm_energy_kernel(b_i, b_j, eps_occ_sum, eps_vir_col, eps_vir_row)


@nki_op(
    "trnblas::batched_pair",
    mutates_args={},
    fake_fn=lambda B, eps_occ_row, eps_vir_col, eps_vir_row: B.new_empty(
        (_PAIR_TILE, B.shape[0] * B.shape[0]), dtype=torch.float32
    ),
)
def batched_pair(
    B: torch.Tensor,
    eps_occ_row: torch.Tensor,
    eps_vir_col: torch.Tensor,
    eps_vir_row: torch.Tensor,
) -> torch.Tensor:
    return _d._batched_pair_kernel(B, eps_occ_row, eps_vir_col, eps_vir_row)


@nki_op(
    "trnblas::j_batched",
    mutates_args={},
    fake_fn=lambda b_i, B, eps_occ_i, eps_occ_row, eps_vir_col, eps_vir_row: B.new_empty(
        (_PAIR_TILE, B.shape[0]), dtype=torch.float32
    ),
)
def j_batched(
    b_i: torch.Tensor,
    B: torch.Tensor,
    eps_occ_i: torch.Tensor,
    eps_occ_row: torch.Tensor,
    eps_vir_col: torch.Tensor,
    eps_vir_row: torch.Tensor,
) -> torch.Tensor:
    return _d._j_batched_kernel(b_i, B, eps_occ_i, eps_occ_row, eps_vir_col, eps_vir_row)


# Keyed by the short op name used in `dispatch._launch` / `_call_kernel`.
OPS = {
    **{_d._gemm_op_name(*tile): _register_gemm(tile) for tile in _d._TILE_CANDIDATES},
    "gemm_ragged": gemm_ragged,
    "syrk": syrk,
    "mp2_energy": mp2_energy,
    "fused_gemm_energy": fused_gemm_energy,
    "batched_pair": batched_pair,
    "j_batched": j_batched,
}
