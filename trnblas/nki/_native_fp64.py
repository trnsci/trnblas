"""
Native TorchNeuron registration of the `trnblas.fp64` kernels
(trnsci/trnsci#169), following `_native.py`: each `trnblas::fp64_*` op calls
a kernel of `fp64_kernels.py` unchanged, with a fake (meta) implementation
giving the output shape. Imported lazily by `fp64_kernels.call`, and only on
the native path; eager calls on plain neuron tensors use the cached
`torch_neuronx.wrap_nki` handle instead (`dispatch._eager_handle`).

No `from __future__ import annotations`: `nki_op` reads the live annotations.
"""

import torch
from torch_neuronx import nki_op

from . import fp64_kernels as _k

_NCH = 4


def _fake_partials(ncol):
    def fake(*xs):
        return xs[0].new_empty((128, ncol * _NCH), dtype=torch.float32)

    return fake


def _register_reduce(name: str, arity: int, ncol: int):
    fake = _fake_partials(ncol)
    if arity == 1:

        def op1(x: torch.Tensor) -> torch.Tensor:
            return _k.kernel(name)(x)

        fn = op1
    elif arity == 2:

        def op2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return _k.kernel(name)(x, y)

        fn = op2
    else:

        def op4(
            xh: torch.Tensor, xl: torch.Tensor, yh: torch.Tensor, yl: torch.Tensor
        ) -> torch.Tensor:
            return _k.kernel(name)(xh, xl, yh, yl)

        fn = op4
    fn.__name__ = name.strip("_")
    return nki_op(f"trnblas::{name.strip('_')}", fn, mutates_args={}, fake_fn=fake)


_ARITY = {"sum32": 1, "nrm32": 1, "dot32": 2, "sumff": 2, "nrmff": 2, "dotff": 4}

OPS = {
    name: _register_reduce(name, _ARITY[kind], 3 if kind.startswith("nrm") else 2)
    for (kind, _arm), name in _k.REDUCE_KERNELS.items()
}


@nki_op(
    "trnblas::fp64_probe_alu",
    mutates_args={},
    fake_fn=lambda a, b: a.new_empty((10, 128, a.shape[1])),
)
def fp64_probe_alu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _k.kernel("_fp64_probe_alu_kernel")(a, b)


@nki_op(
    "trnblas::fp64_probe_fma",
    mutates_args={},
    fake_fn=lambda a, b, c: a.new_empty((5, 128, a.shape[1])),
)
def fp64_probe_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return _k.kernel("_fp64_probe_fma_kernel")(a, b, c)


@nki_op(
    "trnblas::fp64_probe_eft",
    mutates_args={},
    fake_fn=lambda a, b: a.new_empty((4, 128, a.shape[1])),
)
def fp64_probe_eft(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _k.kernel("_fp64_probe_eft_kernel")(a, b)


@nki_op(
    "trnblas::fp64_probe_pe",
    mutates_args={},
    fake_fn=lambda at, b: at.new_empty((2, at.shape[1], b.shape[1])),
)
def fp64_probe_pe(at: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _k.kernel("_fp64_probe_pe_kernel")(at, b)


@nki_op(
    "trnblas::fp64_ff_bits",
    mutates_args={},
    fake_fn=lambda hilo, ea, eb: hilo.new_empty(
        (hilo.shape[1] + 128, 2 * hilo.shape[2]), dtype=torch.int32
    ),
)
def fp64_ff_bits(hilo: torch.Tensor, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    return _k.kernel("_fp64_ff_bits_kernel")(hilo, ea, eb)


OPS.update(
    {
        "_fp64_ff_bits_kernel": fp64_ff_bits,
        "_fp64_probe_alu_kernel": fp64_probe_alu,
        "_fp64_probe_fma_kernel": fp64_probe_fma,
        "_fp64_probe_eft_kernel": fp64_probe_eft,
        "_fp64_probe_pe_kernel": fp64_probe_pe,
    }
)
