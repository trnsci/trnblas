"""
BLAS Level 1 — Vector operations.

axpy, dot, nrm2, scal, asum, iamax
All operate on torch tensors with NKI dispatch for Trainium acceleration.
"""

from __future__ import annotations

import torch

from .nki.dispatch import _fp64_host_guard


@_fp64_host_guard("axpy")
def axpy(alpha: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """y = alpha * x + y"""
    return alpha * x + y


@_fp64_host_guard("dot")
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Dot product: x^T y

    Returns scalar tensor.
    """
    return torch.dot(x.reshape(-1), y.reshape(-1))


@_fp64_host_guard("nrm2")
def nrm2(x: torch.Tensor) -> torch.Tensor:
    """Euclidean norm: ||x||_2"""
    return torch.linalg.norm(x.reshape(-1))


@_fp64_host_guard("scal")
def scal(alpha: float, x: torch.Tensor) -> torch.Tensor:
    """x = alpha * x"""
    return alpha * x


def _abs1(x: torch.Tensor) -> torch.Tensor:
    """BLAS |x|: |Re x| + |Im x| for complex input (scasum/icamax), |x| for real."""
    if x.is_complex():
        return x.real.abs() + x.imag.abs()
    return torch.abs(x)


@_fp64_host_guard("asum")
def asum(x: torch.Tensor) -> torch.Tensor:
    """Sum of absolute values, as BLAS ``?asum``: sum(|x_i|) for real x.

    For complex x this is sum(|Re x_i| + |Im x_i|), as ``scasum`` /
    ``dzasum`` / ``cublasScasum`` (not the modulus; changed in 0.11.0, #75).
    The result is real.
    """
    return _abs1(x).sum()


@_fp64_host_guard("iamax")
def iamax(x: torch.Tensor) -> int:
    """0-based index of the first element of largest absolute value.

    As BLAS ``i?amax``, but 0-based (cuBLAS ``cublasIsamax`` is 1-based). For
    complex x the magnitude is |Re x_i| + |Im x_i|, as ``icamax`` /
    ``izamax`` (not the modulus; changed in 0.11.0, #75). A matrix is
    flattened in row-major order.
    """
    return torch.argmax(_abs1(x.reshape(-1))).item()
