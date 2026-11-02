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


@_fp64_host_guard("asum")
def asum(x: torch.Tensor) -> torch.Tensor:
    """Sum of absolute values: sum(|x_i|)"""
    return torch.abs(x).sum()


@_fp64_host_guard("iamax")
def iamax(x: torch.Tensor) -> int:
    """Index of element with maximum absolute value."""
    return torch.argmax(torch.abs(x.reshape(-1))).item()
