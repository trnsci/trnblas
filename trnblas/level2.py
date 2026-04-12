"""
BLAS Level 2 — Matrix-vector operations.

gemv, symv, trmv, ger
"""

from __future__ import annotations

import torch
from typing import Optional


def gemv(
    alpha: float,
    A: torch.Tensor,
    x: torch.Tensor,
    beta: float = 0.0,
    y: Optional[torch.Tensor] = None,
    trans: bool = False,
) -> torch.Tensor:
    """General matrix-vector multiply: y = alpha * op(A) * x + beta * y

    op(A) = A if trans=False, A^T if trans=True.
    """
    mat = A.T if trans else A
    result = alpha * torch.mv(mat, x)
    if y is not None and beta != 0.0:
        result = result + beta * y
    return result


def symv(
    alpha: float,
    A: torch.Tensor,
    x: torch.Tensor,
    beta: float = 0.0,
    y: Optional[torch.Tensor] = None,
    uplo: str = "upper",
) -> torch.Tensor:
    """Symmetric matrix-vector multiply: y = alpha * A * x + beta * y

    A is symmetric; only the upper or lower triangle is referenced.
    """
    if uplo == "upper":
        sym = torch.triu(A) + torch.triu(A, diagonal=1).T
    else:
        sym = torch.tril(A) + torch.tril(A, diagonal=-1).T
    result = alpha * torch.mv(sym, x)
    if y is not None and beta != 0.0:
        result = result + beta * y
    return result


def trmv(
    A: torch.Tensor,
    x: torch.Tensor,
    uplo: str = "upper",
    trans: bool = False,
    diag: str = "nonunit",
) -> torch.Tensor:
    """Triangular matrix-vector multiply: x = op(A) * x

    uplo: 'upper' or 'lower'
    diag: 'unit' or 'nonunit'
    """
    if uplo == "upper":
        tri = torch.triu(A)
    else:
        tri = torch.tril(A)
    if diag == "unit":
        tri = tri - torch.diag(torch.diag(tri)) + torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    mat = tri.T if trans else tri
    return torch.mv(mat, x)


def ger(
    alpha: float,
    x: torch.Tensor,
    y: torch.Tensor,
    A: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rank-1 update: A = alpha * x * y^T + A"""
    result = alpha * torch.outer(x, y)
    if A is not None:
        result = result + A
    return result
