"""
BLAS Level 3 — Matrix-matrix operations.

gemm, symm, syrk, trsm, trmm

These are the critical path for scientific computing on Trainium:
- DF-MP2 tensor contractions → batched GEMM
- Fock matrix build → SYMM
- Cholesky-based density fitting → TRSM
- Metric contraction J^{-1/2} → SYRK

NKI dispatch for GEMM provides stationary tile reuse on the Tensor Engine.
"""

from __future__ import annotations

import torch


def gemm(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float = 0.0,
    C: torch.Tensor | None = None,
    transA: bool = False,
    transB: bool = False,
) -> torch.Tensor:
    """General matrix multiply: C = alpha * op(A) @ op(B) + beta * C

    op(X) = X if trans=False, X^T if trans=True.

    This is the workhorse for DF-MP2:
        (ia|P) = Σ_μν C_μi @ (μν|P) @ C_νa    → two GEMMs per aux index
        B_ia^P = (ia|Q) @ J^{-1/2}_{QP}        → one GEMM
        E_MP2 = Σ B_ia^P @ B_jb^P              → batched GEMM

    On Trainium, dispatches to NKI GEMM with stationary tile reuse.
    """
    from .nki.dispatch import _use_nki, nki_gemm

    a = A.T if transA else A
    b = B.T if transB else B

    if _use_nki() and a.dim() == 2 and b.dim() == 2:
        result = alpha * nki_gemm(a, b)
    else:
        result = alpha * torch.matmul(a, b)

    if C is not None and beta != 0.0:
        result = result + beta * C

    return result


def batched_gemm(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float = 0.0,
    C: torch.Tensor | None = None,
    transA: bool = False,
    transB: bool = False,
) -> torch.Tensor:
    """Batched GEMM: C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i]

    A, B have shape (batch, M, K) and (batch, K, N) respectively.

    Critical for DF-MP2 energy evaluation where contractions are
    independent across auxiliary basis indices.
    """
    from .nki.dispatch import nki_batched_gemm

    a = A.transpose(-2, -1) if transA else A
    b = B.transpose(-2, -1) if transB else B
    result = alpha * nki_batched_gemm(a, b)
    if C is not None and beta != 0.0:
        result = result + beta * C
    return result


def symm(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float = 0.0,
    C: torch.Tensor | None = None,
    side: str = "left",
    uplo: str = "upper",
) -> torch.Tensor:
    """Symmetric matrix multiply: C = alpha * A @ B + beta * C (side='left')
                              or C = alpha * B @ A + beta * C (side='right')

    A is symmetric; only upper or lower triangle referenced.
    Used in Fock matrix construction: F = H_core + J - K
    where J and K involve symmetric density matrix contractions.
    """
    if uplo == "upper":
        sym = torch.triu(A) + torch.triu(A, diagonal=1).T
    else:
        sym = torch.tril(A) + torch.tril(A, diagonal=-1).T

    if side == "left":
        result = alpha * torch.matmul(sym, B)
    else:
        result = alpha * torch.matmul(B, sym)

    if C is not None and beta != 0.0:
        result = result + beta * C
    return result


def syrk(
    alpha: float,
    A: torch.Tensor,
    beta: float = 0.0,
    C: torch.Tensor | None = None,
    trans: bool = False,
    uplo: str = "upper",
) -> torch.Tensor:
    """Symmetric rank-k update: C = alpha * A @ A^T + beta * C  (trans=False)
                            or C = alpha * A^T @ A + beta * C  (trans=True)

    Used in metric contraction for density fitting:
    J_{PQ} = Σ_μν (P|μν)(μν|Q) which is A^T @ A form.

    On Trainium, dispatches to NKI SYRK — a single-operand matmul kernel
    that avoids the materialised `A.T.contiguous()` copy of the naive
    `gemm(A, A.T)` call pattern.
    """
    from .nki import nki_syrk

    A_eff = A.T.contiguous() if trans else A
    result = alpha * nki_syrk(A_eff)

    if C is not None and beta != 0.0:
        result = result + beta * C

    # Symmetrize (fp32 reduction-order rounding can leave tiny asymmetry).
    result = 0.5 * (result + result.T)
    return result


def trsm(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    side: str = "left",
    uplo: str = "upper",
    trans: bool = False,
    diag: str = "nonunit",
) -> torch.Tensor:
    """Triangular solve with multiple RHS: op(A) @ X = alpha * B  (side='left')
                                       or X @ op(A) = alpha * B  (side='right')

    Returns X.

    Critical for Cholesky-based density fitting:
    Given J = L @ L^T, solve L @ X = (μν|P) for X,
    then B_ia^P = C^T @ X gives the DF coefficients.

    On Trainium + side='left', dispatches to a blocked NKI path: tiny
    diagonal panels solve on torch.linalg.solve_triangular; the trailing
    off-diagonal updates run through nki_gemm. side='right' and the
    non-NKI path both use torch.linalg.solve_triangular directly.
    """
    from .nki import nki_trsm

    return nki_trsm(A, B, side=side, uplo=uplo, trans=trans, diag=diag, alpha=alpha)


def trmm(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    side: str = "left",
    uplo: str = "upper",
    trans: bool = False,
    diag: str = "nonunit",
) -> torch.Tensor:
    """Triangular matrix multiply: B = alpha * op(A) @ B  (side='left')
    or B = alpha * B @ op(A)  (side='right')
    """
    if uplo == "upper":
        tri = torch.triu(A)
    else:
        tri = torch.tril(A)

    if diag == "unit":
        tri = (
            tri
            - torch.diag(torch.diag(tri))
            + torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        )

    mat = tri.T if trans else tri

    if side == "left":
        return alpha * torch.matmul(mat, B)
    else:
        return alpha * torch.matmul(B, mat)
