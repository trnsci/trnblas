"""
BLAS Level 3 — Matrix-matrix operations.

gemm, symm, syrk, trsm, trmm

These are the critical path for scientific computing on Trainium:
- DF-MP2 tensor contractions → batched GEMM
- Fock matrix build → SYMM
- Cholesky-based density fitting → TRSM
- Metric contraction J^{-1/2} → SYRK

NKI dispatch for GEMM provides stationary tile reuse on the Tensor Engine.
On the native (TorchNeuron) path, `auto` routes the GEMM family by measured
time instead: aten on the `neuron` device for device-resident operands, the
host CPU or upload + aten for host operands.
"""

from __future__ import annotations

import torch

from ._flags import diag_flag, side_flag, trans_flag, uplo_flag


def _owned_inplace_ok(result: torch.Tensor) -> bool:
    """True when `result` (a fresh tensor the wrapper owns) may be updated in place.

    Not under autograd, and only for contiguous floating/complex results, so
    the returned tensor keeps the dense layout and dtype that `alpha * result`
    gave.
    """
    return (
        not (torch.is_grad_enabled() and result.requires_grad)
        and result.is_contiguous()
        and (result.is_floating_point() or result.is_complex())
    )


def _scale_add(
    result: torch.Tensor, alpha: float, beta: float, C: torch.Tensor | None
) -> torch.Tensor:
    """`alpha * result + beta * C` for a fresh `result` the wrapper owns.

    Written in place when that gives the same tensor: for host results, each
    out-of-place step is another pass into freshly allocated memory, which is
    bound by page faults (~4.6 GB/s on trn2, #61), not by the arithmetic.
    """
    add = C is not None and beta != 0.0
    inplace = _owned_inplace_ok(result)
    if inplace and add:
        inplace = (
            C.dtype == result.dtype
            and C.device == result.device
            and torch.broadcast_shapes(result.shape, C.shape) == result.shape
        )
    if not inplace:
        result = alpha * result
        return result + beta * C if add else result
    if alpha != 1.0:
        result.mul_(alpha)
    if add:
        result.add_(C, alpha=beta)
    return result


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

    op(X) = X if trans=False, X^T if trans=True. For batched (>= 3-D)
    operands, op() transposes the last two dimensions, as in `batched_gemm`.

    This is the workhorse for DF-MP2:
        (ia|P) = Σ_μν C_μi @ (μν|P) @ C_νa    → two GEMMs per aux index
        B_ia^P = (ia|Q) @ J^{-1/2}_{QP}        → one GEMM
        E_MP2 = Σ B_ia^P @ B_jb^P              → batched GEMM

    On the XLA path (Trn1) and under set_backend("nki"), dispatches to NKI
    GEMM with stationary tile reuse. On the native path (Trn2/Trn3), `auto`
    runs device-resident operands as aten on the device and host operands on
    the host CPU or as upload + aten, by measured time.
    """
    from .nki.dispatch import _neuron_mm_lhs, _use_nki, nki_gemm

    transA, transB = trans_flag(transA, "transA"), trans_flag(transB, "transB")
    # `.mT`, not `.T`: on a batched operand `.T` reverses every dimension
    # (#72). 1-D operands are left as they are (`.T` was a no-op there).
    a = A.mT if transA and A.dim() >= 2 else A
    b = B.mT if transB and B.dim() >= 2 else B
    a = _neuron_mm_lhs(a, b)

    if _use_nki() and a.dim() == 2 and b.dim() == 2:
        result = nki_gemm(a, b)
    else:
        result = torch.matmul(a, b)

    return _scale_add(result, alpha, beta, C)


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

    transA, transB = trans_flag(transA, "transA"), trans_flag(transB, "transB")
    a = A.transpose(-2, -1) if transA else A
    b = B.transpose(-2, -1) if transB else B
    return _scale_add(nki_batched_gemm(a, b), alpha, beta, C)


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
    from .nki.dispatch import _aten_on_device, _gemm_route, _on_neuron_device

    side, uplo = side_flag(side), uplo_flag(uplo)

    def _symm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if uplo == "upper":
            sym = torch.triu(A) + torch.triu(A, diagonal=1).T
        else:
            sym = torch.tril(A) + torch.tril(A, diagonal=-1).T
        return torch.matmul(sym, B) if side == "left" else torch.matmul(B, sym)

    n = A.shape[0]
    out_numel = B.numel()
    route = _gemm_route(
        A,
        B,
        op="symm",
        macs=n * out_numel,
        nbytes=(A.numel() + B.numel() + out_numel) * A.element_size(),
    )
    # No NKI SYMM kernel: the "nki" route is torch on the operands' device,
    # as before. On the native path, large host problems run as aten on the
    # device (#62).
    if route == "aten" and not _on_neuron_device(A, B):
        result = _aten_on_device(_symm, A, B)
    else:
        result = _symm(A, B)
    return _scale_add(result, alpha, beta, C)


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

    On the XLA path (Trn1) and under set_backend("nki"), dispatches to NKI
    SYRK — a single-operand matmul kernel that avoids the materialised
    `A.T.contiguous()` copy of the naive `gemm(A, A.T)` call pattern. On
    the native path, `auto` routes it like `gemm`.
    """
    from .nki import nki_syrk
    from .nki.dispatch import _symmetrize

    trans = trans_flag(trans)
    uplo_flag(uplo)  # validated; the result is the full symmetric matrix (#74)
    A_eff = A.T.contiguous() if trans else A
    # Symmetrize (fp32 reduction-order rounding can leave tiny asymmetry).
    # Without C, symmetrize A A^T before scaling: alpha * sym(S) is exactly
    # symmetric, and a large host-input result is symmetrised on the device
    # (#61). With C, symmetrize the sum, as before.
    add = C is not None and beta != 0.0
    result = _scale_add(nki_syrk(A_eff, symmetrize=not add), alpha, beta, C)
    return _symmetrize(result) if add else result


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

    On Trainium + side='left', dispatches to a blocked path: tiny diagonal
    panels solve on torch.linalg.solve_triangular; the trailing off-diagonal
    updates run through nki_gemm (XLA path, set_backend("nki")) or aten on
    the device (native `auto`). On the native path, host operands solve with
    LAPACK on the host. side='right' and the non-NKI path both use
    torch.linalg.solve_triangular directly.
    """
    from .nki import nki_trsm

    side, uplo, diag = side_flag(side), uplo_flag(uplo), diag_flag(diag)
    trans = trans_flag(trans)
    return nki_trsm(A, B, side=side, uplo=uplo, trans=trans, diag=diag, alpha=alpha)


def metric_inv_half(J: torch.Tensor) -> torch.Tensor:
    """Cholesky inverse factor of an SPD metric: X = L^-T where J = L @ L^T.

    X @ X^T = J^-1, so `(ia|Q) @ X` is the density-fitting metric
    contraction B_ia^P = sum_Q (ia|Q) [J^-1/2]_QP used by DF-MP2.

    On Trainium's native path with J on the `neuron` device (and naux up to
    the measured crossover), the factor is computed on the host with LAPACK
    in float64 and uploaded once; otherwise it is `torch.linalg.cholesky`
    followed by `trsm(1.0, L, I, uplo="lower", trans=True)`.
    `TRNBLAS_METRIC_SOLVE=auto|host|device` overrides the choice.
    """
    from .nki.dispatch import nki_metric_inv_half

    return nki_metric_inv_half(J)


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
    side, uplo, diag = side_flag(side), uplo_flag(uplo), diag_flag(diag)
    trans = trans_flag(trans)
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
