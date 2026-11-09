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

import warnings

import torch

from ._flags import diag_flag, side_flag, trans_flag, uplo_flag
from .nki.dispatch import _fp64_host_guard


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


@_fp64_host_guard("gemm")
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
    from .nki.dispatch import _neuron_matmul, _use_nki, nki_gemm

    transA, transB = trans_flag(transA, "transA"), trans_flag(transB, "transB")
    # `.mT`, not `.T`: on a batched operand `.T` reverses every dimension
    # (#72). 1-D operands are left as they are (`.T` was a no-op there).
    a = A.mT if transA and A.dim() >= 2 else A
    b = B.mT if transB and B.dim() >= 2 else B

    if _use_nki() and a.dim() == 2 and b.dim() == 2:
        # nki_gemm applies `_neuron_mm_lhs` on its aten route; the LNC kernel
        # takes a transposed `a` as is (#79).
        result = nki_gemm(a, b)
    else:
        result = _neuron_matmul(a, b)

    return _scale_add(result, alpha, beta, C)


@_fp64_host_guard("batched_gemm")
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


def gemm_fp64(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    slices: int = 8,
    transA: bool = False,
    transB: bool = False,
) -> torch.Tensor:
    """fp64-class matrix product op(A) @ op(B) on the bf16 Tensor Engine (#22).

    Ozaki splitting: the host splits A row by row and B column by column into
    `slices` (k) bf16 integer slices of 7 bits, exactly. On the device each
    diagonal s + t <= k - 1 of slice pairs runs as bf16 matmuls with exact fp32
    PSUM accumulation, in groups of at most 4096 terms (same-diagonal pairs
    concatenated along K), and each group is folded into a VectorE TwoSum
    (hi, lo) pair. The host recombines in fp64 with the power-of-two scales.

    `slices` is the precision knob: k (k + 1) / 2 slice GEMMs. Measured on
    Trn2 (max |C - C_exact| / (|A| |B|), normal data): about 1e-12 at k = 6,
    1e-14 at k = 7, 2e-16 to 3e-16 at k = 8 (the default; host fp64 BLAS
    level). Scaling is per row of A and column of B, so entries far below
    their row / column maximum lose relative precision: data spread over
    10^+-4 needs k = 10 for host-fp64 accuracy. 1 <= slices <= 12.

    The result is deterministic: every element is a fixed sequence of exact
    integer group sums and fp32 TwoSums, so it does not depend on the tiling or
    the LNC setting, and equals `trnblas.ozaki.reference_gemm` bit for bit.

    A (M, K) and B (K, N) are 2-D real tensors of any float dtype (computed as
    fp64); op() transposes when transA / transB. Returns float64 (M, N) on A's
    device. The kernel runs on the native TorchNeuron path (Trn2/Trn3);
    elsewhere, under set_backend("pytorch"), and for non-finite operands the
    result is host fp64 `A @ B`. For alpha / beta, scale the fp64 result.
    """
    from .nki.dispatch import nki_gemm_fp64
    from .ozaki import check_slices

    slices = check_slices(slices)
    transA, transB = trans_flag(transA, "transA"), trans_flag(transB, "transB")
    for name, X in (("A", A), ("B", B)):
        if not isinstance(X, torch.Tensor) or X.dim() != 2:
            raise ValueError(f"gemm_fp64: {name} must be a 2-D tensor")
        if not X.is_floating_point():
            raise TypeError(f"gemm_fp64: {name} must be a real floating tensor, got {X.dtype}")
    a = A.mT if transA else A
    b = B.mT if transB else B
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"gemm_fp64: inner dimensions differ: op(A) {tuple(a.shape)}, op(B) {tuple(b.shape)}"
        )
    device = A.device
    a = a.detach().to("cpu", torch.float64)
    b = b.detach().to("cpu", torch.float64)
    return nki_gemm_fp64(a, b, slices).to(device)


@_fp64_host_guard("symm", kernel=False)
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
    from .nki.dispatch import _aten_on_device, _gemm_route, _neuron_matmul, _on_neuron_device

    side, uplo = side_flag(side), uplo_flag(uplo)

    def _symm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if uplo == "upper":
            sym = torch.triu(A) + torch.triu(A, diagonal=1).T
        else:
            sym = torch.tril(A) + torch.tril(A, diagonal=-1).T
        return _neuron_matmul(sym, B) if side == "left" else _neuron_matmul(B, sym)

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


@_fp64_host_guard("syrk")
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

    Returns the full symmetric matrix. As in BLAS (``dsyrk``,
    ``cublasSsyrk``), only the `uplo` triangle of an ``n x n`` C is
    referenced; the other triangle is ignored. (Through 0.10.0 `uplo` was
    ignored and the sum was symmetrised, #74.) A C whose off-diagonal entries
    are all in the opposite triangle still uses that triangle, with a
    ``DeprecationWarning``; pass the matching `uplo`. A C that broadcasts
    against the result (scalar, row, column) is added and the sum
    symmetrised.
    """
    from .nki import nki_syrk
    from .nki.dispatch import _is_neuron, _out_block, _symmetrize

    trans = trans_flag(trans)
    uplo = uplo_flag(uplo)
    A_eff = A.T.contiguous() if trans else A
    if trans and _is_neuron(A) and _out_block(*A_eff.shape, A.element_size()):
        # #112: a device copy into that layout is itself unsafe; A^T as a view
        # (A's own short rows) is a safe operand of the blocked product.
        A_eff = A.T
    add = C is not None and beta != 0.0
    n = A_eff.shape[0]
    if add and C.dim() == 2 and tuple(C.shape) == (n, n):
        # BLAS: only the `uplo` triangle of C is referenced (#74). alpha * S
        # + beta * sym(C) is exactly symmetric when both terms are.
        Cs = _sym_from_triangle(C, _resolve_syrk_uplo(C, uplo))
        return _scale_add(nki_syrk(A_eff, symmetrize=True), alpha, beta, Cs)
    # Symmetrize (fp32 reduction-order rounding can leave tiny asymmetry).
    # Without C, symmetrize A A^T before scaling: alpha * sym(S) is exactly
    # symmetric, and a large host-input result is symmetrised on the device
    # (#61). A broadcast C (scalar, row, column) has no triangle: symmetrise
    # the sum, as before.
    result = _scale_add(nki_syrk(A_eff, symmetrize=not add), alpha, beta, C)
    return _symmetrize(result) if add else result


def _sym_from_triangle(C: torch.Tensor, uplo: str) -> torch.Tensor:
    """The symmetric (not Hermitian) matrix stored in the `uplo` triangle of C."""
    if uplo == "upper":
        return torch.triu(C) + torch.triu(C, diagonal=1).mT
    return torch.tril(C) + torch.tril(C, diagonal=-1).mT


def _resolve_syrk_uplo(C: torch.Tensor, uplo: str) -> str:
    """`uplo`, or the opposite triangle when only that one holds entries.

    Through 0.10.0 `syrk` ignored `uplo` and symmetrised the sum, so a C
    stored in the lower triangle with the default ``uplo="upper"`` gave a
    result built from that triangle. Strict BLAS would reference only the
    diagonal of such a C. Keep using the stored triangle, with a warning; a C
    with entries in both triangles references only `uplo` (same rule as
    trnsparse#52/#53).
    """
    has_upper = bool(torch.triu(C, diagonal=1).ne(0).any())
    has_lower = bool(torch.tril(C, diagonal=-1).ne(0).any())
    other = "lower" if uplo == "upper" else "upper"
    in_uplo, in_other = (has_upper, has_lower) if uplo == "upper" else (has_lower, has_upper)
    if in_other and not in_uplo:
        warnings.warn(
            f"syrk: C stores its off-diagonal entries only in the {other} triangle "
            f"but uplo={uplo!r}; using the {other} triangle. Pass uplo={other!r}. "
            "Since 0.11.0 only the uplo triangle of C is referenced when both "
            "are stored, and this fallback will be removed (#74).",
            DeprecationWarning,
            stacklevel=3,
        )
        return other
    return uplo


@_fp64_host_guard("trsm")
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


@_fp64_host_guard("metric_inv_half")
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


@_fp64_host_guard("trmm", kernel=False)
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

    from .nki.dispatch import _neuron_matmul

    if side == "left":
        return alpha * _neuron_matmul(mat, B)
    else:
        return alpha * _neuron_matmul(B, mat)
