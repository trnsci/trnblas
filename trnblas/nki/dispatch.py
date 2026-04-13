"""
NKI dispatch for BLAS operations.

Backend selection mirrors trnfft: auto/pytorch/nki.
The GEMM kernel is the primary acceleration target — it uses stationary
tile reuse on the Tensor Engine for 2x fewer SBUF loads vs naive.
"""

from __future__ import annotations

import os
import warnings

import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

# When set, kernel-path failures re-raise instead of falling back to PyTorch.
# Used by the validation suite to catch silent kernel breakage during iteration.
_REQUIRE_NKI = os.environ.get("TRNBLAS_REQUIRE_NKI", "").lower() in ("1", "true", "yes")


class NkiFallbackWarning(UserWarning):
    """Emitted once per distinct error when the NKI path silently falls
    back to torch.matmul. Prevents the class of bug where PATH / plugin
    misconfiguration causes every NKI call to hit torch without any
    user-visible signal — the v0.4.x-era 'libneuronpjrt-path' silent
    fallback is the motivating example.
    """


_fallback_warned: set[str] = set()


def _warn_fallback(exc: BaseException) -> None:
    """Emit NkiFallbackWarning once per unique error signature."""
    key = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    if key in _fallback_warned:
        return
    _fallback_warned.add(key)
    warnings.warn(
        f"NKI kernel dispatch failed — falling back to torch.matmul "
        f"(set TRNBLAS_REQUIRE_NKI=1 to re-raise). First error: {key}",
        NkiFallbackWarning,
        stacklevel=3,
    )

# Tile shapes for the systolic array (NKI 2.24 limits):
# stationary partition ≤ 128 (= K), free ≤ 128 (= M); moving free ≤ 512 (= N).
_TILE_M = 128
_TILE_K = 128
_TILE_N = 512

_backend = "auto"


def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires neuronxcc")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def nki_batched_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched GEMM. A: (batch, M, K), B: (batch, K, N) → C: (batch, M, N).

    Loops over the batch dim, dispatching the 2D `_nki_gemm_impl` per
    slice. Every slice after the first hits the NEFF cache (identical
    kernel signature), so per-slice cost is HBM transfer + Tensor Engine
    dispatch only. A true 3D-batched NKI kernel is a future optimisation
    if benchmarks justify it.

    For DF-MP2 tensor contractions over auxiliary basis indices, the
    natural use case is one batched_gemm with batch=N_aux per orbital
    pair — exactly this loop's sweet spot.
    """
    if not _use_nki():
        return torch.bmm(A, B)
    return torch.stack([_nki_gemm_impl(A[i], B[i]) for i in range(A.shape[0])])


def _torch_mp2_energy(
    T_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for the fused MP2 energy reduction.

    T_flat: (ic*nvir, nocc*nvir). eps_occ_chunk: (ic,). eps_occ_full:
    (nocc,). eps_vir: (nvir,). Returns a 0-d tensor — sum of
    T*(2T - T.T)/denom over the chunk. Mirrors the expression in
    examples/df_mp2.py so the NKI path can be swapped in transparently.
    """
    ic = eps_occ_chunk.shape[0]
    nocc = eps_occ_full.shape[0]
    nvir = eps_vir.shape[0]
    T = T_flat.reshape(ic, nvir, nocc, nvir).permute(0, 2, 1, 3)
    denom = (
        eps_occ_chunk.view(ic, 1, 1, 1)
        + eps_occ_full.view(1, nocc, 1, 1)
        - eps_vir.view(1, 1, nvir, 1)
        - eps_vir.view(1, 1, 1, nvir)
    )
    return (T * (2.0 * T - T.transpose(-2, -1)) / denom).sum()


def nki_mp2_energy(
    T_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Fused MP2 energy reduction — NKI-dispatched (#15).

    On NKI backend: a single kernel streams T_flat tiles on-chip and
    computes T*(2T - T.T)/denom + sum in one pass, avoiding the four
    HBM round-trips of the torch expression.

    On PyTorch backend (or when the kernel can't handle the shape
    yet): falls back to the torch reference.
    """
    if not _use_nki():
        return _torch_mp2_energy(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)
    try:
        return _nki_mp2_energy_impl(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return _torch_mp2_energy(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)


def _nki_mp2_energy_impl(
    T_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    # Pass eps_* in the orientation each access needs. NKI's partition
    # dim is physical; we can't reshape a partition=1 SBUF tile to
    # partition=N in the kernel ('illegal partition step' BIR error).
    #   eps_vir_col: (NVIR, 1) — strip loads pick (P_TILE, 1) slices
    #   eps_vir_row: (1, NVIR) — full free-dim vector for broadcast
    #   eps_occ_*:   (1, N)    — (1,1) scalars picked from the row
    (t, eo_c, eo_f, ev_col, ev_row), orig_device = _to_xla(
        T_flat.contiguous(),
        eps_occ_chunk.reshape(1, -1).contiguous(),
        eps_occ_full.reshape(1, -1).contiguous(),
        eps_vir.reshape(-1, 1).contiguous(),
        eps_vir.reshape(1, -1).contiguous(),
    )
    partial = _mp2_energy_kernel(t, eo_c, eo_f, ev_col, ev_row)
    return partial.to(orig_device).sum()


def nki_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """GEMM with NKI dispatch.

    On NKI backend: uses tiled GEMM with stationary A reuse.
    On PyTorch backend: torch.matmul.

    NKI GEMM strategy (stationary tile reuse):
        1. Load A tile to SBUF as stationary (stays in systolic array)
        2. Stream B tiles as moving → accumulate in PSUM
        3. One A load serves multiple B tiles

    For DF-MP2 tensor contractions where A is the MO coefficient matrix
    (reused across auxiliary basis indices), this cuts SBUF loads in half.
    """
    if _use_nki():
        return _nki_gemm_impl(A, B)
    return torch.matmul(A, B)


def nki_syrk(A: torch.Tensor) -> torch.Tensor:
    """SYRK via single-operand NKI matmul. Returns A @ A.T.

    On NKI: dispatches `_syrk_kernel`, which loads A directly for both
    operand roles (avoids the A.T.contiguous() HBM write that would
    happen if we just called `nki_gemm(A, A.T)`).

    On PyTorch: falls back to `torch.matmul(A, A.T)`.
    """
    if _use_nki():
        return _nki_syrk_impl(A)
    return torch.matmul(A, A.T)


def nki_trsm(
    A: torch.Tensor,
    B: torch.Tensor,
    side: str = "left",
    uplo: str = "upper",
    trans: bool = False,
    diag: str = "nonunit",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Blocked triangular solve: op(A) X = alpha * B (side='left') or
    X op(A) = alpha * B (side='right').

    On NKI + side='left': blocked panel algorithm — the diagonal panel
    solve stays on torch.linalg.solve_triangular (tiny P×P, intrinsically
    sequential), while the trailing off-diagonal update is one nki_gemm
    call per block. GEMM dominates the work for large M, so this
    captures most of the speedup without writing a substitution kernel.

    Falls back to torch for side='right' (uncommon in chemistry hot
    paths) or when _use_nki() is False.
    """
    if side != "left" or not _use_nki():
        return _trsm_torch(alpha, A, B, side, uplo, trans, diag)
    try:
        return _nki_trsm_left(A, B, uplo, trans, diag, alpha)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return _trsm_torch(alpha, A, B, side, uplo, trans, diag)


def _trsm_torch(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    side: str,
    uplo: str,
    trans: bool,
    diag: str,
) -> torch.Tensor:
    """Pure-torch TRSM reference. Mirrors the body of the original
    `trnblas.trsm` so the NKI dispatch wrapper has a pinned fallback
    that is independent of the public wrapper's evolution.
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
        upper_flag = (uplo == "upper" and not trans) or (uplo == "lower" and trans)
        return alpha * torch.linalg.solve_triangular(mat, B, upper=upper_flag)
    upper_flag = (uplo == "lower" and not trans) or (uplo == "upper" and trans)
    return alpha * torch.linalg.solve_triangular(mat.T, B.T, upper=upper_flag).T


def _nki_trsm_left(
    A: torch.Tensor,
    B: torch.Tensor,
    uplo: str,
    trans: bool,
    diag: str,
    alpha: float,
    block: int = 128,
) -> torch.Tensor:
    """Blocked left-side TRSM. Diagonal panels solved via
    torch.linalg.solve_triangular (small, strictly sequential);
    trailing updates via nki_gemm (dominant work for large M).
    """
    if trans:
        mat = A.T.contiguous()
        eff_upper = uplo == "lower"
    else:
        mat = A
        eff_upper = uplo == "upper"

    M = B.shape[0]
    unit = diag == "unit"

    # Small M: skip blocking — direct solve is cheap enough that
    # blocking only adds Python-loop overhead.
    if M <= block:
        X = torch.linalg.solve_triangular(
            mat, B, upper=eff_upper, unitriangular=unit
        )
        return alpha * X

    X = B.clone()
    if not eff_upper:
        # Lower triangular: forward substitution.
        for k in range(0, M, block):
            ke = min(k + block, M)
            X[k:ke] = torch.linalg.solve_triangular(
                mat[k:ke, k:ke],
                X[k:ke],
                upper=False,
                unitriangular=unit,
            )
            if ke < M:
                X[ke:] = X[ke:] - nki_gemm(
                    mat[ke:, k:ke].contiguous(), X[k:ke]
                )
    else:
        # Upper triangular: back substitution.
        for k in range(M, 0, -block):
            ks = max(k - block, 0)
            X[ks:k] = torch.linalg.solve_triangular(
                mat[ks:k, ks:k],
                X[ks:k],
                upper=True,
                unitriangular=unit,
            )
            if ks > 0:
                X[:ks] = X[:ks] - nki_gemm(
                    mat[:ks, ks:k].contiguous(), X[ks:k]
                )
    return alpha * X


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _to_xla(*tensors):
    """Move tensors to the XLA device for NKI kernel dispatch."""
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


def _nki_gemm_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """NKI GEMM implementation.

    Pads M/K up to TILE_M/TILE_K and N up to TILE_N (only when N > TILE_N
    and not already a multiple) before dispatching the aligned kernel.
    The result is sliced back to the original (M, N).

    Set `TRNBLAS_REQUIRE_NKI=1` to re-raise on kernel errors instead of
    falling back to `torch.matmul`; useful in the validation loop to
    surface silent kernel breakage.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    _, N = B.shape
    M_pad = _round_up(M, _TILE_M)
    K_pad = _round_up(K, _TILE_K)
    # When N <= TILE_N, the kernel uses TILE_N = N (single N-tile, no remainder).
    # Otherwise we need N to be a clean multiple of TILE_N.
    N_pad = N if N <= _TILE_N else _round_up(N, _TILE_N)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, K_pad, dtype=A.dtype, device=A.device)
            A_p[:M, :K] = A
            B_p = torch.zeros(K_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:K, :N] = B
            (a, b), orig_device = _to_xla(A_p.contiguous(), B_p.contiguous())
        else:
            (a, b), orig_device = _to_xla(A.contiguous(), B.contiguous())
        c = _gemm_kernel(a, b)
        result = c.to(orig_device)
        return result[:M, :N] if needs_pad else result
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return torch.matmul(A, B)


def _nki_syrk_impl(A: torch.Tensor) -> torch.Tensor:
    """NKI SYRK implementation. Returns A @ A.T for A of shape (M, K).

    Pads M to TILE_M and K to TILE_K multiples. For M > TILE_N, pads to
    a multiple of TILE_N in the N (= M) direction so the kernel can tile
    output cleanly. Falls back to torch.matmul on kernel errors unless
    TRNBLAS_REQUIRE_NKI=1.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    M_pad = _round_up(M, _TILE_M)
    K_pad = _round_up(K, _TILE_K)
    # Output is (M_pad, M_pad); same TILE_N logic as GEMM applies.
    N_pad = M_pad if M_pad <= _TILE_N else _round_up(M_pad, _TILE_N)
    # M_pad must also equal N_pad (output is square); enforce.
    M_pad = max(M_pad, N_pad)
    needs_pad = (M_pad != M) or (K_pad != K)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, K_pad, dtype=A.dtype, device=A.device)
            A_p[:M, :K] = A
            (a,), orig_device = _to_xla(A_p.contiguous())
        else:
            (a,), orig_device = _to_xla(A.contiguous())
        c = _syrk_kernel(a)
        result = c.to(orig_device)
        return result[:M, :M] if needs_pad else result
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return torch.matmul(A, A.T)


if HAS_NKI:

    @nki.jit
    def _gemm_kernel(a, b):
        """Real GEMM: C = A @ B with stationary tile reuse.

        Caller guarantees M, K are multiples of 128 and N is either ≤ 512
        or a multiple of 512 (handled by the dispatch wrapper's HBM
        padding). PSUM accumulates over K-tiles before the single store
        per (m, n) tile pair.

        NKI 2.24 calling convention (`nisa.nc_matmul`):
            stationary: (TILE_K, TILE_M)  partition=K ≤ 128, free ≤ 128
            moving:     (TILE_K, TILE_N)  partition=K, free ≤ 512
            psum:       (TILE_M, TILE_N)  fp32, in nl.psum
        """
        M, K = a.shape
        _, N = b.shape

        TILE_M = _TILE_M
        TILE_K = _TILE_K
        TILE_N = N if N <= _TILE_N else _TILE_N

        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    # Load A row-tile transposed so partition dim = K.
                    a_t = nl.load_transpose2d(
                        a[m_off:m_off + TILE_M, k_off:k_off + TILE_K]
                    )
                    # B is already K-major.
                    b_tile = nl.load(
                        b[k_off:k_off + TILE_K, n_off:n_off + TILE_N]
                    )

                    psum[...] += nisa.nc_matmul(a_t, b_tile)

                c_sbuf = nl.copy(psum, dtype=a.dtype)
                nl.store(
                    c[m_off:m_off + TILE_M, n_off:n_off + TILE_N],
                    value=c_sbuf,
                )

        return c

    @nki.jit
    def _mp2_energy_kernel(
        T_flat, eps_occ_chunk, eps_occ_full, eps_vir_col, eps_vir_row
    ):
        """Fused MP2 energy reduction (#15).

        Computes Σ_{i<ic, j<nocc, a,b<nvir} T[i,j,a,b] *
        (2 T[i,j,a,b] - T[i,j,b,a]) / denom[i,j,a,b] where
        T[i,j,a,b] = T_flat[i*nvir + a, j*nvir + b].

        Sub-tiles the nvir partition dim into strips of P_TILE ≤ 128
        (largest divisor of NVIR under the NKI partition limit).
        Strip partials are accumulated in a 1×1 SBUF register per
        (i, j), so there is exactly one HBM store per (i, j) —
        cuts store traffic by NSTRIP× vs the per-strip-store variant.

        eps_* args are shape (1, N) so `nl.load` interprets them as
        partition=1, free=N (a 1D load is treated as partition=len,
        which would exceed the 128-partition limit for NVIR > 128).

        NKI only supports reduction along the free dim. A full
        `(P_TILE, NVIR) → scalar` reduce would need a partition-axis
        reduce which NKI rejects. Instead: reduce free-only to get
        per-partition partials `(P_TILE, 1)`, accumulate across
        strips in an SBUF row-accumulator, emit
        `(IC, NOCC, P_TILE)` to HBM — caller `.sum()` handles the
        final partition-axis reduction host-side (partial is small;
        ≤ 258 KB at large bench shape).
        """
        NVIR = eps_vir_row.shape[1]
        IC = eps_occ_chunk.shape[1]
        NOCC = eps_occ_full.shape[1]

        P_TILE = min(NVIR, 128)
        while NVIR % P_TILE != 0:
            P_TILE -= 1
        NSTRIP = NVIR // P_TILE

        # Output layout: partition axis (P_TILE) FIRST so nl.store
        # writes the (P_TILE, 1) SBUF tile with partition-to-partition
        # alignment. Host caller's .sum() is layout-agnostic.
        e_partial = nl.ndarray(
            (P_TILE, IC, NOCC), dtype=nl.float32, buffer=nl.shared_hbm
        )
        # Full eps_vir as a free-dim vector (partition=1, free=NVIR)
        # for the per-b axis of denom.
        ev_row = nl.load(eps_vir_row[0:1, 0:NVIR])

        for i in nl.affine_range(IC):
            eo_i = nl.load(eps_occ_chunk[0:1, i:i + 1])
            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ_full[0:1, j:j + 1])
                eo_sum = nl.add(eo_i, eo_j)

                # Per-strip SBUF slots. Each affine_range iteration
                # writes its own column; a single nl.sum over the
                # NSTRIP free axis at the end reduces strips.
                # (In-place += across affine_range hits NKI's
                # "Unexpected output dependencies" — the compiler
                # wants the strip index in the dst access explicitly.)
                acc_rows = nl.zeros(
                    (P_TILE, NSTRIP), dtype=nl.float32, buffer=nl.sbuf
                )

                for s in nl.affine_range(NSTRIP):
                    a_off = s * P_TILE
                    t = nl.load(
                        T_flat[i * NVIR + a_off : i * NVIR + a_off + P_TILE,
                               j * NVIR : (j + 1) * NVIR]
                    )
                    t_T = nl.load_transpose2d(
                        T_flat[i * NVIR : (i + 1) * NVIR,
                               j * NVIR + a_off : j * NVIR + a_off + P_TILE]
                    )
                    # (P_TILE, 1) partition-axis load of eps_vir's
                    # a-strip — matches the output axis of the (P_TILE,
                    # NVIR) tile we're operating on. No partition
                    # reshape needed anywhere.
                    ev_col = nl.load(
                        eps_vir_col[a_off : a_off + P_TILE, 0:1]
                    )
                    # denom[a, b] = eo_sum - ev_col[a] - ev_row[b]
                    # Broadcast: (1,1) - (P_TILE,1) = (P_TILE,1);
                    #            then - (1,NVIR) = (P_TILE,NVIR).
                    denom = nl.subtract(
                        nl.subtract(eo_sum, ev_col),
                        ev_row,
                    )
                    term = nl.divide(
                        nl.multiply(
                            t,
                            nl.subtract(nl.multiply(t, 2.0), t_T),
                        ),
                        denom,
                    )
                    # Free-dim reduce: (P_TILE, NVIR) → (P_TILE, 1).
                    strip_partial = nl.sum(term, axis=1, keepdims=True)
                    # Write the strip's partial into its own slot (s
                    # indexes the free axis of acc_rows).
                    acc_rows[0:P_TILE, s:s + 1] = strip_partial

                # Reduce across strips (free dim) → (P_TILE, 1).
                acc_row = nl.sum(acc_rows, axis=1, keepdims=True)

                # Store (P_TILE,) per-partition partials for this (i, j)
                # into the partition-major output; axes align directly.
                nl.store(
                    e_partial[0:P_TILE, i:i + 1, j:j + 1],
                    value=acc_row,
                )

        return e_partial

    @nki.jit
    def _syrk_kernel(a):
        """Symmetric rank-k: C = a @ a.T with single-A HBM load.

        Structurally identical to _gemm_kernel, but the "moving"
        operand is loaded from the same `a` HBM region via a
        second `load_transpose2d` — avoiding the materialised
        `a.T.contiguous()` that `nki_gemm(A, A.T)` would otherwise
        issue. K partition dim is at the load_transpose2d limit of 128.

        Caller guarantees M, K are multiples of 128 and M is either
        ≤ 512 or a multiple of 512 (handled by _nki_syrk_impl's HBM
        padding).
        """
        M, K = a.shape

        TILE_M = _TILE_M
        TILE_K = _TILE_K
        TILE_N = M if M <= _TILE_N else _TILE_N

        c = nl.ndarray((M, M), dtype=a.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(M // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum = nl.zeros(
                    (TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum
                )

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    # Stationary: a[m_off:m_off+TILE_M, k_off:k_off+TILE_K]
                    # transposed → (TILE_K, TILE_M), partition=K ≤ 128.
                    a_stat = nl.load_transpose2d(
                        a[m_off:m_off + TILE_M, k_off:k_off + TILE_K]
                    )
                    # Moving: a.T[k_off:k_off+TILE_K, n_off:n_off+TILE_N]
                    # = a[n_off:n_off+TILE_N, k_off:k_off+TILE_K].T
                    # load_transpose2d swaps axes → (TILE_K, TILE_N),
                    # partition=K ≤ 128.
                    a_mov = nl.load_transpose2d(
                        a[n_off:n_off + TILE_N, k_off:k_off + TILE_K]
                    )
                    psum[...] += nisa.nc_matmul(a_stat, a_mov)

                c_sbuf = nl.copy(psum, dtype=a.dtype)
                nl.store(
                    c[m_off:m_off + TILE_M, n_off:n_off + TILE_N],
                    value=c_sbuf,
                )

        return c
