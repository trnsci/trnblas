"""
NKI dispatch for BLAS operations.

Backend selection mirrors trnfft: auto/pytorch/nki.
The GEMM kernel is the primary acceleration target — it uses stationary
tile reuse on the Tensor Engine for 2x fewer SBUF loads vs naive.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

# When set, kernel-path failures re-raise instead of falling back to PyTorch.
# Used by the validation suite to catch silent kernel breakage during iteration.
_REQUIRE_NKI = os.environ.get("TRNBLAS_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. For
# correctness iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNBLAS_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


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

# Autotuner — sweeps tile candidates on hardware once per shape bucket and
# caches the winner to disk. Disabled in simulator mode and by TRNBLAS_AUTOTUNE=0.
_AUTOTUNE_ENABLED: bool = os.environ.get("TRNBLAS_AUTOTUNE", "1") != "0"
_AUTOTUNE_CACHE_FILE: Path = Path(
    os.environ.get("TRNBLAS_AUTOTUNE_CACHE", "/var/tmp/trnblas-autotune/cache.json")
)
# Tile candidates: {64,128} × {128} × {128,256,512}.
# Default (128,128,512) is always included; filtered by divisibility at sweep time.
_TILE_CANDIDATES: list[tuple[int, int, int]] = [
    (64, 128, 128),
    (64, 128, 256),
    (64, 128, 512),
    (128, 128, 128),
    (128, 128, 256),
    (128, 128, 512),
]
# Registry: tile config → cached @nki.jit kernel closure.
_gemm_kernel_registry: dict[tuple[int, int, int], Any] = {}
# Persistent autotune results: shape bucket → winning tile config.
_autotune_mem: dict[tuple[int, int, int], tuple[int, int, int]] = {}
_autotune_loaded: bool = False

_backend = "auto"


def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires nki>=0.3.0 (Neuron SDK 2.29+)")
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
    t_in = T_flat.contiguous()
    eo_c_in = eps_occ_chunk.reshape(1, -1).contiguous()
    eo_f_in = eps_occ_full.reshape(1, -1).contiguous()
    ev_col_in = eps_vir.reshape(-1, 1).contiguous()
    ev_row_in = eps_vir.reshape(1, -1).contiguous()

    if _use_simulator():
        partial_np = nki.simulate(_mp2_energy_kernel)(
            t_in.cpu().numpy(),
            eo_c_in.cpu().numpy(),
            eo_f_in.cpu().numpy(),
            ev_col_in.cpu().numpy(),
            ev_row_in.cpu().numpy(),
        )
        partial = torch.from_numpy(np.asarray(partial_np)).to(T_flat.device)
    else:
        (t, eo_c, eo_f, ev_col, ev_row), orig_device = _to_xla(
            t_in, eo_c_in, eo_f_in, ev_col_in, ev_row_in
        )
        partial = _mp2_energy_kernel(t, eo_c, eo_f, ev_col, ev_row).to(orig_device)
    return partial.sum()


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
    if block >= M:
        X = torch.linalg.solve_triangular(mat, B, upper=eff_upper, unitriangular=unit)
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
                X[ke:] = X[ke:] - nki_gemm(mat[ke:, k:ke].contiguous(), X[k:ke])
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
                X[:ks] = X[:ks] - nki_gemm(mat[:ks, ks:k].contiguous(), X[ks:k])
    return alpha * X


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _to_xla(*tensors):
    """Move tensors to the XLA device for NKI kernel dispatch."""
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


def _make_gemm_kernel(tile_m: int, tile_k: int, tile_n: int):
    """Return a new @nki.jit GEMM kernel with tile_m/tile_k/tile_n baked into
    the closure at trace time.  Each unique config produces a separately-cached
    NEFF; the XLA/NEFF cache amortises compile across the nocc² pair loop.
    Only callable when HAS_NKI is True.
    """
    tm, tk, tn = tile_m, tile_k, tile_n

    def _kernel(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = tm
        TILE_K = tk
        TILE_N = tn if tn < N else N  # single-tile shortcut when N fits
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=c_sbuf)
        return c

    return nki.jit(_kernel)


def _get_gemm_kernel(tile_m: int, tile_k: int, tile_n: int):
    """Return (possibly cached) @nki.jit kernel for the given tile config."""
    key = (tile_m, tile_k, tile_n)
    if key not in _gemm_kernel_registry:
        _gemm_kernel_registry[key] = _make_gemm_kernel(tile_m, tile_k, tile_n)
    return _gemm_kernel_registry[key]


def _ceil_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _autotune_bucket(M: int, K: int, N: int) -> tuple[int, int, int]:
    """Coarse shape bucket — same bucket for shapes within a DF-MP2 run."""
    return (_ceil_pow2(M), _ceil_pow2(K), _ceil_pow2(N))


def _load_autotune_cache() -> None:
    global _autotune_mem, _autotune_loaded
    if _autotune_loaded:
        return
    try:
        if _AUTOTUNE_CACHE_FILE.exists():
            raw = json.loads(_AUTOTUNE_CACHE_FILE.read_text())
            _autotune_mem = {
                tuple(map(int, k.split(","))): tuple(v)  # type: ignore[assignment]
                for k, v in raw.items()
            }
    except Exception:
        pass
    _autotune_loaded = True


def _save_autotune_cache() -> None:
    try:
        _AUTOTUNE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _AUTOTUNE_CACHE_FILE.write_text(
            json.dumps({",".join(map(str, k)): list(v) for k, v in _autotune_mem.items()})
        )
    except Exception:
        pass


def _sweep_tile_configs(M_pad: int, K_pad: int, N_pad: int, a_xla, b_xla) -> tuple[int, int, int]:
    """Time each candidate tile config on hardware; return the fastest.

    Skips configs that don't evenly divide the padded shape. Falls back to
    the default if every candidate errors or gets filtered out.
    """
    import time

    best: tuple[int, int, int] = (_TILE_M, _TILE_K, _TILE_N)
    best_t = float("inf")
    for tm, tk, tn in _TILE_CANDIDATES:
        if M_pad % tm or K_pad % tk or N_pad % tn:
            continue
        k = _get_gemm_kernel(tm, tk, tn)
        try:
            k(a_xla, b_xla)  # warm-up (compile + first run)
            t0 = time.perf_counter()
            for _ in range(3):
                k(a_xla, b_xla)
            t = (time.perf_counter() - t0) / 3
            if t < best_t:
                best, best_t = (tm, tk, tn), t
        except Exception:
            continue
    return best


def _sweep_on_default_pad(
    M: int, K: int, N: int, A: torch.Tensor, B: torch.Tensor
) -> tuple[int, int, int]:
    """Pad with default tile sizes to produce aligned sweep inputs, then sweep."""
    M_p = _round_up(M, _TILE_M)
    K_p = _round_up(K, _TILE_K)
    N_p = N if N <= _TILE_N else _round_up(N, _TILE_N)
    A_p = torch.zeros(M_p, K_p, dtype=A.dtype, device=A.device)
    A_p[:M, :K] = A
    B_p = torch.zeros(K_p, N_p, dtype=B.dtype, device=B.device)
    B_p[:K, :N] = B
    (a, b), _ = _to_xla(A_p, B_p)
    return _sweep_tile_configs(M_p, K_p, N_p, a, b)


def _nki_gemm_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """NKI GEMM implementation with tile-shape autotuner (#26).

    On hardware: sweeps tile candidates once per shape bucket and caches the
    winner to disk (``/var/tmp/trnblas-autotune/cache.json`` by default).
    Subsequent calls for the same bucket hit the in-process dict and then the
    NEFF cache — no re-sweep.

    Padding is computed *after* tile selection so alignment matches the chosen
    tile sizes.  In simulator mode or when ``TRNBLAS_AUTOTUNE=0`` is set,
    the default (128, 128, 512) config is used without sweeping.

    Set ``TRNBLAS_REQUIRE_NKI=1`` to re-raise kernel errors instead of falling
    back to ``torch.matmul``.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    _, N = B.shape

    # Tile selection: autotuner on hardware, defaults elsewhere.
    if _AUTOTUNE_ENABLED and not _use_simulator():
        _load_autotune_cache()
        bucket = _autotune_bucket(M, K, N)
        if bucket not in _autotune_mem:
            tile_m, tile_k, tile_n = _sweep_on_default_pad(M, K, N, A, B)
            _autotune_mem[bucket] = (tile_m, tile_k, tile_n)
            _save_autotune_cache()
        tile_m, tile_k, tile_n = _autotune_mem[bucket]
    else:
        tile_m, tile_k, tile_n = _TILE_M, _TILE_K, _TILE_N

    # Pad to chosen tile sizes.
    M_pad = _round_up(M, tile_m)
    K_pad = _round_up(K, tile_k)
    # When N ≤ tile_n the kernel uses a single N-tile (no remainder needed).
    N_pad = N if tile_n >= N else _round_up(N, tile_n)
    needs_pad = (M_pad != M) or (K_pad != K) or (N_pad != N)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, K_pad, dtype=A.dtype, device=A.device)
            A_p[:M, :K] = A
            B_p = torch.zeros(K_pad, N_pad, dtype=B.dtype, device=B.device)
            B_p[:K, :N] = B
            A_feed, B_feed = A_p.contiguous(), B_p.contiguous()
        else:
            A_feed, B_feed = A.contiguous(), B.contiguous()

        kernel = _get_gemm_kernel(tile_m, tile_k, tile_n)
        if _use_simulator():
            out_np = nki.simulate(kernel)(A_feed.cpu().numpy(), B_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            (a, b), orig_device = _to_xla(A_feed, B_feed)
            result = kernel(a, b).to(orig_device)
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
            A_feed = A_p.contiguous()
        else:
            A_feed = A.contiguous()

        if _use_simulator():
            out_np = nki.simulate(_syrk_kernel)(A_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            (a,), orig_device = _to_xla(A_feed)
            c = _syrk_kernel(a)
            result = c.to(orig_device)
        return result[:M, :M] if needs_pad else result
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return torch.matmul(A, A.T)


if HAS_NKI:
    # Populate default tile config in the kernel registry.
    # _gemm_kernel is kept as a module-level alias for backward compatibility
    # (simulator path and any external callers that reference it directly).
    _gemm_kernel = _get_gemm_kernel(_TILE_M, _TILE_K, _TILE_N)

    @nki.jit
    def _mp2_energy_kernel(T_flat, eps_occ_chunk, eps_occ_full, eps_vir_col, eps_vir_row):
        """Fused MP2 energy reduction (#15, M2 correctness fix).

        Computes Σ_{i<ic, j<nocc, a,b<nvir} T[i,j,a,b] *
        (2 T[i,j,a,b] - T[i,j,b,a]) / denom[i,j,a,b] where
        T[i,j,a,b] = T_flat[i*nvir + a, j*nvir + b].

        Sub-tiles the nvir partition dim into strips of P_TILE ≤ 128
        (largest divisor of NVIR under the NKI partition limit).
        Strip partials are accumulated in a (P_TILE, 1) SBUF register
        per (i, j).

        **#35 cross-pair batching.** Rather than storing each (i, j)
        pair's partial immediately to HBM (IC×NOCC stores per chunk),
        all NOCC j-partials for a given i are accumulated in an SBUF
        tile `acc_j (P_TILE, NOCC)` and flushed in a single HBM store
        after the j-loop — IC stores per chunk total. Hypothesis: the
        per-pair store was a serialization fence preventing the compiler
        from pipelining consecutive pairs. SBUF cost: P_TILE×NOCC×4
        bytes (≤ 43 KB at the large bench shape; well under budget).

        eps_* args are shape (1, N) so `nl.load` interprets them as
        partition=1, free=N (a 1D load is treated as partition=len,
        which would exceed the 128-partition limit for NVIR > 128).

        **NKI 0.3.0 broadcast fix.** Partition-mismatched
        tensor-tensor arith (`(1,1) ⊕ (P_TILE,1)`, `(P_TILE,1) ⊕
        (1,NVIR)`) is rejected by the MLIR verifier. `denom` is built
        by lifting all three eps operands to `(P_TILE, NVIR)` via
        `nl.broadcast_to` first, so every subsequent subtract sees
        matching partition dims. This replaces the M1 pattern that
        re-skipped the 5 MP2 tests during the 0.3.0 migration.

        NKI only supports reduction along the free dim. A full
        `(P_TILE, NVIR) → scalar` reduce would need a partition-axis
        reduce which NKI rejects. Instead: reduce free-only to get
        per-partition partials `(P_TILE, 1)`, batch NOCC of them in
        SBUF, emit `(P_TILE, IC*NOCC)` to HBM — caller `.sum()`
        handles the final partition-axis reduction host-side.
        """
        NVIR = eps_vir_row.shape[1]
        IC = eps_occ_chunk.shape[1]
        NOCC = eps_occ_full.shape[1]

        P_TILE = min(NVIR, 128)
        while NVIR % P_TILE != 0:
            P_TILE -= 1
        NSTRIP = NVIR // P_TILE

        # Output layout: (P_TILE, IC*NOCC) — 2D flat so each i-row's
        # NOCC partials can be stored in one nl.store call.  Partition
        # axis FIRST so the store aligns with SBUF partition layout.
        # Host caller's .sum() is shape-agnostic.
        e_partial = nl.ndarray((P_TILE, IC * NOCC), dtype=nl.float32, buffer=nl.shared_hbm)
        # Full eps_vir as a free-dim vector (partition=1, free=NVIR)
        # for the per-b axis of denom.
        ev_row = nl.load(eps_vir_row[0:1, 0:NVIR])

        for i in nl.affine_range(IC):
            eo_i = nl.load(eps_occ_chunk[0:1, i : i + 1])
            # Batch all NOCC pair partials for this i into SBUF before
            # a single HBM store.  IC×NOCC stores → IC stores per chunk.
            acc_j = nl.zeros((P_TILE, NOCC), dtype=nl.float32, buffer=nl.sbuf)
            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ_full[0:1, j : j + 1])
                eo_sum = nl.add(eo_i, eo_j)

                # Per-strip SBUF slots. Each affine_range iteration
                # writes its own column; a single nl.sum over the
                # NSTRIP free axis at the end reduces strips.
                # (In-place += across affine_range hits NKI's
                # "Unexpected output dependencies" — the compiler
                # wants the strip index in the dst access explicitly.)
                acc_rows = nl.zeros((P_TILE, NSTRIP), dtype=nl.float32, buffer=nl.sbuf)

                for s in nl.affine_range(NSTRIP):
                    a_off = s * P_TILE
                    t = nl.load(
                        T_flat[
                            i * NVIR + a_off : i * NVIR + a_off + P_TILE, j * NVIR : (j + 1) * NVIR
                        ]
                    )
                    t_T = nl.load_transpose2d(
                        T_flat[
                            i * NVIR : (i + 1) * NVIR, j * NVIR + a_off : j * NVIR + a_off + P_TILE
                        ]
                    )
                    # (P_TILE, 1) partition-axis load of eps_vir's
                    # a-strip — matches the output axis of the (P_TILE,
                    # NVIR) tile we're operating on.
                    ev_col = nl.load(eps_vir_col[a_off : a_off + P_TILE, 0:1])
                    # denom[a, b] = eo_sum - ev_col[a] - ev_row[b]
                    #
                    # NKI 0.3.0 MLIR verifier rejects tensor-tensor
                    # arithmetic when partition dims don't match (e.g.
                    # `(1, 1) - (P_TILE, 1)` or `(P_TILE, 1) - (1, N)`).
                    # Use `nl.broadcast_to` to explicitly lift every
                    # operand to the target `(P_TILE, NVIR)` shape first,
                    # then subtract at matching partition dims.
                    eo_sum_bc = nl.broadcast_to(eo_sum, (P_TILE, NVIR))
                    ev_col_bc = nl.broadcast_to(ev_col, (P_TILE, NVIR))
                    ev_row_bc = nl.broadcast_to(ev_row, (P_TILE, NVIR))
                    denom = nl.subtract(
                        nl.subtract(eo_sum_bc, ev_col_bc),
                        ev_row_bc,
                    )
                    # NKI 0.3.0 drops tensor-tensor nl.divide;
                    # substitute multiply × reciprocal.
                    term = nl.multiply(
                        nl.multiply(
                            t,
                            nl.subtract(nl.multiply(t, 2.0), t_T),
                        ),
                        nl.reciprocal(denom),
                    )
                    # Free-dim reduce: (P_TILE, NVIR) → (P_TILE, 1).
                    strip_partial = nl.sum(term, axis=1, keepdims=True)
                    # Write the strip's partial into its own slot (s
                    # indexes the free axis of acc_rows).
                    acc_rows[0:P_TILE, s : s + 1] = strip_partial

                # Reduce across strips (free dim) → (P_TILE, 1).
                acc_row = nl.sum(acc_rows, axis=1, keepdims=True)

                # Accumulate this pair's partial into SBUF; no HBM
                # traffic until after all NOCC j-iterations complete.
                acc_j[0:P_TILE, j : j + 1] = acc_row

            # One HBM store for all NOCC pairs in this i-row.
            nl.store(
                e_partial[0:P_TILE, i * NOCC : (i + 1) * NOCC],
                value=acc_j,
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

                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    # Stationary: a[m_off:m_off+TILE_M, k_off:k_off+TILE_K]
                    # transposed → (TILE_K, TILE_M), partition=K ≤ 128.
                    a_stat = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    # Moving: a.T[k_off:k_off+TILE_K, n_off:n_off+TILE_N]
                    # = a[n_off:n_off+TILE_N, k_off:k_off+TILE_K].T
                    # load_transpose2d swaps axes → (TILE_K, TILE_N),
                    # partition=K ≤ 128.
                    a_mov = nl.load_transpose2d(a[n_off : n_off + TILE_N, k_off : k_off + TILE_K])
                    nisa.nc_matmul(dst=psum, stationary=a_stat, moving=a_mov, accumulate=True)

                # PSUM → SBUF via tensor_copy (NKI 0.3.0 rejects
                # dma_copy reading from PSUM; nl.copy returns a view).
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(
                    c[m_off : m_off + TILE_M, n_off : n_off + TILE_N],
                    value=c_sbuf,
                )

        return c
