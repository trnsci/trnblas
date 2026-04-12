"""
NKI dispatch for BLAS operations.

Backend selection mirrors trnfft: auto/pytorch/nki.
The GEMM kernel is the primary acceleration target — it uses stationary
tile reuse on the Tensor Engine for 2x fewer SBUF loads vs naive.
"""

from __future__ import annotations

import os

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
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.matmul(A, B)


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
