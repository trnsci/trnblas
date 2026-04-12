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


def _aligned(M: int, K: int, N: int) -> bool:
    return (M % _TILE_M == 0) and (K % _TILE_K == 0) and (N % _TILE_N == 0)


def _to_xla(*tensors):
    """Move tensors to the XLA device for NKI kernel dispatch."""
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


def _nki_gemm_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """NKI GEMM implementation.

    Aligned shapes (M%128 == K%128 == 0, N%512 == 0) dispatch to the
    `_gemm_kernel` on the Tensor Engine. Other shapes fall back to
    `torch.matmul` until edge-tile handling lands.

    Set `TRNBLAS_REQUIRE_NKI=1` to re-raise on kernel errors instead of
    falling back — used by the validation loop to surface silent breakage.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    _, N = B.shape
    if not _aligned(M, K, N):
        if _REQUIRE_NKI:
            raise RuntimeError(
                f"TRNBLAS_REQUIRE_NKI set but shape {(M, K, N)} is not "
                f"aligned to (TILE_M={_TILE_M}, TILE_K={_TILE_K}, TILE_N={_TILE_N})"
            )
        return torch.matmul(A, B)
    try:
        (a, b), orig_device = _to_xla(A.contiguous(), B.contiguous())
        c = _gemm_kernel(a, b)
        return c.to(orig_device)
    except Exception:
        if _REQUIRE_NKI:
            raise
        return torch.matmul(A, B)


if HAS_NKI:

    @nki.jit
    def _gemm_kernel(a, b):
        """Real GEMM: C = A @ B with stationary tile reuse.

        Aligned shapes only — caller guarantees M%128 == K%128 == 0,
        N%512 == 0. PSUM accumulates over K-tiles before the single store
        per (m, n) tile pair.

        NKI 2.24 calling convention (`nisa.nc_matmul`):
            stationary: (TILE_K, TILE_M)  partition=K ≤ 128, free ≤ 128
            moving:     (TILE_K, TILE_N)  partition=K, free ≤ 512
            psum:       (TILE_M, TILE_N)  fp32, in nl.psum
        """
        M, K = a.shape
        _, N = b.shape

        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // _TILE_M):
            for n in nl.affine_range(N // _TILE_N):
                m_off = m * _TILE_M
                n_off = n * _TILE_N

                psum = nl.zeros((_TILE_M, _TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // _TILE_K):
                    k_off = k * _TILE_K

                    # Load A row-tile transposed so partition dim = K.
                    a_t = nl.load_transpose2d(
                        a[m_off:m_off + _TILE_M, k_off:k_off + _TILE_K]
                    )
                    # B is already K-major.
                    b_tile = nl.load(
                        b[k_off:k_off + _TILE_K, n_off:n_off + _TILE_N]
                    )

                    psum[...] += nisa.nc_matmul(a_t, b_tile)

                c_sbuf = nl.copy(psum, dtype=a.dtype)
                nl.store(
                    c[m_off:m_off + _TILE_M, n_off:n_off + _TILE_N],
                    value=c_sbuf,
                )

        return c
