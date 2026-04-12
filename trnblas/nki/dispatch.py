"""
NKI dispatch for BLAS operations.

Backend selection mirrors trnfft: auto/pytorch/nki.
The GEMM kernel is the primary acceleration target — it uses stationary
tile reuse on the Tensor Engine for 2x fewer SBUF loads vs naive.
"""

from __future__ import annotations

import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

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


def _nki_gemm_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """NKI GEMM implementation.

    TODO: Wire to actual NKI kernel with tiling once validated on trn1/trn2.
    The kernel should:
    - Tile A into 128×128 blocks (SBUF capacity)
    - For each A tile, stream all corresponding B tiles through the systolic array
    - Accumulate partial products in PSUM
    - Handle edge tiles for non-128-aligned dimensions

    See neuron-complex-ops kernels_optimized.py for the complex variant
    of this pattern — real GEMM is simpler (no real/imag split needed).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    # Fallback until kernel is validated on hardware
    return torch.matmul(A, B)


if HAS_NKI:

    @nki.jit
    def gemm_kernel(A_ref, B_ref, C_ref, M: int, N: int, K: int):
        """Tiled GEMM kernel for Trainium NeuronCore.

        C[M,N] = A[M,K] @ B[K,N]

        Tiling: 128×128 tiles for SBUF, accumulate in PSUM.
        A tiles are stationary (loaded once, reused across N tiles).
        B tiles are streamed (moving operand in systolic array).

        STUB: Scaffolded for on-hardware validation.
        """
        TILE = 128

        for m_tile in nl.affine_range(M // TILE):
            for k_tile in nl.affine_range(K // TILE):
                # Load A tile — stationary in systolic array
                a_tile = nl.load(
                    A_ref[m_tile * TILE:(m_tile + 1) * TILE,
                          k_tile * TILE:(k_tile + 1) * TILE]
                )

                for n_tile in nl.affine_range(N // TILE):
                    # Stream B tile — moving operand
                    b_tile = nl.load(
                        B_ref[k_tile * TILE:(k_tile + 1) * TILE,
                              n_tile * TILE:(n_tile + 1) * TILE]
                    )

                    # Matmul → accumulate in PSUM
                    c_partial = nisa.nc_matmul(a_tile, b_tile)

                    # Store (would accumulate across k_tiles in practice)
                    nl.store(
                        C_ref[m_tile * TILE:(m_tile + 1) * TILE,
                              n_tile * TILE:(n_tile + 1) * TILE],
                        c_partial
                    )
