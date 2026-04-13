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
    except Exception:
        if _REQUIRE_NKI:
            raise
        return _torch_mp2_energy(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)


def _nki_mp2_energy_impl(
    T_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    (t, eo_c, eo_f, ev), orig_device = _to_xla(
        T_flat.contiguous(),
        eps_occ_chunk.contiguous(),
        eps_occ_full.contiguous(),
        eps_vir.contiguous(),
    )
    partial = _mp2_energy_kernel(t, eo_c, eo_f, ev)
    # Kernel returns (ic, nocc, nstrip); reduce host-side.
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

    @nki.jit
    def _mp2_energy_kernel(T_flat, eps_occ_chunk, eps_occ_full, eps_vir):
        """Fused MP2 energy reduction (#15).

        Computes Σ_{i<ic, j<nocc, a,b<nvir} T[i,j,a,b] *
        (2 T[i,j,a,b] - T[i,j,b,a]) / denom[i,j,a,b] where
        T[i,j,a,b] = T_flat[i*nvir + a, j*nvir + b].

        Sub-tiles the nvir partition dim into strips of P_TILE ≤ 128
        (largest divisor of NVIR under the NKI partition limit).
        Strip partials are accumulated in a 1×1 SBUF register per
        (i, j), so there is exactly one HBM store per (i, j) —
        cuts store traffic by NSTRIP× vs the per-strip-store variant.

        Returns (IC, NOCC) fp32; host reduces to scalar.
        """
        NVIR = eps_vir.shape[0]
        IC = eps_occ_chunk.shape[0]
        NOCC = eps_occ_full.shape[0]

        P_TILE = min(NVIR, 128)
        while NVIR % P_TILE != 0:
            P_TILE -= 1
        NSTRIP = NVIR // P_TILE

        e_partial = nl.ndarray((IC, NOCC), dtype=nl.float32, buffer=nl.shared_hbm)
        ev_free = nl.load(eps_vir[0:NVIR])

        for i in nl.affine_range(IC):
            eo_i = nl.load(eps_occ_chunk[i:i + 1])
            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ_full[j:j + 1])
                eo_sum = nl.add(eo_i, eo_j)

                acc = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)

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
                    ev_part = nl.load(eps_vir[a_off : a_off + P_TILE])
                    denom_col = nl.subtract(eo_sum, ev_part)
                    denom = nl.subtract(
                        denom_col.reshape((P_TILE, 1)),
                        ev_free.reshape((1, NVIR)),
                    )
                    term = nl.divide(
                        nl.multiply(
                            t,
                            nl.subtract(nl.multiply(t, 2.0), t_T),
                        ),
                        denom,
                    )
                    acc[...] = nl.add(acc, nl.sum(term, axis=(0, 1)))

                nl.store(e_partial[i:i + 1, j:j + 1], value=acc)

        return e_partial
