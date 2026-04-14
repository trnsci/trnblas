"""
Density-Fitted MP2 energy using trnblas.

Demonstrates the BLAS operations needed for DF-MP2 quantum chemistry
on large molecules (>3000 basis functions).

DF-MP2 core operations:
    1. Three-center integrals: (ia|P) = Σ_μν C_μi (μν|P) C_νa    → two GEMMs
    2. Metric contraction: B_ia^P = Σ_Q (ia|Q) [J^{-1/2}]_QP     → batched GEMM
    3. Energy: E_MP2 = Σ_{ijab} B_ia^P B_jb^P / Δ_{ijab}         → batched GEMM

All three map directly to trnblas Level 3 operations. The per-i Python
loops have been folded into single `batched_gemm` calls, which on
Trainium dispatch all slices through the cached NKI kernel.

Usage:
    python examples/df_mp2.py --demo                  # Small water-like
    python examples/df_mp2.py --nbasis 100 --nocc 20  # Custom shape
    python examples/df_mp2.py --bench                 # Cold/warm timing
                                                       # across 3 shapes
"""

import argparse
import time

import torch

import trnblas


def _bcast(M: torch.Tensor, batch: int) -> torch.Tensor:
    """Materialise a per-batch copy of a 2D matrix as a (batch, *, *) tensor."""
    return M.unsqueeze(0).expand(batch, *M.shape).contiguous()


def _energy_reduction(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
    *,
    mem_budget_bytes: int = 1_500_000_000,
    use_fused: bool = False,
) -> float:
    """MP2 energy from the metric-contracted three-index tensor B.

    Computes E = Σ_{ijab} T_ijab (2 T_ijab - T_ijba) / Δ_ijab where
    T_ijab = Σ_P B[i,a,P] B[j,b,P], via one GEMM per i-chunk:
        T_chunk = B[i_chunk_flat] @ B_flat.T
    chunked along i so each T_chunk stays under `mem_budget_bytes`.
    The energy reduction (T * (2T - T.T) / denom).sum() materialises 3
    additional tensors of T_chunk's shape, so the actual peak is ~4×.
    Default budget targets ~6 GB peak, fits trn1.2xlarge HBM with room
    for ERI + Python + system. At medium this resolves to one chunk.
    """
    nocc, nvir, naux = B.shape
    bytes_per_i = 4 * nvir * nocc * nvir  # one row-block of T_full (fp32)
    i_block = max(1, min(nocc, int(mem_budget_bytes // bytes_per_i)))

    B_flat = B.reshape(nocc * nvir, naux).contiguous()

    # `use_fused`: route the per-chunk `(T * (2T - T.T) / denom).sum()`
    # through `trnblas.nki.nki_mp2_energy`. Measured on trn1 (warm NEFF
    # cache): ~1.48× speedup on the energy step at both medium and large
    # DF-MP2 shapes — real improvement, but below the RFC's 3× bar, so
    # the default path stays torch until a future #15 milestone hits it.
    e_mp2 = torch.zeros((), dtype=B.dtype, device=B.device)
    if not use_fused:
        eps_o_pair = eps_occ.view(nocc, 1, 1, 1) + eps_occ.view(1, nocc, 1, 1)
        eps_v_pair = eps_vir.view(1, 1, nvir, 1) + eps_vir.view(1, 1, 1, nvir)

    for i_start in range(0, nocc, i_block):
        i_end = min(i_start + i_block, nocc)
        ic = i_end - i_start
        B_chunk = B_flat[i_start * nvir : i_end * nvir]  # (ic·nvir, naux)
        T_flat = trnblas.gemm(1.0, B_chunk, B_flat, transB=True)  # (ic·nvir, nocc·nvir)
        if use_fused:
            e_mp2 = e_mp2 + trnblas.nki.nki_mp2_energy(
                T_flat, eps_occ[i_start:i_end], eps_occ, eps_vir
            )
        else:
            T = T_flat.reshape(ic, nvir, nocc, nvir).permute(0, 2, 1, 3)
            denom = eps_o_pair[i_start:i_end] - eps_v_pair
            e_mp2 = e_mp2 + (T * (2.0 * T - T.transpose(-2, -1)) / denom).sum()
    return float(e_mp2)


def df_mp2_energy(
    C_occ: torch.Tensor,  # (nbasis, nocc) — occupied MO coefficients
    C_vir: torch.Tensor,  # (nbasis, nvir) — virtual MO coefficients
    eri_3c: torch.Tensor,  # (nbasis, nbasis, naux) — 3-center integrals
    J_metric: torch.Tensor,  # (naux, naux) — Coulomb metric J_{PQ}
    eps_occ: torch.Tensor,  # (nocc,) — occupied orbital energies
    eps_vir: torch.Tensor,  # (nvir,) — virtual orbital energies
    timings: dict | None = None,
    use_fused: bool = False,
) -> float:
    """Compute DF-MP2 correlation energy.

    Returns E_MP2 (scalar). Optionally fills `timings` with per-step seconds.
    When `use_fused=True`, the energy-reduction step routes through
    `trnblas.nki.nki_mp2_energy`.
    """
    nbasis, nocc = C_occ.shape
    naux = J_metric.shape[0]

    # Step 1: Cholesky of metric → L; J^{-1/2} via L^{-T}: solve L^T @ X = I
    t0 = time.perf_counter()
    L = torch.linalg.cholesky(J_metric)
    J_inv_half = trnblas.trsm(
        1.0,
        L,
        torch.eye(naux, dtype=J_metric.dtype, device=J_metric.device),
        uplo="lower",
        trans=True,
    )
    t_chol = time.perf_counter() - t0

    # Step 2: Half-transform (μν|P) → (ia|P)
    t0 = time.perf_counter()
    # 2a. (iν|P) = C_occ^T @ (μν|P) — reshape ERI to (nbasis, nbasis*naux),
    #             one GEMM, reshape back.
    eri_flat = eri_3c.reshape(nbasis, -1)
    iv_P = trnblas.gemm(1.0, C_occ, eri_flat, transA=True)  # (nocc, nbasis*naux)
    iv_P = iv_P.reshape(nocc, nbasis, naux)

    # 2b. (ia|P) = C_vir^T @ (iν|P) — one batched GEMM over occupied dim.
    #     C_vir.T broadcast across batch=nocc; iv_P is already (nocc, nbasis, naux).
    C_vir_T_b = _bcast(C_vir.T, nocc)  # (nocc, nvir, nbasis)
    ia_P = trnblas.batched_gemm(1.0, C_vir_T_b, iv_P)  # (nocc, nvir, naux)
    t_half = time.perf_counter() - t0

    # Step 3: Metric contraction B[i] = ia_P[i] @ J_inv_half — one batched GEMM.
    t0 = time.perf_counter()
    J_b = _bcast(J_inv_half, nocc)  # (nocc, naux, naux)
    B = trnblas.batched_gemm(1.0, ia_P, J_b)  # (nocc, nvir, naux)
    t_metric = time.perf_counter() - t0

    # Step 4: Energy via one GEMM (chunked over i if memory-tight).
    #   T(i,j)_{ab} = Σ_P B[i,a,P] B[j,b,P]
    # Reshape B → X of shape (nocc·nvir, naux); then T_full = X @ X.T is
    # one GEMM, and T_full[i·nvir+a, j·nvir+b] = T(i,j)_{ab}. No batching
    # over (i,j) needed — that was the wrong shape for this contraction.
    # For shapes where the full T_full doesn't fit HBM, chunk over i.
    t0 = time.perf_counter()
    e_mp2 = _energy_reduction(B, eps_occ, eps_vir, use_fused=use_fused)
    t_energy = time.perf_counter() - t0

    if timings is not None:
        timings.update(chol=t_chol, half=t_half, metric=t_metric, energy=t_energy)
    return float(e_mp2)


# Approximate flop count for a DF-MP2 evaluation at (nbasis, nocc, naux):
#   2a:                     2 · nbasis^2     · nocc       · naux
#   2b:  nocc batched ×     2 · nvir         · nbasis     · naux
#   3:   nocc batched ×     2 · nvir         · naux       · naux
#   4:   nocc² batched ×    2 · nvir         · nvir       · naux
def _flops(nbasis: int, nocc: int, naux: int) -> int:
    nvir = nbasis - nocc
    f_2a = 2 * nbasis * nbasis * nocc * naux
    f_2b = 2 * nocc * nvir * nbasis * naux
    f_3 = 2 * nocc * nvir * naux * naux
    f_4 = 2 * nocc * nocc * nvir * nvir * naux
    return f_2a + f_2b + f_3 + f_4


def _make_inputs(nbasis: int, nocc: int, naux: int, seed: int = 42, device: str = "cpu"):
    # Seed the CPU RNG so cold/warm + CPU/GPU runs are reproducible
    # bit-for-bit. Build on CPU then move — this keeps the random draw
    # identical across devices (torch.randn(..., device="cuda") uses a
    # separate RNG stream, which would make GPU energies drift from CPU
    # for the same seed).
    torch.manual_seed(seed)
    C_occ = torch.randn(nbasis, nocc) * 0.1
    C_vir = torch.randn(nbasis, nbasis - nocc) * 0.1
    eri_3c = torch.randn(nbasis, nbasis, naux) * 0.01
    J_raw = torch.randn(naux, naux) * 0.01
    J_metric = J_raw @ J_raw.T + naux * torch.eye(naux)
    eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
    eps_vir = torch.sort(torch.rand(nbasis - nocc))[0] + 0.1
    if device != "cpu":
        C_occ = C_occ.to(device)
        C_vir = C_vir.to(device)
        eri_3c = eri_3c.to(device)
        J_metric = J_metric.to(device)
        eps_occ = eps_occ.to(device)
        eps_vir = eps_vir.to(device)
    return C_occ, C_vir, eri_3c, J_metric, eps_occ, eps_vir


_BENCH_SHAPES = {
    "small": (128, 16, 384),  # ~75MB ERI
    "medium": (512, 64, 1536),  # ~1.5GB ERI
    "large": (768, 96, 2304),  # ~5GB ERI
}


def bench(shape_name: str, device: str = "cpu", use_fused: bool = False):
    nbasis, nocc, naux = _BENCH_SHAPES[shape_name]
    nvir = nbasis - nocc
    flops = _flops(nbasis, nocc, naux)
    inputs = _make_inputs(nbasis, nocc, naux, device=device)

    print(f"[shape={shape_name} nbasis={nbasis} nocc={nocc} nvir={nvir} naux={naux}]")
    print(
        f"  approx flops: {flops / 1e9:.1f} G  backend: {trnblas.get_backend()}  "
        f"device: {device}  fused_energy: {use_fused}"
    )

    for label in ("cold", "warm"):
        t = {}
        t0 = time.perf_counter()
        e = df_mp2_energy(*inputs, timings=t, use_fused=use_fused)
        # Ensure async GPU work completes before stopping the timer.
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        total = time.perf_counter() - t0
        tflops = flops / total / 1e12
        print(
            f"  {label}: chol {t['chol']:.3f}s  half {t['half']:.3f}s  "
            f"metric {t['metric']:.3f}s  energy {t['energy']:.3f}s  "
            f"total {total:.3f}s  ~{tflops:.2f} TFLOPS  E={e:.6e}"
        )


def main():
    parser = argparse.ArgumentParser(description="DF-MP2 using trnblas")
    parser.add_argument("--demo", action="store_true", help="Small demo (water-like)")
    parser.add_argument(
        "--bench", action="store_true", help="Cold/warm timing across small/medium/large shapes"
    )
    parser.add_argument(
        "--shape", choices=list(_BENCH_SHAPES), help="Restrict --bench to one shape"
    )
    parser.add_argument("--nbasis", type=int, default=24)
    parser.add_argument("--nocc", type=int, default=5)
    parser.add_argument("--naux", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inputs (cpu, cuda, cuda:0, ...). "
        "CPU by default; set cuda to benchmark against cuBLAS "
        "on an NVIDIA GPU instance.",
    )
    parser.add_argument(
        "--fused-energy",
        action="store_true",
        help="Route the energy-reduction step through trnblas.nki.nki_mp2_energy "
        "(fused NKI kernel, #15 M2).",
    )
    args = parser.parse_args()

    if args.bench:
        shapes = [args.shape] if args.shape else list(_BENCH_SHAPES)
        for s in shapes:
            bench(s, device=args.device, use_fused=args.fused_energy)
        return

    if args.demo:
        args.nbasis, args.nocc = 24, 5

    nbasis, nocc = args.nbasis, args.nocc
    nvir = nbasis - nocc
    naux = args.naux or 3 * nbasis
    inputs = _make_inputs(nbasis, nocc, naux)

    print("DF-MP2 calculation:")
    print(f"  Basis functions: {nbasis}")
    print(f"  Occupied MOs:    {nocc}")
    print(f"  Virtual MOs:     {nvir}")
    print(f"  Auxiliary basis: {naux}")
    print(f"  Backend:         {trnblas.get_backend()}\n")

    timings: dict = {}
    t0 = time.perf_counter()
    e_mp2 = df_mp2_energy(*inputs, timings=timings, use_fused=args.fused_energy)
    total = time.perf_counter() - t0
    for k, v in timings.items():
        print(f"  {k:15s}: {v:.3f}s")
    print(f"\n  E_MP2 = {e_mp2:.10f} (synthetic data)")
    print(f"  Total: {total:.3f}s")


if __name__ == "__main__":
    main()
