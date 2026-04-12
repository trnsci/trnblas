"""
Density-Fitted MP2 energy using trnblas.

Demonstrates the BLAS operations needed for DF-MP2 quantum chemistry,
targeting the Janesko lab (TCU) use case with >3000 basis function molecules.

DF-MP2 core operations:
    1. Three-center integrals: (ia|P) = Σ_μν C_μi (μν|P) C_νa    → two GEMMs
    2. Metric contraction: B_ia^P = Σ_Q (ia|Q) [J^{-1/2}]_QP     → GEMM + Cholesky
    3. Energy: E_MP2 = Σ_{ijab} B_ia^P B_jb^P / Δ_{ijab}         → batched GEMM

All three map directly to trnblas Level 3 operations.

Usage:
    python examples/df_mp2.py --demo          # Small water molecule
    python examples/df_mp2.py --nbasis 100    # Larger system
"""

import argparse
import time
import torch
import trnblas


def df_mp2_energy(
    C_occ: torch.Tensor,     # (nbasis, nocc) — occupied MO coefficients
    C_vir: torch.Tensor,     # (nbasis, nvir) — virtual MO coefficients
    eri_3c: torch.Tensor,    # (nbasis, nbasis, naux) — 3-center integrals
    J_metric: torch.Tensor,  # (naux, naux) — Coulomb metric J_{PQ}
    eps_occ: torch.Tensor,   # (nocc,) — occupied orbital energies
    eps_vir: torch.Tensor,   # (nvir,) — virtual orbital energies
) -> float:
    """Compute DF-MP2 correlation energy.

    Returns E_MP2 (scalar).
    """
    nbasis, nocc = C_occ.shape
    nvir = C_vir.shape[1]
    naux = J_metric.shape[0]

    # Step 1: Cholesky of metric → L such that J = L @ L^T
    t0 = time.perf_counter()
    L = torch.linalg.cholesky(J_metric)
    # J^{-1/2} via L^{-T}: solve L^T @ X = I
    J_inv_half = trnblas.trsm(1.0, L, torch.eye(naux), uplo="lower", trans=True)
    t_chol = time.perf_counter() - t0

    # Step 2: Half-transform (μν|P) → (ia|P)
    # First: (iν|P) = C_occ^T @ (μν|P)  — contract over μ
    t0 = time.perf_counter()
    # Reshape eri_3c to (nbasis, nbasis*naux) for single GEMM
    eri_flat = eri_3c.reshape(nbasis, -1)
    iv_P = trnblas.gemm(1.0, C_occ, eri_flat, transA=True)  # (nocc, nbasis*naux)
    iv_P = iv_P.reshape(nocc, nbasis, naux)

    # Then: (ia|P) = Σ_ν (iν|P) C_νa — contract over ν
    # For each i: (a|P) = C_vir^T @ (ν|P)_i
    ia_P = torch.zeros(nocc, nvir, naux)
    for i in range(nocc):
        ia_P[i] = trnblas.gemm(1.0, C_vir, iv_P[i], transA=True)  # (nvir, naux)
    t_half = time.perf_counter() - t0

    # Step 3: Metric contraction B_ia^P = (ia|Q) @ J^{-1/2}_{QP}
    t0 = time.perf_counter()
    # ia_P is (nocc, nvir, naux), J_inv_half is (naux, naux)
    # B_ia^P = ia_P @ J_inv_half
    B = torch.zeros_like(ia_P)
    for i in range(nocc):
        B[i] = trnblas.gemm(1.0, ia_P[i], J_inv_half)  # (nvir, naux)
    t_metric = time.perf_counter() - t0

    # Step 4: MP2 energy
    # E_MP2 = Σ_{ijab} (B_ia^P B_jb^P) (2 * B_ia^P B_jb^P - B_ib^P B_ja^P) / Δ_{ijab}
    t0 = time.perf_counter()
    e_mp2 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            # T_ab = Σ_P B_ia^P B_jb^P = B[i] @ B[j]^T  → (nvir, nvir)
            T = trnblas.gemm(1.0, B[i], B[j], transB=True)

            # Energy denominators
            denom = (eps_occ[i] + eps_occ[j]).unsqueeze(-1).unsqueeze(-1) \
                  - eps_vir.unsqueeze(0).unsqueeze(-1) \
                  - eps_vir.unsqueeze(0).unsqueeze(0)
            # Avoid broadcasting issues
            denom = eps_occ[i] + eps_occ[j] - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)

            e_mp2 += (T * (2 * T - T.T) / denom).sum().item()
    t_energy = time.perf_counter() - t0

    print(f"  Cholesky:        {t_chol:.3f}s")
    print(f"  Half-transform:  {t_half:.3f}s")
    print(f"  Metric contract: {t_metric:.3f}s")
    print(f"  Energy sum:      {t_energy:.3f}s")

    return e_mp2


def main():
    parser = argparse.ArgumentParser(description="DF-MP2 using trnblas")
    parser.add_argument("--demo", action="store_true", help="Small demo (water-like)")
    parser.add_argument("--nbasis", type=int, default=24, help="Basis set size")
    parser.add_argument("--nocc", type=int, default=5, help="Occupied orbitals")
    parser.add_argument("--naux", type=int, default=None, help="Auxiliary basis size (default: 3x nbasis)")
    args = parser.parse_args()

    if args.demo:
        args.nbasis = 24
        args.nocc = 5

    nbasis = args.nbasis
    nocc = args.nocc
    nvir = nbasis - nocc
    naux = args.naux or 3 * nbasis

    print(f"DF-MP2 calculation:")
    print(f"  Basis functions: {nbasis}")
    print(f"  Occupied MOs:    {nocc}")
    print(f"  Virtual MOs:     {nvir}")
    print(f"  Auxiliary basis:  {naux}")
    print(f"  Backend:         {trnblas.get_backend()}")
    print()

    # Generate synthetic data (in real use, these come from PySCF)
    torch.manual_seed(42)
    C_occ = torch.randn(nbasis, nocc) * 0.1
    C_vir = torch.randn(nbasis, nvir) * 0.1
    eri_3c = torch.randn(nbasis, nbasis, naux) * 0.01

    # Make metric SPD
    J_raw = torch.randn(naux, naux) * 0.01
    J_metric = J_raw @ J_raw.T + naux * torch.eye(naux)

    # Orbital energies (occupied < 0, virtual > 0)
    eps_occ = -torch.sort(torch.rand(nocc))[0] - 0.5
    eps_vir = torch.sort(torch.rand(nvir))[0] + 0.1

    t_start = time.perf_counter()
    e_mp2 = df_mp2_energy(C_occ, C_vir, eri_3c, J_metric, eps_occ, eps_vir)
    t_total = time.perf_counter() - t_start

    print(f"\n  E_MP2 = {e_mp2:.10f} (synthetic data)")
    print(f"  Total: {t_total:.3f}s")


if __name__ == "__main__":
    main()
