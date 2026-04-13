"""Real-molecule DF-MP2 via PySCF → trnblas (#11).

Runs PySCF's SCF + density fitting on a small molecule, feeds the
integrals into trnblas's `df_mp2_energy`, and compares to PySCF's
own DF-MP2 reference energy. The point is *correctness validation*
on real chemistry, not performance — PySCF on CPU will beat the
trnblas reference path here regardless.

Usage:
    python examples/df_mp2_pyscf.py                      # H2O / STO-3G
    python examples/df_mp2_pyscf.py --mol h2o --basis cc-pvdz
    python examples/df_mp2_pyscf.py --mol ch4 --basis cc-pvdz
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pyscf_bridge import build_df_mp2_inputs, make_mol, pyscf_reference_energy
from df_mp2 import df_mp2_energy


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="h2o", choices=["h2o", "ch4", "nh3"])
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--auxbasis", default="weigend")
    args = ap.parse_args()

    mol = make_mol(args.mol, args.basis)
    nbasis = mol.nao
    nocc = mol.nelectron // 2

    print(f"[{args.mol.upper()} / basis={args.basis} / aux={args.auxbasis}]")
    print(f"  nbasis={nbasis}  nocc={nocc}  nvir={nbasis - nocc}")

    t0 = time.perf_counter()
    inputs = build_df_mp2_inputs(mol, auxbasis=args.auxbasis)
    t_build = time.perf_counter() - t0
    print(f"  naux={inputs['J_metric'].shape[0]}  build {t_build:.2f}s")

    timings: dict = {}
    t0 = time.perf_counter()
    e_trnblas = df_mp2_energy(**inputs, timings=timings)
    t_trnblas = time.perf_counter() - t0

    t0 = time.perf_counter()
    e_pyscf = pyscf_reference_energy(mol, auxbasis=args.auxbasis)
    t_pyscf = time.perf_counter() - t0

    diff = abs(e_trnblas - e_pyscf)
    rel = diff / abs(e_pyscf) if e_pyscf != 0 else float("nan")

    print(
        f"  trnblas: chol {timings['chol']:.3f}s  half {timings['half']:.3f}s  "
        f"metric {timings['metric']:.3f}s  energy {timings['energy']:.3f}s  "
        f"total {t_trnblas:.3f}s"
    )
    print(f"  E_trnblas = {e_trnblas:.10f} Ha")
    print(f"  E_pyscf   = {e_pyscf:.10f} Ha   ({t_pyscf:.2f}s)")
    print(f"  |diff|    = {diff:.2e} Ha   (rel {rel:.1e})")


if __name__ == "__main__":
    main()
