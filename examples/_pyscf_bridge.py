"""PySCF → trnblas input bridge for DF-MP2 end-to-end validation (#11).

Runs an RHF calculation via PySCF, builds the density-fitted
3-center ERI and Coulomb metric, and returns the six tensors that
`df_mp2_energy` consumes. Also exposes `pyscf_reference_energy`
for correctness comparisons.
"""

from __future__ import annotations

import numpy as np
import torch
from pyscf import df, gto, mp, scf


def build_df_mp2_inputs(mol: gto.Mole, auxbasis: str = "weigend") -> dict:
    """Run SCF + build DF integrals for a PySCF molecule.

    Returns a dict with the six tensors `df_mp2_energy` takes, all
    fp32 torch tensors on CPU. Uses RHF + density-fitting with the
    given auxiliary basis. The 3-center ERI comes from
    `df.incore.aux_e2` which returns `(nbasis, nbasis, naux)`
    already — the layout trnblas expects.
    """
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis).run(verbose=0)
    nocc = mol.nelectron // 2

    C_occ = torch.from_numpy(mf.mo_coeff[:, :nocc]).float().contiguous()
    C_vir = torch.from_numpy(mf.mo_coeff[:, nocc:]).float().contiguous()
    eps_occ = torch.from_numpy(mf.mo_energy[:nocc]).float()
    eps_vir = torch.from_numpy(mf.mo_energy[nocc:]).float()

    auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    eri_np = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    eri_3c = torch.from_numpy(np.ascontiguousarray(eri_np)).float()
    J_np = auxmol.intor("int2c2e")
    J_metric = torch.from_numpy(np.ascontiguousarray(J_np)).float()

    return dict(
        C_occ=C_occ,
        C_vir=C_vir,
        eri_3c=eri_3c,
        J_metric=J_metric,
        eps_occ=eps_occ,
        eps_vir=eps_vir,
    )


def pyscf_reference_energy(mol: gto.Mole, auxbasis: str = "weigend") -> float:
    """Trusted reference: PySCF's own DF-MP2 correlation energy."""
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis).run(verbose=0)
    return float(mp.dfmp2.DFMP2(mf).kernel()[0])


_MOLECULES = {
    "h2o": "O 0 0 0; H 0 0 0.96; H 0.926 0 -0.24",
    "ch4": "C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H 0.63 -0.63 -0.63; H -0.63 0.63 -0.63",
    "nh3": "N 0 0 0; H 0 0.94 -0.38; H 0.82 -0.47 -0.38; H -0.82 -0.47 -0.38",
    # Glycine (H2N-CH2-COOH) — neutral form, standard geometry in Angstroms.
    # 10 atoms, 40 electrons, nocc=20.
    # sto-3g: 30 basis functions;  cc-pVDZ: ~95 functions.
    "glycine": (
        "N  0.000  1.493  0.000;"
        "H  0.823  1.930  0.293;"
        "H -0.823  1.930  0.293;"
        "C  0.000  0.000  0.000;"
        "H  1.029 -0.346  0.000;"
        "H -1.029 -0.346  0.000;"
        "C  0.000 -0.541  1.451;"
        "O  0.000 -1.751  1.586;"
        "O  0.000  0.268  2.423;"
        "H  0.000 -0.143  3.297"
    ),
    # Water trimer — approximately cyclic geometry in Angstroms.
    # 30 electrons, nocc=15; nontrivial auxiliary basis (3× water naux).
    "h2o_trimer": (
        "O  0.000  0.000  0.000;"
        "H  0.756  0.586  0.000;"
        "H -0.756  0.586  0.000;"
        "O  2.440  0.000  0.000;"
        "H  3.196  0.586  0.000;"
        "H  1.684  0.586  0.000;"
        "O  1.220  2.114  0.000;"
        "H  1.976  2.700  0.000;"
        "H  0.464  2.700  0.000"
    ),
}


def make_mol(name: str, basis: str) -> gto.Mole:
    """Build one of the preset molecules at the requested basis."""
    if name not in _MOLECULES:
        raise ValueError(f"unknown molecule {name!r}; pick one of {list(_MOLECULES)}")
    return gto.M(atom=_MOLECULES[name], basis=basis, verbose=0)
