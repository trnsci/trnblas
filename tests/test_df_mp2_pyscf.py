"""Real-molecule DF-MP2 correctness tests against PySCF (#11).

Requires the `[pyscf]` optional extra. Run with:
    pytest tests/test_df_mp2_pyscf.py -m pyscf -v
"""

import os
import sys

import pytest

pyscf = pytest.importorskip("pyscf")

# examples/ isn't a package — add it to sys.path so we can reuse the bridge
# and the DF-MP2 driver without copy-pasting.
_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, os.path.abspath(_EXAMPLES))

from _pyscf_bridge import build_df_mp2_inputs, make_mol, pyscf_reference_energy  # noqa: E402
from df_mp2 import df_mp2_energy  # noqa: E402

pytestmark = pytest.mark.pyscf


@pytest.mark.parametrize(
    "mol_name, basis, tol_ha",
    [
        ("h2o", "sto-3g", 1e-6),
        ("h2o", "cc-pvdz", 1e-5),
        ("ch4", "cc-pvdz", 1e-5),
        ("nh3", "cc-pvdz", 1e-5),
    ],
)
def test_matches_pyscf_reference(mol_name, basis, tol_ha):
    mol = make_mol(mol_name, basis)
    inputs = build_df_mp2_inputs(mol)
    e_trnblas = df_mp2_energy(**inputs)
    e_pyscf = pyscf_reference_energy(mol)
    diff = abs(e_trnblas - e_pyscf)
    assert diff < tol_ha, (
        f"{mol_name}/{basis}: E_trnblas={e_trnblas:.10f}, "
        f"E_pyscf={e_pyscf:.10f}, diff={diff:.2e} > {tol_ha:.0e}"
    )
