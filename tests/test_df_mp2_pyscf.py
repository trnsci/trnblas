"""Real-molecule DF-MP2 correctness tests against PySCF (#11, #20).

Requires the `[pyscf]` optional extra. Run with:
    pytest tests/test_df_mp2_pyscf.py -m pyscf -v

Slow (cc-pVTZ, larger molecules) are also marked @pytest.mark.slow and
are skipped in the default run. Run with:
    pytest tests/test_df_mp2_pyscf.py -m "pyscf and slow" -v

The cc-pVTZ and glycine/cc-pVDZ cases are the fp32-precision envelope
tests (#20): they identify where FP32 accumulation error becomes visible
relative to PySCF's FP64 reference. See docs/architecture.md for the
full precision discussion.
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


def _check(mol_name, basis, tol_ha):
    """Run trnblas DF-MP2 and assert it matches PySCF within tol_ha."""
    mol = make_mol(mol_name, basis)
    inputs = build_df_mp2_inputs(mol)
    e_trnblas = df_mp2_energy(**inputs)
    e_pyscf = pyscf_reference_energy(mol)
    diff = abs(e_trnblas - e_pyscf)
    assert diff < tol_ha, (
        f"{mol_name}/{basis}: E_trnblas={e_trnblas:.10f}, "
        f"E_pyscf={e_pyscf:.10f}, diff={diff:.2e} > {tol_ha:.0e}"
    )
    return e_trnblas, e_pyscf, diff


# ---------------------------------------------------------------------------
# Phase 1 baseline — small molecules, moderate bases (fast, default CI)
# ---------------------------------------------------------------------------


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
    """Correctness baseline — small molecules at sto-3g and cc-pVDZ."""
    _check(mol_name, basis, tol_ha)


# ---------------------------------------------------------------------------
# Phase 2 / precision envelope — larger molecules and triple-zeta (#20)
#
# These tests are marked @pytest.mark.slow — they take 30-120 s each on
# CPU because PySCF builds larger integrals (not because trnblas is slow).
# Run:  pytest -m "pyscf and slow" -v
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_name, basis, tol_ha",
    [
        # Glycine (H2N-CH2-COOH): 10 heavy atoms, nocc=20.
        # sto-3g: 30 basis, fast sanity check for the geometry.
        # cc-pvdz: ~95 functions, nocc=20, nvir=75 → ~2.25M pair-energy terms.
        #          FP32 accumulation drift expected; tol loosened to 5e-5 Ha.
        ("glycine", "sto-3g", 1e-5),
        ("glycine", "cc-pvdz", 5e-5),
        # Water trimer ((H2O)3): 3× larger naux than monomer.
        # nocc=15, ~30 basis at sto-3g; tests auxiliary basis scaling.
        ("h2o_trimer", "sto-3g", 1e-5),
        # H2O at cc-pVTZ: triple-zeta, nocc=5, nvir≈55; first triple-zeta case.
        # FP32 accumulation over more contraction steps; tol=1e-4 Ha.
        # This is the gating test for #10 (double-double): if diff < 1e-6, close #10.
        ("h2o", "cc-pvtz", 1e-4),
    ],
)
def test_precision_envelope(mol_name, basis, tol_ha):
    """FP32 precision envelope — larger molecules and triple-zeta (#20).

    Records |E_trnblas - E_pyscf| to characterise where fp32 accumulation
    error exceeds µHartree. Passes as long as diff < tol_ha; the actual
    measured diff in CI output informs whether double-double (#10) is needed.
    """
    e_trnblas, e_pyscf, diff = _check(mol_name, basis, tol_ha)
    # Print for CI log — the actual diff is the data point for #10 / #20.
    print(
        f"\n[{mol_name}/{basis}] "
        f"E_trnblas={e_trnblas:.10f}  E_pyscf={e_pyscf:.10f}  "
        f"|diff|={diff:.2e} Ha  (tol={tol_ha:.0e})"
    )
