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
    python examples/df_mp2_pyscf.py --mol glycine --basis cc-pvtz --energy pair

``--device auto`` (the default) moves the integrals to the Trainium ``neuron``
device once when the native TorchNeuron path is available, so every stage runs
on device-resident tensors. With host tensors, each NKI launch copies its
operands to the device and the result back (2-4x slower end to end on trn2).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pyscf_bridge import _MOLECULES, build_df_mp2_inputs, make_mol, pyscf_reference_energy
from df_mp2 import df_mp2_energy

_ENERGY_FLAGS = {
    "chunk": {},
    "fused": {"use_fused": True},
    "pair": {"use_batched_pair": True},
}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        from trnblas.nki.dispatch import _get_dispatch_path

        return "neuron" if _get_dispatch_path() == "native" else "cpu"
    except Exception:
        return "cpu"


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="h2o", choices=sorted(_MOLECULES))
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--auxbasis", default="weigend")
    ap.add_argument("--device", default="auto", help="auto (neuron when native), cpu, neuron")
    ap.add_argument(
        "--energy",
        default="chunk",
        choices=sorted(_ENERGY_FLAGS),
        help="chunk: GEMM + torch reduction; fused: GEMM + nki_mp2_energy; "
        "pair: nki_batched_pair_energy",
    )
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

    device = _resolve_device(args.device)
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"  device={device}  energy={args.energy}")

    timings: dict = {}
    t0 = time.perf_counter()
    e_trnblas = df_mp2_energy(**inputs, timings=timings, **_ENERGY_FLAGS[args.energy])
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
