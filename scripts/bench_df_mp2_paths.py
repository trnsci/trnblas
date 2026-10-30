#!/usr/bin/env python
"""Real-molecule DF-MP2: trnblas NKI vs aten-on-device vs CPU, per stage.

Two steps, so the PySCF SCF never runs inside a device process:

    python scripts/bench_df_mp2_paths.py prep --mol h2o --basis cc-pvdz --cache DIR
    python scripts/bench_df_mp2_paths.py run  --mol h2o --basis cc-pvdz --cache DIR \\
        --path nki --energy pair

``prep``  Runs PySCF RHF + density fitting and its own DF-MP2 reference, and
          saves the six `df_mp2_energy` inputs plus the reference energy.
``run``   Times `examples/df_mp2.df_mp2_energy` on one path:

          nki       trnblas NKI kernels, inputs resident on the `neuron` device
          nki-host  trnblas NKI kernels, inputs left on the host (what
                    `examples/df_mp2_pyscf.py` does): every launch round-trips
          aten      the same algorithm, `set_backend("pytorch")`, inputs on the
                    `neuron` device, so every GEMM is aten on the device
          cpu       the same algorithm, `set_backend("pytorch")`, CPU tensors

          ``--energy`` picks the energy step: ``chunk`` (GEMM + torch reduction),
          ``fused`` (GEMM + `nki_mp2_energy`), ``pair`` (`nki_batched_pair_energy`).
          ``--metric`` sets ``TRNBLAS_METRIC_SOLVE`` (``auto``: host LAPACK for
          the metric factor on the native path up to the crossover;
          ``device``: Cholesky + TRSM on the device, the earlier behaviour).

Each repetition prints one line
``RESULT {json}`` with the per-stage seconds (device-synchronised), total, and
|E - E_PySCF|. Rep 0 includes compilation; later reps are warm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
sys.path.insert(0, os.path.abspath(_EXAMPLES))

_KEYS = ("C_occ", "C_vir", "eri_3c", "J_metric", "eps_occ", "eps_vir")


def _cache_file(cache: str, mol: str, basis: str, auxbasis: str) -> str:
    aux = "" if auxbasis == "weigend" else f"_{auxbasis}"
    return os.path.join(cache, f"{mol}_{basis}{aux}.pt")


def _prep_synthetic(name: str):
    """``synth-<nbasis>-<nocc>-<naux>``: `df_mp2._make_inputs` data, FP64 CPU reference."""
    from df_mp2 import _make_inputs, df_mp2_energy

    import trnblas

    nbasis, nocc, naux = (int(x) for x in name.split("-")[1:])
    inputs = dict(zip(_KEYS, _make_inputs(nbasis, nocc, naux), strict=True))
    trnblas.set_backend("pytorch")
    e_ref = df_mp2_energy(**{k: v.double() for k, v in inputs.items()})
    return inputs, e_ref


def prep(args) -> None:
    t0 = time.perf_counter()
    if args.mol.startswith("synth-"):
        inputs, e_ref = _prep_synthetic(args.mol)
    else:
        from _pyscf_bridge import build_df_mp2_inputs, make_mol, pyscf_reference_energy

        mol = make_mol(args.mol, args.basis)
        inputs = build_df_mp2_inputs(mol, auxbasis=args.auxbasis)
        e_ref = pyscf_reference_energy(mol, auxbasis=args.auxbasis)
    os.makedirs(args.cache, exist_ok=True)
    torch.save(
        {"inputs": inputs, "e_ref": e_ref},
        _cache_file(args.cache, args.mol, args.basis, args.auxbasis),
    )
    nbasis, nocc = inputs["C_occ"].shape
    print(
        f"PREP {args.mol}/{args.basis} nbasis={nbasis} nocc={nocc} "
        f"naux={inputs['J_metric'].shape[0]} e_ref={e_ref:.10f} ({time.perf_counter() - t0:.1f}s)"
    )


def run(args) -> None:
    from df_mp2 import _sync, df_mp2_energy

    import trnblas

    os.environ["TRNBLAS_METRIC_SOLVE"] = args.metric
    blob = torch.load(_cache_file(args.cache, args.mol, args.basis, args.auxbasis))
    inputs, e_ref = blob["inputs"], blob["e_ref"]

    if args.path in ("nki", "nki-host"):
        trnblas.set_backend("nki")
    else:
        trnblas.set_backend("pytorch")

    h2d = 0.0
    if args.path in ("nki", "aten"):
        t0 = time.perf_counter()
        inputs = {k: v.to("neuron") for k, v in inputs.items()}
        _sync(inputs["C_occ"].device)
        h2d = time.perf_counter() - t0

    flags = {
        "chunk": {},
        "fused": {"use_fused": True},
        "pair": {"use_batched_pair": True},
    }[args.energy]

    nbasis, nocc = inputs["C_occ"].shape
    naux = inputs["J_metric"].shape[0]
    for rep in range(args.reps):
        t = {}
        t0 = time.perf_counter()
        e = df_mp2_energy(**inputs, timings=t, **flags)
        total = time.perf_counter() - t0
        rec = {
            "mol": args.mol,
            "basis": args.basis,
            "auxbasis": args.auxbasis,
            "nbasis": nbasis,
            "nocc": nocc,
            "naux": naux,
            "path": args.path,
            "energy_mode": args.energy,
            "metric_solve": args.metric,
            "rep": rep,
            "h2d": round(h2d, 4),
            **{k: round(v, 4) for k, v in t.items()},
            "total": round(total, 4),
            "e": e,
            "abs_err": abs(e - e_ref),
        }
        print("RESULT " + json.dumps(rec), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "run"])
    ap.add_argument("--mol", default="h2o")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--auxbasis", default="weigend")
    ap.add_argument("--cache", default="/var/tmp/trnblas-dfmp2-inputs")
    ap.add_argument("--path", choices=["nki", "nki-host", "aten", "cpu"], default="nki")
    ap.add_argument("--energy", choices=["chunk", "fused", "pair"], default="chunk")
    ap.add_argument("--metric", choices=["auto", "host", "device"], default="auto")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "prep":
        prep(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
