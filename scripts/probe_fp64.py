#!/usr/bin/env python
"""Device probes of `trnblas.fp64` for a Trainium window (trnsci/trnsci#169).

Run on the native TorchNeuron path (Trn2, one logical core). Each command
writes JSON and prints a markdown table; each takes well under 20 minutes.

    python scripts/probe_fp64.py e0 --json e0.json           # run FIRST: gates the rest
    python scripts/probe_fp64.py e2 --n 1000000 --json e2_1e6.json
    python scripts/probe_fp64.py e2 --n 100000  --json e2_1e5.json
    python scripts/probe_fp64.py e2 --n 10000000 --no-reference --json e2_1e7.json
    python scripts/probe_fp64.py e3 --json e3.json            # gemm_fp64 combine on device
    python scripts/probe_fp64.py lnc --json lnc2.json         # at LNC=2
    NEURON_LOGICAL_NC_CONFIG=1 python scripts/probe_fp64.py lnc --json lnc1.json
    python scripts/probe_fp64.py compare lnc2.json lnc1.json  # digests must agree
    python scripts/probe_fp64.py cols --json cols.json        # dot_cols / nrm2_cols (#142)
    python scripts/probe_fp64.py cols --n 100000 --R 33 --free 256 512 1024 2048

`--device cpu` runs the CPU reference as the "device" (dry run, no hardware).
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("e0", help="rounding / FMA / TensorE probe")
    p2 = sub.add_parser("e2", help="arms: accuracy and ns per element")
    p2.add_argument("--n", type=int, default=1000000)
    p2.add_argument("--reps", type=int, default=20)
    p2.add_argument("--budget-s", type=float, default=1080.0)
    p2.add_argument("--no-reference", action="store_true", help="skip the CPU bitwise check")
    p3 = sub.add_parser("e3", help="gemm_fp64 combine: fp64 bits on the device vs host")
    p3.add_argument("--reps", type=int, default=10)
    p5 = sub.add_parser("cols", help="column-batched dot / nrm2 vs per-column and pairwise")
    p5.add_argument("--n", type=int, nargs="*", default=None, help="lengths (with --R)")
    p5.add_argument("--R", type=int, nargs="*", default=None, help="column counts")
    p5.add_argument("--free", type=int, nargs="*", default=None, help="group widths to sweep")
    p5.add_argument("--reps", type=int, default=10)
    p5.add_argument("--iter-ms", type=float, default=None, help="block-CG ms per iteration")
    pl = sub.add_parser("lnc", help="partials digests (run at LNC=2 and LNC=1)")
    pc = sub.add_parser("compare", help="compare two lnc JSON files")
    pc.add_argument("a")
    pc.add_argument("b")
    for p in (p0, p2, p3, p5, pl):
        p.add_argument("--json", default=None)
        p.add_argument("--device", default="neuron", choices=("neuron", "cpu"))
    args = ap.parse_args(argv)

    from trnblas.nki import fp64_probe as Q

    if args.cmd == "compare":
        docs = []
        for f in (args.a, args.b):
            with open(f) as fh:
                docs.append(json.load(fh))
        a, b = docs
        diff = [k for k in a["digests"] if a["digests"][k] != b["digests"].get(k)]
        print(
            f"LNC {a['lnc']} vs {b['lnc']}: {len(a['digests']) - len(diff)} equal, {len(diff)} differ"
        )
        for k in diff:
            print("  differs:", k)
        bad_ref = [k for d in (a, b) for k, v in d["equal_ref"].items() if not v]
        print("not equal to the CPU reference:", sorted(set(bad_ref)) or "none")
        return 0 if not diff else 1
    if args.cmd == "e0":
        run = Q.device_run if args.device == "neuron" else Q.ideal_run
        rep = Q.run_e0(run)
        print(Q.format_table(rep))
        print(f"\nvector RN + EFT gate: {'PASS' if rep['gate_vector_rn_and_eft'] else 'FAIL'}")
        print(json.dumps(rep["detail"]["pe"]["verdict"]))
        ok = rep["gate_vector_rn_and_eft"]
    elif args.cmd == "e2":
        rep = Q.e2(
            args.n,
            device=args.device,
            reps=args.reps,
            budget_s=args.budget_s,
            check_reference=not args.no_reference,
        )
        print(Q.format_e2(rep))
        ok = True
    elif args.cmd == "e3":
        rep = Q.e3(device=args.device, reps=args.reps)
        print(Q.format_e3(rep))
        ok = all(r["bitwise_equal_ozaki_combine"] for r in rep["shapes"].values())
    elif args.cmd == "cols":
        shapes = None
        if args.n or args.R:
            shapes = [(n, R) for n in (args.n or [100000]) for R in (args.R or [33])]
        kw = {"frees": tuple(args.free)} if args.free else {}
        if args.iter_ms:
            kw["iter_ms"] = args.iter_ms
        rep = Q.cols_bench(shapes, device=args.device, reps=args.reps, **kw)
        print(Q.format_cols(rep))
        ok = all(v for r in rep["shapes"].values() for k, v in r.items() if k.startswith("bitwise"))
    else:
        rep = Q.lnc_digests(device=args.device)
        bad = [k for k, v in rep["equal_ref"].items() if not v]
        print(
            f"LNC={rep['lnc']}: {len(rep['digests'])} digests; not equal to CPU reference: {bad or 'none'}"
        )
        ok = not bad
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
