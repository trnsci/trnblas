"""Probe NKI API surface + simulator availability.

Run on an AWS Neuron DLAMI instance. Emits:
- neuronxcc version
- full nl.* / nisa.* symbol catalog (helps future kernels know what's available)
- standalone `nki` package info if present
- whether NKI_SIMULATOR=1 is live and what it does to the existing
  scripts/probe_nki.py (subprocess call)

Usage via SSM:
    /opt/aws_neuronx_venv_pytorch_*/bin/python scripts/probe_nki_api.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    print("===== environment =====")
    print(f"python: {sys.version.split()[0]}")
    print(f"NKI_SIMULATOR={os.environ.get('NKI_SIMULATOR', '(unset)')}")

    print("\n===== versions =====")
    try:
        import neuronxcc

        ver = getattr(neuronxcc, "__version__", "unknown")
        print(f"neuronxcc: {ver}")
    except ImportError as exc:
        print(f"neuronxcc: not importable ({exc})")

    try:
        import nki  # standalone package (if installed)

        ver = getattr(nki, "__version__", "unknown")
        print(f"nki (standalone): {ver}")
    except ImportError:
        print("nki (standalone): not importable")

    print("\n===== nl.* symbols =====")
    try:
        import neuronxcc.nki.language as nl

        names = sorted(n for n in dir(nl) if not n.startswith("_"))
        print(f"count: {len(names)}")
        print(json.dumps(names, indent=None))
    except ImportError as exc:
        print(f"cannot import nl: {exc}")

    print("\n===== nisa.* symbols =====")
    try:
        import neuronxcc.nki.isa as nisa

        names = sorted(n for n in dir(nisa) if not n.startswith("_"))
        print(f"count: {len(names)}")
        print(json.dumps(names, indent=None))
    except ImportError as exc:
        print(f"cannot import nisa: {exc}")

    # Primitives our kernels depend on — confirm each still resolves.
    print("\n===== trnblas primitive presence check =====")
    import neuronxcc.nki as _nki_root
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl

    checks = {
        "@nki.jit": hasattr(_nki_root, "jit"),
        "nl.load": hasattr(nl, "load"),
        "nl.load_transpose2d": hasattr(nl, "load_transpose2d"),
        "nl.store": hasattr(nl, "store"),
        "nl.affine_range": hasattr(nl, "affine_range"),
        "nl.ndarray": hasattr(nl, "ndarray"),
        "nl.zeros": hasattr(nl, "zeros"),
        "nl.copy": hasattr(nl, "copy"),
        "nl.sum": hasattr(nl, "sum"),
        "nl.add": hasattr(nl, "add"),
        "nl.subtract": hasattr(nl, "subtract"),
        "nl.multiply": hasattr(nl, "multiply"),
        "nl.divide": hasattr(nl, "divide"),
        "nl.psum": hasattr(nl, "psum"),
        "nl.sbuf": hasattr(nl, "sbuf"),
        "nl.shared_hbm": hasattr(nl, "shared_hbm"),
        "nl.float32": hasattr(nl, "float32"),
        "nisa.nc_matmul": hasattr(nisa, "nc_matmul"),
    }
    for name, present in checks.items():
        mark = "ok" if present else "MISSING"
        print(f"  {name}: {mark}")

    # Call-chain: run the existing probe_nki.py under whatever env vars
    # the caller set. If NKI_SIMULATOR=1 is in the parent env, we
    # inherit it; child sees NKI run in simulator mode.
    probe = Path(__file__).parent / "probe_nki.py"
    print(f"\n===== invoking {probe.name} (inherits env) =====")
    try:
        r = subprocess.run(
            [sys.executable, str(probe)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(f"exit: {r.returncode}")
        print("stdout:")
        print(r.stdout)
        if r.stderr:
            print("stderr (head):")
            print("\n".join(r.stderr.splitlines()[:30]))
    except subprocess.TimeoutExpired:
        print("TIMEOUT (probe hung for 300s)")
        return 1
    except Exception as exc:
        print(f"failed to spawn: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
