"""Device probes of `trnblas.fp64` on Trainium (trnsci/trnsci#169).

Native TorchNeuron path only. Every test writes its JSON report into
TRNBLAS_FP64_PROBE_DIR (default: the pytest tmp dir) as e0.json,
e2_<n>.json, e3.json, lnc<L>.json. Run E0 first: the other results assume what it
checks (fp32 add / subtract / multiply round to nearest even on VectorE,
TwoSum and Dekker TwoProd exact). Each test runs in well under 20 minutes
on one logical core (the E2 runs stop starting new kernels after
TRNBLAS_FP64_BUDGET_S, default 1080 s, and report `truncated`).

    pytest -m neuron tests/test_fp64_neuron.py -k e0 -s        # first
    pytest -m neuron tests/test_fp64_neuron.py -s
    NEURON_LOGICAL_NC_CONFIG=1 TRNBLAS_FP64_PROBE_DIR=... pytest -m neuron \\
        tests/test_fp64_neuron.py -k lnc -s                     # whole chip
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from trnblas import fp64 as F
from trnblas.nki import HAS_NKI
from trnblas.nki import dispatch as D
from trnblas.nki import fp64_kernels as K
from trnblas.nki import fp64_probe as Q


def _native() -> bool:
    if not HAS_NKI or D._use_simulator():
        return False
    try:
        return D._get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]


@pytest.fixture
def outdir(tmp_path) -> Path:
    d = Path(os.environ.get("TRNBLAS_FP64_PROBE_DIR", tmp_path))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump(outdir: Path, name: str, rep: dict) -> None:
    (outdir / name).write_text(json.dumps(rep, indent=1, default=str))


def test_e0_rounding_fma_and_tensor_engine(outdir):
    """E0: per engine and identity, IEEE round-to-nearest-even of fp32 add,
    subtract, multiply (ties, near overflow, subnormals, cancellation, wide
    random), TwoSum / Dekker TwoProd exactness, FMA detection on every
    two-op form, and TensorE + PSUM product / accumulation rounding. Fails
    if VectorE add / subtract / multiply or the EFTs are not exact outside
    the subnormal family: then nothing else in trnblas.fp64 holds."""
    rep = Q.run_e0(Q.device_run)
    _dump(outdir, "e0.json", rep)
    print("\n" + Q.format_table(rep))
    print(json.dumps(rep["detail"]["pe"]["verdict"]))
    assert rep["seconds"] < 1200
    assert rep["gate_vector_rn_and_eft"], "VectorE rounding or TwoSum/TwoProd is not exact"


def test_kernels_equal_cpu_reference_bitwise():
    """Every reduction kernel at n = 100003 (one tile per chunk) and the
    default arm at n = 600001 (three tiles per chunk) equals the CPU
    reference bit for bit."""
    rng = np.random.default_rng(0)
    for n, keys in ((100003, sorted(K.REDUCE_KERNELS)), (600001, [("dot32", "accurate")])):
        x64 = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 6, n)
        y64 = rng.standard_normal(n)
        fx, fy = F.from_fp64(torch.from_numpy(x64)), F.from_fp64(torch.from_numpy(y64))
        x32 = torch.from_numpy(x64.astype(np.float32))
        y32 = torch.from_numpy(y64.astype(np.float32))
        ops = {
            "dot32": [x32, y32],
            "sum32": [x32],
            "nrm32": [x32],
            "dotff": [fx.hi, fx.lo, fy.hi, fy.lo],
            "sumff": [fx.hi, fx.lo],
            "nrmff": [fx.hi, fx.lo],
        }
        for kind, arm in keys:
            xs = ops[kind]
            got = K.run_partials(kind, [t.to("neuron") for t in xs], arm)
            if arm == "exact":
                ref = F.reference_exact_partials(kind, xs)
            else:
                ref = F.reference_partials(kind, xs, arm)
            assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), (n, kind, arm)


def test_public_api_on_device_tensors():
    x = torch.from_numpy(np.random.default_rng(1).standard_normal(250000).astype(np.float32))
    y = x.flip(0).contiguous()
    xd, yd = x.to("neuron"), y.to("neuron")
    assert torch.equal(F.dot(xd, yd), F.dot(x, y))  # auto: kernel == reference
    assert torch.equal(F.nrm2(xd), F.nrm2(x))
    assert torch.equal(F.sum(xd), F.sum(x))
    f = F.sum(xd, finish="ff")
    assert torch.equal(f.hi, F.sum(x, finish="ff").hi)


@pytest.mark.parametrize("n", [100000, 1000000, 10000000], ids=["1e5", "1e6", "1e7"])
def test_e2_arms_accuracy_and_throughput(n, outdir):
    """E2: each arm's correct bits and ns per element (end to end and kernel
    only, with the first-call compile time), against the host fp64 round
    trip it replaces. Accuracy gates: the double-word arms reach >= 45 bits
    on these well-conditioned CG-like vectors; every kernel equals the CPU
    reference bit for bit; the exact arm is exactly rounded."""
    budget = float(os.environ.get("TRNBLAS_FP64_BUDGET_S", "1080"))
    rep = Q.e2(n, budget_s=budget, check_reference=n <= 1000000)
    _dump(outdir, f"e2_{n}.json", rep)
    print("\n" + Q.format_e2(rep))
    assert rep["seconds"] < 1200
    for key, row in rep["kernels"].items():
        if row.get("bitwise_vs_cpu_reference") is not None:
            assert row["bitwise_vs_cpu_reference"], key
        if key.endswith(("sloppy", "accurate")):
            assert row["bits"] >= 45, (key, row["bits"])
        if key.endswith("exact") and row["window_held"]:
            assert row["bits"] == 53.0, key  # exactly rounded


def test_lnc_digests(outdir):
    """Partials digests at this LNC (run at LNC=2 and at LNC=1 with the
    same TRNBLAS_FP64_PROBE_DIR); with both files present they must agree,
    and both must equal the CPU reference."""
    rep = Q.lnc_digests()
    lnc = rep["lnc"]
    _dump(outdir, f"lnc{lnc}.json", rep)
    assert all(rep["equal_ref"].values()), [k for k, v in rep["equal_ref"].items() if not v]
    other = outdir / f"lnc{'1' if lnc == '2' else '2'}.json"
    if other.exists():
        o = json.loads(other.read_text())
        assert o["digests"] == rep["digests"]


def test_e3_combine_on_device(outdir):
    """E3: the gemm_fp64 combine with the fp64 bits built on the device is
    bitwise equal to `ozaki.combine`, with its time against today's
    download + host combine, per launch-block shape."""
    rep = Q.e3()
    _dump(outdir, "e3.json", rep)
    print("\n" + Q.format_e3(rep))
    assert rep["seconds"] < 1200
    for k, r in rep["shapes"].items():
        assert r["bitwise_equal_ozaki_combine"], k
