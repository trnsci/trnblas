"""Device probe: Ozaki II (trnblas#132) against Ozaki I gemm_fp64 and host fp64.

Run on a Trn2 instance (native TorchNeuron path), one logical core, LNC=2:

    NEURON_RT_VISIBLE_CORES=0 python scripts/probe_ozaki2.py micro --out o2_micro.json
    NEURON_RT_VISIBLE_CORES=0 python scripts/probe_ozaki2.py shapes --arms ozaki1 b7 \
        --out o2_b7.json
    NEURON_RT_VISIBLE_CORES=0 python scripts/probe_ozaki2.py shapes --arms b8 b9 \
        --out o2_b8b9.json

`micro` (Q1, ~6 min): at M = N = 2048, K = 16384 (several PSUM groups for
every family), the residue-GEMM kernel time per modulus for b7 / b8 / b9
against Ozaki I's kernel time per slice GEMM (k = 8) and aten bf16 `mm`;
then at K = 4096 and the 53-bit counts, residue GEMM, CRT (ns per output
element) and the fused kernel (both in one launch).

`shapes` (Q2, Q3): at each shape and matched accuracy (configs below, from
the CPU analysis), first-call time (compile), kernel ms (resident operands,
residue GEMM and CRT separately, and the fused kernel), e2e ms of gemm_fp64, the host-CRT variant,
host fp64 at 2 and 12 threads, max_cw on 256 sampled rows against an
independent near-exact reference, and bitwise equality of those rows with
the host reference (`ozaki2.reference_gemm` / `ozaki.reference_gemm`).

Every run writes its JSON after each case and stops scheduling new cases
once `--budget-s` (default 1080 s) has passed, so one run stays under 20 min.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _ozaki_ref import max_cw  # noqa: E402

import trnblas  # noqa: E402
from trnblas import ozaki, ozaki2  # noqa: E402
from trnblas.nki import dispatch as D  # noqa: E402
from trnblas.nki import ozaki2_kernels as K2  # noqa: E402

SHAPES = {
    "1024^3": (1024, 1024, 1024),
    "2048^3": (2048, 2048, 2048),
    "4096^3": (4096, 4096, 4096),
    "3267x8400x241": (3267, 8400, 241),
    "8192x3508x241": (8192, 3508, 241),
}
# matched accuracy on normal data (CPU analysis, max_cw bits):
# ~40 bits (1e-12), ~47 bits (1e-14), ~53 bits (host fp64 level)
CONFIGS = {
    "ozaki1": [6, 7, 8],
    "b7": [13, 15, 17],
    "b8": [11, 13, 15],
    "b9": [10, 11, 13],
}


def sync():
    D._device_synchronize()


def timed(fn, reps=3):
    fn()
    sync()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def inputs(M, K, N, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((M, K))), torch.from_numpy(
        rng.standard_normal((K, N))
    )


def oz1_kernel_fn(A, B, k):
    """Ozaki I kernel launches on resident device slices (as `_ozaki_launcher`)."""
    da, db = ozaki.split_a(A, k), ozaki.split_b(B, k)
    M, N = A.shape[0], B.shape[1]
    if N < ozaki.SWAP_N and M > N:
        da, db, M, N = db, da, N, M
    mbs, kbs, nbs = ozaki.plan(da.xp, da.kp, N, k)
    cache: dict = {}
    blocks = [
        (D._oz_device_block(da, cache, "a", kk, m), D._oz_device_block(db, cache, "b", kk, n))
        for m in mbs
        for kk in kbs
        for n in nbs
    ]

    def run():
        for at, bt in blocks:
            D._call_kernel(D._ozaki_gemm_kernel, "ozaki_gemm", at, bt)

    return run


def oz2_kernel_fns(A, B, n, fam):
    """(residue GEMM, CRT) launches on resident device residues."""
    ra, rb = ozaki2.split_ab(A, B, n, fam)
    M, N = A.shape[0], B.shape[1]
    if N < ozaki.SWAP_N and M > N:
        ra, rb, M, N = rb, ra, N, M
    mbs, kbs, nbs = ozaki2.plan(ra.xp, ra.kp, N, n)
    cache: dict = {}
    blocks = [
        (D._oz_device_block(ra, cache, "a", kk, m), D._oz_device_block(rb, cache, "b", kk, nn))
        for m in mbs
        for kk in kbs
        for nn in nbs
    ]
    g, c, fu = (t.get(fam) for t in (K2.GEMM_KERNELS, K2.CRT_KERNELS, K2.FUSED_KERNELS))
    res = [D._call_kernel(g, f"ozaki2_gemm_{fam}", at, bt) for at, bt in blocks]
    sync()

    def gemm():
        for at, bt in blocks:
            D._call_kernel(g, f"ozaki2_gemm_{fam}", at, bt)

    def crt():
        for r in res:
            D._call_kernel(c, f"ozaki2_crt_{fam}", r)

    def fused():
        for at, bt in blocks:
            D._call_kernel(fu, f"ozaki2_fused_{fam}", at, bt)

    return gemm, crt, fused


def host_ms(A, B):
    out = {}
    nt = torch.get_num_threads()
    for t in (2, 12):
        torch.set_num_threads(t)
        out[f"host_{t}t_ms"] = timed(lambda: A @ B, reps=2)
    torch.set_num_threads(nt)
    return out


def check_rows(C, A, B, ref_fn, rows=256):
    idx = np.sort(np.random.default_rng(1).choice(A.shape[0], min(rows, A.shape[0]), False))
    As = A[idx]
    return {
        "max_cw": max_cw(C[idx], As, B),
        "host_max_cw": max_cw(As @ B, As, B),
        "bitwise": bool(torch.equal(C[idx], ref_fn(As, B))),
    }


def case(name, shape, arm, c):
    M, K, N = shape
    A, B = inputs(M, K, N)
    rec = {"shape": shape, "name": name, "arm": arm, "count": c}
    t = time.perf_counter()
    if arm == "ozaki1":
        trnblas.gemm_fp64(A, B, slices=c)
        rec["first_call_s"] = time.perf_counter() - t
        rec["e2e_ms"] = timed(lambda: trnblas.gemm_fp64(A, B, slices=c), reps=2)
        C = trnblas.gemm_fp64(A, B, slices=c)
        rec["kernel_ms"] = timed(oz1_kernel_fn(A, B, c))
        rec.update(check_rows(C, A, B, lambda a, b: ozaki.reference_gemm(a, b, c)))
    else:
        K2.gemm(A, B, c, arm, "device")
        rec["first_call_s"] = time.perf_counter() - t
        rec["e2e_ms"] = timed(lambda: K2.gemm(A, B, c, arm, "device"), reps=2)
        if M * N <= 2048 * 2048:  # the host CRT downloads 4 n bytes per output
            rec["e2e_host_crt_ms"] = timed(lambda: K2.gemm(A, B, c, arm, "host"), reps=1)
        C = K2.gemm(A, B, c, arm, "device")
        t = time.perf_counter()
        ozaki2.split_ab(A, B, c, arm)
        rec["host_split_ms"] = (time.perf_counter() - t) * 1e3
        gemm, crt, fused = oz2_kernel_fns(A, B, c, arm)
        rec["residue_gemm_ms"] = timed(gemm)
        rec["crt_ms"] = timed(crt)
        rec["kernel_ms"] = rec["residue_gemm_ms"] + rec["crt_ms"]
        try:
            rec["fused_ms"] = timed(fused)
            rec["e2e_fused_ms"] = timed(lambda: K2.gemm(A, B, c, arm, "fused"), reps=1)
            rec["bitwise_fused"] = bool(torch.equal(K2.gemm(A, B, c, arm, "fused"), C))
        except Exception as exc:  # the fused kernel is optional for the decision
            rec["fused_error"] = repr(exc)
        rec["bf16_gemms"] = c
        rec.update(check_rows(C, A, B, lambda a, b: ozaki2.reference_gemm(a, b, c, arm)))
        rec["bitwise_host_crt"] = bool(
            torch.equal(
                K2.gemm(A[:128], B, c, arm, "host"),
                ozaki2.reference_gemm(A[:128], B, c, arm, "host"),
            )
        )
    rec.update(host_ms(A, B))
    rec["bf16_tflops"] = (
        2 * M * K * N * rec.get("bf16_gemms", c * (c + 1) // 2) / rec["kernel_ms"] / 1e9
    )
    return rec


def micro():
    """Q1: per-GEMM cost of each family's group close at 2048 x 16384 x 2048."""
    M = N = 2048
    K = 16384
    A, B = inputs(M, K, N)
    recs = []
    a16 = D._to_device(A.to(torch.bfloat16))[0][0]
    b16 = D._to_device(B.to(torch.bfloat16))[0][0]
    try:
        ms = timed(lambda: torch.mm(a16, b16, out_dtype=torch.float32))
        recs.append({"what": "aten bf16 mm (fp32 out)", "ms": ms})
    except Exception:  # no out_dtype overload on this backend
        recs.append({"what": "aten bf16 mm (bf16 out)", "ms": timed(lambda: torch.mm(a16, b16))})
    run = oz1_kernel_fn(A, B, 8)
    ms = timed(run)
    recs.append({"what": "ozaki1 k=8 kernel", "ms": ms, "per_gemm_ms": ms / 36})
    for fam in ("b7", "b8", "b9"):
        gemm, _, _ = oz2_kernel_fns(A, B, 4, fam)
        ms = timed(gemm)
        recs.append(
            {
                "what": f"ozaki2 {fam} residue gemm, 4 moduli",
                "group_terms": ozaki2.family(fam).group_terms,
                "ms": ms,
                "per_gemm_ms": ms / 4,
            }
        )
    for fam, n in (("b7", 13), ("b7", 17), ("b8", 15), ("b9", 13)):
        gemm, crt, fused = oz2_kernel_fns(A[:, :4096], B[:4096], n, fam)
        g_ms, c_ms = timed(gemm), timed(crt)
        rec = {
            "what": f"{fam} n={n} at 2048x4096x2048",
            "residue_gemm_ms": g_ms,
            "crt_ms": c_ms,
            "crt_ns_per_element": c_ms * 1e6 / (M * N),
        }
        try:
            rec["fused_ms"] = timed(fused)
        except Exception as exc:
            rec["fused_error"] = repr(exc)
        recs.append(rec)
        print(recs[-1], flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["micro", "shapes"])
    ap.add_argument("--arms", nargs="*", default=["ozaki1", "b7"])
    ap.add_argument("--counts", nargs="*", type=int, help="override the configs' counts")
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES))
    ap.add_argument("--out", default="probe_ozaki2.json")
    ap.add_argument("--budget-s", type=float, default=1080)
    a = ap.parse_args()
    if not (D.HAS_NKI and D._get_dispatch_path() == "native"):
        sys.exit("needs the native TorchNeuron path (Trn2)")
    D._REQUIRE_NKI = True
    meta = {
        "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t0 = time.time()
    out = Path(a.out)
    if a.mode == "micro":
        out.write_text(json.dumps({"meta": meta, "micro": micro()}, indent=1))
        return
    recs = []
    for name in a.shapes:
        for arm in a.arms:
            for c in a.counts or CONFIGS[arm]:
                if time.time() - t0 > a.budget_s:
                    print(f"budget reached, skipping {name} {arm} {c}", flush=True)
                    continue
                try:
                    recs.append(case(name, SHAPES[name], arm, c))
                except Exception as exc:  # record and go on
                    recs.append({"name": name, "arm": arm, "count": c, "error": repr(exc)})
                print(json.dumps(recs[-1]), flush=True)
                out.write_text(json.dumps({"meta": meta, "cases": recs}, indent=1))
    meta["seconds"] = time.time() - t0
    out.write_text(json.dumps({"meta": meta, "cases": recs}, indent=1))


if __name__ == "__main__":
    main()
