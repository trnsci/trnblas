"""Device-resident GEMM: aten vs the NKI GEMM kernels (trnblas#79).

Usage::

    bench_lnc_gemm.py --shapes MxKxN,... [--dtypes fp32,bf16] [--routes r,r] [--pipe] [--json FILE]

Routes, each a warm median (ms) of one call plus a device synchronize, on
operands already on the ``neuron`` device (inputs are built on the host and
copied over, aws-neuron-sdk#1439):

    aten      ``a @ b``, A row-major
    aten_tn   ``at.T @ b``, A a transposed view (the DF-MP2 half-transform form)
    old       the single-program kernel (``gemm_128_128_512`` / ``gemm_ragged``)
    lnc       ``_gemm_lnc_kernel`` (A row-major, Tensor Engine transposes)
    tn        ``_gemm_lnc_tn_kernel`` (A^T given, no transposes)
    auto      the public ``trnblas.gemm`` under ``set_backend("auto")``, A a
              transposed view

``--pipe`` also times ``n`` back-to-back calls with one synchronize at the end
(``*_pipe``: steady-state time per call, the host issue cost overlapped) and the
host issue time alone (``*_issue``). Relative errors (``*_err``) are against a
float64 product on the host. Run with ``NEURON_RT_VISIBLE_CORES`` set to one
logical core; LNC comes from ``NEURON_LOGICAL_NC_CONFIG``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import trnblas
from trnblas.nki import dispatch as D

DT = {"fp32": torch.float32, "bf16": torch.bfloat16}


def sync():
    D._device_synchronize()


def bench(fn, budget=0.5, lo=3, hi=30):
    r = fn()
    sync()
    del r
    t0 = time.perf_counter()
    r = fn()
    sync()
    first = time.perf_counter() - t0
    del r
    reps = max(lo, min(hi, int(budget / max(first, 1e-6))))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        sync()
        ts.append(time.perf_counter() - t0)
        del r
    return statistics.median(ts) * 1e3, min(ts) * 1e3


def _routes(ops, old, a, b, at):
    return {
        "aten": lambda: a @ b,
        "aten_tn": lambda: at.T @ b,
        "old": lambda: old(a, b),
        "lnc": lambda: ops["gemm_lnc"](a, b),
        "tn": lambda: ops["gemm_lnc_tn"](at, b),
        "auto": lambda: trnblas.gemm(1.0, at, b, transA=True),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", required=True)
    p.add_argument("--dtypes", default="fp32,bf16")
    p.add_argument("--routes", default="aten,aten_tn,lnc,tn")
    p.add_argument("--pipe", action="store_true")
    p.add_argument("--json")
    p.add_argument("--label", default="")
    args = p.parse_args()

    D._get_dispatch_path()
    ops = D._get_native_ops()
    D._REQUIRE_NKI = True
    trnblas.set_backend("auto")

    def old(a, b):
        M, K = a.shape
        N = b.shape[1]
        if M % 128 == 0 and K % 128 == 0 and (N <= 512 or N % 512 == 0):
            return ops["gemm_128_128_512"](a, b)
        return ops["gemm_ragged"](a, b)

    rows = []
    for s in args.shapes.split(","):
        M, K, N = map(int, s.split("x"))
        for dn in args.dtypes.split(","):
            g = torch.Generator().manual_seed(0)
            A = torch.randn(M, K, generator=g).to(DT[dn])
            B = torch.randn(K, N, generator=g).to(DT[dn])
            ref = A.double() @ B.double() if M * K * N <= 2**33 else None
            a, b = A.to("neuron"), B.to("neuron")
            at = A.T.contiguous().to("neuron")
            routes = _routes(ops, old, a, b, at)
            row = {"shape": s, "dtype": dn, "label": args.label}
            for rn in args.routes.split(","):
                fn = routes[rn]
                t0 = time.perf_counter()
                c = fn()
                sync()
                row[rn + "_first_s"] = round(time.perf_counter() - t0, 2)
                if ref is not None:
                    err = (c.to("cpu", torch.float64) - ref).norm() / ref.norm()
                    row[rn + "_err"] = float(err)
                del c
                med, mn = bench(fn)
                row[rn], row[rn + "_min"] = round(med, 3), round(mn, 3)
                if args.pipe:
                    n = max(5, min(50, int(500 / max(med, 1e-3))))
                    sync()
                    t0 = time.perf_counter()
                    rs = [fn() for _ in range(n)]
                    t1 = time.perf_counter()
                    sync()
                    t2 = time.perf_counter()
                    del rs
                    row[rn + "_pipe"] = round((t2 - t0) / n * 1e3, 3)
                    row[rn + "_issue"] = round((t1 - t0) / n * 1e3, 3)
            print(json.dumps(row), flush=True)
            rows.append(row)
            del a, b, at, routes  # free device memory between cases (#1447)
    if args.json:
        with open(args.json, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
