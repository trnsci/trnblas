#!/usr/bin/env python
"""Crossover for the DF metric factor X = L^-T (J = L L^T): host LAPACK vs device.

For each naux, J is an SPD fp32 matrix resident on the `neuron` device. Times
(warm median of reps 1..n, device-synchronised, ms):

    device   torch.linalg.cholesky(J) + trnblas trsm(L, I) on the device
             (split into chol / trsm), `set_backend("nki")`
    host     J -> host, float64 LAPACK potrf + trsm, X -> device
             (`TRNBLAS_METRIC_SOLVE=host`)

and the accuracy max|X^T J X - I| (float64, on the host) of each.

    python scripts/bench_metric_solve.py --naux 128 256 512 1024 --reps 7
    python scripts/bench_metric_solve.py --cond 1e7   # RI-like conditioning

Each naux prints one ``RESULT {json}`` line.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


def _spd(naux: int, cond: float, seed: int = 0) -> torch.Tensor:
    """SPD fp32 matrix with eigenvalues log-spaced over [1/cond, 1] * naux."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(naux, naux, generator=g, dtype=torch.float64))
    w = torch.logspace(0, -torch.log10(torch.tensor(cond)).item(), naux, dtype=torch.float64)
    J = (Q * (naux * w)) @ Q.T
    return (0.5 * (J + J.T)).float()


def _residual(J: torch.Tensor, X: torch.Tensor) -> float:
    Jd = J.detach().cpu().double()
    Xd = X.detach().cpu().double()
    return float((Xd.T @ Jd @ Xd - torch.eye(Jd.shape[0], dtype=torch.float64)).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--naux", type=int, nargs="+", default=[71, 128, 300, 555, 1024, 1536, 2304, 3072, 4096]
    )
    ap.add_argument("--cond", type=float, default=1e4)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    import trnblas
    from trnblas.nki import dispatch

    trnblas.set_backend("nki")
    dev = "neuron"

    for naux in args.naux:
        J = _spd(naux, args.cond).to(dev)
        dispatch._device_synchronize()

        dev_tot, dev_chol, dev_trsm = [], [], []
        X_dev = None
        for _ in range(args.reps):
            t0 = time.perf_counter()
            L = torch.linalg.cholesky(J)
            dispatch._device_synchronize()
            t1 = time.perf_counter()
            eye = torch.eye(naux, dtype=J.dtype, device=J.device)
            X_dev = dispatch.nki_trsm(L, eye, uplo="lower", trans=True)
            dispatch._device_synchronize()
            t2 = time.perf_counter()
            dev_chol.append(t1 - t0)
            dev_trsm.append(t2 - t1)
            dev_tot.append(t2 - t0)

        os.environ["TRNBLAS_METRIC_SOLVE"] = "host"
        host_tot = []
        X_host = None
        for _ in range(args.reps):
            t0 = time.perf_counter()
            X_host = dispatch.nki_metric_inv_half(J)
            dispatch._device_synchronize()
            host_tot.append(time.perf_counter() - t0)
        os.environ.pop("TRNBLAS_METRIC_SOLVE")
        assert X_host.device.type == dev

        def warm(xs):
            return round(1e3 * statistics.median(xs[1:] if len(xs) > 1 else xs), 3)

        rec = {
            "naux": naux,
            "cond": args.cond,
            "device_ms": warm(dev_tot),
            "device_chol_ms": warm(dev_chol),
            "device_trsm_ms": warm(dev_trsm),
            "device_ms_min": round(1e3 * min(dev_tot[1:] or dev_tot), 3),
            "device_ms_max": round(1e3 * max(dev_tot[1:] or dev_tot), 3),
            "host_ms": warm(host_tot),
            "host_ms_min": round(1e3 * min(host_tot[1:] or host_tot), 3),
            "host_ms_max": round(1e3 * max(host_tot[1:] or host_tot), 3),
            "device_cold_ms": round(1e3 * dev_tot[0], 1),
            "resid_device": _residual(J, X_dev),
            "resid_host": _residual(J, X_host),
        }
        print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
