"""Measure every candidate route of the native level-3 wrappers (trnblas#62, #63).

Usage::

    bench_auto_routes.py <op> <dtype> [--shapes S;S;...] [--routes r,r] [--json FILE]

``op`` is gemm | batched | syrk | trsm | symm; ``dtype`` is fp32 | bf16. A shape
is ``x``-separated (gemm ``MxKxN``, batched ``BxMxKxN``, syrk ``NxK``, trsm and
symm ``NxNRHS``). Routes, each timed as a warm median from host inputs to a
host result (``host_*``) or from device-resident inputs to a device result
(``dev_*``):

    host_cpu   the op in torch on the host (CPU, the pinned host threads)
    host_aten  upload, aten on the device, download through `_to_host`
    host_nki   the NKI kernel path from host inputs (upload, kernel, download)
    host_hyb   trsm only: diagonal panels on the host, updates as aten on the device
    dev_aten   aten on device-resident operands
    dev_nki    the NKI kernel on device-resident operands
    dev_host   trsm only: device operands solved on the host (down, LAPACK, up)
    auto_host  the public wrapper under `set_backend("auto")`, host inputs
    auto_dev   the public wrapper under `set_backend("auto")`, device inputs

The routes call the building blocks directly, so they do not depend on the
routing being measured; ``auto_*`` show what the library picks. NKI kernel
errors are re-raised (``_REQUIRE_NKI``), never silently replaced by torch.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import torch
import torch_neuronx  # noqa: F401

import trnblas
from trnblas.nki import dispatch as D

DEV = "neuron"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

DEFAULT_SHAPES = {
    "gemm": (
        "64x64x64;128x128x128;256x256x256;384x384x384;512x512x512;768x768x768;"
        "1024x1024x1024;1536x1536x1536;2048x2048x2048;3072x3072x3072;4096x4096x4096;"
        "8192x8192x8192;1000x1000x1000;1500x700x2900;3000x3000x3000;"
        "8192x64x8192;64x8192x64;4096x4096x64;64x4096x4096;16384x128x128;"
        "16x128x49152;32x256x196608;7168x768x768;1792x384x384"
    ),
    "batched": (
        "8x56x64x192;16x112x128x384;32x224x256x768;32x224x768x768;"
        "64x128x128x128;128x64x64x64;16x1024x1024x1024;4x2048x2048x2048"
    ),
    "syrk": (
        "256x256;512x512;1024x256;1024x1024;2048x512;2048x2048;4096x1024;8192x2048;"
        "768x4096;1000x3000"
    ),
    "trsm": (
        "96x96;192x192;384x384;768x768;1152x1152;1536x1536;2304x2304;512x8192;1024x4096;"
        "1024x16384;2048x8192;256x65536"
    ),
    "symm": "256x256;1024x1024;2048x2048;4096x4096;1024x8192",
}
ROUTES = (
    "host_cpu",
    "host_aten",
    "host_nki",
    "host_hyb",
    "dev_aten",
    "dev_nki",
    "dev_host",
    "auto_host",
    "auto_dev",
)


def sync():
    D._device_synchronize()


def host(*shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype)


def bench(fn, budget=0.4, lo=3, hi=40):
    """Warm median (and min) in ms; the repeat count adapts to the op's speed."""
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


def rel(x, ref):
    x = x.detach().to("cpu", torch.float64)
    return float((x - ref).norm() / ref.norm())


def up(*ts):
    return [t.to(DEV) for t in ts]


def down(t):
    return D._to_host(t, "cpu")


# ── per-op candidates: returns {route: fn}, check(out) → rel err ─────────────


def gemm_case(M, K, N, dt):
    A, B = host(M, K, dtype=dt), host(K, N, dtype=dt, seed=1)
    a, b = up(A, B)
    rows = min(M, 64)
    ref = A[:rows].double() @ B.double()
    fns = {
        "host_cpu": lambda: torch.matmul(A, B),
        "host_aten": lambda: down(torch.matmul(*up(A, B))),
        "host_nki": lambda: D._nki_gemm_impl(A, B),
        "dev_aten": lambda: torch.matmul(a, b),
        "dev_nki": lambda: D._nki_gemm_impl(a, b),
        "auto_host": lambda: trnblas.gemm(1.0, A, B),
        "auto_dev": lambda: trnblas.gemm(1.0, a, b),
    }
    return fns, lambda out: rel(out[:rows], ref), (A, B, a, b)


def batched_case(Bt, M, K, N, dt):
    A, B = host(Bt, M, K, dtype=dt), host(Bt, K, N, dtype=dt, seed=1)
    a, b = up(A, B)
    ref = A[-1].double() @ B[-1].double()

    def nki_loop(X, Y):
        out = None
        for i in range(X.shape[0]):
            c = D._nki_gemm_impl(X[i], Y[i])
            if out is None:
                out = D._host_empty((X.shape[0], *c.shape), c.dtype, c.device)
            out[i].copy_(c)
        return out

    fns = {
        "host_cpu": lambda: torch.matmul(A, B),
        "host_aten": lambda: down(torch.matmul(*up(A, B))),
        "host_nki": lambda: nki_loop(A, B),
        "dev_aten": lambda: torch.matmul(a, b),
        "dev_nki": lambda: nki_loop(a, b),
        "auto_host": lambda: trnblas.batched_gemm(1.0, A, B),
        "auto_dev": lambda: trnblas.batched_gemm(1.0, a, b),
    }
    return fns, lambda out: rel(out[-1], ref), (A, B, a, b)


def _sym_dev(r):
    return 0.5 * (r + r.T)


def syrk_case(N, K, dt):
    A = host(N, K, dtype=dt)
    (a,) = up(A)
    rows = min(N, 64)
    ref = A[:rows].double() @ A.double().T

    def host_aten():
        (x,) = up(A)
        r = torch.matmul(x, x.T)
        if N >= D._SYRK_DEVICE_SYM_MIN_N:
            return down(_sym_dev(r))
        return D._symmetrize(down(r).contiguous())

    fns = {
        "host_cpu": lambda: D._symmetrize(torch.matmul(A, A.T)),
        "host_aten": host_aten,
        "host_nki": lambda: D._nki_syrk_impl(A, symmetrize=True),
        "dev_aten": lambda: _sym_dev(torch.matmul(a, a.T)),
        "dev_nki": lambda: D._nki_syrk_impl(a, symmetrize=True),
        "auto_host": lambda: trnblas.syrk(1.0, A),
        "auto_dev": lambda: trnblas.syrk(1.0, a),
    }
    return fns, lambda out: rel(out[:rows], ref), (A, a)


def _with_inner_gemm(inner, fn):
    """Run `fn` with the blocked TRSM's trailing-update GEMM pinned to `inner`."""

    def run():
        saved = D.nki_gemm
        D.nki_gemm = inner
        try:
            return fn()
        finally:
            D.nki_gemm = saved

    return run


def trsm_case(N, NRHS, dt):
    # The DF-MP2 metric shape: X = L^-T, i.e. lower, trans=True.
    g = torch.Generator().manual_seed(0)
    L = torch.tril(torch.randn(N, N, generator=g)) * (1.0 / N**0.5) + 2.0 * torch.eye(N)
    L = L.to(dt)
    B = host(N, NRHS, dtype=dt, seed=1)
    l, b = up(L, B)
    kw = dict(uplo="lower", trans=True, diag="nonunit", alpha=1.0)
    Ld = L.double()

    def check(out):
        cols = min(NRHS, 64)
        x = out[:, :cols].detach().to("cpu", torch.float64)
        return float((Ld.T @ x - B[:, :cols].double()).norm() / B[:, :cols].double().norm())

    def blocked(Lx, Bx):
        return D._nki_trsm_left(Lx, Bx, kw["uplo"], kw["trans"], kw["diag"], 1.0)

    fns = {
        "host_cpu": lambda: D._trsm_torch(1.0, L, B, "left", "lower", True, "nonunit"),
        "host_aten": _with_inner_gemm(torch.matmul, lambda: down(blocked(*up(L, B)))),
        "host_nki": _with_inner_gemm(D._nki_gemm_impl, lambda: blocked(L, B)),
        "dev_aten": _with_inner_gemm(torch.matmul, lambda: blocked(l, b)),
        "dev_nki": _with_inner_gemm(D._nki_gemm_impl, lambda: blocked(l, b)),
        # host panels, device updates (aten), host operands
        "host_hyb": _with_inner_gemm(
            lambda x, y: D._aten_on_device(torch.matmul, x, y), lambda: blocked(L, B)
        ),
        # device operands solved on the host: down, LAPACK, up
        "dev_host": lambda: D._trsm_torch(
            1.0, l.cpu(), b.cpu(), "left", "lower", True, "nonunit"
        ).to(DEV),
        "auto_host": lambda: trnblas.trsm(1.0, L, B, uplo="lower", trans=True),
        "auto_dev": lambda: trnblas.trsm(1.0, l, b, uplo="lower", trans=True),
    }
    return fns, check, (L, B, l, b)


def symm_case(N, NRHS, dt):
    S = host(N, N, dtype=dt)
    S = (S + S.T) * 0.5
    B = host(N, NRHS, dtype=dt, seed=1)
    s, b = up(S, B)
    rows = min(N, 64)
    ref = S[:rows].double() @ B.double()
    fns = {
        "host_cpu": lambda: _symm_torch(S, B),
        "host_aten": lambda: down(_symm_torch(*up(S, B))),
        "dev_aten": lambda: _symm_torch(s, b),
        "auto_host": lambda: trnblas.symm(1.0, S, B),
        "auto_dev": lambda: trnblas.symm(1.0, s, b),
    }
    return fns, lambda out: rel(out[:rows], ref), (S, B, s, b)


def _symm_torch(A, B):
    sym = torch.triu(A) + torch.triu(A, diagonal=1).T
    return torch.matmul(sym, B)


CASES = {
    "gemm": gemm_case,
    "batched": batched_case,
    "syrk": syrk_case,
    "trsm": trsm_case,
    "symm": symm_case,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("op", choices=list(CASES))
    p.add_argument("dtype", choices=list(DTYPES))
    p.add_argument("--shapes")
    p.add_argument("--routes", default=",".join(ROUTES))
    p.add_argument("--json")
    p.add_argument("--label", default="")
    args = p.parse_args()
    dt = DTYPES[args.dtype]
    routes = args.routes.split(",")
    D._REQUIRE_NKI = True
    trnblas.set_backend("auto")
    shapes = (args.shapes or DEFAULT_SHAPES[args.op]).split(";")
    for shape in shapes:
        dims = tuple(int(x) for x in shape.split("x"))
        try:
            fns, check, keep = CASES[args.op](*dims, dt)
        except Exception as e:  # noqa: BLE001
            print(f"{args.op} {shape} {args.dtype} SETUP FAILED {type(e).__name__}: {e}"[:300])
            continue
        res = {}
        for r in routes:
            if r not in fns:
                continue
            try:
                med, mn = bench(fns[r])
                err = check(fns[r]())
                res[r] = (med, mn, err)
                print(
                    f"{args.label} {args.op:7s} {shape:16s} {args.dtype} {r:9s} "
                    f"{med:9.3f} ms (min {mn:9.3f})  err {err:.1e}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                res[r] = None
                print(
                    f"{args.label} {args.op:7s} {shape:16s} {args.dtype} {r:9s} FAILED "
                    f"{type(e).__name__}: {str(e)[:160]}",
                    flush=True,
                )
            gc.collect()
        if args.json:
            with open(args.json, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "label": args.label,
                            "op": args.op,
                            "shape": shape,
                            "dtype": args.dtype,
                            "res": res,
                        }
                    )
                    + "\n"
                )
        del fns, keep
        gc.collect()


if __name__ == "__main__":
    main()
