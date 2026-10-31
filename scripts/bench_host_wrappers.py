"""Host-in / host-out trnblas wrappers on one logical core (trnblas#61).

Usage: bench_host_wrappers.py <label> [case ...]
Prints one line per case: warm median ms, effective host-link GB/s, check error.
"""

import os
import resource
import statistics
import sys
import time

import torch
import torch_neuronx  # noqa: F401

import trnblas
from trnblas.nki import dispatch as D

REPS = int(os.environ.get("REPS", "5"))
label = sys.argv[1]
want = set(sys.argv[2:])


def sync():
    D._device_synchronize()


def host(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def bench(fn):
    r = fn()
    sync()
    del r
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        r = fn()
        sync()
        ts.append(time.perf_counter() - t0)
        del r
    return statistics.median(ts), min(ts)


def rel(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


def report(name, t, nbytes, err, extra=""):
    med, mn = t
    print(
        f"{label:10s} {name:34s} {med * 1e3:9.1f} ms (min {mn * 1e3:8.1f})  link {nbytes / 1e6:8.0f} MB  err {err:.1e} {extra}",
        flush=True,
    )


def gemm_case(M, K, N, alpha=1.0, beta=0.0):
    A, B = host(M, K), host(K, N, seed=1)
    C = host(M, N, seed=2) if beta else None
    t = bench(lambda: trnblas.gemm(alpha, A, B, beta=beta, C=C))
    out = trnblas.gemm(alpha, A, B, beta=beta, C=C)
    ref = alpha * (A[:128].double() @ B.double()) + (beta * C[:128].double() if beta else 0)
    err = rel(out[:128], ref)
    pinned = out.is_pinned()
    del out
    return t, 4 * (M * K + K * N + M * N), err, f"pinned={pinned}"


cases = {}
cases["gemm 4096^3"] = lambda: gemm_case(4096, 4096, 4096)
cases["gemm 8192^3"] = lambda: gemm_case(8192, 8192, 8192)
cases["gemm 8192x1024x8192"] = lambda: gemm_case(8192, 1024, 8192)
cases["gemm 8192^3 a=.5 b=1 C"] = lambda: gemm_case(8192, 8192, 8192, 0.5, 1.0)


def batched():
    A, B = host(16, 1024, 1024), host(16, 1024, 1024, seed=1)
    t = bench(lambda: trnblas.batched_gemm(1.0, A, B))
    out = trnblas.batched_gemm(1.0, A, B)
    err = max(rel(out[i], A[i].double() @ B[i].double()) for i in (0, 15))
    return t, 4 * 3 * A.numel(), err, f"pinned={out.is_pinned()}"


cases["batched_gemm 16x1024^3"] = batched


def syrk():
    A = host(8192, 2048)
    t = bench(lambda: trnblas.syrk(1.0, A))
    out = trnblas.syrk(1.0, A)
    err = rel(out[:128], A[:128].double() @ A.double().T)
    return (
        t,
        4 * (A.numel() + 8192 * 8192),
        err,
        f"pinned={out.is_pinned()} sym={bool(torch.equal(out, out.T))}",
    )


cases["syrk 8192x2048"] = syrk


def trsm():
    n, nrhs = 1024, 16384
    L = torch.tril(host(n, n)) * 0.01 + 3 * torch.eye(n)
    B = host(n, nrhs, seed=1)
    t = bench(lambda: trnblas.trsm(1.0, L, B, uplo="lower"))
    X = trnblas.trsm(1.0, L, B, uplo="lower")
    err = rel(L.double() @ X.double(), B.double())
    # link bytes: per block, upload L panel + X block, download the update
    nb = 0
    for k in range(0, n, 128):
        ke = min(k + 128, n)
        if ke < n:
            nb += 4 * ((n - ke) * 128 + 128 * nrhs + (n - ke) * nrhs)
    return t, nb, err, f"pinned={X.is_pinned()}"


cases["trsm L 1024 x 16384 rhs"] = trsm


def breakdown():
    """gemm 8192^3 pieces: upload, NKI kernel, download two ways, aten mm alt."""
    M = K = N = 8192
    A, B = host(M, K), host(K, N, seed=1)
    up = bench(lambda: (A.to("neuron"), B.to("neuron")))
    report("  upload A+B (.to)", up, 4 * (M * K + K * N), 0.0)
    a, b = A.to("neuron"), B.to("neuron")
    k = D._get_gemm_kernel(128, 128, 512)
    ker = bench(lambda: D._call_kernel(k, D._gemm_op_name(128, 128, 512), a, b))
    report("  NKI kernel, resident", ker, 0, 0.0)
    at = bench(lambda: torch.matmul(a, b))
    report("  aten mm, resident", at, 0, 0.0)
    c = torch.matmul(a, b)
    sync()
    report("  download .cpu()", bench(lambda: c.cpu()), 4 * M * N, 0.0)
    report("  download _to_host", bench(lambda: D._to_host(c, "cpu")), 4 * M * N, 0.0)
    h = c.cpu()
    report("  host 0.5 * C (fresh)", bench(lambda: 0.5 * h), 0, 0.0)
    report("  host C.mul_(0.5)", bench(lambda: h.mul_(0.5)), 0, 0.0)
    alt = bench(lambda: D._to_host(torch.matmul(A.to("neuron"), B.to("neuron")), "cpu"))
    report("  ALT up + aten mm + _to_host", alt, 4 * (M * K + K * N + M * N), 0.0)
    return None


cases["breakdown"] = breakdown

trnblas.set_backend("auto")
for name, fn in cases.items():
    if want and not any(w in name for w in want):
        continue
    try:
        r = fn()
        if r is not None:
            report(name, *r)
    except Exception as e:
        print(f"{label:10s} {name:34s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
print(
    f"{label:10s} maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB",
    flush=True,
)
