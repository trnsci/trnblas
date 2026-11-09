"""Independent fp64 reference for the Ozaki GEMM tests (#22).

`ref_exact` is near-correctly-rounded: an fp64 Ozaki split with slices narrow
enough that every host fp64 slice GEMM is exact, summed with Neumaier
compensation, smallest first. It shares no code with `trnblas.ozaki`.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def ref_exact(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    K = A.shape[1]
    b = math.floor((55 - math.log2(max(K, 2))) / 2)
    k = math.ceil(80 / b)

    def sp(X, axis):
        amax = np.abs(X).max(axis=axis, keepdims=True)
        e = np.where(amax > 0, np.ceil(np.log2(np.where(amax > 0, amax, 1.0))), 0.0)
        R = X * np.exp2(-e)
        out = []
        for s in range(k):
            sh = (b - 1) if s == 0 else b
            T = R * 2.0**sh
            Ii = np.rint(T)
            R = T - Ii
            out.append(Ii * np.exp2(e - ((b - 1) + b * s)))
        return out

    As, Bs = sp(A, 1), sp(B, 0)
    pairs = sorted(
        [(s, t) for s in range(k) for t in range(k) if s + t <= k - 1], key=lambda p: -sum(p)
    )
    S = c = None
    for s, t in pairs:
        P = As[s] @ Bs[t]
        if S is None:
            S, c = P, np.zeros_like(P)
            continue
        T = S + P
        big = np.abs(S) >= np.abs(P)
        c += np.where(big, (S - T) + P, (P - T) + S)
        S = T
    return S + c


def max_cw(C, A, B) -> float:
    """max |C - C_exact| / (|A| |B|), componentwise."""
    A, B = A.cpu().numpy(), B.cpu().numpy()
    den = np.abs(A) @ np.abs(B)
    den = np.where(den > 0, den, 1.0)
    return float((np.abs(np.asarray(C.cpu()) - ref_exact(A, B)) / den).max())


def inputs(M, K, N, dist="normal", seed=0):
    """fp64 host A (M, K), B (K, N): normal(0, 1), or "logu4" (entries spread
    over 10^+-4, the hard case for per-row / per-column scaling)."""
    rng = np.random.default_rng(seed)
    if dist == "normal":
        A, B = rng.standard_normal((M, K)), rng.standard_normal((K, N))
    else:
        A = rng.standard_normal((M, K)) * 10.0 ** rng.uniform(-4, 4, (M, K))
        B = rng.standard_normal((K, N)) * 10.0 ** rng.uniform(-4, 4, (K, N))
    return torch.from_numpy(A), torch.from_numpy(B)
