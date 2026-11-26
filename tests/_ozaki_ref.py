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


def device_split_ops() -> dict:
    """CPU stand-ins for the device-split ops of gemm_fp64 (#123), built on
    `ozaki.split_from_bits` (the kernels' arithmetic op for op), for tests
    that run the native route on the host."""
    from trnblas import ozaki

    def x_of(xw):
        return xw.contiguous().view(torch.float64).numpy()

    def exp_field(X):
        return ((X.view(np.uint64) >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.float32)

    def pad(D, k, rp, cp):
        out = torch.zeros((k, rp, cp), dtype=torch.bfloat16)
        out[:, : D.shape[1], : D.shape[2]] = torch.from_numpy(D)
        return out

    def emax(xw):
        return torch.from_numpy(exp_field(x_of(xw)).max(axis=1, initial=0)[:, None])

    def emax_col(xw):
        return torch.from_numpy(exp_field(x_of(xw)).max(axis=0, initial=0)[None, :])

    def split(xw, rcs):
        X = x_of(xw)
        k = rcs.shape[1]
        D, e = ozaki.split_from_bits(X, k)
        assert np.array_equal(rcs[: X.shape[0], 0].numpy(), (1003 + e).astype(np.float32))
        rp, cp = (ozaki._round_up(n, 128) for n in X.shape)
        return pad(D, k, rp, cp)

    def split_col(xw, rcs):
        X = x_of(xw)
        k = rcs.shape[0]
        D, e = ozaki.split_from_bits(np.ascontiguousarray(X.T), k)
        assert np.array_equal(rcs[0, : X.shape[1]].numpy(), (1003 + e).astype(np.float32))
        rp, cp = (ozaki._round_up(n, 128) for n in X.shape)
        return pad(np.ascontiguousarray(D.transpose(0, 2, 1)), k, rp, cp)

    return {
        "ozaki_emax": emax,
        "ozaki_emax_col": emax_col,
        "ozaki_split": split,
        "ozaki_split_col": split_col,
    }


DEVICE_SPLIT_KERNELS = (
    "_ozaki_emax_kernel",
    "_ozaki_emax_col_kernel",
    "_ozaki_split_kernel",
    "_ozaki_split_col_kernel",
)


def ozaki2_kernel_refs() -> dict:
    """CPU stand-ins for the Ozaki II device-split and combine ops (#132):
    each returns its kernel's raw output (layout included), built on the
    host arithmetic of `trnblas.ozaki2` (`norm_terms`, `ozaki.split_from_bits`
    with given exponents, `residues_from_digits`, `combine_kernel_reference`).
    For tests that run the native route on the host, and to check the
    kernels' raw outputs under the simulator."""
    from trnblas import ozaki, ozaki2

    def x_of(xw):
        xw = xw.detach().cpu() if isinstance(xw, torch.Tensor) else torch.from_numpy(xw)
        return xw.contiguous().view(torch.float64).numpy()

    def arr(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    def limbs(tau):
        t = tau.astype(np.int64)
        return [(t >> (9 * i)) & 511 if i < 3 else t >> 27 for i in range(4)]

    def norm(xw, rcs):
        X = x_of(xw)
        R, C = X.shape
        e = arr(rcs)[:R, 0].astype(np.int64) - 1003
        tau = ozaki2.norm_terms(X, e, 1)
        T = -(-C // 512)
        levels = 0
        while levels < 9 and T * 2 ** (9 + levels + 1) <= 2**24:
            levels += 1
        Wo = 512 >> levels
        pad = np.zeros((R, T * 512))
        pad[:, :C] = tau
        out = [l.reshape(R, -1, Wo).sum(axis=1) for l in limbs(pad)]
        return torch.from_numpy(np.concatenate(out, axis=1).astype(np.float32))

    def norm_col(xw, rcs):
        X = x_of(xw)
        R, C = X.shape
        e = arr(rcs)[0, :C].astype(np.int64) - 1003
        tau = ozaki2.norm_terms(X, e, 0)
        NT = -(-R // 128)
        NG = max(1, -(-NT // 255))
        out = np.zeros((4, NG, C))
        for i, l in enumerate(limbs(tau)):
            for g in range(NG):
                out[i, g] = l[g * 255 * 128 : (g + 1) * 255 * 128].sum(axis=0)
        return torch.from_numpy(out.reshape(4 * NG, C).astype(np.float32))

    def split(fam, per):
        def op(xw, rcs, mods):
            X = x_of(xw)
            n = arr(mods).shape[1]
            rc = arr(rcs)
            if per == "col":
                X = np.ascontiguousarray(X.T)
                rc = rc.T
            R, C = X.shape
            k = rc.shape[1]
            ep = rc[:R, 0].astype(np.int64) - 1003
            D, _ = ozaki.split_from_bits(X, k, e=ep)
            res = ozaki2.residues_from_digits(D, ozaki2.family(fam).moduli[:n])
            out = np.zeros((n, ozaki._round_up(R, 128), ozaki._round_up(C, 128)), np.float32)
            out[:, :R, :C] = res
            if per == "col":
                out = np.ascontiguousarray(out.transpose(0, 2, 1))
            return torch.from_numpy(out).to(torch.bfloat16)

        return op

    ops = {
        "ozaki2_norm": norm,
        "ozaki2_norm_col": norm_col,
        "ozaki2_combine": lambda p, ea, eb: torch.from_numpy(
            ozaki2.combine_kernel_reference(p, ea, eb)
        ),
    }
    for fam in ("b7", "b8", "b9"):
        ops[f"ozaki2_split_{fam}"] = split(fam, "row")
        ops[f"ozaki2_split_col_{fam}"] = split(fam, "col")
    return ops


OZAKI2_DEVICE_KERNELS = ("_ozaki_emax_kernel", "_ozaki_emax_col_kernel")
