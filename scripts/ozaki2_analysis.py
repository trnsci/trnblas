"""CPU analysis of Ozaki scheme II against Ozaki I for gemm_fp64 (trnblas#132 Q1, Q2).

Accuracy. Both schemes scale per row of A and per column of B, so a sampled
(rows x columns) block of C is exactly the full problem's; only K matters.
For each K and data set, A (128 x K) and B (K x 128) are drawn and:
- the reference is near-exact, as a double-double: an fp64 split into
  slices narrow enough that every host GEMM is exact (220 bits per row /
  column), Neumaier-summed;
- each scheme's *inherent* result is formed exactly and kept as a
  double-double (Ozaki I: the exact sum of its slice products; Ozaki II: the
  exact CRT integer), so bits beyond fp64 (72, 96) can be measured;
- the *fp64 output* is the bitwise host reference of the device arithmetic
  (`ozaki.reference_gemm`, `ozaki2.reference_gemm`).
bits = -log2(max |C - C_exact| / (|A| |B|)).

Arms: Ozaki I k slices (k (k + 1) / 2 bf16 GEMMs); Ozaki II bf16 residue
families b7 / b8 / b9 (one bf16 GEMM per modulus, PSUM groups of 4096 / 1024
/ 256 terms); Ozaki II FP8 E4M3 (f8, arXiv:2603.10634: 3 FP8 GEMMs per
modulus, run with perf_mode=double_row at 2x, i.e. 1.5 bf16-equivalent).

Cost model (Trn2, one logical core at LNC=2), in bf16-equivalent TensorE
time: the measured Ozaki I kernel time per bf16 slice GEMM at each shape
(#124: 0.024 ms at 1024^3 ... 1.43 ms at 4096^3; 96 TFLOP/s beyond), and
VectorE at 0.019 ms per instruction per 2048^2 fp32 elements (#22). The
kernel is modelled as max(TensorE, VectorE) (the engines overlap; at 4096-term
groups the measured Ozaki I kernel matched its TensorE floor).

Usage: python scripts/ozaki2_analysis.py [--out DIR] [--Ks ...] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from trnblas import ozaki, ozaki2

KS = (1024, 2048, 3508, 4096, 8192, 8400, 16384)
DATA = ("normal", "logu4", "logu8")
TARGET_BITS = (24, 48, 53, 72, 96)
SAMPLE = 128
ARMS = {
    "ozaki1": range(2, 17),
    "b7": range(3, 29),
    "b8": range(3, 33),
    "b9": range(3, 33),
    "f8": range(3, 33),
}
SQUARES = (1024, 2048, 4096, 8192, 16384)
NARROW = ((3267, 8400, 241), (8192, 3508, 241))

# measured Ozaki I kernel ms per bf16 slice GEMM (#124, k = 8: kernel / 36)
MS_PER_GEMM = {
    (1024, 1024, 1024): 0.86 / 36,
    (2048, 2048, 2048): 6.60 / 36,
    (4096, 4096, 4096): 51.55 / 36,
    (3267, 8400, 241): 12.94 / 36,
    (8192, 3508, 241): 10.99 / 36,
}
BF16_TFLOPS = 96e12  # squares beyond 4096 (measured 96 at 4096^3)
VE_MS_PER_OP_2048 = 0.019  # one VectorE instruction over 2048^2 fp32
TWOSUM_OPS = 9
# host fp64 (#124/#125: torch fp64 on a trn2.3xlarge host): ms at measured
# shapes, else 75 / 415 GFLOP/s (2 / 12 vCPU, the 4096^3 rates)
HOST_MS = {
    (1024, 1024, 1024): (29, 5),
    (2048, 2048, 2048): (235, 46),
    (4096, 4096, 4096): (1829, 331),
    (3267, 8400, 241): (198, 35),
    (8192, 3508, 241): (206, 35),
}
HOST_GFLOPS = (75.0, 415.0)
# e2e overheads of the measured Ozaki I device route (#125): e2e - kernel
# ~= 1.8 ns per input element (raw-bit upload + device split) + 4.1 ns per
# output element (download + host combine of the fp32 (hi, lo) pair)
OZ1_NS_IN, OZ1_NS_OUT = 1.8, 4.1
USD_PER_S = 53.64 / 86400  # trn2.3xlarge, window rate


# ── data and references ──────────────────────────────────────────────────────


def draw(K, dist, seed):
    rng = np.random.default_rng([K, seed, DATA.index(dist)])
    A = rng.standard_normal((SAMPLE, K))
    B = rng.standard_normal((K, SAMPLE))
    if dist != "normal":
        e = float(dist[4:])
        A *= 10.0 ** rng.uniform(-e, e, A.shape)
        B *= 10.0 ** rng.uniform(-e, e, B.shape)
    return A, B


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def dd_add(h, lo, x):
    """(h, lo) + x for fp64 arrays, double-double."""
    s, e = two_sum(h, x)
    return two_sum(s, lo + e)


def _slices(X, axis, b, k):
    """fp64 slices of X: b bits each, per row (axis 1) / column (axis 0)."""
    amax = np.abs(X).max(axis=axis, keepdims=True)
    e = np.where(amax > 0, np.frexp(np.where(amax > 0, amax, 1.0))[1], 0)
    R = np.ldexp(X, -e)
    out = []
    for s in range(k):
        T = R * 2.0 ** ((b - 1) if s == 0 else b)
        D = np.rint(T)
        R = T - D
        out.append(np.ldexp(D, e - ((b - 1) + b * s)))
    return out


def ref_dd(A, B, bits=220):
    """Near-exact A @ B as a double-double (hi, lo): exact host slice GEMMs,
    dd-summed smallest first."""
    K = A.shape[1]
    b = math.floor((52 - math.log2(max(K, 2))) / 2)
    k = math.ceil(bits / b)
    As, Bs = _slices(A, 1, b, k), _slices(B, 0, b, k)
    h = lo = np.zeros((A.shape[0], B.shape[1]))
    for d in range(2 * k - 2, -1, -1):
        for s in range(max(0, d - k + 1), min(d, k - 1) + 1):
            h, lo = dd_add(h, lo, As[s] @ Bs[d - s])
    return h, lo


def bits_of(hi, lo, ref, den):
    err = np.abs((hi - ref[0]) + (lo - ref[1])) / den
    m = float(err.max())
    return 200.0 if m == 0 else -math.log2(m)


# ── the schemes, exactly ─────────────────────────────────────────────────────


def oz1_inherent(A, B, k):
    """Ozaki I's exact slice-product sum (b = 7 digits, s + t <= k - 1), dd."""

    def digits(X, axis):  # the split of `ozaki.split`, any k
        amax = np.abs(X).max(axis=axis, keepdims=True)
        e = np.frexp(amax)[1]
        R = np.ldexp(X, 6 - e)
        D = []
        for _ in range(k):
            d = np.rint(R)
            D.append(d)
            R = (R - d) * 128
        return D, e

    Da, ea = digits(A, 1)
    Db, eb = digits(B, 0)
    h = lo = np.zeros((A.shape[0], B.shape[1]))
    for d in range(k - 1, -1, -1):
        G = sum(Da[s] @ Db[d - s] for s in range(d + 1))  # exact integers
        h, lo = dd_add(h, lo, G * 2.0 ** (7 * (k - 1 - d)))
    sc = (ea - 12 - 7 * (k - 1)) + eb
    return np.ldexp(h, sc), np.ldexp(lo, sc)


def oz2_inherent(At, Bt, n, fam):
    """Ozaki II's exact CRT integer, unscaled, as a dd."""
    ra, rb = ozaki2.split_ab(At, Bt, n, fam)
    res = ozaki2.reference_residue_gemm(ra.d, rb.d, fam)[:, :SAMPLE, :SAMPLE]
    c = ozaki2.crt(fam, n)
    R = res.reshape(n, -1).T.astype(np.float64)
    Lq = R @ np.concatenate([c.hW, c.omega64[:, None]], axis=1)
    q = np.rint(Lq[:, -1])
    D = Lq[:, :-1] - q[:, None] * c.hM[None, :]
    sc = 2.0**c.hbeta
    for j in range(c.hn - 1):
        cy = np.rint(D[:, j] / sc)
        D[:, j] -= cy * sc
        D[:, j + 1] += cy
    h = lo = np.zeros(R.shape[0])
    for j in range(c.hn - 1, -1, -1):
        h, lo = dd_add(h, lo, D[:, j] * 2.0 ** (c.hbeta * j))
    e = -ra.e[:SAMPLE, None] - rb.e[None, :SAMPLE]
    return np.ldexp(h.reshape(SAMPLE, SAMPLE), e), np.ldexp(lo.reshape(SAMPLE, SAMPLE), e)


def run_case(K, dist, quick=False):
    t0 = time.time()
    A, B = draw(K, dist, 0)
    ref = ref_dd(A, B)
    den = np.abs(A) @ np.abs(B)
    den = np.where(den > 0, den, 1.0)
    At, Bt = torch.from_numpy(A), torch.from_numpy(B)
    zero = np.zeros_like(den)
    out = {"K": K, "data": dist, "host": bits_of(A @ B, zero, ref, den), "arms": {}}
    for arm, counts in ARMS.items():
        rows = {}
        for c in counts:
            if arm == "ozaki1":
                inh = bits_of(*oz1_inherent(A, B, c), ref, den)
                fp64 = (
                    bits_of(ozaki.reference_gemm(At, Bt, c).numpy(), zero, ref, den)
                    if c <= ozaki.MAX_SLICES
                    else None
                )
            else:
                if c > ozaki2.family(arm).max_moduli:
                    break
                inh = bits_of(*oz2_inherent(At, Bt, c, arm), ref, den)
                fp64 = None
                if ozaki2.family(arm).kernel and inh < 70:
                    C = ozaki2.reference_gemm(At, Bt, c, arm, "device").numpy()
                    fp64 = bits_of(C, zero, ref, den)
            rows[c] = {"inherent": inh, "fp64": fp64}
            if inh > 100 or (quick and inh > 60):
                break
        out["arms"][arm] = rows
    out["seconds"] = time.time() - t0
    print(f"K={K} {dist}: {out['seconds']:.0f} s", flush=True)
    return out


# ── costs ────────────────────────────────────────────────────────────────────


def bf16_eq(arm, c):
    if arm == "ozaki1":
        return c * (c + 1) // 2
    f = ozaki2.family(arm)
    return c * f.gemms_per_modulus / (2 if arm == "f8" else 1)  # FP8 double_row: 2x


def ms_per_gemm(shape):
    if shape in MS_PER_GEMM:
        return MS_PER_GEMM[shape]
    M, K, N = shape
    return 2 * M * K * N / BF16_TFLOPS * 1e3


def ve_ops(arm, c, K):
    """VectorE instructions per output element."""
    if arm == "ozaki1":
        return TWOSUM_OPS * ozaki.n_groups(c, -(-K // 128))
    if arm == "f8":
        # per modulus: 3 group closes of 4 per group, the Karatsuba / square
        # combine (3) and its reduction and canonical form (3 + 3), plus CRT
        f = ozaki2.family("f8")
        red = c * (3 * 4 * ozaki2.n_groups(K, f) + 9)
        return red + ozaki2.vector_ops(K, c, "f8")["crt"]
    v = ozaki2.vector_ops(K, c, arm)
    return v["reduce"] + v["crt"]


def model(shape, arm, c):
    M, K, N = shape
    te = bf16_eq(arm, c) * ms_per_gemm(shape)
    ve = ve_ops(arm, c, K) * VE_MS_PER_OP_2048 * (M * N / 2048**2)
    kern = max(te, ve)
    if arm == "ozaki1":
        over = (OZ1_NS_IN * (M * K + K * N) + OZ1_NS_OUT * M * N) * 1e-6
    else:  # assumed: a device residue split ~2x Ozaki I's, keep fp32 outputs
        keep = ozaki2.crt(arm, c).keep
        over = (1.25 + 2 * 0.55) * (M * K + K * N) * 1e-6 + OZ1_NS_OUT * keep / 2 * M * N * 1e-6
    return {"te_ms": te, "ve_ms": ve, "kernel_ms": kern, "e2e_ms": kern + over}


def host_ms(shape):
    if shape in HOST_MS:
        return HOST_MS[shape]
    M, K, N = shape
    return tuple(2 * M * K * N / (g * 1e9) * 1e3 for g in HOST_GFLOPS)


def needed(case, arm, bits):
    """Smallest count whose inherent result reaches `bits`."""
    for c, r in sorted(case["arms"][arm].items()):
        if r["inherent"] >= bits:
            return c
    return None


def _fmt_count(case, arm, bits):
    c = needed(case, arm, bits)
    if c is None:
        return "-"
    fp = case["arms"][arm][c]["fp64"]
    fps = f", fp64 out {fp:.0f}" if fp is not None and bits >= 48 else ""
    return f"{c} ({bf16_eq(arm, c):g}{fps})"


def crossover(arm, bits, dist, byk, host_idx):
    """Smallest square (of SQUARES) whose modelled e2e beats the host."""
    for n in SQUARES:
        case = byk.get((n, dist))
        c = needed(case, arm, bits) if case else None
        if c is None:
            continue
        if model((n, n, n), arm, c)["e2e_ms"] < host_ms((n, n, n))[host_idx]:
            return n
    return None


def tables(cases):
    byk = {(c["K"], c["data"]): c for c in cases}
    arms = list(ARMS)
    out = ["### 1. Count needed per target (bf16-equivalent GEMMs in brackets)\n"]
    out.append(
        "Smallest k (Ozaki I) or moduli count n (Ozaki II) whose inherent result reaches the "
        "target, max-componentwise bits = -log2(max |C - C_exact| / (|A||B|)). `fp64 out` is "
        "the bits of the fp64 result the device route returns at that count.\n"
    )
    out.append("| K | data | host fp64 bits | bits | " + " | ".join(arms) + " |")
    out.append("|---" * (4 + len(arms)) + "|")
    for case in cases:
        for bits in TARGET_BITS:
            cells = [_fmt_count(case, a, bits) for a in arms]
            out.append(
                f"| {case['K']} | {case['data']} | {case['host']:.1f} | {bits} | "
                + " | ".join(cells)
                + " |"
            )
    out.append("\n### 2. Modelled kernel and e2e time per GEMM, normal data\n")
    out.append(
        "kernel = max(TensorE, VectorE); TE in bf16-equivalent time (measured Ozaki I ms per "
        "slice GEMM at that shape); `*` marks a VectorE-bound kernel. e2e adds the measured "
        "Ozaki I upload / split / download / combine overheads (Ozaki II: an assumed device "
        "residue split at twice Ozaki I's split cost, and `keep` fp32 outputs).\n"
    )
    out.append(
        "| shape | bits | "
        + " | ".join(f"{a} kernel / e2e ms" for a in arms)
        + " | host 2 / 12 vCPU ms | best | best $/GEMM |"
    )
    out.append("|---" * (5 + len(arms)) + "|")
    shapes = [(n, n, n) for n in SQUARES] + list(NARROW)
    for shape in shapes:
        case = byk.get((shape[1], "normal"))
        if case is None:
            continue
        h2, h12 = host_ms(shape)
        for bits in TARGET_BITS:
            cells, best = [], None
            for a in arms:
                c = needed(case, a, bits)
                if c is None:
                    cells.append("-")
                    continue
                m = model(shape, a, c)
                star = "*" if m["ve_ms"] > m["te_ms"] else ""
                cells.append(f"{m['kernel_ms']:.2f}{star} / {m['e2e_ms']:.1f}")
                if best is None or m["e2e_ms"] < best[1]["e2e_ms"]:
                    best = (a, m)
            usd = f"{best[1]['e2e_ms'] * 1e-3 * USD_PER_S:.1e}" if best else "-"
            out.append(
                f"| {'x'.join(map(str, shape))} | {bits} | "
                + " | ".join(cells)
                + f" | {h2:.0f} / {h12:.0f} | {best[0] if best else '-'} | {usd} |"
            )
    out.append("\n### 3. Crossover against host fp64 (smallest square that wins, modelled e2e)\n")
    out.append("| data | bits | " + " | ".join(f"{a} vs 2 / 12 vCPU" for a in arms) + " |")
    out.append("|---" * (2 + len(arms)) + "|")
    for dist in DATA:
        for bits in TARGET_BITS:
            cells = []
            for a in arms:
                x2, x12 = (crossover(a, bits, dist, byk, i) for i in (0, 1))
                cells.append(f"{x2 or '-'} / {x12 or '-'}")
            out.append(f"| {dist} | {bits} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ozaki2_analysis")
    ap.add_argument("--Ks", nargs="*", type=int, default=list(KS))
    ap.add_argument("--data", nargs="*", default=list(DATA))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for K in a.Ks:
        for dist in a.data:
            cases.append(run_case(K, dist, a.quick))
            (out / "ozaki2_analysis.json").write_text(json.dumps(cases, indent=1))
    md = tables(cases)
    (out / "ozaki2_analysis.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
