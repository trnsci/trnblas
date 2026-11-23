#!/usr/bin/env python
"""Accuracy table of `trnblas.fp64` on CPU (trnsci/trnsci#169).

Every arm runs through the CPU reference (`backend="pytorch"`), which is the
device's arithmetic op for op, so the table is also the device's accuracy if
the E0 probe passes. The reference is mpmath at `--prec` bits (cross-checked
against the exactly rounded `math.fsum` of exact terms). The figure is the
number of correct bits, -log2(|r - ref| / |ref|), capped at 53.

Arms on fp32-resident data (the device's working vectors):
  fp32        plain fp32 lane sums + halving fold (exact host finish of the leaves)
  comp        Ogita-Rump-Oishi Sum2 / Dot2 (TwoSum compensation)
  dw-sloppy   double-word lanes, SloppyDWPlusDW
  dw-accurate double-word lanes, AccurateDWPlusDW (the default)
  host fp64   numpy float64 (np.sum / np.dot) of the same data
  exact-int   integer-emulated exact path (`fp64.exact_sum` / `exact_dot`)
Arms on float64 data: the operands enter as float-float (`from_fp64`), so the
representation alone costs up to 2^-48 per element.

    python scripts/fp64_accuracy.py --json acc.json [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction

import numpy as np
import torch

from trnblas import fp64 as F

try:
    import mpmath
except ImportError:  # pragma: no cover
    mpmath = None


def bits(r: float, ref) -> float:
    if mpmath is not None and isinstance(ref, mpmath.mpf):
        if ref == 0:
            return 53.0 if r == 0 else 0.0
        if not math.isfinite(r):
            return 0.0
        rel = abs((mpmath.mpf(r) - ref) / ref)
        if rel == 0:
            return 53.0
        return float(max(0.0, min(53.0, -mpmath.log(rel, 2))))
    ref = float(ref)
    if ref == 0:
        return 53.0 if r == 0 else 0.0
    if not math.isfinite(r):
        return 0.0
    rel = abs(r - ref) / abs(ref)
    return 53.0 if rel == 0 else max(0.0, min(53.0, -math.log2(rel)))


def two_prod64(x: np.ndarray, y: np.ndarray):
    """Exact fp64 products as p + e (Dekker, numpy float64)."""
    p = x * y
    c = 134217729.0 * x
    xh = c - (c - x)
    xl = x - xh
    c = 134217729.0 * y
    yh = c - (c - y)
    yl = y - yh
    e = xl * yl - (((p - xh * yh) - xl * yh) - xh * yl)
    return p, e


def reference(kind: str, x: np.ndarray, y: np.ndarray | None, prec: int):
    """(mpmath value, exactly rounded fp64 cross-check)."""
    if kind == "sum":
        cr = math.fsum(x.tolist())
    else:
        p, e = two_prod64(x, y)
        cr = math.fsum(np.concatenate([p, e]).tolist())
    if mpmath is None:
        return cr, cr
    with mpmath.workprec(prec):
        if kind == "sum":
            v = mpmath.fsum(mpmath.mpf(float(t)) for t in x.tolist())
        else:
            v = mpmath.fdot(
                (mpmath.mpf(float(a)) for a in x.tolist()),
                (mpmath.mpf(float(b)) for b in y.tolist()),
            )
        assert float(v) == cr or abs(float(v) - cr) <= abs(cr) * 2**-52, (float(v), cr)
    return v, cr


# ── data ─────────────────────────────────────────────────────────────────────


def cg_vectors(n: int, iters: int = 25, seed: int = 0):
    """r_k, r_(k-1), p_k, A p_k of CG on a shifted 1-D Poisson matrix (fp64)."""
    rng = np.random.default_rng(seed)
    sigma = 1e-3

    def A(v):
        w = (2 + sigma) * v
        w[1:] -= v[:-1]
        w[:-1] -= v[1:]
        return w

    b = rng.standard_normal(n)
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    rr = r @ r
    r_prev = r
    for _ in range(iters):
        Ap = A(p)
        alpha = rr / (p @ Ap)
        x += alpha * p
        r_prev = r
        r = r - alpha * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
    return {"r": r, "r_prev": r_prev, "p": p, "Ap": A(p)}


def gendot(n: int, cond: float, seed: int = 0, dtype=np.float64):
    """Ogita, Rump and Oishi (2005), Algorithm 6.1: x, y with condition
    number about `cond` (2 sum|x y| / |x . y|), in `dtype`."""
    rng = np.random.default_rng(seed)
    cast = (lambda v: float(np.float32(v))) if dtype == np.float32 else float
    b = math.log2(cond)
    n2 = n // 2
    e = np.round(rng.uniform(0, 1, n2) * b / 2).astype(int)
    e[0] = round(b / 2) + 1
    e[-1] = 0
    x = [cast(v) for v in (2 * rng.uniform(0, 1, n2) - 1) * np.exp2(e)]
    y = [cast(v) for v in (2 * rng.uniform(0, 1, n2) - 1) * np.exp2(e)]
    acc = sum((Fraction(a) * Fraction(c) for a, c in zip(x, y, strict=True)), Fraction(0))
    ee = np.round(np.linspace(b / 2, 0, n - n2)).astype(int)
    for i in range(n - n2):
        xi = cast((2 * rng.uniform() - 1) * 2.0 ** ee[i])
        if xi == 0:
            xi = cast(2.0 ** ee[i])
        yi = cast(((2 * rng.uniform() - 1) * 2.0 ** ee[i] - float(acc)) / xi)
        x.append(xi)
        y.append(yi)
        acc += Fraction(xi) * Fraction(yi)
    perm = rng.permutation(n)
    x = np.array(x, dtype=np.float64)[perm]
    y = np.array(y, dtype=np.float64)[perm]
    return x, y


def achieved_cond(x, y) -> float:
    p, e = two_prod64(x, y)
    d = math.fsum(np.concatenate([p, e]).tolist())
    return 2 * float(np.abs(x * y).sum()) / abs(d) if d else math.inf


def xc_quadrature(n: int, seed: int = 0):
    """Weights and integrand of a DFT exchange-correlation grid sum
    E_x = sum_i w_i f(rho_i): radial Mura-Knowles-like grids around a few
    atoms, w_i = 4 pi r^2 dr (spanning ~1e-14 to 1e3), rho a sum of
    Gaussians, f = LDA exchange -C rho^(4/3)."""
    rng = np.random.default_rng(seed)
    m = n
    q = (np.arange(m) + 0.5) / m
    R = 2.0
    r = -R * np.log(1 - q**3)  # Mura-Knowles radial map
    dr = R * 3 * q**2 / (1 - q**3) / m
    w = 4 * np.pi * r**2 * dr
    alphas = np.array([1e3, 50.0, 3.0, 0.3])
    coef = np.array([30.0, 5.0, 1.0, 0.2])
    rho = (coef[None, :] * np.exp(-alphas[None, :] * r[:, None] ** 2)).sum(axis=1)
    rho = rho * (1 + 1e-3 * rng.standard_normal(m))
    f = -0.7386 * np.abs(rho) ** (4.0 / 3.0)
    return w, f


def pair_energies(n: int, seed: int = 0):
    """DF-MP2-style pair energies: n same-sign contributions, log-normal
    magnitudes over ~8 decades (condition number 1, long accumulation)."""
    rng = np.random.default_rng(seed)
    return -np.exp(rng.normal(-9.0, 2.5, n))


def energy_terms(n: int, seed: int = 0, cancel: float = 1e-3):
    """A long total-energy sum with cancellation: 90% negative and 10%
    positive log-normal contributions, the positive ones scaled so that the
    total is `cancel` times the negative part (condition number ~ 2 / cancel,
    as when large attractive and repulsive pieces cancel)."""
    rng = np.random.default_rng(seed)
    neg = -np.exp(rng.normal(-9.0, 2.5, n - n // 10))
    pos = np.exp(rng.normal(-7.0, 2.5, n // 10))
    pos *= (1 - cancel) * math.fsum((-neg).tolist()) / math.fsum(pos.tolist())
    e = np.concatenate([neg, pos])
    return e[rng.permutation(n)]


def cond_of(kind, x, y=None) -> float:
    if kind == "sum":
        t = math.fsum(x.tolist())
        return float(np.abs(x).sum()) / abs(t) if t else math.inf
    return achieved_cond(x, y) / 2


# ── arms ─────────────────────────────────────────────────────────────────────

ARMS32 = [
    ("fp32", "plain"),
    ("comp", "comp"),
    ("dw-sloppy", "sloppy"),
    ("dw-accurate", "accurate"),
]


def run_arms(kind, x64, y64, prec, data32=True, data64=True):
    """Correct bits per arm; x64 / y64 are float64 arrays. Also records the
    condition number sum|t_i| / |sum t_i| of each data kind and the exact
    path's limb counts (its cost driver)."""
    out = {}
    if data32:
        x32 = x64.astype(np.float32)
        y32 = None if y64 is None else y64.astype(np.float32)
        xd = x32.astype(np.float64)
        yd = None if y32 is None else y32.astype(np.float64)
        ref, _ = reference(kind, xd, yd, prec)
        tx = torch.from_numpy(x32)
        ty = None if y32 is None else torch.from_numpy(y32)
        for label, arm in ARMS32:
            r = F.sum(tx, accumulate=arm) if kind == "sum" else F.dot(tx, ty, accumulate=arm)
            out[f"fp32 data: {label}"] = bits(float(r), ref)
        r = float(np.sum(xd)) if kind == "sum" else float(np.dot(xd, yd))
        out["fp32 data: host fp64"] = bits(r, ref)
        if kind == "sum":
            r = F.exact_sum(torch.from_numpy(xd))
        else:
            r = F.exact_dot(torch.from_numpy(xd), torch.from_numpy(yd))
        out["fp32 data: exact-int"] = bits(float(r), ref)
        out["cond (fp32 data)"] = cond_of(kind, xd, yd)
    if data64:
        ref, _ = reference(kind, x64, y64, prec)
        tx = torch.from_numpy(x64)
        ty = None if y64 is None else torch.from_numpy(y64)
        for label, arm in (("dw-sloppy", "sloppy"), ("dw-accurate", "accurate")):
            if kind == "sum" and arm == "sloppy":
                continue
            try:
                r = F.sum(tx, accumulate=arm) if kind == "sum" else F.dot(tx, ty, accumulate=arm)
                r = float(r)
            except OverflowError:
                r = math.nan
            out[f"fp64 data: {label}"] = bits(r, ref)
        r = float(np.sum(x64)) if kind == "sum" else float(np.dot(x64, y64))
        out["fp64 data: host fp64"] = bits(r, ref)
        r = F.exact_sum(tx) if kind == "sum" else F.exact_dot(tx, ty)
        out["fp64 data: exact-int"] = bits(float(r), ref)
        out["cond (fp64 data)"] = cond_of(kind, x64, y64)
        if kind == "sum":
            out["int limbs"] = f"{F.int_limbs_needed(x64)} x 21 bit"
        else:
            lx = F.int_limbs_needed(x64, F.DOT_LIMB_BITS)
            ly = F.int_limbs_needed(y64, F.DOT_LIMB_BITS)
            out["int limbs"] = f"{lx} x {ly} products of 12 bit"
    return out


def combine_rows(prec):
    """gemm_fp64-combine-shaped partials: 8 K blocks of Ozaki (hi, lo)."""
    from trnblas import ozaki

    rng = np.random.default_rng(7)
    rows = {}
    for label, scale in (("normal data", 0.0), ("entries spread 1e+-4", 4.0)):
        M, K, N, kb, k = 48, 4096, 40, 8, 8
        A = rng.standard_normal((M, K)) * 10.0 ** rng.uniform(-scale, scale, (M, K))
        B = rng.standard_normal((K, N)) * 10.0 ** rng.uniform(-scale, scale, (K, N))
        At, Bt = torch.from_numpy(A), torch.from_numpy(B)
        da, db = ozaki.split_a(At, k), ozaki.split_b(Bt, k)
        at, bt = ozaki.kmajor(da.d, da.kmajor), ozaki.kmajor(db.d, db.kmajor)
        step = at.shape[1] // kb
        parts = [
            F.stacked(
                torch.from_numpy(
                    ozaki.reference_from_digits(at[:, i : i + step], bt[:, i : i + step])
                )[:, :M, :N]
            )
            for i in range(0, at.shape[1], step)
        ]
        exp = da.e[:M, None] + db.e[None, :N] + ozaki.exponent_offset(k)
        # reference: the exact sum of the partials (what any combine should return)
        vals = np.stack(
            [np.ldexp(p.hi.double().numpy(), exp) for p in parts]
            + [np.ldexp(p.lo.double().numpy(), exp) for p in parts]
        )
        ref = np.array(
            [math.fsum(vals[:, i, j].tolist()) for i in range(M) for j in range(N)]
        ).reshape(M, N)
        host = sum(F.to_fp64(p, exp) for p in parts).numpy()
        ff = F.combine(parts, exp=exp).numpy()
        ex = F.combine_exact(parts, exp=exp).numpy()

        def stats(r, ref=ref):
            b = [bits(float(a), float(c)) for a, c in zip(r.ravel(), ref.ravel(), strict=True)]
            return float(np.min(b)), float(np.median(b))

        rows[f"combine, {label}"] = {
            "host fp64 (today)": stats(host),
            "dw-accurate combine": stats(ff),
            "exact-int combine": stats(ex),
        }
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None)
    ap.add_argument("--prec", type=int, default=400)
    ap.add_argument("--quick", action="store_true", help="smaller sizes (CI)")
    args = ap.parse_args(argv)
    t0 = time.time()
    table = {}
    sizes = [10**4, 10**5] if args.quick else [10**5, 10**6]
    for n in sizes:
        v = cg_vectors(n)
        table[f"CG r.r, n={n:.0e}"] = run_arms("dot", v["r"], v["r"], args.prec)
        table[f"CG p.Ap, n={n:.0e}"] = run_arms("dot", v["p"], v["Ap"], args.prec)
        table[f"CG r_k.r_k-1 (near-orthogonal, a Gram-Schmidt dot), n={n:.0e}"] = run_arms(
            "dot", v["r"], v["r_prev"], args.prec
        )
        table[f"CG sum(r), n={n:.0e}"] = run_arms("sum", v["r"], None, args.prec)
    nd = 2000 if args.quick else 10**4
    for cond in (1e4, 1e8, 1e16, 1e24, 1e30):
        x, y = gendot(nd, cond, seed=int(math.log10(cond)))
        c = achieved_cond(x, y)
        row = run_arms("dot", x, y, args.prec, data32=False)
        x32, y32 = gendot(nd, cond, seed=int(math.log10(cond)), dtype=np.float32)
        row.update(run_arms("dot", x32, y32, args.prec, data64=False))
        row["cond (fp64 data)"] = c / 2
        table[f"GenDot cond {cond:.0e}, n={nd}"] = row
    rng = np.random.default_rng(3)
    nw = 10**5 if args.quick else 10**6
    xw = rng.standard_normal(nw) * np.exp2(rng.integers(-60, 60, nw))
    yw = rng.standard_normal(nw) * np.exp2(rng.integers(-60, 60, nw))
    table[f"wide range 2^+-60 sum, n={nw:.0e}"] = run_arms("sum", xw, None, args.prec)
    table[f"wide range 2^+-60 dot, n={nw:.0e}"] = run_arms("dot", xw, yw, args.prec)
    xo = rng.standard_normal(10**4) * np.exp2(rng.integers(-200, 200, 10**4))
    table["beyond fp32 range 2^+-200 sum, n=1e4"] = run_arms(
        "sum", xo, None, args.prec, data32=False
    )
    w, f = xc_quadrature(nw)
    table[f"XC quadrature sum w f(rho), n={nw:.0e}"] = run_arms("dot", w, f, args.prec)
    ne = 10**6 if args.quick else 10**7
    table[f"pair-energy sum (one sign), n={ne:.0e}"] = run_arms(
        "sum", pair_energies(ne), None, args.prec
    )
    table[f"energy sum, cancellation 1e-3, n={ne:.0e}"] = run_arms(
        "sum", energy_terms(ne), None, args.prec
    )
    comb = combine_rows(args.prec)
    rep = {"table": table, "combine": comb, "seconds": time.time() - t0, "prec": args.prec}
    print(format_markdown(rep))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    return rep


def format_markdown(rep) -> str:
    cols = [
        "fp32 data: fp32",
        "fp32 data: comp",
        "fp32 data: dw-sloppy",
        "fp32 data: dw-accurate",
        "fp32 data: host fp64",
        "fp32 data: exact-int",
        "fp64 data: dw-sloppy",
        "fp64 data: dw-accurate",
        "fp64 data: host fp64",
        "fp64 data: exact-int",
    ]
    hdr = ["case", "cond"] + [c.replace("data: ", "") for c in cols] + ["exact-int limbs"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for case, row in rep["table"].items():
        cond = row.get("cond (fp64 data)", row.get("cond (fp32 data)"))
        cells = [case, f"{cond:.1e}" if cond is not None else ""]
        cells += [f"{row[c]:.1f}" if c in row else "" for c in cols]
        cells.append(str(row.get("int limbs", "")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "| combine (8 K blocks, per element) | host fp64 (today) | dw-accurate | exact-int |"
    )
    lines.append("|---|---|---|---|")
    for case, row in rep["combine"].items():
        lines.append(
            f"| {case} (min / median bits) | "
            + " | ".join(
                "{:.1f} / {:.1f}".format(*row[k])
                for k in ("host fp64 (today)", "dw-accurate combine", "exact-int combine")
            )
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
