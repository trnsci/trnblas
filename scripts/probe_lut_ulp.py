#!/usr/bin/env python
"""Measured accuracy of the ScalarE activation-table (LUT) functions on Trn2
(trnsci/trnsci#173), against the GpSimd and VectorE alternatives and Newton-
refined variants.

One kernel evaluates every variant on a (128, F) fp32 tile: ScalarE
`activation` sqrt / rsqrt / reciprocal / exp / log; GpSimd
`tensor_scalar(op0=rsqrt)` and `tensor_tensor(op=power)`; VectorE
`nisa.reciprocal` (VectorE `tensor_tensor` divide fails trn2 ISA validation, NCC_IINAR001); and one or two VectorE Newton
steps after the rsqrt / reciprocal LUT (plus one Heron step after the sqrt
LUT). The host runs it once per function on inputs covering that function's
valid range (log-uniform over the whole range, a dense band near 1 and the
range edges), and measures each row against the float64 result: ULP error
|got - f(x)| / ulp(RN32(f(x))), relative error, and the fraction that is
correctly rounded.

    python -P scripts/probe_lut_ulp.py verify               # MLIR verify, LNC=1 and 2
    python -P scripts/probe_lut_ulp.py run --json lut.json  # on a neuron device
    python -P scripts/probe_lut_ulp.py run --simulate       # nki.simulate dry run

Ranges are the ScalarE valid input ranges of the NKI 0.7 activation table:
sqrt [2^-116, 2^118], rsqrt [2^-87, 2^97], reciprocal +-[2^-42, 2^42],
log [2^-64, 2^64]. exp has no documented range; it is probed where the result
is a normal fp32, x in [-87.3, 88.7].
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

P = 128
TW = 512  # columns per tile
F = 2048  # columns per launch (4 tiles; at LNC=2 each program takes 2)
F32 = np.float32

# (function, implementation, input-range restriction or None)
ROWS = [
    ("sqrt", "ScalarE activation sqrt", None),
    ("rsqrt", "ScalarE activation rsqrt", None),
    ("reciprocal", "ScalarE activation reciprocal", None),
    ("exp", "ScalarE activation exp", None),
    ("log", "ScalarE activation log", None),
    ("rsqrt", "GpSimd tensor_scalar(op0=rsqrt)", None),
    ("reciprocal", "VectorE nisa.reciprocal", None),
    ("rsqrt", "ScalarE rsqrt + 1 VectorE Newton step", None),
    ("rsqrt", "ScalarE rsqrt + 2 VectorE Newton steps", None),
    ("sqrt", "x * (ScalarE rsqrt + 1 Newton step)", "rsqrt"),
    ("sqrt", "x * (ScalarE rsqrt + 2 Newton steps)", "rsqrt"),
    ("sqrt", "ScalarE sqrt + 1 Heron step (VectorE reciprocal)", None),
    ("reciprocal", "ScalarE reciprocal + 1 VectorE Newton step", None),
    ("rsqrt", "GpSimd rsqrt + 1 VectorE Newton step", None),
    ("sqrt", "x * GpSimd rsqrt", None),
    ("sqrt", "GpSimd tensor_tensor power x^0.5", None),
    ("rsqrt", "GpSimd tensor_tensor power x^-0.5", None),
]
NOUT = len(ROWS)

# documented ScalarE ranges and ULP bounds (NKI activation-function table)
RANGES = {
    "sqrt": (2.0**-116, 2.0**118),
    "rsqrt": (2.0**-87, 2.0**97),
    "reciprocal": (2.0**-42, 2.0**42),
    "log": (2.0**-64, 2.0**64),
    "exp": (-87.3, 88.7),
}
DOC_ULP = {"sqrt": 65536, "rsqrt": 40000, "reciprocal": 400, "exp": 400, "log": 40}


if HAS_NKI:

    def _t(shape=(P, TW)):
        return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)

    def _tt(a, b, op, engine=None):
        r = _t()
        if engine is None:
            nisa.tensor_tensor(dst=r, data1=a, data2=b, op=op, engine=nisa.vector_engine)
        else:
            nisa.tensor_tensor(dst=r, data1=a, data2=b, op=op, engine=engine)
        return r

    def _act(x, op):
        r = _t()
        nisa.activation(dst=r, op=op, data=x)
        return r

    def _ts2(x, m, c):
        """VectorE x * m + c (two roundings; E0 found no FMA here)."""
        r = _t()
        nisa.tensor_scalar(
            dst=r,
            data=x,
            op0=nl.multiply,
            operand0=m,
            op1=nl.add,
            operand1=c,
            engine=nisa.vector_engine,
        )
        return r

    def _newton_rsqrt(x, y):
        """y (1.5 - 0.5 x y^2), all on VectorE."""
        t = _tt(y, y, nl.multiply)
        t = _tt(t, x, nl.multiply)
        t = _ts2(t, -0.5, 1.5)
        return _tt(y, t, nl.multiply)

    def _lut_tile(x, out, c0):
        xt = nl.load(x[0:P, c0 : c0 + TW])
        half = _t()
        nisa.memset(dst=half, value=0.5)
        mhalf = _t()
        nisa.memset(dst=mhalf, value=-0.5)
        rows = []
        s_lut = _act(xt, nl.sqrt)
        r_lut = _act(xt, nl.rsqrt)
        q_lut = _act(xt, nl.reciprocal)
        rows += [s_lut, r_lut, q_lut, _act(xt, nl.exp), _act(xt, nl.log)]
        g = _t()
        nisa.tensor_scalar(dst=g, data=xt, op0=nl.rsqrt, operand0=0, engine=nisa.gpsimd_engine)
        rows.append(g)
        vr = _t()
        nisa.reciprocal(dst=vr, data=xt)
        rows.append(vr)
        r1 = _newton_rsqrt(xt, r_lut)
        r2 = _newton_rsqrt(xt, r1)
        rows += [r1, r2, _tt(xt, r1, nl.multiply), _tt(xt, r2, nl.multiply)]
        # Heron: 0.5 (s + x / s), the divide as a VectorE reciprocal
        rs = _t()
        nisa.reciprocal(dst=rs, data=s_lut)
        h = _tt(s_lut, _tt(xt, rs, nl.multiply), nl.add)
        rows.append(_tt(h, half, nl.multiply))
        # reciprocal Newton: q (2 - x q)
        t = _ts2(_tt(xt, q_lut, nl.multiply), -1.0, 2.0)
        rows.append(_tt(q_lut, t, nl.multiply))
        rows += [_newton_rsqrt(xt, g), _tt(xt, g, nl.multiply)]
        rows.append(_tt(xt, half, nl.power, engine=nisa.gpsimd_engine))
        rows.append(_tt(xt, mhalf, nl.power, engine=nisa.gpsimd_engine))
        for i, r in enumerate(rows):
            nl.store(out[i, 0:P, c0 : c0 + TW], value=r)

    @nki.jit
    def _lut_probe_kernel(x):
        """(NOUT, 128, F) fp32: every ROWS variant of x (128, F), F a multiple
        of TW. Gridless: at LNC=2 the programs split the column tiles."""
        nt = x.shape[1] // TW
        out = nl.ndarray((NOUT, P, x.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        per = nt // n_prog
        if n_prog > 1:
            for j in nl.affine_range(per):
                _lut_tile(x, out, (nl.program_id(0) * per + j) * TW)
        else:
            for j in nl.affine_range(nt):
                _lut_tile(x, out, j * TW)
        return out


# ── inputs ──────────────────────────────────────────────────────────────────


def _logu(rng, lo, hi, n):
    """Random fp32 log-uniform in [lo, hi): uniform exponent, random 23-bit mantissa."""
    e = rng.integers(int(np.floor(np.log2(lo))), int(np.floor(np.log2(hi))), size=n)
    m = 1.0 + rng.integers(0, 1 << 23, size=n) * 2.0**-23
    v = np.ldexp(m, e)
    return np.clip(v, lo, np.nextafter(hi, 0))


def inputs(fn: str, seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(128, F) fp32 inputs for `fn` and named masks of input subsets."""
    rng = np.random.default_rng(1000 * seed + list(RANGES).index(fn))
    n = P * F
    q = n // 8
    lo, hi = RANGES[fn]
    if fn == "exp":
        parts = [
            rng.uniform(lo, hi, size=4 * q),  # the whole range
            rng.uniform(-1.0, 1.0, size=2 * q),
            rng.uniform(-(2.0**-10), 2.0**-10, size=q),
            rng.uniform(lo, lo + 2, size=q // 2),  # edges
            rng.uniform(hi - 2, hi, size=q - q // 2),
        ]
    else:
        band = (0.5, 2.0) if fn == "log" else (1.0, 4.0)  # two binades
        parts = [
            _logu(rng, lo, hi, 4 * q),  # the whole range, log-uniform
            rng.uniform(*band, size=2 * q),
            _logu(rng, lo, 4 * lo, q // 2),  # edges: bottom and top two binades
            _logu(rng, hi / 4, hi, q - q // 2),
            rng.uniform(0.9, 1.1, size=q) if fn == "log" else rng.uniform(*band, size=q),
        ]
    x = np.concatenate(parts).astype(F32)
    if fn == "reciprocal":
        x = x * rng.choice([-1.0, 1.0], size=n).astype(F32)
    x = x.reshape(P, F)
    flat = np.abs(x.reshape(-1)).astype(np.float64)
    masks = {"full range": np.ones(n, bool)}
    if fn == "log":
        masks["away from 1 (x outside [0.5, 2])"] = (flat < 0.5) | (flat > 2.0)
        masks["near 1 (x in [0.9, 1.1])"] = (flat >= 0.9) & (flat <= 1.1)
    elif fn == "exp":
        masks["|x| <= 1"] = flat <= 1.0
    else:
        masks["edges (bottom and top two binades)"] = (flat < 4 * lo) | (flat >= hi / 4)
    return x, masks


def reference(fn: str, x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    with np.errstate(all="ignore"):
        return {
            "sqrt": np.sqrt,
            "rsqrt": lambda v: 1.0 / np.sqrt(v),
            "reciprocal": lambda v: 1.0 / v,
            "exp": np.exp,
            "log": np.log,
        }[fn](x64)


def stats(got: np.ndarray, ref: np.ndarray, x: np.ndarray, mask: np.ndarray) -> dict:
    g = got.reshape(-1)[mask].astype(np.float64)
    r = ref.reshape(-1)[mask]
    xs = x.reshape(-1)[mask]
    rn = r.astype(F32)
    with np.errstate(all="ignore"):
        ulp = np.spacing(np.abs(rn)).astype(np.float64)
        ulp = np.where(ulp == 0, np.spacing(F32(0)), ulp)
        e_ulp = np.abs(g - r) / ulp
        e_rel = np.abs(g - r) / np.abs(r)
    finite = np.isfinite(g)
    e_ulp = np.where(finite, e_ulp, np.inf)
    e_rel = np.where(finite, e_rel, np.inf)
    i = int(np.argmax(np.where(np.isnan(e_ulp), np.inf, e_ulp)))
    return {
        "n": int(mask.sum()),
        "max_ulp": float(np.max(e_ulp)),
        "p999_ulp": float(np.quantile(e_ulp, 0.999)),
        "median_ulp": float(np.median(e_ulp)),
        "max_rel": float(np.max(np.where(r == 0, 0.0, e_rel))),
        "correctly_rounded": float(np.mean(got.reshape(-1)[mask] == rn)),
        "nonfinite": int((~finite).sum()),
        "worst": {"x": float(xs[i]), "got": float(g[i]), "f(x)": float(r[i])},
    }


def run(launch, seed: int = 0) -> dict:
    rep = {"rows": {}, "ranges": {k: list(v) for k, v in RANGES.items()}, "doc_ulp": DOC_ULP}
    for fn in RANGES:
        x, masks = inputs(fn, seed)
        t0 = time.perf_counter()
        out = launch(x)
        dt = time.perf_counter() - t0
        out2 = launch(x)
        rep.setdefault("launch_s", {})[fn] = dt
        ref = reference(fn, x)
        for k, (f, impl, restrict) in enumerate(ROWS):
            if f != fn:
                continue
            m_restrict = None
            if restrict is not None:
                lo, hi = RANGES[restrict]
                ax = np.abs(x.reshape(-1)).astype(np.float64)
                m_restrict = (ax >= lo) & (ax < hi)
            row = {"function": fn, "impl": impl, "subsets": {}}
            row["repeat_bitwise_equal"] = bool(
                (out[k].view(np.uint32) == out2[k].view(np.uint32)).all()
            )
            for name, m in masks.items():
                mm = m if m_restrict is None else (m & m_restrict)
                if mm.sum():
                    row["subsets"][name] = stats(out[k], ref, x, mm)
            if restrict is not None:
                row["restricted_to"] = f"{restrict} range"
            rep["rows"][f"{k:02d} {impl}"] = row
    return rep


def format_table(rep: dict) -> str:
    lines = [
        "| function | implementation | subset | n | max ULP | p99.9 ULP | median ULP | max rel err | correctly rounded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fn in RANGES:
        for row in rep["rows"].values():
            if row["function"] != fn:
                continue
            for name, s in row["subsets"].items():
                lines.append(
                    f"| {fn} | {row['impl']} | {name} | {s['n']} | {s['max_ulp']:.4g} | "
                    f"{s['p999_ulp']:.3g} | {s['median_ulp']:.3g} | {s['max_rel']:.3g} | "
                    f"{100 * s['correctly_rounded']:.1f}% |"
                )
    return "\n".join(lines)


def _verify():
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    x = np.ones((P, F), F32)
    for lnc in (1, 2):
        source_to_mlir(
            _lut_probe_kernel, x, target="trn2", lnc=lnc, frontend=FrontendKind.TRACER
        ).verify()
        print(f"verify lnc={lnc}: ok")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    pr = sub.add_parser("run")
    pr.add_argument("--json", default=None)
    pr.add_argument("--simulate", action="store_true")
    pr.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        _verify()
        return 0
    if args.simulate:

        def launch(x):
            return np.asarray(nki.simulate(_lut_probe_kernel)(x))

    else:
        import torch
        import torch_neuronx

        h = torch_neuronx.wrap_nki(_lut_probe_kernel)

        def launch(x):
            t = torch.from_numpy(np.ascontiguousarray(x)).to("neuron")
            return h(t).to("cpu").numpy()

    rep = run(launch, args.seed)
    print(format_table(rep))
    for fn, dt in rep["launch_s"].items():
        print(f"first launch for {fn}: {dt:.2f} s")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
