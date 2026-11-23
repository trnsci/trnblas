"""
E0 device probe of trnsci/trnsci#169: does each engine round fp32 add and
multiply to nearest even, does a fused multiply-add exist (single rounding),
and how does TensorE + PSUM multiply and accumulate fp32?

Everything in `trnblas.fp64` assumes round-to-nearest-even fp32 add,
subtract and multiply with gradual underflow, and no fusion inside the
instructions it issues. This module builds adversarial inputs (rounding
ties, near-overflow, subnormals, cancellation, wide random), runs the probe
kernels of `fp64_kernels.py` through a caller-supplied `run(name, *arrays)`,
and compares every output with IEEE results computed in numpy (float32
arithmetic in numpy is IEEE round-to-nearest-even with gradual underflow).
`run_e0` returns a JSON-able dict: a pass / fail table per engine and
identity, with mismatch counts and a classification of what the mismatches
look like (truncation, flush to zero, single rounding, ...).

`ideal_run` emulates an ideal IEEE device in numpy (every probe passes);
the CPU tests run the probe logic against it and against `nki.simulate`.
"""

from __future__ import annotations

import time

import numpy as np

W = 512  # probe tile width
F32 = np.float32


def _f32(a):
    return np.ascontiguousarray(a, dtype=F32)


def _ulp(a):
    return np.spacing(np.abs(a).astype(F32)).astype(F32)


def inputs(seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Families of (a, b) operand tiles (128, W) for the ALU / EFT probes."""
    rng = np.random.default_rng(seed)
    sh = (128, W)

    def wide(lo, hi):
        e = rng.integers(lo, hi, size=sh)
        m = rng.uniform(1.0, 2.0, size=sh) * rng.choice([-1.0, 1.0], size=sh)
        return _f32(np.ldexp(m, e))

    fam = {}
    fam["random"] = (wide(-40, 40), wide(-40, 40))
    # addition ties: b = odd multiple of half an ulp of a, below one ulp of a
    a = wide(-20, 20)
    k = rng.choice([1.0, 3.0, -1.0, -3.0], size=sh)
    b = _f32(k * _ulp(a) / 2)
    fam["ties_add"] = (a, b)
    # multiplication ties: 13-bit significands, so 26-bit products whose low
    # bits are exactly half an ulp in about a quarter of the cases
    ia = rng.integers(1 << 12, 1 << 13, size=sh).astype(np.float64)
    ib = rng.integers(1 << 12, 1 << 13, size=sh).astype(np.float64)
    fam["ties_mul"] = (
        _f32(np.ldexp(ia, rng.integers(-20, 0, size=sh))),
        _f32(np.ldexp(ib, rng.integers(-20, 0, size=sh)) * rng.choice([-1.0, 1.0], size=sh)),
    )
    # near overflow: sums and products at the top of the range
    fam["near_overflow"] = (
        _f32(np.ldexp(rng.uniform(1.0, 2.0, size=sh), 126 + rng.integers(0, 2, size=sh))),
        _f32(np.ldexp(rng.uniform(0.5, 2.0, size=sh), rng.integers(-1, 2, size=sh))),
    )
    # subnormal operands and results: tiny sums, products that underflow
    fam["subnormal"] = (
        _f32(np.ldexp(rng.uniform(-2.0, 2.0, size=sh), rng.integers(-149, -120, size=sh))),
        _f32(np.ldexp(rng.uniform(-2.0, 2.0, size=sh), rng.integers(-30, 2, size=sh))),
    )
    # cancellation (Sterbenz) and the TwoProd domain edges
    a = wide(-10, 10)
    fam["cancel"] = (a, _f32(-a * (1 + rng.integers(-64, 64, size=sh) * 2.0**-23)))
    fam["split_edge"] = (
        _f32(np.ldexp(rng.uniform(1.0, 2.0, size=sh), rng.integers(100, 116, size=sh))),
        _f32(np.ldexp(rng.uniform(1.0, 2.0, size=sh), rng.integers(-120, -100, size=sh))),
    )
    return fam


# ── IEEE expectations ────────────────────────────────────────────────────────


def _rn(x64):
    """float32 RN of float64 values (exact inputs assumed)."""
    with np.errstate(over="ignore"):
        return _f32(x64)


def _alu_expected(a, b):
    with np.errstate(all="ignore"):
        col = b[:, 0:1]
        return [a + b, a - b, a * b] * 2 + [a + col, a * col, a + col, a * col]


ALU_ROWS = [
    ("vector", "add"),
    ("vector", "subtract"),
    ("vector", "multiply"),
    ("gpsimd", "add"),
    ("gpsimd", "subtract"),
    ("gpsimd", "multiply"),
    ("vector tensor_scalar", "add"),
    ("vector tensor_scalar", "multiply"),
    ("scalar activation", "add (bias)"),
    ("scalar activation", "multiply (scale)"),
]


def _classify(got, exp, a, b, op):
    """Describe how `got` differs from the IEEE result `exp`."""
    g = got.view(np.uint32)
    e = exp.view(np.uint32)
    bad = g != e
    nan_both = np.isnan(got) & np.isnan(exp)
    bad &= ~nan_both
    n = int(bad.sum())
    out = {"mismatches": n, "of": int(got.size)}
    if not n:
        return out
    gb, eb = got[bad].astype(np.float64), exp[bad].astype(np.float64)
    tiny = np.finfo(F32).tiny
    kinds = {
        "flush_to_zero": int(((gb == 0) & (np.abs(eb) < tiny) & (eb != 0)).sum()),
        "toward_zero": int((np.abs(gb) < np.abs(eb)).sum()),
        "away_from_zero": int((np.abs(gb) > np.abs(eb)).sum()),
        "inf_vs_finite": int((np.isinf(gb) != np.isinf(eb)).sum()),
    }
    with np.errstate(all="ignore"):
        ulps = np.abs(gb - eb) / np.spacing(np.abs(eb).astype(F32)).astype(np.float64)
    out["kinds"] = kinds
    out["max_ulp"] = float(np.nanmax(ulps)) if ulps.size else 0.0
    i = int(np.flatnonzero(bad)[0])
    out["example"] = {
        "a": float(a.flat[i]),
        "b": float(b.flat[i]),
        "got": float(got.flat[i]),
        "ieee_rn": float(exp.flat[i]),
    }
    return out


def _fma_inputs(seed: int = 1):
    """(a, b, c) for the FMA probe: per partition, b_p random and c_p chosen
    so that a b_p + c_p cancels heavily; c is also the elementwise
    -RN(a b_p) for the scalar_tensor_tensor form (then a single rounding
    gives the exact product error, two roundings give 0)."""
    rng = np.random.default_rng(seed)
    a0 = rng.uniform(1.0, 2.0, size=(128, 1))
    # half the partitions just below 2 * 4096 / 4097, where 4097 a crosses a
    # binade and the fused and unfused Veltkamp steps differ
    a0[64:] = rng.uniform(1.99951, 1.99999, size=(64, 1))
    j = np.arange(W)[None, :]
    a = _f32(a0 * (1 + j * 2.0**-20))
    b = _f32(np.broadcast_to(rng.uniform(1.0, 2.0, size=(128, 1)), (128, W)))
    with np.errstate(all="ignore"):
        prod = _f32(a * b)
    return a, b, _f32(-prod)


def _fma_expected(a, b, c):
    """Per FMA-probe row: (single-rounding, two-rounding) expectations."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    bp, cp = b64[:, 0:1], c.astype(np.float64)[:, 0:1]
    c64 = c.astype(np.float64)
    single, double = [], []
    for cc in (cp, cp, c64, cp):
        single.append(_rn(a64 * bp + cc))  # a b exact in fp64; + c exact here
        double.append(_rn(_rn(a64 * bp).astype(np.float64) + cc))
    single.append(_rn(a64 * 4097.0 - a64))  # = 4096 a, exact
    double.append(_rn(_rn(a64 * 4097.0).astype(np.float64) - a64))
    return single, double


FMA_ROWS = [
    "vector tensor_scalar (mul, add) (P,1) operands",
    "gpsimd tensor_scalar (mul, add) (P,1) operands",
    "vector scalar_tensor_tensor (mul (P,1), add tensor)",
    "scalar activation (scale (P,1), bias (P,1))",
    "vector scalar_tensor_tensor (mul 4097, subtract a): fused Veltkamp step",
]


def _fma_verdict(got, single, double):
    gs = got.view(np.uint32)
    s_eq = bool((gs == single.view(np.uint32)).all())
    d_eq = bool((gs == double.view(np.uint32)).all())
    differ = int((single.view(np.uint32) != double.view(np.uint32)).sum())
    if s_eq and not d_eq:
        v = "fused: single rounding (an FMA)"
    elif d_eq and not s_eq:
        v = "two roundings (no FMA)"
    elif s_eq and d_eq:
        v = "inconclusive (inputs do not separate)"
    else:
        v = "neither: intermediate kept at some other precision"
    return {
        "verdict": v,
        "eq_single": int((gs == single.view(np.uint32)).sum()),
        "eq_double": int((gs == double.view(np.uint32)).sum()),
        "separating_elements": differ,
        "of": int(got.size),
    }


def _eft_check(out, a, b):
    """TwoSum / TwoProd exactness against fp64 (exact for fp32 operands)."""
    s, e, p, f = (o.astype(np.float64) for o in out)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    with np.errstate(all="ignore"):
        exact_sum = a64 + b64  # exact: two fp32 numbers
        exact_prod = a64 * b64  # exact: 48 bits
        finite_s = np.isfinite(exact_sum) & (np.abs(exact_sum) < 2.0**127)
        dom = (
            (np.abs(a64) <= 2.0**115)
            & (np.abs(b64) <= 2.0**115)
            & (np.abs(exact_prod) >= 2.0**-102)
            & (np.abs(exact_prod) < 2.0**127)
        ) | (exact_prod == 0)
        ts = finite_s & (s + e != exact_sum)
        tp = dom & (p + f != exact_prod)
    return {
        "two_sum_inexact": int(ts.sum()),
        "two_sum_checked": int(finite_s.sum()),
        "two_prod_inexact": int(tp.sum()),
        "two_prod_checked": int(dom.sum()),
    }


PE_K = 1024  # 8 K tiles


def pe_inputs(seed: int = 2):
    """TensorE probe operands: b (PE_K, 512) and three stationary operands
    (PE_K, 128). Column blocks of b, each read from one launch:
      0-127   single products a_m b_n (`at_single`: row 0 random, rest 0);
      128-255 in-tile sum of ties (`at_ones`): 1, then 127 terms of 2^-24
              (half an ulp of 1) in K tile 0: RN per add keeps 1, a wider
              accumulator gives 1 + 127 2^-24 rounded once;
      256-383 cross-tile ties (`at_ones`): 1 in tile 0, 2^-24 in each of
              tiles 1-7 (one nc_matmul each): RN per accumulate keeps 1, a
              wider PSUM gives RN(1 + 7 2^-24) = 1 + 2^-22;
      384-511 random terms (`at_rand`): precision and order of 1024-term sums."""
    rng = np.random.default_rng(seed)
    K, M, N = PE_K, 128, 512
    b = np.zeros((K, N), F32)
    at_single = np.zeros((K, M), F32)
    at_single[0, :] = _f32(rng.uniform(1.0, 2.0, M) * rng.choice([-1, 1], M))
    b[0, 0:128] = _f32(rng.uniform(1.0, 2.0, 128))
    at_ones = np.ones((K, M), F32)
    b[0, 128:256] = 1.0
    b[1:128, 128:256] = 2.0**-24
    b[0, 256:384] = 1.0
    for t in range(1, K // 128):
        b[128 * t, 256:384] = 2.0**-24
    at_rand = _f32(rng.standard_normal((K, M)))
    b[:, 384:512] = _f32(rng.standard_normal((K, 128)))
    return {"at_single": at_single, "at_ones": at_ones, "at_rand": at_rand}, b


_PE_BLOCKS = {
    "single_product": ("at_single", slice(0, 128)),
    "in_tile_sum_of_ties": ("at_ones", slice(128, 256)),
    "cross_tile_sum_of_ties": ("at_ones", slice(256, 384)),
    "random_1024_terms": ("at_rand", slice(384, 512)),
}


def _pe_sequential(at, b, cs):
    """fp32 RN accumulation in K order, and per K tile then across tiles."""
    seq = np.zeros((at.shape[1], cs.stop - cs.start), F32)
    tiles = np.zeros_like(seq)
    for kt in range(at.shape[0] // 128):
        ps = np.zeros_like(seq)
        for k in range(kt * 128, (kt + 1) * 128):
            term = _f32(np.outer(at[k], b[k, cs]))
            seq = _f32(seq + term)
            ps = _f32(ps + term)
        tiles = _f32(tiles + ps)
    return seq, tiles


def _pe_report(outs: dict, ats: dict, b) -> dict:
    rep = {}
    for name, (key, cs) in _PE_BLOCKS.items():
        at = ats[key]
        exact = at.astype(np.float64).T @ b[:, cs].astype(np.float64)
        seq, tiles = _pe_sequential(at, b, cs)
        r = {}
        for i, which in enumerate(("psum_accumulate", "per_tile_then_vector")):
            g = outs[key][i][:, cs]
            g64 = g.astype(np.float64)
            r[which] = {
                "eq_rn_of_exact": int((_f32(exact) == g).sum()),
                "eq_sequential_fp32": int(((seq if i == 0 else tiles) == g).sum()),
                "of": int(g.size),
                "max_rel_err_vs_exact": float(
                    np.max(np.abs(g64 - exact) / np.maximum(np.abs(exact), 1e-300))
                ),
                "sample": float(g[0, 0]),
                "exact_sample": float(exact[0, 0]),
            }
        rep[name] = r
    return rep


def _pe_verdict(rep):
    sp = rep["single_product"]["psum_accumulate"]
    v = {"fp32_multiply_is_rn": sp["eq_rn_of_exact"] == sp["of"]}
    for key, wide in (
        ("in_tile_sum_of_ties", 1 + 127 * 2.0**-24),
        ("cross_tile_sum_of_ties", 1 + 2.0**-22),
    ):
        s = rep[key]["psum_accumulate"]["sample"]
        if s == 1.0:
            v[key] = "fp32 RN-even per add"
        elif abs(s - float(_f32(wide))) == 0:
            v[key] = "wider than fp32, rounded once"
        else:
            v[key] = f"other: {s!r}"
    return v


def run_e0(run, seed: int = 0) -> dict:
    """The E0 probe. `run(kernel_name, *arrays) -> np.ndarray` launches a
    probe kernel. Returns the report (JSON-able) with `table`: rows
    [engine / form, identity or op, family, PASS / FAIL, details]."""
    t0 = time.time()
    table = []
    detail = {"alu": {}, "eft": {}, "fma": {}, "pe": {}}
    for fam, (a, b) in inputs(seed).items():
        out = np.asarray(run("_fp64_probe_alu_kernel", a, b))
        for (eng, op), got, exp in zip(ALU_ROWS, out, _alu_expected(a, b), strict=True):
            c = _classify(got, _f32(exp), a, b, op)
            detail["alu"][f"{eng}|{op}|{fam}"] = c
            table.append(
                [eng, f"{op} is IEEE RN", fam, "PASS" if not c["mismatches"] else "FAIL", c]
            )
        eo = np.asarray(run("_fp64_probe_eft_kernel", a, b))
        r = _eft_check(eo, a, b)
        detail["eft"][fam] = r
        table.append(
            ["vector", "TwoSum exact", fam, "PASS" if not r["two_sum_inexact"] else "FAIL", r]
        )
        table.append(
            [
                "vector",
                "Dekker TwoProd exact",
                fam,
                "PASS" if not r["two_prod_inexact"] else "FAIL",
                r,
            ]
        )
    a, b, c = _fma_inputs(seed + 1)
    out = np.asarray(run("_fp64_probe_fma_kernel", a, b, c))
    single, double = _fma_expected(a, b, c)
    for row, got, s, d in zip(FMA_ROWS, out, single, double, strict=True):
        v = _fma_verdict(got, s, d)
        detail["fma"][row] = v
        table.append([row, "fused multiply-add?", "cancel", v["verdict"], v])
    ats, b = pe_inputs(seed + 2)
    outs = {k: np.asarray(run("_fp64_probe_pe_kernel", at, b)) for k, at in ats.items()}
    pr = _pe_report(outs, ats, b)
    pv = _pe_verdict(pr)
    detail["pe"] = {"blocks": pr, "verdict": pv}
    sp = pr["single_product"]["psum_accumulate"]
    table.append(
        [
            "tensor+psum",
            "fp32 matmul product is RN(a b)",
            "single_product",
            "PASS" if pv["fp32_multiply_is_rn"] else "FAIL",
            sp,
        ]
    )
    for key in ("in_tile_sum_of_ties", "cross_tile_sum_of_ties"):
        table.append(["tensor+psum", "accumulation rounding", key, pv[key], pr[key]])
    table.append(
        ["tensor+psum", "1024-term sums vs fp64", "random", "see detail", pr["random_1024_terms"]]
    )
    gate = all(row[3] == "PASS" for row in table if row[0] == "vector" and row[2] != "subnormal")
    return {
        "probe": "E0",
        "seconds": time.time() - t0,
        "gate_vector_rn_and_eft": gate,
        "table": table,
        "detail": detail,
    }


def format_table(rep: dict) -> str:
    """Markdown pass / fail table of an E0 report."""
    lines = ["| engine / form | identity | family | result |", "|---|---|---|---|"]
    for eng, ident, fam, res, _d in rep["table"]:
        lines.append(f"| {eng} | {ident} | {fam} | {res} |")
    return "\n".join(lines)


def ideal_run(name: str, *arrays: np.ndarray) -> np.ndarray:
    """An ideal IEEE round-to-nearest device without fusion, in numpy."""
    with np.errstate(all="ignore"):
        if name == "_fp64_probe_alu_kernel":
            a, b = arrays
            return np.stack(_alu_expected(a, b)).astype(F32)
        if name == "_fp64_probe_fma_kernel":
            return np.stack(_fma_expected(*arrays)[1]).astype(F32)
        if name == "_fp64_probe_eft_kernel":
            import torch

            from .. import fp64

            a, b = (torch.from_numpy(x) for x in arrays)
            s, e = fp64.two_sum(a, b)
            p, f = fp64.two_prod(a, b)
            return np.stack([t.numpy() for t in (s, e, p, f)])
        if name == "_fp64_probe_pe_kernel":
            at, b = arrays
            seq, tiles = _pe_sequential(at, b, slice(0, b.shape[1]))
            return np.stack([seq, tiles])
    raise KeyError(name)


# ── Device runner ────────────────────────────────────────────────────────────


def device_run(name: str, *arrays: np.ndarray) -> np.ndarray:
    """Launch probe kernel `name` on the neuron device; host numpy result."""
    import torch

    from . import fp64_kernels as K

    ts = [torch.from_numpy(np.ascontiguousarray(a)).to("neuron") for a in arrays]
    return K.call(name, *ts).to("cpu").numpy()


# ── E2: accuracy and throughput of the arms ──────────────────────────────────

E2_KERNELS = [
    ("dot32", "plain"),
    ("dot32", "comp"),
    ("dot32", "sloppy"),
    ("dot32", "accurate"),
    ("sum32", "plain"),
    ("sum32", "comp"),
    ("sum32", "sloppy"),
    ("sum32", "accurate"),
    ("nrm32", "accurate"),
    ("dotff", "sloppy"),
    ("dotff", "accurate"),
    ("sumff", "accurate"),
    ("nrmff", "accurate"),
]


def e2_vectors(n: int, seed: int = 0):
    """CG-like float64 vectors (residual r and A p of a shifted 1-D Poisson
    CG after 8 iterations): smooth plus rough parts, several decades."""
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    r = b.copy()
    p = r.copy()
    rr = float(r @ r)

    def A(v):
        w = 2.001 * v
        w[1:] -= v[:-1]
        w[:-1] -= v[1:]
        return w

    for _ in range(8):
        Ap = A(p)
        alpha = rr / float(p @ Ap)
        r = r - alpha * Ap
        rr2 = float(r @ r)
        p = r + (rr2 / rr) * p
        rr = rr2
    return r, A(p)


def _bits(r: float, ref: float) -> float:
    if ref == 0:
        return 53.0 if r == 0 else 0.0
    if not np.isfinite(r):
        return 0.0
    rel = abs(r - ref) / abs(ref)
    return 53.0 if rel == 0 else float(max(0.0, min(53.0, -np.log2(rel))))


def _exact_ref(kind: str, xs64: list) -> float:
    """Exactly rounded fp64 value of the reduction of the (fp64) operands."""
    import math

    if kind.startswith("sum"):
        return math.fsum(xs64[0].tolist())
    if kind.startswith("nrm"):
        x = xs64[0]
        m = float(np.abs(x).max())
        s = 2.0 ** np.frexp(m)[1] if m else 1.0
        xs = x / s  # a power of two: exact
        p = xs * xs
        c = 134217729.0 * xs
        h = c - (c - xs)
        lo = xs - h
        e = lo * lo - (((p - h * h) - lo * h) - h * lo)
        return s * math.sqrt(math.fsum(np.concatenate([p, e]).tolist()))
    x, y = xs64
    p = x * y
    c = 134217729.0 * x
    xh = c - (c - x)
    xl = x - xh
    c = 134217729.0 * y
    yh = c - (c - y)
    yl = y - yh
    e = xl * yl - (((p - xh * yh) - xl * yh) - xh * yl)
    return math.fsum(np.concatenate([p, e]).tolist())


def _timeit(fn, reps: int, sync) -> float:
    ts = []
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def e2(
    n: int,
    *,
    device: str = "neuron",
    reps: int = 20,
    kernels=None,
    budget_s: float = 1080.0,
    check_reference: bool = True,
) -> dict:
    """E2 at length n: for each (kind, arm) kernel, correct bits against the
    exactly rounded reference, bitwise equality with the CPU reference, the
    first-call (compile) time and the warm time per element, end to end
    (layout + kernel + partials download + host finish) and kernel only;
    plus the host fp64 round trip each replaces (device -> host copy of the
    fp32 operands and numpy float64). Stops starting new kernels after
    `budget_s` seconds (partial report, `truncated`). `device="cpu"` runs the
    pytorch reference as the "device" (for tests)."""
    import torch

    from .. import fp64
    from . import fp64_kernels as K

    t_start = time.time()
    if device == "neuron":
        from . import dispatch as D

        sync = D._device_synchronize
    else:

        def sync():
            return None

    r64, ap64 = e2_vectors(n)
    r32, ap32 = r64.astype(np.float32), ap64.astype(np.float32)
    X = torch.from_numpy(r32).to(device)
    Y = torch.from_numpy(ap32).to(device)
    fr, fa = fp64.from_fp64(torch.from_numpy(r64)), fp64.from_fp64(torch.from_numpy(ap64))
    XH, XL = fr.hi.to(device), fr.lo.to(device)
    YH, YL = fa.hi.to(device), fa.lo.to(device)
    rep = {"n": n, "device": device, "plan": list(fp64.plan(n)), "kernels": {}, "host_fp64": {}}
    rd = r32.astype(np.float64)
    ad = ap32.astype(np.float64)

    def host_dot():
        x = X.to("cpu").numpy().astype(np.float64)
        y = Y.to("cpu").numpy().astype(np.float64)
        return float(np.dot(x, y))

    def host_sum():
        return float(np.sum(X.to("cpu").numpy().astype(np.float64)))

    def host_nrm():
        return float(np.linalg.norm(X.to("cpu").numpy().astype(np.float64)))

    for name, fn, ref in (
        ("dot", host_dot, _exact_ref("dot", [rd, ad])),
        ("sum", host_sum, _exact_ref("sum", [rd])),
        ("nrm2", host_nrm, _exact_ref("nrm", [rd])),
    ):
        t = _timeit(fn, reps, sync)
        rep["host_fp64"][name] = {
            "ms": t * 1e3,
            "ns_per_elem": t / n * 1e9,
            "bits": _bits(fn(), ref),
        }
    refs = {
        "dot32": _exact_ref("dot", [rd, ad]),
        "sum32": _exact_ref("sum", [rd]),
        "nrm32": _exact_ref("nrm", [rd]),
        "dotff": _exact_ref("dot", [r64, ap64]),
        "sumff": _exact_ref("sum", [r64]),
        "nrmff": _exact_ref("nrm", [r64]),
    }
    ops = {
        "dot32": [X, Y],
        "sum32": [X],
        "nrm32": [X],
        "dotff": [XH, XL, YH, YL],
        "sumff": [XH, XL],
        "nrmff": [XH, XL],
    }
    pl = fp64.plan(n)
    for kind, arm in kernels or E2_KERNELS:
        if time.time() - t_start > budget_s:
            rep["truncated"] = True
            break
        xs = ops[kind]
        key = f"{kind}/{arm}"
        sim = device != "neuron"

        def run(xs=xs, kind=kind, arm=arm, sim=sim):
            if sim:
                return fp64.reference_partials(kind, xs, arm)
            return K.run_partials(kind, xs, arm)

        t0 = time.perf_counter()
        parts = run()
        first = time.perf_counter() - t0
        val = fp64.finish_host(parts, kind)
        e2e = _timeit(lambda run=run, kind=kind: fp64.finish_host(run(), kind), reps, sync)
        rows = [fp64.layout(t, pl).contiguous() for t in xs]
        if sim:
            kern = e2e
            lay = 0.0
        else:
            name = K.REDUCE_KERNELS[(kind, arm)]
            kern = _timeit(lambda name=name, rows=rows: K.call(name, *rows), reps, sync)
            lay = _timeit(lambda xs=xs: [fp64.layout(t, pl) for t in xs], reps, sync)
        row = {
            "bits": _bits(val, refs[kind]),
            "value": val,
            "first_call_s": first,
            "e2e_ms": e2e * 1e3,
            "e2e_ns_per_elem": e2e / n * 1e9,
            "kernel_ms": kern * 1e3,
            "kernel_ns_per_elem": kern / n * 1e9,
            "layout_ms": lay * 1e3,
        }
        if check_reference:
            ref = fp64.reference_partials(kind, [t.to("cpu") for t in xs], arm)
            row["bitwise_vs_cpu_reference"] = bool(
                torch.equal(parts.view(torch.int32), ref.view(torch.int32))
            )
        rep["kernels"][key] = row
    rep["seconds"] = time.time() - t_start
    return rep


def format_e2(rep: dict) -> str:
    n = rep["n"]
    lines = [
        f"E2 at n = {n} ({rep['device']}, plan {rep['plan']})",
        "| arm | bits | e2e ns/elem | kernel ns/elem | first call s | bitwise vs CPU ref |",
        "|---|---|---|---|---|---|",
    ]
    for k, r in rep["kernels"].items():
        lines.append(
            f"| {k} | {r['bits']:.1f} | {r['e2e_ns_per_elem']:.3f} | "
            f"{r['kernel_ns_per_elem']:.3f} | {r['first_call_s']:.1f} | "
            f"{r.get('bitwise_vs_cpu_reference', '')} |"
        )
    for k, r in rep["host_fp64"].items():
        lines.append(
            f"| host fp64 round trip {k} | {r['bits']:.1f} | {r['ns_per_elem']:.3f} | | | |"
        )
    return "\n".join(lines)


# ── LNC digests ──────────────────────────────────────────────────────────────


def lnc_digests(ns=(100003, 1000000), device: str = "neuron") -> dict:
    """SHA-256 of every reduction kernel's partials for fixed inputs, and
    whether they equal the CPU reference. Run once at LNC=2 and once at LNC=1
    (NEURON_LOGICAL_NC_CONFIG=1, whole chip): the digests must agree."""
    import hashlib
    import os

    import torch

    from .. import fp64
    from . import fp64_kernels as K

    out = {"lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"), "digests": {}, "equal_ref": {}}
    for n in ns:
        r64, ap64 = e2_vectors(n, seed=1)
        x = torch.from_numpy(r64.astype(np.float32))
        y = torch.from_numpy(ap64.astype(np.float32))
        fr, fa = fp64.from_fp64(torch.from_numpy(r64)), fp64.from_fp64(torch.from_numpy(ap64))
        ops = {
            "dot32": [x, y],
            "sum32": [x],
            "nrm32": [x],
            "dotff": [fr.hi, fr.lo, fa.hi, fa.lo],
            "sumff": [fr.hi, fr.lo],
            "nrmff": [fr.hi, fr.lo],
        }
        for (kind, arm), _name in sorted(K.REDUCE_KERNELS.items()):
            xs = ops[kind]
            if device == "neuron":
                parts = K.run_partials(kind, [t.to("neuron") for t in xs], arm)
            else:
                parts = fp64.reference_partials(kind, xs, arm)
            key = f"{kind}/{arm}/n={n}"
            out["digests"][key] = hashlib.sha256(parts.numpy().tobytes()).hexdigest()
            ref = fp64.reference_partials(kind, xs, arm)
            out["equal_ref"][key] = bool(
                torch.equal(parts.view(torch.int32), ref.view(torch.int32))
            )
    return out
