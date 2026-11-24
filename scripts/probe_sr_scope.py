#!/usr/bin/env python
"""What `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` changes on Trn2 (trnsci/trnsci#174).

Two probe kernels on (128, F) tiles:

* `_sr_bf16_kernel` writes bf16 results of: fp32 -> bf16 casts by VectorE,
  GpSimd and ScalarE (`tensor_copy` / `activation(copy)`), DMA casts (HBM fp32
  -> SBUF bf16 and HBM fp32 -> HBM bf16), bf16 arithmetic (VectorE add and
  multiply, GpSimd add, ScalarE activation with an fp32 (P,1) scale), and the
  eviction of an fp32 PSUM matmul result to bf16 by VectorE and by ScalarE.
* `_sr_f32_kernel` writes fp32 results of the same PSUM matmul (the pre-
  rounding values of the eviction rows) and of fp32 arithmetic on VectorE,
  GpSimd and ScalarE, to see whether SR reaches fp32 results at all.
* `_sr_bf16_seeded_kernel` is `_sr_bf16_kernel` after `nisa.set_rng_seed`.

Half the partitions of the cast input hold x = v + k/64 ulp_bf16(v) with
k = partition - 64 (v a random bf16 in [1, 2)): under stochastic rounding the
fraction rounded up should be k/64, and exact values (k = 0) must stay put.

    python -P scripts/probe_sr_scope.py verify             # MLIR verify, LNC=1 and 2
    python -P scripts/probe_sr_scope.py run --json a.json  # one process (env as given)
    python -P scripts/probe_sr_scope.py driver --dir D     # processes with SR off / on /
                                                            # fixed NEURON_RT_STOCHASTIC_ROUNDING_SEED
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
TW = 512
F = 1024  # two column tiles; at LNC=2 each program takes one

BF16_ROWS = [
    "cast VectorE tensor_copy fp32->bf16",
    "cast GpSimd tensor_copy fp32->bf16",
    "cast ScalarE activation(copy) fp32->bf16",
    "cast DMA HBM fp32 -> SBUF bf16",
    "cast DMA HBM fp32 -> HBM bf16",
    "bf16 add VectorE",
    "bf16 add GpSimd",
    "bf16 multiply VectorE",
    "bf16 ScalarE activation(copy, scale=(P,1) fp32)",
    "PSUM fp32 -> bf16 eviction VectorE tensor_copy",
    "PSUM fp32 -> bf16 eviction ScalarE activation(copy)",
]
F32_ROWS = [
    "PSUM fp32 matmul result (VectorE copy to fp32)",
    "fp32 multiply VectorE",
    "fp32 multiply GpSimd",
    "fp32 add VectorE",
    "fp32 ScalarE activation(copy, scale=(P,1))",
]
NB, NF = len(BF16_ROWS), len(F32_ROWS)


if HAS_NKI:

    def _sb(dtype, shape=(P, TW)):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def _matmul(pa, pb, c0):
        at = nl.load(pa[0:P, 0:P])
        bt = nl.load(pb[0:P, c0 : c0 + TW])
        ps = nl.zeros((P, TW), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=at, moving=bt, accumulate=True)
        return ps

    def _bf16_tile(x, a, b, pa, pb, out, c0):
        xt = nl.load(x[0:P, c0 : c0 + TW])
        at = nl.load(a[0:P, c0 : c0 + TW])
        bt = nl.load(b[0:P, c0 : c0 + TW])
        col = _sb(nl.float32, (P, 1))
        nisa.tensor_copy(dst=col, src=xt[:, 0:1])
        rows = []
        for eng in (nisa.vector_engine, nisa.gpsimd_engine):
            r = _sb(nl.bfloat16)
            nisa.tensor_copy(dst=r, src=xt, engine=eng)
            rows.append(r)
        r = _sb(nl.bfloat16)
        nisa.activation(dst=r, op=nl.copy, data=xt)
        rows.append(r)
        r = _sb(nl.bfloat16)
        nisa.dma_copy(dst=r, src=x[0:P, c0 : c0 + TW])
        rows.append(r)
        rows.append(None)  # DMA HBM -> HBM, written directly below
        for op, eng in (
            (nl.add, nisa.vector_engine),
            (nl.add, nisa.gpsimd_engine),
            (nl.multiply, nisa.vector_engine),
        ):
            r = _sb(nl.bfloat16)
            nisa.tensor_tensor(dst=r, data1=at, data2=bt, op=op, engine=eng)
            rows.append(r)
        r = _sb(nl.bfloat16)
        nisa.activation(dst=r, op=nl.copy, data=at, scale=col)
        rows.append(r)
        ps = _matmul(pa, pb, c0)
        r = _sb(nl.bfloat16)
        nisa.tensor_copy(dst=r, src=ps, engine=nisa.vector_engine)
        rows.append(r)
        r = _sb(nl.bfloat16)
        nisa.activation(dst=r, op=nl.copy, data=ps)
        rows.append(r)
        for i, r in enumerate(rows):
            if r is None:
                nisa.dma_copy(dst=out[i, 0:P, c0 : c0 + TW], src=x[0:P, c0 : c0 + TW])
            else:
                nl.store(out[i, 0:P, c0 : c0 + TW], value=r)

    def _f32_tile(x, a, b, pa, pb, out, c0):
        xt = nl.load(x[0:P, c0 : c0 + TW])
        at = nl.load(a[0:P, c0 : c0 + TW])
        col = _sb(nl.float32, (P, 1))
        nisa.tensor_copy(dst=col, src=xt[:, 0:1])
        a32 = _sb(nl.float32)
        nisa.tensor_copy(dst=a32, src=at)
        rows = []
        ps = _matmul(pa, pb, c0)
        r = _sb(nl.float32)
        nisa.tensor_copy(dst=r, src=ps, engine=nisa.vector_engine)
        rows.append(r)
        for eng in (nisa.vector_engine, nisa.gpsimd_engine):
            r = _sb(nl.float32)
            nisa.tensor_tensor(dst=r, data1=xt, data2=xt, op=nl.multiply, engine=eng)
            rows.append(r)
        r = _sb(nl.float32)
        nisa.tensor_tensor(dst=r, data1=xt, data2=a32, op=nl.add, engine=nisa.vector_engine)
        rows.append(r)
        r = _sb(nl.float32)
        nisa.activation(dst=r, op=nl.copy, data=xt, scale=col)
        rows.append(r)
        for i, r in enumerate(rows):
            nl.store(out[i, 0:P, c0 : c0 + TW], value=r)

    def _tiles(body, x, a, b, pa, pb, out):
        nt = x.shape[1] // TW
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        per = nt // n_prog
        if n_prog > 1:
            for j in nl.affine_range(per):
                body(x, a, b, pa, pb, out, (nl.program_id(0) * per + j) * TW)
        else:
            for j in nl.affine_range(nt):
                body(x, a, b, pa, pb, out, j * TW)

    @nki.jit
    def _sr_bf16_kernel(x, a, b, pa, pb):
        out = nl.ndarray((NB, P, x.shape[1]), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        _tiles(_bf16_tile, x, a, b, pa, pb, out)
        return out

    @nki.jit
    def _sr_bf16_seeded_kernel(x, a, b, pa, pb, seed):
        s = nl.load(seed[0:1, 0:1])
        nisa.set_rng_seed(src_seeds=s)
        out = nl.ndarray((NB, P, x.shape[1]), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        _tiles(_bf16_tile, x, a, b, pa, pb, out)
        return out

    @nki.jit
    def _sr_f32_kernel(x, a, b, pa, pb):
        out = nl.ndarray((NF, P, x.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm)
        _tiles(_f32_tile, x, a, b, pa, pb, out)
        return out


# ── inputs and host expectations ────────────────────────────────────────────


def inputs(seed: int = 0):
    import torch

    rng = np.random.default_rng(seed)
    x = np.empty((P, F), np.float32)
    # partitions 0-63: wide random fp32 with all 23 mantissa bits random
    e = rng.integers(-30, 30, size=(64, F))
    m = 1.0 + rng.integers(0, 1 << 23, size=(64, F)) * 2.0**-23
    x[:64] = np.ldexp(m * rng.choice([-1.0, 1.0], size=(64, F)), e)
    # partitions 64-127: v + k/64 ulp_bf16(v), v a bf16 in [1, 2), k = p - 64
    v = 1.0 + rng.integers(0, 128, size=(64, F)) * 2.0**-7
    k = np.arange(64)[:, None]
    x[64:] = v + k / 64 * 2.0**-7
    # bf16 operands: a in [1, 2), b = +-a' 2^-s with s in 1..12 (sums and
    # products that need rounding to bf16)
    a = 1.0 + rng.integers(0, 128, size=(P, F)) * 2.0**-7
    b = np.ldexp(1.0 + rng.integers(0, 128, size=(P, F)) * 2.0**-7, -rng.integers(1, 13, (P, F)))
    b *= rng.choice([-1.0, 1.0], size=(P, F))
    pa = rng.uniform(-1, 1, size=(P, P))
    pb = rng.uniform(-1, 1, size=(P, F))
    bf = torch.bfloat16
    return {
        "x": torch.from_numpy(x),
        "a": torch.from_numpy(a.astype(np.float32)).to(bf),
        "b": torch.from_numpy(b.astype(np.float32)).to(bf),
        "pa": torch.from_numpy(pa.astype(np.float32)).to(bf),
        "pb": torch.from_numpy(pb.astype(np.float32)).to(bf),
    }


def _tile_col(x):
    """The kernels' (P,1) scale: column 0 of each TW-column tile, broadcast."""
    return x[:, ::TW].repeat_interleave(TW, dim=1)


def expected_bf16(ins, psum32):
    """Round-to-nearest-even bf16 expectation of every BF16_ROWS row (host torch
    casts are RNE), from the exact fp32 pre-rounding values."""
    import torch

    bf = torch.bfloat16
    x, a, b = ins["x"], ins["a"].float(), ins["b"].float()
    col = _tile_col(x)
    cast = x.to(bf)
    prod_scale = (a * col).to(bf)  # fp32 product (RN), then bf16
    ev = torch.from_numpy(psum32).to(bf)
    return [cast] * 5 + [(a + b).to(bf)] * 2 + [(a * b).to(bf), prod_scale, ev, ev]


def expected_f32(ins, psum32):
    import torch

    x, a = ins["x"], ins["a"].float()
    col = _tile_col(x)
    return [torch.from_numpy(psum32), x * x, x * x, x + a, x * col]


def _bits16(t):
    import torch

    return t.contiguous().view(torch.int16).numpy().astype(np.int32) & 0xFFFF


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _row_stats_bf16(reps: list[np.ndarray], exp_bits: np.ndarray, x: np.ndarray | None):
    """reps: R arrays of uint16 bits (P, F) of one row."""
    st = np.stack(reps)  # (R, P, F)
    diff = st != exp_bits[None]
    out = {
        "differs_from_RNE": float(diff.mean()),
        "differs_from_RNE_rep0": float(diff[0].mean()),
        "elements_varying_across_repeats": float((st != st[0:1]).any(axis=0).mean()),
        "digests": [_digest(r) for r in reps],
    }
    # far misses: more than one bf16 ulp from the RNE result (same sign assumed)
    d = np.abs(st.astype(np.int64) - exp_bits[None].astype(np.int64))
    out["max_bf16_ulp_from_RNE"] = int(d.max())
    if x is not None:
        # partitions 64-127: fraction rounded up vs k / 64; floor to bf16: truncate the low 16 bits (all values positive)
        floor_bits = (x[64:].view(np.uint32) >> 16).astype(np.int32)
        up = st[:, 64:] != floor_bits[None]
        frac = up.mean(axis=(0, 2))
        k = np.arange(64) / 64
        out["p_up_vs_k/64_max_abs_dev"] = float(np.max(np.abs(frac - k)))
        out["p_up_at_k=0 (exact inputs)"] = float(frac[0])
        out["p_up_at_k=16 (1/4)"] = float(frac[16])
        out["p_up_at_k=32 (tie)"] = float(frac[32])
        out["p_up_at_k=48 (3/4)"] = float(frac[48])
        out["mean_bias_in_bf16_ulps"] = float(
            np.mean((st[:, 64:].astype(np.float64) - floor_bits[None]) - k[None, :, None])
        )
    return out


def _np(t):
    """Host numpy array of a torch tensor (bf16 as ml_dtypes.bfloat16)."""
    import torch

    if t.dtype == torch.bfloat16:
        import ml_dtypes

        return t.float().numpy().astype(ml_dtypes.bfloat16)
    return t.numpy()


def _sim_handle(kernel):
    """nki.simulate stand-in for a wrap_nki handle (host torch in and out)."""
    import torch

    def h(*ts):
        r = np.asarray(nki.simulate(kernel)(*[_np(t) for t in ts]))
        t = torch.from_numpy(r.astype(np.float32))
        return t.to(torch.bfloat16) if r.dtype.itemsize == 2 else t

    return h


def run_process(reps: int, seeds=(42, 42, 43), simulate: bool = False) -> dict:
    import torch

    ins = inputs(0)
    order = ("x", "a", "b", "pa", "pb")
    if simulate:
        dev, wrap, where = ins, _sim_handle, "cpu"
    else:
        import torch_neuronx

        dev = {k: v.to("neuron") for k, v in ins.items()}
        wrap, where = torch_neuronx.wrap_nki, "neuron"
    hb = wrap(_sr_bf16_kernel)
    hs = wrap(_sr_bf16_seeded_kernel)
    hf = wrap(_sr_f32_kernel)
    args = [dev[k] for k in order]
    t0 = time.perf_counter()
    f32 = [hf(*args).to("cpu").numpy() for _ in range(reps)]
    b16 = [_bits16(hb(*args).to("cpu")) for _ in range(reps)]
    seeded = {}
    for i, s in enumerate(seeds):
        st = torch.tensor([[s]], dtype=torch.int32).to(where)
        seeded[f"{i}:seed={s}"] = [_bits16(hs(*args, st).to("cpu")) for _ in range(2)]
    dt = time.perf_counter() - t0
    psum32 = f32[0][0]
    exp_b = [_bits16(t) for t in expected_bf16(ins, psum32)]
    exp_f = [t.numpy() for t in expected_f32(ins, psum32)]
    x = ins["x"].numpy()
    rep = {
        "env": {
            k: os.environ.get(k)
            for k in (
                "NEURON_RT_STOCHASTIC_ROUNDING_EN",
                "NEURON_RT_STOCHASTIC_ROUNDING_SEED",
                "NEURON_LOGICAL_NC_CONFIG",
            )
        },
        "reps": reps,
        "seconds": dt,
        "bf16": {},
        "f32": {},
        "seeded": {},
    }
    for i, name in enumerate(BF16_ROWS):
        rep["bf16"][name] = _row_stats_bf16([r[i] for r in b16], exp_b[i], x if i < 5 else None)
    for i, name in enumerate(F32_ROWS):
        st = np.stack([r[i] for r in f32]).view(np.uint32)
        e = exp_f[i].view(np.uint32)
        rep["f32"][name] = {
            "differs_from_IEEE_RN": float((st != e[None]).mean()) if i else None,
            "elements_varying_across_repeats": float((st != st[0:1]).any(axis=0).mean()),
            "digests": [_digest(r) for r in st],
        }
    for key, rs in seeded.items():
        rep["seeded"][key] = {
            name: {
                "digests": [_digest(r[i]) for r in rs],
                "differs_from_RNE": float(np.mean([(r[i] != exp_b[i]).mean() for r in rs])),
            }
            for i, name in enumerate(BF16_ROWS)
        }
    return rep


def _verify():
    import torch
    from nki.compiler.frontend import FrontendKind, source_to_mlir

    ins = inputs(0)
    args = [_np(ins[k]) for k in ("x", "a", "b", "pa", "pb")]
    seed = np.array([[42]], np.int32)
    for lnc in (1, 2):
        for k, extra in (
            (_sr_bf16_kernel, []),
            (_sr_bf16_seeded_kernel, [seed]),
            (_sr_f32_kernel, []),
        ):
            source_to_mlir(
                k, *args, *extra, target="trn2", lnc=lnc, frontend=FrontendKind.TRACER
            ).verify()
        print(f"verify lnc={lnc}: ok")


DRIVER = [
    ("off_a", {}),
    ("off_b", {}),
    ("on_a", {"NEURON_RT_STOCHASTIC_ROUNDING_EN": "1"}),
    ("on_b", {"NEURON_RT_STOCHASTIC_ROUNDING_EN": "1"}),
    (
        "on_seed1234_a",
        {"NEURON_RT_STOCHASTIC_ROUNDING_EN": "1", "NEURON_RT_STOCHASTIC_ROUNDING_SEED": "1234"},
    ),
    (
        "on_seed1234_b",
        {"NEURON_RT_STOCHASTIC_ROUNDING_EN": "1", "NEURON_RT_STOCHASTIC_ROUNDING_SEED": "1234"},
    ),
    (
        "on_seed5678",
        {"NEURON_RT_STOCHASTIC_ROUNDING_EN": "1", "NEURON_RT_STOCHASTIC_ROUNDING_SEED": "5678"},
    ),
]


def driver(d: str, reps: int, timeout_s: float) -> int:
    os.makedirs(d, exist_ok=True)
    docs = {}
    for name, extra in DRIVER:
        env = dict(os.environ)
        for k in ("NEURON_RT_STOCHASTIC_ROUNDING_EN", "NEURON_RT_STOCHASTIC_ROUNDING_SEED"):
            env.pop(k, None)
        env.update(extra)
        js = os.path.join(d, f"sr_{name}.json")
        cmd = [sys.executable, "-P", os.path.abspath(__file__), "run", "--reps", str(reps)]
        cmd += ["--json", js]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, env=env, timeout=timeout_s, capture_output=True, text=True)
        print(f"{name}: rc={r.returncode} in {time.perf_counter() - t0:.0f}s", flush=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:])
            continue
        with open(js) as fh:
            docs[name] = json.load(fh)
    summary = summarize(docs)
    with open(os.path.join(d, "sr_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(format_summary(summary))
    return 0


def summarize(docs: dict) -> dict:
    out = {"bf16": {}, "f32": {}, "seeded": {}}
    names = list(docs)
    for sect in ("bf16", "f32"):
        rows = docs[names[0]][sect]
        for row in rows:
            ent = {}
            for n in names:
                s = docs[n][sect][row]
                ent[n] = {
                    "differs": s.get("differs_from_RNE", s.get("differs_from_IEEE_RN")),
                    "vary_in_process": s["elements_varying_across_repeats"],
                    "first_digest": s["digests"][0],
                    **{k: v for k, v in s.items() if k.startswith("p_up") or k.startswith("mean")},
                }
            out[sect][row] = ent
    for n in names:
        out["seeded"][n] = {
            row: [v["digests"] for v in docs[n]["seeded"].values()]
            for row in docs[n]["seeded"][next(iter(docs[n]["seeded"]))]
        }
    return out


def format_summary(s: dict) -> str:
    lines = []
    for sect in ("bf16", "f32"):
        for row, ent in s[sect].items():
            names = list(ent)
            dig = [ent[n]["first_digest"] for n in names]
            groups = {}
            for n, g in zip(names, dig, strict=True):
                groups.setdefault(g, []).append(n)
            parts = [
                f"{n}: diff {ent[n]['differs'] if ent[n]['differs'] is None else round(ent[n]['differs'], 4)}"
                f" vary {round(ent[n]['vary_in_process'], 4)}"
                for n in names
            ]
            lines.append(f"[{sect}] {row}\n    " + "; ".join(parts))
            lines.append(f"    first-call digest groups: {list(groups.values())}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    pr = sub.add_parser("run")
    pr.add_argument("--json", default=None)
    pr.add_argument("--reps", type=int, default=8)
    pr.add_argument("--simulate", action="store_true", help="nki.simulate dry run")
    pd = sub.add_parser("driver")
    pd.add_argument("--dir", required=True)
    pd.add_argument("--reps", type=int, default=8)
    pd.add_argument("--timeout-s", type=float, default=240.0)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        _verify()
        return 0
    if args.cmd == "driver":
        return driver(args.dir, args.reps, args.timeout_s)
    rep = run_process(args.reps, simulate=args.simulate)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    for sect in ("bf16", "f32"):
        for row, s in rep[sect].items():
            keys = {k: v for k, v in s.items() if k != "digests"}
            print(f"[{sect}] {row}: {keys}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
