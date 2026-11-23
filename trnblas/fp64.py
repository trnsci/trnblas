"""
fp64-class vector arithmetic on fp32 hardware (trnsci/trnsci#169, step 1).

Trainium has no fp64 arithmetic: device float64 ops compute at fp32. This
module gives reductions, dot products, norms, axpy and the combine step of
`gemm_fp64` at better than fp32 precision, three ways:

1. **Float-float** (double-word fp32, `FF`): a value is the unevaluated sum
   hi + lo of two fp32 numbers, about 48 significant bits and fp32's exponent
   range. Built from error-free transformations: Knuth's TwoSum, Dekker's
   FastTwoSum, and Dekker's TwoProd with the Veltkamp split (FMA-free, since
   FMA behaviour on the Trainium engines is unverified). The double-word
   additions and products follow Joldes, Muller and Popescu, "Tight and
   rigorous error bounds for basic building blocks of double-word
   arithmetic", ACM TOMS 44(2), 2017 (AccurateDWPlusDW, DWPlusFP, DWTimesDW,
   DWTimesFP); the compensated dot follows Ogita, Rump and Oishi, "Accurate
   sum and dot product", SIAM J. Sci. Comput. 26(6), 2005.
2. **Integer-emulated, exact** (`exact_sum`, `exact_dot`): fixed-point limbs
   cut from the raw fp64 bits (as the raw-bits Ozaki split does, trnblas#125)
   and accumulated in int32 lanes with explicit carries. Exact whatever the
   condition number; the cost grows with the exponent spread of the data.
3. **Hybrid** (the default `finish="host"`): float-float partials on the
   device, one per (partition, chunk), then an exactly rounded fp64 sum of
   those 2 x 4 x 128 fp32 numbers on the host (`math.fsum`). This is the
   "pairwise + host fp64 leaves" pattern of trnsolver#140, raised in
   precision: the device leaves are float-float, not fp32.

Backends: `pytorch` is a CPU reference that performs the device's fp32
operations one for one, in the same order, so a device result equals it bit
for bit when the engines' fp32 add / subtract / multiply round to nearest
even and no intermediate is subnormal (both checked by the device probes in
`tests/test_fp64_neuron.py`). `nki` runs the kernels of
`trnblas/nki/fp64_kernels.py`. Under `auto` a device tensor on the native
TorchNeuron path takes the kernels and a host tensor the reference, so both
give the same bits.

Determinism: a reduction over n has one fixed order that depends only on n.
The vector is laid out as (128, F) rows, cut into `NCH` = 4 chunks of
columns; each lane (partition, column) of a chunk accumulates its elements
in column-tile order, the W lanes of a chunk fold in a fixed halving tree,
and the (128, NCH) chunk partials are finished on the host exactly (or
folded in a fixed tree, `finish="ff"`). At LNC=2 the two programs take two
chunks each, at LNC=1 one program takes all four, with the same operations,
so the result is the same bits at any LNC (the trnsolver#128 requirement).

Range and precision limits of float-float (all from the fp32 format):

* 48 significant bits (relative representation error <= 2^-48 ~ 3.6e-15,
  against 2^-53 ~ 1.1e-16 for fp64). On fp64 input data this alone bounds a
  dot product's relative error by about 2^-48 * cond, where
  cond = 2 sum|x_i y_i| / |x.y|: float-float is not an fp64 substitute for
  ill-conditioned dots of fp64 data (see the accuracy table on #169); the
  exact path is.
* Exponent range: |x| < 2^128 (OverflowError beyond), and full precision only
  for |x| >= 2^-102 (`FF_MIN_FULL`); below it lo is subnormal and precision
  falls off gradually (and may flush to zero on the device).
* TwoProd is exact for |a|, |b| <= 2^115 (`SPLIT_MAX`, the Veltkamp split
  overflows above) and |a b| >= 2^-102. `nrm2` scales each chunk lane group
  by a power of two first, so it has no such limit.
* Non-finite inputs give a nan result (inf - inf in the error terms): the
  policy is "inf or nan in, nan out" for every reduction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import torch

F32 = torch.float32
F64 = torch.float64

P = 128  # partitions
NCH = 4  # chunks per reduction: LNC-invariant for 1, 2 or 4 programs
W_MAX = 512  # columns per tile
W_ALIGN = 16  # tile width granularity for short vectors

SPLITTER = 4097.0  # 2^12 + 1: Veltkamp split of a 24-bit significand into 12 + 12
SPLIT_MAX = 2.0**115  # |a| above this overflows a * SPLITTER (fp32 max ~ 2^128)
FF_MAX = float(np.finfo(np.float32).max)
FF_MIN_FULL = 2.0**-102  # smallest |x| whose lo is still a normal fp32 number
U32 = 2.0**-24  # fp32 unit roundoff
U_FF = 2.0**-48  # float-float representation error bound (relative)


class FF(NamedTuple):
    """A float-float tensor: value = hi + lo, both float32 of one shape."""

    hi: torch.Tensor
    lo: torch.Tensor

    @property
    def shape(self):
        return self.hi.shape

    @property
    def device(self):
        return self.hi.device

    def to(self, device) -> FF:
        return FF(self.hi.to(device), self.lo.to(device))


# ── Error-free transformations (fp32, round to nearest even) ─────────────────
# Every function below is a fixed sequence of single fp32 operations; the NKI
# kernels issue the same operations in the same order (one instruction each).


def two_sum(a: torch.Tensor, b: torch.Tensor):
    """Knuth's TwoSum: s + e == a + b exactly, s = fl(a + b). 6 flops."""
    s = a + b
    bb = s - a
    t = s - bb
    e1 = a - t
    e2 = b - bb
    return s, e1 + e2


def fast_two_sum(a: torch.Tensor, b: torch.Tensor):
    """Dekker's FastTwoSum (|a| >= |b| or a == 0): s + e == a + b. 3 flops."""
    s = a + b
    z = s - a
    return s, b - z


def split(a: torch.Tensor):
    """Veltkamp split: a == h + l, each with at most 12 significant bits."""
    c = a * SPLITTER
    d = c - a
    h = c - d
    return h, a - h


def two_prod(a: torch.Tensor, b: torch.Tensor):
    """Dekker's TwoProd without FMA: p + e == a * b exactly, p = fl(a b).

    Exact for |a|, |b| <= SPLIT_MAX and no underflow (|a b| >= 2^-102).
    17 flops, in the order of Ogita, Rump and Oishi (2005), Algorithm 3.3."""
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    t = p - ah * bh
    t = t - al * bh
    t = t - ah * bl
    return p, al * bl - t


def two_prod_fma(a: torch.Tensor, b: torch.Tensor):
    """TwoProd with a fused multiply-add: e = fma(a, b, -p). The reference
    computes the fma exactly through float64 (a b is exact there). The device
    variant is only valid if the probe shows the engine's fused op does not
    round the intermediate product."""
    p = a * b
    e = (a.to(F64) * b.to(F64) - p.to(F64)).to(F32)
    return p, e


def ff_add(x: FF | tuple, y: FF | tuple) -> FF:
    """AccurateDWPlusDW (Joldes et al. 2017, Alg. 6): relative error <= 3u^2."""
    sh, sl = two_sum(x[0], y[0])
    th, tl = two_sum(x[1], y[1])
    sl = sl + th
    vh, vl = fast_two_sum(sh, sl)
    vl = vl + tl
    return FF(*fast_two_sum(vh, vl))


def ff_add_sloppy(x: FF | tuple, y: FF | tuple) -> FF:
    """SloppyDWPlusDW (Joldes et al. 2017, Alg. 5): 11 flops; accurate unless
    x and y nearly cancel. Used only by the `accumulate="sloppy"` variant."""
    sh, sl = two_sum(x[0], y[0])
    v = x[1] + y[1]
    w = sl + v
    return FF(*fast_two_sum(sh, w))


def ff_add_fp(x: FF | tuple, b: torch.Tensor) -> FF:
    """DWPlusFP (Joldes et al. 2017, Alg. 4): relative error <= 2u^2."""
    sh, sl = two_sum(x[0], b)
    v = x[1] + sl
    return FF(*fast_two_sum(sh, v))


def ff_mul(x: FF | tuple, y: FF | tuple) -> FF:
    """DWTimesDW without FMA (Joldes et al. 2017, Alg. 10 with TwoProd)."""
    ch, cl1 = two_prod(x[0], y[0])
    tl1 = x[0] * y[1]
    tl2 = x[1] * y[0]
    cl2 = tl1 + tl2
    cl3 = cl1 + cl2
    return FF(*fast_two_sum(ch, cl3))


def ff_mul_fp(x: FF | tuple, b: torch.Tensor) -> FF:
    """DWTimesFP without FMA (Joldes et al. 2017, Alg. 7 with TwoProd)."""
    ch, cl1 = two_prod(x[0], b)
    cl2 = x[1] * b
    th, tl1 = fast_two_sum(ch, cl2)
    tl2 = tl1 + cl1
    return FF(*fast_two_sum(th, tl2))


# ── Conversion ───────────────────────────────────────────────────────────────


def _as_f64_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.dtype != F64:
            raise TypeError(f"from_fp64 takes float64; got {x.dtype}")
        return x.detach().to("cpu").contiguous().numpy()
    a = np.ascontiguousarray(x)
    if a.dtype != np.float64:
        raise TypeError(f"from_fp64 takes float64; got {a.dtype}")
    return a


def split_bits(x: np.ndarray, underflow: str = "keep") -> tuple[np.ndarray, np.ndarray]:
    """(hi, lo) float32 arrays of float64 `x`, from its raw bits.

    hi = RN32(x) is rounded on the integer fields (52-bit fraction to 23 bits,
    round half to even, carry into the exponent); lo = RN32(x - hi), where
    x - hi is an integer below 2^29 times a power of two, exact in float64.
    So hi + lo is the canonical double-word of x, |lo| <= ulp(hi) / 2, and
    |x - hi - lo| <= 2^-48 |x|.

    Raises OverflowError for a finite |x| that rounds past the fp32 range.
    `underflow="raise"` raises ValueError for 0 < |x| < FF_MIN_FULL (where lo,
    or hi, is subnormal and precision is lost); "keep" keeps the gradual
    IEEE result. inf / nan give hi = x (as float32), lo = 0.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    bits = x.view(np.uint64)
    sign = (bits >> np.uint64(63)).astype(np.uint32)
    E = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    frac = bits & np.uint64((1 << 52) - 1)
    finite = E < 0x7FF
    nonzero = finite & ((E > 0) | (frac > 0))
    e = E - 1023  # unbiased exponent of a normal x
    full = nonzero & (e >= -126) & (e <= 127)  # hi is a normal fp32 number
    m = frac | np.uint64(1 << 52)
    keep = (m >> np.uint64(29)).astype(np.int64)  # 24 bits
    rem = (m & np.uint64((1 << 29) - 1)).astype(np.int64)
    half = 1 << 28
    up = (rem > half) | ((rem == half) & ((keep & 1) == 1))
    keep = keep + up
    carry = keep == (1 << 24)
    keep = np.where(carry, 1 << 23, keep)
    eh = e + carry
    if bool((nonzero & (eh > 127)).any()):
        bad = x[nonzero & (eh > 127)]
        raise OverflowError(
            f"float-float: |x| = {abs(bad.flat[0]):.6g} is outside the fp32 range (< 2^128)"
        )
    if underflow == "raise" and bool((nonzero & (np.abs(x) < FF_MIN_FULL)).any()):
        raise ValueError(
            f"float-float: 0 < |x| < 2^-102 (FF_MIN_FULL) loses precision ({underflow=})"
        )
    if underflow not in ("keep", "raise"):
        raise ValueError(f"underflow must be 'keep' or 'raise'; got {underflow!r}")
    hb = (
        (sign << np.uint32(31))
        | (np.clip(eh + 127, 0, 255).astype(np.uint32) << np.uint32(23))
        | (np.clip(keep - (1 << 23), 0, None).astype(np.uint32))
    )
    hi = np.where(full, hb, np.uint32(0)).view(np.float32)
    # lo = RN32((rem - up 2^29) 2^(e - 52)), signed: exact in float64 first.
    r = (rem - up.astype(np.int64) * (1 << 29)).astype(np.float64)
    lo64 = np.ldexp(r, np.where(full, e - 52, 0).astype(np.int32))
    lo64 = np.where(sign.astype(bool) & (r != 0), -lo64, lo64)  # x - hi = +0 when exact
    lo = np.where(full, lo64.astype(np.float32), np.float32(0))
    # Outside the normal fp32 range of hi (tiny x, zeros, inf / nan): the
    # IEEE conversions directly (the same canonical pair, gradual underflow).
    other = ~full
    if bool(other.any()):
        h2 = x.astype(np.float32)
        with np.errstate(invalid="ignore"):
            l2 = np.where(finite, (x - h2.astype(np.float64)).astype(np.float32), np.float32(0))
        hi = np.where(other, h2, hi)
        lo = np.where(other, l2, lo)
    return hi, lo


def from_fp64(x, *, underflow: str = "keep", device=None) -> FF:
    """Float-float of a float64 tensor or array, exactly as `split_bits`.

    Done on the host (a device float64 tensor is first copied to the host,
    which is exact). `device` places the result (default: x's device)."""
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
    hi, lo = split_bits(_as_f64_numpy(x), underflow=underflow)
    return FF(torch.from_numpy(hi).to(device), torch.from_numpy(lo).to(device))


def to_fp64(x: FF | torch.Tensor, exp=None) -> torch.Tensor:
    """float64 RN(hi + lo) on the host, times 2^exp if given (exact ldexp,
    `exp` an integer array broadcastable to the shape)."""
    if isinstance(x, torch.Tensor):
        v = x.detach().to("cpu").to(F64).numpy()
    else:
        v = x.hi.detach().to("cpu").to(F64).numpy() + x.lo.detach().to("cpu").to(F64).numpy()
    if exp is not None:
        v = np.ldexp(v, np.asarray(exp, dtype=np.int32))
    return torch.from_numpy(np.ascontiguousarray(v))


def as_ff(x) -> FF:
    """FF from an FF, a (hi, lo) pair, a float32 tensor (lo = 0) or float64."""
    if isinstance(x, FF):
        return x
    if isinstance(x, tuple) and len(x) == 2:
        return FF(*x)
    if isinstance(x, torch.Tensor) and x.dtype == F32:
        return FF(x, torch.zeros_like(x))
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return from_fp64(x)
    raise TypeError(f"cannot make a float-float from {type(x).__name__}")


# ── The fixed reduction layout ───────────────────────────────────────────────


class Plan(NamedTuple):
    """Reduction layout of an n-vector: (P, F) rows, F = NCH * tpc * W."""

    n: int
    F: int  # padded columns per partition
    W: int  # tile width (lanes per partition)
    tpc: int  # tiles per chunk


def plan(n: int) -> Plan:
    """The reduction layout for length n (depends on n only): the fewest
    tiles per chunk with W <= W_MAX, then the narrowest W (a multiple of
    W_ALIGN) that holds the vector, so the padding is under NCH tpc W_ALIGN
    columns per partition."""
    if n < 0:
        raise ValueError("n must be >= 0")
    F = max(1, -(-n // P))
    tpc = -(-F // (NCH * W_MAX))
    cols = -(-F // (NCH * tpc))  # columns per tile, before alignment
    W = -(-cols // W_ALIGN) * W_ALIGN
    return Plan(n, NCH * tpc * W, W, tpc)


def plan_from_cols(F: int) -> tuple[int, int]:
    """(W, tpc) of a padded column count F, as the kernels derive it:
    tpc = ceil(F / (NCH W_MAX)), W = F / (NCH tpc) (inverts `plan`)."""
    tpc = -(-F // (NCH * W_MAX))
    return F // (NCH * tpc), tpc


def layout(x: torch.Tensor, pl: Plan | None = None) -> torch.Tensor:
    """x (n,) float32 -> (P, F) zero-padded rows (on x's device). Element i
    sits at (i // F, i % F); zeros add nothing to any reduction."""
    x = x.reshape(-1)
    pl = pl or plan(x.numel())
    pad = P * pl.F - x.numel()
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(P, pl.F)


# Accumulation arms of a reduction. Each lane (partition, column) adds its
# elements' terms in tile order, then the lanes fold in a halving tree:
#   plain     fp32 add                              (1 flop / term)
#   comp      Ogita-Rump-Oishi Sum2 / Dot2: TwoSum into s, error into c
#             (branch-free Neumaier compensation)     (7 flops / term)
#   sloppy    SloppyDWPlusDW                         (11 flops / term)
#   accurate  AccurateDWPlusDW                       (20 flops / term)
# A float32 summand (sum of fp32 x) takes DWPlusFP (10 flops) in both
# double-word arms, so for `sum` they differ only in the fold.
ARMS = ("plain", "comp", "sloppy", "accurate")


def _halves(h, l, w, k):
    return (h[..., :k], l[..., :k]), (h[..., w - k : w], l[..., w - k : w])


def _fold(acc: FF, dim: int = -1, mode: str = "accurate") -> FF:
    """Fixed halving tree over `dim`: width w -> w - w // 2 by adding the last
    w // 2 entries onto the first, in arm `mode`. Keeps the dim at size 1."""
    h, l = acc.hi.movedim(dim, -1).clone(), acc.lo.movedim(dim, -1).clone()
    w = h.shape[-1]
    while w > 1:
        k = w // 2
        a, b = _halves(h, l, w, k)
        if mode == "plain":
            s = FF(a[0] + b[0], a[1])
        elif mode == "comp":
            sh, q = two_sum(a[0], b[0])
            c = a[1] + b[1]
            s = FF(sh, c + q)
        elif mode == "sloppy":
            s = ff_add_sloppy(a, b)
        else:
            s = ff_add(a, b)
        h[..., :k], l[..., :k] = s.hi, s.lo
        w -= k
    return FF(h[..., :1].movedim(-1, dim), l[..., :1].movedim(-1, dim))


def _term(kind: str, mode: str, xs: list, scale=None):
    """The term one element contributes, as the kernels compute it: an fp32
    tensor (plain arm, or a float32 summand) or an FF pair."""
    if kind == "sum32":
        return xs[0]
    if kind == "sumff":
        return FF(xs[0], xs[1])
    if kind == "dot32":
        if mode == "plain":
            return xs[0] * xs[1]
        return FF(*two_prod(xs[0], xs[1]))
    if kind == "dotff":
        return ff_mul((xs[0], xs[1]), (xs[2], xs[3]))
    if kind == "nrm32":
        a = xs[0] * scale
        if mode == "plain":
            return a * a
        return FF(*two_prod(a, a))
    if kind == "nrmff":
        a = xs[0] * scale
        b = xs[1] * scale
        p, e = two_prod(a, a)
        t = a * b
        t = t + t
        e = e + t
        return FF(*fast_two_sum(p, e))
    raise ValueError(kind)


def _accumulate(acc: FF, term, mode: str) -> FF:
    """One lane step of arm `mode` (see ARMS)."""
    if mode == "plain":
        t = term if isinstance(term, torch.Tensor) else term[0]
        return FF(acc.hi + t, acc.lo)
    if mode == "comp":
        if isinstance(term, torch.Tensor):
            s, q = two_sum(acc.hi, term)
            return FF(s, acc.lo + q)
        s, q = two_sum(acc.hi, term[0])
        u = q + term[1]
        return FF(s, acc.lo + u)
    if isinstance(term, torch.Tensor):
        return ff_add_fp(acc, term)
    if mode == "sloppy":
        return ff_add_sloppy(acc, term)
    return ff_add(acc, term)


_ARITY = {"sum32": 1, "sumff": 2, "dot32": 2, "dotff": 4, "nrm32": 1, "nrmff": 2}


def _nrm_scale(rows: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Per (partition, chunk) power-of-two scale from the largest |hi|:
    s = 2^(127 - Ec), Ec = min(biased exponent of the max, 253) (so s is a
    normal fp32 number; a zero or subnormal maximum gives s = 2^127); k = Ec - 127.
    `rows[0]` is (P, NCH, tpc, W) float32. Returns s (P, NCH, 1), k (P, NCH)."""
    a = (rows[0].contiguous().view(torch.int32) & 0x7FFFFFFF).view(F32)
    m = a.amax(dim=(2, 3))  # exact, order-free
    E = (m.view(torch.int32) >> 23).to(F32)
    Ec = torch.minimum(E, torch.tensor(253.0))
    s = ((254.0 - Ec).to(torch.int32) << 23).view(F32)
    return s[:, :, None], Ec - 127.0


def reference_partials(kind: str, xs: Sequence[torch.Tensor], accumulate: str = "accurate"):
    """The kernels' output on the host, bit for bit: float32 (P, 2 NCH)
    (hi, lo of chunk c in columns 2c, 2c + 1), or (P, 3 NCH) for the norms
    (third column: the chunk's scale exponent k)."""
    if len(xs) != _ARITY[kind]:
        raise ValueError(f"{kind} takes {_ARITY[kind]} operands")
    if accumulate not in ARMS:
        raise ValueError(f"accumulate must be one of {ARMS}; got {accumulate!r}")
    pl = plan(xs[0].numel())
    rows = [layout(t.to("cpu", F32), pl).view(P, NCH, pl.tpc, pl.W) for t in xs]
    scale = k = None
    if kind.startswith("nrm"):
        scale, k = _nrm_scale(rows)
    zero = torch.zeros((P, NCH, pl.W), dtype=F32)
    acc = FF(zero, zero.clone())
    for t in range(pl.tpc):
        term = _term(kind, accumulate, [r[:, :, t, :] for r in rows], scale)
        acc = _accumulate(acc, term, accumulate)
    f = _fold(acc, mode=accumulate)
    cols = [f.hi[..., 0], f.lo[..., 0]] + ([k] if k is not None else [])
    return torch.stack(cols, dim=-1).reshape(P, -1).contiguous()


def finish_host(parts: torch.Tensor, kind: str = "sum") -> float:
    """Exactly rounded float64 sum of the chunk partials (`math.fsum`), the
    hybrid finish. For the norms, sum (hi + lo) 4^k, then sqrt."""
    a = parts.detach().to("cpu").to(F64).numpy()
    if not bool(np.isfinite(a).all()):
        return math.nan  # inf or nan in, nan out (see the module docstring)
    if kind.startswith("nrm"):
        a = a.reshape(P, NCH, 3)
        k2 = (2 * a[..., 2]).astype(np.int32)
        vals = np.concatenate([np.ldexp(a[..., 0], k2).ravel(), np.ldexp(a[..., 1], k2).ravel()])
        return math.sqrt(math.fsum(vals.tolist()))
    return math.fsum(a.ravel().tolist())


def finish_ff(parts: torch.Tensor) -> FF:
    """Float-float fold of the (P, 2 NCH) partials: chunks, then partitions,
    in fixed AccurateDWPlusDW halving trees (what a device-side fold computes)."""
    a = parts.detach().to("cpu", F32).reshape(P, NCH, 2)
    f = _fold(FF(a[..., 0], a[..., 1]), dim=1)  # (P, 1)
    f = _fold(FF(f.hi[:, 0], f.lo[:, 0]), dim=0)  # (1,)
    return FF(f.hi.reshape(()), f.lo.reshape(()))


# ── Backend routing ──────────────────────────────────────────────────────────


def _dispatch():
    from .nki import dispatch

    return dispatch


def _route(tensors, backend: str | None) -> str:
    """ "pytorch" (the reference), "nki" (native kernels) or "sim" (nki.simulate)."""
    d = _dispatch()
    b = backend or d.get_backend()
    if b not in ("auto", "pytorch", "nki"):
        raise ValueError(f"backend must be auto, pytorch or nki; got {b!r}")
    if b == "pytorch" or not d.HAS_NKI:
        if b == "nki":
            raise RuntimeError("trnblas.fp64: backend 'nki' needs the nki package")
        return "pytorch"
    if d._use_simulator():
        return "sim"
    on_dev = all(t.device.type == "neuron" for t in tensors)
    if (on_dev or b == "nki") and d._get_dispatch_path() == "native":
        return "nki"
    if b == "nki":
        raise RuntimeError(
            "trnblas.fp64: the NKI kernels need the native TorchNeuron path (Trn2/Trn3) "
            "or TRNBLAS_USE_SIMULATOR=1"
        )
    return "pytorch"


def _operands(kind: str, args) -> tuple[str, list]:
    """Normalise the inputs of sum / dot / nrm2 to float32 tensors: fp32 in,
    the fp32 kind; FF or float64 in, the float-float kind."""
    ffs = [isinstance(a, (FF, tuple)) or a.dtype == F64 for a in args]
    if not any(ffs):
        for a in args:
            if a.dtype != F32:
                raise TypeError(f"trnblas.fp64 takes float32, float64 or FF; got {a.dtype}")
        return kind + "32", [a.reshape(-1) for a in args]
    out = []
    for a in args:
        f = as_ff(a)
        out += [f.hi.reshape(-1), f.lo.reshape(-1)]
    if len({t.numel() for t in out}) != 1:
        raise ValueError("operands must have the same number of elements")
    return kind + "ff", out


def partials(kind: str, xs: Sequence[torch.Tensor], backend=None, accumulate="accurate"):
    """Chunk partials (P, 2 NCH) or (P, 3 NCH) on the selected backend."""
    if len({t.numel() for t in xs}) != 1:
        raise ValueError("operands must have the same number of elements")
    route = _route(xs, backend)
    if route == "pytorch":
        return reference_partials(kind, xs, accumulate)
    from .nki import fp64_kernels as K

    return K.run_partials(kind, xs, accumulate, simulate=(route == "sim"))


def _finish(parts: torch.Tensor, kind: str, finish: str):
    if finish == "host":
        return torch.tensor(finish_host(parts, kind), dtype=F64)
    if finish == "ff":
        if kind.startswith("nrm"):
            raise ValueError("nrm2 has no float-float finish (per-chunk scales); use 'host'")
        return finish_ff(parts)
    if finish == "partials":
        return parts
    raise ValueError(f"finish must be 'host', 'ff' or 'partials'; got {finish!r}")


def _check_arm(accumulate: str) -> None:
    if accumulate not in ARMS:
        raise ValueError(f"accumulate must be one of {ARMS}; got {accumulate!r}")


def sum(x, *, backend=None, finish: str = "host", accumulate: str = "accurate"):
    """Float-float sum of x (float32, float64 or FF), in the fixed order.

    finish="host" (default): float64 0-d tensor, the exactly rounded sum of
    the device's chunk partials (hybrid). "ff": an FF scalar from a fixed
    float-float fold. "partials": the raw (P, 2 NCH) partials.
    `accumulate` picks the arm (`ARMS`): "accurate" (default), "sloppy",
    "comp" (Sum2) or "plain" (fp32), for measurement."""
    _check_arm(accumulate)
    kind, xs = _operands("sum", [x])
    return _finish(partials(kind, xs, backend, accumulate), kind, finish)


def dot(x, y, *, backend=None, finish: str = "host", accumulate: str = "accurate"):
    """Float-float dot product x . y (float32, float64 or FF operands).

    float32 operands: exact TwoProd products and float-float accumulation, a
    compensated dot (Ogita, Rump and Oishi) kept as a double word. float64 /
    FF operands: DWTimesDW products. `accumulate`: as for `sum` ("comp" is
    Dot2, "plain" is an fp32 dot in the same order). See `sum` for `finish`."""
    _check_arm(accumulate)
    kind, xs = _operands("dot", [x, y])
    return _finish(partials(kind, xs, backend, accumulate), kind, finish)


def nrm2(x, *, backend=None, accumulate: str = "accurate") -> torch.Tensor:
    """Euclidean norm, float64 0-d tensor. Each (partition, chunk) group is
    scaled by a power of two from its largest |x| before squaring, so there
    is no overflow or underflow for any finite fp32 / float-float input."""
    _check_arm(accumulate)
    kind, xs = _operands("nrm", [x])
    return _finish(partials(kind, xs, backend, accumulate), kind, "host")


def axpy(alpha, x, y, *, backend=None) -> FF:
    """Float-float alpha * x + y, elementwise (FF result).

    alpha: a Python float, float64 0-d tensor or FF scalar; x, y: float32,
    float64 or FF. alpha x is DWTimesFP / DWTimesDW, + y is DWPlusFP /
    AccurateDWPlusDW. Elementwise, so it is LNC-invariant on any backend.
    The NKI kernel is a follow-up; every backend runs this reference."""
    if isinstance(alpha, FF):
        a = alpha
    else:
        a = from_fp64(torch.tensor(float(alpha), dtype=F64))
    xf = isinstance(x, (FF, tuple)) or x.dtype == F64
    yf = isinstance(y, (FF, tuple)) or y.dtype == F64
    xx = as_ff(x) if xf else x
    dev = xx.hi.device if xf else xx.device
    ah, al = a.hi.to(dev), a.lo.to(dev)
    if xf:
        prod = ff_mul((ah.expand_as(xx.hi), al.expand_as(xx.hi)), xx)
    else:
        prod = ff_mul_fp((ah.expand_as(xx), al.expand_as(xx)), xx)
    if yf:
        return ff_add(prod, as_ff(y))
    return ff_add_fp(prod, y)


def combine(parts, *, exp=None, out: str = "fp64"):
    """Sum a list of float32 / float-float partials elementwise in a fixed
    pairwise tree (AccurateDWPlusDW), the shape of `gemm_fp64`'s combine.

    `parts`: a sequence of same-shape float32 tensors or FF pairs (`stacked`
    views a (2, ...) Ozaki kernel output as one). `exp`: optional
    integer exponents, broadcastable to the shape, applied exactly at the
    float64 conversion (the row / column scales 2^(eA_i + eB_j + off)).
    out="fp64": float64 tensor RN(hi + lo) 2^exp; out="ff": the FF sum.

    `combine_exact` is the integer-emulated alternative (exactly rounded)."""
    terms = [_part_ff(p) for p in parts]
    if not terms:
        raise ValueError("combine needs at least one part")
    while len(terms) > 1:
        k = len(terms) // 2
        w = len(terms)
        terms = [ff_add(terms[i], terms[w - k + i]) for i in range(k)] + terms[k : w - k]
    res = terms[0]
    if out == "ff":
        return res
    if out != "fp64":
        raise ValueError("out must be 'fp64' or 'ff'")
    return to_fp64(res, exp)


def _part_ff(p) -> FF:
    if isinstance(p, FF):
        return p
    if isinstance(p, tuple):
        return FF(*p)
    if isinstance(p, torch.Tensor) and p.dtype == F32:
        return FF(p, torch.zeros_like(p))
    raise TypeError("combine parts must be float32 tensors or FF pairs")


def stacked(hilo: torch.Tensor) -> FF:
    """FF view of a (2, ...) stacked (hi, lo) float32 tensor (Ozaki output)."""
    return FF(hilo[0], hilo[1])


# ── Integer-emulated exact path (raw fp64 bits, int32 limbs) ─────────────────
#
# A float64 x is m 2^(E - 1075) (m the 53-bit significand, E the biased
# exponent; subnormals take E = 1). Relative to a window top T (|x| < 2^T for
# every element) and L limbs of B bits, x is cut into B-bit fixed-point digits
#   Q_j = floor(|x| 2^(B (j + 1) - T)) mod 2^B,  j = 0 .. L - 1,
# exact when the window reaches x's last bit (T - B L <= E - 1075). Each lane
# (partition, column) adds sign * Q_j into its int32 limb j over the column
# tiles of its chunk: |sum| < tpc 2^B, inside int32 for tpc <= 2^(31 - B).
# Carries then normalise limbs 1 .. L - 1 into [0, 2^B) and a top limb (L)
# takes what overflows limb 0; the W lanes fold by int32 adds in the halving
# tree, normalising after every level. The host finishes the (P, NCH, L + 1)
# limbs as one Python integer, so the result is the exactly rounded sum.
# Every step is an exact int32 add (exact mod 2^32 on GpSimd), an arithmetic
# shift or a mask: the device pitfalls (int arithmetic through fp32,
# aws-neuron-sdk#1308; transposes, #1466; mask-index ops above 2^24, #1467)
# are not touched. The digit extraction is the raw-bits floor chain of the
# Ozaki device split (trnblas#125), with B = 21.

INT_LIMB_BITS = 21  # sum digits: the Ozaki split's 21-bit chunks
DOT_LIMB_BITS = 12  # dot digits: 12 x 12-bit products are exact in fp32
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


class _Bits(NamedTuple):
    sign: np.ndarray  # int64 0 / 1
    m: np.ndarray  # uint64 significand (with the implicit bit)
    lsb: np.ndarray  # int64 exponent of the significand's last bit
    top: int  # smallest T with |x| < 2^T for every element (or None)
    low: int  # smallest exponent of a nonzero last bit (or None)


def _bits(x) -> _Bits:
    a = _as_f64_numpy(x).ravel()
    if not bool(np.isfinite(a).all()):
        raise ValueError("exact path: inf / nan input")
    b = a.view(np.uint64)
    sign = (b >> np.uint64(63)).astype(np.int64)
    E = ((b >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    f = b & np.uint64((1 << 52) - 1)
    m = np.where(E > 0, f | np.uint64(1 << 52), f)
    lsb = np.maximum(E, 1) - 1075
    nz = m != 0
    if not bool(nz.any()):
        return _Bits(sign, m, lsb, None, None)
    mm = m[nz].astype(np.int64)  # < 2^53: exact as float64
    nbits = np.frexp(mm.astype(np.float64))[1].astype(np.int64)  # bit length
    tz = np.frexp((mm & -mm).astype(np.float64))[1].astype(np.int64) - 1  # trailing zeros
    top = int((lsb[nz] + nbits).max())
    low = int((lsb[nz] + tz).min())
    return _Bits(sign, m, lsb, top, low)


def int_limbs_needed(x, bits: int = INT_LIMB_BITS) -> int:
    """Limbs for an exact window over x's nonzero bits (spread + 1 bits)."""
    b = x if isinstance(x, _Bits) else _bits(x)
    if b.top is None:
        return 1
    return max(1, -(-(b.top - b.low) // bits))


def _digits(b: _Bits, T: int, L: int, B: int) -> np.ndarray:
    """(L, n) int64 signed digits sign * Q_j of the window [T - B L, T)."""
    n = b.m.size
    out = np.zeros((L, n), dtype=np.int64)
    mask = np.uint64((1 << B) - 1)
    for j in range(L):
        # bit position of digit j's last bit, relative to x's last bit
        r = (T - B * (j + 1)) - b.lsb
        right = r >= 0
        rr = np.clip(r, 0, 63).astype(np.uint64)
        q_r = np.where(r >= 64, np.uint64(0), (b.m >> rr) & mask)
        ll = np.clip(-r, 0, B).astype(np.uint64)
        q_l = np.where(-r >= B, np.uint64(0), (b.m & (mask >> ll)) << ll)
        q = np.where(right, q_r, q_l).astype(np.int64)
        out[j] = np.where(b.sign == 1, -q, q)
    return out


def _carry(limbs: np.ndarray, B: int) -> np.ndarray:
    """Carry pass over axis 0: limbs 0 .. L - 1 are the digits, most
    significant first, normalised into [0, 2^B); index L is the signed
    overflow limb above limb 0. An arithmetic shift and a subtract per limb."""
    L = limbs.shape[0] - 1
    out = limbs.copy()
    for j in range(L - 1, -1, -1):
        c = out[j] >> B  # arithmetic shift: floor division
        out[j] = out[j] - (c << B)
        if j:
            out[j - 1] += c
        else:
            out[L] += c
    return out


def _check32(a: np.ndarray, what: str) -> None:
    if bool((a < INT32_MIN).any() or (a > INT32_MAX).any()):
        raise OverflowError(f"exact path: int32 limb overflow in {what}")


def _int_reduce(digits: np.ndarray, B: int, n: int) -> np.ndarray:
    """The device's lane accumulation and fold of (L, n) digits in the
    `plan(n)` layout, int32 throughout. Returns (P, NCH, L + 1) limbs."""
    L = digits.shape[0]
    pl = plan(n)
    d = np.zeros((L, P * pl.F), dtype=np.int64)
    d[:, :n] = digits
    d = d.reshape(L, P, NCH, pl.tpc, pl.W)
    acc = np.zeros((L + 1, P, NCH, pl.W), dtype=np.int64)
    acc[:L] = d.sum(axis=3)  # integer adds: order-free and exact
    _check32(acc, "lane accumulation")
    acc = _carry(acc, B)
    w = pl.W
    while w > 1:
        k = w // 2
        acc[..., :k] = acc[..., :k] + acc[..., w - k : w]
        acc = _carry(acc, B)
        _check32(acc, "fold")
        w -= k
    return np.moveaxis(acc[..., 0], 0, -1)


def _limbs_value(limbs: np.ndarray, B: int) -> int:
    """Python integer sum_{p,c} (sum_j limb_j 2^(B (L-1-j)) + top 2^(B L))."""
    L = limbs.shape[-1] - 1
    tot = 0
    for j in range(L):
        tot += int(limbs[..., j].sum()) << (B * (L - 1 - j))
    return tot + (int(limbs[..., L].sum()) << (B * L))


def _round_fp64(num: int, e2: int) -> float:
    """Exactly rounded float64 of num * 2^e2."""
    if num == 0:
        return 0.0
    if e2 >= 0:
        return float(Fraction(num * (1 << e2)))
    return float(Fraction(num, 1 << -e2))


def exact_sum(x, *, limbs: int | None = None) -> torch.Tensor:
    """Exactly rounded float64 sum of float64 x, by int32 limb accumulation
    of 21-bit digits of the raw bits (CPU reference of the device algorithm).

    `limbs=None` takes enough for the exponent spread of x (exact); a smaller
    count truncates below the window (|error| <= n 2^(T - 21 L)). Raises
    ValueError for inf / nan."""
    b = _bits(x)
    n = b.m.size
    if b.top is None:
        return torch.tensor(0.0, dtype=F64)
    B = INT_LIMB_BITS
    L = limbs or int_limbs_needed(b, B)
    T = b.top
    pl = plan(n)
    if pl.tpc > (1 << (31 - B)) - 1:
        raise ValueError("exact_sum: vector too long for one carry-free lane pass")
    lim = _int_reduce(_digits(b, T, L, B), B, n)
    return torch.tensor(_round_fp64(_limbs_value(lim, B), T - B * L), dtype=F64)


def exact_dot(x, y, *, limbs: tuple[int, int] | None = None) -> torch.Tensor:
    """Exactly rounded float64 dot product of float64 x, y (CPU reference).

    Each operand is cut into 12-bit digits over its own exponent window;
    digit products (< 2^24) are exact in fp32 and convert exactly to int32,
    and product (s, t) is added into limb s + t of base 2^12. With L_x and
    L_y digits per element that is L_x L_y products per element, so it is
    slow on the device (a GEMV on VectorE); it is exact for any condition
    number. Lanes carry-normalise every `_DOT_CARRY_TILES` column tiles."""
    bx, by = _bits(x), _bits(y)
    n = bx.m.size
    if by.m.size != n:
        raise ValueError("operands must have the same number of elements")
    if bx.top is None or by.top is None:
        return torch.tensor(0.0, dtype=F64)
    B = DOT_LIMB_BITS
    Lx = limbs[0] if limbs else int_limbs_needed(bx, B)
    Ly = limbs[1] if limbs else int_limbs_needed(by, B)
    dx = _digits(bx, bx.top, Lx, B)
    dy = _digits(by, by.top, Ly, B)
    L = Lx + Ly - 1
    pl = plan(n)
    pad = P * pl.F - n
    dx = np.pad(dx, ((0, 0), (0, pad))).reshape(Lx, P, NCH, pl.tpc, pl.W)
    dy = np.pad(dy, ((0, 0), (0, pad))).reshape(Ly, P, NCH, pl.tpc, pl.W)
    acc = np.zeros((L + 1, P, NCH, pl.W), dtype=np.int64)
    for t in range(pl.tpc):
        for s in range(Lx):
            for u in range(Ly):
                acc[s + u] += dx[s, :, :, t] * dy[u, :, :, t]
        if (t + 1) % _DOT_CARRY_TILES == 0 or t == pl.tpc - 1:
            _check32(acc, "dot lane accumulation")
            acc = _carry(acc, B)
    w = pl.W
    while w > 1:
        k = w // 2
        acc[..., :k] = acc[..., :k] + acc[..., w - k : w]
        acc = _carry(acc, B)
        _check32(acc, "dot fold")
        w -= k
    lim = np.moveaxis(acc[..., 0], 0, -1)
    # digit s of x has weight 2^(Tx - B (s + 1)); product limb q = s + u has
    # weight 2^(Tx + Ty - B (q + 2)); limb index q -> position L - 1 - q.
    e2 = bx.top + by.top - B * (L + 1)
    return torch.tensor(_round_fp64(_limbs_value(lim, B), e2), dtype=F64)


# Products are < 2^24 and limb s + u collects up to min(Lx, Ly) of them per
# element, so int32 limbs take 2^31 / (L 2^24) = 128 / L tiles between carry
# passes; 8 tiles keeps that safe for up to 16 digits per operand.
_DOT_CARRY_TILES = 8


def combine_exact(parts, *, exp=None) -> torch.Tensor:
    """Exactly rounded float64 of the elementwise sum of float32 / FF parts
    (times 2^exp): the gemm_fp64 combine with no rounding before the last.

    CPU reference of the integer-emulated combine: every part value is an
    fp32 number, so the sum is a dyadic rational computed exactly here and
    rounded once (on the device: 21-bit digits of each element's window and
    int32 limb adds, as `exact_sum`, per element instead of per vector)."""
    vals = []
    for p in parts:
        f = _part_ff(p)
        vals += [f.hi.detach().to("cpu").to(F64).numpy(), f.lo.detach().to("cpu").to(F64).numpy()]
    stack = np.stack(vals).reshape(len(vals), -1)
    out = np.empty(stack.shape[1], dtype=np.float64)
    for i in range(stack.shape[1]):
        out[i] = math.fsum(stack[:, i].tolist())  # exactly rounded
    shape = _part_ff(parts[0]).hi.shape
    res = out.reshape(shape)
    if exp is not None:
        # fsum's rounding happens before the scaling; exact unless the scaled
        # result leaves the normal float64 range.
        res = np.ldexp(res, np.broadcast_to(np.asarray(exp, dtype=np.int32), shape))
    return torch.from_numpy(np.ascontiguousarray(res))


__all__ = [
    "FF",
    "FF_MAX",
    "FF_MIN_FULL",
    "ARMS",
    "NCH",
    "Plan",
    "SPLIT_MAX",
    "U_FF",
    "as_ff",
    "axpy",
    "combine",
    "combine_exact",
    "dot",
    "exact_dot",
    "exact_sum",
    "fast_two_sum",
    "ff_add",
    "ff_add_fp",
    "ff_add_sloppy",
    "ff_mul",
    "ff_mul_fp",
    "finish_ff",
    "finish_host",
    "from_fp64",
    "int_limbs_needed",
    "layout",
    "nrm2",
    "partials",
    "plan",
    "reference_partials",
    "split",
    "split_bits",
    "stacked",
    "sum",
    "to_fp64",
    "two_prod",
    "two_prod_fma",
    "two_sum",
]
