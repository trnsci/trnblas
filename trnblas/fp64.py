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

Accumulation arms (`ARMS`, for measurement): "plain" fp32, "comp" (Sum2 /
Dot2 of Ogita, Rump and Oishi: TwoSum compensation, i.e. branch-free
Neumaier), "sloppy" (SloppyDWPlusDW; Stloukal and Jezequel, arXiv:2611.08371,
report that sloppy double-word addition suffices for their workload) and
"accurate" (AccurateDWPlusDW, the default).

Which to use (CPU accuracy table on trnsci/trnsci#169): double-word gives
48-52 correct bits on well-conditioned reductions (CG r.r and p.Ap, norms,
DFT grid quadratures, same-sign energy sums), where plain fp32 gives 20-30;
its error grows like 2^-48 cond, about 5 bits worse than an fp64 sum, so for
ill-conditioned dots (cond >~ 1e8) and for the gemm_fp64 combine use the
exact path. The integer path is exact at any condition number; its cost is
the number of limbs, set by the exponent spread of the data.

Extension path: triple-word fp32 (~72 bits) with the branch-free multi-word
algorithms of arXiv:2607.11391 would cover the gap between double-word and
the exact path (cancelling energy sums, cond up to ~2^20) at about twice the
double-word cost; the layout, arms and kernels here take it as a fourth
accumulation arm. Not implemented in step 1.
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
    `rows[0]` is (P, NCH, tpc, W) float32. Returns s (P, NCH, 1), k (P, NCH)
    (per column, with a trailing R axis, for (P, NCH, tpc, W, R) rows)."""
    a = (rows[0].contiguous().view(torch.int32) & 0x7FFFFFFF).view(F32)
    m = a.amax(dim=(2, 3))  # exact, order-free
    E = (m.view(torch.int32) >> 23).to(F32)
    Ec = torch.minimum(E, torch.tensor(253.0))
    s = ((254.0 - Ec).to(torch.int32) << 23).view(F32)
    # (P, NCH, 1), or (P, NCH, 1, R) for the column-batched (P, NCH, tpc, W, R)
    return s.unsqueeze(2), Ec - 127.0


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


def _operands(kind: str, args, exact: bool = False) -> tuple[str, list]:
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
    """Chunk partials (P, 2 NCH) or (P, 3 NCH) on the selected backend
    (int32 (P, NCH (EXACT_LIMBS + 3)) for the exact arm)."""
    if len({t.numel() for t in xs}) != 1:
        raise ValueError("operands must have the same number of elements")
    route = _route(xs, backend)
    if route == "pytorch":
        if accumulate == "exact":
            return reference_exact_partials(kind, xs)
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
    if accumulate not in ARMS + ("exact",):
        raise ValueError(f"accumulate must be one of {ARMS + ('exact',)}; got {accumulate!r}")


def _exact_finish(parts, kind, xs, finish):
    """Finish of the exact arm: exactly rounded float64 (host fallback when a
    group's window did not hold every term)."""
    if finish == "partials":
        return parts
    if finish != "host":
        raise ValueError("the exact arm finishes on the host ('host' or 'partials')")
    v = finish_exact(parts)
    if v is None:  # sticky: recompute exactly on the host from the operands
        h = [t.detach().to("cpu").to(F64) for t in xs]
        return exact_sum(h[0]) if kind == "sum32" else exact_dot(h[0], h[1])
    return torch.tensor(v, dtype=F64)


def sum(x, *, backend=None, finish: str = "host", accumulate: str = "accurate"):
    """Float-float sum of x (float32, float64 or FF), in the fixed order.

    finish="host" (default): float64 0-d tensor, the exactly rounded sum of
    the device's chunk partials (hybrid). "ff": an FF scalar from a fixed
    float-float fold. "partials": the raw (P, 2 NCH) partials.
    `accumulate` picks the arm (`ARMS`): "accurate" (default), "sloppy",
    "comp" (Sum2) or "plain" (fp32), for measurement; or "exact" (float32
    operands): the integer-limb arm, exactly rounded."""
    _check_arm(accumulate)
    kind, xs = _operands("sum", [x])
    if accumulate == "exact":
        return _exact_finish(partials(kind, xs, backend, accumulate), kind, xs, finish)
    return _finish(partials(kind, xs, backend, accumulate), kind, finish)


def dot(x, y, *, backend=None, finish: str = "host", accumulate: str = "accurate"):
    """Float-float dot product x . y (float32, float64 or FF operands).

    float32 operands: exact TwoProd products and float-float accumulation, a
    compensated dot (Ogita, Rump and Oishi) kept as a double word. float64 /
    FF operands: DWTimesDW products. `accumulate`: as for `sum` ("comp" is
    Dot2, "plain" is an fp32 dot in the same order). See `sum` for `finish`."""
    _check_arm(accumulate)
    kind, xs = _operands("dot", [x, y])
    if accumulate == "exact":
        return _exact_finish(partials(kind, xs, backend, accumulate), kind, xs, finish)
    return _finish(partials(kind, xs, backend, accumulate), kind, finish)


def nrm2(x, *, backend=None, accumulate: str = "accurate") -> torch.Tensor:
    """Euclidean norm, float64 0-d tensor. Each (partition, chunk) group is
    scaled by a power of two from its largest |x| before squaring, so there
    is no overflow or underflow for any finite fp32 / float-float input."""
    _check_arm(accumulate)
    if accumulate == "exact":
        raise ValueError("nrm2 has no exact arm")
    kind, xs = _operands("nrm", [x])
    return _finish(partials(kind, xs, backend, accumulate), kind, "host")


# ── Column-batched reductions (trnblas#142) ──────────────────────────────────
#
# `dot_cols(X, Y)` and `nrm2_cols(X)` reduce the R columns of (n, R) operands
# in one kernel launch, for block solvers (trnsolver `cg_block`) that need R
# inner products or norms per iteration. Column j keeps the layout of its own
# reduction: element (i, j) sits at lane (i // F, i % F) of plan(n), as
# `layout(X[:, j])` puts element i, and the columns sit side by side on the
# free axis. Every step of a reduction is elementwise over the lanes and the
# halving tree folds over W alone, so column j gets the fp32 operations of
# `dot(X[:, j], Y[:, j])` / `nrm2(X[:, j])` in the same order, and the same
# bits; the host finish is the same function, per column. The nrm2 scales
# stay per (partition, chunk, column).


def layout_cols(X: torch.Tensor, pl: Plan | None = None) -> torch.Tensor:
    """X (n, R) float32 -> (P, F, R) zero-padded rows (on X's device): element
    (i, j) at (i // F, i % F, j), so [:, :, j] is `layout(X[:, j], pl)`. A
    row-major X needs no copy beyond the padding."""
    n, R = X.shape
    pl = pl or plan(n)
    pad = P * pl.F - n
    if pad:
        X = torch.nn.functional.pad(X, (0, 0, 0, pad))
    return X.reshape(P, pl.F, R)


def _ncol(kind: str) -> int:
    return 3 if kind.startswith("nrm") else 2


def reference_partials_cols(kind: str, xs: Sequence[torch.Tensor], accumulate: str = "accurate"):
    """`reference_partials` of every column at once, on the host: float32
    (P, ncol NCH, R) with [:, :, j] bitwise equal to
    `reference_partials(kind, [x[:, j] for x in xs], accumulate)` (the same
    fp32 operations, with a trailing column axis). Operands (n, R)."""
    if len(xs) != _ARITY[kind]:
        raise ValueError(f"{kind} takes {_ARITY[kind]} operands")
    if accumulate not in ARMS:
        raise ValueError(f"accumulate must be one of {ARMS}; got {accumulate!r}")
    n, R = xs[0].shape
    pl = plan(n)
    rows = [layout_cols(t.to("cpu", F32), pl).view(P, NCH, pl.tpc, pl.W, R) for t in xs]
    scale = k = None
    if kind.startswith("nrm"):
        scale, k = _nrm_scale(rows)  # (P, NCH, 1, R), (P, NCH, R)
    zero = torch.zeros((P, NCH, pl.W, R), dtype=F32)
    acc = FF(zero, zero.clone())
    for t in range(pl.tpc):
        term = _term(kind, accumulate, [r[:, :, t] for r in rows], scale)
        acc = _accumulate(acc, term, accumulate)
    f = _fold(acc, dim=2, mode=accumulate)
    cols = [f.hi[:, :, 0], f.lo[:, :, 0]] + ([k] if k is not None else [])
    return torch.stack(cols, dim=2).reshape(P, _ncol(kind) * NCH, R).contiguous()


def reference_exact_partials_cols(kind: str, xs: Sequence[torch.Tensor]) -> torch.Tensor:
    """`reference_exact_partials` per column: int32 (P, NCH (L + 3), R)."""
    R = xs[0].shape[1]
    cols = [reference_exact_partials(kind, [t[:, j] for t in xs]) for j in range(R)]
    if not cols:
        return torch.zeros((P, NCH * EXACT_COLS, 0), dtype=torch.int32)
    return torch.stack(cols, dim=2).contiguous()


def _operands_cols(kind: str, args) -> tuple[str, list]:
    """`_operands` for (n, R) operands: the same kinds, shapes kept."""
    ffs = [isinstance(a, (FF, tuple)) or a.dtype == F64 for a in args]
    if not any(ffs):
        for a in args:
            if a.dtype != F32:
                raise TypeError(f"trnblas.fp64 takes float32, float64 or FF; got {a.dtype}")
        out, kind = list(args), kind + "32"
    else:
        out = []
        for a in args:
            f = as_ff(a)
            out += [f.hi, f.lo]
        kind = kind + "ff"
    if out[0].dim() != 2:
        raise ValueError(f"the column reductions take (n, R) operands; got {tuple(out[0].shape)}")
    if len({tuple(t.shape) for t in out}) != 1:
        raise ValueError("operands must have the same (n, R) shape")
    return kind, out


def partials_cols(kind: str, xs: Sequence[torch.Tensor], backend=None, accumulate="accurate"):
    """Per-column chunk partials (P, ncol NCH, R) on the selected backend
    ([:, :, j] equals `partials` of column j; int32 (P, NCH (L + 3), R) for
    the exact arm). One kernel launch for all R columns."""
    if len({tuple(t.shape) for t in xs}) != 1:
        raise ValueError("operands must have the same (n, R) shape")
    R = xs[0].shape[1]
    if R == 0:
        if accumulate == "exact":
            return torch.zeros((P, NCH * EXACT_COLS, 0), dtype=torch.int32)
        return torch.zeros((P, _ncol(kind) * NCH, 0), dtype=F32)
    route = _route(xs, backend)
    if route == "pytorch":
        if accumulate == "exact":
            return reference_exact_partials_cols(kind, xs)
        return reference_partials_cols(kind, xs, accumulate)
    from .nki import fp64_kernels as K

    return K.run_partials_cols(kind, xs, accumulate, simulate=(route == "sim"))


def _finish_cols(parts: torch.Tensor, kind: str, finish: str):
    if finish == "partials":
        return parts
    R = parts.shape[2]
    if finish == "host":
        return torch.tensor([finish_host(parts[:, :, j], kind) for j in range(R)], dtype=F64)
    if finish == "ff":
        if kind.startswith("nrm"):
            raise ValueError("nrm2 has no float-float finish (per-chunk scales); use 'host'")
        fs = [finish_ff(parts[:, :, j]) for j in range(R)]
        if not fs:
            return FF(torch.zeros(0, dtype=F32), torch.zeros(0, dtype=F32))
        return FF(torch.stack([f.hi for f in fs]), torch.stack([f.lo for f in fs]))
    raise ValueError(f"finish must be 'host', 'ff' or 'partials'; got {finish!r}")


def _exact_finish_cols(parts, kind, xs, finish):
    if finish == "partials":
        return parts
    R = parts.shape[2]
    vals = [
        _exact_finish(parts[:, :, j].contiguous(), kind, [t[:, j] for t in xs], finish)
        for j in range(R)
    ]
    return torch.stack(vals) if vals else torch.zeros(0, dtype=F64)


def dot_cols(X, Y, *, backend=None, finish: str = "host", accumulate: str = "accurate"):
    """Column-wise float-float dot products: X[:, j] . Y[:, j] for (n, R)
    operands (float32, float64 or FF), float64 (R,) by default.

    Column j is bitwise equal to `dot(X[:, j], Y[:, j], ...)` with the same
    arguments: the same layout, fp32 operations and order (see the section
    comment above), and the same host finish. All R columns take one kernel
    launch (`dot` takes one each) and one download of the partials.
    finish="ff": an FF of shape (R,); "partials": the raw (P, 2 NCH, R).
    `accumulate` as for `dot`, including "exact" (float32 operands), whose
    columns finish as `dot`'s: exactly rounded, a column whose window did not
    hold every term recomputed on the host."""
    _check_arm(accumulate)
    kind, xs = _operands_cols("dot", [X, Y])
    parts = partials_cols(kind, xs, backend, accumulate)
    if accumulate == "exact":
        return _exact_finish_cols(parts, kind, xs, finish)
    return _finish_cols(parts, kind, finish)


def nrm2_cols(X, *, backend=None, accumulate: str = "accurate") -> torch.Tensor:
    """Column 2-norms of an (n, R) operand (float32, float64 or FF), float64
    (R,): column j bitwise equal to `nrm2(X[:, j])`, with its own power-of-two
    scale per (partition, chunk). One kernel launch for all R columns."""
    _check_arm(accumulate)
    if accumulate == "exact":
        raise ValueError("nrm2 has no exact arm")
    kind, xs = _operands_cols("nrm", [X])
    return _finish_cols(partials_cols(kind, xs, backend, accumulate), kind, "host")


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


# ── The device's exact arm: fp32 terms into int32 limbs ─────────────────────
#
# `accumulate="exact"` for float32 operands (sum32, dot32): every term (x, or
# both parts p, e of TwoProd(x, y)) is cut into EXACT_LIMBS 21-bit digits of a
# fixed-point window per (partition, chunk) group, and the digits are added
# into int32 lanes (GpSimd int32 add, exact mod 2^32). The window top is
# T = Eg - 126 for Eg the largest biased fp32 exponent of the group's terms,
# so every term has |v| < 2^T, and digit j of v is
#   Q_j = floor(|v| 2^(21 (j + 1) - T)) mod 2^21,
# from r_0 = |v| 2^(21 - T) (two power-of-two scalings, exact) and the floor
# chain Q_j = floor(r_j), r_(j+1) = 2^21 (r_j - Q_j) of the Ozaki device split
# (each r_j < 2^21, so the floor by the corrected fp32 -> uint32 cast is exact).
# A term with bits below the window (r_L != 0), or one so small that all its
# digits are zero (r_0 < 2^-100), sets the group's sticky flag: the result is
# then not exact and the caller recomputes it on the host. The window covers
# 21 L = 126 bits: a dot of fp32 data stays exact while the group's products
# span up to ~78 bits of exponent. After the lane pass one carry pass makes
# digits 0 .. L - 1 canonical ([0, 2^21)) with a signed overflow limb, by an
# arithmetic shift built from logical shifts and masks; the W lanes then fold
# by int32 adds with no further carries (< 2^30). Output per group:
# L + 1 limbs, T and the sticky flag (int32), as (P, NCH (L + 3)).

EXACT_LIMBS = 6
EXACT_COLS = EXACT_LIMBS + 3


def _exact_terms(kind: str, rows: list, t: int) -> list:
    if kind == "sum32":
        return [rows[0][:, :, t, :]]
    p, e = two_prod(rows[0][:, :, t, :], rows[1][:, :, t, :])
    return [p, e]


def reference_exact_partials(kind: str, xs: Sequence[torch.Tensor]) -> torch.Tensor:
    """The exact-arm kernels' output on the host: int32 (P, NCH (L + 3))."""
    if kind not in ("sum32", "dot32"):
        raise ValueError("the exact arm takes float32 sum or dot operands")
    L = EXACT_LIMBS
    pl = plan(xs[0].numel())
    rows = [layout(t.to("cpu", F32), pl).view(P, NCH, pl.tpc, pl.W) for t in xs]
    # group maxima: of |x| (sum) or of |fl(x y)| (dot), as the kernels' pass 1
    lead = rows[0] if kind == "sum32" else rows[0] * rows[1]
    m = (lead.contiguous().view(torch.int32) & 0x7FFFFFFF).amax(dim=(2, 3))
    Eg = (m >> 23).numpy().astype(np.int64)  # (P, NCH)
    T = Eg - 126
    acc = np.zeros((L + 1, P, NCH, pl.W), dtype=np.int64)
    sticky = np.zeros((P, NCH, pl.W), dtype=bool)
    scale = np.ldexp(1.0, (21 - T)[:, :, None])
    for t in range(pl.tpc):
        for v in _exact_terms(kind, rows, t):
            a = np.abs(v.numpy().astype(np.float64))
            r = a * scale  # exact in float64
            sticky |= (a != 0) & (r < 2.0**-100)
            sg = np.where(v.numpy() < 0, -1, 1)
            for j in range(L):
                q = np.floor(r)
                acc[j] += sg * q.astype(np.int64)
                r = (r - q) * 2.0**21
            sticky |= r != 0
    _check32(acc, "exact lane accumulation")
    acc = _carry(acc[list(range(L)) + [L]], 21)  # digits 0 .. L - 1, overflow limb L
    w = pl.W
    while w > 1:
        k = w // 2
        acc[..., :k] = acc[..., :k] + acc[..., w - k : w]
        sticky[..., :k] |= sticky[..., w - k : w]
        _check32(acc, "exact fold")
        w -= k
    out = np.zeros((P, NCH, EXACT_COLS), dtype=np.int64)
    out[:, :, : L + 1] = np.moveaxis(acc[..., 0], 0, -1)
    out[:, :, L + 1] = T
    out[:, :, L + 2] = sticky[..., 0]
    return torch.from_numpy(out.reshape(P, -1).astype(np.int32))


def finish_exact(parts: torch.Tensor):
    """Exactly rounded float64 of the exact arm's partials, or None if a
    group's sticky flag is set (the window did not hold every term)."""
    L = EXACT_LIMBS
    a = parts.detach().to("cpu").numpy().astype(np.int64).reshape(P, NCH, EXACT_COLS)
    if bool(a[:, :, L + 2].any()):
        return None
    total = Fraction(0)
    for T in np.unique(a[:, :, L + 1]):
        sel = a[:, :, L + 1] == T
        num = _limbs_value(a[sel][:, : L + 1], 21)
        total += Fraction(num) * Fraction(2) ** int(T - 21 * L)
    return float(total)


# ── Device fp64 construction: RN64(hi + lo) 2^E as raw bits ─────────────────
#
# The combine of gemm_fp64 is RN64(hi + lo) * 2^(eA_i + eB_j + off) per
# element, today on the host (66 of 181 ms at 4096^3, trnblas#125). The bits
# of that float64 can be built on the device from exact fp32 arithmetic and
# bitvec ops only (`ff_to_fp64_bits` is the arithmetic, op for op):
#   1. (s, t) = TwoSum(hi, lo): s + t = hi + lo exactly, |t| <= ulp(s) / 2.
#   2. In units of the float64 ulp of s's binade, |s + t| = S 2^29 + v with S
#      the 24-bit significand of s and v = t sign(s) 2^(52 - e_s) (exact:
#      times +-1 and a power of two), |v| <= 2^28. RN64 rounds v to an integer q, ties to even (S 2^29
#      is even, so the parity of the sum is that of q); for |v| < 2^23 by the
#      fp32 2^23 trick, above it v is already an integer.
#   3. If |s| is a power of two and |s + t| falls below it (v < -1/4),
#      the result is in the binade below: v doubles and the exponent drops.
#   4. significand = A 2^29 + B with A = S m - [q < 0] (24 bits) and
#      B = q mod 2^29 (two's complement AND); the exponent is
#      e_s + 1023 + E - (m - 1). hi word = sign | exponent << 20 | (A - 2^23)
#      >> 3, lo word = ((A - 2^23) & 7) << 29 | B.
# No integer add goes through fp32: integers are only converted when exact.
# Elements whose result is zero are +-0; elements outside the handled range
# (non-finite, s subnormal or below 2^-75, result exponent outside the normal
# float64 range) are flagged for the host.


def ff_to_fp64_bits(hi, lo, E) -> tuple[np.ndarray, np.ndarray]:
    """float64 bits of RN64(hi + lo) 2^E from float32 hi, lo and integer E
    (|E| < 2^24), by the device's arithmetic. Returns (uint64 bits, flag):
    flag[i] is True where the element is outside the handled range (the bits
    are then meaningless and the caller computes it on the host)."""
    f = np.float32
    h = np.ascontiguousarray(hi, dtype=f)
    l = np.ascontiguousarray(lo, dtype=f)
    Ef = np.broadcast_to(np.asarray(E, dtype=f), h.shape)
    with np.errstate(all="ignore"):
        s = h + l
        bb = s - h
        tt = s - bb
        e1 = h - tt
        e2 = l - bb
        t = e1 + e2
        sb = s.view(np.uint32)
        sign = sb >> np.uint32(31)
        Es = ((sb >> np.uint32(23)) & np.uint32(0xFF)).astype(f)
        frac = sb & np.uint32(0x7FFFFF)
        S = (frac | np.uint32(0x800000)).astype(f)
        fz = (frac.astype(f) == f(0)).astype(f)
        # v1 = t 2^(52 - e_s): scale bits (306 - Es) << 23, valid for Es in [52, 254]
        sc1 = (np.clip(f(306) - Es, 1, 254).astype(np.uint32) << np.uint32(23)).view(f)
        ss = f(1) - f(2) * sign.astype(f)  # +-1: work on |s + t| = |s| + t sign(s)
        v1 = (t * ss) * sc1
        m2 = fz * (v1 < f(-0.25)).astype(f)  # binade drop
        mf = m2 + f(1)
        v = v1 * mf
        av = (v.view(np.uint32) & np.uint32(0x7FFFFFFF)).view(f)
        r = (av + f(8388608.0)) - f(8388608.0)
        small = (av < f(8388608.0)).astype(f)
        neg = (v < f(0)).astype(f)
        sg = f(1) - f(2) * neg
        rs = r * sg
        d = (rs - v) * small
        q = v + d
        qi = q.astype(np.int32)  # exact: an integer below 2^29 in magnitude
        B = qi.view(np.uint32) & np.uint32((1 << 29) - 1)
        borrow = (q < f(0)).astype(f)
        A = S * mf - borrow
        Au = (A - f(8388608.0)).astype(np.uint32)
        E64 = Es + f(896) - m2 + Ef
        bad = (
            ~np.isfinite(s)
            | ~np.isfinite(t)
            | ((s != 0) & ((Es < 52) | (Es > 254)))
            | ((s != 0) & ((E64 < 1) | (E64 > 2046)))
        )
        E64u = np.clip(E64, 0, 2047).astype(np.uint32)
        hiw = (sign << np.uint32(31)) | (E64u << np.uint32(20)) | (Au >> np.uint32(3))
        low = ((Au & np.uint32(7)) << np.uint32(29)) | B
        zero = s == 0
        hiw = np.where(zero, sign << np.uint32(31), hiw)
        low = np.where(zero, np.uint32(0), low)
    bits = (hiw.astype(np.uint64) << np.uint64(32)) | low.astype(np.uint64)
    return bits, bad


def ff_bits_kernel_reference(hilo, ea, eb) -> np.ndarray:
    """What `_fp64_ff_bits_kernel(hilo, ea, eb)` returns, on the host: int32
    (R + 128, 2 N). Rows 0 .. R - 1 are the words of the float64
    RN64(hi + lo) 2^(ea_i + eb_j) (`ff_to_fp64_bits`), and out[R + p, b] = 1
    where an element of row 128 b + p is flagged; the rest of the flag rows
    are 0. The words of a flagged row are unspecified (the caller recombines
    it on the host); at a non-finite element they differ from the kernel's. `hilo` (2, R, N) float32 with R a multiple of 128, `ea` (R, 1)
    and `eb` (1, N) float32 integers. The CPU stand-in of the kernel (tests,
    and the host-only native route)."""

    def host(x):
        return np.asarray(x.detach().to("cpu") if isinstance(x, torch.Tensor) else x)

    hl, ea, eb = host(hilo), host(ea), host(eb)
    _, R, N = hl.shape
    if R % 128:
        raise ValueError(f"rows must be a multiple of 128, got {R}")
    E = ea.reshape(R, 1).astype(np.float32) + eb.reshape(1, N).astype(np.float32)
    bits, bad = ff_to_fp64_bits(hl[0], hl[1], E)
    out = np.zeros((R + 128, 2 * N), dtype=np.int32)
    out[:R] = bits.view(np.int32).reshape(R, 2 * N)
    out[R:, : R // 128] = bad.any(axis=1).astype(np.int32).reshape(R // 128, 128).T
    return out


def combine_bits(hilo, eA, eB, slices: int, M: int, N: int) -> torch.Tensor:
    """`ozaki.combine` (one launch block) through `ff_to_fp64_bits`, with the
    flagged elements computed on the host: float64 (M, N), equal to
    `ozaki.combine(hilo, eA, eB, slices, M, N)` bit for bit."""
    from . import ozaki

    hl = hilo.detach().to("cpu").numpy() if isinstance(hilo, torch.Tensor) else hilo
    E = np.asarray(eA[:M], np.int64)[:, None] + ozaki.exponent_offset(slices)
    E = E + np.asarray(eB[:N], np.int64)[None, :]
    if np.abs(E).max(initial=0) >= 1 << 24:
        return ozaki.combine(torch.from_numpy(hl), eA, eB, slices, M, N)
    bits, bad = ff_to_fp64_bits(hl[0, :M, :N], hl[1, :M, :N], E)
    out = bits.view(np.float64)
    if bool(bad.any()):
        i, j = np.nonzero(bad)
        v = hl[0, i, j].astype(np.float64) + hl[1, i, j].astype(np.float64)
        out[i, j] = np.ldexp(v, E[i, j].astype(np.int32))
    return torch.from_numpy(np.ascontiguousarray(out))


__all__ = [
    "FF",
    "FF_MAX",
    "FF_MIN_FULL",
    "ARMS",
    "EXACT_COLS",
    "EXACT_LIMBS",
    "NCH",
    "Plan",
    "SPLIT_MAX",
    "U_FF",
    "as_ff",
    "axpy",
    "combine",
    "combine_bits",
    "combine_exact",
    "dot",
    "dot_cols",
    "exact_dot",
    "exact_sum",
    "fast_two_sum",
    "ff_bits_kernel_reference",
    "ff_to_fp64_bits",
    "ff_add",
    "ff_add_fp",
    "ff_add_sloppy",
    "ff_mul",
    "ff_mul_fp",
    "finish_exact",
    "finish_ff",
    "finish_host",
    "from_fp64",
    "int_limbs_needed",
    "layout",
    "layout_cols",
    "nrm2",
    "nrm2_cols",
    "partials",
    "partials_cols",
    "plan",
    "reference_exact_partials",
    "reference_exact_partials_cols",
    "reference_partials",
    "reference_partials_cols",
    "split",
    "split_bits",
    "stacked",
    "sum",
    "to_fp64",
    "two_prod",
    "two_prod_fma",
    "two_sum",
]
