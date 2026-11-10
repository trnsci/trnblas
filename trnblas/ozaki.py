"""
fp64-class GEMM by Ozaki splitting on the bf16 Tensor Engine (trnblas#22).

Trainium has no fp64 arithmetic. The Tensor Engine multiplies bf16 operands
exactly and accumulates in fp32 PSUM, so a GEMM over small integers is exact
as long as no partial sum exceeds 2^24. This module holds the host half of
`trnblas.gemm_fp64`:

- `split`: fp64 -> `slices` bf16 integer slices ("digits") of b = 7 bits per
  row of A / column of B, plus one power-of-two exponent per row / column.
  Exact: the slices times their scales sum to the input up to a truncation
  below 2^-(6 + 7 (slices - 1)) of the row / column maximum. On the device
  route the same digits come from the device split (#123);
  `split_from_bits` is its arithmetic on the host.
- `diag_groups`: the static accumulation schedule the kernel unrolls. For each
  diagonal d = s + t of slice pairs (s of A, t of B, d <= slices - 1), the
  slice products share one scale, so their K tiles are concatenated and cut
  into PSUM groups of at most 4096 terms (32 K tiles of 128). Each group sum
  is an exact integer (4096 * 64 * 64 = 2^24).
- `combine`: the kernel's fp32 (hi, lo) pair times the row / column scales,
  in fp64.
- `reference_from_digits`: the kernel's arithmetic on the host, bit for bit:
  exact group sums, each scaled by 2^(7 (slices - 1 - d)) and folded into an
  fp32 TwoSum (hi, lo) pair, diagonals smallest first. Every output element
  sees the same operation sequence whatever the tiling, so the device result
  does not depend on the LNC split and should equal this reference exactly.

Accuracy is set by `slices` (k): about 1e-12 of |A||B| at k = 6, 1e-14 at
k = 7 and 2e-16 to 3e-16 (fp64-BLAS level) at k = 8 on well-scaled data
(measured on Trn2, trnblas#22). The scaling is per row of A and per column of
B, so entries much smaller than their row / column maximum lose relative
precision: wide-dynamic-range data (entries spread over 10^+-4) needs k = 10
to reach host fp64 accuracy. Beyond k ~ 10 the double-fp32 output (about
2^-48 relative) is the floor.
"""

from __future__ import annotations

import numpy as np
import torch

SLICE_BITS = 7  # b: |digit| <= 2^(b-1) = 64, so a product is <= 2^12
TILE_K = 128  # K rows per Tensor Engine pass (partition dim)
GROUP_TERMS = 4096  # 4096 * 2^12 = 2^24: every partial sum is exact in fp32
GROUP_TILES = GROUP_TERMS // TILE_K
MIN_SLICES = 1
MAX_SLICES = 12  # fp32 range: 2^(7 (k-1)) * 2^24 * K stays far below 2^127
DEFAULT_SLICES = 8


def check_slices(slices) -> int:
    """`slices` as an int in [MIN_SLICES, MAX_SLICES], else ValueError."""
    if isinstance(slices, bool) or not isinstance(slices, (int, np.integer)):
        raise TypeError(f"slices must be an int, got {type(slices).__name__}")
    k = int(slices)
    if not MIN_SLICES <= k <= MAX_SLICES:
        raise ValueError(f"slices must be in [{MIN_SLICES}, {MAX_SLICES}], got {k}")
    return k


def n_products(slices: int) -> int:
    """Slice GEMMs per product: pairs (s, t) with s + t <= slices - 1."""
    return slices * (slices + 1) // 2


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


# Elements per block of `split` / `combine`. Measured on a Trn2 host (split of
# 4096^2 at k = 8, ns per element): with 1-2 workers 2^15 is best (L2-resident
# buffers, 7.3 at 2 threads on 2 physical cores); with more workers small blocks
# thrash the GIL between numpy calls (12 threads: 38 at 2^15, 4.3 at 2^17), so
# blocks grow to 2^17 (6 threads: 3.9). Workers are capped at _MAX_WORKERS.
_BLOCK_ELEMS = 1 << 15
_BLOCK_ELEMS_MANY = 1 << 17
_MAX_WORKERS = 8


def _torch_empty(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def _host_threads() -> int:
    return max(1, min(_MAX_WORKERS, torch.get_num_threads()))


def _block_elems() -> int:
    return _BLOCK_ELEMS if _host_threads() <= 2 else _BLOCK_ELEMS_MANY


def _parallel_blocks(n: int, step: int, fn) -> None:
    """fn(r0, r1, scratch) over [0, n) in blocks of `step`, on torch's host
    thread count (numpy releases the GIL inside its loops). Blocks are
    disjoint; `scratch` is a dict private to the worker, so buffers are
    allocated once per worker, not once per block (fresh pages fault)."""
    starts = list(range(0, n, step))
    nt = min(_host_threads(), len(starts))

    def worker(w):
        scratch: dict = {}
        for r0 in starts[w::nt]:
            fn(r0, min(n, r0 + step), scratch)

    if nt <= 1:
        worker(0)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(nt) as ex:
        list(ex.map(worker, range(nt)))


def _buf(scratch: dict, name: str, shape, dtype) -> np.ndarray:
    """A scratch view of `shape` (reused across blocks of one worker)."""
    size = int(np.prod(shape))
    b = scratch.get(name)
    if b is None or b.size < size:
        b = scratch[name] = np.empty(size, dtype=dtype)
    return b[:size].reshape(shape)


def split(
    X: torch.Tensor, slices: int, per: str = "col", pad: tuple = (TILE_K, 1), empty=None
) -> tuple[torch.Tensor, np.ndarray]:
    """Split a 2-D (R, C) matrix into `slices` integer digits, in its layout.

    `per="col"` scales each column j by 2^-e_j, with e_j the frexp exponent
    of the column's largest magnitude (so |X[:, j]| < 2^e_j); `per="row"`
    scales each row. Digit 0 is rint(X 2^(6 - e)), and each next digit is
    rint of 2^7 times the remainder (all exact in fp64), so every digit is an
    integer with |digit| <= 64, exact in bf16, a digit product is at most
    2^12, and (per column)

        X[:, j] = 2^(e_j - 6) * sum_s digit_s[:, j] * 2^(-7 s) + truncation,

    with |truncation| <= 2^(e_j - 7 slices) (at k = 8, at most 2^-55 of the
    column maximum).

    Returns `(digits, e)`: bf16 digits of shape (slices, Rp, Cp), R and C
    padded with zeros to multiples of `pad`, and int64 exponents (C,) or (R,).
    `empty(shape, dtype)` allocates the digits (default `torch.empty`; the
    device route passes the caching pinned host allocator, whose memory is
    already faulted in and uploads faster). A row-major X is read twice (max,
    then digits), in cache-sized blocks of rows, threaded over torch's host
    thread count; a transposed view is first made contiguous, which is slow,
    so pass the contiguous orientation and pick `per` to match. Cost: about
    5 numpy passes per slice over L2-resident blocks, ~1.2 ns per element
    per slice on one Trn2 host core. Non-finite input is not representable:
    the caller checks.
    """
    k = check_slices(slices)
    if per not in ("row", "col"):
        raise ValueError(f"per must be 'row' or 'col', got {per!r}")
    if X.dim() != 2:
        raise ValueError(f"split expects a 2-D matrix, got shape {tuple(X.shape)}")
    X = X.detach().to(device="cpu", dtype=torch.float64).contiguous()
    R, C = X.shape
    Rp, Cp = _round_up(max(R, 1), pad[0]), _round_up(max(C, 1), pad[1])
    digits = (empty or _torch_empty)((k, Rp, Cp), torch.bfloat16)
    ne = R if per == "row" else C
    if R == 0 or C == 0:
        digits.zero_()
        return digits, np.zeros(ne, dtype=np.int64)
    if Rp > R:
        digits[:, R:].zero_()
    if Cp > C:
        digits[:, :R, C:].zero_()
    Xn = X.numpy()
    step = max(1, _block_elems() // C)
    if per == "row":
        amax = np.empty(R)

        def row_max(r0, r1, _scratch):
            xb = Xn[r0:r1]
            np.maximum(xb.max(axis=1), -xb.min(axis=1), out=amax[r0:r1])

        _parallel_blocks(R, step, row_max)
    else:
        parts: list = []

        def col_max(r0, r1, _scratch):
            xb = Xn[r0:r1]
            parts.append(np.maximum(xb.max(axis=0), -xb.min(axis=0)))

        _parallel_blocks(R, max(step, 256), col_max)
        amax = np.maximum.reduce(parts)
    e = np.frexp(amax)[1].astype(np.int64)  # amax < 2^e; amax = 0 -> e = 0
    shift = (SLICE_BITS - 1) - e
    if shift.max() <= 1023 and shift.min() >= -1022:
        scales = [np.ldexp(1.0, shift)]
    else:  # subnormal / huge rows or columns: two factors, neither over/underflows
        scales = [np.ldexp(1.0, shift // 2), np.ldexp(1.0, shift - shift // 2)]
    if per == "row":
        scales = [sc[:, None] for sc in scales]
    out16 = digits.view(torch.int16).numpy()  # bf16 bits
    radix = float(1 << SLICE_BITS)

    def block(r0, r1, scratch):
        n = r1 - r0
        Rb = _buf(scratch, "R", (n, C), np.float64)
        D = _buf(scratch, "D", (n, C), np.float64)
        F = _buf(scratch, "F", (n, C), np.float32)
        sc = [s[r0:r1] if per == "row" else s for s in scales]
        np.multiply(Xn[r0:r1], sc[0], out=Rb)
        for f in sc[1:]:
            np.multiply(Rb, f, out=Rb)  # |Rb| < 2^6, exact (power-of-two scales)
        for s in range(k):
            if s:
                np.multiply(Rb, radix, out=Rb)  # |Rb| <= 2^6
            np.rint(Rb, out=D)  # |digit| <= 64
            if s + 1 < k:
                np.subtract(Rb, D, out=Rb)  # |Rb| <= 1/2, exact
            F[...] = D  # exact
            # A small integer's bf16 bits are the high half of its fp32 bits.
            out16[s, r0:r1, :C] = F.view(np.int16)[:, 1::2]

    _parallel_blocks(R, step, block)
    return digits, e


def split_from_bits(X: np.ndarray, slices: int) -> tuple[np.ndarray, np.ndarray]:
    """The device split's algorithm (`_ozaki_split_kernel`, #123) on the host,
    op for op in float32 and uint32: digits of a per-row split from the raw
    fp64 bits of X (R, C), as float32 (k, R, C), and the row exponents (R,).

    It equals `split(X, slices, per="row")` bit for bit (as values; a zero
    digit may differ in sign, which no product or sum can see), except for a
    row whose largest element is subnormal (device exponent field 0): there
    `e` comes from the bit length, which the device route leaves to the host.
    Non-finite X is not representable. For tests and documentation; the
    host split (`split`) is faster on the host.
    """
    k = check_slices(slices)
    f32 = np.float32
    X = np.ascontiguousarray(X, dtype=np.float64)
    R, C = X.shape
    w = X.view(np.uint32).reshape(R, C, 2)
    lo, hi = w[..., 0], w[..., 1]

    def as_f32(u):
        return np.asarray(u, dtype=np.uint32).view(np.float32)

    E = (hi >> 20) & 0x7FF
    emax = E.max(axis=1) if C else np.zeros(R, dtype=np.uint32)
    e = np.where(emax > 0, emax.astype(np.int64) - 1022, 0)
    # |x| 2^(20 - e) = P1 + P2 + P3 (disjoint bits); exponent clamped at -70
    Xe = np.maximum(E, 1).astype(f32) - (1003 + e).astype(f32)[:, None]
    S1 = as_f32((np.maximum(Xe, f32(-70)) + f32(127)).astype(np.uint32) << 23)
    z = (E == 0).astype(f32)
    m1 = as_f32(((hi & 0xFFFFF) << 3) | (lo >> 29) | 0x3F800000)
    m2 = as_f32(((lo << 3) >> 9) | 0x3F800000)
    m3 = as_f32(((lo & 0x3F) << 17) | 0x3F800000)
    y = [
        (m1 - z) * S1,
        ((m2 - f32(1)) * f32(2.0**-23)) * S1,
        ((m3 - f32(1)) * f32(2.0**-46)) * S1,
    ]
    Q = []  # 21-bit chunks of floor(|x| 2^(20 - e + 21 j))
    fr = y
    for j in range((k + 3) // 3):
        t = fr if j == 0 else [v * f32(2.0**21) for v in fr]
        c = [np.rint(v) for v in t]  # the device's fp32 -> uint32 cast
        dl = [v - ci for v, ci in zip(t, c, strict=True)]
        fl = [(ci - (di < 0)).astype(np.uint32) for ci, di in zip(c, dl, strict=True)]
        fr = [di + (di < 0).astype(f32) for di in dl]
        Q.append(fl[0] | fl[1] | fl[2])  # disjoint bits
    frac = (fr[0] + fr[1]) + fr[2]
    F = [((Q[s // 3] >> (14 - 7 * (s % 3))) & 127).astype(f32) for s in range(k + 1)]
    low = 14 - 7 * (k % 3)
    rest = (Q[k // 3] & ((1 << low) - 1)).astype(f32) if low else np.zeros_like(frac)
    G = f32(2) * F[k] + (np.maximum(rest, frac) > 0).astype(f32)
    Rr = [None] * k  # F_s + carry_s, rounded half to even
    for s in range(k - 1, -1, -1):
        v = G * f32(1 / 256) + F[s]
        Rr[s] = (v + f32(12582912.0)) - f32(12582912.0)
        if s:
            G = F[s] * f32(2) + np.minimum(G, f32(1))
    sg = (hi >> 31).astype(f32) * f32(-2) + f32(1)
    D = np.empty((k, R, C), dtype=f32)
    D[0] = Rr[0] * sg
    for s in range(1, k):
        D[s] = ((Rr[s - 1] - F[s - 1]) * f32(-128) + Rr[s]) * sg
    return D, e


class Digits:
    """One operand's slices: `d` bf16 (k, Kp, X) when `kmajor` (K on the
    partition axis, as the kernel takes it), else (k, X, Kp); `e` the int64
    exponents of its rows of A / columns of B."""

    __slots__ = ("d", "e", "kmajor")

    def __init__(self, d: torch.Tensor, e: np.ndarray, kmajor: bool):
        self.d, self.e, self.kmajor = d, e, kmajor

    def block(self, k0: int, k1: int, x0: int, x1: int) -> torch.Tensor:
        """Rows k0:k1 of K and x0:x1 of M (A) / N (B), in this layout."""
        return self.d[:, k0:k1, x0:x1] if self.kmajor else self.d[:, x0:x1, k0:k1]

    @property
    def kp(self) -> int:
        return self.d.shape[1] if self.kmajor else self.d.shape[2]

    @property
    def xp(self) -> int:
        return self.d.shape[2] if self.kmajor else self.d.shape[1]


def split_a(A: torch.Tensor, slices: int, empty=None) -> Digits:
    """A (M, K) by rows. A transposed view (A.mT row-major) splits as is into
    the kernel's K-major layout; a row-major A splits in its own layout and is
    transposed later (on the device, where that is cheap)."""
    if A.mT.is_contiguous() and not A.is_contiguous():
        d, e = split(A.mT, slices, per="col", pad=(TILE_K, TILE_K), empty=empty)
        return Digits(d, e, True)
    d, e = split(A, slices, per="row", pad=(TILE_K, TILE_K), empty=empty)
    return Digits(d, e, False)


def split_b(B: torch.Tensor, slices: int, empty=None) -> Digits:
    """B (K, N) by columns; as `split_a`, a transposed view splits as is. N is
    padded to a multiple of 128 too, so B's slices can also be the kernel's
    first operand (`run` computes C^T = B^T A^T when N is narrow)."""
    if B.mT.is_contiguous() and not B.is_contiguous():
        d, e = split(B.mT, slices, per="row", pad=(TILE_K, TILE_K), empty=empty)
        return Digits(d, e, False)
    d, e = split(B, slices, per="col", pad=(TILE_K, TILE_K), empty=empty)
    return Digits(d, e, True)


def kmajor(d: torch.Tensor, is_kmajor: bool) -> torch.Tensor:
    """`d` in the kernel's (k, Kp, X) layout (contiguous)."""
    return d.contiguous() if is_kmajor else d.transpose(1, 2).contiguous()


def diag_groups(kt: int, d: int) -> list:
    """PSUM groups of diagonal d for kt K tiles per slice, in kernel order.

    The pairs of diagonal d are (s, d - s), s = 0..d. A group is a list of
    runs (s, t, kt0, n): K tiles kt0 .. kt0 + n - 1 of pair (s, t), at most
    GROUP_TILES tiles in all. With kt >= GROUP_TILES each pair is cut on its
    own into full groups plus a remainder; with kt < GROUP_TILES (K < 4096) a
    group holds GROUP_TILES // kt whole pairs of the same diagonal, laid end to
    end along K. (This uniform cut lets the kernel run groups as loops.)
    """
    if kt >= GROUP_TILES:
        return [
            [(s, d - s, k0, min(GROUP_TILES, kt - k0))]
            for s in range(d + 1)
            for k0 in range(0, kt, GROUP_TILES)
        ]
    per = GROUP_TILES // kt
    pairs = [(s, d - s) for s in range(d + 1)]
    return [[(s, t, 0, kt) for s, t in pairs[i : i + per]] for i in range(0, d + 1, per)]


def schedule(slices: int, kt: int) -> list:
    """[(d, scale, groups)] in accumulation order: diagonals smallest first
    (d = slices - 1 down to 0), each group's integer sum times
    scale = 2^(7 (slices - 1 - d))."""
    return [
        (d, float(2 ** (SLICE_BITS * (slices - 1 - d))), diag_groups(kt, d))
        for d in range(slices - 1, -1, -1)
    ]


def n_groups(slices: int, kt: int) -> int:
    """PSUM groups (TwoSum evictions) per output tile."""
    return sum(len(g) for _, _, g in schedule(slices, kt))


def exponent_offset(slices: int) -> int:
    """C = (hi + lo) * 2^(eA_i + eB_j + exponent_offset(slices))."""
    return -2 * (SLICE_BITS - 1) - SLICE_BITS * (slices - 1)


def _two_sum32(hi: np.ndarray, lo: np.ndarray, p: np.ndarray):
    """Knuth TwoSum of fp32 hi + p, error added into lo; the kernel's op order."""
    s = hi + p
    bb = s - hi
    t = s - bb
    e1 = hi - t
    e2 = p - bb
    e = e1 + e2
    return s, lo + e


def reference_from_digits(at: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """The kernel's result on the host: float32 array (2, Mp, N) of (hi, lo).

    `at` (k, Kp, Mp) and `b` (k, Kp, N) are the bf16 digits (`split`). Group
    sums are exact integers (computed in fp64), scaled by a power of two and
    rounded to fp32 exactly, then folded with fp32 TwoSum in the kernel's
    order, so this matches the device bit for bit when VectorE fp32 add /
    subtract round to nearest.
    """
    k, Kp, _ = at.shape
    if Kp % TILE_K:
        raise ValueError(f"Kp must be a multiple of {TILE_K}")
    A = at.to(torch.float64).numpy()
    B = b.to(torch.float64).numpy()
    hi = lo = None
    for _d, scale, groups in schedule(k, Kp // TILE_K):
        for grp in groups:
            G = None
            for s, t, k0, n in grp:
                r0, r1 = k0 * TILE_K, (k0 + n) * TILE_K
                P = A[s, r0:r1].T @ B[t, r0:r1]
                G = P if G is None else G + P  # integers < 2^24: exact
            p = (G * scale).astype(np.float32)  # exact: integer times 2^m
            if hi is None:  # the kernel starts from (0, 0) and TwoSums every group
                hi, lo = np.zeros_like(p), np.zeros_like(p)
            hi, lo = _two_sum32(hi, lo, p)
    return np.stack([hi, lo])


def combine(
    hilo: torch.Tensor,
    eA: np.ndarray,
    eB: np.ndarray,
    slices: int,
    M: int,
    N: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """fp64 C (M, N) from the kernel's (2, >=M, >=N) fp32 (hi, lo) output:
    (hi + lo) * 2^(eA_i + eB_j + exponent_offset(slices)), exact scaling.
    Written into `out` (a float64 (M, N) view) when given."""
    hl = hilo.detach().to("cpu").numpy()
    res = torch.empty((M, N), dtype=torch.float64) if out is None else out
    C = res.numpy()
    if M == 0 or N == 0:
        return res
    off = exponent_offset(slices)
    ea = np.asarray(eA[:M], dtype=np.int64) + off
    eb = np.asarray(eB[:N], dtype=np.int64)
    separable = max(abs(int(ea.max())), abs(int(ea.min()))) < 900 and (
        max(abs(int(eb.max())), abs(int(eb.min()))) < 900
    )
    if separable:  # power-of-two factors; no intermediate over/underflows
        ra = np.ldexp(1.0, ea)[:, None]
        cb = np.ldexp(1.0, eb)[None, :]

    def block(r0, r1, _scratch):
        v = C[r0:r1]
        np.add(hl[0, r0:r1, :N], hl[1, r0:r1, :N], out=v, dtype=np.float64)
        if separable:
            np.multiply(v, ra[r0:r1], out=v)
            np.multiply(v, cb, out=v)
        else:
            np.copyto(v, np.ldexp(v, ea[r0:r1, None] + eb[None, :]))

    _parallel_blocks(M, max(1, _block_elems() // N), block)
    return res


# Each kernel launch keeps its operands and its result below 2 GiB: under the
# 4 GiB partition-span limit of #112 (aws-neuron-sdk#1457) with the margin of
# the 2 GiB blocks measured safe there. Rows are at most Mp * 2 bytes, far
# under the 32 MiB row limit at any block that fits.
BLOCK_BYTES = 1 << 31


def _blocks(total: int, step: int) -> list:
    """[(start, size)] cutting [0, total) into blocks of `step` and a
    remainder. Fixed steps (not near-equal ones) keep the set of launch
    shapes, hence of compiled NEFFs, small across problem sizes."""
    return [(i, min(step, total - i)) for i in range(0, total, step)]


# Below this many columns of B (and with more rows of A than columns of B),
# `run` computes C^T = B^T A^T: the kernel's moving operand is then A, up to
# 512 columns wide per matmul, instead of B's few columns (N = 241: 22 against
# about 90 TFLOP/s).
SWAP_N = 512

# Largest output block per kernel launch. The compile time of the kernel grows
# with its whole problem (M and N tiles times K: 1.4 s at 1024^3, 29 s at
# 2048^3, 240 s at 4096^3 for k = 8, first call), so launches are capped and
# one NEFF serves every M and N of a given (slices, K).
LAUNCH_M = 1024
LAUNCH_N = 1024


def plan(
    Mp: int,
    Kp: int,
    N: int,
    slices: int,
    limit: int = BLOCK_BYTES,
    max_m: int = LAUNCH_M,
    max_n: int = LAUNCH_N,
):
    """Launch blocks (m_blocks, k_blocks, n_blocks), each [(start, size)].

    Each launch has at most max_m x max_n outputs, and its operand bytes,
    2 k Kb Mb (A^T slices) and 2 k Kb Nb (B slices), and result bytes, 8 Mb Nb,
    stay within `limit`. M and N blocks change nothing in the arithmetic. K
    blocks (only when one 128-row M tile times all of K would not fit, K > ~1M
    at k = 8) are separate (hi, lo) results summed in fp64 on the host, in
    order.
    """
    k = check_slices(slices)
    cap = limit // (2 * k)  # elements per slice-stack operand
    Kb = Kp if Kp * 1024 <= cap else max(TILE_K, (cap // 1024) // TILE_K * TILE_K)
    Mb = max(TILE_K, min(Mp, max_m // TILE_K * TILE_K, (cap // Kb) // TILE_K * TILE_K))
    Nb = max(1, min(N, max_n, cap // Kb))
    if 8 * Mb * Nb > limit:
        Nb = max(1, min(Nb, limit // (8 * Mb)))
    return _blocks(Mp, Mb), _blocks(Kp, Kb), _blocks(N, Nb)


def run(
    A: torch.Tensor,
    B: torch.Tensor,
    slices: int,
    launch,
    limit: int = BLOCK_BYTES,
    empty=None,
    max_m: int = LAUNCH_M,
    max_n: int = LAUNCH_N,
    split_ab=None,
) -> torch.Tensor:
    """C = A @ B in fp64 by Ozaki splitting. A (M, K) and B (K, N) are finite
    fp64 on the host. `launch(da, db, (m0, mb), (k0, kb), (n0, nb)) ->
    (2, mb, nb) fp32` computes one block's (hi, lo) from the operands' slices
    (`Digits`): the NKI kernel, or `reference_launch`. For narrow B
    (N < SWAP_N < M) the launches compute blocks of C^T = B^T A^T instead.
    `empty(shape, dtype)` allocates the slices and the result.
    `split_ab(A, B, slices, empty) -> (Digits, Digits)` makes the slices
    (default: `split_a` and `split_b` on the host; the device route splits
    on the device, #123, with the same digits)."""
    k = check_slices(slices)
    M, K = A.shape
    N = B.shape[1]
    if M == 0 or N == 0 or K == 0:
        return torch.zeros((M, N), dtype=torch.float64)
    empty = empty or _torch_empty
    if split_ab is None:
        da, db = split_a(A, k, empty), split_b(B, k, empty)
    else:
        da, db = split_ab(A, B, k, empty)
    C = empty((M, N), torch.float64)
    out = C
    if N < SWAP_N and M > N:  # C^T = B^T A^T: B's slices are the first operand
        da, db, M, N, out = db, da, N, M, C.mT
    mbs, kbs, nbs = plan(da.xp, da.kp, N, k, limit, max_m, max_n)
    for m0, mb in mbs:
        mr = min(mb, M - m0)
        if mr <= 0:
            continue
        for n0, nb in nbs:
            acc = None
            for k0, kb in kbs:
                hilo = launch(da, db, (m0, mb), (k0, kb), (n0, nb))
                ea, eb = da.e[m0 : m0 + mr], db.e[n0 : n0 + nb]
                if len(kbs) == 1:  # straight into C (or C^T)
                    combine(hilo, ea, eb, k, mr, nb, out=out[m0 : m0 + mr, n0 : n0 + nb])
                    continue
                part = combine(hilo, ea, eb, k, mr, nb)
                acc = part if acc is None else acc.add_(part)
            if acc is not None:
                out[m0 : m0 + mr, n0 : n0 + nb] = acc
    return C


def host_blocks(da: Digits, db: Digits, m, kk, n):
    """One launch's K-major (k, kb, mb) and (k, kb, nb) slices, on the host."""
    (m0, mb), (k0, kb), (n0, nb) = m, kk, n
    a = kmajor(da.block(k0, k0 + kb, m0, m0 + mb), da.kmajor)
    b = kmajor(db.block(k0, k0 + kb, n0, n0 + nb), db.kmajor)
    return a, b


def reference_launch(da: Digits, db: Digits, m, kk, n) -> torch.Tensor:
    """`run`'s launch with the host reference of the kernel arithmetic."""
    return torch.from_numpy(reference_from_digits(*host_blocks(da, db, m, kk, n)))


def reference_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    slices: int = DEFAULT_SLICES,
    limit: int = BLOCK_BYTES,
    max_m: int = LAUNCH_M,
    max_n: int = LAUNCH_N,
) -> torch.Tensor:
    """Host reference of `trnblas.gemm_fp64`: the same split, blocking and
    arithmetic as the device (`reference_from_digits`), in fp64 (M, N). For
    tests and for checking the device bit for bit; it costs about
    k (k + 1) / 2 host fp64 GEMMs."""
    A = A.detach().to("cpu", torch.float64)
    B = B.detach().to("cpu", torch.float64)
    return run(A, B, slices, reference_launch, limit, max_m=max_m, max_n=max_n)
