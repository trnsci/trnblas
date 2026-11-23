"""
fp64-class GEMM by the Ozaki scheme II on the bf16 Tensor Engine (trnblas#132).

Prototype, behind `trnblas.gemm_fp64(..., scheme="ozaki2")`; `scheme="ozaki1"`
(`trnblas/ozaki.py`) stays the default. The scheme is from Ozaki, Uchino and
Imamura, "Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix
multiplication using an integer modular technique" (arXiv:2504.08009):

1. **Scale to integers.** Row i of A is scaled by 2^mu_i and column j of B by
   2^nu_j, then rounded to integers Abar, Bbar (exact in fp64). The exponents
   come from the Cauchy-Schwarz bound: ||Abar_i||_2 ||Bbar_j||_2 <= M / 4,
   with M the product of the moduli, so |(Abar Bbar)_ij| <= M / 4 and the
   integer product is determined by its residues.
2. **Residues.** For each modulus m_l, the balanced residues of Abar and Bbar
   (|r| <= (m_l - 1) / 2), exact in bf16.
3. **Residue GEMMs.** One bf16 GEMM per modulus, fp32 PSUM accumulation in K
   groups short enough that every partial sum is an exact integer
   (`Family.group_tiles`), and a modular reduction at each group close, so
   the accumulator stays a small integer.
4. **CRT.** Abar Bbar = sum_l w_l C_l mod M (w_l the CRT weights), symmetric,
   then C = 2^(-mu_i - nu_j) (Abar Bbar) in fp64.

The number of products is the number of moduli n, linear in the precision
(about 7 bits per bf16 GEMM at the b7 family), against k (k + 1) / 2 for
Ozaki I's k slices.

**Families** (`FAMILIES`): pairwise coprime odd moduli below 2^7, 2^8 or 2^9.
The residue bound forces the PSUM group length: 4096 terms (b7, |r| <= 63),
1024 (b8, |r| <= 127), 256 (b9, |r| <= 255), from L r^2 < 2^24.

**Exactness and determinism.** Every group sum is an exact fp32 integer, every
reduction step is exact fp32 integer arithmetic, and each modulus ends with a
canonical balanced residue, so the residues (and everything after them) are
a function of the integers Abar, Bbar only: independent of the tiling, the
LNC split and even the rounding of the reciprocal multiply.

**CRT on the device (`crt_device`)** uses fp32 integer ops only: limb sums
L_c = sum_l W[l, c] r_l of the weights' beta-bit limbs (exact), the quotient
q = rint(sum_l fl(w_l / M) r_l) (exact: |C / M| <= 1/4), D_c = L_c - q M_c,
carry normalisation to balanced beta-bit digits, and pairs of digits packed
into exact fp32 values. The host sums the top `keep` packed values in fp64.
`crt_host` is the same reconstruction in fp64 words on the host.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch

from . import ozaki as _oz

TILE_K = 128
MAGIC = 12582912.0  # 1.5 * 2^23: (x + MAGIC) - MAGIC = rint(x) in fp32 for |x| < 2^22
F32_INT = 1 << 24  # every integer of magnitude <= 2^24 is exact in fp32


def _coprime_odd(mmax: int, count: int, fixed: tuple = ()) -> tuple:
    """`fixed`, then pairwise coprime odd integers greedily from mmax down,
    `count` in all."""
    out: list = list(fixed[:count])
    m = mmax if mmax % 2 else mmax - 1
    while len(out) < count and m >= 3:
        if all(math.gcd(m, x) == 1 for x in out):
            out.append(m)
        m -= 2
    if len(out) < count:  # pragma: no cover - tables are sized to fit
        raise ValueError(f"only {len(out)} coprime moduli below {mmax}")
    return tuple(out)


@dataclass(frozen=True)
class Family:
    """A moduli family: pairwise coprime moduli, balanced residues
    |r| <= rmax, PSUM groups of `group_tiles` K tiles of 128 with
    group_terms * (product bound) + mmax < 2^24.

    b7 / b8 / b9 are bf16 families: odd moduli <= mmax, |r| <= (mmax - 1) / 2
    (exact in bf16 up to 256), one bf16 GEMM per modulus, product bound rmax^2.
    f8 is analysis-only (no kernel): the FP8 E4M3 set of Uchino, Ozaki and
    Imamura (arXiv:2603.10634), six square moduli then odd moduli <= 511,
    each residue split into two FP8 digits |d| <= 16 (s = 16 or sqrt(p)),
    3 FP8 GEMMs per modulus (Karatsuba, or the s^2 = 0 trick for squares),
    product bound 16^2."""

    name: str
    mmax: int
    group_tiles: int
    max_moduli: int
    fixed: tuple = ()  # moduli taken first (f8: the squares)
    gemms_per_modulus: int = 1
    product_bound: int = 0  # bound on one PSUM product (0: rmax^2)

    @property
    def rmax(self) -> int:
        top = max((self.mmax, *self.fixed))
        return top // 2

    @property
    def group_terms(self) -> int:
        return self.group_tiles * TILE_K

    @property
    def moduli(self) -> tuple:
        return _coprime_odd(self.mmax, self.max_moduli, self.fixed)

    @property
    def kernel(self) -> bool:
        return self.gemms_per_modulus == 1

    @property
    def residue_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.rmax <= 256 else torch.float32


FAMILIES = {
    "b7": Family("b7", 127, 32, 28),  # 4096 * 63^2 = 16.26e6
    "b8": Family("b8", 255, 8, 32),  # 1024 * 127^2 = 16.52e6
    "b9": Family("b9", 511, 2, 32),  # 256 * 255^2 = 16.65e6
    # 65536 * 16^2 = 2^24 at most (the double_row FP8 path accumulates the same)
    "f8": Family("f8", 511, 511, 32, (1089, 1024, 961, 841, 625, 529), 3, 256),
}
for _f in FAMILIES.values():
    assert _f.group_terms * (_f.product_bound or _f.rmax**2) + _f.mmax < F32_INT, _f
    assert _f.rmax <= 256 or not _f.kernel  # bf16 has 8 significant bits
DEFAULT_FAMILY = "b7"
DEFAULT_MODULI = 17  # b7: ~53 bits on well-scaled data, the level of Ozaki I k = 8 (#132)
MIN_MODULI = 2


def family(name) -> Family:
    if isinstance(name, Family):
        return name
    if name not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {name!r}")
    return FAMILIES[name]


def check_moduli(n, fam) -> int:
    """`n` (the moduli count, the precision knob) as an int in range, else error."""
    f = family(fam)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"moduli must be an int, got {type(n).__name__}")
    n = int(n)
    if not MIN_MODULI <= n <= f.max_moduli:
        raise ValueError(f"moduli must be in [{MIN_MODULI}, {f.max_moduli}] for {f.name}, got {n}")
    return n


# ── CRT constants ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Crt:
    """CRT constants of the first n moduli of a family.

    Host (fp64 words of `hbeta` bits): `hW` (n, hn) the weights' words,
    `hM` (hn,) M's words, `omega64` (n,) w_l / M.
    Device (fp32 limbs of `beta` bits): `W` (n, nl) limbs, `Ml` (nl,) M's
    limbs, `omega` (n,) fl32(w_l / M); the output packs digit pairs
    (2 beta bits each) and keeps packed values j0 .. npk - 1 (`keep`)."""

    fam: Family
    n: int
    moduli: tuple
    M: int
    w: tuple
    beta: int
    nl: int
    W: np.ndarray
    Ml: np.ndarray
    omega: np.ndarray
    npk: int
    j0: int
    hbeta: int
    hn: int
    hW: np.ndarray
    hM: np.ndarray
    omega64: np.ndarray

    @property
    def keep(self) -> int:
        return self.npk - self.j0

    @property
    def log2M(self) -> float:
        return math.log2(self.M)


def _limbs(x: int, beta: int, count: int) -> list:
    return [(x >> (beta * c)) & ((1 << beta) - 1) for c in range(count)]


@functools.cache
def crt(fam, n: int) -> Crt:
    f = family(fam)
    n = check_moduli(n, f)
    mods = f.moduli[:n]
    M = math.prod(mods)
    w = tuple((M // m) * pow(M // m, -1, m) % M for m in mods)
    rn = n * f.rmax  # bound on sum_l |r_l|, and on |q|
    # device limbs: |D_c| <= 2 rn (2^beta - 1) < 2^24, and digit pairs pack
    # into fp32 (2 beta + 1 <= 24 bits)
    beta = min(12, math.floor(23 - math.log2(rn)))
    nl = -(-M.bit_length() // beta)
    W = np.array([_limbs(x, beta, nl) for x in w], dtype=np.float32)
    Ml = np.array(_limbs(M, beta, nl), dtype=np.float32)
    omega = np.array([x / M for x in w], dtype=np.float32)
    npk = -(-nl // 2)
    # keep the packed values down to 13 bits below sqrt(M / 4), the size of
    # the rounding error of the scaled operands (|dA| <= 1/2 times ||Bbar_j||)
    floor_bits = (M.bit_length() - 2) // 2 - 13
    j0 = max(0, min(npk - 1, floor_bits // (2 * beta)))
    # host words: n products of hbeta-bit words with |r| <= rmax, plus q M_c,
    # stay below 2^53
    hbeta = math.floor(52 - math.log2(rn))
    hn = -(-M.bit_length() // hbeta)
    hW = np.array([_limbs(x, hbeta, hn) for x in w], dtype=np.float64)
    hM = np.array(_limbs(M, hbeta, hn), dtype=np.float64)
    omega64 = np.array([x / M for x in w], dtype=np.float64)
    return Crt(f, n, mods, M, w, beta, nl, W, Ml, omega, npk, j0, hbeta, hn, hW, hM, omega64)


# ── scaling and residues ─────────────────────────────────────────────────────


def bound(n: int, fam=DEFAULT_FAMILY) -> float:
    """T = sqrt(M / 4) (rounded down a little): each scaled row of A and
    column of B has 2-norm <= T, so |sum_k Abar_ik Bbar_kj| <= T^2 <= M / 4."""
    M = crt(fam, n).M
    return float(math.isqrt(M // 4)) * (1 - 2.0**-40)


def scale_exponents(X: np.ndarray, axis: int, T: float) -> np.ndarray:
    """int64 exponents mu (one per row for axis=1, per column for axis=0)
    with || rint(X 2^mu) ||_2 <= T along `axis`: 2^mu ||x|| <= T - sqrt(K) / 2
    (rounding adds at most 1/2 per entry). Zero rows / columns get 0."""
    K = X.shape[axis]
    amax = np.abs(X).max(axis=axis) if K else np.zeros(X.shape[1 - axis])
    e = np.frexp(amax)[1].astype(np.int64)  # amax < 2^e
    Y = np.ldexp(X, -e[:, None] if axis == 1 else -e[None, :])  # |Y| < 1, exact
    nrm = np.sqrt(np.einsum("ij,ij->i" if axis == 1 else "ij,ij->j", Y, Y))
    nrm *= 1 + 2.0**-30  # covers the rounding of the fp64 norm
    budget = T - 0.5 * math.sqrt(max(K, 1))
    if budget <= 0:
        raise ValueError(f"too few moduli for K = {K}")
    with np.errstate(divide="ignore"):
        ex = np.frexp(budget / np.where(nrm > 0, nrm, 1.0))[1].astype(np.int64)
    mu = ex - 1 - e  # 2^(ex - 1) <= budget / nrm
    return np.where(amax > 0, mu, 0)


def _canon(r: np.ndarray, m: int) -> np.ndarray:
    """Balanced residue in [-(m - 1) / 2, (m - 1) / 2] of r in [0, m)."""
    return r - m * (r > (m - 1) // 2)


@functools.cache
def _pow2_mod(m: int) -> np.ndarray:
    """2^e mod m for e = 0 .. 2099 (every fp64 exponent)."""
    return np.array([pow(2, e, m) for e in range(2100)], dtype=np.int64)


def residues_of(Xbar: np.ndarray, moduli) -> list:
    """Balanced residues (int64 arrays) of the integer-valued fp64 Xbar.
    |Xbar| < 2^53 converts to int64 exactly; a larger one is mant 2^ex with
    an int64 mantissa (< 2^53) and ex > 0, so Xbar mod m is
    (mant mod m)(2^ex mod m) mod m."""
    big = Xbar.size and float(np.abs(Xbar).max()) >= 2.0**53
    if not big:
        xi = Xbar.astype(np.int64)
        return [_canon(xi % m, m) for m in moduli]
    f, e = np.frexp(Xbar)
    mant = np.ldexp(f, 53).astype(np.int64)
    ex = e.astype(np.int64) - 53
    small = ex <= 0
    xi = np.where(small, Xbar, 0.0).astype(np.int64)
    exi = np.where(small, 0, ex)
    return [_canon(np.where(small, xi % m, (mant % m) * _pow2_mod(m)[exi] % m), m) for m in moduli]


class Residues:
    """One operand's residues: `d` bf16 (n, Kp, Xp), K-major (the kernel's
    layout, like `ozaki.Digits` with kmajor=True), `e` the int64 scale
    exponents (mu of A's rows / nu of B's columns)."""

    __slots__ = ("d", "e", "kmajor")

    def __init__(self, d: torch.Tensor, e: np.ndarray):
        self.d, self.e, self.kmajor = d, e, True

    def block(self, k0, k1, x0, x1) -> torch.Tensor:
        return self.d[:, k0:k1, x0:x1]

    @property
    def kp(self) -> int:
        return self.d.shape[1]

    @property
    def xp(self) -> int:
        return self.d.shape[2]


def split(X: torch.Tensor, n: int, fam=DEFAULT_FAMILY, per: str = "col", empty=None) -> Residues:
    """Residues of a 2-D fp64 X in the kernel's K-major layout.

    per="col": X is (K, N) (B), scaled per column, residues (n, Kp, Np).
    per="row": X is (M, K) (A), scaled per row, residues of X^T (n, Kp, Mp).
    Both padded with zeros to multiples of 128. Finite X only."""
    f = family(fam)
    n = check_moduli(n, f)
    if per not in ("row", "col"):
        raise ValueError(f"per must be 'row' or 'col', got {per!r}")
    Xn = X.detach().to("cpu", torch.float64).numpy()
    if per == "row":
        Xn = Xn.T  # (K, M) view: scaled per column, like B
    K, C = Xn.shape
    Kp, Cp = _oz._round_up(max(K, 1), TILE_K), _oz._round_up(max(C, 1), TILE_K)
    dt = f.residue_dtype
    d = (empty or _oz._torch_empty)((n, Kp, Cp), dt)
    d.zero_()
    if K == 0 or C == 0:
        return Residues(d, np.zeros(C, dtype=np.int64))
    mu = scale_exponents(Xn, 0, bound(n, f))
    out16 = d.view(torch.int16).numpy() if dt == torch.bfloat16 else None
    outf = d.numpy() if out16 is None else None
    mods = f.moduli[:n]
    step = max(1, _oz._block_elems() // C)

    def block(r0, r1, _scratch):
        Xb = np.rint(np.ldexp(np.ascontiguousarray(Xn[r0:r1]), mu[None, :]))  # exact
        for s, r in enumerate(residues_of(Xb, mods)):
            if outf is not None:
                outf[s, r0:r1, :C] = r
                continue
            # a small integer's bf16 bits are the high half of its fp32 bits
            out16[s, r0:r1, :C] = r.astype(np.float32).view(np.int16)[:, 1::2]

    _oz._parallel_blocks(K, step, block)
    return Residues(d, mu)


def split_ab(A: torch.Tensor, B: torch.Tensor, n: int, fam=DEFAULT_FAMILY, empty=None):
    return split(A, n, fam, per="row", empty=empty), split(B, n, fam, per="col", empty=empty)


# ── residue GEMM: exact reference and the kernel's arithmetic ────────────────


def reference_residue_gemm(at: torch.Tensor, b: torch.Tensor, fam=DEFAULT_FAMILY) -> np.ndarray:
    """The kernel's result: float32 (n, Mp, N) balanced residues of
    at[l]^T b[l] mod m_l. Exact: host fp64 GEMMs of integers below 2^53."""
    f = family(fam)
    n = at.shape[0]
    A = at.to(torch.float64).numpy()
    B = b.to(torch.float64).numpy()
    out = np.empty((n, A.shape[2], B.shape[2]), dtype=np.float32)
    for l, m in enumerate(f.moduli[:n]):
        P = A[l].T @ B[l]
        out[l] = _canon(np.mod(P, m), m)
    return out


def residue_gemm_ops(at: torch.Tensor, b: torch.Tensor, fam=DEFAULT_FAMILY) -> np.ndarray:
    """The kernel's group-close arithmetic op for op in fp32 (for tests):
    per modulus, exact group sums of `group_tiles` K tiles, each reduced as
    T = S + acc; q = rint(T fl(1/m)); acc = T - q m; then a final canonical
    reduction. Equals `reference_residue_gemm` (asserted by the tests)."""
    f = family(fam)
    n, Kp, _ = at.shape
    A = at.to(torch.float64).numpy()
    B = b.to(torch.float64).numpy()
    f32 = np.float32
    L = f.group_terms
    out = []
    for l, m in enumerate(f.moduli[:n]):
        inv, mm = f32(1.0 / m), f32(-m)
        acc = np.zeros((A.shape[2], B.shape[2]), dtype=f32)
        for k0 in range(0, Kp, L):
            S = A[l, k0 : k0 + L].T @ B[l, k0 : k0 + L]
            assert np.abs(S).max(initial=0) + np.abs(acc).max(initial=0) < F32_INT
            T = S.astype(f32) + acc
            q = ((T * inv) + f32(MAGIC)) - f32(MAGIC)
            acc = (q * mm) + T
        q = ((acc * inv) + f32(MAGIC)) - f32(MAGIC)
        out.append((q * mm) + acc)
    return np.stack(out)


# ── CRT ──────────────────────────────────────────────────────────────────────


def crt_device(res: np.ndarray, c: Crt) -> np.ndarray:
    """The device CRT (`_ozaki2_crt_*_kernel`) op for op in fp32: float32
    (keep, R, C) packed digit pairs of the integer X = Abar Bbar, from
    balanced residues res (n, R, C):
    X ~= sum_j out[j] 2^(2 beta (j0 + j)), the dropped part below
    2^(2 beta j0 - 1) in magnitude. Every step is an exact integer operation
    in fp32 except the quotient estimate, whose rint is exact."""
    f32 = np.float32
    r = [np.asarray(res[l], dtype=f32) for l in range(c.n)]
    L = []
    for j in range(c.nl):
        acc = None
        for l in range(c.n):
            wv = c.W[l, j]
            if wv == 0:
                continue
            acc = r[l] * wv if acc is None else (r[l] * wv) + acc
        L.append(np.zeros_like(r[0]) if acc is None else acc)
    qa = r[0] * c.omega[0]
    for l in range(1, c.n):
        qa = (r[l] * c.omega[l]) + qa
    q = (qa + f32(MAGIC)) - f32(MAGIC)
    D = [L[j] if c.Ml[j] == 0 else (q * -c.Ml[j]) + L[j] for j in range(c.nl)]
    sc, isc = f32(2.0**c.beta), f32(2.0**-c.beta)
    for j in range(c.nl - 1):
        cy = ((D[j] * isc) + f32(MAGIC)) - f32(MAGIC)
        D[j] = (cy * -sc) + D[j]
        D[j + 1] = D[j + 1] + cy
    out = []
    for p in range(c.j0, c.npk):
        lo, hi = 2 * p, 2 * p + 1
        out.append(D[lo] if hi >= c.nl else (D[hi] * sc) + D[lo])
    return np.stack(out)


def unpack_device(packed: np.ndarray, c: Crt) -> np.ndarray:
    """fp64 integer X from `crt_device`'s packed values (Horner, top first):
    exact through the top two values, one rounding per further value."""
    P = np.asarray(packed, dtype=np.float64)
    s = P[-1].copy()
    step = 2.0 ** (2 * c.beta)
    for j in range(P.shape[0] - 2, -1, -1):
        s *= step
        s += P[j]
    return s  # times 2^(2 beta j0)


def crt_host(res: np.ndarray, c: Crt) -> np.ndarray:
    """X = Abar Bbar in fp64 from balanced residues res (n, R, C), on the host
    in hbeta-bit fp64 words: word sums by one exact BLAS product (every
    partial sum is an integer below 2^53), q = rint(sum_l (w_l / M) r_l),
    D_c = L_c - q M_c, carries, then top-down fp64 Horner."""
    n, R, C = res.shape
    out = np.empty((R, C), dtype=np.float64)
    if R == 0 or C == 0:
        return out
    step = max(1, _oz._block_elems() // C)
    Wq = np.concatenate([c.hW, c.omega64[:, None]], axis=1)  # (n, hn + 1)
    sc = 2.0**c.hbeta

    def block(r0, r1, _scratch):
        Rb = np.ascontiguousarray(res[:, r0:r1].reshape(n, -1).T, dtype=np.float64)
        Lq = Rb @ Wq  # (rows, hn + 1): word sums (exact) and x / M
        q = np.rint(Lq[:, -1])
        D = Lq[:, :-1] - q[:, None] * c.hM[None, :]
        for j in range(c.hn - 1):
            cy = np.rint(D[:, j] / sc)
            D[:, j] -= cy * sc
            D[:, j + 1] += cy
        s = D[:, -1].copy()
        for j in range(c.hn - 2, -1, -1):
            s *= sc
            s += D[:, j]
        out[r0:r1] = s.reshape(r1 - r0, C)

    _oz._parallel_blocks(R, step, block)
    return out


def unscale(X: np.ndarray, eA: np.ndarray, eB: np.ndarray, shift: int = 0) -> np.ndarray:
    """C = X 2^(shift - eA_i - eB_j), exact (barring under/overflow)."""
    ea = shift - np.asarray(eA, dtype=np.int64)
    eb = -np.asarray(eB, dtype=np.int64)
    return np.ldexp(X, ea[:, None] + eb[None, :])


# ── blocking and the driver ──────────────────────────────────────────────────


def plan(Mp: int, Kp: int, N: int, n: int, limit: int = _oz.BLOCK_BYTES, max_m=None, max_n=None):
    """Launch blocks (m_blocks, k_blocks, n_blocks), as `ozaki.plan` with n
    residue stacks: operand bytes 2 n Kb Mb and 2 n Kb Nb, and the device
    residues 4 n Mb Nb, each within `limit` (#112)."""
    max_m = max_m or _oz.LAUNCH_M
    max_n = max_n or _oz.LAUNCH_N
    cap = limit // (2 * n)
    Kb = Kp if Kp * 1024 <= cap else max(TILE_K, (cap // 1024) // TILE_K * TILE_K)
    Mb = max(TILE_K, min(Mp, max_m // TILE_K * TILE_K, (cap // Kb) // TILE_K * TILE_K))
    Nb = max(1, min(N, max_n, cap // Kb))
    if 4 * n * Mb * Nb > limit:
        Nb = max(1, min(Nb, limit // (4 * n * Mb)))
    return _oz._blocks(Mp, Mb), _oz._blocks(Kp, Kb), _oz._blocks(N, Nb)


CRT_MODES = ("device", "host")


def run(
    A: torch.Tensor,
    B: torch.Tensor,
    n: int,
    launch,
    fam=DEFAULT_FAMILY,
    crt_mode: str = "device",
    limit: int = _oz.BLOCK_BYTES,
    empty=None,
    max_m=None,
    max_n=None,
    split_fn=None,
) -> torch.Tensor:
    """C = A @ B in fp64 by Ozaki II. A (M, K), B (K, N) finite fp64 on the
    host. `launch(ra, rb, (m0, mb), (k0, kb), (n0, nb))` returns host fp32:
    the packed CRT output (keep, mb, nb) when crt_mode == "device", the
    residues (n, mb, nb) when "host". Narrow B (N < SWAP_N < M) computes
    C^T = B^T A^T. K blocks (K > ~1M) are summed in fp64, in order."""
    f = family(fam)
    n = check_moduli(n, f)
    if crt_mode not in CRT_MODES:
        raise ValueError(f"crt must be one of {CRT_MODES}, got {crt_mode!r}")
    c = crt(f, n)
    M, K = A.shape
    N = B.shape[1]
    if M == 0 or N == 0 or K == 0:
        return torch.zeros((M, N), dtype=torch.float64)
    empty = empty or _oz._torch_empty
    ra, rb = (split_fn or split_ab)(A, B, n, f, empty)
    C = empty((M, N), torch.float64)
    out = C.numpy()
    if N < _oz.SWAP_N and M > N:
        ra, rb, M, N, out = rb, ra, N, M, out.T
    mbs, kbs, nbs = plan(ra.xp, ra.kp, N, n, limit, max_m, max_n)
    shift = 2 * c.beta * c.j0 if crt_mode == "device" else 0
    for m0, mb in mbs:
        mr = min(mb, M - m0)
        if mr <= 0:
            continue
        for n0, nb in nbs:
            nr = min(nb, N - n0)
            acc = None
            for k0, kb in kbs:
                got = launch(ra, rb, (m0, mb), (k0, kb), (n0, nb)).numpy()
                if crt_mode == "device":
                    X = unpack_device(got[:, :mr, :nr], c)
                else:
                    X = crt_host(got[:, :mr, :nr], c)
                part = unscale(X, ra.e[m0 : m0 + mr], rb.e[n0 : n0 + nr], shift)
                acc = part if acc is None else acc + part
            out[m0 : m0 + mr, n0 : n0 + nr] = acc
    return C


def host_blocks(ra: Residues, rb: Residues, m, kk, nn):
    (m0, mb), (k0, kb), (n0, nb) = m, kk, nn
    return ra.block(k0, k0 + kb, m0, m0 + mb), rb.block(k0, k0 + kb, n0, n0 + nb)


def reference_launch(fam=DEFAULT_FAMILY, crt_mode: str = "device"):
    """`run`'s launch with the host reference of the kernels' arithmetic."""
    f = family(fam)

    def launch(ra, rb, m, kk, nn):
        at, bt = host_blocks(ra, rb, m, kk, nn)
        res = reference_residue_gemm(at, bt, f)
        if crt_mode == "host":
            return torch.from_numpy(res)
        return torch.from_numpy(crt_device(res, crt(f, at.shape[0])))

    return launch


def reference_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    n: int = DEFAULT_MODULI,
    fam=DEFAULT_FAMILY,
    crt_mode: str = "device",
    limit: int = _oz.BLOCK_BYTES,
    max_m=None,
    max_n=None,
) -> torch.Tensor:
    """Host reference of `gemm_fp64(..., scheme="ozaki2")`: the same scaling,
    residues, blocking and CRT as the device, in fp64 (M, N). The device
    result should equal it bit for bit. Costs n host fp64 GEMMs."""
    A = A.detach().to("cpu", torch.float64)
    B = B.detach().to("cpu", torch.float64)
    return run(A, B, n, reference_launch(fam, crt_mode), fam, crt_mode, limit, None, max_m, max_n)


# ── cost model (for the #132 analysis) ───────────────────────────────────────


def n_groups(K: int, fam=DEFAULT_FAMILY) -> int:
    """PSUM group closes (modular reductions) per modulus per output tile."""
    kt = -(-K // TILE_K)
    return -(-kt // family(fam).group_tiles)


def vector_ops(K: int, n: int, fam=DEFAULT_FAMILY) -> dict:
    """VectorE instructions per output element: group-close reductions (4
    each: add, scale+round, unround, multiply-subtract), the per-modulus
    canonical reduction (3) and the device CRT."""
    c = crt(fam, n)
    red = n * (4 * n_groups(K, fam) + 3)
    crt_ops = int((c.W != 0).sum()) + c.n + 1 + int((c.Ml != 0).sum()) + 4 * (c.nl - 1) + c.keep
    return {"reduce": red, "crt": crt_ops}


def fp8_split(r: np.ndarray, p: int):
    """FP8 E4M3 digits of balanced residues mod p (arXiv:2603.10634, analysis
    only): (d1, d2, s) with r = s d1 + d2 and |d1|, |d2| <= 16 exact in E4M3.
    Square p = s^2: d1 = rint(r / s), and the d1 d1' product drops out mod p.
    Otherwise s = 16, d1 = sign(r) ceil(|r| / 16), and |d1 + d2| <= 16 too
    (the Karatsuba middle product (d1 + d2)(d1' + d2'))."""
    s = math.isqrt(p)
    if s * s == p:
        d1 = np.rint(r / s)
        return d1, r - s * d1, s
    d1 = np.sign(r) * np.ceil(np.abs(r) / 16)
    return d1, r - 16 * d1, 16


def fp8_product_mod(ra: np.ndarray, rb: np.ndarray, p: int) -> np.ndarray:
    """(ra @ rb) mod p from 3 products of FP8 digits, as the FP8 arm forms it
    (each product exact: digits <= 16, K <= 65536 terms per fp32 group)."""
    a1, a2, s = fp8_split(ra, p)
    b1, b2, _ = fp8_split(rb, p)
    if s * s == p:
        X = s * (a1 @ b2) + s * (a2 @ b1) + a2 @ b2
    else:
        c1, c2, c3 = a1 @ b1, a2 @ b2, (a1 + a2) @ (b1 + b2)
        X = s * s * c1 + c2 + s * (c3 - c1 - c2)
    return _canon(np.mod(X, p), p)
