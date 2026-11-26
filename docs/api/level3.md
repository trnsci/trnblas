# Level 3 — Matrix-matrix operations

Level 3 BLAS: `O(n³)` matrix-matrix operations. This is the hot path for
scientific computing workloads and the primary target for NKI acceleration.

## `gemm(alpha, A, B, beta=0.0, C=None, transA=False, transB=False)`

General matrix-matrix multiply: `C = α·op(A)·op(B) + β·C`.

On Trn1 (XLA path) and under `set_backend("nki")`, dispatches to the NKI
GEMM kernel with stationary tile reuse. On Trn2/Trn3 (native path), `auto`
runs device-resident operands as aten on the `neuron` device (faster than the
NKI GEMM at every measured shape) and host operands on the host CPU or as
upload + aten, by measured time (v0.9.0). Falls back to `torch.matmul` on
CPU/GPU.

## `batched_gemm(alpha, A, B, beta=0.0, C=None, transA=False, transB=False)`

Batched GEMM over the leading dimension — used for DF-MP2 tensor contractions
over auxiliary basis indices. Routed like `gemm` on the native path.

## `gemm_fp64(A, B, *, slices=8, transA=False, transB=False, scheme="ozaki1", moduli=None)`

fp64-class product `op(A) @ op(B)` on the bf16 Tensor Engine, by Ozaki
splitting ([#22](https://github.com/trnsci/trnblas/issues/22)). Returns
float64 on A's device. `slices` (k, 1–12) is the precision knob.

How it works:

1. **Split (device, exact).** A is split row by row and B column by column
   into k bf16 integer slices of 7 bits each, |digit| ≤ 64, scaled by one
   power of two per row / column (the digits of `trnblas.ozaki.split`). Each
   operand is uploaded as its raw fp64 bits (8 bytes per element, a plain
   copy) and split on the device (#123) by `_ozaki_emax_kernel` (row
   exponents; `_ozaki_emax_col_kernel` for columns) and `_ozaki_split_kernel`
   (`_ozaki_split_col_kernel`), in the operand's own layout. Device float64
   arithmetic is fp32, so the kernels use only bitvec ops and exact fp32
   arithmetic: three disjoint fp32 pieces of |x| · 2^(20 − e) from the bit
   fields, an exact floor chain to 21-bit fixed-point chunks, the 7-bit
   fields by shifts, and the balanced digits with their round-half-even
   carries (`trnblas.ozaki.split_from_bits` is the same arithmetic on the
   host, op for op). The digits equal the host split's (as values: a zero
   digit can be −0 on one side and +0 on the other, which no product or
   PSUM sum can see), so the result is bitwise the same either way.
2. **Slice products (device).** The kernel (`_ozaki_gemm_kernel`) runs the
   k(k+1)/2 pairs (s, t) with s + t ≤ k − 1 as bf16 matmuls with fp32 PSUM
   accumulation. The pairs of one diagonal s + t share a scale, so their K
   tiles are laid end to end and cut into PSUM groups of at most 4096 terms:
   4096 · 64 · 64 = 2²⁴, so every group sum is an exact integer. When K < 4096,
   one group holds several pairs.
3. **Eviction (device).** Each group is scaled by 2^(7(k−1−d)) on its way out
   of PSUM and folded into an SBUF fp32 TwoSum (hi, lo) pair on VectorE,
   diagonals smallest first.
4. **Combine (device, #125).** C = RN64(hi + lo) · 2^(e_i + e_j − 12 − 7(k−1)),
   the float64 words built on the device by `_fp64_ff_bits_kernel`
   (`trnblas.fp64`, #134) from exact fp32 and bitvec ops, bitwise equal to
   the host combine `trnblas.ozaki.combine`. The (hi, lo) blocks of one band
   of output rows stay on the device, are joined along N, and one launch per
   band writes the band's words, which are downloaded straight into the
   result's storage. Rows the kernel flags (a non-finite element, or a
   result outside the normal float64 range: overflow or gradual underflow)
   are recombined on the host, as are products with several K blocks;
   `TRNBLAS_OZAKI_DEVICE_COMBINE=0` keeps the host combine. Measured on
   Trn2 for one 4096 × 4096 block: 14.5 ms against 79.4 ms for the
   download and host combine.

Properties:

- **Accuracy** (max |C − C_exact| / (|A||B|), normal data, measured on Trn2):
  about 1e-12 at k = 6 (21 slice GEMMs), 1e-14 at k = 7 (28), and
  2e-16 to 4e-16 at k = 8 (36), the host fp64 BLAS level. The scaling is per
  row / column, so entries far below their row or column maximum lose
  relative precision: data spread over 10^±4 needs k = 10 for host-fp64
  accuracy (k = 8 gives about 1e-14 there).
- **Deterministic.** Every element is the same sequence of exact integer group
  sums and fp32 TwoSums whatever the tiling, so the result is bitwise the same
  at LNC=1 and LNC=2 and equals `trnblas.ozaki.reference_gemm` (the host
  reference of the kernel arithmetic) bit for bit.
- **Launches and memory (#112).** Each kernel launch computes at most a
  1024 × 1024 output block (`ozaki.LAUNCH_M`, `LAUNCH_N`): the kernel's compile
  time grows with its whole problem (240 s for one 4096³ launch at k = 8), so a
  capped block keeps one NEFF per (slices, K, block shape) for every M and N.
  At k = 8 each operand's slices take 16 bytes per element (2× fp64). Slices
  under 2 GiB are uploaded once per call and cut into blocks on the device;
  larger ones are uploaded block by block. Every launch's operands and result
  stay under 2 GiB (`ozaki.plan`). M and N blocks do not change the
  arithmetic; K is blocked only beyond ~1M at k = 8 (fp64 partial sums on the
  host).
- **Layouts.** Each operand is split in its contiguous layout (per row for
  a row-major A or Bᵀ, per column for Aᵀ or a row-major B, as the host split
  does); slices that are not K-major are transposed on the device (about
  500 GB/s). The raw fp64 bits are never transposed on the device: a device
  `permute(...).contiguous()` of int32 bit patterns did not preserve them.
- **Host split fallback.** The host split (`trnblas.ozaki.split`, same
  digits) runs under `TRNBLAS_OZAKI_DEVICE_SPLIT=0`, for an operand whose raw
  bits or digit stack reach 2 GiB (#112), and for an operand with a row of A
  (column of B) whose largest element is subnormal: its exponent comes from
  the bit length, which the device route leaves to the host.
- **Routes.** The kernel runs on the native TorchNeuron path (Trn2/Trn3) and
  under `TRNBLAS_USE_SIMULATOR=1`. On CPU, on Trn1 (XLA), under
  `set_backend("pytorch")`, and for non-finite operands (found from the
  device exponent maxima), `gemm_fp64` is host fp64 `A @ B`. Under
  `TRNBLAS_REQUIRE_NKI=1` the off-device case raises.
- **Cost** (one logical core at LNC=2):
  - The device split costs about 0.55 ns per element, after an 8-byte
    upload per element. The host split costs about 11 ns per element at
    k = 8 on one Trn2 host core (two vCPUs).
  - The kernel runs at about 87–96 TFLOP/s of bf16 slice products.
  - The first call per operand and launch shape compiles; the NEFFs are
    cached afterwards.

  End to end at k = 8:
  - Against host fp64 on 2 vCPUs: 3.0× on (3267, 8400) @ (8400, 241), 2.6×
    on (8192, 3508) @ (3508, 241), and 10.1× at 4096³.
  - Against a 12-vCPU host: 1.5–2.2× from 2048³ up, and about 2× slower on
    the narrow DF-MP2 shapes. There, the upload and the split of A already
    take as long as the 12-vCPU host GEMM.

  See [Benchmarks](../benchmarks.md#device-split-123).

`alpha` / `beta` are not taken: scale the fp64 result. Batched operands are
not supported.

### `scheme="ozaki2"` (prototype, [#132](https://github.com/trnsci/trnblas/issues/132))

The Ozaki scheme II (Ozaki, Uchino and Imamura, arXiv:2504.08009) in place
of the slicing above; `scheme="ozaki1"` stays the default. Host half:
`trnblas/ozaki2.py`; kernels: `trnblas/nki/ozaki2_kernels.py`.

1. **Scale (device, #132).** Row i of A by 2^mu_i and column j of B by
   2^nu_j, then round to integers. The exponents come from the
   Cauchy–Schwarz bound ||Ā_i||·||B̄_j|| ≤ M/4, with M the product of the
   moduli. The norm bound is an exact integer sum, so it does not depend on
   the order of summation, the operand's layout or the LNC split. Each
   element contributes τ = ⌊w²/2⁶⌋, where w is |x|·2^(20−e) taken from the
   bit fields, cut to 12 significant bits, and rounded up by one unit. w²
   is then exact in fp32. The device sums τ in 9-bit limbs, and every
   partial sum is an integer below 2²⁴ (`_ozaki2_norm_kernel`,
   `_ozaki2_norm_col_kernel`). The host finishes the sums in int64 and
   computes the exponents (`trnblas.ozaki2.scale_from_sums`). The host
   split uses the same rule. The bound is within about 2⁻¹⁰ of the fp64
   norm's square. So the exponents equal those of an fp64-norm rule, except
   within that distance of a power-of-two boundary, and they are never
   larger. On 12,000 test rows they were all equal.
2. **Residues (device).** These are balanced residues modulo `moduli` (n)
   pairwise coprime odd moduli below 2^7 (127, 125, 123, ...), with |r| ≤ 63,
   exact in bf16. The split kernels (`_ozaki2_split_<fam>_kernel`, and
   `_col` for columns) reuse the Ozaki I device split (#123) on a scaled
   grid. With k digits and the row exponent e' = 6 + 7(k−1) − mu, the
   balanced base-128 digits are RNE(x·2^mu), exactly. Every digit but the
   last is exact, and the last rounds half to even, which sets the parity
   of the whole. A residue is then a short dot product of digits and the
   balanced residues of powers of two. Two digits are joined into 14-bit
   terms where that keeps every partial sum below 2²², and single digits
   are used for the wider b8/b9 moduli. One rounding,
   q = rint(t·fl(1/m)), follows. It gives the nearest integer to t/m,
   because t/m is at least 1/(2m) from a half-integer and the rounding
   error is smaller than that. The result is then t − q·m. The residues
   are a function of the integers alone, so they equal the host split's
   (`trnblas.ozaki2.split_from_bits` is the device arithmetic on the
   host). The host split still runs:
   - for a row or column whose largest element is subnormal;
   - for an operand whose raw bits or residues reach 2 GiB (#112);
   - for more than 12 digits (b8 beyond about 22 moduli);
   - under `TRNBLAS_OZAKI2_DEVICE_SPLIT=0`.
3. **Residue GEMMs (device, one launch for all moduli).** One bf16 GEMM per
   modulus with fp32 PSUM groups of 4096 terms (4096·63² < 2^24, exact),
   and a 4-instruction VectorE modular reduction at each group close, then
   a canonical residue.
4. **CRT (device, in the same launch: `crt_mode="fused"`, the default).**
   Exact fp32 integer ops: limb sums of the CRT weights, the quotient by M,
   carries to balanced digits, packed digit pairs.
5. **Combine (device).** C = RN64(X')·2^(shift − mu_i − nu_j), where X' is
   the kept packed integer. This is one correctly rounded rounding; the
   earlier host Horner rounded once per packed value. The float64 words are
   built on the device by `_ozaki2_combine_kernel` from exact fp32 and
   bitvec ops:
   - the sign of the top nonzero packed value, borrows to nonnegative
     24-bit limbs, and shifts that make the top limb nonzero;
   - the 72-bit window from the leading bit, and a sticky bit;
   - round to nearest even, with carries into the exponent.

   `trnblas.ozaki2.packed_to_fp64_bits` is the same arithmetic on the host.
   It is checked against exact rational rounding, ties and carries
   included. Results outside the normal float64 range are flagged and
   computed on the host as ldexp(RN64(X'), E). Packed widths other than 24
   bits (b8 beyond 16 moduli, b9 beyond 8) are combined on the host by the
   same rule (`trnblas.ozaki2.combine_packed`), and so is
   `TRNBLAS_OZAKI_DEVICE_COMBINE=0`.

`moduli` is the precision knob, about 3.3 bits per modulus (n GEMMs, against
k(k+1)/2 for Ozaki I). The default of 17 gives about 53 bits of |A||B|, the
level of `slices=8` (36 GEMMs). The result is deterministic: residues are
canonical, so every bit is a function of the scaled integers only,
independent of the tiling, the layout and the LNC split, and equals
`trnblas.ozaki2.reference_gemm`.

On Trn2 (one logical core, LNC=2, kernel time on resident operands, #132),
the device result was bitwise equal to `reference_gemm` at LNC=1 and LNC=2.
At 53 bits, the fused kernel (`crt_mode="fused"`) takes 0.57× Ozaki I's
kernel time at 4096³ and 0.53× at 8192³. On the DF-MP2 shapes
(3267, 8400, 241) and (8192, 3508, 241) it takes 0.53× and 0.57×. At
2048³ and below it is no faster (1.0× at 2048³, 2.0× at 1024³).

Until #132 the operands were scaled and reduced on the host, at about 46 to
155 ns per element, so the prototype lost end to end. The device split and
combine are new, and their Trn2 timing (`scripts/probe_ozaki2.py e2e`) is
pending. `scheme="ozaki1"` stays the default until that timing shows an
end-to-end win. A `scheme="auto"` choice by shape is the follow-up.

## `symm(alpha, A, B, beta=0.0, C=None, side="left", uplo="upper")`

Symmetric matrix-matrix multiply. No NKI kernel: torch on the operands'
device; on the native path, large host problems run as upload + aten.

## `syrk(alpha, A, beta=0.0, C=None, uplo="upper", trans=False)`

Symmetric rank-k update: `C = α·A·Aᵀ + β·C` (or `AᵀA` when `trans=True`).

On Trn1 (XLA path) and under `set_backend("nki")`, dispatches to a
dedicated NKI kernel (`trnblas.nki.nki_syrk`): loads `A` once into SBUF for both operand roles, avoiding the
`A.T.contiguous()` HBM write that `gemm(A, A.T)` would otherwise issue.
Return value is the dense symmetric matrix (both triangles populated);
a small post-kernel `0.5·(C + Cᵀ)` symmetrisation protects against fp32
reduction-order asymmetry. On the native path, `auto` routes it like
`gemm` (aten on the device, or host CPU for small host calls).

As in BLAS (`dsyrk`, `cublasSsyrk`), only the `uplo` triangle of an `n×n`
`C` is referenced; the other triangle is ignored (since 0.11.0, #74). A `C`
whose off-diagonal entries all lie in the opposite triangle still uses that
triangle and emits a `DeprecationWarning`: pass the matching `uplo`. A `C`
that broadcasts against the result (scalar, row, column) is added and the
sum symmetrised.

## `trsm(alpha, A, B, side="left", uplo="upper", trans=False, diag="nonunit")`

Triangular solve: solves `op(A)·X = α·B` (or `X·op(A) = α·B` when `side="right"`).

On Trainium + `side="left"`, dispatches to a blocked panel algorithm
(`trnblas.nki.nki_trsm`): tiny diagonal panels solve via
`torch.linalg.solve_triangular`; trailing off-diagonal updates run
through `nki_gemm` (Trn1 / `set_backend("nki")`) or aten on the device
(native `auto`). On the native path, host operands solve with LAPACK on
the host instead (v0.9.0). Covers all combinations of `uplo ∈ {"upper", "lower"}`,
`trans ∈ {True, False}`, `diag ∈ {"unit", "nonunit"}`. `side="right"`
falls back to direct `torch.linalg.solve_triangular`.

## `metric_inv_half(J)`

Cholesky inverse factor of an SPD metric: `X = L^-T` where `J = L·Lᵀ`, so
`A @ X` is the density-fitting metric contraction. On the native path with
`J` on the `neuron` device and naux ≤ 1024, the factor is computed in
float64 LAPACK on the host and uploaded once; otherwise it is
`torch.linalg.cholesky` + `trsm` (v0.8.0). `TRNBLAS_METRIC_SOLVE=auto|host|device`
overrides the choice.

## `trmm(alpha, A, B, side="left", uplo="upper", trans=False, diag="nonunit")`

Triangular matrix-matrix multiply: `B = α·op(A)·B`.

## Flag values

`uplo` takes `"upper"` / `"lower"`, `side` takes `"left"` / `"right"` and
`diag` takes `"unit"` / `"nonunit"`; the BLAS letters (`"U"`/`"L"`,
`"L"`/`"R"`, `"U"`/`"N"`) are accepted too, in either case. `trans`,
`transA` and `transB` are bools; the BLAS characters `"N"` / `"T"` are
accepted. Any other value raises `ValueError` (the same rules apply to the
Level 2 ops).
