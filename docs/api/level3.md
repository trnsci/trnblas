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

## `gemm_fp64(A, B, *, slices=8, transA=False, transB=False)`

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
4. **Combine (host).** C = (hi + lo) · 2^(e_i + e_j − 12 − 7(k−1)) in fp64.

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
