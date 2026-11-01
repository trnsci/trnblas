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

## `trmm(alpha, A, B, side="left", uplo="upper", trans=False, unit=False)`

Triangular matrix-matrix multiply: `B = α·op(A)·B`.
