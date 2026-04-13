# Level 3 — Matrix-matrix operations

Level 3 BLAS: `O(n³)` matrix-matrix operations. This is the hot path for
scientific computing workloads and the primary target for NKI acceleration.

## `gemm(alpha, A, B, beta=0.0, C=None, transA=False, transB=False)`

General matrix-matrix multiply: `C = α·op(A)·op(B) + β·C`.

Dispatches to NKI GEMM kernel with stationary tile reuse on Trainium;
falls back to `torch.matmul` on CPU/GPU.

## `batched_gemm(alpha, A, B, beta=0.0, C=None, transA=False, transB=False)`

Batched GEMM over the leading dimension — used for DF-MP2 tensor contractions
over auxiliary basis indices.

## `symm(alpha, A, B, beta=0.0, C=None, side="left", uplo="upper")`

Symmetric matrix-matrix multiply.

## `syrk(alpha, A, beta=0.0, C=None, uplo="upper", trans=False)`

Symmetric rank-k update: `C = α·A·Aᵀ + β·C` (or `AᵀA` when `trans=True`).

Dispatches to a dedicated NKI kernel (`trnblas.nki.nki_syrk`) on
Trainium: loads `A` once into SBUF for both operand roles, avoiding the
`A.T.contiguous()` HBM write that `gemm(A, A.T)` would otherwise issue.
Return value is the dense symmetric matrix (both triangles populated);
a small post-kernel `0.5·(C + Cᵀ)` symmetrisation protects against fp32
reduction-order asymmetry.

## `trsm(alpha, A, B, side="left", uplo="upper", trans=False, diag="nonunit")`

Triangular solve: solves `op(A)·X = α·B` (or `X·op(A) = α·B` when `side="right"`).

On Trainium + `side="left"`, dispatches to a blocked panel algorithm
(`trnblas.nki.nki_trsm`): tiny diagonal panels solve via
`torch.linalg.solve_triangular`; trailing off-diagonal updates run
through `nki_gemm`. Covers all combinations of `uplo ∈ {"upper", "lower"}`,
`trans ∈ {True, False}`, `diag ∈ {"unit", "nonunit"}`. `side="right"`
falls back to direct `torch.linalg.solve_triangular`.

## `trmm(alpha, A, B, side="left", uplo="upper", trans=False, unit=False)`

Triangular matrix-matrix multiply: `B = α·op(A)·B`.
