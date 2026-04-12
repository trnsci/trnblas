# Level 2 — Matrix-vector operations

Level 2 BLAS: `O(n²)` matrix-vector operations. Currently PyTorch-only; no
NKI acceleration because the Tensor Engine is optimised for matrix-matrix work.

## `gemv(alpha, A, x, beta=0.0, y=None, trans=False)`

General matrix-vector multiply: `y = α·op(A)·x + β·y`.

## `symv(alpha, A, x, beta=0.0, y=None, uplo="upper")`

Symmetric matrix-vector multiply (A symmetric, only `uplo` triangle referenced).

## `trmv(A, x, uplo="upper", trans=False, unit=False)`

Triangular matrix-vector multiply: `x = op(A)·x`.

## `ger(alpha, x, y, A=None)`

Rank-1 update: `A = α·x·yᵀ + A`.
