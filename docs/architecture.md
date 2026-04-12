# Architecture

## Layout

```
trnblas/
├── trnblas/
│   ├── __init__.py      # Re-exports all BLAS operations
│   ├── level1.py        # axpy, dot, nrm2, scal, asum, iamax
│   ├── level2.py        # gemv, symv, trmv, ger
│   ├── level3.py        # gemm, batched_gemm, symm, syrk, trsm, trmm
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py  # auto/pytorch/nki dispatch + NKI GEMM kernel
├── tests/
├── examples/df_mp2.py
└── benchmarks/
```

## DF-MP2 → BLAS mapping

The DF-MP2 algorithm maps directly to trnblas Level 3 operations:

| DF-MP2 Step        | Math                                  | trnblas Call                                     |
|--------------------|---------------------------------------|--------------------------------------------------|
| Half-transform     | `(iν|P) = C_occ^T @ (μν|P)`           | `gemm(1.0, C_occ, eri, transA=True)`             |
| MO transform       | `(ia|P) = C_vir^T @ (iν|P)`           | `gemm(1.0, C_vir, iv_P, transA=True)`            |
| Cholesky           | `J = L @ L^T`                         | `torch.linalg.cholesky`                          |
| Metric solve       | `L^T @ X = I`                         | `trsm(1.0, L, I, uplo="lower", trans=True)`      |
| Metric contract    | `B = (ia|Q) @ J^{-1/2}`               | `gemm(1.0, ia_P, J_inv_half)`                    |
| Energy             | `T_ab = B_i @ B_j^T`                  | `gemm(1.0, B[i], B[j], transB=True)`             |

## NKI GEMM strategy

The NKI GEMM kernel uses stationary tile reuse on the Tensor Engine:

1. Load A tile (128×128) to SBUF as stationary operand.
2. Stream B tiles through the systolic array as moving operand.
3. Accumulate partial products in PSUM.
4. One A load serves all B tiles → 2× fewer SBUF loads vs naive.

For DF-MP2, the MO coefficient matrix `C` is the natural stationary operand
since it's reused across all auxiliary basis indices `P`.

## Known gaps

- **NKI GEMM is a stub.** Falls back to `torch.matmul` until validated on
  trn1/trn2. Kernel scaffold is in `trnblas/nki/dispatch.py`.
- **No FP64.** Trainium's Tensor Engine maxes out at FP32. Double-double
  emulation is the path forward for chemistry workloads.
- **Level 1/2 are PyTorch-only.** The Tensor Engine is wasted on vector ops;
  Level 3 is where NKI acceleration pays off.
