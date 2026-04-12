# trnblas

BLAS operations for AWS Trainium via NKI (Neuron Kernel Interface).

Trainium ships no BLAS library. `trnblas` provides Level 1-3 BLAS operations with NKI kernel acceleration on the Tensor Engine, targeting scientific computing workloads that are GEMM-dominated.

Part of the **trn-\*** scientific computing suite by [Playground Logic](https://playgroundlogic.co).

## Why

NVIDIA has cuBLAS with 152 optimized routines. Trainium has `torch.matmul`. That's fine for ML training but insufficient for scientific computing codes that need TRSM, SYRK, SYMM, and batched GEMM with specific transpose/scaling semantics.

trnblas closes this gap — same BLAS API surface, NKI-accelerated GEMM on Trainium, PyTorch fallback everywhere else.

## Install

```bash
pip install trnblas

# With Neuron hardware support
pip install trnblas[neuron]
```

## Usage

```python
import torch
import trnblas

# Level 3 — Matrix multiply (the hot path)
C = trnblas.gemm(alpha=1.0, A=A, B=B, beta=0.5, C=C_init, transA=True)

# Batched GEMM (DF-MP2 tensor contractions)
C = trnblas.batched_gemm(1.0, A_batch, B_batch)

# Symmetric matrix multiply (Fock builds)
F = trnblas.symm(1.0, density, H_core, side="left")

# Triangular solve (Cholesky-based density fitting)
X = trnblas.trsm(1.0, L, B, uplo="lower")

# Symmetric rank-k update (metric construction)
J = trnblas.syrk(1.0, integrals, trans=True)

# Level 2 — Matrix-vector
y = trnblas.gemv(1.0, A, x, beta=1.0, y=y)

# Level 1 — Vector operations
y = trnblas.axpy(alpha, x, y)
d = trnblas.dot(x, y)
n = trnblas.nrm2(x)
```

## DF-MP2 Example

```bash
# Run the density-fitted MP2 example (Janesko/TCU use case)
python examples/df_mp2.py --demo
python examples/df_mp2.py --nbasis 100 --nocc 20
```

The example demonstrates all core BLAS operations in a realistic quantum chemistry workflow: Cholesky factorization, triangular solve, half-transform GEMMs, metric contraction, and energy evaluation.

## Operations

| Level | Operation | Description |
|-------|-----------|-------------|
| 1 | `axpy` | y = αx + y |
| 1 | `dot` | x^T y |
| 1 | `nrm2` | ‖x‖₂ |
| 1 | `scal` | x = αx |
| 1 | `asum` | Σ\|xᵢ\| |
| 1 | `iamax` | argmax \|xᵢ\| |
| 2 | `gemv` | y = α op(A) x + βy |
| 2 | `symv` | y = α A x + βy (A symmetric) |
| 2 | `trmv` | x = op(A) x (A triangular) |
| 2 | `ger` | A = α x yᵀ + A |
| 3 | `gemm` | C = α op(A) op(B) + βC |
| 3 | `batched_gemm` | Batched GEMM |
| 3 | `symm` | C = α A B + βC (A symmetric) |
| 3 | `syrk` | C = α A Aᵀ + βC |
| 3 | `trsm` | Solve op(A) X = αB |
| 3 | `trmm` | B = α op(A) B |

## Status

- [x] Level 1-3 BLAS with PyTorch backend
- [x] GEMM with NKI dispatch stub
- [x] DF-MP2 example (Janesko/TCU use case)
- [ ] NKI GEMM kernel validation on trn1/trn2
- [ ] NKI GEMM with stationary tile reuse
- [ ] Batched GEMM NKI kernel
- [ ] Double-double FP64 emulation
- [ ] Benchmarks vs cuBLAS

## Related Projects

| Project | What |
|---------|------|
| [trnfft](https://github.com/scttfrdmn/trnfft) | FFT + complex ops for Trainium (Williamson/OSU use case) |
| trnsolver *(planned)* | Linear solvers and eigendecomposition |

## License

Apache 2.0 — Playground Logic LLC
