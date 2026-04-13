# trnblas

[![CI](https://github.com/trnsci/trnblas/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnblas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trnblas)](https://pypi.org/project/trnblas/)
[![Python](https://img.shields.io/pypi/pyversions/trnblas)](https://pypi.org/project/trnblas/)
[![License](https://img.shields.io/github/license/trnsci/trnblas)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://trnsci.github.io/trnblas/)

BLAS operations for AWS Trainium via NKI (Neuron Kernel Interface).

Trainium ships no BLAS library. `trnblas` provides Level 1-3 BLAS operations with NKI kernel acceleration on the Tensor Engine, targeting scientific computing workloads that are GEMM-dominated.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Current phase

trnblas follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Active work is tracked in phase-labeled GitHub issues:

- **[Phase 1 — correctness](https://github.com/trnsci/trnblas/issues/21)**: **complete as of v0.4.0** (GEMM, SYRK, MP2 energy reduction kernels hardware-validated on trn1; end-to-end DF-MP2 validated against PySCF at nanohartree tolerance).
- **[Phase 2 — precision](https://github.com/trnsci/trnblas/issues/22)** (next): double-double FP64 GEMM for chemistry workloads. Unblocks [trnsolver#27](https://github.com/trnsci/trnsolver/issues/27) and [trntensor#28](https://github.com/trnsci/trntensor/issues/28).
- **[Phase 3 — perf](https://github.com/trnsci/trnblas/issues/23)**: tile sweeps, fused DF-MP2 kernels, true 3D batched GEMM, NEFF cache reuse.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnblas/issues/24)**: tensor-parallel GEMM across NeuronCores.
- **[Phase 5 — generation](https://github.com/trnsci/trnblas/issues/25)**: trn2 FP16-accumulate GEMM path.

Suite-wide tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

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
# Run the density-fitted MP2 example
python examples/df_mp2.py --demo
python examples/df_mp2.py --nbasis 100 --nocc 20
```

The example demonstrates all core BLAS operations in a realistic quantum chemistry workflow: Cholesky factorization, triangular solve, half-transform GEMMs, metric contraction, and energy evaluation.

### Real-molecule validation (via PySCF)

```bash
pip install trnblas[pyscf]
python examples/df_mp2_pyscf.py                       # H2O / STO-3G
python examples/df_mp2_pyscf.py --mol ch4 --basis cc-pvdz
```

Runs SCF + density fitting via PySCF, feeds the integrals through trnblas, and compares to PySCF's own DF-MP2 reference energy. Matches to < 10⁻⁷ Hartree on H2O, CH4, NH3 at cc-pvdz.

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
- [x] DF-MP2 example
- [ ] NKI GEMM kernel validation on trn1/trn2
- [ ] NKI GEMM with stationary tile reuse
- [ ] Batched GEMM NKI kernel
- [ ] Double-double FP64 emulation
- [ ] Benchmarks vs cuBLAS

## Related Projects

| Project | What |
|---------|------|
| [trnfft](https://github.com/trnsci/trnfft) | FFT + complex ops for Trainium |
| [trnrand](https://github.com/trnsci/trnrand) | Random number generation (Philox/Sobol) for Trainium |
| [trnsolver](https://github.com/trnsci/trnsolver) | Linear solvers and eigendecomposition |

## License

Apache 2.0 — Copyright 2026 Scott Friedman
