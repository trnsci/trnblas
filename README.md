# trnblas

[![CI](https://github.com/trnsci/trnblas/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnblas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/trnsci/trnblas/graph/badge.svg)](https://codecov.io/gh/trnsci/trnblas)
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
- **[Phase 2 — precision](https://github.com/trnsci/trnblas/issues/22)** (deferred): double-double FP64 GEMM for chemistry workloads. FP32 measured sufficient for DF-MP2 through glycine/cc-pVDZ and H₂O/cc-pVTZ (see [Precision envelope](docs/architecture.md#precision-envelope)). Unblocks [trnsolver#27](https://github.com/trnsci/trnsolver/issues/27) and [trntensor#28](https://github.com/trnsci/trntensor/issues/28).
- **[Phase 3 — perf](https://github.com/trnsci/trnblas/issues/23)**: **complete** (tile autotuner, fused DF-MP2 pair-energy kernels, NEFF cache reuse). On Trn2 the native path adds measured `auto` routing (v0.9.0) and the fused `pair_energy_sym` kernel, 1.8× faster than aten on the energy step at 96/672/2304.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnblas/issues/24)**: **complete as of v0.9.0** for one Trn2 chip: [`trnblas.parallel`](docs/api/parallel.md) SPMD GEMM and DF-MP2 across NeuronCores (2.5–2.9× on 4 cores).
- **[Phase 5 — generation](https://github.com/trnsci/trnblas/issues/25)**: trn2 FP16-accumulate GEMM path.

Suite-wide tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Why

NVIDIA has cuBLAS with 152 optimized routines. Trainium has `torch.matmul`. That's fine for ML training but insufficient for scientific computing codes that need TRSM, SYRK, SYMM, and batched GEMM with specific transpose/scaling semantics.

trnblas closes this gap — same BLAS API surface, NKI kernels on Trainium, PyTorch fallback everywhere else. On Trn2/Trn3 (native TorchNeuron), `auto` routes each GEMM-family call by measured time: device-resident operands run as aten on the `neuron` device (the NKI GEMM is 2.4–3.2× slower there), host operands run on the host CPU below a size threshold and as upload + aten above it. The fused DF-MP2 pair-energy kernel (`pair_energy_sym`) still runs at large sizes, where it beats aten.

## Install

```bash
pip install trnblas

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnblas[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnblas[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Kernels dispatch on whichever Neuron PyTorch lineage is installed:

- **TorchNeuron (native)**: Trn2/Trn3, Neuron SDK 2.33+, Python ≥ 3.12,
  `torch` 2.14, `torch-neuronx` 2.14.*. Each kernel is a PyTorch custom op
  (`torch.ops.trnblas.*`) on the `neuron` device; no `torch_xla`.
- **PyTorch/XLA**: Trn1 (and any host on the `torch-neuronx` 2.9 lineage).
  Kernels dispatch through `torch_xla`, unchanged.

trnblas picks the native path when the native `torch_neuronx` and a `neuron`
device are present, and the XLA path when `torch_xla` is installed; without
`nki` it uses `torch.matmul`. Set `TRNSCI_NKI_DISPATCH=native` or `xla`
(shared by all trnsci libraries) to force a path. With an editable install
(`pip install -e`), set `NKI_ENABLE_TRACE_CACHE=0`: the NKI kernel cache only
versions installed wheels and can serve stale kernels. See
[installation](docs/installation.md).

### Known platform issue: NKI trace cache and kernel factories

`import torch_neuronx` turns on the NKI trace cache, and the cache key ignores
constants captured in a kernel's closure: two `@nki.jit` kernels built by one
factory, differing only in a captured value, share one trace, and the second
silently returns the first's results
([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454)).
trnblas's kernels do not rely on closure constants (or on module globals that
can differ between processes): every value that specialises a kernel is a
kernel argument, and `tests/test_trace_cache_safety.py` enforces this. If you
write your own NKI kernels with a factory, set `NKI_ENABLE_TRACE_CACHE=0`
before importing `torch_neuronx`.

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

The example demonstrates all core BLAS operations in a realistic quantum chemistry workflow: the metric factor J^-1/2 (`trnblas.metric_inv_half`, Cholesky + triangular solve; float64 LAPACK on the host for naux ≤ 1024 on the native path), half-transform GEMMs, metric contraction, and energy evaluation.

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
- [x] NKI GEMM kernel validation on trn1/trn2
- [x] NKI GEMM with stationary tile reuse
- [x] Batched GEMM NKI kernel
- [x] Native TorchNeuron dispatch (Trn2/Trn3) with measured `auto` routing
- [x] Multi-core DF-MP2 and GEMM (`trnblas.parallel`)
- [x] Benchmarks vs cuBLAS
- [ ] Double-double FP64 emulation

## Related Projects

| Project | What |
|---------|------|
| [trnfft](https://github.com/trnsci/trnfft) | FFT + complex ops for Trainium |
| [trnrand](https://github.com/trnsci/trnrand) | Random number generation (Philox/Sobol) for Trainium |
| [trnsolver](https://github.com/trnsci/trnsolver) | Linear solvers and eigendecomposition |
| [trnsparse](https://github.com/trnsci/trnsparse) | Sparse formats and SpMM |
| [trntensor](https://github.com/trnsci/trntensor) | einsum and tensor decompositions |

## License

Apache 2.0 — Copyright 2026 Scott Friedman


## Disclaimer

trnsci is an **independent open-source project**. It is not sponsored by, endorsed by, or affiliated with Amazon.com, Inc., Amazon Web Services, Inc., or Annapurna Labs Ltd.

"AWS", "Amazon", "Trainium", "Inferentia", "NeuronCore", "Neuron SDK", and related identifiers are trademarks of their respective owners and are used here solely for descriptive and interoperability purposes. Use does not imply endorsement, partnership, or any other relationship.

All work, opinions, analyses, benchmark results, architectural commentary, and editorial judgments in this repository and on [trnsci.dev](https://trnsci.dev) are those of the project's contributors. They do not represent the views, positions, or commitments of Amazon, AWS, or Annapurna Labs.

Feedback directed at the Neuron SDK or Trainium hardware is good-faith ecosystem commentary from independent users. It is not privileged information, is not pre-reviewed by AWS, and should not be read as authoritative about product roadmap, behavior, or quality.

For official AWS guidance, see [aws-neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/) and the [AWS Trainium product page](https://aws.amazon.com/ai/machine-learning/trainium/).
