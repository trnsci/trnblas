# trnblas

BLAS operations for AWS Trainium via NKI (Neuron Kernel Interface).

Trainium ships no BLAS library. `trnblas` provides Level 1–3 BLAS operations
with NKI kernel acceleration on the Tensor Engine, targeting scientific
computing workloads that are GEMM-dominated.

Part of the **trn-\*** scientific computing suite by
[Playground Logic](https://playgroundlogic.co).

## Why

NVIDIA has cuBLAS with 152 optimized routines. Trainium has `torch.matmul`.
That's fine for ML training but insufficient for scientific computing codes
that need TRSM, SYRK, SYMM, and batched GEMM with specific transpose/scaling
semantics.

trnblas closes this gap — same BLAS API surface, NKI-accelerated GEMM on
Trainium, PyTorch fallback everywhere else.

## Primary use case

DF-MP2 quantum chemistry on large molecules (>3000 basis functions), where
sustained GEMM throughput for tensor contractions dominates wall-time. See the
[Architecture](architecture.md) page for the algorithm-to-BLAS mapping.

## Related projects

- [trnfft](https://github.com/scttfrdmn/trnfft) — FFT + complex ops for Trainium.
- [trnrand](https://github.com/trnsci/trnrand) — Random number generation (Philox/Sobol) for Trainium.
- `trnsolver` *(planned)* — Linear solvers and eigendecomposition.
