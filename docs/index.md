# trnblas

BLAS operations for AWS Trainium via NKI (Neuron Kernel Interface).

Trainium ships no BLAS library. `trnblas` provides Level 1–3 BLAS operations
with NKI kernel acceleration on the Tensor Engine, targeting scientific
computing workloads that are GEMM-dominated.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

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

As of v0.4.0, trnblas's DF-MP2 is validated against PySCF to nanohartree
precision on H2O, CH4, and NH3 at cc-pVDZ. Run the end-to-end example with:

```bash
pip install trnblas[pyscf]
python examples/df_mp2_pyscf.py --mol ch4 --basis cc-pvdz
```

## Related projects

- [trnfft](https://github.com/trnsci/trnfft) — FFT + complex ops for Trainium.
- [trnrand](https://github.com/trnsci/trnrand) — Random number generation (Philox/Sobol) for Trainium.
- `trnsolver` *(planned)* — Linear solvers and eigendecomposition.
