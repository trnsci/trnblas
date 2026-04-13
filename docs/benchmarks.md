# Benchmarks

Performance results for trnblas kernels — comparing the PyTorch CPU fallback and NKI Trainium path (when available) across canonical Level 3 BLAS workloads.

## Status

Baseline PyTorch-fallback numbers run on every CI build. NKI numbers are pending on-hardware validation on trn1 / trn2 — run `scripts/run_neuron_tests.sh` to generate them locally once a Neuron CI instance is provisioned (see [AWS Setup](aws_setup.md)).

Until the on-hardware data is stable enough to publish here, refer to the [examples/df_mp2.py](https://github.com/trnsci/trnblas/blob/main/examples/df_mp2.py) `--bench` output for an end-to-end DF-MP2 wall-time pass, and `scripts/run_df_mp2_bench.sh` for the SSM-driven runner.

## Reproducing locally

```bash
pytest benchmarks/ --benchmark-only
```

Or the full DF-MP2 bench:

```bash
./scripts/run_df_mp2_bench.sh
```

## Results table (placeholder)

| Op | Size | PyTorch (CPU) | NKI (Trainium) | Speedup |
|---|---|---|---|---|
| gemm | 1024×1024 | TBD | TBD | TBD |
| gemm | 4096×4096 | TBD | TBD | TBD |
| batched_gemm | 64×256×256 | TBD | TBD | TBD |
| trsm | 1024×1024 | TBD | TBD | TBD |
| syrk | 1024×1024 | TBD | TBD | TBD |

Numbers will be populated once the NKI GEMM kernel validates on trn1 / trn2 and the benchmark harness is wired into CI.
