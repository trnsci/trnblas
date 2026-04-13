# Benchmarks

All numbers on `trn1.2xlarge`, `neuronxcc 2.24.5133`, warm NEFF cache
unless noted. Source: the `examples/df_mp2.py --bench` runs captured in
the v0.4.0 [CHANGELOG](https://github.com/trnsci/trnblas/blob/main/CHANGELOG.md#040--2026-04-12).

## NKI GEMM — per-call kernel timing

Warm cache, mean of 5 calls. Aligned shapes (multiples of 128).

| Shape (M×K×N)  | Per-call |
|----------------|---------:|
| 512 × 512 × 512   | 1.6 ms |
| 1024 × 1024 × 1024 | 4.5 ms |

## NKI batched GEMM

Warm cache, batch=32 of 256×128×256. Per-slice cost after the first is
HBM transfer + Tensor Engine dispatch only (NEFF cache hit).

| Metric    | Value   |
|-----------|--------:|
| Total     | 39.3 ms |
| Per-slice | 1.23 ms |

## DF-MP2 end-to-end

Synthetic inputs. Energy reproducible bit-for-bit across runs.

| Shape              | Flops    | Cold   | Warm   | TFLOPS | E_MP2          |
|--------------------|---------:|-------:|-------:|-------:|---------------:|
| small (128/16/384) | 3.4 G    | 0.025s | 0.008s |  0.43  | -1.619250e-04  |
| medium (512/64/1536) | 2757 G | 12.9s  | 9.77s  |  0.28  | -2.487221      |
| large (768/96/2304) | 20352 G | 65.9s  | 62.8s  |  0.32  | -4.351183e+01  |

At large the energy step is 92% of wall-time — memory-bandwidth bound on
the T tensor + intermediates. The fused `nki_mp2_energy` kernel is
correct but does not yet beat this torch path; see
[#15](https://github.com/trnsci/trnblas/issues/15) for the perf
follow-up.

## NEFF cache warmup

Same suite run twice on a freshly started instance:

| Pass                                    | Wall time         |
|-----------------------------------------|------------------:|
| Cold (first run after instance start)   | 7.01s             |
| Warm (NEFF cache hit + warm XLA graph)  | 2.52s (2.8× faster) |

The cache at `/var/tmp/neuron-compile-cache/` persists across instance
stop/start (EBS-backed), so kernel compile cost is paid exactly once
per shape per cache lifetime.

## Reproducing locally

```bash
# Micro-benchmark harness (CPU baselines + NKI when available):
pytest benchmarks/ --benchmark-only

# Full DF-MP2 bench on trn1 (provisions + runs + stops instance):
AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh --shape medium
```

See [AWS Setup](aws_setup.md) for the one-time Terraform provisioning.

## Out of scope

- **`syrk` / `trsm` NKI numbers:** those ops are PyTorch-only in v0.4.x;
  v0.5.0 will add NKI kernels and a dedicated row here.
- **cuBLAS head-to-head:** requires GPU access; tracked under
  [#4](https://github.com/trnsci/trnblas/issues/4).
