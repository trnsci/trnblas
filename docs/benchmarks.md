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

## NKI SYRK — per-call timing (#18)

`trnblas.syrk` on Trainium dispatches to a dedicated kernel (single-A
HBM load via two `load_transpose2d` calls) rather than
`gemm(A, A.T)`. Correctness: 7/7 `@pytest.mark.neuron` tests pass on
trn1; outputs match `torch.matmul(A, A.T)` to `atol=1e-3, rtol=1e-4`.

Warm-cache per-call timings and effective TFLOPS (mean of 5 runs):

| Shape (M×K) | trn1 warm | trn1 TFLOPS | A10G warm | A10G TFLOPS | A10G vs trn1 |
|-------------|----------:|------------:|----------:|------------:|-------------:|
| 512×512     | 2.45 ms   | 0.11        | 0.11 ms   | 2.39        | 22×          |
| 1024×512    | 6.15 ms   | 0.17        | 0.16 ms   | 6.90        | 38×          |
| 1024×1024   | 7.91 ms   | 0.27        | 0.21 ms   | 10.07       | 38×          |
| 2048×512    | 21.93 ms  | 0.20        | 0.53 ms   | 8.11        | 41×          |

Same pattern as the DF-MP2 end-to-end: the NKI kernel is correct and
well-tiled, but A10G's cuBLAS remains ~30× faster per-call on
Ampere-era single-GPU hardware at these sizes. Reproducible:

```bash
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh     # trn1 correctness
# Then ad-hoc:
python examples/bench_syrk.py                     # cpu
python examples/bench_syrk.py --device cuda       # on a g5.xlarge
```

## NKI batched GEMM

Warm cache, batch=32 of 256×128×256. Per-slice cost after the first is
HBM transfer + Tensor Engine dispatch only (NEFF cache hit).

| Metric    | Value   |
|-----------|--------:|
| Total     | 39.3 ms |
| Per-slice | 1.23 ms |

## DF-MP2 end-to-end — Trainium1 vs NVIDIA A10G

Synthetic inputs, same seed, same three shapes on both platforms.
Energy matches bit-for-bit within fp32 reduction-order noise.

**Vintage parity:** Trainium1 launched Oct 2022; NVIDIA A10G
(GA102 Ampere) launched Apr 2021 — closest single-GPU match on AWS.
A10G via `g5.xlarge` (~$1/hr), trn1 via `trn1.2xlarge` (~$1.34/hr).

| Shape                | Flops   | trn1 warm | A10G warm | trn1 TFLOPS | A10G TFLOPS | A10G vs trn1 |
|----------------------|--------:|----------:|----------:|------------:|------------:|-------------:|
| small (128/16/384)   | 3.4 G   | 0.008s    | 0.001s    | 0.43        | 2.25        | 8×           |
| medium (512/64/1536) | 2 757 G | 9.77s     | 0.266s    | 0.28        | 10.36       | **37×**      |
| large (768/96/2304)  | 20 352 G | 62.84s   | 2.018s    | 0.32        | 10.09       | **31×**      |

**Energy bit-exact across platforms:**

| Shape  | trn1 E_MP2 | A10G E_MP2 |
|--------|------------|------------|
| small  | -1.619250e-04 | -1.619250e-04 |
| medium | -2.487221     | -2.487220 |
| large  | -4.351183e+01 | -4.351184e+01 |

(Medium/large differ by 1 ULP in the last significant figure — expected
fp32 reduction-order variance between platforms, not a correctness
gap.)

### Reading this table

At medium/large, **cuBLAS on A10G is ~30× faster than the current
trnblas torch-matmul path on trn1**. This is the target to close with
NKI kernels. The trn1 numbers above use the v0.4.0 NKI GEMM (validated
but not yet a perf win over torch.matmul fallback in this pipeline); the
fused `nki_mp2_energy` kernel also matches torch rather than beating it
at this scale. Narrowing the gap is the v0.5.0+ kernel work, tracked
under [#15](https://github.com/trnsci/trnblas/issues/15),
[#18](https://github.com/trnsci/trnblas/issues/18) (syrk), and
[#19](https://github.com/trnsci/trnblas/issues/19) (trsm).

At large the energy step is 100% of A10G wall-time (2.016s of 2.018s)
— memory-bandwidth bound on the `T` tensor + intermediates. Same shape
on trn1 is 92%, so the bottleneck is architectural, not platform-specific.

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

# Same workload on A10G (cuBLAS reference for the same vintage):
AWS_PROFILE=aws ./scripts/run_cuda_bench.sh --shape medium
```

See [AWS Setup](aws_setup.md) for the one-time Terraform provisioning
for each instance (`infra/terraform/` for trn1, `infra/terraform-cuda/`
for the A10G).

## Out of scope

- **`syrk` / `trsm` NKI numbers:** those ops are PyTorch-only in v0.4.x;
  v0.5.0 will add NKI kernels and a dedicated row here.
- **cuBLAS head-to-head:** requires GPU access; tracked under
  [#4](https://github.com/trnsci/trnblas/issues/4).
