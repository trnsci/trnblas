# Benchmarks

!!! warning "v0.4.x trn1 numbers were CPU torch.matmul, not NKI (fixed in v0.4.3)"

    Releases v0.4.0 / v0.4.1 / v0.4.2 published "trn1 NKI" tables in this
    page and in [CHANGELOG](https://github.com/trnsci/trnblas/blob/main/CHANGELOG.md).
    A PJRT-plugin path resolution bug (our SSM runners didn't put the
    Neuron venv's `bin/` on `$PATH`) caused every NKI dispatch to fail
    with `FileNotFoundError: 'libneuronpjrt-path'`; the
    `_nki_*_impl.try/except` wrappers silently fell back to
    `torch.matmul` for every one of those runs. As a result, each "trn1
    NKI" warm number on this page through v0.4.2 reflects **trn1's
    8-vCPU Xeon**, not the Trainium Tensor Engine.

    Fix landed in v0.4.3 (commit `d1b481f`): PATH prepend in SSM
    runners + `NkiFallbackWarning` + `test_nki_really_runs.py` that
    forces `TRNBLAS_REQUIRE_NKI=1`. The tables below are re-measured
    from the same commit under real NKI dispatch (NEFF compile visible
    on cold, 10-15000× cold/warm ratios confirm the kernel actually
    runs).

    The MP2 energy kernel (`trnblas.nki.nki_mp2_energy`) turned out to
    have a partition-limit bug that was masked by the silent fallback;
    its tests are skipped pending rewrite (tracked in
    [#15](https://github.com/trnsci/trnblas/issues/15)). Not in the
    production DF-MP2 path.

All numbers on `trn1.2xlarge`, `neuronxcc 2.24.5133`, warm NEFF cache
unless noted.

## NKI GEMM — per-call kernel timing

Warm cache, mean of 5 calls. Aligned shapes (multiples of 128). Real
NKI dispatch verified — `test_compile_vs_cache_timing[1024³]` reports
`cold=26.7ms warm=2.3ms speedup=11.8×`, which is a NEFF-compile
signature not reproducible on CPU.

| Shape (M×K×N)      | Warm     |
|--------------------|---------:|
| 512 × 512 × 512    | 1.3 ms   |
| 1024 × 1024 × 1024 | 2.3 ms   |

## NKI TRSM — per-call timing (#19)

`trnblas.trsm` on Trainium uses a blocked panel algorithm: diagonal
panels solved via `torch.linalg.solve_triangular` (tiny P×P, intrinsically
sequential); trailing off-diagonal updates run through `nki_gemm`
(dominant work for large M). Block size fixed at 128; autotuning is
Phase 3 work (#26). Correctness: 7/7 `@pytest.mark.neuron` tests pass
on trn1 across `{lower, upper} × {trans, not}` + unit-diag.

Warm-cache per-call timings (mean of 5, using the DF-MP2 call pattern
`uplo="lower", trans=True`; real NKI + trailing GEMM, v0.4.3-measured):

| Shape (M × N) | trn1 NKI warm | trn1 TFLOPS | A10G warm | A10G TFLOPS | A10G vs trn1 |
|---------------|--------------:|------------:|----------:|------------:|-------------:|
| 512 × 512     | 5.59 ms       | 0.02        | 0.21 ms   | 0.65        | 27×          |
| 1024 × 512    | 13.27 ms      | 0.04        | 0.36 ms   | 1.50        | 37×          |
| 1024 × 1024   | 18.72 ms      | 0.06        | 0.47 ms   | 2.29        | 40×          |
| 2048 × 512    | 35.82 ms      | 0.06        | 0.81 ms   | 2.67        | 44×          |

Cold (first call, includes NEFF compile of each trailing-GEMM tile
signature): 5.8–12.8 s.

Lower TFLOPS than GEMM/SYRK is inherent to TRSM — the sequential
panel solve limits parallelism. On trn1 the blocked structure adds
Python-loop + per-block `nki_gemm` dispatch overhead on top; closing
that gap is a Phase 3 follow-up (autotuner #26 and eventually a pure
NKI substitution kernel).

## NKI SYRK — per-call timing (#18)

`trnblas.syrk` on Trainium dispatches to a dedicated kernel (single-A
HBM load via two `load_transpose2d` calls) rather than
`gemm(A, A.T)`. Correctness: 7/7 `@pytest.mark.neuron` tests pass on
trn1; outputs match `torch.matmul(A, A.T)` to `atol=1e-3, rtol=1e-4`.

Warm-cache per-call timings and effective TFLOPS (mean of 5 runs on
real NKI, v0.4.3-measured):

| Shape (M×K) | trn1 NKI warm | trn1 TFLOPS | A10G warm | A10G TFLOPS | A10G vs trn1 |
|-------------|--------------:|------------:|----------:|------------:|-------------:|
| 512×512     | 2.14 ms       | 0.13        | 0.11 ms   | 2.39        | 19×          |
| 1024×512    | 6.21 ms       | 0.17        | 0.16 ms   | 6.90        | 39×          |
| 1024×1024   | 5.71 ms       | 0.38        | 0.21 ms   | 10.07       | 27×          |
| 2048×512    | 23.89 ms      | 0.18        | 0.53 ms   | 8.11        | 45×          |

Cold (first call, includes NEFF compile): 1.6–11.4 s depending on shape.

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

| Shape                | Flops   | trn1 NKI warm | A10G warm | A10G vs trn1 |
|----------------------|--------:|--------------:|----------:|-------------:|
| small (128/16/384)   | 3.4 G   | 0.091 s       | 0.001 s   | 91×          |
| medium (512/64/1536) | 2 757 G | 9.910 s       | 0.266 s   | **37×**      |
| large (768/96/2304)  | 20 352 G | (not re-run) | 2.018 s   | —            |

**Energy bit-exact across platforms:** E_MP2 matches to fp32 noise for
small (-1.619250e-04) and medium (-2.487220) under real NKI dispatch.

### Reading this table

At medium, **cuBLAS on A10G is ~37× faster than trnblas NKI GEMM on
trn1** — the Ampere GPU is built for matmul-dominant workloads, while
trn1's Tensor Engine has a higher per-call dispatch overhead. At small,
the gap balloons to 91× because NKI dispatch overhead dominates the
actual ~3 Gflops of compute.

**Uncomfortable honest comparison:** trn1's **host Xeon** (8 vCPU)
running `torch.matmul` (the silent-fallback path that v0.4.x
accidentally measured) produces roughly the same warm DF-MP2 numbers as
real NKI dispatch on this workload — the CPU is competitive at
512–1024 scale because NKI kernel launch is ~1-3 ms per call and
trn1.2xlarge's Xeon is fast enough to do 512³ GEMM in the same time.
Trainium's advantage here is a cost story
(trn1.2xlarge at $1.34/hr vs g5.xlarge at $1.006/hr, with the difference
being the 32 GB HBM and 2 NeuronCores that matter more at larger,
memory-bandwidth-bound workloads than these benches touch).

Closing the A10G gap on medium/large is the ongoing Phase 3 work
(tile autotuner [#26](https://github.com/trnsci/trnblas/issues/26),
energy kernel rewrite [#15](https://github.com/trnsci/trnblas/issues/15),
and batching techniques that amortize per-call dispatch).

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
