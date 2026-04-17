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

## DF-MP2 energy step — 3-way kernel comparison

**Small shape (nbasis=128, nocc=16, nvir=112, naux=384 — 256 pairs),
trn1.2xlarge, warm NEFF cache, v0.5.2:**

| Energy path | Warm energy | Warm total | vs torch |
|---|---:|---:|---:|
| torch (chunk-GEMM baseline) | 0.018 s | 0.096 s | 1× |
| fused-gemm (per-pair, v0.5.1) | 0.381 s | 0.454 s | 21× slower |
| **batched-pair (v0.5.2)** | **0.005 s** | **0.081 s** | **3.6× faster** |

The batched-pair kernel is the first energy path that beats the chunk-GEMM
torch baseline end-to-end on the energy step. Cold energy (first call,
includes NEFF compile): 6.7 s for batched-pair — paid once per instance
lifetime, amortised across all subsequent calls.

Energies agree to FP32 noise: torch / fused-gemm = -1.619250e-04,
batched-pair = -1.619249e-04.

**Medium shape (nbasis=512, nocc=64, nvir=448, naux=1536 — 4096 pairs),
trn1.2xlarge, warm NEFF cache:**

| Energy path | Version | Warm energy | Warm total | vs torch |
|---|---|---:|---:|---:|
| torch (chunk-GEMM baseline) | v0.5.2 | 8.035 s | 9.795 s | 1× |
| fused-gemm (per-pair) | v0.5.2 | 9.174 s | 10.877 s | 1.11× slower |
| batched-pair (CPU fallback†) | v0.5.2 | 5.239 s | 7.111 s | 1.38× faster |
| **batched-pair (chunked NKI‡)** | **v0.5.4** | **1.536 s** | **4.784 s** | **5.2× faster** |

† v0.5.2: NEFF compile for medium-shape batched-pair kernel failed — `nl.affine_range`
traces all loop iterations eagerly at compile time, producing an 18 GB XLA graph IR that
exceeded the trn1 root volume's 16 GB free space. The warm row used the
cached-failed-NEFF path → torch.matmul fallback on CPU.

‡ v0.5.4: Chunked dispatch (issue #46) — outer `i`-loop moved to Python; one
`@nki.jit` call per i-row processes all `nocc` j-pairs. 64 i-dispatches × ~24 ms
each = 1.536 s warm energy (XLA dispatch overhead dominates; Tensor Engine executes
each kernel in ~1 ms). Cold energy = 34 min (77 NEFF compilations at ~27 s each;
paid once per instance lifetime). Device HBM note: 64 loaded energy NEFFs ×
244 MB DMA spill ≈ 15.6 GB fills the 16 GB device; a `Failed to allocate 1.5 GB`
warning is logged during warm setup but computation succeeds.

Energies agree to FP32 noise: -2.487220e+00 (torch), -2.487219e+00 (fused-gemm),
-2.487221e+00 (batched-pair fallback), -2.487218e+00 (chunked NKI).

**Reading the medium numbers:** The chunked dispatch closes the medium-shape gap —
5.2× faster than torch chunk-GEMM, 3.4× faster than the v0.5.2 CPU fallback.
Each i-dispatch is ~24 ms (XLA overhead + ~1 ms Tensor Engine execution), so
64 dispatches costs 1.536 s instead of the 409 s that a per-pair loop would require.
The full-batch kernel's 18 GB XLA graph is replaced by 64 × ~1.4 GB graphs that
compile and cache normally.

## DF-MP2 end-to-end — Trainium1 vs NVIDIA A10G

Synthetic inputs, same seed, same three shapes on both platforms.
Energy matches bit-for-bit within fp32 reduction-order noise.

**Vintage parity:** Trainium1 launched Oct 2022; NVIDIA A10G
(GA102 Ampere) launched Apr 2021 — closest single-GPU match on AWS.
A10G via `g5.xlarge` (~$1/hr), trn1 via `trn1.2xlarge` (~$1.34/hr).

| Shape                | Flops   | trn1 NKI warm | A10G warm | A10G vs trn1 |
|----------------------|--------:|--------------:|----------:|-------------:|
| small (128/16/384)   | 3.4 G   | 0.091 s       | 0.001 s   | 91×          |
| medium (512/64/1536) | 2 757 G | **4.784 s** (v0.5.4†) | 0.266 s | **18×** |
| large (768/96/2304)  | 20 352 G | (not re-run) | 2.018 s   | —            |

† v0.5.4 chunked dispatch (warm total 4.784 s). Prior v0.5.3 used CPU fallback
(9.910 s). The 18× gap vs A10G is down from 37× in v0.5.3.

**Energy bit-exact across platforms:** E_MP2 matches to fp32 noise for
small (-1.619250e-04) and medium (-2.487218) under real NKI dispatch.

### Reading this table

At medium with v0.5.4 chunked dispatch, **cuBLAS on A10G is ~18× faster than
trnblas NKI on trn1** — down from 37× with the v0.5.3 CPU fallback. The chunked
dispatch (64 NKI calls at ~25 ms each) produces 1.536 s warm energy vs 8.035 s
torch baseline, bringing the end-to-end total to 4.784 s. At small, the gap
balloons to 91× because NKI dispatch overhead dominates the actual ~3 GFlops of
compute.

**The XLA dispatch overhead is the honest ceiling.** Each chunked i-call costs
~24 ms (fixed XLA overhead) + ~1 ms (Tensor Engine execution). At medium shape,
64 calls = 1.6 s of dispatch overhead out of the 1.536 s energy step — the
hardware is effectively executing for 64 ms and idling for 1.472 s waiting for
the host. Reducing dispatch count (larger chunks, fewer i-rows per call) requires
solving the XLA graph size problem for larger batches — which is the remaining
Phase 3 frontier.

Closing the A10G gap further requires either batching more pairs per dispatch
(reduces dispatch count) or investing in trn2 (2× NeuronCores, lower per-call
overhead). See [#25 — trn2 benchmarks](https://github.com/trnsci/trnblas/issues/25)
and [#26 — tile autotuner](https://github.com/trnsci/trnblas/issues/26).

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

## Tile-shape autotuner (v0.5.0)

`nki_gemm` now sweeps six tile candidates `{64,128} × {128} × {128,256,512}` on
the first call per shape bucket and caches the winner to
`/var/tmp/trnblas-autotune/cache.json` (overrideable via `TRNBLAS_AUTOTUNE_CACHE`).

### How it works

| Step | Detail |
|---|---|
| Shape bucket | `ceil_pow2(M) × ceil_pow2(K) × ceil_pow2(N)` — all shapes in a DF-MP2 run land in the same bucket |
| Sweep | 3 warm runs per candidate; candidates that don't evenly divide the padded shape are skipped |
| Winner | Stored in-process in `_autotune_mem`; written to JSON cache |
| Cache hit | Same bucket → dict lookup only, no re-sweep |
| Escape hatch | `TRNBLAS_AUTOTUNE=0` → fixed `(128,128,512)`, identical to v0.4.x |

The sweep runs once per shape bucket per instance lifetime (the cache file persists
on EBS across stop/start). DF-MP2's `nocc²` pair loop sees zero sweep overhead after
the first call.

### Measured numbers (trn1.2xlarge, warm NEFF cache)

Hardware sweep timings are recorded after the first DF-MP2 bench run with v0.5.0.
Numbers will be added here once the hardware run completes
([#26](https://github.com/trnsci/trnblas/issues/26) tracking).

## Fused GEMM+energy kernel (v0.5.1)

`nki_fused_gemm_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)` fuses the
two GEMMs (T and T_T) and the VE energy expression into a single `@nki.jit`.
Eliminates the `(nvir, nvir)` T_flat HBM round-trip.

**Measured on trn1 (small shape: nocc=16, 256 pairs):**

| Path | Warm energy step |
|---|---|
| Chunk-GEMM baseline | 0.13 s |
| Per-pair fused (#41, v0.5.1) | 27.8 s |

The per-pair kernel is correct (energies match to 6 significant figures) but
**215× slower** — root cause is Neuron XLA's ~100ms per-NEFF-dispatch
overhead multiplied by 256 pairs. Fixed in v0.5.2 below.

## Batched-pair energy kernel (v0.5.2, #43)

`nki_batched_pair_energy(B, eps_occ, eps_vir)` computes all NOCC² pair
energies in a single `@nki.jit` dispatch, reducing overhead from O(nocc²) to
O(1).

**Measured on trn1 (SHA `7dabe88`, warm NEFF cache, nocc=4 / 16 pairs):**

| Metric | Value |
|---|---:|
| Batched warm | 1.9 ms |
| Per-pair loop (16 pairs, warm) | 25.4 ms |
| Speedup (warm cache) | **13.5×** |

**Reading the 13.5× number:** With a warm NEFF cache each `nki_fused_gemm_energy`
call takes ~1.6 ms (Tensor Engine compute only). 16 pairs × 1.6ms = 25.4ms vs
one batched dispatch at 1.9ms. In production on a cold instance (first DF-MP2
call, nocc=16 / 256 pairs), each per-pair invocation costs ~100ms → 256 × 100ms
= 25.6s vs one batched dispatch → speedup ~1340×. The Spike B measurement
(800× at NOCC=4, 16 pairs) used the cold-cache scenario.

All 10 `TestBatchedPairEnergy` tests passed on trn1 (aligned, unaligned,
vs-fused-gemm cross-check, zero-B). Total suite: 62/62.

## Out of scope

- **cuBLAS head-to-head at batched-pair scale:** planned once PR #44 merges
  and trn1 numbers are available.
- **trn2 benchmarks:** infrastructure provisioned (`infra/terraform-trn2/`),
  hardware investigation deferred ([#25](https://github.com/trnsci/trnblas/issues/25)).
