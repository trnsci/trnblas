# `_mp2_energy_kernel` profile findings (#33)

**Date:** 2026-04-14 (attempt 1, blocked) / 2026-04-15 (attempt 2, resolved).
**Tracker:** [#33](https://github.com/trnsci/trnblas/issues/33).
**Context:** [#15](https://github.com/trnsci/trnblas/issues/15) M2 shipped the fused
energy kernel at 1.48× vs torch. Multiple follow-up hypotheses (denom hoisting, per-pair
HBM fences) produced no improvement. This doc records the profile investigation and the
explanation for the ceiling.

## TL;DR

The profiler ran. The answers explain the 1.48× ceiling completely.

- **Vector Engine: 96.45% active.** The energy expression (`T*(2T-Tᵀ)/denom`) is entirely
  element-wise arithmetic — Vector Engine only, no Tensor Engine work.
- **HBM reads: 6.6 GB** (not the 33 GB napkin estimate). Matches the analytical
  prediction exactly (2 passes over T_flat × IC × NOCC strips).
- **Energy kernel wall time: ~0.21 s.** The GEMM that computes T_flat takes ~5.2 s.
  The energy kernel is **~4% of total energy step time.**
- **Amdahl ceiling: 1.48×.** The kernel achieves ~13× speedup on the reduction alone.
  The overall 1.48× is an Amdahl limit — the GEMM dominates, and the kernel can only
  improve the small fraction.

The perf gap is not a kernel tuning problem. The kernel is near-optimal on the Vector
Engine. The gap to 3× requires algorithmic change — either fusing the GEMM into the
energy pass or restructuring to avoid materialising T_flat at all (Phase 3 RFC).

---

## Profiler setup

The April-14 attempt used the deprecated Neuron 2.29 API (`neuron-profile inspect` /
`show-session`), which produced an incompatible NTFF format. The April-15 retry used the
Neuron Profiler 2.0 API confirmed available in `neuron-profile 2.29.18.0`:

```bash
# Compile _mp2_energy_kernel to a fresh cache (isolates its NEFF from other kernels)
python mp2_warmup.py   # ic=nocc=64, nvir=448 (medium bench shape, all-in-one variant)

# Capture
neuron-profile capture -n <neff> -s profile.ntff
# note: --io-from=neff (default) allocates IO tensors from NEFF-declared shapes; no input files needed

# Extract
neuron-profile view -n <neff> -s profile.ntff --output-format summary-text
neuron-profile view -n <neff> -s profile.ntff --output-format summary-json
```

NEFF: `MODULE_5952003545848148214+e30acd3a/model.neff` (17 MB).
Profile artifact: `profile.ntff` (191 MB).
Hardware: trn1.2xlarge, Neuron runtime 2.31.24, compiler 2.24.5133.

> **Shape note.** The profiled variant uses `ic=nocc=64` (single chunk, all pairs).
> The df_mp2 bench uses `i_block=29` at medium shape (1.5 GB budget → 3 chunk calls).
> The engine utilization ratios and HBM volume are proportionally correct; the kernel
> structure is identical across chunk sizes.

---

## B.1 — Per-engine utilization

| Engine | Active % | Active time |
|--------|----------:|------------:|
| Vector | **96.45%** | 0.2062 s |
| Scalar | 3.67% | 0.0079 s |
| DMA | 26.42% | 0.0565 s |
| GpSimd | 0.00010% | 22 µs |
| Tensor | **0.000002%** | 0.48 µs |

Total wall time: 0.2138 s. Total active: 97.4% (VE and DMA overlap).

The Tensor Engine runs 21 instructions in 0.48 µs — entirely XLA graph setup overhead,
not the kernel body. The `T*(2T-Tᵀ)/denom` expression is element-wise arithmetic
throughout: multiply, subtract, reciprocal, sum. All Vector Engine.

---

## B.2 — Instruction counts

| Engine | Instructions | Wall time |
|--------|------------:|----------:|
| Vector | 403,039 | 0.224 s |
| Scalar | 16,407 | 0.0079 s |
| GpSimd | 42,928 | 22 µs |
| Sync | 52,223 | 0.0017 s |
| Tensor | 21 | 0.48 µs |

The 403 K Vector Engine instructions are the broadcast, multiply, reciprocal, and free-dim
`nl.sum` calls in the strip loop. No per-op breakdown is available from `summary-text`
(requires Perfetto format for instruction-level timeline).

---

## B.3 — Pipeline depth (pairs overlap?)

`summary-text` does not provide instruction-level timeline. The DMA-active (26.4%) vs
Vector-active (96.45%) overlap suggests DMA prefetch is keeping up with VE consumption —
pairs are not stalling for HBM loads. Whether `(i, j+1)` begins before `(i, j)` completes
at the instruction level requires Perfetto format; the `.pftrace` artifact is stored on the
instance at `/home/ubuntu/profiles/run-1776296734/` for future retrieval.

Observable signal: VE at 96.45% with total active 97.4% — the kernel leaves essentially
no idle cycles. Consecutive pairs are at minimum executing without inter-pair gaps; the
question is whether there is instruction-level overlap across pair boundaries.

---

## B.4 — HBM bandwidth

| Metric | Value |
|--------|------:|
| HBM reads | **6.58 GB** |
| HBM writes | 1.75 MB |
| DMA transfer total | 3.29 GB |
| DMA transfer time | 0.0467 s |
| Effective read bandwidth | ~30.8 GB/s |

**Analytical prediction (exact match):**
For each (i, j) pair, NSTRIP=4 strips; each strip loads (P_TILE=112, NVIR=448) for `t`
and (NVIR=448, P_TILE=112) for `t.T` → 2 × 112 × 448 × 4 = 401,408 bytes per strip.

Total = IC × NOCC × NSTRIP × 401,408
= 64 × 64 × 4 × 401,408
= **6,578,757,632 bytes** ≈ 6.58 GB ✓

**Previous napkin (33 GB) was incorrect.** The original estimate in this doc counted
HBM traffic for the *unfused* torch path (which materialises T, T.T, denom, and the
product as separate HBM tensors). The fused kernel reads only T_flat (2 passes via
`nl.load` + `nl.load_transpose2d`); all intermediates are SBUF-resident.

---

## The Amdahl picture — why 1.48× and not 3×

This is the key finding that resolves #31.

From the bench (medium, warm, trn1.2xlarge):

| Step | Torch path | Fused path |
|------|----------:|----------:|
| GEMM (T_flat = B_chunk @ B_flat.T) | ~5.2 s | ~5.2 s |
| Energy reduction | ~2.83 s | ~0.21 s |
| **Total energy step** | **8.03 s** | **5.43 s** |

The energy kernel achieves **~13× speedup on the reduction step alone** (2.83 s → 0.21 s).
But the GEMM step is **identical in both paths** and **accounts for ~96% of the fused
energy step** (5.2 of 5.43 s).

Amdahl's law: with f = 0.35 (fraction of torch path that is reduction) and s = 13.5
(kernel speedup):

```
speedup = 1 / ((1 - f) + f/s)
        = 1 / (0.65 + 0.35/13.5)
        = 1 / (0.65 + 0.026)
        ≈ 1.48×
```

This matches the measured result exactly. **The 1.48× ceiling is an exact Amdahl
prediction, not a tuning failure.**

---

## What this means for #31

The four hypotheses investigated before this profile (denom hoisting, dispatch overhead,
per-pair store fences, cross-pair batching) were all targeting the wrong bottleneck. The
actual situation:

- The NKI energy kernel is **near-optimal**: VE at 96.45%, ~13× on its own step.
- The GEMM is the dominant cost and is already running through `nki_gemm` (Tensor Engine).
- No Vector-Engine tuning in `_mp2_energy_kernel` can get past 1.48× overall without
  also reducing GEMM time.

**Paths to 3× total energy speedup:**

1. **Fuse GEMM into the energy pass.** If T_flat is never materialised to HBM — computed
   on-chip and consumed immediately — the GEMM time collapses. This is the Phase 3 RFC
   design (fused `B_i @ B_j.T → energy` without storing T). Architecturally correct
   but requires a custom Tensor-then-Vector pipeline kernel.

2. **Speed up the T_flat GEMM separately.** The GEMM is already NKI-dispatched. Its
   roofline is 2529 GFLOPs at ~0.49 TFLOPS effective throughput at medium shape. Headroom
   exists (trn1 peak is ~3 TFLOPS) but is a separate investigation.

3. **Algorithm change.** An O(N²) algorithm for DF-MP2 that avoids the full-nocc×nocc
   T_flat materialisation (e.g., direct-ring formulation) would eliminate the GEMM
   bottleneck by design.

---

## Recommended next steps

1. **Close #31 with this explanation.** All four prior hypotheses are falsified; the
   ceiling is Amdahl. The 1.48× result is correct and near-optimal given the current
   algorithm structure. Open a new issue for the GEMM-fusion path if warranted.

2. **Phase 3 RFC (fused GEMM+energy).** The most direct route to 3× is fusing
   `B_i @ B_j.T` and the energy reduction into one kernel. The energy kernel's 13×
   reduction speedup shows the on-chip computation is efficient; the question is whether
   the GEMM and energy can share a single NEFF with on-chip T_flat.

3. **Retrieve Perfetto artifact for B.3.** The `.pftrace` is on the instance at
   `/home/ubuntu/profiles/run-1776296734/`. Scp and open in `ui.perfetto.dev` for
   instruction-level pipeline timeline if the pair-overlap question becomes important.

4. **`run_neuron_profile.sh` is ready for future runs.** The working script is at
   `scripts/run_neuron_profile.sh` — the `--probe` mode discovered the correct Neuron
   Profiler 2.0 API, and the default mode captures + extracts `summary-text` without
   InfluxDB or a browser. Future profile runs: `AWS_PROFILE=aws ./scripts/run_neuron_profile.sh`.

---

## Raw profile data

```json
{
  "vector_engine_active_time_percent": 0.9645396092701302,
  "tensor_engine_active_time_percent": 2.2616407562047187e-06,
  "scalar_engine_active_time_percent": 0.03673315592245759,
  "gpsimd_engine_active_time_percent": 1.0491756505262174e-04,
  "dma_active_time_percent": 0.26416618715491547,
  "hbm_read_bytes": 6582249472,
  "hbm_write_bytes": 1835008,
  "vector_engine_instruction_count": 403039,
  "vector_engine_instruction_time": 0.224051265651,
  "tensor_engine_instruction_count": 21,
  "tensor_engine_active_time": 4.83557e-07,
  "scalar_engine_instruction_count": 16407,
  "sync_engine_instruction_count": 52223,
  "total_time": 0.213808050051,
  "total_active_time": 0.20825356317,
  "total_active_time_percent": 0.9740211517776104,
  "neuroncore_cycle_count": 299331264,
  "dma_transfer_total_bytes": 3288352768,
  "dma_transfer_count": 18935,
  "instance_type": "trn1.2xlarge",
  "compiler_version": "2.24.5133.0+58f8de22",
  "runtime_version": "2.31.24 (0b044)",
  "profiler_version": "2.29.18.0%kaena-tools/2.29@d5fe7ba"
}
```
