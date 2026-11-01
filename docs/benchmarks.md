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
paid once per instance lifetime). **Device HBM note (confirmed 2026-04-21):** at
medium shape, all 64 loaded energy NEFFs remain resident after the cold pass —
12.6 GB DMA spill + 900 MB model code = 15.9 GB total. A warm pass in the same
process fails with `Failed to allocate 1.500GB (usage: tensors)` — no headroom
remains. Warm timing must be measured in a separate process that loads from the
EBS NEFF cache; `run_bench.sh` does this via `--passes cold` then `--passes warm`.

Energies agree to FP32 noise: -2.487220e+00 (torch), -2.487219e+00 (fused-gemm),
-2.487221e+00 (batched-pair fallback), -2.487218e+00 (chunked NKI).

**Reading the medium numbers:** The chunked dispatch closes the medium-shape gap —
5.2× faster than torch chunk-GEMM, 3.4× faster than the v0.5.2 CPU fallback.
Each i-dispatch is ~24 ms (XLA overhead + ~1 ms Tensor Engine execution), so
64 dispatches costs 1.536 s instead of the 409 s that a per-pair loop would require.
The full-batch kernel's 18 GB XLA graph is replaced by 64 × ~1.4 GB graphs that
compile and cache normally.

## DF-MP2 end-to-end on Trn2 (native TorchNeuron): NKI vs aten vs CPU

trn2.3xlarge, SDK 2.33 (torch-neuronx 2.14, NKI 0.7), one LNC=2 logical core,
fp32, warm median (the first rep compiles). CPU is `set_backend("pytorch")` on
**2 host threads** (`OMP_NUM_THREADS=2`, `taskset -c 4,5`), so it is a floor for
the box rather than a CPU-node baseline. "aten" is the same algorithm with
`set_backend("pytorch")` on `neuron`-device tensors. "NKI" is
`set_backend("nki")`, device-resident inputs, `--energy pair`. Reproduce with
`scripts/bench_df_mp2_paths.py` (PySCF RHF + DF, weigend aux unless "RI" =
cc-pVTZ-RI). aten and NKI after the change were measured in the same session.

| molecule / basis | nbasis/nocc/naux | CPU | aten | NKI before | NKI after |
|---|---|---|---|---|---|
| H₂O / cc-pVDZ | 24/5/71 | 0.3 ms | 16.9 ms | 15.2 ms | 8.9 ms |
| CH₄ / cc-pVTZ | 86/5/93 | 1.2 ms | 8.3 ms | 22.2 ms | 8.2 ms |
| glycine / cc-pVDZ | 95/20/300 | 66 ms | 29.8 ms | 27.3 ms | 15.7 ms |
| glycine / cc-pVTZ | 220/20/300 | 220 ms | 15.0 ms | 61.8 ms | 19.7 ms |
| glycine / cc-pVTZ, RI | 220/20/555 | 223 ms | 28.8 ms | 93.1 ms | 23.1 ms |
| (H₂O)₃ / cc-pVTZ, RI | 174/15/423 | 84 ms | 19.8 ms | 48.8 ms | 20.5 ms |
| synthetic | 512/64/1536 | 18.4 s | 303 ms | 8.0 s | 226 ms |

- Below about nbasis 100 with nocc 5, the CPU wins: the whole calculation
  takes about 1 ms or less there, and any device path costs 5–10 ms of fixed overhead.
- On the real molecules, the NKI path after this change runs the same aten
  GEMMs as the aten path, and every energy step there is below the
  sym-kernel crossover. The two match within run-to-run noise, which comes
  almost entirely from the Cholesky + TRSM stage (5–26 ms on the device).
- At 512/64/1536, `pair_energy_sym` takes the energy step from 207 ms (aten)
  to 120 ms, which is 1.34× end to end. At 96/672/2304 the energy step alone
  drops from 1.22 s (aten chunked reduction, warm, device mostly free) to
  0.69 s. An earlier version of this page gave 63 s for aten there. That is
  the same reduction run while 10–14 GB of the 24 GB logical core are already
  held; see the next section.
- Host-resident inputs (the old `df_mp2_pyscf.py` default) cost 2–5× end to
  end: every NKI launch copies its operands over and back.

## DF-MP2 on Trn2: metric factor on the host

`trnblas.metric_inv_half(J)` returns X = L⁻ᵀ for J = L Lᵀ. On the native path
with J on the device and naux ≤ 1024, it copies J to the host, runs LAPACK in
float64 and uploads X once. Above 1024 it runs `torch.linalg.cholesky` + `trsm`
on the device, as before. `TRNBLAS_METRIC_SOLVE=auto|host|device` overrides
the choice.

Crossover (`scripts/bench_metric_solve.py`). Setup: trn2.3xlarge, one LNC=2
core, fp32 J on the device, the host on 2 threads with
`OMP_WAIT_POLICY=PASSIVE`. Times are warm medians in ms. The residual is
max|XᵀJX − I| at cond(J) = 1e4.

| naux | device (chol + trsm) | host (down + fp64 LAPACK + up) | residual, device / host |
|---|---|---|---|
| 71 | 2.6 | 0.5 | 3e-5 / 3e-7 |
| 128 | 3.1 | 0.4 | 3e-5 / 3e-7 |
| 300 | 5.9 | 1.6 | 4e-5 / 2e-7 |
| 423 | 9.4 | 3.1 | 3e-5 / 2e-7 |
| 555 | 13.5 | 9.0 | 7e-5 / 2e-7 |
| 1024 | 38.9 | 40.2 | 4e-5 / 1e-7 |
| 1536 | 93.6 | 115 | 5e-5 / 1e-7 |
| 2304 | 215 | 278 | 4e-5 / 1e-7 |

- **Up to naux 555 the host route is faster.** Above 1536 the device is
  faster. At cond(J) = 1e7, typical of RI metrics, the device residual grows
  to 1e-2–4e-2 while the host stays at 5e-6.
- **The device route compiles once per new naux.** That takes 33–210 s at
  naux 1024–6144 the first time; after that it is cached on disk.

End to end, all with `OMP_WAIT_POLICY=PASSIVE`. Times are warm best-of ms. The
NKI paths use device-resident inputs and `--energy pair`. aten and CPU use
`--energy chunk`. The "device chol" column is #55's behaviour
(`TRNBLAS_METRIC_SOLVE=device`).

| molecule / basis | nbasis/nocc/naux | CPU (2 thr) | aten | NKI, device chol | NKI, host chol | vs aten | |ΔE| (host chol) |
|---|---|---|---|---|---|---|---|
| H₂O / cc-pVDZ | 24/5/71 | 0.4 | 5.6 | 5.5 | 3.1 | 1.81× | 3.2e-08 |
| H₂O / cc-pVTZ | 58/5/71 | 0.8 | 4.6 | 4.4 | 3.3 | 1.39× | 3.5e-07 |
| NH₃ / cc-pVDZ | 29/5/82 | 0.5 | 5.7 | 3.7 | 3.1 | 1.84× | 1.4e-07 |
| NH₃ / cc-pVTZ | 72/5/82 | 1.1 | 6.7 | 5.4 | 3.2 | 2.09× | 3.5e-07 |
| CH₄ / cc-pVDZ | 34/5/93 | 0.7 | 5.6 | 5.4 | 3.3 | 1.70× | 8.5e-09 |
| CH₄ / cc-pVTZ | 86/5/93 | 2.4 | 4.3 | 4.0 | 3.3 | 1.30× | 3.2e-07 |
| glycine / cc-pVDZ | 95/20/300 | 27.4 | 8.1 | 8.5 | 4.5 | 1.80× | 2.9e-07 |
| glycine / cc-pVTZ | 220/20/300 | 230 | 17.3 | 11.7 | 9.6 | 1.80× | 4.1e-08 |
| glycine / cc-pVTZ, RI | 220/20/555 | 276 | 22.3 | 20.8 | 16.5 | 1.35× | 1.5e-07 |
| (H₂O)₃ / cc-pVTZ, RI | 174/15/423 | 93.2 | 15.6 | 12.6 | 14.2 | 1.10× | 2.5e-07 |
| synthetic | 512/64/1536 | 21.4 s | 309 | 232 | 228 | 1.35× | 9.3e-09 |
| synthetic | 768/96/2304 | 120 s | 25.0 s (1.55 s)* | 947 | 948 | 26× (1.6×)* | 6.3e-07 |

\* With the default 1.5 GB chunk budget, the aten energy step at 768/96/2304
takes 24.7 s. A 0.75 GB budget brings it to 1.29 s. The default chunks land
when the ERI, the broadcast metric and B already occupy most of the logical
core, and the eager chunk loop then slows 20–50×. A standalone repro shows
1.22 s with 8 GB held, 35 s with 10 GB and 64 s with 14 GB; at 18 GB it fails
with "Failed to load model" after five minutes. naux 1536 and 2304 are above
the crossover, so the two NKI columns run the same code there.

- **Small molecules (nocc 5):** the host route takes the metric stage from
  1.6–3 ms to 0.4–0.6 ms. The CPU still wins at this size.
- **Glycine and the water trimer:** the device beats the 2-thread CPU by
  6.6–24×, and trnblas beats aten by 1.1–1.8×.
- **Accuracy:** |ΔE| against PySCF stays at 8e-9 to 4e-7 Ha on every path,
  in the FP32 regime. The FP32 GEMMs and energy dominate the error, so the
  more accurate float64 metric factor doesn't show in ΔE here.
- **`OMP_WAIT_POLICY=PASSIVE`:** with the default OpenMP wait policy, torch's
  pool spin-waits on the same 2 pinned cores the Neuron runtime threads use.
  Small uploads then take ~4 ms instead of 0.06 ms, and every device total
  grows by 3–6 ms (H₂O: 8.4 vs 3.1 ms). On the default policy the host route
  still wins: 1.0–2.0× faster than aten, and up to 2× faster than the device
  route, on the real molecules.

## Multi-core on one Trn2 chip (#24)

The setup is trn2.3xlarge, SDK 2.33 (native TorchNeuron), LNC=2, so there are
4 logical cores, and the whole chip was held for each run. Host threads were
pinned to 2 CPUs (`taskset -c 4,5`, `OMP_NUM_THREADS=2`). Times are warm
medians in ms, fp32. The inputs are synthetic `examples/df_mp2._make_inputs`,
and "packed" also symmetrises (μν|P). Reproduce with
`scripts/bench_parallel.py` (SPMD, under `torchrun`) and
`scripts/bench_multicore.py` (single process, and the CPU baselines).

### DF-MP2 end to end, from host inputs

- **single-core** is `examples/df_mp2.df_mp2_energy`
  (`use_batched_pair=True`) on `neuron:0`. Its time is warm compute plus
  one upload of the inputs.
- **SPMD** is `trnblas.parallel.df_mp2_energy`, one process per core.
- **packed** passes the lower triangle of the ERI (`eri_symmetric=True`).

| nbasis/nocc/naux | single-core | SPMD 1 / 2 / 4 | SPMD packed 1 / 2 / 4 | best vs single-core |
|---|---|---|---|---|
| 512/64/1536 | 348 | 346 / 203 / 162 | 309 / 171 / 138 | **2.5×** |
| 768/96/2304 | 1323 | 1326 / 831 / 620 | 1192 / 696 / 459 | **2.9×** |

On 2 CPU threads (`torch` / MKL), the energy step alone takes 14.5 s at
512/64/1536 and 94 s at 768/96/2304.

Scaling efficiency of SPMD against its own 1-rank time, at 2 / 4 ranks:

| shape | full ERI | packed ERI | energy step alone |
|---|---|---|---|
| 512/64/1536 | 85% / 53% | 90% / 56% | 92% / 89% (124 / 67 / 35 ms) |
| 768/96/2304 | 80% / 53% | 86% / 65% | 91% / 90% (691 / 380 / 193 ms) |

- **The energy step scales almost linearly.** All-gathering B (147–567 MB)
  takes 2–11 ms.
- **The rest is fixed cost.** The ERI upload is bound by the host link
  (~13 GB/s for the chip, 1.6 / 5.4 GB). The metric factor runs on rank 0
  (90 / 200 ms) while the other ranks upload. The packed ERI halves the
  upload, and at 768/96/2304 the upload and the metric factor are then about equal.
- Energies agree with the single-core path to 1e-7 relative.

### DF-MP2 pair energy, single process (`TRNBLAS_CORES`)

B is resident on `neuron:0` and staged through the host to the other cores
(`scripts/bench_multicore.py energy`, sym route).

| nocc/nvir/naux | 1 core | 2 cores | 4 cores | SPMD, same step, 4 ranks |
|---|---|---|---|---|
| 64/448/1536 | 120 | 106 | 108 | 35 |
| 96/672/2304 | 685 | 485 | 389 | 193 |

With the data already on each core, the same row split runs 111 / 62 / 40
and 664 / 362 / 189 ms. The difference from the table above is the host
staging. That is why the single-process path is opt-in, and why `auto` splits
only above 1e13 flop.

### GEMM, C = A @ B fp32

**resident** means the shards are already on their cores. **gathered** adds
the all-gather (or, for K, the all-reduce) that gives every rank the full C.
Single-core aten is 4.13 ms at 4096³ and 30.6 ms at 8192³. The NKI kernel
takes 12.1 ms at 4096³. CPU on 2 threads takes 356 ms and 2.79 s.

| 8192³ | 1 | 2 | 4 | 4-core efficiency |
|---|---|---|---|---|
| single process, resident, row (1 thread / thread per core) | 30.6 | 18.0 / 17.8 | 9.3 / 11.6 | 82% |
| single process, resident, col | 30.6 | 18.1 / 18.1 | 12.8 / 12.7 | 60% |
| SPMD row, compute / gathered | 30.2 / 31.2 | 19.1 / 24.1 | 10.6 / 13.5 | 71% / 58% |
| SPMD col, compute / gathered | 30.2 / 33.8 | 19.6 / 26.3 | 10.9 / 15.6 | 69% / 54% |
| SPMD K, compute / all-reduced | 29.9 / 34.7 | 19.4 / 26.8 | 10.6 / 15.0 | 71% / 58% |
| single process, host A, B → host C (row) | 148 | 166 | 204 | — |
| single process, A, B on `neuron:0`, staged (row) | 30.6 | 209 | 456 | — |

At 4096³, resident compute is 4.1 / 2.4 / 1.6–2.3 ms across the modes. The
host-input split runs 42–61 ms and the staged split 53–173 ms.

- **A transparent multi-core `gemm` on `neuron:0` data loses at every size.**
  Staging the operands through the host costs more than the GEMM itself.
  The default `gemm` is therefore unchanged.
- **Host-input GEMM is bound by the host link.** Splitting it only adds
  copies, so it gets slower with more cores.
- **The split pays only when the shards already live on their cores.** For
  compute alone, threads and processes come out the same: at these sizes
  there are few launches, so the host is not the bottleneck. Processes win
  as soon as data has to move, because they can use collectives.

### Transfers and collectives

| | rate |
|---|---|
| host → device, from 1 thread, 4 threads, or 4 processes at once | 12.5–13.0 GB/s total |
| device → host, `copy_` into a reused / pinned buffer | 13.0–13.3 GB/s |
| device → host, `.cpu()` or into fresh memory (> 64 MB) | 2.8–3.1 GB/s |
| all-gather 4 × 147 MB / broadcast 588 MB / reduce-scatter 680 MB | 5–6 / 10 / 12 ms |

## Native `auto` routing of the GEMM family (#62, #63)

trn2.3xlarge, SDK 2.33 (native TorchNeuron), one LNC=2 logical core, host
threads pinned to 2 CPUs (`taskset -c 4,5`, `OMP_NUM_THREADS=2`), warm
medians in ms, inputs built on the host. Every candidate route was timed
with `scripts/bench_auto_routes.py <op> <dtype>`; "`auto` before" is the
v0.8 routing (host operands → upload + NKI kernel), "after" is this
routing. The host CPUs share physical cores with another job on this box, so
host-CPU times varied by up to 2-5x between runs (bf16 most); the
thresholds were fitted to the quieter run.

Host operands, host result:

| op, fp32 | host CPU | upload+aten | upload+NKI | `auto` before | `auto` after |
|---|---|---|---|---|---|
| gemm 128³ | **0.02** | 0.27 | 0.46 | 0.43 | 0.04 |
| gemm 384³ | **0.47** | 0.68 | 0.84 | 0.81 | 0.58 |
| gemm 512³ | 0.87 | **0.54** | 0.67 | 0.73 | 0.79 |
| gemm 4096³ | 438 | **19.5** | 27.6 | 27.5 | 19.6 |
| gemm 8192³ | 3126 | **88.4** | 182 | 182 | 87.2 |
| gemm 16×128×49152 | **1.05** | 2.70 | 2.69 | 2.71 | 1.50 |
| gemm 32×256×196608 | 30.1 | **18.6** | 19.0 | 18.8 | 18.5 |
| batched 16×112×128×384 | 1.51 | **1.12** | 66.7 | 18.4 | 1.20 |
| batched 32×224×768×768 | 31.0 | **10.2** | 101 | 125 | 9.86 |
| syrk 256×256 | **0.34** | 0.55 | 0.80 | 0.77 | 0.57 |
| syrk 8192×2048 | 1077 | **34.9** | 58.6 | 58.1 | 35.1 |
| symm 4096² | 606 | **20.5** | – | 636 | 21.4 |
| trsm n=768, 768 rhs | **2.69** | 12.5 | 11.7 | 8.72 | 5.75 |
| trsm n=2304, 2304 rhs | **59.9** | 96.9 | 100 | 121 | 77.2 |
| trsm n=1024, 16384 rhs | 150 | 211 | **127** | 113 | 163 |

- Device-resident operands go to aten, as before: aten beat the NKI kernel at
  every GEMM, batched and SYRK shape and dtype, by 1.1x (32×256×196608) to 6x
  (8192³ bf16: 7.7 vs 47 ms), and 20-310x for batched GEMM, which launches
  the kernel once per slice. `set_backend("nki")` still runs the kernel.
- bf16 moves the host/device crossover up: the host's bf16 matmul (AMX) ran
  4096³ in 115 ms and 32×256×196608 in 4.0 ms (upload + aten: 10.2 ms).
- The TRSM device routes are bound by the diagonal-panel solves (aten and
  NKI updates cost the same), so host operands solve with LAPACK. The one
  loss is a wide right-hand side, 1024 × 16384, where the old host-panel +
  NKI-update path was faster.
- The rule is `macs < threads/2 · (fixed + per_byte · link_bytes)` per (op,
  dtype); see `_HOST_GEMM_MODEL` in `trnblas/nki/dispatch.py`.

Umbrella DF-MP2 profile (`examples/quantum_chemistry/df_mp2_profile.py` from
trnsci/trnsci main, `--modes cpu,host-auto,neuron`), warm total in ms. "before"
is v0.8 in the same session, run right after the "after" rows:

| size (ao/aux/occ), pipeline | cpu (after / before session) | host-auto before | host-auto after | neuron after |
|---|---|---|---|---|
| m (128/384/16), example | 41.3 / 62.8 | 153.2 | 29.0 | 21.9 |
| m (128/384/16), fused | 24.1 / 24.3 | 29.8 | 28.9 | 13.2 |
| l (256/768/32), example | 801 / 811 | 1516 | 710 | 55.3 |
| l (256/768/32), fused | 878 / 845 | 754 | 798 | 46.7 |

- host-auto at l, example: half-transform 137 → 23 ms, metric 187 → 15 ms.
  The energy step (trntensor, on the host in both runs) also went from 1208
  to 673 ms; its cause was not isolated.
- At xs and s (32-64 AOs) host-auto now matches cpu (1.0 / 2.5 ms); an
  earlier v0.8 run took 12.8 / 32.7 ms there.
- The profile labels any non-kernel GEMM route "aten", including the host
  route.
- neuron mode (device-resident inputs) is unchanged within noise: v0.8 took
  21.0 / 17.1 ms (m) and 57.6 / 47.9 ms (l).

## DF-MP2 end-to-end — Trainium1 vs NVIDIA A10G

Synthetic inputs, same seed, same three shapes on both platforms.
Energy matches bit-for-bit within fp32 reduction-order noise.

**Vintage parity:** Trainium1 launched Oct 2022; NVIDIA A10G
(GA102 Ampere) launched Apr 2021 — closest single-GPU match on AWS.
A10G via `g5.xlarge` (~$1/hr), trn1 via `trn1.2xlarge` (~$1.34/hr).

| Shape                | Flops   | trn1 compile-cold | trn1 EBS-warm‡ | trn1 HBM-warm | A10G warm | A10G vs trn1 |
|----------------------|--------:|------------------:|---------------:|--------------:|----------:|-------------:|
| small (128/16/384)   | 3.4 G   | —                 | —              | 0.091 s       | 0.001 s   | 91×          |
| medium (512/64/1536) | 2 757 G | **2100.4 s**††    | **119.8 s**‡‡  | **4.784 s**†  | 0.266 s   | **18×**      |
| large (768/96/2304)  | 20 352 G | **174.5 s**†††   | **80.5 s**§    | —             | 2.018 s   | **40×**      |

†† Medium compile-cold (2026-04-22, empty cache): chol 32.4 s, half 24.6 s, metric 2.0 s,
energy 2041.3 s = **2100.4 s total** (77 unique NEFF compilations at ~27 s each). Prior
measurement of 238.5 s (2026-04-21) was from a partially-warm EBS cache where GEMM/SYRK/TRSM
NEFFs hit cache and only the energy kernel compiled fresh.

‡ **EBS-warm** = fresh process, all NEFFs loaded from EBS NEFF cache (no compilation),
but not yet resident in device HBM. This is the timing experienced by any fresh process
after the instance has been used at least once at this shape.

‡‡ Medium EBS-warm (2026-04-22, fresh compilation): chol 11.5 s, half 9.1 s, metric 0.5 s,
energy 98.7 s = 119.8 s total. Energy still ~99 s because 64 NEFFs load serially at
~1.5 s/NEFF ≈ 96 s DMA + kernel time.

† HBM-warm = NEFFs already resident in device HBM (in-process second pass). Energy step
costs only the kernel dispatch: 64 i-dispatches × ~24 ms = 1.536 s. v0.5.4 chunked
dispatch. **HBM-warm is not reproducible at medium via a separate process:** after the
cold pass, 64 energy NEFFs + GEMM/SYRK/TRSM NEFFs fill 15.9 GB of the 16 GB HBM, leaving
no headroom for tensor allocation in a second pass. The 4.784 s warm figure is from an
earlier tracing run; current architecture cannot re-measure it without HBM OOM.

††† Large compile-cold (2026-04-24, empty energy cache): chol 38.5 s, half 96.5 s,
metric 3.8 s, **energy 35.6 s (PyTorch CPU fallback†††‡)**, total 174.5 s. Half-transform
96.5 s reflects GEMM NEFF compilation for large matrices. E = −4.351185×10¹ Ha.

†††‡ **Large energy uses PyTorch CPU fallback, not NKI.** `_j_batched_kernel` inner loop
tile count `N_A × N_B × N_K = 6 × 6 × 18 = 648` exceeds the NeuronCore NRT hardware
load limit (~192; medium's `4 × 4 × 12 = 192` loads cleanly). Confirmed: j_chunk = 32,
16, 1 all fail with `NRT_RESOURCE`. The dispatch proactively detects this and returns
`_torch_batched_pair_energy` immediately (avoids ~30 min of wasted NEFF compilation).
Chol, half-transform, and metric steps use NKI; energy alone falls back to trn1 CPUs.

§ Large EBS-warm (2026-04-24): chol 12.5 s (NKI), half 30.4 s (NKI EBS load),
metric 2.0 s (NKI), energy 35.5 s (PyTorch CPU, same as cold — no cache benefit), total
80.5 s. The 40× A10G gap reflects the energy step running on trn1 CPUs vs A10G cuBLAS.

† v0.5.4 chunked dispatch. Prior v0.5.3 used CPU fallback (9.910 s). The 18× gap vs A10G
is down from 37× in v0.5.3.

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

Opt-in since #50 (`TRNBLAS_AUTOTUNE=1` or `trnblas.set_autotune(True)`); the
default is the fixed `(128,128,512)` heuristic with no first-call sweep. When
enabled, `nki_gemm` sweeps six tile candidates `{64,128} × {128} × {128,256,512}` on
the first call per shape bucket and caches the winner to
`/var/tmp/trnblas-autotune/cache.json` (overrideable via `TRNBLAS_AUTOTUNE_CACHE`).

### How it works

| Step | Detail |
|---|---|
| Shape bucket | `ceil_pow2(M) × ceil_pow2(K) × ceil_pow2(N)` — all shapes in a DF-MP2 run land in the same bucket |
| Sweep | 3 warm runs per candidate; candidates that don't evenly divide the padded shape are skipped |
| Winner | Stored in-process in `_autotune_mem`; written to JSON cache |
| Cache hit | Same bucket → dict lookup only, no re-sweep |
| Default (since #50) | sweep off → fixed `(128,128,512)`; enable with `TRNBLAS_AUTOTUNE=1` |

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
- **trn2 FP16-accumulate GEMM:** not benchmarked; tracked in
  [#25](https://github.com/trnsci/trnblas/issues/25). Trn2 (native
  TorchNeuron) numbers for the existing paths are in the sections above.
