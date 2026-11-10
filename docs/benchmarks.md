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

## LNC-split GEMM on Trn2 (#79)

`_gemm_lnc_tn_kernel` (A given as A^T) and `_gemm_lnc_kernel` (row-major A)
replace the single-program kernel's design on the native path:

- launched gridless, one program per physical core at LNC=2
  (`num_programs()` / `program_id`): rows of C split in half, or columns when
  M < 256; an odd split overlaps by one row/column, written identically by both;
- a 512-row A^T block (at most 64 KiB per partition) stays stationary in SBUF
  over all of K, and each B tile feeds every M tile; FP32 PSUM, at most 4
  banks per block; B loads up to 2048 columns wide (small M);
- no DMA transposes: A^T tiles are plain `nl.load`s; row-major A is staged
  by DMA a block ahead and transposed on the Tensor Engine;
- every M/K/N remainder in-kernel; no closure or global state in the kernels.

trn2.3xlarge, SDK 2.33, one LNC=2 logical core (`NEURON_RT_VISIBLE_CORES=1`),
host threads pinned to one physical core (`taskset -c 3,9`), device-resident
operands built on the host, warm medians of one call plus a synchronize, in
ms (`scripts/bench_lnc_gemm.py`). The chip was shared with three other
agents' jobs; aten's own time for one shape moved by up to 15% between runs
(4096³ bf16: 1.04–1.21 ms), so ratios within ~5% are ties.

| M×K×N | dtype | aten (A row-major) | aten (A transposed view) | old NKI kernel | `_gemm_lnc_kernel` | `_gemm_lnc_tn_kernel` |
|---|---|---|---|---|---|---|
| 1024³ | fp32 | **0.154** | 0.166 | 0.349 | 0.245 | 0.234 |
| 1024³ | bf16 | **0.105** | 0.121 | 0.233 | 0.184 | 0.179 |
| 2048³ | fp32 | **0.552** | **0.552** | 1.75 | 0.667 | 0.657 |
| 2048³ | bf16 | 0.229 | **0.222** | 0.610 | 0.306 | 0.291 |
| 4096³ | fp32 | **3.72** | 3.86 | 11.5 | 3.90 | 3.88 |
| 4096³ | bf16 | **1.04** | 1.24 | 6.19 | 1.18 | 1.12 |
| 8192³ | fp32 | **31.9** | **31.7** | (116, #65) | 32.0 | 32.3 |
| 8192³ | bf16 | 7.47 | 8.62 | (47, #65) | 7.62 | **7.37** |
| DF-MP2 half m, 16×128×49152 | fp32 | **0.195** | 0.203 | 0.327 | 0.239 | 0.230 |
| DF-MP2 half l, 32×256×196608 | fp32 | 0.921 | 0.930 | 1.06 | **0.599** | 0.603 |
| DF-MP2 half xl, 48×384×442368 | fp32 | 2.68 | 2.69 | – | **1.56** | 1.57 |
| DF-MP2 half xxl, 64×512×786432 | fp32 | **3.95** | 6.29 | – | 3.97 | 3.97 |
| DF-MP2 half l, 32×256×196608 | bf16 | **0.295** | 0.306 | 0.990 | 0.428 | 0.421 |
| DF-MP2 half xl, 48×384×442368 | bf16 | **0.739** | 0.743 | – | 1.07 | 1.06 |
| DF-MP2 half xxl, 64×512×786432 | bf16 | **1.66** | 2.07 | – | 2.02 | 2.02 |
| DF-MP2 metric m, 1792×384×384 | fp32 | **0.119** | 0.129 | 0.254 | 0.198 | 0.190 |
| DF-MP2 metric l, 7168×768×768 | fp32 | 0.355 | **0.325** | 0.730 | 0.466 | 0.411 |
| DF-MP2 metric xl, 16128×1152×1152 | fp32 | 1.28 | **1.22** | 2.65 | 1.58 | 1.33 |
| DF-MP2 metric xxl, 28672×1536×1536 | fp32 | 3.91 | **3.77** | – | 4.31 | 3.98 |
| DF-MP2 metric l, 7168×768×768 | bf16 | 0.173 | **0.163** | 0.356 | 0.267 | 0.252 |
| DF-MP2 metric xl, 16128×1152×1152 | bf16 | 0.405 | **0.388** | 0.980 | 0.539 | 0.483 |
| DF-MP2 metric xxl, 28672×1536×1536 | bf16 | **1.13** | 1.16 | – | 1.57 | 1.32 |

- The new kernels are 1.2–5.5x faster than the old one at every shape both ran.
- **Against aten they win only on the DF-MP2 half-transform in fp32**
  (`C_occ.T @ eri`, a transposed small-M left operand and a very wide right
  one): 1.5x at `l`, 1.7x at `xl`, and 1.6x against the transposed view at
  `xxl` (aten on a contiguous copy of `C_occ.T` ties there). `auto` routes
  that region to `_gemm_lnc_tn_kernel` (below). 8192³ bf16 from a transposed
  view is 1.17x faster than aten on the same view, but it is one shape and
  6144³ / 4096×8192×4096 bf16 did not repeat it, so it is not routed.
- At 4096³ and 8192³ the kernels are within ±5% of aten. At the metric
  shapes (K = N ≤ 1536) aten is 5–25% faster, and below 4096³ aten wins.
- Each call of a TorchNeuron custom op costs about 0.1 ms of host time
  (time to issue: 0.10–0.12 ms, against 0.014–0.024 ms for an aten matmul).
  A synchronous call pays it in full, which is most of the gap at ≤ 2048³.
  Issued back to back, so the host cost overlaps the device, 4096³ bf16 is
  0.97 ms (kernel) against 0.98–1.15 ms (aten) and 1024³ fp32 0.126 against
  0.091.

Device time alone (`neuron-explorer capture` of each op's NEFF, LNC=2; no
host issue cost), ms, with the Tensor Engine's busy fraction and the mean
activity-throttle limit (1.0 = never throttled):

| shape, dtype | aten | aten, transposed view | `_gemm_lnc_tn_kernel` | `_gemm_lnc_kernel` |
|---|---|---|---|---|
| 4096³ fp32 | 3.61 (99.4%, 0.99) | **3.56** (99.3%, 0.99) | 3.69 (96.1%, 0.97) | 3.71 (98.7%, 0.98) |
| 4096³ bf16 | 0.972 (97.6%, 0.97) | 1.03 (96.8%, 0.96) | **0.965** (95.2%, 0.95) | 1.02 (98.0%, 0.95) |
| 16128×1152×1152 bf16 | 0.333 (95.5%, 0.93) | – | **0.327** (90.9%, 0.90) | 0.392 (94.6%, 0.90) |
| 32×256×196608 fp32 | – | 0.839 (97.9%) | **0.438** (89.7%, 0.88) | – |
| 64×512×786432 fp32 | 3.75 (99.1%, 0.86) | 6.11 (96.2%, **0.61**) | **3.52** (94.6%, 0.90) | – |

- On the device the kernels are at parity with aten from 4096³ up and on
  the larger metric shapes; the synchronous-call gap there is the ~0.1 ms
  custom-op issue cost.
- `_gemm_lnc_tn_kernel` at 4096³ fp32 loses 2–4% at its three M-block
  boundaries: the next block's A^T load (8 MiB per core) is not overlapped
  with the current block, so the Tensor Engine idles 25–38 µs, and each idle
  is followed by ~30 µs of activity throttling (half rate). Issuing the next
  load explicitly before the current block's matmuls, or spread across its
  N tiles, measured 1–3% slower (it queues ahead of the current block's B
  tiles), so the kernel does not.
- aten on the transposed view at DF-MP2 `xxl` spends a third of its time
  throttled (mean limit 0.61, 266 throttle events); the kernel runs the same
  bytes at 0.90. At `l`, aten issues 7680 matmuls against the kernel's 1529
  (Tensor Engine busy 98% for 0.84 ms vs 90% for 0.44 ms).
- The first version of the kernel transposed A on the Tensor Engine inside
  each M block, waiting on 128×128 DMA loads: PE stalls of 3–7 µs at every
  block boundary tripped the throttle each time (6.5% of the run at half
  rate), 4.07 vs 3.70 ms at 4096³ fp32. Staging the next block's rows by
  DMA a block ahead removed most of it (3.90).

Win-region boundary, fp32, A a transposed view (aten on the same view):

| M×K×N | K·N | aten | `_gemm_lnc_tn_kernel` | |
|---|---|---|---|---|
| 16×128×49152 | 6.3M | **0.203** | 0.230 | aten |
| 16×256×49152 | 12.6M | 0.280 | 0.273 | tie |
| 32×256×65536 | 16.8M | 0.375 | **0.306** | kernel |
| 32×256×98304 | 25.2M | 0.519 | **0.375** | kernel |
| 16×128×196608 | 25.2M | 0.471 | **0.440** | kernel |
| 64×512×49152 | 25.2M | **0.317** | 0.387 | aten (N < 256 K) |
| 128×512×65536 | 33.6M | **0.379** | 0.460 | aten (M > 64) |
| 128×512×196608 | 101M | **0.898** | 1.03 | aten (M > 64) |
| 192×512×65536 | 33.6M | **0.613** | 0.710 | aten |
| 8×64×393216 | 25.2M | **0.801** | 0.890 | aten (M < 16, K < 128) |

The rule (`_lnc_gemm_wins`): fp32, A a transposed view, 16 ≤ M ≤ 64,
128 ≤ K ≤ 512, N ≥ 256·K and K·N ≥ 2^24. bf16 never qualifies: aten on the
transposed view is 1.1–2.6x faster at every bf16 small-M shape measured but
`xxl` (2.07 vs 2.02 ms, where aten on contiguous A takes 1.66).

Umbrella DF-MP2 profile (`examples/quantum_chemistry/df_mp2_profile.py`,
trnsci/trnsci main, `--modes neuron --pipelines example`), warm ms, same
session; "before" is `TRNBLAS_NATIVE_GEMM=aten` (the previous `auto` for
device-resident operands):

| size (ao/aux/occ) | half-transform before → after | total before → after | half `gemm` route after |
|---|---|---|---|
| m (128/384/16) | 0.7 → 0.7 | 18.0 → 14.8 | aten |
| l (256/768/32) | 1.5 → 1.4 | 52.4 → 52.7 | kernel |
| xl (384/1152/48) | 3.6 → 2.6 | 152.4 → 148.5 | kernel |

The half-transform is a small part of the pipeline (the metric and energy
steps dominate), so the end-to-end change is within run-to-run noise: `m`
takes the same routes in both runs and still moved 18.0 → 14.8 ms.
B and the energy match to the same relative error (1.9–2.0e-7, ≤ 6.9e-8).

**LNC=1** (`NEURON_LOGICAL_NC_CONFIG=1`, one physical core): the same
picture at about twice the time. 4096³ fp32: aten 7.28, kernel 7.45 (LNC=2:
3.72 / 3.88, so the kernel scales 1.92x from one to two physical cores);
DF-MP2 half `l` fp32: 1.74 vs 0.99; `xxl`: 8.68 (transposed view) / 6.58
(contiguous) vs 6.40.

**Accuracy.** Relative error against a float64 product equals aten's
(fp32 1.2–2.8e-7, bf16 1.6–1.7e-3, the bf16 output rounding) at
1×1×1, 5×7×3, 127×129×513, 129×4097×130, 257×640×2049, 300×1000×700 (and
±1 in each dimension), 1023×1025×1031, 2047×63×1999, 300×8300×200,
17×12289×300 and 129×16385×257 (fp32 3.1e-7 against aten's 2.1e-7), at
LNC=2 and LNC=1, with the NKI trace cache on and off.

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

## fp64-class GEMM (`gemm_fp64`)

[`gemm_fp64`](api/level3.md) (#22), measured 2026-11-09 on Trn2
(trn2.3xlarge, torch-neuronx 2.14, neuronx-cc 2.28, NKI 0.7) on **one logical
core at LNC=2**, normal(0, 1) data, warm NEFF cache. Host baselines are torch
fp64 `A @ B` on 2 pinned vCPUs (one physical core, the suite's usual host)
and on all 12 vCPUs.

- `max_cw` is max |C − C_exact| / (|A||B|) on 256 sampled rows, against an
  independent near-correctly-rounded fp64 reference. Host fp64 itself scores
  5e-17 to 2.4e-16.
- `kernel` is the device kernel on resident slices: the sum of its 1024 × 1024
  launches. `bf16 TFLOP/s` counts the k(k+1)/2 slice products; `fp64-equiv`
  counts 2MNK.
- `split` is the host split of A and B. `upload + rest` covers the upload,
  the device transposes and block copies, and launch overheads.
  `download + combine` is the (hi, lo) download and the fp64 recombine. `e2e`
  is `gemm_fp64` from host fp64 to host fp64.
- `bitwise` means the device (hi, lo) equals `ozaki.reference_from_digits`
  bit for bit. It held in every run, at LNC=1 and LNC=2, with the NKI trace
  cache on and off.

| shape (M, K, N) | k (GEMMs) | max_cw | kernel ms | bf16 TFLOP/s | fp64-equiv GFLOP/s | split ms | upload + rest ms | download + combine ms | e2e ms | host fp64 2 vCPU ms | kernel / e2e vs 2 vCPU | host fp64 12 vCPU ms | e2e 12 vCPU ms | bitwise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (1024, 1024, 1024) | 8 (36) | 3.8e-16 | 0.86 | 89 | 2483 | 28 | 5 | 5 | 38 | 35 | 41× / 0.93× | 5 | 23 | yes |
| (1024, 1024, 1024) | 7 (28) | 1.4e-14 | 0.68 | 89 | 3175 | 23 | 8 | 4 | 36 | 35 | 52× / 0.99× |  |  | yes |
| (1024, 1024, 1024) | 6 (21) | 1.9e-12 | 0.52 | 87 | 4134 | 20 | 14 | 4 | 39 | 35 | 68× / 0.91× |  |  | yes |
| (3267, 8400, 241) | 8 (36) | 2.2e-16 | 12.94 | 39 | 1023 | 492 | 42 | 11 | 559 | 197 | 15× / 0.35× | 34 | 259 | yes |
| (3267, 8400, 241) | 7 (28) | 6.0e-15 | 10.02 | 39 | 1320 | 311 | 176 | 11 | 508 | 197 | 20× / 0.39× |  |  | yes |
| (3267, 8400, 241) | 6 (21) | 7.7e-13 | 7.49 | 40 | 1766 | 271 | 165 | 11 | 454 | 197 | 26× / 0.43× |  |  | yes |
| (8192, 3508, 241) | 8 (36) | 1.8e-16 | 10.99 | 49 | 1260 | 367 | 178 | 26 | 583 | 208 | 19× / 0.36× | 34 | 279 | yes |
| (8192, 3508, 241) | 7 (28) | 8.3e-15 | 8.54 | 49 | 1621 | 312 | 178 | 27 | 525 | 208 | 24× / 0.40× |  |  | yes |
| (8192, 3508, 241) | 6 (21) | 1.2e-12 | 6.40 | 49 | 2164 | 277 | 176 | 27 | 486 | 208 | 33× / 0.43× |  |  | yes |
| (2048, 2048, 2048) | 8 (36) | 2.4e-16 | 6.60 | 94 | 2605 | 130 | 8 | 28 | 173 | 237 | 36× / 1.37× | 45 | 87 | yes |
| (2048, 2048, 2048) | 7 (28) | 1.3e-14 | 5.18 | 93 | 3319 | 88 | 50 | 21 | 164 | 237 | 46× / 1.45× |  |  | yes |
| (2048, 2048, 2048) | 6 (21) | 1.7e-12 | 3.90 | 93 | 4409 | 80 | 33 | 21 | 138 | 237 | 61× / 1.72× |  |  | yes |
| (3072, 3072, 3072) | 8 (36) | 2.4e-16 | 22.11 | 94 | 2623 | 290 | 33 | 56 | 401 | 773 | 35× / 1.93× | 142 | 212 | yes |
| (3072, 3072, 3072) | 7 (28) | 1.0e-14 | 17.26 | 94 | 3359 | 229 | 80 | 57 | 382 | 773 | 45× / 2.02× |  |  | yes |
| (3072, 3072, 3072) | 6 (21) | 1.5e-12 | 13.03 | 93 | 4451 | 177 | 109 | 46 | 345 | 773 | 59× / 2.24× |  |  | yes |
| (4096, 4096, 4096) | 8 (36) | 1.9e-16 | 51.55 | 96 | 2666 | 515 | 49 | 129 | 745 | 1820 | 35× / 2.44× | 321 | 372 | yes |
| (4096, 4096, 4096) | 7 (28) | 9.8e-15 | 40.24 | 96 | 3416 | 356 | 199 | 99 | 694 | 1820 | 45× / 2.62× |  |  | yes |
| (4096, 4096, 4096) | 6 (21) | 1.2e-12 | 30.43 | 95 | 4516 | 317 | 202 | 80 | 629 | 1820 | 60× / 2.89× |  |  | yes |

Readings:

- **Accuracy.** k = 8 gives the host fp64 BLAS level (1.8e-16 to 3.8e-16),
  k = 7 about 1e-14, and k = 6 about 1e-12. The accuracy depends on k, not on
  the shape: the slices are exact, and every PSUM group sums exact integers.
- **The kernel** runs at 87–96 TFLOP/s of bf16 slice products on squares (one
  logical core). That is 35–41× host fp64 on 2 vCPUs at k = 8, and 6.2× the
  12-vCPU host at 4096³. On the narrow DF-MP2 shapes it reaches 39–49 TFLOP/s:
  they run as Cᵀ = Bᵀ Aᵀ with only 2 stationary tiles, and look DMA-bound.
  LNC=2 against LNC=1: 1.98× at 2048³ and 1.18× at (3267, 8400, 241).
- **End to end, the host split dominates.** It costs about 11 ns per element
  at k = 8 on one host core (2 vCPUs) and about 3–4 ns on 12. On squares,
  `gemm_fp64` is 1.4–2.4× host fp64 on 2 vCPUs at k = 8 (2048–4096³), but
  only 0.5–0.9× the 12-vCPU host (slower). **On the narrow DF-MP2 shapes it is 2.8× slower
  than host fp64 on 2 vCPUs, and 7–8× slower than on 12.** A has only 2N = 482
  flops per element there, so the split of A alone outweighs the host GEMM.
  The device split (#123, next section) removes this.
- **First call.** One NEFF per (k, K, launch shape). Each new shape compiles
  for a few seconds to about 30 s, plus the device-side slicing ops. An
  uncapped single 4096³ launch took 240 s to compile, hence the 1024 × 1024
  launch cap.

### Device split (#123)

The same runs with the split on the device, measured 2026-11-09 on the same
box (one logical core at LNC=2, normal data, warm NEFF cache). The data is the
same as in the table above (seed 0), so `max_cw` and `kernel` carry over: every
result was bitwise equal to the host-split route, and the device digits were
checked against `ozaki.split` in each run.

- `device split A + B` is the raw-bit upload of A and B (8 bytes per element)
  plus the four split kernels (exponent maxima, digits). `ns / element`
  counts the kernels only.
- `e2e, host split` is the same build with `TRNBLAS_OZAKI_DEVICE_SPLIT=0`,
  measured in the same process.
- Host baselines are torch fp64 `A @ B` on 2 pinned vCPUs. The 12-vCPU host
  baseline and the 12-vCPU column (`gemm_fp64` with 12 host threads, k = 8)
  were measured in a whole-chip slice with the host otherwise quiet.

| shape (M, K, N) | k (GEMMs) | max_cw | device split A + B ms (raw upload; kernels) | ns / element | kernel ms | **e2e ms** | e2e, host split ms | host fp64 2 vCPU ms | **e2e vs 2 vCPU** | host fp64 12 vCPU ms | e2e, 12 vCPU host ms | e2e vs 12 vCPU | bitwise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (1024, 1024, 1024) | 8 (36) | 3.8e-16 | 3.0 (1.4; 1.6) | 0.76 | 0.86 | **8.9** | 40 | 29 | **3.23×** | 5 | 10 | 0.54× | yes |
| (1024, 1024, 1024) | 7 (28) | 1.4e-14 | 3.0 (1.4; 1.5) | 0.73 | 0.68 | **8.5** | 36 | 29 | **3.40×** | 5 |  |  | yes |
| (1024, 1024, 1024) | 6 (21) | 1.9e-12 | 3.0 (1.4; 1.5) | 0.72 | 0.52 | **8.4** | 34 | 29 | **3.44×** | 5 |  |  | yes |
| (3267, 8400, 241) | 8 (36) | 2.2e-16 | 36.2 (17.9; 18.3) | 0.62 | 12.94 | **65.5** | 562 | 198 | **3.03×** | 35 | 68 | 0.51× | yes |
| (3267, 8400, 241) | 7 (28) | 6.0e-15 | 35.4 (17.9; 17.5) | 0.59 | 10.02 | **60.3** | 511 | 198 | **3.29×** | 35 |  |  | yes |
| (3267, 8400, 241) | 6 (21) | 7.7e-13 | 34.2 (17.9; 16.3) | 0.55 | 7.49 | **57.2** | 475 | 198 | **3.46×** | 35 |  |  | yes |
| (8192, 3508, 241) | 8 (36) | 1.8e-16 | 34.7 (18.0; 16.7) | 0.56 | 10.99 | **78.4** | 587 | 206 | **2.63×** | 35 | 79 | 0.44× | yes |
| (8192, 3508, 241) | 7 (28) | 8.3e-15 | 33.9 (18.0; 15.8) | 0.53 | 8.54 | **74.9** | 547 | 206 | **2.76×** | 35 |  |  | yes |
| (8192, 3508, 241) | 6 (21) | 1.2e-12 | 32.9 (18.0; 14.9) | 0.50 | 6.4 | **70.3** | 492 | 206 | **2.94×** | 35 |  |  | yes |
| (2048, 2048, 2048) | 8 (36) | 2.4e-16 | 10.0 (5.1; 4.9) | 0.59 | 6.6 | **39.0** | 185 | 235 | **6.02×** | 46 | 30 | 1.53× | yes |
| (2048, 2048, 2048) | 7 (28) | 1.3e-14 | 9.8 (5.1; 4.7) | 0.56 | 5.18 | **37.4** | 164 | 235 | **6.29×** | 46 |  |  | yes |
| (2048, 2048, 2048) | 6 (21) | 1.7e-12 | 9.6 (5.1; 4.5) | 0.53 | 3.9 | **33.9** | 147 | 235 | **6.93×** | 46 |  |  | yes |
| (3072, 3072, 3072) | 8 (36) | 2.4e-16 | 21.9 (11.4; 10.5) | 0.56 | 22.11 | **93.1** | 397 | 774 | **8.31×** | 147 | 75 | 1.95× | yes |
| (3072, 3072, 3072) | 7 (28) | 1.0e-14 | 21.4 (11.4; 10.0) | 0.53 | 17.26 | **86.7** | 385 | 774 | **8.92×** | 147 |  |  | yes |
| (3072, 3072, 3072) | 6 (21) | 1.5e-12 | 20.8 (11.4; 9.4) | 0.50 | 13.03 | **83.5** | 353 | 774 | **9.26×** | 147 |  |  | yes |
| (4096, 4096, 4096) | 8 (36) | 1.9e-16 | 38.4 (20.2; 18.2) | 0.54 | 51.55 | **180.8** | 739 | 1829 | **10.12×** | 331 | 154 | 2.15× | yes |
| (4096, 4096, 4096) | 7 (28) | 9.8e-15 | 37.5 (20.2; 17.4) | 0.52 | 40.24 | **166.7** | 684 | 1829 | **10.97×** | 331 |  |  | yes |
| (4096, 4096, 4096) | 6 (21) | 1.2e-12 | 36.5 (20.2; 16.3) | 0.49 | 30.43 | **157.0** | 629 | 1829 | **11.65×** | 331 |  |  | yes |

Readings:

- **The split is about 20× cheaper on the device.** It costs 0.50 to 0.62 ns
  per element on one logical core (1.8× slower at LNC=1), against about 11 ns
  for the host split on one host core. It is VectorE-bound: about 136
  VectorE instructions per element at k = 8, using bitvec ops and exact fp32
  arithmetic only. At about one element per cycle per lane on each of the two
  physical cores, that matches the measured rate.
- **`gemm_fp64` now beats host fp64 on 2 vCPUs at every shape.**
  - DF-MP2 shapes: 2.6 to 3.5× (they were 2.3 to 2.8× slower).
  - Squares from 2048³ up: 6 to 12×, 10.1× at 4096³ for k = 8. At 1024³:
    3.2 to 3.4×.
  - Against the 12-vCPU host it is 1.5 to 2.2× faster on squares from 2048³
    up, and still about 2× slower on the DF-MP2 shapes.
- **Why the narrow shapes stop at about 2× the 12-vCPU host.** At
  (3267, 8400, 241) the e2e of about 65 ms is made of:
  - the raw upload of A, 18 ms at about 12.5 GB/s;
  - the split of A, 16 ms;
  - the GEMM kernel, 13 ms;
  - the device transposes of A's digits, 4 ms;
  - the host combine into Cᵀ and the download, about 7 ms.

  The upload plus the split alone, about 34 ms, already equal the 12-vCPU
  host GEMM, and eager uploads do not overlap kernels: an upload queued behind
  a 16 ms split kernel took 24 ms. Beating 12 host vCPUs on these shapes needs
  more than one logical core, or operands that stay resident across calls.
- **Squares.** The host combine (fp64 (hi + lo) times the scales, about 4 ns
  per element on 2 vCPUs, 66 ms at 4096³) is now the largest host cost. It is
  the next candidate to move.
- **First call.** Each split kernel compiles per operand shape: 4 s at 2048²,
  16 s at 4096², and 34 s for the 3267 × 8400 operand. NEFFs are cached
  afterwards.

## Out of scope

- **cuBLAS head-to-head at batched-pair scale:** planned once PR #44 merges
  and trn1 numbers are available.
- **trn2 FP16-accumulate GEMM:** not benchmarked; tracked in
  [#25](https://github.com/trnsci/trnblas/issues/25). Trn2 (native
  TorchNeuron) numbers for the existing paths are in the sections above.
