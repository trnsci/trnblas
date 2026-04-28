# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.5] — 2026-04-27

### Added

- **`--passes` flag for `df_mp2.py --bench`** (`cold`/`warm`/`both`, default `both`).
  On Trainium at medium/large shapes, running both passes in the same process OOMs:
  after the cold pass, all loaded NEFFs remain resident in HBM (64 × 244 MB = 15.6 GB
  DMA spill at medium shape), leaving no headroom for tensor allocation in the warm pass.
  The fix is two separate process invocations; `run_bench.sh` now does this automatically.

- **`scripts/run_bench.sh`** — runs `df_mp2.py --bench --batched-pair-energy` on the
  trn1 CI instance via SSM, cold and warm as separate processes. Supports
  `--shape medium|large` (default: both). Follows the base64-SSM pattern from
  `run_pyscf_tests.sh`.

### Hardware (2026-04-21 / 2026-04-22, trn1.2xlarge, neuronxcc 2.24.5133)

**Medium-shape timing** (`nbasis=512, nocc=64, nvir=448, naux=1536`):

| Step | True compile-cold† | Partial-warm cold†† | EBS-warm‡ |
|---|---:|---:|---:|
| Cholesky | 32.4 s | 29.7 s | 11.5 s |
| Half-transform | 24.6 s | 103.5 s | 9.1 s |
| Metric contraction | 2.0 s | 4.0 s | 0.5 s |
| Energy (64 i-dispatches) | 2041.3 s | 101.3 s | 98.7 s |
| **Total** | **2100.4 s** | **238.5 s** | **119.8 s** |

† True compile-cold (2026-04-22): empty cache (`rm -rf /var/tmp/neuron-compile-cache/`).
All 77 unique NEFFs compiled from scratch — energy dominates at 2041 s (77 compilations
× ~27 s each). `E = −2.487218×10⁰ Ha`.

†† Partial-warm cold (2026-04-21): GEMM/SYRK/TRSM NEFFs hit EBS cache from prior
test-suite runs; only the energy kernel compiled fresh. The 8.8× difference (238 s vs
2100 s) reflects this cache state.

‡ EBS-warm (2026-04-22, fresh compilation): all NEFFs loaded from `/var/tmp/neuron-compile-cache/`
(EBS, no compilation). Half-transform NEFF loads in 9 s (vs 24.6 s compile). Energy
remains ~99 s: 64 energy NEFFs still load serially from EBS at ~1.5 s/NEFF.

**HBM constraint confirmed:** after the medium cold pass, 64 energy NEFFs + GEMM/SYRK/TRSM
NEFFs fill 15.9 GB of the 16 GB device. A subsequent in-process warm pass fails with
`Failed to allocate 1.500GB (usage: tensors)`. The prior 4.784 s warm figure (1.536 s
energy) is in-process HBM-warm; it cannot be reproduced via separate-process EBS loading
(takes ~120 s as shown above).

**Large-shape cold: NRT hardware limit on energy; PyTorch fallback (2026-04-22/24).**

`_j_batched_kernel` inner loop tile count `N_A × N_B × N_K = 6 × 6 × 18 = 648`
exceeds the NeuronCore NRT load capacity regardless of j_chunk size (confirmed via
j_chunk = 1, 16, 32 — all fail with `NRT_RESOURCE`).  Fix: proactive PyTorch fallback
in `_nki_batched_pair_energy_chunked_impl` when `inner_ops > _NRT_INNER_OP_LIMIT (300)`.

**Large-shape timing (2026-04-24, trn1.2xlarge, neuronxcc 2.24.5133):**

| Step | Cold | Warm |
|---|---:|---:|
| Cholesky | 38.5 s (NKI) | 12.5 s (NKI) |
| Half-transform | 96.5 s (NKI compile) | 30.4 s (NKI EBS-warm) |
| Metric contraction | 3.8 s (NKI) | 2.0 s (NKI) |
| **Energy (PyTorch CPU fallback)** | **35.6 s** | **35.5 s** |
| **Total** | **174.5 s** | **80.5 s** |

`~0.12 TFLOPS cold, ~0.25 TFLOPS warm. E = −4.351185×10¹ Ha.`

Energy cold ≈ warm (35.5 s both) — confirming PyTorch CPU (no NKI caching benefit).
Half-transform cold 96.5 s reflects GEMM NEFF compilation for large-matrix shapes.
NKI is used for chol, half-transform, and metric; energy falls back to trn1 CPUs.
NRT fix required for energy NKI: redesign to keep inner tile count ≤ ~200 (options:
move b-strip loop to Python, or use TILE=256 to reduce N_A/N_B from 6 to 3).

## [0.5.4] — 2026-04-17

### Added

- **Chunked batched-pair dispatch (#46, `_j_batched_kernel`).** Extends
  `nki_batched_pair_energy` to medium/large shapes where the full-batch XLA
  graph exceeds available disk during neuronxcc compilation.

  **Root cause recap:** `nl.affine_range` traces all loop iterations eagerly
  into the XLA IR. At small shape (nocc=16, 256 pairs), this produces a ~240 MB
  JSON that compiles fine. At medium shape (nocc=64, 4096 pairs, nvir=448,
  naux=1536), the same tracing generates an 18 GB JSON — larger than the 16 GB
  free space on the trn1 EBS root volume.

  **Fix:** adaptive routing in `_nki_batched_pair_energy_impl` based on estimated
  iteration count (`est_iters = nocc² × N_A² × N_K`). When
  `est_iters > _BATCHED_PAIR_CHUNK_THRESHOLD` (4096), the implementation switches
  to `_nki_batched_pair_energy_chunked_impl`: a Python `for i in range(nocc)` loop
  over `_j_batched_kernel` — a 4-level `@nki.jit` (j → a → b → k) that processes
  all `nocc` j-pairs for one i-row per dispatch.

  **NEFF economics:** all NOCC i-calls share identical input shapes
  (`(nvir_pad, naux_pad)`, `(nocc, nvir_pad, naux_pad)`, `(1,1)`, `(1,nocc)`,
  `(nvir_pad,1)`, `(1,nvir_pad)`) → **one NEFF compile**, NOCC warm dispatches.
  Per-call XLA graph at medium shape: ~1.4 GB (well within available disk).
  Expected warm overhead for nocc=64: 64 × ~100 ms ≈ 6.4 s
  (vs 4096 × ~100 ms ≈ 409 s for per-pair).

  **Backward compatibility:** shapes with `est_iters ≤ 4096` (e.g. small bench,
  nocc=16, small basis) continue to use `_batched_pair_kernel` (full-batch,
  O(1) dispatch) — the 3.6× speedup at small shape is preserved.

  **Public API:** unchanged — `nki_batched_pair_energy(B, eps_occ, eps_vir)`
  routes automatically.

  **Tests:** `test_correctness_chunked_path` (shapes above and below threshold),
  `test_chunked_and_full_batch_agree` (threshold=0 forces chunked, validates
  agreement with full-batch on small shape), `test_chunked_dispatch_overhead`
  (`@pytest.mark.neuron`, timing report with nocc=16 chunked shape).

### Fixed

- **`_j_batched_kernel` SBUF annotation error.** The pre-sliced `eps_occ_i`
  argument (a `(1,1)` XLA tensor) carried a `shared_hbm` memory annotation.
  Using it directly in `nl.add(eps_occ_i, eps_j)` without an explicit
  `nl.load` propagated the `shared_hbm` annotation through the NKI compute
  graph, corrupting the PSUM allocation for `psum_t` and causing neuronxcc
  to fail with:
  ```
  tensor_copy src must be SBUF or PSUM, got shared_hbm
  ```
  Fix: `eo_i = nl.load(eps_occ_i[0:1, 0:1])` before the j-loop, matching the
  pattern in `_batched_pair_kernel` where `eps_occ_i` is always loaded via
  `nl.load` before arithmetic.

### Hardware (v0.5.4, trn1.2xlarge, neuronxcc 2.24.5133)

Chunked dispatch correctness and performance confirmed on hardware
(67 neuron tests pass, 0 failures):

- **`test_chunked_dispatch_overhead`** (nocc=16, nvir=384, naux=512, dispatches=16):
  cold=49.4 ms, warm=38.7 ms per dispatch, per-pair-loop=2031.6 ms,
  speedup=52.6× vs per-pair loop.

- **Medium-shape DF-MP2 bench** (nbasis=512, nocc=64, nvir=448, naux=1536, 4096 pairs),
  `--batched-pair-energy --shape medium`, SHA 2274dec:

  | Step | Cold | Warm |
  |---|---:|---:|
  | chol | 36.475 s | 0.061 s |
  | half-transform | 144.509 s | 2.814 s |
  | metric | 2.747 s | 0.318 s |
  | **energy (chunked NKI)** | **2049.166 s** | **1.536 s** |
  | **total** | **2232.940 s** | **4.784 s** |

  TFLOPS (warm): ~0.58. Energy: E = -2.487218e+00.

  Cold energy = 34 min: 77 NEFF compilations at ~27 s each (paid once per
  instance lifetime on the EBS-backed NEFF cache). Warm energy = 1.536 s
  = 64 i-dispatches × ~24 ms each (XLA overhead dominates; Tensor Engine
  executes each kernel in ~1 ms).

  **vs v0.5.3 fallback:** warm energy 5.239 s → 1.536 s (**3.4× faster**).
  **vs torch chunk-GEMM baseline:** warm energy 8.035 s → 1.536 s (**5.2× faster**).

  Device HBM note: 64 loaded energy NEFFs × 244 MB DMA spill = ~15.6 GB,
  filling the 16 GB device. A `Failed to allocate 1.500 GB` warning is logged
  during warm setup; computation succeeds (Neuron runtime handles gracefully).

- **PySCF FP32 precision envelope (#20).** All 8 `test_precision_envelope` cases
  pass on hardware (2026-04-18, `pytest -m "pyscf and slow"`):

  | Molecule    | Basis   | |ΔE| Ha    | Tol    |
  |-------------|---------|----------:|-------:|
  | H₂O         | sto-3g  | < 1e-6    | 1e-6   |
  | H₂O         | cc-pVDZ | < 1e-5    | 1e-5   |
  | CH₄         | cc-pVDZ | < 1e-5    | 1e-5   |
  | NH₃         | cc-pVDZ | < 1e-5    | 1e-5   |
  | glycine     | sto-3g  | 1.71e-08  | 1e-5   |
  | glycine     | cc-pVDZ | 3.51e-07  | 5e-5   |
  | (H₂O)₃     | sto-3g  | 4.17e-08  | 1e-5   |
  | H₂O         | cc-pVTZ | 1.99e-07  | 1e-4   |

  **Double-double decision:** both gate cases (glycine/cc-pVDZ = 3.51e-07 Ha,
  h2o/cc-pVTZ = 1.99e-07 Ha) are well below the 1 µHartree threshold.
  FP32 is sufficient for DF-MP2 at these molecule/basis combinations.
  [#10](https://github.com/trnsci/trnblas/issues/10) closed as "not needed";
  [#22](https://github.com/trnsci/trnblas/issues/22) (double-double emulation)
  deferred indefinitely.

## [0.5.3] — 2026-04-17

### Added

- **Medium-shape bench numbers** (`docs/benchmarks.md`). 3-way comparison at
  nbasis=512, nocc=64, nvir=448, naux=1536 (4096 pairs). Torch warm total
  9.795s; fused-gemm 10.877s (+11%); batched-pair 7.111s (CPU fallback — NEFF
  compile blocked, see below). See benchmarks.md for the full table.

### Investigation

- **Medium-shape batched-pair NEFF compile fails: XLA graph is 18 GB.**
  `nl.affine_range` traces all loop iterations eagerly into the XLA IR at
  compile time. At nocc=64, nvir=448, naux=1536 (4096 pairs × ~192 inner tile
  operations), the NKI source JSON reaches 18 GB. The trn1 root volume had
  16 GB free — both `/tmp` and `/var/tmp` are on the same 96 GB EBS root
  volume; there is no filesystem routing fix.

  For context: small-shape (nocc=16, 256 pairs) produces a ~240 MB JSON and
  compiles fine. Medium is 75× larger due to more pairs and larger nvir/naux.

  **Fix (deferred, issue #46):** chunked dispatch — call the batched-pair
  kernel with ~256 pairs per dispatch (16 calls for nocc=64). Each chunk's
  XLA graph is ~240 MB; total dispatch overhead is ~1.6 s (vs 409 s per-pair).

- **`TMPDIR=/var/tmp` added to SSM runners** (`run_df_mp2_bench.sh`,
  `run_neuron_tests.sh`, `run_pyscf_tests.sh`) and Terraform user-data as a
  defensive measure. Does not unblock medium-shape batched-pair compilation
  (both paths share the same filesystem), but is correct hygiene for future
  NKI kernels that produce large intermediate files.

## [0.5.2] — 2026-04-16

### Added

- **Batched-pair energy kernel (#43, `nki_batched_pair_energy`).** A single
  `@nki.jit` dispatch computes the full DF-MP2 pair energy for all NOCC²
  orbital pairs, eliminating the ~100ms × nocc² Neuron XLA per-dispatch
  overhead that made `nki_fused_gemm_energy` impractical (215× slower than
  chunk-GEMM at NOCC=16 / 256 pairs).

  **Design:** The kernel has five levels of `nl.affine_range` (i → j → a-strip
  → b-strip → k-tile). For each (i, j, a, b) combination, two GEMMs are
  computed as in `_fused_gemm_energy_kernel`:
  - GEMM 1: `T[a,b] = B[i][a_strip,:] @ B[j][b_strip,:].T`
  - GEMM 2: `T_T[a,b] = B[j][a_strip,:] @ B[i][b_strip,:].T`
  Both land in SBUF via `tensor_copy`; the VE energy expression and free-dim
  reductions all run in SBUF/PSUM. Output is `(TILE, NOCC²)` — host calls
  `.sum()` for the scalar energy. No partition-axis (axis=0) reductions inside
  the kernel.

  **3D NKI indexing validated on trn1 (2026-04-17):** `nl.load_transpose2d(
  B[i, a_off:a_off+T, k_off:k_off+K])` with `i` as a `nl.affine_range` loop
  variable compiles correctly and produces accurate results (both `nl.load` and
  `nl.load_transpose2d` confirmed via Spike A / Spike B scripts).

  **Spike B warm time:** 2ms for NOCC=4 (16 pairs), vs ~1.6s for the per-pair
  loop — 800× reduction in dispatch overhead at the same NOCC.

  **Public API:** `trnblas.nki.nki_batched_pair_energy(B, eps_occ, eps_vir)` →
  0-d scalar. `B: (nocc, nvir, naux)`, `eps_occ: (nocc,)`, `eps_vir: (nvir,)`.

  **Example integration:** `examples/df_mp2.py --batched-pair-energy` routes
  the energy step through the new kernel.

  **Tests:** `TestBatchedPairEnergy` in `tests/test_nki_gemm.py`: aligned and
  unaligned correctness (atol=1e-2 over the full NOCC² sum),
  `test_matches_fused_gemm_energy` (against the per-pair sum),
  zero-B, and `test_dispatch_overhead` (cold/warm/per-pair-loop timing).

- **`[tool.uv]` dev-machine support.** Added `exclude-dependencies` for
  `neuronxcc`, `torch-neuronx`, and `nki` so `uv run` / `uv sync` resolves
  on machines without Trainium hardware.

## [0.5.1] — 2026-04-15

### Added

- **Fused GEMM+energy kernel (#38, `nki_fused_gemm_energy`).** A single
  `@nki.jit` kernel handles one DF-MP2 orbital pair — both GEMMs (T and T_T)
  and the VE energy expression — without writing the `(nvir, nvir)` T_flat
  intermediate to HBM.

  **Two-GEMM T_T strategy:** `T.T[a,b] = T[b,a] = (B_j @ B_i.T)[a,b]`.
  Rather than `nl.load_transpose2d` of T from HBM (which re-introduces the
  HBM round-trip), T_T is computed as a second GEMM tile in the same kernel
  body.  Both T and T_T land in SBUF via `tensor_copy` — no HBM write for
  either intermediate.

  **Kernel design:**
  - `TILE = 128` everywhere (`nl.load_transpose2d` constrains both dims to ≤ 128).
  - Outer a-loop, inner b-loop; two sequential PSUM allocations per (a, b) tile
    (one for T, one for T_T); VE energy expression fully SBUF-resident.
  - Cross-b batching: `acc_b (TILE, N_B_TILES)` in SBUF accumulates all b-strip
    partials before one `nl.store` per a-strip — same pattern as `_mp2_energy_kernel`.
  - NEFF cache amortises the two-GEMM compile across all `nocc²` pairs (same
    shape every invocation).
  - NKI 0.3.0 broadcast fix applied to `denom` construction (same as `_mp2_energy_kernel`).

  **Public API:** `trnblas.nki.nki_fused_gemm_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)` → scalar.

  **Example integration:** `examples/df_mp2.py --fused-gemm-energy` routes the
  energy step through the per-pair kernel.  Default path remains the chunk-GEMM
  path — see benchmark note below.

  **On-hardware benchmark (trn1, small shape: nbasis=128, nocc=16, nvir=112,
  naux=384; 256 pairs):**

  | Step | Baseline warm | Fused warm |
  |---|---|---|
  | energy | **0.13s** | **27.8s** |
  | total | 3.98s | 31.5s |

  The fused kernel is correct (energies agree to 6 significant figures) but
  the per-pair loop is **215× slower** on the energy step.  Root cause:
  Neuron XLA imposes ~100ms per-NEFF-dispatch overhead, independent of kernel
  compute time.  With 256 pairs × 100ms = 25.6s ≈ 27.8s observed.  The
  chunk-GEMM baseline amortises this with two dispatches total.

  Pre-transferring B to the XLA device and accumulating on-device (eliminating
  per-pair CPU syncs) produces the same warm timing because Neuron XLA's
  per-dispatch overhead is in the dispatch pipeline itself, not in the
  CPU→XLA transfer.

  **Follow-on:** production speedup requires a batched kernel that processes
  all nocc² pairs in one `@nki.jit` invocation — tracked in #43.

  **Tests:** `TestFusedGemmEnergy` in `tests/test_nki_gemm.py`:
  aligned/unaligned correctness (atol=1e-2), symmetry (`E(i,j) == E(j,i)`),
  zero-B_i, NEFF cache reuse (cold vs warm timing).

### Fixed

- **NKI closure variable limitation in autotuner (#26 regression).** The
  v0.5.0 `_make_gemm_kernel` factory returned a `@nki.jit` closure that
  referenced tile sizes (`tm`, `tk`, `tn`) as Python free variables.
  NKI's AST-based compiler reads source from the on-disk file and resolves
  names from the local namespace only — it cannot traverse closure cells,
  producing `error: unbound variable 'tm'` for every tile config.

  **Fix:** replaced the factory with six static `@nki.jit` kernel definitions
  at module level (`_gemm_kernel_64_128_128` … `_gemm_kernel_128_128_512`),
  each with literal integer tile constants.  All six are registered in
  `_gemm_kernel_registry` at import time; `_get_gemm_kernel()` is now a
  dict lookup.  `_make_gemm_kernel` is removed.  Autotuner behaviour
  (sweep, cache, escape-hatch) is unchanged.

  **Root cause note:** NKI `@nki.jit` functions must have tile constants
  visible as literal integers or module-level globals at AST trace time.
  Closure variables from an enclosing factory scope are not reachable.

## [0.5.0] — 2026-04-15

### Added

- **GEMM tile-shape autotuner (#26).** `_nki_gemm_impl` now sweeps six tile
  candidates `{64,128} × {128} × {128,256,512}` on the first call per shape
  bucket and caches the winner to disk
  (`/var/tmp/trnblas-autotune/cache.json`, overrideable via
  `TRNBLAS_AUTOTUNE_CACHE`). Subsequent calls for the same bucket hit the
  in-process dict first, then the NEFF cache — no re-sweep. Padding is now
  computed *after* tile selection so alignment always matches the chosen tile
  sizes. Opt out with `TRNBLAS_AUTOTUNE=0` to restore v0.4.x fixed
  `(128,128,512)` behaviour. Backward compatible: `_gemm_kernel` alias
  preserved; `_TILE_M/K/N` fallback constants kept for SYRK/TRSM.
  - `_make_gemm_kernel(tile_m, tile_k, tile_n)` — factory returning a new
    `@nki.jit` closure with tile values baked in at trace time; each config
    produces a separately-cached NEFF.
  - `_get_gemm_kernel(...)` — registry wrapper (call once per config, reuse).
  - `_autotune_bucket(M, K, N)` — `ceil_pow2` coarse bucket; all shapes in
    a DF-MP2 run land in the same bucket.
  - `_sweep_tile_configs` / `_sweep_on_default_pad` — hardware timing (3
    warm runs each), safe fallback to default on per-config errors.
  - Persistent JSON cache with atomic directory creation.
  - `TestAutotuner` class in `tests/test_nki_gemm.py`: escape-hatch,
    cache-hit (no re-sweep), persistent-cache round-trip, per-config
    correctness, hardware sweep.

### Changed

- **#33 resolved — `_mp2_energy_kernel` profile via Neuron Profiler 2.0.**
  The April-14 profiling attempt was blocked by the deprecated Neuron 2.29
  API (`inspect`/`show-session` → NTFF v130 format mismatch). Retried
  April-15 using the new `neuron-profile capture` + `view --output-format
  summary-text` API (no InfluxDB, no browser). Key findings at medium bench
  shape (trn1.2xlarge, neuronxcc 2.24.5133):
  - **Vector Engine: 96.45% active** — the entire reduction (`T*(2T-Tᵀ)/denom`)
    is element-wise arithmetic; Tensor Engine runs 21 instructions in 0.48 µs.
  - **HBM reads: 6.58 GB** — matches the analytical 2-pass prediction exactly
    (previous napkin of 33 GB was for the unfused torch path).
  - **Kernel wall time: ~0.21 s** vs ~2.83 s for the torch reduction — **~13×
    speedup on the reduction step alone.**
  - **Amdahl ceiling: 1.48×.** The GEMM (T_flat = B_chunk @ B_flat.T) takes
    ~5.2 s in both paths and is 96% of fused energy step time. The 1.48× overall
    result is an exact Amdahl prediction (f=0.35, s=13.5 → 1.48×), not a tuning
    failure. Path to 3× requires fusing the GEMM into the energy pass (Phase 3
    RFC) or a different algorithm.
  See `docs/design/mp2_energy_profile_findings.md` for full data and next steps.
  `scripts/run_neuron_profile.sh` updated to Neuron Profiler 2.0 API with
  base64-encoded SSM commands (bypasses all shell-quoting issues; supports
  `--probe` mode for API discovery).

- **#35 — cross-pair batching in `_mp2_energy_kernel` (store-fence
  hypothesis falsified).** Restructured the kernel to accumulate all
  NOCC pair partials for each `i`-row in SBUF before a single batched
  HBM store, replacing IC×NOCC per-pair stores with IC stores per chunk
  (4,096 → 64 stores at medium; 9,216 → 96 at large). Measured result
  on trn1 (warm NEFF cache):
  - medium: energy 5.43 s → **5.38 s** (within noise; 1.49× vs torch)
  - large: energy 30.27 s → **30.13 s** (within noise; 1.47× vs torch)
  The per-pair HBM-store fence hypothesis is now falsified — the
  compiler appears to tolerate the store traffic without serializing
  across pairs. The kernel is kept with batched stores (no functional
  regression, 64× less store traffic); the 1.47× ceiling stands.
  Ceiling now explained by Amdahl (#33 profile); no remaining unknown
  hypotheses. Tracked on [#31](https://github.com/trnsci/trnblas/issues/31).

- **Migrated to NKI 0.3.0 / Neuron SDK 2.29.** Canonical `nki.*`
  namespace; the legacy `neuronxcc.nki.*` shim is no longer used.
  `[neuron]` extra now requires `nki>=0.3.0`. Kernels updated for
  the NKI 0.3.0 breaking-change surface:
  - `nc_matmul` → `nisa.nc_matmul(dst=, stationary=, moving=, accumulate=)`
    (all kwargs; internal accumulate replaces external `psum[...] += ...`).
  - `nl.copy(psum, ...)` returns a view; use `nl.ndarray + nisa.tensor_copy`
    to move PSUM → SBUF before `nl.store`.
  - Tensor-tensor `nl.divide` dropped; use `multiply × reciprocal`.
  Kernels migrated: `_gemm_kernel`, `_syrk_kernel`, and the kernel
  factory in `scripts/autotune_gemm.py`. 32 neuron tests pass.
- `_mp2_energy_kernel` re-skipped pending #15 M2 redesign — NKI 0.3.0's
  stricter tensor-tensor broadcast rules reject the current kernel's
  `(1,1) - (P_TILE,1)` partition-dim pattern. M1 work (namespace +
  free-dim reduction + partition-major HBM output) is preserved in
  the kernel source; only the scalar-vs-partition subtract needs
  rewriting.
- **#15 M2.1a — `_mp2_energy_kernel` broadcast correctness fix.**
  `denom` construction now lifts all three eps operands to
  `(P_TILE, NVIR)` via `nl.broadcast_to` before subtracting, so every
  op sees matching partition dims. 5 `TestNkiKernel` MP2 cases
  re-enabled (37/37 neuron tests pass on trn1). Same kernel
  structure as M1 — same tile geometry, per-strip accumulator,
  `(P_TILE, IC, NOCC)` output layout.
- **#15 M2.2 — measured DF-MP2 perf on trn1 (warm NEFF cache).**
  `trnblas.nki.nki_mp2_energy` now beats the torch reduction but
  misses the RFC's 3–5× target:
  - medium (nbasis=512, nocc=64, nvir=448): energy 8.03 s → 5.43 s
    (**1.48×**); total 9.79 s → 7.29 s.
  - large (nbasis=768, nocc=96, nvir=672): energy 44.57 s → 30.27 s
    (**1.47×**); total 50.00 s → 35.76 s.
  - Bit-parity with torch reference at `atol=1e-4, rtol=1e-4` at
    both shapes.
  Speedup ratio is roughly shape-invariant — per-(i,j) launch cost
  scales with `nocc²` the same way torch does. RFC is updated to
  "Shipped (M2.1)" status; perf follow-ups (larger free-dim tiles,
  atomic-add variant, multi-engine pipeline evidence) tracked on
  the same [#15](https://github.com/trnsci/trnblas/issues/15)
  issue. `examples/df_mp2.py --fused-energy` exposes the kernel;
  default path remains torch until a future milestone hits 3×.
- `examples/df_mp2.py` — `--fused-energy` opt-in flag; threads
  `use_fused` through `df_mp2_energy` / `_energy_reduction`.
- `scripts/run_df_mp2_bench.sh` — `--compare` runs torch + fused
  back-to-back in one SSM session for A/B perf measurement;
  'stopping' instance state no longer blocks back-to-back runs.

### Added

- **NKI CPU simulator dispatch** via `TRNBLAS_USE_SIMULATOR=1`.
  Routes kernels through `nki.simulate(kernel)(numpy_args)` on CPU,
  bypassing `torch_xla` + NEFF compile. Iteration loop drops from
  ~8–12 min per cycle to seconds. `_nki_{gemm,syrk,mp2_energy}_impl`
  all carry the simulator branch; `_nki_trsm_left` plumbs through
  transitively via `nki_gemm`. Correctness-only — no perf modelling,
  no SBUF capacity checks. See `docs/developing_kernels.md`.
- `tests/test_nki_sim.py` — curated simulator-backed correctness
  suite, marker `nki_simulator`. Skips unless
  `TRNBLAS_USE_SIMULATOR=1` + `nki` is importable.
- `scripts/run_simulator_tests.sh` — SSM runner that runs the
  simulator suite on the trn1 DLAMI.
- **`nki-simulator` CI job on `ubuntu-latest`.** Runs the
  `nki_simulator`-marked suite against `nki>=0.3.0` from the AWS
  pip index (`--extra-index-url https://pip.repos.neuron.amazonaws.com`)
  on every push + PR. Zero AWS cost for the correctness gate;
  hardware SSM now reserved for perf + MLIR verification. Of the
  five NKI 0.3.0 breaking-changes trnblas navigated, four would
  have surfaced on this gate pre-merge. The fifth
  (partition-broadcast strictness, MLIR-level) still requires
  hardware — NKI 0.3.0 has no documented device-free NEFF compile
  API. Main `test` matrix now excludes `-m nki_simulator` to avoid
  running the suite twice.
- `docs/developing_kernels.md` — kernel authoring guide: three
  dispatch modes (pytorch / hardware / simulator), simulator
  limitations, NKI 0.3.0 migration reference,
  architecture-exploitation design discipline.

### Superseded by NKI 0.3.0 migration (history for completeness)

- `nki_mp2_energy` M1 landed: kernel now correctly produces
  `(P_TILE, IC, NOCC)` per-partition partials on real NKI under
  `TRNBLAS_REQUIRE_NKI=1`. All 5 previously-skipped
  `TestNkiKernel` tests pass on trn1 across
  `nvir ∈ {8, 16, 64, 256, 448}`. Host `.sum()` reduces to the
  final scalar energy (partial is ≤ 258 KB, noise cost).
- `docs/design/fused_df_mp2_energy_kernel.md` — architectural RFC
  for the M2 fused pair-energy kernel (Phase 3 follow-up that uses
  M1's reduction pattern as a building block).

### Architectural features exploited in M1 (per the design discipline)

- **SBUF persistence** across the strip loop (per-partition buffer
  lives on-chip between all NSTRIP iterations).
- **Scalar Engine free-dim reduction** via `nl.sum(axis=1)` — the
  only reduction axis NKI permits.
- **Partition-major HBM output** so the `(P_TILE, 1)` SBUF tile
  stores with axis-to-axis alignment (no partition-dim reshape,
  which the BIR verifier rejects).
- **Amortised dispatch**: IC × NOCC (i, j) pairs per kernel launch.

### NKI constraints navigated (documented for future kernels)

| Error | Pattern rejected | Pattern that works |
|---|---|---|
| `partitions … exceed 128` | `nl.load(1D_tensor)` of length >128 | Reshape to `(1, N)` at caller |
| `Reduction on partition axes is not supported` | `nl.sum(tile, axis=(0,1))` | Free-dim reduce only; accumulate per-partition |
| `illegal partition step` (BIR) | Reshape SBUF partition↔free | Plan tensor layout so partition aligns throughout; never reshape |
| `Unexpected output dependencies, missing indices in dst` | `acc[...] = nl.add(acc, x)` inside `affine_range` | Per-iteration SBUF slots, reduce after the loop |

Each constraint was surfaced by real hardware under
`TRNBLAS_REQUIRE_NKI=1` — the silent-fallback era would have masked
all of them.

## [0.4.3] — 2026-04-13

### Correction: v0.4.x "trn1 NKI" numbers were silent torch.matmul fallback

The SSM runners in v0.4.0–v0.4.2 invoked the Neuron venv's python
directly without prepending its `bin/` to `$PATH`. `torch_neuronx`'s
initializer calls `subprocess.run(["libneuronpjrt-path"])` to locate
the PJRT plugin library; that binary lives in the venv's `bin/` and
couldn't be resolved. Every NKI dispatch raised `FileNotFoundError`,
which our `_nki_*_impl` `try/except` wrappers swallowed and fell back
to `torch.matmul`. As a result, **every "trn1 NKI" perf number
published in v0.4.0 / v0.4.1 / v0.4.2 was trn1's 8-vCPU Xeon, not the
Tensor Engine.**

Correctness tests still passed because `torch.matmul` gives the same
answer as `nki_gemm`; only perf attribution was wrong. The v0.4.2
cross-vendor comparison vs A10G was also mislabeled — we were
comparing A10G's GPU to trn1's Xeon.

Real NKI dispatch is now verified (commit `d1b481f`): cold call
includes NEFF compile (seconds), warm dispatches show real Tensor
Engine execution. `docs/benchmarks.md` tables are re-measured and
prefaced with a retraction banner.

### Fixed

- `scripts/run_neuron_tests.sh`, `scripts/run_df_mp2_bench.sh` —
  prepend `$NEURON_VENV/bin` to `$PATH` in the SSM `env` line so
  `torch_neuronx`'s PJRT plugin lookup can resolve. Tests now also
  run with `TRNBLAS_REQUIRE_NKI=1` so future silent-fallback
  regressions fail loudly.
- `trnblas.nki.nki_mp2_energy` kernel tests skipped (#15) — the
  kernel has a partition-limit bug (`nl.load(eps_vir[0:NVIR])`
  exceeds 128 partitions for `nvir > 128`) that was masked by the
  silent fallback. Not in the production DF-MP2 path; kernel
  rewrite tracked under #15.

### Added

- `trnblas.nki.NkiFallbackWarning` — emitted once per distinct error
  when the NKI path silently falls back to torch. Makes misconfigured
  environments visible without requiring `TRNBLAS_REQUIRE_NKI=1`.
  Emitted via `warnings.warn` with a custom category.
- `tests/test_nki_really_runs.py` — anti-regression test that forces
  `TRNBLAS_REQUIRE_NKI=1` and asserts a GEMM dispatch completes.
  Would have caught the v0.4.0 regression on day one.

### Changed — re-measured benchmark numbers

Under real NKI dispatch (commit `fd56274`, trn1.2xlarge, neuronxcc
2.24.5133):

| Op | Shape | v0.4.x "trn1 NKI" (was CPU fallback) | v0.4.3 trn1 NKI (real) |
|----|-------|-------------------------------------:|-----------------------:|
| GEMM warm | 1024³ | 4.5 ms | 2.3 ms |
| SYRK warm | 512²   | 2.45 ms | 2.14 ms |
| SYRK warm | 1024² | 7.91 ms | 5.71 ms |
| TRSM warm | 512²   | 6.05 ms | 5.59 ms |
| TRSM warm | 2048×512 | 27.75 ms | 35.82 ms |
| DF-MP2 medium warm | — | 9.77 s | 9.91 s |

The relative A10G vs trn1 ratios are in a similar 19–45× range; the
cross-vendor story's shape is unchanged, only the attribution is
fixed.

### Added (carried from [Unreleased] into this release)

- `trnblas.nki.nki_trsm` — blocked panel TRSM (#19). Diagonal panels
  solve via `torch.linalg.solve_triangular` (small, sequential);
  trailing off-diagonal updates run through `nki_gemm` (dominant work
  for large M). Covers all `{lower, upper} × {trans, not} ×
  {unit, nonunit}` combinations for `side="left"`; `side="right"` falls
  back to torch. 7/7 `@pytest.mark.neuron` tests pass on trn1 under
  real NKI dispatch.
- `trnblas.nki.nki_syrk` — NKI SYRK kernel (#18). Loads `A` once
  from HBM and reuses it for both operand roles via two
  `load_transpose2d` calls, avoiding the materialised
  `A.T.contiguous()` that `nki_gemm(A, A.T)` would otherwise write.
  7/7 `@pytest.mark.neuron` tests pass on trn1 under real NKI.
- `examples/bench_syrk.py`, `examples/bench_trsm.py` — per-op
  timing scripts (cpu / cuda / trn1) feeding the cross-vendor table.
- `scripts/autotune_gemm.py` — GEMM tile-config study harness (#26).
  Paused during this correction release; resume in v0.5.0.
- `scripts/probe_nki.py` — one-shot NKI health probe (diagnostic
  for the silent-fallback class of bug).

## [0.4.2] — 2026-04-13

### Added

- cuBLAS head-to-head infrastructure (#4). New
  `infra/terraform-cuda/` module provisions a single-A10G
  `g5.xlarge` CI instance (GA102 Ampere, Apr 2021 — vintage-matched
  to Trainium1 Oct 2022). New `scripts/run_cuda_bench.sh` SSM runner
  mirrors `run_df_mp2_bench.sh` with trap-stop cleanup.
- `examples/df_mp2.py --device {cpu,cuda}` flag. Inputs are built on
  CPU with a fixed seed and then moved to the requested device, so
  GPU energies match CPU bit-for-bit (within fp32 reduction-order
  noise). Added `torch.cuda.synchronize()` before stopping the
  wall-clock so async kernels complete.

### Changed

- `df_mp2_energy` now respects the input tensor device. The
  `torch.eye` in the metric inversion step and the scalar energy
  accumulator previously hardcoded CPU, which broke `--device cuda`.
- `docs/benchmarks.md` — DF-MP2 table replaced with a side-by-side
  trn1 vs A10G comparison (new headline: A10G is **30–37× faster**
  than the current trn1 torch-matmul path at medium/large, with
  bit-exact energies — the gap to close via NKI kernels in v0.5.0+).
- `docs/aws_setup.md` — new "GPU companion instance" subsection +
  g5.xlarge cost row.

### Fixed

- `.gitignore` terraform-state rule extended to cover all
  `infra/terraform*/` dirs (was scoped to the Trainium module only).

## [0.4.1] — 2026-04-13

### Fixed

- `trnblas.__version__` was stuck at `"0.3.0"` while `pyproject.toml`
  advanced through `0.3.1` / `0.4.0`. Now tracks the current release
  (`"0.4.1"`).

### Changed

- Documentation site stabilised to match v0.4.0 state:
  - `docs/installation.md` — new `[pyscf]` extra section,
    `TRNBLAS_REQUIRE_NKI` env var table, updated `neuronxcc >= 2.24`
    and `torch-neuronx >= 2.9` pins.
  - `docs/api/nki.md` — expanded GEMM section with HBM padding
    behaviour + measured per-call timings; new sections for
    `nki_batched_gemm` and `nki_mp2_energy` with perf caveats.
  - `docs/architecture.md` — "Known gaps" refreshed with current
    Level 3 coverage status and issue cross-references.
  - `docs/benchmarks.md` — placeholder replaced with measured
    trn1.2xlarge numbers (GEMM per-call, batched GEMM per-slice,
    DF-MP2 small/medium/large, NEFF cache warmup).
  - `docs/index.md` — pointer to the PySCF real-molecule demo.

## [0.4.0] — 2026-04-12

### Added

- Real-molecule DF-MP2 validation against PySCF (#11). New
  `examples/_pyscf_bridge.py` runs RHF + builds DF integrals;
  `examples/df_mp2_pyscf.py` is a runnable demo comparing trnblas
  vs PySCF's own `mp.dfmp2.DFMP2` reference. New
  `tests/test_df_mp2_pyscf.py` (marker: `pyscf`, skipped if PySCF
  isn't installed) parameterises H2O/STO-3G, H2O/cc-pvdz,
  CH4/cc-pvdz, NH3/cc-pvdz. Measured agreement on all four:
  |E_trnblas - E_pyscf| < 10⁻⁷ Hartree (nanohartree precision).
  New optional extra `pip install trnblas[pyscf]`.

- `trnblas.nki.nki_mp2_energy` — fused MP2 energy-reduction NKI
  kernel (#15). Streams T_flat tiles on-chip via partition-dim
  sub-tiling (P_TILE picked as the largest divisor of nvir ≤ 128;
  covers all bench shapes). Loads a (P_TILE, nvir) strip + its
  within-block transpose (via `nl.load_transpose2d`), builds
  `denom` on-chip, reduces into a per-(i,j) SBUF accumulator,
  single HBM store per (i,j). Five on-hardware correctness tests
  (`tests/test_nki_mp2_energy.py`, `@pytest.mark.neuron`) cover
  `nvir ∈ {8, 16, 64, 256, 448}` — all pass on trn1.
  **Perf status:** bit-exact with the torch reference but matches
  (not beats) it at medium on trn1 — the per-(i,j) dispatch/load
  chain swamps compute savings. `examples/df_mp2.py` keeps the
  torch path for now; NKI dispatch re-wire deferred to a kernel
  restructuring pass (batch multiple (i,j) per dispatch).

- `examples/df_mp2.py` refactored to use `trnblas.batched_gemm` for all
  per-occupied-orbital loops (steps 2b, 3, and 4-per-i). Energy
  reduction in step 4 fully vectorised over (j, a, b), eliminating the
  per-pair `.item()` round-trip.
- `--bench` mode in the example: runs cold + warm passes for three
  synthetic shapes (small/medium/large), reports per-step timings and
  effective TFLOPS.
- `scripts/run_df_mp2_bench.sh` — SSM-driven runner for the bench,
  parallel to `run_neuron_tests.sh`.

### Performance — DF-MP2 on trn1.2xlarge

After collapsing the energy step (was 4096 sequential batched dispatches
via a Python loop; now one chunked GEMM via the algebraic identity
`T_full = X @ X.T` where `X = B.reshape(nocc·nvir, naux)`), see #14:

| Shape | Flops | Cold | Warm | TFLOPS | Speedup vs prior |
|-------|------:|-----:|-----:|-------:|-----------------:|
| small (128/16/384)    | 3.4 G    | 0.025s | 0.008s  | 0.43 | — |
| medium (512/64/1536)  | 2757 G   | 12.92s | 9.77s   | 0.28 | 2.2× |
| large (768/96/2304)   | 20352 G  | 65.88s | 62.84s  | 0.32 | (newly feasible) |

Energy reproducible bit-for-bit across runs:
- small: -1.619250e-04
- medium: -2.487221
- large: -4.351183e+01

The energy step still dominates large's wall (57s of 63s = 92%) — it's
memory-bandwidth bound on the huge T tensor + intermediates. Fusing it
into a custom NKI kernel would be the next optimisation; tracked as a
future v0.4 follow-up to #14.

### Added

- Terraform module (`infra/terraform/`) provisioning a Trainium CI instance
  with SSM access; instance kept stopped between runs (~$10/mo EBS-only).
- `scripts/run_neuron_tests.sh` — local SSM-driven runner for
  `pytest -m neuron`; starts the instance, runs tests, **always stops it
  via trap**.
- AWS setup docs (`docs/aws_setup.md`) covering provisioning, running
  tests, cost, and troubleshooting.

### Removed

- `.github/workflows/neuron.yml` workflow_dispatch scaffold. Per the
  trnfft pattern, GitHub Actions does not touch AWS — all Neuron testing
  is human-initiated locally with `AWS_PROFILE=aws`.

### Changed

- NKI GEMM kernel (`trnblas/nki/dispatch.py:_gemm_kernel`) wired to actual
  `nisa.nc_matmul` calls with PSUM accumulation across K-tiles and
  stationary A-tile reuse — supersedes the previous stub that overwrote
  per K-tile.
- Dispatch wrapper now handles arbitrary shapes via HBM padding: M/K
  rounded to 128, N rounded to 512 (when N > 512); kernel uses
  `TILE_N = min(N, 512)` for single-tile small-N. Result is sliced back
  to the original (M, N). Removes the alignment-rejection fallback path.
- `TRNBLAS_REQUIRE_NKI=1` env-var added — re-raises on kernel exceptions
  instead of silently falling back to `torch.matmul`. Lets the
  validation suite surface kernel breakage.

- `trnblas.batched_gemm` dispatches per-slice through the cached 2D
  `_gemm_kernel` via new `nki_batched_gemm` wrapper. Every slice after
  the first hits the NEFF cache (identical signature), so per-slice cost
  is HBM transfer + Tensor Engine dispatch only. The natural batched
  dispatch shape for DF-MP2 contractions over auxiliary basis indices.

- Bumped `neuronxcc` floor from `>=2.15` to `>=2.24` to unify with the
  rest of the trnsci suite (matches trnfft / trnrand). `torch-neuronx`
  floor bumped to `>=2.9` to match.

- Repository transferred from `scttfrdmn/trnblas` to the `trnsci`
  GitHub organisation (`trnsci/trnblas`). Documentation now served at
  <https://trnsci.dev/trnblas/>. Canonical `CONTRIBUTING.md` and
  `CODE_OF_CONDUCT.md` adopted to match the trnsci suite.

### Performance (validated on trn1.2xlarge, neuronxcc 2.24.5133)

17/17 `pytest -m neuron` tests pass. Cached-NEFF speedup measured by
running the suite twice on the same instance:

| Pass | Wall time |
|------|----------:|
| Cold (first run after instance start) | 7.01s |
| Warm (NEFF cache hit + warm XLA graph) | 2.52s (2.8× faster) |

Per-call kernel timing (warm cache, mean of 5):

| Shape (M×K×N) | Per-call |
|---------------|---------:|
| 512×512×512    | 1.6 ms |
| 1024×1024×1024 | 4.5 ms |

Batched dispatch (warm, batch=32 of 256×128×256):

| Metric | Value |
|--------|------:|
| Total | 39.3 ms |
| Per-slice | 1.23 ms |

NEFF cache at `/var/tmp/neuron-compile-cache/` persists across instance
stop/start (EBS-backed), so kernel compile cost is paid exactly once per
shape per cache lifetime.

## [0.2.0] - 2026-04-11

### Added

- MkDocs Material documentation site at
  [scttfrdmn.github.io/trnblas](https://scttfrdmn.github.io/trnblas/) with
  Installation, Quickstart, API reference (Level 1/2/3, NKI backend), and
  Architecture pages.
- GitHub Actions CI matrix (Python 3.10, 3.11, 3.12).
- Neuron hardware CI workflow scaffold (`workflow_dispatch`) — SSM wiring
  deferred until a persistent CI Trainium instance is available.
- PyPI publishing workflow (OIDC trusted publishers, sdist + wheel on
  release) — matches trnfft pattern.
- Benchmark suite scaffold (`benchmarks/bench_blas.py`, pytest-benchmark).
- Issue and PR templates under `.github/`.
- README badges — CI status, PyPI version, Python versions, License, Docs.
- Cross-link to trnblas in trnfft's Related Projects table
  (`scttfrdmn/trnfft@7330b3f`).

## [0.1.0] - 2026-04-11

### Added

- Level 1 BLAS: `axpy`, `dot`, `nrm2`, `scal`, `asum`, `iamax`.
- Level 2 BLAS: `gemv`, `symv`, `trmv`, `ger`.
- Level 3 BLAS: `gemm`, `batched_gemm`, `symm`, `syrk`, `trsm`, `trmm`.
- NKI dispatch layer with `auto`, `pytorch`, and `nki` backend selection.
- NKI GEMM kernel stub with stationary tile reuse strategy (scaffolded for
  on-hardware validation on trn1/trn2).
- DF-MP2 example (`examples/df_mp2.py`) demonstrating the quantum-chemistry
  use case with half-transform GEMMs, Cholesky, triangular solve, and energy
  evaluation.
- Test suite covering Level 1/2/3 BLAS correctness against PyTorch/NumPy
  references, with SPD matrix fixtures for symmetric/triangular routines.

[Unreleased]: https://github.com/trnsci/trnblas/compare/v0.4.3...HEAD
[0.4.3]: https://github.com/trnsci/trnblas/releases/tag/v0.4.3
[0.4.2]: https://github.com/trnsci/trnblas/releases/tag/v0.4.2
[0.4.1]: https://github.com/trnsci/trnblas/releases/tag/v0.4.1
[0.4.0]: https://github.com/trnsci/trnblas/releases/tag/v0.4.0
[0.3.0]: https://github.com/trnsci/trnblas/releases/tag/v0.3.0
[0.2.0]: https://github.com/trnsci/trnblas/releases/tag/v0.2.0
[0.1.0]: https://github.com/trnsci/trnblas/commits/main
