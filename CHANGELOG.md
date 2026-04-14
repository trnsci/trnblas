# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
