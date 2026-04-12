# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
- DF-MP2 example (`examples/df_mp2.py`) demonstrating the Janesko/TCU use case
  with half-transform GEMMs, Cholesky, triangular solve, and energy evaluation.
- Test suite covering Level 1/2/3 BLAS correctness against PyTorch/NumPy
  references, with SPD matrix fixtures for symmetric/triangular routines.

[Unreleased]: https://github.com/scttfrdmn/trnblas/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/scttfrdmn/trnblas/releases/tag/v0.3.0
[0.2.0]: https://github.com/scttfrdmn/trnblas/releases/tag/v0.2.0
[0.1.0]: https://github.com/scttfrdmn/trnblas/commits/main
