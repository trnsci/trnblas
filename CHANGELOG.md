# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/scttfrdmn/trnblas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scttfrdmn/trnblas/releases/tag/v0.1.0
