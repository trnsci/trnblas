# trnblas

BLAS operations for AWS Trainium via NKI.
Part of the trnsci scientific computing suite.

## What This Is

A cuBLAS-equivalent for Trainium. Provides Level 1 (vector), Level 2
(matrix-vector), and Level 3 (matrix-matrix) BLAS operations with NKI
kernel acceleration on the Tensor Engine.

**Primary use case:** DF-MP2 quantum chemistry on large molecules
(>3000 basis functions), where sustained GEMM throughput for tensor
contractions dominates wall-time. On 192 Zen4 cores (c8a.48xlarge), a
single calculation takes ~24 hours at ~$200. trnblas on Trainium
targets 3-4x cost reduction by exploiting the systolic array for the
GEMM-dominated hot path.

## Architecture

```
trnblas/
├── trnblas/
│   ├── __init__.py          # Re-exports all BLAS operations
│   ├── level1.py            # axpy, dot, nrm2, scal, asum, iamax
│   ├── level2.py            # gemv, symv, trmv, ger
│   ├── level3.py            # gemm, batched_gemm, symm, syrk, trsm, trmm, metric_inv_half
│   ├── parallel.py          # SPMD multi-core GEMM + DF-MP2 (torchrun)
│   └── nki/
│       ├── __init__.py
│       ├── dispatch.py      # auto/pytorch/nki dispatch, NKI kernels, native/XLA path selection
│       ├── _native.py       # TorchNeuron custom ops (nki_op), loaded lazily
│       └── multicore.py     # opt-in single-process pair-energy split (set_cores)
├── tests/
│   ├── conftest.py          # Fixtures: random matrices, SPD matrices
│   ├── test_level1.py       # Vector operation tests
│   ├── test_level2.py       # Matrix-vector tests
│   └── test_level3.py       # Matrix-matrix tests (GEMM, TRSM, SYRK, etc.)
├── examples/
│   └── df_mp2.py            # DF-MP2 energy using trnblas
├── pyproject.toml
├── README.md
├── LICENSE                  # Apache 2.0
└── CLAUDE.md                # This file
```

## DF-MP2 → BLAS Mapping

The DF-MP2 algorithm maps directly to trnblas Level 3 operations:

| DF-MP2 Step | Math | trnblas Call |
|-------------|------|-------------|
| Half-transform | (iν\|P) = C_occ^T @ (μν\|P) | `gemm(1.0, C_occ, eri, transA=True)` |
| MO transform | (ia\|P) = C_vir^T @ (iν\|P) | `gemm(1.0, C_vir, iv_P, transA=True)` |
| Cholesky + metric solve | J = L @ L^T, L^T @ X = I | `metric_inv_half(J)` (host fp64 LAPACK for naux ≤ 1024 on the native path; else `torch.linalg.cholesky` + `trsm`) |
| Metric contract | B = (ia\|Q) @ J^{-1/2} | `gemm(1.0, ia_P, J_inv_half)` |
| Energy | T_ab = B_i @ B_j^T | `gemm(1.0, B[i], B[j], transB=True)` |

## NKI GEMM Strategy

The NKI GEMM kernel uses stationary tile reuse on the Tensor Engine:

1. Load A tile (128×128) to SBUF as stationary operand
2. Stream B tiles through the systolic array as moving operand
3. Accumulate partial products in PSUM
4. One A load serves all B tiles → 2x fewer SBUF loads vs naive

For DF-MP2, the MO coefficient matrix C is the natural stationary operand
since it's reused across all auxiliary basis indices P.

## Known Gaps & Design Notes

- **NKI GEMM is validated but loses to aten on Trn2.** On the native
  (TorchNeuron) path, `auto` routes GEMM-family calls by measured time:
  device-resident operands go to aten (2.4–3.2× faster than the NKI GEMM),
  host operands to the host CPU or upload + aten (v0.9.0). The NKI kernels
  run on the XLA path (Trn1) and under `set_backend("nki")`. The fused
  `pair_energy_sym` kernel is 1.8× faster than aten on the DF-MP2 energy
  step (an earlier 92× figure was wrong).

- **No FP64.** Trainium's Tensor Engine maxes out at FP32. For chemistry
  workloads needing higher precision, double-double arithmetic (emulated FP64
  from two FP32 values) is the path — not implemented yet.

- **Level 3 dominates.** Level 1/2 are included for API completeness but
  won't get NKI kernels initially. The Tensor Engine is wasted on vector ops.

- **DF-MP2 example energy step** defaults to a chunked GEMM + torch
  reduction; `--batched-pair-energy` runs `nki_batched_pair_energy` (one
  dispatch, `pair_energy_sym` on the native path at large sizes).
  `trnblas.parallel.df_mp2_energy` is the multi-core version.

## Dependencies

- `torch>=2.1` — tensor operations and CPU fallback
- `numpy>=1.24` — test reference
- `neuronxcc` — NKI kernels (optional, only on Neuron hardware)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v               # CPU fallback mode
python examples/df_mp2.py --demo   # Quick DF-MP2 demo
```

## Neuron testing

`pytest -m neuron` runs against real Trainium hardware. The flow is
human-initiated and local — **GitHub Actions does not touch AWS**.

```bash
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh        # trn1 (default)
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn2   # or inf2
```

The script starts a tagged Trainium instance (`trnblas-ci-trn1` by
default), runs pytest via SSM, and **always stops the instance** via a
trap. Provisioning is one-time via `infra/terraform/`. See
[`docs/aws_setup.md`](docs/aws_setup.md) for setup, cost, and
troubleshooting.

## Naming Convention

Sibling repos in the trn-* suite:
- `trnfft` — FFT + complex ops (https://github.com/trnsci/trnfft)
- `trnblas` — BLAS operations (this repo)
- `trnsolver` — Linear solvers, eigendecomposition (https://github.com/trnsci/trnsolver)
- `trnsparse` — Sparse formats + SpMM (https://github.com/trnsci/trnsparse)
- `trntensor` — einsum + tensor decompositions (https://github.com/trnsci/trntensor)
- `trnrand` — Random number generation (https://github.com/trnsci/trnrand)

All repos: Python/NKI, Apache 2.0.

## Blog posts

When asked to draft a blog post for this library for the [trnsci blog](https://trnsci.dev/blog/):

1. Read the editorial brief at [`docs/blog/AUTHOR_BRIEF.md`](https://github.com/trnsci/trnsci/blob/main/docs/blog/AUTHOR_BRIEF.md) in the umbrella repo (trnsci/trnsci). It defines voice (authorless, library-as-subject), stance (architecture-first, transparency-always), and the nine required section headings.

2. Find the prompt block for this library in [`BLOG_PROMPTS.md`](https://github.com/trnsci/trnsci/blob/main/BLOG_PROMPTS.md) at the umbrella repo root. It carries library-specific context and suggested architectural angles.

3. Draft the post following the brief. Open a PR against `trnsci/trnsci` at `docs/blog/posts/<YYYY-MM-DD>-<slug>.md`. Scott (suite director) reviews before merge.

The umbrella repo — not this one — owns the blog. Per-library retrospective posts are unsigned; library is the subject, no byline. See the brief for the full set of rules.
