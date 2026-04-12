# trnblas

BLAS operations for AWS Trainium via NKI.
Part of the trn-* scientific computing suite by Playground Logic.

## What This Is

A cuBLAS-equivalent for Trainium. Provides Level 1 (vector), Level 2
(matrix-vector), and Level 3 (matrix-matrix) BLAS operations with NKI
kernel acceleration on the Tensor Engine.

**Primary use case:** DF-MP2 quantum chemistry with Prof. Ben Janesko at TCU.
Molecules with >3000 basis functions require sustained GEMM throughput for
tensor contractions. On 192 Zen4 cores (c8a.48xlarge), a single calculation
takes ~24 hours at ~$200. trnblas on Trainium targets 3-4x cost reduction
by exploiting the systolic array for the GEMM-dominated hot path.

## Architecture

```
trnblas/
├── trnblas/
│   ├── __init__.py          # Re-exports all BLAS operations
│   ├── level1.py            # axpy, dot, nrm2, scal, asum, iamax
│   ├── level2.py            # gemv, symv, trmv, ger
│   ├── level3.py            # gemm, batched_gemm, symm, syrk, trsm, trmm
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py      # auto/pytorch/nki dispatch + NKI GEMM kernel
├── tests/
│   ├── conftest.py          # Fixtures: random matrices, SPD matrices
│   ├── test_level1.py       # Vector operation tests
│   ├── test_level2.py       # Matrix-vector tests
│   └── test_level3.py       # Matrix-matrix tests (GEMM, TRSM, SYRK, etc.)
├── examples/
│   └── df_mp2.py            # DF-MP2 energy using trnblas (Janesko use case)
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
| Cholesky | J = L @ L^T | `torch.linalg.cholesky` |
| Metric solve | L^T @ X = I | `trsm(1.0, L, I, uplo="lower", trans=True)` |
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

- **NKI GEMM is a stub.** Falls back to `torch.matmul` until validated on
  trn1/trn2. The kernel scaffold is in `nki/dispatch.py`.

- **No FP64.** Trainium's Tensor Engine maxes out at FP32. For chemistry
  workloads needing higher precision, double-double arithmetic (emulated FP64
  from two FP32 values) is the path — not implemented yet.

- **Level 3 dominates.** Level 1/2 are included for API completeness but
  won't get NKI kernels initially. The Tensor Engine is wasted on vector ops.

- **DF-MP2 example uses Python loops** over occupied orbital pairs. A
  production implementation would batch these as a single batched_gemm call.

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
- `trnfft` — FFT + complex ops (https://github.com/scttfrdmn/trnfft)
- `trnblas` — BLAS operations (this repo)
- `trnsolver` — Linear solvers, eigendecomposition (planned)
- `trnrand` — Random number generation (https://github.com/trnsci/trnrand)

All repos: Python/NKI, Apache 2.0, Playground Logic.
