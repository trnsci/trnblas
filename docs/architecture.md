# Architecture

## Layout

```
trnblas/
├── trnblas/
│   ├── __init__.py      # Re-exports all BLAS operations
│   ├── level1.py        # axpy, dot, nrm2, scal, asum, iamax
│   ├── level2.py        # gemv, symv, trmv, ger
│   ├── level3.py        # gemm, batched_gemm, symm, syrk, trsm, trmm
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py  # auto/pytorch/nki dispatch + NKI GEMM kernel
├── tests/
├── examples/df_mp2.py
└── benchmarks/
```

## DF-MP2 → BLAS mapping

The DF-MP2 algorithm maps directly to trnblas Level 3 operations:

| DF-MP2 Step        | Math                                  | trnblas Call                                     |
|--------------------|---------------------------------------|--------------------------------------------------|
| Half-transform     | `(iν|P) = C_occ^T @ (μν|P)`           | `gemm(1.0, C_occ, eri, transA=True)`             |
| MO transform       | `(ia|P) = C_vir^T @ (iν|P)`           | `gemm(1.0, C_vir, iv_P, transA=True)`            |
| Cholesky           | `J = L @ L^T`                         | `torch.linalg.cholesky`                          |
| Metric solve       | `L^T @ X = I`                         | `trsm(1.0, L, I, uplo="lower", trans=True)`      |
| Metric contract    | `B = (ia|Q) @ J^{-1/2}`               | `gemm(1.0, ia_P, J_inv_half)`                    |
| Energy             | `T_ab = B_i @ B_j^T`                  | `gemm(1.0, B[i], B[j], transB=True)`             |

## NKI GEMM strategy

The NKI GEMM kernel uses stationary tile reuse on the Tensor Engine:

1. Load A tile (128×128) to SBUF as stationary operand.
2. Stream B tiles through the systolic array as moving operand.
3. Accumulate partial products in PSUM.
4. One A load serves all B tiles → 2× fewer SBUF loads vs naive.

For DF-MP2, the MO coefficient matrix `C` is the natural stationary operand
since it's reused across all auxiliary basis indices `P`.

## Known gaps

- **Level 3 NKI coverage is partial.** `gemm`, `batched_gemm`, `syrk`, and
  `trsm` (left-side blocked) have NKI kernels. `symm` and `trmm` still
  dispatch straight to PyTorch. Neither is in the DF-MP2 hot path.
- **Batched-pair energy (v0.5.2–v0.5.4) solved the dispatch overhead.**
  `nki_batched_pair_energy`
  ([#43](https://github.com/trnsci/trnblas/issues/43),
  [#46](https://github.com/trnsci/trnblas/issues/46)) replaces the nocc²-loop
  dispatch with a single `@nki.jit` call (small shape) or chunked i-loop
  (medium/large). Warm: **3.6× faster than torch at small shape, 5.2× at
  medium shape.**
- **No FP64.** Trainium's Tensor Engine maxes out at FP32.
  **Decision (2026-04-18):** FP32 is sufficient — both gate cases are well
  below 1 µHartree.
  [#10](https://github.com/trnsci/trnblas/issues/10) closed "not needed";
  [#22](https://github.com/trnsci/trnblas/issues/22) (double-double) deferred
  indefinitely. See [Precision envelope](#precision-envelope) below.
- **Level 1/2 are PyTorch-only.** The Tensor Engine is wasted on vector ops;
  Level 3 is where NKI acceleration pays off. Not planned to change.

## Precision envelope

Trainium's Tensor Engine is FP32-only. trnblas uses FP32 throughout. The
question for chemistry workloads is when FP32 accumulation error becomes
visible relative to a FP64 reference (PySCF).

### Measured |E_trnblas − E_pyscf| (v0.5.4, trn1.2xlarge, neuronxcc 2.24.5133)

All values from `pytest -m "pyscf and slow"` on hardware (2026-04-18, [#20](https://github.com/trnsci/trnblas/issues/20)).

| Molecule | Basis   | nocc | nvir | Pair-energy terms | |ΔE| Ha   |
|----------|---------|-----:|-----:|------------------:|----------:|
| H₂O      | sto-3g  |    5 |    2 |           100     | < 1e-6    |
| H₂O      | cc-pVDZ |    5 |   19 |         ~9000     | < 1e-5    |
| CH₄      | cc-pVDZ |    5 |   29 |        ~21000     | < 1e-5    |
| NH₃      | cc-pVDZ |    5 |   21 |        ~11000     | < 1e-5    |
| glycine  | sto-3g  |   20 |   10 |       ~160000     | 1.71e-08  |
| glycine  | cc-pVDZ |   20 |   75 |      ~9000000     | **3.51e-07** |
| (H₂O)₃  | sto-3g  |   15 |   12 |       ~324000     | 4.17e-08  |
| H₂O      | cc-pVTZ |    5 |   53 |        ~70000     | **1.99e-07** |

All 8 cases pass their tolerances. The `test_precision_envelope` test class in
`tests/test_df_mp2_pyscf.py` covers all rows.

### Decision gate for double-double (#10, #22)

**Decision (2026-04-18):** FP32 is sufficient. Both gate cases are well below
the 1 µHartree threshold: glycine/cc-pVDZ = 3.51e-07 Ha and h2o/cc-pVTZ =
1.99e-07 Ha. [#10](https://github.com/trnsci/trnblas/issues/10) is closed as
"not needed". [#22](https://github.com/trnsci/trnblas/issues/22) (double-double
emulation) is deferred indefinitely.

For reference, the original gate:
- |ΔE| < 1 µHartree at cc-pVTZ and glycine/cc-pVDZ → FP32 sufficient (this outcome)
- |ΔE| > 10 µHartree → double-double emulation warranted

The FP32 precision floor for a single GEMM (M×K×N) is approximately
`sqrt(K) × machine_epsilon ≈ sqrt(K) × 6e-8`. For DF-MP2, the dominant
accumulation is over the auxiliary basis P (K = naux), so the per-element
error in T = B_i @ B_j.T is O(sqrt(naux) × 6e-8). With naux = 1536
(medium shape) this is ~2.4e-6 per element, which can accumulate to
~µHartree in the energy sum for large nvir.

The pair-loop accumulation error grows as O(nocc² × nvir² × sqrt(naux) × ε).
At glycine/cc-pVDZ (nocc=20, nvir=75, naux≈300), the theoretical FP32
floor is ~5e-5 Ha — which sets the tolerance in `test_precision_envelope`.
