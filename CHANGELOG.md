# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Tests

- **Device conformance runs are signal-only** (trnsci/trnsci#35; test code
  only, no library change). On `--backend nki` with a non-CPU
  `--conformance-device`, `tests/test_conformance.py` marks the known platform
  failures `xfail(strict=False)` at run time, by rule, one tracking issue per
  category: complex128 at fp32 precision (#103), complex64 `mv` / `triu` /
  `tril` unsupported on device (#127), float64 `syrk` checked at fp64
  tolerance under explicit `nki` (#100), and the stride-0 `to_dev()` harness
  limit (#128). A new failure is then the only red, and a category that starts
  passing shows as XPASS. CPU CI runs every case unmarked. Device runs print a
  summary (passed, xfailed and XPASS by category, NEW failures);
  `--conformance-json PATH` writes it, and `scripts/conformance_expect.py`
  merges chunked runs. `tests/test_device_known.py` pins the rules on CPU.

## [0.13.0] — 2026-11-10

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: `trnblas.gemm_fp64`, an fp64-class GEMM by Ozaki splitting on the bf16 Tensor Engine (#22, #124), with the operand split on the device (#123, #125): 2e-16 to 4e-16 of |A||B| at the default k = 8, and 3.0× host fp64 on 2 vCPUs at (3267, 8400, 241), 10.3× at 4096³. `_host_empty` falls back to pageable memory when pinning raises.

### Added

- **`gemm_fp64` splits its operands on the device** (#123, refs #22,
  trnsci/trnsci#129). Each fp64 operand is uploaded as its raw bits (8 bytes
  per element, a plain copy) and split by four new NKI kernels:
  `_ozaki_emax_kernel` / `_ozaki_emax_col_kernel` compute the row (column)
  maxima of the exponent field, and `_ozaki_split_kernel` /
  `_ozaki_split_col_kernel` produce the digits of `ozaki.split`. Device
  float64 arithmetic is fp32, so the kernels use only VectorE bitvec ops and
  exact fp32 arithmetic:
  - three disjoint fp32 pieces of |x| · 2^(20 − e), built from the bit fields;
  - an exact floor chain to 21-bit fixed-point chunks, with the floor taken
    from the fp32 → uint32 cast and corrected, so it holds whichever way the
    cast rounds;
  - the 7-bit fields by shifts, then the balanced digits with their
    round-half-even carries.

  The new `trnblas.ozaki.split_from_bits` is the same arithmetic on the host.
  On Trn2 the digits equal the host split's (as values) per row and per
  column, for every k tested, on random fp64 bit patterns and on half-way
  ties. `gemm_fp64` stays bitwise equal to `ozaki.reference_gemm`, at LNC=1
  and LNC=2. The split costs about 0.55 ns per element on one logical core,
  against about 11 ns on one host core.

  End to end at k = 8 (one logical core at LNC=2, host fp64 on 2 vCPUs):
  - (3267, 8400, 241): 68 ms against 207 ms for host fp64, 3.0× faster (it
    was 2.8× slower with the host split);
  - (8192, 3508, 241): 2.7× faster;
  - 4096³: 10.3× faster.

  The host split still runs under `TRNBLAS_OZAKI_DEVICE_SPLIT=0`, for
  operands of 2 GiB or more, and for a row or column whose maximum is
  subnormal. inf and nan are found from the device exponent maxima and take
  host fp64, as before. See docs/benchmarks.md.

- **`trnblas.gemm_fp64(A, B, *, slices=8, transA=False, transB=False)`: an
  fp64-class GEMM on the bf16 Tensor Engine by Ozaki splitting** (#22,
  trnsci/trnsci#129). The host splits fp64 A row by row and B column by
  column into `slices` exact 7-bit bf16 integer slices (`trnblas/ozaki.py`).
  The new NKI kernel `_ozaki_gemm_kernel` runs the k(k+1)/2 slice products of
  the diagonals s + t ≤ k − 1 with fp32 PSUM accumulation in groups of at most
  4096 terms, which keeps every group sum an exact integer (same-diagonal
  pairs share a group when K < 4096), and folds each group into a VectorE
  TwoSum (hi, lo) pair. The host recombines in fp64. Measured on Trn2 (one
  logical core, LNC=2, normal data, max |C − C_exact| / (|A||B|)): 2e-16 to
  4e-16 at the default k = 8 (host fp64 BLAS level), about 1e-14 at k = 7 and
  1e-12 at k = 6. The kernel runs at 87–96 TFLOP/s of bf16 slice products on
  squares, 35–41× host fp64 on 2 vCPUs at k = 8. End to end the host split
  dominates (about 11 ns per element at k = 8 on one host core, 3–4 ns on 12
  vCPUs): 1.4–2.4× host fp64 on 2 vCPUs on squares 2048–4096³ but 0.5–0.9×
  the 12-vCPU host, and 2.8× slower than host fp64 on 2 vCPUs on the
  narrow DF-MP2 shapes (see docs/benchmarks.md). Results are deterministic, bitwise
  equal at LNC=1 and LNC=2, and equal to the host reference
  `trnblas.ozaki.reference_gemm` bit for bit. Launches are capped at
  1024 × 1024 outputs (one NEFF per (slices, K, block), first-call compile
  15 s instead of 240 s at 4096³) and keep every operand under 2 GiB (#112).
  Narrow products (N < 512 < M) run as Cᵀ = Bᵀ Aᵀ. Off the native path (CPU,
  Trn1, `set_backend("pytorch")`) and for non-finite operands, `gemm_fp64` is
  host fp64 `A @ B`.

### Fixed

- **Tests: the device suites' `_rel` helpers dropped the imaginary part of
  complex results** (#113; test code only, no library change).
  `tests/test_nki_native.py` and `tests/test_neuron_span_device.py` took
  `.double()` of their operands, so a complex result was compared on its real
  part only. In 0.12.0 and 0.12.1 the complex128 cases of
  `TestNativeKernelOpOffCastFloat64Warning` (`gemm-c128`, `syrk-c128`, #106)
  and the left complex128 `trsm` check also built their inputs as
  `A64.to(complex128)`, with a zero imaginary part. `_rel` now compares in
  complex128 when either side is complex, those cases use genuinely complex
  operands, and a CPU test (`tests/test_rel_helpers.py`) covers the helpers.
  pytest now turns torch's "Casting complex values to real discards the
  imaginary part" warning into an error, so a recurrence fails the run.
- **Host results on the native path no longer need pinnable memory** (#61
  follow-up). `_host_empty` asked for `pin_memory=True` whenever the native
  path returned a large CPU result, which raises on a torch build with no
  accelerator to pin for (e.g. "Found no NVIDIA driver" with a CUDA build on
  a GPU-less Linux host). It now falls back to pageable memory when pinning
  raises, as `multicore._stage_to_host` already did. Where pinning works
  (Trn2) the behaviour is unchanged.

## [0.12.1] — 2026-11-07

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: device matmuls with 32 MiB or longer rows (an operand's 128 partitions spanning 4 GiB) are no longer silently wrong (#112, aws-neuron-sdk#1457), and `parallel.df_mp2_energy`'s symmetric half-transform at nocc ≥ 128 is no longer wrong (#116).

### Fixed

- **Device matmuls with 32 MiB or longer rows were silently wrong (#112,
  aws-neuron-sdk#1457).** On the native TorchNeuron path (Trn2, SDK 2.33),
  a device matmul returns wrong values with no error (or aborts with NRT
  status 1203) once an operand's 128 DMA partitions span 4 GiB or more,
  i.e. `min(rows, 128) * row_bytes >= 2**32`. That holds for a contiguous left
  operand as well, so the trnblas#57 guard (transposed left operand only)
  did not cover it. Example: fp32 `(96, 128) @ (128, 8,389,632)` gave rel
  err 0.8. Affected 0.6.0–0.12.0: `gemm`, `batched_gemm`, `symm`, `syrk`,
  `trmm`, `gemv`, the `trsm` updates, `nki_gemm` (aten, LNC-kernel and NKI-kernel
  routes) and `trnblas.parallel`'s GEMMs, when a device-resident (or uploaded)
  operand or the result has fp32 rows of 32 MiB or more (bf16: 16M columns)
  and at least 128 of them. These products are now cut into contiguous row
  blocks of the long operand, each block spanning at most 2 GiB: K is split
  (partial sums), or M / N (concatenated results). No operand is copied.
  Splitting K changes the fp32 summation order of the affected products
  only. Every other product is the same aten call as before (bitwise
  unchanged). At those sizes the LNC and NKI GEMM kernels and the SYRK
  kernel are not used. Host tensors are not affected. Device elementwise
  ops on such tensors (for example `gemm`'s `alpha` / `beta` update of a
  result that is itself 4 GiB with 32 MiB rows) are a separate issue,
  trnsci/trnsci#145.
- **`trnblas.parallel.df_mp2_energy`: wrong half-transform at nocc ≥ 128
  with nbasis × naux ≥ 2²³ (#116).** With `eri_symmetric=True`, the strided
  in-place update of the partial half-transform `part` (nocc_pad, nbasis ×
  naux) has the same ≥ 4 GiB partition span on the device and returned wrong
  values (rel err 0.54 on Trn2 at nbasis 64, naux 131,088, nocc 128). It now
  updates blocks of at most 64 occupied rows there, and is one add as before
  everywhere else. The half-transform GEMMs are covered by the fix above.
  Affected 0.9.0–0.12.0.

## [0.12.0] — 2026-11-07

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: under explicit nki, ops with a kernel now warn once per op when device float64 / complex128 operands miss the float32 kernel cast (#103, #106); parallel.pair_energy raises under TRNBLAS_REQUIRE_NKI=1.

### Fixed

- **Explicit `nki`, ops with a kernel: device float64 off the cast now warns
  (#103).** Under `set_backend("nki")`, float64 / complex128 operands on the
  `neuron` device that did not take the float32 kernel cast ran on device
  aten at fp32 precision with no warning: complex128 (`gemm`,
  `batched_gemm`, `syrk`, the energy APIs), float64 mixed with another dtype
  (a float64 `C` on a float32 `gemm` / `syrk`, mixed `trsm` operands,
  float64 amplitudes with integer or complex `eps`), batched or 1-D `gemm`
  operands, a 1-D `nki_batched_gemm` operand, `side="right"` `trsm`, the
  `metric_inv_half` device route (whose only warning was the inner
  `trsm`'s, and none once `trsm` had warned), and the
  `nki_batched_pair_energy` aten route. `trnblas.parallel.pair_energy` also
  rounded float64 `eps` to float32 silently and did not raise for device
  float64 under `TRNBLAS_REQUIRE_NKI=1`. Each now emits the policy's one
  `UserWarning` per op, in the shared dedup set and with the "precision is
  reduced to float32" wording; `parallel.pair_energy` raises under
  `TRNBLAS_REQUIRE_NKI=1` like the other public ops. Values, routing,
  `auto`, `pytorch`, host operands and the host-only routes (left `trsm` on
  complex128, the `metric_inv_half` host route, which compute at full
  precision and stay quiet) are unchanged.

## [0.11.1] — 2026-11-04

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: trsm and metric_inv_half keep autograd on native auto host routes (#101); no-kernel ops warn on device float64 under explicit nki.

### Fixed

- **Autograd on the native `auto` host routes (#101).** On a Neuron host
  (TorchNeuron dispatch) under the default `auto` backend, `trsm` on host
  operands went through `_trsm_on_host`, which detached `A` and `B` before
  the LAPACK solve, so the result had no graph and gradients silently
  stopped. trnsolver's `eigh_generalized` inherited it (its eigenvalues came
  back with `requires_grad=False`). Affected 0.9.0–0.11.0. The host solve
  now keeps autograd (the copies, casts and `torch.linalg.solve_triangular`
  are all differentiable); values, routing, the float64 policy and the
  kernel dtype guard are unchanged. The same `.detach()` was in the host
  route of `metric_inv_half` for a device-resident metric `J`
  (`_host_inv_chol_factor`, 0.8.0–0.11.0, under `auto` and `nki`), fixed
  the same way. CPU-only hosts, the `pytorch` backend, and `trsm` under
  `nki` were not affected.
- **Explicit `nki`, ops with no kernel: device float64 now warns.** Under
  `set_backend("nki")`, the ops with no NKI kernel (`axpy`, `dot`, `nrm2`,
  `scal`, `asum`, `iamax`, `gemv`, `symv`, `trmv`, `ger`, `symm`, `trmm`)
  computed float64 / complex128 operands on the `neuron` device on device
  aten at fp32 precision with no warning (probe table on #91). They now
  emit the policy's one `UserWarning` per op ("precision is reduced to
  float32"), as the kernel ops do for their float32 cast. Values, routing,
  `auto`, `pytorch`, host operands and `TRNBLAS_REQUIRE_NKI=1` (which still
  raises first) are unchanged.

## [0.11.0] — 2026-11-04

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: the trnsci float64 policy with one dtype check at the NKI kernel boundary (#89, #91); `syrk` honours `uplo` as in BLAS (#74); complex `asum`/`iamax` use |Re|+|Im| (#75); mixed-dtype GEMM raises like `torch.matmul` (#90).

### Changed

- **float64 policy (suite-wide, #89; behaviour change).** float64 under
  `auto` now computes in torch at float64 accuracy (previously cast to fp32
  on the NKI path). Trainium computes float64 at fp32 precision, so no NKI
  kernel gives float64 accuracy; the four rules, the same in every trnsci
  library, are:
  - `set_backend("auto")`: float64 operands (amplitudes, `B`, or orbital
    energies) run in host torch at float64 accuracy. Device aten computes
    float64 at fp32 precision on Trainium (measured on Trn2), so float64 or
    complex128 operands already on the `neuron` device are copied to the
    host, computed there, and returned on the device in float64, with one
    `UserWarning` per op ("float64 on the neuron device is computed on the
    host for float64 accuracy (device arithmetic is fp32); pass float32 to
    stay on device"). This covers every public op, Level 1-3 and
    `trnblas.nki.*`, through one guard (`_fp64_host_guard`).
    `TRNBLAS_FP64_ON_DEVICE=1` keeps device aten (fp32 precision) for callers
    who prefer speed. Host-resident float64 stays in host torch, and a forced
    `TRNBLAS_NATIVE_GEMM=aten|lnc` no longer uploads it to device aten
    unless `TRNBLAS_FP64_ON_DEVICE=1`. For the DF-MP2 energy
    APIs this replaces the fp32 rounding of float64 orbital energies on the
    `pair_energy_sym` and multi-core routes (through 0.10.0); pass float32
    `eps` to keep those calls on the kernel.
  - explicit `set_backend("nki")`: float64 operands are cast to float32 for
    the kernel, and a `UserWarning` ("precision is reduced to float32") is
    emitted once per op. **This changes GEMM-family behaviour**: from 0.9.1
    (#68/#69) through 0.10.0, float64 `gemm`, `batched_gemm`, `syrk`, left
    `trsm` and the device route of `metric_inv_half` ran in torch at float64
    accuracy under `nki`; they now run the FP32 kernels and return float64
    values at float32 accuracy. Use `auto` for float64 accuracy. `symm` has
    no kernel and still runs in torch. A float64 operand mixed with another
    dtype is not cast (torch semantics, as before; #90 is separate).
  - `TRNBLAS_REQUIRE_NKI=1`: float64 raises `NotImplementedError` (the
    #68 `_no_kernel_for_dtype` convention) before any cast; it takes
    precedence over the `nki` cast. float64 / complex128 on the `neuron`
    device raise at every public op (backends `auto` and `nki`, ahead of
    `TRNBLAS_FP64_ON_DEVICE=1`), with no host copy. Before, the routes with
    no kernel computed them in device aten at fp32 precision with no error:
    `gemm`, `batched_gemm`, `syrk`, `symm` and left `trsm` under `auto` on
    the native path, `symm` under `nki`, and Level 1/2.
  - Result dtype: every route returns what the pytorch backend returns for
    the same inputs. The GEMM family already did; the energy kernels
    (`nki_mp2_energy`, `nki_fused_gemm_energy`, `nki_batched_pair_energy`
    including the `pair_energy_sym`, chunked and multi-core routes) returned
    float32 (multi-core: `B.dtype`) through 0.10.0 and now return the
    promoted dtype: float64 for float64 inputs, bfloat16/float16 for
    half-precision amplitudes and orbital energies (the kernel still
    accumulates in FP32; only the returned scalar is cast), float64 for
    float32 amplitudes with float64 `eps_vir`. For `nki_batched_pair_energy`
    and `nki_fused_gemm_energy` the occupied energies enter the pytorch
    reference as Python floats, so the dtype is
    `torch.promote_types(B.dtype, eps_vir.dtype)`. The multi-core aten route
    (`TRNBLAS_CORES`) now keeps the caller's `eps` dtype instead of rounding
    it to `B.dtype`, as the single-core aten route already did.
  - `trnblas.parallel.pair_energy` returns a Python float from an fp32
    all-reduce and still rounds `eps` to `B`'s dtype; only float64 `B`
    under explicit `nki` changes (cast to float32 for the sym kernel, with
    the warning).

  Lands in the release after 0.10.0. `tests/test_conformance.py` keeps the
  fp64 tolerance under `--backend auto` and relaxes it (and skips the fp64
  gradchecks) only under `--backend nki`. Needs device validation.

### Fixed

- **Float64, integer, bool and complex tensors reached the FP32 DF-MP2 energy
  kernels** (silent). `nki_mp2_energy`, `nki_fused_gemm_energy` and
  `nki_batched_pair_energy` had no dtype check, so under `set_backend("nki")`
  or `auto` a float64 (or integer/complex) amplitude tensor, or float64
  orbital energies, went into the FP32 kernels and came back as garbage, or
  failed and fell back to torch with a warning. Now integer / bool / complex
  amplitudes run the torch reference (the pytorch backend's result;
  device-resident `B` on the sym route runs as aten), bfloat16/float16
  operands are upcast exactly to float32 for the kernel, and float64
  follows the policy above. Under `TRNBLAS_REQUIRE_NKI=1` each of these
  raises `NotImplementedError` instead. Affects every release with these
  kernels, through 0.10.0. Found by the suite-wide audit (trnsci/trnsci#35).
- **One dtype check at the NKI kernel boundary.** Every launch (`_launch`,
  `_call_kernel` on the native and XLA paths, the simulator via the new
  `_simulate`, and every `_native.OPS` entry, including the direct
  `pair_energy_sym` calls in `multicore` / `parallel`) checks its operands
  against a per-kernel table `_KERNEL_DTYPES` and raises `TypeError` naming
  the op and dtypes for anything else, or for a kernel without an entry.
  `tests/test_kernel_dtype_guard.py` registers every op through a stub
  `nki_op` that refuses unsupported dtypes and drives every public API with
  float32/float64/bfloat16/float16/int32/int64/bool/complex inputs on the
  native, XLA and simulator paths, under `auto` and `nki`, plus the float64
  policy (no kernel under `auto`; float32 kernel operands and exactly one
  warning per op under `nki`; `NotImplementedError` under
  `TRNBLAS_REQUIRE_NKI=1`) and the result-dtype rule. Open question: #90
  (mixed-dtype GEMM operands).

- **`syrk` honours `uplo`: only the `uplo` triangle of C is referenced, as in
  BLAS `dsyrk`/`cublasSsyrk` (#74).** Through 0.10.0 `uplo` was validated but
  ignored, and with `beta != 0` the result was `sym(alpha·A·Aᵀ + beta·C)`,
  `sym(X) = (X + Xᵀ)/2`. A C stored in one triangle (the BLAS convention, the
  other triangle zero) got half its off-diagonal entries (rel. err 0.04–0.19
  in the audit), and the other triangle of a non-symmetric C leaked into the
  result. Now an `n×n` C is read as the symmetric matrix stored in its `uplo`
  triangle. **Results change** for callers passing a full non-symmetric C:
  only its `uplo` triangle is used. A symmetric C gives the same result as
  before. A C that broadcasts (scalar, row, column) is unchanged.

### Deprecated

- **`syrk` with C stored only in the triangle opposite `uplo` (#74).** When
  every off-diagonal entry of C is in the other triangle (for example a
  lower-stored C with the default `uplo="upper"`), `syrk` uses that triangle
  and emits a `DeprecationWarning` asking for the matching `uplo`. Strict
  BLAS would read only the diagonal of such a C; the fallback will be removed
  in a later minor release. Same rule as trnsparse `spmv_symmetric`
  (trnsci/trnsparse#52). A diagonal-only C gives no warning.
- **Complex `asum` and `iamax` use the BLAS magnitude |Re| + |Im| (#75).
  Results change for complex input.** Through 0.10.0 they used the modulus
  |x|, unlike `scasum`/`dzasum`/`icamax`/`izamax` and their cuBLAS
  equivalents. Now `asum(x)` returns `sum(|Re x_i| + |Im x_i|)`, which is up
  to √2 larger than the old value, and `iamax(x)` picks the largest
  `|Re x_i| + |Im x_i|`, so it can return a different index (for
  `[3, 2+2j]` it now returns 1, not 0). Real input is unchanged. This is a
  correctness fix to match BLAS, with no deprecation period. To keep the old
  values, use `x.abs().sum()` and `x.abs().argmax()`. `iamax` stays 0-based
  (cuBLAS is 1-based); the docstring and `docs/api/level1.md` now say so.
### Fixed

- **Mixed-dtype GEMM operands raise the same error as `torch.matmul` on every
  backend (#90).** The NKI dtype check looked at each operand on its own, so
  `gemm(1.0, A_fp32, B_bf16)` (and `batched_gemm`, `symm`) took the NKI kernel
  route under `set_backend("nki")`, the XLA path or `TRNBLAS_NATIVE_GEMM=nki`/`lnc`,
  and returned a result in the left operand's dtype, where the pytorch backend
  raises `RuntimeError`. Affects 0.2.0 through 0.10.0. Mixed operands now go to
  torch before any kernel route and raise torch's error (also under
  `TRNBLAS_REQUIRE_NKI=1`). A mixed `trsm` computes in torch, as on the pytorch
  backend.

## [0.10.0] — 2026-11-02

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: **correctness fixes — upgrade recommended.** In 0.2.0–0.9.1, `gemm` with transA/transB on batched operands reversed every dimension (#72) and BLAS letter flags (`uplo="U"`, `diag="U"`, `trans="N"`) ran the opposite branch (#73); on the NKI route `batched_gemm` ignored batch broadcasting and integer `metric_inv_half` returned zeros (#84, #86). New: LNC=2 GEMM kernels that beat aten on the DF-MP2 half-transform (#88), and cached `wrap_nki` handles that cut eager NKI dispatch from ~0.11 to ~0.03 ms (#92, aws-neuron-sdk#1456).

### Performance

- **Eager NKI kernel calls skip the per-call handle rebuild and the custom-op
  layer (trnblas#88).** On the native path, a `trnblas::` custom-op call cost
  about 0.11 ms of host time before the kernel was queued, against 0.016 ms
  for aten. Most of that went to nki's `Kernel.__call__`, which rebuilds the
  torch handle on every call, plus the `torch.library` layer. `_call_kernel`
  now calls a `torch_neuronx.wrap_nki` handle built once per kernel. On trn2
  that is 0.027 ms per GEMM call. A 128x128x512 kernel GEMM plus synchronize
  drops from 0.188 to 0.118 ms, `trnblas.gemm` from 0.221 to 0.136 ms, and
  `syrk` 256x128 from 0.187 to 0.114 ms. `torch.compile`, requires-grad,
  int64 and tensor-subclass inputs keep the custom op.
  `TRNBLAS_NKI_EAGER_HANDLE=0` turns this off. `scripts/bench_native_dispatch.py
  call` measures it.

### Fixed

- **`batched_gemm` on the NKI route follows torch.matmul batch broadcasting (#84).**
  The per-slice NKI loop indexed `A[i], B[i]` over `A.shape[0]`. With
  `set_backend("nki")` on Trainium, a broadcast batch `(1, M, K) x (B, K, N)`
  silently returned only `A[0] @ B[0]` with shape `(1, M, N)`, and 2-D or 4-D
  operands raised `ValueError`. The pytorch and aten routes were correct.
  Affects 0.3.0 through 0.9.1. The loop now runs over the broadcast batch
  index on expanded views, with no copies. Found by the conformance suite on
  Trn2 (trnsci/trnsci#35).
- **`metric_inv_half` with an integer `J` on the `neuron` device returned a
  truncated integer factor** (silent, #86). The native host route computes
  L^-T in float64 and cast it back to `J.dtype`, which turned most entries
  into 0. The torch route raises for integer input. Every route now raises
  `NotImplementedError`. Affects 0.8.0 through 0.9.1. Found by the round-3
  device harness on Trn2 (trnsci/trnsci#35).


- **BLAS flag spellings silently ran the opposite operation** (silent). Each
  wrapper compared `uplo` / `side` / `diag` against one word and took the other
  branch for anything else, and `trans*` used truthiness: `uplo="U"` ran the
  lower triangle, `diag="U"` a non-unit diagonal, `trans="N"` the transpose
  (rel. err 0.04–2.4 in `symv`, `symm`, `trmv`, `trmm`, `trsm`, `syrk`, `gemv`,
  `gemm`, `batched_gemm`). The documented words and the BLAS letters are now
  accepted case-insensitively and anything else raises `ValueError`. Every
  spelling that was correct before gives the same result. Affects every
  release, 0.2.0–0.9.1, on every backend (#73). The API docs listed a
  non-existent `unit=` keyword for `trmv` / `trmm`; it is `diag=`.
- **`gemm(..., transA=True)` / `transB=True` on batched (>= 3-D) operands
  reversed every dimension** (silent). `op(X)` was built with `X.T`, which on a
  non-2-D tensor turns `(b, m, k)` into `(k, m, b)`; `torch.matmul` then
  multiplied the reversed tensor whenever the shapes broadcast (square
  batches), and raised otherwise. `op()` now transposes the last two
  dimensions, as `batched_gemm` does. Affects every release, 0.2.0–0.9.1, on
  every backend; 2-D calls are unchanged (#72).

### Added

- **LNC-split NKI GEMM for Trn2 (#79).** Two new kernels on the native path:
  `_gemm_lnc_tn_kernel` (A given as A^T, e.g. a transposed view) and
  `_gemm_lnc_kernel` (row-major A). Both are launched gridless and split C
  over both physical cores of an LNC=2 logical core (`num_programs()` /
  `program_id`; one program at LNC=1), keep a 512-row A^T block stationary in
  SBUF over all of K with FP32 PSUM accumulation, load B in up to 2048-wide
  tiles, and handle every M/K/N remainder in-kernel. No DMA transposes
  (aws-neuron-sdk#1452): A^T comes from plain `nl.load`s, and row-major A is
  transposed on the Tensor Engine. Results match aten's error against float64
  at every awkward size tested. On the device the kernels run at aten's
  Tensor Engine rate from ~4096³ up, but each custom-op launch costs ~0.1 ms
  more host time than aten, so `auto` takes them only in the measured win
  region: device-resident fp32 with A a transposed view, 16 ≤ M ≤ 64,
  128 ≤ K ≤ 512, N ≥ 256·K and K·N ≥ 2^24. That is the DF-MP2 half-transform
  `C_occ.T @ eri` from the umbrella profile's `l` size up (1.5-1.7x faster
  than aten there). Everything else keeps aten. `TRNBLAS_NATIVE_GEMM=lnc`
  forces the kernels for plain 2-D GEMMs. Tables: `docs/benchmarks.md` and
  #79; harness: `scripts/bench_lnc_gemm.py`.

- **Reference-conformance suite (trnsci/trnsci#105).** `tests/test_conformance.py`
  checks every public op against independent float64 torch formulas at sizes
  0/1/127/129/1000, fp32/fp64/complex (bf16/fp16 for the GEMM family), every
  flag value and BLAS letter, non-contiguous and stride-0 inputs, batch
  broadcast, beta=0 with NaN in C/y, input mutation and output aliasing, and
  fp64 gradcheck. About 6500 cases, under 10 s on CPU. Needs-decision
  behaviour is `xfail(strict=True)` (#74, #75). `--backend nki|auto
  --conformance-device neuron` runs the same cases on Trainium.
- **Guard against the NKI trace-cache closure bug ([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454),
  trnsci/trnsci#35).** `tests/test_trace_cache_safety.py` (CPU) finds every
  `@nki.jit` kernel and fails if its body, or a helper it calls, reads a
  closure variable or a module global that is not a constant expression
  (reviewed exceptions are allow-listed with a reason), and checks each
  kernel's `__closure__` at runtime against a stub `nki`.
- Tests: `conftest.py` no longer turns the NKI trace cache off. Neuron tests
  run with it on, as under `import torch_neuronx`, with a fresh
  `NKI_TRACE_CACHE_URL` directory per session (no stale editable-install
  traces). `NKI_ENABLE_TRACE_CACHE=0` still turns it off.
- README: known platform issue note for #1454.

## [0.9.1] — 2026-11-01

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: **correctness fix:** under `set_backend("nki")`, float64 `gemm` / `syrk` / `trsm` went into the FP32 NKI kernels and came back non-finite with no error (0.9.0 and earlier on the native path). They now compute in torch, or raise under `TRNBLAS_REQUIRE_NKI` (#69, #68). `auto` and `pytorch` were unaffected. Adds tail-tile regression tests (#67).

### Fixed

- **Float64 GEMM / SYRK / TRSM under `set_backend("nki")` returned non-finite
  values** (silent). Measured on **0.9.0** on Trn2. The `"nki"` route
  had no dtype check in any release, so every release with the NKI GEMM path
  is affected. `"auto"` and `"pytorch"` were correct, because `auto` keeps
  float64 off the device. Float64 (and any other dtype the kernels do not
  compute) now runs in torch on the operands' device, and TRSM on the host.
  Under `TRNBLAS_REQUIRE_NKI=1` it raises `NotImplementedError` instead (#68).

## [0.9.0] — 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: multi-core DF-MP2 via `trnblas.parallel` (2.5–2.9× on 4 cores, #60); pinned-memory downloads make host-input wrappers 1.7–15× faster (#64); measured native `auto` routing — aten on device, CPU for small host calls (#65).

### Added

#### Multi-core on one Trn2 chip (#24)

- **`trnblas.parallel`**: SPMD tensor-parallel GEMM and DF-MP2, one process
  per logical NeuronCore, under `torchrun` with TorchNeuron's `neuron`
  process-group backend. Cross-core data moves only through on-chip
  collectives.
  - `df_mp2_energy(...)` takes the same host inputs as `examples/df_mp2.py`.
    It splits the ERI upload by μ rows, reduce-scatters the half-transform
    over the occupied index, all-gathers B and splits the pair energy by
    occupied rows. The metric factor on rank 0 overlaps the upload.
  - `eri_symmetric=True` uploads only the lower triangle of (μν|P), or takes
    it packed (`pack_tril`, PySCF's `_cderi.T`).
  - Measured on trn2.3xlarge (4 logical cores), end to end from host inputs
    against the single-core library path:
    - 512/64/1536: 348 → 138 ms (2.5×);
    - 768/96/2304: 1323 → 459 ms (2.9×).
  - The energy step alone scales 3.6× on 4 cores.
  - `gemm_row_parallel`, `gemm_col_parallel`, `gemm_k_parallel` and
    `all_gather_rows` work on shards resident on each rank. At 8192³ the local
    GEMM goes from 30.2 to 10.6 ms on 4 ranks, or 13.5 ms with C all-gathered.
  - Everything also runs on CPU with `gloo`, which is how it is tested.
- **`trnblas.set_cores(n | "auto" | None)` / `TRNBLAS_CORES`**: opt-in,
  single-process split of `nki_batched_pair_energy` by occupied rows across
  visible cores, with B staged through a reused pinned host buffer.
  - 96/672/2304: 685 → 389 ms on 4 cores.
  - 64/448/1536: about even.
  - `auto` splits above 1e13 flop.
  - Unset, nothing changes.
- `scripts/bench_parallel.py` and `scripts/bench_multicore.py`, and a
  multi-core section in `docs/benchmarks.md`.

#### Host results at the host-link rate (#61)

- **Host-input wrappers return results in cached host memory.** On the
  native path, a device result of 1 MiB or more copies into a fresh tensor
  from the accelerator's caching host allocator (`pin_memory=True`), not
  through `.to("cpu")`. The memory is already faulted in, so the copy runs at
  13-14 GB/s, not 4 GB/s. Every call still returns a fresh tensor: a block
  is recycled only after its owner is freed. `TRNBLAS_PINNED_OUTPUT=0`
  restores `.to("cpu")`. Freed blocks stay cached, rounded up to a power of
  two, until `torch.accelerator.empty_host_cache()`.
- **In-place epilogues** in `gemm`, `batched_gemm`, `syrk` and the blocked
  `trsm`. `alpha`, `beta * C`, the batch stack, the SYRK symmetrisation and
  the TRSM updates no longer allocate another fresh host tensor each. Each of
  those was a page-fault-bound pass at about 4.6 GB/s. The out-of-place form
  is kept under autograd, for dtype promotion and for broadcasts that grow
  the result.
- Helpers `_to_host` / `_host_empty` in `trnblas/nki/dispatch.py`.
  `scripts/bench_host_link.py` (transfer options) and
  `scripts/bench_host_wrappers.py` (wrappers end to end).

### Changed

#### Measured `auto` routing for the native GEMM family (#62, #63)

- **`auto` picks host CPU, upload + aten, or aten on the device by measured
  time**, for `gemm`, `batched_gemm`, `syrk`, `symm` and left `trsm` on the
  native (TorchNeuron) path. Measured on trn2 across 24 GEMM, 8 batched, 10
  SYRK, 12 TRSM and 5 SYMM shapes (square, ragged, skinny, DF-MP2), fp32 and
  bf16 (`scripts/bench_auto_routes.py`).
  - Device-resident operands: aten, as before. aten beat the NKI kernel at
    every GEMM / batched / SYRK shape and dtype, by 1.1-6x (20-310x for
    batched).
  - Host operands: the host CPU below a per-(op, dtype) threshold on
    multiply-adds and host-link bytes, scaled by `torch.get_num_threads()`;
    above it, one upload, aten, and one download through the cached host
    memory of #61. `gemm` 8192³: 182 → 87 ms; `batched_gemm` 32×224×768×768:
    125 → 10 ms; `symm` 4096²: 636 → 21 ms; small GEMMs no longer pay the
    ~0.4 ms upload/launch floor (64³: 0.38 → 0.02 ms).
  - Transposed and batch-broadcast host operands upload their dense base
    once and are transposed or expanded on the device.
  - `trsm` on host operands solves with LAPACK on the host (1.5-12x faster
    up to n = 2304); device-resident operands keep the blocked device solve.
    Wide right-hand sides lose here: 1024 × 16384 was 113 ms with the old
    host-panel + NKI-update path and is 150-163 ms on the host.
  - `syrk` through aten symmetrises on the device at every size.
- **`set_backend("nki")` now always runs the NKI kernels**, for
  device-resident operands too (before, those went to aten). Use `auto` for
  the measured routes.
- `TRNBLAS_NATIVE_GEMM` takes `auto | nki | aten | host`.

### Fixed

- `trsm` on bf16 / fp16 host operands under `auto` raised
  (`triangular_solve_cpu` has no bf16); it now solves in fp32 on the host and
  returns the input dtype.

### Not changed

- **`trnblas.gemm` stays single-core.** Splitting a GEMM whose operands sit on
  `neuron:0` means staging them through the host, which made it 7–40× slower
  at 4096³–8192³. Host-input GEMM is bound by the host link and got slower
  with more cores.

## [0.8.0] — 2026-10-30

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron; LNC=1 and LNC=2) — see trnsci/trnsci#35.

Headline performance: native DF-MP2 factors the metric on the host (float64 LAPACK, naux ≤ 1024) and runs 1.1–1.8× faster end to end than aten on glycine and (H₂O)₃ (#57); `pair_energy_sym` now runs at LNC=1 as well as LNC=2 (#58).

### Added

#### Native DF-MP2: metric factor on the host (#50)

- **`trnblas.metric_inv_half(J)`** (and `trnblas.nki.nki_metric_inv_half`)
  returns the Cholesky inverse factor X = L^-T of an SPD metric J = L L^T, so
  `A @ X` is the density-fitting metric contraction. On the native path with J
  on the `neuron` device and naux ≤ 1024, it copies J to the host once, runs
  LAPACK Cholesky + triangular solve in float64 and uploads X once. Above
  that, and everywhere else (host tensors, the XLA path, the simulator, the
  pytorch backend), it is `torch.linalg.cholesky` + `trsm` as before.
  `TRNBLAS_METRIC_SOLVE=auto|host|device` overrides the choice. Measured on trn2
  with 2 host threads, the host route is 2–8× faster up to naux 555, ties at
  1024 and loses from 1536. Its residual max|XᵀJX − I| is 100–300× smaller
  (2000–8000× at cond(J) = 1e7). `examples/df_mp2.py` uses it for step 1.
- `scripts/bench_metric_solve.py`: the host vs device crossover by naux.
  `scripts/bench_df_mp2_paths.py --metric auto|host|device`.
- `docs/benchmarks.md`: on the trn2 box, `OMP_WAIT_POLICY=PASSIVE` cuts 2–6 ms
  from every device path on small molecules. With torch's OpenMP pool
  spin-waiting on the same few cores, the Neuron runtime's threads are
  starved, and a 20 KB upload takes ~4 ms instead of 0.06 ms.

### Fixed

- **`trnblas::pair_energy_sym` now compiles and runs at LNC=1**
  (`NEURON_LOGICAL_NC_CONFIG=1`). It was launched as an explicit
  `kernel[2]` grid, which neuronx-cc rejects at LNC=1 with `NCC_EVRF066`
  ("configured with LNC=2, but neuronx-cc is using LNC=1"). It now launches
  gridless and splits the j window by `nl.num_programs()`, falling back to
  one program. Warm time at 64/448/1536 is unchanged at LNC=2 (119.7 vs
  119.9 ms); at LNC=1 it is 231 ms on one core (#50).
- **The 96/672/2304 aten figure in the [0.7.0] pair-energy entry.** It was
  63 s, but that is not aten's steady state. With the device mostly free, the
  warm chunked aten energy there takes 1.22 s, so the sym kernel (0.69 s) is
  1.8× faster, not 92×. The 35–64 s only appears when 10–14 GB of the 24 GB
  logical core are already held, as in a full 768/96/2304 DF-MP2 run (trnblas#50).
- **`gemm` with a transposed left operand and a right operand over 4 GiB on
  the `neuron` device** no longer aborts the process. aten::mm on that layout
  hits an execution timeout and "NRT status 1203, A DMA engine encountered an
  unrecoverable error" on torch-neuronx 2.14.3 / runtime 2.35. DF-MP2's
  half-transform `C_occ.T @ eri_flat` hits it from nbasis 768 / naux 2304.
  `gemm` now copies the (small) left operand to row-major first in that case.

## [0.7.0] — 2026-10-30

Native TorchNeuron dispatch (Trn2/Trn3) alongside the PyTorch/XLA path (Trn1, maintenance); validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0) — see trnsci/trnsci#35.

Headline performance: native GEMM routes to aten for device-resident operands (2.4–3.2× faster than the NKI GEMM on Trn2), and a new fused LNC=2 `pair_energy_sym` kernel for DF-MP2 (#55).

### Added

#### Native DF-MP2: aten routing for plain GEMMs, LNC=2 symmetric pair-energy kernel

Measured end to end on trn2 with real molecules (`scripts/bench_df_mp2_paths.py`).
On the native `neuron` device, aten `matmul` beats the standalone NKI GEMM at
every shape tried (2.4–3.2×). The NKI GEMM runs as a single program and
stages A through DMA transposes. In a spike, the same loop launched as an
LNC=2 grid (one program per physical core) with a pre-transposed A came
within 8–15% of aten at 2048³ and above, so those two choices account for
most of the gap.

- **Behaviour change: GEMM-family calls on device-resident operands go to aten
  on the native path.** `gemm`, `batched_gemm`, `syrk` and the trailing updates
  in `trsm` call `torch.matmul` when every operand is already on the `neuron`
  device. Host tensors keep the NKI kernels, and so do the XLA path and the
  simulator. `TRNBLAS_NATIVE_GEMM=nki` restores the NKI kernels for
  device-resident operands.
- **`trnblas::pair_energy_sym`**, a new native-only fused energy kernel. It is
  launched as an LNC=2 grid (one program per physical core) and reads a
  pre-transposed B, so its loads are plain DMAs. B[i] and B[j] stay in SBUF.
  Each pair needs one GEMM, and T^T comes from Tensor Engine transposes of the
  SBUF-resident T. Pair symmetry (e_ij = e_ji) halves the pair count. It
  launches once per occupied row, which keeps the trace and compile small
  (10–45 s instead of 12 min for a single launch). It is 1.75× faster than aten
  at nocc=64/nvir=448/naux=1536 (118 vs 207 ms), and 92× faster at
  96/672/2304 (0.69 vs 63 s), where the aten chunked reduction falls off a
  cliff.
- **Routing for `nki_batched_pair_energy` on device-resident native inputs:**
  the sym kernel above 3 GFLOP per launch, otherwise aten (chunked GEMM + torch
  reduction). Every real molecule measured so far is below that crossover.
  `TRNBLAS_PAIR_ENERGY=auto|sym|aten|legacy` overrides it. Host inputs, the
  XLA path and the simulator keep the existing full-batch / chunked kernels.
- `examples/df_mp2_pyscf.py` gains `--device` (default `auto`: move the
  integrals to the `neuron` device once on the native path) and
  `--energy chunk|fused|pair`. `examples/df_mp2.py` synchronizes the device
  before each stage timer, so per-stage times on `neuron`/CUDA are honest.
- `scripts/bench_df_mp2_paths.py`: per-stage timings for NKI / aten-on-device /
  CPU on PySCF molecules and synthetic shapes.

#### Native PyTorch dispatch on TorchNeuron (#50, trnsci/trnsci#35)

On TorchNeuron (Neuron SDK 2.33+, `torch-neuronx` 2.14, PyTorch 2.14,
Trn2/Trn3), every NKI kernel is now registered as a PyTorch custom op with
`torch_neuronx.nki_op` and dispatched on `device="neuron"`, with no
`torch_xla` import. Ops: `trnblas::gemm_ragged`, one
`trnblas::gemm_<tm>_<tk>_<tn>` per GEMM tile config (default
`gemm_128_128_512`), `syrk`, `mp2_energy`, `fused_gemm_energy`,
`batched_pair`, `j_batched`. Kernel bodies are unchanged.

- The dispatch path is detected on the first kernel launch: native when
  `torch_neuronx` exports `nki_op` and a `neuron` device is available, XLA
  when `torch_xla` is installed, otherwise none. The XLA path (Trn1) is
  unchanged, as are the `torch.matmul` fallback and the
  `TRNBLAS_USE_SIMULATOR` path.
- Custom ops are registered lazily (`trnblas/nki/_native.py` is imported
  only when the native path is selected), so CPU-only installs and the XLA
  lineage see no import-time side effects.
- All host wrappers share one launch helper: GEMM (default and in-kernel
  ragged, still with no torch-side pad), batched GEMM, SYRK, TRSM (through
  GEMM), the MP2 / fused-pair / batched-pair energy kernels, and the opt-in
  autotune sweep (which synchronizes the device around its timings on the
  native path).
- **Path-aware batched-pair routing.** `nki_batched_pair_energy` chose the
  single full-batch kernel below `_BATCHED_PAIR_CHUNK_THRESHOLD = 4096`
  because one launch amortizes the XLA path's ~100 ms dispatch floor. The
  native path uses 3072, measured on trn2: the full-batch kernel is 2–11×
  faster warm at every shape, but above est_iters 3072 its first-call compile
  was 43–178 s, against 2–4 s for the chunked kernel. The XLA value is
  unchanged.
  `TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD` overrides both, for re-benchmarking
  (`scripts/bench_native_dispatch.py pair`).
- `TRNSCI_NKI_DISPATCH=auto|native|xla` forces the dispatch path (a
  suite-wide variable shared by all trnsci libraries). Forcing a path that
  isn't available raises instead of falling back.
- Dormant XLA deprecation hook: with `TRNSCI_XLA_DEPRECATION=1`, selecting
  the XLA path emits one `DeprecationWarning`. Off by default; the suite
  turns it on at TorchNeuron GA (trnsci/trnsci#35).
- New `[torchneuron]` extra (`torch>=2.14`, `torch-neuronx>=2.14`,
  `nki>=0.6`); `[neuron]` stays the PyTorch/XLA lineage.
- Docs: install notes cover both extras and lineages and recommend
  `NKI_ENABLE_TRACE_CACHE=0` for editable installs.
- Tests: `conftest.py` sets `NKI_ENABLE_TRACE_CACHE=0` by default; CPU tests
  for path selection, launch routing of every host wrapper, lazy
  registration, op schemas and fake shapes, and the deprecation hook;
  `tests/test_nki_compile.py` runs `source_to_mlir(...).verify()` for every
  kernel on trn1 and trn2 (simulator CI job); `@pytest.mark.neuron`
  native-path tests (`tests/test_nki_native.py`, not yet run on hardware).
- `scripts/bench_native_dispatch.py`: native re-benchmarks of batched-pair
  full vs chunked and NKI `gemm` vs aten `A @ B`, with host-created inputs.

## [0.6.0] — 2026-10-28

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron device) — see trnsci/trnsci#35.

### Changed

- **Behaviour change: the GEMM tile autotuner is now opt-in (#50).** By default,
  `nki_gemm` / `gemm` / `batched_gemm` use the fixed heuristic tile (128, 128, 512),
  which the kernel clamps per dimension for small shapes, and run no first-call
  sweep. Opt in with `TRNBLAS_AUTOTUNE=1` or `trnblas.set_autotune(True)`. Why: on
  trn2 (SDK 2.33, native device, fp32, cold caches per process) the sweep made the
  first call 7–12× slower, and the tile it picked gave no warm benefit:

  | case | first call | warm |
  |---|---:|---:|
  | 2048³ autotune on | 25.1 s | 7.5 ms |
  | 2048³ autotune off | 3.4 s | 8.1 ms |
  | 4096³ autotune on | 182.7 s | 36.3 ms |
  | 4096³ autotune off | 15.6 s | 35.4 ms |

- **Unaligned GEMM shapes handle their edge tiles in-kernel (#50).** The new
  `_gemm_kernel_ragged` splits each of M/K/N into full tiles plus one static
  remainder tile. The K remainder is a shorter matmul (stationary and moving
  operands both start at partition 0) accumulated into the same PSUM tile. M and
  N remainders are narrower PSUM/SBUF tiles. This removes the torch-level
  pad/slice and its extra copies, halving warm time on odd shapes on Trn2:

  | case (autotune off) | torch pad, warm | ragged kernel, warm |
  |---|---:|---:|
  | 2047³ | 14.9 ms | 7.1 ms |
  | 4095³ | 68.9 ms | 32.0 ms |

  A roughly 10× first-call penalty seen on odd shapes during this work was not
  the pad path. It came from the benchmark's own lazy
  `torch.randn(..., device="neuron")`, which is shape-sensitive
  (aws-neuron/aws-neuron-sdk#1439). With host-created inputs, cold first calls
  cost the same at odd and aligned sizes (4095³ 16.1 s, 4096³ 15.3 s).

  Aligned shapes still use the unchanged default kernel. `TRNBLAS_GEMM_RAGGED=0`
  restores the torch pad path. If the ragged kernel raises, dispatch warns
  (`NkiFallbackWarning`) and falls back to the padded NKI path, or re-raises
  under `TRNBLAS_REQUIRE_NKI=1`. Checked with `nki.simulate` and
  `nki.compiler.frontend.source_to_mlir(...).verify()` (parser and tracer
  frontends, trn2) and on Trn2 hardware (neuron suite, awkward-size sweep).

- **GEMM pad-to-tile / slice-back factored into `_run_padded_gemm` (#50).** The
  NKI GEMM kernels have no remainder handling, so `nki_gemm` / `gemm` /
  `batched_gemm` on the `nki` backend zero-pad M and K to the tile size (128)
  and N to the N-tile (512) when N > 512, then slice the result back to (M, N).
  Already-aligned contiguous operands are passed through with no copy. The
  autotune sweep now uses the same helper. In this refactor alone the behaviour
  was unchanged: dispatch
  already padded this way, and 8191³ pads to exactly 8192³. The 1605 s first call
  for 8191³ on Trn2 (#50) is therefore not a ragged-edge kernel path. The likely
  cause is the first-call autotune sweep, which compiles and times up to six tile
  configs at 8192³. The 8192³ run that followed reused both the cached winner
  (same pow2 bucket) and the NEFF. Hardware confirmation is pending.

### Added

- `scripts/probe_first_call.py N [--mode gemm|full|direct-jit|direct-op|hop|mlir] [--cpu]`:
  prints one `PHASE <name> <seconds>` line for each phase of a cold first GEMM call,
  to find where the odd-shape first-call cost goes (#50).

### Tests

- `tests/test_gemm_padding.py`: CPU tests for the pad/slice wrapper with a
  stand-in kernel. They cover odd M/K/N, every autotune tile candidate,
  transposed views, `gemm` alpha/beta/C with all transpose combinations,
  `batched_gemm`, and the aligned no-copy case.
- `tests/test_gemm_padding.py`: edge-tile index math (`_ragged_dim_plan` /
  `_ragged_tiles` cover every index exactly once; a torch emulation of the ragged
  kernel's tile walk matches `torch.matmul`) and the autotune-off default.
- `tests/test_nki_gemm.py`: neuron-marked `TestRaggedKernel`, which runs the ragged
  kernel directly on odd shapes (1000³, 2047³, 127×129×513, 1×1×1, …) and checks
  that dispatch takes no torch pad path. `TestAutotuner` now enables the sweep
  explicitly.
- `tests/test_nki_gemm.py`: neuron-marked 1000³ check against `torch.matmul`
  with `TRNBLAS_REQUIRE_NKI` semantics, so a silent fallback fails the test.

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
