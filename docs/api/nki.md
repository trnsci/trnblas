# NKI Backend

The NKI dispatch layer controls whether BLAS operations run on the native
Trainium Tensor Engine or fall back to PyTorch.

## Backend selection

```python
import trnblas

trnblas.set_backend("auto")     # NKI on Trainium, PyTorch elsewhere (default)
trnblas.set_backend("pytorch")  # force PyTorch fallback
trnblas.set_backend("nki")      # force NKI (requires nki>=0.3.0)
```

`trnblas.HAS_NKI` is `True` when the `nki` package (NKI 0.3.0+, Neuron
SDK 2.29+) is importable. trnblas uses the canonical `nki.*`
namespace directly; the legacy `neuronxcc.nki.*` shim is not used.

## Environment variables

| Variable | Effect |
|---|---|
| `TRNBLAS_REQUIRE_NKI=1` | Re-raise kernel exceptions instead of falling back to `torch.matmul` — surfaces silent breakage in validation runs. |
| `TRNBLAS_USE_SIMULATOR=1` | Route kernel dispatch through `nki.simulate(kernel)(numpy_args)` on CPU. Bypasses `torch_xla` + NEFF compile; used for fast correctness iteration. See [docs/developing_kernels.md](../developing_kernels.md). |

## GEMM kernel

`trnblas.nki.nki_gemm` — NKI-dispatched GEMM with stationary tile reuse:

- A tile (128×128) loaded once to SBUF, held stationary in the systolic array.
- B tiles streamed through as the moving operand.
- Partial products accumulated in PSUM.
- HBM padding: M/K rounded to 128, N rounded to 512 (when N > 512); kernel
  uses `TILE_N = min(N, 512)` for single-tile small-N. Result is sliced back
  to the original (M, N).

**Status:** validated on trn1.2xlarge with neuronxcc 2.24. 17/17 hardware
tests pass. Cached-NEFF speedup ~2.8× on warm runs; per-call kernel
timings land at 1.6 ms (512³) and 4.5 ms (1024³) on warm cache.

## Batched GEMM

`trnblas.nki.nki_batched_gemm` — per-slice dispatch through the cached 2D
`_gemm_kernel`. Every slice after the first hits the NEFF cache (identical
signature), so per-slice cost is HBM transfer + Tensor Engine dispatch only.

## Fused MP2 energy-reduction kernel

`trnblas.nki.nki_mp2_energy(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)`
— computes

```
E_chunk = Σ_{i, j, a, b} T[i,j,a,b] * (2·T[i,j,a,b] - T[i,j,b,a]) / Δ[i,j,a,b]
```

where `T[i,j,a,b] = T_flat[i*nvir + a, j*nvir + b]` and
`Δ[i,j,a,b] = eps_occ[i] + eps_occ[j] - eps_vir[a] - eps_vir[b]`.

Partition-dim sub-tiling: `P_TILE` picked at trace time as the largest
divisor of `nvir` that is ≤ 128 (the NKI partition limit). All three
DF-MP2 bench shapes (nvir = 112 / 448 / 672) share `P_TILE = 112`, so
one compiled kernel serves all.

**Returns:** `(P_TILE, ic, nocc)` fp32 tensor of per-partition partials;
caller reduces host-side via `.sum()`.

**Status:** on-hardware correctness validated across
`nvir ∈ {8, 16, 64, 256, 448}` — all 5 tests pass on trn1 (under the
NKI 0.3.0 MLIR verifier after the #15 M2 broadcast-fix commit
[`c1769c6`](https://github.com/trnsci/trnblas/commit/c1769c6)).

**Measured perf** (trn1.2xlarge, warm NEFF cache, SDK 2.29):

| DF-MP2 shape | Energy step — torch | Energy step — fused | Speedup |
|---|---:|---:|---:|
| medium (nbasis=512, nocc=64, nvir=448) | 8.03 s | 5.43 s | **1.48×** |
| large (nbasis=768, nocc=96, nvir=672) | 44.57 s | 30.27 s | **1.47×** |

Bit-parity with the torch reference at both shapes (`atol=1e-4`,
`rtol=1e-4`). The fused kernel beats torch consistently but the
speedup ratio is roughly shape-invariant — per-(i,j) dispatch cost
scales with `nocc²` same as the torch path, so the RFC's 3–5× target
isn't hit. The remaining perf work (larger free-dim tiles, atomic-add
variant, multi-engine pipeline evidence) is tracked under
[#15](https://github.com/trnsci/trnblas/issues/15). Production
`examples/df_mp2.py` exposes the kernel via `--fused-energy`; default
remains the torch path until a future milestone hits the 3× bar.
