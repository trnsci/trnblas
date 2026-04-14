# Developing NKI kernels

trnblas ships five NKI kernels (`gemm`, `batched_gemm`, `syrk`,
`trsm`, `mp2_energy`) in `trnblas/nki/dispatch.py`. This page is
for contributors designing / debugging them.

## Three dispatch modes

| Mode | Trigger | When to use |
|------|---------|-------------|
| **PyTorch fallback** | `HAS_NKI = False` (non-Neuron host), or an `_nki_*_impl` exception gets caught | Laptops, GPUs, CI's ubuntu-latest — the default for anyone who doesn't have Neuron installed |
| **NKI hardware** | `HAS_NKI = True` + default env. Kernel runs through `torch_xla` → NEFF compile → Trainium dispatch | Real perf numbers, final validation |
| **NKI simulator** | `TRNBLAS_USE_SIMULATOR=1` + `HAS_NKI = True`. Kernel runs through `nki.simulate(kernel)(numpy_args)` on CPU | Fast correctness iteration during kernel design |

The three modes share the same kernel source: `@nki.jit`-decorated
functions inside `if HAS_NKI:` blocks.

## Simulator workflow

NKI 0.3.0 Stable (Neuron SDK 2.29, April 2026) ships a CPU simulator
that runs kernels without Trainium hardware. It collapses the
iteration loop from ~8–12 min per attempt (instance start + SSM +
NEFF compile) to seconds — critical for kernel design where each
new semantic constraint costs one round-trip to discover.

Run simulator tests on GH Actions `ubuntu-latest` (the `nki-simulator`
job — see below), on the existing trn1 CI instance via SSM:

```bash
AWS_PROFILE=aws ./scripts/run_simulator_tests.sh
```

Or inline on any Linux x86_64 box that has `nki>=0.3.0` installed:

```bash
TRNBLAS_USE_SIMULATOR=1 pytest tests/ -m nki_simulator -v
```

## CI coverage

Three gates, in increasing cost order:

| Gate | Runner | Catches | Misses |
|------|--------|---------|--------|
| `test` matrix (py 3.10/3.11/3.12) | `ubuntu-latest` | Pure-Python correctness against the `torch.matmul` reference. Fast (~1 s). | Anything NKI-kernel-specific. |
| `nki-simulator` | `ubuntu-latest` | Python trace-level kernel errors: wrong `nc_matmul` kwargs, dropped ops (`nl.divide`), shape mismatches, tile-size violations. Seconds per kernel. | MLIR verifier errors — simulator explicitly skips compile. Perf. |
| `neuron` (SSM, manual) | `trn1.2xlarge` | Full NEFF compile + on-hardware execution. MLIR verification. Real perf numbers. | Nothing (this is the ground truth). |

**Of the five NKI 0.3.0 migration breaking-changes trnblas navigated,
four would surface on the `nki-simulator` gate.** Only the
partition-broadcast strictness in `_mp2_energy_kernel` (MLIR-level
verification) still requires the hardware round-trip. The gate isn't
a perfect substitute for hardware, but it catches the 80% that cost
the most AWS round-trips during design.

Hardware (`scripts/run_neuron_tests.sh`, `scripts/run_simulator_tests.sh`)
remains the final sign-off — in particular for any kernel that touches
partition-dim broadcasting, PSUM→SBUF staging, or new `nc_matmul`
patterns.

### Simulator limitations

From [`nki.simulate` API docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.simulate.html):

- **No compile.** NKI compiler errors won't surface — kernels that
  simulate clean can still fail the NEFF compile on hardware.
- **Meta-programming mismatch.** The simulator accepts arbitrary
  Python; the compiler enforces a restricted subset.
- **Memory model is loose.** SBUF / PSUM capacity overflows aren't
  detected; kernels that simulate clean can still OOM the SBUF at
  runtime.
- **No parallelism / latency.** Multi-engine pipelining (TE + VE +
  Scalar + GPSIMD running concurrently) isn't modelled. Simulator
  output gives no perf signal.
- **Not-implemented simulator APIs:** `nki.collectives`,
  `local_gather`, `nc_stream_shuffle` with `mask=255`, `nc_matmul_mx`,
  `quantize_mx`. trnblas uses none of these as of NKI 0.3.0.

**Use the simulator for correctness + constraint iteration. Use
hardware for perf numbers and final sign-off.**

## Namespace: `nki.*` is canonical

NKI 0.3.0 promotes `nki.*` as the official namespace; the legacy
`neuronxcc.nki.*` shim still works in 2.29 but is deprecated.
trnblas imports exclusively from `nki.*`:

```python
import nki
import nki.isa as nisa
import nki.language as nl
```

Minimum dependency: `nki>=0.3.0` (Neuron SDK 2.29+) — declared in
the `[neuron]` extra of `pyproject.toml`.

## Breaking changes we navigated in NKI 0.3.0

All kernels migrated. Recorded here so future kernels start correct:

| Area | Before (Beta 2) | After (0.3.0) |
|------|-----------------|---------------|
| Namespace | `neuronxcc.nki.*` | `nki.*` |
| `nc_matmul` call | `psum[...] += nisa.nc_matmul(a, b)` returns tile | `nisa.nc_matmul(dst=psum, stationary=a, moving=b, accumulate=True)` writes in-place |
| Args are keyword-only | Positional OK | `stationary` / `moving` / `dst` all required kwargs |
| PSUM → HBM via nl.store | `c_sbuf = nl.copy(psum, ...); nl.store(hbm, value=c_sbuf)` | `dma_copy` refuses PSUM source. Use `nisa.tensor_copy(src=psum, dst=sbuf_tile)` then `nl.store(hbm, value=sbuf_tile)` |
| Tensor-tensor `nl.divide(a, b)` | Supported | Dropped. Use `nl.multiply(a, nl.reciprocal(b))` |

See [NKI 0.3.0 migration guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/migration/nki-0-3-0-update-guide.html)
for the complete breaking-change list.

## Design discipline

Kernels in trnblas should **exploit Trainium architecture** rather
than port cuBLAS equivalents. Before designing a new kernel, state
which of these features it uses:

- **Multi-engine pipelining** — Tensor + Vector + Scalar + GPSIMD
  engines run concurrently. Express dataflow so the compiler can
  schedule them in parallel.
- **Explicit SBUF hierarchy** — 24 MB on-chip, 128 partitions ×
  192 KB free. Keep operands resident across many ops; avoid HBM
  round-trips explicitly.
- **Persistent operands** — load once, reuse across many ops.
- **PSUM accumulator** — 32-bit accumulator tile independent of
  SBUF, sized for systolic-array output. Use the `accumulate=True`
  path on `nc_matmul` to keep results there.
- **Fused non-matmul reductions** — patterns like
  `T * (2T − Tᵀ) / denom .sum()` have no cuBLAS equivalent. NKI
  lets us express them.

If the answer is "this is the NKI version of the cuBLAS call," the
framing is wrong — rethink what the kernel should be doing.
