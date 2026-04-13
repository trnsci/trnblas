# RFC: Fused DF-MP2 pair-energy kernel

**Status:** Design · **Phase:** 3 · **Tracker:** [#15](https://github.com/trnsci/trnblas/issues/15)

This RFC describes the shape of the single most valuable Phase 3
optimization in trnblas: collapsing the DF-MP2 pair-energy reduction
into one NKI kernel. The work is worth an RFC because the design is
Trainium-specific — the four-engine separation and NEFF-cached plan
model enable a pipeline that isn't cleanly expressible on GPU.

## Motivation — the problem in measured numbers

From [#15](https://github.com/trnsci/trnblas/issues/15), timings for
`examples/df_mp2.py` at two problem sizes:

| Shape | Total wall | Energy-step wall | Energy fraction |
|---|---:|---:|---:|
| medium | 9.77 s | 8.46 s | 87% |
| large | 62.84 s | 57.77 s | **92%** |

The dominant cost isn't the upstream GEMMs that produce `T_chunk` —
it's the *reduction* of T into the scalar MP2 energy. At large, the
reduction is 92% of total wall time.

The reduction expression:

```python
e_mp2 += (T * (2.0 * T - T.transpose(-2, -1)) / denom).sum()
```

For a 64512×64512 fp32 T_chunk (16.6 GB), a naive read gives this
a theoretical HBM-bandwidth floor of ~70 ms per chunk at trn1's
~700 GB/s. Measured is **5–6 seconds per chunk**. That's 70× slower
than HBM. The gap is dispatch + kernel-launch overhead per
intermediate tensor.

## Why the current path is slow

The expression as the runtime sees it:

| # | Op | Intermediate shape | HBM traffic |
|---|---|---|---|
| 1 | `2T` | 16.6 GB | read + write |
| 2 | `T.T` | 16.6 GB | read + (maybe) write |
| 3 | `2T - T.T` | 16.6 GB | read 2, write 1 |
| 4 | `T * (...)` | 16.6 GB | read 2, write 1 |
| 5 | `... / denom` | 16.6 GB | read 2, write 1 |
| 6 | `.sum()` | scalar | read, reduce |

Per chunk that's ~50 GB of intermediate HBM traffic, and every op
crosses the XLA boundary — each pays dispatch overhead and waits for
the previous kernel to fully write HBM before reading. None of the
intermediates are needed by anything outside the reduction.

## The fusion target

One NKI kernel takes `(T_chunk, denom, eps_o_pair_slice)` and emits
one scalar. The entire expression runs SBUF-resident:

```
@nki.jit
def df_mp2_energy_kernel(T_chunk, denom, out_scalar):
    for tile in iterate_tiles(T_chunk.shape, tile_shape):
        T_tile    = nl.load(T_chunk[tile])           # SBUF
        denom_tile = nl.load(denom[tile])             # SBUF
        T_T_tile  = nl.transpose_sbuf(T_tile)         # SBUF-local transpose

        # Vector Engine, SBUF-resident through all of these:
        diff = nl.subtract(nl.multiply(T_tile, 2.0), T_T_tile)
        prod = nl.multiply(T_tile, diff)
        quot = nl.divide(prod, denom_tile)

        # Reduce-sum into PSUM accumulator
        tile_sum = nl.sum(quot, axis=(0, 1))          # PSUM
        nl.atomic_add(out_scalar, tile_sum)           # atomic
```

Zero HBM writes for intermediates. One kernel launch per chunk. The
reduction accumulates in PSUM across tiles before touching HBM.

## Engine orchestration

Concrete role for each engine:

| Engine | Role in this kernel |
|---|---|
| **Vector** | All the elementwise work: `2T − T.T`, multiply, divide. Runs on SBUF tiles directly — this is what Vector is for. |
| **Scalar** | Final scalar atomic-add for the chunk. PSUM handles the intra-tile reduction; Scalar ties the chunk's partial sums together. |
| **GpSimd** | Not used. No integer or bitwise work in this expression. |
| **Tensor** | **Deliberately idle during the reduction.** |

**The "Tensor is idle" point is the architectural thesis of this RFC.**
On Trainium, the four engines run concurrently within one kernel
invocation. Whle this fused reduction runs on Vector + Scalar, the
Tensor Engine is free. In the pair-energy loop:

```
for i in occupied:
    for j in occupied:
        T[i,j] = GEMM(B[i], B[j].T)                        # Tensor
        e_mp2 += fused_reduction(T[i,j], denom[i,j])       # Vector + Scalar
```

These two operations for consecutive `(i,j)` pairs **can overlap**:
while pair (0,1)'s reduction runs on Vector, pair (0,2)'s GEMM can
begin on Tensor. That's a two-pair-at-a-time pipeline. On GPU, the
SFU and Tensor Cores share the warp scheduler; you can't orchestrate
this overlap from library code the same way.

## NEFF cache reuse across the pair loop

The pair loop invokes the fused kernel `nocc²` times per chunk. Every
invocation has the same tile shape (determined by `nvir` and the
autotuner-chosen tile size). NEFF caches the compiled kernel after
the first call; subsequent invocations pay only dispatch + kernel
runtime.

Combined with the plan-time tile-shape autotuner ([#26](https://github.com/trnsci/trnblas/issues/26)):
- **First pair:** pay compile cost (~2 s), choose tile shape, cache.
- **Remaining pairs:** ~launch latency + actual compute, ~tens of
  milliseconds.

For medium (nocc=33, so ~1089 pairs per chunk), the amortized compile
is ≤ 2 ms/pair. The dispatch-per-op overhead that dominates today
disappears entirely.

## Why GPU libraries don't ship this

Honest contrast:

- **cuBLAS / cuBLASLt** — no such kernel. Users hand-write a CUDA
  reduction kernel, tune it per GPU generation, ship it alongside.
  The trnblas equivalent ships it as a library primitive.
- **JAX `jit`** — can fuse elementwise chains, but reductions break
  fusion in current XLA/MLIR backends. The chain `a * (b - c.T) / d`
  fuses; wrap it in `.sum()` and XLA typically emits two kernels.
- **PyTorch 2 `torch.compile`** — inductor on GPU can fuse this
  end-to-end; on Trainium via the XLA backend, reduction is a
  kernel boundary today.

The RFC claim isn't "impossible elsewhere" — it's "as a first-class,
library-provided, NEFF-cached primitive, composed into the pair-energy
loop with zero per-pair dispatch tax." That's a design decision, not
just a compiler trick.

## Validation claims

Phase 3 on-hardware testing must prove:

1. **Medium-shape wall ≤ 3 s** (down from 9.77 s — ≥ 3.3× total
   speedup, driven by ≥ 5× on the energy step).
2. **Large-shape wall ≤ 10 s** (down from 62.84 s — ≥ 6.3× total).
3. **Bit-parity vs piecewise reference.** The fused kernel's per-
   chunk output matches the piecewise PyTorch path within
   `atol=1e-4, rtol=1e-4` (reduction order differs, so
   allclose not equal).
4. **NEFF cache reuse.** Second-pair invocation with the same tile
   shape runs in ≤ launch_latency + expected kernel runtime (≤ a
   few ms). Compile cost amortized to zero.
5. **Stream overlap evidence.** Profiler trace shows Tensor-Engine
   time for pair `(i+1, j+1)` GEMM overlapping with Vector-Engine
   time for pair `(i, j)` reduction. Without this evidence, the
   pipeline is nominal, not real.
6. **Vintage-matched GPU baseline.** A10G cuBLAS + hand-fused CUDA
   reduction on the same molecules. Interesting number is per-dollar
   throughput: trn1.2xlarge at $1.34/hr vs A10G at $1.20/hr, end-to-
   end wall time normalized per problem.

## Open questions

- **Tile shape.** For 64512² T-chunks, a `(128, 128)` tile fits all
  three SBUF-resident operands (T, T.T, denom) at ~192 KB. A
  `(128, 512)` tile reduces transpose edge effects but needs ~768 KB
  — still under SBUF's 48 MB but crowds the denom tile. Autotuner
  picks at plan time.
- **SBUF-local transpose.** Does Vector Engine have a cheap
  transpose primitive, or is it stride-2 loads from SBUF? Answer
  depends on NKI 2.24 primitives; worth spiking before committing
  the tile shape.
- **trn1 vs trn2 trade-offs.** Wider PSUM on trn2 enables larger
  tiles, which shifts the transpose / compute ratio. Phase 5
  ([#25](https://github.com/trnsci/trnblas/issues/25)) handles the
  generation-specific path; this RFC targets trn1 first.
- **Relationship to [trntensor #13](https://github.com/trnsci/trntensor/issues/13).**
  trntensor's fused-einsum work wants a similar contraction+divide
  primitive for users writing einsum syntax. Suggestion: trnblas
  owns the GEMM+reduce hot path as a `blas`-level primitive;
  trntensor's einsum frontend detects the DF-MP2 shape and dispatches
  into it. Coordinated via the umbrella
  [`examples/quantum_chemistry/`](https://github.com/trnsci/trnsci/tree/main/examples/quantum_chemistry)
  integration demo.

## Cross-suite alignment

- **Phase 1** ([#21](https://github.com/trnsci/trnblas/issues/21)):
  GEMM + batched_gemm closed. This RFC composes on top of the
  already-validated GEMM primitive; no kernel-level dependency.
- **Phase 3 tile-shape autotuner** ([#26](https://github.com/trnsci/trnblas/issues/26)):
  this kernel benefits directly. Plan-time autotuning picks tile
  shape + feeds it into the NEFF cache key.
- **trnrand Phase 3** ([RFC](https://trnsci.dev/trnrand/design/sbuf_resident_generator/)):
  shares the SBUF-residency + NEFF-cache thesis. Different workloads,
  same architectural pattern.
- **Umbrella [nki_validation_status](https://trnsci.dev/nki_validation_status/)**:
  links to this RFC once accepted.

## References

- [trnsci ROADMAP — Phase 3](https://trnsci.dev/roadmap/#phase-3-single-chip-performance) — suite framing.
- [trnblas #15 — DF-MP2 energy reduction: fuse into a custom NKI kernel](https://github.com/trnsci/trnblas/issues/15) — issue tracker with the original benchmark numbers.
- [trnblas #26 — NKI GEMM tile-shape autotuner](https://github.com/trnsci/trnblas/issues/26) — companion autotuner work; picks the tile shape this kernel uses.
- [trnrand Phase 3 RFC — SBUF-resident streaming Generator](https://trnsci.dev/trnrand/design/sbuf_resident_generator/) — same architectural thesis, different workload.
- DF-MP2 for large molecular systems (>3000 basis functions) — the target workload motivating the problem size.
