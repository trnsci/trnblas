# Multi-core: `trnblas.parallel`

One Trn2 chip at LNC=2 exposes four logical NeuronCores. `trnblas.parallel`
splits GEMM and DF-MP2 across them with **one process per core**, launched by
`torchrun` with TorchNeuron's `neuron` process-group backend. Data moves
between cores only through on-chip collectives (#24).

```bash
torchrun --nproc_per_node=4 my_script.py
```

```python
import torch.distributed as dist
import torch_neuronx.distributed.backend  # registers the "neuron" backend

import trnblas.parallel as tp

dist.init_process_group("neuron")
# Host tensors, the same on every rank. With eri_symmetric=True, eri_3c may
# be the packed lower triangle (npair, naux), which is PySCF's _cderi.T.
e = tp.df_mp2_energy(C_occ, C_vir, eri_3c, J, eps_occ, eps_vir, eri_symmetric=True)
```

## Why process per core, and why collectives

These were measured on trn2.3xlarge with SDK 2.33.

| path | rate |
|---|---|
| host → device, 1 or 4 processes at once | ~13 GB/s for the whole chip |
| device → host, `copy_` into a reused or pinned buffer | ~13 GB/s |
| device → host, `.cpu()` or a fresh buffer, > 64 MB | 2.8–3.1 GB/s |
| all-gather, 4 × 147 MB | 5–6 ms, 70–83 GB/s received per rank |
| reduce-scatter, 680 MB per rank | 12 ms |
| broadcast, 588 MB | 10 ms |

- **The host link is the scarce resource.** It does not get faster with more
  processes or threads, so `df_mp2_energy` sends each ERI byte over it once,
  split by μ rows across ranks. With `eri_symmetric=True` it sends only the
  lower triangle, which halves the bytes.
- **Moving data between cores through the host costs 5–10× more than a
  collective.** A `neuron:0 → neuron:k` `.to()` does not move the data, so
  inside one process the host is the only route. That is why the
  single-process split below gains little.

## `df_mp2_energy` dataflow

1. Rank 0 forms X = J^-1/2 while ranks 1..w-1 upload their ERI rows, so the
   metric factor overlaps the upload.
2. Each ERI-holding rank computes its partial half-transform
   C_occ[μ_r]ᵀ (μ_r ν|P). A reduce-scatter over the occupied index sums the
   partials and leaves each rank its own block of occupied rows.
3. X is broadcast. Each rank computes (ia|P) and B = (ia|P) X for its own rows.
4. B is all-gathered. Each rank computes the pair energy of its own rows,
   using `pair_energy_sym` on the native path, and one all-reduce sums the
   result.

Each core holds about 1/w of the ERI. At 768/96/2304 the single-core
`examples/df_mp2.py` only fits if its inputs are uploaded once per process.
Re-uploading them for each repetition ran the logical core out of memory.

## Tensor-parallel GEMM primitives

The operands are local shards on each rank's core. The functions are collective.

| function | sharding | communication |
|---|---|---|
| `gemm_row_parallel(A_rows, B, gather=False)` | A along M | all-gather of C (optional) |
| `gemm_col_parallel(A, B_cols, gather=False)` | B along N | all-gather of C (optional) |
| `gemm_k_parallel(A_k, B_k, scatter=False)` | contraction dim K | all-reduce, or reduce-scatter of rows |
| `all_gather_rows(x)` | — | all-gather along dim 0 |

All of these also run on CPU tensors with the `gloo` backend, which is how
they are unit-tested.

## Single-process split: `set_cores` / `TRNBLAS_CORES`

`trnblas.set_cores(n | "auto" | None)` and `TRNBLAS_CORES=<n>|auto` split
the DF-MP2 pair energy (`nki_batched_pair_energy`) by occupied rows across
`n` visible cores, from one process. B is staged through the host to the
other cores. The setting is opt-in, and unset it changes nothing. See
[Benchmarks](../benchmarks.md#multi-core-on-one-trn2-chip-24) for when it
helps.
