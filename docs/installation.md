# Installation

## From PyPI

```bash
pip install trnblas
```

## With Neuron hardware support

```bash
# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnblas[torchneuron]" \
  --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnblas[neuron]" \
  --extra-index-url https://pip.repos.neuron.amazonaws.com
```

`[torchneuron]` pins `torch>=2.14`, `torch-neuronx>=2.14`, and `nki>=0.6`
(torch-neuronx 2.14 requires Python ≥ 3.12). `[neuron]` is the XLA lineage
(`nki>=0.3.0`, `neuronxcc>=2.24`, `torch-neuronx>=2.9`). These are only
needed on Trainium/Inferentia instances. On CPU or GPU, trnblas falls back
to `torch.matmul` automatically. The `nki`, `neuronxcc` and `torch-neuronx`
wheels ship on the AWS pip index rather than PyPI, hence `--extra-index-url`.

### Two Neuron PyTorch lineages

`torch-neuronx` ships in two lineages, and trnblas dispatches its NKI
kernels on whichever one is installed:

| Lineage | Instances | Stack | Kernel dispatch |
|---|---|---|---|
| TorchNeuron (native) | Trn2, Trn3 | Neuron SDK 2.33+, Python ≥ 3.12, `torch` 2.14, `torch-neuronx` 2.14.* | Each kernel is a PyTorch custom op (`torch.ops.trnblas.gemm_ragged`, `syrk`, `j_batched`, ...) registered with `torch_neuronx.nki_op`; tensors live on `device="neuron"`. Works in eager mode and inside `torch.compile(backend="neuron")`. |
| PyTorch/XLA | Trn1 (also Inf2) | `torch-neuronx` 2.9 lineage, `torch_xla` | Unchanged: kernels run on the `torch_xla` device. |

Selection is automatic, on the first kernel launch:

1. **native**: `torch_neuronx` exports `nki_op` and a `neuron` device is
   available (`torch_neuronx.device_count() > 0`). This path never imports
   `torch_xla`.
2. **xla**: `torch_xla` is installed.
3. otherwise there is no device path, and the `nki` backend falls back to
   `torch.matmul` with an `NkiFallbackWarning` (unchanged from earlier
   releases). Use `set_backend("pytorch")` or `TRNBLAS_USE_SIMULATOR=1`.

The `trnblas::` custom ops are registered only when the native path is
selected, so importing trnblas on CPU or on the XLA lineage has no
TorchNeuron side effects. Both lineages publish a package named
`torch-neuronx`, so pick the extra (or Deep Learning AMI Neuron) that
matches the instance.

To force a path, set `TRNSCI_NKI_DISPATCH` before the first launch. The
variable is shared by every trnsci library:

| Value | Effect |
|---|---|
| `auto` (default) | The order above. |
| `native` | Use TorchNeuron; raise `RuntimeError` if no `neuron` device is available. |
| `xla` | Use torch_xla even if TorchNeuron is present; raise `RuntimeError` if torch_xla is missing. |

`TRNSCI_XLA_DEPRECATION=1` makes the XLA path emit one `DeprecationWarning`
per process. It is off by default; the suite turns it on when TorchNeuron
reaches GA (Trn1/Inf2 stay supported on the XLA path in maintenance mode).

### Editable installs and the NKI kernel cache

TorchNeuron enables the NKI trace cache by default
(`NKI_ENABLE_TRACE_CACHE=1`), which persists compiled kernels across
processes. The cache only versions source that belongs to an installed
wheel (a `site-packages` tree with `.dist-info`); an editable install is
not tracked and can get a stale kernel after you edit `dispatch.py`. When
developing with `pip install -e`, turn the cache off before starting
Python (the test suite's `conftest.py` does this by default):

```bash
export NKI_ENABLE_TRACE_CACHE=0
```

See the TorchNeuron
[environment variables](https://github.com/aws-neuron/torch-neuronx/blob/main/docs/environment_variables.md)
reference.

### Simulator-only install (CPU, no hardware)

NKI 0.3.0 ships a CPU simulator (`nki.simulate(kernel)(numpy_args)`)
that runs kernels without a Neuron device. The GH Actions `nki-simulator`
job uses this install line; mirror it locally for fast kernel iteration:

```bash
pip install -e ".[dev]"
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com "nki>=0.3.0"
```

`torch-neuronx` is **not** required for the simulator path — dispatch
routes NumPy directly through `nki.simulate` and bypasses `torch_xla`.

## With PySCF (real-molecule DF-MP2 validation)

```bash
pip install trnblas[pyscf]
```

Pulls in [PySCF](https://pyscf.org/) so the end-to-end DF-MP2 example
(`examples/df_mp2_pyscf.py`) and the correctness test
(`tests/test_df_mp2_pyscf.py`, `@pytest.mark.pyscf`) can run real
molecules and compare against PySCF's own `mp.dfmp2.DFMP2` reference.

## From source

```bash
git clone https://github.com/trnsci/trnblas
cd trnblas
pip install -e ".[dev]"
pytest tests/ -v
```

## Runtime environment variables

| Variable | Effect |
|----------|--------|
| `TRNBLAS_REQUIRE_NKI=1` | Re-raise on NKI kernel errors instead of silently falling back to `torch.matmul`. Useful in the validation suite to surface kernel breakage. Under `set_backend("nki")`, dtypes the kernels do not compute (float64, for example) raise `NotImplementedError` instead of running in torch (v0.9.1). Unset (default): kernel exceptions fall back to PyTorch. |
| `TRNBLAS_USE_SIMULATOR=1` | Route kernel dispatch through `nki.simulate(kernel)(numpy_args)` on CPU. Bypasses `torch_xla` + NEFF compile; used for fast correctness iteration. Requires `nki>=0.3.0`. |
| `TRNSCI_NKI_DISPATCH=auto\|native\|xla` | Suite-wide. Forces the native or XLA dispatch path; `auto` (default) detects. Forcing an unavailable path raises. |
| `TRNSCI_XLA_DEPRECATION=1` | Suite-wide, off by default. The XLA path emits one `DeprecationWarning`. |
| `TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD=<n>` | Overrides the batched-pair full-batch vs chunked routing threshold (default 4096 on XLA, 3072 on native). For re-benchmarking. |
| `TRNBLAS_NATIVE_GEMM=auto\|nki\|aten\|host` | Native path only. `auto` (default) routes `gemm` / `batched_gemm` / `syrk` / `symm` / `trsm` by measured time (#62, #63): device-resident operands run as aten on the `neuron` device; host operands run on the host CPU below a size threshold and as upload + aten + download above it; `trsm` solves with LAPACK on the host. `nki` forces the NKI kernels, `aten` forces aten on the device, `host` keeps host operands on the host. An explicit `set_backend("nki")` always runs the NKI kernels. |
| `TRNBLAS_PAIR_ENERGY=auto\|sym\|aten\|legacy` | Native path, device-resident inputs only. Picks the `nki_batched_pair_energy` implementation: the LNC=2 symmetric kernel, aten (chunked GEMM + reduction), or the full-batch / chunked kernels. `auto` takes `sym` above 3 GFLOP per row launch. |
| `TRNBLAS_METRIC_SOLVE=auto\|host\|device` | Native path. Where `metric_inv_half` factors the metric: `auto` (default) uses float64 LAPACK on the host for J on the `neuron` device with naux ≤ 1024; above that, and for host tensors, `torch.linalg.cholesky` + `trsm` (v0.8.0). |
| `TRNBLAS_CORES=<n>\|auto` | Opt-in single-process split of the DF-MP2 pair energy across `n` visible cores (same as `trnblas.set_cores`). Unset: one core. See [Multi-core](api/parallel.md). |
| `TRNBLAS_PINNED_OUTPUT=1\|0` | Native path only. `1` (default): host results of 1 MiB or more from the device (`gemm`, `batched_gemm`, `syrk`, `trsm` and the other host-input wrappers) are copied into a fresh tensor from the accelerator's caching host allocator (`pin_memory=True`). That is about 3.5x faster than `.to("cpu")`, which page-faults fresh memory. Freed blocks stay cached, rounded up to a power of two, until `torch.accelerator.empty_host_cache()`. `0` returns plain `.to("cpu")` results. |
| `NKI_ENABLE_TRACE_CACHE=0` | TorchNeuron (not trnblas) setting. Turns off the cross-process NKI kernel cache, which cannot version editable installs. |

## Requirements

- Python ≥ 3.10
- `torch >= 2.1`
- `numpy >= 1.24`
- `neuronxcc >= 2.24` (optional, for on-hardware NKI kernels — pinned to
  the 2.24+ `nisa.nc_matmul` calling convention used across the trnsci
  suite)
- `torch-neuronx >= 2.9` (optional, pulled in by the `[neuron]` extra), or
  `torch-neuronx >= 2.14` on Python ≥ 3.12 for native dispatch (the
  `[torchneuron]` extra)
- `pyscf >= 2.4` (optional, pulled in by the `[pyscf]` extra)
