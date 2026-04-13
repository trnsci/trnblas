# Installation

## From PyPI

```bash
pip install trnblas
```

## With Neuron hardware support

```bash
pip install trnblas[neuron]
```

This pulls in `neuronxcc` and `torch-neuronx`, which are only needed on
Trainium/Inferentia instances. On CPU or GPU, trnblas falls back to
`torch.matmul` automatically.

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
| `TRNBLAS_REQUIRE_NKI=1` | Re-raise on NKI kernel errors instead of silently falling back to `torch.matmul`. Useful in the validation suite to surface kernel breakage. Unset (default): kernel exceptions fall back to PyTorch. |

## Requirements

- Python ≥ 3.10
- `torch >= 2.1`
- `numpy >= 1.24`
- `neuronxcc >= 2.24` (optional, for on-hardware NKI kernels — pinned to
  the 2.24+ `nisa.nc_matmul` calling convention used across the trnsci
  suite)
- `torch-neuronx >= 2.9` (optional, pulled in by the `[neuron]` extra)
- `pyscf >= 2.4` (optional, pulled in by the `[pyscf]` extra)
