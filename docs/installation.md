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

## From source

```bash
git clone https://github.com/scttfrdmn/trnblas
cd trnblas
pip install -e ".[dev]"
pytest tests/ -v
```

## Requirements

- Python ≥ 3.10
- `torch >= 2.1`
- `numpy >= 1.24`
- `neuronxcc >= 2.15` (optional, for on-hardware NKI kernels)
