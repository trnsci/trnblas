# Quickstart

```python
import torch
import trnblas

A = torch.randn(256, 128)
B = torch.randn(128, 64)

# Level 3 — Matrix multiply (the hot path)
C = trnblas.gemm(alpha=1.0, A=A, B=B)

# Batched GEMM (DF-MP2 tensor contractions)
A_batch = torch.randn(32, 128, 64)
B_batch = torch.randn(32, 64, 32)
C_batch = trnblas.batched_gemm(1.0, A_batch, B_batch)

# Triangular solve (Cholesky-based density fitting)
L = torch.linalg.cholesky(A.T @ A + torch.eye(128))
B2 = torch.randn(128, 16)
X = trnblas.trsm(1.0, L, B2, uplo="lower")

# Symmetric rank-k update (metric construction)
J = trnblas.syrk(1.0, A, trans=True)

# Level 2 — Matrix-vector
x = torch.randn(128)
y = trnblas.gemv(1.0, A, x)

# Level 1 — Vector operations
u = torch.randn(256)
v = torch.randn(256)
w = trnblas.axpy(2.0, u, v)
d = trnblas.dot(u, v)
n = trnblas.nrm2(u)
```

## Backend selection

```python
import trnblas

trnblas.set_backend("auto")     # NKI on Trainium, PyTorch elsewhere (default)
trnblas.set_backend("pytorch")  # force PyTorch
trnblas.set_backend("nki")      # force NKI (requires Neuron hardware)
```

## DF-MP2 example

```bash
python examples/df_mp2.py --demo
python examples/df_mp2.py --nbasis 100 --nocc 20
```
