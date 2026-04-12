# NKI Backend

The NKI dispatch layer controls whether BLAS operations run on the native
Trainium Tensor Engine or fall back to PyTorch.

## Backend selection

```python
import trnblas

trnblas.set_backend("auto")     # NKI on Trainium, PyTorch elsewhere (default)
trnblas.set_backend("pytorch")  # force PyTorch fallback
trnblas.set_backend("nki")      # force NKI (requires neuronxcc)
```

`trnblas.HAS_NKI` is `True` when `neuronxcc` is importable.

## GEMM kernel

The NKI GEMM kernel lives in `trnblas/nki/dispatch.py`. It uses stationary
tile reuse:

- A tile (128×128) loaded once to SBUF, held stationary in the systolic array.
- B tiles streamed through as the moving operand.
- Partial products accumulated in PSUM.

**Status:** scaffolded but not yet validated on trn1/trn2 hardware. Falls
back to `torch.matmul` until the kernel ships. See the roadmap issues for
on-hardware validation work.
