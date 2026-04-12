# Level 1 — Vector operations

Level 1 BLAS: `O(n)` vector operations. These run on PyTorch (CPU/GPU/Neuron
via XLA); the Tensor Engine is wasted on vector work so no NKI kernels.

## `axpy(alpha, x, y)`

`y = α·x + y`. Returns the updated `y`.

## `dot(x, y)`

Scalar inner product `x^T y`.

## `nrm2(x)`

Euclidean norm `‖x‖₂`.

## `scal(alpha, x)`

In-place scaling `x = α·x`.

## `asum(x)`

Sum of absolute values `Σ|xᵢ|`.

## `iamax(x)`

Index of element with largest absolute value.
