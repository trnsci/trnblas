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

Sum of absolute values `Σ|xᵢ|`, as BLAS `?asum`. For complex `x` this is
`Σ(|Re xᵢ| + |Im xᵢ|)`, as `scasum` / `dzasum` / `cublasScasum`, not the
modulus (since 0.11.0, #75). The result is real.

## `iamax(x)`

0-based index of the first element of largest absolute value, as BLAS
`i?amax` (cuBLAS is 1-based). For complex `x` the magnitude is
`|Re xᵢ| + |Im xᵢ|`, as `icamax` / `izamax` (since 0.11.0, #75).
