"""NKI kernel dispatch for Trainium BLAS acceleration."""

from .dispatch import HAS_NKI, set_backend, get_backend, nki_gemm, nki_batched_gemm

__all__ = ["HAS_NKI", "set_backend", "get_backend", "nki_gemm", "nki_batched_gemm"]
