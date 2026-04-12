"""NKI kernel dispatch for Trainium BLAS acceleration."""

from .dispatch import (
    HAS_NKI,
    set_backend,
    get_backend,
    nki_gemm,
    nki_batched_gemm,
    nki_mp2_energy,
)

__all__ = [
    "HAS_NKI",
    "set_backend",
    "get_backend",
    "nki_gemm",
    "nki_batched_gemm",
    "nki_mp2_energy",
]
