"""NKI kernel dispatch for Trainium BLAS acceleration."""

from .dispatch import (
    HAS_NKI,
    get_backend,
    nki_batched_gemm,
    nki_gemm,
    nki_mp2_energy,
    nki_syrk,
    nki_trsm,
    set_backend,
)

__all__ = [
    "HAS_NKI",
    "set_backend",
    "get_backend",
    "nki_gemm",
    "nki_batched_gemm",
    "nki_mp2_energy",
    "nki_syrk",
    "nki_trsm",
]
