"""NKI kernel dispatch for Trainium BLAS acceleration."""

from .dispatch import (
    HAS_NKI,
    set_backend,
    get_backend,
    nki_gemm,
    nki_batched_gemm,
    nki_mp2_energy,
    nki_syrk,
    nki_trsm,
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
