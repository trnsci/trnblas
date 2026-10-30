"""NKI kernel dispatch for Trainium BLAS acceleration."""

from .dispatch import (
    HAS_NKI,
    get_autotune,
    get_backend,
    nki_batched_gemm,
    nki_batched_pair_energy,
    nki_fused_gemm_energy,
    nki_gemm,
    nki_metric_inv_half,
    nki_mp2_energy,
    nki_syrk,
    nki_trsm,
    set_autotune,
    set_backend,
)

__all__ = [
    "HAS_NKI",
    "set_backend",
    "get_backend",
    "set_autotune",
    "get_autotune",
    "nki_gemm",
    "nki_batched_gemm",
    "nki_batched_pair_energy",
    "nki_fused_gemm_energy",
    "nki_metric_inv_half",
    "nki_mp2_energy",
    "nki_syrk",
    "nki_trsm",
]
