"""
trnblas — BLAS operations for AWS Trainium via NKI.

Provides Level 1 (vector), Level 2 (matrix-vector), and Level 3 (matrix-matrix)
BLAS operations with NKI kernel acceleration on Trainium hardware.
Part of the trnsci scientific computing suite.

Target workloads: DF-MP2 tensor contractions, Fock matrix builds,
Cholesky-based density fitting, and general scientific linear algebra.
"""

__version__ = "0.4.3"

# Level 1 — Vector operations
from .level1 import asum, axpy, dot, iamax, nrm2, scal

# Level 2 — Matrix-vector operations
from .level2 import gemv, ger, symv, trmv

# Level 3 — Matrix-matrix operations
from .level3 import batched_gemm, gemm, symm, syrk, trmm, trsm

# Backend control
from .nki import HAS_NKI, get_backend, set_backend

__all__ = [
    # Level 1
    "axpy",
    "dot",
    "nrm2",
    "scal",
    "asum",
    "iamax",
    # Level 2
    "gemv",
    "symv",
    "trmv",
    "ger",
    # Level 3
    "gemm",
    "batched_gemm",
    "symm",
    "syrk",
    "trsm",
    "trmm",
    # Backend
    "HAS_NKI",
    "set_backend",
    "get_backend",
]
