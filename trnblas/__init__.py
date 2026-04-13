"""
trnblas — BLAS operations for AWS Trainium via NKI.

Provides Level 1 (vector), Level 2 (matrix-vector), and Level 3 (matrix-matrix)
BLAS operations with NKI kernel acceleration on Trainium hardware.
Part of the trnsci scientific computing suite.

Target workloads: DF-MP2 tensor contractions, Fock matrix builds,
Cholesky-based density fitting, and general scientific linear algebra.
"""

__version__ = "0.4.2"

# Level 1 — Vector operations
from .level1 import axpy, dot, nrm2, scal, asum, iamax

# Level 2 — Matrix-vector operations
from .level2 import gemv, symv, trmv, ger

# Level 3 — Matrix-matrix operations
from .level3 import gemm, batched_gemm, symm, syrk, trsm, trmm

# Backend control
from .nki import HAS_NKI, set_backend, get_backend

__all__ = [
    # Level 1
    "axpy", "dot", "nrm2", "scal", "asum", "iamax",
    # Level 2
    "gemv", "symv", "trmv", "ger",
    # Level 3
    "gemm", "batched_gemm", "symm", "syrk", "trsm", "trmm",
    # Backend
    "HAS_NKI", "set_backend", "get_backend",
]
