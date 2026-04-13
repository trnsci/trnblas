"""Anti-regression test: ensure NKI dispatch actually runs on Trainium,
not silently falling back to torch.matmul.

Motivated by the v0.4.x-era silent fallback (fix in v0.4.3): the SSM
runner invoked the venv's python without putting its bin dir on $PATH,
so `torch_neuronx.initializer` couldn't resolve `libneuronpjrt-path`,
every NKI dispatch hit the fallback, and the "trn1 perf" story in the
benchmarks page was actually CPU torch.matmul.

This test forces `TRNBLAS_REQUIRE_NKI=1` semantics and asserts that a
GEMM call completes without falling back. It would have failed
loudly during v0.4.0 if it had existed then.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.neuron


def test_nki_dispatches_without_fallback():
    """A single GEMM must reach the NKI kernel, not the torch fallback."""
    # Set the env var BEFORE importing dispatch so _REQUIRE_NKI picks it up.
    os.environ["TRNBLAS_REQUIRE_NKI"] = "1"

    # Re-import so the module-level _REQUIRE_NKI is re-evaluated.
    import importlib

    import trnblas.nki.dispatch as dispatch_mod

    importlib.reload(dispatch_mod)
    assert dispatch_mod._REQUIRE_NKI, (
        "TRNBLAS_REQUIRE_NKI didn't take effect after reload — test setup bug"
    )

    import trnblas

    trnblas.set_backend("nki")

    # If the NKI dispatch silently falls back today, this call raises
    # the underlying exception (PATH / plugin / kernel).
    A = torch.randn(128, 128)
    B = torch.randn(128, 128)
    out = trnblas.gemm(1.0, A, B)

    torch.testing.assert_close(out, A @ B, atol=1e-3, rtol=1e-4)
