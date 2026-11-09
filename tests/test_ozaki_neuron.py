"""fp64-class Ozaki GEMM (#22) on Trainium (native TorchNeuron path).

Accuracy against an independent fp64 reference, bitwise equality with the
host reference of the kernel arithmetic (`ozaki.reference_gemm`), and the
LNC check: every element is a fixed sequence of exact integer group sums and
fp32 TwoSums, so the result must not depend on the LNC split. Run the suite
at LNC=2 (default) and at LNC=1 (`NEURON_LOGICAL_NC_CONFIG=1`); both runs
must equal the host reference bit for bit, and with TRNBLAS_LNC_DIGEST_DIR
set the second run also compares its digests with the first run's directly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch
from _ozaki_ref import inputs, max_cw

import trnblas
from trnblas import ozaki
from trnblas.nki import HAS_NKI
from trnblas.nki import dispatch as D


def _native() -> bool:
    if not HAS_NKI or D._use_simulator():
        return False
    try:
        return D._get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]


@pytest.fixture
def require_nki(monkeypatch):
    """Kernel errors re-raise (no silent host fallback)."""
    monkeypatch.setattr(D, "_REQUIRE_NKI", True)


# (M, K, N): an odd M-tile count with an N tail and K groups spanning pairs;
# a DF-MP2-like narrow product with K over two 4096-term groups; a square.
SHAPES = [(515, 3000, 241), (300, 8400, 241), (1024, 1024, 1024)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_k8_is_fp64_accurate(require_nki, shape):
    A, B = inputs(*shape)
    C = trnblas.gemm_fp64(A, B)
    assert C.dtype == torch.float64 and C.device.type == "cpu"
    host = max_cw(A @ B, A, B)
    err = max_cw(C, A, B)
    assert err <= 1e-15, (err, host)  # measured 2e-16 to 4e-16 (host fp64 1e-16 to 3e-16)


@pytest.mark.parametrize("k,tol", [(6, 2e-11), (7, 2e-13), (10, 1e-15)])
def test_slices_set_the_accuracy(require_nki, k, tol):
    A, B = inputs(400, 2000, 300, seed=1)
    assert max_cw(trnblas.gemm_fp64(A, B, slices=k), A, B) <= tol


def test_wide_range_data_at_k10(require_nki):
    A, B = inputs(256, 3000, 200, dist="logu4", seed=2)
    host = max_cw(A @ B, A, B)
    assert max_cw(trnblas.gemm_fp64(A, B, slices=10), A, B) <= max(4 * host, 4e-15)


@pytest.mark.parametrize(
    "shape,k", [((515, 3000, 241), 8), ((256, 1000, 1000), 6), ((130, 128, 1), 3)]
)
def test_bitwise_equals_host_reference(require_nki, shape, k):
    """At any LNC: the device result is the host reference, bit for bit."""
    A, B = inputs(*shape, seed=3)
    assert torch.equal(trnblas.gemm_fp64(A, B, slices=k), ozaki.reference_gemm(A, B, k))


def test_transposed_operands(require_nki):
    A, B = inputs(300, 700, 260, seed=4)
    ref = ozaki.reference_gemm(A, B, 8)
    At, Bt = A.T.contiguous(), B.T.contiguous()
    assert torch.equal(trnblas.gemm_fp64(At, B, transA=True), ref)
    assert torch.equal(trnblas.gemm_fp64(A, Bt, transB=True), ref)


def test_lnc_digest(require_nki):
    """sha256 of the kernel's (hi, lo) for a fixed problem; with
    TRNBLAS_LNC_DIGEST_DIR, written per LNC and compared with the other LNC's."""
    A, B = inputs(1000, 5000, 700, seed=5)
    da, db = ozaki.split_a(A, 8), ozaki.split_b(B, 8)
    hilo = D._ozaki_launch(da.d, da.kmajor, db.d, db.kmajor)
    digest = hashlib.sha256(hilo.numpy().tobytes()).hexdigest()
    ref = ozaki.reference_from_digits(ozaki.kmajor(da.d, da.kmajor), ozaki.kmajor(db.d, db.kmajor))
    assert digest == hashlib.sha256(ref.tobytes()).hexdigest()
    out = os.environ.get("TRNBLAS_LNC_DIGEST_DIR")
    if not out:
        return
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"ozaki_lnc{lnc}.sha256").write_text(digest)
    other = Path(out) / f"ozaki_lnc{'1' if lnc == '2' else '2'}.sha256"
    if other.exists():
        assert other.read_text() == digest


def test_device_resident_operands(require_nki):
    A, B = inputs(200, 600, 150, seed=6)
    try:
        Ad, Bd = A.to("neuron"), B.to("neuron")
    except Exception as exc:  # pragma: no cover - fp64 storage on the device
        pytest.skip(f"no float64 device tensors: {exc}")
    C = trnblas.gemm_fp64(Ad, Bd)
    assert C.device.type == "neuron" and C.dtype == torch.float64
    assert torch.equal(C.cpu(), ozaki.reference_gemm(A, B, 8))
