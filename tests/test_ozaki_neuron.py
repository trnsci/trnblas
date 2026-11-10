"""fp64-class Ozaki GEMM (#22) on Trainium (native TorchNeuron path).

Accuracy against an independent fp64 reference, bitwise equality with the
host reference of the kernel arithmetic (`ozaki.reference_gemm`), and the
LNC check: every element is a fixed sequence of exact integer group sums and
fp32 TwoSums, so the result must not depend on the LNC split. Run the suite
at LNC=2 (default) and at LNC=1 (`NEURON_LOGICAL_NC_CONFIG=1`); both runs
must equal the host reference bit for bit, and with TRNBLAS_LNC_DIGEST_DIR
set the second run also compares its digests with the first run's directly.
The device split (#123) is checked the same way: its digits are those of
`ozaki.split`, bit for bit, at any LNC.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
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
    hilo = D._ozaki_launcher()(da, db, (0, da.xp), (0, da.kp), (0, db.xp))
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


def _digest_check(name: str, digest: str) -> None:
    out = os.environ.get("TRNBLAS_LNC_DIGEST_DIR")
    if not out:
        return
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"{name}_lnc{lnc}.sha256").write_text(digest)
    other = Path(out) / f"{name}_lnc{'1' if lnc == '2' else '2'}.sha256"
    if other.exists():
        assert other.read_text() == digest


def _split_inputs(kind, shape, per, seed):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.standard_normal(shape)
    if kind == "wide":
        return rng.standard_normal(shape) * 10.0 ** rng.uniform(-30, 30, shape)
    if kind == "ties":  # half-way cases on the k = 8 digit grid of a row max ~ 1
        T = rng.integers(-(2**20), 2**20, shape).astype(np.float64)
        T += rng.choice([0.0, 0.5, -0.5, 0.25], shape)
        T *= 2.0**-63
        T[:, 0] = T[0, :] = 1.0 - 2.0**-52
        return T
    bits = rng.integers(0, 2**63, shape, dtype=np.uint64)  # every exponent and sign
    sub = (slice(None), slice(1, None, 3)) if per == "row" else (slice(1, None, 3), slice(None))
    bits[sub] &= ~np.uint64(0x7FF0000000000000)  # subnormal / zero elements
    X = bits.view(np.float64).copy()
    X[~np.isfinite(X)] = 1.0
    X[::2] *= -1
    return X


@pytest.mark.parametrize(
    "kind,shape,k",
    [
        ("normal", (515, 3000), 8),
        ("wide", (300, 700), 3),
        ("bits", (256, 1100), 12),
        ("ties", (200, 640), 8),
    ],
)
@pytest.mark.parametrize("per", ["row", "col"])
def test_device_split_digits_are_the_host_split(require_nki, kind, shape, k, per):
    """The device split (#123) gives `ozaki.split`'s digits and exponents bit
    for bit, per row and per column, at any LNC."""
    X = torch.from_numpy(_split_inputs(kind, shape, per, seed=k))
    d, e = D._oz_device_split(X, k, per)
    ref, e_ref = ozaki.split(X, k, per=per, pad=(ozaki.TILE_K, ozaki.TILE_K))
    got = d.cpu()
    assert np.array_equal(got.float().numpy(), ref.float().numpy())
    assert np.array_equal(e, e_ref)
    _digest_check(
        f"ozaki_split_{kind}_{per}", hashlib.sha256(got.view(torch.int16).numpy()).hexdigest()
    )


def test_device_split_falls_back(require_nki):
    A, B = inputs(130, 300, 70, seed=7)
    A[4] = 1e-310  # a subnormal row maximum: the host splits A
    assert torch.equal(trnblas.gemm_fp64(A, B, slices=5), ozaki.reference_gemm(A, B, 5))
    B[3, 3] = float("inf")  # inf / nan: host fp64
    C = trnblas.gemm_fp64(A, B)
    assert torch.equal(C.isnan(), (A @ B).isnan())


def test_device_resident_operands(require_nki):
    A, B = inputs(200, 600, 150, seed=6)
    try:
        Ad, Bd = A.to("neuron"), B.to("neuron")
    except Exception as exc:  # pragma: no cover - fp64 storage on the device
        pytest.skip(f"no float64 device tensors: {exc}")
    C = trnblas.gemm_fp64(Ad, Bd)
    assert C.device.type == "neuron" and C.dtype == torch.float64
    assert torch.equal(C.cpu(), ozaki.reference_gemm(A, B, 8))
