"""Ozaki II prototype (#132) on Trainium (native TorchNeuron path).

Bitwise equality with the host reference of the kernels' arithmetic
(`ozaki2.reference_gemm`, both CRT modes), the residue GEMM at the residue
bound (every PSUM partial sum at its exactness limit), accuracy, and the LNC
check: run at LNC=2 (default) and at LNC=1 (`NEURON_LOGICAL_NC_CONFIG=1`);
both must equal the host reference, and with TRNBLAS_LNC_DIGEST_DIR set the
second run compares its digests with the first run's. Timing is in
`scripts/probe_ozaki2.py`.
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
from trnblas import ozaki2
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
    monkeypatch.setattr(D, "_REQUIRE_NKI", True)


def _k2():
    from trnblas.nki import ozaki2_kernels

    return ozaki2_kernels


SHAPES = [(515, 3000, 241), (300, 8400, 241), (1024, 1024, 1024)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_default_is_fp64_accurate_and_bitwise(require_nki, shape):
    A, B = inputs(*shape)
    C = trnblas.gemm_fp64(A, B, scheme="ozaki2")
    assert C.dtype == torch.float64 and C.device.type == "cpu"
    assert torch.equal(C, ozaki2.reference_gemm(A, B))
    host = max_cw(A @ B, A, B)
    assert max_cw(C, A, B) <= max(2 * host, 1e-15)


@pytest.mark.parametrize("fam,n", [("b7", 13), ("b8", 15), ("b9", 13)])
@pytest.mark.parametrize("crt_mode", ["device", "host", "fused"])
def test_families_bitwise(require_nki, fam, n, crt_mode):
    A, B = inputs(400, 5000, 300, seed=1)
    C = _k2().gemm(A, B, n, fam, crt_mode)
    ref_mode = "host" if crt_mode == "host" else "device"  # fused: the device CRT's bits
    assert torch.equal(C, ozaki2.reference_gemm(A, B, n, fam, ref_mode))


@pytest.mark.parametrize("fam", ["b7", "b8", "b9"])
def test_residue_gemm_at_the_bound(require_nki, fam):
    """Every product at rmax^2 in the first tile: each PSUM group sum sits at
    its exactness limit and the group-close reductions see their largest
    operands."""
    f = ozaki2.FAMILIES[fam]
    K = 2 * f.group_terms + 384
    at = torch.zeros((3, K, 256), dtype=torch.bfloat16)
    b = torch.zeros((3, K, 130), dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(2)
    at.copy_(torch.randint(-f.rmax, f.rmax + 1, at.shape, generator=g).to(torch.bfloat16))
    b.copy_(torch.randint(-f.rmax, f.rmax + 1, b.shape, generator=g).to(torch.bfloat16))
    at[:, :, :128] = f.rmax
    b[:, :, :2] = f.rmax
    b[:, :, 2:4] = -f.rmax
    k2 = _k2()
    out = D._call_kernel(
        k2.GEMM_KERNELS[fam], f"ozaki2_gemm_{fam}", D._to_device(at)[0][0], D._to_device(b)[0][0]
    )
    got = D._to_host(out, "cpu").numpy()
    assert np.array_equal(got, ozaki2.reference_residue_gemm(at, b, fam))


def test_wide_range(require_nki):
    A, B = inputs(256, 3000, 200, dist="logu4", seed=3)
    C = trnblas.gemm_fp64(A, B, scheme="ozaki2", moduli=20)
    assert torch.equal(C, ozaki2.reference_gemm(A, B, 20))
    assert max_cw(C, A, B) <= max(2 * max_cw(A @ B, A, B), 2e-15)


def test_lnc_digest(require_nki):
    """sha256 of the device CRT output for a fixed problem; with
    TRNBLAS_LNC_DIGEST_DIR, written per LNC and compared with the other LNC's."""
    A, B = inputs(1000, 5000, 700, seed=5)
    ra, rb = ozaki2.split_ab(A, B, 16, "b7")
    got = _k2()._launcher("b7", "device")(ra, rb, (0, ra.xp), (0, ra.kp), (0, 700))
    ref = ozaki2.reference_launch("b7", "device")(ra, rb, (0, ra.xp), (0, ra.kp), (0, 700))
    digest = hashlib.sha256(got.numpy().tobytes()).hexdigest()
    assert digest == hashlib.sha256(ref.numpy().tobytes()).hexdigest()
    out = os.environ.get("TRNBLAS_LNC_DIGEST_DIR")
    if not out:
        return
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"ozaki2_lnc{lnc}.sha256").write_text(digest)
    other = Path(out) / f"ozaki2_lnc{'1' if lnc == '2' else '2'}.sha256"
    if other.exists():
        assert other.read_text() == digest
