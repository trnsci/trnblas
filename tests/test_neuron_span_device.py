"""On-device regression tests for #112 (aws-neuron-sdk#1457) at the real sizes.

A device matmul is silently wrong once an operand's (or the result's) 128 DMA
partitions span 4 GiB or more: fp32 rows of 32 MiB or more. These tests run
the public entry points at N = 8,389,632 (32 MiB + 4 KiB rows) against a host
fp32 reference. B is uploaded above a same-size decoy that holds different
data (the setup that exposes the bug reliably), or produced on the device by
`clone`. The control at N = 8,388,480 (32 MiB - 512 B) checks that the
products below the rule are still exactly the plain aten call.

Each case needs about 9-14 GB of one logical core's device memory and about
10 GB of host memory; run one test process per core.
"""

import gc

import pytest
import torch

import trnblas
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

DEV = "neuron"
N_BAD = 8_389_632  # 32 MiB + 4 KiB fp32 rows
N_OK = 8_388_480  # 32 MiB - 512 B


def _host(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _rel(out, ref):
    """||out - ref|| / ||ref|| in float64, or complex128 when either is complex
    (`.double()` would drop the imaginary part, #113)."""
    wide = torch.complex128 if (out.is_complex() or ref.is_complex()) else torch.float64
    out, ref = out.cpu().to(wide), ref.cpu().to(wide)
    return ((out - ref).norm() / ref.norm()).item()


@pytest.fixture(autouse=True)
def _free():
    yield
    gc.collect()


@pytest.fixture(params=["upload", "device"])
def source(request):
    return request.param


def _device_operand(X: torch.Tensor, source: str):
    """(decoy, X on the device): uploaded above a decoy, or a device `clone`."""
    decoy = (-X).to(DEV)  # different data directly below
    if source == "upload":
        return decoy, X.to(DEV)
    return decoy, X.to(DEV).clone()


class TestSpan4GiB:
    def test_gemm_long_b_rows(self, source):
        A, B = _host(96, 128), _host(128, N_BAD, seed=1)
        decoy, Bd = _device_operand(B, source)
        out = trnblas.gemm(1.0, A.to(DEV), Bd)
        assert out.device.type == "neuron"
        assert _rel(out, A @ B) < 1e-5

    def test_gemm_transposed_lhs_lnc_shape(self, source):
        # M = 64, K = 128 transposed view: the `auto` LNC-kernel region (#79).
        At, B = _host(128, 64), _host(128, N_BAD, seed=1)
        decoy, Bd = _device_operand(B, source)
        out = trnblas.gemm(1.0, At.to(DEV), Bd, transA=True)
        assert _rel(out, At.T @ B) < 1e-5

    def test_gemm_long_a_rows(self, source):
        A, B = _host(128, N_BAD), _host(N_BAD, 64, seed=1)
        decoy, Ad = _device_operand(A, source)
        out = trnblas.gemm(1.0, Ad, B.to(DEV))
        assert _rel(out, A @ B) < 1e-5

    def test_gemm_long_result_rows(self):
        # (128, 128) @ (128, N): the result itself has the bad layout.
        A, B = _host(128, 128), _host(128, N_BAD, seed=1)
        out = trnblas.gemm(1.0, A.to(DEV), B.to(DEV))
        assert out.shape == (128, N_BAD)
        assert _rel(out, A @ B) < 1e-5

    def test_batched_gemm(self, source):
        A, B = _host(1, 96, 128), _host(1, 128, N_BAD, seed=1)
        decoy, Bd = _device_operand(B, source)
        out = trnblas.batched_gemm(1.0, A.to(DEV), Bd)
        assert _rel(out, A @ B) < 1e-5

    def test_syrk_long_rows(self, source):
        A = _host(128, N_BAD)
        decoy, Ad = _device_operand(A, source)
        out = trnblas.syrk(1.0, Ad)
        assert _rel(out, A @ A.T) < 1e-5

    def test_gemm_host_operands(self):
        # Host in, host out: the aten route uploads, blocks, and downloads.
        A, B = _host(96, 128), _host(128, N_BAD, seed=1)
        out = trnblas.gemm(1.0, A, B)
        assert out.device.type == "cpu"
        assert _rel(out, A @ B) < 1e-5

    def test_below_rule_is_the_plain_call(self):
        A, B = _host(96, 128), _host(128, N_OK, seed=1)
        decoy, Bd = _device_operand(B, "upload")
        Ad = A.to(DEV)
        assert not D._neuron_mm_unsafe(Ad, Bd)
        out = trnblas.gemm(1.0, Ad, Bd)
        assert torch.equal(out.cpu(), torch.matmul(Ad, Bd).cpu())
        assert _rel(out, A @ B) < 1e-5
