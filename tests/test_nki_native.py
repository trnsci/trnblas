"""
Native TorchNeuron dispatch on hardware (Trn2/Trn3, torch-neuronx 2.14+).

Skipped unless the dispatch path resolves to "native": these do not run on
the XLA lineage (Trn1) or off-hardware. The existing kernel suites
(test_nki_gemm, test_nki_syrk, test_nki_mp2_energy, ...) also exercise the
native path automatically on a TorchNeuron host; this file covers what is
specific to it: custom-op registration, no torch_xla, eager op calls,
torch.compile, and the path-aware batched-pair routing.

Inputs are created on the host and copied to the device: on-device
`torch.randn` is shape-sensitive (aws-neuron/aws-neuron-sdk#1439) and would
skew any timing or first-call behaviour.
"""

import sys

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


@pytest.fixture
def require_nki(monkeypatch):
    """nki backend with kernel errors re-raised (no silent torch fallback)."""
    old = trnblas.get_backend()
    trnblas.set_backend("nki")
    monkeypatch.setattr(D, "_REQUIRE_NKI", True)
    yield
    trnblas.set_backend(old)


def _host(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _rel(out, ref):
    return ((out.double() - ref.double()).norm() / ref.double().norm()).item()


def _pair_reference(B, eps_occ, eps_vir):
    return float(D._torch_batched_pair_energy(B.double(), eps_occ.double(), eps_vir.double()))


class TestNativeRegistration:
    def test_launch_does_not_import_torch_xla(self, require_nki):
        trnblas.gemm(1.0, _host(128, 128), _host(128, 128, seed=1))
        assert D._get_dispatch_path() == "native"
        assert "torch_xla" not in sys.modules

    def test_ops_registered_in_torch_ops(self):
        ops = D._get_native_ops()
        for name in ops:
            assert hasattr(torch.ops.trnblas, name), name


class TestNativeOpsEager:
    @pytest.mark.parametrize("M,K,N", [(127, 129, 513), (1000, 1000, 1000), (1, 1, 1)])
    def test_gemm_ragged_op(self, M, K, N):
        A, B = _host(M, K), _host(K, N, seed=1)
        out = torch.ops.trnblas.gemm_ragged(A.to(DEV), B.to(DEV))
        assert out.device.type == "neuron"
        assert _rel(out.cpu(), A.double() @ B.double()) < 1e-5

    def test_gemm_default_tile_op(self):
        A, B = _host(256, 256), _host(256, 1024, seed=1)
        out = torch.ops.trnblas.gemm_128_128_512(A.to(DEV), B.to(DEV)).cpu()
        assert _rel(out, A.double() @ B.double()) < 1e-5

    def test_syrk_op(self):
        A = _host(256, 128)
        out = torch.ops.trnblas.syrk(A.to(DEV)).cpu()
        assert _rel(out, A.double() @ A.double().T) < 1e-5

    def test_public_gemm_on_resident_inputs_stays_on_device(self, require_nki):
        A, B = _host(300, 200), _host(200, 600, seed=1)
        C = trnblas.gemm(1.0, A.to(DEV), B.to(DEV))
        assert C.device.type == "neuron"
        assert _rel(C.cpu(), A.double() @ B.double()) < 1e-5


class TestNativeTorchCompile:
    def test_gemm_ragged_inside_compile(self):
        gemm_ragged = D._get_native_ops()["gemm_ragged"]

        @torch.compile(backend="neuron", fullgraph=True)
        def scaled(a, b):
            return gemm_ragged(a, b) * 2.0

        A, B = _host(127, 129), _host(129, 513, seed=1)
        out = scaled(A.to(DEV), B.to(DEV)).cpu()
        assert _rel(out, 2.0 * (A.double() @ B.double())) < 1e-5

    def test_syrk_inside_compile(self):
        syrk = D._get_native_ops()["syrk"]

        @torch.compile(backend="neuron", fullgraph=True)
        def f(a):
            return syrk(a) + 1.0

        A = _host(256, 128)
        out = f(A.to(DEV)).cpu()
        assert _rel(out, A.double() @ A.double().T + 1.0) < 1e-5


class TestNativeEnergyKernels:
    def test_batched_pair_routes_chunked_and_matches(self, require_nki, monkeypatch):
        # Forced chunked: the default would take full-batch at this small shape.
        monkeypatch.setenv("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", "0")
        nocc, nvir, naux = 4, 100, 200
        B = 0.1 * _host(nocc, nvir, naux)
        eps_occ = -1.0 - torch.rand(nocc, generator=torch.Generator().manual_seed(2))
        eps_vir = 0.5 + torch.rand(nvir, generator=torch.Generator().manual_seed(3))
        called = []
        real = D._nki_batched_pair_energy_chunked_impl
        monkeypatch.setattr(
            D,
            "_nki_batched_pair_energy_chunked_impl",
            lambda *a: called.append(1) or real(*a),
        )
        e = float(trnblas.nki.nki_batched_pair_energy(B, eps_occ, eps_vir))
        assert called == [1]
        assert e == pytest.approx(_pair_reference(B, eps_occ, eps_vir), rel=1e-4)

    def test_batched_pair_default_takes_full_batch(self, require_nki, monkeypatch):
        assert D._batched_pair_chunk_threshold() == 3072
        monkeypatch.setattr(
            D,
            "_nki_batched_pair_energy_chunked_impl",
            lambda *a: pytest.fail("est_iters 4 must take the full-batch kernel"),
        )
        nocc, nvir, naux = 2, 128, 128
        B = 0.1 * _host(nocc, nvir, naux)
        eps_occ = torch.tensor([-1.2, -1.0])
        eps_vir = 0.5 + torch.rand(nvir, generator=torch.Generator().manual_seed(3))
        e = float(trnblas.nki.nki_batched_pair_energy(B, eps_occ, eps_vir))
        assert e == pytest.approx(_pair_reference(B, eps_occ, eps_vir), rel=1e-4)

    def test_fused_gemm_energy(self, require_nki):
        nvir, naux = 100, 200
        bi, bj = 0.1 * _host(nvir, naux), 0.1 * _host(nvir, naux, seed=1)
        eps_vir = 0.5 + torch.rand(nvir, generator=torch.Generator().manual_seed(3))
        e = float(trnblas.nki.nki_fused_gemm_energy(bi, bj, -1.1, -1.0, eps_vir))
        ref = float(
            D._torch_fused_pair_energy(bi.double(), bj.double(), -1.1, -1.0, eps_vir.double())
        )
        assert e == pytest.approx(ref, rel=1e-4)

    def test_mp2_energy(self, require_nki):
        ic, nocc, nvir = 2, 3, 128
        T = 0.1 * _host(ic * nvir, nocc * nvir)
        eo_c = torch.tensor([-1.2, -1.1])
        eo_f = torch.tensor([-1.2, -1.1, -1.0])
        ev = 0.5 + torch.rand(nvir, generator=torch.Generator().manual_seed(3))
        e = float(trnblas.nki.nki_mp2_energy(T, eo_c, eo_f, ev))
        ref = float(D._torch_mp2_energy(T.double(), eo_c.double(), eo_f.double(), ev.double()))
        assert e == pytest.approx(ref, rel=1e-4)


class TestNativeDevicePolicy:
    """Device-resident operands: GEMM-family → aten, large pair energy → sym kernel."""

    def test_resident_gemm_skips_nki_kernel(self, require_nki, monkeypatch):
        monkeypatch.delenv("TRNBLAS_NATIVE_GEMM", raising=False)
        monkeypatch.setattr(D, "_backend", "auto")
        monkeypatch.setattr(D, "_nki_gemm_impl", lambda *a: pytest.fail("NKI GEMM launched"))
        A, B = _host(300, 200), _host(200, 600, seed=1)
        C = trnblas.gemm(1.0, A.to(DEV), B.to(DEV))
        assert C.device.type == "neuron"
        assert _rel(C.cpu(), A.double() @ B.double()) < 1e-5

    def test_resident_gemm_env_nki_runs_kernel(self, require_nki, monkeypatch):
        monkeypatch.setenv("TRNBLAS_NATIVE_GEMM", "nki")
        called = []
        real = D._nki_gemm_impl
        monkeypatch.setattr(D, "_nki_gemm_impl", lambda *a: called.append(1) or real(*a))
        A, B = _host(300, 200), _host(200, 600, seed=1)
        C = trnblas.gemm(1.0, A.to(DEV), B.to(DEV))
        assert called == [1]
        assert _rel(C.cpu(), A.double() @ B.double()) < 1e-5

    @pytest.mark.parametrize(
        "nocc,nvir,naux", [(1, 19, 71), (2, 128, 128), (5, 19, 71), (4, 130, 129), (7, 256, 384)]
    )
    def test_pair_energy_sym_kernel(self, require_nki, monkeypatch, nocc, nvir, naux):
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "sym")
        B = 0.1 * _host(nocc, nvir, naux)
        eps_occ = -1.0 - torch.rand(nocc, generator=torch.Generator().manual_seed(2))
        eps_vir = 0.5 + torch.rand(nvir, generator=torch.Generator().manual_seed(3))
        called = []
        real = D._nki_pair_energy_sym_impl
        monkeypatch.setattr(D, "_nki_pair_energy_sym_impl", lambda *a: called.append(1) or real(*a))
        e = trnblas.nki.nki_batched_pair_energy(B.to(DEV), eps_occ.to(DEV), eps_vir.to(DEV))
        assert called == [1]
        assert float(e) == pytest.approx(_pair_reference(B, eps_occ, eps_vir), rel=1e-5)

    def test_pair_energy_small_resident_goes_to_aten(self, require_nki, monkeypatch):
        monkeypatch.delenv("TRNBLAS_PAIR_ENERGY", raising=False)
        monkeypatch.setattr(
            D, "_nki_pair_energy_sym_impl", lambda *a: pytest.fail("sym kernel below crossover")
        )
        nocc, nvir, naux = 5, 19, 71
        B = 0.1 * _host(nocc, nvir, naux)
        eps_occ = -1.0 - torch.rand(nocc, generator=torch.Generator().manual_seed(2))
        eps_vir = 0.5 + torch.rand(nvir, generator=torch.Generator().manual_seed(3))
        e = trnblas.nki.nki_batched_pair_energy(B.to(DEV), eps_occ.to(DEV), eps_vir.to(DEV))
        assert float(e) == pytest.approx(_pair_reference(B, eps_occ, eps_vir), rel=1e-5)


def _visible_cores() -> int:
    try:
        import torch_neuronx

        return int(torch_neuronx.device_count())
    except Exception:
        return 0


class TestNativeMultiCorePairEnergy:
    """The #24 row split; needs NEURON_RT_VISIBLE_CORES covering 2+ logical cores."""

    @pytest.fixture(autouse=True)
    def _cores(self):
        if _visible_cores() < 2:
            pytest.skip("needs 2+ visible logical NeuronCores")
        yield
        trnblas.set_cores(None)

    @pytest.mark.parametrize("route", ["sym", "aten"])
    @pytest.mark.parametrize("nocc,nvir,naux", [(5, 19, 71), (9, 130, 129), (16, 256, 384)])
    def test_split_matches_single_core(self, require_nki, monkeypatch, route, nocc, nvir, naux):
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", route)
        B = 0.1 * _host(nocc, nvir, naux)
        eps_occ = -1.0 - torch.rand(nocc, generator=torch.Generator().manual_seed(2))
        eps_vir = 0.5 + torch.rand(nvir, generator=torch.Generator().manual_seed(3))
        args = (B.to(DEV), eps_occ.to(DEV), eps_vir.to(DEV))
        ref = _pair_reference(B, eps_occ, eps_vir)
        for k in sorted({2, min(4, _visible_cores())}):
            trnblas.set_cores(k)
            e = trnblas.nki.nki_batched_pair_energy(*args)
            assert e.device == args[0].device
            assert float(e) == pytest.approx(ref, rel=1e-5)


class TestNativeAutoRouting:
    """`auto` on host inputs: small → host CPU, large → upload + aten (#62, #63);
    an explicit `set_backend("nki")` still runs the kernel."""

    @pytest.fixture
    def auto(self, require_nki, monkeypatch):
        monkeypatch.delenv("TRNBLAS_NATIVE_GEMM", raising=False)
        monkeypatch.setattr(D, "_backend", "auto")
        monkeypatch.setattr(D, "_nki_gemm_impl", lambda *a: pytest.fail("NKI GEMM launched"))
        monkeypatch.setattr(D, "_nki_syrk_impl", lambda *a, **k: pytest.fail("NKI SYRK launched"))

    def test_small_host_gemm_stays_on_host(self, auto, monkeypatch):
        monkeypatch.setattr(D, "_upload", lambda t: pytest.fail("uploaded"))
        A, B = _host(64, 64), _host(64, 64, seed=1)
        C = trnblas.gemm(1.0, A, B)
        assert C.device.type == "cpu"
        assert _rel(C, A.double() @ B.double()) < 1e-5

    def test_large_host_wrappers_run_aten(self, auto):
        A, B = _host(1024, 1024), _host(1024, 1024, seed=1)
        C = trnblas.gemm(1.0, A, B, transA=True)
        assert C.device.type == "cpu" and C.is_pinned()
        assert _rel(C, A.double().T @ B.double()) < 1e-5
        S = trnblas.syrk(1.0, A)
        assert torch.equal(S, S.T) and _rel(S, A.double() @ A.double().T) < 1e-5
        # Batch-broadcast left operand, as in the DF-MP2 example's half-transform.
        Cv = _host(256, 224, seed=2)
        X = _host(8, 256, 768, seed=3)
        Z = trnblas.batched_gemm(1.0, Cv.T.unsqueeze(0).expand(8, -1, -1), X)
        assert _rel(Z, Cv.double().T @ X.double()) < 1e-5
        n = 1024
        L = torch.tril(_host(n, n)) * 0.01 + 3 * torch.eye(n)
        R = _host(n, 2048, seed=4)
        T = trnblas.trsm(1.0, L, R, uplo="lower", trans=True)
        assert _rel(L.double().T @ T.double(), R.double()) < 1e-5
        Sy = A + A.T
        Y = trnblas.symm(1.0, Sy, B)
        assert _rel(Y, Sy.double() @ B.double()) < 1e-5

    def test_explicit_nki_runs_kernel_on_resident_and_host(self, require_nki, monkeypatch):
        monkeypatch.delenv("TRNBLAS_NATIVE_GEMM", raising=False)
        called = []
        real = D._nki_gemm_impl
        monkeypatch.setattr(D, "_nki_gemm_impl", lambda *a: called.append(1) or real(*a))
        A, B = _host(300, 200), _host(200, 600, seed=1)
        C = trnblas.gemm(1.0, A.to(DEV), B.to(DEV))
        C2 = trnblas.gemm(1.0, A, B)
        assert called == [1, 1]
        assert _rel(C.cpu(), A.double() @ B.double()) < 1e-5
        assert _rel(C2, A.double() @ B.double()) < 1e-5


class TestNativeMetricInvHalf:
    """`metric_inv_half` on a device-resident J: host LAPACK route vs device route."""

    @staticmethod
    def _spd(n):
        R = _host(n, n, seed=5).double()
        return (R @ R.T / n + torch.eye(n, dtype=torch.float64)).float()

    @pytest.mark.parametrize("route", ["host", "device"])
    @pytest.mark.parametrize("n", [71, 300])
    def test_route_on_device(self, require_nki, monkeypatch, route, n):
        monkeypatch.setenv("TRNBLAS_METRIC_SOLVE", route)
        called = []
        real = D._host_inv_chol_factor
        monkeypatch.setattr(D, "_host_inv_chol_factor", lambda J: called.append(1) or real(J))
        J = self._spd(n)
        X = trnblas.metric_inv_half(J.to(DEV))
        assert X.device.type == "neuron" and X.dtype == torch.float32
        assert called == ([1] if route == "host" else [])
        Xd, Jd = X.cpu().double(), J.double()
        resid = (Xd.T @ Jd @ Xd - torch.eye(n, dtype=torch.float64)).abs().max().item()
        assert resid < 1e-4

    def test_auto_takes_host_below_crossover(self, require_nki, monkeypatch):
        monkeypatch.delenv("TRNBLAS_METRIC_SOLVE", raising=False)
        assert D._metric_solve_route(torch.eye(71).to(DEV)) == "host"


class TestNativeHostOutput:
    """Host results come from the caching host allocator, fresh per call (#61)."""

    def test_large_host_gemm_result_is_cached_host_memory(self, require_nki, monkeypatch):
        monkeypatch.delenv("TRNBLAS_PINNED_OUTPUT", raising=False)
        A, B = _host(1024, 512), _host(512, 1024, seed=1)  # 4 MiB result
        C1 = trnblas.gemm(1.0, A, B)
        C2 = trnblas.gemm(2.0, A, B)
        assert C1.device.type == "cpu" and C1.is_pinned() and C1.is_contiguous()
        assert C1.data_ptr() != C2.data_ptr()
        ref = A.double() @ B.double()
        assert _rel(C1, ref) < 1e-5
        assert _rel(C2, 2.0 * ref) < 1e-5
        # A freed result's block is recycled, and the live one is untouched.
        del C2
        C3 = trnblas.gemm(3.0, A, B)
        assert _rel(C1, ref) < 1e-5
        assert _rel(C3, 3.0 * ref) < 1e-5

    def test_env_off_returns_pageable(self, require_nki, monkeypatch):
        monkeypatch.setenv("TRNBLAS_PINNED_OUTPUT", "0")
        A, B = _host(1024, 512), _host(512, 1024, seed=1)
        C = trnblas.gemm(1.0, A, B)
        assert not C.is_pinned()
        assert _rel(C, A.double() @ B.double()) < 1e-5

    def test_small_result_skips_cache(self, require_nki):
        A, B = _host(64, 64), _host(64, 64, seed=1)
        assert not trnblas.gemm(1.0, A, B).is_pinned()

    def test_host_syrk_batched_trsm(self, require_nki):
        A = _host(1024, 256)
        S = trnblas.syrk(1.0, A)
        assert S.is_pinned() and torch.equal(S, S.T)
        assert _rel(S, A.double() @ A.double().T) < 1e-5
        X, Y = _host(4, 512, 256), _host(4, 256, 512, seed=1)
        Z = trnblas.batched_gemm(0.5, X, Y)
        assert Z.is_pinned()
        assert _rel(Z, 0.5 * (X.double() @ Y.double())) < 1e-5
        n = 512
        L = torch.tril(_host(n, n)) * 0.01 + 3 * torch.eye(n)
        R = _host(n, 1024, seed=2)
        T = trnblas.trsm(1.0, L, R, uplo="lower")
        assert _rel(L.double() @ T.double(), R.double()) < 1e-5
