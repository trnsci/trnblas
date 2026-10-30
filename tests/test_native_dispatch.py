"""
Dispatch-path selection (native TorchNeuron vs XLA vs CPU), launch routing,
lazy registration of the `trnblas::` custom ops, and the dormant XLA
deprecation hook.

CPU-only: `torch_neuronx` and `torch_xla` are replaced with stand-ins via
`sys.modules` / `find_spec` monkeypatching, so these run in the default
`pytest -m "not neuron"` job on any host.
"""

import importlib.util
import subprocess
import sys
import types
import warnings

import pytest
import torch

import trnblas.nki
from trnblas.nki import dispatch

_REAL_FIND_SPEC = importlib.util.find_spec

GEMM_OPS = {f"gemm_{tm}_{tk}_{tn}" for tm, tk, tn in dispatch._TILE_CANDIDATES}
EXPECTED_OPS = GEMM_OPS | {
    "gemm_ragged",
    "syrk",
    "mp2_energy",
    "fused_gemm_energy",
    "batched_pair",
    "j_batched",
    "pair_energy_sym",
}


@pytest.fixture(autouse=True)
def reset_dispatch_state(monkeypatch):
    """Each test sees an undetected path and an unregistered op table."""
    monkeypatch.setattr(dispatch, "_dispatch_path", None)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "_xla_deprecation_warned", False)
    monkeypatch.delenv("TRNSCI_NKI_DISPATCH", raising=False)
    monkeypatch.delenv("TRNSCI_XLA_DEPRECATION", raising=False)
    monkeypatch.delenv("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", raising=False)
    monkeypatch.delenv("TRNBLAS_PAIR_ENERGY", raising=False)
    monkeypatch.delenv("TRNBLAS_NATIVE_GEMM", raising=False)
    # Drop any `_native` left by an earlier test so registration re-runs.
    monkeypatch.delitem(sys.modules, "trnblas.nki._native", raising=False)
    monkeypatch.delattr(trnblas.nki, "_native", raising=False)
    yield


def _fake_native_torch_neuronx(device_count=1, recorded=None):
    """Stand-in for the TorchNeuron `torch_neuronx` (exports `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            if recorded is not None:
                recorded[name] = (f, mutates_args, fake_fn)
            return f

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: device_count
    return mod


def _fake_xla_torch_neuronx():
    """Stand-in for the XLA-lineage `torch_neuronx` (no `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")
    mod.trace = lambda *a, **k: None
    return mod


def _set_torch_xla_present(monkeypatch, present: bool):
    def find_spec(name, *args, **kwargs):
        if name == "torch_xla":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)


def _accept_neuron_device(monkeypatch):
    # Off-hardware, "neuron" is not a registered torch device type.
    real_device = torch.device

    def device(spec, *args, **kwargs):
        if spec == dispatch._NEURON_DEVICE:
            return real_device("cpu")
        return real_device(spec, *args, **kwargs)

    monkeypatch.setattr(dispatch.torch, "device", device)


class TestDispatchPathSelection:
    def test_native_when_nki_op_and_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "native"

    def test_native_probe_does_not_touch_torch_xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        # Any import of torch_xla would raise ImportError; any spec lookup
        # would fail the test.
        monkeypatch.setitem(sys.modules, "torch_xla", None)

        def find_spec(name, *args, **kwargs):
            assert name != "torch_xla", "native probe must not look up torch_xla"
            return _REAL_FIND_SPEC(name, *args, **kwargs)

        monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)
        assert dispatch._get_dispatch_path() == "native"

    def test_xla_when_torch_neuronx_has_no_nki_op(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_xla_when_native_has_no_device(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(device_count=0)
        )
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_not_native_when_neuron_device_type_unregistered(self, monkeypatch):
        # torch_neuronx imported but "neuron" is not a torch device type
        # (e.g. another backend owns PrivateUse1).
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_not_native_when_device_count_raises(self, monkeypatch):
        mod = _fake_native_torch_neuronx()

        def boom():
            raise RuntimeError("no runtime")

        mod.device_count = boom
        monkeypatch.setitem(sys.modules, "torch_neuronx", mod)
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_cpu_when_nothing_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", None)  # import raises
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_detection_is_cached(self, monkeypatch):
        calls = []

        def detect():
            calls.append(1)
            return "xla"

        monkeypatch.setattr(dispatch, "_detect_dispatch_path", detect)
        assert dispatch._get_dispatch_path() == "xla"
        assert dispatch._get_dispatch_path() == "xla"
        assert len(calls) == 1


class TestDispatchOverride:
    """TRNSCI_NKI_DISPATCH=auto|native|xla (suite-wide) pins the path."""

    def _both_lineages(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)

    def test_auto_prefers_native(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "auto")
        assert dispatch._get_dispatch_path() == "native"

    def test_force_xla_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "XLA")
        assert dispatch._get_dispatch_path() == "xla"

    def test_force_native_without_backend_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "native")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=native"):
            dispatch._get_dispatch_path()

    def test_force_xla_without_torch_xla_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "xla")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=xla"):
            dispatch._get_dispatch_path()

    def test_unknown_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cuda")
        with pytest.raises(ValueError, match="auto, native, or xla"):
            dispatch._get_dispatch_path()


class TestXlaDeprecationHook:
    """Dormant until TorchNeuron GA (trnsci/trnsci#35): off unless
    TRNSCI_XLA_DEPRECATION=1, then a single DeprecationWarning."""

    def _xla_only(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)

    def test_off_by_default(self, monkeypatch):
        self._xla_only(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "xla"

    def test_fires_once_when_enabled(self, monkeypatch):
        self._xla_only(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert dispatch._get_dispatch_path() == "xla"
            # Re-detection in the same process must not warn again.
            dispatch._dispatch_path = None
            assert dispatch._get_dispatch_path() == "xla"
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert str(dep[0].message) == (
            "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
            "Trn1/Inf2 remain supported in maintenance mode."
        )

    def test_never_fires_on_native(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "native"


class TestLaunchRouting:
    def test_native_launch_uses_custom_op_not_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        seen = {}

        def op(a, b):
            seen["args"] = (a, b)
            return a + b

        monkeypatch.setattr(dispatch, "_native_ops", {"fake": op})

        def no_xla(*t):
            raise AssertionError("_to_xla must not run on the native path")

        def no_kernel(*t):
            raise AssertionError("raw kernel must not be called on the native path")

        monkeypatch.setattr(dispatch, "_to_xla", no_xla)
        out = dispatch._launch(no_kernel, "fake", torch.ones(2, 1), torch.ones(2, 1))
        assert torch.equal(out, torch.full((2, 1), 2.0))
        assert out.device.type == "cpu"

    def test_xla_launch_calls_kernel_via_to_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        moved = []

        def fake_to_xla(*tensors):
            moved.append(len(tensors))
            return list(tensors), tensors[0].device

        monkeypatch.setattr(dispatch, "_to_xla", fake_to_xla)

        def no_ops():
            raise AssertionError("native ops must not load on the XLA path")

        monkeypatch.setattr(dispatch, "_get_native_ops", no_ops)
        out = dispatch._launch(lambda x: x * 3, "unused", torch.ones(4, 1))
        assert moved == [1]
        assert torch.equal(out, torch.full((4, 1), 3.0))

    def test_cpu_launch_keeps_pre_native_behaviour(self, monkeypatch):
        # No device path: the launch goes through `_to_xla`, which raises
        # when torch_xla is absent, the same failure as before.
        monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        with pytest.raises(ImportError):
            dispatch._launch(lambda x: x, "unused", torch.ones(1, 1))


@pytest.fixture
def native_stub(monkeypatch):
    """Native path on CPU: every op is a torch stand-in that records its name.

    The raw kernels are sentinels that fail if called, so a host wrapper that
    bypasses the custom op (or `_to_xla`) fails the test.
    """
    calls: list[str] = []
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(dispatch, "_RAGGED_ENABLED", True)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")

    def no_xla(*t):
        raise AssertionError("_to_xla must not run on the native path")

    monkeypatch.setattr(dispatch, "_to_xla", no_xla)

    def sentinel(name):
        def k(*a):
            raise AssertionError(f"raw kernel {name} called on the native path")

        return k

    for name in (
        "_gemm_kernel_ragged",
        "_syrk_kernel",
        "_mp2_energy_kernel",
        "_fused_gemm_energy_kernel",
        "_batched_pair_kernel",
        "_j_batched_kernel",
    ):
        monkeypatch.setattr(dispatch, name, sentinel(name), raising=False)
    monkeypatch.setattr(
        dispatch,
        "_gemm_kernel_registry",
        {t: sentinel(str(t)) for t in dispatch._TILE_CANDIDATES},
    )

    def rec(name, fn):
        def op(*a):
            calls.append(name)
            return fn(*a)

        return op

    pair = 128
    ops = {n: rec(n, lambda a, b: a @ b) for n in GEMM_OPS | {"gemm_ragged"}}
    ops["syrk"] = rec("syrk", lambda a: a @ a.T)
    ops["mp2_energy"] = rec(
        "mp2_energy", lambda t, ec, ef, vc, vr: torch.zeros(1, ec.shape[1] * ef.shape[1])
    )
    ops["fused_gemm_energy"] = rec(
        "fused_gemm_energy", lambda bi, bj, eo, vc, vr: torch.zeros(bi.shape[0], 1)
    )
    ops["batched_pair"] = rec(
        "batched_pair", lambda B, eo, vc, vr: torch.zeros(pair, B.shape[0] ** 2)
    )
    ops["j_batched"] = rec(
        "j_batched", lambda bi, B, eoi, eor, vc, vr: torch.zeros(pair, B.shape[0])
    )
    ops["pair_energy_sym"] = rec(
        "pair_energy_sym", lambda bi, Bw, eoi, eow, vc, vr: torch.zeros(pair, Bw.shape[0])
    )
    assert set(ops) == EXPECTED_OPS
    monkeypatch.setattr(dispatch, "_native_ops", ops)
    return calls


class TestHostWrappersOnNativePath:
    """Every host wrapper reaches its `trnblas::` op, never `_to_xla`."""

    def test_gemm_aligned_uses_default_tile_op(self, native_stub):
        A, B = torch.randn(256, 128), torch.randn(128, 512)
        out = dispatch.nki_gemm(A, B)
        assert native_stub == ["gemm_128_128_512"]
        torch.testing.assert_close(out, A @ B)

    def test_gemm_ragged_uses_ragged_op_without_pad(self, native_stub, monkeypatch):
        pads = []
        orig_pad = dispatch._pad_gemm_operands
        monkeypatch.setattr(
            dispatch, "_pad_gemm_operands", lambda *a: pads.append(1) or orig_pad(*a)
        )
        A, B = torch.randn(127, 129), torch.randn(129, 513)
        out = dispatch.nki_gemm(A, B)
        assert native_stub == ["gemm_ragged"]
        assert pads == [], "native ragged path must not pad in torch"
        torch.testing.assert_close(out, A @ B)

    def test_batched_gemm(self, native_stub):
        A, B = torch.randn(3, 128, 128), torch.randn(3, 128, 128)
        out = dispatch.nki_batched_gemm(A, B)
        assert native_stub == ["gemm_128_128_512"] * 3
        torch.testing.assert_close(out, torch.bmm(A, B))

    def test_syrk(self, native_stub):
        A = torch.randn(256, 128)
        out = dispatch.nki_syrk(A)
        assert native_stub == ["syrk"]
        torch.testing.assert_close(out, A @ A.T)

    def test_trsm_rides_gemm_op(self, native_stub):
        n = 256
        L = torch.tril(torch.randn(n, n)) + n * torch.eye(n)
        B = torch.randn(n, 128)
        X = dispatch.nki_trsm(L, B, uplo="lower")
        assert native_stub and set(native_stub) <= GEMM_OPS | {"gemm_ragged"}
        torch.testing.assert_close(L @ X, B, atol=1e-3, rtol=1e-4)

    def test_mp2_energy(self, native_stub):
        ic, nocc, nvir = 2, 3, 8
        dispatch.nki_mp2_energy(
            torch.randn(ic * nvir, nocc * nvir),
            -torch.rand(ic),
            -torch.rand(nocc),
            torch.rand(nvir),
        )
        assert native_stub == ["mp2_energy"]

    def test_fused_gemm_energy(self, native_stub):
        dispatch.nki_fused_gemm_energy(
            torch.randn(100, 200), torch.randn(100, 200), -0.5, -0.4, torch.rand(100)
        )
        assert native_stub == ["fused_gemm_energy"]

    def test_batched_pair_small_takes_full_batch_on_native(self, native_stub):
        # est_iters = 32^2 * 1 * 3 = 3072 <= native threshold: one launch.
        nocc = 32
        dispatch.nki_batched_pair_energy(
            torch.randn(nocc, 128, 384), -torch.rand(nocc), torch.rand(128)
        )
        assert native_stub == ["batched_pair"]

    def test_batched_pair_large_takes_chunked_on_native(self, native_stub):
        # est_iters = 64^2 * 1 * 1 = 4096 > 3072: one j_batched launch per i
        # (still full-batch on XLA, whose threshold is 4096).
        nocc = 64
        dispatch.nki_batched_pair_energy(
            torch.randn(nocc, 128, 128), -torch.rand(nocc), torch.rand(128)
        )
        assert native_stub == ["j_batched"] * nocc

    def test_batched_pair_threshold_env_override(self, native_stub, monkeypatch):
        monkeypatch.setenv("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", "0")
        dispatch.nki_batched_pair_energy(torch.randn(3, 128, 128), -torch.rand(3), torch.rand(128))
        assert native_stub == ["j_batched"] * 3

    def test_autotune_sweep_launches_ops(self, native_stub, monkeypatch, tmp_path):
        monkeypatch.setattr(dispatch, "_AUTOTUNE_ENABLED", True)
        monkeypatch.setattr(dispatch, "_autotune_mem", {})
        monkeypatch.setattr(dispatch, "_autotune_loaded", True)
        monkeypatch.setattr(dispatch, "_AUTOTUNE_CACHE_FILE", tmp_path / "c.json")
        synced = []
        monkeypatch.setattr(dispatch, "_device_synchronize", lambda: synced.append(1))
        A, B = torch.randn(128, 128), torch.randn(128, 512)
        out = dispatch.nki_gemm(A, B)
        torch.testing.assert_close(out, A @ B)
        assert set(native_stub) <= GEMM_OPS
        assert len(set(native_stub)) == len(dispatch._TILE_CANDIDATES)
        assert synced, "native sweep timing must synchronize the device"


class TestBatchedPairThreshold:
    def test_xla_value_unchanged(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        assert dispatch._batched_pair_chunk_threshold() == 4096

    def test_native_value(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
        assert dispatch._batched_pair_chunk_threshold() == 3072

    def test_simulator_keeps_xla_value_without_detection(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_USE_SIMULATOR", True)
        monkeypatch.setattr(dispatch, "HAS_NKI", True)

        def no_detect():
            raise AssertionError("simulator mode must not probe the device")

        monkeypatch.setattr(dispatch, "_get_dispatch_path", no_detect)
        assert dispatch._batched_pair_chunk_threshold() == 4096

    def test_env_override(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        monkeypatch.setenv("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", "12")
        assert dispatch._batched_pair_chunk_threshold() == 12


class TestLazyRegistration:
    def test_import_does_not_register_ops(self):
        # Fresh interpreter: importing trnblas (and the dispatch module)
        # must not load `_native` or import torch_neuronx / torch_xla.
        code = (
            "import sys, trnblas, trnblas.nki.dispatch; "
            "bad = [m for m in ('trnblas.nki._native', 'torch_neuronx', 'torch_xla') "
            "if m in sys.modules]; "
            "assert not bad, bad"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_ops_registered_on_first_native_use(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        assert "trnblas.nki._native" not in sys.modules
        assert recorded == {}

        ops = dispatch._get_native_ops()

        assert set(ops) == EXPECTED_OPS
        assert set(recorded) == {f"trnblas::{name}" for name in EXPECTED_OPS}
        assert all(mutates == {} for _, mutates, _ in recorded.values())
        assert all(fake is not None for _, _, fake in recorded.values())

        # Second call reuses the table; no re-registration.
        recorded.clear()
        assert dispatch._get_native_ops() is ops
        assert recorded == {}

    def test_op_signatures_build_a_schema(self, monkeypatch):
        # nki_op derives the schema with torch.library.infer_schema; check
        # every wrapper is fully annotated (Tensor args, Tensor return).
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()
        assert len(recorded) == len(EXPECTED_OPS)
        for name, (fn, mutates, _) in recorded.items():
            schema = torch.library.infer_schema(fn, mutates_args=mutates)
            assert schema.endswith("-> Tensor"), (name, schema)
            assert "Tensor" in schema.split("->")[0], (name, schema)

    def test_fake_fns_give_kernel_output_shapes(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()

        def meta(*shape):
            return torch.empty(shape, dtype=torch.float32, device="meta")

        def fake(name):
            return recorded[f"trnblas::{name}"][2]

        for name in GEMM_OPS | {"gemm_ragged"}:
            out = fake(name)(meta(300, 200), meta(200, 600))
            assert out.shape == (300, 600) and out.device.type == "meta", name
        assert fake("syrk")(meta(256, 128)).shape == (256, 256)
        # mp2: (P_TILE, IC*NOCC), P_TILE = largest divisor of NVIR <= 128.
        ic, nocc = 2, 3
        for nvir, p_tile in ((128, 128), (200, 100), (24, 24), (131, 1)):
            out = fake("mp2_energy")(
                meta(ic * nvir, nocc * nvir),
                meta(1, ic),
                meta(1, nocc),
                meta(nvir, 1),
                meta(1, nvir),
            )
            assert out.shape == (p_tile, ic * nocc), nvir
        assert fake("fused_gemm_energy")(
            meta(256, 128), meta(256, 128), meta(1, 1), meta(256, 1), meta(1, 256)
        ).shape == (256, 1)
        assert fake("batched_pair")(
            meta(5, 128, 256), meta(1, 5), meta(128, 1), meta(1, 128)
        ).shape == (128, 25)
        assert fake("pair_energy_sym")(
            meta(256, 128), meta(4, 256, 128), meta(1, 1), meta(1, 4), meta(128, 1), meta(1, 128)
        ).shape == (128, 4)
        assert fake("j_batched")(
            meta(128, 256), meta(4, 128, 256), meta(1, 1), meta(1, 4), meta(128, 1), meta(1, 128)
        ).shape == (128, 4)


class TestNativeGemmPolicy:
    """Plain GEMM-family calls on device-resident operands route to aten (native path).

    CPU tensors stand in for `neuron` tensors by patching `_on_neuron_device`.
    """

    @pytest.fixture
    def nki_backend(self, monkeypatch):
        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_backend", "nki")
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.delenv("TRNBLAS_NATIVE_GEMM", raising=False)

        def boom(*a, **k):
            raise AssertionError("NKI GEMM kernel launched")

        monkeypatch.setattr(dispatch, "_nki_gemm_impl", boom)
        monkeypatch.setattr(dispatch, "_nki_syrk_impl", boom)

    def test_device_resident_goes_to_aten(self, monkeypatch, nki_backend):
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
        A, B = torch.randn(7, 5), torch.randn(5, 3)
        torch.testing.assert_close(dispatch.nki_gemm(A, B), A @ B)
        torch.testing.assert_close(dispatch.nki_syrk(A), A @ A.T)
        Ab, Bb = torch.randn(4, 7, 5), torch.randn(4, 5, 3)
        torch.testing.assert_close(dispatch.nki_batched_gemm(Ab, Bb), torch.bmm(Ab, Bb))
        L = torch.tril(torch.randn(300, 300)) + 300 * torch.eye(300)
        X = dispatch.nki_trsm(L, torch.eye(300), uplo="lower", trans=True)
        torch.testing.assert_close(L.T @ X, torch.eye(300), atol=1e-5, rtol=0)

    def test_host_tensors_keep_nki(self, monkeypatch, nki_backend):
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: False)
        with pytest.raises(AssertionError, match="NKI GEMM kernel launched"):
            dispatch.nki_gemm(torch.randn(4, 4), torch.randn(4, 4))

    def test_env_nki_forces_kernel(self, monkeypatch, nki_backend):
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
        monkeypatch.setenv("TRNBLAS_NATIVE_GEMM", "nki")
        with pytest.raises(AssertionError, match="NKI GEMM kernel launched"):
            dispatch.nki_gemm(torch.randn(4, 4), torch.randn(4, 4))

    def test_env_rejects_unknown(self, monkeypatch, nki_backend):
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
        monkeypatch.setenv("TRNBLAS_NATIVE_GEMM", "fast")
        with pytest.raises(ValueError, match="TRNBLAS_NATIVE_GEMM"):
            dispatch.nki_gemm(torch.randn(4, 4), torch.randn(4, 4))

    def test_simulator_keeps_nki(self, monkeypatch, nki_backend):
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: True)
        assert not dispatch._gemm_via_aten(torch.randn(2, 2))


class TestPairEnergySym:
    """Host-side logic of the native LNC=2 symmetric pair-energy kernel."""

    @pytest.mark.parametrize("nocc", range(1, 10))
    def test_window_weights_cover_every_ordered_pair_once(self, nocc):
        dims = dispatch._pair_sym_dims(nocc, 128, 128)
        cover = torch.zeros(nocc, nocc, dtype=torch.float64)
        for i in range(nocc):
            for d, w in enumerate(dims["weights"]):
                j = (i + d) % nocc
                cover[i, j] += w / 2
                cover[j, i] += w / 2
        torch.testing.assert_close(cover, torch.ones(nocc, nocc, dtype=torch.float64))
        assert dims["nd_pad"] % 2 == 0

    @pytest.mark.parametrize("shape", [(1, 3, 5), (2, 7, 9), (5, 19, 71), (4, 130, 129)])
    def test_reference_walk_matches_pair_loop(self, shape):
        g = torch.Generator().manual_seed(0)
        nocc, nvir, naux = shape
        B = 0.1 * torch.randn(nocc, nvir, naux, generator=g)
        eo = -0.5 - torch.rand(nocc, generator=g)
        ev = 0.2 + torch.rand(nvir, generator=g)
        ref = dispatch._torch_batched_pair_energy(B.double(), eo.double(), ev.double())
        got = dispatch._pair_sym_reference(B, eo, ev)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=0)

    def test_chunked_aten_energy_matches_pair_loop(self):
        g = torch.Generator().manual_seed(1)
        B = 0.1 * torch.randn(6, 11, 17, generator=g, dtype=torch.float64)
        eo = -0.5 - torch.rand(6, generator=g, dtype=torch.float64)
        ev = 0.2 + torch.rand(11, generator=g, dtype=torch.float64)
        ref = dispatch._torch_batched_pair_energy(B, eo, ev)
        # A budget of two i-rows forces three chunks.
        got = dispatch._torch_chunked_pair_energy(B, eo, ev, mem_budget_bytes=2 * 4 * 11 * 6 * 11)
        torch.testing.assert_close(got, ref)

    @pytest.fixture
    def native_device(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")

    def test_route_by_work(self, native_device):
        assert dispatch._pair_energy_route(torch.empty(20, 200, 555)) == "aten"
        assert dispatch._pair_energy_route(torch.empty(64, 448, 1536)) == "sym"

    def test_route_aten_when_sbuf_too_small(self, native_device):
        # nvir_pad 1024, naux_pad 4096: 4 * 1024 * (2 * 32 + 8) bytes per partition.
        assert dispatch._pair_energy_route(torch.empty(64, 1000, 4000)) == "aten"

    def test_route_env_override(self, native_device, monkeypatch):
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "legacy")
        assert dispatch._pair_energy_route(torch.empty(64, 448, 1536)) == "legacy"
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "sym")
        assert dispatch._pair_energy_route(torch.empty(2, 8, 8)) == "sym"
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "bogus")
        with pytest.raises(ValueError, match="TRNBLAS_PAIR_ENERGY"):
            dispatch._pair_energy_route(torch.empty(2, 8, 8))

    def test_route_legacy_off_native(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: False)
        assert dispatch._pair_energy_route(torch.empty(64, 448, 1536)) == "legacy"
