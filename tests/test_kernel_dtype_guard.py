"""No NKI kernel is ever reached with an operand dtype it does not take
(trnsci/trnsci#35; earlier local fixes #68/#69, #86).

The GEMM / SYRK kernels take float32, bfloat16 and float16 (FP32 PSUM, output
in the input dtype). The DF-MP2 energy kernels take float32 only. Before this
guard, float64, integer, bool and complex tensors reached the energy kernels
unchanged under `set_backend("nki")` / `"auto"` and came back as garbage.

CPU-only. Every `trnblas::` custom op is registered through a stand-in
`torch_neuronx.nki_op` whose op is a stub: it records any operand dtype not
in the kernel's allowed set below (and raises), then returns the op's
zero-filled fake output. The raw `@nki.jit` kernels are replaced with the same
stubs for the XLA and simulator paths. Every public API that can reach a
kernel is then driven with float32, float64, bfloat16, float16, int32, int64,
bool, complex64 and complex128 inputs under `nki` and `auto`, and the result
dtype is compared with the pytorch backend's.
"""

import sys
import types
import warnings

import pytest
import torch

import trnblas
import trnblas.nki
from trnblas.nki import dispatch, multicore

F32, BF16, F16, F64 = torch.float32, torch.bfloat16, torch.float16, torch.float64
GEMM = frozenset({F32, BF16, F16})
FP32 = frozenset({F32})

# What each kernel really takes; independent of dispatch._KERNEL_DTYPES so a
# loosened table fails here. A new kernel without an entry fails
# `test_every_op_has_an_allowed_set`.
ALLOWED = {
    **{dispatch._gemm_op_name(*t): GEMM for t in dispatch._TILE_CANDIDATES},
    "gemm_ragged": GEMM,
    "gemm_lnc": GEMM,
    "gemm_lnc_tn": GEMM,
    "syrk": GEMM,
    "mp2_energy": FP32,
    "fused_gemm_energy": FP32,
    "batched_pair": FP32,
    "j_batched": FP32,
    "pair_energy_sym": FP32,
}

# Raw kernel attribute on `dispatch` for each op (XLA / simulator paths).
RAW_KERNELS = {
    "gemm_ragged": "_gemm_kernel_ragged",
    "syrk": "_syrk_kernel",
    "mp2_energy": "_mp2_energy_kernel",
    "fused_gemm_energy": "_fused_gemm_energy_kernel",
    "batched_pair": "_batched_pair_kernel",
    "j_batched": "_j_batched_kernel",
    "pair_energy_sym": "_pair_energy_sym_kernel",
}

ALL_DTYPES = [F32, F64, BF16, F16, torch.int32, torch.int64, torch.bool]
ALL_DTYPES += [torch.complex64, torch.complex128]


class Stubs:
    """Records every kernel call and every operand dtype violation."""

    def __init__(self):
        self.calls: list[str] = []
        self.violations: list[tuple[str, list[str]]] = []
        self.fakes: dict = {}

    def stub(self, name, fake_fn):
        def op(*args):
            dts = [a.dtype for a in args if isinstance(a, torch.Tensor)]
            self.calls.append(name)
            if any(d not in ALLOWED[name] for d in dts):
                self.violations.append((name, [str(d) for d in dts]))
                raise AssertionError(f"kernel {name} reached with operand dtypes {dts}")
            out = fake_fn(*args)
            return out.zero_() if out.is_floating_point() else out.fill_(0)

        return op


def _fake_torch_neuronx(stubs: Stubs):
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        short = name.split("::", 1)[1]

        def dec(f):
            stubs.fakes[short] = fake_fn
            return stubs.stub(short, fake_fn)

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: 1
    return mod


@pytest.fixture
def stubs(monkeypatch):
    """Native dispatch path on CPU with stub custom ops; `_native` imported fresh."""
    s = Stubs()
    monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_torch_neuronx(s))
    monkeypatch.delitem(sys.modules, "trnblas.nki._native", raising=False)
    monkeypatch.delattr(trnblas.nki, "_native", raising=False)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setattr(dispatch, "_AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(dispatch, "_RAGGED_ENABLED", True)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    for var in (
        "TRNBLAS_NATIVE_GEMM",
        "TRNBLAS_PAIR_ENERGY",
        "TRNBLAS_METRIC_SOLVE",
        "TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD",
        "TRNBLAS_CORES",
    ):
        monkeypatch.delenv(var, raising=False)

    def no_xla(*t):
        raise AssertionError("_to_xla must not run on the native path")

    monkeypatch.setattr(dispatch, "_to_xla", no_xla)

    def sentinel(name):
        def k(*a):
            raise AssertionError(f"raw kernel {name} called on the native path")

        return k

    for attr in RAW_KERNELS.values():
        monkeypatch.setattr(dispatch, attr, sentinel(attr), raising=False)
    monkeypatch.setattr(
        dispatch,
        "_gemm_kernel_registry",
        {t: sentinel(str(t)) for t in dispatch._TILE_CANDIDATES},
    )
    dispatch._get_native_ops()  # registers every op through the stand-in
    prev = dispatch.get_backend()
    yield s
    dispatch.set_backend(prev)


def _raw_kernel_stubs(monkeypatch, s: Stubs):
    """Replace the raw kernels (XLA / simulator) with dtype-checking stubs."""
    for name, attr in RAW_KERNELS.items():
        monkeypatch.setattr(dispatch, attr, s.stub(name, s.fakes[name]), raising=False)
    monkeypatch.setattr(
        dispatch,
        "_gemm_kernel_registry",
        {
            t: s.stub(dispatch._gemm_op_name(*t), s.fakes[dispatch._gemm_op_name(*t)])
            for t in dispatch._TILE_CANDIDATES
        },
    )


@pytest.fixture
def xla_stubs(stubs, monkeypatch):
    """XLA dispatch path: `_to_xla` is the identity, raw kernels are stubs."""
    monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
    monkeypatch.setattr(dispatch, "_to_xla", lambda *t: (list(t), t[0].device))

    def no_ops():
        raise AssertionError("native ops must not load on the XLA path")

    monkeypatch.setattr(dispatch, "_get_native_ops", no_ops)
    _raw_kernel_stubs(monkeypatch, stubs)
    return stubs


@pytest.fixture
def sim_stubs(stubs, monkeypatch):
    """Simulator path: a stand-in `nki.simulate` runs the raw-kernel stubs."""
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", True)
    monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
    fake_nki = types.SimpleNamespace(
        simulate=lambda k: lambda *arrs: k(*[torch.from_numpy(a) for a in arrs]).numpy()
    )
    monkeypatch.setattr(dispatch, "nki", fake_nki, raising=False)
    _raw_kernel_stubs(monkeypatch, stubs)
    return stubs


# ── inputs ───────────────────────────────────────────────────────────────────


def _rand(shape, dtype, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    if not (dtype.is_floating_point or dtype.is_complex):
        return torch.randint(-3, 4, shape, generator=g).to(dtype)
    return (torch.randn(shape, generator=g, dtype=F64) * scale).to(dtype)


def _tri(n, dtype):
    A = torch.triu(_rand((n, n), F64, 1, 0.01)) + 4 * torch.eye(n, dtype=F64)
    return A.to(dtype) if dtype != torch.bool else A.bool()


def _spd(n, dtype):
    X = _rand((n, n), F64, 2, 0.01)
    J = X @ X.T + n * torch.eye(n, dtype=F64)
    return J.to(dtype) if dtype != torch.bool else J.bool()


def _eps(nocc, nvir, dtype):
    eo = -torch.linspace(1.0, 2.0, nocc, dtype=F64)
    ev = torch.linspace(0.5, 1.5, nvir, dtype=F64)
    if dtype == torch.bool or not (dtype.is_floating_point or dtype.is_complex):
        eo, ev = torch.round(eo * 2) - 1, torch.round(ev * 2) + 1
    return eo.to(dtype), ev.to(dtype)


NOCC, NVIR, NAUX, IC = 3, 5, 7, 2

# name -> (callable, builder(data dtype, eps dtype) -> args, is_energy)
APIS = {
    "gemm": (
        lambda A, B: trnblas.gemm(1.0, A, B),
        lambda d, e: (_rand((9, 7), d), _rand((7, 5), d, 1)),
        False,
    ),
    "gemm_transA": (
        lambda A, B: trnblas.gemm(1.0, A, B, transA=True),
        lambda d, e: (_rand((7, 9), d), _rand((7, 5), d, 1)),
        False,
    ),
    "batched_gemm": (
        lambda A, B: trnblas.batched_gemm(1.0, A, B),
        lambda d, e: (_rand((2, 9, 7), d), _rand((1, 7, 5), d, 1)),
        False,
    ),
    "syrk": (lambda A: trnblas.syrk(1.0, A), lambda d, e: (_rand((9, 7), d),), False),
    "syrk_trans": (
        lambda A: trnblas.syrk(1.0, A, trans=True),
        lambda d, e: (_rand((7, 9), d),),
        False,
    ),
    "symm": (
        lambda A, B: trnblas.symm(1.0, A, B),
        lambda d, e: (_rand((6, 6), d), _rand((6, 4), d, 1)),
        False,
    ),
    "trsm": (
        lambda A, B: trnblas.trsm(1.0, A, B),
        lambda d, e: (_tri(130, d), _rand((130, 3), d, 1)),
        False,
    ),
    "metric_inv_half": (trnblas.metric_inv_half, lambda d, e: (_spd(130, d),), False),
    "nki_gemm": (trnblas.nki.nki_gemm, lambda d, e: (_rand((9, 7), d), _rand((7, 5), d, 1)), False),
    "nki_batched_gemm": (
        trnblas.nki.nki_batched_gemm,
        lambda d, e: (_rand((2, 9, 7), d), _rand((2, 7, 5), d, 1)),
        False,
    ),
    "nki_syrk": (trnblas.nki.nki_syrk, lambda d, e: (_rand((9, 7), d),), False),
    "nki_trsm": (trnblas.nki.nki_trsm, lambda d, e: (_tri(130, d), _rand((130, 3), d, 1)), False),
    "nki_metric_inv_half": (trnblas.nki.nki_metric_inv_half, lambda d, e: (_spd(130, d),), False),
    "nki_mp2_energy": (
        trnblas.nki.nki_mp2_energy,
        lambda d, e: (
            _rand((IC * NVIR, NOCC * NVIR), d, scale=0.1),
            _eps(NOCC, NVIR, e)[0][:IC],
            _eps(NOCC, NVIR, e)[0],
            _eps(NOCC, NVIR, e)[1],
        ),
        True,
    ),
    "nki_fused_gemm_energy": (
        lambda bi, bj, ev: trnblas.nki.nki_fused_gemm_energy(bi, bj, -1.1, -1.0, ev),
        lambda d, e: (
            _rand((NVIR, NAUX), d, scale=0.1),
            _rand((NVIR, NAUX), d, 1, scale=0.1),
            _eps(NOCC, NVIR, e)[1],
        ),
        True,
    ),
    "nki_batched_pair_energy": (
        trnblas.nki.nki_batched_pair_energy,
        lambda d, e: (_rand((NOCC, NVIR, NAUX), d, scale=0.1), *_eps(NOCC, NVIR, e)),
        True,
    ),
}

# (api, data dtype, eps dtype): every dtype for the data with float32 eps, and
# for the energy APIs float32 data with every eps dtype.
CASES = [(n, d, F32) for n in APIS for d in ALL_DTYPES] + [
    (n, F32, e) for n, (_, _, energy) in APIS.items() if energy for e in ALL_DTYPES if e != F32
]
PAIR_CASES = [c for c in CASES if c[0] == "nki_batched_pair_energy"]


def _id(case):
    n, d, e = case
    return f"{n}-{str(d)[6:]}-eps_{str(e)[6:]}"


def _run(fn, args, backend):
    dispatch.set_backend(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args)


def _pytorch(fn, args):
    try:
        return _run(fn, args, "pytorch"), None
    except Exception as exc:  # the pytorch backend itself refuses this dtype
        return None, exc


def _expected_dtype(name, ref, s: Stubs, is_energy):
    # The energy kernels return float32 (unchanged by this guard). Where the
    # pytorch backend returns another dtype (float64 eps, half-precision eps
    # and amplitudes) that is a needs-decision question, not changed here.
    if is_energy and s.calls:
        return F32
    return ref.dtype


def _drive(s: Stubs, name, d, e, backend):
    fn, build, is_energy = APIS[name]
    args = build(d, e)
    ref, ref_exc = _pytorch(fn, args)
    s.calls.clear()
    try:
        got = _run(fn, args, backend)
    except Exception as exc:
        assert not s.violations, s.violations
        if ref_exc is None:
            raise AssertionError(f"{name} {d}/{e}: pytorch backend ran, {backend} raised") from exc
        return
    assert not s.violations, s.violations
    if ref is None:
        return  # pytorch refuses it; the nki route may compute in torch
    if isinstance(ref, torch.Tensor):
        assert got.dtype == _expected_dtype(name, ref, s, is_energy), (name, got.dtype, ref.dtype)
        assert got.shape == ref.shape
        if not s.calls:  # torch route: the pytorch backend's answer
            torch.testing.assert_close(got, ref, equal_nan=True)


# ── tests ────────────────────────────────────────────────────────────────────


def test_every_op_has_an_allowed_set(stubs):
    ops = dispatch._get_native_ops()
    assert set(stubs.fakes) == set(ops), "registered ops differ from _native.OPS"
    missing = set(ops) - set(ALLOWED)
    assert not missing, f"kernels without an allowed-dtype entry in this test: {missing}"
    table = getattr(dispatch, "_KERNEL_DTYPES", {})
    for op in ops:
        assert op in table, f"dispatch._KERNEL_DTYPES has no entry for {op}"
        assert set(table[op]) <= set(ALLOWED[op]), (op, table[op])


@pytest.mark.parametrize("name,d,e", CASES, ids=[_id(c) for c in CASES])
@pytest.mark.parametrize("backend", ["nki", "auto"])
def test_native_path(stubs, name, d, e, backend):
    _drive(stubs, name, d, e, backend)


@pytest.mark.parametrize("name,d,e", CASES, ids=[_id(c) for c in CASES])
def test_native_path_policy_nki(stubs, monkeypatch, name, d, e):
    # `auto` with TRNBLAS_NATIVE_GEMM=nki forces the kernels for host operands.
    monkeypatch.setenv("TRNBLAS_NATIVE_GEMM", "nki")
    _drive(stubs, name, d, e, "auto")


@pytest.mark.parametrize("_n,d,e", PAIR_CASES, ids=[_id(c) for c in PAIR_CASES])
@pytest.mark.parametrize("backend", ["nki", "auto"])
def test_pair_energy_sym_route(stubs, monkeypatch, _n, d, e, backend):
    # B treated as device-resident so the sym kernel route is taken.
    monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "sym")
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    _drive(stubs, "nki_batched_pair_energy", d, e, backend)


@pytest.mark.parametrize("_n,d,e", PAIR_CASES, ids=[_id(c) for c in PAIR_CASES])
def test_pair_energy_chunked_route(stubs, monkeypatch, _n, d, e):
    monkeypatch.setenv("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", "0")
    _drive(stubs, "nki_batched_pair_energy", d, e, "nki")


@pytest.mark.parametrize("d", ALL_DTYPES, ids=str)
def test_multicore_sym_rows(stubs, d):
    """`multicore.sym_rows` (also used by `trnblas.parallel`) calls the op directly."""
    B = _rand((NOCC, NVIR, NAUX), d, scale=0.1)
    eo, ev = _eps(NOCC, NVIR, F64)
    if d in GEMM:
        out = multicore.sym_rows(B, eo, ev, (0, NOCC), torch.device("cpu"))
        assert out.dtype == F32
    else:
        with pytest.raises(TypeError):
            multicore.sym_rows(B, eo, ev, (0, NOCC), torch.device("cpu"))
    assert not stubs.violations, stubs.violations


@pytest.mark.parametrize("name,d,e", CASES, ids=[_id(c) for c in CASES])
def test_xla_path(xla_stubs, name, d, e):
    _drive(xla_stubs, name, d, e, "auto")


@pytest.mark.parametrize("name,d,e", CASES, ids=[_id(c) for c in CASES])
def test_simulator_path(sim_stubs, name, d, e):
    _drive(sim_stubs, name, d, e, "auto")


# ── TRNBLAS_REQUIRE_NKI=1: refuse instead of computing in torch or casting ────

REQUIRE_RAISES = [F64, torch.int32, torch.int64, torch.bool, torch.complex64]


@pytest.mark.parametrize("d", REQUIRE_RAISES, ids=str)
@pytest.mark.parametrize(
    "name",
    [
        "gemm",
        "batched_gemm",
        "syrk",
        "trsm",
        "nki_mp2_energy",
        "nki_fused_gemm_energy",
        "nki_batched_pair_energy",
    ],
)
def test_require_nki_raises_for_unsupported_data(stubs, monkeypatch, name, d):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    fn, build, _ = APIS[name]
    with pytest.raises(NotImplementedError, match="NKI"):
        _run(fn, build(d, F32), "nki")
    assert not stubs.calls


@pytest.mark.parametrize(
    "name", ["nki_mp2_energy", "nki_fused_gemm_energy", "nki_batched_pair_energy"]
)
def test_require_nki_raises_for_float64_eps(stubs, monkeypatch, name):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    fn, build, _ = APIS[name]
    with pytest.raises(NotImplementedError, match="float64"):
        _run(fn, build(F32, F64), "nki")
    assert not stubs.calls


def test_require_nki_raises_on_sym_route(stubs, monkeypatch):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "sym")
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    fn, build, _ = APIS["nki_batched_pair_energy"]
    with pytest.raises(NotImplementedError):
        _run(fn, build(F64, F32), "nki")


@pytest.mark.parametrize("d", [F32, BF16, F16], ids=str)
def test_require_nki_still_runs_supported_dtypes(stubs, monkeypatch, d):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    out = _run(trnblas.gemm, (1.0, _rand((9, 7), d), _rand((7, 5), d)), "nki")
    assert out.dtype == d
    assert stubs.calls == ["gemm_ragged"]
    out = _run(
        trnblas.nki.nki_batched_pair_energy, APIS["nki_batched_pair_energy"][1](d, F32), "nki"
    )
    assert "batched_pair" in stubs.calls
    assert not stubs.violations


# ── the boundary itself ───────────────────────────────────────────────────────


@pytest.mark.parametrize("d", [F64, torch.int32, torch.bool, torch.complex64], ids=str)
def test_boundary_refuses_unsupported_dtypes(stubs, monkeypatch, d):
    """`_launch`, `_call_kernel`, `_simulate` and every `_native.OPS` entry
    raise TypeError naming the op, even when called directly."""
    x = _rand((4, 4), d)
    ops = dispatch._get_native_ops()
    for op_name, op in ops.items():
        n_args = len(
            stubs.fakes[op_name].__code__.co_varnames[: stubs.fakes[op_name].__code__.co_argcount]
        )
        with pytest.raises(TypeError, match=op_name):
            op(*([x] * n_args))
    with pytest.raises(TypeError, match="syrk"):
        dispatch._launch(dispatch._syrk_kernel, "syrk", x)
    with pytest.raises(TypeError, match="mp2_energy"):
        dispatch._call_kernel(dispatch._mp2_energy_kernel, "mp2_energy", x, x, x, x, x)
    monkeypatch.setattr(dispatch, "nki", types.SimpleNamespace(simulate=None), raising=False)
    with pytest.raises(TypeError, match="gemm_ragged"):
        dispatch._simulate(dispatch._gemm_kernel_ragged, "gemm_ragged", x, x)
    assert not stubs.calls


def test_boundary_refuses_unknown_op(stubs):
    with pytest.raises(TypeError, match="no entry"):
        dispatch._call_kernel(lambda *a: a[0], "not_a_kernel", torch.ones(2, 2))


KERNEL_FAMILIES = {
    "gemm_ragged",
    "syrk",
    "mp2_energy",
    "fused_gemm_energy",
    "batched_pair",
    "j_batched",
    "pair_energy_sym",
}


def test_supported_dtypes_reach_every_kernel(stubs, monkeypatch):
    """The stubs are live: float32 calls reach every kernel family."""
    seen = set()
    for name in APIS:
        _drive(stubs, name, F32, F32, "nki")
        seen |= set(stubs.calls)
    monkeypatch.setenv("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", "0")
    _drive(stubs, "nki_batched_pair_energy", F32, F32, "nki")
    seen |= set(stubs.calls)
    monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "sym")
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    _drive(stubs, "nki_batched_pair_energy", F32, F32, "nki")
    seen |= set(stubs.calls)
    assert seen >= KERNEL_FAMILIES, KERNEL_FAMILIES - seen
