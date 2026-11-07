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

Float64 policy (suite-wide, trnblas#89): under `auto` float64 runs in torch at
float64 accuracy (no kernel call); under explicit `nki` float64 operands are
cast to float32 for the kernel with one UserWarning per op; under
TRNBLAS_REQUIRE_NKI=1 float64 raises NotImplementedError. Every route returns
the pytorch backend's result dtype (bf16/fp16 and mixed promotion included).
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
    # Record the real state first, so teardown removes the stub-backed `_native`
    # imported below (a later test in the session would otherwise get stub ops).
    monkeypatch.setitem(sys.modules, "trnblas.nki._native", None)
    monkeypatch.setattr(trnblas.nki, "_native", None, raising=False)
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


def _has_f64(name, d, e):
    return d == F64 or (APIS[name][2] and e == F64)


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
    if backend == "auto" and _has_f64(name, d, e):
        assert not s.calls, f"{name} {d}/{e}: float64 under auto reached {s.calls}"
    if isinstance(ref, torch.Tensor):
        # Every route returns the pytorch backend's dtype (trnblas#89).
        assert got.dtype == ref.dtype, (name, got.dtype, ref.dtype)
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
@pytest.mark.parametrize("backend", ["nki", "auto"])
def test_xla_path(xla_stubs, name, d, e, backend):
    _drive(xla_stubs, name, d, e, backend)


@pytest.mark.parametrize("name,d,e", CASES, ids=[_id(c) for c in CASES])
@pytest.mark.parametrize("backend", ["nki", "auto"])
def test_simulator_path(sim_stubs, name, d, e, backend):
    _drive(sim_stubs, name, d, e, backend)


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


# ── float64 policy (trnblas#89): auto -> torch, nki -> fp32 cast + one warning ─

# APIs whose float64 call reaches a kernel under explicit `nki`, and the op
# name the warning carries. `symm` has no kernel: torch, no warning.
F64_KERNEL_APIS = {
    "gemm": "gemm",
    "gemm_transA": "gemm",
    "batched_gemm": "batched_gemm",
    "syrk": "syrk",
    "syrk_trans": "syrk",
    "trsm": "trsm",
    "metric_inv_half": "trsm",
    "nki_gemm": "gemm",
    "nki_batched_gemm": "batched_gemm",
    "nki_syrk": "syrk",
    "nki_trsm": "trsm",
    "nki_metric_inv_half": "trsm",
    "nki_mp2_energy": "mp2_energy",
    "nki_fused_gemm_energy": "fused_gemm_energy",
    "nki_batched_pair_energy": "batched_pair_energy",
}
ENERGY_APIS = ["nki_mp2_energy", "nki_fused_gemm_energy", "nki_batched_pair_energy"]
# (api, data dtype, eps dtype) with a float64 operand.
F64_CASES = [(n, F64, F32) for n in F64_KERNEL_APIS] + [(n, F64, F64) for n in ENERGY_APIS]
F64_CASES += [(n, F32, F64) for n in ENERGY_APIS]
AUTO_F64_CASES = F64_CASES + [("symm", F64, F32)]
PAIR_F64_CASES = [c for c in F64_CASES if c[0] == "nki_batched_pair_energy"]


def _cast_warnings(record):
    return [w for w in record if "precision is reduced to float32" in str(w.message)]


@pytest.fixture
def fresh_warned(monkeypatch):
    monkeypatch.setattr(dispatch, "_fp32_cast_warned", set())


def _path_fixture(request, path):
    return request.getfixturevalue(
        {"native": "stubs", "xla": "xla_stubs", "sim": "sim_stubs"}[path]
    )


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("name,d,e", AUTO_F64_CASES, ids=[_id(c) for c in AUTO_F64_CASES])
def test_auto_float64_runs_torch_at_fp64(request, fresh_warned, path, name, d, e):
    s = _path_fixture(request, path)
    fn, build, _ = APIS[name]
    args = build(d, e)
    ref = _run(fn, args, "pytorch")
    s.calls.clear()
    dispatch.set_backend("auto")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(*args)
    assert not s.calls, f"float64 under auto reached kernels {s.calls}"
    assert not _cast_warnings(rec)
    assert got.dtype == ref.dtype
    # float64 accuracy: torch.testing's float64 tolerances (rtol 1e-7, atol 1e-7).
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("name,d,e", F64_CASES, ids=[_id(c) for c in F64_CASES])
def test_nki_float64_casts_to_fp32_and_warns_once(request, fresh_warned, path, name, d, e):
    s = _path_fixture(request, path)
    fn, build, _ = APIS[name]
    args = build(d, e)
    ref = _run(fn, args, "pytorch")
    s.calls.clear()
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        outs = [fn(*args) for _ in range(3)]
    assert s.calls, f"{name}: float64 under nki did not reach a kernel"
    assert not s.violations, s.violations  # every kernel operand was float32
    msgs = _cast_warnings(rec)
    assert len(msgs) == 1, [str(w.message) for w in msgs]
    assert msgs[0].category is UserWarning
    assert F64_KERNEL_APIS[name] in str(msgs[0].message)
    for got in outs:
        assert got.dtype == ref.dtype, (got.dtype, ref.dtype)
        assert got.shape == ref.shape


@pytest.mark.parametrize("_n,d,e", PAIR_F64_CASES, ids=[_id(c) for c in PAIR_F64_CASES])
def test_nki_float64_sym_route_casts(stubs, fresh_warned, monkeypatch, _n, d, e):
    monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "sym")
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    fn, build, _ = APIS["nki_batched_pair_energy"]
    args = build(d, e)
    ref = _run(fn, args, "pytorch")
    stubs.calls.clear()
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = [fn(*args) for _ in range(2)]
    assert "pair_energy_sym" in stubs.calls
    assert not stubs.violations
    assert len(_cast_warnings(rec)) == 1
    assert all(g.dtype == ref.dtype for g in got)


def test_nki_symm_float64_runs_torch_without_warning(stubs, fresh_warned):
    fn, build, _ = APIS["symm"]
    args = build(F64, F32)
    ref = _run(fn, args, "pytorch")
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(*args)
    assert not stubs.calls and not _cast_warnings(rec)
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


def test_nki_mixed_float64_gemm_keeps_torch_semantics(stubs, fresh_warned):
    """float64 @ float32 is not cast: it runs (and raises) as the pytorch backend does."""
    A, B = _rand((9, 7), F64), _rand((7, 5), F32, 1)
    with pytest.raises(RuntimeError):
        _run(trnblas.nki.nki_gemm, (A, B), "pytorch")
    with pytest.raises(RuntimeError):
        _run(trnblas.nki.nki_gemm, (A, B), "nki")
    assert not stubs.calls


@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("name,d,e", F64_CASES, ids=[_id(c) for c in F64_CASES])
def test_require_nki_float64_raises_before_cast(
    stubs, fresh_warned, monkeypatch, backend, name, d, e
):
    """TRNBLAS_REQUIRE_NKI=1 takes precedence over the explicit-nki cast."""
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    fn, build, _ = APIS[name]
    args = build(d, e)
    dispatch.set_backend(backend)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        if backend == "nki" or APIS[name][2]:
            with pytest.raises(NotImplementedError):
                fn(*args)
        else:
            # Native `auto` GEMM family never picks a kernel for host operands
            # (aten / host CPU, fp32 included), so there is nothing to refuse:
            # float64 computes in torch.
            fn(*args)
    assert not stubs.calls
    assert not _cast_warnings(rec)


@pytest.mark.parametrize("name,d,e", F64_CASES, ids=[_id(c) for c in F64_CASES])
def test_require_nki_float64_raises_on_xla_auto(xla_stubs, monkeypatch, name, d, e):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    fn, build, _ = APIS[name]
    with pytest.raises(NotImplementedError):
        _run(fn, build(d, e), "auto")
    assert not xla_stubs.calls


# ── result dtype == pytorch backend dtype (trnblas#89) ────────────────────────

RESULT_DTYPE_PAIRS = [
    (F32, F32),
    (F64, F64),
    (F32, F64),
    (F64, F32),
    (BF16, BF16),
    (F16, F16),
    (BF16, F32),
    (F16, F32),
    (F32, BF16),
    (F32, F16),
    (BF16, F64),
]
PAIR_ROUTES = ["default", "chunked", "sym", "multicore_sym", "multicore_aten"]
RESULT_CASES = [(n, "default", d, e) for n in ENERGY_APIS[:2] for d, e in RESULT_DTYPE_PAIRS]
RESULT_CASES += [
    ("nki_batched_pair_energy", r, d, e) for r in PAIR_ROUTES for d, e in RESULT_DTYPE_PAIRS
]


@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize(
    "name,route,d,e",
    RESULT_CASES,
    ids=[f"{c[1]}-{_id((c[0], c[2], c[3]))}" for c in RESULT_CASES],
)
def test_energy_result_dtype_matches_pytorch(stubs, monkeypatch, backend, name, route, d, e):
    if route == "chunked":
        monkeypatch.setenv("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", "0")
    if route in ("sym", "multicore_sym", "multicore_aten"):
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "aten" if route == "multicore_aten" else "sym")
        monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    if route.startswith("multicore"):
        cpu = torch.device("cpu")
        monkeypatch.setattr(multicore, "pair_energy_devices", lambda B: [cpu, cpu])
    fn, build, _ = APIS[name]
    args = build(d, e)
    ref = _run(fn, args, "pytorch")
    got = _run(fn, args, backend)
    assert not stubs.violations, stubs.violations
    assert got.dtype == ref.dtype, (got.dtype, ref.dtype)
    assert got.shape == ref.shape


@pytest.mark.parametrize("d", [F32, F64, BF16, F16], ids=str)
@pytest.mark.parametrize("name", ["gemm", "batched_gemm", "syrk", "trsm", "nki_gemm"])
@pytest.mark.parametrize("backend", ["nki", "auto"])
def test_gemm_result_dtype_matches_pytorch(stubs, name, d, backend):
    fn, build, _ = APIS[name]
    args = build(d, F32)
    ref, ref_exc = _pytorch(fn, args)
    if ref_exc is not None:  # torch has no half-precision CPU triangular solve
        assert name == "trsm" and d in (BF16, F16)
        return
    got = _run(fn, args, backend)
    assert got.dtype == ref.dtype == d


# ── float64 on the neuron device (trnblas#89 amendment) ──────────────────────
#
# Device aten computes float64 at fp32 precision on Trainium, so under `auto`
# a float64 / complex128 operand on the neuron device is copied to the host,
# computed there in host torch at float64, and returned on its device in
# float64, with one UserWarning per op. TRNBLAS_FP64_ON_DEVICE=1 keeps device
# aten. The neuron device is faked on CPU: `dispatch._is_neuron` reports the
# tensors the test marks, and the host / return transfers are spies.

HOST_MSG = "float64 on the neuron device is computed on the host for float64 accuracy"

# Every public op (trnblas.* and trnblas.nki.*): name -> (callable, builder).
DEV_APIS = {n: (fn, build) for n, (fn, build, _) in APIS.items()}
DEV_APIS.update(
    {
        "axpy": (
            lambda x, y: trnblas.axpy(-1.5, x, y),
            lambda d, e: (_rand((9,), d), _rand((9,), d, 1)),
        ),
        "dot": (trnblas.dot, lambda d, e: (_rand((9,), d), _rand((9,), d, 1))),
        "nrm2": (trnblas.nrm2, lambda d, e: (_rand((9,), d),)),
        "scal": (lambda x: trnblas.scal(0.25, x), lambda d, e: (_rand((9,), d),)),
        "asum": (trnblas.asum, lambda d, e: (_rand((9,), d),)),
        "iamax": (trnblas.iamax, lambda d, e: (_rand((9,), d),)),
        "gemv": (
            lambda A, x, y: trnblas.gemv(1.5, A, x, beta=0.5, y=y),
            lambda d, e: (_rand((6, 4), d), _rand((4,), d, 1), _rand((6,), d, 2)),
        ),
        "symv": (
            lambda A, x: trnblas.symv(1.5, A, x),
            lambda d, e: (_rand((5, 5), d), _rand((5,), d, 1)),
        ),
        "trmv": (trnblas.trmv, lambda d, e: (_tri(5, d), _rand((5,), d, 1))),
        "ger": (
            lambda x, y, A: trnblas.ger(2.0, x, y, A),
            lambda d, e: (_rand((6,), d), _rand((4,), d, 1), _rand((6, 4), d, 2)),
        ),
        "trmm": (
            lambda A, B: trnblas.trmm(1.5, A, B),
            lambda d, e: (_tri(6, d), _rand((6, 4), d, 1)),
        ),
        "syrk_C": (
            lambda A, C: trnblas.syrk(1.5, A, beta=0.5, C=C),
            lambda d, e: (_rand((6, 4), d), _spd(6, d)),
        ),
    }
)
# The trnblas.* entry point whose warning each case carries.
DEV_OP = {n: n.removesuffix("_transA").removesuffix("_trans").removesuffix("_C") for n in DEV_APIS}
_NO_C128 = set(ENERGY_APIS) | {"metric_inv_half", "nki_metric_inv_half", "iamax"}
DEV_CASES = [(n, F64, F32) for n in DEV_APIS]
DEV_CASES += [(n, F32, F64) for n in ENERGY_APIS]
DEV_CASES += [(n, torch.complex128, F32) for n in DEV_APIS if n not in _NO_C128]


class FakeNeuron:
    """Marks CPU tensors as neuron-resident and records the host transfers."""

    def __init__(self):
        self.keep: list[torch.Tensor] = []  # keeps ids unique while marked
        self.ids: set[int] = set()
        self.down: list[torch.Tensor] = []
        self.up: list[torch.device] = []

    def mark(self, t):
        if isinstance(t, torch.Tensor):
            self.keep.append(t)
            self.ids.add(id(t))
        return t

    def is_neuron(self, t):
        return id(t) in self.ids

    def to_host(self, t):
        self.down.append(t)
        return t.clone()  # a distinct, unmarked (host) tensor

    def from_host(self, t, device):
        self.up.append(torch.device(device))
        return self.mark(t.clone())


@pytest.fixture
def fake_neuron(stubs, monkeypatch):
    f = FakeNeuron()
    monkeypatch.setattr(dispatch, "_is_neuron", f.is_neuron)
    monkeypatch.setattr(dispatch, "_fp64_to_host", f.to_host)
    monkeypatch.setattr(dispatch, "_fp64_from_host", f.from_host)
    monkeypatch.setattr(dispatch, "_fp64_host_warned", set())
    monkeypatch.setattr(dispatch, "_fp32_cast_warned", set())
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", False)

    def no_upload(*a, **k):
        raise AssertionError("float64 under auto must not be uploaded to device aten")

    monkeypatch.setattr(dispatch, "_upload", no_upload)
    f.stubs = stubs
    return f


def _host_warnings(record):
    return [w for w in record if HOST_MSG in str(w.message)]


def _dev_case(name, d, e):
    fn, build = DEV_APIS[name]
    args = build(d, e)
    ref, exc = _pytorch(fn, args)
    if exc is not None:
        pytest.skip(f"the pytorch backend refuses {name} {d}: {exc}")
    return fn, args, ref


@pytest.mark.parametrize("name,d,e", DEV_CASES, ids=[_id(c) for c in DEV_CASES])
def test_auto_device_float64_computes_on_host(fake_neuron, name, d, e):
    fn, args, ref = _dev_case(name, d, e)
    dev_args = [fake_neuron.mark(a) for a in args]
    dispatch.set_backend("auto")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        outs = [fn(*dev_args) for _ in range(3)]
    assert not fake_neuron.stubs.calls, fake_neuron.stubs.calls  # no kernel
    # Host path: every neuron-resident operand came down on every call.
    assert len(fake_neuron.down) == 3 * len(dev_args)
    msgs = _host_warnings(rec)
    assert len(msgs) == 1, [str(w.message) for w in msgs]  # once per op
    assert msgs[0].category is UserWarning
    assert f"trnblas {DEV_OP[name]}:" in str(msgs[0].message)
    for got in outs:
        if not isinstance(ref, torch.Tensor):  # iamax: a Python int
            assert got == ref
            continue
        assert fake_neuron.is_neuron(got)  # back on the device ...
        assert fake_neuron.up[-1] == args[0].device  # ... the operands came from
        assert got.dtype == ref.dtype and got.dtype in (F64, torch.complex128)
        # float64 accuracy against the float64 CPU reference.
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["gemm", "nki_batched_pair_energy", "trsm", "dot"])
def test_fp64_on_device_override_keeps_device_path(fake_neuron, monkeypatch, name):
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", True)
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)  # device routes
    fn, args, ref = _dev_case(name, F64, F32)
    dev_args = [fake_neuron.mark(a) for a in args]
    dispatch.set_backend("auto")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(*dev_args)
    assert not fake_neuron.down and not fake_neuron.up  # no host copy
    assert not _host_warnings(rec)
    assert not fake_neuron.stubs.calls  # device aten, not a kernel
    assert got.dtype == ref.dtype == F64


@pytest.mark.parametrize("backend", ["nki", "pytorch"])
@pytest.mark.parametrize("name", ["gemm", "symm", "nki_mp2_energy", "axpy"])
def test_device_float64_host_path_is_auto_only(fake_neuron, backend, name):
    fn, args, ref = _dev_case(name, F64, F32)
    dev_args = [fake_neuron.mark(a) for a in args]
    dispatch.set_backend(backend)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(*dev_args)
    assert not fake_neuron.down and not fake_neuron.up
    assert not _host_warnings(rec)
    assert got.dtype == ref.dtype


# Explicit `nki` on an op with no NKI kernel (Level 1/2, symm, trmm): device
# float64 / complex128 runs on device aten at fp32 precision. That is the same
# precision loss as the kernel ops' float32 cast, so it warns once per op too.
NO_KERNEL_OPS = {
    "axpy", "dot", "nrm2", "scal", "asum", "iamax",
    "gemv", "symv", "trmv", "ger", "symm", "trmm",
}  # fmt: skip
NO_KERNEL_DEV_CASES = [(n, d, e) for n, d, e in DEV_CASES if n in NO_KERNEL_OPS]


@pytest.mark.parametrize("name,d,e", NO_KERNEL_DEV_CASES, ids=[_id(c) for c in NO_KERNEL_DEV_CASES])
def test_nki_device_float64_without_kernel_warns_once(fake_neuron, name, d, e):
    fn, args, ref = _dev_case(name, d, e)
    dev_args = [fake_neuron.mark(a) for a in args]
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        outs = [fn(*dev_args) for _ in range(3)]
    msgs = _cast_warnings(rec)
    assert len(msgs) == 1, [str(w.message) for w in rec]
    assert msgs[0].category is UserWarning
    text = str(msgs[0].message)
    assert f"trnblas {name}:" in text and "no NKI kernel" in text
    assert not _host_warnings(rec)
    assert not fake_neuron.down and not fake_neuron.up  # stays on the device
    assert not fake_neuron.stubs.calls  # no kernel
    for got in outs:
        if isinstance(ref, torch.Tensor):
            assert got.dtype == ref.dtype


@pytest.mark.parametrize("name", sorted(NO_KERNEL_OPS))
def test_no_kernel_cast_warning_is_nki_device_only(fake_neuron, name):
    """No fp32 warning for host float64 under `nki`, device float32 under
    `nki`, or device float64 under `auto` (host, its own warning) / `pytorch`."""
    for backend, d, marked in [
        ("nki", F64, False),
        ("nki", F32, True),
        ("auto", F64, True),
        ("pytorch", F64, True),
    ]:
        fn, args, _ = _dev_case(name, d, F32)
        if marked:
            args = [fake_neuron.mark(a) for a in args]
        dispatch.set_backend(backend)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fn(*args)
        assert not _cast_warnings(rec), (backend, d, marked, [str(w.message) for w in rec])


@pytest.mark.parametrize("name", ["gemm", "syrk", "trsm"])
def test_kernel_ops_keep_their_own_cast_warning(fake_neuron, name):
    """Device float64 under `nki` on an op with a kernel: the kernel's float32
    cast warning, once, not the no-kernel one."""
    fn, args, _ = _dev_case(name, F64, F32)
    dev_args = [fake_neuron.mark(a) for a in args]
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(3):
            fn(*dev_args)
    msgs = _cast_warnings(rec)
    assert len(msgs) == 1, [str(w.message) for w in rec]
    assert "no NKI kernel" not in str(msgs[0].message)
    assert fake_neuron.stubs.calls  # the FP32 kernel ran


# TRNBLAS_REQUIRE_NKI=1 with float64 / complex128 on the neuron device: every
# public op raises, before any host copy, kernel or device aten. The routes
# with no kernel (`auto` gemm / batched_gemm / syrk / symm / trsm on device
# aten, Level 1/2) used to compute them at fp32 precision with no error.
REQUIRE_DEV_CASES = [(n, d, e) for n, d, e in DEV_CASES]


@pytest.mark.parametrize("backend", ["auto", "nki"])
@pytest.mark.parametrize("name,d,e", REQUIRE_DEV_CASES, ids=[_id(c) for c in REQUIRE_DEV_CASES])
def test_require_nki_device_float64_raises(fake_neuron, monkeypatch, backend, name, d, e):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)  # device routes
    fn, args, _ = _dev_case(name, d, e)
    dev_args = [fake_neuron.mark(a) for a in args]
    dispatch.set_backend(backend)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError, match="TRNBLAS_REQUIRE_NKI") as err:
            fn(*dev_args)
    assert f"trnblas {DEV_OP[name]}:" in str(err.value)
    assert "neuron device" in str(err.value)
    assert not fake_neuron.down and not fake_neuron.up  # no host copy
    assert not fake_neuron.stubs.calls  # no kernel
    assert not _host_warnings(rec) and not _cast_warnings(rec)


@pytest.mark.parametrize("name", ["gemm", "symm", "dot", "trsm"])
def test_require_nki_device_float64_raises_ahead_of_the_override(fake_neuron, monkeypatch, name):
    """REQUIRE_NKI takes precedence over TRNBLAS_FP64_ON_DEVICE=1 (device aten)."""
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", True)
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    fn, args, _ = _dev_case(name, F64, F32)
    dispatch.set_backend("auto")
    with pytest.raises(NotImplementedError, match="TRNBLAS_REQUIRE_NKI"):
        fn(*[fake_neuron.mark(a) for a in args])


@pytest.mark.parametrize("name", ["gemm", "symm", "dot", "trsm"])
def test_require_nki_device_refusal_leaves_other_inputs(fake_neuron, monkeypatch, name):
    """Under REQUIRE_NKI: device float32 runs, host float64 keeps its host
    torch route, and the pytorch backend ignores the variable."""
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    dispatch.set_backend("auto")
    fn, args32, _ = _dev_case(name, F32, F32)
    fn(*[fake_neuron.mark(a) for a in args32])
    fn, args, ref = _dev_case(name, F64, F32)
    torch.testing.assert_close(fn(*args), ref, rtol=1e-12, atol=1e-12)
    dispatch.set_backend("pytorch")
    got = fn(*[fake_neuron.mark(a) for a in args])
    assert got.dtype == ref.dtype
    assert not fake_neuron.down


@pytest.mark.parametrize("name", ["gemm", "symm", "nki_mp2_energy", "metric_inv_half"])
def test_host_float64_unchanged(fake_neuron, name):
    fn, args, ref = _dev_case(name, F64, F32)  # host-resident: not marked
    dispatch.set_backend("auto")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(*args)
    assert not fake_neuron.down and not fake_neuron.up and not _host_warnings(rec)
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


def test_device_float32_stays_on_device(fake_neuron):
    fn, args, _ = _dev_case("gemm", F32, F32)
    dispatch.set_backend("auto")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*[fake_neuron.mark(a) for a in args])
    assert not fake_neuron.down and not _host_warnings(rec)


@pytest.mark.parametrize("policy", ["aten", "lnc"])
@pytest.mark.parametrize("name", ["gemm", "batched_gemm", "syrk", "symm", "trsm", "nki_gemm"])
def test_forced_aten_policy_keeps_host_float64_on_host(fake_neuron, monkeypatch, policy, name):
    """TRNBLAS_NATIVE_GEMM=aten|lnc no longer uploads host float64 to device
    aten (fp32 precision); TRNBLAS_FP64_ON_DEVICE=1 restores the upload."""
    monkeypatch.setenv("TRNBLAS_NATIVE_GEMM", policy)
    fn, args, ref = _dev_case(name, F64, F32)
    dispatch.set_backend("auto")
    got = _run(fn, args, "auto")  # `_upload` raises if reached
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
    uploads = []
    monkeypatch.setattr(dispatch, "_upload", lambda t: uploads.append(t) or t)
    monkeypatch.setattr(dispatch, "_FP64_ON_DEVICE", True)
    _run(fn, args, "auto")
    # trsm takes no `lnc` policy: host operands solve on the host either way.
    assert uploads or (name, policy) == ("trsm", "lnc")


# Explicit `nki` on an op WITH a kernel (#103): float64 / complex128 on the
# neuron device that does not take the kernel's float32 cast (complex128,
# float64 mixed with another dtype, a shape or side with no kernel route, an
# fp64 `C`, the size-chosen energy aten route) runs on device aten at fp32
# precision. It warns once per op, in the shared dedup set and wording, and
# TRNBLAS_REQUIRE_NKI=1 raises. Routes that compute only on the host at full
# precision (left `trsm` on complex128, the `metric_inv_half` host route) do
# not warn.
C128 = torch.complex128
I64 = torch.int64
DEVICE_ATEN_MSG = "on the neuron device run on device aten; precision is reduced to float32"


def _device_aten_warnings(record):
    return [w for w in record if DEVICE_ATEN_MSG in str(w.message)]


def _right_trsm(A, B):
    return trnblas.trsm(1.0, A, B, side="right")


def _nki_gemm_C(A, B, C):
    return trnblas.gemm(1.5, A, B, beta=0.5, C=C)


def _syrk_C(A, C):
    return trnblas.syrk(1.5, A, beta=0.5, C=C)


def _mp2(d, e):
    return APIS["nki_mp2_energy"][1](d, e)


def _fused(d, e):
    return APIS["nki_fused_gemm_energy"][1](d, e)


def _pair(d, e):
    return APIS["nki_batched_pair_energy"][1](d, e)


# id -> (op in the warning, callable, args builder, extra setup)
KERNEL_ROUTES = {
    # complex128: `_kernel_dtypes_ok` is false and `_nki_casts_fp64` needs all-float64.
    "gemm-c128": ("gemm", APIS["gemm"][0], lambda: APIS["gemm"][1](C128, F32), None),
    "nki_gemm-c128": ("gemm", APIS["nki_gemm"][0], lambda: APIS["nki_gemm"][1](C128, F32), None),
    "batched_gemm-c128": (
        "batched_gemm",
        APIS["batched_gemm"][0],
        lambda: APIS["batched_gemm"][1](C128, F32),
        None,
    ),
    "nki_batched_gemm-c128": (
        "batched_gemm",
        APIS["nki_batched_gemm"][0],
        lambda: APIS["nki_batched_gemm"][1](C128, F32),
        None,
    ),
    "syrk-c128": ("syrk", APIS["syrk"][0], lambda: APIS["syrk"][1](C128, F32), None),
    "syrk_trans-c128": ("syrk", APIS["syrk_trans"][0], lambda: APIS["syrk"][1](C128, F32), None),
    "nki_syrk-c128": ("syrk", APIS["nki_syrk"][0], lambda: APIS["nki_syrk"][1](C128, F32), None),
    # Shapes with no kernel route under `nki`: batched / 1-D `gemm` operands go
    # straight to torch.matmul, all-float64 included.
    "gemm-3d-f64": (
        "gemm",
        lambda A, B: trnblas.gemm(1.0, A, B),
        lambda: (_rand((2, 9, 7), F64), _rand((2, 7, 5), F64, 1)),
        None,
    ),
    "gemm-1d-f64": (
        "gemm",
        lambda A, b: trnblas.gemm(1.0, A, b),
        lambda: (_rand((9, 7), F64), _rand((7,), F64, 1)),
        None,
    ),
    "nki_batched_gemm-1d-f64": (
        "batched_gemm",
        trnblas.nki.nki_batched_gemm,
        lambda: (_rand((2, 9, 7), F64), _rand((7,), F64, 1)),
        None,
    ),
    # float64 mixed with another dtype: an fp64 C added to the fp32 kernel result.
    "gemm-f32-C_f64": (
        "gemm",
        _nki_gemm_C,
        lambda: (_rand((9, 7), F32), _rand((7, 5), F32, 1), _rand((9, 5), F64, 2)),
        None,
    ),
    "gemm-f32-C_c128": (
        "gemm",
        _nki_gemm_C,
        lambda: (_rand((9, 7), F32), _rand((7, 5), F32, 1), _rand((9, 5), C128, 2)),
        None,
    ),
    "syrk-f32-C_f64": ("syrk", _syrk_C, lambda: (_rand((6, 4), F32), _spd(6, F64)), None),
    # trsm: side='right' never takes the blocked kernel route.
    "trsm-right-f64": (
        "trsm",
        _right_trsm,
        lambda: (_tri(130, F64), _rand((3, 130), F64, 1)),
        None,
    ),
    "trsm-right-c128": (
        "trsm",
        _right_trsm,
        lambda: (_tri(130, C128), _rand((3, 130), C128, 1)),
        None,
    ),
    # trsm, float64 mixed with another dtype: `_trsm_route` -> "torch".
    "trsm-f64-B_f32": (
        "trsm",
        APIS["trsm"][0],
        lambda: (_tri(130, F64), _rand((130, 3), F32, 1)),
        None,
    ),
    "nki_trsm-f32-B_c128": (
        "trsm",
        APIS["nki_trsm"][0],
        lambda: (_tri(130, F32), _rand((130, 3), C128, 1)),
        None,
    ),
    # metric_inv_half device route: the Cholesky runs on device aten. Before,
    # float64 warned only as "trsm" (nothing once trsm had warned), complex128
    # not at all.
    "metric_inv_half-device-f64": (
        "metric_inv_half",
        APIS["metric_inv_half"][0],
        lambda: APIS["metric_inv_half"][1](F64, F32),
        None,
    ),
    "nki_metric_inv_half-device-c128": (
        "metric_inv_half",
        APIS["nki_metric_inv_half"][0],
        lambda: APIS["nki_metric_inv_half"][1](C128, F32),
        None,
    ),
    # Energy kernels: complex128, and float64 mixed with a dtype that is not
    # float64 (`_nki_casts_fp64(*bad)` false), run the torch reference.
    "nki_mp2_energy-c128": ("mp2_energy", APIS["nki_mp2_energy"][0], lambda: _mp2(C128, F32), None),
    "nki_mp2_energy-eps_c128": (
        "mp2_energy",
        APIS["nki_mp2_energy"][0],
        lambda: _mp2(F32, C128),
        None,
    ),
    "nki_mp2_energy-f64-eps_i64": (
        "mp2_energy",
        APIS["nki_mp2_energy"][0],
        lambda: _mp2(F64, I64),
        None,
    ),
    "nki_fused_gemm_energy-c128": (
        "fused_gemm_energy",
        APIS["nki_fused_gemm_energy"][0],
        lambda: _fused(C128, F32),
        None,
    ),
    "nki_fused_gemm_energy-f64-eps_c128": (
        "fused_gemm_energy",
        APIS["nki_fused_gemm_energy"][0],
        lambda: _fused(F64, C128),
        None,
    ),
    "nki_batched_pair_energy-c128": (
        "batched_pair_energy",
        APIS["nki_batched_pair_energy"][0],
        lambda: _pair(C128, F32),
        None,
    ),
    "nki_batched_pair_energy-f64-eps_i64": (
        "batched_pair_energy",
        APIS["nki_batched_pair_energy"][0],
        lambda: _pair(F64, I64),
        None,
    ),
    # Device B on the native aten route (chosen by size, or forced): float64
    # is not cast there, and float64 eps are not rounded.
    "nki_batched_pair_energy-aten-f64": (
        "batched_pair_energy",
        APIS["nki_batched_pair_energy"][0],
        lambda: _pair(F64, F32),
        "pair_aten",
    ),
    "nki_batched_pair_energy-aten-eps_f64": (
        "batched_pair_energy",
        APIS["nki_batched_pair_energy"][0],
        lambda: _pair(F32, F64),
        "pair_aten",
    ),
    "nki_batched_pair_energy-aten-c128": (
        "batched_pair_energy",
        APIS["nki_batched_pair_energy"][0],
        lambda: _pair(C128, F32),
        "pair_device",
    ),
    "nki_batched_pair_energy-multicore_aten-f64": (
        "batched_pair_energy",
        APIS["nki_batched_pair_energy"][0],
        lambda: _pair(F64, F32),
        "pair_multicore_aten",
    ),
}


def _route_setup(monkeypatch, setup):
    if setup is None:
        return
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    if setup == "pair_aten":
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "aten")
    if setup == "pair_multicore_aten":
        monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", "aten")
        cpu = torch.device("cpu")
        monkeypatch.setattr(multicore, "pair_energy_devices", lambda B: [cpu, cpu])


def _kernel_route_case(monkeypatch, case):
    op, fn, build, setup = KERNEL_ROUTES[case]
    _route_setup(monkeypatch, setup)
    args = build()
    ref, exc = _pytorch(fn, args)
    assert exc is None, f"{case}: the pytorch backend raised {exc!r}"
    return op, fn, args, ref


@pytest.mark.parametrize("case", sorted(KERNEL_ROUTES))
def test_nki_device_float64_off_the_cast_warns_once(fake_neuron, monkeypatch, case):
    op, fn, args, ref = _kernel_route_case(monkeypatch, case)
    dev_args = [fake_neuron.mark(a) for a in args]
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        outs = [fn(*dev_args) for _ in range(3)]
    msgs = [w for w in _cast_warnings(rec) if f"trnblas {op}:" in str(w.message)]
    assert len(msgs) == 1, [str(w.message) for w in rec]
    assert msgs[0].category is UserWarning
    assert DEVICE_ATEN_MSG in str(msgs[0].message)
    assert msgs[0].filename == __file__  # points at the caller, not trnblas
    assert not _host_warnings(rec)
    assert not fake_neuron.down and not fake_neuron.up  # stays on the device
    assert not fake_neuron.stubs.violations
    for got in outs:
        assert got.dtype == ref.dtype
        if not fake_neuron.stubs.calls:  # the stub kernels return zeros
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("case", sorted(KERNEL_ROUTES))
def test_nki_device_float64_off_the_cast_quiet_elsewhere(fake_neuron, monkeypatch, case):
    """No device-aten warning for the same inputs on the host under `nki`, and
    no fp32 warning on the device under `pytorch` or `auto` (host warning only)."""
    _, fn, args, _ = _kernel_route_case(monkeypatch, case)
    for backend, marked in [("nki", False), ("pytorch", True), ("auto", True)]:
        a = [fake_neuron.mark(t) for t in args] if marked else args
        dispatch.set_backend(backend)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fn(*a)
        bad = _device_aten_warnings(rec) if backend == "nki" else _cast_warnings(rec)
        assert not bad, (backend, marked, [str(w.message) for w in rec])


@pytest.mark.parametrize("backend", ["auto", "nki"])
@pytest.mark.parametrize("case", sorted(KERNEL_ROUTES))
def test_require_nki_device_float64_off_the_cast_raises(fake_neuron, monkeypatch, backend, case):
    op, fn, args, _ = _kernel_route_case(monkeypatch, case)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    dispatch.set_backend(backend)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError, match="TRNBLAS_REQUIRE_NKI"):
            fn(*[fake_neuron.mark(a) for a in args])
    assert not fake_neuron.down and not fake_neuron.stubs.calls
    assert not _cast_warnings(rec)


def test_nki_kernel_cast_and_route_warning_share_one_entry(fake_neuron):
    """One warning per op across routes: a complex128 gemm after a float64
    gemm (kernel cast) does not warn again, and vice versa for trsm."""
    calls = [_dev_case(n, d, F32)[:2] for n, d in [("gemm", F64), ("gemm", C128)]]
    calls += [_dev_case("nki_gemm", d, F32)[:2] for d in (C128, F64)]  # same op as gemm
    calls.append((_right_trsm, (_tri(130, F64), _rand((3, 130), F64))))
    calls.append(_dev_case("trsm", F64, F32)[:2])
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for fn, args in calls:
            fn(*[fake_neuron.mark(a) for a in args])
    msgs = [str(w.message) for w in _cast_warnings(rec)]
    assert len(msgs) == 2, msgs
    assert msgs[0].startswith("trnblas gemm:") and "on the FP32 NKI kernel" in msgs[0]
    assert msgs[1].startswith("trnblas trsm:") and "device aten" in msgs[1]


def test_nki_metric_inv_half_warns_under_its_own_name(fake_neuron):
    """The device route's trsm cast used to be its only warning, keyed "trsm":
    after a plain trsm warned, metric_inv_half on device float64 was silent."""
    fn, args, _ = _dev_case("trsm", F64, F32)
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*[fake_neuron.mark(a) for a in args])
        trnblas.metric_inv_half(fake_neuron.mark(_spd(130, F64)))
    msgs = [str(w.message) for w in _cast_warnings(rec)]
    assert [m.split(":")[0] for m in msgs] == ["trnblas trsm", "trnblas metric_inv_half"], msgs


@pytest.mark.parametrize(
    "name,d", [("trsm", C128), ("nki_trsm", C128), ("metric_inv_half", F64)], ids=str
)
def test_nki_device_float64_on_a_host_route_stays_quiet(fake_neuron, monkeypatch, name, d):
    """Left trsm on complex128 and the metric_inv_half host route compute on
    the host at full precision: no fp32 warning."""
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)  # metric host route
    fn, args, ref = _dev_case(name, d, F32)
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(*[fake_neuron.mark(a) for a in args])
    assert not _cast_warnings(rec), [str(w.message) for w in rec]
    assert not fake_neuron.stubs.calls
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)


def test_nki_mixed_float64_gemm_that_raises_does_not_warn(fake_neuron):
    """float64 @ float32 raises as torch.matmul does; nothing ran, no warning."""
    A, B = fake_neuron.mark(_rand((9, 7), F64)), fake_neuron.mark(_rand((7, 5), F32, 1))
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            trnblas.gemm(1.0, A, B)
    assert not _cast_warnings(rec)


# trnblas.parallel.pair_energy is not wrapped by `_fp64_host_guard` (its
# all-reduce needs the caller's group); it applies the same rules itself.
PARALLEL_CASES = {
    "aten-f64": (F64, F32, "aten", True),
    "aten-c128": (C128, F32, "aten", True),
    "sym-eps_f64": (F32, F64, "sym", True),
    "sym-f64": (F64, F32, "sym", False),  # the kernel cast's own warning
}


@pytest.mark.parametrize("case", sorted(PARALLEL_CASES))
def test_parallel_pair_energy_device_float64_under_nki(fake_neuron, monkeypatch, case):
    from trnblas import parallel

    d, e, route, device_msg = PARALLEL_CASES[case]
    monkeypatch.setenv("TRNBLAS_PAIR_ENERGY", route)
    monkeypatch.setattr(dispatch, "_on_neuron_device", lambda *t: True)
    monkeypatch.setattr(parallel, "_rank_world", lambda group: (0, 1))
    monkeypatch.setattr(parallel.dist, "all_reduce", lambda t, group=None: t)
    args = [fake_neuron.mark(a) for a in _pair(d, e)]
    dispatch.set_backend("nki")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(3):
            parallel.pair_energy(*args)
    msgs = _cast_warnings(rec)
    assert len(msgs) == 1, [str(w.message) for w in rec]
    assert str(msgs[0].message).startswith("trnblas pair_energy:")
    assert ("device aten" in str(msgs[0].message)) == device_msg
    assert msgs[0].filename == __file__
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    with pytest.raises(NotImplementedError, match="TRNBLAS_REQUIRE_NKI"):
        parallel.pair_energy(*args)
