"""Tail-tile audit (trnsci/trnfft#100, trnsci/trnsci#35): every NKI kernel at
awkward sizes, against a float64 reference.

trnfft found kernels that dropped the partial last tile in M, K or N and
returned wrong numbers without an error. These cases use sizes that are not
tile multiples (1, 127, 129, 200, 1000, 4097; odd K and N; free dims that
are not multiples of 512) on every kernel reachable with
``set_backend("nki")``: the ragged GEMM, the padded tile-config GEMMs, SYRK,
blocked TRSM, the fused MP2 energy, the fused per-pair GEMM energy, the
full-batch and chunked batched-pair kernels, the LNC=2 symmetric pair
kernel and its multi-core split.

Kernel errors are re-raised (``TRNBLAS_REQUIRE_NKI``), so a fallback to
torch cannot hide a failure. Inputs are created on the host
(aws-neuron-sdk#1439). ``CASES`` is also used by a standalone probe that
reports OK / SILENT / LOUD per case.
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch

import trnblas
from trnblas.nki import HAS_NKI
from trnblas.nki import dispatch as D

DEV = "neuron"


def _native() -> bool:
    if not HAS_NKI or D._use_simulator():
        return False
    try:
        return D._get_dispatch_path() == "native"
    except Exception:
        return False


def _host(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _uniform(n, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(n, generator=g)


@contextlib.contextmanager
def nki_mode(env: dict | None = None, attrs: dict | None = None):
    """``set_backend("nki")`` with kernel errors re-raised, plus temporary env
    vars and `dispatch` attributes; everything is restored afterwards."""
    old_backend = trnblas.get_backend()
    old_env = {k: os.environ.get(k) for k in (env or {})}
    attrs = {"_REQUIRE_NKI": True, **(attrs or {})}
    old_attrs = {k: getattr(D, k) for k in attrs}
    trnblas.set_backend("nki")
    os.environ.update(env or {})
    for k, v in attrs.items():
        setattr(D, k, v)
    try:
        yield
    finally:
        trnblas.set_backend(old_backend)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in old_attrs.items():
            setattr(D, k, v)


# Each case returns (out, ref): out from the kernel path, ref in float64.


def _gemm(M, K, N, device_inputs=False, ragged=True):
    def run():
        A, B = _host(M, K), _host(K, N, seed=1)
        a, b = (A.to(DEV), B.to(DEV)) if device_inputs else (A, B)
        with nki_mode(attrs={"_RAGGED_ENABLED": ragged}):
            out = trnblas.gemm(1.0, a, b)
        return out.cpu(), A.double() @ B.double()

    return run


def _gemm_tile(tile, M, K, N):
    def run():
        A, B = _host(M, K), _host(K, N, seed=1)

        def launch(a, b):
            return D._launch(D._get_gemm_kernel(*tile), D._gemm_op_name(*tile), a, b)

        with nki_mode():
            out = D._run_padded_gemm(A, B, launch, *tile)
        return out.cpu(), A.double() @ B.double()

    return run


def _batched_gemm(bt, M, K, N):
    def run():
        A, B = _host(bt, M, K), _host(bt, K, N, seed=1)
        with nki_mode():
            out = trnblas.batched_gemm(1.0, A, B)
        return out.cpu(), A.double() @ B.double()

    return run


def _syrk(M, K, symmetrize=False):
    def run():
        A = _host(M, K)
        with nki_mode():
            out = D.nki_syrk(A, symmetrize=symmetrize)
        return out.cpu(), A.double() @ A.double().T

    return run


def _trsm(M, nrhs, uplo, trans):
    def run():
        g = torch.Generator().manual_seed(3)
        A = 0.1 * torch.randn(M, M, generator=g) / max(1, M) ** 0.5
        A = A + torch.diag(1.0 + torch.rand(M, generator=g))
        B = _host(M, nrhs, seed=4)
        with nki_mode():
            X = trnblas.trsm(1.0, A, B, uplo=uplo, trans=trans)
        tri = torch.triu(A) if uplo == "upper" else torch.tril(A)
        op = tri.T if trans else tri
        ref = torch.linalg.solve(op.double(), B.double())
        return X.cpu(), ref

    return run


def _eps(nocc, nvir, seed=5):
    return _uniform(nocc, -2.0, -0.5, seed), _uniform(nvir, 0.5, 3.0, seed + 1)


def _mp2_energy(ic, nocc, nvir):
    def run():
        eo, ev = _eps(nocc, nvir)
        T = 0.1 * _host(ic * nvir, nocc * nvir, seed=7)
        with nki_mode():
            out = D.nki_mp2_energy(T, eo[:ic], eo, ev)
        ref = D._torch_mp2_energy(T.double(), eo[:ic].double(), eo.double(), ev.double())
        return out.cpu().reshape(1), ref.reshape(1)

    return run


def _fused_pair(nvir, naux):
    def run():
        bi, bj = 0.1 * _host(nvir, naux, seed=8), 0.1 * _host(nvir, naux, seed=9)
        _, ev = _eps(1, nvir)
        with nki_mode():
            out = D.nki_fused_gemm_energy(bi, bj, -1.0, -0.75, ev)
        ref = D._torch_fused_pair_energy(bi.double(), bj.double(), -1.0, -0.75, ev.double())
        return out.cpu().reshape(1).double(), ref.reshape(1).double()

    return run


def _pair_energy(nocc, nvir, naux, route, device_inputs=False, chunked=False, cores=None):
    def run():
        B = 0.1 * _host(nocc, nvir, naux, seed=10)
        eo, ev = _eps(nocc, nvir)
        env = {"TRNBLAS_PAIR_ENERGY": route}
        if chunked:
            env["TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD"] = "0"
        b, e_o, e_v = (B.to(DEV), eo.to(DEV), ev.to(DEV)) if device_inputs else (B, eo, ev)
        from trnblas.nki import multicore

        old = multicore._cores_override
        if cores is not None:
            multicore.set_cores(cores)
        try:
            with nki_mode(env=env):
                out = D.nki_batched_pair_energy(b, e_o, e_v)
        finally:
            multicore._cores_override = old
        ref = D._torch_batched_pair_energy(B.double(), eo.double(), ev.double())
        return out.cpu().reshape(1).double(), ref.reshape(1)

    return run


GEMM_SHAPES = [
    (1, 1, 1),
    (127, 129, 513),
    (129, 127, 1),
    (200, 200, 600),
    (1000, 64, 64),
    (4097, 8, 8),
    (7, 300, 5),
    (129, 1000, 129),
    (5, 4097, 3),
    (3, 5, 4097),
    (200, 129, 1000),
]

CASES: list[tuple[str, object, float]] = []
CASES += [(f"gemm_ragged {s}", _gemm(*s), 1e-4) for s in GEMM_SHAPES]
CASES += [
    (f"gemm_ragged_devin {s}", _gemm(*s, device_inputs=True), 1e-4)
    for s in [(129, 127, 513), (4097, 8, 8)]
]
CASES += [
    (f"gemm_padded {s}", _gemm(*s, ragged=False), 1e-4)
    for s in [(1, 1, 1), (129, 127, 513), (200, 200, 600), (4097, 8, 8), (5, 4097, 3)]
]
CASES += [
    (f"gemm_tile{t} {s}", _gemm_tile(t, *s), 1e-4)
    for t in D._TILE_CANDIDATES
    for s in [(129, 127, 513), (200, 129, 1000)]
]
CASES += [
    (f"batched_gemm {s}", _batched_gemm(*s), 1e-4) for s in [(3, 129, 127, 200), (2, 1, 1, 1)]
]
CASES += [
    (f"syrk {s}", _syrk(*s), 1e-4)
    for s in [(1, 1), (127, 3), (129, 129), (200, 1000), (1000, 7), (4097, 8)]
]
CASES += [("syrk_sym (1000, 129)", _syrk(1000, 129, symmetrize=True), 1e-4)]
CASES += [
    (f"trsm {M}x{n} {uplo} trans={tr}", _trsm(M, n, uplo, tr), 1e-4)
    for M, n, uplo, tr in [
        (1, 3, "lower", False),
        (129, 3, "lower", False),
        (200, 129, "upper", False),
        (1000, 7, "lower", True),
        (1000, 513, "upper", True),
    ]
]
CASES += [
    (f"mp2_energy ic,nocc,nvir={s}", _mp2_energy(*s), 1e-4)
    for s in [(1, 1, 1), (2, 3, 127), (1, 2, 129), (2, 3, 200), (1, 2, 131), (1, 1, 1000)]
]
CASES += [
    (f"fused_gemm_energy nvir,naux={s}", _fused_pair(*s), 1e-4)
    for s in [(1, 1), (127, 129), (129, 200), (200, 1000)]
]
CASES += [
    (f"batched_pair nocc,nvir,naux={s}", _pair_energy(*s, "legacy"), 1e-4)
    for s in [(1, 1, 1), (3, 129, 127), (2, 200, 129)]
]
CASES += [
    (f"j_batched(chunked) nocc,nvir,naux={s}", _pair_energy(*s, "legacy", chunked=True), 1e-4)
    for s in [(1, 1, 1), (3, 129, 200)]
]
CASES += [
    (f"pair_energy_sym nocc,nvir,naux={s}", _pair_energy(*s, "sym", device_inputs=True), 1e-4)
    for s in [(1, 1, 1), (2, 127, 129), (3, 129, 200), (5, 200, 131), (3, 500, 129), (7, 129, 1)]
]
CASES += [
    (
        f"pair_energy_multicore[{r}] nocc,nvir,naux={s}",
        _pair_energy(*s, r, device_inputs=True, cores=2),
        1e-4,
    )
    for r in ("sym", "aten")
    for s in [(1, 1, 1), (5, 129, 200), (7, 200, 131)]
]


def max_rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    """max |out - ref| / max |ref| (inf on NaN or shape mismatch)."""
    if tuple(out.shape) != tuple(ref.shape):
        return float("inf")
    d = (out.double() - ref.double()).abs().max().item()
    s = ref.double().abs().max().item()
    if d != d:
        return float("inf")
    return d / s if s > 0 else d


def _needs_multicore(name: str) -> bool:
    return name.startswith("pair_energy_multicore")


@pytest.mark.neuron
@pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path")
@pytest.mark.parametrize("name,run,tol", CASES, ids=[c[0] for c in CASES])
def test_tail(name, run, tol):
    if _needs_multicore(name):
        from trnblas.nki.multicore import _visible_cores

        if _visible_cores() < 2:
            pytest.skip("needs >= 2 visible logical cores")
    out, ref = run()
    err = max_rel_err(out, ref)
    assert err < tol, f"{name}: max rel err {err:.3e} (tol {tol:.0e})"
