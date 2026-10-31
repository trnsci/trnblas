"""Multi-core DF-MP2 pair energy (#24): core selection and the row split.

CPU-only: the per-core devices are all `cpu`, and the `pair_energy_sym`
custom op is replaced with a torch emulation of the kernel, so the split,
the per-core B windows and the weights are checked against the single-core
reference on any host. The on-hardware check is in `test_nki_native.py`.
"""

import pytest
import torch

import trnblas
from trnblas.nki import dispatch, multicore


@pytest.fixture(autouse=True)
def reset_cores(monkeypatch):
    monkeypatch.delenv("TRNBLAS_CORES", raising=False)
    multicore.set_cores(None)
    yield
    multicore.set_cores(None)


def _devices(cores):
    """B's own core first, then stand-ins for other cores: `cpu:1` compares
    unequal to B's `cpu`, so the host-staging branch runs."""
    return [torch.device("cpu")] + [torch.device("cpu", 1)] * (cores - 1)


def _inputs(nocc, nvir, naux, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(nocc, nvir, naux, generator=g) * 0.1
    eps_occ = -torch.sort(torch.rand(nocc, generator=g))[0] - 0.5
    eps_vir = torch.sort(torch.rand(nvir, generator=g))[0] + 0.1
    return B, eps_occ, eps_vir


def _sym_op(bt_i, Bw, eps_occ_i, eps_occ_w, eps_vir_col, eps_vir_row):
    """Torch emulation of `_pair_energy_sym_kernel`: (128, ND) partials, sum in row 0."""
    nd = Bw.shape[0]
    out = torch.zeros(128, nd, dtype=torch.float64)
    bi = bt_i.double()
    for d in range(nd):
        T = bi.T @ Bw[d].double()
        denom = float(eps_occ_i[0, 0]) + float(eps_occ_w[0, d]) - eps_vir_col.double()
        denom = denom - eps_vir_row.double()
        out[0, d] = (T * (2.0 * T - T.T) / denom).sum()
    return out.to(bt_i.dtype)


@pytest.fixture
def emulated_sym(monkeypatch):
    monkeypatch.setattr(dispatch, "_get_native_ops", lambda: {"pair_energy_sym": _sym_op})


class TestCoreSetting:
    def test_default_is_one(self):
        assert trnblas.get_cores() == 1

    def test_env(self, monkeypatch):
        monkeypatch.setenv("TRNBLAS_CORES", "4")
        assert trnblas.get_cores() == 4
        monkeypatch.setenv("TRNBLAS_CORES", "auto")
        assert trnblas.get_cores() == "auto"

    def test_set_cores_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TRNBLAS_CORES", "4")
        trnblas.set_cores(2)
        assert trnblas.get_cores() == 2

    @pytest.mark.parametrize("bad", ["0", "-1", "two"])
    def test_bad_env_raises(self, monkeypatch, bad):
        monkeypatch.setenv("TRNBLAS_CORES", bad)
        with pytest.raises(ValueError):
            trnblas.get_cores()

    @pytest.mark.parametrize("bad", [0, -2, 1.5, "all"])
    def test_bad_set_cores_raises(self, bad):
        with pytest.raises(ValueError):
            trnblas.set_cores(bad)

    def test_host_tensor_never_splits(self):
        trnblas.set_cores(4)
        B, _, _ = _inputs(8, 16, 32)
        assert multicore.pair_energy_devices(B) is None


class TestOccChunks:
    @pytest.mark.parametrize("nocc,parts", [(1, 4), (5, 4), (16, 4), (96, 4), (7, 2), (9, 3)])
    def test_cover_and_balance(self, nocc, parts):
        chunks = multicore.occ_chunks(nocc, parts)
        assert chunks[0][0] == 0 and chunks[-1][1] == nocc
        assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))
        sizes = [e - s for s, e in chunks]
        assert len(chunks) == min(parts, nocc)
        assert max(sizes) - min(sizes) <= 1


class TestPairEnergySplit:
    @pytest.mark.parametrize(
        "nocc,nvir,naux,cores",
        [(5, 24, 40, 2), (8, 128, 128, 4), (9, 50, 70, 4), (16, 30, 130, 3), (2, 20, 20, 4)],
    )
    def test_sym_matches_single_core(self, emulated_sym, nocc, nvir, naux, cores):
        B, eo, ev = _inputs(nocc, nvir, naux)
        ref = dispatch._torch_batched_pair_energy(B.double(), eo.double(), ev.double())
        devices = _devices(cores)
        got = multicore.pair_energy_multicore(B, eo, ev, "sym", devices)
        assert got.shape == ()
        assert float(got) == pytest.approx(float(ref), rel=1e-5)

    @pytest.mark.parametrize("nocc,cores", [(6, 4), (11, 3), (3, 2)])
    def test_aten_matches_single_core(self, nocc, cores):
        B, eo, ev = _inputs(nocc, 20, 36)
        ref = dispatch._torch_chunked_pair_energy(B, eo, ev)
        devices = _devices(cores)
        got = multicore.pair_energy_multicore(B, eo, ev, "aten", devices)
        assert float(got) == pytest.approx(float(ref), rel=1e-5)

    def test_bad_route_raises(self):
        B, eo, ev = _inputs(4, 8, 8)
        with pytest.raises(ValueError):
            multicore.pair_energy_multicore(B, eo, ev, "legacy", [torch.device("cpu")] * 2)
