"""`trnblas.parallel` (#24) on CPU: 1-3 gloo ranks, compared with one process.

The SPMD routines only use torch.distributed collectives, so the same code
that runs one rank per NeuronCore runs here with CPU ranks. The on-chip run
is `scripts/bench_parallel.py` under torchrun.
"""

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _inputs(nbasis, nocc, naux):
    from df_mp2 import _make_inputs

    return _make_inputs(nbasis, nocc, naux)


def _sym_inputs(nbasis, nocc, naux):
    """`_inputs` with (mu nu|P) symmetrised in mu, nu, like real integrals."""
    C_occ, C_vir, eri, J, eo, ev = _inputs(nbasis, nocc, naux)
    return C_occ, C_vir, 0.5 * (eri + eri.transpose(0, 1)), J, eo, ev


def _worker(rank, world, port, case, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    import trnblas
    import trnblas.parallel as tp

    trnblas.set_backend("pytorch")
    out = {}
    try:
        if case == "mp2":
            for shape in [(24, 5, 40), (30, 7, 64), (16, 2, 20)]:
                out[shape] = tp.df_mp2_energy(*_inputs(*shape))
                sym = list(_sym_inputs(*shape))
                out[shape + ("sym",)] = tp.df_mp2_energy(*sym, eri_symmetric=True)
                sym[2] = tp.pack_tril(sym[2])
                out[shape + ("packed",)] = tp.df_mp2_energy(*sym, eri_symmetric=True)
        else:
            g = torch.Generator().manual_seed(1)
            A = torch.randn(12, 10, generator=g)
            B = torch.randn(10, 8, generator=g)
            mr, nr, kr = 12 // world, 8 // world, 10 // world
            out["row"] = tp.gemm_row_parallel(A[rank * mr : (rank + 1) * mr], B, gather=True)
            out["col"] = tp.gemm_col_parallel(A, B[:, rank * nr : (rank + 1) * nr], gather=True)
            ks = slice(rank * kr, (rank + 1) * kr)
            out["k"] = tp.gemm_k_parallel(A[:, ks], B[ks])
            out["k_scatter"] = tp.gemm_k_parallel(A[:, ks], B[ks], scatter=True)
            # Plain lists: tensors sent through an mp queue keep the sender alive.
            out = {k: v.tolist() for k, v in out.items()}
        q.put((rank, out))
    finally:
        dist.destroy_process_group()


def _run(world, case):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ps = [ctx.Process(target=_worker, args=(r, world, port, case, q)) for r in range(world)]
    for p in ps:
        p.start()
    got = dict(q.get(timeout=120) for _ in ps)
    for p in ps:
        p.join(timeout=60)
        assert p.exitcode == 0
    return got


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_df_mp2_matches_single_process(world):
    from df_mp2 import df_mp2_energy

    import trnblas

    trnblas.set_backend("pytorch")
    got = _run(world, "mp2")
    for shape in got[0]:
        make = _sym_inputs if shape[-1] in ("sym", "packed") else _inputs
        ref = df_mp2_energy(*make(*shape[:3]))
        for r in range(world):
            assert got[r][shape] == pytest.approx(ref, rel=1e-5), (shape, r)


def test_gemm_primitives():
    world = 2
    got = _run(world, "gemm")
    g = torch.Generator().manual_seed(1)
    A = torch.randn(12, 10, generator=g)
    B = torch.randn(10, 8, generator=g)
    C = A @ B
    for r in range(world):
        o = got[r]
        for key in ("row", "col", "k"):
            torch.testing.assert_close(torch.tensor(o[key]), C, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            torch.tensor(o["k_scatter"]), C[r * 6 : (r + 1) * 6], rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("nbasis,holders", [(7, 3), (24, 3), (512, 3), (5, 1), (3, 4)])
def test_eri_rows_cover_and_balance(nbasis, holders):
    from trnblas.parallel import _eri_rows

    for symmetric in (False, True):
        rows = _eri_rows(nbasis, holders, symmetric)
        assert rows[0][0] == 0 and rows[-1][1] == nbasis
        assert all(a[1] == b[0] for a, b in zip(rows, rows[1:], strict=False))
    rows = _eri_rows(512, 3, True)
    loads = [sum(mu + 1 for mu in range(a, b)) for a, b in rows]
    assert max(loads) / min(loads) < 1.02


def test_packed_needs_symmetric():
    import trnblas.parallel as tp

    C_occ, C_vir, eri, J, eo, ev = _sym_inputs(8, 2, 10)
    with pytest.raises(ValueError):
        tp.df_mp2_energy(C_occ, C_vir, tp.pack_tril(eri), J, eo, ev)
