"""CPU checks of the LNC-split GEMM's static tiling (#79).

`_gemm_lnc_kernel` / `_gemm_lnc_tn_kernel` build their tile walk from the
plain-Python helpers below while they trace. These tests replay that walk on
index sets and check that every element of C is computed, by the programs of
an LNC=1 or LNC=2 logical core, with tiles inside the hardware limits:
<= 128 stationary columns per M tile, <= 512 moving columns per PSUM tile,
<= 4 PSUM banks per M block.
"""

import itertools

import pytest
import torch

from trnblas.nki import dispatch as d

SIZES = [1, 2, 5, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1031, 2048, 2049, 4097]


def _walk(M, N, K, n_prog, itemsize=4):
    """Yield (program, row range, col range, K tile sizes) for every PSUM tile."""
    split, MP, NPP = d._lnc_split(M, N, n_prog)
    k_tiles = d._lnc_sizes(K, 128)
    tm = d._lnc_tm(K, itemsize)
    for pid in range(n_prog if split else 1):
        r0 = pid * (M - MP) if split == "m" else 0
        c0 = pid * (N - NPP) if split == "n" else 0
        for m_off, msizes in d._lnc_blocks_of(MP, tm):
            assert 1 <= len(msizes) <= tm and all(1 <= ms <= 128 for ms in msizes)
            W, NF, full, tail = d._lnc_nplan(NPP, len(msizes))
            supers = [(nt * W, full) for nt in range(NF)] + ([(NF * W, tail)] if tail else [])
            for n_off, nsizes in supers:
                assert len(msizes) * len(nsizes) <= 4  # PSUM banks per block
                assert all(1 <= ns <= 512 for ns in nsizes)
                cols = n_off
                for ns in nsizes:
                    rows = r0 + m_off
                    for ms in msizes:
                        yield pid, (rows, rows + ms), (c0 + cols, c0 + cols + ns), k_tiles
                        rows += ms
                    cols += ns


@pytest.mark.parametrize("n_prog", [1, 2])
@pytest.mark.parametrize("M,N", list(itertools.product(SIZES, [1, 3, 512, 513, 2049, 5000])))
def test_tiles_cover_c_exactly_per_program(M, N, n_prog):
    count = torch.zeros(M, N, dtype=torch.int32)
    per_prog = {}
    for pid, (r0, r1), (c0, c1), k_tiles in _walk(M, N, 300, n_prog):
        assert 0 <= r0 < r1 <= M and 0 <= c0 < c1 <= N
        assert sum(k_tiles) == 300
        per_prog.setdefault(pid, torch.zeros(M, N, dtype=torch.int32))[r0:r1, c0:c1] += 1
        count[r0:r1, c0:c1] += 1
    # Within one program every element is computed exactly once ...
    for c in per_prog.values():
        assert int(c.max()) == 1
    # ... and across programs every element at least once, at most once per
    # program (the one-row/column overlap of an odd split).
    assert int(count.min()) >= 1 and int(count.max()) <= n_prog
    if n_prog == 2 and len(per_prog) == 2:
        overlap = int((count == 2).sum())
        split = d._lnc_split(M, N, 2)[0]
        assert overlap == (N * (M % 2) if split == "m" else M * (N % 2))


@pytest.mark.parametrize("K", [1, 63, 128, 129, 1000, 4096, 4097, 8192, 16384])
def test_block_fits_sbuf_budget(K):
    for itemsize in (4, 2):
        tm = d._lnc_tm(K, itemsize)
        assert 1 <= tm <= 4
        assert tm == 1 or K * tm * itemsize <= 64 * 1024


def test_split_prefers_rows_then_columns():
    assert d._lnc_split(4096, 4096, 2) == ("m", 2048, 4096)
    assert d._lnc_split(4097, 10, 2) == ("m", 2049, 10)
    assert d._lnc_split(64, 786432, 2) == ("n", 64, 393216)
    assert d._lnc_split(64, 1, 2) == ("", 64, 1)
    assert d._lnc_split(4096, 4096, 1) == ("", 4096, 4096)


def test_wide_b_loads_for_small_m():
    # One M tile: B loads are 4 sub-tiles (2048 columns) wide.
    W, NF, full, tail = d._lnc_nplan(393216, 1)
    assert (W, NF, full, tail) == (2048, 192, [512] * 4, [])
    W, NF, full, tail = d._lnc_nplan(5000, 1)
    assert W == 2048 and NF == 2 and tail == [512, 392]
    # Four M tiles: one 512-wide sub-tile per super-tile.
    assert d._lnc_nplan(4096, 4) == (512, 8, [512], [])
    assert d._lnc_nplan(300, 4) == (300, 1, [300], [])
