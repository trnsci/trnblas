"""SPMD tensor-parallel GEMM and DF-MP2 across NeuronCores, one process per core (#24).

Run under ``torchrun --nproc_per_node=<cores>`` with the TorchNeuron
``neuron`` process-group backend. Each rank owns one logical NeuronCore, and
data crosses cores only through on-chip collectives. Measured on one Trn2 chip
(trn2.3xlarge, LNC=2, 4 logical cores, SDK 2.33):

- all-gather: 4 x 147 MB in 5-6 ms, 70-83 GB/s received per rank;
- reduce-scatter: 680 MB per rank in 12 ms;
- broadcast: 588 MB in 10 ms.

Copies between host and device run at about 13 GB/s for the whole chip,
whether one process or four make them. That makes the host the expensive
link, so these routines move each host byte once. Staging between cores
through the host (the single-process alternative, `trnblas.set_cores`) costs
5-10x more per byte.

Every function is collective: each rank of ``group`` must call it with the
same shapes. Tensors are on the rank's own core (``device()``) unless noted.

    import torch.distributed as dist
    import torch_neuronx.distributed.backend  # registers "neuron"
    import trnblas.parallel as tp

    dist.init_process_group("neuron")
    e = tp.df_mp2_energy(C_occ, C_vir, eri_3c, J, eps_occ, eps_vir)  # host inputs

The routines also run on any torch.distributed backend with CPU tensors
(gloo), which is how they are unit-tested.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from .nki.multicore import aten_rows, occ_chunks, sym_rows


def device() -> torch.device:
    """This rank's device: its NeuronCore when on the native path, else the CPU."""
    try:
        import torch_neuronx

        if torch_neuronx.device_count() > 0:
            return torch.device("neuron", torch_neuronx.current_device())
    except Exception:
        pass
    return torch.device("cpu")


def _rank_world(group) -> tuple[int, int]:
    return dist.get_rank(group), dist.get_world_size(group)


def _reduce_scatter(out: torch.Tensor, inp: torch.Tensor, group) -> None:
    """``reduce_scatter_tensor`` (sum) along dim 0; all-reduce + slice on gloo,
    which has no reduce-scatter."""
    if dist.get_backend(group) != "gloo":
        dist.reduce_scatter_tensor(out, inp, group=group)
        return
    r, w = _rank_world(group)
    full = inp.clone()
    dist.all_reduce(full, group=group)
    out.copy_(full.reshape(w, *out.shape)[r])


def _sync(dev: torch.device) -> None:
    if dev.type == "neuron":
        import torch_neuronx

        torch_neuronx.synchronize(dev.index)


# ── Tensor-parallel GEMM primitives ─────────────────────────────────────────


def all_gather_rows(x: torch.Tensor, group=None) -> torch.Tensor:
    """Concatenate every rank's ``x`` (same shape) along dim 0, on every rank."""
    _, w = _rank_world(group)
    out = torch.empty((w * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def gemm_row_parallel(A_rows: torch.Tensor, B: torch.Tensor, gather: bool = False, group=None):
    """C rows = A rows @ B: A sharded along M, B replicated. No communication
    unless ``gather``, which all-gathers the full C (every rank's A_rows must
    then have the same number of rows)."""
    C = A_rows @ B
    return all_gather_rows(C, group) if gather else C


def gemm_col_parallel(A: torch.Tensor, B_cols: torch.Tensor, gather: bool = False, group=None):
    """C cols = A @ B cols: B sharded along N, A replicated. With ``gather``,
    every rank gets the full C (equal shard widths)."""
    C = A @ B_cols
    if not gather:
        return C
    parts = all_gather_rows(C.T.contiguous(), group)  # (w * n_r, M)
    return parts.T.contiguous()


def gemm_k_parallel(A_k: torch.Tensor, B_k: torch.Tensor, scatter: bool = False, group=None):
    """C = sum over ranks of A_k @ B_k: the contraction dim K is sharded.

    Returns the full C on every rank (all-reduce), or with ``scatter`` this
    rank's block of C's rows (reduce-scatter; M must divide by the world size).
    """
    P = A_k @ B_k
    if scatter:
        _, w = _rank_world(group)
        out = torch.empty((P.shape[0] // w, P.shape[1]), dtype=P.dtype, device=P.device)
        _reduce_scatter(out, P, group)
        return out
    dist.all_reduce(P, group=group)
    return P


# ── DF-MP2 ──────────────────────────────────────────────────────────────────


def pair_energy(
    B_full: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor, group=None
) -> float:
    """MP2 pair energy with occupied rows split over the ranks.

    ``B_full`` (nocc, nvir, naux) is on this rank's device on every rank. Rank r
    computes rows ``occ_chunks(nocc, world)[r]`` (the sym kernel on the native
    path above its crossover, else chunked aten) and one all-reduce sums them.
    """
    from .nki import dispatch as D

    r, w = _rank_world(group)
    nocc = B_full.shape[0]
    chunks = occ_chunks(nocc, w)
    eo_h = eps_occ.detach().cpu().to(B_full.dtype)
    ev_h = eps_vir.detach().cpu().to(B_full.dtype)
    dev = B_full.device
    if r < len(chunks):
        route = D._pair_energy_route(B_full) if D._use_nki() else "aten"
        fn = sym_rows if route == "sym" else aten_rows
        e = fn(B_full, eo_h, ev_h, chunks[r], dev).reshape(1).to(torch.float32)
    else:
        e = torch.zeros(1, dtype=torch.float32, device=dev)
    dist.all_reduce(e, group=group)
    return float(e.cpu())


def _eri_rows(nbasis: int, holders: int, symmetric: bool) -> list[tuple[int, int]]:
    """Contiguous ERI row ranges for ``holders`` ranks, balanced by uploaded bytes.

    Full rows all weigh ``nbasis``; with ``symmetric`` row mu uploads only its
    lower triangle, ``mu + 1`` entries.
    """
    if not symmetric:
        return occ_chunks(nbasis, holders)
    total = nbasis * (nbasis + 1) // 2
    bounds, acc, start = [], 0, 0
    for mu in range(nbasis):
        acc += mu + 1
        if acc * holders >= total * (len(bounds) + 1) and len(bounds) < holders - 1:
            bounds.append((start, mu + 1))
            start = mu + 1
    bounds.append((start, nbasis))
    return bounds


def _tril_offset(mu: int) -> int:
    """Row ``mu``'s start in the packed lower triangle (pair index mu (mu+1)/2 + nu)."""
    return mu * (mu + 1) // 2


def pack_tril(eri_3c: torch.Tensor) -> torch.Tensor:
    """(nbasis, nbasis, naux) symmetric ERI -> packed lower triangle (npair, naux).

    Pair (mu, nu) with nu <= mu is row ``mu (mu + 1) / 2 + nu``: PySCF's
    ``lib.pack_tril`` order, so PySCF's ``_cderi`` (naux, npair) is this
    transposed.
    """
    n = eri_3c.shape[0]
    return torch.cat([eri_3c[mu, : mu + 1] for mu in range(n)])


def _half_transform_part(
    C_occ_pad: torch.Tensor,
    eri_3c: torch.Tensor,
    rows: tuple[int, int],
    dev: torch.device,
    symmetric: bool,
    nbasis: int,
) -> torch.Tensor:
    """This rank's partial sum of (i nu|P) = sum_mu C[mu, i] (mu nu|P) over ``rows`` of mu.

    Returns (nocc_pad, nbasis * naux) on ``dev``. ``eri_3c`` is the full
    (nbasis, nbasis, naux) tensor, or the packed lower triangle (npair, naux)
    (`pack_tril`). With ``symmetric`` only the lower triangle (nu <= mu) of
    each row is uploaded, in one copy, and unpacked on the device; the
    mirrored entries' contribution comes from the same data:
    sum_{nu < mu} C[nu, i] (mu nu|P) lands in (i mu|P).
    """
    naux = eri_3c.shape[-1]
    m0, m1 = rows
    n_r = m1 - m0
    Cr = C_occ_pad[m0:m1].to(dev)  # (n_r, nocc_pad)
    if not symmetric:
        eri_r = eri_3c[m0:m1].reshape(n_r, nbasis * naux).to(dev)
        return Cr.T.contiguous() @ eri_r
    p0, p1 = _tril_offset(m0), _tril_offset(m1)
    if eri_3c.dim() == 2:
        packed = eri_3c[p0:p1]
    else:
        packed = torch.cat([eri_3c[mu, : mu + 1] for mu in range(m0, m1)])
    P = torch.zeros(p1 - p0 + 1, naux, dtype=packed.dtype, device=dev)  # last row: zeros
    P[: p1 - p0].copy_(packed)
    # (r, nu) -> packed row, or the zero row above the diagonal.
    idx = torch.full((n_r, nbasis), p1 - p0, dtype=torch.int64)
    for r in range(n_r):
        mu = m0 + r
        idx[r, : mu + 1] = torch.arange(_tril_offset(mu) - p0, _tril_offset(mu) - p0 + mu + 1)
    diag_idx = idx[torch.arange(n_r), torch.arange(m0, m1)].to(dev)
    L = P.index_select(0, idx.reshape(-1).to(dev)).reshape(n_r, nbasis, naux)
    diag = P.index_select(0, diag_idx)  # (n_r, naux): (mu mu|P)
    del P
    part = Cr.T.contiguous() @ L.reshape(n_r, nbasis * naux)  # nu <= mu terms
    # Mirrored terms: (i mu|P) += sum_{nu < mu} C[nu, i] (mu nu|P).
    CT = C_occ_pad.T.contiguous().to(dev)  # (nocc_pad, nbasis)
    M = torch.matmul(CT, L)  # (n_r, nocc_pad, naux), includes nu == mu
    M = M - Cr.unsqueeze(2) * diag.unsqueeze(1)
    part.reshape(-1, nbasis, naux)[:, m0:m1, :] += M.permute(1, 0, 2)
    return part


def df_mp2_energy(
    C_occ: torch.Tensor,
    C_vir: torch.Tensor,
    eri_3c: torch.Tensor,
    J_metric: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
    group=None,
    timings: dict | None = None,
    eri_symmetric: bool = False,
) -> float:
    """DF-MP2 correlation energy, split over the ranks of ``group``.

    Same inputs and result as ``examples/df_mp2.py:df_mp2_energy``, as host
    tensors (every rank passes the same ones; each rank reads only its rows
    of ``eri_3c``). Returns the energy on every rank.

    ``eri_symmetric=True`` declares (mu nu|P) = (nu mu|P), as for real
    three-centre integrals; then only the lower triangle is read and
    uploaded, which halves the bytes over the host link. ``eri_3c`` may then
    also be the packed lower triangle (npair, naux) of `pack_tril` (PySCF's
    ``_cderi`` transposed), which saves the host-side packing copy.

    Dataflow, with r = rank and w = world size:

    1. Rank 0 forms X = J^-1/2 (`trnblas.metric_inv_half`) while ranks
       1..w-1 upload the ERI (with one rank, it does both). Host-to-device
       copies share one ~13 GB/s link, so the ERI upload is the floor and
       the metric factor overlaps it.
    2a. Each ERI-holding rank uploads its rows mu_r and forms the partial
       (i nu|P) = C_occ[mu_r]^T (mu_r nu|P); a reduce-scatter over i sums
       the partials and leaves rank r its block of occupied rows.
    2b, 3. X is broadcast; (ia|P) and B = (ia|P) X for those rows, locally.
    4. B is all-gathered, rank r computes the pair energy of its rows, and
       an all-reduce sums one scalar.

    The ERI crosses the host link once in total, not once per core.
    """
    import trnblas

    if eri_3c.dim() == 2 and not eri_symmetric:
        raise ValueError("a packed (npair, naux) eri_3c needs eri_symmetric=True")
    r, w = _rank_world(group)
    dev = device()
    nbasis, nocc = C_occ.shape
    nvir = C_vir.shape[1]
    naux = J_metric.shape[0]
    occ_r = -(-nocc // w)  # rows per rank, nocc padded to w * occ_r
    nocc_pad = occ_r * w
    t = {}
    clock = time.perf_counter()

    def mark(name):
        nonlocal clock
        if timings is not None:
            _sync(dev)
            now = time.perf_counter()
            t[name] = now - clock
            clock = now

    # 1 + 2a. Rank 0: metric factor. ERI holders: their rows' partial half-transform.
    holders = list(range(1, w)) if w > 1 else [0]
    X = torch.empty(naux, naux, dtype=torch.float32, device=dev)
    if r == 0:
        X.copy_(trnblas.metric_inv_half(J_metric.to(dev)))
    Cp = torch.zeros(nbasis, nocc_pad, dtype=C_occ.dtype)
    Cp[:, :nocc] = C_occ
    if r in holders:
        rows = _eri_rows(nbasis, len(holders), eri_symmetric)[holders.index(r)]
        part = _half_transform_part(Cp, eri_3c, rows, dev, eri_symmetric, nbasis)
    else:
        part = torch.zeros(nocc_pad, nbasis * naux, dtype=torch.float32, device=dev)
    mark("chol_upload_2a")
    iv_r = torch.empty(occ_r, nbasis * naux, dtype=part.dtype, device=dev)
    _reduce_scatter(iv_r, part, group)
    del part
    dist.broadcast(X, 0, group=group)
    mark("rs_iv_bcast_X")

    # 2b + 3 for this rank's occupied rows.
    CvT = C_vir.T.contiguous().to(dev)
    ia_r = torch.matmul(CvT, iv_r.reshape(occ_r, nbasis, naux))  # (occ_r, nvir, naux)
    del iv_r
    B_r = (ia_r.reshape(occ_r * nvir, naux) @ X).reshape(occ_r, nvir, naux)
    del ia_r
    mark("local_2b_3")

    # 4. All-gather B, energy rows, all-reduce.
    B_full = all_gather_rows(B_r, group)[:nocc].contiguous()
    del B_r
    mark("ag_B")
    e = pair_energy(B_full, eps_occ, eps_vir, group)
    mark("energy")
    if timings is not None:
        timings.update(t)
    return e


__all__ = [
    "all_gather_rows",
    "device",
    "df_mp2_energy",
    "gemm_col_parallel",
    "gemm_k_parallel",
    "gemm_row_parallel",
    "pack_tril",
    "pair_energy",
]
