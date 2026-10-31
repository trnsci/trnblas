"""Opt-in multi-core DF-MP2 pair energy across logical NeuronCores (#24).

One Trn2 chip at LNC=2 exposes four logical cores (``neuron:0..3``). The
DF-MP2 energy step is the workload's only O(nocc^2 nvir^2 naux) stage, and its
occupied rows are independent: row i of `_pair_energy_sym_kernel` (or the
i block of the chunked aten reduction) only reads B[i] and a window of B[j].
So the split is batch-over-pairs: each core takes a contiguous range of rows,
gets the B rows that range reads, and returns one scalar. Nothing else
crosses cores.

Data movement follows what Trn2 measurements showed for the other suite
libraries (trnsolver#54, trnrand#68): a ``neuron:0 -> neuron:k`` ``.to()``
leaves the data on the source core, and a copy out of a busy core waits
behind its queued kernels. So B is copied to the host once, before any
kernel is queued, and each other core uploads only the rows it reads. All
launches are then issued from the calling thread, round-robin over the
cores; launches are asynchronous, so the cores run together without host
threads (and without concurrent NKI tracing, which is not thread-safe).

Enable with ``TRNBLAS_CORES=<n>|auto`` or `set_cores`. Unset, the default
energy path is unchanged and single core. Staging through the host caps the
gain (1.76x on 4 cores at 96/672/2304, ~1.1x at 64/448/1536); the
process-per-core `trnblas.parallel` moves B with on-chip collectives instead
and gets 3.6x on the same step.
"""

from __future__ import annotations

import os

import torch

_CORES_ENV = "TRNBLAS_CORES"
_cores_override: int | str | None = None


def set_cores(n: int | str | None) -> None:
    """Logical NeuronCores for the DF-MP2 pair energy: an int, ``"auto"``, or None.

    None (the default) defers to ``TRNBLAS_CORES``; when that is unset too,
    everything runs on the input's core. ``"auto"`` uses every visible core
    when the energy is large enough to gain (see `_AUTO_MIN_FLOP`).
    """
    global _cores_override
    if n is not None and n != "auto" and (not isinstance(n, int) or n < 1):
        raise ValueError(f"set_cores expects a positive int, 'auto' or None; got {n!r}")
    _cores_override = n


def get_cores() -> int | str:
    """The active core setting (1 when nothing is set)."""
    if _cores_override is not None:
        return _cores_override
    raw = os.environ.get(_CORES_ENV, "").strip().lower()
    if raw in ("", "1"):
        return 1
    if raw == "auto":
        return "auto"
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{_CORES_ENV} must be a positive int or 'auto'; got {raw!r}") from None
    if n < 1:
        raise ValueError(f"{_CORES_ENV} must be a positive int or 'auto'; got {raw!r}")
    return n


# `auto` splits only when the whole energy step is at least this many flops
# (2 * nocc^2 * nvir^2 * naux, the unsymmetrised count). Measured on Trn2
# (scripts/bench_multicore.py energy, sym route, B on neuron:0, ms on 1/2/4
# cores): 64/448/1536 (2.5e12) 120 / 106 / 108; 96/672/2304 (1.9e13)
# 685 / 485 / 389. Staging B through the host costs ~13 GB/s each way, so
# only the larger case gains.
_AUTO_MIN_FLOP = 1.0e13


def _visible_cores() -> int:
    try:
        import torch_neuronx

        return int(torch_neuronx.device_count())
    except Exception:
        return 1


def pair_energy_devices(B: torch.Tensor) -> list[torch.device] | None:
    """The cores to split the pair energy of B over, B's own core first, or None."""
    setting = get_cores()
    if setting == 1 or B.device.type != "neuron":
        return None
    visible = _visible_cores()
    if setting == "auto":
        nocc, nvir, naux = B.shape
        if 2.0 * nocc * nocc * nvir * nvir * naux < _AUTO_MIN_FLOP:
            return None
        n = visible
    else:
        n = min(int(setting), visible)
    n = min(n, B.shape[0])
    if n <= 1:
        return None
    home = B.device.index if B.device.index is not None else _current_index()
    order = [home] + [k for k in range(visible) if k != home]
    return [torch.device("neuron", k) for k in order[:n]]


def _current_index() -> int:
    try:
        import torch_neuronx

        return int(torch_neuronx.current_device())
    except Exception:
        return 0


def occ_chunks(nocc: int, parts: int) -> list[tuple[int, int]]:
    """Split ``range(nocc)`` into at most ``parts`` contiguous, near-even ranges."""
    parts = max(1, min(parts, nocc))
    base, extra = divmod(nocc, parts)
    out, start = [], 0
    for c in range(parts):
        stop = start + base + (1 if c < extra else 0)
        out.append((start, stop))
        start = stop
    return out


def _gather_rows(x: torch.Tensor, idx: list[int], dev: torch.device) -> torch.Tensor:
    """``x[idx]`` on ``dev``, uploading each contiguous run of ``idx`` as one copy.

    ``idx`` is a wrapped range (``(i0 + r) % n``), so there are one or two runs
    (more only when the window is longer than ``x``).
    """
    runs, start = [], 0
    for r in range(1, len(idx) + 1):
        if r == len(idx) or idx[r] != idx[r - 1] + 1:
            runs.append(x[idx[start] : idx[r - 1] + 1].to(dev))
            start = r
    return runs[0] if len(runs) == 1 else torch.cat(runs)


# One reusable host staging buffer (shape, dtype) -> tensor; see `_stage_to_host`.
_STAGE: dict = {}


def _stage_to_host(t: torch.Tensor) -> torch.Tensor:
    """``t`` copied into a reused (pinned when possible) host buffer.

    Measured on Trn2 (SDK 2.33): a device-to-host copy into freshly
    allocated host memory, ``t.cpu()`` included, moves 2.8-3.1 GB/s above
    ~64 MB (first-touch page faults), while ``copy_`` into an already-touched
    or pinned buffer moves 13 GB/s. The buffer is overwritten by the next
    call, so callers use it only within one operation. One buffer is kept.
    """
    key = (tuple(t.shape), t.dtype)
    buf = _STAGE.get(key)
    if buf is None:
        _STAGE.clear()
        try:
            buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        except RuntimeError:
            buf = torch.empty(t.shape, dtype=t.dtype)
        _STAGE[key] = buf
    buf.copy_(t)
    return buf


def pair_energy_multicore(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
    route: str,
    devices: list[torch.device],
) -> torch.Tensor:
    """`nki_batched_pair_energy` with the occupied rows split over ``devices``.

    ``route`` is ``"sym"`` (`_pair_energy_sym_kernel`, one launch per row) or
    ``"aten"`` (chunked GEMM + torch reduction). ``devices[0]`` must be B's
    core. Returns a 0-d tensor on B's device. Each row's terms are the same
    as on one core; only the order of the final scalar sum changes.

    Order of work: B is copied to the host first (a copy out of a core waits
    for kernels queued there), then B's own core gets all its launches, then
    each other core in turn gets its rows uploaded and its launches queued.
    Host-to-device copies are serial (~13 GB/s for the chip from one
    process), so the uploads to later cores overlap the earlier cores' work.
    """
    if route not in ("sym", "aten"):
        raise ValueError(f"multi-core pair energy supports routes 'sym' and 'aten'; got {route!r}")
    nocc = B.shape[0]
    chunks = occ_chunks(nocc, len(devices))
    devices = devices[: len(chunks)]
    home = B.device
    B_h = _stage_to_host(B.detach()) if any(d != home for d in devices) else None
    eo_h = eps_occ.detach().cpu().to(B.dtype)
    ev_h = eps_vir.detach().cpu().to(B.dtype)

    parts = []
    for rows, dev in zip(chunks, devices, strict=True):
        src = B if dev == home else B_h
        if route == "sym":
            parts.append(sym_rows(src, eo_h, ev_h, rows, dev))
        else:
            parts.append(aten_rows(src, eo_h, ev_h, rows, dev))
    total = sum(float(p) for p in parts)
    return torch.tensor(total, dtype=B.dtype).to(home)


def sym_rows(
    src: torch.Tensor,
    eo_h: torch.Tensor,
    ev_h: torch.Tensor,
    rows: tuple[int, int],
    dev: torch.device,
) -> torch.Tensor:
    """Weighted `pair_energy_sym` energy of occupied rows ``rows`` on ``dev``.

    ``src`` is B (on ``dev`` or on the host); only the rows the range reads,
    i0 .. i1 + nd - 2 (mod nocc), are copied to ``dev``. ``eo_h``/``ev_h`` are
    host eps vectors. The launches are queued and a 0-d tensor on ``dev`` is
    returned without waiting.
    """
    from . import dispatch as D

    nocc, nvir, naux = src.shape
    i0, i1 = rows
    dims = D._pair_sym_dims(nocc, nvir, naux)
    nv, na, nd = dims["nvir_pad"], dims["naux_pad"], dims["nd_pad"]
    op = D._get_native_ops()["pair_energy_sym"]
    span = i1 - i0 + nd - 1
    idx = [(i0 + r) % nocc for r in range(span)]
    win = _gather_rows(src, idx, dev)
    if (nv, na) != (nvir, naux):
        Bp = torch.zeros(span, nv, na, dtype=src.dtype, device=dev)
        Bp[:, :nvir, :naux] = win
    else:
        Bp = win
    Bw = Bp.permute(0, 2, 1).contiguous()  # (span, naux_pad, nvir_pad): B[j].T
    del Bp, win
    eps_w = torch.cat([eo_h, eo_h])[idx]
    ex = torch.stack([eps_w[r : r + nd] for r in range(i1 - i0)]).to(dev)  # (rows, nd)
    eo = eo_h[i0:i1].reshape(1, -1).to(dev)
    evp = torch.zeros(nv, dtype=src.dtype)
    evp[:nvir] = ev_h
    evc = evp.reshape(nv, 1).to(dev)
    evr = evp.reshape(1, nv).to(dev)
    w = torch.tensor(dims["weights"], dtype=src.dtype).to(dev)
    acc = None
    for r in range(i1 - i0):
        p = op(Bw[r], Bw[r : r + nd], eo[:, r : r + 1], ex[r : r + 1], evc, evr)
        acc = p if acc is None else acc + p
    return (acc.sum(0) * w).sum()


def aten_rows(
    src: torch.Tensor,
    eo_h: torch.Tensor,
    ev_h: torch.Tensor,
    rows: tuple[int, int],
    dev: torch.device,
    mem_budget_bytes: int = 1_500_000_000,
) -> torch.Tensor:
    """Chunked aten energy (`_torch_chunked_pair_energy`) of rows ``rows`` on ``dev``.

    Every row pairs with all j, so ``dev`` gets all of B.
    """
    nocc, nvir, naux = src.shape
    i0, i1 = rows
    i_block = max(1, min(nocc, int(mem_budget_bytes // (4 * nvir * nocc * nvir))))
    B_flat = src.to(dev).reshape(nocc * nvir, naux)
    eo, ev = eo_h.to(dev), ev_h.to(dev)
    eps_o_pair = eo.view(nocc, 1, 1, 1) + eo.view(1, nocc, 1, 1)
    eps_v_pair = ev.view(1, 1, nvir, 1) + ev.view(1, 1, 1, nvir)
    e = None
    for a in range(i0, i1, i_block):
        b = min(a + i_block, i1)
        T = (B_flat[a * nvir : b * nvir] @ B_flat.T).reshape(b - a, nvir, nocc, nvir)
        T = T.permute(0, 2, 1, 3)
        t = (T * (2.0 * T - T.transpose(-2, -1)) / (eps_o_pair[a:b] - eps_v_pair)).sum()
        e = t if e is None else e + t
    return e
