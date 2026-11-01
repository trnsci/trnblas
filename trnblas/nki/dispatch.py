"""
NKI dispatch for BLAS operations.

Backend selection mirrors trnfft: auto/pytorch/nki.
The GEMM kernel is the primary acceleration target — it uses stationary
tile reuse on the Tensor Engine for 2x fewer SBUF loads vs naive.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

# When set, kernel-path failures re-raise instead of falling back to PyTorch.
# Used by the validation suite to catch silent kernel breakage during iteration.
_REQUIRE_NKI = os.environ.get("TRNBLAS_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. For
# correctness iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNBLAS_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


class NkiFallbackWarning(UserWarning):
    """Emitted once per distinct error when the NKI path silently falls
    back to torch.matmul. Prevents the class of bug where PATH / plugin
    misconfiguration causes every NKI call to hit torch without any
    user-visible signal — the v0.4.x-era 'libneuronpjrt-path' silent
    fallback is the motivating example.
    """


_fallback_warned: set[tuple[str, str]] = set()


def _warn_fallback(
    exc: BaseException,
    what: str = "NKI kernel dispatch failed — falling back to torch.matmul",
) -> None:
    """Emit NkiFallbackWarning once per unique error signature."""
    lines = str(exc).splitlines()
    key = f"{type(exc).__name__}: {(lines[0] if lines else '')[:200]}"
    if (what, key) in _fallback_warned:
        return
    _fallback_warned.add((what, key))
    warnings.warn(
        f"{what} (set TRNBLAS_REQUIRE_NKI=1 to re-raise). First error: {key}",
        NkiFallbackWarning,
        stacklevel=3,
    )


# Tile shapes for the systolic array (NKI 2.24 limits):
# stationary partition ≤ 128 (= K), free ≤ 128 (= M); moving free ≤ 512 (= N).
_TILE_M = 128
_TILE_K = 128
_TILE_N = 512

# Batched-pair energy kernel — full-batch vs chunked routing (#46).
# nl.affine_range traces all iterations eagerly into the XLA graph IR.
# Empirically: 768 iters (small bench, nocc=16) → ~240 MB JSON (fine);
# 786,432 iters (medium, nocc=64) → 18 GB (exceeds available disk).
# When est_iters = nocc² × N_A² × N_K exceeds this threshold, route to
# the chunked impl (_j_batched_kernel, Python i-loop, NOCC dispatches).
#
# The value is path-aware (see `_batched_pair_chunk_threshold`). XLA keeps
# 4096. On the native TorchNeuron path (~0.2 ms dispatch floor) the trade is
# compile time vs warm time, measured on trn2 (SDK 2.33, #52):
#
#   nocc,nvir,naux  est_iters  full first/warm    chunked first/warm
#   4,128,128           16      8.3 s / 0.90 ms    1.5 s / 1.86 ms
#   16,128,384         768      8.8 s / 2.33 ms    2.1 s / 25.7 ms
#   32,128,384        3072     35.8 s / 6.67 ms    2.4 s / 71.9 ms
#   16,256,512        4096     43.2 s / 7.0 ms     4.3 s / 42.3 ms
#   64,128,128        4096    177.9 s / 21.8 ms    3.2 s / 168.7 ms
#
# Full-batch is 2-11x faster warm everywhere, but its compile grows with the
# trace. Native takes full-batch up to est_iters 3072 (compile <= ~36 s) and the
# chunked kernel above it, where measured compiles reach 43-178 s. est_iters
# predicts compile only roughly: both 4096 rows differ 4x, tracking
# nocc^2 * N_A^2 more than N_K.
# TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD overrides both (for re-benchmarking, #50).
_BATCHED_PAIR_CHUNK_THRESHOLD = 4096
_BATCHED_PAIR_CHUNK_THRESHOLD_NATIVE = 3072

# J-sub-chunking: within the chunked impl, `_j_batched_kernel` receives the full
# B:(nocc × nvir_pad × naux_pad) tensor.  At large shapes (nocc=96, nvir_pad≈768,
# naux_pad=2304, ≈680 MB FP32) this exhausts neuronx-cc's compiler RAM on
# trn1.2xlarge (32 GB).  When the total B size exceeds this threshold, the j-loop
# is split into sub-chunks so each call receives B:(j_chunk × nvir_pad × naux_pad)
# within the ~256 MB range that compiles cleanly (medium B ≈ 251 MB is fine).
_J_CHUNK_MAX_B_BYTES = 256 * 1024 * 1024

# NRT hardware inner tile count limit: N_A × N_B × N_K per j-iteration.
# When this exceeds ~192 the compiled NEFF cannot be loaded by the NeuronCore
# NRT regardless of j_chunk size.  Confirmed empirically (2026-04-22, trn1):
#   Medium: 4 × 4 × 12 = 192  → NRT load OK
#   Large:  6 × 6 × 18 = 648  → NRT_RESOURCE for j_chunk = 1, 16, 32
# Conservative threshold: 300 (safely above 192, safely below 648).
# When inner_ops > this value, proactively fall back to torch to avoid
# ~30 min of wasted NEFF compilation before a guaranteed NRT failure.
_NRT_INNER_OP_LIMIT = 300

# Autotuner — sweeps tile candidates on hardware once per shape bucket and
# caches the winner to disk. Opt-in since #50: TRNBLAS_AUTOTUNE=1 or
# set_autotune(True). On trn2 the first-call sweep made cold GEMM 7-12x slower
# (2048^3: 25.1 s vs 3.4 s; 4096^3: 182.7 s vs 15.6 s) and the tile it picked
# gave no warm benefit, so the default is the fixed heuristic (128, 128, 512),
# clamped per dimension by the kernel for small shapes. Never used in simulator mode.
_AUTOTUNE_ENABLED: bool = os.environ.get("TRNBLAS_AUTOTUNE", "0").lower() in ("1", "true", "yes")

# In-kernel ragged-edge GEMM (#50). Unaligned shapes go to _gemm_kernel_ragged,
# which handles the M/K/N remainder tiles in-kernel, so there is no torch-level
# pad/slice on the device. TRNBLAS_GEMM_RAGGED=0 restores the pad-in-torch path.
# If the ragged kernel fails, dispatch falls back to the padded path (or
# re-raises under TRNBLAS_REQUIRE_NKI=1).
_RAGGED_ENABLED: bool = os.environ.get("TRNBLAS_GEMM_RAGGED", "1").lower() not in (
    "0",
    "false",
    "no",
)
# ── Native-path routing of the plain GEMM family (#62, #63) ──────────────────
#
# `gemm`, `batched_gemm`, `syrk`, `symm` and left `trsm` have three candidate
# routes on the native path: the NKI kernel, aten on the `neuron` device, and
# (for host inputs) the op in torch on the host CPU. `_gemm_route` picks one
# from measured times (scripts/bench_auto_routes.py; tables in #63 / #62):
#
# trn2.3xlarge, SDK 2.33, one LNC=2 logical core, 2 pinned host threads, warm
# medians in ms. Host operands, host result (upload + op + download):
#
#   op / shape (fp32)          host CPU   upload+aten   upload+NKI
#   gemm 256^3                    0.12        0.33         0.42
#   gemm 384^3                    0.47        0.68         0.84
#   gemm 512^3                    0.87        0.54         0.67
#   gemm 4096^3                    438        19.5         27.6
#   gemm 8192^3                   3126        88.4          182
#   gemm 16x128x49152             1.05        2.70         2.69
#   gemm 32x256x196608            30.1        18.6         19.0
#   batched 16x112x128x384        1.51        1.12         66.7
#   batched 16x1024^3              131        17.7         99.6
#   syrk 256x256                  0.34        0.55         0.81
#   syrk 8192x2048                1077        34.9         58.6
#   symm 4096^2                    606        20.5            -
#   trsm 768, 768 rhs             2.69        12.5         11.7
#   trsm 2304, 2304 rhs           59.9        96.9          100
#
# bf16 moves the host/device crossover up 2-30x: the host's bf16 matmul
# (AMX) ran 4096^3 in 115 ms and 32x256x196608 in 4.0 ms (device 10.2 ms).
# Device-resident operands: aten beat the NKI kernel at every GEMM, batched
# and SYRK shape and dtype measured, by 1.1x (32x256x196608, syrk 1024x256)
# to 6x (8192^3 bf16: 7.7 vs 47 ms; fp32 29.4 vs 116 ms), and by 20-310x for
# batched_gemm (one kernel launch per slice); there is no crossover. The NKI
# GEMM stays reachable through `set_backend("nki")` / TRNBLAS_NATIVE_GEMM=nki.
# TRSM: see `_trsm_route`.
#
# The host CPUs on the measurement box share physical cores with other jobs,
# and host-CPU times swung 2-5x between runs (bf16 most). The thresholds are
# fitted to the quieter run; on a contended host the device route is better
# than they predict near the crossover.
#
# TRNBLAS_NATIVE_GEMM=auto|nki|aten|host overrides `auto` (for validation and
# re-benchmarking; `host` applies to all-host inputs only). An explicit
# `set_backend("nki")` always runs the NKI kernels. The XLA path, the simulator
# and the "cpu" dispatch path keep the kernel route, as before. The fused
# kernels (MP2 energy, batched pair, pair_energy_sym) are not affected.
_NATIVE_GEMM_CHOICES = ("auto", "nki", "aten", "host")

# Host-vs-device model: for all-host operands, the host CPU wins when
#   macs < scale * (fixed + per_byte * link_bytes)
# with link_bytes = upload + download bytes and scale = host threads /
# _HOST_GEMM_MODEL_THREADS (the CPU side is compute-bound and scales with
# threads; it was measured on 2). `fixed` is the ~0.3 ms launch-and-transfer
# floor of the device route in host multiply-adds; `per_byte` prices the link
# against the CPU. Fitted per (op, dtype) to the sweep; bf16 has its own row
# because the host's bf16 matmul (AMX) is 4-6x faster than its fp32 one.
# float16 uses the fp32 row (not measured).
_HOST_GEMM_MODEL_THREADS = 2
_HOST_GEMM_MODEL: dict[tuple[str, str], tuple[float, float]] = {
    ("gemm", "fp32"): (6.0e7, 4.0),
    ("gemm", "bf16"): (2.5e8, 25.0),
    ("batched_gemm", "fp32"): (6.0e7, 4.0),
    ("batched_gemm", "bf16"): (2.5e8, 100.0),
    ("syrk", "fp32"): (6.0e7, 4.0),
    ("syrk", "bf16"): (1.5e8, 25.0),
    ("symm", "fp32"): (6.0e7, 4.0),
    ("symm", "bf16"): (2.5e8, 25.0),
}
# Only these dtypes have a device route under `auto`; anything else on the
# host (float64, complex, integer) stays on the host, where it ran correctly
# before (Trainium has no float64 matmul).
_DEVICE_GEMM_DTYPES = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp32"}


def _native_gemm_policy() -> str:
    value = os.environ.get("TRNBLAS_NATIVE_GEMM", "auto").strip().lower()
    if value not in _NATIVE_GEMM_CHOICES:
        raise ValueError(
            f"TRNBLAS_NATIVE_GEMM must be one of {_NATIVE_GEMM_CHOICES}; got {value!r}"
        )
    return value


def _on_neuron_device(*tensors: torch.Tensor) -> bool:
    return all(t.device.type == "neuron" for t in tensors)


def _on_host(*tensors: torch.Tensor) -> bool:
    return all(t.device.type == "cpu" for t in tensors)


def _upload_nbytes(t: torch.Tensor) -> int:
    """Bytes `_upload(t)` sends over the host link (0 if already on the device)."""
    if t.device.type == "neuron":
        return 0
    if t.dim() >= 3 and t.stride(0) == 0:
        return _upload_nbytes(t[0])
    return t.numel() * t.element_size()


def _upload(t: torch.Tensor) -> torch.Tensor:
    """`t` on the `neuron` device, sending each distinct byte once.

    A strided upload is far slower than a dense one on trn2 (~60x for the
    column-major metric factor, see `_host_inv_chol_factor`), and a host-side
    `.contiguous()` is a page-fault-bound pass. So a transposed view uploads
    its dense base and is transposed back on the device, and a batch-broadcast
    operand (stride 0 on the batch dim, e.g. `C.T.expand(nocc, ...)`) uploads
    one slice and is expanded on the device.
    """
    if t.device.type == "neuron":
        return t
    if t.is_contiguous():
        return t.to(_NEURON_DEVICE)
    if t.dim() >= 3 and t.stride(0) == 0:
        return _upload(t[0]).unsqueeze(0).expand(t.shape)
    if t.dim() >= 2:
        tt = t.transpose(-2, -1)
        if tt.is_contiguous():
            return tt.to(_NEURON_DEVICE).transpose(-2, -1)
    return t.contiguous().to(_NEURON_DEVICE)


def _host_gemm_wins(op: str, macs: float, nbytes: int, dtype: torch.dtype) -> bool:
    """True when `op` on all-host operands is faster on the host CPU than
    upload + aten + download (see `_HOST_GEMM_MODEL`)."""
    kind = _DEVICE_GEMM_DTYPES.get(dtype)
    if kind is None:
        return True
    fixed, per_byte = _HOST_GEMM_MODEL[(op, kind)]
    scale = max(1, torch.get_num_threads()) / _HOST_GEMM_MODEL_THREADS
    return macs < scale * (fixed + per_byte * nbytes)


def _kernel_dtypes_ok(*tensors: torch.Tensor) -> bool:
    """True when every operand has a dtype the NKI GEMM kernels compute."""
    return all(t.dtype in _DEVICE_GEMM_DTYPES for t in tensors)


def _no_kernel_for_dtype(what: str, *tensors: torch.Tensor) -> None:
    """Under TRNBLAS_REQUIRE_NKI, refuse a dtype the kernels do not take.

    Without it the caller runs torch instead. Before 0.9.1 float64 operands
    went into the FP32 kernels under ``set_backend("nki")`` and came back
    non-finite, with no error (#68).
    """
    if _REQUIRE_NKI:
        dts = sorted({str(t.dtype) for t in tensors})
        raise NotImplementedError(
            f"trnblas {what}: the NKI kernels take float32/bfloat16/float16, got {dts}. "
            "Unset TRNBLAS_REQUIRE_NKI to compute in torch."
        )


def _gemm_route(
    *tensors: torch.Tensor, op: str = "gemm", macs: float | None = None, nbytes: int | None = None
) -> str:
    """'torch', 'nki', 'aten' or 'host' for a plain GEMM-family call.

    'torch' is the pytorch backend (torch on whatever device the data is on).
    `macs` / `nbytes` (multiply-adds and host-link bytes) are needed only for
    all-host operands under `auto`; they default to `tensors[0] @ tensors[1]`.
    """
    if not _use_nki():
        return "torch"
    if _backend == "nki" or _use_simulator() or _get_dispatch_path() != "native":
        if not _kernel_dtypes_ok(*tensors):
            _no_kernel_for_dtype(op, *tensors)
            return "torch"
        return "nki"
    policy = _native_gemm_policy()
    if policy == "nki" and not _kernel_dtypes_ok(*tensors):
        _no_kernel_for_dtype(op, *tensors)
        return "torch"
    if policy in ("nki", "aten"):
        return policy
    if not _on_host(*tensors):
        return "aten"  # device-resident (or mixed): aten at every measured shape
    if policy == "host":
        return "host"
    if macs is None:
        a, b = tensors[0], tensors[1]
        macs = a.shape[-2] * a.shape[-1] * b.shape[-1] * max(1, math.prod(a.shape[:-2]))
        nbytes = (
            sum(_upload_nbytes(t) for t in tensors[:2])
            + (macs // max(1, a.shape[-1])) * a.element_size()
        )
    return "host" if _host_gemm_wins(op, macs, nbytes, tensors[0].dtype) else "aten"


def _gemm_via_aten(*tensors: torch.Tensor) -> bool:
    """True when a plain GEMM on these operands does not run the NKI kernel
    (aten on the device, or torch on the host). Kept for callers that only
    distinguish kernel from not-kernel (e.g. the umbrella DF-MP2 profile)."""
    return _gemm_route(*tensors) in ("aten", "host")


def _aten_on_device(fn, *tensors: torch.Tensor) -> torch.Tensor:
    """`fn(*tensors)` as aten on the `neuron` device; the result is returned on
    the device of the first input (host results come down through `_to_host`)."""
    orig = tensors[0].device
    return _to_host(fn(*[_upload(t) for t in tensors]), orig)


# aten::mm on the native `neuron` device aborts (execution timeout, then
# "NRT status 1203, A DMA engine encountered an unrecoverable error") when the
# left operand is a transposed view and the right operand is larger than
# 4 GiB: (96, 768)-view @ (768, 1399808) fails, (768, 1397760) runs, and a
# contiguous left operand runs at 5.4 GB. DF-MP2's half-transform
# `C_occ.T @ eri_flat` hits it from nbasis 768 / naux 2304. Until the runtime
# is fixed, copy the (small) left operand to row-major first.
_NEURON_MM_BIG_RHS_BYTES = 1 << 32


def _neuron_mm_lhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a`, made contiguous when `a @ b` would hit the strided-lhs/>4 GiB abort."""
    if (
        not a.is_contiguous()
        and b.numel() * b.element_size() >= _NEURON_MM_BIG_RHS_BYTES
        and _on_neuron_device(a, b)
    ):
        return a.contiguous()
    return a


_AUTOTUNE_CACHE_FILE: Path = Path(
    os.environ.get("TRNBLAS_AUTOTUNE_CACHE", "/var/tmp/trnblas-autotune/cache.json")
)
# Tile candidates: {64,128} × {128} × {128,256,512}.
# Default (128,128,512) is always included; filtered by divisibility at sweep time.
_TILE_CANDIDATES: list[tuple[int, int, int]] = [
    (64, 128, 128),
    (64, 128, 256),
    (64, 128, 512),
    (128, 128, 128),
    (128, 128, 256),
    (128, 128, 512),
]
# Registry: tile config → @nki.jit kernel (populated at import time in if HAS_NKI block).
_gemm_kernel_registry: dict[tuple[int, int, int], Any] = {}
# Persistent autotune results: shape bucket → winning tile config.
_autotune_mem: dict[tuple[int, int, int], tuple[int, int, int]] = {}
_autotune_loaded: bool = False

_backend = "auto"


def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires nki>=0.3.0 (Neuron SDK 2.29+)")
    _backend = backend


def get_backend() -> str:
    return _backend


def set_autotune(enabled: bool) -> None:
    """Enable/disable the first-call GEMM tile sweep (off by default since #50)."""
    global _AUTOTUNE_ENABLED
    _AUTOTUNE_ENABLED = bool(enabled)


def get_autotune() -> bool:
    return _AUTOTUNE_ENABLED


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def nki_batched_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched GEMM. A: (batch, M, K), B: (batch, K, N) → C: (batch, M, N).

    Loops over the batch dim, dispatching the 2D `_nki_gemm_impl` per
    slice. Every slice after the first hits the NEFF cache (identical
    kernel signature), so per-slice cost is HBM transfer + Tensor Engine
    dispatch only. A true 3D-batched NKI kernel is a future optimisation
    if benchmarks justify it.

    For DF-MP2 tensor contractions over auxiliary basis indices, the
    natural use case is one batched_gemm with batch=N_aux per orbital
    pair — exactly this loop's sweet spot.
    """
    if A.shape[0] == 0:
        return torch.matmul(A, B)
    route = _gemm_route(A, B, op="batched_gemm")
    if route in ("torch", "host"):
        return torch.matmul(A, B)
    if route == "aten":
        return _aten_on_device(torch.matmul, A, B)
    # Each slice is copied into one preallocated result as it arrives (a
    # `torch.stack` would be one more pass into fresh host memory, #61).
    out = None
    for i in range(A.shape[0]):
        c = _nki_gemm_impl(A[i], B[i])
        if out is None:
            out = _host_empty((A.shape[0], *c.shape), c.dtype, c.device)
        out[i].copy_(c)
    return out


def _torch_mp2_energy(
    T_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for the fused MP2 energy reduction.

    T_flat: (ic*nvir, nocc*nvir). eps_occ_chunk: (ic,). eps_occ_full:
    (nocc,). eps_vir: (nvir,). Returns a 0-d tensor — sum of
    T*(2T - T.T)/denom over the chunk. Mirrors the expression in
    examples/df_mp2.py so the NKI path can be swapped in transparently.
    """
    ic = eps_occ_chunk.shape[0]
    nocc = eps_occ_full.shape[0]
    nvir = eps_vir.shape[0]
    T = T_flat.reshape(ic, nvir, nocc, nvir).permute(0, 2, 1, 3)
    denom = (
        eps_occ_chunk.view(ic, 1, 1, 1)
        + eps_occ_full.view(1, nocc, 1, 1)
        - eps_vir.view(1, 1, nvir, 1)
        - eps_vir.view(1, 1, 1, nvir)
    )
    return (T * (2.0 * T - T.transpose(-2, -1)) / denom).sum()


def nki_mp2_energy(
    T_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Fused MP2 energy reduction — NKI-dispatched (#15).

    On NKI backend: a single kernel streams T_flat tiles on-chip and
    computes T*(2T - T.T)/denom + sum in one pass, avoiding the four
    HBM round-trips of the torch expression.

    On PyTorch backend (or when the kernel can't handle the shape
    yet): falls back to the torch reference.
    """
    if not _use_nki():
        return _torch_mp2_energy(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)
    try:
        return _nki_mp2_energy_impl(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return _torch_mp2_energy(T_flat, eps_occ_chunk, eps_occ_full, eps_vir)


def _nki_mp2_energy_impl(
    T_flat: torch.Tensor,
    eps_occ_chunk: torch.Tensor,
    eps_occ_full: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    # Pass eps_* in the orientation each access needs. NKI's partition
    # dim is physical; we can't reshape a partition=1 SBUF tile to
    # partition=N in the kernel ('illegal partition step' BIR error).
    #   eps_vir_col: (NVIR, 1) — strip loads pick (P_TILE, 1) slices
    #   eps_vir_row: (1, NVIR) — full free-dim vector for broadcast
    #   eps_occ_*:   (1, N)    — (1,1) scalars picked from the row
    t_in = T_flat.contiguous()
    eo_c_in = eps_occ_chunk.reshape(1, -1).contiguous()
    eo_f_in = eps_occ_full.reshape(1, -1).contiguous()
    ev_col_in = eps_vir.reshape(-1, 1).contiguous()
    ev_row_in = eps_vir.reshape(1, -1).contiguous()

    if _use_simulator():
        partial_np = nki.simulate(_mp2_energy_kernel)(
            t_in.cpu().numpy(),
            eo_c_in.cpu().numpy(),
            eo_f_in.cpu().numpy(),
            ev_col_in.cpu().numpy(),
            ev_row_in.cpu().numpy(),
        )
        partial = torch.from_numpy(np.asarray(partial_np)).to(T_flat.device)
    else:
        partial = _launch(
            _mp2_energy_kernel, "mp2_energy", t_in, eo_c_in, eo_f_in, ev_col_in, ev_row_in
        )
    return partial.sum()


def nki_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """GEMM with NKI dispatch.

    On NKI backend: uses tiled GEMM with stationary A reuse.
    On PyTorch backend: torch.matmul.

    NKI GEMM strategy (stationary tile reuse):
        1. Load A tile to SBUF as stationary (stays in systolic array)
        2. Stream B tiles as moving → accumulate in PSUM
        3. One A load serves multiple B tiles

    For DF-MP2 tensor contractions where A is the MO coefficient matrix
    (reused across auxiliary basis indices), this cuts SBUF loads in half.
    """
    route = _gemm_route(A, B)
    if route == "nki":
        return _nki_gemm_impl(A, B)
    if route == "aten":
        return _aten_on_device(lambda a, b: torch.matmul(_neuron_mm_lhs(a, b), b), A, B)
    return torch.matmul(A, B)


def nki_syrk(A: torch.Tensor, symmetrize: bool = False) -> torch.Tensor:
    """SYRK via single-operand NKI matmul. Returns A @ A.T.

    On NKI: dispatches `_syrk_kernel`, which loads A directly for both
    operand roles (avoids the A.T.contiguous() HBM write that would
    happen if we just called `nki_gemm(A, A.T)`).

    On PyTorch: falls back to `torch.matmul(A, A.T)`.

    ``symmetrize=True`` returns ``0.5 * (C + C.T)`` (bitwise the same values),
    which is exactly symmetric. On the native path a large host-input result
    is symmetrised on the device before it comes down (#61).
    """
    M, K = A.shape
    route = _gemm_route(A, op="syrk", macs=M * M * K, nbytes=(M * K + M * M) * A.element_size())
    if route == "nki":
        return _nki_syrk_impl(A, symmetrize=symmetrize)
    if route == "aten" and A.device.type != "neuron":
        a = _upload(A)
        r = torch.matmul(a, a.T)
        if symmetrize:
            # Symmetrise on the device, before the result comes down (#61):
            # a few aten ops there, no kernel compile, bitwise 0.5 * (C + C.T).
            return _to_host(0.5 * (r + r.T), A.device)
        r = _to_host(r, A.device)
    else:
        r = torch.matmul(A, A.T)
    return _symmetrize(r) if symmetrize else r


def nki_trsm(
    A: torch.Tensor,
    B: torch.Tensor,
    side: str = "left",
    uplo: str = "upper",
    trans: bool = False,
    diag: str = "nonunit",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Blocked triangular solve: op(A) X = alpha * B (side='left') or
    X op(A) = alpha * B (side='right').

    On NKI + side='left': blocked panel algorithm — the diagonal panel
    solve stays on torch.linalg.solve_triangular (tiny P×P, intrinsically
    sequential), while the trailing off-diagonal update is one nki_gemm
    call per block. GEMM dominates the work for large M, so this
    captures most of the speedup without writing a substitution kernel.

    Falls back to torch for side='right' (uncommon in chemistry hot
    paths) or when _use_nki() is False.

    On the native path `auto` picks the route for the whole solve
    (`_trsm_route`): host operands solve with LAPACK on the host, and
    device-resident operands run the blocked solve with aten updates on the
    device. `TRNBLAS_NATIVE_GEMM=aten` uploads host operands once and runs
    the blocked solve on the device. The NKI route (explicit
    `set_backend("nki")`, the XLA path) uses the NKI GEMM for every update.
    """
    if side != "left" or not _use_nki() or B.dim() != 2:
        return _trsm_torch(alpha, A, B, side, uplo, trans, diag)
    route = _trsm_route(A, B)
    if route == "host":
        return _trsm_on_host(alpha, A, B, uplo, trans, diag)
    try:
        if route == "aten":
            if _on_neuron_device(A, B):
                return _nki_trsm_left(A, B, uplo, trans, diag, alpha, mm=torch.matmul)
            X = _nki_trsm_left(_upload(A), _upload(B), uplo, trans, diag, alpha, mm=torch.matmul)
            return _to_host(X, B.device)
        return _nki_trsm_left(A, B, uplo, trans, diag, alpha, mm=_nki_gemm_impl)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return _trsm_torch(alpha, A, B, side, uplo, trans, diag)


# TRSM on the native path (#62, #63). The blocked solve on the device is
# bound by its diagonal-panel `solve_triangular` launches, not by the GEMM
# updates: with aten or NKI updates it costs the same (fp32, device-resident:
# 12 ms at n = 768, 91 ms at 2304, 200 ms at 1024 x 16384 rhs). From host
# operands, LAPACK on 2 host threads was 1.6-30x faster than every device
# route up to n = 2304 in the quieter run, and within noise of the best device
# route on wide right-hand sides (1024 x 16384: 150-160 ms vs 127-153 ms for
# host panels + NKI updates); 2048 x 8192 favoured upload + blocked aten (215
# vs 258 ms). With the host CPUs contended (see above), the device route won
# from n = 1536 (41 vs 55 ms) and on wide right-hand sides (1024 x 16384: 197
# vs 275 ms). So `auto` solves host operands on the host, and keeps
# device-resident operands on the device (copying them down to solve was
# within noise of the device route, sometimes slower). bf16/fp16 host
# operands solve in fp32 (LAPACK has no bf16 TRSM; `auto` used to raise).


def _trsm_route(A: torch.Tensor, B: torch.Tensor) -> str:
    """'nki', 'aten' or 'host' for a left TRSM (with `_use_nki()`)."""
    if _backend == "nki" or _use_simulator() or _get_dispatch_path() != "native":
        route = "nki"
    else:
        route = _native_gemm_policy()
        if route not in ("nki", "aten"):
            return "host" if _on_host(A, B) else "aten"
    if route == "nki" and not _kernel_dtypes_ok(A, B):
        # The NKI GEMM updates are FP32-only (#68): solve with torch instead.
        _no_kernel_for_dtype("trsm", A, B)
        return "host"
    return route


def _trsm_on_host(
    alpha: float, A: torch.Tensor, B: torch.Tensor, uplo: str, trans: bool, diag: str
) -> torch.Tensor:
    """Left TRSM with LAPACK on the host; the result has B's device and dtype."""
    dev, dt = B.device, B.dtype
    cdt = torch.float32 if dt in (torch.bfloat16, torch.float16) else dt
    Ah = A.detach().to(device="cpu", dtype=cdt)
    Bh = B.detach().to(device="cpu", dtype=cdt)
    X = _trsm_torch(alpha, Ah, Bh, "left", uplo, trans, diag)
    if X.dtype != dt:
        X = X.to(dt)
    return X if dev.type == "cpu" else X.contiguous().to(dev)


def _trsm_torch(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    side: str,
    uplo: str,
    trans: bool,
    diag: str,
) -> torch.Tensor:
    """Pure-torch TRSM reference. Mirrors the body of the original
    `trnblas.trsm` so the NKI dispatch wrapper has a pinned fallback
    that is independent of the public wrapper's evolution.
    """
    if uplo == "upper":
        tri = torch.triu(A)
    else:
        tri = torch.tril(A)

    if diag == "unit":
        tri = (
            tri
            - torch.diag(torch.diag(tri))
            + torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        )

    mat = tri.T if trans else tri

    if side == "left":
        upper_flag = (uplo == "upper" and not trans) or (uplo == "lower" and trans)
        return alpha * torch.linalg.solve_triangular(mat, B, upper=upper_flag)
    upper_flag = (uplo == "lower" and not trans) or (uplo == "upper" and trans)
    return alpha * torch.linalg.solve_triangular(mat.T, B.T, upper=upper_flag).T


def _nki_trsm_left(
    A: torch.Tensor,
    B: torch.Tensor,
    uplo: str,
    trans: bool,
    diag: str,
    alpha: float,
    block: int = 128,
    mm=None,
) -> torch.Tensor:
    """Blocked left-side TRSM. Diagonal panels solved via
    torch.linalg.solve_triangular (small, strictly sequential);
    trailing updates via `mm` (dominant work for large M; default `nki_gemm`).
    """
    if mm is None:
        mm = nki_gemm
    if trans:
        mat = A.T.contiguous()
        eff_upper = uplo == "lower"
    else:
        mat = A
        eff_upper = uplo == "upper"

    M = B.shape[0]
    unit = diag == "unit"

    # Small M: skip blocking — direct solve is cheap enough that
    # blocking only adds Python-loop overhead.
    if block >= M:
        X = torch.linalg.solve_triangular(mat, B, upper=eff_upper, unitriangular=unit)
        return alpha * X

    # In place throughout: on host operands every out-of-place step would be
    # another pass into fresh, page-faulting memory (#61).
    X = _host_empty_like(B).copy_(B)
    if not eff_upper:
        # Lower triangular: forward substitution.
        for k in range(0, M, block):
            ke = min(k + block, M)
            X[k:ke] = torch.linalg.solve_triangular(
                mat[k:ke, k:ke],
                X[k:ke],
                upper=False,
                unitriangular=unit,
            )
            if ke < M:
                X[ke:].sub_(mm(mat[ke:, k:ke].contiguous(), X[k:ke]))
    else:
        # Upper triangular: back substitution.
        for k in range(M, 0, -block):
            ks = max(k - block, 0)
            X[ks:k] = torch.linalg.solve_triangular(
                mat[ks:k, ks:k],
                X[ks:k],
                upper=True,
                unitriangular=unit,
            )
            if ks > 0:
                X[:ks].sub_(mm(mat[:ks, ks:k].contiguous(), X[ks:k]))
    return X.mul_(alpha) if alpha != 1.0 else X


# ── DF metric factor J^-1/2 = L^-T: host LAPACK vs device ────────────────────
#
# DF-MP2 needs X = L^-T with J = L L^T (so B = (ia|Q) X gives B B^T =
# (ia|Q) J^-1 (jb|Q)^T). On the native path with a device-resident J, the
# device route is `torch.linalg.cholesky` + the blocked `trsm` against the
# identity, i.e. ~naux/128 small solves and aten updates, each its own
# launch: 5-26 ms and noisy at naux 71-555 (#55). The host route copies J
# down once (naux^2 floats), runs LAPACK potrf + trsm in float64, and
# uploads X once. Float64 also helps accuracy: RI metrics have condition
# numbers of 1e6-1e9, where an FP32 Cholesky loses most of its digits.
#
# Measured on trn2 (SDK 2.33, fp32 J resident on the device, host on 2
# threads with OMP_WAIT_POLICY=PASSIVE, warm median ms;
# scripts/bench_metric_solve.py; resid = max|X^T J X - I| at cond(J) = 1e4):
#
#   naux    device (chol + trsm)    host (down + fp64 + up)   resid dev / host
#     71      2.6  (1.8 + 0.8)          0.5                    3e-5 / 3e-7
#    300      5.9  (1.9 + 4.0)          1.6                    4e-5 / 2e-7
#    423      9.4  (3.3 + 6.2)          3.1                    3e-5 / 2e-7
#    555     13.5  (4.6 + 8.9)          9.0                    7e-5 / 2e-7
#   1024     38.9  (23.4 + 15.4)       40.2                    4e-5 / 1e-7
#   1536     93.6  (52.1 + 41.6)      115                      5e-5 / 1e-7
#   2304    215    (124 + 91)         278                      4e-5 / 1e-7
#
# Host wins by 1.5-6x up to naux 555 and ties at 1024; from 1536 the device
# is 1.2-1.3x faster. `_METRIC_HOST_MAX_NAUX` sits at the tie, where the host
# route's 100-300x smaller residual decides it (2000-8000x at cond 1e7).
# The device route also compiles once per new naux (33-210 s at naux
# 1024-6144 the first time; cached on disk afterwards); the host route
# never compiles.
# TRNBLAS_METRIC_SOLVE=auto|host|device overrides. Host tensors, the XLA
# path, the simulator and the pytorch backend keep the device route (which
# is plain torch there), so the XLA path and the aten baseline are unchanged.
_METRIC_SOLVE_CHOICES = ("auto", "host", "device")
_METRIC_HOST_MAX_NAUX = 1024


def _metric_solve_route(J: torch.Tensor) -> str:
    """'host' or 'device' for `nki_metric_inv_half` on this J."""
    choice = os.environ.get("TRNBLAS_METRIC_SOLVE", "auto").strip().lower()
    if choice not in _METRIC_SOLVE_CHOICES:
        raise ValueError(
            f"TRNBLAS_METRIC_SOLVE must be one of {_METRIC_SOLVE_CHOICES}; got {choice!r}"
        )
    if not _use_nki() or _use_simulator() or not _on_neuron_device(J):
        return "device"
    if choice != "auto":
        return choice
    return "host" if J.shape[0] <= _METRIC_HOST_MAX_NAUX else "device"


def _host_inv_chol_factor(J: torch.Tensor) -> torch.Tensor:
    """L^-T for J = L L^T on the host in float64 (torch's LAPACK potrf + trsm).

    torch's own LAPACK, not SciPy's: a second BLAS runtime in the process
    brings a second spinning thread pool, which on a few pinned cores made
    torch's own LAPACK calls several times slower when the two interleaved.
    """
    Jd = J.detach().to(device="cpu", dtype=torch.float64)
    L, info = torch.linalg.cholesky_ex(Jd)
    if int(info) != 0:
        raise torch.linalg.LinAlgError(
            f"metric Cholesky failed: leading minor of order {int(info)} is not positive-definite"
        )
    eye = torch.eye(Jd.shape[0], dtype=torch.float64)
    # solve_triangular returns a column-major result; make it row-major on the
    # host. Uploading the strided tensor costs ~3.8 ms at naux 71 on trn2
    # (the device materialises the layout), against 0.06 ms contiguous.
    return torch.linalg.solve_triangular(L.T, eye, upper=True).contiguous()


def _device_inv_chol_factor(J: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(J)
    eye = torch.eye(J.shape[0], dtype=J.dtype, device=J.device)
    return nki_trsm(L, eye, uplo="lower", trans=True)


def nki_metric_inv_half(J: torch.Tensor) -> torch.Tensor:
    """Cholesky inverse factor X = L^-T of an SPD metric J = L L^T.

    X X^T = J^-1, so `A @ X` is the density-fitting metric contraction
    (`B = (ia|Q) J^-1/2` in DF-MP2). On the native path with J on the
    `neuron` device and naux <= `_METRIC_HOST_MAX_NAUX`, the factor is
    computed on the host in float64 and uploaded once (one download of J,
    one upload of X); otherwise `torch.linalg.cholesky` + `nki_trsm`.
    The result has J's dtype and device.
    """
    if not (J.is_floating_point() or J.is_complex()):
        # The host route would compute in float64 and truncate the factor back
        # to an integer dtype (#86); the torch route raises. Raise on every route.
        raise NotImplementedError(
            f"metric_inv_half: expected a floating point or complex tensor; got {J.dtype}"
        )
    if _metric_solve_route(J) == "host":
        X = _host_inv_chol_factor(J)
        return X.to(dtype=J.dtype).to(J.device)
    return _device_inv_chol_factor(J)


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _to_xla(*tensors):
    """Move tensors to the XLA device for NKI kernel dispatch."""
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


# ── Device dispatch path: native TorchNeuron vs XLA ──────────────────────────
#
# Two Neuron PyTorch lineages coexist:
#   native  torch-neuronx 2.14+ (TorchNeuron, Trn2/Trn3): Trainium is a
#           PyTorch `neuron` device; kernels are custom ops registered via
#           `torch_neuronx.nki_op` (see `_native.py`).
#   xla     torch-neuronx 2.9 lineage (PyTorch/XLA; Trn1 and Inf2 stay
#           here): kernels dispatch on the torch_xla device via `_to_xla()`.
# Anything else is "cpu": no device path, so a kernel launch outside the
# simulator raises (from `_to_xla`) exactly as it did before the native
# path existed, and the public wrappers fall back to torch as before.
# Detection runs on the first launch, not at import, and the native probe
# never imports torch_xla.

_NEURON_DEVICE = "neuron"
_dispatch_path: str | None = None
_native_ops: dict | None = None
_xla_deprecation_warned: bool = False

_XLA_DEPRECATION_MESSAGE = (
    "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
    "Trn1/Inf2 remain supported in maintenance mode."
)


def _native_available() -> bool:
    try:
        import torch_neuronx
    except Exception:
        return False
    # The XLA-lineage torch_neuronx has no `nki_op`.
    if not hasattr(torch_neuronx, "nki_op"):
        return False
    try:
        torch.device(_NEURON_DEVICE)  # torch_neuronx renames PrivateUse1 to "neuron"
        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def _detect_dispatch_path() -> str:
    # TRNSCI_NKI_DISPATCH=auto|native|xla is shared by every trnsci library, so
    # one variable pins the whole suite on hosts that have both lineages.
    forced = os.environ.get("TRNSCI_NKI_DISPATCH", "auto").strip().lower()
    if forced == "native":
        if not _native_available():
            raise RuntimeError(
                "TRNSCI_NKI_DISPATCH=native, but no TorchNeuron backend with a "
                "'neuron' device is available"
            )
        return "native"
    if forced == "xla":
        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError("TRNSCI_NKI_DISPATCH=xla, but torch_xla is not installed")
        return "xla"
    if forced != "auto":
        raise ValueError(f"TRNSCI_NKI_DISPATCH must be auto, native, or xla; got {forced!r}")
    if _native_available():
        return "native"
    if importlib.util.find_spec("torch_xla") is not None:
        return "xla"
    return "cpu"


def _maybe_warn_xla_deprecated() -> None:
    """Dormant XLA deprecation hook (trnsci/trnsci#35).

    Off by default; the suite turns it on (TRNSCI_XLA_DEPRECATION=1) when
    TorchNeuron reaches GA. Emits a single DeprecationWarning per process.
    """
    global _xla_deprecation_warned
    if _xla_deprecation_warned:
        return
    if os.environ.get("TRNSCI_XLA_DEPRECATION", "").strip().lower() not in ("1", "true", "yes"):
        return
    _xla_deprecation_warned = True
    warnings.warn(_XLA_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def _get_dispatch_path() -> str:
    """Return "native", "xla", or "cpu" (cached after the first call)."""
    global _dispatch_path
    if _dispatch_path is None:
        _dispatch_path = _detect_dispatch_path()
        if _dispatch_path == "xla":
            _maybe_warn_xla_deprecated()
    return _dispatch_path


def _get_native_ops() -> dict:
    """Import `_native` (registering the trnblas:: custom ops) on first use."""
    global _native_ops
    if _native_ops is None:
        from . import _native

        _native_ops = _native.OPS
    return _native_ops


def _to_device(*tensors):
    """Move tensors to the kernel device for the selected dispatch path.

    Returns ``(device_tensors, orig_device)``. Native: ``device="neuron"``.
    XLA (and "cpu", which then raises as before): ``_to_xla``, unchanged.
    """
    if _get_dispatch_path() == "native":
        orig = tensors[0].device
        return [t.to(_NEURON_DEVICE) for t in tensors], orig
    return _to_xla(*tensors)


# ── Host-link downloads (#61) ────────────────────────────────────────────────
#
# Copying device -> host into fresh host memory is bound by first-touch page
# faults, not by the link. Measured on trn2 (SDK 2.33, one logical core,
# 2 host threads), one copy down:
#
#   destination                                     64 MB   256 MB    1 GB
#   `.cpu()` / `.to("cpu")` / `torch.empty` + copy_   3.8     3.9      4.1 GB/s
#   `torch.empty` + `zero_()` (prefault) + copy_      3.6     3.8      4.0
#   fresh mmap with MADV_HUGEPAGE + copy_             7.6     8.0      8.0
#   reused host buffer, copy_                        12.9    13.8     14.2
#   fresh `torch.empty(pin_memory=True)` + copy_     12.8    13.9     14.2
#
# `pin_memory=True` goes through the accelerator's caching host allocator.
# Each call still gets a fresh tensor, and a block is recycled only after the
# tensor that owned it has been freed, so results never alias. The memory
# comes back already faulted in. Uploads from pageable memory already run at
# the link rate (13.5-13.8 GB/s), so uploads are unchanged.
#
# Costs: blocks are rounded up to a power of two, and freed blocks stay cached
# until `torch.accelerator.empty_host_cache()`. TRNBLAS_PINNED_OUTPUT=0 turns
# this off (plain `.to(device)`). Results under `_PINNED_OUTPUT_MIN_BYTES`
# always take `.to(device)`. The XLA path and the simulator are unchanged.
_PINNED_OUTPUT_MIN_BYTES = 1 << 20


def _pinned_output_enabled() -> bool:
    value = os.environ.get("TRNBLAS_PINNED_OUTPUT", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _use_host_cache(device, nbytes: int) -> bool:
    """True when a CPU result of `nbytes` should come from the caching host
    allocator: native path, CPU destination, large enough, not disabled."""
    return (
        torch.device(device).type == "cpu"
        and nbytes >= _PINNED_OUTPUT_MIN_BYTES
        and _pinned_output_enabled()
        and not _use_simulator()
        and _get_dispatch_path() == "native"
    )


def _host_empty(shape, dtype: torch.dtype, device) -> torch.Tensor:
    """`torch.empty(shape, dtype=dtype, device=device)`; a large CPU tensor on the
    native path comes from the caching host allocator (fresh, already faulted in)."""
    shape = tuple(shape)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if _use_host_cache(device, nbytes):
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.empty(shape, dtype=dtype, device=device)


def _host_empty_like(t: torch.Tensor) -> torch.Tensor:
    """Contiguous `_host_empty` with `t`'s shape, dtype and device."""
    return _host_empty(t.shape, t.dtype, t.device)


def _to_host(t: torch.Tensor, device) -> torch.Tensor:
    """`t.to(device)`. A large device -> CPU copy lands in a fresh tensor from
    the caching host allocator, about 3.5x faster than `.to("cpu")` (#61)."""
    if t.device.type == "cpu" or not _use_host_cache(device, t.numel() * t.element_size()):
        return t.to(device)
    return _host_empty(t.shape, t.dtype, device).copy_(t)


def _call_kernel(kernel, op_name: str, *device_tensors: torch.Tensor) -> torch.Tensor:
    """Run one kernel on tensors already on the kernel device (see `_to_device`).

    `op_name` is the `trnblas::<op_name>` custom op used on the native path;
    the XLA path calls `kernel` directly on torch_xla tensors.
    """
    if _get_dispatch_path() == "native":
        return _get_native_ops()[op_name](*device_tensors)
    return kernel(*device_tensors)


def _launch(kernel, op_name: str, *tensors: torch.Tensor) -> torch.Tensor:
    """Run one kernel launch on the Neuron device and return the result on
    the device of the first input. Every host wrapper goes through here (or
    through `_to_device` + `_call_kernel` when operands stay resident across
    several launches)."""
    xs, orig = _to_device(*tensors)
    return _to_host(_call_kernel(kernel, op_name, *xs), orig)


def _device_synchronize() -> None:
    """Wait for queued native kernels (for timing). No-op on the XLA path,
    whose sweep timing is unchanged."""
    if _get_dispatch_path() != "native":
        return
    import torch_neuronx

    sync = getattr(torch_neuronx, "synchronize", None)
    if sync is None:
        sync = torch_neuronx.neuron.synchronize
    sync()


def _gemm_op_name(tile_m: int, tile_k: int, tile_n: int) -> str:
    """Native custom-op name for a GEMM tile-config kernel."""
    return f"gemm_{tile_m}_{tile_k}_{tile_n}"


def _batched_pair_chunk_threshold() -> int:
    """est_iters above which batched-pair takes the chunked kernel (path-aware).

    XLA keeps 4096. Native uses 3072: full-batch is 2-11x faster warm, but
    above that its first-call compile was 43-178 s on trn2 (see the table at
    `_BATCHED_PAIR_CHUNK_THRESHOLD`).
    """
    env = os.environ.get("TRNBLAS_BATCHED_PAIR_CHUNK_THRESHOLD", "").strip()
    if env:
        return int(env)
    if not _use_simulator() and _get_dispatch_path() == "native":
        return _BATCHED_PAIR_CHUNK_THRESHOLD_NATIVE
    return _BATCHED_PAIR_CHUNK_THRESHOLD


def _get_gemm_kernel(tile_m: int, tile_k: int, tile_n: int):
    """Return the @nki.jit kernel for the given tile config.

    NKI's AST-based compiler resolves names from source rather than following
    Python closure chains, so tile constants must be literal integers baked
    into each kernel's function body at definition time.  All six candidates
    are defined as module-level @nki.jit functions in the ``if HAS_NKI:``
    block below and registered in ``_gemm_kernel_registry`` at import time.
    """
    key = (tile_m, tile_k, tile_n)
    if key not in _gemm_kernel_registry:
        raise KeyError(
            f"No pre-compiled GEMM kernel for tile config {key}. "
            f"Valid configs: {sorted(_gemm_kernel_registry)}"
        )
    return _gemm_kernel_registry[key]


def _ceil_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _autotune_bucket(M: int, K: int, N: int) -> tuple[int, int, int]:
    """Coarse shape bucket — same bucket for shapes within a DF-MP2 run."""
    return (_ceil_pow2(M), _ceil_pow2(K), _ceil_pow2(N))


def _load_autotune_cache() -> None:
    global _autotune_mem, _autotune_loaded
    if _autotune_loaded:
        return
    try:
        if _AUTOTUNE_CACHE_FILE.exists():
            raw = json.loads(_AUTOTUNE_CACHE_FILE.read_text())
            _autotune_mem = {
                tuple(map(int, k.split(","))): tuple(v)  # type: ignore[assignment]
                for k, v in raw.items()
            }
    except Exception:
        pass
    _autotune_loaded = True


def _save_autotune_cache() -> None:
    try:
        _AUTOTUNE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _AUTOTUNE_CACHE_FILE.write_text(
            json.dumps({",".join(map(str, k)): list(v) for k, v in _autotune_mem.items()})
        )
    except Exception:
        pass


def _sweep_tile_configs(M_pad: int, K_pad: int, N_pad: int, a_dev, b_dev) -> tuple[int, int, int]:
    """Time each candidate tile config on hardware; return the fastest.

    ``a_dev`` / ``b_dev`` are already on the kernel device (`_to_device`).
    Skips configs that don't evenly divide the padded shape. Falls back to
    the default if every candidate errors or gets filtered out. On the native
    path each timing is fenced with a device synchronize (eager launches are
    asynchronous); the XLA path times exactly as before.
    """
    import time

    best: tuple[int, int, int] = (_TILE_M, _TILE_K, _TILE_N)
    best_t = float("inf")
    for tm, tk, tn in _TILE_CANDIDATES:
        if M_pad % tm or K_pad % tk or N_pad % tn:
            continue
        k = _get_gemm_kernel(tm, tk, tn)
        op_name = _gemm_op_name(tm, tk, tn)
        try:
            _call_kernel(k, op_name, a_dev, b_dev)  # warm-up (compile + first run)
            _device_synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                _call_kernel(k, op_name, a_dev, b_dev)
            _device_synchronize()
            t = (time.perf_counter() - t0) / 3
            if t < best_t:
                best, best_t = (tm, tk, tn), t
        except Exception:
            continue
    return best


def _sweep_on_default_pad(
    M: int, K: int, N: int, A: torch.Tensor, B: torch.Tensor
) -> tuple[int, int, int]:
    """Pad with default tile sizes to produce aligned sweep inputs, then sweep."""
    A_p, B_p, _ = _pad_gemm_operands(A, B, _TILE_M, _TILE_K, _TILE_N)
    (a, b), _ = _to_device(A_p, B_p)
    return _sweep_tile_configs(A_p.shape[0], A_p.shape[1], B_p.shape[1], a, b)


def _gemm_padded_shape(
    M: int, K: int, N: int, tile_m: int, tile_k: int, tile_n: int
) -> tuple[int, int, int]:
    """Padded (M, K, N) the GEMM kernels accept for a given tile config.

    The kernels loop ``M // tile_m``, ``K // tile_k`` and ``N // TILE_N`` with
    no remainder handling, so M and K must be exact tile multiples. Along N the
    kernel clamps ``TILE_N = N`` when ``N <= tile_n`` (single N-tile), so N is
    left alone in that case and rounded up to ``tile_n`` otherwise.
    """
    M_pad = _round_up(M, tile_m)
    K_pad = _round_up(K, tile_k)
    N_pad = N if tile_n >= N else _round_up(N, tile_n)
    return M_pad, K_pad, N_pad


def _pad_gemm_operands(
    A: torch.Tensor, B: torch.Tensor, tile_m: int, tile_k: int, tile_n: int
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Zero-pad A:(M, K) and B:(K, N) to the kernel's tile multiples (#50).

    Zero padding along K is exact (padded terms contribute 0 to every dot
    product); padded M rows / N columns only produce output that the caller
    slices off. When the shape is already aligned no padded copy is made:
    the operands are returned as-is (``.contiguous()`` is a no-op for
    contiguous inputs; transposed views get one dense copy, which the kernel
    needs anyway).
    """
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(
            f"GEMM inner dimensions differ: A is {tuple(A.shape)}, B is {tuple(B.shape)}"
        )
    M_pad, K_pad, N_pad = _gemm_padded_shape(M, K, N, tile_m, tile_k, tile_n)
    if (M_pad, K_pad, N_pad) == (M, K, N):
        return A.contiguous(), B.contiguous(), False
    A_p = torch.zeros(M_pad, K_pad, dtype=A.dtype, device=A.device)
    A_p[:M, :K] = A
    B_p = torch.zeros(K_pad, N_pad, dtype=B.dtype, device=B.device)
    B_p[:K, :N] = B
    return A_p, B_p, True


def _run_padded_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    kernel_fn,
    tile_m: int = _TILE_M,
    tile_k: int = _TILE_K,
    tile_n: int = _TILE_N,
) -> torch.Tensor:
    """Pad A/B to tile multiples, run ``kernel_fn(A_feed, B_feed)``, slice to (M, N).

    ``kernel_fn`` takes the padded torch operands and returns a torch tensor of
    shape (M_pad, N_pad). Keeping the device hop (XLA / simulator) inside
    ``kernel_fn`` lets CPU tests drive this wrapper with a stand-in kernel.
    """
    M = A.shape[0]
    N = B.shape[1]
    A_feed, B_feed, needs_pad = _pad_gemm_operands(A, B, tile_m, tile_k, tile_n)
    result = kernel_fn(A_feed, B_feed)
    return result[:M, :N] if needs_pad else result


def _ragged_dim_plan(D: int, tile: int) -> tuple[int, int, int]:
    """Edge-tile plan for one GEMM dimension, mirroring `_gemm_kernel_ragged`.

    Returns ``(T, n_full, rem)``: tile size ``T = min(tile, D)``, the number of
    full tiles ``n_full = D // T`` (always >= 1), and the remainder tile size
    ``rem = D - n_full * T`` (0 when aligned). The remainder tile starts at
    ``n_full * T``. Every index in ``[0, D)`` is covered exactly once.
    """
    if D < 1:
        raise ValueError(f"GEMM dimension must be >= 1, got {D}")
    T = tile if tile < D else D
    n_full = D // T
    return T, n_full, D - n_full * T


def _ragged_tiles(D: int, tile: int) -> list[tuple[int, int]]:
    """(offset, size) list of the tiles `_gemm_kernel_ragged` visits along D."""
    T, n_full, rem = _ragged_dim_plan(D, tile)
    tiles = [(i * T, T) for i in range(n_full)]
    if rem > 0:
        tiles.append((n_full * T, rem))
    return tiles


def _ragged_gemm_reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Torch emulation of `_gemm_kernel_ragged`'s tile walk (for CPU tests).

    Same loop structure and the same per-tile slices as the kernel: the
    stationary tile is A[m, k].T (partitions = k size), the moving tile is
    B[k, n] (partitions = k size, so the counts match), and partial products
    accumulate into an FP32 "PSUM" tile of shape (m size, n size).
    """
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, dtype=A.dtype)
    for m_off, msz in _ragged_tiles(M, _TILE_M):
        for n_off, nsz in _ragged_tiles(N, _TILE_N):
            psum = torch.zeros(msz, nsz, dtype=torch.float64)
            for k_off, ksz in _ragged_tiles(K, _TILE_K):
                a_t = A[m_off : m_off + msz, k_off : k_off + ksz].T
                b_t = B[k_off : k_off + ksz, n_off : n_off + nsz]
                assert a_t.shape[0] == b_t.shape[0] == ksz <= _TILE_K
                assert msz <= _TILE_M and nsz <= _TILE_N
                psum += a_t.T.double() @ b_t.double()
            C[m_off : m_off + msz, n_off : n_off + nsz] = psum.to(A.dtype)
    return C


def _nki_gemm_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """NKI GEMM implementation with tile-shape autotuner (#26).

    With autotune enabled: sweeps tile candidates once per shape bucket and caches the
    winner to disk (``/var/tmp/trnblas-autotune/cache.json`` by default).
    Subsequent calls for the same bucket hit the in-process dict and then the
    NEFF cache — no re-sweep.

    The sweep is opt-in (``TRNBLAS_AUTOTUNE=1`` / ``set_autotune(True)``) since
    #50; by default and in simulator mode the (128, 128, 512) heuristic is used.
    With the default tiles, unaligned shapes run ``_gemm_kernel_ragged`` (edge
    tiles in-kernel, no torch pad). Otherwise operands are padded *after* tile
    selection so alignment matches the chosen tile sizes.

    Set ``TRNBLAS_REQUIRE_NKI=1`` to re-raise kernel errors instead of falling
    back to ``torch.matmul``.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    _, N = B.shape

    # Tile selection: autotuner on hardware, defaults elsewhere.
    if _AUTOTUNE_ENABLED and not _use_simulator():
        _load_autotune_cache()
        bucket = _autotune_bucket(M, K, N)
        if bucket not in _autotune_mem:
            tile_m, tile_k, tile_n = _sweep_on_default_pad(M, K, N, A, B)
            _autotune_mem[bucket] = (tile_m, tile_k, tile_n)
            _save_autotune_cache()
        tile_m, tile_k, tile_n = _autotune_mem[bucket]
    else:
        tile_m, tile_k, tile_n = _TILE_M, _TILE_K, _TILE_N

    def _run(kernel, op_name: str, A_feed: torch.Tensor, B_feed: torch.Tensor) -> torch.Tensor:
        if _use_simulator():
            out_np = nki.simulate(kernel)(A_feed.cpu().numpy(), B_feed.cpu().numpy())
            return torch.from_numpy(np.asarray(out_np)).to(A.device)
        return _launch(kernel, op_name, A_feed, B_feed)

    def _launch_tiled(A_feed: torch.Tensor, B_feed: torch.Tensor) -> torch.Tensor:
        return _run(
            _get_gemm_kernel(tile_m, tile_k, tile_n),
            _gemm_op_name(tile_m, tile_k, tile_n),
            A_feed,
            B_feed,
        )

    # Ragged shapes with the default tiles: the edge tiles are handled in-kernel
    # (#50), so there is no torch pad/slice and no per-shape pad/slice graphs on
    # the device.
    aligned = _gemm_padded_shape(M, K, N, tile_m, tile_k, tile_n) == (M, K, N)
    if _RAGGED_ENABLED and not aligned and (tile_m, tile_k, tile_n) == (_TILE_M, _TILE_K, _TILE_N):
        try:
            return _run(_gemm_kernel_ragged, "gemm_ragged", A.contiguous(), B.contiguous())
        except Exception as exc:
            if _REQUIRE_NKI:
                raise
            _warn_fallback(exc, "ragged GEMM kernel failed — falling back to padded NKI GEMM")

    try:
        return _run_padded_gemm(A, B, _launch_tiled, tile_m, tile_k, tile_n)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return torch.matmul(A, B)


def _torch_batched_pair_energy(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for the batched pair energy kernel.

    B: (nocc, nvir, naux). Returns a 0-d scalar — total MP2 energy
    summed over all (i, j) orbital pairs.
    """
    e = torch.zeros((), dtype=B.dtype, device=B.device)
    nocc = B.shape[0]
    for i in range(nocc):
        for j in range(nocc):
            T = B[i] @ B[j].T
            denom = (
                float(eps_occ[i]) + float(eps_occ[j]) - eps_vir.unsqueeze(1) - eps_vir.unsqueeze(0)
            )
            e = e + (T * (2.0 * T - T.T) / denom).sum()
    return e


def nki_batched_pair_energy(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Batched DF-MP2 pair energy — O(1) dispatch for all NOCC² pairs (#43).

    Replaces the per-pair loop over `nki_fused_gemm_energy` with a single
    @nki.jit dispatch that computes all orbital-pair energies in one NEFF.
    Eliminates the ~100ms × nocc² Neuron XLA per-dispatch overhead.

    Routing between the full-batch kernel and the chunked per-i kernel is
    path-aware (`_batched_pair_chunk_threshold`): the native TorchNeuron path
    switches to the chunked kernel above est_iters 3072 (XLA: 4096), where the
    full-batch compile becomes prohibitive.

    On NKI backend: one `_batched_pair_kernel` call covering all pairs.
    On PyTorch backend (or hardware errors): falls back to the torch
    reference — equivalent result, O(nocc²) Python overhead but no
    Trainium cost.

    B: (nocc, nvir, naux) — DF-fitted B matrix for all occupied orbitals.
    eps_occ: (nocc,) — occupied orbital energies.
    eps_vir: (nvir,) — virtual orbital energies.
    Returns: 0-d scalar tensor (total MP2 pair energy, same sign convention
    as nki_fused_gemm_energy).
    """
    if not _use_nki():
        return _torch_batched_pair_energy(B, eps_occ, eps_vir)
    route = _pair_energy_route(B)
    if route in ("sym", "aten"):
        from .multicore import pair_energy_devices, pair_energy_multicore

        devices = pair_energy_devices(B)
        if devices is not None:
            return pair_energy_multicore(B, eps_occ, eps_vir, route, devices)
    if route == "aten":
        return _torch_chunked_pair_energy(B, eps_occ, eps_vir)
    if route == "sym":
        try:
            return _nki_pair_energy_sym_impl(B, eps_occ, eps_vir)
        except Exception as exc:
            if _REQUIRE_NKI:
                raise
            _warn_fallback(exc, "pair_energy_sym kernel failed — falling back to aten")
            return _torch_chunked_pair_energy(B, eps_occ, eps_vir)
    try:
        return _nki_batched_pair_energy_impl(B, eps_occ, eps_vir)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return _torch_batched_pair_energy(B, eps_occ, eps_vir)


# ── Native batched-pair energy: symmetric kernel, both cores at LNC=2 (`_pair_energy_sym_kernel`) ──
#
# On device-resident inputs on the native path, `nki_batched_pair_energy` picks
# between the LNC=2 symmetric kernel and aten (chunked GEMM + torch reduction):
# the kernel launches once per occupied row, so it pays ~nocc launch floors and
# only wins once each row carries enough work. Measured on trn2 (SDK 2.33,
# fp32, device-resident, warm; aten = `_torch_chunked_pair_energy` on the
# device; row GFLOP = 2 * (nocc//2 + 1) * nvir_pad^2 * naux_pad):
#
#   nocc,nvir,naux   row GFLOP   aten        sym kernel
#   16,128,384          0.11      1.9 ms      5.5 ms
#   20,200,555          0.92      4.3 ms      7.5 ms
#   32,256,768          1.71     12.1 ms     13.8 ms
#   32,384,1024         5.13     27.3 ms     20.5 ms
#   48,384,1152         8.50     61.9 ms     39.4 ms
#   64,448,1536        26.6     207 ms      118 ms
#   96,672,2304        66.9     1.22 s      0.69 s
#
# (#55 recorded 63 s for aten at 96/672/2304. That number is not aten's
# steady state: with a few GB free on the logical core the warm chunked
# reduction takes 1.22 s, and it degrades to 35-64 s only when 10-14 GB of
# the 24 GB core are already held, as in a full 768/96/2304 DF-MP2 run. See
# trnblas#50.)
# The crossover sits between 1.7 and 5.1 row GFLOP; `_PAIR_SYM_MIN_ROW_FLOP`
# takes 3e9. TRNBLAS_PAIR_ENERGY overrides: auto | sym | aten | legacy (the
# full-batch / chunked kernels).
_PAIR_ENERGY_CHOICES = ("auto", "sym", "aten", "legacy")
_PAIR_SYM_MIN_ROW_FLOP = 3.0e9
# SBUF bytes per partition the kernel keeps live: B[i].T and B[j].T
# (NK * NVIR each) plus the T strip block (NA * NVIR), all fp32. trn2 has
# 192 KiB per partition; leave room for the compiler's temporaries.
_PAIR_SYM_SBUF_BUDGET = 144 * 1024


def _pair_sym_dims(nocc: int, nvir: int, naux: int) -> dict:
    """Padded sizes, j-window length and per-d weights for `_pair_energy_sym_kernel`.

    Row i takes j = i + d (mod nocc) for d = 0..nocc//2, so every unordered
    pair {i, j} is visited by exactly one (i, d) — twice for d = nocc/2 when
    nocc is even. e_ij = e_ji, so off-diagonal d get weight 2 except that
    doubled d, which gets 1. The window is padded to an even length (one j
    per LNC=2 program) with weight-0 entries.
    """
    nvir_pad = _round_up(nvir, 128)
    naux_pad = _round_up(naux, 128)
    half = nocc // 2
    nd = half + 1
    nd_pad = nd + (nd % 2)
    weights = [2.0] * nd_pad
    weights[0] = 1.0
    if half > 0 and nocc % 2 == 0:
        weights[half] = 1.0
    for d in range(nd, nd_pad):
        weights[d] = 0.0
    na, nk = nvir_pad // 128, naux_pad // 128
    sbuf_bytes = 4 * nvir_pad * (2 * nk + na)
    return dict(
        nvir_pad=nvir_pad,
        naux_pad=naux_pad,
        nd_pad=nd_pad,
        weights=weights,
        sbuf_bytes=sbuf_bytes,
        row_flop=2.0 * nd * nvir_pad * nvir_pad * naux_pad,
    )


def _pair_energy_route(B: torch.Tensor) -> str:
    """'sym', 'aten' or 'legacy' for `nki_batched_pair_energy` on these inputs."""
    choice = os.environ.get("TRNBLAS_PAIR_ENERGY", "auto").strip().lower()
    if choice not in _PAIR_ENERGY_CHOICES:
        raise ValueError(
            f"TRNBLAS_PAIR_ENERGY must be one of {_PAIR_ENERGY_CHOICES}; got {choice!r}"
        )
    if _use_simulator() or not _on_neuron_device(B) or _get_dispatch_path() != "native":
        return "legacy"
    if choice != "auto":
        return choice
    dims = _pair_sym_dims(*B.shape)
    if dims["sbuf_bytes"] > _PAIR_SYM_SBUF_BUDGET:
        return "aten"
    return "sym" if dims["row_flop"] >= _PAIR_SYM_MIN_ROW_FLOP else "aten"


def _torch_chunked_pair_energy(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
    mem_budget_bytes: int = 1_500_000_000,
) -> torch.Tensor:
    """MP2 energy as chunked GEMM + torch reduction (the aten path).

    T_chunk = B[i_chunk] @ B_flat.T, then sum T * (2T - T^T) / denom, with the
    i chunk sized so one T_chunk stays under `mem_budget_bytes`.
    """
    nocc, nvir, naux = B.shape
    i_block = max(1, min(nocc, int(mem_budget_bytes // (4 * nvir * nocc * nvir))))
    B_flat = B.reshape(nocc * nvir, naux)
    eps_o_pair = eps_occ.view(nocc, 1, 1, 1) + eps_occ.view(1, nocc, 1, 1)
    eps_v_pair = eps_vir.view(1, 1, nvir, 1) + eps_vir.view(1, 1, 1, nvir)
    e = torch.zeros((), dtype=B.dtype, device=B.device)
    for i0 in range(0, nocc, i_block):
        i1 = min(i0 + i_block, nocc)
        T = (B_flat[i0 * nvir : i1 * nvir] @ B_flat.T).reshape(i1 - i0, nvir, nocc, nvir)
        T = T.permute(0, 2, 1, 3)
        e = e + (T * (2.0 * T - T.transpose(-2, -1)) / (eps_o_pair[i0:i1] - eps_v_pair)).sum()
    return e


def _pair_sym_feeds(B: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor, dims: dict):
    """Device operands for `_pair_energy_sym_kernel`: B[j].T stacked twice (so each
    row's j window is a contiguous slice), the matching eps window row, and the
    padded eps_vir column/row."""
    nocc, nvir, naux = B.shape
    nv, na = dims["nvir_pad"], dims["naux_pad"]
    if (nv, na) != (nvir, naux):
        Bp = torch.zeros(nocc, nv, na, dtype=B.dtype, device=B.device)
        Bp[:, :nvir, :naux] = B
    else:
        Bp = B
    Bt = Bp.permute(0, 2, 1).contiguous()  # (nocc, naux_pad, nvir_pad)
    Bx = torch.cat([Bt, Bt], dim=0)
    eo = eps_occ.reshape(1, nocc).to(B.dtype)
    ex = torch.cat([eo, eo], dim=1)
    ev = torch.zeros(nv, dtype=B.dtype, device=B.device)
    ev[:nvir] = eps_vir
    return Bt, Bx, eo, ex, ev.reshape(nv, 1).contiguous(), ev.reshape(1, nv).contiguous()


def _pair_sym_reference(B: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor):
    """Torch emulation of the sym kernel's (i, d) walk and weights (for CPU tests)."""
    nocc = B.shape[0]
    dims = _pair_sym_dims(*B.shape)
    Bt, Bx, eo, ex, evc, evr = _pair_sym_feeds(B, eps_occ, eps_vir, dims)
    nd_pad = dims["nd_pad"]
    w = torch.tensor(dims["weights"], dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64)
    for i in range(nocc):
        for d in range(nd_pad):
            bi, bj = Bt[i].double(), Bx[i + d].double()
            T = bi.T @ bj
            denom = float(eo[0, i]) + float(ex[0, i + d]) - evc.double() - evr.double()
            total = total + w[d] * (T * (2.0 * T - T.T) / denom).sum()
    return total


def _nki_pair_energy_sym_impl(
    B: torch.Tensor, eps_occ: torch.Tensor, eps_vir: torch.Tensor
) -> torch.Tensor:
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    nocc = B.shape[0]
    dims = _pair_sym_dims(*B.shape)
    if dims["sbuf_bytes"] > _PAIR_SYM_SBUF_BUDGET:
        raise ValueError(
            f"pair_energy_sym needs {dims['sbuf_bytes']} SBUF bytes per partition "
            f"(budget {_PAIR_SYM_SBUF_BUDGET}) for B of shape {tuple(B.shape)}"
        )
    Bt, Bx, eo, ex, evc, evr = _pair_sym_feeds(B, eps_occ, eps_vir, dims)
    nd_pad = dims["nd_pad"]
    op = _get_native_ops()["pair_energy_sym"]
    acc = None
    for i in range(nocc):
        p = op(
            Bt[i],
            Bx[i : i + nd_pad],
            eo[:, i : i + 1],
            ex[:, i : i + nd_pad].contiguous(),
            evc,
            evr,
        )
        acc = p if acc is None else acc + p
    w = torch.tensor(dims["weights"], dtype=acc.dtype).to(acc.device)
    return (acc.sum(0) * w).sum()


def _nki_batched_pair_energy_impl(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    nocc, nvir, naux = B.shape

    # Route to chunked dispatch when the full-batch XLA graph would exceed disk.
    # est_iters mirrors the nl.affine_range trace depth for _batched_pair_kernel.
    _N_A = _round_up(nvir, 128) // 128
    _N_K = _round_up(naux, 128) // 128
    est_iters = (nocc * nocc) * _N_A * _N_A * _N_K
    if est_iters > _batched_pair_chunk_threshold():
        return _nki_batched_pair_energy_chunked_impl(B, eps_occ, eps_vir)

    TILE = 128
    TILE_K = 128

    nvir_pad = _round_up(nvir, TILE)
    naux_pad = _round_up(naux, TILE_K)
    needs_pad = (nvir_pad != nvir) or (naux_pad != naux)

    eps_occ_row = eps_occ.reshape(1, nocc).contiguous()
    ev_col = eps_vir.reshape(-1, 1).contiguous()
    ev_row = eps_vir.reshape(1, -1).contiguous()

    if needs_pad:
        B_feed = torch.zeros(nocc, nvir_pad, naux_pad, dtype=B.dtype, device=B.device)
        B_feed[:, :nvir, :naux] = B
        if nvir_pad != nvir:
            ev_col_feed = torch.zeros(nvir_pad, 1, dtype=eps_vir.dtype, device=eps_vir.device)
            ev_col_feed[:nvir] = ev_col
            ev_row_feed = torch.zeros(1, nvir_pad, dtype=eps_vir.dtype, device=eps_vir.device)
            ev_row_feed[0, :nvir] = ev_row
        else:
            ev_col_feed, ev_row_feed = ev_col, ev_row
    else:
        B_feed = B.contiguous()
        ev_col_feed, ev_row_feed = ev_col, ev_row

    if _use_simulator():
        partial_np = nki.simulate(_batched_pair_kernel)(
            B_feed.cpu().numpy(),
            eps_occ_row.cpu().numpy(),
            ev_col_feed.cpu().numpy(),
            ev_row_feed.cpu().numpy(),
        )
        partial = torch.from_numpy(np.asarray(partial_np)).to(B.device)
    else:
        partial = _launch(
            _batched_pair_kernel,
            "batched_pair",
            B_feed,
            eps_occ_row,
            ev_col_feed.to(B.device),
            ev_row_feed.to(B.device),
        )

    return partial.sum()


def _nki_batched_pair_energy_chunked_impl(
    B: torch.Tensor,
    eps_occ: torch.Tensor,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Chunked batched-pair dispatch: Python i-loop over _j_batched_kernel (#46).

    Used when est_iters > _BATCHED_PAIR_CHUNK_THRESHOLD — i.e. the full-batch
    XLA graph would be too large to compile.

    NRT fallback: if N_A × N_B × N_K > _NRT_INNER_OP_LIMIT, the compiled NEFF
    cannot be loaded by the NeuronCore NRT on trn1 regardless of j_chunk size.
    Falls back to `_torch_batched_pair_energy` immediately (skips ~30 min of
    wasted compilation).  Large shape (inner_ops=648) hits this path; medium
    (inner_ops=192) does not.

    J-sub-chunking: if B:(nocc × nvir_pad × naux_pad) exceeds _J_CHUNK_MAX_B_BYTES,
    the j-loop is split so each call gets B:(j_chunk × nvir_pad × naux_pad) ≤ 256 MB.
    Medium (B ≈ 201 MB): j_chunk_size = nocc, original call signature preserved.
    """
    nocc, nvir, naux = B.shape
    TILE = 128
    TILE_K = 128
    nvir_pad = _round_up(nvir, TILE)
    naux_pad = _round_up(naux, TILE_K)
    needs_pad = (nvir_pad != nvir) or (naux_pad != naux)

    # ---- NRT hardware inner-complexity check ----------------------------------
    # _j_batched_kernel is rejected by the NeuronCore NRT when the inner loop
    # tile count N_A × N_B × N_K exceeds the hardware limit (~192 on trn1).
    # Proactively fall back to avoid ~30 min of wasted NEFF compilation followed
    # by guaranteed NRT_RESOURCE.  j_chunk=1,16,32 all confirmed failing for
    # large shape (inner_ops=648); the limit is on inner_ops, not j_chunk×inner_ops.
    _n_a = nvir_pad // TILE
    _n_k = naux_pad // TILE_K
    _inner_ops = _n_a * _n_a * _n_k  # N_A × N_B × N_K
    if _inner_ops > _NRT_INNER_OP_LIMIT:
        _warn_fallback(
            RuntimeError(
                f"_j_batched_kernel inner tile count {_inner_ops} "
                f"(N_A×N_B×N_K = {_n_a}×{_n_a}×{_n_k}) "
                f"exceeds trn1 NRT hardware limit (~{_NRT_INNER_OP_LIMIT}). "
                f"j_chunk=1,16,32 all confirmed NRT_RESOURCE. Falling back to torch."
            )
        )
        return _torch_batched_pair_energy(B, eps_occ, eps_vir)

    eps_occ_row = eps_occ.reshape(1, nocc).contiguous()
    ev_col = eps_vir.reshape(-1, 1).contiguous()
    ev_row = eps_vir.reshape(1, -1).contiguous()

    if needs_pad:
        B_feed = torch.zeros(nocc, nvir_pad, naux_pad, dtype=B.dtype, device=B.device)
        B_feed[:, :nvir, :naux] = B
        if nvir_pad != nvir:
            ev_col_feed = torch.zeros(nvir_pad, 1, dtype=eps_vir.dtype, device=eps_vir.device)
            ev_col_feed[:nvir] = ev_col
            ev_row_feed = torch.zeros(1, nvir_pad, dtype=eps_vir.dtype, device=eps_vir.device)
            ev_row_feed[0, :nvir] = ev_row
        else:
            ev_col_feed, ev_row_feed = ev_col, ev_row
    else:
        B_feed = B.contiguous()
        ev_col_feed, ev_row_feed = ev_col, ev_row

    # ---- j-sub-chunking ---------------------------------------------------
    # When B:(nocc × nvir_pad × naux_pad) > 256 MB, neuronx-cc OOMs during
    # NEFF compilation.  Split the j-loop so each call gets ≤ 256 MB.
    # Medium (B ≈ 201 MB < 256 MB): j_chunk_size = nocc, no sub-chunking.
    # Large shapes with inner_ops > _NRT_INNER_OP_LIMIT are caught above and
    # returned to torch before reaching this code.
    bytes_per_j = nvir_pad * naux_pad * 4  # float32
    if bytes_per_j * nocc > _J_CHUNK_MAX_B_BYTES:
        _raw = max(1, _J_CHUNK_MAX_B_BYTES // bytes_per_j)
        j_chunk_size = 1 << (_raw.bit_length() - 1)  # floor to power-of-2
    else:
        j_chunk_size = nocc  # no sub-chunking
    nocc_padded = ((nocc + j_chunk_size - 1) // j_chunk_size) * j_chunk_size
    if nocc_padded > nocc:
        _jpad = nocc_padded - nocc
        # Zero B rows → T = B_i @ 0.T = 0 → energy contribution = 0 (safe).
        B_feed = torch.cat(
            [B_feed, torch.zeros(_jpad, nvir_pad, naux_pad, dtype=B.dtype, device=B_feed.device)],
            dim=0,
        )
        eps_occ_row = torch.cat(
            [eps_occ_row, torch.zeros(1, _jpad, dtype=eps_occ.dtype, device=eps_occ_row.device)],
            dim=1,
        )

    if _use_simulator():
        e_total = torch.zeros((), dtype=B.dtype, device=B.device)
        for i in range(nocc):
            b_i_feed = B_feed[i].contiguous()
            eo_i = eps_occ[i : i + 1].reshape(1, 1).contiguous()
            if j_chunk_size == nocc:
                partial_np = nki.simulate(_j_batched_kernel)(
                    b_i_feed.cpu().numpy(),
                    B_feed.cpu().numpy(),
                    eo_i.cpu().numpy(),
                    eps_occ_row.cpu().numpy(),
                    ev_col_feed.cpu().numpy(),
                    ev_row_feed.cpu().numpy(),
                )
                e_total = e_total + float(torch.from_numpy(np.asarray(partial_np)).sum())
            else:
                for j_start in range(0, nocc_padded, j_chunk_size):
                    B_j = B_feed[j_start : j_start + j_chunk_size].contiguous()
                    eo_j = eps_occ_row[0:1, j_start : j_start + j_chunk_size]
                    partial_np = nki.simulate(_j_batched_kernel)(
                        b_i_feed.cpu().numpy(),
                        B_j.cpu().numpy(),
                        eo_i.cpu().numpy(),
                        eo_j.cpu().numpy(),
                        ev_col_feed.cpu().numpy(),
                        ev_row_feed.cpu().numpy(),
                    )
                    e_total = e_total + float(torch.from_numpy(np.asarray(partial_np)).sum())
        return e_total

    # Hardware: move fixed tensors to the device once; slice b_i, eo_i, and
    # j-chunks per i on the device (native or XLA, via `_to_device`).
    # When j_chunk_size == nocc: original single-call path (preserves NEFF cache).
    # When j_chunk_size < nocc: j-sub-chunked path; all j-chunks share the same
    # B_j shape → ONE NEFF compile, nocc × (nocc_padded/j_chunk_size) warm dispatches.
    (B_xla, eo_row_xla, evc_xla, evr_xla), orig = _to_device(
        B_feed,
        eps_occ_row,
        ev_col_feed.to(B.device),
        ev_row_feed.to(B.device),
    )
    e_total = torch.zeros((), dtype=B.dtype, device=B.device)
    for i in range(nocc):
        b_i_xla = B_xla[i]  # (nvir_pad, naux_pad)
        eo_i_xla = eo_row_xla[0:1, i : i + 1]  # (1, 1)
        if j_chunk_size == nocc:
            # Original call signature — no slice, preserves existing NEFF cache.
            partial = _call_kernel(
                _j_batched_kernel,
                "j_batched",
                b_i_xla,
                B_xla,
                eo_i_xla,
                eo_row_xla,
                evc_xla,
                evr_xla,
            ).to(orig)
            e_total = e_total + partial.sum()
        else:
            for j_start in range(0, nocc_padded, j_chunk_size):
                B_j = B_xla[j_start : j_start + j_chunk_size]
                eo_j = eo_row_xla[0:1, j_start : j_start + j_chunk_size]
                partial = _call_kernel(
                    _j_batched_kernel, "j_batched", b_i_xla, B_j, eo_i_xla, eo_j, evc_xla, evr_xla
                ).to(orig)
                e_total = e_total + partial.sum()
    return e_total


def _torch_fused_pair_energy(
    b_i: torch.Tensor,
    b_j: torch.Tensor,
    eps_occ_i: float,
    eps_occ_j: float,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for the fused GEMM+energy kernel.

    b_i, b_j: (nvir, naux). Returns a 0-d scalar tensor — the MP2
    pair energy contribution for orbital pair (i, j).
    """
    T = b_i @ b_j.T  # (nvir, nvir)
    denom = (
        eps_occ_i
        + eps_occ_j
        - eps_vir.unsqueeze(1)  # (nvir, 1)
        - eps_vir.unsqueeze(0)  # (1, nvir)
    )
    return (T * (2.0 * T - T.T) / denom).sum()


def nki_fused_gemm_energy(
    b_i: torch.Tensor,
    b_j: torch.Tensor,
    eps_occ_i: float,
    eps_occ_j: float,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    """Fused GEMM+energy kernel for one DF-MP2 orbital pair (#38, v0.5.1).

    On NKI backend: one NEFF per pair — the two GEMMs (T and T_T) and
    the VE energy expression all run inside a single @nki.jit, with T
    and T_T produced into PSUM then tensor_copy'd to SBUF.  The 6.58 GB
    T_flat HBM round-trip present in the two-kernel path is eliminated.

    The T_T = B_j @ B_i.T second GEMM avoids nl.load_transpose2d from
    HBM (which would defeat the HBM-elimination goal) by computing T_T
    as a fresh GEMM in the same kernel body.

    On PyTorch backend (or hardware errors): falls back to the torch
    reference — T materialised in HBM then element-wise ops.

    b_i, b_j: (nvir, naux).  eps_occ_i, eps_occ_j: scalar floats.
    eps_vir: (nvir,).  Returns a 0-d scalar tensor.
    """
    if not _use_nki():
        return _torch_fused_pair_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
    try:
        return _nki_fused_gemm_energy_impl(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return _torch_fused_pair_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir)


def _nki_fused_gemm_energy_impl(
    b_i: torch.Tensor,
    b_j: torch.Tensor,
    eps_occ_i: float,
    eps_occ_j: float,
    eps_vir: torch.Tensor,
) -> torch.Tensor:
    if not HAS_NKI:
        raise RuntimeError("NKI not available")

    nvir, naux = b_i.shape
    # load_transpose2d constrains both partition and free dims to ≤ 128.
    TILE = 128
    TILE_K = 128

    nvir_pad = _round_up(nvir, TILE)
    naux_pad = _round_up(naux, TILE_K)
    needs_pad = (nvir_pad != nvir) or (naux_pad != naux)

    # eps_occ_sum: (1, 1) — the scalar sum passed as a 2D tile so NKI
    # can load it cleanly as partition=1, free=1.
    eo_sum = torch.tensor([[eps_occ_i + eps_occ_j]], dtype=torch.float32)

    # eps_vir in column orientation (NVIR, 1) for a-strip loads and
    # row orientation (1, NVIR) for b-strip loads.
    ev_col = eps_vir.reshape(-1, 1).contiguous()
    ev_row = eps_vir.reshape(1, -1).contiguous()

    if needs_pad:
        bi_feed = torch.zeros(nvir_pad, naux_pad, dtype=b_i.dtype, device=b_i.device)
        bi_feed[:nvir, :naux] = b_i
        bj_feed = torch.zeros(nvir_pad, naux_pad, dtype=b_j.dtype, device=b_j.device)
        bj_feed[:nvir, :naux] = b_j
        if nvir_pad != nvir:
            ev_col_feed = torch.zeros(nvir_pad, 1, dtype=eps_vir.dtype, device=eps_vir.device)
            ev_col_feed[:nvir] = ev_col
            ev_row_feed = torch.zeros(1, nvir_pad, dtype=eps_vir.dtype, device=eps_vir.device)
            ev_row_feed[0, :nvir] = ev_row
        else:
            ev_col_feed = ev_col
            ev_row_feed = ev_row
    else:
        bi_feed = b_i.contiguous()
        bj_feed = b_j.contiguous()
        ev_col_feed = ev_col
        ev_row_feed = ev_row

    if _use_simulator():
        partial_np = nki.simulate(_fused_gemm_energy_kernel)(
            bi_feed.cpu().numpy(),
            bj_feed.cpu().numpy(),
            eo_sum.cpu().numpy(),
            ev_col_feed.cpu().numpy(),
            ev_row_feed.cpu().numpy(),
        )
        partial = torch.from_numpy(np.asarray(partial_np)).to(b_i.device)
    else:
        partial = _launch(
            _fused_gemm_energy_kernel,
            "fused_gemm_energy",
            bi_feed,
            bj_feed,
            eo_sum,
            ev_col_feed.to(b_i.device),
            ev_row_feed.to(b_i.device),
        )

    # Slice unpadded rows (zero-pad rows contribute zero energy, but
    # slice keeps the caller's sum from touching padding artefacts).
    if nvir_pad != nvir:
        partial = partial[:nvir, :]
    return partial.sum()


# Host-side 0.5 * (C + C.T) reads C transposed, which is cache-hostile. For
# 8192^2 fp32 on 2 threads (trn2 host) it takes 726 ms out of place, 596 ms
# into cached memory and 224 ms blocked in place; at 1024^2 it takes 5.0 /
# 1.6 ms. On the device it takes 2.9 ms at 8192^2 and 0.4 ms at 1024^2, plus
# a one-off compile of up to ~2.5 s per shape, cached on disk afterwards.
# Results with n >= this go through the device; smaller ones use the blocked
# host version. Both give bitwise the same values as `0.5 * (C + C.T)`.
_SYRK_DEVICE_SYM_MIN_N = 1024
_SYM_BLOCK = 512


def _symmetrize(r: torch.Tensor) -> torch.Tensor:
    """`0.5 * (r + r.T)`. A fresh, owned, dense CPU `r` is updated in place, in
    blocks (no fresh host memory, cache-friendly transposes)."""
    if not (
        r.device.type == "cpu"
        and r.dim() == 2
        and r.is_contiguous()
        and r.is_floating_point()
        and not (torch.is_grad_enabled() and r.requires_grad)
    ):
        return 0.5 * (r + r.T)
    n, bs = r.shape[0], _SYM_BLOCK
    for i in range(0, n, bs):
        ie = min(i + bs, n)
        d = r[i:ie, i:ie]
        d.copy_(0.5 * (d + d.T))
        for j in range(ie, n, bs):
            je = min(j + bs, n)
            up, lo = r[i:ie, j:je], r[j:je, i:ie]
            up.add_(lo.T).mul_(0.5)
            lo.copy_(up.T)
    return r


def _nki_syrk_impl(A: torch.Tensor, symmetrize: bool = False) -> torch.Tensor:
    """NKI SYRK implementation. Returns A @ A.T for A of shape (M, K).

    Pads M to TILE_M and K to TILE_K multiples. For M > TILE_N, pads to
    a multiple of TILE_N in the N (= M) direction so the kernel can tile
    output cleanly. Falls back to torch.matmul on kernel errors unless
    TRNBLAS_REQUIRE_NKI=1.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    M, K = A.shape
    M_pad = _round_up(M, _TILE_M)
    K_pad = _round_up(K, _TILE_K)
    # Output is (M_pad, M_pad); same TILE_N logic as GEMM applies.
    N_pad = M_pad if M_pad <= _TILE_N else _round_up(M_pad, _TILE_N)
    # M_pad must also equal N_pad (output is square); enforce.
    M_pad = max(M_pad, N_pad)
    needs_pad = (M_pad != M) or (K_pad != K)

    try:
        if needs_pad:
            A_p = torch.zeros(M_pad, K_pad, dtype=A.dtype, device=A.device)
            A_p[:M, :K] = A
            A_feed = A_p.contiguous()
        else:
            A_feed = A.contiguous()

        if _use_simulator():
            out_np = nki.simulate(_syrk_kernel)(A_feed.cpu().numpy())
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            (a,), orig = _to_device(A_feed)
            result = _call_kernel(_syrk_kernel, "syrk", a)
            if symmetrize and M >= _SYRK_DEVICE_SYM_MIN_N:
                # Symmetrise on the device, before the result comes down.
                r = result[:M, :M] if needs_pad else result
                return _to_host(0.5 * (r + r.T), orig)
            result = _to_host(result, orig)
        if needs_pad:
            result = result[:M, :M]
        if not symmetrize:
            return result
        if not result.is_contiguous():
            result = _host_empty_like(result).copy_(result)
        return _symmetrize(result)
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        r = torch.matmul(A, A.T)
        return _symmetrize(r) if symmetrize else r


if HAS_NKI:
    # ---------------------------------------------------------------------------
    # GEMM tile-config kernels — one @nki.jit per candidate in _TILE_CANDIDATES.
    #
    # NKI's AST-based compiler reads source from the on-disk file and resolves
    # names from the local namespace only; closure variables from an enclosing
    # factory function are treated as "unbound" and cause a compile error.
    # The fix is to define each variant as a module-level function with literal
    # integer constants (no closures), and register all six in
    # _gemm_kernel_registry at import time.
    # ---------------------------------------------------------------------------

    @nki.jit
    def _gemm_kernel_64_128_128(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = 64
        TILE_K = 128
        TILE_N = 128 if N > 128 else N
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=c_sbuf)
        return c

    @nki.jit
    def _gemm_kernel_64_128_256(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = 64
        TILE_K = 128
        TILE_N = 256 if N > 256 else N
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=c_sbuf)
        return c

    @nki.jit
    def _gemm_kernel_64_128_512(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = 64
        TILE_K = 128
        TILE_N = 512 if N > 512 else N
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=c_sbuf)
        return c

    @nki.jit
    def _gemm_kernel_128_128_128(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = 128
        TILE_K = 128
        TILE_N = 128 if N > 128 else N
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=c_sbuf)
        return c

    @nki.jit
    def _gemm_kernel_128_128_256(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = 128
        TILE_K = 128
        TILE_N = 256 if N > 256 else N
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=c_sbuf)
        return c

    @nki.jit
    def _gemm_kernel_128_128_512(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = 128
        TILE_K = 128
        TILE_N = 512 if N > 512 else N
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K
                    a_t = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    b_tile = nl.load(b[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=c_sbuf)
        return c

    @nki.jit
    def _gemm_kernel_ragged(a, b):
        """GEMM with in-kernel edge tiles for unaligned M/K/N (#50).

        Tile sizes are the default (128, 128, 512), clamped to the dimension
        when it is smaller. All shapes are static per compile: each dimension
        splits into ``D // T`` full tiles plus one remainder tile of
        ``D - (D // T) * T`` (see ``_ragged_dim_plan``). The K remainder is
        accumulated into the same PSUM tile as a shorter matmul whose
        stationary and moving operands both have ``KR`` partitions, starting
        at partition 0. M/N remainder tiles are narrower PSUM/SBUF tiles.
        The body is written out once per (M-tile kind, N-tile kind) because
        the frontend resolves names from source and does not support kernel
        helper closures.
        """
        M, K = a.shape
        _, N = b.shape
        TM = 128 if M > 128 else M
        TK = 128 if K > 128 else K
        TN = 512 if N > 512 else N
        MF = M // TM
        KF = K // TK
        NF = N // TN
        MR = M - MF * TM
        KR = K - KF * TK
        NR = N - NF * TN
        M0 = MF * TM
        K0 = KF * TK
        N0 = NF * TN
        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

        # Full M tiles.
        for m in nl.affine_range(MF):
            m_off = m * TM
            # Full M x full N.
            for n in nl.affine_range(NF):
                n_off = n * TN
                psum = nl.zeros((TM, TN), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(KF):
                    k_off = k * TK
                    a_t = nl.load_transpose2d(a[m_off : m_off + TM, k_off : k_off + TK])
                    b_t = nl.load(b[k_off : k_off + TK, n_off : n_off + TN])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_t, accumulate=True)
                if KR > 0:
                    a_r = nl.load_transpose2d(a[m_off : m_off + TM, K0:K])
                    b_r = nl.load(b[K0:K, n_off : n_off + TN])
                    nisa.nc_matmul(dst=psum, stationary=a_r, moving=b_r, accumulate=True)
                c_sbuf = nl.ndarray((TM, TN), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(c[m_off : m_off + TM, n_off : n_off + TN], value=c_sbuf)
            # Full M x N remainder.
            if NR > 0:
                psum_n = nl.zeros((TM, NR), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(KF):
                    k_off = k * TK
                    a_t = nl.load_transpose2d(a[m_off : m_off + TM, k_off : k_off + TK])
                    b_t = nl.load(b[k_off : k_off + TK, N0:N])
                    nisa.nc_matmul(dst=psum_n, stationary=a_t, moving=b_t, accumulate=True)
                if KR > 0:
                    a_r = nl.load_transpose2d(a[m_off : m_off + TM, K0:K])
                    b_r = nl.load(b[K0:K, N0:N])
                    nisa.nc_matmul(dst=psum_n, stationary=a_r, moving=b_r, accumulate=True)
                c_n = nl.ndarray((TM, NR), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_n, dst=c_n)
                nl.store(c[m_off : m_off + TM, N0:N], value=c_n)

        # M remainder tile.
        if MR > 0:
            # M remainder x full N.
            for n in nl.affine_range(NF):
                n_off = n * TN
                psum_m = nl.zeros((MR, TN), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(KF):
                    k_off = k * TK
                    a_t = nl.load_transpose2d(a[M0:M, k_off : k_off + TK])
                    b_t = nl.load(b[k_off : k_off + TK, n_off : n_off + TN])
                    nisa.nc_matmul(dst=psum_m, stationary=a_t, moving=b_t, accumulate=True)
                if KR > 0:
                    a_r = nl.load_transpose2d(a[M0:M, K0:K])
                    b_r = nl.load(b[K0:K, n_off : n_off + TN])
                    nisa.nc_matmul(dst=psum_m, stationary=a_r, moving=b_r, accumulate=True)
                c_m = nl.ndarray((MR, TN), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_m, dst=c_m)
                nl.store(c[M0:M, n_off : n_off + TN], value=c_m)
            # M remainder x N remainder (corner).
            if NR > 0:
                psum_mn = nl.zeros((MR, NR), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(KF):
                    k_off = k * TK
                    a_t = nl.load_transpose2d(a[M0:M, k_off : k_off + TK])
                    b_t = nl.load(b[k_off : k_off + TK, N0:N])
                    nisa.nc_matmul(dst=psum_mn, stationary=a_t, moving=b_t, accumulate=True)
                if KR > 0:
                    a_r = nl.load_transpose2d(a[M0:M, K0:K])
                    b_r = nl.load(b[K0:K, N0:N])
                    nisa.nc_matmul(dst=psum_mn, stationary=a_r, moving=b_r, accumulate=True)
                c_mn = nl.ndarray((MR, NR), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_mn, dst=c_mn)
                nl.store(c[M0:M, N0:N], value=c_mn)
        return c

    # Register all six variants.  The default (128, 128, 512) is also kept as
    # the module-level alias _gemm_kernel for backward compatibility.
    _gemm_kernel_registry.update(
        {
            (64, 128, 128): _gemm_kernel_64_128_128,
            (64, 128, 256): _gemm_kernel_64_128_256,
            (64, 128, 512): _gemm_kernel_64_128_512,
            (128, 128, 128): _gemm_kernel_128_128_128,
            (128, 128, 256): _gemm_kernel_128_128_256,
            (128, 128, 512): _gemm_kernel_128_128_512,
        }
    )
    _gemm_kernel = _gemm_kernel_registry[(_TILE_M, _TILE_K, _TILE_N)]

    @nki.jit
    def _mp2_energy_kernel(T_flat, eps_occ_chunk, eps_occ_full, eps_vir_col, eps_vir_row):
        """Fused MP2 energy reduction (#15, M2 correctness fix).

        Computes Σ_{i<ic, j<nocc, a,b<nvir} T[i,j,a,b] *
        (2 T[i,j,a,b] - T[i,j,b,a]) / denom[i,j,a,b] where
        T[i,j,a,b] = T_flat[i*nvir + a, j*nvir + b].

        Sub-tiles the nvir partition dim into strips of P_TILE ≤ 128
        (largest divisor of NVIR under the NKI partition limit).
        Strip partials are accumulated in a (P_TILE, 1) SBUF register
        per (i, j).

        **#35 cross-pair batching.** Rather than storing each (i, j)
        pair's partial immediately to HBM (IC×NOCC stores per chunk),
        all NOCC j-partials for a given i are accumulated in an SBUF
        tile `acc_j (P_TILE, NOCC)` and flushed in a single HBM store
        after the j-loop — IC stores per chunk total. Hypothesis: the
        per-pair store was a serialization fence preventing the compiler
        from pipelining consecutive pairs. SBUF cost: P_TILE×NOCC×4
        bytes (≤ 43 KB at the large bench shape; well under budget).

        eps_* args are shape (1, N) so `nl.load` interprets them as
        partition=1, free=N (a 1D load is treated as partition=len,
        which would exceed the 128-partition limit for NVIR > 128).

        **NKI 0.3.0 broadcast fix.** Partition-mismatched
        tensor-tensor arith (`(1,1) ⊕ (P_TILE,1)`, `(P_TILE,1) ⊕
        (1,NVIR)`) is rejected by the MLIR verifier. `denom` is built
        by lifting all three eps operands to `(P_TILE, NVIR)` via
        `nl.broadcast_to` first, so every subsequent subtract sees
        matching partition dims. This replaces the M1 pattern that
        re-skipped the 5 MP2 tests during the 0.3.0 migration.

        NKI only supports reduction along the free dim. A full
        `(P_TILE, NVIR) → scalar` reduce would need a partition-axis
        reduce which NKI rejects. Instead: reduce free-only to get
        per-partition partials `(P_TILE, 1)`, batch NOCC of them in
        SBUF, emit `(P_TILE, IC*NOCC)` to HBM — caller `.sum()`
        handles the final partition-axis reduction host-side.
        """
        NVIR = eps_vir_row.shape[1]
        IC = eps_occ_chunk.shape[1]
        NOCC = eps_occ_full.shape[1]

        P_TILE = min(NVIR, 128)
        while NVIR % P_TILE != 0:
            P_TILE -= 1
        NSTRIP = NVIR // P_TILE

        # Output layout: (P_TILE, IC*NOCC) — 2D flat so each i-row's
        # NOCC partials can be stored in one nl.store call.  Partition
        # axis FIRST so the store aligns with SBUF partition layout.
        # Host caller's .sum() is shape-agnostic.
        e_partial = nl.ndarray((P_TILE, IC * NOCC), dtype=nl.float32, buffer=nl.shared_hbm)
        # Full eps_vir as a free-dim vector (partition=1, free=NVIR)
        # for the per-b axis of denom.
        ev_row = nl.load(eps_vir_row[0:1, 0:NVIR])

        for i in nl.affine_range(IC):
            eo_i = nl.load(eps_occ_chunk[0:1, i : i + 1])
            # Batch all NOCC pair partials for this i into SBUF before
            # a single HBM store.  IC×NOCC stores → IC stores per chunk.
            acc_j = nl.zeros((P_TILE, NOCC), dtype=nl.float32, buffer=nl.sbuf)
            for j in nl.affine_range(NOCC):
                eo_j = nl.load(eps_occ_full[0:1, j : j + 1])
                eo_sum = nl.add(eo_i, eo_j)

                # Per-strip SBUF slots. Each affine_range iteration
                # writes its own column; a single nl.sum over the
                # NSTRIP free axis at the end reduces strips.
                # (In-place += across affine_range hits NKI's
                # "Unexpected output dependencies" — the compiler
                # wants the strip index in the dst access explicitly.)
                acc_rows = nl.zeros((P_TILE, NSTRIP), dtype=nl.float32, buffer=nl.sbuf)

                for s in nl.affine_range(NSTRIP):
                    a_off = s * P_TILE
                    t = nl.load(
                        T_flat[
                            i * NVIR + a_off : i * NVIR + a_off + P_TILE, j * NVIR : (j + 1) * NVIR
                        ]
                    )
                    t_T = nl.load_transpose2d(
                        T_flat[
                            i * NVIR : (i + 1) * NVIR, j * NVIR + a_off : j * NVIR + a_off + P_TILE
                        ]
                    )
                    # (P_TILE, 1) partition-axis load of eps_vir's
                    # a-strip — matches the output axis of the (P_TILE,
                    # NVIR) tile we're operating on.
                    ev_col = nl.load(eps_vir_col[a_off : a_off + P_TILE, 0:1])
                    # denom[a, b] = eo_sum - ev_col[a] - ev_row[b]
                    #
                    # NKI 0.3.0 MLIR verifier rejects tensor-tensor
                    # arithmetic when partition dims don't match (e.g.
                    # `(1, 1) - (P_TILE, 1)` or `(P_TILE, 1) - (1, N)`).
                    # Use `nl.broadcast_to` to explicitly lift every
                    # operand to the target `(P_TILE, NVIR)` shape first,
                    # then subtract at matching partition dims.
                    eo_sum_bc = nl.broadcast_to(eo_sum, (P_TILE, NVIR))
                    ev_col_bc = nl.broadcast_to(ev_col, (P_TILE, NVIR))
                    ev_row_bc = nl.broadcast_to(ev_row, (P_TILE, NVIR))
                    denom = nl.subtract(
                        nl.subtract(eo_sum_bc, ev_col_bc),
                        ev_row_bc,
                    )
                    # NKI 0.3.0 drops tensor-tensor nl.divide;
                    # substitute multiply × reciprocal.
                    term = nl.multiply(
                        nl.multiply(
                            t,
                            nl.subtract(nl.multiply(t, 2.0), t_T),
                        ),
                        nl.reciprocal(denom),
                    )
                    # Free-dim reduce: (P_TILE, NVIR) → (P_TILE, 1).
                    strip_partial = nl.sum(term, axis=1, keepdims=True)
                    # Write the strip's partial into its own slot (s
                    # indexes the free axis of acc_rows).
                    acc_rows[0:P_TILE, s : s + 1] = strip_partial

                # Reduce across strips (free dim) → (P_TILE, 1).
                acc_row = nl.sum(acc_rows, axis=1, keepdims=True)

                # Accumulate this pair's partial into SBUF; no HBM
                # traffic until after all NOCC j-iterations complete.
                acc_j[0:P_TILE, j : j + 1] = acc_row

            # One HBM store for all NOCC pairs in this i-row.
            nl.store(
                e_partial[0:P_TILE, i * NOCC : (i + 1) * NOCC],
                value=acc_j,
            )

        return e_partial

    @nki.jit
    def _syrk_kernel(a):
        """Symmetric rank-k: C = a @ a.T with single-A HBM load.

        Structurally identical to _gemm_kernel, but the "moving"
        operand is loaded from the same `a` HBM region via a
        second `load_transpose2d` — avoiding the materialised
        `a.T.contiguous()` that `nki_gemm(A, A.T)` would otherwise
        issue. K partition dim is at the load_transpose2d limit of 128.

        Caller guarantees M, K are multiples of 128 and M is either
        ≤ 512 or a multiple of 512 (handled by _nki_syrk_impl's HBM
        padding).
        """
        M, K = a.shape

        TILE_M = _TILE_M
        TILE_K = _TILE_K
        TILE_N = M if M <= _TILE_N else _TILE_N

        c = nl.ndarray((M, M), dtype=a.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(M // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    # Stationary: a[m_off:m_off+TILE_M, k_off:k_off+TILE_K]
                    # transposed → (TILE_K, TILE_M), partition=K ≤ 128.
                    a_stat = nl.load_transpose2d(a[m_off : m_off + TILE_M, k_off : k_off + TILE_K])
                    # Moving: a.T[k_off:k_off+TILE_K, n_off:n_off+TILE_N]
                    # = a[n_off:n_off+TILE_N, k_off:k_off+TILE_K].T
                    # load_transpose2d swaps axes → (TILE_K, TILE_N),
                    # partition=K ≤ 128.
                    a_mov = nl.load_transpose2d(a[n_off : n_off + TILE_N, k_off : k_off + TILE_K])
                    nisa.nc_matmul(dst=psum, stationary=a_stat, moving=a_mov, accumulate=True)

                # PSUM → SBUF via tensor_copy (NKI 0.3.0 rejects
                # dma_copy reading from PSUM; nl.copy returns a view).
                c_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=c_sbuf)
                nl.store(
                    c[m_off : m_off + TILE_M, n_off : n_off + TILE_N],
                    value=c_sbuf,
                )

        return c

    @nki.jit
    def _fused_gemm_energy_kernel(b_i, b_j, eps_occ_sum, eps_vir_col, eps_vir_row):
        """Fused GEMM+energy kernel for one DF-MP2 pair (#38, v0.5.1).

        b_i, b_j: (NVIR_pad, NAUX_pad) — DF-fitted B matrices, zero-padded
            to TILE multiples by the Python wrapper.
        eps_occ_sum: (1, 1) — eps_occ_i + eps_occ_j pre-summed by caller.
        eps_vir_col: (NVIR_pad, 1) — eps_vir in partition-axis orientation
            (a-strip loads pick TILE rows).
        eps_vir_row: (1, NVIR_pad) — eps_vir in free-axis orientation
            (b-strip loads pick TILE columns).

        Returns e_partial: (NVIR_pad, 1).  Caller `.sum()` for the scalar.

        **Two-GEMM T_T strategy:**  T.T[a,b] = T[b,a] = (B_j @ B_i.T)[a,b].
        Rather than nl.load_transpose2d of T from HBM (which defeats the
        HBM-elimination goal), T_T is computed as a second GEMM tile in the
        same @nki.jit.  Both T and T_T land in SBUF via tensor_copy — no
        HBM write for either intermediate.

        **TILE = 128 everywhere:** nl.load_transpose2d constrains both input
        dimensions to ≤ 128.  TILE_M = TILE_N = TILE_K = 128.

        **Accumulation pattern:** For each a-strip we accumulate b-strip
        partials into acc_b (TILE, N_B_TILES) in SBUF, then do one
        nl.sum(axis=1) → (TILE, 1) before a single nl.store per a-strip.
        This is the same cross-pair batching strategy used in
        _mp2_energy_kernel to minimise HBM traffic.

        **NKI 0.3.0 broadcast fix:** All denom operands are lifted to
        (TILE, TILE) via nl.broadcast_to before arithmetic (same fix as
        _mp2_energy_kernel for the MLIR partition-matching requirement).
        """
        NVIR, NAUX = b_i.shape
        TILE = 128
        TILE_K = 128
        N_A_TILES = NVIR // TILE
        N_B_TILES = NVIR // TILE
        N_K_TILES = NAUX // TILE_K

        e_partial = nl.ndarray((NVIR, 1), dtype=nl.float32, buffer=nl.shared_hbm)

        # Scalar eps_occ_sum: load once, reuse across all (a, b) tiles.
        eo_sum = nl.load(eps_occ_sum[0:1, 0:1])

        for a in nl.affine_range(N_A_TILES):
            a_off = a * TILE
            # eps_vir for the a-strip: (TILE, 1) — partition axis slice.
            ev_col_a = nl.load(eps_vir_col[a_off : a_off + TILE, 0:1])

            # Batch all b-strip partials in SBUF before a single HBM store
            # per a-strip.  Shape (TILE, N_B_TILES): column b holds the
            # (TILE, 1) sum for that b-strip.
            acc_b = nl.zeros((TILE, N_B_TILES), dtype=nl.float32, buffer=nl.sbuf)

            for b in nl.affine_range(N_B_TILES):
                b_off = b * TILE
                # eps_vir for the b-strip: (1, TILE) — free axis slice.
                ev_row_b = nl.load(eps_vir_row[0:1, b_off : b_off + TILE])

                # --- GEMM 1: T[a_strip, b_strip] = B_i[a_strip,:] @ B_j[b_strip,:].T ---
                # Stationary: B_i[a_off:, k_off:] → load_transpose2d → (TILE_K, TILE).
                # Moving:     B_j[b_off:, k_off:] → load_transpose2d → (TILE_K, TILE).
                # PSUM result shape: (TILE, TILE).
                psum_t = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(N_K_TILES):
                    k_off = k * TILE_K
                    bi_stat = nl.load_transpose2d(b_i[a_off : a_off + TILE, k_off : k_off + TILE_K])
                    bj_mov = nl.load_transpose2d(b_j[b_off : b_off + TILE, k_off : k_off + TILE_K])
                    nisa.nc_matmul(dst=psum_t, stationary=bi_stat, moving=bj_mov, accumulate=True)
                t_sbuf = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_t, dst=t_sbuf)

                # --- GEMM 2: T_T[a_strip,b_strip] = B_j[a_strip,:] @ B_i[b_strip,:].T ---
                # T.T[a,b] = T[b,a] = (B_j @ B_i.T)[a,b].
                # Stationary: B_j[a_off:, k_off:] → load_transpose2d.
                # Moving:     B_i[b_off:, k_off:] → load_transpose2d.
                psum_tt = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(N_K_TILES):
                    k_off = k * TILE_K
                    bj_stat = nl.load_transpose2d(b_j[a_off : a_off + TILE, k_off : k_off + TILE_K])
                    bi_mov = nl.load_transpose2d(b_i[b_off : b_off + TILE, k_off : k_off + TILE_K])
                    nisa.nc_matmul(dst=psum_tt, stationary=bj_stat, moving=bi_mov, accumulate=True)
                t_t_sbuf = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum_tt, dst=t_t_sbuf)

                # --- VE: energy = T * (2T - T_T) / denom, sum over b-strip ---
                # denom[a, b] = eo_sum - ev_vir_col[a] - ev_vir_row[b]
                #
                # NKI 0.3.0 MLIR verifier rejects arithmetic between tiles
                # with mismatched partition dims.  Broadcast all operands to
                # (TILE, TILE) first (same fix as _mp2_energy_kernel).
                eo_bc = nl.broadcast_to(eo_sum, (TILE, TILE))
                evc_bc = nl.broadcast_to(ev_col_a, (TILE, TILE))
                evr_bc = nl.broadcast_to(ev_row_b, (TILE, TILE))
                denom = nl.subtract(nl.subtract(eo_bc, evc_bc), evr_bc)

                two_t = nl.multiply(t_sbuf, 2.0)
                diff = nl.subtract(two_t, t_t_sbuf)
                numer = nl.multiply(t_sbuf, diff)
                energy_tile = nl.multiply(numer, nl.reciprocal(denom))

                # Sum over b-strip (free dim): (TILE, TILE) → (TILE, 1).
                b_partial = nl.sum(energy_tile, axis=1, keepdims=True)
                acc_b[0:TILE, b : b + 1] = b_partial

            # Sum over all b-strip columns (free dim): (TILE, N_B_TILES) → (TILE, 1).
            e_a = nl.sum(acc_b, axis=1, keepdims=True)
            nl.store(e_partial[a_off : a_off + TILE, 0:1], value=e_a)

        return e_partial

    @nki.jit
    def _batched_pair_kernel(B, eps_occ_row, eps_vir_col, eps_vir_row):
        """Batched DF-MP2 pair energy — one NEFF for all NOCC² pairs (#43).

        Eliminates the O(nocc²) × 100ms Neuron XLA per-dispatch overhead by
        computing all orbital-pair energies inside a single @nki.jit.

        B: (NOCC, NVIR_pad, NAUX_pad) — zero-padded to TILE multiples.
        eps_occ_row: (1, NOCC) — occupied orbital energies.
        eps_vir_col: (NVIR_pad, 1) — virtual energies, column orientation.
        eps_vir_row: (1, NVIR_pad) — virtual energies, row orientation.
        Returns: e_partial (TILE, NOCC²) — host calls .sum() for scalar.

        **Loop structure (5 levels of nl.affine_range):**
          i-loop  →  j-loop  →  a-strip  →  b-strip  →  k-tile (GEMM)
          acc_j (TILE, NOCC) batches j-partials per i-row before HBM store.
          acc_a (TILE, N_A) batches a-strip sums per j before j-slot write.
          acc_b (TILE, N_B) batches b-strip partials per a-strip.

        **3D indexing:** Spike A confirmed nl.load_transpose2d(B[i,...]) and
        nl.load_transpose2d(B[j,...]) compile and produce correct results when
        i, j are nl.affine_range loop variables.  Validated on trn1 2026-04-17.

        **NKI partition-axis rule:** no axis=0 reductions inside kernel.
        All reductions are free-dim (axis=1). Host does final .sum().

        **GEMM tile size:** TILE = TILE_K = 128 (nl.load_transpose2d limit on
        trn1). Output partition dim is therefore always 128.
        """
        NOCC = B.shape[0]
        NVIR = B.shape[1]  # NVIR_pad — multiple of TILE
        NAUX = B.shape[2]  # NAUX_pad — multiple of TILE_K

        TILE = 128
        TILE_K = 128
        N_A_TILES = NVIR // TILE
        N_B_TILES = NVIR // TILE
        N_K_TILES = NAUX // TILE_K

        # Output: (TILE, NOCC²).  Host .sum() gives total MP2 energy.
        e_partial = nl.ndarray((TILE, NOCC * NOCC), dtype=nl.float32, buffer=nl.shared_hbm)

        for i in nl.affine_range(NOCC):
            eps_i = nl.load(eps_occ_row[0:1, i : i + 1])  # (1, 1)

            # Batch all NOCC j-partials for this i-row before HBM store.
            acc_j = nl.zeros((TILE, NOCC), dtype=nl.float32, buffer=nl.sbuf)

            for j in nl.affine_range(NOCC):
                eps_j = nl.load(eps_occ_row[0:1, j : j + 1])  # (1, 1)
                eps_occ_sum = nl.add(eps_i, eps_j)  # (1, 1)

                # Batch all N_A_TILES a-strip partials before j-slot write.
                acc_a = nl.zeros((TILE, N_A_TILES), dtype=nl.float32, buffer=nl.sbuf)

                for a in nl.affine_range(N_A_TILES):
                    a_off = a * TILE
                    ev_col_a = nl.load(eps_vir_col[a_off : a_off + TILE, 0:1])  # (TILE, 1)

                    # Batch all N_B_TILES b-strip partials per a-strip.
                    acc_b = nl.zeros((TILE, N_B_TILES), dtype=nl.float32, buffer=nl.sbuf)

                    for b in nl.affine_range(N_B_TILES):
                        b_off = b * TILE
                        ev_row_b = nl.load(eps_vir_row[0:1, b_off : b_off + TILE])  # (1, TILE)

                        # GEMM 1: T[a_strip, b_strip] = B[i][a_strip,:] @ B[j][b_strip,:].T
                        # B[i, a_off:, k_off:] with affine_range vars i, a, k — 3D indexing.
                        psum_t = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                        for k in nl.affine_range(N_K_TILES):
                            k_off = k * TILE_K
                            bi_stat = nl.load_transpose2d(
                                B[i, a_off : a_off + TILE, k_off : k_off + TILE_K]
                            )  # (TILE_K, TILE)
                            bj_mov = nl.load_transpose2d(
                                B[j, b_off : b_off + TILE, k_off : k_off + TILE_K]
                            )  # (TILE_K, TILE)
                            nisa.nc_matmul(
                                dst=psum_t, stationary=bi_stat, moving=bj_mov, accumulate=True
                            )
                        t_sbuf = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(src=psum_t, dst=t_sbuf)

                        # GEMM 2: T_T[a_strip,b_strip] = B[j][a_strip,:] @ B[i][b_strip,:].T
                        # T_T[a,b] = T[b,a] = (B[j] @ B[i].T)[a,b].
                        psum_tt = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                        for k in nl.affine_range(N_K_TILES):
                            k_off = k * TILE_K
                            bj_stat = nl.load_transpose2d(
                                B[j, a_off : a_off + TILE, k_off : k_off + TILE_K]
                            )  # (TILE_K, TILE)
                            bi_mov = nl.load_transpose2d(
                                B[i, b_off : b_off + TILE, k_off : k_off + TILE_K]
                            )  # (TILE_K, TILE)
                            nisa.nc_matmul(
                                dst=psum_tt, stationary=bj_stat, moving=bi_mov, accumulate=True
                            )
                        t_t_sbuf = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(src=psum_tt, dst=t_t_sbuf)

                        # VE: energy = T * (2T - T_T) / denom, sum over b-strip.
                        eo_bc = nl.broadcast_to(eps_occ_sum, (TILE, TILE))
                        evc_bc = nl.broadcast_to(ev_col_a, (TILE, TILE))
                        evr_bc = nl.broadcast_to(ev_row_b, (TILE, TILE))
                        denom = nl.subtract(nl.subtract(eo_bc, evc_bc), evr_bc)
                        two_t = nl.multiply(t_sbuf, 2.0)
                        diff = nl.subtract(two_t, t_t_sbuf)
                        numer = nl.multiply(t_sbuf, diff)
                        energy_tile = nl.multiply(numer, nl.reciprocal(denom))
                        # Free-dim reduce: (TILE, TILE) → (TILE, 1).
                        b_partial = nl.sum(energy_tile, axis=1, keepdims=True)
                        acc_b[0:TILE, b : b + 1] = b_partial

                    # Free-dim reduce b-strips: (TILE, N_B_TILES) → (TILE, 1).
                    e_a = nl.sum(acc_b, axis=1, keepdims=True)
                    acc_a[0:TILE, a : a + 1] = e_a

                # Free-dim reduce a-strips: (TILE, N_A_TILES) → (TILE, 1).
                pair_energy = nl.sum(acc_a, axis=1, keepdims=True)
                acc_j[0:TILE, j : j + 1] = pair_energy

            # One HBM store for all NOCC j-partials in this i-row.
            nl.store(e_partial[0:TILE, i * NOCC : (i + 1) * NOCC], value=acc_j)

        return e_partial

    @nki.jit
    def _j_batched_kernel(b_i, B, eps_occ_i, eps_occ_row, eps_vir_col, eps_vir_row):
        """Chunked batched-pair kernel — one NEFF call per i-row (#46).

        Processes all NOCC j-pairs for a single occupied orbital i.
        The host Python loop iterates over i; all calls share the same
        input shapes → ONE NEFF compile, NOCC warm dispatches.

        b_i:         (NVIR_pad, NAUX_pad) — B[i] slice, 2D (stationary operand).
        B:           (NOCC, NVIR_pad, NAUX_pad) — full matrix (j-indexing inside).
        eps_occ_i:   (1, 1) — eo[i], pre-sliced on host.
        eps_occ_row: (1, NOCC) — all eo[j] values.
        eps_vir_col: (NVIR_pad, 1) — virtual energies, column layout.
        eps_vir_row: (1, NVIR_pad) — virtual energies, row layout.
        Returns: e_partial (TILE, NOCC) — host accumulates across i-rows.

        Loop structure (4 levels of nl.affine_range):
          j-loop → a-strip → b-strip → k-tile (GEMM)

        Differences from _batched_pair_kernel:
        - No outer i affine_range; i is resolved by the host.
        - b_i is 2D — simpler load_transpose2d indexing.
        - eps_occ_i is pre-sliced (1,1), not loaded from eps_occ_row inside kernel.
        - Output (TILE, NOCC) vs (TILE, NOCC²).
        - 3D indexing B[j, ...] with affine_range j — validated on trn1 (Spike A/B).
        """
        NOCC = B.shape[0]
        NVIR = b_i.shape[0]  # NVIR_pad
        NAUX = b_i.shape[1]  # NAUX_pad
        TILE = 128
        TILE_K = 128
        N_A_TILES = NVIR // TILE
        N_B_TILES = NVIR // TILE
        N_K_TILES = NAUX // TILE_K

        e_partial = nl.ndarray((TILE, NOCC), dtype=nl.float32, buffer=nl.shared_hbm)

        # Load eps_occ_i into SBUF explicitly so nl.add operates on SBUF
        # operands, not shared_hbm. Without this load the shared_hbm
        # annotation propagates through the NKI compute graph and corrupts
        # the PSUM allocation for psum_t, causing:
        #   tensor_copy src must be SBUF or PSUM, got shared_hbm
        eo_i = nl.load(eps_occ_i[0:1, 0:1])  # (1, 1), now SBUF-resident

        # Batch all NOCC j-partials before single HBM store.
        acc_j = nl.zeros((TILE, NOCC), dtype=nl.float32, buffer=nl.sbuf)

        for j in nl.affine_range(NOCC):
            eps_j = nl.load(eps_occ_row[0:1, j : j + 1])  # (1, 1)
            eps_occ_sum = nl.add(eo_i, eps_j)  # (1, 1)

            acc_a = nl.zeros((TILE, N_A_TILES), dtype=nl.float32, buffer=nl.sbuf)

            for a in nl.affine_range(N_A_TILES):
                a_off = a * TILE
                ev_col_a = nl.load(eps_vir_col[a_off : a_off + TILE, 0:1])  # (TILE, 1)
                acc_b = nl.zeros((TILE, N_B_TILES), dtype=nl.float32, buffer=nl.sbuf)

                for b in nl.affine_range(N_B_TILES):
                    b_off = b * TILE
                    ev_row_b = nl.load(eps_vir_row[0:1, b_off : b_off + TILE])  # (1, TILE)

                    # GEMM 1: T[a,b] = b_i[a_strip,:] @ B[j][b_strip,:].T
                    psum_t = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(N_K_TILES):
                        k_off = k * TILE_K
                        bi_stat = nl.load_transpose2d(
                            b_i[a_off : a_off + TILE, k_off : k_off + TILE_K]
                        )  # (TILE_K, TILE) — 2D indexing
                        bj_mov = nl.load_transpose2d(
                            B[j, b_off : b_off + TILE, k_off : k_off + TILE_K]
                        )  # (TILE_K, TILE) — 3D with affine_range j
                        nisa.nc_matmul(
                            dst=psum_t, stationary=bi_stat, moving=bj_mov, accumulate=True
                        )
                    t_sbuf = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(src=psum_t, dst=t_sbuf)

                    # GEMM 2: T_T[a,b] = B[j][a_strip,:] @ b_i[b_strip,:].T
                    psum_tt = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(N_K_TILES):
                        k_off = k * TILE_K
                        bj_stat = nl.load_transpose2d(
                            B[j, a_off : a_off + TILE, k_off : k_off + TILE_K]
                        )  # (TILE_K, TILE) — 3D with affine_range j
                        bi_mov = nl.load_transpose2d(
                            b_i[b_off : b_off + TILE, k_off : k_off + TILE_K]
                        )  # (TILE_K, TILE) — 2D indexing
                        nisa.nc_matmul(
                            dst=psum_tt, stationary=bj_stat, moving=bi_mov, accumulate=True
                        )
                    t_t_sbuf = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(src=psum_tt, dst=t_t_sbuf)

                    # VE energy expression (identical to _batched_pair_kernel).
                    eo_bc = nl.broadcast_to(eps_occ_sum, (TILE, TILE))
                    evc_bc = nl.broadcast_to(ev_col_a, (TILE, TILE))
                    evr_bc = nl.broadcast_to(ev_row_b, (TILE, TILE))
                    denom = nl.subtract(nl.subtract(eo_bc, evc_bc), evr_bc)
                    two_t = nl.multiply(t_sbuf, 2.0)
                    diff = nl.subtract(two_t, t_t_sbuf)
                    numer = nl.multiply(t_sbuf, diff)
                    energy_tile = nl.multiply(numer, nl.reciprocal(denom))
                    b_partial = nl.sum(energy_tile, axis=1, keepdims=True)
                    acc_b[0:TILE, b : b + 1] = b_partial

                e_a = nl.sum(acc_b, axis=1, keepdims=True)
                acc_a[0:TILE, a : a + 1] = e_a

            pair_energy = nl.sum(acc_a, axis=1, keepdims=True)
            acc_j[0:TILE, j : j + 1] = pair_energy

        nl.store(e_partial[0:TILE, 0:NOCC], value=acc_j)
        return e_partial

    @nki.jit
    def _pair_energy_sym_kernel(bt_i, Bw, eps_occ_i, eps_occ_w, eps_vir_col, eps_vir_row):
        """DF-MP2 pair energies for one occupied row i over a window of j (native).

        Launched gridless. At LNC=2 (the Trn2 default) NKI traces it once per
        physical core of the logical core (`num_programs() == 2`) and each core
        takes half of the j window; at LNC=1 there is one program and it takes
        the whole window. The logical-core size comes from
        NEURON_LOGICAL_NC_CONFIG, so the same launch works at both (an explicit
        `kernel[2]` grid fails to compile at LNC=1 with NCC_EVRF066). The
        standalone GEMM kernels above run as one program, which is most of why
        aten `matmul` (which uses both cores) beats them.

        bt_i:        (NAUX, NVIR) = B[i].T, so Tensor Engine operands are plain
                     loads (no DMA transposes). Loaded into SBUF once per launch.
        Bw:          (ND, NAUX, NVIR) j window, B[j].T for j = i + d (mod nocc),
                     d = 0..ND-1. ND is even. Each B[j].T is loaded into SBUF once.
        eps_occ_i:   (1, 1). eps_occ_w: (1, ND), eps for the window.
        eps_vir_col: (NVIR, 1); eps_vir_row: (1, NVIR).
        NAUX and NVIR are multiples of 128 (zero padded by the host).

        Per pair, one GEMM fills T = B[i] B[j].T (NVIR x NVIR) in SBUF a 128-row
        strip at a time (fresh PSUM tile per output block, accumulate over K).
        T.T blocks come from Tensor Engine transposes of the SBUF-resident T, so
        the second GEMM of `_batched_pair_kernel` is gone. Returns per-partition
        partials (128, ND); the host weights each d (pair symmetry e_ij = e_ji)
        and sums.
        """
        ND = Bw.shape[0]
        NAUX = Bw.shape[1]
        NVIR = Bw.shape[2]
        TILE = 128
        TN = 512 if NVIR % 512 == 0 else 128
        NA = NVIR // TILE
        NB = NVIR // TN
        NK = NAUX // TILE
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        if ND % n_prog != 0:  # ND is even, so this only guards n_prog > 2
            n_prog = 1
        pid = nl.program_id(0) if n_prog > 1 else 0
        NJ = ND // n_prog
        e_partial = nl.ndarray((TILE, ND), dtype=nl.float32, buffer=nl.shared_hbm)
        eps_i = nl.load(eps_occ_i[0:1, 0:1])
        ev_row = nl.load(eps_vir_row[0:1, 0:NVIR])
        bi_sb = nl.ndarray((TILE, NK * NVIR), dtype=nl.float32, buffer=nl.sbuf)
        for k in nl.affine_range(NK):
            bi_sb[0:TILE, k * NVIR : (k + 1) * NVIR] = nl.load(
                bt_i[k * TILE : (k + 1) * TILE, 0:NVIR]
            )
        acc_j = nl.zeros((TILE, NJ), dtype=nl.float32, buffer=nl.sbuf)
        for jj in nl.affine_range(NJ):
            j = pid * NJ + jj
            eps_j = nl.load(eps_occ_w[0:1, j : j + 1])
            eo = nl.add(eps_i, eps_j)
            bj_sb = nl.ndarray((TILE, NK * NVIR), dtype=nl.float32, buffer=nl.sbuf)
            for k in nl.affine_range(NK):
                bj_sb[0:TILE, k * NVIR : (k + 1) * NVIR] = nl.load(
                    Bw[j, k * TILE : (k + 1) * TILE, 0:NVIR]
                )
            t_all = nl.ndarray((TILE, NA * NVIR), dtype=nl.float32, buffer=nl.sbuf)
            for a in nl.affine_range(NA):
                a_off = a * TILE
                for b in nl.affine_range(NB):
                    b_off = b * TN
                    pt = nl.zeros((TILE, TN), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(NK):
                        nisa.nc_matmul(
                            dst=pt,
                            stationary=bi_sb[0:TILE, k * NVIR + a_off : k * NVIR + a_off + TILE],
                            moving=bj_sb[0:TILE, k * NVIR + b_off : k * NVIR + b_off + TN],
                            accumulate=True,
                        )
                    nisa.tensor_copy(
                        src=pt, dst=t_all[0:TILE, a * NVIR + b_off : a * NVIR + b_off + TN]
                    )
            acc_ab = nl.zeros((TILE, NA * NA), dtype=nl.float32, buffer=nl.sbuf)
            for a in nl.affine_range(NA):
                a_off = a * TILE
                ev_a = nl.load(eps_vir_col[a_off : a_off + TILE, 0:1])
                for b in nl.affine_range(NA):
                    b_off = b * TILE
                    # T.T[a-strip, b-strip] = (T[b-strip, a-strip]).T
                    ptt = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=ptt, data=t_all[0:TILE, b * NVIR + a_off : b * NVIR + a_off + TILE]
                    )
                    tt = nl.ndarray((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(src=ptt, dst=tt)
                    t = t_all[0:TILE, a * NVIR + b_off : a * NVIR + b_off + TILE]
                    denom = nl.subtract(
                        nl.subtract(
                            nl.broadcast_to(eo, (TILE, TILE)), nl.broadcast_to(ev_a, (TILE, TILE))
                        ),
                        nl.broadcast_to(ev_row[0:1, b_off : b_off + TILE], (TILE, TILE)),
                    )
                    num = nl.multiply(t, nl.subtract(nl.multiply(t, 2.0), tt))
                    term = nl.multiply(num, nl.reciprocal(denom))
                    acc_ab[0:TILE, a * NA + b : a * NA + b + 1] = nl.sum(
                        term, axis=1, keepdims=True
                    )
            acc_j[0:TILE, jj : jj + 1] = nl.sum(acc_ab, axis=1, keepdims=True)
        nl.store(e_partial[0:TILE, pid * NJ : pid * NJ + NJ], value=acc_j)
        return e_partial
