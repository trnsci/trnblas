#!/usr/bin/env python
"""Phase-by-phase timing of a cold first ``trnblas.gemm`` call (#50).

Run one N per fresh process, with fresh compile caches, so every phase is cold:

    python scripts/probe_first_call.py N [--mode MODE] [--input-device neuron|cpu]
    python scripts/probe_first_call.py N --cpu          # CPU dry run / pre-flight

Each phase prints exactly one line, ``PHASE <name> <seconds>``. A phase that
fails prints ``PHASE <name> ERROR <first line>`` and the remaining phases still
run where they can. Device work is fenced with a sync after every phase:
``torch_neuronx.synchronize()`` for native tensors, plus
``xm.mark_step(); xm.wait_device_ops()`` for XLA tensors.

Modes (one per process):

  gemm (default)  Replays what ``trnblas.gemm(1.0, A, B)`` does on the nki
                  backend, using the real dispatch functions in order:
                  backend and tile choice -> aligned check / ``_gemm_padded_shape``
                  -> ``.contiguous()`` -> ``dispatch._to_xla`` -> kernel call
                  (``_gemm_kernel_ragged`` if the shape is unaligned, else the
                  default kernel) -> sync, which does the neuronx-cc compile and
                  the first execute -> ``.to(orig_device)`` -> slice ->
                  ``alpha *``. On the XLA path, torch_xla's CompileTime and
                  ExecuteTime metrics are also printed so compile and execute
                  can be separated. It ends with a timed warm full
                  ``trnblas.gemm``.
  full            One cold ``trnblas.gemm(1.0, A, B)`` as a black box (the baseline
                  that the per-phase sum should match).
  direct-jit      The kernel's ``@nki.jit`` entry point called on XLA tensors,
                  with no trnblas Python around it. Inputs are already moved
                  before the timer starts.
  direct-op       The ragged kernel wrapped in ``torch_neuronx.nki_op`` and
                  called on native ``neuron`` tensors (TorchNeuron path, no torch_xla).
  hop             No GEMM. Moves (N, N) tensors through the same ``_to_xla`` hop,
                  runs a trivial op on XLA, and brings the result back. If this
                  alone shows the odd/aligned gap, the cost is in the device hop
                  and graph compile around the kernel, not in the kernel.
  mlir            CPU-only, static. Kernel trace and lowering via
                  ``nki.compiler.frontend.source_to_mlir`` for (N, N, N); prints
                  the op count and IR size. Run it with N=4095 and N=4096.

Every device mode also prints a ``PHASE nki_trace_lower_cpu`` line, which is
the same trace and lowering to MLIR on CPU, so it can be compared with the
NEFF compile.

Recommended device sequence (fresh process and fresh caches for each line):

    for n in 4095 4096 2047 2048; do
      for m in full gemm direct-jit direct-op hop; do
        rm -rf "$NEURON_CC_CACHE" /var/tmp/neuron-compile-cache ~/.cache/neuron* 2>/dev/null
        python scripts/probe_first_call.py $n --mode $m
      done
    done
"""

# ruff: noqa: E402  (env vars must be set before NKI / trnblas import)
from __future__ import annotations

import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("N", type=int)
ap.add_argument(
    "--mode", default="gemm", choices=["gemm", "full", "direct-jit", "direct-op", "hop", "mlir"]
)
ap.add_argument("--cpu", action="store_true", help="CPU dry run: pytorch backend / stand-ins")
ap.add_argument("--input-device", default="neuron", choices=["neuron", "cpu"])
ap.add_argument("--alpha", type=float, default=1.0)
ARGS = ap.parse_args()
N = ARGS.N
CPU = ARGS.cpu or ARGS.mode == "mlir"

os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")
if not CPU:
    os.environ.setdefault("TRNBLAS_REQUIRE_NKI", "1")

_T0 = time.perf_counter()


def phase(name, fn=None, *, sync=True):
    """Run fn, fence with sync, print one PHASE line, and return fn's result."""
    t = time.perf_counter()
    try:
        out = fn() if fn is not None else None
        if sync:
            _sync(out)
    except Exception as e:  # keep probing the remaining phases
        lines = str(e).strip().splitlines()
        print(
            f"PHASE {name} ERROR {type(e).__name__}: {lines[0][:240] if lines else ''}", flush=True
        )
        return _FAILED
    print(f"PHASE {name} {time.perf_counter() - t:.4f}", flush=True)
    return out


class _Failed:
    pass


_FAILED = _Failed()


def _sync(obj=None):
    if CPU:
        return
    try:
        import torch_neuronx

        torch_neuronx.synchronize()
    except Exception:
        pass
    if _HAS_XLA:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        xm.wait_device_ops()


import torch

phase(
    "import_torch_neuronx", (lambda: __import__("torch_neuronx")) if not CPU else None, sync=False
)
try:
    import torch_xla  # noqa: F401

    _HAS_XLA = not CPU
except Exception:
    _HAS_XLA = False
print(f"INFO N={N} mode={ARGS.mode} cpu={CPU} torch_xla={_HAS_XLA}", flush=True)

trnblas = phase("import_trnblas", lambda: __import__("trnblas"), sync=False)
from trnblas.nki import dispatch as D

print(
    f"INFO HAS_NKI={D.HAS_NKI} autotune={D.get_autotune()} ragged={D._RAGGED_ENABLED} "
    f"require_nki={D._REQUIRE_NKI}",
    flush=True,
)

DEV = "cpu" if CPU else ARGS.input_device


def _xla_metrics(tag):
    """Print torch_xla CompileTime / ExecuteTime totals (seconds) and counts."""
    if not _HAS_XLA:
        return
    try:
        import torch_xla.debug.metrics as met

        for m in ("CompileTime", "ExecuteTime"):
            d = met.metric_data(m)
            if d:
                print(f"PHASE {tag}_{m} {d[1] / 1e9:.4f}", flush=True)
                print(f"INFO {tag}_{m}_count {d[0]}", flush=True)
    except Exception as e:
        print(f"INFO xla metrics unavailable: {e}", flush=True)


def _cpu_kernel(kernel):
    """CPU stand-in: nki.simulate if NKI is importable, else torch.matmul."""
    if D.HAS_NKI:
        import nki
        import numpy as np

        return lambda a, b: torch.from_numpy(np.asarray(nki.simulate(kernel)(a.numpy(), b.numpy())))
    return lambda a, b: a @ b


def _hop(*ts):
    """The dispatch's device hop (native or XLA, `_to_device`); identity on CPU."""
    if CPU:
        return list(ts), ts[0].device
    return D._to_device(*ts)


def nki_trace_lower_cpu(M, K, N_, kernel):
    import nki.compiler.frontend as fe
    import numpy as np

    a = np.zeros((M, K), np.float32)
    b = np.zeros((K, N_), np.float32)
    out = {}
    for frontend in fe._FRONTENDS:
        t = time.perf_counter()
        mod = fe.source_to_mlir(kernel, a, b, target="trn2", frontend=frontend)
        text = str(mod)
        dt = time.perf_counter() - t
        nops = sum(
            1 for ln in text.splitlines() if "=" in ln or ln.strip().startswith(("nki", '"'))
        )
        name = getattr(frontend, "value", str(frontend))
        print(f"PHASE nki_trace_lower_cpu_{name} {dt:.4f}", flush=True)
        print(
            f"INFO mlir_{name} lines={text.count(chr(10))} approx_ops={nops} bytes={len(text)}",
            flush=True,
        )
        out[name] = text
    return out


def _kernel_for(M, K, N_):
    tiles = (D._TILE_M, D._TILE_K, D._TILE_N)
    aligned = D._gemm_padded_shape(M, K, N_, *tiles) == (M, K, N_)
    if not aligned and D._RAGGED_ENABLED:
        return D._gemm_kernel_ragged, "ragged", aligned
    return D._get_gemm_kernel(*tiles), "default", aligned


# ---------------------------------------------------------------------------
g = torch.Generator().manual_seed(N)
A_h = phase("host_randn", lambda: torch.randn(N, N, generator=g), sync=False)
B_h = torch.randn(N, N, generator=g)
A = phase("h2d_inputs", lambda: A_h.to(DEV))
B = B_h.to(DEV)
_sync()
ref = A_h.double() @ B_h.double()


def check(C, tag):
    try:
        Cc = C.detach().to("cpu").double()
        rel = ((Cc - ref).norm() / ref.norm()).item()
        print(f"INFO {tag} shape={tuple(C.shape)} rel_err={rel:.3e}", flush=True)
    except Exception as e:
        print(f"INFO {tag} check failed: {e}", flush=True)


if ARGS.mode == "mlir":
    if not D.HAS_NKI:
        sys.exit("mlir mode needs nki installed (linux x86_64)")
    kern, kname, aligned = _kernel_for(N, N, N)
    print(f"INFO kernel={kname} aligned={aligned}", flush=True)
    nki_trace_lower_cpu(N, N, N, kern)
    sys.exit(0)

if ARGS.mode == "full":
    trnblas.set_backend("pytorch" if CPU else "nki")
    C = phase("full_trnblas_gemm_cold", lambda: trnblas.gemm(ARGS.alpha, A, B))
    check(C, "full")
    C = phase("full_trnblas_gemm_warm", lambda: trnblas.gemm(ARGS.alpha, A, B))
    _xla_metrics("xla")
    sys.exit(0)

if ARGS.mode == "hop":

    def _roundtrip():
        (a, b), orig = _hop(A, B)
        return (a * 2 + b).to(orig)

    C = phase("hop_roundtrip_cold", _roundtrip)
    _xla_metrics("xla_after_cold")
    phase("hop_roundtrip_warm", _roundtrip)
    _xla_metrics("xla_after_warm")
    (a, b), orig = _hop(A, B)
    phase("hop_in_only", lambda: _hop(A, B))
    phase("hop_out_only", lambda: a.to(orig))
    sys.exit(0)

if ARGS.mode == "direct-op":
    if CPU:
        kern = _cpu_kernel(D._gemm_kernel_ragged) if D.HAS_NKI else (lambda a, b: a @ b)
        call = kern
    else:
        from torch_neuronx import nki_op

        @nki_op("trnblas_probe::gemm_ragged", mutates_args={})
        def _gemm_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return D._gemm_kernel_ragged(a, b)

        call = _gemm_op
    if D.HAS_NKI:
        phase(
            "nki_trace_lower_cpu",
            lambda: nki_trace_lower_cpu(N, N, N, D._gemm_kernel_ragged),
            sync=False,
        )
    C = phase("direct_op_first_call", lambda: call(A, B))
    check(C, "direct_op")
    phase("direct_op_warm_call", lambda: call(A, B))
    sys.exit(0)

if ARGS.mode == "direct-jit":
    kern, kname, aligned = _kernel_for(N, N, N)
    print(f"INFO kernel={kname} aligned={aligned}", flush=True)
    (a, b), orig = _hop(A.contiguous(), B.contiguous())
    _sync()
    if D.HAS_NKI:
        phase("nki_trace_lower_cpu", lambda: nki_trace_lower_cpu(N, N, N, kern), sync=False)
    call = _cpu_kernel(kern) if CPU else kern
    out = phase("direct_jit_call_nosync", lambda: call(a, b), sync=False)
    phase("direct_jit_sync_compile_execute", None)
    _xla_metrics("xla_first")
    if out is not _FAILED:
        check(out, "direct_jit")
    phase("direct_jit_warm", lambda: call(a, b))
    _xla_metrics("xla_after_warm")
    sys.exit(0)

# ---- mode gemm: replay trnblas.gemm -> nki_gemm -> _nki_gemm_impl in order ----
trnblas.set_backend("pytorch" if CPU else "nki")
alpha = ARGS.alpha


def _select():
    use_nki = D._use_nki()
    if D._AUTOTUNE_ENABLED and not D._use_simulator():
        print(
            "INFO autotune enabled: the sweep would run here; probe uses default tiles", flush=True
        )
    return use_nki, (D._TILE_M, D._TILE_K, D._TILE_N)


use_nki, tiles = phase("dispatch_select", _select, sync=False)
print(f"INFO use_nki={use_nki} tiles={tiles}", flush=True)
kern, kname, aligned = _kernel_for(N, N, N)
print(f"INFO kernel={kname} aligned={aligned}", flush=True)

# level3.gemm: op(A), op(B) are views (no trans here); _nki_gemm_impl routes on alignment.
if kname == "ragged":
    feeds = phase("input_prep_contiguous", lambda: (A.contiguous(), B.contiguous()))
else:
    feeds = phase(
        "input_prep_pad_or_contiguous",
        lambda: D._pad_gemm_operands(A, B, *tiles)[:2],
    )
A_f, B_f = feeds
moved = phase("to_xla_hop_in", lambda: _hop(A_f, B_f))
(a, b), orig = moved
if D.HAS_NKI:
    phase(
        "nki_trace_lower_cpu",
        lambda: nki_trace_lower_cpu(*A_f.shape, B_f.shape[1], kern),
        sync=False,
    )
call = _cpu_kernel(kern) if CPU else kern
out = phase("kernel_call_nosync", lambda: call(a, b), sync=False)
phase("sync_neff_compile_and_first_execute", None)
_xla_metrics("xla_first")
res = phase("to_orig_device_hop_out", lambda: out.to(orig))
res = phase("slice_result", lambda: res if aligned or kname == "ragged" else res[:N, :N])
C = phase("alpha_scale", lambda: alpha * res)
check(C, "gemm_replay")
C2 = phase("warm_full_trnblas_gemm", lambda: trnblas.gemm(alpha, A, B))
_xla_metrics("xla_total")
print(f"PHASE total_process {time.perf_counter() - _T0:.4f}", flush=True)
