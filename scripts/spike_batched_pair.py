"""Batched-pair kernel pre-implementation spike (#43).

Three kernels that probe the NKI primitives required for the batched-pair
fused GEMM+energy design before committing to a full implementation.

## The design under investigation

Current path (post-v0.5.1): one NEFF per (i,j) pair, called in a Python loop.
  dispatcher → _fused_gemm_energy_kernel  ×  nocc²  dispatches
  Each dispatch: ~100ms Neuron XLA overhead (measured on trn1, April 2026).
  For nocc=16: 256 pairs × 100ms = 25.6s warm — energy dominates total time.

Phase 4 target (#43) — one NEFF for ALL pairs:
  dispatcher → _batched_pair_kernel  ×  1  dispatch
  Inner loops over i, j inside the kernel body.
  O(1) dispatch overhead regardless of nocc.

## Spike questions

  A. 3D batch indexing (spike_A_3d_index):
     Can NKI index a 3D tensor as B[i, a:a+T, k:k+T] where i is an
     affine_range loop variable?  The existing flat trick
     (T_flat[i * stride + offset : ...]) works for 2D; the 3D case is
     new because the compiler must emit an affine base-address expression
     over a batch dim stride, not just within a row.
     Expected: compiles if NKI's affine IR supports batch-dim strides.
     Fail mode: "illegal affine expression" or incorrect result.

  B. Nested pair loops with safe SBUF accumulation (spike_B_pair_loop):
     Full NOCC×NOCC pair loop inside a single @nki.jit.  Each pair (i,j)
     does one tile GEMM + energy VE; partials stored to a per-pair SBUF
     slot `e_partial[0, i*NOCC+j : i*NOCC+j+1]`.  Post-loop single HBM
     store; host sums the (1, NOCC²) vector.
     Tests: (1) do nested affine_range loops at nocc=4 compile without
     unrolling explosion? (2) Is `e_partial[0, k:k+1] = value` valid
     when k is a function of two loop variables?
     Fail mode: compile error on the computed-offset SBUF store, or
     catastrophic compile time at nocc=4.

  C. Running scalar accumulation (spike_C_running_add):
     Attempt `e_acc = nl.add(e_acc, pair_energy)` inside an affine_range
     loop — the "natural" accumulation pattern.  Expected to FAIL with NKI's
     "Unexpected output dependencies" error (the same error that blocked
     in-place `acc_rows +=` in _mp2_energy_kernel #35 fix).  Confirms
     that the safe pattern (Spike B) is the right one for production.

## NOCC / NVIR / NAUX sizing

Spike uses small shapes to keep compile time tractable and isolate compiler
behaviour from SBUF capacity issues:
  NOCC = 4  →  16 pairs; unrolled graph still tiny
  NVIR = 128, NAUX = 128  →  single tile; no tile loops needed
For production, NVIR ~ 512, NAUX ~ 384 with tile loops.
"""

import sys
import time

import numpy as np
import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False
    print("NKI not available — run on Trainium hardware or with NEURON_VENV.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Spike shapes
# ---------------------------------------------------------------------------
NOCC = 4
NVIR = 128  # exact multiple of TILE (128)
NAUX = 128  # exact multiple of TILE_K (128)
TILE = 128  # free/partition limit for load_transpose2d
TILE_K = 128


# ---------------------------------------------------------------------------
# Torch reference: single-pair energy
# ---------------------------------------------------------------------------
def _ref_pair_energy(b_i, b_j, eps_occ_i, eps_occ_j, eps_vir):
    """T*(2T - T.T)/denom for one (i,j) pair.  b_i, b_j: (NVIR, NAUX)."""
    T = b_i @ b_j.T  # (NVIR, NVIR)
    denom = eps_occ_i + eps_occ_j - eps_vir[:, None] - eps_vir[None, :]
    return float((T * (2.0 * T - T.T) / denom).sum())


def _ref_total_energy(B, eps_occ, eps_vir):
    """Sum over all (i,j) pairs."""
    e = 0.0
    for i in range(NOCC):
        for j in range(NOCC):
            e += _ref_pair_energy(B[i], B[j], eps_occ[i], eps_occ[j], eps_vir)
    return e


# ---------------------------------------------------------------------------
# Spike A: 3D batch indexing
# ---------------------------------------------------------------------------
if HAS_NKI:

    @nki.jit
    def spike_A_3d_index(B, e_out):
        """Test: load B[i, 0:TILE, 0:TILE_K] with i in affine_range.

        B: (NOCC, NVIR, NAUX) — 3D tensor.
        e_out: (1, NOCC) — one output slot per i-row.

        Each i-iteration loads B[i]'s single tile via nl.load (not
        load_transpose2d first, to isolate the 3D-index question from the
        transpose constraint), computes a trivial free-axis sum, and stores
        the scalar to e_out[0, i:i+1].  If the 3D index compiles and
        produces the correct row sum, spike A passes.
        """
        N = 4  # NOCC literal — must be literal for affine_range
        T = 128  # NVIR literal
        K = 128  # NAUX literal

        out = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.shared_hbm)

        for i in nl.affine_range(N):
            # 3D batch index: i is an affine_range variable.
            # Partition dim = K (≤ 128), free dim = T (≤ 512).
            tile = nl.load(B[i, 0:T, 0:K])  # shape (T, K) in SBUF
            row_sum = nl.sum(tile, axis=1, keepdims=True)  # (T, 1)
            col_sum = nl.sum(row_sum, axis=0, keepdims=True)  # (1, 1)
            out[0:1, i : i + 1] = col_sum

        return out

    @nki.jit
    def spike_A_3d_transpose(B, e_out):
        """Test: nl.load_transpose2d(B[i, 0:TILE, 0:TILE_K]) with affine_range i.

        load_transpose2d on a 3D slice is the exact call needed for the
        production GEMM: stationary = load_transpose2d(B[i, a:a+T, k:k+K]).
        This spike uses NOCC=4, NVIR=TILE, NAUX=TILE_K so the trivial
        (a=0, k=0) slice is the only tile — no tile loops.

        Result: the (TILE_K, TILE) transposed tile is loaded; we store its
        element-wise sum to out[0, i:i+1].  Correctness: must match sum of
        B[i] computed on CPU.
        """
        N = 4
        T = 128
        K = 128

        out = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.shared_hbm)

        for i in nl.affine_range(N):
            # load_transpose2d expects partition ≤ 128 (the K dim here).
            # Result shape in SBUF: (K, T) with partition=K.
            b_t = nl.load_transpose2d(B[i, 0:T, 0:K])  # (K, T) in SBUF
            row_sum = nl.sum(b_t, axis=1, keepdims=True)  # (K, 1)
            col_sum = nl.sum(row_sum, axis=0, keepdims=True)  # (1, 1)
            out[0:1, i : i + 1] = col_sum

        return out


# ---------------------------------------------------------------------------
# Spike B: nested pair loops with safe SBUF accumulation
# ---------------------------------------------------------------------------
if HAS_NKI:

    @nki.jit
    def spike_B_pair_loop(B, eps_occ_row, eps_vir_col, eps_vir_row, e_out):
        """Test: NOCC×NOCC pair loop + per-pair SBUF slot accumulation.

        B: (NOCC, NVIR, NAUX)  — all pairs share this tensor.
        eps_occ_row: (1, NOCC) — occupied energies in a row vector.
        eps_vir_col: (NVIR, 1), eps_vir_row: (1, NVIR) — virtual energies.
        e_out: (1, 1) — scalar energy; host slices [0, 0].

        Pattern:
          e_pairs = nl.zeros((1, NOCC*NOCC), ..., buffer=nl.sbuf)
          for i in nl.affine_range(NOCC):
              for j in nl.affine_range(NOCC):
                  ... compute pair energy tile (1,1) ...
                  e_pairs[0, i*NOCC+j : i*NOCC+j+1] = e_pair
          sum = nl.sum(e_pairs, axis=1, keepdims=True)
          nl.store(e_out[0:1, 0:1], value=sum)

        All shape params are literals (NKI AST constraint).
        """
        N = 4  # NOCC
        T = 128  # NVIR
        K = 128  # NAUX

        out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)

        # Per-pair SBUF accumulator: (1, N²) — each slot holds one pair's energy.
        e_pairs = nl.zeros((1, N * N), dtype=nl.float32, buffer=nl.sbuf)

        for i in nl.affine_range(N):
            eps_i = nl.load(eps_occ_row[0:1, i : i + 1])  # (1, 1)

            for j in nl.affine_range(N):
                eps_j = nl.load(eps_occ_row[0:1, j : j + 1])  # (1, 1)
                # eps_occ_sum: scalar tile (1,1).
                eps_occ_sum = nl.add(eps_i, eps_j)

                # GEMM: T[a,b] = B[i] @ B[j].T
                # Stationary = B[i].T loaded via load_transpose2d → (K, T).
                # Moving     = B[j].T loaded via load_transpose2d → (K, T).
                # nc_matmul(stationary=(K,T), moving=(K,T)) → psum (T,T).
                psum = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                # NOCC=4, NVIR=NAUX=128 → single k-tile, no k-loop needed.
                b_i_t = nl.load_transpose2d(B[i, 0:T, 0:K])  # (K, T)
                b_j_t = nl.load_transpose2d(B[j, 0:T, 0:K])  # (K, T)
                nisa.nc_matmul(dst=psum, stationary=b_i_t, moving=b_j_t, accumulate=False)

                # PSUM → SBUF.
                t_sbuf = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=t_sbuf)

                # Energy expression:  T * (2T - T.T) / denom
                # T.T is the transpose of t_sbuf — we approximate by loading
                # B[j] @ B[i].T, which equals T.T exactly.
                psum2 = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=psum2, stationary=b_j_t, moving=b_i_t, accumulate=False)
                t_T_sbuf = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum2, dst=t_T_sbuf)

                # denom[a,b] = (eps_occ_i + eps_occ_j) - eps_vir[a] - eps_vir[b]
                evc = nl.load(eps_vir_col[0:T, 0:1])  # (T, 1)
                evr = nl.load(eps_vir_row[0:1, 0:T])  # (1, T)
                eo_bc = nl.broadcast_to(eps_occ_sum, (T, T))
                evc_bc = nl.broadcast_to(evc, (T, T))
                evr_bc = nl.broadcast_to(evr, (T, T))
                denom = nl.subtract(nl.subtract(eo_bc, evc_bc), evr_bc)

                # Energy tile: T * (2T - T_T) / denom  →  (T, T).
                two_T_minus_T_T = nl.subtract(nl.multiply(t_sbuf, 2.0), t_T_sbuf)
                energy_tile = nl.multiply(
                    nl.multiply(t_sbuf, two_T_minus_T_T),
                    nl.reciprocal(denom),
                )

                # Reduce (T, T) → scalar (1, 1) in two steps.
                row_sums = nl.sum(energy_tile, axis=1, keepdims=True)  # (T, 1)
                e_pair = nl.sum(row_sums, axis=0, keepdims=True)  # (1, 1)

                # Store to per-pair SBUF slot.
                # k = i * N + j  — affine expression of two loop variables.
                e_pairs[0:1, i * N + j : i * N + j + 1] = e_pair

        # Reduce all pairs to scalar and flush to HBM.
        total = nl.sum(e_pairs, axis=1, keepdims=True)  # (1, 1)
        nl.store(out[0:1, 0:1], value=total)
        return out


# ---------------------------------------------------------------------------
# Spike C: running nl.add accumulation (expected to fail)
# ---------------------------------------------------------------------------
if HAS_NKI:

    @nki.jit
    def spike_C_running_add(B, e_out):
        """Test: e_acc = nl.add(e_acc, pair_energy) inside affine_range.

        EXPECTED TO FAIL with 'Unexpected output dependencies' or similar.
        The NKI compiler's affine IR treats a loop body as a template that
        gets unrolled / parallelised; a running accumulator creates a loop-
        carried dependency that is not expressible as an affine recurrence.
        The same failure was observed in _mp2_energy_kernel M1 (pre-#35)
        when we tried `acc_rows += strip_partial` inside nl.affine_range.

        If this passes, it means NKI has added support for loop-carried
        accumulation — update the production kernel design accordingly.
        If it fails, Spike B's per-slot write is the right pattern.
        """
        N = 4
        T = 128
        K = 128

        out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        # Running scalar accumulator — this is the loop-carried dep.
        e_acc = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)

        for i in nl.affine_range(N):
            for _j in nl.affine_range(N):
                # Trivial "energy": just load one element as proxy.
                tile = nl.load(B[i, 0:T, 0:K])
                row_sum = nl.sum(tile, axis=1, keepdims=True)
                scalar = nl.sum(row_sum, axis=0, keepdims=True)
                # Loop-carried add — the suspected failure point.
                e_acc = nl.add(e_acc, scalar)

        nl.store(out[0:1, 0:1], value=e_acc)
        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _to_xla(*tensors):
    """Move tensors to the XLA device; return (xla_tensors, cpu_device)."""
    import torch_xla.core.xla_model as xm  # type: ignore

    dev = xm.xla_device()
    return tuple(t.to(dev) for t in tensors), tensors[0].device


def run_spike_A():
    print("\n=== Spike A: 3D batch indexing ===")
    torch.manual_seed(42)
    B_cpu = torch.randn(NOCC, NVIR, NAUX, dtype=torch.float32)

    # Reference row sums
    ref_sums = B_cpu.sum(dim=(1, 2)).tolist()  # (NOCC,)
    ref_trans = [float(B_cpu[i].sum()) for i in range(NOCC)]

    # Part 1: nl.load 3D
    print("  Part 1: nl.load(B[i, 0:T, 0:K])  with affine_range i", end=" ... ", flush=True)
    try:
        e_out = torch.zeros(1, NOCC, dtype=torch.float32)
        (B_xla, out_xla), cpu = _to_xla(B_cpu, e_out)
        result = spike_A_3d_index(B_xla, out_xla)
        result_cpu = result.to(cpu).squeeze().tolist()
        match = all(abs(r - e) < 1.0 for r, e in zip(result_cpu, ref_sums, strict=True))
        print("PASS" if match else f"FAIL (got {result_cpu} vs ref {ref_sums})")
    except Exception as exc:
        print(f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})")

    # Part 2: nl.load_transpose2d 3D
    print(
        "  Part 2: nl.load_transpose2d(B[i, 0:T, 0:K])  with affine_range i",
        end=" ... ",
        flush=True,
    )
    try:
        e_out = torch.zeros(1, NOCC, dtype=torch.float32)
        (B_xla, out_xla), cpu = _to_xla(B_cpu, e_out)
        result = spike_A_3d_transpose(B_xla, out_xla)
        result_cpu = result.to(cpu).squeeze().tolist()
        # load_transpose2d swaps axes → same elements, same sum
        match = all(abs(r - e) < 1.0 for r, e in zip(result_cpu, ref_trans, strict=True))
        print("PASS" if match else f"FAIL (got {result_cpu} vs ref {ref_trans})")
    except Exception as exc:
        print(f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})")


def run_spike_B():
    print("\n=== Spike B: nested pair loops + per-pair SBUF accumulation ===")
    torch.manual_seed(42)
    B_cpu = torch.randn(NOCC, NVIR, NAUX, dtype=torch.float32)
    eps_occ = torch.linspace(-0.9, -0.1, NOCC)
    eps_vir = torch.linspace(0.1, 1.0, NVIR)

    ref = _ref_total_energy(B_cpu, eps_occ, eps_vir)
    print(f"  Reference energy: {ref:.6e}")

    eps_occ_row = eps_occ.reshape(1, NOCC).contiguous()
    eps_vir_col = eps_vir.reshape(NVIR, 1).contiguous()
    eps_vir_row = eps_vir.reshape(1, NVIR).contiguous()
    e_out_cpu = torch.zeros(1, 1, dtype=torch.float32)

    print("  Compile + warm run", end=" ... ", flush=True)
    t0 = time.perf_counter()
    try:
        (B_xla, eo_xla, evc_xla, evr_xla, out_xla), cpu = _to_xla(
            B_cpu, eps_occ_row, eps_vir_col, eps_vir_row, e_out_cpu
        )
        result_xla = spike_B_pair_loop(B_xla, eo_xla, evc_xla, evr_xla, out_xla)
        result = float(result_xla.to(cpu)[0, 0])
        elapsed = time.perf_counter() - t0
        err = abs(result - ref) / (abs(ref) + 1e-30)
        status = "PASS" if err < 1e-4 else f"FAIL (err={err:.2e})"
        print(f"{status}  E={result:.6e}  ({elapsed:.2f}s incl. compile)")
    except Exception as exc:
        print(f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})")
        return

    print("  Warm run (NEFF cached)", end=" ... ", flush=True)
    t1 = time.perf_counter()
    try:
        (B_xla2, eo_xla2, evc_xla2, evr_xla2, out_xla2), cpu2 = _to_xla(
            B_cpu, eps_occ_row, eps_vir_col, eps_vir_row, e_out_cpu
        )
        result2_xla = spike_B_pair_loop(B_xla2, eo_xla2, evc_xla2, evr_xla2, out_xla2)
        result2 = float(result2_xla.to(cpu2)[0, 0])
        elapsed2 = time.perf_counter() - t1
        print(f"E={result2:.6e}  ({elapsed2:.3f}s)")
    except Exception as exc:
        print(f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})")


def run_spike_C():
    print("\n=== Spike C: running nl.add accumulation (expected FAIL) ===")
    torch.manual_seed(42)
    B_cpu = torch.randn(NOCC, NVIR, NAUX, dtype=torch.float32)
    e_out_cpu = torch.zeros(1, 1, dtype=torch.float32)

    print("  nl.add(e_acc, scalar) inside affine_range", end=" ... ", flush=True)
    try:
        (B_xla, out_xla), cpu = _to_xla(B_cpu, e_out_cpu)
        result = spike_C_running_add(B_xla, out_xla)
        result_cpu = float(result.to(cpu)[0, 0])
        print(f"PASS (unexpected — accumulation supported; result={result_cpu:.3e})")
        print("  → Update production kernel design: running nl.add is valid.")
    except Exception as exc:
        msg = str(exc).splitlines()[0][:120]
        print(f"FAIL as expected: {type(exc).__name__}: {msg}")
        print("  → Confirmed: per-slot SBUF write (Spike B pattern) is required.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batched-pair kernel spike (#43)")
    parser.add_argument(
        "--spike",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which spike to run (default: all)",
    )
    args = parser.parse_args()

    spikes = {"A": run_spike_A, "B": run_spike_B, "C": run_spike_C}

    if args.spike == "all":
        for _name, fn in spikes.items():
            fn()
    else:
        spikes[args.spike]()

    print()
