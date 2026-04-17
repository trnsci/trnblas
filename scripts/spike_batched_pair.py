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

  A. 3D batch indexing (spike_A_3d_index / spike_A_3d_transpose):
     Can NKI index a 3D tensor as B[i, a:a+T, k:k+T] where i is an
     affine_range loop variable?  The existing flat trick
     (T_flat[i * stride + offset : ...]) works for 2D; the 3D case is
     new because the compiler must emit an affine base-address expression
     over a batch dim stride, not just within a row.
     Expected: compiles if NKI's affine IR supports batch-dim strides.
     Fail mode: "illegal affine expression" or incorrect result.

  B. Nested pair loops with safe SBUF accumulation (spike_B_pair_loop):
     Full NOCC×NOCC pair loop inside a single @nki.jit.  Each pair (i,j)
     does 2 tile GEMMs + energy VE; partial sum (T,1) stored to per-j SBUF
     slot acc_j[0:T, j:j+1].  After j-loop, acc_j flushed to
     out[0:T, i*N:(i+1)*N].  Host sums the (T, N²) output.
     Tests: (1) 3D B[i,...] and B[j,...] inside nested pair loops;
     (2) acc_j[0:T, j:j+1] write with single loop variable j; (3) HBM
     store with offset i*N (affine in i).
     Fail mode: compile error on any of the three, or wrong energy.

  C. Running nl.add accumulation (spike_C_running_add):
     Attempt `e_acc = nl.add(e_acc, pair_energy)` inside an affine_range
     loop — the "natural" accumulation pattern.  Expected to FAIL with
     "tensor_reduce cannot reduce partition dimension" or similar.
     The same failure was seen in _mp2_energy_kernel M1 before the
     per-slot SBUF write fix (#35).  Confirms Spike B is the right pattern.

## NKI partition-axis reduction rule (learned from first spike run)

NKI only supports free-dim (axis=1) reductions.  `nl.sum(x, axis=0)` —
reducing the partition dim — is rejected by the compiler at trace time.
The workaround: output a 2D `(T, N)` partial array and let the host call
`.sum()`.  This is the same pattern used in _mp2_energy_kernel (which
returns `(P_TILE, IC*NOCC)` and the caller does `.sum()`).

## NOCC / NVIR / NAUX sizing

Spike uses small shapes to keep compile time tractable and isolate compiler
behaviour from SBUF capacity issues:
  NOCC = 4  →  16 pairs; loop graph stays compact
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
#
# NKI rule: no partition-axis (axis=0) reductions.
# Fix: output (T, N) partials and let host sum axis=0.
# ---------------------------------------------------------------------------
if HAS_NKI:

    @nki.jit
    def spike_A_3d_index(B):
        """Test: nl.load(B[i, 0:T, 0:K]) with i in affine_range.

        B: (NOCC, NVIR, NAUX) — 3D tensor.
        Returns: (T, N) — column i holds row_sum of B[i] along NAUX axis.
          row_sum[a] = sum_k B[i, a, k]
        Host: result[:, i].sum() should equal B[i].sum().

        Only free-dim (axis=1) reductions used — no partition-axis reduce.
        """
        N = 4  # NOCC literal — must be literal for affine_range
        T = 128  # NVIR literal
        K = 128  # NAUX literal

        # (T, N): one column per i; host sums partition axis
        out = nl.ndarray((T, N), dtype=nl.float32, buffer=nl.shared_hbm)

        for i in nl.affine_range(N):
            # 3D batch index: i is an affine_range variable.
            # Partition dim = T (≤ 128), free dim = K (≤ 512).
            tile = nl.load(B[i, 0:T, 0:K])  # (T, K) in SBUF
            row_sum = nl.sum(tile, axis=1, keepdims=True)  # (T, 1) — free-dim only
            nl.store(out[0:T, i : i + 1], value=row_sum)

        return out

    @nki.jit
    def spike_A_3d_transpose(B):
        """Test: nl.load_transpose2d(B[i, 0:T, 0:K]) with affine_range i.

        load_transpose2d on a 3D slice is the exact call needed for the
        production GEMM: stationary = load_transpose2d(B[i, a:a+T, k:k+K]).
        This spike uses NOCC=4, NVIR=TILE, NAUX=TILE_K so the trivial
        (a=0, k=0) slice is the only tile — no tile loops.

        Returns: (K, N) — column i holds nl.sum(b_t, axis=1) for B[i].
          b_t[k, a] = B[i, a, k] (load_transpose2d swaps dims).
          row_sum[k] = sum_a B[i, a, k].
        Host: result[:, i].sum() should equal B[i].sum() (same total).

        Only free-dim (axis=1) reductions used.
        """
        N = 4
        T = 128
        K = 128

        out = nl.ndarray((K, N), dtype=nl.float32, buffer=nl.shared_hbm)

        for i in nl.affine_range(N):
            # load_transpose2d: partition ≤ 128 (K), free ≤ 512 (T).
            # Result: (K, T) in SBUF.
            b_t = nl.load_transpose2d(B[i, 0:T, 0:K])  # (K, T)
            row_sum = nl.sum(b_t, axis=1, keepdims=True)  # (K, 1) — free-dim
            nl.store(out[0:K, i : i + 1], value=row_sum)

        return out


# ---------------------------------------------------------------------------
# Spike B: nested pair loops with safe SBUF accumulation
#
# Pattern from _mp2_energy_kernel:
#   - Inner j-loop writes (T,1) partial to acc_j[0:T, j:j+1]  (j-only index)
#   - Outer i-loop stores acc_j to out[0:T, i*N:(i+1)*N]       (i-only offset)
#   - Host: result.sum() gives total energy
# ---------------------------------------------------------------------------
if HAS_NKI:

    @nki.jit
    def spike_B_pair_loop(B, eps_occ_row, eps_vir_col, eps_vir_row):
        """Test: NOCC×NOCC pair loop + acc_j SBUF pattern.

        B: (NOCC, NVIR, NAUX).  eps_occ_row: (1, NOCC).
        eps_vir_col: (NVIR, 1).  eps_vir_row: (1, NVIR).
        Returns: (T, N²) — one (T,1) column per pair; host sums all dims.

        Inner j-loop uses acc_j[0:T, j:j+1] (j-only SBUF index) — the
        same pattern validated by _mp2_energy_kernel.  The 3D B[i,...] and
        B[j,...] indexing is the new primitive under test here.
        """
        N = 4  # NOCC
        T = 128  # NVIR
        K = 128  # NAUX

        # Output: (T, N²) — partition=T rows, N² columns.  Host sums both.
        out = nl.ndarray((T, N * N), dtype=nl.float32, buffer=nl.shared_hbm)

        for i in nl.affine_range(N):
            eps_i = nl.load(eps_occ_row[0:1, i : i + 1])  # (1, 1)

            # Per-j SBUF accumulator: (T, N) — j-partials for this i-row.
            acc_j = nl.zeros((T, N), dtype=nl.float32, buffer=nl.sbuf)

            for j in nl.affine_range(N):
                eps_j = nl.load(eps_occ_row[0:1, j : j + 1])  # (1, 1)
                eps_occ_sum = nl.add(eps_i, eps_j)  # (1, 1)

                # GEMM: T[a,b] = B[i] @ B[j].T
                # b_i_t = B[i].T → (K, T); b_j_t = B[j].T → (K, T)
                # nc_matmul: stationary=(K,T), moving=(K,T) → psum (T,T)
                psum = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                b_i_t = nl.load_transpose2d(B[i, 0:T, 0:K])  # (K, T)
                b_j_t = nl.load_transpose2d(B[j, 0:T, 0:K])  # (K, T)
                nisa.nc_matmul(dst=psum, stationary=b_i_t, moving=b_j_t, accumulate=False)
                t_sbuf = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum, dst=t_sbuf)

                # GEMM: T_T[a,b] = T[b,a] = B[j] @ B[i].T
                psum2 = nl.zeros((T, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=psum2, stationary=b_j_t, moving=b_i_t, accumulate=False)
                t_T_sbuf = nl.ndarray((T, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(src=psum2, dst=t_T_sbuf)

                # denom[a,b] = eps_occ_sum - eps_vir[a] - eps_vir[b]
                evc = nl.load(eps_vir_col[0:T, 0:1])  # (T, 1)
                evr = nl.load(eps_vir_row[0:1, 0:T])  # (1, T)
                eo_bc = nl.broadcast_to(eps_occ_sum, (T, T))
                evc_bc = nl.broadcast_to(evc, (T, T))
                evr_bc = nl.broadcast_to(evr, (T, T))
                denom = nl.subtract(nl.subtract(eo_bc, evc_bc), evr_bc)

                # energy_tile[a,b] = T[a,b] * (2*T[a,b] - T[b,a]) / denom[a,b]
                two_T_minus_T_T = nl.subtract(nl.multiply(t_sbuf, 2.0), t_T_sbuf)
                energy_tile = nl.multiply(
                    nl.multiply(t_sbuf, two_T_minus_T_T),
                    nl.reciprocal(denom),
                )

                # Reduce free dim only: (T, T) → (T, 1).
                # No partition-axis reduce — host handles that.
                row_sums = nl.sum(energy_tile, axis=1, keepdims=True)  # (T, 1)

                # Store to per-j SBUF slot (j-only index — validated pattern).
                acc_j[0:T, j : j + 1] = row_sums

            # Flush all NOCC j-partials for this i-row to HBM.
            # Offset i*N is affine in the outer loop variable — valid.
            nl.store(out[0:T, i * N : (i + 1) * N], value=acc_j)

        return out


# ---------------------------------------------------------------------------
# Spike C: running nl.add accumulation (expected to fail)
# ---------------------------------------------------------------------------
if HAS_NKI:

    @nki.jit
    def spike_C_running_add(B, e_out):
        """Test: e_acc = nl.add(e_acc, free_sum) inside affine_range.

        EXPECTED TO FAIL.  Running accumulation creates a loop-carried
        dependency that NKI's affine IR cannot express.  The same failure
        was seen in _mp2_energy_kernel pre-#35 when `acc_rows += strip`
        was attempted.

        Spike C now uses only free-dim reductions (no axis=0 sum) so the
        failure point is specifically the loop-carried nl.add — not the
        partition-axis reduce from the first spike run.
        """
        N = 4
        T = 128
        K = 128

        out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        # Running accumulator (T, 1) — the loop-carried dep.
        e_acc = nl.zeros((T, 1), dtype=nl.float32, buffer=nl.sbuf)

        for i in nl.affine_range(N):
            for _j in nl.affine_range(N):
                tile = nl.load(B[i, 0:T, 0:K])  # (T, K)
                free_sum = nl.sum(tile, axis=1, keepdims=True)  # (T, 1) — free dim
                # Loop-carried add — the suspected failure point.
                e_acc = nl.add(e_acc, free_sum)

        nl.store(out[0:T, 0:1], value=e_acc)
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

    # Reference: per-i total element sum
    ref_sums = [float(B_cpu[i].sum()) for i in range(NOCC)]

    # Part 1: nl.load 3D — output (T, N), host sums column per i
    print("  Part 1: nl.load(B[i, 0:T, 0:K])  with affine_range i", end=" ... ", flush=True)
    try:
        (B_xla,), cpu = _to_xla(B_cpu)
        result = spike_A_3d_index(B_xla)
        result_cpu = result.to(cpu)
        # result_cpu[:, i].sum() should match B[i].sum()
        got = [float(result_cpu[:, i].sum()) for i in range(NOCC)]
        match = all(abs(g - r) / (abs(r) + 1e-8) < 1e-4 for g, r in zip(got, ref_sums, strict=True))
        print("PASS" if match else f"FAIL (got {got} vs ref {ref_sums})")
    except Exception as exc:
        print(f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})")

    # Part 2: nl.load_transpose2d 3D — output (K, N), host sums column per i
    print(
        "  Part 2: nl.load_transpose2d(B[i, 0:T, 0:K])  with affine_range i",
        end=" ... ",
        flush=True,
    )
    try:
        (B_xla,), cpu = _to_xla(B_cpu)
        result = spike_A_3d_transpose(B_xla)
        result_cpu = result.to(cpu)
        # result_cpu[:, i].sum() = sum_k sum_a B[i,a,k] = B[i].sum()
        got = [float(result_cpu[:, i].sum()) for i in range(NOCC)]
        match = all(abs(g - r) / (abs(r) + 1e-8) < 1e-4 for g, r in zip(got, ref_sums, strict=True))
        print("PASS" if match else f"FAIL (got {got} vs ref {ref_sums})")
    except Exception as exc:
        print(f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})")


def run_spike_B():
    print("\n=== Spike B: nested pair loops + acc_j SBUF accumulation ===")
    torch.manual_seed(42)
    B_cpu = torch.randn(NOCC, NVIR, NAUX, dtype=torch.float32)
    eps_occ = torch.linspace(-0.9, -0.1, NOCC)
    eps_vir = torch.linspace(0.1, 1.0, NVIR)

    ref = _ref_total_energy(B_cpu, eps_occ, eps_vir)
    print(f"  Reference energy: {ref:.6e}")

    eps_occ_row = eps_occ.reshape(1, NOCC).contiguous()
    eps_vir_col = eps_vir.reshape(NVIR, 1).contiguous()
    eps_vir_row = eps_vir.reshape(1, NVIR).contiguous()

    print("  Compile + warm run", end=" ... ", flush=True)
    t0 = time.perf_counter()
    try:
        (B_xla, eo_xla, evc_xla, evr_xla), cpu = _to_xla(
            B_cpu, eps_occ_row, eps_vir_col, eps_vir_row
        )
        result_xla = spike_B_pair_loop(B_xla, eo_xla, evc_xla, evr_xla)
        # result is (T, N²); sum all dims for total energy
        result = float(result_xla.to(cpu).sum())
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
        (B_xla2, eo_xla2, evc_xla2, evr_xla2), cpu2 = _to_xla(
            B_cpu, eps_occ_row, eps_vir_col, eps_vir_row
        )
        result2_xla = spike_B_pair_loop(B_xla2, eo_xla2, evc_xla2, evr_xla2)
        result2 = float(result2_xla.to(cpu2).sum())
        elapsed2 = time.perf_counter() - t1
        print(f"E={result2:.6e}  ({elapsed2:.3f}s)")
    except Exception as exc:
        print(f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})")


def run_spike_C():
    print("\n=== Spike C: running nl.add accumulation (expected FAIL) ===")
    torch.manual_seed(42)
    B_cpu = torch.randn(NOCC, NVIR, NAUX, dtype=torch.float32)

    print("  nl.add(e_acc, free_sum) inside affine_range", end=" ... ", flush=True)
    try:
        e_out_cpu = torch.zeros(NVIR, 1, dtype=torch.float32)
        (B_xla, out_xla), cpu = _to_xla(B_cpu, e_out_cpu)
        result = spike_C_running_add(B_xla, out_xla)
        result_cpu = float(result.to(cpu).sum())
        print(f"PASS (unexpected — loop-carried add supported; result={result_cpu:.3e})")
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
