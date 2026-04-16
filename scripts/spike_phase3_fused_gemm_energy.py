"""Phase 3 pre-implementation spike (#38).

Three kernels that probe the NKI primitives required for the fused
GEMM+energy design before committing to a full implementation.

## The design under investigation

Current two-dispatch path:

    _gemm_kernel:        B_chunk @ B_flat.T → T_flat (HBM write, 6.58 GB)
    _mp2_energy_kernel:  T_flat (HBM read, 6.58 GB) → scalar energy

Phase 3 target — one NEFF, no HBM round-trip for T_flat:

    TE:  nc_matmul(B_i, B_j.T) → PSUM
    VE:  tensor_copy(PSUM → SBUF) + T*(2T-T_T)/denom + sum → HBM scalar

## Spike questions

  A. PSUM → SBUF → VE chain (spike_A_psum_to_ve):
     Can a single @nki.jit run a GEMM tile, tensor_copy the PSUM result
     to SBUF, and then immediately apply VE energy-reduction ops on that
     SBUF tile — without any HBM intermediate for T_flat?
     Expected: yes (TE→VE handoff via SBUF is a documented NKI pattern).
     Fail mode: MLIR verifier rejects the TE→VE use-def chain.

  B. T_T without HBM (spike_B_two_gemm):
     The energy reduction needs both T[i,j,a,b] and T[i,j,b,a] = T.T.
     In the current _mp2_energy_kernel, T.T is loaded from HBM via
     nl.load_transpose2d. In Phase 3, T is SBUF-resident (from PSUM).
     Strategy: compute T_T = B_j @ B_i.T as a second GEMM tile in the
     same kernel — both T and T_T end up in SBUF, no HBM needed.
     Expected: both GEMMs tile into PSUM sequentially; the second reuses
     the same PSUM buffer (zero'd between uses).
     Fail mode: PSUM aliasing error or second GEMM overwrites first SBUF tile.
     Rejection: if compute cost of two GEMMs > one GEMM + HBM reload.

  C. TE/VE concurrency (spike_C_te_ve_overlap):
     In a kernel that does GEMM for pair k+1 and energy reduction for
     pair k in the same loop iteration, does the profiler show TE and VE
     active simultaneously?
     This is NOT testable from Python alone — it requires a Neuron Profiler
     2.0 capture and a Perfetto trace.
     This kernel provides the NEFF to profile; run it through
     neuron-profile capture + view --output-format summary-json.
     Success criterion: dma_active_time_percent > 10% while both
     vector_engine_active_time_percent and tensor_engine_active_time_percent
     are elevated (TE/VE overlap in instruction timeline).

## Usage

    # Run all three spikes and print results:
    python scripts/spike_phase3_fused_gemm_energy.py

    # Run on trn1 via SSM (see scripts/run_neuron_tests.sh for instance setup):
    AWS_PROFILE=aws ./scripts/run_phase3_spike.sh

    # Profile spike_C for TE/VE overlap evidence:
    AWS_PROFILE=aws ./scripts/run_phase3_spike.sh --profile-spike-c

## Interpreting results

  A: Pass if output matches torch.matmul(A, B).sum() * 2 within atol=1e-2.
     The *2.0 is the VE multiply applied to the PSUM result.
  B: Pass if output matches T*(2T-T_T) sum within atol=1e-2. T and T_T
     computed from two GEMM tiles; energy expression applied on-chip.
  C: Requires profiler — see scripts/run_phase3_spike.sh --profile-spike-c.
     Python-side: pass if result is numerically correct. The overlap
     question is answered by the NTFF trace.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Spike A: PSUM → SBUF → VE in one @nki.jit
# ---------------------------------------------------------------------------
# Tests whether TE output (PSUM) can feed VE energy ops in one kernel.
# Simplified energy expression: 2 * T (scalar multiply, all VE, clearly SBUF-resident).
# If this compiles + runs correctly, the TE→VE data path is confirmed.

if HAS_NKI:

    @nki.jit
    def _spike_a_psum_to_ve(a, b):
        """GEMM(a, b) → PSUM → tensor_copy → SBUF → VE multiply(2.0) → sum → HBM.

        No HBM intermediate for the tile result. Proves the TE→VE chain
        works within one @nki.jit without materialising T_flat.

        Output declared inside and returned (same pattern as _mp2_energy_kernel).
        Caller guarantees: a is (128, 128), b is (128, 128).
        """
        TILE_M, TILE_K, TILE_N = 128, 128, 128

        psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        a_t = nl.load_transpose2d(a[0:TILE_M, 0:TILE_K])
        b_tile = nl.load(b[0:TILE_K, 0:TILE_N])
        nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)

        # TE → SBUF handoff (documented NKI pattern).
        t_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(src=psum, dst=t_sbuf)

        # VE ops on SBUF-resident T — no HBM read of T_flat.
        scaled = nl.multiply(t_sbuf, 2.0)

        # Free-dim reduce: (TILE_M, TILE_N) → (TILE_M, 1).
        row_sums = nl.sum(scaled, axis=1, keepdims=True)
        out = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nl.store(out[0:TILE_M, 0:1], value=row_sums)
        return out

    # ---------------------------------------------------------------------------
    # Spike B: T and T_T from two GEMMs — no HBM for either
    # ---------------------------------------------------------------------------
    # Tests whether two GEMM tiles can each PSUM→SBUF in one kernel, then VE
    # energy ops use both SBUF tiles. Answers "how do we get T.T without HBM?".

    @nki.jit
    def _spike_b_two_gemm(b_i, b_j, denom):
        """Two-GEMM spike for Phase 3 T.T strategy.

        Computes in one @nki.jit:
          T     = B_i @ B_j.T   (GEMM 1, PSUM → t_sbuf)
          T_T   = B_j @ B_i.T   (GEMM 2, PSUM → t_t_sbuf)
          energy = T * (2*T - T_T) / denom   (VE, fully SBUF-resident)
          return sum(energy, axis=1)          (free-dim reduce → HBM)

        No HBM intermediate for T or T_T. nc_matmul convention:
          stationary = load_transpose2d(X): (TILE_K, TILE_M), partition=K
          moving     = load_transpose2d(Y): (TILE_K, TILE_N), partition=K
          result in PSUM: (TILE_M, TILE_N) = X @ Y.T

        Two separate nl.psum allocations — spike tests whether the compiler
        allows sequential PSUM reuse or rejects double-allocation.

        Output declared inside and returned (same pattern as _mp2_energy_kernel).
        Caller guarantees: b_i, b_j are (128, 128). denom is (128, 128).
        """
        TILE_M, TILE_K = 128, 128

        # GEMM 1: T = B_i @ B_j.T
        # stationary=B_i.T=(TILE_K,TILE_M), moving=B_j.T=(TILE_K,TILE_M)
        # → PSUM = B_i @ B_j.T = (TILE_M, TILE_M)
        psum_t = nl.zeros((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.psum)
        bi_stat = nl.load_transpose2d(b_i[0:TILE_M, 0:TILE_K])
        bj_mov = nl.load_transpose2d(b_j[0:TILE_M, 0:TILE_K])
        nisa.nc_matmul(dst=psum_t, stationary=bi_stat, moving=bj_mov, accumulate=True)
        t_sbuf = nl.ndarray((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(src=psum_t, dst=t_sbuf)

        # GEMM 2: T_T = B_j @ B_i.T
        # stationary=B_j.T=(TILE_K,TILE_M), moving=B_i.T=(TILE_K,TILE_M)
        # → PSUM = B_j @ B_i.T = (TILE_M, TILE_M)
        psum_tt = nl.zeros((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.psum)
        bj_stat = nl.load_transpose2d(b_j[0:TILE_M, 0:TILE_K])
        bi_mov = nl.load_transpose2d(b_i[0:TILE_M, 0:TILE_K])
        nisa.nc_matmul(dst=psum_tt, stationary=bj_stat, moving=bi_mov, accumulate=True)
        t_t_sbuf = nl.ndarray((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(src=psum_tt, dst=t_t_sbuf)

        # VE: energy expression — all SBUF-resident, no HBM reads.
        denom_tile = nl.load(denom[0:TILE_M, 0:TILE_M])
        two_t = nl.multiply(t_sbuf, 2.0)
        diff = nl.subtract(two_t, t_t_sbuf)
        numer = nl.multiply(t_sbuf, diff)
        energy_tile = nl.multiply(numer, nl.reciprocal(denom_tile))

        # Free-dim reduce → (TILE_M, 1).
        row_sums = nl.sum(energy_tile, axis=1, keepdims=True)
        out = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nl.store(out[0:TILE_M, 0:1], value=row_sums)
        return out

    # ---------------------------------------------------------------------------
    # Spike C: TE/VE concurrency within a pair loop
    # ---------------------------------------------------------------------------
    # Kernel that interleaves GEMM (pair k+1) and energy reduction (pair k)
    # in a loop body. Whether TE and VE genuinely overlap is answered by the
    # profiler — this Python file gives the NEFF to capture.

    @nki.jit
    def _spike_c_te_ve_overlap(b_pairs, denom):
        """Interleaved GEMM + energy across NPAIRS iterations.

        Loop body for pair k:
          - GEMM tile k into PSUM (TE work)
          - tensor_copy psum → sbuf  (TE→VE handoff)
          - Energy reduction on sbuf tile k (VE work)
          - Store pair k result into column k of out

        Profiler should show: TE active while VE is active for adjacent pairs.

        Output declared inside and returned as (TILE_M, NPAIRS); caller
        transposes to (NPAIRS, TILE_M, 1) for comparison with reference.

        Caller guarantees:
          b_pairs: (NPAIRS, NVIR, TILE_K), each slice is B[i] or B[j]
          denom:   (NVIR, NVIR) — unused in simplified energy (2*T)

        NVIR = TILE_M = 128, TILE_K = 128, NPAIRS ≥ 2.
        """
        NPAIRS = b_pairs.shape[0]
        TILE_M, TILE_K = 128, 128

        out = nl.ndarray((TILE_M, NPAIRS), dtype=nl.float32, buffer=nl.shared_hbm)

        for k in nl.affine_range(NPAIRS):
            # GEMM for pair k: B[k] @ B[k].T (SYRK, TE).
            psum = nl.zeros((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.psum)
            b_stat = nl.load_transpose2d(b_pairs[k, 0:TILE_M, 0:TILE_K])
            b_mov = nl.load_transpose2d(b_pairs[k, 0:TILE_M, 0:TILE_K])
            nisa.nc_matmul(dst=psum, stationary=b_stat, moving=b_mov, accumulate=True)

            # PSUM → SBUF (TE→VE handoff).
            t_sbuf = nl.ndarray((TILE_M, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(src=psum, dst=t_sbuf)

            # VE: simplified energy (2*T sum) on SBUF-resident tile.
            scaled = nl.multiply(t_sbuf, 2.0)
            row_sums = nl.sum(scaled, axis=1, keepdims=True)
            nl.store(out[0:TILE_M, k : k + 1], value=row_sums)

        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _check_nki() -> bool:
    if not HAS_NKI:
        print("HAS_NKI=False — neuronxcc not importable; spikes require trn1.")
        return False
    return True


def _to_xla(*tensors):
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


def run_spike_a() -> bool:
    """Spike A: PSUM → SBUF → VE chain (no HBM intermediate)."""
    print("\n=== Spike A: PSUM → SBUF → VE chain ===")
    TILE = 128
    rng = torch.Generator()
    rng.manual_seed(42)
    A = torch.randn(TILE, TILE, generator=rng)
    B = torch.randn(TILE, TILE, generator=rng)

    # Reference: 2 * (A @ B), summed per row.
    ref = (2.0 * (A @ B)).sum(dim=1, keepdim=True)

    try:
        (a, b), orig = _to_xla(A.contiguous(), B.contiguous())
        t0 = time.perf_counter()
        result = _spike_a_psum_to_ve(a, b).to(orig)
        dt = time.perf_counter() - t0
        ok = torch.allclose(result, ref, atol=1e-2, rtol=1e-3)
        print(f"  Result: {'PASS' if ok else 'FAIL'}  ({dt * 1000:.1f} ms)")
        if not ok:
            print(f"  max_diff = {(result - ref).abs().max().item():.4e}")
        return ok
    except Exception as exc:
        print(f"  COMPILE/RUNTIME ERROR: {exc}")
        return False


def run_spike_b() -> bool:
    """Spike B: T and T_T from two GEMMs — no HBM for either."""
    print("\n=== Spike B: two-GEMM T and T_T without HBM ===")
    TILE = 128
    rng = torch.Generator()
    rng.manual_seed(7)
    B_i = torch.randn(TILE, TILE, generator=rng)
    B_j = torch.randn(TILE, TILE, generator=rng)
    denom = torch.ones(TILE, TILE) * 2.0  # fixed denominator for reference clarity

    # Reference: T = B_i @ B_j.T, T_T = B_j @ B_i.T
    T = B_i @ B_j.T
    T_T = B_j @ B_i.T
    ref = (T * (2.0 * T - T_T) / denom).sum(dim=1, keepdim=True)

    try:
        (bi, bj, d), orig = _to_xla(B_i.contiguous(), B_j.contiguous(), denom.contiguous())
        t0 = time.perf_counter()
        result = _spike_b_two_gemm(bi, bj, d).to(orig)
        dt = time.perf_counter() - t0
        ok = torch.allclose(result, ref, atol=1e-2, rtol=1e-3)
        print(f"  Result: {'PASS' if ok else 'FAIL'}  ({dt * 1000:.1f} ms)")
        if not ok:
            print(f"  max_diff = {(result - ref).abs().max().item():.4e}")
        return ok
    except Exception as exc:
        print(f"  COMPILE/RUNTIME ERROR: {exc}")
        return False


def run_spike_c(npairs: int = 8) -> bool:
    """Spike C: TE/VE concurrency kernel — correctness check only.

    TE/VE overlap requires profiler evidence (see run_phase3_spike.sh
    --profile-spike-c). This function confirms the kernel runs correctly.
    """
    print(f"\n=== Spike C: TE/VE interleaved loop (NPAIRS={npairs}) ===")
    TILE = 128
    rng = torch.Generator()
    rng.manual_seed(3)
    B_pairs = torch.randn(npairs, TILE, TILE, generator=rng)
    denom = torch.ones(TILE, TILE)

    # Reference: for each pair k, 2 * (B_pairs[k] @ B_pairs[k].T) summed per row.
    ref_list = []
    for k in range(npairs):
        Bk = B_pairs[k]
        T = Bk @ Bk.T
        ref_list.append((2.0 * T).sum(dim=1, keepdim=True))
    ref = torch.stack(ref_list)  # (npairs, TILE, 1)

    try:
        (bp, d), orig = _to_xla(B_pairs.contiguous(), denom.contiguous())
        t0 = time.perf_counter()
        # Kernel returns (TILE_M, NPAIRS); transpose to (NPAIRS, TILE_M, 1) for ref comparison.
        raw = _spike_c_te_ve_overlap(bp, d).to(orig)
        dt = time.perf_counter() - t0
        result = raw.T.unsqueeze(-1)  # (npairs, TILE, 1)
        ok = torch.allclose(result, ref, atol=1e-2, rtol=1e-3)
        print(f"  Result: {'PASS' if ok else 'FAIL'}  ({dt * 1000:.1f} ms, {npairs} pairs)")
        if not ok:
            for k in range(npairs):
                diff = (result[k] - ref[k]).abs().max().item()
                if diff > 1e-2:
                    print(f"  pair {k}: max_diff={diff:.4e}")
        print(
            "  NOTE: TE/VE overlap answer requires neuron-profile trace.\n"
            "  Run: AWS_PROFILE=aws ./scripts/run_phase3_spike.sh --profile-spike-c"
        )
        return ok
    except Exception as exc:
        print(f"  COMPILE/RUNTIME ERROR: {exc}")
        return False


def main() -> int:
    print("trnblas Phase 3 spike — #38 fused GEMM+energy pre-implementation probe")
    print(f"HAS_NKI = {HAS_NKI}")

    if not _check_nki():
        print("\nSkipping all spikes (no NKI). Run on trn1.")
        return 0

    results = {
        "A": run_spike_a(),
        "B": run_spike_b(),
        "C": run_spike_c(),
    }

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  Spike {name}: {'PASS' if ok else 'FAIL'}")

    all_pass = all(results.values())
    if all_pass:
        print("\nAll spikes passed. Phase 3 implementation path is clear:")
        print("  A → TE→VE handoff via PSUM→SBUF works in one @nki.jit")
        print("  B → Two-GEMM strategy for T and T_T is viable (no HBM needed)")
        print("  C → TE/VE overlap requires profiler — run with --profile-spike-c")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\nFailed spikes: {failed}. Review error output for implementation constraints.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
