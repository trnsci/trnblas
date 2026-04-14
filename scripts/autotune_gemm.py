"""One-shot NKI GEMM tile-config study (#26).

Sweeps a small grid of (TILE_M, TILE_K, TILE_N) tuples against a set of
representative (M, K, N) shapes, measures warm-cache per-call timing,
reports winner per shape, and flags whether a uniform better default
exists.

This is a **study**, not a runtime autotuner. Run once on trn1:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh  # makes sure hardware is up
    # then via SSM: /opt/aws_neuronx_venv_pytorch_*/bin/python scripts/autotune_gemm.py

Use the emitted summary to update the hardcoded tiles in
`trnblas/nki/dispatch.py` (constants at the top of the file) and/or
add a small shape heuristic in `_nki_gemm_impl`.

On CPU the script exits with a note that the NKI path is required —
tile configs are a no-op through torch.matmul.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from trnblas.nki.dispatch import _REQUIRE_NKI, HAS_NKI, _round_up, _to_xla

# Tile configurations to sweep. NKI 2.24 limits: partition dim ≤ 128,
# stationary free dim ≤ 128, moving free dim ≤ 512.
TILE_CONFIGS: list[tuple[int, int, int]] = [
    (64, 64, 128),
    (64, 64, 256),
    (64, 64, 512),
    (64, 128, 128),
    (64, 128, 256),
    (64, 128, 512),
    (128, 64, 128),
    (128, 64, 256),
    (128, 64, 512),
    (128, 128, 128),
    (128, 128, 256),
    (128, 128, 512),  # current default
]

# Representative shapes — GEMM sizes that show up in DF-MP2 + batched paths.
SHAPES: list[tuple[str, tuple[int, int, int]]] = [
    ("square_512", (512, 512, 512)),
    ("square_1024", (1024, 1024, 1024)),
    ("df_half_xform", (512, 1024, 1536)),
    ("df_metric", (1536, 1536, 1024)),
    ("tall_batched_slice", (256, 128, 256)),
]

WARMS = 5  # per-config measurement passes after cold


def _make_gemm_kernel(tile_m: int, tile_k: int, tile_n: int) -> Callable:
    """Factory: build a compiled NKI GEMM kernel for a specific tile tuple."""
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def _kernel(a, b):
        M, K = a.shape
        _, N = b.shape
        TILE_M = tile_m
        TILE_K = tile_k
        # Match the existing kernel's small-N behaviour: if N fits in one
        # moving tile, don't require a full multiple.
        TILE_N = N if tile_n >= N else tile_n

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

    return _kernel


def _pad(A: torch.Tensor, B: torch.Tensor, tile_m: int, tile_k: int, tile_n: int):
    M, K = A.shape
    _, N = B.shape
    M_pad = _round_up(M, tile_m)
    K_pad = _round_up(K, tile_k)
    N_pad = N if tile_n >= N else _round_up(N, tile_n)
    if (M_pad, K_pad, N_pad) == (M, K, N):
        return A.contiguous(), B.contiguous(), (M, N)
    A_p = torch.zeros(M_pad, K_pad, dtype=A.dtype, device=A.device)
    A_p[:M, :K] = A
    B_p = torch.zeros(K_pad, N_pad, dtype=B.dtype, device=B.device)
    B_p[:K, :N] = B
    return A_p.contiguous(), B_p.contiguous(), (M, N)


def measure(
    kernel: Callable,
    A: torch.Tensor,
    B: torch.Tensor,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> tuple[float, float]:
    """Return (cold_seconds, warm_mean_seconds)."""
    A_p, B_p, (M, N) = _pad(A, B, tile_m, tile_k, tile_n)
    (a, b), orig = _to_xla(A_p, B_p)

    # Cold: first invocation triggers NEFF compile (possibly) + run.
    t0 = time.perf_counter()
    c = kernel(a, b)
    _ = c.to(orig)
    cold = time.perf_counter() - t0

    times = []
    for _ in range(WARMS):
        t0 = time.perf_counter()
        c = kernel(a, b)
        _ = c.to(orig)
        times.append(time.perf_counter() - t0)
    warm = sum(times) / len(times)
    return cold, warm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs", type=int, default=None, help="Limit to first N configs for smoke-test"
    )
    args = ap.parse_args()

    if not HAS_NKI:
        print(
            "HAS_NKI is False — NKI isn't available; tile configs are a no-op "
            "on the torch.matmul fallback. Run on a Neuron instance."
        )
        return

    configs = TILE_CONFIGS[: args.configs] if args.configs else TILE_CONFIGS

    # Collect: results[shape_label][tile_tuple] = warm_ms
    results: dict[str, dict[tuple[int, int, int], float]] = {}
    errors: dict[tuple[str, tuple[int, int, int]], str] = {}

    for label, (M, K, N) in SHAPES:
        print(f"\n[shape {label}  ({M}, {K}, {N})]")
        torch.manual_seed(0)
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        flops = 2 * M * K * N
        results[label] = {}
        for tile in configs:
            try:
                kernel = _make_gemm_kernel(*tile)
                cold, warm = measure(kernel, A, B, *tile)
                tflops = flops / warm / 1e12
                results[label][tile] = warm
                print(
                    f"  tile={tile}  cold={cold * 1000:7.2f}ms  "
                    f"warm={warm * 1000:7.3f}ms  ({tflops:5.2f} TFLOPS)"
                )
            except Exception as exc:
                errors[(label, tile)] = str(exc).splitlines()[0][:100]
                print(f"  tile={tile}  ERROR: {errors[(label, tile)]}")

    print("\n" + "=" * 72)
    print("Winner per shape (min warm time):")
    print("=" * 72)
    best_per_shape: dict[str, tuple[int, int, int]] = {}
    for label, runs in results.items():
        if not runs:
            print(f"  {label:20s}  (all configs failed)")
            continue
        best_tile = min(runs.keys(), key=lambda t: runs[t])
        best_warm = runs[best_tile]
        default_warm = runs.get((128, 128, 512))
        if default_warm is None:
            pct = ""
        else:
            speedup = default_warm / best_warm
            pct = f"  (default {default_warm * 1000:.3f}ms → {speedup:.2f}× faster)"
        print(f"  {label:20s}  tile={best_tile}  warm={best_warm * 1000:.3f}ms{pct}")
        best_per_shape[label] = best_tile

    print("\nUniform winner?")
    winners = set(best_per_shape.values())
    if len(winners) == 1:
        print(f"  YES → (TILE_M, TILE_K, TILE_N) = {winners.pop()}")
    else:
        print(f"  NO → {len(winners)} distinct winners across shapes:")
        for label, tile in best_per_shape.items():
            print(f"      {label}: {tile}")


if __name__ == "__main__":
    main()
