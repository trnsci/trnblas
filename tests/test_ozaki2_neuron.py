"""Ozaki II prototype (#132) on Trainium (native TorchNeuron path).

Bitwise equality with the host reference of the kernels' arithmetic
(`ozaki2.reference_gemm`, both CRT modes), the residue GEMM at the residue
bound (every PSUM partial sum at its exactness limit), accuracy, and the LNC
check: run at LNC=2 (default) and at LNC=1 (`NEURON_LOGICAL_NC_CONFIG=1`);
both must equal the host reference, and with TRNBLAS_LNC_DIGEST_DIR set the
second run compares its digests with the first run's. Timing is in
`scripts/probe_ozaki2.py`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from _ozaki_ref import inputs, max_cw

import trnblas
from trnblas import ozaki2
from trnblas.nki import HAS_NKI
from trnblas.nki import dispatch as D


def _native() -> bool:
    if not HAS_NKI or D._use_simulator():
        return False
    try:
        return D._get_dispatch_path() == "native"
    except Exception:
        return False


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]


@pytest.fixture
def require_nki(monkeypatch):
    monkeypatch.setattr(D, "_REQUIRE_NKI", True)


def _k2():
    from trnblas.nki import ozaki2_kernels

    return ozaki2_kernels


SHAPES = [(515, 3000, 241), (300, 8400, 241), (1024, 1024, 1024)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_default_is_fp64_accurate_and_bitwise(require_nki, shape):
    A, B = inputs(*shape)
    C = trnblas.gemm_fp64(A, B, scheme="ozaki2")
    assert C.dtype == torch.float64 and C.device.type == "cpu"
    assert torch.equal(C, ozaki2.reference_gemm(A, B))
    host = max_cw(A @ B, A, B)
    assert max_cw(C, A, B) <= max(2 * host, 1e-15)


@pytest.mark.parametrize("fam,n", [("b7", 13), ("b8", 15), ("b9", 13)])
@pytest.mark.parametrize("crt_mode", ["device", "host", "fused"])
def test_families_bitwise(require_nki, fam, n, crt_mode):
    A, B = inputs(400, 5000, 300, seed=1)
    C = _k2().gemm(A, B, n, fam, crt_mode)
    ref_mode = "host" if crt_mode == "host" else "device"  # fused: the device CRT's bits
    assert torch.equal(C, ozaki2.reference_gemm(A, B, n, fam, ref_mode))


@pytest.mark.parametrize("fam", ["b7", "b8", "b9"])
def test_residue_gemm_at_the_bound(require_nki, fam):
    """Every product at rmax^2 in the first tile: each PSUM group sum sits at
    its exactness limit and the group-close reductions see their largest
    operands."""
    f = ozaki2.FAMILIES[fam]
    K = 2 * f.group_terms + 384
    at = torch.zeros((3, K, 256), dtype=torch.bfloat16)
    b = torch.zeros((3, K, 130), dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(2)
    at.copy_(torch.randint(-f.rmax, f.rmax + 1, at.shape, generator=g).to(torch.bfloat16))
    b.copy_(torch.randint(-f.rmax, f.rmax + 1, b.shape, generator=g).to(torch.bfloat16))
    at[:, :, :128] = f.rmax
    b[:, :, :2] = f.rmax
    b[:, :, 2:4] = -f.rmax
    k2 = _k2()
    out = D._call_kernel(
        k2.GEMM_KERNELS[fam], f"ozaki2_gemm_{fam}", D._to_device(at)[0][0], D._to_device(b)[0][0]
    )
    got = D._to_host(out, "cpu").numpy()
    assert np.array_equal(got, ozaki2.reference_residue_gemm(at, b, fam))


def test_wide_range(require_nki):
    A, B = inputs(256, 3000, 200, dist="logu4", seed=3)
    C = trnblas.gemm_fp64(A, B, scheme="ozaki2", moduli=20)
    assert torch.equal(C, ozaki2.reference_gemm(A, B, 20))
    assert max_cw(C, A, B) <= max(2 * max_cw(A @ B, A, B), 2e-15)


def test_lnc_digest(require_nki):
    """sha256 of the device CRT output for a fixed problem; with
    TRNBLAS_LNC_DIGEST_DIR, written per LNC and compared with the other LNC's."""
    A, B = inputs(1000, 5000, 700, seed=5)
    ra, rb = ozaki2.split_ab(A, B, 16, "b7")
    got = _k2()._launcher("b7", "device")(ra, rb, (0, ra.xp), (0, ra.kp), (0, 700))
    ref = ozaki2.reference_launch("b7", "device")(ra, rb, (0, ra.xp), (0, ra.kp), (0, 700))
    digest = hashlib.sha256(got.numpy().tobytes()).hexdigest()
    assert digest == hashlib.sha256(ref.numpy().tobytes()).hexdigest()
    out = os.environ.get("TRNBLAS_LNC_DIGEST_DIR")
    if not out:
        return
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"ozaki2_lnc{lnc}.sha256").write_text(digest)
    other = Path(out) / f"ozaki2_lnc{'1' if lnc == '2' else '2'}.sha256"
    if other.exists():
        assert other.read_text() == digest


# ── the all-device route (#132): device split, fused kernel, device combine ──


def _digest_check(name: str, digest: str) -> None:
    out = os.environ.get("TRNBLAS_LNC_DIGEST_DIR")
    if not out:
        return
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"{name}_lnc{lnc}.sha256").write_text(digest)
    other = Path(out) / f"{name}_lnc{'1' if lnc == '2' else '2'}.sha256"
    if other.exists():
        assert other.read_text() == digest


def _split_data(kind, shape, seed, per="row"):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.standard_normal(shape)
    if kind == "wide":
        return rng.standard_normal(shape) * 10.0 ** rng.uniform(-30, 30, shape)
    bits = rng.integers(0, 2**63, shape, dtype=np.uint64)  # every exponent and sign
    # subnormal / zero elements across each row (column), none the maximum
    sub = (slice(None), slice(1, None, 3)) if per == "row" else (slice(1, None, 3), slice(None))
    bits[sub] &= ~np.uint64(0x7FF0000000000000)
    X = bits.view(np.float64).copy()
    X[~np.isfinite(X)] = 1.0
    X[::2] *= -1
    X[1] = 0.0
    return X


@pytest.mark.parametrize(
    "fam,n,kind,shape",
    [
        ("b7", 17, "normal", (515, 3000)),
        ("b7", 17, "wide", (300, 700)),
        ("b7", 24, "bits", (256, 1100)),
        ("b8", 14, "normal", (300, 8400)),
        ("b9", 13, "wide", (200, 640)),
    ],
)
@pytest.mark.parametrize("per", ["row", "col"])
def test_device_split_is_the_host_split(require_nki, fam, n, kind, shape, per):
    """The device split (exponents, exact norm sums, digits on the scaled
    grid, residues) gives the host split's residues and scale exponents bit
    for bit, per row and per column, at any LNC."""
    X = torch.from_numpy(_split_data(kind, shape, n, per))
    got = _k2()._oz2_device_split(X, n, fam, per)
    assert got is not None
    d, mu = got
    r = ozaki2.split(X, n, fam, per=per)
    host = r.d.float().numpy()
    dev = d.cpu().float().numpy()
    assert np.array_equal(dev, host.transpose(0, 2, 1) if per == "row" else host)
    assert np.array_equal(mu, r.e)
    _digest_check(
        f"ozaki2_split_{fam}_{n}_{kind}_{per}",
        hashlib.sha256(d.cpu().view(torch.int16).numpy().tobytes()).hexdigest(),
    )


@pytest.mark.parametrize("per", ["row", "col"])
def test_norm_kernels_raw_output(require_nki, per):
    """The norm-sum kernels' raw output (lanes, groups) equals the CPU
    stand-in: integer sums below 2^24, exact on VectorE and GpSimd."""
    from _ozaki_ref import ozaki2_kernel_refs

    from trnblas import ozaki

    X = _split_data("wide", (300, 1100) if per == "row" else (33000, 3), 11)
    xw = torch.from_numpy(np.ascontiguousarray(X).view(np.int32).copy())
    axis = 1 if per == "row" else 0
    e, _ = ozaki2.exponents_of(X, axis)
    rc = np.full(ozaki._round_up(len(e), 128), 1003.0, np.float32)
    rc[: len(e)] = 1003 + e
    rc = torch.from_numpy(rc[:, None] if per == "row" else rc[None, :].copy())
    k2 = _k2()
    name = "ozaki2_norm" if per == "row" else "ozaki2_norm_col"
    kern = k2._ozaki2_norm_kernel if per == "row" else k2._ozaki2_norm_col_kernel
    got = D._to_host(D._call_kernel(kern, name, *D._to_device(xw, rc)[0]), "cpu")
    assert torch.equal(got, ozaki2_kernel_refs()[name](xw, rc))


def test_combine_kernel_raw_output(require_nki):
    """RN64 of packed integers on the device: ties, sticky bits, carries,
    zero, negative and small values; rows out of range flagged."""
    import random

    R, N, K = 384, 1100, 4
    rnd = random.Random(1)
    xs = [0, 1, -1, (1 << 53) + 1, (1 << 54) - 1]
    for b in range(54, 94, 3):
        xs += [(1 << b) + (1 << (b - 53)), (1 << b) + (1 << (b - 53)) + 1, (1 << (b + 1)) - 1]
    xs += [rnd.randrange(-(1 << 94), 1 << 94) >> rnd.randrange(0, 94) for _ in range(5000)]
    P = np.zeros((K, R * N), np.float32)
    for j in range(R * N):
        x = xs[rnd.randrange(len(xs))]
        for p in range(K - 1):
            lo = x & 0xFFFFFF
            lo -= (lo >= 1 << 23) << 24
            P[p, j] = lo
            x = (x - lo) >> 24
        P[K - 1, j] = x
    P = torch.from_numpy(P.reshape(K, R, N))
    ea = torch.tensor([[rnd.randrange(-300, 300) + ozaki2.combine_offset(K)] for _ in range(R)])
    ea = ea.float()
    ea[7] += 900
    ea[200] -= 1300
    eb = torch.tensor([[rnd.randrange(-300, 300) for _ in range(N)]]).float()
    k2 = _k2()
    got = D._to_host(
        D._call_kernel(k2._ozaki2_combine_kernel, "ozaki2_combine", *D._to_device(P, ea, eb)[0]),
        "cpu",
    ).numpy()
    ref = ozaki2.combine_kernel_reference(P, ea, eb)
    assert np.array_equal(got[R:], ref[R:])
    ok = ref[R:, : R // 128].T.reshape(-1) == 0
    assert np.array_equal(got[:R][ok], ref[:R][ok])


@pytest.mark.parametrize(
    "shape,trans",
    [
        ((3267, 8400, 241), (False, False)),
        ((8192, 3508, 241), (False, False)),
        ((1024, 1024, 1024), (True, True)),
        ((2100, 1000, 1500), (False, True)),
    ],
    ids=["dfmp2_a", "dfmp2_b", "square_tt", "bands"],
)
def test_all_device_route_bitwise(require_nki, shape, trans):
    """gemm_fp64(scheme="ozaki2") all on the device equals the host
    reference and the host-split / host-combine routes bit for bit."""
    A, B = inputs(*shape, seed=12)
    a = A.T.contiguous() if trans[0] else A
    b = B.T.contiguous() if trans[1] else B
    C = trnblas.gemm_fp64(a, b, transA=trans[0], transB=trans[1], scheme="ozaki2")
    k2 = _k2()
    assert torch.equal(C, k2.gemm(A, B, device_split=False))
    assert torch.equal(C, k2.gemm(A, B, device_combine=False))
    idx = np.sort(np.random.default_rng(1).choice(shape[0], min(256, shape[0]), False))
    assert torch.equal(C[idx], ozaki2.reference_gemm(A[idx], B))
    _digest_check(
        f"ozaki2_e2e_{'x'.join(map(str, shape))}", hashlib.sha256(C.numpy().tobytes()).hexdigest()
    )


def test_all_device_time(require_nki):
    """End to end at 4096^3 (b7, 17 moduli): all on the device against the
    host split and against Ozaki I (printed; the full table is
    `scripts/probe_ozaki2.py e2e`). The gate is equality."""
    import time

    A, B = inputs(4096, 4096, 4096, seed=13)
    k2 = _k2()

    def best(fn, reps=3):
        fn()
        t = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            t = min(t, time.perf_counter() - t0)
        return out, t * 1e3

    dev, t_dev = best(lambda: trnblas.gemm_fp64(A, B, scheme="ozaki2"))
    host, t_host = best(lambda: k2.gemm(A, B, device_split=False), reps=1)
    _, t_oz1 = best(lambda: trnblas.gemm_fp64(A, B))
    print(
        f"\n4096^3 e2e: ozaki2 all-device {t_dev:.1f} ms, host split {t_host:.1f} ms, "
        f"ozaki1 {t_oz1:.1f} ms"
    )
    assert torch.equal(dev, host)
