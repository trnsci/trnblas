"""
NKI kernels of the Ozaki scheme II prototype (trnblas#132, `trnblas/ozaki2.py`).

- `_ozaki2_gemm_<fam>_kernel(at, b)`: the residue GEMMs of all n moduli in
  one launch. `at` (n, Kp, Mp) and `b` (n, Kp, N) bf16 hold the balanced
  residues of A^T and B; returns (n, Mp, N) fp32 balanced residues of
  at[l]^T b[l] mod m_l. Per modulus and output tile: bf16 matmuls into fp32
  PSUM in groups of `group_tiles` K tiles (every partial sum an exact
  integer below 2^24), and at each group close a VectorE reduction
  T = S + acc; q = rint(T fl(1/m)); acc = T - q m (4 instructions, exact
  integers), then one canonical reduction. Bitwise `ozaki2.residue_gemm_ops`
  and `ozaki2.reference_residue_gemm`.
- `_ozaki2_crt_<fam>_kernel(res)`: the device CRT of `ozaki2.crt_device`,
  element-wise, exact fp32 integer ops: (keep, Mp, N) fp32 packed digits.

One kernel pair per moduli family (b7: groups of 4096 terms, b8: 1024,
b9: 256); the moduli are the family's first n = at.shape[0], as static
constants. Both are gridless and split row tiles between the programs of an
LNC=2 core by program_id; every output element sees a fixed op sequence, so
the result is LNC-invariant. The matmul helpers are Ozaki I's
(`dispatch._oz_mm`, `_oz_psum`), unchanged.
"""

from __future__ import annotations

from .. import ozaki2 as _o2

try:  # importable without nki (CPU installs, and tests that stub the native path)
    import nki
    import nki.isa as nisa
    import nki.language as nl

    _HAS_NKI = True
except ImportError:
    _HAS_NKI = False

if _HAS_NKI:
    from .dispatch import _lnc_sizes, _oz_mm, _oz_psum, _ozs_range

    _OZ2_MAGIC = _o2.MAGIC

    def _oz2_round(dst, src, inv, shape):
        """dst = rint(src * inv) (fp32; |src * inv| < 2^22)."""
        u = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=u, data=src, op0=nl.multiply, operand0=inv, op1=nl.add, operand1=_OZ2_MAGIC
        )
        nisa.tensor_scalar(dst=dst, data=u, op0=nl.subtract, operand0=_OZ2_MAGIC)

    def _oz2_reduce(ps, acc, m, TM, nsizes):
        """Group close: acc = (ps + acc) mod m, a representative within
        0.52 m of zero. Exact: |ps| + |acc| < 2^24 and q m < 2^24."""
        for i in range(TM):
            for j in range(len(nsizes)):
                shape = (128, nsizes[j])
                t = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
                q = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=t, data1=ps[i][j], data2=acc[i][j], op=nl.add)
                _oz2_round(q, t, 1.0 / m, shape)
                nisa.scalar_tensor_tensor(
                    dst=acc[i][j],
                    data=q,
                    op0=nl.multiply,
                    operand0=-float(m),
                    op1=nl.add,
                    operand1=t,
                )

    def _oz2_canon(acc, m, TM, nsizes):
        """acc = the balanced residue of acc (|acc| <= 0.52 m) mod odd m:
        rint(acc / m) is exact here, so the result is canonical."""
        for i in range(TM):
            for j in range(len(nsizes)):
                shape = (128, nsizes[j])
                q = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
                _oz2_round(q, acc[i][j], 1.0 / m, shape)
                nisa.scalar_tensor_tensor(
                    dst=acc[i][j],
                    data=q,
                    op0=nl.multiply,
                    operand0=-float(m),
                    op1=nl.add,
                    operand1=acc[i][j],
                )

    def _oz2_tile(at, b, out, m0, TM, n0, nsizes, moduli, GT):
        """TM M tiles x N sub-tiles `nsizes` of every modulus's residues, from
        row m0 and column n0. Moduli are a static loop (O(n) traced code),
        the PSUM groups of each a runtime loop."""
        n = at.shape[0]
        KT = at.shape[1] // 128
        G = KT // GT
        R = KT - G * GT
        noff = [sum(nsizes[:j]) for j in range(len(nsizes))]
        for l in range(n):
            m = moduli[l]
            acc = [
                [nl.zeros((128, ns), dtype=nl.float32, buffer=nl.sbuf) for ns in nsizes]
                for _ in range(TM)
            ]
            if G:
                for g in nl.sequential_range(G):
                    ps = _oz_psum(TM, nsizes)
                    for kk in nl.affine_range(GT):
                        _oz_mm(at, b, ps, l, l, (g * GT + kk) * 128, m0, TM, n0, nsizes)
                    _oz2_reduce(ps, acc, m, TM, nsizes)
            if R:
                ps = _oz_psum(TM, nsizes)
                for kk in nl.affine_range(R):
                    _oz_mm(at, b, ps, l, l, (G * GT + kk) * 128, m0, TM, n0, nsizes)
                _oz2_reduce(ps, acc, m, TM, nsizes)
            _oz2_canon(acc, m, TM, nsizes)
            for i in range(TM):
                r = m0 + i * 128
                for j in range(len(nsizes)):
                    c = n0 + noff[j]
                    nl.store(out[l, r : r + 128, c : c + nsizes[j]], value=acc[i][j])

    def _oz2_mblock(at, b, out, m0, TM, c0, NP, moduli, GT):
        """Columns [c0, c0 + NP) of one M block (as Ozaki I's `_oz_mblock`)."""
        NF = NP // 1024
        for nb in nl.affine_range(NF):
            _oz2_tile(at, b, out, m0, TM, c0 + nb * 1024, [512, 512], moduli, GT)
        if NP - NF * 1024:
            sizes = _lnc_sizes(NP - NF * 1024, 512)
            _oz2_tile(at, b, out, m0, TM, c0 + NF * 1024, sizes, moduli, GT)

    def _oz2_gemm(at, b, out, moduli, GT):
        """The LNC split of `_ozaki_gemm_kernel`: halves of the 128-row M
        tiles per program (an odd middle tile computed by both, identically),
        or halves of the columns with at most two M tiles."""
        Mp = at.shape[2]
        N = b.shape[2]
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        MT = Mp // 128
        if n_prog > 1 and MT <= 2 and N >= 2:
            NPP = (N + n_prog - 1) // n_prog
            _oz2_mblock(at, b, out, 0, MT, nl.program_id(0) * (N - NPP), NPP, moduli, GT)
            return
        MPT = (MT + n_prog - 1) // n_prog
        r0 = nl.program_id(0) * ((MT - MPT) * 128) if n_prog > 1 else 0
        NB = MPT // 2
        for mb in nl.affine_range(NB):
            _oz2_mblock(at, b, out, r0 + mb * 256, 2, 0, N, moduli, GT)
        if MPT % 2:
            _oz2_mblock(at, b, out, r0 + NB * 256, 1, 0, N, moduli, GT)

    # ── device CRT ───────────────────────────────────────────────────────────

    _OZ2_CW = 512  # columns per CRT tile

    def _oz2_crt_tile(res, out, r0, c0, w, c):
        """`ozaki2.crt_device` on rows r0 .. r0 + 127, columns c0 .. c0 + w - 1."""
        shape = (128, w)

        def f():
            return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)

        L = [None] * c.nl
        qa = f()
        for l in range(c.n):
            r = nl.load(res[l, r0 : r0 + 128, c0 : c0 + w])
            for j in range(c.nl):
                wv = float(c.W[l, j])
                if wv == 0.0:
                    continue
                if L[j] is None:
                    L[j] = f()
                    nisa.tensor_scalar(dst=L[j], data=r, op0=nl.multiply, operand0=wv)
                else:
                    nisa.scalar_tensor_tensor(
                        dst=L[j],
                        data=r,
                        op0=nl.multiply,
                        operand0=wv,
                        op1=nl.add,
                        operand1=L[j],
                    )
            om = float(c.omega[l])
            if l == 0:
                nisa.tensor_scalar(dst=qa, data=r, op0=nl.multiply, operand0=om)
            else:
                nisa.scalar_tensor_tensor(
                    dst=qa, data=r, op0=nl.multiply, operand0=om, op1=nl.add, operand1=qa
                )
        q = f()
        nisa.tensor_scalar(
            dst=q, data=qa, op0=nl.add, operand0=_OZ2_MAGIC, op1=nl.subtract, operand1=_OZ2_MAGIC
        )
        D = []
        for j in range(c.nl):
            if L[j] is None:
                L[j] = f()
                nisa.memset(dst=L[j], value=0.0)
            mv = float(c.Ml[j])
            if mv != 0.0:
                nisa.scalar_tensor_tensor(
                    dst=L[j], data=q, op0=nl.multiply, operand0=-mv, op1=nl.add, operand1=L[j]
                )
            D.append(L[j])
        sc = float(2.0**c.beta)
        for j in range(c.nl - 1):
            cy = f()
            _oz2_round(cy, D[j], 1.0 / sc, shape)
            nisa.scalar_tensor_tensor(
                dst=D[j], data=cy, op0=nl.multiply, operand0=-sc, op1=nl.add, operand1=D[j]
            )
            nisa.tensor_tensor(dst=D[j + 1], data1=D[j + 1], data2=cy, op=nl.add)
        for p in range(c.j0, c.npk):
            lo, hi = 2 * p, 2 * p + 1
            if hi >= c.nl:
                v = D[lo]
            else:
                v = f()
                nisa.scalar_tensor_tensor(
                    dst=v, data=D[hi], op0=nl.multiply, operand0=sc, op1=nl.add, operand1=D[lo]
                )
            nl.store(out[p - c.j0, r0 : r0 + 128, c0 : c0 + w], value=v)

    def _oz2_crt(res, out, c):
        """All tiles: row tiles split between the LNC programs (Mp is a
        multiple of 128), columns in tiles of 512 plus a tail."""
        Mp = res.shape[1]
        N = res.shape[2]
        x0, per, _, _ = _ozs_range(Mp, 128)
        CF = N // _OZ2_CW
        tail = N - CF * _OZ2_CW
        for i in nl.affine_range(per):
            r0 = x0 + i * 128
            for cb in nl.affine_range(CF):
                _oz2_crt_tile(res, out, r0, cb * _OZ2_CW, _OZ2_CW, c)
            if tail:
                _oz2_crt_tile(res, out, r0, CF * _OZ2_CW, tail, c)

    # One kernel pair per family, module-level with literal family names (the
    # NKI frontend resolves no closure variables).

    @nki.jit
    def _ozaki2_gemm_b7_kernel(at, b):
        """Residue GEMMs, b7 family (moduli < 2^7, groups of 4096 terms)."""
        out = nl.ndarray(
            (at.shape[0], at.shape[2], b.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm
        )
        f = _o2.FAMILIES["b7"]
        _oz2_gemm(at, b, out, f.moduli, f.group_tiles)
        return out

    @nki.jit
    def _ozaki2_gemm_b8_kernel(at, b):
        """Residue GEMMs, b8 family (moduli < 2^8, groups of 1024 terms)."""
        out = nl.ndarray(
            (at.shape[0], at.shape[2], b.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm
        )
        f = _o2.FAMILIES["b8"]
        _oz2_gemm(at, b, out, f.moduli, f.group_tiles)
        return out

    @nki.jit
    def _ozaki2_gemm_b9_kernel(at, b):
        """Residue GEMMs, b9 family (moduli < 2^9, groups of 256 terms)."""
        out = nl.ndarray(
            (at.shape[0], at.shape[2], b.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm
        )
        f = _o2.FAMILIES["b9"]
        _oz2_gemm(at, b, out, f.moduli, f.group_tiles)
        return out

    @nki.jit
    def _ozaki2_crt_b7_kernel(res):
        """Device CRT, b7 family: (keep, Mp, N) fp32 from (n, Mp, N) residues."""
        c = _o2.crt("b7", res.shape[0])
        out = nl.ndarray(
            (c.keep, res.shape[1], res.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm
        )
        _oz2_crt(res, out, c)
        return out

    @nki.jit
    def _ozaki2_crt_b8_kernel(res):
        """Device CRT, b8 family."""
        c = _o2.crt("b8", res.shape[0])
        out = nl.ndarray(
            (c.keep, res.shape[1], res.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm
        )
        _oz2_crt(res, out, c)
        return out

    @nki.jit
    def _ozaki2_crt_b9_kernel(res):
        """Device CRT, b9 family."""
        c = _o2.crt("b9", res.shape[0])
        out = nl.ndarray(
            (c.keep, res.shape[1], res.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm
        )
        _oz2_crt(res, out, c)
        return out

    GEMM_KERNELS = {
        "b7": _ozaki2_gemm_b7_kernel,
        "b8": _ozaki2_gemm_b8_kernel,
        "b9": _ozaki2_gemm_b9_kernel,
    }
    CRT_KERNELS = {
        "b7": _ozaki2_crt_b7_kernel,
        "b8": _ozaki2_crt_b8_kernel,
        "b9": _ozaki2_crt_b9_kernel,
    }
else:  # pragma: no cover - CPU-only installs
    GEMM_KERNELS = {}
    CRT_KERNELS = {}


# ── host side: launches and the gemm_fp64(scheme="ozaki2") route ─────────────


def _launcher(fam: str, crt_mode: str):
    """`ozaki2.run`'s launch: the residue-GEMM kernel on K-major device
    residues (uploaded once per call when they fit, as Ozaki I's slices),
    then the device CRT (crt_mode "device") or the residues themselves
    ("host"), downloaded. Under the simulator, both kernels on host blocks."""
    import numpy as np
    import torch

    from . import dispatch as D

    gemm_op, crt_op = f"ozaki2_gemm_{fam}", f"ozaki2_crt_{fam}"
    cache: dict = {}

    def launch(ra, rb, m, kk, nn):
        if D._use_simulator():
            at, bt = _o2.host_blocks(ra, rb, m, kk, nn)
            D._check_kernel_operands(gemm_op, at, bt)
            res = D._simulate_bf16(GEMM_KERNELS[fam], at, bt)
            if crt_mode == "host":
                return res
            return torch.from_numpy(np.asarray(nki.simulate(CRT_KERNELS[fam])(res.numpy())))
        D._check_kernel_operands(gemm_op, ra.d, rb.d)
        at = D._oz_device_block(ra, cache, "a", kk, m)
        bt = D._oz_device_block(rb, cache, "b", kk, nn)
        # .get: without nki (stubbed native ops in tests) the tables are empty
        res = D._call_kernel(GEMM_KERNELS.get(fam), gemm_op, at, bt)
        if crt_mode == "device":
            res = D._call_kernel(CRT_KERNELS.get(fam), crt_op, res)
        return D._to_host(res, "cpu")

    return launch


def gemm(A, B, n: int = _o2.DEFAULT_MODULI, fam: str = _o2.DEFAULT_FAMILY, crt_mode="device"):
    """fp64 (M, N) = A @ B by Ozaki II for 2-D host fp64 A, B: the kernels on
    the native path (or the simulator), else host fp64 matmul (the routes of
    Ozaki I, `dispatch._ozaki_route`). Non-finite operands take host fp64.
    The operands are scaled and reduced to residues on the host (the device
    split of #123 has no Ozaki II counterpart yet)."""
    import torch

    from . import dispatch as D

    f = _o2.family(fam)
    n = _o2.check_moduli(n, f)
    if not f.kernel:
        raise ValueError(f"family {f.name} is analysis-only (no kernel)")
    if D._ozaki_route() == "host":
        return torch.matmul(A, B)
    if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
        return torch.matmul(A, B)
    try:
        return _o2.run(A, B, n, _launcher(f.name, crt_mode), f, crt_mode, empty=D._ozaki_host_empty)
    except Exception as exc:
        if D._REQUIRE_NKI:
            raise
        D._warn_fallback(exc, "Ozaki II fp64 GEMM kernel failed — falling back to host fp64")
        return torch.matmul(A, B)
