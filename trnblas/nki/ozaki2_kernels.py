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
    from .dispatch import (
        _lnc_sizes,
        _oz_mm,
        _oz_psum,
        _ozs_col_width,
        _ozs_range,
        _ozs_split_cols,
        _ozs_split_rows,
        _ozs_tile,
    )

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

    def _oz2_tile(at, b, out, m0, TM, n0, nsizes, moduli, GT, c=None):
        """TM M tiles x N sub-tiles `nsizes` of every modulus's residues, from
        row m0 and column n0. Moduli are a static loop (O(n) traced code),
        the PSUM groups of each a runtime loop. With CRT constants `c`
        (fused kernels), each canonical residue tile goes into the tile's
        CRT sums instead of HBM, and the packed CRT output is stored at the
        end: the same ops as the residue kernel then the CRT kernel."""
        n = at.shape[0]
        if c is not None:
            Lf = [[[None] * c.nl for _ in nsizes] for _ in range(TM)]
            qf = [
                [nl.ndarray((128, ns), dtype=nl.float32, buffer=nl.sbuf) for ns in nsizes]
                for _ in range(TM)
            ]
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
                    if c is not None:
                        _oz2_crt_acc(Lf[i][j], qf[i][j], acc[i][j], l, c, (128, nsizes[j]))
                        continue
                    cc = n0 + noff[j]
                    nl.store(out[l, r : r + 128, cc : cc + nsizes[j]], value=acc[i][j])
        if c is not None:
            for i in range(TM):
                for j in range(len(nsizes)):
                    shape = (128, nsizes[j])
                    _oz2_crt_store(Lf[i][j], qf[i][j], out, m0 + i * 128, n0 + noff[j], c, shape)

    def _oz2_mblock(at, b, out, m0, TM, c0, NP, moduli, GT, c=None):
        """Columns [c0, c0 + NP) of one M block (as Ozaki I's `_oz_mblock`)."""
        NF = NP // 1024
        for nb in nl.affine_range(NF):
            _oz2_tile(at, b, out, m0, TM, c0 + nb * 1024, [512, 512], moduli, GT, c)
        if NP - NF * 1024:
            sizes = _lnc_sizes(NP - NF * 1024, 512)
            _oz2_tile(at, b, out, m0, TM, c0 + NF * 1024, sizes, moduli, GT, c)

    def _oz2_gemm(at, b, out, moduli, GT, c=None):
        """The LNC split of `_ozaki_gemm_kernel`: halves of the 128-row M
        tiles per program (an odd middle tile computed by both, identically),
        or halves of the columns with at most two M tiles."""
        Mp = at.shape[2]
        N = b.shape[2]
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        MT = Mp // 128
        if n_prog > 1 and MT <= 2 and N >= 2:
            NPP = (N + n_prog - 1) // n_prog
            _oz2_mblock(at, b, out, 0, MT, nl.program_id(0) * (N - NPP), NPP, moduli, GT, c)
            return
        MPT = (MT + n_prog - 1) // n_prog
        r0 = nl.program_id(0) * ((MT - MPT) * 128) if n_prog > 1 else 0
        NB = MPT // 2
        for mb in nl.affine_range(NB):
            _oz2_mblock(at, b, out, r0 + mb * 256, 2, 0, N, moduli, GT, c)
        if MPT % 2:
            _oz2_mblock(at, b, out, r0 + NB * 256, 1, 0, N, moduli, GT, c)

    # ── device CRT ───────────────────────────────────────────────────────────

    _OZ2_CW = 512  # columns per CRT tile

    def _oz2_crt_acc(L, qa, r, l, c, shape):
        """Add modulus l's residue tile r into the limb sums L (a list of
        nl tiles, allocated on first use) and the quotient estimate qa."""
        for j in range(c.nl):
            wv = float(c.W[l, j])
            if wv == 0.0:
                continue
            if L[j] is None:
                L[j] = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=L[j], data=r, op0=nl.multiply, operand0=wv)
            else:
                nisa.scalar_tensor_tensor(
                    dst=L[j], data=r, op0=nl.multiply, operand0=wv, op1=nl.add, operand1=L[j]
                )
        om = float(c.omega[l])
        if l == 0:
            nisa.tensor_scalar(dst=qa, data=r, op0=nl.multiply, operand0=om)
        else:
            nisa.scalar_tensor_tensor(
                dst=qa, data=r, op0=nl.multiply, operand0=om, op1=nl.add, operand1=qa
            )

    def _oz2_crt_store(L, qa, out, r0, c0, c, shape):
        """Finish the CRT of one tile (`ozaki2.crt_device` after the sums):
        q = rint(qa), D = L - q M, carries, packed pairs stored from row r0,
        column c0 of out (keep, Mp, N)."""
        w = shape[1]

        def f():
            return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)

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

    def _oz2_crt_tile(res, out, r0, c0, w, c):
        """`ozaki2.crt_device` on rows r0 .. r0 + 127, columns c0 .. c0 + w - 1."""
        shape = (128, w)
        L = [None] * c.nl
        qa = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
        for l in range(c.n):
            r = nl.load(res[l, r0 : r0 + 128, c0 : c0 + w])
            _oz2_crt_acc(L, qa, r, l, c, shape)
        _oz2_crt_store(L, qa, out, r0, c0, c, shape)

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

    def _oz2_fused(at, b, fam):
        f = _o2.FAMILIES[fam]
        c = _o2.crt(fam, at.shape[0])
        out = nl.ndarray((c.keep, at.shape[2], b.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm)
        _oz2_gemm(at, b, out, f.moduli, f.group_tiles, c)
        return out

    @nki.jit
    def _ozaki2_fused_b7_kernel(at, b):
        """Residue GEMMs and the device CRT in one kernel, b7: (keep, Mp, N),
        bitwise the residue kernel followed by the CRT kernel."""
        return _oz2_fused(at, b, "b7")

    @nki.jit
    def _ozaki2_fused_b8_kernel(at, b):
        """Fused residue GEMMs + CRT, b8."""
        return _oz2_fused(at, b, "b8")

    @nki.jit
    def _ozaki2_fused_b9_kernel(at, b):
        """Fused residue GEMMs + CRT, b9."""
        return _oz2_fused(at, b, "b9")

    # ── device split (#132): norm bound, then digits and residues ───────────
    #
    # `ozaki2.norm_terms` on the device: per element tau = floor(w^2 / 2^6),
    # cut into four 9-bit limbs and summed exactly (every partial sum an
    # integer below 2^24), so the sums do not depend on the order. The host
    # finishes them (int64) and computes the scale exponents
    # (`ozaki2.scale_from_sums`); the split kernels then cut the scaled
    # integers into the Ozaki I digits and reduce them to residues
    # (`dispatch._ozs_residues`).

    _OZ2_NW = 512  # fp64 columns per tile of the row norm kernel

    def _o2_f(shape):
        return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)

    def _o2_u(shape):
        return nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)

    def _o2_floor(x, shape):
        """floor(x) for x in [0, 2^23) as fp32: cast to uint32 (nearest or
        toward zero), then minus [x - cast < 0]."""
        c = _o2_u(shape)
        nisa.tensor_copy(dst=c, src=x)
        dl = _o2_f(shape)
        nisa.scalar_tensor_tensor(
            dst=dl, data=c, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=x
        )
        fl = _o2_f(shape)
        nisa.scalar_tensor_tensor(
            dst=fl, data=dl, op0=nl.less, operand0=0.0, op1=nl.subtract, operand1=c, reverse1=True
        )
        return fl

    def _oz2_norm_acc(t, rc, acc, W):
        """Add the 9-bit limbs of tau of one (128, W) tile of fp64 bits `t`
        into acc[0..3] (`ozaki2.norm_terms`); rc = 1003 + e, (128, 1) per
        row or (128, W) per column."""
        sh = (128, W)
        f, u = _o2_f, _o2_u
        lo = t[:, 0 : 2 * W : 2]
        hi = t[:, 1 : 2 * W : 2]
        E = u(sh)
        nisa.tensor_scalar(
            dst=E, data=hi, op0=nl.right_shift, operand0=20, op1=nl.bitwise_and, operand1=0x7FF
        )
        X = f(sh)
        if rc.shape[1] == 1:
            nisa.tensor_scalar(
                dst=X, data=E, op0=nl.maximum, operand0=1.0, op1=nl.subtract, operand1=rc
            )
        else:
            nisa.scalar_tensor_tensor(
                dst=X, data=E, op0=nl.maximum, operand0=1.0, op1=nl.subtract, operand1=rc
            )
        b1 = u(sh)
        nisa.tensor_scalar(
            dst=b1, data=X, op0=nl.maximum, operand0=-70.0, op1=nl.add, operand1=127.0
        )
        s1 = u(sh)
        nisa.tensor_scalar(dst=s1, data=b1, op0=nl.left_shift, operand0=23)
        z = f(sh)
        nisa.tensor_scalar(dst=z, data=E, op0=nl.equal, operand0=0.0)
        a1 = u(sh)
        nisa.tensor_scalar(
            dst=a1, data=hi, op0=nl.bitwise_and, operand0=0xFFFFF, op1=nl.left_shift, operand1=3
        )
        a2 = u(sh)
        nisa.tensor_scalar(
            dst=a2, data=lo, op0=nl.right_shift, operand0=29, op1=nl.bitwise_or, operand1=0x3F800000
        )
        m1 = u(sh)
        nisa.tensor_tensor(dst=m1, data1=a1, data2=a2, op=nl.bitwise_or)
        q1 = f(sh)  # m1 - z (1 - 2^-23): a subnormal x keeps one unit of 2^-23
        nisa.scalar_tensor_tensor(
            dst=q1,
            data=z,
            op0=nl.multiply,
            operand0=-(1.0 - 2.0**-23),
            op1=nl.add,
            operand1=m1.view(nl.float32),
        )
        P1 = f(sh)
        nisa.tensor_tensor(dst=P1, data1=q1, data2=s1.view(nl.float32), op=nl.multiply)
        pb = P1.view(nl.uint32)
        t12 = u(sh)  # P1 cut to 12 significant bits
        nisa.tensor_scalar(
            dst=t12, data=pb, op0=nl.right_shift, operand0=12, op1=nl.left_shift, operand1=12
        )
        eb = u(sh)
        nisa.tensor_scalar(
            dst=eb, data=pb, op0=nl.right_shift, operand0=23, op1=nl.bitwise_and, operand1=0xFF
        )
        ef = f(sh)
        nisa.tensor_copy(dst=ef, src=eb)
        uf = f(sh)
        nisa.tensor_scalar(
            dst=uf, data=ef, op0=nl.maximum, operand0=12.0, op1=nl.subtract, operand1=11.0
        )
        ui = u(sh)
        nisa.tensor_copy(dst=ui, src=uf)
        u12 = u(sh)  # one unit of the 12th bit
        nisa.tensor_scalar(dst=u12, data=ui, op0=nl.left_shift, operand0=23)
        w = f(sh)
        nisa.tensor_tensor(dst=w, data1=t12.view(nl.float32), data2=u12.view(nl.float32), op=nl.add)
        y = f(sh)  # w^2 / 2^6, exact
        nisa.scalar_tensor_tensor(
            dst=y, data=w, op0=nl.multiply, operand0=2.0**-6, op1=nl.multiply, operand1=w
        )
        rem = y
        for i in (3, 2, 1):
            ts = f(sh)
            nisa.tensor_scalar(dst=ts, data=rem, op0=nl.multiply, operand0=2.0 ** (-9 * i))
            li = _o2_floor(ts, sh)
            nisa.tensor_tensor(dst=acc[i], data1=acc[i], data2=li, op=nl.add)
            r2 = f(sh)
            nisa.scalar_tensor_tensor(
                dst=r2,
                data=li,
                op0=nl.multiply,
                operand0=-(2.0 ** (9 * i)),
                op1=nl.add,
                operand1=rem,
            )
            rem = r2
        l0 = _o2_floor(rem, sh)
        nisa.tensor_tensor(dst=acc[0], data1=acc[0], data2=l0, op=nl.add)

    def _oz2_acc4(W):
        acc = []
        for _ in range(4):
            a = _o2_f((128, W))
            nisa.memset(dst=a, value=0.0)
            acc.append(a)
        return acc

    def _oz2_norm_rows(xw, rcs, out, r0, n, levels):
        """Rows r0 .. r0 + n - 1: limb sums over the column tiles, folded over
        `levels` halvings of the 512 lanes, into out[r0 : r0 + n, i Wo : (i + 1) Wo]."""
        C = xw.shape[1] // 2
        W = _OZ2_NW
        CF = C // W
        rc = nl.load(rcs[r0 : r0 + 128, 0:1])
        acc = _oz2_acc4(W)
        if CF:
            for c in nl.sequential_range(CF):
                _oz2_norm_acc(_ozs_tile(xw, r0, n, c * W, W, W), rc, acc, W)
        if C - CF * W:
            _oz2_norm_acc(_ozs_tile(xw, r0, n, CF * W, C - CF * W, W), rc, acc, W)
        Wo = W >> levels
        for i in range(4):
            cur = acc[i]
            h = W
            for _ in range(levels):
                h = h // 2
                nxt = _o2_f((128, h))
                nisa.tensor_tensor(dst=nxt, data1=cur[:, 0:h], data2=cur[:, h : 2 * h], op=nl.add)
                cur = nxt
            nl.store(out[r0 : r0 + n, i * Wo : (i + 1) * Wo], value=cur[0:n, 0:Wo])

    @nki.jit
    def _ozaki2_norm_kernel(xw, rcs):
        """The exact norm sums of `ozaki2.norm_sums` per row: `xw` (R, 2C)
        int32, the bits of a finite fp64 (R, C) matrix; `rcs` (Rp, 1) fp32,
        1003 + e_i. Returns (R, 4 Wo) fp32 integers: column i Wo + j is lane
        j's sum of limb i (weight 2^(9 i)); Wo = 512 >> levels, the lanes left
        after folding while every sum stays below 2^24. Rows by program_id."""
        R = xw.shape[0]
        C = xw.shape[1] // 2
        T = -(-C // _OZ2_NW)
        levels = 0
        while levels < 9 and T * 2 ** (9 + levels + 1) <= 2**24:
            levels += 1
        Wo = _OZ2_NW >> levels
        out = nl.ndarray((R, 4 * Wo), dtype=nl.float32, buffer=nl.shared_hbm)
        r0, MPT, rt, tail = _ozs_range(R, 128)
        if MPT:
            for i in nl.affine_range(MPT):
                _oz2_norm_rows(xw, rcs, out, r0 + i * 128, 128, levels)
        if tail:
            _oz2_norm_rows(xw, rcs, out, rt, tail, levels)
        return out

    _OZ2_NG = 255  # row tiles per partition reduction: 128 * 255 * 2^9 < 2^24

    def _oz2_norm_cols(xw, rcs, out, c0, cw, W, NG):
        """Columns c0 .. c0 + cw - 1: limb sums per partition over groups of
        _OZ2_NG row tiles, each group reduced across partitions (GpSimd; an
        exact integer sum) into out[i NG + g, c0 : c0 + cw]."""
        R = xw.shape[0]
        RT = R // 128
        tail = R - RT * 128
        rc = _o2_f((128, W))
        if cw < W:
            nisa.memset(dst=rc, value=1003.0)
        nisa.dma_copy(dst=rc[:, 0:cw], src=rcs[0:1, c0 : c0 + cw].broadcast(0, 128))
        NT = RT + (1 if tail else 0)
        for g in range(NG):
            t0 = g * _OZ2_NG
            t1 = min(NT, t0 + _OZ2_NG)
            acc = _oz2_acc4(W)
            full = min(t1, RT) - t0
            if full > 0:
                for r in nl.sequential_range(full):
                    rr = (t0 + r) * 128
                    _oz2_norm_acc(_ozs_tile(xw, rr, 128, c0, cw, W), rc, acc, W)
            if tail and t1 == NT:
                _oz2_norm_acc(_ozs_tile(xw, RT * 128, tail, c0, cw, W), rc, acc, W)
            for i in range(4):
                m = _o2_f((1, W))
                nisa.tensor_partition_reduce(dst=m, op=nl.add, data=acc[i])
                nl.store(out[i * NG + g : i * NG + g + 1, c0 : c0 + cw], value=m[0:1, 0:cw])

    @nki.jit
    def _ozaki2_norm_col_kernel(xw, rcs):
        """The exact norm sums per column: `xw` (R, 2C) int32 bits, `rcs`
        (1, Cp) fp32, 1003 + e_j. Returns (4 NG, C) fp32 integers: row
        i NG + g is row group g's sum of limb i (weight 2^(9 i)), NG =
        ceil(row tiles / 255). The column tiles split between the programs."""
        R = xw.shape[0]
        C = xw.shape[1] // 2
        W = _ozs_col_width(C)
        NT = -(-R // 128)
        NG = max(1, -(-NT // _OZ2_NG))
        out = nl.ndarray((4 * NG, C), dtype=nl.float32, buffer=nl.shared_hbm)
        c0, NCT, ct, tail = _ozs_range(C, W)
        if NCT:
            for c in nl.affine_range(NCT):
                _oz2_norm_cols(xw, rcs, out, c0 + c * W, W, W, NG)
        if tail:
            _oz2_norm_cols(xw, rcs, out, ct, tail, W, NG)
        return out

    def _oz2_split(xw, rcs, mods, fam):
        """Residues (n, Rp, Cp) bf16 per row (the Ozaki I row split with the
        residue stage; `ozaki2.split_from_bits`): `rcs` (Rp, k), 1003 + e'_i
        in column 0, k the digit count; n = mods.shape[1]."""
        R = xw.shape[0]
        C = xw.shape[1] // 2
        k = rcs.shape[1]
        n = mods.shape[1]
        rmods = _o2.FAMILIES[fam].moduli[:n]
        Rp = ((R + 127) // 128) * 128
        Cp = ((C + 127) // 128) * 128
        out = nl.ndarray((n, Rp, Cp), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        r0, MPT, rt, tail = _ozs_range(R, 128)
        if MPT:
            for i in nl.affine_range(MPT):
                _ozs_split_rows(xw, rcs, out, r0 + i * 128, 128, k, rmods)
        if tail:
            _ozs_split_rows(xw, rcs, out, rt, tail, k, rmods)
        return out

    def _oz2_split_col(xw, rcs, mods, fam):
        """Residues per column, in X's layout: `rcs` (k, Cp), 1003 + e'_j in
        row 0 (the Ozaki I column split with the residue stage)."""
        R = xw.shape[0]
        C = xw.shape[1] // 2
        k = rcs.shape[0]
        n = mods.shape[1]
        rmods = _o2.FAMILIES[fam].moduli[:n]
        W = _ozs_col_width(C)
        Rp = ((R + 127) // 128) * 128
        Cp = ((C + 127) // 128) * 128
        out = nl.ndarray((n, Rp, Cp), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        if C <= W:
            _ozs_split_cols(xw, rcs, out, 0, C, Cp, k, W, _ozs_range(R, 128), rmods)
            return out
        RT = R // 128
        rows = (0, RT, RT * 128, R - RT * 128)
        c0, NCT, ct, tail = _ozs_range(C, W)
        if NCT:
            for c in nl.affine_range(NCT):
                _ozs_split_cols(xw, rcs, out, c0 + c * W, W, W, k, W, rows, rmods)
        if tail:
            _ozs_split_cols(xw, rcs, out, ct, tail, Cp - ct, k, W, rows, rmods)
        return out

    @nki.jit
    def _ozaki2_split_b7_kernel(xw, rcs, mods):
        """Device residue split per row, b7 (`_oz2_split`)."""
        return _oz2_split(xw, rcs, mods, "b7")

    @nki.jit
    def _ozaki2_split_b8_kernel(xw, rcs, mods):
        """Device residue split per row, b8."""
        return _oz2_split(xw, rcs, mods, "b8")

    @nki.jit
    def _ozaki2_split_b9_kernel(xw, rcs, mods):
        """Device residue split per row, b9."""
        return _oz2_split(xw, rcs, mods, "b9")

    @nki.jit
    def _ozaki2_split_col_b7_kernel(xw, rcs, mods):
        """Device residue split per column, b7 (`_oz2_split_col`)."""
        return _oz2_split_col(xw, rcs, mods, "b7")

    @nki.jit
    def _ozaki2_split_col_b8_kernel(xw, rcs, mods):
        """Device residue split per column, b8."""
        return _oz2_split_col(xw, rcs, mods, "b8")

    @nki.jit
    def _ozaki2_split_col_b9_kernel(xw, rcs, mods):
        """Device residue split per column, b9."""
        return _oz2_split_col(xw, rcs, mods, "b9")

    # ── device combine (#132): RN64 of the packed CRT output ────────────────
    # `ozaki2.packed_to_fp64_bits`, op for op (exact fp32 and bitvec ops).

    _OZ2_BW = 512  # columns per tile of the combine

    def _o2_ts(data, op0, operand0, shape, op1=None, operand1=None, dtype=None):
        r = nl.ndarray(shape, dtype=dtype or nl.float32, buffer=nl.sbuf)
        if op1 is None:
            nisa.tensor_scalar(dst=r, data=data, op0=op0, operand0=operand0)
        else:
            nisa.tensor_scalar(
                dst=r, data=data, op0=op0, operand0=operand0, op1=op1, operand1=operand1
            )
        return r

    def _o2_tt(a, b, op, shape, dtype=None):
        r = nl.ndarray(shape, dtype=dtype or nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=a, data2=b, op=op)
        return r

    def _o2_stt(data, op0, operand0, op1, operand1, shape):
        r = _o2_f(shape)
        nisa.scalar_tensor_tensor(
            dst=r, data=data, op0=op0, operand0=operand0, op1=op1, operand1=operand1
        )
        return r

    def _o2_cvt(src, dtype, shape):
        r = nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=r, src=src)
        return r

    def _o2_sign(x, sh):
        gt = _o2_ts(x, nl.greater, 0.0, sh)
        r = _o2_f(sh)
        nisa.scalar_tensor_tensor(
            dst=r, data=x, op0=nl.less, operand0=0.0, op1=nl.subtract, operand1=gt, reverse1=True
        )
        return r

    def _o2_pow2(field, sh):
        """2^(field - 127) from an fp32 exponent field in [1, 254]."""
        fi = _o2_cvt(field, nl.uint32, sh)
        return _o2_ts(fi, nl.left_shift, 23, sh, dtype=nl.uint32).view(nl.float32)

    def _oz2_comb_tile(packed, ea_col, eb, out, flag, r0, c0, w):
        """Words of one (128, w) tile into out[r0 : r0 + 128, 2 c0 : 2 (c0 + w)];
        flag (128, 1) accumulates the tile's out-of-range maximum."""
        sh = (128, w)
        f32, u32 = nl.float32, nl.uint32
        K = packed.shape[0]
        P = [nl.load(packed[p, r0 : r0 + 128, c0 : c0 + w]) for p in range(K)]
        ebt = _o2_f(sh)
        nisa.dma_copy(dst=ebt, src=eb[0:1, c0 : c0 + w].broadcast(0, 128))
        E = _o2_f(sh)
        nisa.tensor_scalar(dst=E, data=ebt, op0=nl.add, operand0=ea_col)
        sg = _o2_sign(P[K - 1], sh)
        for p in range(K - 2, -1, -1):
            z = _o2_ts(sg, nl.equal, 0.0, sh)
            sg = _o2_tt(_o2_tt(z, _o2_sign(P[p], sh), nl.multiply, sh), sg, nl.add, sh)
        U = [_o2_tt(P[p], sg, nl.multiply, sh) for p in range(K)]
        for p in range(K - 1):
            b = _o2_ts(U[p], nl.less, 0.0, sh)
            U[p] = _o2_stt(b, nl.multiply, 2.0**24, nl.add, U[p], sh)
            U[p + 1] = _o2_tt(U[p + 1], b, nl.subtract, sh)
        ns = _o2_f(sh)
        nisa.memset(dst=ns, value=0.0)
        for _ in range(K - 1):
            z = _o2_ts(U[K - 1], nl.equal, 0.0, sh)
            nzf = _o2_ts(z, nl.multiply, -1.0, sh, op1=nl.add, operand1=1.0)
            for p in range(K - 1, 0, -1):
                U[p] = _o2_tt(
                    _o2_tt(U[p], nzf, nl.multiply, sh),
                    _o2_tt(U[p - 1], z, nl.multiply, sh),
                    nl.add,
                    sh,
                )
            U[0] = _o2_tt(U[0], nzf, nl.multiply, sh)
            ns = _o2_tt(ns, z, nl.add, sh)
        zero = _o2_f(sh)
        nisa.memset(dst=zero, value=0.0)
        A = U[K - 1]
        B = U[K - 2] if K >= 2 else zero
        C = U[K - 3] if K >= 3 else zero
        D = U[K - 4] if K >= 4 else zero
        rest = zero
        for p in range(K - 4):
            rest = _o2_tt(rest, _o2_ts(U[p], nl.greater, 0.0, sh), nl.maximum, sh)
        Ea = _o2_cvt(
            _o2_ts(
                A.view(u32), nl.right_shift, 23, sh, op1=nl.bitwise_and, operand1=0xFF, dtype=u32
            ),
            f32,
            sh,
        )
        Eac = _o2_ts(Ea, nl.maximum, 127.0, sh)
        p_sh = _o2_pow2(_o2_ts(Eac, nl.multiply, -1.0, sh, op1=nl.add, operand1=277.0), sh)
        p_inv = _o2_pow2(_o2_ts(Eac, nl.multiply, -1.0, sh, op1=nl.add, operand1=253.0), sh)
        p_lam = _o2_pow2(_o2_ts(Eac, nl.add, 1.0, sh), sh)
        fB = _o2_floor(_o2_tt(B, p_inv, nl.multiply, sh), sh)
        S = _o2_tt(_o2_tt(A, p_sh, nl.multiply, sh), fB, nl.add, sh)
        fC = _o2_floor(_o2_tt(C, p_inv, nl.multiply, sh), sh)
        mB = _o2_tt(B, _o2_tt(fB, p_lam, nl.multiply, sh), nl.subtract, sh)
        T = _o2_tt(_o2_tt(mB, p_sh, nl.multiply, sh), fC, nl.add, sh)
        fD = _o2_floor(_o2_tt(D, p_inv, nl.multiply, sh), sh)
        mC = _o2_tt(C, _o2_tt(fC, p_lam, nl.multiply, sh), nl.subtract, sh)
        W = _o2_tt(_o2_tt(mC, p_sh, nl.multiply, sh), fD, nl.add, sh)
        mD = _o2_tt(D, _o2_tt(fD, p_lam, nl.multiply, sh), nl.subtract, sh)
        sticky = _o2_tt(_o2_ts(mD, nl.greater, 0.0, sh), rest, nl.maximum, sh)
        w1 = _o2_ts(W, nl.multiply, 2.0**-19, sh)
        r1 = _o2_ts(w1, nl.add, _OZ2_MAGIC, sh)  # two instructions: both round
        r = _o2_ts(r1, nl.subtract, _OZ2_MAGIC, sh)
        d = _o2_tt(w1, r, nl.subtract, sh)
        q = _o2_tt(_o2_tt(_o2_ts(d, nl.equal, 0.5, sh), sticky, nl.multiply, sh), r, nl.add, sh)
        c1 = _o2_ts(q, nl.equal, 32.0, sh)
        q = _o2_stt(c1, nl.multiply, -32.0, nl.add, q, sh)
        T = _o2_tt(T, c1, nl.add, sh)
        c2 = _o2_ts(T, nl.equal, 2.0**24, sh)
        T = _o2_stt(c2, nl.multiply, -(2.0**24), nl.add, T, sh)
        S = _o2_tt(S, c2, nl.add, sh)
        c3 = _o2_ts(S, nl.equal, 2.0**24, sh)
        S = _o2_stt(c3, nl.multiply, -(2.0**23), nl.add, S, sh)
        nz = _o2_ts(A, nl.not_equal, 0.0, sh)
        E64 = _o2_tt(
            _o2_stt(ns, nl.multiply, -24.0, nl.add, _o2_tt(E, Eac, nl.add, sh), sh),
            c3,
            nl.add,
            sh,
        )
        b1 = _o2_tt(
            _o2_ts(E64, nl.less, 1.0, sh), _o2_ts(E64, nl.greater, 2046.0, sh), nl.maximum, sh
        )
        bad = _o2_tt(b1, nz, nl.multiply, sh)
        bm = _o2_f((128, 1))
        nisa.tensor_reduce(dst=bm, op=nl.maximum, data=bad, axis=1, keepdims=True)
        nisa.tensor_tensor(dst=flag, data1=flag, data2=bm, op=nl.maximum)
        E64c = _o2_ts(
            _o2_tt(E64, nz, nl.multiply, sh), nl.maximum, 0.0, sh, op1=nl.minimum, operand1=2047.0
        )
        Su = _o2_cvt(_o2_tt(_o2_ts(S, nl.subtract, 2.0**23, sh), nz, nl.multiply, sh), u32, sh)
        sgn = _o2_cvt(_o2_ts(sg, nl.less, 0.0, sh), u32, sh)
        hw = _o2_ts(sgn, nl.left_shift, 31, sh, dtype=u32)
        Eu = _o2_ts(_o2_cvt(E64c, u32, sh), nl.left_shift, 20, sh, dtype=u32)
        nisa.tensor_tensor(dst=hw, data1=hw, data2=Eu, op=nl.bitwise_or)
        s3 = _o2_ts(Su, nl.right_shift, 3, sh, dtype=u32)
        nisa.tensor_tensor(dst=hw, data1=hw, data2=s3, op=nl.bitwise_or)
        lw = _o2_ts(Su, nl.bitwise_and, 7, sh, op1=nl.left_shift, operand1=29, dtype=u32)
        Tu = _o2_ts(_o2_cvt(T, u32, sh), nl.left_shift, 5, sh, dtype=u32)
        nisa.tensor_tensor(dst=lw, data1=lw, data2=Tu, op=nl.bitwise_or)
        qu = _o2_cvt(q, u32, sh)
        nisa.tensor_tensor(dst=lw, data1=lw, data2=qu, op=nl.bitwise_or)
        Tw = _o2_u((128, 2 * w))
        nisa.tensor_copy(dst=Tw[:, 0 : 2 * w : 2], src=lw)
        nisa.tensor_copy(dst=Tw[:, 1 : 2 * w : 2], src=hw)
        nl.store(out[r0 : r0 + 128, 2 * c0 : 2 * (c0 + w)], value=Tw.view(nl.int32))

    def _oz2_comb_rows(packed, ea, eb, out, b):
        """Row tile b: all column tiles, then the tile's flags into
        out[R : R + 128, b]."""
        R = packed.shape[1]
        N = packed.shape[2]
        r0 = b * 128
        CF = N // _OZ2_BW
        tail = N - CF * _OZ2_BW
        ea_col = nl.load(ea[r0 : r0 + 128, 0:1])
        flag = _o2_f((128, 1))
        nisa.memset(dst=flag, value=0.0)
        if CF:
            for c in nl.sequential_range(CF):
                _oz2_comb_tile(packed, ea_col, eb, out, flag, r0, c * _OZ2_BW, _OZ2_BW)
        if tail:
            _oz2_comb_tile(packed, ea_col, eb, out, flag, r0, CF * _OZ2_BW, tail)
        fi = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=fi, src=flag)
        nl.store(out[R : R + 128, b : b + 1], value=fi)

    @nki.jit
    def _ozaki2_combine_kernel(packed, ea, eb):
        """float64 bits of RN64(sum_p P_p 2^(24 p)) 2^(ea_i + eb_j - offset)
        (`ozaki2.packed_to_fp64_bits`): `packed` (K, R, N) fp32, the packed
        CRT output (24-bit pairs), R a multiple of 128 with R / 128 <= 2 N;
        `ea` (R, 1) and `eb` (1, N) fp32 integers (their sum is the kernel's
        Eab, `ozaki2.combine_offset` included). Returns (R + 128, 2 N) int32:
        the float64 words of rows 0 .. R - 1, and out[R + p, b] = 1 where
        row 128 b + p has an element outside the normal float64 range.
        Elementwise; at LNC=2 the programs take halves of the row tiles."""
        R = packed.shape[1]
        N = packed.shape[2]
        out = nl.ndarray((R + 128, 2 * N), dtype=nl.int32, buffer=nl.shared_hbm)
        RT = R // 128
        zt = nl.ndarray((128, 2 * N), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=zt, value=0)
        nl.store(out[R : R + 128, 0 : 2 * N], value=zt)
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        per = (RT + n_prog - 1) // n_prog
        if n_prog > 1:
            b0 = nl.program_id(0) * (RT - per)
            for i in nl.affine_range(per):
                _oz2_comb_rows(packed, ea, eb, out, b0 + i)
        else:
            for i in nl.affine_range(RT):
                _oz2_comb_rows(packed, ea, eb, out, i)
        return out

    SPLIT_KERNELS = {
        "b7": _ozaki2_split_b7_kernel,
        "b8": _ozaki2_split_b8_kernel,
        "b9": _ozaki2_split_b9_kernel,
    }
    SPLIT_COL_KERNELS = {
        "b7": _ozaki2_split_col_b7_kernel,
        "b8": _ozaki2_split_col_b8_kernel,
        "b9": _ozaki2_split_col_b9_kernel,
    }

    FUSED_KERNELS = {
        "b7": _ozaki2_fused_b7_kernel,
        "b8": _ozaki2_fused_b8_kernel,
        "b9": _ozaki2_fused_b9_kernel,
    }
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
    FUSED_KERNELS = {}
    SPLIT_KERNELS = {}
    SPLIT_COL_KERNELS = {}


# ── host side: launches and the gemm_fp64(scheme="ozaki2") route ─────────────

import os  # noqa: E402

# The device split of the operands (#132): TRNBLAS_OZAKI2_DEVICE_SPLIT=0 keeps
# the host split (the same residues and exponents).
_OZ2_DEVICE_SPLIT = os.environ.get("TRNBLAS_OZAKI2_DEVICE_SPLIT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _run(kernel, op: str, *xs):
    """One kernel launch: under the simulator on host arrays (result on the
    host, bf16 results as torch bf16), else on device tensors."""
    import numpy as np
    import torch

    from . import dispatch as D

    if D._use_simulator():
        arrs = [x.detach().cpu().contiguous().numpy() if hasattr(x, "numpy") else x for x in xs]
        out = np.ascontiguousarray(nki.simulate(kernel)(*arrs))
        if out.dtype.itemsize == 2 and out.dtype != np.int16:  # ml_dtypes bfloat16
            return torch.from_numpy(out.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(out)
    return D._call_kernel(kernel, op, *[x if x.device.type != "cpu" else D._upload(x) for x in xs])


def _launcher(fam: str, crt_mode: str, keep_on_device: bool = False):
    """`ozaki2.run`'s launch: the residue-GEMM kernel on K-major device
    residues (uploaded once per call when they fit, as Ozaki I's slices),
    then the device CRT (crt_mode "device") or the residues themselves
    ("host"); "fused" runs both in one kernel (same bits as "device"). The
    result comes down to the host, or stays on the device with
    `keep_on_device` (the device combine). Under the simulator, the kernels
    on host blocks."""
    import numpy as np
    import torch

    from . import dispatch as D

    gemm_op, crt_op, fused_op = (f"ozaki2_{k}_{fam}" for k in ("gemm", "crt", "fused"))
    cache: dict = {}

    def launch(ra, rb, m, kk, nn):
        if D._use_simulator():
            at, bt = _o2.host_blocks(ra, rb, m, kk, nn)
            D._check_kernel_operands(gemm_op, at, bt)
            if crt_mode == "fused":
                return D._simulate_bf16(FUSED_KERNELS[fam], at, bt)
            res = D._simulate_bf16(GEMM_KERNELS[fam], at, bt)
            if crt_mode == "host":
                return res
            return torch.from_numpy(np.asarray(nki.simulate(CRT_KERNELS[fam])(res.numpy())))
        D._check_kernel_operands(gemm_op, ra.d, rb.d)
        at = D._oz_device_block(ra, cache, "a", kk, m)
        bt = D._oz_device_block(rb, cache, "b", kk, nn)
        # .get: without nki (stubbed native ops in tests) the tables are empty
        if crt_mode == "fused":
            out = D._call_kernel(FUSED_KERNELS.get(fam), fused_op, at, bt)
        else:
            out = D._call_kernel(GEMM_KERNELS.get(fam), gemm_op, at, bt)
            if crt_mode == "device":
                out = D._call_kernel(CRT_KERNELS.get(fam), crt_op, out)
        return out if keep_on_device else D._to_host(out, "cpu")

    return launch


def combine_packed_device(packed, eA, eB, c, rows: int, N: int, shift: int, out=None):
    """`ozaki2.combine_packed` of one band with `_ozaki2_combine_kernel`:
    float64 (rows, N) on the host, bit for bit the host combine (flagged
    rows are recombined on the host). `packed` (keep, Rp, >= N) fp32 on the
    device (on the host under the simulator), Rp a multiple of 128; `out`
    an optional host int32 (Rp + 128, 2 N) buffer to download into."""
    import numpy as np
    import torch

    from . import dispatch as D

    Rp = packed.shape[1]
    K = packed.shape[0]
    E_a = shift - np.asarray(eA[:rows], np.int64) + _o2.combine_offset(K)
    E_b = -np.asarray(eB[:N], np.int64)
    big = max(np.abs(E_a).max(initial=0), np.abs(E_b).max(initial=0)) >= 1 << 22
    if 2 * c.beta != _o2.PACK_BITS or big or Rp % 128 or Rp // 128 > 2 * N:
        host = packed.detach().to("cpu").numpy()
        return torch.from_numpy(_o2.combine_packed(host, eA, eB, c, rows, N, shift))
    P = packed if packed.shape[2] == N else packed[:, :, :N]
    P = P.contiguous()
    ea = np.zeros((Rp, 1), np.float32)
    ea[:rows, 0] = E_a
    eb = np.ascontiguousarray(E_b.astype(np.float32)[None, :])
    dev = P.device
    ea_t, eb_t = torch.from_numpy(ea).to(dev), torch.from_numpy(eb).to(dev)
    res = _run(_ozaki2_combine_kernel if _HAS_NKI else None, "ozaki2_combine", P, ea_t, eb_t)
    host = (res.to("cpu") if not D._use_simulator() else res) if out is None else out.copy_(res)
    o = host.numpy()
    C = o[:Rp].view(np.float64)[:rows]
    flags = o[Rp : Rp + 128, : Rp // 128].T.reshape(-1)[:rows]
    bad = np.nonzero(flags)[0]
    if len(bad):
        sub = packed.detach().to("cpu").numpy()[:, bad]
        C[bad] = _o2.combine_packed(sub, np.asarray(eA)[bad], eB, c, len(bad), N, shift)
    return torch.from_numpy(C)


def _oz2_device_split(X, n: int, fam: str, per: str):
    """`ozaki2.split` of a contiguous host fp64 X (R, C) on the device
    (#132), per row or per column: (residues (n, Rp, Cp) bf16 in X's layout
    (a device tensor; a host one under the simulator), mu). Steps: the raw
    bits up once; the exponent maxima (`_ozaki_emax*_kernel`, #123); the exact
    norm sums (`_ozaki2_norm*_kernel`), finished on the host; the scale
    exponents (`ozaki2.scale_from_sums`, the host split's own rule); the
    residues (`_ozaki2_split*_kernel`). None when X must take the host split
    (size, a subnormal row / column maximum, more than 12 digits); raises
    `dispatch._OzNonFinite` for inf / nan."""
    import numpy as np
    import torch

    from .. import ozaki as _oz
    from . import dispatch as D

    f = _o2.family(fam)
    R, C = X.shape
    Rp, Cp = _oz._round_up(R, 128), _oz._round_up(C, 128)
    if 8 * R * C >= D._OZ_RESIDENT_BYTES or 2 * n * Rp * Cp >= D._OZ_RESIDENT_BYTES:
        return None
    rows = per == "row"
    sim = D._use_simulator()
    bits = X.view(torch.int32)  # (R, 2C): the raw bits, no copy
    xw = bits if sim else D._to_device(bits)[0][0]
    emax_k = D._ozaki_emax_kernel if rows else D._ozaki_emax_col_kernel
    emax = _run(emax_k, "ozaki_emax" if rows else "ozaki_emax_col", xw)
    emax = emax.to("cpu").reshape(-1).numpy() if not sim else emax.reshape(-1).numpy()
    if (emax >= 2047).any():
        raise D._OzNonFinite()
    zero = np.nonzero(emax == 0)[0]
    if len(zero):
        idx = torch.from_numpy(zero)
        if bool((X[idx] if rows else X[:, idx]).ne(0).any()):
            return None  # a subnormal maximum: the host's
    nz = emax > 0
    e = np.where(nz, emax.astype(np.int64) - 1022, 0)
    cnt, npad = (R, Rp) if rows else (C, Cp)
    rc = np.full(npad, 1003.0, dtype=np.float32)
    rc[:cnt] = (1003 + e).astype(np.float32)
    rcn = torch.from_numpy(rc[:, None] if rows else rc[None, :].copy())
    norm_k = None
    if _HAS_NKI:
        norm_k = _ozaki2_norm_kernel if rows else _ozaki2_norm_col_kernel
    sums = _run(norm_k, "ozaki2_norm" if rows else "ozaki2_norm_col", xw, rcn)
    sums = sums.to("cpu").numpy().astype(np.int64)
    if rows:
        L = sums.reshape(R, 4, -1).sum(axis=2)
    else:
        L = sums.reshape(4, -1, C).sum(axis=1).T
    S = sum(L[:, i] << (_o2.NORM_LIMB * i) for i in range(4))
    mu = _o2.scale_from_sums(S, e, nz, C if rows else R, _o2.bound(n, f))
    k = _o2.digits_for(mu, e, nz)
    if k > _oz.MAX_SLICES:
        return None
    ep = (_oz.SLICE_BITS - 1) + _oz.SLICE_BITS * (k - 1) - mu
    rcs = torch.full((npad, k), 1003.0, dtype=torch.float32)
    rcs[:cnt, 0] = torch.from_numpy((1003 + ep).astype(np.float32))
    if not rows:
        rcs = rcs.mT.contiguous()  # (k, Cp): row 0 is broadcast down the partitions
    mods = torch.tensor([f.moduli[:n]], dtype=torch.float32)
    kern = (SPLIT_KERNELS if rows else SPLIT_COL_KERNELS).get(f.name)
    d = _run(kern, f"ozaki2_split{'' if rows else '_col'}_{f.name}", xw, rcs, mods)
    return d, mu


def _split_fn(device_split: bool):
    """`ozaki2.run`'s split: each operand on the device in its own layout
    (per row for a row-major A or B^T, per column for A^T or a row-major B),
    else the host split (the same residues) after a finiteness check."""
    import torch

    from . import dispatch as D

    def split_ab(A, B, n, f, empty):
        if not device_split:
            return _o2.split_ab(A, B, n, f, empty)

        def one(X, Xh, per, kmajor, host_per):
            got = _oz2_device_split(X, n, f.name, per)
            if got is None:
                if not bool(torch.isfinite(X).all()):
                    raise D._OzNonFinite()
                return _o2.split(Xh, n, f, per=host_per, empty=empty)
            return _o2.Residues(got[0], got[1], kmajor)

        if A.mT.is_contiguous() and not A.is_contiguous():
            ra = one(A.mT, A, "col", True, "row")
        else:
            ra = one(A.contiguous(), A, "row", False, "row")
        if B.mT.is_contiguous() and not B.is_contiguous():
            rb = one(B.mT, B, "row", False, "col")
        else:
            rb = one(B.contiguous(), B, "col", True, "col")
        return ra, rb

    return split_ab


def gemm(
    A,
    B,
    n: int = _o2.DEFAULT_MODULI,
    fam: str = _o2.DEFAULT_FAMILY,
    crt_mode="fused",
    device_split=None,
    device_combine=None,
):
    """fp64 (M, N) = A @ B by Ozaki II for 2-D host fp64 A, B: the kernels on
    the native path (or the simulator), else host fp64 matmul (the routes of
    Ozaki I, `dispatch._ozaki_route`). Non-finite operands take host fp64.

    All on the device by default (#132): the operands are split on the
    device from their raw bits (`_oz2_device_split`; `device_split`,
    default TRNBLAS_OZAKI2_DEVICE_SPLIT), the residue GEMMs and the CRT run
    in one kernel (crt_mode="fused"), and the combine runs on the device
    (`combine_packed_device`; `device_combine`, default
    TRNBLAS_OZAKI_DEVICE_COMBINE, for 24-bit packed values: every b7
    count, b8 up to 16 moduli, b9 up to 8). Every route gives the same
    bits as `ozaki2.reference_gemm`."""
    import torch

    from .. import ozaki as _oz
    from . import dispatch as D

    f = _o2.family(fam)
    n = _o2.check_moduli(n, f)
    if not f.kernel:
        raise ValueError(f"family {f.name} is analysis-only (no kernel)")
    if D._ozaki_route() == "host":
        return torch.matmul(A, B)
    dsplit = _OZ2_DEVICE_SPLIT if device_split is None else device_split
    if not dsplit and not (torch.isfinite(A).all() and torch.isfinite(B).all()):
        return torch.matmul(A, B)
    c = _o2.crt(f, n)
    dcomb = D._OZ_DEVICE_COMBINE if device_combine is None else device_combine
    dcomb = dcomb and crt_mode != "host" and 2 * c.beta == _o2.PACK_BITS
    shift = 2 * c.beta * c.j0

    def comb(P, ea, eb, _n, rows, N, out=None):
        try:
            return combine_packed_device(P, ea, eb, c, rows, N, shift, out=out)
        except Exception as exc:  # the band on the host (same bits)
            if D._REQUIRE_NKI:
                raise
            D._warn_fallback(exc, "Ozaki II device combine failed — combining on the host")
            host = P.detach().to("cpu").numpy()
            return torch.from_numpy(_o2.combine_packed(host, ea, eb, c, rows, N, shift))

    try:
        return _o2.run(
            A,
            B,
            n,
            _launcher(f.name, crt_mode, keep_on_device=dcomb),
            f,
            crt_mode,
            empty=D._ozaki_host_empty,
            max_m=_oz.LAUNCH_M,
            max_n=_oz.LAUNCH_N,
            split_fn=_split_fn(dsplit),
            device_combine=comb if dcomb else None,
        )
    except D._OzNonFinite:
        return torch.matmul(A, B)
    except Exception as exc:
        if D._REQUIRE_NKI:
            raise
        D._warn_fallback(exc, "Ozaki II fp64 GEMM kernel failed — falling back to host fp64")
        return torch.matmul(A, B)
