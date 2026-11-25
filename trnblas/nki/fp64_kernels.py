"""
NKI kernels of `trnblas.fp64` (trnsci/trnsci#169): float-float reductions
and the device probes.

Every kernel issues the operations of the CPU reference (`trnblas/fp64.py`)
one fp32 instruction each, in the same order, so a device result equals
`fp64.reference_partials` bit for bit when the engines round fp32 add,
subtract and multiply to nearest even and nothing is subnormal. Rounding-
sensitive steps are never fused into a two-op instruction: whether such an
instruction rounds its intermediate is what `_fp64_probe_fma_kernel` measures.

Reduction layout (`fp64.plan`): the operands arrive as (128, F) float32 rows,
F = NCH * tpc * W. Chunk c is columns [c tpc W, (c + 1) tpc W); each of its
(128, W) lanes accumulates tpc column tiles in order, then the W lanes fold
in a halving tree. The output is (128, 2 NCH) float32, (hi, lo) of chunk c in
columns 2c and 2c + 1, or (128, 3 NCH) for the norms (the chunk's scale
exponent in column 3c + 2). Gridless: at LNC=2 program p takes chunks
2p, 2p + 1, at LNC=1 one program takes all four, with identical operations,
so the partials are the same bits at any LNC.

Engines: the reduction kernels leave engine selection to the compiler (VectorE
for tensor_tensor in practice). The probes pin VectorE, GpSimd and ScalarE
explicitly.
"""

from __future__ import annotations

import os

import numpy as np
import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

_NCH = 4  # = trnblas.fp64.NCH (tests pin the equality)
_W_MAX = 512  # = trnblas.fp64.W_MAX
_SPLITTER = 4097.0  # = trnblas.fp64.SPLITTER

# kernel name -> operand dtypes (the boundary check, as dispatch._KERNEL_DTYPES)
KERNEL_DTYPES = {}

# (kind, arm) -> kernel attribute name; filled below under HAS_NKI
REDUCE_KERNELS: dict[tuple[str, str], str] = {
    ("sum32", "plain"): "_fp64_sum32_plain_kernel",
    ("sum32", "comp"): "_fp64_sum32_comp_kernel",
    ("sum32", "sloppy"): "_fp64_sum32_sloppy_kernel",
    ("sum32", "accurate"): "_fp64_sum32_accurate_kernel",
    ("dot32", "plain"): "_fp64_dot32_plain_kernel",
    ("dot32", "comp"): "_fp64_dot32_comp_kernel",
    ("dot32", "sloppy"): "_fp64_dot32_sloppy_kernel",
    ("dot32", "accurate"): "_fp64_dot32_accurate_kernel",
    ("sumff", "accurate"): "_fp64_sumff_accurate_kernel",
    ("dotff", "accurate"): "_fp64_dotff_accurate_kernel",
    ("dotff", "sloppy"): "_fp64_dotff_sloppy_kernel",
    ("nrm32", "accurate"): "_fp64_nrm32_accurate_kernel",
    ("nrmff", "accurate"): "_fp64_nrmff_accurate_kernel",
    ("sum32", "exact"): "_fp64_sum32_exact_kernel",
    ("dot32", "exact"): "_fp64_dot32_exact_kernel",
}
_EXACT_L = 6  # = trnblas.fp64.EXACT_LIMBS
for _name in REDUCE_KERNELS.values():
    KERNEL_DTYPES[_name] = frozenset({torch.float32})
KERNEL_DTYPES["_fp64_probe_alu_kernel"] = frozenset({torch.float32})
KERNEL_DTYPES["_fp64_probe_fma_kernel"] = frozenset({torch.float32})
KERNEL_DTYPES["_fp64_probe_eft_kernel"] = frozenset({torch.float32})
KERNEL_DTYPES["_fp64_probe_pe_kernel"] = frozenset({torch.float32})
KERNEL_DTYPES["_fp64_ff_bits_kernel"] = frozenset({torch.float32})

_FB_W = 512  # columns per tile of the fp64-bits combine


def _fv_plan(F):
    """(W, tpc) of the padded column count F (`fp64.plan_from_cols`)."""
    tpc = (F + _NCH * _W_MAX - 1) // (_NCH * _W_MAX)
    return F // (_NCH * tpc), tpc


if HAS_NKI:
    # ---------------------------------------------------------------------
    # Elementwise building blocks. Each helper allocates its results and
    # issues one instruction per fp32 operation of the reference function
    # of the same name in trnblas/fp64.py.
    # ---------------------------------------------------------------------

    def _fv_t(shape):
        return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)

    def _fv_op(a, b, op, shape):
        r = _fv_t(shape)
        nisa.tensor_tensor(dst=r, data1=a, data2=b, op=op)
        return r

    def _fv_two_sum(a, b, shape):
        s = _fv_op(a, b, nl.add, shape)
        bb = _fv_op(s, a, nl.subtract, shape)
        t = _fv_op(s, bb, nl.subtract, shape)
        e1 = _fv_op(a, t, nl.subtract, shape)
        e2 = _fv_op(b, bb, nl.subtract, shape)
        return s, _fv_op(e1, e2, nl.add, shape)

    def _fv_fast_two_sum_into(a, b, dh, dl):
        """dh, dl = FastTwoSum(a, b); a and b are not read after dh is written."""
        nisa.tensor_tensor(dst=dh, data1=a, data2=b, op=nl.add)
        z = _fv_op(dh, a, nl.subtract, dh.shape)
        nisa.tensor_tensor(dst=dl, data1=b, data2=z, op=nl.subtract)

    def _fv_fast_two_sum(a, b, shape):
        s = _fv_t(shape)
        e = _fv_t(shape)
        _fv_fast_two_sum_into(a, b, s, e)
        return s, e

    def _fv_split(a, shape):
        c = _fv_t(shape)
        nisa.tensor_scalar(dst=c, data=a, op0=nl.multiply, operand0=_SPLITTER)
        d = _fv_op(c, a, nl.subtract, shape)
        h = _fv_op(c, d, nl.subtract, shape)
        return h, _fv_op(a, h, nl.subtract, shape)

    def _fv_two_prod(a, b, shape):
        p = _fv_op(a, b, nl.multiply, shape)
        ah, al = _fv_split(a, shape)
        bh, bl = _fv_split(b, shape)
        t = _fv_op(p, _fv_op(ah, bh, nl.multiply, shape), nl.subtract, shape)
        t = _fv_op(t, _fv_op(al, bh, nl.multiply, shape), nl.subtract, shape)
        t = _fv_op(t, _fv_op(ah, bl, nl.multiply, shape), nl.subtract, shape)
        return p, _fv_op(_fv_op(al, bl, nl.multiply, shape), t, nl.subtract, shape)

    def _fv_add_into(xh, xl, yh, yl, dh, dl, shape):
        """dh, dl = AccurateDWPlusDW((xh, xl), (yh, yl))."""
        sh, sl = _fv_two_sum(xh, yh, shape)
        th, tl = _fv_two_sum(xl, yl, shape)
        sl2 = _fv_op(sl, th, nl.add, shape)
        vh, vl = _fv_fast_two_sum(sh, sl2, shape)
        vl2 = _fv_op(vl, tl, nl.add, shape)
        _fv_fast_two_sum_into(vh, vl2, dh, dl)

    def _fv_add_sloppy_into(xh, xl, yh, yl, dh, dl, shape):
        """dh, dl = SloppyDWPlusDW((xh, xl), (yh, yl))."""
        sh, sl = _fv_two_sum(xh, yh, shape)
        v = _fv_op(xl, yl, nl.add, shape)
        w = _fv_op(sl, v, nl.add, shape)
        _fv_fast_two_sum_into(sh, w, dh, dl)

    def _fv_add_fp_into(xh, xl, b, dh, dl, shape):
        """dh, dl = DWPlusFP((xh, xl), b)."""
        sh, sl = _fv_two_sum(xh, b, shape)
        v = _fv_op(xl, sl, nl.add, shape)
        _fv_fast_two_sum_into(sh, v, dh, dl)

    def _fv_mul(xh, xl, yh, yl, shape):
        """DWTimesDW (fp64.ff_mul)."""
        ch, cl1 = _fv_two_prod(xh, yh, shape)
        tl1 = _fv_op(xh, yl, nl.multiply, shape)
        tl2 = _fv_op(xl, yh, nl.multiply, shape)
        cl2 = _fv_op(tl1, tl2, nl.add, shape)
        cl3 = _fv_op(cl1, cl2, nl.add, shape)
        return _fv_fast_two_sum(ch, cl3, shape)

    # ---------------------------------------------------------------------
    # Reductions
    # ---------------------------------------------------------------------

    def _fv_term(kind, mode, tiles, scale, shape):
        """(hi, lo) of the element terms, lo None for an fp32 term
        (fp64._term)."""
        if kind == "sum32":
            return tiles[0], None
        if kind == "sumff":
            return tiles[0], tiles[1]
        if kind == "dot32":
            if mode == "plain":
                return _fv_op(tiles[0], tiles[1], nl.multiply, shape), None
            return _fv_two_prod(tiles[0], tiles[1], shape)
        if kind == "dotff":
            return _fv_mul(tiles[0], tiles[1], tiles[2], tiles[3], shape)
        a = _fv_t(shape)
        nisa.tensor_scalar(dst=a, data=tiles[0], op0=nl.multiply, operand0=scale)
        if kind == "nrm32":
            if mode == "plain":
                return _fv_op(a, a, nl.multiply, shape), None
            return _fv_two_prod(a, a, shape)
        # nrmff
        b = _fv_t(shape)
        nisa.tensor_scalar(dst=b, data=tiles[1], op0=nl.multiply, operand0=scale)
        p, e = _fv_two_prod(a, a, shape)
        t = _fv_op(a, b, nl.multiply, shape)
        t2 = _fv_op(t, t, nl.add, shape)
        e2 = _fv_op(e, t2, nl.add, shape)
        return _fv_fast_two_sum(p, e2, shape)

    def _fv_accumulate(Ah, Al, th, tl, mode, shape):
        """One lane step of arm `mode`, in place on (Ah, Al) (fp64._accumulate)."""
        if mode == "plain":
            nisa.tensor_tensor(dst=Ah, data1=Ah, data2=th, op=nl.add)
            return
        if mode == "comp":
            s, q = _fv_two_sum(Ah, th, shape)
            if tl is None:
                nisa.tensor_tensor(dst=Al, data1=Al, data2=q, op=nl.add)
            else:
                u = _fv_op(q, tl, nl.add, shape)
                nisa.tensor_tensor(dst=Al, data1=Al, data2=u, op=nl.add)
            nisa.tensor_copy(dst=Ah, src=s)
            return
        if tl is None:
            _fv_add_fp_into(Ah, Al, th, Ah, Al, shape)
        elif mode == "sloppy":
            _fv_add_sloppy_into(Ah, Al, th, tl, Ah, Al, shape)
        else:
            _fv_add_into(Ah, Al, th, tl, Ah, Al, shape)

    def _fv_fold(Ah, Al, W, mode):
        """Halving tree over the W lanes (fp64._fold), in place; the result
        is in column 0."""
        w = W
        while w > 1:
            k = w // 2
            sh = (128, k)
            ah, al = Ah[:, 0:k], Al[:, 0:k]
            bh, bl = Ah[:, w - k : w], Al[:, w - k : w]
            if mode == "plain":
                nisa.tensor_tensor(dst=ah, data1=ah, data2=bh, op=nl.add)
            elif mode == "comp":
                s, q = _fv_two_sum(ah, bh, sh)
                nisa.tensor_tensor(dst=al, data1=al, data2=bl, op=nl.add)
                nisa.tensor_tensor(dst=al, data1=al, data2=q, op=nl.add)
                nisa.tensor_copy(dst=ah, src=s)
            elif mode == "sloppy":
                _fv_add_sloppy_into(ah, al, bh, bl, ah, al, sh)
            else:
                _fv_add_into(ah, al, bh, bl, ah, al, sh)
            w -= k

    def _fv_scale(x, c0, W, tpc, out, col):
        """Power-of-two scale of one chunk per partition (fp64._nrm_scale):
        s = 2^(127 - Ec), Ec = min(E(max |x|), 253); stores k = Ec - 127 in
        out[:, col]. Returns s as a (128, 1) float32 tile."""
        M = _fv_t((128, W))
        nisa.memset(dst=M, value=0.0)
        for t in nl.sequential_range(tpc):
            xt = nl.load(x[0:128, c0 + t * W : c0 + (t + 1) * W])
            a = nl.ndarray((128, W), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=a, data=xt.view(nl.uint32), op0=nl.bitwise_and, operand0=0x7FFFFFFF
            )
            nisa.tensor_tensor(dst=M, data1=M, data2=a.view(nl.float32), op=nl.maximum)
        m = _fv_t((128, 1))
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=M, axis=1, keepdims=True)
        E = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=E, data=m.view(nl.uint32), op0=nl.right_shift, operand0=23)
        Ef = _fv_t((128, 1))
        nisa.tensor_copy(dst=Ef, src=E)
        Ec = _fv_t((128, 1))
        nisa.tensor_scalar(dst=Ec, data=Ef, op0=nl.minimum, operand0=253.0)
        sb = _fv_t((128, 1))  # 254 - Ec: a small integer, exact
        nisa.tensor_scalar(
            dst=sb, data=Ec, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=254.0
        )
        su = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=su, src=sb)
        s = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s, data=su, op0=nl.left_shift, operand0=23)
        k = _fv_t((128, 1))
        nisa.tensor_scalar(dst=k, data=Ec, op0=nl.subtract, operand0=127.0)
        nl.store(out[0:128, col : col + 1], value=k)
        return s.view(nl.float32)

    def _fv_chunk(ins, out, c, W, tpc, kind, mode):
        ncol = 3 if kind[:3] == "nrm" else 2
        c0 = c * (tpc * W)
        shape = (128, W)
        scale = None
        if ncol == 3:
            scale = _fv_scale(ins[0], c0, W, tpc, out, ncol * c + 2)
        Ah = _fv_t(shape)
        Al = _fv_t(shape)
        nisa.memset(dst=Ah, value=0.0)
        nisa.memset(dst=Al, value=0.0)
        for t in nl.sequential_range(tpc):
            tiles = [nl.load(v[0:128, c0 + t * W : c0 + (t + 1) * W]) for v in ins]
            th, tl = _fv_term(kind, mode, tiles, scale, shape)
            _fv_accumulate(Ah, Al, th, tl, mode, shape)
        _fv_fold(Ah, Al, W, mode)
        nl.store(out[0:128, ncol * c : ncol * c + 1], value=Ah[:, 0:1])
        nl.store(out[0:128, ncol * c + 1 : ncol * c + 2], value=Al[:, 0:1])

    def _fv_reduce(ins, kind, mode):
        F = ins[0].shape[1]
        W, tpc = _fv_plan(F)
        ncol = 3 if kind[:3] == "nrm" else 2
        out = nl.ndarray((128, ncol * _NCH), dtype=nl.float32, buffer=nl.shared_hbm)
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        cpp = _NCH // n_prog
        if n_prog > 1:
            for j in nl.affine_range(cpp):
                _fv_chunk(ins, out, nl.program_id(0) * cpp + j, W, tpc, kind, mode)
        else:
            for j in nl.affine_range(_NCH):
                _fv_chunk(ins, out, j, W, tpc, kind, mode)
        return out

    # ---------------------------------------------------------------------
    # The exact arm (fp64.reference_exact_partials): 21-bit digits of each
    # term in a per-group fixed-point window, int32 lanes on GpSimd.
    # ---------------------------------------------------------------------

    def _fx_i(shape):
        return nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)

    def _fx_pow2(field, shape):
        """2^(field - 127) as float32 from an integer-valued float32 field."""
        u = nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=u, src=field)
        b = nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=b, data=u, op0=nl.left_shift, operand0=23)
        return b.view(nl.float32)

    def _fx_digits(v, sa, sb, acc, sticky, shape):
        """Add the digits of terms v into the limb lanes acc[0 .. L - 1]."""
        u32 = nl.uint32
        av = nl.ndarray(shape, dtype=u32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=av, data=v.view(u32), op0=nl.bitwise_and, operand0=0x7FFFFFFF)
        a = av.view(nl.float32)
        neg = _fv_t(shape)
        nisa.tensor_scalar(dst=neg, data=v, op0=nl.less, operand0=0.0)
        sg = _fv_t(shape)
        nisa.tensor_scalar(
            dst=sg, data=neg, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0
        )
        r0 = _fv_t(shape)
        nisa.tensor_scalar(dst=r0, data=a, op0=nl.multiply, operand0=sa)
        r = _fv_t(shape)
        nisa.tensor_scalar(dst=r, data=r0, op0=nl.multiply, operand0=sb)
        tiny = _fv_t(shape)
        nisa.tensor_scalar(dst=tiny, data=r, op0=nl.less, operand0=2.0**-100)
        nzv = _fv_t(shape)
        nisa.tensor_scalar(dst=nzv, data=a, op0=nl.not_equal, operand0=0.0)
        lost = _fv_op(tiny, nzv, nl.multiply, shape)
        nisa.tensor_tensor(dst=sticky, data1=sticky, data2=lost, op=nl.maximum)
        for j in range(_EXACT_L):
            c = nl.ndarray(shape, dtype=u32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=c, src=r)
            d = _fv_t(shape)
            nisa.scalar_tensor_tensor(
                dst=d, data=c, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=r
            )
            q = _fv_t(shape)  # floor(r) = c - [d < 0]
            nisa.scalar_tensor_tensor(
                dst=q,
                data=d,
                op0=nl.less,
                operand0=0.0,
                op1=nl.subtract,
                operand1=c,
                reverse1=True,
            )
            b = _fv_t(shape)  # frac(r) = d + [d < 0]
            nisa.scalar_tensor_tensor(
                dst=b, data=d, op0=nl.less, operand0=0.0, op1=nl.add, operand1=d
            )
            r = _fv_t(shape)
            nisa.tensor_scalar(dst=r, data=b, op0=nl.multiply, operand0=2.0**21)
            dq = _fv_op(q, sg, nl.multiply, shape)
            di = _fx_i(shape)
            nisa.tensor_copy(dst=di, src=dq)
            nisa.tensor_tensor(
                dst=acc[j], data1=acc[j], data2=di, op=nl.add, engine=nisa.gpsimd_engine
            )
        below = _fv_t(shape)
        nisa.tensor_scalar(dst=below, data=r, op0=nl.not_equal, operand0=0.0)
        nisa.tensor_tensor(dst=sticky, data1=sticky, data2=below, op=nl.maximum)

    def _fx_carry(acc, zero, shape):
        """Digits 0 .. L - 1 into [0, 2^21), carries upward, the last into
        acc[L]: c = a >> 21 (arithmetic: logical shift | sign extension)."""
        u32 = nl.uint32
        for j in range(_EXACT_L - 1, -1, -1):
            au = acc[j].view(u32)
            lsh = nl.ndarray(shape, dtype=u32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=lsh, data=au, op0=nl.right_shift, operand0=21)
            sbit = nl.ndarray(shape, dtype=u32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sbit, data=au, op0=nl.right_shift, operand0=31)
            ext = _fx_i(shape)  # 0 - sign: 0 or -1
            nisa.tensor_tensor(
                dst=ext,
                data1=zero,
                data2=sbit.view(nl.int32),
                op=nl.subtract,
                engine=nisa.gpsimd_engine,
            )
            hi = nl.ndarray(shape, dtype=u32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=hi, data=ext.view(u32), op0=nl.bitwise_and, operand0=0xFFFFF800)
            cu = nl.ndarray(shape, dtype=u32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=cu, data1=lsh, data2=hi, op=nl.bitwise_or)
            nisa.tensor_scalar(dst=au, data=au, op0=nl.bitwise_and, operand0=(1 << 21) - 1)
            up = acc[j - 1] if j else acc[_EXACT_L]
            nisa.tensor_tensor(
                dst=up, data1=up, data2=cu.view(nl.int32), op=nl.add, engine=nisa.gpsimd_engine
            )

    def _fx_chunk(ins, out, c, W, tpc, kind):
        L = _EXACT_L
        ncol = L + 3
        c0 = c * (tpc * W)
        sh = (128, W)
        M = _fv_t(sh)
        nisa.memset(dst=M, value=0.0)
        for t in nl.sequential_range(tpc):
            xt = nl.load(ins[0][0:128, c0 + t * W : c0 + (t + 1) * W])
            if kind == "dot32":
                yt = nl.load(ins[1][0:128, c0 + t * W : c0 + (t + 1) * W])
                lead = _fv_op(xt, yt, nl.multiply, sh)
            else:
                lead = xt
            a = nl.ndarray(sh, dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=a, data=lead.view(nl.uint32), op0=nl.bitwise_and, operand0=0x7FFFFFFF
            )
            nisa.tensor_tensor(dst=M, data1=M, data2=a.view(nl.float32), op=nl.maximum)
        m = _fv_t((128, 1))
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=M, axis=1, keepdims=True)
        Eu = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=Eu, data=m.view(nl.uint32), op0=nl.right_shift, operand0=23)
        Eg = _fv_t((128, 1))
        nisa.tensor_copy(dst=Eg, src=Eu)
        fa = _fv_t((128, 1))  # field of 2^min(147 - Eg, 127)
        nisa.tensor_scalar(
            dst=fa, data=Eg, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=274.0
        )
        nisa.tensor_scalar(dst=fa, data=fa, op0=nl.minimum, operand0=254.0)
        fb = _fv_t((128, 1))  # field of 2^max(20 - Eg, 0)
        nisa.tensor_scalar(
            dst=fb, data=Eg, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=147.0
        )
        nisa.tensor_scalar(dst=fb, data=fb, op0=nl.maximum, operand0=127.0)
        sa = _fx_pow2(fa, (128, 1))
        sb = _fx_pow2(fb, (128, 1))
        acc = []
        for _j in range(L + 1):
            z = _fx_i(sh)
            nisa.memset(dst=z, value=0)
            acc.append(z)
        zero = _fx_i(sh)
        nisa.memset(dst=zero, value=0)
        sticky = _fv_t(sh)
        nisa.memset(dst=sticky, value=0.0)
        for t in nl.sequential_range(tpc):
            xt = nl.load(ins[0][0:128, c0 + t * W : c0 + (t + 1) * W])
            if kind == "dot32":
                yt = nl.load(ins[1][0:128, c0 + t * W : c0 + (t + 1) * W])
                p, e = _fv_two_prod(xt, yt, sh)
                _fx_digits(p, sa, sb, acc, sticky, sh)
                _fx_digits(e, sa, sb, acc, sticky, sh)
            else:
                _fx_digits(xt, sa, sb, acc, sticky, sh)
        _fx_carry(acc, zero, sh)
        w = W
        while w > 1:
            k = w // 2
            for j in range(L + 1):
                nisa.tensor_tensor(
                    dst=acc[j][:, 0:k],
                    data1=acc[j][:, 0:k],
                    data2=acc[j][:, w - k : w],
                    op=nl.add,
                    engine=nisa.gpsimd_engine,
                )
            nisa.tensor_tensor(
                dst=sticky[:, 0:k], data1=sticky[:, 0:k], data2=sticky[:, w - k : w], op=nl.maximum
            )
            w -= k
        for j in range(L + 1):
            nl.store(out[0:128, ncol * c + j : ncol * c + j + 1], value=acc[j][:, 0:1])
        Tf = _fv_t((128, 1))
        nisa.tensor_scalar(dst=Tf, data=Eg, op0=nl.subtract, operand0=126.0)
        Ti = _fx_i((128, 1))
        nisa.tensor_copy(dst=Ti, src=Tf)
        nl.store(out[0:128, ncol * c + L + 1 : ncol * c + L + 2], value=Ti)
        Si = _fx_i((128, 1))
        nisa.tensor_copy(dst=Si, src=sticky[:, 0:1])
        nl.store(out[0:128, ncol * c + L + 2 : ncol * c + L + 3], value=Si)

    def _fx_reduce(ins, kind):
        F = ins[0].shape[1]
        W, tpc = _fv_plan(F)
        out = nl.ndarray((128, (_EXACT_L + 3) * _NCH), dtype=nl.int32, buffer=nl.shared_hbm)
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        cpp = _NCH // n_prog
        if n_prog > 1:
            for j in nl.affine_range(cpp):
                _fx_chunk(ins, out, nl.program_id(0) * cpp + j, W, tpc, kind)
        else:
            for j in nl.affine_range(_NCH):
                _fx_chunk(ins, out, j, W, tpc, kind)
        return out

    @nki.jit
    def _fp64_sum32_exact_kernel(x):
        return _fx_reduce((x,), "sum32")

    @nki.jit
    def _fp64_dot32_exact_kernel(x, y):
        return _fx_reduce((x, y), "dot32")

    @nki.jit
    def _fp64_sum32_plain_kernel(x):
        return _fv_reduce((x,), "sum32", "plain")

    @nki.jit
    def _fp64_sum32_comp_kernel(x):
        return _fv_reduce((x,), "sum32", "comp")

    @nki.jit
    def _fp64_sum32_sloppy_kernel(x):
        return _fv_reduce((x,), "sum32", "sloppy")

    @nki.jit
    def _fp64_sum32_accurate_kernel(x):
        return _fv_reduce((x,), "sum32", "accurate")

    @nki.jit
    def _fp64_dot32_plain_kernel(x, y):
        return _fv_reduce((x, y), "dot32", "plain")

    @nki.jit
    def _fp64_dot32_comp_kernel(x, y):
        return _fv_reduce((x, y), "dot32", "comp")

    @nki.jit
    def _fp64_dot32_sloppy_kernel(x, y):
        return _fv_reduce((x, y), "dot32", "sloppy")

    @nki.jit
    def _fp64_dot32_accurate_kernel(x, y):
        return _fv_reduce((x, y), "dot32", "accurate")

    @nki.jit
    def _fp64_sumff_accurate_kernel(xh, xl):
        return _fv_reduce((xh, xl), "sumff", "accurate")

    @nki.jit
    def _fp64_dotff_accurate_kernel(xh, xl, yh, yl):
        return _fv_reduce((xh, xl, yh, yl), "dotff", "accurate")

    @nki.jit
    def _fp64_dotff_sloppy_kernel(xh, xl, yh, yl):
        return _fv_reduce((xh, xl, yh, yl), "dotff", "sloppy")

    @nki.jit
    def _fp64_nrm32_accurate_kernel(x):
        return _fv_reduce((x,), "nrm32", "accurate")

    @nki.jit
    def _fp64_nrmff_accurate_kernel(xh, xl):
        return _fv_reduce((xh, xl), "nrmff", "accurate")

    # ---------------------------------------------------------------------
    # Device probes (E0 of #169). Inputs a, b, c are (128, W) float32;
    # the host compares every output with IEEE fp32 round-to-nearest-even
    # results computed in numpy (`trnblas.fp64_probe`).
    # ---------------------------------------------------------------------

    @nki.jit
    def _fp64_probe_alu_kernel(a, b):
        """Single fp32 add / subtract / multiply on VectorE and GpSimd
        (tensor_tensor), the (P, 1)-operand forms (tensor_scalar, VectorE and
        GpSimd) and ScalarE (activation: a * scale, a + bias with a per-
        partition operand from column 0 of b). Output (10, 128, W):
        0-2 VectorE add, sub, mul; 3-5 GpSimd add, sub, mul;
        6-7 VectorE tensor_scalar add, mul by b[:, 0];
        8-9 ScalarE activation a + b[:, 0], a * b[:, 0]."""
        W = a.shape[1]
        out = nl.ndarray((10, 128, W), dtype=nl.float32, buffer=nl.shared_hbm)
        at = nl.load(a[0:128, 0:W])
        bt = nl.load(b[0:128, 0:W])
        col = _fv_t((128, 1))
        nisa.tensor_copy(dst=col, src=bt[:, 0:1])
        i = 0
        for eng in (nisa.vector_engine, nisa.gpsimd_engine):
            for op in (nl.add, nl.subtract, nl.multiply):
                r = _fv_t((128, W))
                nisa.tensor_tensor(dst=r, data1=at, data2=bt, op=op, engine=eng)
                nl.store(out[i, 0:128, 0:W], value=r)
                i += 1
        for op in (nl.add, nl.multiply):
            r = _fv_t((128, W))
            nisa.tensor_scalar(dst=r, data=at, op0=op, operand0=col, engine=nisa.vector_engine)
            nl.store(out[i, 0:128, 0:W], value=r)
            i += 1
        r = _fv_t((128, W))
        nisa.activation(dst=r, op=nl.copy, data=at, bias=col, scale=1.0)
        nl.store(out[8, 0:128, 0:W], value=r)
        r = _fv_t((128, W))
        nisa.activation(dst=r, op=nl.copy, data=at, scale=col)
        nl.store(out[9, 0:128, 0:W], value=r)
        return out

    @nki.jit
    def _fp64_probe_fma_kernel(a, b, c):
        """Two-op instructions that could be fused multiply-adds: each
        computes a * b[:, 0] + c (c[:, 0] where the form needs a per-partition
        operand). Single rounding gives RN(a b + c); two give RN(RN(a b) + c).
        Output (5, 128, W): 0 VectorE tensor_scalar (mul, add) with (P, 1)
        operands; 1 GpSimd tensor_scalar; 2 VectorE scalar_tensor_tensor
        (mul by b[:, 0], add tensor c); 3 ScalarE activation (scale b[:, 0],
        bias c[:, 0]); 4 VectorE scalar_tensor_tensor (a * 4097) - a, the
        fused form of the Veltkamp split's first two steps."""
        W = a.shape[1]
        out = nl.ndarray((5, 128, W), dtype=nl.float32, buffer=nl.shared_hbm)
        at = nl.load(a[0:128, 0:W])
        bt = nl.load(b[0:128, 0:W])
        ct = nl.load(c[0:128, 0:W])
        bcol = _fv_t((128, 1))
        ccol = _fv_t((128, 1))
        nisa.tensor_copy(dst=bcol, src=bt[:, 0:1])
        nisa.tensor_copy(dst=ccol, src=ct[:, 0:1])
        for i, eng in ((0, nisa.vector_engine), (1, nisa.gpsimd_engine)):
            r = _fv_t((128, W))
            nisa.tensor_scalar(
                dst=r,
                data=at,
                op0=nl.multiply,
                operand0=bcol,
                op1=nl.add,
                operand1=ccol,
                engine=eng,
            )
            nl.store(out[i, 0:128, 0:W], value=r)
        r = _fv_t((128, W))
        nisa.scalar_tensor_tensor(
            dst=r, data=at, op0=nl.multiply, operand0=bcol, op1=nl.add, operand1=ct
        )
        nl.store(out[2, 0:128, 0:W], value=r)
        r = _fv_t((128, W))
        nisa.activation(dst=r, op=nl.copy, data=at, bias=ccol, scale=bcol)
        nl.store(out[3, 0:128, 0:W], value=r)
        r = _fv_t((128, W))
        nisa.scalar_tensor_tensor(
            dst=r, data=at, op0=nl.multiply, operand0=4097.0, op1=nl.subtract, operand1=at
        )
        nl.store(out[4, 0:128, 0:W], value=r)
        return out

    @nki.jit
    def _fp64_probe_eft_kernel(a, b):
        """The error-free transformations as the reduction kernels issue
        them: TwoSum and Dekker TwoProd of a, b. Output (4, 128, W): s, e of
        TwoSum, p, e of TwoProd."""
        W = a.shape[1]
        sh = (128, W)
        out = nl.ndarray((4, 128, W), dtype=nl.float32, buffer=nl.shared_hbm)
        at = nl.load(a[0:128, 0:W])
        bt = nl.load(b[0:128, 0:W])
        s, e = _fv_two_sum(at, bt, sh)
        p, f = _fv_two_prod(at, bt, sh)
        for i, r in enumerate((s, e, p, f)):
            nl.store(out[i, 0:128, 0:W], value=r)
        return out

    @nki.jit
    def _fp64_probe_pe_kernel(at, b):
        """TensorE + PSUM: (2, M, N) of at^T b for at (K, M), b (K, N) fp32,
        K a multiple of 128 (unrolled: keep K small). Output 0: one nc_matmul per 128-row K tile, all
        accumulated in one PSUM tile (accumulate=True). Output 1: each K tile
        into a fresh PSUM tile, evicted and summed on VectorE in order (so
        products and in-PSUM sums of one tile can be told apart from the
        cross-tile accumulation)."""
        K, M = at.shape
        N = b.shape[1]
        KT = K // 128
        out = nl.ndarray((2, M, N), dtype=nl.float32, buffer=nl.shared_hbm)
        ps = nl.zeros((M, N), dtype=nl.float32, buffer=nl.psum)
        acc = _fv_t((M, N))
        nisa.memset(dst=acc, value=0.0)
        for kt in range(KT):
            a_t = nl.load(at[kt * 128 : (kt + 1) * 128, 0:M])
            b_t = nl.load(b[kt * 128 : (kt + 1) * 128, 0:N])
            nisa.nc_matmul(dst=ps, stationary=a_t, moving=b_t, accumulate=True)
            p1 = nl.zeros((M, N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p1, stationary=a_t, moving=b_t, accumulate=True)
            t = _fv_t((M, N))
            nisa.tensor_copy(dst=t, src=p1)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=t, op=nl.add)
        r = _fv_t((M, N))
        nisa.tensor_copy(dst=r, src=ps)
        nl.store(out[0, 0:M, 0:N], value=r)
        nl.store(out[1, 0:M, 0:N], value=acc)
        return out


if HAS_NKI:
    # ---------------------------------------------------------------------
    # fp64 bits of RN64(hi + lo) 2^E on the device (fp64.ff_to_fp64_bits,
    # op for op): the gemm_fp64 combine without the host pass.
    # ---------------------------------------------------------------------

    def _fb_u(shape):
        return nl.ndarray(shape, dtype=nl.uint32, buffer=nl.sbuf)

    def _fb_ts(data, op0, operand0, shape, op1=None, operand1=None, dtype=None):
        r = nl.ndarray(shape, dtype=dtype or nl.float32, buffer=nl.sbuf)
        if op1 is None:
            nisa.tensor_scalar(dst=r, data=data, op0=op0, operand0=operand0)
        else:
            nisa.tensor_scalar(
                dst=r, data=data, op0=op0, operand0=operand0, op1=op1, operand1=operand1
            )
        return r

    def _fb_cvt(src, dtype, shape):
        r = nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=r, src=src)
        return r

    def _fb_tile(hilo, ea_col, eb, out, flag, r0, c0, w):
        """Bits of one (128, w) tile into out[r0 : r0 + 128, 2 c0 : 2 (c0 + w)];
        flag (128, 1) accumulates the tile's out-of-range maximum."""
        sh = (128, w)
        u32, f32 = nl.uint32, nl.float32
        h = nl.load(hilo[0, r0 : r0 + 128, c0 : c0 + w])
        lo = nl.load(hilo[1, r0 : r0 + 128, c0 : c0 + w])
        ebt = _fv_t(sh)
        nisa.dma_copy(dst=ebt, src=eb[0:1, c0 : c0 + w].broadcast(0, 128))
        E = _fv_t(sh)
        nisa.tensor_scalar(dst=E, data=ebt, op0=nl.add, operand0=ea_col)
        s, t = _fv_two_sum(h, lo, sh)
        sb = s.view(u32)
        sign_u = _fb_ts(sb, nl.right_shift, 31, sh, dtype=u32)
        Es = _fb_cvt(
            _fb_ts(sb, nl.right_shift, 23, sh, op1=nl.bitwise_and, operand1=0xFF, dtype=u32),
            f32,
            sh,
        )
        frac = _fb_ts(sb, nl.bitwise_and, 0x7FFFFF, sh, dtype=u32)
        S = _fb_cvt(_fb_ts(frac, nl.bitwise_or, 0x800000, sh, dtype=u32), f32, sh)
        fz = _fb_ts(_fb_cvt(frac, f32, sh), nl.equal, 0.0, sh)
        sce = _fb_ts(Es, nl.multiply, -1.0, sh, op1=nl.add, operand1=306.0)
        scc = _fb_ts(sce, nl.maximum, 1.0, sh, op1=nl.minimum, operand1=254.0)
        sc = _fb_ts(_fb_cvt(scc, u32, sh), nl.left_shift, 23, sh, dtype=u32).view(f32)
        ss = _fb_ts(_fb_cvt(sign_u, f32, sh), nl.multiply, -2.0, sh, op1=nl.add, operand1=1.0)
        v1 = _fv_op(_fv_op(t, ss, nl.multiply, sh), sc, nl.multiply, sh)
        m2 = _fv_op(fz, _fb_ts(v1, nl.less, -0.25, sh), nl.multiply, sh)
        mf = _fb_ts(m2, nl.add, 1.0, sh)
        v = _fv_op(v1, mf, nl.multiply, sh)
        av = _fb_ts(v.view(u32), nl.bitwise_and, 0x7FFFFFFF, sh, dtype=u32).view(f32)
        r1 = _fb_ts(av, nl.add, 8388608.0, sh)  # two instructions: both round
        r = _fb_ts(r1, nl.subtract, 8388608.0, sh)
        small = _fb_ts(av, nl.less, 8388608.0, sh)
        neg = _fb_ts(v, nl.less, 0.0, sh)
        sg = _fb_ts(neg, nl.multiply, -2.0, sh, op1=nl.add, operand1=1.0)
        rs = _fv_op(r, sg, nl.multiply, sh)
        d = _fv_op(_fv_op(rs, v, nl.subtract, sh), small, nl.multiply, sh)
        q = _fv_op(v, d, nl.add, sh)
        qi = _fb_cvt(q, nl.int32, sh)
        B = _fb_ts(qi.view(u32), nl.bitwise_and, (1 << 29) - 1, sh, dtype=u32)
        borrow = _fb_ts(q, nl.less, 0.0, sh)
        A = _fv_op(_fv_op(S, mf, nl.multiply, sh), borrow, nl.subtract, sh)
        Au = _fb_cvt(_fb_ts(A, nl.subtract, 8388608.0, sh), u32, sh)
        nz = _fb_ts(s, nl.not_equal, 0.0, sh)
        E64 = _fv_op(_fv_op(_fb_ts(Es, nl.add, 896.0, sh), m2, nl.subtract, sh), E, nl.add, sh)
        # out of range: s or t non-finite; s != 0 with Es < 52 or E64 outside [1, 2046]
        tE = _fb_cvt(
            _fb_ts(
                t.view(u32), nl.right_shift, 23, sh, op1=nl.bitwise_and, operand1=0xFF, dtype=u32
            ),
            f32,
            sh,
        )
        b1 = _fv_op(_fb_ts(Es, nl.less, 52.0, sh), _fb_ts(E64, nl.less, 1.0, sh), nl.maximum, sh)
        b2 = _fv_op(b1, _fb_ts(E64, nl.greater, 2046.0, sh), nl.maximum, sh)
        b3 = _fv_op(b2, nz, nl.multiply, sh)
        b4 = _fv_op(b3, _fb_ts(Es, nl.equal, 255.0, sh), nl.maximum, sh)
        bad = _fv_op(b4, _fb_ts(tE, nl.equal, 255.0, sh), nl.maximum, sh)
        bm = _fv_t((128, 1))
        nisa.tensor_reduce(dst=bm, op=nl.maximum, data=bad, axis=1, keepdims=True)
        nisa.tensor_tensor(dst=flag, data1=flag, data2=bm, op=nl.maximum)
        E64z = _fv_op(E64, nz, nl.multiply, sh)
        E64c = _fb_ts(E64z, nl.maximum, 0.0, sh, op1=nl.minimum, operand1=2047.0)
        Eu = _fb_ts(_fb_cvt(E64c, u32, sh), nl.left_shift, 20, sh, dtype=u32)
        hw = _fb_ts(sign_u, nl.left_shift, 31, sh, dtype=u32)
        nisa.tensor_tensor(dst=hw, data1=hw, data2=Eu, op=nl.bitwise_or)
        a3 = _fb_ts(Au, nl.right_shift, 3, sh, dtype=u32)
        nisa.tensor_tensor(dst=hw, data1=hw, data2=a3, op=nl.bitwise_or)
        lw = _fb_ts(Au, nl.bitwise_and, 7, sh, op1=nl.left_shift, operand1=29, dtype=u32)
        nisa.tensor_tensor(dst=lw, data1=lw, data2=B, op=nl.bitwise_or)
        T = _fb_u((128, 2 * w))
        nisa.tensor_copy(dst=T[:, 0 : 2 * w : 2], src=lw)
        nisa.tensor_copy(dst=T[:, 1 : 2 * w : 2], src=hw)
        nl.store(out[r0 : r0 + 128, 2 * c0 : 2 * (c0 + w)], value=T.view(nl.int32))

    def _fb_rows(hilo, ea, eb, out, b):
        """Row tile b (rows 128 b ..): all column tiles, then the tile's flags
        into out[R : R + 128, b]."""
        R = hilo.shape[1]
        N = hilo.shape[2]
        r0 = b * 128
        CF = N // _FB_W
        tail = N - CF * _FB_W
        ea_col = nl.load(ea[r0 : r0 + 128, 0:1])
        flag = _fv_t((128, 1))
        nisa.memset(dst=flag, value=0.0)
        if CF:
            for c in nl.sequential_range(CF):
                _fb_tile(hilo, ea_col, eb, out, flag, r0, c * _FB_W, _FB_W)
        if tail:
            _fb_tile(hilo, ea_col, eb, out, flag, r0, CF * _FB_W, tail)
        fi = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=fi, src=flag)
        nl.store(out[R : R + 128, b : b + 1], value=fi)

    @nki.jit
    def _fp64_ff_bits_kernel(hilo, ea, eb):
        """float64 bits of RN64(hi + lo) 2^(ea_i + eb_j) (`fp64.ff_to_fp64_bits`):
        `hilo` (2, R, N) float32 (the Ozaki kernel's output, R a multiple of
        128, R / 128 <= 2 N), `ea` (R, 1) and `eb` (1, N) float32 integer
        exponents. Returns (R + 128, 2 N) int32: rows 0 .. R - 1 are the
        little-endian words of the R x N float64 results (a plain view on the
        host), and out[R + p, b] flags row 128 b + p (1: an element of it is
        out of the handled range; take the host combine). Elementwise, so any
        LNC split gives the same bits; at LNC=2 the programs take halves of
        the row tiles (an odd middle tile is computed by both, identically)."""
        R = hilo.shape[1]
        N = hilo.shape[2]
        out = nl.ndarray((R + 128, 2 * N), dtype=nl.int32, buffer=nl.shared_hbm)
        RT = R // 128
        z = nl.ndarray((128, 2 * N), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=z, value=0)
        nl.store(out[R : R + 128, 0 : 2 * N], value=z)
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        per = (RT + n_prog - 1) // n_prog
        if n_prog > 1:
            b0 = nl.program_id(0) * (RT - per)
            for i in nl.affine_range(per):
                _fb_rows(hilo, ea, eb, out, b0 + i)
        else:
            for i in nl.affine_range(RT):
                _fb_rows(hilo, ea, eb, out, i)
        return out


# ── Launching ────────────────────────────────────────────────────────────────


def kernel(name: str):
    """The @nki.jit kernel `name` of this module."""
    if not HAS_NKI:
        raise RuntimeError("trnblas.fp64: the nki package is not importable")
    return globals()[name]


def check_operands(name: str, *tensors) -> None:
    """TypeError unless every operand dtype is allowed for kernel `name`."""
    allowed = KERNEL_DTYPES.get(name)
    if allowed is None:
        raise TypeError(f"trnblas: NKI kernel {name!r} has no dtype entry")
    bad = [str(t.dtype) for t in tensors if t.dtype not in allowed]
    if bad:
        raise TypeError(f"trnblas: NKI kernel {name!r} takes float32; got {bad}")


def call(name: str, *tensors: torch.Tensor, simulate: bool = False) -> torch.Tensor:
    """Run kernel `name` and return its result: on the native path on
    neuron tensors (uploading host operands, result left on the device), or
    under `nki.simulate` (host numpy, result on the host)."""
    check_operands(name, *tensors)
    k = kernel(name)
    if simulate:
        arrs = [t.detach().to("cpu").contiguous().numpy() for t in tensors]
        return torch.from_numpy(np.ascontiguousarray(nki.simulate(k)(*arrs)))
    from . import dispatch as D

    xs = [t if t.device.type == "neuron" else t.to("neuron") for t in tensors]
    xs = [t.contiguous() for t in xs]
    if D._eager_call_ok(xs):
        h = D._eager_handle(k)
        if h is not None:
            return h(*xs)
    from . import _native_fp64

    return _native_fp64.OPS[name](*xs)


def run_partials(kind: str, xs, accumulate: str, simulate: bool = False) -> torch.Tensor:
    """The chunk partials of `fp64.partials` from the kernel for (kind,
    accumulate), as a host float32 tensor."""
    from .. import fp64

    name = REDUCE_KERNELS.get((kind, accumulate))
    if name is None:
        raise NotImplementedError(
            f"trnblas.fp64: no NKI kernel for {kind} with accumulate={accumulate!r}; "
            f"kernels: {sorted(REDUCE_KERNELS)}"
        )
    pl = fp64.plan(xs[0].numel())
    rows = [fp64.layout(t, pl).contiguous() for t in xs]
    out = call(name, *rows, simulate=simulate)
    return out.to("cpu")


def lnc() -> int:
    """The LNC setting a kernel launch will see (NEURON_LOGICAL_NC_CONFIG)."""
    return int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))


def combine_bits_device(
    hilo: torch.Tensor, eA, eB, slices: int, M: int, N: int, simulate=False, out=None
):
    """`ozaki.combine` of one launch block, with the RN64 and the scaling on
    the device (`_fp64_ff_bits_kernel`): float64 (M, N) on the host, equal to
    `ozaki.combine(hilo, eA, eB, slices, M, N)` bit for bit (flagged rows are
    recomputed on the host). `hilo` (2, Rp, >= N) float32, Rp a multiple of
    128, on the device (on the host under `simulate`). `out`: optional host
    int32 (Rp + 128, 2 N) buffer to download into (pinned, reused)."""
    from .. import ozaki

    Rp = hilo.shape[1]
    off = ozaki.exponent_offset(slices)
    E_a = np.asarray(eA[:M], np.int64) + off
    E_b = np.asarray(eB[:N], np.int64)
    big = max(np.abs(E_a).max(initial=0), np.abs(E_b).max(initial=0)) >= 1 << 22
    if big or Rp % 128 or Rp // 128 > 2 * N:
        return ozaki.combine(hilo.to("cpu"), eA, eB, slices, M, N)
    hl = hilo if hilo.shape[2] == N else hilo[:, :, :N]
    hl = hl.contiguous()
    ea = np.zeros((Rp, 1), np.float32)
    ea[:M, 0] = E_a
    eb = E_b.astype(np.float32)[None, :]
    dev = hl.device
    res = call(
        "_fp64_ff_bits_kernel",
        hl,
        torch.from_numpy(ea).to(dev),
        torch.from_numpy(np.ascontiguousarray(eb)).to(dev),
        simulate=simulate,
    )
    host = res.to("cpu") if out is None else out.copy_(res)
    o = host.numpy()
    C = o[:Rp].view(np.float64)[:M]  # a view: no copy
    flags = o[Rp : Rp + 128, : Rp // 128].T.reshape(-1)[:M]
    bad = np.nonzero(flags)[0]
    if len(bad):
        sub = hilo.to("cpu")[:, torch.from_numpy(bad)]
        C[bad] = ozaki.combine(sub, np.asarray(eA)[bad], eB, slices, len(bad), N).numpy()
    return torch.from_numpy(C)
