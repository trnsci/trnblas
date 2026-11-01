"""Reference-conformance suite for every public trnblas op (trnsci/trnsci#105).

Each case compares a trnblas call with an *independent* reference: plain
torch formulas (``torch.matmul``, ``torch.linalg.solve_triangular``,
``torch.linalg.cholesky``, ...) evaluated on the host in float64/complex128.
It never compares one trnblas backend against another, so it also catches
bugs that every backend shares (the class trnsci/trnsci#35 kept finding:
trnblas#72 batched trans, trnblas#73 BLAS flag letters).

Axes: awkward sizes (0, 1, 127, 129, 1000), dtypes (fp32, fp64, complex64,
complex128, plus bf16/fp16 for the GEMM family), every value of every
flag-like keyword (trans*/uplo/side/diag, the BLAS letters, alpha/beta/C),
non-contiguous inputs (transposed, step-sliced, stride-0), empty and size-1
shapes, batch/broadcast, input mutation and output aliasing, determinism,
and fp64 ``gradcheck``/``gradgradcheck``.

Known needs-decision behaviour is ``xfail(strict=True)`` with the issue
number, so the case flips to XPASS (a failure) once it is decided:
trnblas#74 (``syrk`` ignores ``uplo``) and trnblas#75 (complex ``asum`` /
``iamax`` use the modulus, not BLAS |Re|+|Im|).

CPU CI runs it with the defaults (``pytest -m "not neuron"``). On a Trainium
box the same cases run through the NKI/auto paths with device inputs::

    pytest tests/test_conformance.py --backend nki --conformance-device neuron
    pytest tests/test_conformance.py --backend auto --conformance-device neuron
    pytest tests/test_conformance.py --backend nki          # host inputs

With ``--backend nki|auto`` the fp32/complex64 tolerances are relaxed 50x
(PSUM accumulation order). fp64 gradchecks are skipped for device inputs.
Under ``TRNBLAS_REQUIRE_NKI=1``, fp64/complex/int GEMM/SYRK/TRSM raise by
design (#68); run those configurations with ``-k "not float64 and not
complex"``.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import trnblas as tb

F32, F64, C64, C128 = torch.float32, torch.float64, torch.complex64, torch.complex128
DTYPES = [F32, F64, C64, C128]
DT_ID = {
    F32: "f32",
    F64: "f64",
    C64: "c64",
    C128: "c128",
    torch.bfloat16: "bf16",
    torch.float16: "f16",
}
BASE_TOL = {
    F32: 5e-5,
    C64: 5e-5,
    F64: 1e-11,
    C128: 1e-11,
    torch.bfloat16: 3e-2,
    torch.float16: 5e-3,
}
DEVICE_TOL_FACTOR = 50.0

# Set by the module fixture from --backend / --conformance-device.
_CFG = {"backend": "pytorch", "device": torch.device("cpu")}


@pytest.fixture(scope="module", autouse=True)
def conformance_backend(request):
    backend = request.config.getoption("--backend")
    device = request.config.getoption("--conformance-device")
    if device != "cpu":
        import torch_neuronx  # noqa: F401  (registers the `neuron` device)
    old = tb.get_backend()
    tb.set_backend(backend)
    _CFG.update(backend=backend, device=torch.device(device))
    yield
    tb.set_backend(old)
    _CFG.update(backend="pytorch", device=torch.device("cpu"))


# ---------------------------------------------------------------- inputs
def _gen(seed):
    return torch.Generator().manual_seed(seed)


def rnd(g, *shape, dtype=F64):
    if dtype.is_complex:
        re = torch.randn(shape, generator=g, dtype=F64)
        im = torch.randn(shape, generator=g, dtype=F64)
        return torch.complex(re, im).to(dtype)
    return torch.randn(shape, generator=g, dtype=F64).to(dtype)


def spd(g, n, dtype=F64):
    a = rnd(g, n, n)
    return (a @ a.T + n * torch.eye(n, dtype=F64)).to(dtype)


def well_tri(g, n, dtype=F64):
    """Both triangles well conditioned (off-diagonal ~ 1/sqrt(n), diag in [2, 3])."""
    a = rnd(g, n, n, dtype=dtype) * (0.5 / max(n, 1) ** 0.5)
    d = (2.0 + torch.rand(n, generator=g, dtype=F64)).to(dtype)
    return a + torch.diag(d)


def layout(t, kind):
    """Same values as `t`, in another memory layout."""
    if kind == "contig":
        return t
    if kind == "transposed":  # column-major copy of a 2-D tensor
        return t.mT.contiguous().mT
    if kind == "step":  # every other element of a larger buffer
        big = torch.zeros(tuple(2 * s + 1 for s in t.shape), dtype=t.dtype)
        idx = tuple(slice(1, None, 2) for _ in t.shape)
        big[idx] = t
        return big[idx]
    raise ValueError(kind)


def to_dev(v):
    """Move a host tensor to the conformance device, keeping its strides."""
    dev = _CFG["device"]
    if not isinstance(v, torch.Tensor) or dev.type == "cpu":
        return v
    if v.is_contiguous():
        return v.to(dev)
    return torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=dev).copy_(v)


def hi(t):
    t = t.detach().cpu()
    return t.to(C128) if t.is_complex() else t.to(F64)


# ------------------------------------------------------------- references
def _tri(A, uplo, diag):
    t = torch.triu(A) if uplo == "upper" else torch.tril(A)
    if diag == "unit":
        t = t - torch.diag(torch.diag(t)) + torch.eye(A.shape[0], dtype=A.dtype)
    return t


def _sym(A, uplo):
    """The symmetric (not Hermitian) matrix stored in the `uplo` triangle."""
    t = torch.triu(A) if uplo == "upper" else torch.tril(A)
    return t + t.mT - torch.diag(torch.diag(A))


def _addC(r, beta, C):
    return r + beta * C if (C is not None and beta != 0.0) else r


def R_gemm(alpha, A, B, beta=0.0, C=None, transA=False, transB=False):
    a = A.mT if transA and A.dim() >= 2 else A
    b = B.mT if transB and B.dim() >= 2 else B
    return _addC(alpha * torch.matmul(a, b), beta, C)


def R_symm(alpha, A, B, beta=0.0, C=None, side="left", uplo="upper"):
    S = _sym(A, uplo)
    return _addC(alpha * (S @ B if side == "left" else B @ S), beta, C)


def R_syrk(alpha, A, beta=0.0, C=None, trans=False, uplo="upper"):
    """BLAS semantics: only the `uplo` triangle of C is referenced; trnblas
    returns the full symmetric matrix."""
    P = A.mT @ A if trans else A @ A.mT
    return _addC(alpha * P, beta, None if C is None else _sym(C, uplo))


def R_trsm(alpha, A, B, side="left", uplo="upper", trans=False, diag="nonunit"):
    T = _tri(A, uplo, diag)
    M = T.mT if trans else T
    if side == "left":
        return torch.linalg.solve(M, alpha * B)
    return torch.linalg.solve(M.mT, (alpha * B).mT).mT


def R_trmm(alpha, A, B, side="left", uplo="upper", trans=False, diag="nonunit"):
    T = _tri(A, uplo, diag)
    M = T.mT if trans else T
    return alpha * (M @ B if side == "left" else B @ M)


def R_gemv(alpha, A, x, beta=0.0, y=None, trans=False):
    return _addC(alpha * ((A.mT if trans else A) @ x), beta, y)


def R_symv(alpha, A, x, beta=0.0, y=None, uplo="upper"):
    return _addC(alpha * (_sym(A, uplo) @ x), beta, y)


def R_trmv(A, x, uplo="upper", trans=False, diag="nonunit"):
    T = _tri(A, uplo, diag)
    return (T.mT if trans else T) @ x


def R_ger(alpha, x, y, A=None):
    r = alpha * torch.outer(x, y)
    return r + A if A is not None else r


REF = {
    "axpy": lambda alpha, x, y: alpha * x + y,
    "dot": lambda x, y: (x.reshape(-1) * y.reshape(-1)).sum(),
    "nrm2": lambda x: x.reshape(-1).abs().pow(2).sum().sqrt(),
    "scal": lambda alpha, x: alpha * x,
    "asum": lambda x: x.reshape(-1).abs().sum(),  # modulus: documented (see #75)
    "gemv": R_gemv,
    "symv": R_symv,
    "trmv": R_trmv,
    "ger": R_ger,
    "gemm": R_gemm,
    "batched_gemm": R_gemm,
    "symm": R_symm,
    "syrk": R_syrk,
    "trsm": R_trsm,
    "trmm": R_trmm,
    "metric_inv_half": lambda J: torch.linalg.inv(torch.linalg.cholesky(J)).mT,
}


# ------------------------------------------------------------------ check
def _shares(a, b):
    try:
        sa, sb = a.untyped_storage(), b.untyped_storage()
        return sa.nbytes() > 0 and sa.data_ptr() == sb.data_ptr()
    except Exception:
        return False


def check(op, args, kwargs=None, *, dtype, ref=None, scale=None, tol_mult=1.0):
    """Call trnblas `op` on device copies of `args` and compare with the
    float64 reference on the host copies.

    Also checks: output dtype == `dtype`, shape, device, no input mutated,
    output does not alias an input. `scale` overrides the error normaliser
    (default: max |ref|).
    """
    kwargs = kwargs or {}
    ref_fn = ref or REF[op]
    want = ref_fn(
        *[hi(a) if isinstance(a, torch.Tensor) else a for a in args],
        **{k: hi(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()},
    )
    dargs = [to_dev(a) for a in args]
    dkw = {k: to_dev(v) for k, v in kwargs.items()}
    tensors = [t for t in list(dargs) + list(dkw.values()) if isinstance(t, torch.Tensor)]
    before = [t.detach().cpu().clone() for t in tensors]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = getattr(tb, op)(*dargs, **dkw)
    for i, (b, t) in enumerate(zip(before, tensors, strict=True)):
        a = t.detach().cpu()
        same = a.shape == b.shape and bool(((a == b) | (a.isnan() & b.isnan())).all())
        assert same, f"input {i} mutated"
    assert isinstance(out, torch.Tensor), type(out)
    for i, t in enumerate(tensors):
        assert not _shares(out, t), f"output aliases input {i}"
    assert tuple(out.shape) == tuple(want.shape), (tuple(out.shape), tuple(want.shape))
    assert out.dtype == dtype, (out.dtype, dtype)
    assert out.device.type == _CFG["device"].type, out.device
    if want.numel() == 0:
        return out
    o = hi(out)
    s = scale if scale is not None else want.abs().max().item()
    err = (o - want).abs().max().item() / max(s, 1e-30)
    tol = BASE_TOL[dtype] * tol_mult
    if _CFG["backend"] != "pytorch" and dtype in (F32, C64, torch.bfloat16, torch.float16):
        tol *= DEVICE_TOL_FACTOR
    assert err <= tol, f"relerr {err:.3g} > {tol:.1g}"
    return out


def _id(v):
    if isinstance(v, torch.dtype):
        return DT_ID[v]
    if isinstance(v, tuple):
        return "x".join(map(str, v))
    return str(v)


P = pytest.mark.parametrize

# ================================================================ Level 1
VEC_SHAPES = [(), (0,), (1,), (127,), (129,), (1000,), (7, 9)]


@P("dtype", DTYPES, ids=_id)
@P("shape", VEC_SHAPES, ids=_id)
@P("lay", ["contig", "step"])
@P("op", ["axpy", "dot", "nrm2", "scal", "asum"])
def test_level1(op, lay, shape, dtype):
    g = _gen(1)
    x = layout(rnd(g, *shape, dtype=dtype), lay)
    y = rnd(g, *shape, dtype=dtype)
    scale = None
    if op == "axpy":
        args = [-1.5, x, y]
    elif op == "scal":
        args = [0.25, x]
    elif op == "dot":
        args = [x, y]
        # A dot product can cancel; normalise by sum |x_i y_i|.
        scale = (hi(x).abs() * hi(y).abs()).sum().item()
    else:
        args = [x]
    want_dtype = F32 if (op in ("nrm2", "asum") and dtype == C64) else dtype
    if op in ("nrm2", "asum") and dtype == C128:
        want_dtype = F64
    check(op, args, dtype=want_dtype, scale=scale, tol_mult=10.0 if op == "dot" else 1.0)


@P("alpha", [0.0, 1.0, -2.5, 3])
@P("op", ["axpy", "scal"])
def test_level1_alpha(op, alpha):
    g = _gen(2)
    x, y = rnd(g, 129, dtype=F32), rnd(g, 129, dtype=F32)
    check(op, [alpha, x, y] if op == "axpy" else [alpha, x], dtype=F32)


def test_level1_stride0():
    x = torch.tensor([1.5], dtype=F64).expand(10)
    y = torch.arange(10, dtype=F64)
    check("axpy", [2.0, x, y], dtype=F64)
    check("dot", [x, y], dtype=F64)
    check("nrm2", [x], dtype=F64)
    check("asum", [x], dtype=F64)


def _iamax_ref(x):
    return int(torch.argmax(hi(x).reshape(-1).abs()).item())


@P("dtype", DTYPES, ids=_id)
@P("n", [1, 127, 129, 1000])
@P("lay", ["contig", "step"])
def test_iamax(lay, n, dtype):
    x = layout(rnd(_gen(3), n, dtype=dtype), lay)
    out = tb.iamax(to_dev(x))
    assert isinstance(out, int)
    assert out == _iamax_ref(x)


def test_iamax_first_of_ties_and_matrix():
    assert tb.iamax(to_dev(torch.tensor([1.0, -3.0, 3.0, 2.0]))) == 1
    A = rnd(_gen(4), 7, 9)
    assert tb.iamax(to_dev(A.mT)) == _iamax_ref(A.mT)


# BLAS (and cuBLAS icamax/scasum) use |Re| + |Im| for complex input.
_75 = pytest.mark.xfail(
    strict=True,
    reason="trnsci/trnblas#75 needs decision: complex asum/iamax use the modulus, "
    "BLAS uses |Re|+|Im|",
)


@_75
@P("dtype", [C64, C128], ids=_id)
def test_asum_complex_blas(dtype):
    x = torch.tensor([3 + 4j, -1 + 2j, 0.5 - 0.5j], dtype=dtype)
    want = (x.real.abs() + x.imag.abs()).sum()
    assert torch.allclose(hi(tb.asum(to_dev(x))), hi(want))


@_75
@P("dtype", [C64, C128], ids=_id)
def test_iamax_complex_blas(dtype):
    # modulus picks 0 (3 > 2.83); |Re|+|Im| picks 1 (4 > 3).
    x = torch.tensor([3 + 0j, 2 + 2j], dtype=dtype)
    assert tb.iamax(to_dev(x)) == 1


# ================================================================ Level 2
MV_SHAPES = [(1, 1), (127, 129), (129, 127), (1000, 3), (3, 1000), (0, 4), (4, 0)]
BETA_Y = ["beta0", "beta0.5", "beta0_y_nan"]


def _y(g, n, dtype, by):
    if by == "beta0":
        return 0.0, None
    if by == "beta0.5":
        return 0.5, rnd(g, n, dtype=dtype)
    # beta == 0: y must not be read (BLAS), so NaN in it must not leak.
    return 0.0, torch.full((n,), float("nan"), dtype=dtype)


@P("dtype", DTYPES, ids=_id)
@P("by", BETA_Y)
@P("trans", [False, True, "N", "T"])
@P("lay", ["contig", "transposed", "step"])
@P("mn", MV_SHAPES, ids=_id)
def test_gemv(mn, lay, trans, by, dtype):
    g = _gen(10)
    m, n = mn
    A = layout(rnd(g, m, n, dtype=dtype), lay)
    t = trans in (True, "T")
    x = rnd(g, m if t else n, dtype=dtype)
    beta, y = _y(g, n if t else m, dtype, by)
    check(
        "gemv", [1.5, A, x, beta, y], dict(trans=trans), dtype=dtype, ref=_ref_kw(R_gemv, trans=t)
    )


def _ref_kw(fn, **fixed):
    """Reference with some keyword values replaced by their canonical meaning."""

    def r(*a, **kw):
        kw.update(fixed)
        return fn(*a, **kw)

    return r


UPLO = ["upper", "lower", "U", "l"]
CANON_UPLO = {"upper": "upper", "lower": "lower", "U": "upper", "u": "upper", "L": "lower"}
CANON_UPLO["l"] = "lower"


@P("dtype", DTYPES, ids=_id)
@P("by", BETA_Y)
@P("uplo", UPLO)
@P("lay", ["contig", "transposed", "step"])
@P("n", [0, 1, 127, 129])
def test_symv(n, lay, uplo, by, dtype):
    g = _gen(11)
    A = layout(rnd(g, n, n, dtype=dtype), lay)
    x = rnd(g, n, dtype=dtype)
    beta, y = _y(g, n, dtype, by)
    check(
        "symv",
        [1.5, A, x, beta, y],
        dict(uplo=uplo),
        dtype=dtype,
        ref=_ref_kw(R_symv, uplo=CANON_UPLO[uplo]),
    )


@P("dtype", DTYPES, ids=_id)
@P("diag", ["nonunit", "unit", "N", "U"])
@P("trans", [False, True])
@P("uplo", ["upper", "lower"])
@P("lay", ["contig", "transposed"])
@P("n", [0, 1, 127, 129])
def test_trmv(n, lay, uplo, trans, diag, dtype):
    g = _gen(12)
    A = layout(rnd(g, n, n, dtype=dtype), lay)
    x = rnd(g, n, dtype=dtype)
    cdiag = {"N": "nonunit", "U": "unit"}.get(diag, diag)
    check(
        "trmv",
        [A, x],
        dict(uplo=uplo, trans=trans, diag=diag),
        dtype=dtype,
        ref=_ref_kw(R_trmv, diag=cdiag),
    )


@P("dtype", DTYPES, ids=_id)
@P("withA", [False, True])
@P("lay", ["contig", "step"])
@P("mn", MV_SHAPES, ids=_id)
def test_ger(mn, lay, withA, dtype):
    g = _gen(13)
    m, n = mn
    x = layout(rnd(g, m, dtype=dtype), lay)
    y = rnd(g, n, dtype=dtype)
    args = [-0.75, x, y] + ([layout(rnd(g, m, n, dtype=dtype), "transposed")] if withA else [])
    check("ger", args, dtype=dtype)


# ================================================================ Level 3
MKN = [
    (1, 1, 1),
    (127, 129, 1),
    (1, 127, 129),
    (129, 127, 130),
    (1000, 3, 2),
    (2, 1000, 3),
    (0, 4, 5),
    (4, 0, 5),
    (4, 5, 0),
]
BETA_C = ["beta0", "beta0.5", "beta0_C_nan", "beta1"]


def _C(g, shape, dtype, bc):
    if bc == "beta0":
        return 0.0, None
    if bc == "beta0_C_nan":
        # BLAS: C need not be set on input when beta == 0.
        return 0.0, torch.full(shape, float("nan"), dtype=dtype)
    return float(bc[4:]), rnd(g, *shape, dtype=dtype)


@P("dtype", DTYPES, ids=_id)
@P("bc", BETA_C)
@P("transB", [False, True])
@P("transA", [False, True])
@P("mkn", MKN, ids=_id)
def test_gemm(mkn, transA, transB, bc, dtype):
    g = _gen(20)
    m, k, n = mkn
    A = rnd(g, *((k, m) if transA else (m, k)), dtype=dtype)
    B = rnd(g, *((n, k) if transB else (k, n)), dtype=dtype)
    beta, C = _C(g, (m, n), dtype, bc)
    check("gemm", [1.5, A, B, beta, C], dict(transA=transA, transB=transB), dtype=dtype)


@P("trans", ["N", "T", "n", "t"])
def test_gemm_letter_flags(trans):
    g = _gen(21)
    A, B = rnd(g, 6, 6, dtype=F32), rnd(g, 6, 5, dtype=F32)
    t = trans in ("T", "t")
    check(
        "gemm",
        [1.0, A, B],
        dict(transA=trans),
        dtype=F32,
        ref=_ref_kw(R_gemm, transA=t),
    )
    check(
        "batched_gemm",
        [
            1.0,
            rnd(g, 2, 6, 6, dtype=F32),
            rnd(g, 2, 5, 6, dtype=F32) if t else rnd(g, 2, 6, 5, dtype=F32),
        ],
        dict(transB=trans),
        dtype=F32,
        ref=_ref_kw(R_gemm, transB=t),
    )


@P("which", ["A", "B", "C"])
@P("lay", ["transposed", "step"])
@P("dtype", [F32, F64, C64], ids=_id)
def test_gemm_layouts(dtype, lay, which):
    g = _gen(22)
    ops = {"A": rnd(g, 129, 127, dtype=dtype), "B": rnd(g, 127, 65, dtype=dtype)}
    ops["C"] = rnd(g, 129, 65, dtype=dtype)
    ops[which] = layout(ops[which], lay)
    check("gemm", [1.5, ops["A"], ops["B"], 0.5, ops["C"]], dtype=dtype)
    # Through the transpose flags as well.
    At, Bt = ops["A"].mT, ops["B"].mT
    check("gemm", [1.5, At, Bt], dict(transA=True, transB=True), dtype=dtype)


@P("dtype", [torch.bfloat16, torch.float16], ids=_id)
@P("op", ["gemm", "batched_gemm", "symm", "syrk", "trmm"])
def test_level3_half(op, dtype):
    g = _gen(23)
    if op == "gemm":
        args = [1.5, rnd(g, 33, 17, dtype=dtype), rnd(g, 17, 9, dtype=dtype)]
    elif op == "batched_gemm":
        args = [1.5, rnd(g, 3, 33, 17, dtype=dtype), rnd(g, 3, 17, 9, dtype=dtype)]
    elif op == "symm":
        args = [1.5, rnd(g, 17, 17, dtype=dtype), rnd(g, 17, 9, dtype=dtype)]
    elif op == "syrk":
        args = [1.5, rnd(g, 17, 9, dtype=dtype)]
    else:
        args = [1.5, rnd(g, 17, 17, dtype=dtype), rnd(g, 17, 9, dtype=dtype)]
    check(op, args, dtype=dtype, tol_mult=2.0)


# Batched gemm (>= 3-D) and batched_gemm: torch.matmul batch semantics,
# op() transposes the last two dims (trnblas#72).
BATCH_CASES = {
    "3d": ((3, 4, 5), (3, 5, 6), None),
    "4d": ((2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 4, 6)),
    "batch1": ((1, 129, 127), (1, 127, 3), (1, 129, 3)),
    "batch0": ((0, 4, 5), (0, 5, 6), None),
    "bcast_A1": ((1, 4, 5), (3, 5, 6), None),
    "bcast_B2d": ((3, 4, 5), (5, 6), None),
    "C_bcast_2d": ((3, 4, 5), (3, 5, 6), (4, 6)),
    "C_bcast_row": ((3, 4, 5), (3, 5, 6), (6,)),
    "square_4": ((4, 4, 4), (4, 4, 4), (4, 4, 4)),
}


@P("dtype", [F32, F64, C128], ids=_id)
@P("transB", [False, True])
@P("transA", [False, True])
@P("case", list(BATCH_CASES))
@P("op", ["batched_gemm", "gemm"])
def test_batched(op, case, transA, transB, dtype):
    g = _gen(24)
    sa, sb, sc = BATCH_CASES[case]
    A = rnd(g, *sa, dtype=dtype)
    B = rnd(g, *sb, dtype=dtype)
    A = A.mT.contiguous() if transA else A
    B = B.mT.contiguous() if transB else B
    C = rnd(g, *sc, dtype=dtype) if sc else None
    check(
        op,
        [1.5, A, B, 0.5 if C is not None else 0.0, C],
        dict(transA=transA, transB=transB),
        dtype=dtype,
    )


@P("kind", ["stride0", "permuted", "channels_last"])
def test_batched_layouts(kind):
    g = _gen(25)
    if kind == "stride0":
        A, B = rnd(g, 1, 4, 5).expand(3, 4, 5), rnd(g, 3, 5, 6)
    elif kind == "permuted":
        A = rnd(g, 4, 3, 5).permute(1, 0, 2)
        B = rnd(g, 3, 6, 5).transpose(-2, -1)
    else:
        A = rnd(g, 2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
        B = rnd(g, 2, 3, 5, 6).contiguous(memory_format=torch.channels_last)
    check("batched_gemm", [1.5, A, B], dtype=F64)
    check("gemm", [1.5, A, B], dtype=F64)


def test_gemm_C_broadcast_and_vector_rhs():
    g = _gen(26)
    A, B = rnd(g, 4, 5), rnd(g, 5, 6)
    check("gemm", [1.5, A, B, 0.5, rnd(g, 6)], dtype=F64)
    check("gemm", [1.5, A, B, 0.5, rnd(g, 4, 1)], dtype=F64)
    check("gemm", [1.5, A, B, 0.5, torch.tensor(2.0, dtype=F64)], dtype=F64)
    check("gemm", [1.5, A, rnd(g, 5)], dtype=F64)


def test_gemm_aliasing_and_mutation():
    g = _gen(27)
    X = rnd(g, 5, 5)
    check("gemm", [1.5, X, rnd(g, 5, 5), 0.5, X], dtype=F64)  # C is A
    check("gemm", [1.0, X, X], dict(transB=True), dtype=F64)  # A is B
    base = rnd(g, 10, 10, dtype=F32)
    check("gemm", [1.5, rnd(g, 7, 6, dtype=F32), rnd(g, 6, 5, dtype=F32), 0.5, base[2:9, 1:6]],
          dtype=F32)  # fmt: skip
    # C of another dtype promotes, as torch does.
    check(
        "gemm",
        [1.5, rnd(g, 4, 3, dtype=F32), rnd(g, 3, 4, dtype=F32), 0.5, rnd(g, 4, 4)],
        dtype=F64,
        tol_mult=1e5,  # the product is computed in fp32
    )
    # The result is fresh: writing to it leaves the inputs alone.
    A, B = rnd(g, 4, 4), rnd(g, 4, 4)
    sa, sb = A.clone(), B.clone()
    tb.gemm(1.0, to_dev(A), to_dev(B)).add_(1.0)
    assert torch.equal(A, sa) and torch.equal(B, sb)


SIDE = ["left", "right", "L", "r"]
CANON_SIDE = {"left": "left", "right": "right", "L": "left", "r": "right"}


@P("dtype", DTYPES, ids=_id)
@P("bc", ["beta0", "beta0.5", "beta0_C_nan"])
@P("uplo", UPLO)
@P("side", SIDE)
@P("mn", [(1, 1), (127, 3), (129, 129), (0, 3), (3, 0)], ids=_id)
def test_symm(mn, side, uplo, bc, dtype):
    g = _gen(30)
    m, n = mn
    cs = CANON_SIDE[side]
    na = m if cs == "left" else n
    A = rnd(g, na, na, dtype=dtype)
    B = rnd(g, m, n, dtype=dtype)
    beta, C = _C(g, (m, n), dtype, bc)
    check(
        "symm",
        [1.5, A, B, beta, C],
        dict(side=side, uplo=uplo),
        dtype=dtype,
        ref=_ref_kw(R_symm, side=cs, uplo=CANON_UPLO[uplo]),
    )


@P("dtype", [F32, F64], ids=_id)
@P("lay", ["transposed", "step"])
def test_symm_layouts(lay, dtype):
    g = _gen(31)
    A = layout(rnd(g, 65, 65, dtype=dtype), lay)
    B = layout(rnd(g, 65, 33, dtype=dtype), lay)
    check("symm", [1.5, A, B], dict(uplo="lower"), dtype=dtype)


SYRK_NK = [(1, 1), (127, 3), (3, 129), (129, 129), (0, 3), (3, 0)]


@P("dtype", DTYPES, ids=_id)
@P("Ckind", ["none", "symmetric", "symmetric_beta0_nan"])
@P("uplo", ["upper", "lower"])
@P("trans", [False, True, "N", "T"])
@P("nk", SYRK_NK, ids=_id)
def test_syrk(nk, trans, uplo, Ckind, dtype):
    g = _gen(40)
    n, k = nk
    t = trans in (True, "T")
    A = rnd(g, *((k, n) if t else (n, k)), dtype=dtype)
    if Ckind == "none":
        beta, C = 0.0, None
    elif Ckind == "symmetric":
        S = rnd(g, n, n, dtype=dtype)
        beta, C = 0.5, S + S.mT
    else:
        beta, C = 0.0, torch.full((n, n), float("nan"), dtype=dtype)
    check(
        "syrk",
        [1.5, A, beta, C],
        dict(trans=trans, uplo=uplo),
        dtype=dtype,
        ref=_ref_kw(R_syrk, trans=t),
    )


@pytest.mark.xfail(
    strict=True,
    reason="trnsci/trnblas#74 needs decision: syrk ignores uplo; BLAS reads only "
    "the uplo triangle of C",
)
@P("Ckind", ["one_triangle", "nonsymmetric"])
@P("uplo", ["upper", "lower"])
@P("trans", [False, True])
def test_syrk_uplo_blas(trans, uplo, Ckind):
    g = _gen(41)
    n = 9
    A = rnd(g, *((3, n) if trans else (n, 3)))
    S = spd(g, n)
    if Ckind == "one_triangle":
        C = torch.triu(S) if uplo == "upper" else torch.tril(S)
    else:
        C = rnd(g, n, n)
    check("syrk", [1.5, A, 0.5, C], dict(trans=trans, uplo=uplo), dtype=F64)


def test_syrk_layouts_and_broadcast():
    g = _gen(42)
    for lay in ("transposed", "step"):
        A = layout(rnd(g, 65, 33), lay)
        check("syrk", [1.5, A], dtype=F64)
        check("syrk", [1.5, A], dict(trans=True), dtype=F64)
    check(
        "syrk",
        [1.5, rnd(g, 4, 3), 0.5, torch.tensor(2.0, dtype=F64)],
        dtype=F64,
        ref=lambda alpha, A, beta, C: alpha * A @ A.mT + beta * C,
    )


TRI_FLAGS = [
    (side, uplo, trans, diag)
    for side in ("left", "right")
    for uplo in ("upper", "lower")
    for trans in (False, True)
    for diag in ("nonunit", "unit")
]


def _tri_id(f):
    return "-".join(str(x) for x in f)


@P("dtype", DTYPES, ids=_id)
@P("alpha", [1.0, -0.5])
@P("flags", TRI_FLAGS, ids=_tri_id)
@P("mn", [(1, 1), (127, 3), (129, 2), (3, 129), (0, 3), (3, 0)], ids=_id)
@P("op", ["trsm", "trmm"])
def test_triangular(op, mn, flags, alpha, dtype):
    g = _gen(50)
    side, uplo, trans, diag = flags
    m, n = mn
    na = m if side == "left" else n
    A = well_tri(g, na, dtype=dtype) if op == "trsm" else rnd(g, na, na, dtype=dtype)
    B = rnd(g, m, n, dtype=dtype)
    check(
        op,
        [alpha, A, B],
        dict(side=side, uplo=uplo, trans=trans, diag=diag),
        dtype=dtype,
        tol_mult=4.0,
    )


@P("letters", [("L", "U", "N", "N"), ("R", "L", "T", "U"), ("l", "u", "t", "u")], ids=str)
@P("op", ["trsm", "trmm"])
def test_triangular_letter_flags(op, letters):
    g = _gen(51)
    side, uplo, trans, diag = letters
    canon = dict(
        side={"l": "left", "r": "right"}[side.lower()],
        uplo={"u": "upper", "l": "lower"}[uplo.lower()],
        trans=trans.lower() == "t",
        diag={"u": "unit", "n": "nonunit"}[diag.lower()],
    )
    na = 6 if canon["side"] == "left" else 4
    A = well_tri(g, na)
    B = rnd(g, 6, 4)
    check(
        op,
        [1.5, A, B],
        dict(side=side, uplo=uplo, trans=trans, diag=diag),
        dtype=F64,
        ref=_ref_kw(REF[op], **canon),
    )


@P("dtype", [F32, F64], ids=_id)
@P("uplo", ["upper", "lower"])
@P("op", ["trsm", "trmm"])
def test_triangular_blocked_and_layouts(op, uplo, dtype):
    g = _gen(52)
    # 200 crosses the 128-row panel of the blocked TRSM; wide RHS of 1000.
    A = well_tri(g, 200, dtype=dtype)
    check(op, [1.0, A, rnd(g, 200, 3, dtype=dtype)], dict(uplo=uplo), dtype=dtype, tol_mult=4.0)
    check(op, [1.0, well_tri(g, 33, dtype=dtype), rnd(g, 33, 1000, dtype=dtype)], dict(uplo=uplo),
          dtype=dtype, tol_mult=4.0)  # fmt: skip
    At = layout(well_tri(g, 65, dtype=dtype), "transposed")
    Bs = layout(rnd(g, 65, 7, dtype=dtype), "step")
    check(op, [1.0, At, Bs], dict(uplo=uplo, trans=True), dtype=dtype, tol_mult=4.0)


@pytest.mark.parametrize(
    "op,kw",
    [
        ("symv", dict(uplo="X")),
        ("trmv", dict(diag="C")),
        ("trmv", dict(trans="C")),
        ("gemv", dict(trans="C")),
        ("symm", dict(side="top")),
        ("syrk", dict(uplo="both")),
        ("trsm", dict(diag="maybe")),
        ("trmm", dict(uplo="")),
        ("gemm", dict(transA="H")),
        ("batched_gemm", dict(transB="C")),
    ],
)
def test_invalid_flags_raise(op, kw):
    """An unknown flag must raise, not silently pick a branch (trnblas#73)."""
    g = _gen(53)
    A, B, x = rnd(g, 4, 4), rnd(g, 4, 4), rnd(g, 4)
    args = {
        "symv": [1.0, A, x],
        "trmv": [A, x],
        "gemv": [1.0, A, x],
        "symm": [1.0, A, B],
        "syrk": [1.0, A],
        "trsm": [1.0, well_tri(g, 4), B],
        "trmm": [1.0, A, B],
        "gemm": [1.0, A, B],
        "batched_gemm": [1.0, A[None], B[None]],
    }[op]
    with pytest.raises(ValueError):
        getattr(tb, op)(*[to_dev(a) for a in args], **kw)


@P("dtype", [F32, F64], ids=_id)
@P("lay", ["contig", "transposed"])
@P("n", [1, 2, 127, 129])
def test_metric_inv_half(n, lay, dtype):
    J = layout(spd(_gen(60), n, dtype=dtype), lay)
    X = check("metric_inv_half", [J], dtype=dtype, tol_mult=10.0)
    # Defining property: X X^T = J^-1, X upper triangular.
    Xh = hi(X)
    assert torch.equal(Xh, torch.triu(Xh))
    err = (hi(J) @ Xh @ Xh.mT - torch.eye(n, dtype=F64)).abs().max().item()
    assert err < (1e-3 if dtype == F32 else 1e-10)


def test_metric_inv_half_empty():
    out = tb.metric_inv_half(to_dev(torch.zeros(0, 0)))
    assert out.shape == (0, 0)


# ============================================================ determinism
@P(
    "op",
    ["gemm", "batched_gemm", "symm", "syrk", "trsm", "trmm", "gemv", "symv", "trmv", "ger"],
)
def test_determinism(op):
    g = _gen(70)
    m, k, n = 130, 260, 70
    f = F32
    args = {
        "gemm": [1.5, rnd(g, m, k, dtype=f), rnd(g, k, n, dtype=f), 0.5, rnd(g, m, n, dtype=f)],
        "batched_gemm": [1.5, rnd(g, 3, m, k, dtype=f), rnd(g, 3, k, n, dtype=f)],
        "symm": [1.5, rnd(g, m, m, dtype=f), rnd(g, m, n, dtype=f)],
        "syrk": [1.5, rnd(g, m, k, dtype=f)],
        "trsm": [1.5, well_tri(g, m, dtype=f), rnd(g, m, n, dtype=f)],
        "trmm": [1.5, rnd(g, m, m, dtype=f), rnd(g, m, n, dtype=f)],
        "gemv": [1.5, rnd(g, m, k, dtype=f), rnd(g, k, dtype=f)],
        "symv": [1.5, rnd(g, m, m, dtype=f), rnd(g, m, dtype=f)],
        "trmv": [rnd(g, m, m, dtype=f), rnd(g, m, dtype=f)],
        "ger": [1.5, rnd(g, m, dtype=f), rnd(g, n, dtype=f)],
    }[op]
    d = [to_dev(a) for a in args]
    fn = getattr(tb, op)
    assert torch.equal(fn(*d).cpu(), fn(*d).cpu())


# =============================================================== autograd
def _grad_specs():
    g = _gen(80)
    return {
        "axpy": ([1.3, rnd(g, 4), rnd(g, 4)], {}),
        "dot": ([rnd(g, 4), rnd(g, 4)], {}),
        "nrm2": ([rnd(g, 4)], {}),
        "scal": ([1.3, rnd(g, 4)], {}),
        "asum": ([rnd(g, 4)], {}),
        "gemv": ([1.3, rnd(g, 4, 3), rnd(g, 3), 0.7, rnd(g, 4)], {}),
        "gemv:trans": ([1.3, rnd(g, 3, 4), rnd(g, 3), 0.7, rnd(g, 4)], dict(trans=True)),
        "symv": ([1.3, rnd(g, 4, 4), rnd(g, 4), 0.7, rnd(g, 4)], {}),
        "symv:lower": ([1.3, rnd(g, 4, 4), rnd(g, 4)], dict(uplo="lower")),
        "trmv": ([rnd(g, 4, 4), rnd(g, 4)], dict(uplo="lower", trans=True, diag="unit")),
        "ger": ([1.3, rnd(g, 4), rnd(g, 3), rnd(g, 4, 3)], {}),
        "gemm": ([1.3, rnd(g, 4, 3), rnd(g, 3, 5), 0.7, rnd(g, 4, 5)], {}),
        "gemm:trans": (
            [1.3, rnd(g, 3, 4), rnd(g, 5, 3), 0.7, rnd(g, 4, 5)],
            dict(transA=True, transB=True),
        ),
        "gemm:batched_trans": (
            [1.3, rnd(g, 2, 3, 4), rnd(g, 2, 5, 3)],
            dict(transA=True, transB=True),
        ),
        "batched_gemm": ([1.3, rnd(g, 2, 4, 3), rnd(g, 2, 3, 5), 0.7, rnd(g, 2, 4, 5)], {}),
        "symm": ([1.3, rnd(g, 4, 4), rnd(g, 4, 3), 0.7, rnd(g, 4, 3)], {}),
        "symm:right_lower": ([1.3, rnd(g, 3, 3), rnd(g, 4, 3)], dict(side="right", uplo="lower")),
        "syrk": ([1.3, rnd(g, 4, 3), 0.7, spd(g, 4)], {}),
        "syrk:trans": ([1.3, rnd(g, 3, 4)], dict(trans=True)),
        "trsm": ([1.3, well_tri(g, 4), rnd(g, 4, 3)], {}),
        "trsm:lower_trans_unit": (
            [1.3, well_tri(g, 4), rnd(g, 4, 3)],
            dict(uplo="lower", trans=True, diag="unit"),
        ),
        "trsm:right": ([1.3, well_tri(g, 3), rnd(g, 4, 3)], dict(side="right")),
        "trmm": ([1.3, rnd(g, 4, 4), rnd(g, 4, 3)], {}),
        "trmm:right_lower": ([1.3, rnd(g, 3, 3), rnd(g, 4, 3)], dict(side="right", uplo="lower")),
        "metric_inv_half": ([spd(g, 4)], {}),
    }


GRAD_SPECS = _grad_specs()


@P("name", list(GRAD_SPECS))
def test_autograd(name):
    """fp64 gradcheck + gradgradcheck, and gradients equal the reference's."""
    if _CFG["device"].type != "cpu":
        pytest.skip("fp64 gradcheck runs on host inputs only")
    from torch.autograd import gradcheck, gradgradcheck

    op = name.split(":")[0]
    args, kw = GRAD_SPECS[name]
    tidx = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
    # Symmetric-matrix inputs: differentiate through a symmetrising map so the
    # finite differences stay in the subspace where the gradient is defined.
    syms = {"metric_inv_half": {0}, "syrk": {3}}.get(op, set())

    def fill(ts):
        a = list(args)
        for i, t in zip(tidx, ts, strict=True):
            a[i] = 0.5 * (t + t.mT) if i in syms else t
        return a

    def f(*ts):
        return getattr(tb, op)(*fill(ts), **kw)

    def fr(*ts):
        return REF[op](*fill(ts), **kw)

    inputs = tuple(args[i].clone().requires_grad_(True) for i in tidx)
    assert gradcheck(f, inputs, eps=1e-6, atol=1e-6)
    assert gradgradcheck(f, inputs, eps=1e-6, atol=1e-6)
    out = f(*inputs)
    cot = rnd(_gen(81), *out.shape)
    got = torch.autograd.grad(out, inputs, cot, allow_unused=True)
    want = torch.autograd.grad(fr(*inputs), inputs, cot, allow_unused=True)
    for a, b in zip(got, want, strict=True):
        assert a is not None and b is not None
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
