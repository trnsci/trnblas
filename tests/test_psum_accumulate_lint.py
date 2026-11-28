"""Static lint: every PSUM accumulation group opens with a matmul write
(trnsci/trnsci#185).

The `nki.isa.nc_matmul` documentation: the first matmul into a PSUM
location must use `accumulate=False`; `accumulate=True` onto uninitialised
PSUM is undefined behaviour on all targets, and on NeuronCore-v3 (Trn2)
accumulating onto PSUM set by a non-matmul instruction (`nl.zeros`, i.e. a
memset) is undefined behaviour too. Device results matched the references
before the fix, but only by luck of the current compiler and runtime.

This test parses `trnblas/nki/*.py` (no nki import, CPU only) and fails if a
PSUM tile allocated with `nl.zeros(..., buffer=nl.psum)` or
`nl.ndarray(..., buffer=nl.psum)` gets an `nc_matmul(..., accumulate=True)`
before any opening matmul write to it in source order. An opening write is
`accumulate=False` / `None` / omitted, or a non-constant expression such as
`accumulate=(k > 0)` over a static range. It follows PSUM tiles through
plain helper functions in the scanned modules: a helper that returns a PSUM
allocation (`_oz_psum`), and a helper that issues the matmuls on a parameter,
with `accumulate` given by another parameter and branches on a bare
parameter (`if first:`) resolved from constant call arguments (`_oz_run`).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

NKI_DIR = Path(__file__).resolve().parents[1] / "trnblas" / "nki"

OPEN, ACC = "open", "acc"


def _attr(node) -> str | None:
    return node.attr if isinstance(node, ast.Attribute) else None


def _is_psum_alloc(node) -> bool:
    """`nl.zeros(...)` / `nl.ndarray(...)` with `buffer=nl.psum`."""
    if not (isinstance(node, ast.Call) and _attr(node.func) in ("zeros", "ndarray")):
        return False
    return any(kw.arg == "buffer" and _attr(kw.value) == "psum" for kw in node.keywords)


def _is_psum_zeros(node) -> bool:
    return _is_psum_alloc(node) and _attr(node.func) == "zeros"


def _root(node) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _callee(node: ast.Call) -> str | None:
    f = node.func
    return f.id if isinstance(f, ast.Name) else _attr(f)


def _mode(expr, params: set[str]):
    """The write mode of an `accumulate=` value: ACC, OPEN or ("param", name)."""
    if expr is None:
        return OPEN
    if isinstance(expr, ast.Constant):
        return ACC if expr.value is True else OPEN
    if isinstance(expr, ast.Name) and expr.id in params:
        return ("param", expr.id)
    return OPEN  # data-dependent, e.g. `kt > 0` over a static range: opens on its first trip


class _Module:
    """Function table over every scanned module (helpers are imported across them)."""

    def __init__(self, sources: dict[str, str]):
        self.funcs: dict[str, ast.FunctionDef] = {}
        self.where: dict[str, str] = {}
        for name, src in sources.items():
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.FunctionDef):
                    self.funcs[node.name] = node
                    self.where[node.name] = name
        self._factory: dict[str, bool] = {}
        self._events: dict[str, list] = {}

    def factory(self, fname: str) -> bool:
        """Whether `fname` returns a PSUM allocation (e.g. `_oz_psum`)."""
        if fname not in self._factory:
            fn = self.funcs[fname]
            self._factory[fname] = any(
                isinstance(n, ast.Return)
                and n.value is not None
                and any(_is_psum_alloc(m) for m in ast.walk(n.value))
                for n in ast.walk(fn)
            )
        return self._factory[fname]

    def allocates_psum(self, expr) -> bool:
        for n in ast.walk(expr):
            if _is_psum_alloc(n):
                return True
            if isinstance(n, ast.Call) and _callee(n) in self.funcs and self.factory(_callee(n)):
                return True
        return False

    def events(self, fname: str) -> list:
        """Ordered matmul writes of `fname`: (root, mode, guards, line), where
        root is a local or parameter name, mode OPEN / ACC / ("param", p), and
        guards a tuple of (param, truth) from enclosing `if p:` / `if not p:`."""
        if fname in self._events:
            return self._events[fname]
        fn = self.funcs[fname]
        a = fn.args
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        out: list = []

        def visit(node, guards):
            if isinstance(node, ast.If):
                t = node.test
                g = None
                if isinstance(t, ast.Name) and t.id in params:
                    g = (t.id, True)
                elif (
                    isinstance(t, ast.UnaryOp)
                    and isinstance(t.op, ast.Not)
                    and isinstance(t.operand, ast.Name)
                    and t.operand.id in params
                ):
                    g = (t.operand.id, False)
                visit(t, guards)
                for s in node.body:
                    visit(s, guards + ((g,) if g else ()))
                neg = ((g[0], not g[1]),) if g else ()
                for s in node.orelse:
                    visit(s, guards + neg)
                return
            if isinstance(node, ast.Call):
                for child in ast.iter_child_nodes(node):
                    visit(child, guards)
                self._call_events(node, params, guards, out)
                return
            for child in ast.iter_child_nodes(node):
                visit(child, guards)

        for s in fn.body:
            visit(s, ())
        self._events[fname] = out
        return out

    def _call_events(self, call: ast.Call, params, guards, out):
        name = _callee(call)
        line = call.lineno
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        if name == "nc_matmul":
            dst = kw.get("dst", call.args[0] if call.args else None)
            root = _root(dst) if dst is not None else None
            if root:
                out.append((root, _mode(kw.get("accumulate"), params), guards, line))
            return
        if name not in self.funcs or name == "nc_matmul":
            return
        h = self.funcs[name]
        hparams = [x.arg for x in h.args.posonlyargs + h.args.args]
        bound = dict(zip(hparams, call.args, strict=False))
        bound.update(kw)
        for root, mode, hguards, _ in self.events(name):
            # A branch on a constant argument is decided here.
            keep, rest = True, []
            for p, truth in hguards:
                arg = bound.get(p)
                if isinstance(arg, ast.Constant):
                    keep &= bool(arg.value) == truth
                elif isinstance(arg, ast.Name) and arg.id in params:
                    rest.append((arg.id, truth))
            if not keep:
                continue
            arg = bound.get(root)
            if arg is None:
                continue  # writes a tile the helper allocates itself (checked there)
            r = _root(arg)
            if r is None:
                continue
            if isinstance(mode, tuple):
                mode = _mode(bound.get(mode[1]), params)
            out.append((r, mode, guards + tuple(rest), line))

    def allocations(self, fname: str) -> list[tuple[str, int, bool]]:
        """(name, line, is_zeros) for each `name = <PSUM allocation>` in `fname`."""
        res = []
        for n in ast.walk(self.funcs[fname]):
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                t = n.targets[0]
                if isinstance(t, ast.Name) and self.allocates_psum(n.value):
                    zeros = any(_is_psum_zeros(m) for m in ast.walk(n.value))
                    if not zeros:
                        for m in ast.walk(n.value):
                            if isinstance(m, ast.Call) and _callee(m) in self.funcs:
                                f = self.funcs[_callee(m)]
                                zeros |= any(_is_psum_zeros(x) for x in ast.walk(f))
                    res.append((t.id, n.lineno, zeros))
        return res

    def violations(self) -> tuple[list[str], int]:
        """Unopened accumulations, and the number of PSUM tiles checked."""
        bad, checked = [], 0
        for fname in self.funcs:
            allocs = self.allocations(fname)
            if not allocs:
                continue
            evs = self.events(fname)
            for name, line, zeros in allocs:
                # Events on `name` from this allocation up to its next one.
                nxt = min((ln for n2, ln, _ in allocs if n2 == name and ln > line), default=1 << 30)
                mine = [e for e in evs if e[0] == name and line <= e[3] < nxt]
                if not mine:
                    continue  # not a matmul target (e.g. an nc_transpose tile)
                checked += 1
                opened = False
                for _, mode, _, ln in mine:
                    if mode == OPEN:
                        opened = True
                    elif mode == ACC and not opened:
                        what = "nl.zeros" if zeros else "uninitialised"
                        bad.append(
                            f"{self.where[fname]}:{ln} {fname}: nc_matmul accumulate=True onto "
                            f"{what} PSUM tile {name!r} (line {line}) with no earlier "
                            "accumulate=False write"
                        )
                        break
        return bad, checked


def _lint(sources: dict[str, str]):
    return _Module(sources).violations()


def test_trnblas_psum_groups_open_with_a_matmul():
    files = sorted(NKI_DIR.glob("*.py"))
    assert files
    bad, checked = _lint({p.name: p.read_text() for p in files})
    assert not bad, "\n".join(bad)
    # GEMM tiles, ragged edges, LNC GEMM, SYRK, energy kernels, Ozaki I/II, PE probe.
    assert checked >= 25, checked


# --- the lint on known-good and known-bad kernels ---------------------------

_BAD_ZEROS = """
def k(a, b):
    ps = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
    for k in nl.affine_range(4):
        nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=True)
"""
_BAD_NDARRAY = """
def k(a, b):
    ps = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=True)
"""
_GOOD_PEEL = """
def k(a, b):
    ps = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=False)
    for k in nl.affine_range(3):
        nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=True)
"""
_GOOD_STATIC = """
def k(a, b):
    ps = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
    for kt in range(4):
        nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=kt > 0)
"""
_GOOD_TRANSPOSE = """
def k(a):
    pt = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pt, data=a)
"""
_HELPERS = """
def mk(n):
    return [nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(n)]

def mm(ps, a, b, acc):
    for i in range(len(ps)):
        nisa.nc_matmul(dst=ps[i], stationary=a, moving=b, accumulate=acc)

def run(ps, a, b, n, first):
    if first:
        mm(ps, a, b, False)
        for kk in nl.affine_range(n - 1):
            mm(ps, a, b, True)
    else:
        for kk in nl.affine_range(n):
            mm(ps, a, b, True)
"""
_GOOD_HELPERS = """
def k(a, b):
    ps = mk(2)
    run(ps, a, b, 4, True)
    run(ps, a, b, 4, False)
    ps = mk(2)
    run(ps, a, b, 4, True)
"""
_BAD_HELPERS = """
def k(a, b):
    ps = mk(2)
    run(ps, a, b, 4, True)
    ps = mk(2)
    run(ps, a, b, 4, False)
"""
_BAD_HELPER_ACC = """
def k(a, b):
    ps = mk(2)
    mm(ps, a, b, True)
"""


@pytest.mark.parametrize(
    "src,n_bad",
    [
        (_BAD_ZEROS, 1),
        (_BAD_NDARRAY, 1),
        (_GOOD_PEEL, 0),
        (_GOOD_STATIC, 0),
        (_GOOD_TRANSPOSE, 0),
        (_HELPERS + _GOOD_HELPERS, 0),
        (_HELPERS + _BAD_HELPERS, 1),
        (_HELPERS + _BAD_HELPER_ACC, 1),
    ],
    ids=[
        "zeros-then-acc",
        "ndarray-then-acc",
        "peeled",
        "static-range",
        "transpose-only",
        "helpers-good",
        "helpers-second-group-unopened",
        "helper-acc-only",
    ],
)
def test_lint_examples(src, n_bad):
    bad, _ = _lint({"example.py": textwrap.dedent(src)})
    assert len(bad) == n_bad, bad


def test_lint_flags_the_pre_fix_gemm_tile():
    """The pattern of trnsci#185 as it was in the GEMM tile kernels."""
    src = """
    def _gemm_kernel_128_128_512(a, b):
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    a_t = nl.load_transpose2d(a[0:1, 0:1])
                    b_tile = nl.load(b[0:1, 0:1])
                    nisa.nc_matmul(dst=psum, stationary=a_t, moving=b_tile, accumulate=True)
    """
    bad, checked = _lint({"dispatch.py": textwrap.dedent(src)})
    assert checked == 1 and len(bad) == 1 and "nl.zeros" in bad[0], bad
