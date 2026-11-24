"""CPU checks for the device-known conformance categories (trnsci/trnsci#35).

``tests/test_conformance.py`` marks known platform failures ``xfail`` on
device runs only. These tests pin the rule (which cases fall in which
category) and the summary's reason parsing, without a device.
"""

from types import SimpleNamespace

import conftest
import pytest
import test_conformance as tc
import torch


def node(name, **params):
    return SimpleNamespace(originalname=name, name=name, callspec=SimpleNamespace(params=params))


C64, C128, F32, F64 = torch.complex64, torch.complex128, torch.float32, torch.float64


U64, U128 = "complex64-unsupported", "complex128-fp32"


@pytest.mark.parametrize(
    "n, want",
    [
        # gemm: complex64 matmul works; complex128 misses the fp64 tolerance
        # unless the output is empty or exact zeros (k = 0 and beta = 0).
        (node("test_gemm", mkn=(1, 1, 1), bc="beta0", dtype=C128), U128),
        (node("test_gemm", mkn=(0, 4, 5), bc="beta0.5", dtype=C128), None),
        (node("test_gemm", mkn=(4, 0, 5), bc="beta0", dtype=C128), None),
        (node("test_gemm", mkn=(4, 0, 5), bc="beta0_C_nan", dtype=C128), None),
        (node("test_gemm", mkn=(4, 0, 5), bc="beta0.5", dtype=C128), U128),
        (node("test_gemm", mkn=(1, 1, 1), bc="beta1", dtype=C64), None),
        # gemv: complex mv needs both dimensions; with an empty inner
        # dimension only complex128 beta * y misses the tolerance.
        (node("test_gemv", mn=(127, 129), trans=False, by="beta0", dtype=C64), U64),
        (node("test_gemv", mn=(4, 0), trans=False, by="beta0.5", dtype=C64), None),
        (node("test_gemv", mn=(4, 0), trans=False, by="beta0.5", dtype=C128), U128),
        (node("test_gemv", mn=(4, 0), trans="N", by="beta0", dtype=C128), None),
        (node("test_gemv", mn=(0, 4), trans="T", by="beta0.5", dtype=C128), U128),
        (node("test_gemv", mn=(4, 0), trans="T", by="beta0.5", dtype=C128), None),
        # level 1: complex reductions pass; complex128 axpy / scal / dot do not.
        (node("test_level1", op="dot", lay="contig", shape=(127,), dtype=C64), None),
        (node("test_level1", op="dot", lay="contig", shape=(127,), dtype=C128), U128),
        (node("test_level1", op="dot", lay="step", shape=(0,), dtype=C128), None),
        (node("test_level1", op="axpy", lay="contig", shape=(), dtype=C128), U128),
        (node("test_level1", op="scal", lay="step", shape=(0,), dtype=C128), None),
        (node("test_level1", op="nrm2", lay="contig", shape=(1000,), dtype=C128), None),
        (node("test_level1", op="asum", lay="contig", shape=(1000,), dtype=C64), None),
        (node("test_iamax", lay="contig", n=129, dtype=C128), None),
        (node("test_asum_complex_blas", dtype=C64), None),
        (node("test_iamax_complex_blas", dtype=C128), None),
        # symm / trmm triangularise the square A even when the output is empty.
        (node("test_symm", mn=(0, 3), side="right", uplo="upper", bc="beta0", dtype=C64), U64),
        (node("test_symm", mn=(0, 3), side="L", uplo="l", bc="beta0", dtype=C128), None),
        (node("test_symm", mn=(3, 0), side="left", uplo="U", bc="beta0", dtype=C128), U128),
        (node("test_symm", mn=(3, 0), side="r", uplo="lower", bc="beta0", dtype=C64), None),
        (node("test_triangular", op="trmm", mn=(3, 0), flags=("left",), dtype=C64), U64),
        (node("test_triangular", op="trmm", mn=(3, 0), flags=("right",), dtype=C64), None),
        # trsm: the left-side blocked path computes complex correctly; the
        # right side goes through device solve_triangular.
        (node("test_triangular", op="trsm", mn=(127, 3), flags=("left",), dtype=C64), None),
        (node("test_triangular", op="trsm", mn=(127, 3), flags=("left",), dtype=C128), None),
        (node("test_triangular", op="trsm", mn=(127, 3), flags=("right",), dtype=C64), U64),
        (node("test_triangular", op="trsm", mn=(0, 3), flags=("right",), dtype=C128), U128),
        (node("test_triangular", op="trsm", mn=(3, 0), flags=("left",), dtype=C64), None),
        (node("test_triangular", op="trmm", mn=(127, 3), flags=("left",), dtype=F32), None),
        (node("test_triangular", op="trmm", mn=(127, 3), flags=("left",), dtype=F64), None),
        (node("test_symv", n=1, dtype=C64), U64),
        (node("test_symv", n=0, dtype=C128), None),
        (node("test_trmv", n=129, dtype=C128), U128),
        (node("test_ger", mn=(1, 1), dtype=C64), None),
        (node("test_ger", mn=(1, 1), dtype=C128), U128),
        (node("test_ger", mn=(4, 0), dtype=C128), None),
        (node("test_batched", case="3d", dtype=C128), U128),
        (node("test_batched", case="batch0", dtype=C128), None),
        # syrk: the output is n x n for either trans; beta * C reads one
        # triangle of C (complex triu / tril).
        (node("test_syrk", nk=(0, 3), trans=True, Ckind="none", dtype=C128), None),
        (node("test_syrk", nk=(3, 0), trans=True, Ckind="none", dtype=C128), None),
        (node("test_syrk", nk=(3, 0), trans=False, Ckind="symmetric", dtype=C64), U64),
        (node("test_syrk", nk=(3, 0), trans="T", Ckind="symmetric", dtype=C128), U128),
        (node("test_syrk", nk=(127, 3), trans="N", Ckind="none", dtype=C64), None),
        (node("test_syrk", nk=(127, 3), trans="N", Ckind="none", dtype=C128), U128),
        (node("test_syrk", nk=(127, 3), trans=True, Ckind="symmetric_beta0_nan", dtype=C64), None),
        (node("test_syrk_no_warning", uplo="upper", Ckind="diagonal"), "fp64-tolerance-syrk"),
        (node("test_syrk_opposite_triangle_deprecated", uplo="lower"), "fp64-tolerance-syrk"),
        (node("test_batched_layouts", kind="stride0"), "stride0-harness"),
        (node("test_batched_layouts", kind="permuted"), None),
        (
            SimpleNamespace(originalname="test_level1_stride0", name="test_level1_stride0"),
            "stride0-harness",
        ),
    ],
)
def test_category(n, want):
    assert tc._device_known_category(n) == want


def test_every_category_has_an_issue():
    for cat, (issue, what) in tc.DEVICE_KNOWN.items():
        assert issue.startswith("trnsci/trnblas#"), cat
        assert what


def test_summary_categories():
    issue, what = tc.DEVICE_KNOWN["complex64-unsupported"]
    reason = f"[device-known:complex64-unsupported] {issue}: {what}"
    assert conftest._expect_category(reason) == "complex64-unsupported"
    assert conftest._expect_category("trnsci/trnblas#1: pending fix") == "other xfail"
    assert conftest._expect_category(None) == "other xfail"

    def rep(nodeid, **kw):
        return SimpleNamespace(nodeid=nodeid, when="call", **kw)

    stats = {
        "passed": [rep("a")],
        "xfailed": [rep("b", wasxfail=reason), rep("c", wasxfail="other")],
        "xpassed": [rep("d", wasxfail=reason)],
        "failed": [rep("e")],
    }
    s = conftest._expect_summary(stats)
    assert s["passed"] == 1
    assert s["xfailed"] == {"complex64-unsupported": ["b"], "other xfail": ["c"]}
    assert s["xpassed"] == {"complex64-unsupported": ["d"]}
    assert s["new_failures"] == ["e"]
