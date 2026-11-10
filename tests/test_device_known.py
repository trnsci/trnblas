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


@pytest.mark.parametrize(
    "n, want",
    [
        (node("test_gemm", mkn=(1, 1, 1), dtype=C128), "complex128-fp32"),
        (node("test_gemm", mkn=(0, 4, 5), dtype=C128), None),  # empty output
        (node("test_gemm", mkn=(4, 0, 5), dtype=C128), "complex128-fp32"),  # k = 0, not empty
        (node("test_gemm", mkn=(1, 1, 1), dtype=C64), None),  # complex matmul works
        (node("test_gemv", mn=(127, 129), trans=False, dtype=C64), "complex64-unsupported"),
        (node("test_gemv", mn=(4, 0), trans="T", dtype=C64), None),
        (node("test_gemv", mn=(4, 0), trans=False, dtype=C64), "complex64-unsupported"),
        (
            node("test_level1", op="dot", lay="contig", shape=(0,), dtype=C64),
            "complex64-unsupported",
        ),
        (node("test_level1", op="axpy", lay="contig", shape=(127,), dtype=C64), None),
        (node("test_level1", op="scal", lay="step", shape=(0,), dtype=C128), None),
        (node("test_syrk", nk=(0, 3), trans=True, dtype=C128), "complex128-fp32"),
        (node("test_syrk", nk=(0, 3), trans=False, dtype=C128), None),
        (node("test_triangular", mn=(127, 3), dtype=C64), "complex64-unsupported"),
        (node("test_triangular", mn=(127, 3), dtype=F32), None),
        (node("test_triangular", mn=(127, 3), dtype=F64), None),
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
