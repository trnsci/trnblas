"""The device suites' `_rel` helpers compare complex results in full (#113).

`.double()` of a complex tensor drops its imaginary part (with a torch
UserWarning), so a `_rel` built on it checks only the real part: an error in
the imaginary part of a complex128 result passed. CPU-only: the helpers are
plain torch; the suites that use them run on the device.
"""

import importlib.util
import pathlib

import pytest
import torch

HERE = pathlib.Path(__file__).parent


def _helper(module: str):
    spec = importlib.util.spec_from_file_location(f"_rel_{module}", HERE / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._rel


@pytest.mark.parametrize("module", ["test_nki_native", "test_neuron_span_device"])
def test_rel_sees_the_imaginary_part(module):
    rel = _helper(module)
    g = torch.Generator().manual_seed(113)
    re, im = (
        torch.randn(64, 32, generator=g, dtype=torch.float64),
        torch.randn(64, 32, generator=g, dtype=torch.float64),
    )
    ref = torch.complex(re, im)
    assert rel(ref.clone(), ref) == 0.0
    wrong_im = torch.complex(re, -im)  # right real part, wrong imaginary part
    assert rel(wrong_im, ref) == pytest.approx(float((2 * im).norm() / ref.abs().norm()))
    # A real result for a complex reference is the imaginary part lost.
    assert rel(re, ref) == pytest.approx(float(im.norm() / ref.abs().norm()))
    # complex64 against complex128, and real inputs, as before.
    assert rel(ref.to(torch.complex64), ref) < 1e-6
    assert rel(re.float(), re) < 1e-6
    assert rel(2 * re, re) == pytest.approx(1.0)
