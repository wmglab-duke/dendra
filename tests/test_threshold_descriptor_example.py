"""End-to-end check for the public threshold-gradient example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "threshold_descriptor_gradients.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "dendra_threshold_descriptor_example",
        EXAMPLE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu
@pytest.mark.slow
def test_real_hh_threshold_and_chronaxie_gradients_match_hard_searches():
    example = _load_example()
    result = example.run_example()

    torch.testing.assert_close(
        result["proxy_thresholds_mA"],
        result["hard_thresholds_mA"],
        rtol=0.0,
        atol=0.0,
    )
    assert result["proxy_chronaxie_ms"] == pytest.approx(
        result["hard_chronaxie_ms"],
        rel=1e-12,
        abs=1e-12,
    )
    assert bool((result["threshold_relative_errors"] < 0.02).all())
    assert result["chronaxie_relative_error"] < 0.02
    assert bool((result["autograd_threshold_slopes"] < 0).all())
    assert result["autograd_chronaxie_slope"] < 0
    assert bool((result["linearity_errors"] < 0.05).all())
