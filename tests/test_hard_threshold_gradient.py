"""Gradients of hard threshold surfaces despite abrupt spike decisions."""

import math

import pytest
import torch

from dendra.models.analysis.hard_threshold_gradient import (
    HardThresholdBracket,
    hard_threshold_parameter_gradient,
)

CHECKS = dict(
    steps=(0.01, 0.03),
    max_forward_bracket=1e-7,
    max_relative_slope_halfwidth=0.01,
    max_relative_step_disagreement=0.01,
    max_relative_one_sided_disagreement=0.05,
    slope_scale_floor=1e-6,
)


def _binary_activation_threshold(q, formula):
    """The observable jumps from resting to full-spike voltage at threshold."""
    threshold = formula(q)
    lower, upper = 0.0, 4.0
    for _ in range(34):
        middle = (lower + upper) / 2
        peak_voltage = 20.0 if middle >= threshold else -80.0
        if peak_voltage >= 0:
            upper = middle
        else:
            lower = middle
    return HardThresholdBracket(lower, upper)


def test_hard_threshold_slope_survives_all_or_none_peak_jump():
    raw_resistivity = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    coordinate = raw_resistivity.log()

    def formula(q):
        return 1.0 + 0.5 * q

    result = hard_threshold_parameter_gradient(
        lambda q: _binary_activation_threshold(q, formula),
        coordinate,
        **CHECKS,
    )
    assert result.valid
    assert float(result.proxy.detach()) == pytest.approx(
        formula(float(coordinate.detach())), abs=1e-8
    )
    assert result.slope == pytest.approx(0.5, abs=1e-8)
    (gradient,) = torch.autograd.grad(result.proxy, raw_resistivity)
    assert float(gradient) == pytest.approx(0.25, abs=1e-8)


def test_hard_threshold_gradient_rejects_kink_hidden_by_symmetric_secant():
    q = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    result = hard_threshold_parameter_gradient(
        lambda coordinate: _binary_activation_threshold(
            coordinate, lambda x: 1.0 + abs(x)
        ),
        q,
        **CHECKS,
    )
    assert not result.valid
    assert result.rejection_reasons == ("no_consistent_step_pair",)
    assert result.proxy is None
    assert all(
        secant.midpoint_slope == pytest.approx(0.0, abs=1e-8)
        for secant in result.secants
    )
    assert all(secant.relative_one_sided_disagreement > 1 for secant in result.secants)


def test_hard_threshold_gradient_rejects_unresolved_search_brackets():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    result = hard_threshold_parameter_gradient(
        lambda coordinate: HardThresholdBracket(
            1.0 + 0.5 * coordinate - 0.01,
            1.0 + 0.5 * coordinate + 0.01,
        ),
        q,
        **{**CHECKS, "max_forward_bracket": 0.1},
    )
    assert not result.valid
    assert result.rejection_reasons == ("no_consistent_step_pair",)
    assert result.secants[0].relative_slope_halfwidth > 1


def test_coordinate_factory_rebuilds_graph_after_hard_model_mutations():
    raw_resistivity = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    baseline = float(raw_resistivity.detach())

    def coordinate():
        return torch.log(raw_resistivity / baseline)

    def search(q):
        with torch.no_grad():
            raw_resistivity.fill_(baseline * math.exp(q))
        try:
            return _binary_activation_threshold(q, lambda x: 1.0 + 0.5 * x)
        finally:
            with torch.no_grad():
                raw_resistivity.fill_(baseline)

    result = hard_threshold_parameter_gradient(search, coordinate, **CHECKS)
    assert result.valid
    (raw_slope,) = torch.autograd.grad(result.proxy, raw_resistivity)
    assert float(raw_slope) == pytest.approx(0.25, abs=1e-8)


def test_coordinate_factory_detects_evaluator_that_leaves_parameter_changed():
    raw = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def search(q):
        with torch.no_grad():
            raw.fill_(2.0 * math.exp(q))
        return _binary_activation_threshold(q, lambda x: 1.0 + 0.5 * x)

    with pytest.raises(ValueError, match="did not restore"):
        hard_threshold_parameter_gradient(
            search,
            lambda: torch.log(raw / 2.0),
            **CHECKS,
        )
