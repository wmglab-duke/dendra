"""Analytic contracts for Exp2Syn time-constant normalization."""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra.models.mod import exp2syn


def _exp2syn_population(tau1, tau2, *, dtype=torch.float64, parameterized=False):
    population = dn.SingleCompartment(N=1, C=1, dtype=dtype)
    tau1 = torch.tensor(tau1, dtype=dtype)
    tau2 = torch.tensor(tau2, dtype=dtype)
    if parameterized:
        tau1 = torch.nn.Parameter(tau1)
        tau2 = torch.nn.Parameter(tau2)
    population.insert(
        exp2syn.rename("normalization_probe"),
        tau1=tau1,
        tau2=tau2,
    )
    return population


def _initialized_exp2syn(tau1, tau2, *, dtype=torch.float64):
    population = _exp2syn_population(tau1, tau2, dtype=dtype)
    population.initialize()
    return population.mech.normalization_probe


@pytest.mark.parametrize(
    "tau1,tau2,expected_ratio",
    [
        (0.2, 2.0, 0.1),
        (2.0, 2.0, 0.9999),
        (1.999999, 2.0, 0.9999),
        (3.0, 2.0, 0.9999),
        (1.0e-12, 2.0, 1.0e-9),
        (1.0e-8, 2.0, 5.0e-9),
    ],
)
def test_exp2syn_effective_ratio_and_peak_normalization(tau1, tau2, expected_ratio):
    mechanism = _initialized_exp2syn(tau1, tau2)
    effective_tau1 = float(mechanism.DE["A"].tau1.item())
    effective_tau2 = float(mechanism.DE["B"].tau2.item())
    ratio = effective_tau1 / effective_tau2

    assert ratio == pytest.approx(expected_ratio, rel=2.0e-7, abs=1.0e-15)
    assert torch.isfinite(mechanism.factor).all()
    assert torch.all(mechanism.factor > 0)

    peak_time = (
        effective_tau1
        * effective_tau2
        / (effective_tau2 - effective_tau1)
        * math.log(effective_tau2 / effective_tau1)
    )
    denominator = -math.exp(-peak_time / effective_tau1) + math.exp(
        -peak_time / effective_tau2
    )
    assert float(mechanism.factor.item()) == pytest.approx(
        1.0 / denominator, rel=3.0e-11
    )

    weight = 0.37
    mechanism.net_receive(torch.full_like(mechanism.A, weight), netcon=None)
    expected_jump = weight * mechanism.factor
    torch.testing.assert_close(mechanism.A, expected_jump)
    torch.testing.assert_close(mechanism.B, expected_jump)
    peak_conductance = float(mechanism.factor.item()) * weight * denominator
    assert peak_conductance == pytest.approx(weight, rel=3.0e-11)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_exp2syn_equal_taus_stay_finite_at_supported_floating_precisions(dtype):
    mechanism = _initialized_exp2syn(2.0, 2.0, dtype=dtype)
    ratio = mechanism.DE["A"].tau1 / mechanism.DE["B"].tau2

    assert torch.all(ratio > 0)
    assert torch.all(ratio < 1)
    assert torch.isfinite(mechanism.factor).all()
    assert torch.all(mechanism.factor > 0)


def test_exp2syn_effective_tau_drives_a_kinetics_and_preserves_parameter_gradients():
    population = _exp2syn_population(0.2, 2.0, parameterized=True)
    population.train()
    population.initialize()
    mechanism = population.mech.normalization_probe
    state_a = mechanism.DE["A"]
    state_b = mechanism.DE["B"]

    states = {"A": torch.ones_like(mechanism.A), "B": torch.zeros_like(mechanism.B)}
    dt = torch.tensor(0.1, dtype=population.dtype())
    updated = state_a.advance(population.v, dt, states)["A"]
    expected = torch.exp(-dt / state_a.tau1)
    torch.testing.assert_close(updated, expected)

    (updated.sum() + mechanism.factor.sum()).backward()
    for parameter in (state_a.tau1_param, state_b.tau2_param):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
        assert parameter.grad.abs() > 0

    # Reinitialization must repopulate from the live parameters before applying
    # the normalization guard, rather than retaining a stale effective buffer.
    with torch.no_grad():
        state_a.tau1_param.fill_(2.0)
        state_b.tau2_param.fill_(2.0)
    population.initialize()
    assert float(state_a.tau1.item()) == pytest.approx(1.9998)
    assert torch.isfinite(mechanism.factor).all()
    clamped_update = state_a.advance(population.v, dt, states)["A"]
    torch.testing.assert_close(clamped_update, torch.exp(-dt / state_a.tau1))


@pytest.mark.parametrize(
    "tau1,tau2",
    [
        (0.0, 2.0),
        (-1.0, 2.0),
        (float("nan"), 2.0),
        (float("inf"), 2.0),
        (0.1, 0.0),
        (0.1, -2.0),
        (0.1, float("nan")),
        (0.1, float("inf")),
    ],
)
def test_exp2syn_rejects_nonpositive_or_nonfinite_taus(tau1, tau2):
    with pytest.raises(
        ValueError, match="exp2syn tau1 and tau2 must be positive and finite"
    ):
        _initialized_exp2syn(tau1, tau2)
