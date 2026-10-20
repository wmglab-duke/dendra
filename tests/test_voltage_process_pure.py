"""Pure voltage transitions preserve the independent hard/STE fire equations."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import fire, fire_d, fire_r, fire_r_d
from tests.test_builtin_spike_mechanisms import _instance

FIRE_TYPES = {cls.__name__: cls for cls in (fire, fire_r, fire_d, fire_r_d)}
DT = 0.05
STEPS = 6


def _reference_fire(kind, voltage, dt, parameters, values):
    """Independent original equations, including the intentional STE backward."""
    threshold, rest = parameters["threshold"], parameters["rest"]
    hard_spike = voltage > threshold
    available = torch.ones_like(voltage, dtype=torch.bool)
    refractory_before = torch.zeros_like(available)
    updates = {}
    if kind in {"fire_r", "fire_r_d"}:
        timer = torch.where(
            values["is_refractory"],
            values["time_refractory"] - dt,
            values["time_refractory"],
        )
        refractory_before = values["is_refractory"] & (timer > 0)
        available = ~refractory_before
        hard_spike = hard_spike & available
        refractory = refractory_before | hard_spike
        updates.update(
            is_refractory=refractory,
            time_refractory=torch.where(hard_spike, parameters["refractory"], timer),
        )
    else:
        refractory = hard_spike
    hard_voltage = torch.where(refractory, rest, voltage)
    if kind in {"fire", "fire_r"}:
        return hard_voltage, updates
    scale = parameters["ste_scale"]
    gate = torch.sigmoid((voltage - threshold) / parameters["tau_gate"].clamp_min(1e-3))
    spike = gate * available.to(voltage.dtype)
    reset = 1 - (1 - refractory_before.to(voltage.dtype)) * (1 - spike)
    updates["reset_gate"] = refractory.to(voltage.dtype) + scale * (
        reset - reset.detach()
    )
    if kind == "fire_r_d":
        updates["spike_gate"] = hard_spike.to(voltage.dtype) + scale * (
            spike - spike.detach()
        )
    soft_voltage = voltage + reset * (rest - voltage)
    return hard_voltage.detach() + scale * (
        soft_voltage - soft_voltage.detach()
    ), updates


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _parameters(kind, reference):
    result = {
        "threshold": reference.new_tensor(-50.0, requires_grad=True),
        "rest": reference.new_tensor(-65.0, requires_grad=True),
    }
    if kind in {"fire_r", "fire_r_d"}:
        result["refractory"] = reference.new_tensor(0.12, requires_grad=True)
    if kind in {"fire_d", "fire_r_d"}:
        result["tau_gate"] = reference.new_tensor(0.5, requires_grad=True)
        result["ste_scale"] = reference.new_tensor(0.75, requires_grad=True)
    return result


def _carry(kind, voltage):
    result = {}
    if kind in {"fire_r", "fire_r_d"}:
        result.update(
            is_refractory=torch.tensor([[True, True, False]], device=voltage.device),
            time_refractory=voltage.new_tensor([[0.10, 0.05, 0.0]]),
        )
    if kind in {"fire_d", "fire_r_d"}:
        result["reset_gate"] = torch.zeros_like(voltage)
    if kind == "fire_r_d":
        result["spike_gate"] = torch.zeros_like(voltage)
    return result


def _mechanism(kind, voltage):
    with dn.ctx(DTYPE=voltage.dtype, REQUIRE_GRAD=1):
        return _instance(FIRE_TYPES[kind], voltage, **_parameters(kind, voltage))


def _loss(voltage, values):
    loss = (voltage * voltage.new_tensor([[0.7, 1.1, 1.3]])).sum()
    for name, value in values.items():
        if value.is_floating_point():
            loss = loss + (0.2 if name.endswith("gate") else 0.1) * value.sum()
    return loss


@pytest.mark.parametrize("kind", tuple(FIRE_TYPES))
def test_pure_fire_values_and_gradients_use_explicit_carry(kind, cpu_threads):
    voltage = torch.tensor([[-51.0, -50.0, -49.0]], dtype=torch.float64)
    parameters = _parameters(kind, voltage)
    mechanism = _mechanism(kind, voltage)
    values = _carry(kind, voltage)
    # Deliberately disagree with the explicit carry. Reading live module state
    # would therefore produce the wrong reset and refractory countdown.
    for name, value in values.items():
        setattr(mechanism, name, torch.ones_like(value))
    before = {name: value.clone() for name, value in values.items()}
    live_before = {name: getattr(mechanism, name).clone() for name in values}
    actual_v = voltage.clone().requires_grad_()
    actual_dt = voltage.new_tensor(DT, requires_grad=True)
    actual, updates = mechanism.voltage_values(actual_v, actual_dt, values)
    expected_v = voltage.clone().requires_grad_()
    expected_dt = voltage.new_tensor(DT, requires_grad=True)
    raw = {name: getattr(mechanism, name) for name in parameters}
    expected, expected_updates = _reference_fire(
        kind, expected_v, expected_dt, raw, values
    )
    assert updates.keys() == expected_updates.keys()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in updates.items():
        torch.testing.assert_close(value, expected_updates[name], rtol=0, atol=0)
    raw_targets = tuple(getattr(mechanism, name + "_param") for name in parameters)
    actual_gradients = torch.autograd.grad(
        _loss(actual, updates),
        (actual_v, actual_dt, *raw_targets),
        allow_unused=True,
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        _loss(expected, expected_updates),
        (expected_v, expected_dt, *raw_targets),
        allow_unused=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert (actual_gradient is None) == (expected_gradient is None)
        if actual_gradient is not None:
            torch.testing.assert_close(actual_gradient, expected_gradient)
    for name in values:
        torch.testing.assert_close(values[name], before[name], rtol=0, atol=0)
        torch.testing.assert_close(
            getattr(mechanism, name), live_before[name], rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", ("fire", "fire_r"))
def test_hard_fire_smooth_region_voltage_derivative_matches_finite_difference(
    kind, cpu_threads
):
    voltage = torch.tensor([[-55.0, -53.0, -49.0]], dtype=torch.float64)
    mechanism = _mechanism(kind, voltage)
    values = _carry(kind, voltage)
    v = voltage.clone().requires_grad_()
    output, _ = mechanism.voltage_values(v, v.new_tensor(DT), values)
    gradient = torch.autograd.grad(output.sum(), v)[0]
    direction = voltage.new_tensor([[0.5, 1.0, 1.5]])
    step = 1e-5
    losses = [
        mechanism.voltage_values(voltage + sign * step * direction, DT, values)[0]
        .sum()
        .detach()
        for sign in (1, -1)
    ]
    torch.testing.assert_close(
        (gradient * direction).sum(),
        (losses[0] - losses[1]) / (2 * step),
        rtol=1e-8,
        atol=1e-8,
    )
