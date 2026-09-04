"""Hard-forward and surrogate-gradient contracts for built-in spike mechanisms."""

from __future__ import annotations

import pytest
import torch

from dendra.models.mod import (
    apcount,
    apcount_d,
    fire,
    fire_d,
    fire_r,
    fire_r_d,
    spikedetect,
)

DTYPE = torch.float64


def _instance(cls, v_init, *, tensor_dtype=DTYPE, **parameters):
    v_init = torch.as_tensor(v_init, dtype=tensor_dtype)
    if v_init.ndim == 1:
        v_init = v_init.unsqueeze(0)
    mechanism = cls(
        cls.__name__,
        torch.full_like(v_init, 34.0),
        torch.ones_like(v_init),
        tuple(v_init.shape),
        tuple(v_init.shape),
        **parameters,
    )
    mechanism.populate()
    mechanism._init_buffers_s(v_init)
    return mechanism


def _advance(mechanism, voltage):
    mechanism._advance_states(voltage, voltage.new_tensor(1.0))


def test_fire_uses_strict_threshold_and_differentiable_forward_is_exact():
    voltage = torch.tensor([[-51.0, -50.0, -49.0]], dtype=DTYPE)
    hard = _instance(fire, voltage, threshold=-50.0, rest=-65.0)
    differentiable = _instance(
        fire_d,
        voltage,
        threshold=-50.0,
        rest=-65.0,
        tau_gate=0.5,
        ste_scale=1.0,
    )

    expected = torch.tensor([[-51.0, -50.0, -65.0]], dtype=DTYPE)
    torch.testing.assert_close(hard.update_v(voltage), expected)
    torch.testing.assert_close(differentiable.update_v(voltage), expected)
    torch.testing.assert_close(
        differentiable.reset_gate, torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    )


@pytest.mark.parametrize("tau_gate", (0.0, 0.5))
def test_fire_d_surrogate_gradient_matches_closed_form(tau_gate):
    voltage = torch.tensor([[-51.0, -50.0, -49.0]], dtype=DTYPE, requires_grad=True)
    mechanism = _instance(
        fire_d,
        voltage.detach(),
        threshold=-50.0,
        rest=-65.0,
        tau_gate=tau_gate,
        ste_scale=0.75,
    )

    output = mechanism.update_v(voltage)
    output.sum().backward()

    tau = max(tau_gate, 1.0e-3)
    gate = torch.sigmoid((voltage.detach() + 50.0) / tau)
    expected_gradient = 0.75 * (
        1.0 - gate + gate * (1.0 - gate) * (-65.0 - voltage.detach()) / tau
    )
    torch.testing.assert_close(
        voltage.grad, expected_gradient, atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.isfinite(voltage.grad).all()


def test_refractory_fire_hard_and_surrogate_forward_lifecycle_match():
    initial = torch.tensor([[-51.0, -50.0, -49.0]], dtype=DTYPE)
    hard = _instance(fire_r, initial, threshold=-50.0, rest=-65.0, refractory=2.0)
    differentiable = _instance(
        fire_r_d,
        initial,
        threshold=-50.0,
        rest=-65.0,
        refractory=2.0,
        tau_gate=0.5,
        ste_scale=1.0,
    )
    hard._configure_timestep(1.0)
    differentiable._configure_timestep(1.0)

    sequence = (
        initial,
        torch.full_like(initial, -49.0),
        torch.full_like(initial, -60.0),
        torch.full_like(initial, -60.0),
    )
    expected = (
        torch.tensor([[-51.0, -50.0, -65.0]], dtype=DTYPE),
        torch.full_like(initial, -65.0),
        torch.tensor([[-65.0, -65.0, -60.0]], dtype=DTYPE),
        torch.full_like(initial, -60.0),
    )
    for voltage, expected_voltage in zip(sequence, expected):
        hard_voltage = hard.update_v(voltage)
        diff_voltage = differentiable.update_v(voltage)
        torch.testing.assert_close(hard_voltage, expected_voltage)
        torch.testing.assert_close(diff_voltage, expected_voltage)
        assert torch.equal(hard.is_refractory, differentiable.is_refractory)
        torch.testing.assert_close(hard.time_refractory, differentiable.time_refractory)

    assert not hard.is_refractory.any()
    assert not differentiable.is_refractory.any()


def test_fire_r_d_gradients_exist_only_when_available_to_spike():
    initial = torch.tensor([[-51.0]], dtype=DTYPE)
    mechanism = _instance(
        fire_r_d,
        initial,
        threshold=-50.0,
        rest=-65.0,
        refractory=2.0,
        tau_gate=0.5,
    )
    mechanism._configure_timestep(1.0)

    crossing = torch.tensor([[-49.0]], dtype=DTYPE, requires_grad=True)
    output = mechanism.update_v(crossing)
    assert output.item() == -65.0
    assert mechanism.spike_gate.item() == 1.0
    (output + mechanism.spike_gate).sum().backward()
    assert torch.isfinite(crossing.grad).all()
    assert crossing.grad.abs().item() > 0.0

    held_high = torch.tensor([[-49.0]], dtype=DTYPE, requires_grad=True)
    held_output = mechanism.update_v(held_high)
    assert held_output.item() == -65.0
    assert mechanism.spike_gate.item() == 0.0
    (held_output + mechanism.spike_gate).sum().backward()
    torch.testing.assert_close(held_high.grad, torch.zeros_like(held_high))


def test_apcount_variants_count_only_upward_crossings():
    initial = torch.tensor([[-1.0, 0.0, 1.0]], dtype=DTYPE)
    hard = _instance(apcount, initial, threshold=0.0)
    differentiable = _instance(
        apcount_d,
        initial,
        threshold=0.0,
        tau_gate=0.5,
        ste_scale=1.0,
    )

    assert hard.n.dtype == torch.float32
    assert differentiable.n.dtype == DTYPE
    _advance(hard, initial)
    _advance(differentiable, initial)
    torch.testing.assert_close(hard.n, torch.zeros_like(initial, dtype=torch.float32))
    torch.testing.assert_close(differentiable.n, torch.zeros_like(initial))

    expected_counts = (
        torch.tensor([[1.0, 1.0, 0.0]], dtype=DTYPE),
        torch.tensor([[1.0, 1.0, 0.0]], dtype=DTYPE),
        torch.tensor([[1.0, 1.0, 0.0]], dtype=DTYPE),
        torch.tensor([[2.0, 2.0, 1.0]], dtype=DTYPE),
    )
    for voltage, expected in zip(
        (
            torch.ones_like(initial),
            torch.ones_like(initial),
            -torch.ones_like(initial),
            torch.ones_like(initial),
        ),
        expected_counts,
    ):
        _advance(hard, voltage)
        _advance(differentiable, voltage)
        torch.testing.assert_close(hard.n, expected.to(torch.float32))
        torch.testing.assert_close(differentiable.n, expected)


@pytest.mark.parametrize(
    ("dtype", "last_exact_integer"),
    ((torch.float16, 2048), (torch.bfloat16, 256)),
)
def test_hard_apcount_does_not_saturate_with_low_precision_voltage(
    dtype, last_exact_integer
):
    initial = torch.tensor([[-1.0]], dtype=dtype)
    counter = _instance(apcount, initial, tensor_dtype=dtype, threshold=0.0)
    counter.n.fill_(last_exact_integer)

    _advance(counter, torch.tensor([[1.0]], dtype=dtype))

    assert counter.n.dtype == torch.float32
    assert counter.n.item() == last_exact_integer + 1


def test_explicit_carry_dtype_survives_module_dtype_conversion():
    counter = _instance(apcount, torch.tensor([[-1.0]]), threshold=0.0)

    counter.to(dtype=torch.float16)

    assert counter.n.dtype == torch.float32
    assert counter.active.dtype == torch.bool


def test_apcount_d_and_spikedetect_emit_hard_events_with_surrogate_gradients():
    initial = torch.tensor([[-1.0]], dtype=DTYPE)
    counter = _instance(apcount_d, initial, threshold=0.0, tau_gate=0.5)
    detector = _instance(spikedetect, initial, threshold=0.0, tau_gate=0.5)

    counter_voltage = torch.tensor([[0.1]], dtype=DTYPE, requires_grad=True)
    detector_voltage = torch.tensor([[0.1]], dtype=DTYPE, requires_grad=True)
    _advance(counter, counter_voltage)
    _advance(detector, detector_voltage)

    assert counter.spikes.item() == detector.spikes.item() == 1.0
    assert counter.n.item() == 1.0
    counter.n.sum().backward()
    detector.spikes.sum().backward()
    assert counter_voltage.grad.item() > 0.0
    assert detector_voltage.grad.item() > 0.0
    assert torch.isfinite(counter_voltage.grad).all()
    assert torch.isfinite(detector_voltage.grad).all()

    _advance(counter, torch.tensor([[0.2]], dtype=DTYPE))
    _advance(detector, torch.tensor([[0.2]], dtype=DTYPE))
    assert counter.spikes.item() == detector.spikes.item() == 0.0
    assert counter.n.item() == 1.0


def test_spikedetect_initialization_above_threshold_does_not_emit_event():
    initial = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    detector = _instance(spikedetect, initial, threshold=0.0, tau_gate=0.5)

    _advance(detector, initial)
    torch.testing.assert_close(detector.spikes, torch.zeros_like(initial))
    _advance(detector, torch.tensor([[-1.0, -1.0]], dtype=DTYPE))
    _advance(detector, torch.tensor([[1.0, 1.0]], dtype=DTYPE))
    torch.testing.assert_close(detector.spikes, torch.ones_like(initial))
