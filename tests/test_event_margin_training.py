"""Synthetic checks for opt-in hard-forward event training directions."""

import pytest
import torch

from dendra.models.analysis import hard_firing_rate
from dendra.models.analysis.event_margin_training import (
    event_crossing_margin,
    hard_forward_event_training_loss,
)


def test_margin_respects_slot_pair_and_site_masks_and_upstroke_gate():
    # Three independent event slots, with one deliberately forbidden crossing.
    voltage = torch.tensor(
        [
            [[-2.0, -2.0], [1.0, -1.0], [-1.0, -1.0]],
            [[-2.0, -2.0], [-1.0, 1.0], [-1.0, -1.0]],
            [[-2.0, -2.0], [-1.0, -1.0], [-1.0, -1.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    valid = torch.tensor(
        [
            [[True, False], [False, False]],
            [[True, False], [True, False]],
            [[True, True], [True, True]],
        ]
    )
    out = event_crossing_margin(voltage, valid_pairs=valid)
    assert out.hard_crossed.tolist() == [True, False, False]
    assert out.margin.tolist() == pytest.approx([1.0, -1.0, -1.0])
    assert out.pair_index.tolist() == [0, 0, 0]
    assert out.site_index.tolist() == [0, 0, 0]

    # At dt=1 ms, the first upward pair rises 3 mV/ms. This gate rejects it.
    gated = event_crossing_margin(
        voltage,
        valid_pairs=valid,
        dv_threshold_mV_per_ms=4.0,
        dt_ms=1.0,
    )
    assert not bool(gated.hard_crossed.any())
    assert gated.margin[0].item() == pytest.approx(-1.0)


def test_event_decision_matches_independent_hard_firing_rate_crossings():
    generator = torch.Generator().manual_seed(412)
    voltage = torch.randn((29, 3, 4), generator=generator, dtype=torch.double)
    mask = torch.tensor([[True, False, True, False]]).expand(3, 4)
    expected = hard_firing_rate(
        voltage,
        0.1,
        node_mask=mask,
        V_spk=0.2,
        use_dv_gate=True,
        dv_spk=8.0,
        time_window=(2, 27),
    )["count_comp"]
    observed = event_crossing_margin(
        voltage[2:27].permute(1, 0, 2),
        threshold_mV=0.2,
        valid_pairs=mask[:, None, :],
        dv_threshold_mV_per_ms=8.0,
        dt_ms=0.1,
    )
    assert torch.equal(observed.hard_crossed, ((expected > 0) & mask).any(dim=-1))
    assert torch.equal(observed.margin >= 0, observed.hard_crossed)


def test_hard_forward_loss_uses_all_parameter_gradients_for_missing_spike():
    q = torch.tensor([0.2, 0.3], dtype=torch.double, requires_grad=True)
    # This subthreshold carrier is a local model for an evoked voltage pair.
    trace = torch.stack((q.new_tensor(-1.0), q.sum() - 1.0)).reshape(1, 2, 1)
    margin = event_crossing_margin(trace).margin
    assert margin.item() == pytest.approx(-0.5)
    hard_event = torch.tensor([False])
    desired = torch.tensor([True])
    hard_task_loss = torch.tensor(7.25, dtype=torch.double, requires_grad=True)
    result = hard_forward_event_training_loss(
        hard_task_loss,
        margin,
        hard_event,
        desired,
        temperature_mV=0.5,
    )
    assert result.loss.item() == 7.25
    assert result.mismatched_events.tolist() == [True]
    gradient = torch.autograd.grad(result.loss, (q, hard_task_loss), allow_unused=True)
    assert gradient[0].shape == (2,)
    assert bool((gradient[0] < 0).all())  # Descent raises both conductance-like q.
    assert gradient[1] is None  # No derivative is claimed for the hard loss.

    q_next = q.detach() - gradient[0].detach()
    next_trace = torch.stack((q.new_tensor(-1.0), q_next.sum() - 1.0)).reshape(1, 2, 1)
    assert event_crossing_margin(next_trace).hard_crossed.item()


def test_hard_forward_loss_preserves_tensor_hard_loss_precision():
    margin = torch.tensor([-0.25], dtype=torch.float32, requires_grad=True)
    hard_task_loss = torch.tensor(
        1.0 + 2.0**-40,
        dtype=torch.float64,
        requires_grad=True,
    )
    result = hard_forward_event_training_loss(
        hard_task_loss,
        margin,
        torch.tensor([False]),
        torch.tensor([True]),
    )
    assert result.loss.dtype == torch.float64
    assert torch.equal(result.loss.detach(), hard_task_loss.detach())
    margin_grad, hard_grad = torch.autograd.grad(
        result.loss,
        (margin, hard_task_loss),
        allow_unused=True,
    )
    assert margin_grad is not None and margin_grad.dtype == torch.float32
    assert margin_grad.item() < 0
    assert hard_grad is None


def test_unwanted_spike_uses_inactive_carrier_to_move_threshold_up():
    q = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    # Complete hard protocol at the training stimulus A=0.9 has a spike.
    current = torch.stack((q.new_tensor(-1.0), q + 0.9 - 1.0)).reshape(1, 2, 1)
    current_margin = event_crossing_margin(current)
    assert current_margin.hard_crossed.item()
    with pytest.raises(ValueError, match="inactive-carrier"):
        hard_forward_event_training_loss(
            1.0,
            current_margin.margin,
            torch.tensor([True]),
            torch.tensor([False]),
        )
    # A nearby A=0.7 trial is inactive. Use its margin as the gradient carrier.
    carrier = torch.stack((q.new_tensor(-1.0), q + 0.7 - 1.0)).reshape(1, 2, 1)
    carrier_margin = event_crossing_margin(carrier).margin
    assert carrier_margin.item() < 0
    result = hard_forward_event_training_loss(
        1.0,
        carrier_margin,
        torch.tensor([True]),
        torch.tensor([False]),
        temperature_mV=0.1,
    )
    grad = torch.autograd.grad(result.loss, q)[0]
    assert grad > 0  # Descent lowers q, raising the current-stimulus threshold.
    q_next = q.detach() - 0.5 * grad.detach()
    next_current = torch.stack((q.new_tensor(-1.0), q_next + 0.9 - 1.0)).reshape(
        1, 2, 1
    )
    assert not event_crossing_margin(next_current).hard_crossed.item()


def test_matching_events_have_zero_trace_gradient_and_fractional_weights_normalize():
    q = torch.tensor([-0.2, -0.5], dtype=torch.double, requires_grad=True)
    matching = hard_forward_event_training_loss(
        0.0,
        q,
        torch.tensor([False, False]),
        torch.tensor([False, False]),
    )
    assert matching.loss.item() == 0.0
    assert torch.equal(torch.autograd.grad(matching.loss, q)[0], torch.zeros_like(q))

    one = hard_forward_event_training_loss(
        2.0,
        q[:1],
        torch.tensor([False]),
        torch.tensor([True]),
        event_weights=torch.tensor([0.1]),
    )
    assert one.trace_objective.item() == pytest.approx(
        torch.nn.functional.softplus(-q[0]).item()
    )


def test_far_inactive_carrier_fails_closed_instead_of_silent_zero_vjp():
    q = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    # A synthetic analogue of an event-absent carrier ~55 mV below threshold.
    # Suppressing a currently active event would have a softplus derivative
    # of sigmoid(-55) / mV, effectively zero at tau=1 mV.
    carrier = (q - 55.0).reshape(1)
    hard = torch.tensor([True])
    target = torch.tensor([False])
    with pytest.raises(ValueError, match="inactive_carrier_saturated"):
        hard_forward_event_training_loss(
            1.0,
            carrier,
            hard,
            target,
            temperature_mV=1.0,
        )

    # A genuinely near-boundary inactive carrier passes and gives a connected
    # suppression direction. This does not certify hard-step improvement.
    near = (q - 2.0).reshape(1)
    result = hard_forward_event_training_loss(
        1.0,
        near,
        hard,
        target,
        temperature_mV=1.0,
    )
    assert result.local_sensitivity_fraction.item() == pytest.approx(
        torch.sigmoid(near.detach()).item()
    )
    assert torch.autograd.grad(result.loss, q)[0].item() > 0.1

    # An explicitly ignored mismatch does not block another valid slot.
    margins = torch.stack((q - 55.0, q - 2.0))
    ignored = hard_forward_event_training_loss(
        1.0,
        margins,
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        event_weights=torch.tensor([0.0, 1.0]),
    )
    assert ignored.loss.item() == 1.0
    assert torch.autograd.grad(ignored.loss, q)[0].item() > 0.1

    with pytest.raises(ValueError, match="min_local_sensitivity_fraction"):
        hard_forward_event_training_loss(
            1.0,
            near,
            hard,
            target,
            min_local_sensitivity_fraction=1.1,
        )


def test_invalid_empty_window_and_label_shapes_are_rejected():
    with pytest.raises(ValueError, match="valid sample pair"):
        event_crossing_margin(
            torch.zeros((1, 2, 1)),
            valid_pairs=torch.zeros((1, 1, 1), dtype=torch.bool),
        )
    q = torch.tensor([0.0], requires_grad=True)
    with pytest.raises(ValueError, match="identical shapes"):
        hard_forward_event_training_loss(
            1.0,
            q,
            torch.tensor([False, False]),
            torch.tensor([True]),
        )
    with pytest.raises(TypeError, match="boolean"):
        hard_forward_event_training_loss(
            1.0,
            q,
            torch.tensor([0]),
            torch.tensor([True]),
        )
