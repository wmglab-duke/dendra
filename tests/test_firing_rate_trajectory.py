"""Checks for the experimental per-neuron event-train rate trajectory."""

import math

import pytest
import torch

from dendra.models.analysis import hard_firing_rate
from dendra.models.analysis.firing_rate_trajectory import (
    branch_conditioned_firing_rate_trajectory,
    hard_firing_rate_trajectory,
)


def _trace(values):
    return torch.tensor(values, dtype=torch.float64)[:, None, None]


def test_per_neuron_shape_units_readout_and_hard_parity():
    V = torch.full((8, 2, 2), -2.0, dtype=torch.float64)
    V[2, 0, 0] = 2.0  # event at t=1.5 ms, selected
    V[2, 0, 1] = 2.0  # duplicate event, deliberately unselected
    V[4, 1, 1] = 2.0  # event at t=3.5 ms, selected for neuron 1
    mask = torch.tensor([[1, 0], [0, 1]])
    V.requires_grad_()
    soft = branch_conditioned_firing_rate_trajectory(V, 1.0, mask, tau_ms=1.0)
    hard = hard_firing_rate_trajectory(V, 1.0, mask, tau_ms=1.0)
    hard_count = hard_firing_rate(V, 1.0, mask)["count"]

    assert soft["rate_hz"].shape == (8, 2)
    assert soft["time_ms"].tolist() == list(range(8))
    assert soft["count"].tolist() == [1.0, 1.0]
    assert hard_count.tolist() == pytest.approx([1.0, 1.0])
    assert torch.equal(soft["rate_hz"], hard["rate_hz"])
    assert soft["rate_hz"].requires_grad
    assert not hard["rate_hz"].requires_grad
    assert soft["branch_signature"] == (((1,), None), (None, (3,)))
    # An event at 1.5 ms contributes K(0.5 ms) at the 2 ms sample.
    assert soft["rate_hz"][2, 0].item() == pytest.approx(500 * math.exp(-0.5))
    assert soft["rate_hz"][2, 1].item() == 0.0
    assert soft["rate_hz"][4, 1].item() == pytest.approx(500 * math.exp(-0.5))


def test_multiple_selected_readouts_are_mean_not_duplicate_spike_count():
    V = torch.full((7, 1, 2), -2.0, dtype=torch.float64)
    V[2, 0, 0] = 2.0
    V[2, 0, 1] = 2.0
    both = hard_firing_rate_trajectory(V, 1.0, torch.tensor([1, 1]), tau_ms=1.0)
    one = hard_firing_rate_trajectory(V, 1.0, torch.tensor([1, 0]), tau_ms=1.0)
    assert both["count"].item() == one["count"].item() == 1.0
    assert torch.equal(both["rate_hz"], one["rate_hz"])


def test_event_time_gradient_matches_stable_hard_finite_difference():
    base = torch.tensor([-2.0, -2.0, 2.0, -2.0, -2.0, 2.0, -2.0], dtype=torch.float64)
    change = torch.zeros_like(base)
    change[2] = 0.4  # Shifts the first crossing within its sample pair.
    q = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    soft = branch_conditioned_firing_rate_trajectory(
        (base + q * change)[:, None, None], 1.0, torch.tensor([1]), tau_ms=1.5
    )
    (slope,) = torch.autograd.grad(soft["rate_hz"][2, 0], q)

    def hard_at(value):
        return hard_firing_rate_trajectory(
            (base + value * change)[:, None, None],
            1.0,
            torch.tensor([1]),
            tau_ms=1.5,
        )

    h = 1e-5
    lower, upper = hard_at(-h), hard_at(h)
    assert (
        soft["branch_signature"]
        == lower["branch_signature"]
        == upper["branch_signature"]
    )
    secant = (upper["rate_hz"][2, 0] - lower["rate_hz"][2, 0]) / (2 * h)
    assert slope.item() == pytest.approx(secant.item(), rel=1e-8)
    assert slope.item() != 0.0
    assert soft["count"].item() == lower["count"].item() == upper["count"].item()


def test_existing_spike_shift_changes_trajectory_without_changing_count():
    base = _trace([-2, -2, 2, -2, -2, -2, 2, -2, -2])
    shifted = base.clone()
    shifted[6, 0, 0] = -2
    shifted[7, 0, 0] = 2
    early = hard_firing_rate_trajectory(base, 1.0, torch.tensor([1]), tau_ms=1.0)
    late = hard_firing_rate_trajectory(shifted, 1.0, torch.tensor([1]), tau_ms=1.0)
    assert early["count"].item() == late["count"].item() == 2.0
    assert not torch.equal(early["rate_hz"], late["rate_hz"])
    assert early["rate_hz"][7, 0] > late["rate_hz"][7, 0]


def test_silent_trace_has_no_birth_gradient_and_hard_birth_is_visible():
    base = _trace([-2, -2, -0.1, -2, -2])
    q = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    change = _trace([0, 0, 1, 0, 0])
    soft = branch_conditioned_firing_rate_trajectory(
        base + q * change, 1.0, torch.tensor([1]), tau_ms=1.0
    )
    born = hard_firing_rate_trajectory(
        base + 0.2 * change, 1.0, torch.tensor([1]), tau_ms=1.0
    )
    assert torch.equal(soft["rate_hz"], torch.zeros_like(soft["rate_hz"]))
    assert not soft["rate_hz"].requires_grad
    assert soft["count"].item() == 0.0
    assert born["count"].item() == 1.0
    assert born["rate_hz"].max().item() > 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_invalid_kernel_scale_rejected(bad):
    with pytest.raises(ValueError, match="tau_ms"):
        hard_firing_rate_trajectory(_trace([-1, 1]), 1.0, torch.tensor([1]), tau_ms=bad)


def test_every_neuron_requires_a_selected_readout():
    V = torch.zeros((3, 2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="node_mask"):
        hard_firing_rate_trajectory(V, 1.0, torch.tensor([[1], [0]]), tau_ms=1.0)
