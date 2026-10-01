"""Synthetic branch and censor checks for two-trace activation witnesses."""

import pytest
import torch

from dendra.models.analysis.activation_witness import (
    select_pre_event_activation_witness,
)


def test_selects_closest_pre_event_lower_margin_and_preserves_its_vjp():
    q = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    lower = torch.full((8, 4), -1.0, dtype=torch.double)
    lower[3, 0] = -0.5
    lower[4, 1] = q - 0.1  # Closest eligible, with an interior peak.
    lower[4, 2] = -0.01  # Upper crossing is too late to be upstream.
    lower[5, 3] = -0.02  # Peak is at the last admissible frame.
    upper = torch.full_like(lower, -1.0)
    upper[3, 0] = 1.0
    upper[4, 1] = 1.0
    upper[7, 2] = 1.0
    upper[5, 3] = 1.0

    selected = select_pre_event_activation_witness(
        lower,
        upper,
        monitored_upper_event_sample_index=6,
    )
    assert selected.valid
    assert selected.candidate_count == 2
    assert selected.site_index == 1
    assert selected.lower_peak_sample_index == 4
    assert selected.upper_crossing_pair_index == 3
    assert selected.upper_first_crossing_pair_index == 3
    assert selected.upper_crossing_count == 1
    assert selected.lower_margin.item() == pytest.approx(-0.1)
    assert torch.autograd.grad(selected.lower_margin, q)[0].item() == pytest.approx(1.0)


def test_reports_first_hard_upper_crossing_separately_from_max_margin_pair():
    q = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    lower = torch.full((9, 1), -2.0, dtype=torch.double)
    lower[4, 0] = q - 0.1
    # Pair 0 ends exactly on threshold and counts as a hard crossing. Pair 1
    # starts exactly there, so it must not count under the strict lower test.
    # Pair 7 crosses at the monitored event sample and is not pre-event.
    upper = torch.tensor(
        [[-2.0], [0.0], [2.0], [-2.0], [1.0], [-2.0], [4.0], [-2.0], [8.0]],
        dtype=torch.double,
    )
    result = select_pre_event_activation_witness(
        lower,
        upper,
        monitored_upper_event_sample_index=8,
    )
    assert result.valid
    assert result.upper_first_crossing_pair_index == 0
    assert result.upper_crossing_count == 3  # Pairs 0, 3, and 5.
    assert result.upper_crossing_pair_index == 5  # Largest margin, not first.
    assert result.lower_margin.item() == pytest.approx(-0.1)
    assert torch.autograd.grad(result.lower_margin, q)[0].item() == pytest.approx(1.0)


def test_rejects_boundary_peak_or_missing_pre_event_upper_crossing():
    q = torch.tensor(-0.2, dtype=torch.double, requires_grad=True)
    lower = torch.stack((q.new_tensor(-1.0), q.new_tensor(-1.0), q)).reshape(3, 1)
    upper = torch.tensor([[-1.0], [-1.0], [1.0]], dtype=torch.double)
    result = select_pre_event_activation_witness(
        lower,
        upper,
        monitored_upper_event_sample_index=2,
    )
    assert not result.valid
    assert result.rejection_reasons == ("no_eligible_witness",)

    no_pairs = select_pre_event_activation_witness(
        lower,
        upper,
        monitored_upper_event_sample_index=2,
        valid_pairs=torch.tensor([[False], [True]]),
    )
    assert not no_pairs.valid
    assert no_pairs.rejection_reasons == ("no_pre_event_pairs",)


def test_rejects_lower_trace_with_existing_crossing_or_suprathreshold_peak():
    q = torch.tensor(0.1, dtype=torch.double, requires_grad=True)
    lower = torch.stack(
        (q.new_tensor(-1.0), q, q.new_tensor(-1.0), q.new_tensor(-1.0))
    ).reshape(4, 1)
    upper = torch.tensor([[-1.0], [1.0], [-1.0], [-1.0]], dtype=torch.double)
    result = select_pre_event_activation_witness(
        lower,
        upper,
        monitored_upper_event_sample_index=3,
    )
    assert not result.valid
    assert result.rejection_reasons == ("no_eligible_witness",)
