"""An activation trace must expose the selected peak before its check ends."""

import pytest
import torch

from dendra.models.analysis.callback_horizon import callback_peak_horizon_diagnostic
from dendra.models.callbacks import Active, ActiveAL


@pytest.mark.parametrize(
    "site_voltage,expected_observed,expected_steps,expected_drop",
    [
        ([-1.0, -1.0, 0.4, 0.8], False, 0, 0.0),
        ([-1.0, -1.0, 0.8, 0.8], False, 1, 0.0),
        ([-1.0, -1.0, 0.8, 0.2], True, 1, 0.6),
    ],
)
def test_final_or_flat_endpoint_peak_is_censored(
    site_voltage, expected_observed, expected_steps, expected_drop
):
    trace = torch.tensor(site_voltage, dtype=torch.double).reshape(-1, 1, 1)
    callback = Active(threshold=0.0, node_check=[0], dt=0.1)
    result = callback_peak_horizon_diagnostic(callback, trace, dt_ms=0.1)
    assert bool(result.margin.decision.item())
    assert bool(result.peak_observed.item()) is expected_observed
    assert bool(result.horizon_censored.item()) is not expected_observed
    assert result.post_peak_steps.item() == expected_steps
    assert result.terminal_drop_from_peak_mv.item() == pytest.approx(expected_drop)
    assert result.last_checked_step_index == 2
    assert result.last_checked_frame_is_final_trace_frame


def test_callback_window_end_can_censor_even_if_simulation_continues():
    trace = torch.tensor([-1.0, -1.0, 0.4, 0.8, 0.1, -1.0]).reshape(-1, 1, 1)
    callback = Active(threshold=0.0, node_check=[0], dt=1.0, t_end_check=3.0)
    result = callback_peak_horizon_diagnostic(callback, trace, dt_ms=1.0)
    assert result.margin.branch_id == (2, 0)
    assert result.last_checked_step_index == 2
    assert not result.last_checked_frame_is_final_trace_frame
    assert bool(result.horizon_censored.item())


def test_caller_controls_required_post_peak_time_and_drop():
    trace = torch.tensor([-1.0, -1.0, 0.8, 0.2]).reshape(-1, 1, 1)
    callback = Active(node_check=[0], dt=0.1)
    default = callback_peak_horizon_diagnostic(callback, trace, dt_ms=0.1)
    longer = callback_peak_horizon_diagnostic(
        callback, trace, dt_ms=0.1, required_post_peak_ms=0.2
    )
    larger_drop = callback_peak_horizon_diagnostic(
        callback, trace, dt_ms=0.1, required_peak_drop_mv=0.7
    )
    assert bool(default.peak_observed.item())
    assert not bool(longer.peak_observed.item())
    assert not bool(larger_drop.peak_observed.item())


def test_partitioned_active_al_reports_each_selected_event():
    trace = torch.full((4, 1, 3), -1.0)
    trace[2, 0, 0] = 0.7
    trace[3, 0, 0] = 0.1
    trace[3, 0, 1] = 0.5
    trace[2, 0, 2] = 0.6
    trace[3, 0, 2] = 0.1
    callback = ActiveAL(node_check=[0, 1, 2], at_least=1, dt=0.1)
    result = callback_peak_horizon_diagnostic(
        callback, trace, dt_ms=0.1, partition=[1, 2]
    )
    assert result.peak_observed.shape == (1, 2)
    assert result.peak_observed.tolist() == [[True, True]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt_ms": 0.0},
        {"dt_ms": float("nan")},
        {"dt_ms": 0.1, "required_post_peak_ms": -1.0},
        {"dt_ms": 0.1, "required_peak_drop_mv": -1.0},
    ],
)
def test_invalid_clearance_inputs_fail(kwargs):
    trace = torch.zeros((3, 1, 1))
    callback = Active(node_check=[0])
    with pytest.raises(ValueError):
        callback_peak_horizon_diagnostic(callback, trace, **kwargs)


def test_callback_check_step_must_match_replay_step():
    trace = torch.zeros((3, 1, 1))
    callback = Active(node_check=[0], dt=0.2)
    with pytest.raises(ValueError, match="callback.dt must match"):
        callback_peak_horizon_diagnostic(callback, trace, dt_ms=0.1)
