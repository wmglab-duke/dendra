import math

import numpy as np
import pytest
import torch

from dendra.models.analysis import (
    action_potential_width,
    active,
    chronaxie_from_trials,
    compute_rheobase_chronaxie,
    conduction_velocity,
    firing_rate,
    hard_action_potential_width,
    hard_active,
    hard_chronaxie_from_trials,
    hard_conduction_velocity,
    hard_firing_rate,
    hard_spike_arrival_times,
)

DT_MS = 0.1


def _resting_trace(T, F, C):
    return torch.full((T, F, C), -70.0, dtype=torch.float64)


def _add_square_spike(V, *, fiber, compartment, start, width=1):
    """Add an upstroke between start and start + 1, followed by a downstroke."""
    V[start + 1 : start + 1 + width, fiber, compartment] = 30.0


def _traveling_spike_trace(*, T=40, F=1, C=3, start=4, spacing=2, width=2):
    V = _resting_trace(T, F, C)
    for f in range(F):
        for c in range(C):
            _add_square_spike(
                V,
                fiber=f,
                compartment=c,
                start=start + c * spacing,
                width=width,
            )
    return V


def _chronaxie_trials():
    """Three pulse-width groups with deterministic onset and block boundaries."""
    widths = (0.1, 0.2, 0.4)
    thresholds = (4.0, 3.0, 2.0)
    amplitudes = torch.tensor([1, 2, 3, 4, 5] * len(widths), dtype=torch.float64)
    pws = torch.tensor(
        [width for width in widths for _ in range(5)], dtype=torch.float64
    )
    V = _resting_trace(8, len(amplitudes), 2)
    for trial, (amp, pw) in enumerate(zip(amplitudes.tolist(), pws.tolist())):
        threshold = thresholds[widths.index(pw)]
        # Make the highest strength inactive to exercise block-boundary handling.
        if threshold <= amp < 5.0:
            _add_square_spike(V, fiber=trial, compartment=0, start=2)
    return V, amplitudes, pws


@pytest.mark.parametrize("fit_domain", ["current", "charge", "log_current"])
def test_compute_rheobase_chronaxie_recovers_weiss_parameters(fit_domain):
    pws = np.array([0.1, 0.2, 0.5, 1.0])
    expected_rheobase = 2.0
    expected_chronaxie = 0.3
    thresholds = expected_rheobase * (1.0 + expected_chronaxie / pws)

    out = compute_rheobase_chronaxie(
        pws,
        thresholds,
        fit_domain=fit_domain,
        return_dict=True,
    )

    assert out["rheobase"] == pytest.approx(expected_rheobase, rel=2e-5)
    assert out["chronaxie_ms"] == pytest.approx(expected_chronaxie, rel=2e-5)
    assert out["weiss_fit_domain"] == fit_domain


def test_compute_rheobase_chronaxie_preserves_torch_gradients_and_filters_data():
    pws = torch.tensor([0.1, 0.2, 0.5, float("nan")], dtype=torch.float64)
    thresholds = torch.tensor(
        [8.0, 5.0, 3.2, 1.0], dtype=torch.float64, requires_grad=True
    )
    weights = torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=torch.float64)

    rheobase, chronaxie = compute_rheobase_chronaxie(
        pws, thresholds, weights=weights, fit_domain="current"
    )
    (rheobase + chronaxie).backward()

    assert torch.isfinite(rheobase)
    assert torch.isfinite(chronaxie)
    assert thresholds.grad is not None
    assert torch.isfinite(thresholds.grad[:3]).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pws": [0.1], "thresholds": [2.0]}, "at least two"),
        (
            {"pws": [0.1, 0.2], "thresholds": [2.0], "weights": [1.0, 1.0]},
            "same 1D shape",
        ),
        (
            {"pws": [0.1, 0.2], "thresholds": [2.0, 1.5], "fit_domain": "bad"},
            "fit_domain",
        ),
        (
            {
                "pws": [0.1, 0.2],
                "thresholds": [2.0, 1.5],
                "fit_domain": "log_current",
                "log_current_iterations": -1,
            },
            "iterations",
        ),
    ],
)
def test_compute_rheobase_chronaxie_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_rheobase_chronaxie(**kwargs)


def test_hard_spike_arrival_times_detects_first_crossing_and_windows():
    V = _resting_trace(12, 2, 2)
    _add_square_spike(V, fiber=0, compartment=0, start=2)
    _add_square_spike(V, fiber=0, compartment=1, start=5)

    out = hard_spike_arrival_times(V, DT_MS, V_th=0.0, dv_th=100.0)
    assert torch.equal(
        out["has_crossing"], torch.tensor([[True, True], [False, False]])
    )
    assert out["t_cross_ms"][0, 0].item() == pytest.approx(0.27)
    assert out["t_cross_ms"][0, 1].item() == pytest.approx(0.57)
    assert torch.isnan(out["t_cross_ms"][1]).all()

    windowed = hard_spike_arrival_times(
        V,
        torch.tensor(DT_MS),
        time_window_ms=(0.4, 0.8),
        interpolate=False,
    )
    assert not windowed["has_crossing"][0, 0]
    assert windowed["t_cross_ms"][0, 1].item() == pytest.approx(0.6)

    empty = hard_spike_arrival_times(V, DT_MS, time_window=(2, 3))
    assert not empty["has_crossing"].any()
    assert torch.isnan(empty["t_cross_ms"]).all()

    boundary = _resting_trace(10, 1, 1)
    boundary[7, 0, 0] = 0.0
    aligned = hard_spike_arrival_times(
        boundary,
        0.01,
        time_window_ms=(0.07, 0.075),
    )
    assert aligned["has_crossing"].item()
    assert aligned["t_cross_ms"].item() == pytest.approx(0.07)


def test_hard_spike_arrival_times_validates_shape_and_window_selection():
    V = _resting_trace(4, 1, 1)
    with pytest.raises(ValueError, match="shape"):
        hard_spike_arrival_times(V[:, 0], DT_MS)
    with pytest.raises(ValueError, match="only one"):
        hard_spike_arrival_times(
            V, DT_MS, time_window=(0, 3), time_window_ms=(0.0, 0.3)
        )
    for invalid_dt in (0.0, -0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            hard_spike_arrival_times(V, invalid_dt)
    with pytest.raises(ValueError, match="end > start"):
        hard_spike_arrival_times(V, DT_MS, time_window_ms=(0.3, 0.3))
    with pytest.raises(ValueError, match="finite"):
        hard_spike_arrival_times(V, DT_MS, time_window_ms=(0.0, float("nan")))


def test_hard_active_respects_vector_and_per_fiber_masks():
    V = _resting_trace(10, 2, 2)
    _add_square_spike(V, fiber=0, compartment=0, start=2)
    _add_square_spike(V, fiber=1, compartment=1, start=3)

    vector_mask = hard_active(V, DT_MS, node_mask=torch.tensor([1, 0]))
    assert torch.equal(vector_mask["active"], torch.tensor([True, False]))

    matrix_mask = hard_active(
        V,
        DT_MS,
        node_mask=torch.tensor([[0, 1], [0, 1]]),
        use_dv_gate=False,
    )
    assert torch.equal(matrix_mask["active"], torch.tensor([False, True]))

    with pytest.raises(ValueError, match="node_mask"):
        hard_active(V, DT_MS, node_mask=torch.ones(3))


@pytest.mark.parametrize(
    "aggregate, expected",
    [("mean", 1.5), ("max", 2.0), ("sum", 3.0)],
)
def test_hard_firing_rate_counts_spikes_and_aggregates_compartments(
    aggregate, expected
):
    V = _resting_trace(24, 1, 2)
    _add_square_spike(V, fiber=0, compartment=0, start=2)
    _add_square_spike(V, fiber=0, compartment=0, start=12)
    _add_square_spike(V, fiber=0, compartment=1, start=4)

    out = hard_firing_rate(
        V,
        DT_MS,
        node_mask=torch.tensor([1, 1]),
        use_dv_gate=True,
        aggregate=aggregate,
    )
    assert out["count"].item() == pytest.approx(expected)
    assert out["rate_hz"].item() == pytest.approx(1000.0 * expected / 2.3)


def test_hard_firing_rate_refractory_window_and_validation():
    V = _resting_trace(12, 1, 1)
    _add_square_spike(V, fiber=0, compartment=0, start=2)
    _add_square_spike(V, fiber=0, compartment=0, start=5)

    out = hard_firing_rate(V, DT_MS, refractory_ms=0.4)
    assert out["count"].item() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="aggregate"):
        hard_firing_rate(V, DT_MS, aggregate="bad")
    with pytest.raises(ValueError, match="at least 2"):
        hard_firing_rate(V, DT_MS, time_window=(2, 3))
    with pytest.raises(ValueError, match="only one"):
        hard_firing_rate(V, DT_MS, time_window=(0, 3), time_window_ms=(0.0, 0.3))
    with pytest.raises(ValueError, match="end > start"):
        hard_firing_rate(V, DT_MS, time_window_ms=(0.3, 0.3))
    with pytest.raises(ValueError, match="finite"):
        hard_firing_rate(V, DT_MS, time_window_ms=(0.0, float("nan")))


@pytest.mark.parametrize(
    "lengths",
    [100.0, torch.tensor([100.0, 100.0, 100.0]), torch.full((1, 3), 100.0)],
)
def test_hard_conduction_velocity_recovers_known_speed(lengths):
    V = _traveling_spike_trace(spacing=2)
    out = hard_conduction_velocity(V, lengths, DT_MS, dv_th=100.0)

    assert out["ess"].item() == 3.0
    assert out["v_um_per_ms"].item() == pytest.approx(500.0, rel=1e-10)
    assert out["v_m_per_s"].item() == pytest.approx(0.5, rel=1e-10)


def test_hard_conduction_velocity_returns_nan_with_too_few_arrivals():
    V = _traveling_spike_trace()
    out = hard_conduction_velocity(V, 100.0, DT_MS, node_mask=torch.tensor([1, 0, 0]))
    assert out["ess"].item() == 1.0
    assert torch.isnan(out["v_m_per_s"]).all()

    with pytest.raises(ValueError, match="node_mask"):
        hard_conduction_velocity(V, 100.0, DT_MS, node_mask=torch.ones(2))
    with pytest.raises(ValueError, match="lengths_um"):
        hard_conduction_velocity(V, torch.ones(2), DT_MS)


@pytest.mark.parametrize("mode", ["half_height", "half_peak_to_peak"])
def test_hard_action_potential_width_measures_square_pulse(mode):
    V = _resting_trace(50, 1, 2)
    _add_square_spike(V, fiber=0, compartment=0, start=10, width=4)

    out = hard_action_potential_width(
        V,
        DT_MS,
        node_mask=torch.tensor([1, 0]),
        mode=mode,
        dv_th=100.0,
        baseline_pre_ms=0.8,
        baseline_guard_ms=0.1,
        peak_post_ms=0.8,
        width_pre_ms=0.5,
        width_post_ms=1.0,
        full_width_post_ms=1.2,
    )

    assert out["width_ms"].item() == pytest.approx(0.4, abs=0.02)
    assert out["full_width_ms"].item() == pytest.approx(0.5, abs=0.02)
    assert torch.isnan(out["width_ms_comp"][0, 1])


def test_hard_action_potential_width_validates_mode_and_mask():
    V = _traveling_spike_trace(C=2)
    with pytest.raises(ValueError, match="mode"):
        hard_action_potential_width(V, DT_MS, mode="bad")
    with pytest.raises(ValueError, match="node_mask"):
        hard_action_potential_width(V, DT_MS, node_mask=torch.ones(3))


def _one_and_two_spike_width_traces():
    first = _resting_trace(140, 1, 1)
    spike = torch.tensor(
        [-67.0, -61.0, -40.0, 0.0, 30.0, 25.0, 10.0, -20.0, -45.0, -58.0, -65.0, -70.0],
        dtype=first.dtype,
    )
    first[30:42, 0, 0] = spike
    second = first.clone()
    second[80:92, 0, 0] = spike
    return first, second


def test_full_width_excludes_later_spike_within_postspike_window():
    first, two_spikes = _one_and_two_spike_width_traces()
    hard_options = dict(baseline_margin_mV=10.0, full_width_post_ms=8.0)
    # Fix the first event's localization to isolate the width readout.
    soft_options = dict(
        **hard_options,
        t_hat_ms=torch.tensor([[3.3]], dtype=first.dtype),
        p_spike=torch.ones((1, 1), dtype=first.dtype),
    )

    hard_first = hard_action_potential_width(first, DT_MS, **hard_options)
    hard_two = hard_action_potential_width(two_spikes, DT_MS, **hard_options)
    soft_first = action_potential_width(first, DT_MS, **soft_options)
    soft_two = action_potential_width(two_spikes, DT_MS, **soft_options)

    assert hard_first["full_width_ms"].item() == pytest.approx(0.7984, abs=1e-4)
    assert torch.equal(hard_first["full_width_ms"], hard_two["full_width_ms"])
    assert soft_two["full_width_ms"].item() == pytest.approx(
        soft_first["full_width_ms"].item(), abs=0.04
    )
    assert soft_two["full_width_ms"].item() == pytest.approx(
        hard_two["full_width_ms"].item(), abs=0.12
    )


def test_full_width_first_return_remains_differentiable():
    _, two_spikes = _one_and_two_spike_width_traces()
    V = two_spikes.requires_grad_()
    options = dict(
        baseline_margin_mV=10.0,
        full_width_post_ms=8.0,
        t_hat_ms=torch.tensor([[3.3]], dtype=V.dtype),
        p_spike=torch.ones((1, 1), dtype=V.dtype),
    )

    def width(voltage):
        return action_potential_width(voltage, DT_MS, **options)["full_width_ms"].sum()

    grad = torch.autograd.grad(width(V), V)[0]
    direction = torch.zeros_like(V)
    direction[38:41, 0, 0] = torch.tensor([0.25, 1.0, 0.25], dtype=V.dtype)
    autodiff_slope = (grad * direction).sum().item()

    def centered_slope(step_mV):
        return (
            (
                width(V.detach() + step_mV * direction)
                - width(V.detach() - step_mV * direction)
            )
            / (2 * step_mV)
        ).item()

    one_mV_slope = centered_slope(1.0)
    fine_slope = centered_slope(0.1)

    assert torch.isfinite(grad).all()
    assert grad[29:42].abs().max().item() > 1e-4
    assert abs(autodiff_slope) > 1e-4
    assert one_mV_slope == pytest.approx(autodiff_slope, rel=0.1)
    assert fine_slope == pytest.approx(autodiff_slope, rel=0.01)
    assert abs(fine_slope - autodiff_slope) < abs(one_mV_slope - autodiff_slope)


def test_soft_conduction_velocity_is_differentiable_and_honors_weights():
    V = _traveling_spike_trace().requires_grad_()
    node_mask = torch.tensor([[1.0, 0.5, 1.0]], dtype=V.dtype)
    out = conduction_velocity(
        V,
        lengths_um=100.0,
        dt_ms=torch.tensor(DT_MS),
        node_mask=node_mask,
        beta=1.0,
        use_dv_gate=True,
        reg_ess_weight=0.1,
        reg_var_weight=0.1,
    )

    assert out["t_hat_ms"].shape == (1, 3)
    assert torch.all(out["t_hat_ms"][0, 1:] > out["t_hat_ms"][0, :-1])
    assert torch.isfinite(out["v_m_per_s"]).all()
    assert torch.allclose(out["weights"] / out["p_spike"], node_mask)

    (out["speed_m_per_s"].sum() + out["reg"]).backward()
    assert V.grad is not None
    assert torch.isfinite(V.grad).all()


def test_soft_conduction_velocity_validates_shapes():
    V = _traveling_spike_trace()
    with pytest.raises(AssertionError, match="V must"):
        conduction_velocity(V[:, 0], 100.0, DT_MS)
    with pytest.raises(AssertionError, match="node_mask"):
        conduction_velocity(V, 100.0, DT_MS, node_mask=torch.ones(3))


def test_soft_action_potential_width_reuses_localization_and_returns_traces():
    V = _traveling_spike_trace(T=60, start=14, width=4).requires_grad_()
    localized = conduction_velocity(V, 100.0, DT_MS, beta=1.0)
    out = action_potential_width(
        V,
        torch.tensor(DT_MS),
        node_mask=torch.ones((1, 3), dtype=V.dtype),
        t_hat_ms=localized["t_hat_ms"],
        p_spike=localized["p_spike"],
        baseline_pre_ms=0.8,
        baseline_guard_ms=0.1,
        peak_post_ms=0.8,
        width_pre_ms=0.5,
        width_post_ms=1.0,
        full_width_post_ms=1.2,
        return_tail_ms=0.3,
        reg_ess_weight=0.1,
        reg_amp_weight=0.1,
        reg_full_width_weight=0.1,
        reg_return_weight=0.1,
        return_time_traces=True,
    )

    assert out["width_ms"].shape == (1,)
    assert out["q_half"].shape == V.shape
    assert out["q_full"].shape == V.shape
    assert torch.isfinite(out["full_width_ms"]).all()

    (out["width_ms"].sum() + out["reg"]).backward()
    assert torch.isfinite(V.grad).all()

    trough = action_potential_width(
        V.detach(),
        DT_MS,
        mode="half_peak_to_peak",
        baseline_pre_ms=0.8,
        baseline_guard_ms=0.1,
        peak_post_ms=0.8,
        trough_post_ms=1.0,
        full_width_post_ms=1.2,
        return_tail_ms=0.3,
    )
    assert torch.isfinite(trough["width_ms"]).all()


@pytest.mark.parametrize("aggregate", ["mean", "soft_or", "softmax"])
def test_soft_active_supports_all_aggregations(aggregate):
    V = _resting_trace(12, 2, 2)
    _add_square_spike(V, fiber=0, compartment=0, start=3)
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=V.dtype)

    out = active(
        V,
        DT_MS,
        node_mask=mask,
        aggregate=aggregate,
        use_dv_gate=aggregate != "mean",
        reg_ess_weight=0.1,
    )
    assert out["active"][0] > out["active"][1]
    assert out["reg"] > 0


def test_soft_active_handles_empty_region_and_validates_inputs():
    V = _resting_trace(4, 1, 2)
    out = active(V, DT_MS, node_mask=torch.zeros((1, 2)), aggregate="softmax")
    assert out["active"].item() == 0.0

    with pytest.raises(ValueError, match="aggregate"):
        active(V, DT_MS, aggregate="bad")
    with pytest.raises(ValueError, match="at least 1"):
        active(V, DT_MS, time_window=(1, 1), use_dv_gate=False)
    with pytest.raises(ValueError, match="at least 2"):
        active(V, DT_MS, time_window=(1, 2), use_dv_gate=True)


def test_soft_firing_rate_counts_excursions_and_accepts_precomputed_confidence():
    V = _resting_trace(24, 1, 2)
    _add_square_spike(V, fiber=0, compartment=0, start=2)
    _add_square_spike(V, fiber=0, compartment=0, start=12)
    _add_square_spike(V, fiber=0, compartment=1, start=4)
    mask = torch.ones((1, 2), dtype=V.dtype)

    out = firing_rate(
        V,
        DT_MS,
        node_mask=mask,
        use_dv_gate=True,
        confidence_weighted=True,
        reg_ess_weight=0.1,
    )
    assert out["count_comp"][0, 0] > out["count_comp"][0, 1]
    assert out["rate_hz"].item() > 0

    reused = firing_rate(
        V,
        torch.tensor(DT_MS),
        node_mask=mask,
        p_spike=torch.tensor([[1.0, 0.0]], dtype=V.dtype),
        use_dv_gate=False,
        time_window=(0, 20),
    )
    assert reused["count"].item() == pytest.approx(
        reused["count_comp"].mean().item(), rel=1e-5
    )

    with pytest.raises(ValueError, match="at least 2"):
        firing_rate(V[:1], DT_MS)
    with pytest.raises(ValueError, match="Selected time window"):
        firing_rate(V, DT_MS, time_window=(1, 2))


@pytest.mark.parametrize("fit_domain", ["charge", "current", "log_current"])
def test_hard_chronaxie_from_trials_finds_empirical_boundaries(fit_domain):
    V, amplitudes, pws = _chronaxie_trials()
    out = hard_chronaxie_from_trials(
        V,
        amplitudes,
        pws,
        DT_MS,
        node_mask=torch.tensor([1, 0]),
        use_dv_gate=False,
        pw_round_decimals=3,
        weiss_fit_domain=fit_domain,
    )

    assert torch.allclose(out["I_th"], out["I_th"].new_tensor([3.5, 2.5, 1.5]))
    assert torch.equal(
        out["boundaries"]["I_post_inactive"], out["I_th"].new_full((3,), 5.0)
    )
    assert torch.isfinite(out["chronaxie_ms"])
    assert out["weiss_fit_domain"] == fit_domain


@pytest.mark.parametrize(
    "activation_mode,count_event",
    [
        ("soft_max", "crossing"),
        ("soft_count", "crossing"),
        ("soft_count", "peak"),
        ("soft_fraction", "crossing"),
        ("soft_all", "crossing"),
    ],
)
def test_soft_chronaxie_from_trials_supports_activation_modes(
    activation_mode, count_event
):
    V, amplitudes, pws = _chronaxie_trials()
    V.requires_grad_()
    out = chronaxie_from_trials(
        V,
        amplitudes,
        pws,
        node_mask=torch.tensor([1.0, 0.0]),
        activation_mode=activation_mode,
        count_event=count_event,
        at_least=1,
        use_activation_straight_through=activation_mode == "soft_count",
        threshold_method="onset_midpoint",
        compute_block=True,
        reg_bracket_weight=0.01,
        reg_monotone_weight=0.01,
        reg_unimodal_weight=0.01,
        reg_weiss_fit_weight=0.01,
    )

    assert out["I_th"].shape == (3,)
    assert torch.isfinite(out["chronaxie_ms"])
    assert torch.isfinite(out["reg"])
    (out["chronaxie_ms"] + out["reg"]).backward()
    assert V.grad is not None


@pytest.mark.parametrize("partition_combine", ["any", "all"])
def test_soft_chronaxie_partition_activation(partition_combine):
    V, amplitudes, pws = _chronaxie_trials()
    out = chronaxie_from_trials(
        V,
        amplitudes,
        pws,
        node_mask=torch.ones(2),
        activation_mode="partition_soft_count",
        partition_masks=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        partition_combine=partition_combine,
        at_least=1,
        compute_block=False,
        threshold_method="ptarget",
    )
    assert out["p_spike_trial"].shape == amplitudes.shape
    assert torch.isnan(out["boundaries"]["I_post_inactive"]).all()


def test_analysis_trial_descriptors_validate_shapes_and_options():
    V, amplitudes, pws = _chronaxie_trials()
    with pytest.raises(ValueError, match="shape"):
        hard_chronaxie_from_trials(V[:, :, 0], amplitudes, pws)
    with pytest.raises(ValueError, match="node_mask"):
        hard_chronaxie_from_trials(V, amplitudes, pws, node_mask=torch.ones(3))
    with pytest.raises(ValueError, match="activation_mode"):
        hard_chronaxie_from_trials(V, amplitudes, pws, activation_mode="bad")
    with pytest.raises(ValueError, match="at_least"):
        hard_chronaxie_from_trials(
            V, amplitudes, pws, activation_mode="count", at_least=0
        )
    with pytest.raises(ValueError, match="partition_masks"):
        chronaxie_from_trials(
            V,
            amplitudes,
            pws,
            activation_mode="partition_soft_count",
            partition_masks=torch.ones(2),
        )


def test_analysis_outputs_remain_finite_for_quiet_trace():
    V = _resting_trace(10, 1, 2)
    soft_active = active(V, DT_MS, use_dv_gate=False)
    soft_rate = firing_rate(V, DT_MS, use_dv_gate=False)

    assert math.isfinite(soft_active["active"].item())
    assert math.isfinite(soft_rate["rate_hz"].item())
    assert soft_rate["count"].item() == 0.0
