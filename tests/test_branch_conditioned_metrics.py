"""Exact-forward, stable-branch checks for interpolated timing gradients."""

import pytest
import torch

from dendra.models.analysis import (
    _piecewise_linear_window_mean,
    hard_action_potential_width,
    hard_activity_dependent_slowing,
)
from dendra.models.analysis.branch_conditioned_metrics import (
    branch_conditioned_action_potential_width,
    branch_conditioned_activity_dependent_slowing,
    compare_branch_signatures,
)

DT = 0.025
DTYPE = torch.float64


def _ap_trace(q: torch.Tensor, *, shift_ms: float = 0.0) -> torch.Tensor:
    t = torch.arange(240, dtype=DTYPE) * DT
    width = 0.23 * torch.exp(q)
    fibers = []
    for delay in (0.0, 0.08):
        tc = 2.0 + shift_ms + delay
        spike = 112.0 * torch.exp(-((t - tc) / width).square())
        after = 8.0 * torch.exp(-((t - tc - 0.55) / 0.25).square())
        fibers.append((-70.0 + spike - after)[:, None])
    return torch.stack(fibers, dim=1)


def _competing_ap_trace(q: torch.Tensor) -> torch.Tensor:
    times = torch.arange(240, dtype=DTYPE) * DT
    width = 0.18 * torch.exp(0.1 * q)
    early = (70.0 + 40.0 * q) * torch.exp(-((times - 1.8) / width).square())
    late = 110.0 * torch.exp(-((times - 3.8) / width).square())
    return (-70.0 + early + late)[:, None, None]


def _moving_boundary_ap_trace(shift_ms: torch.Tensor) -> torch.Tensor:
    """A drifting baseline and a spike crossing 0 mV near an exact grid time."""
    t = torch.arange(240, dtype=DTYPE) * DT
    voltage = (
        -70.0
        + 10.0 * t
        + 100.0
        * (
            torch.sigmoid((t - (2.0 + shift_ms)) / 0.02)
            - torch.sigmoid((t - (2.5 + shift_ms)) / 0.02)
        )
    )
    return voltage[:, None, None]


def test_piecewise_linear_time_mean_is_continuous_through_sample_boundaries():
    t = torch.arange(100, dtype=DTYPE) * DT
    trace = -70.0 + 10.0 * t
    values = []
    for shift in (-1e-8, 0.0, 1e-8):
        q = torch.tensor(shift, dtype=DTYPE, requires_grad=True)
        mean = _piecewise_linear_window_mean(
            trace,
            torch.tensor(DT, dtype=DTYPE),
            1.0 + q,
            1.85 + q,
        )
        expected = -70.0 + 10.0 * (1.425 + shift)
        assert mean.item() == pytest.approx(expected, abs=1e-10)
        (derivative,) = torch.autograd.grad(mean, q)
        assert derivative.item() == pytest.approx(10.0, abs=1e-8)
        values.append(mean.item())
    assert values[2] - values[0] == pytest.approx(2e-7, abs=1e-10)


def test_continuous_baseline_removes_sample_window_jump_and_matches_hard_forward():
    options = dict(V_spk=0.0, baseline_margin_mV=10.0)
    baseline = {}
    full_width = {}
    signatures = {}
    for shift in (-1e-4, 1e-4):
        voltage = _moving_boundary_ap_trace(torch.tensor(shift, dtype=DTYPE))
        for mean_mode in ("sampled", "continuous_time"):
            hard = hard_action_potential_width(
                voltage,
                DT,
                baseline_mean_mode=mean_mode,
                **options,
            )
            selected = branch_conditioned_action_potential_width(
                voltage,
                DT,
                baseline_mean_mode=mean_mode,
                **options,
            )
            for key in ("V_base_mV", "width_ms", "full_width_ms"):
                torch.testing.assert_close(selected[key], hard[key], rtol=0, atol=0)
            assert selected["branch_matches_hard"].all()
            baseline[(mean_mode, shift)] = hard["V_base_mV"].item()
            full_width[(mean_mode, shift)] = hard["full_width_ms"].item()
            signatures[(mean_mode, shift)] = selected["branch_signature"]
            if mean_mode == "continuous_time":
                assert selected["branch_signature"]["baseline_start"].item() == -2
                assert selected["branch_signature"]["baseline_end"].item() == -2
                for key in (
                    "baseline_start_clipped",
                    "baseline_end_clipped",
                    "baseline_empty",
                ):
                    assert selected["branch_signature"][key].item() == 0

    sampled_jump = abs(baseline[("sampled", 1e-4)] - baseline[("sampled", -1e-4)])
    continuous_change = abs(
        baseline[("continuous_time", 1e-4)] - baseline[("continuous_time", -1e-4)]
    )
    assert sampled_jump > 0.1  # a sample enters/leaves the baseline slice
    assert continuous_change < 0.01  # only the 0.0002-ms crossing shift remains
    sampled_full_change = abs(
        full_width[("sampled", 1e-4)] - full_width[("sampled", -1e-4)]
    )
    continuous_full_change = abs(
        full_width[("continuous_time", 1e-4)] - full_width[("continuous_time", -1e-4)]
    )
    assert continuous_full_change < sampled_full_change / 3
    left = signatures[("continuous_time", -1e-4)]
    right = signatures[("continuous_time", 1e-4)]
    assert right["arrival_pair"].item() - left["arrival_pair"].item() == 1
    assert not compare_branch_signatures(left, right)["all_stable"].all()
    # The pair index changed, but its interpolated crossing and baseline mean
    # followed the same spike continuously through the sample boundary.
    assert continuous_change < 0.01

    voltage = _moving_boundary_ap_trace(torch.tensor(-1e-4, dtype=DTYPE))
    hard_default = hard_action_potential_width(voltage, DT, **options)
    hard_sampled = hard_action_potential_width(
        voltage,
        DT,
        baseline_mean_mode="sampled",
        **options,
    )
    for key in ("V_base_mV", "width_ms", "full_width_ms"):
        torch.testing.assert_close(hard_default[key], hard_sampled[key], rtol=0, atol=0)


def test_continuous_baseline_signature_records_clipping_and_empty_state():
    for shift, expected_empty in ((-1.5, 0), (-1.9, 1)):
        voltage = _moving_boundary_ap_trace(torch.tensor(shift, dtype=DTYPE))
        result = branch_conditioned_action_potential_width(
            voltage,
            DT,
            baseline_mean_mode="continuous_time",
            baseline_margin_mV=10.0,
        )
        signature = result["branch_signature"]
        assert signature["baseline_start"].item() == -2
        assert signature["baseline_end"].item() == -2
        assert signature["baseline_mean_mode"].item() == 1
        assert signature["baseline_start_clipped"].item() == 1
        assert signature["baseline_end_clipped"].item() == 0
        assert signature["baseline_empty"].item() == expected_empty
        assert result["branch_matches_hard"].all()
        assert bool(torch.isnan(result["V_base_mV"]).item()) == bool(expected_empty)


def test_adjacent_peak_sample_switch_can_preserve_hard_width_continuity():
    """A changed peak index alone does not establish a different spike."""
    times = torch.arange(240, dtype=DTYPE) * DT
    results = []
    for shift in (-1e-6, 1e-6):
        center = 2.0125 + shift  # halfway between the 2.000 and 2.025-ms samples
        voltage = (-70.0 + 110.0 * torch.exp(-((times - center) / 0.2).square()))[
            :, None, None
        ]
        options = dict(baseline_mean_mode="continuous_time", baseline_margin_mV=10.0)
        hard = hard_action_potential_width(voltage, DT, **options)
        selected = branch_conditioned_action_potential_width(
            voltage,
            DT,
            **options,
        )
        assert selected["branch_matches_hard"].all()
        for key in ("width_ms", "full_width_ms"):
            torch.testing.assert_close(selected[key], hard[key], rtol=0, atol=0)
        results.append(selected)

    before, after = results
    assert after["branch_signature"]["peak_sample"].item() == (
        before["branch_signature"]["peak_sample"].item() + 1
    )
    assert before["branch_signature"]["arrival_pair"].item() == (
        after["branch_signature"]["arrival_pair"].item()
    )
    assert abs(after["width_ms"].item() - before["width_ms"].item()) < 1e-6
    assert abs(after["full_width_ms"].item() - before["full_width_ms"].item()) < 1e-6


def _ads_trace(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(480, dtype=DTYPE) * DT
    pulses = torch.tensor([1.0, 4.0, 7.0, 10.0], dtype=DTYPE)
    compartments = []
    for c in range(3):
        site = torch.full_like(t, -70.0)
        for p, onset in enumerate(pulses):
            latency = 0.45 + 0.15 * c + 0.05 * p + q * (0.008 * c + 0.018 * p)
            up = torch.sigmoid((t - onset - latency) / 0.035)
            down = torch.sigmoid((t - onset - latency - 0.33) / 0.035)
            site = site + 110.0 * (up - down)
        compartments.append(site)
    return torch.stack(compartments, dim=-1)[:, None, :], pulses


@pytest.mark.parametrize("mode", ["half_height", "half_peak_to_peak"])
@pytest.mark.parametrize("baseline_mean_mode", ["sampled", "continuous_time"])
def test_ap_width_exact_forward_and_voltage_chain_rule(mode, baseline_mean_mode):
    q = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    voltage = _ap_trace(q)
    options = dict(
        V_spk=0.0,
        mode=mode,
        baseline_margin_mV=10.0,
        baseline_mean_mode=baseline_mean_mode,
    )
    hard = hard_action_potential_width(voltage, DT, **options)
    selected = branch_conditioned_action_potential_width(voltage, DT, **options)
    for key in ("width_ms", "full_width_ms", "width_ms_comp", "full_width_ms_comp"):
        torch.testing.assert_close(
            selected[key], hard[key], rtol=0, atol=0, equal_nan=True
        )
    assert selected["branch_matches_hard"].all()
    assert torch.isfinite(selected["width_ms"]).all()
    assert torch.isfinite(selected["full_width_ms"]).all()

    for key in ("width_ms", "full_width_ms"):
        slope = torch.autograd.grad(selected[key].sum(), q, retain_graph=True)[0]
        h = 1e-4
        plus_v = _ap_trace(torch.tensor(h, dtype=DTYPE))
        minus_v = _ap_trace(torch.tensor(-h, dtype=DTYPE))
        plus = hard_action_potential_width(plus_v, DT, **options)[key].sum()
        minus = hard_action_potential_width(minus_v, DT, **options)[key].sum()
        secant = (plus - minus) / (2 * h)
        assert torch.isfinite(slope)
        torch.testing.assert_close(slope, secant, rtol=0.01, atol=1e-4)
        stable = compare_branch_signatures(
            branch_conditioned_action_potential_width(plus_v, DT, **options)[
                "branch_signature"
            ],
            branch_conditioned_action_potential_width(minus_v, DT, **options)[
                "branch_signature"
            ],
        )
        assert stable["all_stable"].all()


def test_ap_width_reports_censored_endpoint_and_event_switch():
    voltage = torch.full((200, 1, 1), -70.0, dtype=DTYPE)
    voltage[80:, 0, 0] = 30.0
    one = branch_conditioned_action_potential_width(voltage, DT, full_width_post_ms=1.0)
    hard = hard_action_potential_width(voltage, DT, full_width_post_ms=1.0)
    torch.testing.assert_close(
        one["full_width_ms"], hard["full_width_ms"], rtol=0, atol=0
    )
    assert one["branch_signature"]["full_fall_kind"].item() == 1
    assert one["branch_matches_hard"].all()

    q = torch.tensor(0.0, dtype=DTYPE)
    before = branch_conditioned_action_potential_width(_ap_trace(q), DT)
    after = branch_conditioned_action_potential_width(_ap_trace(q, shift_ms=0.04), DT)
    stable = compare_branch_signatures(
        before["branch_signature"], after["branch_signature"]
    )
    assert not stable["all_stable"].all()


def test_ap_width_gradient_recomputed_after_selected_spike_switches():
    """A new branch can have a valid local derivative at the next iterate."""
    options = dict(
        V_spk=0.0,
        baseline_margin_mV=10.0,
        full_width_post_ms=1.0,
        amp_min_mV=5.0,
    )

    signatures = []
    for q0 in (-0.5, 0.5):
        q = torch.tensor(q0, dtype=DTYPE, requires_grad=True)
        selected = branch_conditioned_action_potential_width(
            _competing_ap_trace(q),
            DT,
            **options,
        )
        signatures.append(selected["branch_signature"])
        assert selected["branch_matches_hard"].all()

        h = 1e-4
        plus = branch_conditioned_action_potential_width(
            _competing_ap_trace(torch.tensor(q0 + h, dtype=DTYPE)),
            DT,
            **options,
        )
        minus = branch_conditioned_action_potential_width(
            _competing_ap_trace(torch.tensor(q0 - h, dtype=DTYPE)),
            DT,
            **options,
        )
        assert compare_branch_signatures(
            plus["branch_signature"],
            minus["branch_signature"],
        )["all_stable"].all()
        for key in ("width_ms", "full_width_ms"):
            (slope,) = torch.autograd.grad(selected[key].sum(), q, retain_graph=True)
            secant = (plus[key].sum() - minus[key].sum()) / (2 * h)
            torch.testing.assert_close(slope, secant, rtol=0.01, atol=1e-4)

    assert not compare_branch_signatures(
        signatures[0],
        signatures[1],
    )["all_stable"].all()


def test_ap_width_training_steps_reselect_branch_and_reduce_hard_loss():
    options = dict(
        V_spk=0.0,
        baseline_margin_mV=10.0,
        full_width_post_ms=1.0,
        amp_min_mV=5.0,
    )
    target = branch_conditioned_action_potential_width(
        _competing_ap_trace(torch.tensor(0.5, dtype=DTYPE)),
        DT,
        **options,
    )["width_ms"].detach()
    parameter = torch.tensor(-0.5, dtype=DTYPE)
    losses = []
    signatures = []
    for _ in range(3):
        parameter = parameter.detach().requires_grad_()
        measured = branch_conditioned_action_potential_width(
            _competing_ap_trace(parameter),
            DT,
            **options,
        )
        loss = (measured["width_ms"] - target).square().sum()
        (gradient,) = torch.autograd.grad(loss, parameter)
        assert torch.isfinite(gradient)
        assert measured["branch_matches_hard"].all()
        losses.append(float(loss.detach()))
        signatures.append(measured["branch_signature"])
        parameter = parameter - 500.0 * gradient

    assert losses[2] < losses[1] < losses[0]
    assert not compare_branch_signatures(
        signatures[0],
        signatures[1],
    )["all_stable"].all()


def test_ads_exact_forward_and_physical_parameter_directional_derivative():
    q = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    V, pulses = _ads_trace(q)
    options = dict(
        lengths_um=100.0,
        response_window_ms=(0.1, 1.4),
        V_th=-20.0,
        dv_th=None,
        baseline_n_pulses=2,
        tail_n_pulses=2,
    )
    hard = hard_activity_dependent_slowing(V, pulses, DT, **options)
    selected = branch_conditioned_activity_dependent_slowing(V, pulses, DT, **options)
    for key in (
        "arrival_time_ms",
        "latency_ms",
        "baseline_latency_ms",
        "ads_percent",
        "final_ads_percent",
        "tail_ads_percent",
        "v_m_per_s",
        "velocity_change_percent",
        "final_velocity_slowing_percent",
        "tail_velocity_slowing_percent",
        "t_cross_ms_comp",
    ):
        torch.testing.assert_close(
            selected[key], hard[key], rtol=0, atol=0, equal_nan=True
        )
    assert selected["branch_matches_hard"].all()

    h = 1e-4
    plus_v, _ = _ads_trace(torch.tensor(h, dtype=DTYPE))
    minus_v, _ = _ads_trace(torch.tensor(-h, dtype=DTYPE))
    plus = hard_activity_dependent_slowing(plus_v, pulses, DT, **options)
    minus = hard_activity_dependent_slowing(minus_v, pulses, DT, **options)
    sig_plus = branch_conditioned_activity_dependent_slowing(
        plus_v, pulses, DT, **options
    )["branch_signature"]
    sig_minus = branch_conditioned_activity_dependent_slowing(
        minus_v, pulses, DT, **options
    )["branch_signature"]
    assert compare_branch_signatures(sig_plus, sig_minus)["all_stable"].all()
    for key in ("final_ads_percent", "velocity_change_percent"):
        slope = torch.autograd.grad(selected[key].sum(), q, retain_graph=True)[0]
        secant = ((plus[key] - minus[key]) / (2 * h)).sum()
        assert torch.isfinite(slope)
        torch.testing.assert_close(slope, secant, rtol=0.01, atol=1e-5)


def test_ads_split_stabilizers_preserve_hard_forward_and_sign_conventions():
    V, pulses = _ads_trace(torch.tensor(0.0, dtype=DTYPE, requires_grad=True))
    options = dict(
        lengths_um=100.0,
        response_window_ms=(0.1, 1.4),
        V_th=-20.0,
        dv_th=None,
        baseline_n_pulses=2,
        tail_n_pulses=2,
        reference_latency_ms=0.4,
        reference_velocity_m_per_s=0.4,
        eps=1e-12,
        eps_time_ms2=0.005,
        eps_latency_ms=0.2,
        eps_velocity_m_per_s=0.2,
    )
    hard = hard_activity_dependent_slowing(V, pulses, DT, **options)
    selected = branch_conditioned_activity_dependent_slowing(V, pulses, DT, **options)

    for key in (
        "ads_percent",
        "v_m_per_s",
        "velocity_change_percent",
        "velocity_slowing_percent",
    ):
        torch.testing.assert_close(selected[key], hard[key], rtol=0, atol=0)
    torch.testing.assert_close(
        selected["velocity_slowing_percent"],
        -selected["velocity_change_percent"],
        rtol=0,
        atol=0,
    )
    assert selected["branch_matches_hard"].all()


@pytest.mark.parametrize(
    "stabilizer", ["eps", "eps_time_ms2", "eps_latency_ms", "eps_velocity_m_per_s"]
)
@pytest.mark.parametrize("invalid", [-1e-6, float("nan"), float("inf")])
def test_branch_conditioned_ads_rejects_invalid_stabilizers(stabilizer, invalid):
    voltage = torch.zeros((2, 1, 1), dtype=DTYPE)
    pulse_times = torch.zeros(1, dtype=DTYPE)

    with pytest.raises(ValueError, match=stabilizer):
        branch_conditioned_activity_dependent_slowing(
            voltage,
            pulse_times,
            DT,
            **{stabilizer: invalid},
        )


def test_ads_absent_response_remains_nan_and_marks_branch():
    V, pulses = _ads_trace(torch.tensor(0.0, dtype=DTYPE))
    V = V.clone()
    V[round(10.0 / DT) :, 0, 2] = -70.0
    options = dict(
        node_mask=torch.tensor([0, 0, 1]),
        response_window_ms=(0.1, 1.4),
        V_th=-20.0,
        dv_th=None,
    )
    hard = hard_activity_dependent_slowing(V, pulses, DT, **options)
    selected = branch_conditioned_activity_dependent_slowing(V, pulses, DT, **options)
    torch.testing.assert_close(
        selected["latency_ms"], hard["latency_ms"], rtol=0, atol=0, equal_nan=True
    )
    assert torch.isnan(selected["latency_ms"][0, -1])
    assert selected["branch_signature"]["crossing_pair"][0, -1, 2].item() == -1
