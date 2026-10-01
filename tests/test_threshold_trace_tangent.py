"""The trace proxy has a hard forward value and one-pass parameter slopes."""

import pytest
import torch

from dendra.models.analysis.threshold_trace_tangent import (
    TraceTangentChronaxieResult,
    TraceTangentDiagnostics,
    TraceTangentThresholdResult,
    chronaxie_from_threshold_proxies,
    paired_pulse_recovery_ratio_from_trace_tangents,
    trace_tangent_threshold_proxy,
)


def test_trace_projection_recovers_exact_threshold_slope_for_two_parameters():
    q = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor([1.9], dtype=torch.float64, requires_grad=True)
    threshold = 1.1 + torch.exp(q[0]) + 0.3 * q[1].square()
    basis = torch.tensor([[-2.0, 1.0, 3.0], [4.0, -1.0, 2.0]], dtype=torch.float64)
    baseline = torch.tensor([[7.0, -4.0, 1.0], [2.0, 5.0, -3.0]], dtype=torch.float64)
    # This trace depends on the parameters only through A - T(q).
    voltage = baseline + basis * (amplitude - threshold)
    mask = torch.tensor([[1.0, 0.0, 0.25], [0.0, 2.0, 1.0]], dtype=torch.float64)
    second_amplitude = amplitude.detach() + 0.01
    second_voltage = baseline + basis * (second_amplitude - threshold.detach())

    result = trace_tangent_threshold_proxy(
        threshold.detach(),
        amplitude,
        voltage,
        basis,
        min_tangent_norm=1e-9,
        mask=mask,
        check_amplitude=second_amplitude,
        check_voltage=second_voltage,
        max_relative_linearization_error=1e-10,
    )
    assert result.valid
    assert result.proxy.detach().item() == threshold.detach().item()
    dq, da = torch.autograd.grad(result.proxy, (q, amplitude))
    assert torch.allclose(
        dq,
        torch.tensor([torch.exp(q[0]).item(), 0.6 * q[1].item()], dtype=q.dtype),
        rtol=1e-12,
        atol=1e-12,
    )
    assert da.item() == pytest.approx(0.0, abs=1e-12)
    assert result.diagnostics.selected_samples == 4
    assert result.diagnostics.amplitude_tangent_l2 > 0
    assert result.diagnostics.secant_gain == pytest.approx(1.0, abs=1e-12)
    assert result.diagnostics.relative_linearization_error == pytest.approx(
        0.0, abs=1e-12
    )


def test_trace_projection_rejects_weak_or_nonlinear_amplitude_tangent():
    q = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor(1.0, dtype=torch.float64)
    voltage = torch.stack([q, 2 * q])
    weak = trace_tangent_threshold_proxy(
        0.5,
        amplitude,
        voltage,
        torch.zeros_like(voltage),
        min_tangent_norm=1e-8,
    )
    assert not weak.valid
    assert weak.proxy is None
    assert weak.rejection_reasons == ("amplitude_tangent_too_small",)

    nonlinear = trace_tangent_threshold_proxy(
        0.5,
        amplitude,
        voltage,
        torch.ones_like(voltage),
        min_tangent_norm=1e-8,
        check_amplitude=1.1,
        check_voltage=voltage.detach() + 0.2,
        max_relative_linearization_error=0.1,
    )
    assert not nonlinear.valid
    assert nonlinear.rejection_reasons == ("amplitude_tangent_nonlinear",)
    assert nonlinear.diagnostics.relative_linearization_error == pytest.approx(1.0)
    assert nonlinear.diagnostics.secant_gain == pytest.approx(2.0)


def test_trace_projection_masks_unselected_tangent_samples():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    voltage = torch.stack([q, 2 * q])
    tangent = torch.tensor([2.0, float("nan")], dtype=torch.float64)
    result = trace_tangent_threshold_proxy(
        0.7,
        0.3,
        voltage,
        tangent,
        min_tangent_norm=1.0,
        mask=torch.tensor([True, False]),
    )
    assert result.valid
    (slope,) = torch.autograd.grad(result.proxy, q)
    assert slope.item() == pytest.approx(-0.5)


def test_trace_projection_checks_matching_replay_and_mask():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    voltage = torch.stack([q, 2 * q])
    tangent = torch.ones_like(voltage)
    with pytest.raises(ValueError, match="supplied together"):
        trace_tangent_threshold_proxy(
            0.4,
            0.3,
            voltage,
            tangent,
            min_tangent_norm=1e-8,
            check_voltage=voltage.detach(),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        trace_tangent_threshold_proxy(
            0.4,
            0.3,
            voltage,
            tangent,
            min_tangent_norm=1e-8,
            mask=torch.tensor([-1.0, 1.0]),
        )


def test_trace_projection_preserves_high_precision_python_hard_threshold():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    hard = 0.04595585670322179
    result = trace_tangent_threshold_proxy(
        hard,
        0.04,
        q.reshape(1),
        torch.ones(1, dtype=q.dtype),
        min_tangent_norm=1e-8,
    )
    assert result.valid
    assert result.proxy.item() == hard


def test_conditioned_and_baseline_threshold_proxies_compose_into_ratio():
    conditioned_q = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    baseline_q = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor(0.2, dtype=torch.float64)
    conditioned_threshold = 0.12 + 0.03 * conditioned_q
    baseline_threshold = 0.08 + 0.02 * baseline_q
    conditioned_trace = 4 * (amplitude - conditioned_threshold)
    baseline_trace = 7 * (amplitude - baseline_threshold)
    conditioned = trace_tangent_threshold_proxy(
        conditioned_threshold.detach(),
        amplitude,
        conditioned_trace.reshape(1),
        torch.tensor([4.0], dtype=amplitude.dtype),
        min_tangent_norm=1e-8,
    )
    baseline = trace_tangent_threshold_proxy(
        baseline_threshold.detach(),
        amplitude,
        baseline_trace.reshape(1),
        torch.tensor([7.0], dtype=amplitude.dtype),
        min_tangent_norm=1e-8,
    )
    assert conditioned.valid and baseline.valid
    composition = paired_pulse_recovery_ratio_from_trace_tangents(
        conditioned,
        baseline,
    )
    assert composition.valid
    ratio = composition.ratio
    dc, db = torch.autograd.grad(ratio, (conditioned_q, baseline_q))
    tc = conditioned_threshold.item()
    tb = baseline_threshold.item()
    assert ratio.item() == pytest.approx(tc / tb, rel=1e-14)
    assert dc.item() == pytest.approx(0.03 / tb, rel=1e-14)
    assert db.item() == pytest.approx(-tc * 0.02 / tb**2, rel=1e-14)
    assert (dc + db).item() == pytest.approx(0.03 / tb - tc * 0.02 / tb**2, rel=1e-14)


def test_paired_ratio_rejects_invalid_thresholds_and_nonpositive_baseline():
    diagnostics = TraceTangentDiagnostics(1, 1.0, None, None)
    conditioned = TraceTangentThresholdResult(
        torch.tensor(0.15, dtype=torch.float64, requires_grad=True),
        diagnostics,
        (),
    )
    rejected_baseline = TraceTangentThresholdResult(
        None,
        diagnostics,
        ("amplitude_tangent_too_small",),
    )
    rejected = paired_pulse_recovery_ratio_from_trace_tangents(
        conditioned,
        rejected_baseline,
    )
    assert not rejected.valid
    assert rejected.ratio is None
    assert rejected.rejection_reasons == ("baseline:amplitude_tangent_too_small",)

    for value in (0.0, -0.05):
        baseline = TraceTangentThresholdResult(
            torch.tensor(value, dtype=torch.float64, requires_grad=True),
            diagnostics,
            (),
        )
        result = paired_pulse_recovery_ratio_from_trace_tangents(
            conditioned,
            baseline,
        )
        assert not result.valid
        assert result.rejection_reasons == ("baseline_threshold_nonpositive",)


def test_paired_ratio_rejects_malformed_proxy_and_preserves_exact_value():
    diagnostics = TraceTangentDiagnostics(1, 1.0, None, None)
    conditioned = TraceTangentThresholdResult(
        torch.tensor([0.04595585670322179], dtype=torch.float64, requires_grad=True),
        diagnostics,
        (),
    )
    baseline = TraceTangentThresholdResult(
        torch.tensor(0.03111111111111111, dtype=torch.float64, requires_grad=True),
        diagnostics,
        (),
    )
    result = paired_pulse_recovery_ratio_from_trace_tangents(
        conditioned,
        baseline,
    )
    assert result.valid
    assert result.ratio.item() == (conditioned.proxy.item() / baseline.proxy.item())

    bad = TraceTangentThresholdResult(
        torch.ones(2, dtype=torch.float64, requires_grad=True),
        diagnostics,
        (),
    )
    rejected = paired_pulse_recovery_ratio_from_trace_tangents(bad, baseline)
    assert rejected.rejection_reasons == ("conditioned:proxy_not_scalar_tensor",)
    disconnected = TraceTangentThresholdResult(
        torch.tensor(0.15, dtype=torch.float64),
        diagnostics,
        (),
    )
    rejected = paired_pulse_recovery_ratio_from_trace_tangents(
        disconnected,
        baseline,
    )
    assert rejected.rejection_reasons == (
        "conditioned:proxy_has_no_autograd_connection",
    )


def _threshold_result(proxy: torch.Tensor) -> TraceTangentThresholdResult:
    return TraceTangentThresholdResult(
        proxy,
        TraceTangentDiagnostics(1, 1.0, None, None),
        (),
    )


def test_chronaxie_composition_has_exact_hard_fit_and_parameter_gradient():
    q = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    widths = torch.tensor([0.1, 0.2, 0.5, 1.0], dtype=q.dtype)
    rheobase = 0.8 + 0.1 * q[0] - 0.04 * q[1]
    chronaxie = 0.25 - 0.03 * q[0] + 0.02 * q[1]
    hard_thresholds = rheobase * (1 + chronaxie / widths)
    proxies = tuple(_threshold_result(value) for value in hard_thresholds)

    result = chronaxie_from_threshold_proxies(widths, proxies)
    assert isinstance(result, TraceTangentChronaxieResult)
    assert result.valid
    torch.testing.assert_close(result.thresholds, hard_thresholds, atol=0, rtol=0)
    torch.testing.assert_close(result.rheobase, rheobase, atol=2e-15, rtol=0)
    torch.testing.assert_close(
        result.strength_duration_slope,
        rheobase * chronaxie,
        atol=2e-15,
        rtol=0,
    )
    torch.testing.assert_close(result.chronaxie_ms, chronaxie, atol=2e-15, rtol=0)
    (gradient,) = torch.autograd.grad(result.chronaxie_ms, q)
    torch.testing.assert_close(
        gradient,
        torch.tensor([-0.03, 0.02], dtype=q.dtype),
        atol=2e-14,
        rtol=0,
    )


def test_chronaxie_composition_supports_fixed_fit_weights():
    q = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    widths = [0.1, 0.2, 0.5, 1.0]
    width_tensor = torch.tensor(widths, dtype=q.dtype)
    thresholds = 0.7 + (0.21 + 0.04 * q) / width_tensor
    proxies = [_threshold_result(value) for value in thresholds]
    result = chronaxie_from_threshold_proxies(
        widths,
        proxies,
        weights=[1.0, 0.0, 2.0, 3.0],
    )
    assert result.valid
    assert result.rheobase.item() == pytest.approx(0.7, abs=1e-14)
    (slope,) = torch.autograd.grad(result.chronaxie_ms, q)
    assert slope.item() == pytest.approx(0.04 / 0.7, abs=1e-13)


def test_chronaxie_composition_rejects_invalid_proxy_or_fit():
    diagnostics = TraceTangentDiagnostics(1, 1.0, None, None)
    valid = _threshold_result(
        torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
    )
    invalid = TraceTangentThresholdResult(
        None,
        diagnostics,
        ("amplitude_tangent_too_small",),
    )
    result = chronaxie_from_threshold_proxies([0.1, 0.2], [valid, invalid])
    assert not result.valid
    assert result.rejection_reasons == ("threshold_1:amplitude_tangent_too_small",)

    duplicate = chronaxie_from_threshold_proxies([0.2, 0.2], [valid, valid])
    assert not duplicate.valid
    assert duplicate.rejection_reasons == ("insufficient_distinct_pulse_widths",)

    # Thresholds that increase with duration produce a negative, nonphysical
    # current-domain chronaxie while retaining a positive rheobase.
    increasing = [
        _threshold_result(
            torch.tensor(value, dtype=torch.float64, requires_grad=True),
        )
        for value in (0.9, 1.0, 1.1)
    ]
    nonphysical = chronaxie_from_threshold_proxies(
        [0.1, 0.2, 0.4],
        increasing,
    )
    assert not nonphysical.valid
    assert nonphysical.rejection_reasons == ("chronaxie_nonpositive",)


@pytest.mark.parametrize(
    ("widths", "weights", "reason"),
    [
        ([0.1, float("nan")], None, "pulse_width_nonfinite"),
        ([0.1, 0.0], None, "pulse_width_nonpositive"),
        ([0.1, 0.2], [1.0, float("inf")], "weights_nonfinite"),
        ([0.1, 0.2], [1.0, -1.0], "weights_negative"),
        ([0.1, 0.2], [1.0, 0.0], "insufficient_positive_weight_points"),
    ],
)
def test_chronaxie_composition_reports_protocol_rejections(widths, weights, reason):
    proxies = [
        _threshold_result(
            torch.tensor(value, dtype=torch.float64, requires_grad=True),
        )
        for value in (2.0, 1.5)
    ]
    result = chronaxie_from_threshold_proxies(widths, proxies, weights=weights)
    assert not result.valid
    assert result.rejection_reasons == (reason,)
