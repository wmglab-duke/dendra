"""Unit-neutral root checks and compatibility checks for the Active adapter."""

import pytest
import torch

from dendra.models.analysis.threshold_event_root import (
    MarginObservation,
    active_signed_margin,
    bisect_margin_root,
    bisect_signed_margin_root,
    implicit_amplitude_root_proxy,
    implicit_threshold_proxy,
)


def test_active_margin_uses_post_step_samples_in_exact_window():
    # Recorder frame zero is initialized state. Active steps i=1,2 read
    # frames 2,3; positive frames 1 and 4 must not affect this decision.
    trace = torch.tensor([-60.0, 7.0, -5.0, -2.0, 9.0], dtype=torch.float64)
    obs = active_signed_margin(trace, first_checked_step=1, end_checked_step=3)
    assert obs.value == -2.0
    assert obs.peak_index == 3
    trace[3] = 0.0
    assert (
        active_signed_margin(trace, first_checked_step=1, end_checked_step=3).value
        == 0.0
    )


def test_implicit_proxy_matches_analytic_parameter_slope():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    threshold = torch.exp(q)

    def evaluator(x):
        return MarginObservation(
            torch.tensor(x, dtype=torch.float64) - threshold.detach(), 12
        )

    root = bisect_margin_root(
        evaluator, 0.5, 2.0, max_iter=40, max_current_bracket_mA=1e-9
    )
    assert root.valid
    current = torch.tensor([root.root], dtype=torch.float64, requires_grad=True)
    margin = current - torch.exp(q)
    proxy, partial = implicit_threshold_proxy(root, current, margin)
    (slope,) = torch.autograd.grad(proxy, q)
    assert float(proxy.detach()) == pytest.approx(float(threshold.detach()), abs=2e-12)
    assert float(partial) == pytest.approx(1.0)
    assert float(slope) == pytest.approx(float(threshold.detach()), rel=1e-12)


def test_generic_root_uses_caller_amplitude_and_margin_units():
    # The amplitude here is around 1300 in arbitrary units. A derivative
    # tolerance is consequently specified in margin units per amplitude unit.
    q = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def evaluate(amplitude):
        return MarginObservation(
            0.005 * (amplitude - 1250) - 0.7 * float(q.detach()),
            branch_id="selected-response",
        )

    root = bisect_signed_margin_root(
        evaluate,
        1200.0,
        1400.0,
        max_iter=40,
        max_amplitude_bracket=1e-7,
        max_margin_jump=1e-3,
        max_center_residual=1e-3,
    )
    assert root.valid
    assert root.root == pytest.approx(1306.0, abs=1e-8)
    assert root.validation.amplitude_bracket_width <= 1e-7
    assert root.validation.absolute_center_residual <= 1e-3
    assert root.validation.branch_consistent is True
    assert root.validation.secant_margin_partial == pytest.approx(0.005)
    amplitude = torch.tensor(root.root, dtype=torch.float64, requires_grad=True)
    margin = 0.005 * (amplitude - 1250) - 0.7 * q
    proxy, partial = implicit_amplitude_root_proxy(
        root,
        amplitude,
        margin,
        min_positive_margin_partial=1e-6,
    )
    dq, d_amplitude = torch.autograd.grad(proxy, (q, amplitude))
    assert float(proxy.detach()) == pytest.approx(1306.0, abs=1e-8)
    assert partial == pytest.approx(0.005)
    assert float(dq) == pytest.approx(140.0)
    assert float(d_amplitude) == pytest.approx(0.0)


def test_generic_scalar_margin_needs_no_active_peak_index():
    root = bisect_signed_margin_root(
        lambda amplitude: amplitude - 3.0,
        2.0,
        4.0,
        max_iter=30,
        max_amplitude_bracket=1e-8,
        max_margin_jump=1e-7,
        max_center_residual=1e-7,
    )
    assert root.valid
    assert root.validation.branch_consistent is None
    assert root.left.branch_id is None
    assert root.root == pytest.approx(3.0, abs=1e-8)
    amplitude = torch.tensor(root.root, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="does not match"):
        implicit_amplitude_root_proxy(
            root,
            amplitude,
            amplitude - 3.0 + q,
            min_positive_margin_partial=1e-9,
        )


def test_generic_root_rejects_jump_even_if_branch_id_is_unchanged():
    root = bisect_signed_margin_root(
        lambda amplitude: MarginObservation(
            -40.0 if amplitude < 0.37 else 50.0,
            branch_id="same-event",
        ),
        0.3,
        0.4,
        max_iter=30,
        max_amplitude_bracket=1e-9,
        max_margin_jump=1.0,
        max_center_residual=1.0,
    )
    assert not root.valid
    assert "margin_jump" in root.rejection_reasons
    assert "root_residual" in root.rejection_reasons
    assert root.validation.branch_consistent is True
    assert root.validation.margin_span == pytest.approx(90.0)
    amplitude = torch.tensor(root.root, requires_grad=True)
    with pytest.raises(ValueError, match="continuity/branch"):
        implicit_amplitude_root_proxy(
            root,
            amplitude,
            amplitude,
            min_positive_margin_partial=1e-9,
        )


def test_generic_root_rejects_branch_change_and_incomplete_metadata():
    def changed(amplitude):
        return MarginObservation(
            amplitude - 0.37, branch_id="left" if amplitude < 0.37 else "right"
        )

    common = dict(
        max_iter=30,
        max_amplitude_bracket=1e-9,
        max_margin_jump=1.0,
        max_center_residual=1.0,
    )
    root = bisect_signed_margin_root(changed, 0.3, 0.4, **common)
    assert not root.valid
    assert root.rejection_reasons == ("branch_changed",)
    assert root.validation.branch_consistent is False

    def missing(amplitude):
        return MarginObservation(
            amplitude - 0.37, branch_id="event" if amplitude < 0.37 else None
        )

    root = bisect_signed_margin_root(missing, 0.3, 0.4, **common)
    assert not root.valid
    assert "branch_metadata_incomplete" in root.rejection_reasons


def test_generic_root_reports_insufficient_resolution():
    root = bisect_signed_margin_root(
        lambda amplitude: amplitude - 0.25,
        0.0,
        1.0,
        max_iter=2,
        max_amplitude_bracket=0.1,
        max_margin_jump=1.0,
        max_center_residual=1.0,
    )
    assert not root.valid
    assert root.rejection_reasons == ("amplitude_bracket_too_wide",)
    assert root.validation.amplitude_bracket_width == pytest.approx(0.25)
    assert root.validation.iterations == 2


def test_generic_root_rejects_stalled_midpoint_at_machine_precision():
    root = bisect_signed_margin_root(
        lambda amplitude: amplitude - 0.3,
        0.0,
        1.0,
        max_iter=100,
        max_amplitude_bracket=1e-9,
        max_margin_jump=1.0,
        max_center_residual=1.0,
    )
    assert not root.valid
    assert "amplitude_resolution_lost" in root.rejection_reasons


def test_jump_and_peak_switch_reject_implicit_derivative():
    def evaluate(x):
        if x < 0.37:
            return MarginObservation(torch.tensor(-80.0), 1)
        return MarginObservation(torch.tensor(17.0), 243)

    root = bisect_margin_root(
        evaluate, 0.3, 0.4, max_iter=30, max_current_bracket_mA=1e-9
    )
    assert not root.valid
    assert "margin_jump" in root.rejection_reasons
    assert "peak_index_changed" in root.rejection_reasons
    current = torch.tensor([root.root], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="continuity/branch"):
        implicit_threshold_proxy(root, current, current)
