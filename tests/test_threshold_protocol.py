"""Protocol-neutral threshold composition and gradient checks."""

import pytest
import torch

from dendra.models.analysis import compute_rheobase_chronaxie
from dendra.models.analysis.threshold_protocol import (
    DifferentiableThresholdTrial,
    ThresholdTrial,
    differentiable_protocol_threshold,
    stack_valid_thresholds,
)

ROOT_OPTIONS = dict(
    max_amplitude_bracket=1e-8,
    max_margin_jump=1e-3,
    max_center_residual=1e-3,
    max_replay_margin_difference=1e-10,
    min_positive_margin_partial=1e-12,
    max_iter=32,
)


def test_generic_protocol_roots_compose_into_chronaxie_gradient():
    q = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    widths = torch.tensor([0.1, 0.2, 0.4, 0.8], dtype=torch.float64)
    rheobase = 1.0 + 0.2 * q
    inverse_width_coefficient = 0.4 + 0.1 * q
    results = []

    for width in widths:

        def hard_trial(amplitude):
            margin = amplitude - (
                rheobase.detach() + inverse_width_coefficient.detach() / width
            )
            return ThresholdTrial(
                decision=bool(margin >= 0),
                margin=margin,
                branch_id="selected_event",
            )

        def differentiable_trial(amplitude):
            amplitude_leaf = torch.tensor(
                amplitude,
                dtype=torch.float64,
                requires_grad=True,
            )
            margin = amplitude_leaf - (rheobase + inverse_width_coefficient / width)
            return DifferentiableThresholdTrial(
                amplitude_leaf,
                margin,
                branch_id="selected_event",
            )

        result = differentiable_protocol_threshold(
            hard_trial,
            differentiable_trial,
            0.0,
            8.0,
            **ROOT_OPTIONS,
        )
        assert result.valid
        assert result.amplitude_partial == pytest.approx(1.0)
        assert float(result.threshold.detach()) == pytest.approx(
            float((rheobase + inverse_width_coefficient / width).detach()),
            abs=1e-8,
        )
        results.append(result)

    thresholds = stack_valid_thresholds(results)
    fit = compute_rheobase_chronaxie(
        widths,
        thresholds,
        fit_domain="current",
        return_dict=True,
    )
    chronaxie = fit["chronaxie_ms"]
    expected_chronaxie = inverse_width_coefficient / rheobase
    torch.testing.assert_close(chronaxie, expected_chronaxie, rtol=1e-8, atol=1e-8)
    (slope,) = torch.autograd.grad(chronaxie, q)
    expected_slope = (
        0.1 * rheobase - 0.2 * inverse_width_coefficient
    ) / rheobase.square()
    torch.testing.assert_close(slope, expected_slope, rtol=1e-10, atol=1e-10)


def test_generic_paired_threshold_ratio_differentiates_both_searches():
    q = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    intervals = torch.tensor([5.0, 10.0, 20.0], dtype=torch.float64)
    reference_results = []
    conditioned_results = []

    def search(formula):
        def hard_trial(amplitude):
            margin = amplitude - formula.detach()
            return ThresholdTrial(bool(margin >= 0), margin)

        def differentiable_trial(amplitude):
            leaf = torch.tensor(amplitude, dtype=torch.float64, requires_grad=True)
            return DifferentiableThresholdTrial(leaf, leaf - formula)

        return differentiable_protocol_threshold(
            hard_trial,
            differentiable_trial,
            0.0,
            5.0,
            **ROOT_OPTIONS,
        )

    for interval in intervals:
        reference_results.append(search(1.0 + 0.2 * q + 0.1 / interval))
        conditioned_results.append(search(2.0 + 0.5 * q + 0.2 / interval))

    reference = stack_valid_thresholds(reference_results)
    conditioned = stack_valid_thresholds(conditioned_results)
    ratio = conditioned / reference
    (slope,) = torch.autograd.grad(ratio.sum(), q)
    expected_slope = (
        0.5 * reference.detach() - 0.2 * conditioned.detach()
    ) / reference.detach().square()
    torch.testing.assert_close(slope, expected_slope.sum(), rtol=1e-10, atol=1e-10)


def test_protocol_rejects_hard_decision_margin_mismatch():
    def incorrect_hard_trial(amplitude):
        return ThresholdTrial(decision=False, margin=amplitude - 1.0)

    with pytest.raises(ValueError, match="Signed margin and hard decision disagree"):
        differentiable_protocol_threshold(
            incorrect_hard_trial,
            lambda _: None,
            0.5,
            1.5,
            **ROOT_OPTIONS,
        )


def test_protocol_returns_explicit_invalid_status_for_jump_and_replay_mismatch():
    def jump_trial(amplitude):
        active = amplitude >= 1.0
        return ThresholdTrial(
            active, 20.0 if active else -80.0, branch_id="spike" if active else "rest"
        )

    jump = differentiable_protocol_threshold(
        jump_trial,
        lambda _: (_ for _ in ()).throw(AssertionError("invalid root was rerun")),
        0.5,
        1.5,
        **ROOT_OPTIONS,
    )
    assert not jump.valid
    assert jump.threshold is None
    assert "margin_jump" in jump.rejection_reasons
    assert "branch_changed" in jump.rejection_reasons
    with pytest.raises(ValueError, match="Invalid threshold results"):
        stack_valid_thresholds([jump])

    def hard_trial(amplitude):
        return ThresholdTrial(amplitude >= 1.0, amplitude - 1.0)

    def inconsistent_replay(amplitude):
        leaf = torch.tensor(amplitude, dtype=torch.float64, requires_grad=True)
        return DifferentiableThresholdTrial(leaf, leaf - 0.8)

    replay = differentiable_protocol_threshold(
        hard_trial,
        inconsistent_replay,
        0.5,
        1.5,
        **ROOT_OPTIONS,
    )
    assert not replay.valid
    assert replay.rejection_reasons == ("differentiable_replay_mismatch",)


def test_protocol_rejects_autograd_rerun_on_different_event_branch():
    def hard_trial(amplitude):
        return ThresholdTrial(
            amplitude >= 1.0, amplitude - 1.0, branch_id="first_spike"
        )

    def changed_branch(amplitude):
        leaf = torch.tensor(amplitude, dtype=torch.float64, requires_grad=True)
        return DifferentiableThresholdTrial(
            leaf,
            leaf - 1.0,
            branch_id="second_spike",
        )

    result = differentiable_protocol_threshold(
        hard_trial,
        changed_branch,
        0.5,
        1.5,
        **ROOT_OPTIONS,
    )
    assert not result.valid
    assert result.rejection_reasons == ("differentiable_replay_branch_mismatch",)


def test_protocol_can_reject_replay_with_wrong_amplitude_slope():
    def hard_trial(amplitude):
        return ThresholdTrial(amplitude >= 1.0, amplitude - 1.0)

    def replay_with_wrong_slope(amplitude):
        leaf = torch.tensor(amplitude, dtype=torch.float64, requires_grad=True)
        # Match the hard margin at the root but double its local A derivative.
        return DifferentiableThresholdTrial(
            leaf, 2 * (leaf - amplitude) + amplitude - 1.0
        )

    result = differentiable_protocol_threshold(
        hard_trial,
        replay_with_wrong_slope,
        0.5,
        1.5,
        **{**ROOT_OPTIONS, "max_relative_amplitude_partial_error": 0.1},
    )
    assert not result.valid
    assert "amplitude partial disagrees" in result.rejection_reasons[0]
