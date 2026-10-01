"""Vectorized protocol threshold searches with independent lane gradients."""

import pytest
import torch

from dendra.models.analysis.batched_threshold_protocol import (
    BatchedThresholdTrial,
    DifferentiableBatchedThresholdTrial,
    differentiable_batched_protocol_thresholds,
)
from dendra.models.analysis.callback_margin import callback_signed_margin
from dendra.models.analysis.threshold_protocol import stack_valid_thresholds
from dendra.models.callbacks import ActiveAL

OPTIONS = dict(
    max_amplitude_bracket=1e-8,
    max_margin_jump=1e-6,
    max_center_residual=1e-6,
    max_replay_margin_difference=1e-9,
    min_positive_margin_partial=1e-12,
    max_cross_amplitude_partial=1e-10,
    max_iter=30,
)


def test_vectorized_search_uses_one_hard_trial_per_iteration():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    thresholds = torch.stack((1.0 + q, 2.0 + 2.0 * q, 3.0 - 0.5 * q))
    calls = []

    def hard_trial(amplitudes):
        calls.append(amplitudes.detach().clone())
        margins = amplitudes - thresholds.detach()
        return BatchedThresholdTrial(
            margins >= 0,
            margins,
            branch_ids=("a", "b", "c"),
        )

    def differentiable_trial(amplitudes):
        leaf = amplitudes.detach().clone().requires_grad_()
        return DifferentiableBatchedThresholdTrial(
            leaf,
            leaf - thresholds,
            branch_ids=("a", "b", "c"),
        )

    result = differentiable_batched_protocol_thresholds(
        hard_trial,
        differentiable_trial,
        torch.zeros(3, dtype=torch.float64),
        torch.full((3,), 5.0, dtype=torch.float64),
        **OPTIONS,
    )
    assert len(calls) == OPTIONS["max_iter"] + 3
    assert result.valid_mask.tolist() == [True, True, True]
    proxy = stack_valid_thresholds(result.results)
    torch.testing.assert_close(proxy.detach(), thresholds.detach(), rtol=0, atol=1e-8)
    for i, expected in enumerate((1.0, 2.0, -0.5)):
        (slope,) = torch.autograd.grad(proxy[i], q, retain_graph=True)
        assert float(slope) == pytest.approx(expected, abs=1e-12)
        assert result.results[i].amplitude_partial == pytest.approx(1.0)


def test_invalid_lane_has_no_gradient_while_other_lane_remains_usable():
    q = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    def hard_trial(amplitudes):
        first = amplitudes[0] - (1.0 + q.detach())
        second = torch.where(amplitudes[1] < 2.0, -80.0, 20.0)
        margins = torch.stack((first, second))
        branch = ("continuous", "rest" if amplitudes[1] < 2.0 else "spike")
        return BatchedThresholdTrial(margins >= 0, margins, branch)

    def differentiable_trial(amplitudes):
        leaf = amplitudes.detach().clone().requires_grad_()
        margins = leaf - torch.stack((1.0 + q, q.new_tensor(2.0)))
        return DifferentiableBatchedThresholdTrial(
            leaf,
            margins,
            ("continuous", "spike"),
        )

    result = differentiable_batched_protocol_thresholds(
        hard_trial,
        differentiable_trial,
        torch.zeros(2, dtype=torch.float64),
        torch.full((2,), 4.0, dtype=torch.float64),
        **OPTIONS,
    )
    assert result.valid_mask.tolist() == [True, False]
    assert result.results[1].threshold is None
    assert "margin_jump" in result.results[1].rejection_reasons
    (slope,) = torch.autograd.grad(result.results[0].threshold, q)
    assert float(slope) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Invalid threshold results"):
        stack_valid_thresholds(result.results)


def test_cross_lane_amplitude_dependence_is_rejected():
    thresholds = torch.tensor([1.0, 2.0], dtype=torch.float64)

    def hard_trial(amplitudes):
        margins = amplitudes - thresholds
        return BatchedThresholdTrial(margins >= 0, margins)

    def differentiable_trial(amplitudes):
        leaf = amplitudes.detach().clone().requires_grad_()
        margins = leaf - thresholds
        coupled_first = margins[0] + 0.1 * (leaf[1] - leaf[1].detach())
        return DifferentiableBatchedThresholdTrial(
            leaf,
            torch.stack((coupled_first, margins[1])),
        )

    result = differentiable_batched_protocol_thresholds(
        hard_trial,
        differentiable_trial,
        torch.zeros(2, dtype=torch.float64),
        torch.full((2,), 4.0, dtype=torch.float64),
        **OPTIONS,
    )
    assert result.valid_mask.tolist() == [False, True]
    assert result.results[0].rejection_reasons == ("cross_amplitude_coupling",)


def test_batched_search_rejects_stalled_float32_lane_without_replay():
    def hard_trial(amplitudes):
        margins = amplitudes - 0.3
        return BatchedThresholdTrial(margins >= 0, margins)

    def differentiable_trial(amplitudes):
        raise AssertionError("A stalled lane must not enter differentiable replay")

    result = differentiable_batched_protocol_thresholds(
        hard_trial,
        differentiable_trial,
        torch.zeros(1, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
        **{**OPTIONS, "max_iter": 100},
    )
    assert result.valid_mask.tolist() == [False]
    assert "amplitude_resolution_lost" in result.results[0].rejection_reasons


def test_batched_search_rejects_malformed_hard_decision_dtype():
    def hard_trial(amplitudes):
        margins = amplitudes - 1.0
        return BatchedThresholdTrial((margins >= 0).to(torch.int8), margins)

    with pytest.raises(TypeError, match="boolean dtype"):
        differentiable_batched_protocol_thresholds(
            hard_trial,
            lambda _: None,
            torch.zeros(1, dtype=torch.float64),
            torch.full((1,), 2.0, dtype=torch.float64),
            **OPTIONS,
        )


def test_batched_search_can_reject_wrong_replay_amplitude_slope():
    def hard_trial(amplitudes):
        margins = amplitudes - 1.0
        return BatchedThresholdTrial(margins >= 0, margins)

    def replay(amplitudes):
        leaf = amplitudes.detach().clone().requires_grad_()
        margins = 2 * (leaf - leaf.detach()) + amplitudes - 1.0
        return DifferentiableBatchedThresholdTrial(leaf, margins)

    result = differentiable_batched_protocol_thresholds(
        hard_trial,
        replay,
        torch.zeros(1, dtype=torch.float64),
        torch.full((1,), 2.0, dtype=torch.float64),
        **{**OPTIONS, "max_relative_amplitude_partial_error": 0.1},
    )
    assert result.valid_mask.tolist() == [False]
    assert result.results[0].rejection_reasons == ("replay_amplitude_partial_mismatch",)


def test_batched_protocol_uses_actual_active_callback_and_matching_margin():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    targets = torch.stack((1.0 + q, 2.0 - 0.5 * q))

    class Model:
        def __init__(self, voltage):
            self.v = voltage
            self.nc = voltage.shape[-1]
            self.shape = voltage.shape

        def device(self):
            return self.v.device

    def trace(amplitudes):
        rest = torch.full(
            (2, 2), -70.0, dtype=amplitudes.dtype, device=amplitudes.device
        )
        response = torch.stack(
            (amplitudes - targets, torch.full_like(amplitudes, -70.0)),
            dim=-1,
        )
        return torch.stack((rest, rest, response, rest), dim=0)

    def callback():
        return ActiveAL(
            threshold=0.0,
            node_check=[0, 1],
            at_least=1,
            t_start_check=0.0,
            t_end_check=0.3,
            dt=0.1,
        )

    def hard_trial(amplitudes):
        voltage = trace(amplitudes)
        detector = callback()
        model = Model(voltage[0])
        detector.pre_loop_hook(model)
        for frame in voltage[1:]:
            model.v = frame
            detector.post_step_hook(model)
        margin = callback_signed_margin(detector, voltage)
        decisions = detector.is_active()
        assert torch.equal(decisions, margin.decision)
        return BatchedThresholdTrial(
            decisions,
            margin.margin,
            margin.branch_ids,
        )

    def differentiable_trial(amplitudes):
        leaf = amplitudes.detach().clone().requires_grad_()
        margin = callback_signed_margin(callback(), trace(leaf))
        return DifferentiableBatchedThresholdTrial(
            leaf,
            margin.margin,
            margin.branch_ids,
        )

    result = differentiable_batched_protocol_thresholds(
        hard_trial,
        differentiable_trial,
        torch.zeros(2, dtype=torch.float64),
        torch.full((2,), 4.0, dtype=torch.float64),
        **OPTIONS,
    )
    assert result.valid_mask.tolist() == [True, True]
    proxy = stack_valid_thresholds(result.results)
    torch.testing.assert_close(proxy.detach(), targets.detach(), rtol=0, atol=1e-8)
    (slope,) = torch.autograd.grad(proxy.sum(), q)
    assert float(slope) == pytest.approx(0.5, abs=1e-12)
