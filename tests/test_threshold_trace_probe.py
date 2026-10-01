"""Probe selection checks only protocol side and local voltage linearity."""

import pytest
import torch

from dendra.models.analysis.threshold_trace_probe import (
    ProbeTrace,
    select_trace_tangent_threshold_probe,
)


def test_selects_nearest_inactive_probe_with_exact_hard_forward_and_two_slopes():
    q = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    hard = 1.0 + 0.3 * q[0] - 0.2 * q[1]
    threshold = hard.detach().item()
    basis = torch.tensor([2.0, -3.0, 1.5], dtype=q.dtype)
    calls = []

    def trace(amplitude, need_tangent):
        calls.append((amplitude, need_tangent))
        voltage = basis * (amplitude - hard)
        return ProbeTrace(voltage, basis if need_tangent else None)

    result = select_trace_tangent_threshold_probe(
        threshold,
        (threshold - 1e-7, threshold + 1e-7),
        hard_is_active=lambda amplitude: amplitude >= threshold,
        trace_and_tangent=trace,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=1e-10,
        relative_gaps=(1e-5, 1e-4),
    )
    assert result.valid
    assert result.selected_attempt_index == 0
    assert len(result.attempts) == 1
    assert result.proxy.item() == threshold
    assert len(calls) == 2
    assert calls[0][1] is False and calls[1][1] is True
    assert calls[1][0] < threshold - 1e-7
    assert calls[0][0] < calls[1][0]
    (gradient,) = torch.autograd.grad(result.proxy, q)
    assert torch.allclose(
        gradient, torch.tensor([0.3, -0.2], dtype=q.dtype), rtol=1e-12, atol=1e-12
    )
    assert result.attempts[0].diagnostics.relative_linearization_error == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_check_voltage_is_snapshotted_before_final_gradient_bearing_replay():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    threshold = 1.0 + 0.2 * q.detach().item()
    reused_recorder_buffer = torch.zeros(1, dtype=q.dtype)
    calls = []

    def trace(amplitude, need_tangent):
        calls.append(need_tangent)
        if not need_tangent:
            reused_recorder_buffer.fill_(amplitude - threshold)
            return ProbeTrace(reused_recorder_buffer)
        # Simulate a recorder buffer reused by the subsequent forward.
        reused_recorder_buffer.fill_(999.0)
        voltage = (amplitude - (1.0 + 0.2 * q)).reshape(1)
        return ProbeTrace(voltage, torch.ones(1, dtype=q.dtype))

    result = select_trace_tangent_threshold_probe(
        threshold,
        (threshold - 1e-5, threshold + 1e-5),
        hard_is_active=lambda amplitude: amplitude >= threshold,
        trace_and_tangent=trace,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=1e-10,
        relative_gaps=(1e-4,),
    )
    assert result.valid
    assert calls == [False, True]
    assert result.attempts[0].diagnostics.relative_linearization_error == pytest.approx(
        0.0,
        abs=1e-10,
    )
    (derivative,) = torch.autograd.grad(result.proxy, q)
    assert derivative.item() == pytest.approx(0.2, abs=1e-12)


def test_rejects_hard_side_mismatch_without_running_a_trace():
    trace_calls = []

    def trace(amplitude, need_tangent):
        trace_calls.append(amplitude)
        raise AssertionError("Hard rejection should precede a voltage replay")

    result = select_trace_tangent_threshold_probe(
        1.0,
        (0.99, 1.01),
        hard_is_active=lambda amplitude: True,
        trace_and_tangent=trace,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=0.1,
        relative_gaps=(1e-4, 1e-3),
    )
    assert not result.valid
    assert result.proxy is None
    assert result.rejection_reasons == ("no_valid_probe", "probe_hard_side_mismatch")
    assert [attempt.rejection_reasons for attempt in result.attempts] == [
        ("probe_hard_side_mismatch",),
        ("probe_hard_side_mismatch",),
    ]
    assert not trace_calls


def test_rejects_replay_that_crosses_to_opposite_hard_side():
    # A nonmonotone hard classifier could reverse outside a valid bracket.
    def hard_active(amplitude):
        return amplitude >= 1.0 or amplitude < 0.89

    result = select_trace_tangent_threshold_probe(
        1.0,
        (0.9, 1.1),
        hard_is_active=hard_active,
        trace_and_tangent=lambda *_: (_ for _ in ()).throw(AssertionError()),
        min_tangent_norm=1e-9,
        max_relative_linearization_error=0.1,
        relative_gaps=(0.008,),
        check_step_fraction=0.5,
    )
    assert not result.valid
    assert result.attempts[0].hard_probe_active is False
    assert result.attempts[0].hard_check_active is True
    assert result.attempts[0].rejection_reasons == ("check_hard_side_mismatch",)


def test_skips_weak_first_tangent_and_retains_second_graph():
    q = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    threshold = 1.0 + 0.2 * q.detach().item()
    inactive = threshold - 1e-5
    scale = threshold
    first = inactive - 1e-6 * scale

    def trace(amplitude, need_tangent):
        if abs(amplitude - first) < 1e-12 and need_tangent:
            return ProbeTrace(q.reshape(1), torch.zeros(1, dtype=q.dtype))
        voltage = (amplitude - (1.0 + 0.2 * q)).reshape(1)
        return ProbeTrace(
            voltage, torch.ones(1, dtype=q.dtype) if need_tangent else None
        )

    result = select_trace_tangent_threshold_probe(
        threshold,
        (inactive, threshold + 1e-5),
        hard_is_active=lambda amplitude: amplitude >= threshold,
        trace_and_tangent=trace,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=1e-8,
        relative_gaps=(1e-6, 1e-4),
    )
    assert result.valid
    assert result.selected_attempt_index == 1
    assert result.attempts[0].rejection_reasons == ("amplitude_tangent_too_small",)
    assert result.attempts[1].valid
    (derivative,) = torch.autograd.grad(result.proxy, q)
    assert derivative.item() == pytest.approx(0.2, abs=1e-12)


def test_rejects_nonlinear_trace_on_same_hard_side():
    q = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def trace(amplitude, need_tangent):
        voltage = (q + amplitude**2).reshape(1)
        tangent = torch.tensor([2 * amplitude], dtype=q.dtype) if need_tangent else None
        return ProbeTrace(voltage, tangent)

    result = select_trace_tangent_threshold_probe(
        1.0,
        (0.9, 1.1),
        hard_is_active=lambda amplitude: amplitude >= 1.0,
        trace_and_tangent=trace,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=1e-5,
        relative_gaps=(0.01,),
    )
    assert not result.valid
    assert result.rejection_reasons == ("no_valid_probe", "amplitude_tangent_nonlinear")
    assert result.attempts[0].diagnostics.relative_linearization_error > 1e-5


def test_supports_reversed_bracket_and_optional_active_side():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    threshold = 1.5 + 0.1 * q.detach().item()

    def trace(amplitude, need_tangent):
        voltage = (amplitude - (1.5 + 0.1 * q)).reshape(1)
        tangent = torch.ones(1, dtype=q.dtype) if need_tangent else None
        return ProbeTrace(voltage, tangent)

    result = select_trace_tangent_threshold_probe(
        threshold,
        (threshold + 0.01, threshold - 0.01),
        hard_is_active=lambda amplitude: amplitude <= threshold,
        trace_and_tangent=trace,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=1e-10,
        relative_gaps=(1e-3,),
        sides=("active",),
        verify_bracket_endpoints=True,
    )
    assert result.valid
    attempt = result.attempts[0]
    assert attempt.probe_amplitude < threshold - 0.01
    assert attempt.check_amplitude < attempt.probe_amplitude
    assert attempt.hard_probe_active and attempt.hard_check_active
    (gradient,) = torch.autograd.grad(result.proxy, q)
    assert gradient.item() == pytest.approx(0.1, abs=1e-12)


def test_invalid_hard_bracket_has_explicit_rejection():
    def callback(*_):
        return ProbeTrace(torch.tensor([0.0], requires_grad=True))

    kwargs = dict(
        hard_is_active=lambda amplitude: amplitude > 1.0,
        trace_and_tangent=callback,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=0.1,
    )
    assert select_trace_tangent_threshold_probe(
        1.0,
        (0.9, 0.9),
        **kwargs,
    ).rejection_reasons == ("zero_width_hard_bracket",)
    assert select_trace_tangent_threshold_probe(
        1.2,
        (0.9, 1.1),
        **kwargs,
    ).rejection_reasons == ("hard_threshold_outside_bracket",)
    assert select_trace_tangent_threshold_probe(
        1.0,
        (0.9, 1.1),
        verify_bracket_endpoints=True,
        hard_is_active=lambda _: True,
        trace_and_tangent=callback,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=0.1,
    ).rejection_reasons == ("bracket_hard_decisions_inconsistent",)


def test_rejects_unrepresentable_gap_and_invalid_trace_record():
    tiny = select_trace_tangent_threshold_probe(
        1e20,
        (1e20 - 1e3, 1e20 + 1e3),
        hard_is_active=lambda amplitude: amplitude >= 1e20,
        trace_and_tangent=lambda *_: None,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=0.1,
        relative_gaps=(1e-20,),
    )
    # The supplied bracket collapses in binary64 before gap generation.
    assert tiny.rejection_reasons == ("zero_width_hard_bracket",)

    invalid = select_trace_tangent_threshold_probe(
        1.0,
        (0.9, 1.1),
        hard_is_active=lambda amplitude: amplitude >= 1.0,
        trace_and_tangent=lambda *_: None,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=0.1,
        relative_gaps=(1e-4,),
    )
    assert invalid.rejection_reasons == (
        "no_valid_probe",
        "trace_callback_invalid_record",
    )


def test_optional_gradient_stability_gate_compares_probes_with_vjps():
    q = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    def trace(amplitude, need_tangent):
        # Each local amplitude replay is perfectly linear, but a distant
        # probe follows a different parameter-sensitive trace branch.
        parameter_gain = 1.0 if amplitude > 0.989 else 2.0
        voltage = (amplitude - 1.0 + parameter_gain * q).reshape(1)
        tangent = torch.ones(1, dtype=q.dtype) if need_tangent else None
        return ProbeTrace(voltage, tangent)

    kwargs = dict(
        hard_is_active=lambda amplitude: amplitude >= 1.0,
        trace_and_tangent=trace,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=1e-10,
        stability_parameters=(q,),
        max_relative_gradient_disagreement=0.01,
    )
    stable = select_trace_tangent_threshold_probe(
        1.0,
        (0.99, 1.01),
        relative_gaps=(1e-6, 0.01, 0.02),
        **kwargs,
    )
    assert stable.valid
    assert stable.selected_attempt_index == 2
    assert [attempt.parameter_gradient for attempt in stable.attempts] == [
        (-1.0,),
        (-2.0,),
        (-2.0,),
    ]
    assert stable.attempts[1].relative_gradient_disagreement == pytest.approx(0.5)
    assert stable.attempts[2].relative_gradient_disagreement == pytest.approx(0.0)
    assert all(
        attempt.diagnostics.relative_linearization_error < 1e-10
        for attempt in stable.attempts
    )
    (slope,) = torch.autograd.grad(stable.proxy, q)
    assert slope.item() == pytest.approx(-2.0)

    unstable = select_trace_tangent_threshold_probe(
        1.0,
        (0.99, 1.01),
        relative_gaps=(1e-6, 0.01),
        **kwargs,
    )
    assert not unstable.valid
    assert unstable.proxy is None
    assert unstable.rejection_reasons == ("probe_gradient_unstable",)
    assert unstable.attempts[1].relative_gradient_disagreement == pytest.approx(0.5)


def test_rejects_invalid_ladder_configuration():
    kwargs = dict(
        hard_is_active=lambda _: False,
        trace_and_tangent=lambda *_: None,
        min_tangent_norm=1e-9,
        max_relative_linearization_error=0.1,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        select_trace_tangent_threshold_probe(
            1.0, (0.9, 1.1), relative_gaps=(1e-4, 1e-4), **kwargs
        )
    with pytest.raises(ValueError, match="Check step"):
        select_trace_tangent_threshold_probe(
            1.0, (0.9, 1.1), check_step_fraction=0.0, **kwargs
        )
    with pytest.raises(ValueError, match="requires stability parameters"):
        select_trace_tangent_threshold_probe(
            1.0,
            (0.9, 1.1),
            max_relative_gradient_disagreement=0.1,
            **kwargs,
        )


def test_resource_exhaustion_is_not_swallowed_or_retried():
    calls = []

    def trace(amplitude, need_tangent):
        calls.append(amplitude)
        raise RuntimeError("MPS backend out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        select_trace_tangent_threshold_probe(
            1.0,
            (0.9, 1.1),
            hard_is_active=lambda amplitude: amplitude >= 1.0,
            trace_and_tangent=trace,
            min_tangent_norm=1e-9,
            max_relative_linearization_error=0.1,
            relative_gaps=(1e-4, 1e-3),
        )
    assert len(calls) == 1
