"""Choose a trace-tangent threshold probe without a model-specific protocol.

The caller owns the complete hard threshold search, the hard spike decision,
the observation mask, and the voltage/JVP simulation. This module only chooses
an amplitude on a certified side of the hard bracket and checks that a nearby
amplitude remains on that side with an approximately linear voltage response.
Passing these checks does not establish hard parameter-gradient fidelity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .threshold_trace_tangent import (
    TraceTangentDiagnostics,
    TraceTangentThresholdResult,
    trace_tangent_threshold_proxy,
)


@dataclass(frozen=True)
class ProbeTrace:
    """One voltage replay; ``amplitude_tangent`` is needed only at the probe.

    The tangent is a voltage JVP with respect to the scalar stimulus amplitude.
    A same-side amplitude secant is also acceptable. Neither construction
    needs a finite difference along any model-parameter coordinate.
    """

    voltage: torch.Tensor
    amplitude_tangent: torch.Tensor | None = None


@dataclass(frozen=True)
class ProbeAttempt:
    """Detached record of one candidate, including explicit rejection reasons."""

    side: str
    relative_gap: float
    probe_amplitude: float
    check_amplitude: float
    hard_probe_active: bool | None
    hard_check_active: bool | None
    diagnostics: TraceTangentDiagnostics | None
    parameter_gradient: tuple[float, ...] | None
    relative_gradient_disagreement: float | None
    rejection_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.rejection_reasons


@dataclass(frozen=True)
class ProbeSelectionResult:
    """The selected hard-forward proxy and a graph-free history of attempts."""

    threshold: TraceTangentThresholdResult | None
    selected_attempt_index: int | None
    attempts: tuple[ProbeAttempt, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.threshold is not None
            and self.threshold.valid
            and self.selected_attempt_index is not None
            and not self.rejection_reasons
        )

    @property
    def proxy(self) -> torch.Tensor | None:
        return self.threshold.proxy if self.valid else None


def _rejected(
    reasons: tuple[str, ...],
    attempts: list[ProbeAttempt] | None = None,
) -> ProbeSelectionResult:
    return ProbeSelectionResult(None, None, tuple(attempts or ()), reasons)


def _raise_if_resource_exhausted(error: Exception) -> None:
    """Do not retry a probe ladder after the simulator runs out of memory."""
    if isinstance(error, MemoryError) or "out of memory" in str(error).lower():
        raise error


def select_trace_tangent_threshold_probe(
    hard_threshold: float | torch.Tensor,
    hard_bracket: Sequence[float],
    *,
    hard_is_active: Callable[[float], bool],
    trace_and_tangent: Callable[[float, bool], ProbeTrace],
    min_tangent_norm: float,
    max_relative_linearization_error: float,
    mask: torch.Tensor | None = None,
    relative_gaps: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3),
    check_step_fraction: float = 0.5,
    sides: tuple[str, ...] = ("inactive",),
    verify_bracket_endpoints: bool = False,
    stability_parameters: tuple[torch.Tensor, ...] | None = None,
    max_relative_gradient_disagreement: float | None = None,
) -> ProbeSelectionResult:
    r"""Select the first locally valid probe from a dimensionless gap ladder.

    ``hard_bracket`` is ``(inactive_amplitude, active_amplitude)`` from a
    *complete* hard activation search. Its order may be increasing or
    decreasing; this supports either stimulus polarity. The hard threshold
    must lie between the two endpoints. By default the supplied bracket is
    trusted, avoiding two redundant hard simulations. Set
    ``verify_bracket_endpoints`` to replay its hard decisions.

    For each relative gap ``g``, define ``scale = max(abs(T), bracket_width)``
    and move ``g * scale`` beyond the corresponding bracket endpoint. The
    second replay moves another ``check_step_fraction * g * scale`` outward.
    The hard decision callback must confirm both amplitudes are on the
    requested side, even when the bracket's activation response is not
    monotonic outside its endpoints. ``trace_and_tangent(A, True)`` returns
    the differentiable probe trace and its amplitude JVP; the call at the
    second amplitude uses ``False`` and needs only the voltage. The check
    replay runs first and its voltage is snapshotted. The gradient-bearing
    probe replay must be the callback's final simulator forward before the
    caller takes a backward pass, because a later simulator run may mutate
    saved VJP buffers. A callback that computes a JVP with another forward
    should do so *before* its fresh, gradient-bearing probe replay. A fixed
    ``mask`` can select protocol-relevant sites and times. The caller may
    derive a new mask from its hard event protocol on each threshold call;
    this selector holds that mask fixed across the local probe ladder and
    never chooses a voltage event or branch itself.

    The selected candidate delegates the weighted trace projection and
    same-side linearity calculation to ``trace_tangent_threshold_proxy``.
    Rejected traces are not stored in ``attempts``; accepted candidates retain
    one proxy graph so a single backward pass reaches all connected model
    parameters. ``min_tangent_norm`` has the caller's voltage/amplitude units
    and must be chosen for that protocol. A valid result is only a candidate
    descent direction: complete hard-search parameter secants remain the
    independent validation target.

    Optionally pass ``stability_parameters`` to compute a detached gradient
    vector for each locally valid candidate in one VJP per candidate. With
    ``max_relative_gradient_disagreement``, the first two accepted probes on
    the same side whose gradient vectors agree within the stated relative
    Euclidean tolerance confirm the current, latest-forward probe. The selected proxy retains
    its graph for a later training backward pass. This costs extra reverse
    passes during probe selection, but never finite differences in model
    parameters. Agreement across probes still cannot prove agreement with the
    hard threshold gradient: an unsuitable observation mask or horizon can
    induce the same bias at every probe.
    """
    if not callable(hard_is_active) or not callable(trace_and_tangent):
        raise TypeError("Hard decision and trace/JVP callbacks must be callable.")
    if not hasattr(hard_bracket, "__len__") or len(hard_bracket) != 2:
        raise TypeError("Hard bracket must be an (inactive, active) pair.")
    if (
        not relative_gaps
        or any(not math.isfinite(g) or g <= 0 for g in relative_gaps)
        or any(b <= a for a, b in zip(relative_gaps, relative_gaps[1:]))
    ):
        raise ValueError(
            "Relative gaps must be finite, positive, and strictly increasing."
        )
    if not math.isfinite(check_step_fraction) or check_step_fraction <= 0:
        raise ValueError("Check step fraction must be finite and positive.")
    if (
        not sides
        or len(set(sides)) != len(sides)
        or any(side not in ("inactive", "active") for side in sides)
    ):
        raise ValueError("Sides must be distinct 'inactive' or 'active' entries.")
    if not math.isfinite(min_tangent_norm) or min_tangent_norm <= 0:
        raise ValueError("Minimum tangent norm must be finite and positive.")
    if (
        not math.isfinite(max_relative_linearization_error)
        or max_relative_linearization_error < 0
    ):
        raise ValueError("Linearization error limit must be finite and nonnegative.")
    if max_relative_gradient_disagreement is not None and (
        not math.isfinite(max_relative_gradient_disagreement)
        or max_relative_gradient_disagreement < 0
    ):
        raise ValueError("Gradient disagreement limit must be finite and nonnegative.")
    if max_relative_gradient_disagreement is not None and stability_parameters is None:
        raise ValueError("A gradient disagreement limit requires stability parameters.")
    if stability_parameters is not None and (
        not stability_parameters
        or any(
            not isinstance(p, torch.Tensor)
            or not p.is_floating_point()
            or not p.requires_grad
            for p in stability_parameters
        )
    ):
        raise ValueError(
            "Stability parameters must be nonempty differentiable tensors."
        )

    try:
        threshold = float(
            hard_threshold.detach().item()
            if isinstance(hard_threshold, torch.Tensor)
            else hard_threshold
        )
        inactive, active = (float(value) for value in hard_bracket)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return _rejected(("invalid_hard_threshold_or_bracket",))
    if not all(math.isfinite(value) for value in (threshold, inactive, active)):
        return _rejected(("nonfinite_hard_threshold_or_bracket",))
    if inactive == active:
        return _rejected(("zero_width_hard_bracket",))
    if not min(inactive, active) <= threshold <= max(inactive, active):
        return _rejected(("hard_threshold_outside_bracket",))

    if verify_bracket_endpoints:
        try:
            inactive_decision = bool(hard_is_active(inactive))
            active_decision = bool(hard_is_active(active))
        except Exception as error:
            _raise_if_resource_exhausted(error)
            return _rejected(("bracket_hard_decision_failed",))
        if inactive_decision or not active_decision:
            return _rejected(("bracket_hard_decisions_inconsistent",))

    direction = 1.0 if active > inactive else -1.0
    scale = max(abs(threshold), abs(active - inactive))
    attempts: list[ProbeAttempt] = []
    saw_gradient_pair = False
    for side in sides:
        previous_gradient: torch.Tensor | None = None
        anchor = inactive if side == "inactive" else active
        outward = -direction if side == "inactive" else direction
        expected_active = side == "active"
        for relative_gap in relative_gaps:
            gap = relative_gap * scale
            probe = anchor + outward * gap
            check = probe + outward * check_step_fraction * gap
            reasons: list[str] = []
            probe_decision: bool | None = None
            check_decision: bool | None = None
            diagnostics: TraceTangentDiagnostics | None = None
            gradient_values: tuple[float, ...] | None = None
            gradient_vector: torch.Tensor | None = None
            relative_disagreement: float | None = None
            if (
                not math.isfinite(gap)
                or not math.isfinite(probe)
                or not math.isfinite(check)
                or probe == anchor
                or check == probe
            ):
                reasons.append("probe_gap_not_representable")
            else:
                try:
                    probe_decision = bool(hard_is_active(probe))
                except Exception as error:
                    _raise_if_resource_exhausted(error)
                    reasons.append("probe_hard_decision_failed")
                if not reasons and probe_decision != expected_active:
                    reasons.append("probe_hard_side_mismatch")
                if not reasons:
                    try:
                        check_decision = bool(hard_is_active(check))
                    except Exception as error:
                        _raise_if_resource_exhausted(error)
                        reasons.append("check_hard_decision_failed")
                if not reasons and check_decision != expected_active:
                    reasons.append("check_hard_side_mismatch")

            result: TraceTangentThresholdResult | None = None
            if not reasons:
                try:
                    check_trace = trace_and_tangent(check, False)
                except Exception as error:
                    _raise_if_resource_exhausted(error)
                    reasons.append("trace_callback_failed")
                else:
                    if not isinstance(check_trace, ProbeTrace) or not isinstance(
                        check_trace.voltage, torch.Tensor
                    ):
                        reasons.append("trace_callback_invalid_record")
                    else:
                        try:
                            check_voltage = check_trace.voltage.detach().clone()
                        except (TypeError, ValueError, RuntimeError) as error:
                            _raise_if_resource_exhausted(error)
                            reasons.append("check_trace_snapshot_failed")
                if not reasons:
                    try:
                        probe_trace = trace_and_tangent(probe, True)
                    except Exception as error:
                        _raise_if_resource_exhausted(error)
                        reasons.append("trace_callback_failed")
                    else:
                        if (
                            not isinstance(probe_trace, ProbeTrace)
                            or not isinstance(probe_trace.voltage, torch.Tensor)
                            or not isinstance(
                                probe_trace.amplitude_tangent, torch.Tensor
                            )
                        ):
                            reasons.append("trace_callback_invalid_record")
                        else:
                            try:
                                result = trace_tangent_threshold_proxy(
                                    threshold,
                                    probe,
                                    probe_trace.voltage,
                                    probe_trace.amplitude_tangent,
                                    min_tangent_norm=min_tangent_norm,
                                    mask=mask,
                                    check_amplitude=check,
                                    check_voltage=check_voltage,
                                    max_relative_linearization_error=(
                                        max_relative_linearization_error
                                    ),
                                )
                            except (TypeError, ValueError, RuntimeError) as error:
                                _raise_if_resource_exhausted(error)
                                reasons.append("trace_projection_invalid_input")
                            else:
                                diagnostics = result.diagnostics
                                reasons.extend(result.rejection_reasons)

            if result is not None and result.valid and stability_parameters is not None:
                try:
                    gradients = torch.autograd.grad(
                        result.proxy,
                        stability_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                except RuntimeError as error:
                    _raise_if_resource_exhausted(error)
                    reasons.append("gradient_probe_failed")
                else:
                    if any(gradient is None for gradient in gradients):
                        reasons.append("gradient_coordinate_disconnected")
                    else:
                        gradient_vector = torch.cat(
                            [
                                gradient.detach()
                                .to(device="cpu", dtype=torch.float64)
                                .reshape(-1)
                                for gradient in gradients
                            ]
                        )
                        if not bool(torch.isfinite(gradient_vector).all()):
                            reasons.append("gradient_probe_nonfinite")
                            gradient_vector = None
                        else:
                            gradient_values = tuple(
                                float(value) for value in gradient_vector
                            )
                            if previous_gradient is not None:
                                denominator = max(
                                    float(torch.linalg.vector_norm(previous_gradient)),
                                    float(torch.linalg.vector_norm(gradient_vector)),
                                )
                                difference = float(
                                    torch.linalg.vector_norm(
                                        gradient_vector - previous_gradient
                                    )
                                )
                                if not math.isfinite(denominator) or not math.isfinite(
                                    difference
                                ):
                                    reasons.append("gradient_probe_norm_nonfinite")
                                else:
                                    relative_disagreement = (
                                        difference / denominator
                                        if denominator > 0
                                        else 0.0
                                    )
                                    saw_gradient_pair = True

            attempts.append(
                ProbeAttempt(
                    side,
                    relative_gap,
                    probe,
                    check,
                    probe_decision,
                    check_decision,
                    diagnostics,
                    gradient_values,
                    relative_disagreement,
                    tuple(reasons),
                )
            )
            if result is not None and result.valid and not reasons:
                if stability_parameters is None:
                    return ProbeSelectionResult(
                        result, len(attempts) - 1, tuple(attempts), ()
                    )
                if max_relative_gradient_disagreement is None:
                    return ProbeSelectionResult(
                        result, len(attempts) - 1, tuple(attempts), ()
                    )
                if (
                    previous_gradient is not None
                    and relative_disagreement is not None
                    and relative_disagreement <= max_relative_gradient_disagreement
                ):
                    return ProbeSelectionResult(
                        result, len(attempts) - 1, tuple(attempts), ()
                    )
                previous_gradient = gradient_vector

    # The history stores only scalar values and detached diagnostics. It is
    # safe to retain for reporting after rejected trace graphs are released.
    distinct_reasons = tuple(
        dict.fromkeys(
            reason for attempt in attempts for reason in attempt.rejection_reasons
        )
    )
    if max_relative_gradient_disagreement is not None:
        reason = (
            "probe_gradient_unstable"
            if saw_gradient_pair
            else "probe_gradient_confirmation_unavailable"
        )
        return _rejected((reason,) + distinct_reasons, attempts)
    return _rejected(("no_valid_probe",) + distinct_reasons, attempts)
