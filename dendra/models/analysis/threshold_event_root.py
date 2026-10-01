"""Differentiate a scalar amplitude threshold defined by a signed margin.

The caller supplies a margin that is negative below the hard protocol's
threshold and nonnegative above it. Amplitude and margin have caller-defined
units. An optional branch ID records which local event defines the margin.
The original single-site ``Active`` interface remains available as an adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Hashable

import torch


@dataclass(frozen=True, init=False)
class MarginObservation:
    """One scalar signed margin and, optionally, its selected event branch."""

    margin: torch.Tensor
    branch_id: Hashable | None

    def __init__(
        self,
        margin: torch.Tensor | float,
        branch_id: Hashable | None = None,
        *,
        peak_index: int | None = None,
    ) -> None:
        # Older Active-specific callers may supply peak_index by keyword.
        if peak_index is not None:
            if branch_id is not None and branch_id != peak_index:
                raise ValueError("Conflicting branch_id and peak_index.")
            branch_id = peak_index
        margin = torch.as_tensor(margin)
        if margin.numel() != 1:
            raise ValueError("A signed margin must be scalar.")
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "branch_id", branch_id)

    @property
    def value(self) -> float:
        return float(self.margin.detach())

    @property
    def peak_index(self) -> int | None:
        """Older name used by the single-site ``Active`` adapter."""
        return self.branch_id if isinstance(self.branch_id, int) else None


@dataclass(frozen=True)
class RootValidation:
    """Measured local checks; tolerances have caller-defined units."""

    iterations: int
    amplitude_bracket_width: float
    margin_span: float
    absolute_center_residual: float
    secant_margin_partial: float
    branch_consistent: bool | None
    max_amplitude_bracket: float
    max_margin_jump: float
    max_center_residual: float


@dataclass(frozen=True)
class RootResult:
    root: float
    lower: float
    upper: float
    left: MarginObservation
    right: MarginObservation
    center: MarginObservation
    valid: bool
    rejection_reasons: tuple[str, ...]
    validation: RootValidation


def _checked_observation(
    evaluate: Callable[[float], MarginObservation | torch.Tensor | float],
    amplitude: float,
) -> MarginObservation:
    value = evaluate(amplitude)
    observation = (
        value if isinstance(value, MarginObservation) else MarginObservation(value)
    )
    if not math.isfinite(observation.value):
        raise ValueError("Non-finite signed margin.")
    return observation


def bisect_signed_margin_root(
    evaluate: Callable[[float], MarginObservation | torch.Tensor | float],
    lower: float,
    upper: float,
    *,
    max_amplitude_bracket: float,
    max_margin_jump: float,
    max_center_residual: float,
    max_iter: int = 23,
) -> RootResult:
    """Find a negative-to-nonnegative margin root and assess local validity.

    The evaluator must use the same model, stimulus basis, decision rule, and
    observation window at every amplitude. The amplitude tolerance is in
    amplitude units, and the other two are in margin units. If branch IDs are
    supplied, the three final observations must select the same branch.
    These local checks do not establish global monotonicity or exclude a
    second activation/block branch elsewhere in the trial range.
    """
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("Need finite ordered amplitude bounds.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    tolerances = (max_amplitude_bracket, max_margin_jump, max_center_residual)
    if any(not math.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError("Root tolerances must be finite and positive.")

    left = _checked_observation(evaluate, lower)
    right = _checked_observation(evaluate, upper)
    if not (left.value < 0 <= right.value):
        raise ValueError("Need negative lower and nonnegative upper margins.")
    resolution_lost = False
    iterations = 0
    for _ in range(max_iter):
        mid = (lower + upper) / 2
        if mid == lower or mid == upper:
            resolution_lost = True
            break
        sample = _checked_observation(evaluate, mid)
        iterations += 1
        if sample.value >= 0:
            upper, right = mid, sample
        else:
            lower, left = mid, sample
    root = (lower + upper) / 2
    center = _checked_observation(evaluate, root)

    branch_ids = (left.branch_id, right.branch_id, center.branch_id)
    if all(branch_id is None for branch_id in branch_ids):
        branch_consistent = None
    else:
        branch_consistent = (
            all(branch_id is not None for branch_id in branch_ids)
            and branch_ids[0] == branch_ids[1] == branch_ids[2]
        )
    bracket_width = upper - lower
    margin_span = right.value - left.value
    validation = RootValidation(
        iterations=iterations,
        amplitude_bracket_width=bracket_width,
        margin_span=margin_span,
        absolute_center_residual=abs(center.value),
        secant_margin_partial=(
            margin_span / bracket_width if bracket_width > 0 else float("nan")
        ),
        branch_consistent=branch_consistent,
        max_amplitude_bracket=max_amplitude_bracket,
        max_margin_jump=max_margin_jump,
        max_center_residual=max_center_residual,
    )
    reasons = []
    if resolution_lost or bracket_width <= 0:
        reasons.append("amplitude_resolution_lost")
    if bracket_width > max_amplitude_bracket:
        reasons.append("amplitude_bracket_too_wide")
    if margin_span > max_margin_jump:
        reasons.append("margin_jump")
    if branch_consistent is False:
        reasons.append(
            "branch_changed"
            if all(branch_id is not None for branch_id in branch_ids)
            else "branch_metadata_incomplete"
        )
    if abs(center.value) > max_center_residual:
        reasons.append("root_residual")
    return RootResult(
        root, lower, upper, left, right, center, not reasons, tuple(reasons), validation
    )


def implicit_amplitude_root_proxy(
    root: RootResult,
    differentiable_amplitude: torch.Tensor,
    differentiable_margin: torch.Tensor,
    *,
    min_positive_margin_partial: float,
    max_relative_amplitude_partial_error: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Return the bracketed amplitude with implicit slope ``-m_q / m_A``.

    ``differentiable_amplitude`` is the scalar amplitude used to evaluate the
    scalar ``differentiable_margin`` at ``root.root``. For a model parameter
    ``q``, the proxy gradient is ``-partial_q(m) / partial_A(m)``. Its forward
    value is the bracketed root. The collocation amplitude's direct derivative
    is cancelled; it is not itself a model parameter.
    """
    if not root.valid:
        raise ValueError(
            "Root failed local continuity/branch checks: "
            + ", ".join(root.rejection_reasons)
        )
    if (
        not math.isfinite(min_positive_margin_partial)
        or min_positive_margin_partial <= 0
    ):
        raise ValueError("Minimum positive margin partial must be finite and positive.")
    if max_relative_amplitude_partial_error is not None and (
        not math.isfinite(max_relative_amplitude_partial_error)
        or max_relative_amplitude_partial_error < 0
    ):
        raise ValueError(
            "Amplitude partial error tolerance must be finite and nonnegative."
        )
    if differentiable_amplitude.numel() != 1 or differentiable_margin.numel() != 1:
        raise ValueError("Amplitude and margin must be scalar tensors.")
    if (
        not differentiable_amplitude.requires_grad
        or not differentiable_margin.requires_grad
    ):
        raise ValueError("Amplitude and margin must retain an autograd connection.")
    root_constant = differentiable_amplitude.new_tensor(root.root)
    machine_tolerance = (
        8 * torch.finfo(differentiable_amplitude.dtype).eps * max(1.0, abs(root.root))
    )
    if abs(float(differentiable_amplitude.detach()) - root.root) > max(
        machine_tolerance, root.validation.amplitude_bracket_width / 2
    ):
        raise ValueError(
            "Differentiable simulation amplitude differs from bracketed root."
        )
    margin_value = float(differentiable_margin.detach())
    if (
        not math.isfinite(margin_value)
        or abs(margin_value) > root.validation.max_center_residual
        or abs(margin_value - root.center.value) > root.validation.max_center_residual
    ):
        raise ValueError("Differentiable margin does not match the bracketed root.")
    (partial,) = torch.autograd.grad(
        differentiable_margin,
        differentiable_amplitude,
        retain_graph=True,
        allow_unused=True,
    )
    if partial is None:
        raise ValueError("Margin has no autograd connection to amplitude.")
    partial_value = float(partial.detach())
    if not math.isfinite(partial_value) or partial_value <= min_positive_margin_partial:
        raise ValueError("Margin has no finite positive amplitude partial at root.")
    if max_relative_amplitude_partial_error is not None:
        secant = root.validation.secant_margin_partial
        relative_error = abs(partial_value - secant) / max(
            abs(partial_value), abs(secant), min_positive_margin_partial
        )
        if (
            not math.isfinite(relative_error)
            or relative_error > max_relative_amplitude_partial_error
        ):
            raise ValueError(
                "Differentiable amplitude partial disagrees with hard bracket secant."
            )
    # Remove the collocation amplitude's direct derivative while retaining
    # model-parameter derivatives through the margin.
    parameter_margin = (
        differentiable_margin
        - differentiable_margin.detach()
        - partial.detach()
        * (differentiable_amplitude - differentiable_amplitude.detach())
    )
    proxy = root_constant - parameter_margin / partial.detach()
    return proxy, partial_value


def active_signed_margin(
    recorder_voltage: torch.Tensor,
    *,
    first_checked_step: int,
    end_checked_step: int,
    voltage_threshold: float = 0.0,
) -> MarginObservation:
    """Return the single-site ``Active(at_least=1)`` signed voltage margin.

    ``first_checked_step`` and ``end_checked_step`` are ``Active``'s
    ``ind_start``/``ind_end``; the latter is exclusive. Recorder frame zero is
    initialized state; ``Active`` solver step zero reads recorder frame one.
    """
    if recorder_voltage.ndim != 1:
        raise ValueError("Expected one site's 1D Recorder trace.")
    if first_checked_step < 0 or end_checked_step <= first_checked_step:
        raise ValueError("Invalid Active check-step interval.")
    start = first_checked_step + 1
    stop = min(end_checked_step + 1, len(recorder_voltage))
    if start >= stop:
        raise ValueError("Active check window has no recorded solver samples.")
    checked = recorder_voltage[start:stop]
    max_voltage, index = checked.max(dim=0)
    return MarginObservation(
        max_voltage - voltage_threshold, peak_index=start + int(index)
    )


def bisect_margin_root(
    evaluate: Callable[[float], MarginObservation],
    lower: float,
    upper: float,
    *,
    max_iter: int = 23,
    max_voltage_jump_mV: float = 1.0,
    max_center_residual_mV: float = 1.0,
    max_current_bracket_mA: float = 1e-8,
) -> RootResult:
    """Compatibility adapter for the original Active/current root helper."""
    result = bisect_signed_margin_root(
        evaluate,
        lower,
        upper,
        max_iter=max_iter,
        max_amplitude_bracket=max_current_bracket_mA,
        max_margin_jump=max_voltage_jump_mV,
        max_center_residual=max_center_residual_mV,
    )
    legacy_reasons = {
        "amplitude_bracket_too_wide": "current_bracket_too_wide",
        "branch_changed": "peak_index_changed",
        "branch_metadata_incomplete": "peak_index_changed",
    }
    reasons = tuple(
        legacy_reasons.get(reason, reason) for reason in result.rejection_reasons
    )
    return replace(result, rejection_reasons=reasons)


def implicit_threshold_proxy(
    root: RootResult,
    differentiable_current: torch.Tensor,
    differentiable_margin: torch.Tensor,
    *,
    min_abs_current_partial_mV_per_mA: float = 1e-9,
) -> tuple[torch.Tensor, float]:
    """Compatibility adapter for the original Active/current gradient helper."""
    return implicit_amplitude_root_proxy(
        root,
        differentiable_current,
        differentiable_margin,
        min_positive_margin_partial=min_abs_current_partial_mV_per_mA,
    )
