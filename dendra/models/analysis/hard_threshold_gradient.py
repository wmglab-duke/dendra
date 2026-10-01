"""Numerical parameter gradients of a complete hard threshold protocol.

The spike decision may change abruptly with stimulus amplitude. This module
therefore differentiates the *threshold surface* returned by the full hard
search, without requiring a differentiable voltage margin. The caller owns
the model, parameter coordinates, stimulus, decision, and search protocol.

The returned tensor has the reported hard value on its forward pass and
a validated central-difference slope on its backward pass. That backward
slope is a numerical estimate, not an autograd derivative of the simulator.
It is appropriate for a training step only at the reported parameter scale;
the caller should reevaluate the hard objective after updating parameters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardThresholdBracket:
    """A reported hard value and certified bounds around it.

    ``value`` defaults to the bracket midpoint. Supply it when the reported
    descriptor is a nonlinear fit of midpoint thresholds, whose value need
    not equal the midpoint of its propagated uncertainty interval.
    """

    lower: float
    upper: float
    value: float | None = None

    def __post_init__(self):
        lower, upper = float(self.lower), float(self.upper)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError("Hard threshold bounds must be finite and ordered.")
        value = (lower + upper) / 2 if self.value is None else float(self.value)
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("Reported hard value must lie in its finite bounds.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "value", value)

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class ThresholdSecant:
    """Central and one-sided slopes, including hard-search uncertainty."""

    step: float
    minus: HardThresholdBracket
    plus: HardThresholdBracket
    lower_slope: float
    upper_slope: float
    midpoint_slope: float
    relative_slope_halfwidth: float
    left_slope: float
    right_slope: float
    relative_one_sided_disagreement: float


@dataclass(frozen=True)
class HardThresholdGradientResult:
    """Hard-forward proxy, or explicit reasons for withholding its gradient."""

    center: HardThresholdBracket
    secants: tuple[ThresholdSecant, ...]
    selected_step: float | None
    slope: float | None
    proxy: torch.Tensor | None
    rejection_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.proxy is not None and not self.rejection_reasons


def hard_threshold_parameter_gradient(
    evaluate_hard: Callable[[float], HardThresholdBracket],
    coordinate: torch.Tensor | Callable[[], torch.Tensor],
    *,
    steps: Sequence[float],
    max_forward_bracket: float,
    max_relative_slope_halfwidth: float,
    max_relative_step_disagreement: float,
    max_relative_one_sided_disagreement: float,
    slope_scale_floor: float,
) -> HardThresholdGradientResult:
    r"""Attach a checked hard-threshold slope to a scalar parameter coordinate.

    ``evaluate_hard(q)`` must rerun the *complete, identical* threshold
    protocol at coordinate ``q`` and return a certified interval for its
    scalar result. This can be a searched threshold or a fitted descriptor
    with propagated threshold-bracket uncertainty. Use a meaningful
    coordinate, such as log physical conductance or
    resistivity. It must restore any model state it mutates between calls.
    Pass a zero-argument coordinate factory when searches temporarily mutate
    trainable parameters: the graph is then rebuilt after all hard searches,
    and the proxy propagates through the current raw-parameter transform.
    A precomputed tensor is suitable when the evaluator does not mutate any
    tensor needed by its autograd graph.

    For a central step ``h``, the hard brackets imply the secant interval

    ``[(L(q+h)-U(q-h))/(2h), (U(q+h)-L(q-h))/(2h)]``.

    At each step, one-sided slopes from the center bracket screen for a kink
    that symmetric differences alone could conceal. At least two adjacent
    supplied steps must have resolved slope intervals and agree in slope.
    The smallest passing adjacent pair is chosen, and its smaller-step
    midpoint secant supplies the backward derivative. The checks support a
    local numerical gradient; they cannot prove differentiability or detect
    a branch switch outside the sampled parameter neighborhood. They also do
    not establish that the chosen spike-detection horizon is physiologically
    sufficient. This method costs ``1 + 2 * len(steps)`` complete hard
    evaluations per scalar coordinate.

    ``steps`` and all tolerances are caller-chosen and model-independent.
    ``max_forward_bracket`` has threshold-amplitude units; the slope floor has
    threshold-amplitude per coordinate units. Other tolerances are relative.
    """
    factory = coordinate if callable(coordinate) else None
    with torch.no_grad():
        initial_coordinate = factory() if factory is not None else coordinate
    if (
        not isinstance(initial_coordinate, torch.Tensor)
        or initial_coordinate.numel() != 1
        or not initial_coordinate.is_floating_point()
        or not bool(torch.isfinite(initial_coordinate).all())
    ):
        raise ValueError("coordinate must be a finite floating-point scalar tensor.")
    step_values = tuple(sorted(float(step) for step in steps))
    if (
        len(step_values) < 2
        or len(set(step_values)) != len(step_values)
        or any(not math.isfinite(step) or step <= 0 for step in step_values)
    ):
        raise ValueError("Need at least two distinct finite positive steps.")
    positive = (
        max_forward_bracket,
        max_relative_slope_halfwidth,
        max_relative_step_disagreement,
        max_relative_one_sided_disagreement,
        slope_scale_floor,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError(
            "Gradient tolerances and slope floor must be finite and positive."
        )

    q0 = float(initial_coordinate.detach())

    def checked(q: float) -> HardThresholdBracket:
        with torch.no_grad():
            bracket = evaluate_hard(q)
        if not isinstance(bracket, HardThresholdBracket):
            raise TypeError("Hard evaluator must return HardThresholdBracket.")
        return bracket

    center = checked(q0)
    secants = []
    for step in step_values:
        minus = checked(q0 - step)
        plus = checked(q0 + step)
        lo = (plus.lower - minus.upper) / (2 * step)
        hi = (plus.upper - minus.lower) / (2 * step)
        midpoint = (plus.value - minus.value) / (2 * step)
        left = (center.value - minus.value) / step
        right = (plus.value - center.value) / step
        slope_scale = max(abs(midpoint), slope_scale_floor)
        side_scale = max(abs(left), abs(right), slope_scale_floor)
        secants.append(
            ThresholdSecant(
                step,
                minus,
                plus,
                lo,
                hi,
                midpoint,
                (hi - lo) / (2 * slope_scale),
                left,
                right,
                abs(left - right) / side_scale,
            )
        )

    reasons = []
    if center.width > max_forward_bracket:
        reasons.append("forward_bracket_too_wide")

    selected = None
    for fine, coarse in zip(secants[:-1], secants[1:]):
        scale = max(
            abs(fine.midpoint_slope), abs(coarse.midpoint_slope), slope_scale_floor
        )
        step_disagreement = abs(fine.midpoint_slope - coarse.midpoint_slope) / scale
        if (
            fine.relative_slope_halfwidth <= max_relative_slope_halfwidth
            and coarse.relative_slope_halfwidth <= max_relative_slope_halfwidth
            and fine.relative_one_sided_disagreement
            <= max_relative_one_sided_disagreement
            and coarse.relative_one_sided_disagreement
            <= max_relative_one_sided_disagreement
            and step_disagreement <= max_relative_step_disagreement
        ):
            selected = fine
            break
    if selected is None:
        reasons.append("no_consistent_step_pair")
    if reasons:
        return HardThresholdGradientResult(
            center, tuple(secants), None, None, None, tuple(reasons)
        )

    live_coordinate = factory() if factory is not None else coordinate
    if (
        not isinstance(live_coordinate, torch.Tensor)
        or live_coordinate.numel() != 1
        or not live_coordinate.is_floating_point()
        or not live_coordinate.requires_grad
        or not bool(torch.isfinite(live_coordinate.detach()).all())
    ):
        raise ValueError("Final coordinate must retain a finite autograd connection.")
    final_q = float(live_coordinate.detach())
    tolerance = 8 * torch.finfo(live_coordinate.dtype).eps * max(1.0, abs(q0))
    if abs(final_q - q0) > tolerance:
        raise ValueError("Hard evaluator did not restore the parameter coordinate.")

    # The hard value is exact up to its reported bracket. The derivative is
    # the validated numerical secant in the caller's parameter coordinate.
    proxy = live_coordinate.new_tensor(center.value) + (
        live_coordinate - live_coordinate.detach()
    ) * live_coordinate.new_tensor(selected.midpoint_slope)
    return HardThresholdGradientResult(
        center,
        tuple(secants),
        selected.step,
        selected.midpoint_slope,
        proxy,
        (),
    )
