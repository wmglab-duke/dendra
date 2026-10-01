"""Model-agnostic threshold differentiation for a caller-defined protocol.

The caller runs both versions of one trial: a hard trial that returns its
decision and a signed margin, and an autograd trial at the accepted root.
This module knows nothing about model classes, stimulus waveforms, units, or
recording sites. Those choices belong to the protocol supplied by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import torch

from .threshold_event_root import (
    MarginObservation,
    RootResult,
    bisect_signed_margin_root,
    implicit_amplitude_root_proxy,
)


@dataclass(frozen=True)
class ThresholdTrial:
    """Hard decision and matching signed margin from one amplitude trial.

    ``decision`` must be true exactly when ``margin >= 0``. ``branch_id`` may
    identify a selected event, sample, or recording site whose identity must
    remain fixed near the threshold. Use ``None`` when no such ID exists.
    """

    decision: bool
    margin: torch.Tensor | float
    branch_id: Hashable | None = None


@dataclass(frozen=True)
class DifferentiableThresholdTrial:
    """Autograd rerun at the root, including the selected event identity."""

    amplitude: torch.Tensor
    margin: torch.Tensor
    branch_id: Hashable | None = None


@dataclass(frozen=True)
class ProtocolThresholdResult:
    """A threshold proxy or an explicit reason why no derivative was issued."""

    root: RootResult
    threshold: torch.Tensor | None
    amplitude_partial: float | None
    rejection_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.threshold is not None and not self.rejection_reasons


def differentiable_protocol_threshold(
    evaluate_hard: Callable[[float], ThresholdTrial],
    evaluate_differentiable: Callable[[float], DifferentiableThresholdTrial],
    lower: float,
    upper: float,
    *,
    max_amplitude_bracket: float,
    max_margin_jump: float,
    max_center_residual: float,
    max_replay_margin_difference: float,
    min_positive_margin_partial: float,
    max_relative_amplitude_partial_error: float | None = None,
    max_iter: int = 23,
) -> ProtocolThresholdResult:
    """Search and differentiate a protocol-defined activation threshold.

    ``evaluate_hard(A)`` must run the complete hard trial at amplitude ``A``
    from the same initialized state and return its decision and signed margin.
    Their agreement is checked on every sampled trial. The margin must be
    negative below the threshold and nonnegative above it.

    ``evaluate_differentiable(A)`` must rerun that protocol with autograd at
    the bracketed amplitude and return ``DifferentiableThresholdTrial``.
    The amplitude tensor must be the scalar used by the simulation. Its device,
    dtype, and units come from the caller. The replayed margin and selected
    branch must agree with the hard trial at the root. The lower-level
    implicit check also requires replay agreement within
    ``max_center_residual`` margin units.

    ``max_relative_amplitude_partial_error`` optionally compares the replay's
    local amplitude derivative with the hard bracket secant. Choose a bracket
    wide enough to resolve the hard margin before enabling this check; a
    machine-precision secant is not a reliable derivative reference. Return
    the exact amplitude tensor used by the replay, not a view created later.

    Failed local root or replay checks return ``threshold=None`` with explicit
    reasons. A valid result has the bracketed hard threshold on its forward
    pass and an implicit local gradient. The caller must still establish that
    the selected root is the intended threshold if activation is nonmonotone.
    """
    if (
        not math.isfinite(max_replay_margin_difference)
        or max_replay_margin_difference < 0
    ):
        raise ValueError("Replay margin tolerance must be finite and nonnegative.")

    def checked_margin(amplitude: float) -> MarginObservation:
        with torch.no_grad():
            trial = evaluate_hard(amplitude)
        if not isinstance(trial, ThresholdTrial):
            raise TypeError("Hard evaluator must return ThresholdTrial.")
        observation = MarginObservation(trial.margin, branch_id=trial.branch_id)
        if bool(trial.decision) != (observation.value >= 0):
            raise ValueError(
                f"Signed margin and hard decision disagree at amplitude {amplitude}."
            )
        return observation

    root = bisect_signed_margin_root(
        checked_margin,
        lower,
        upper,
        max_amplitude_bracket=max_amplitude_bracket,
        max_margin_jump=max_margin_jump,
        max_center_residual=max_center_residual,
        max_iter=max_iter,
    )
    if not root.valid:
        return ProtocolThresholdResult(root, None, None, root.rejection_reasons)

    replay = evaluate_differentiable(root.root)
    if not isinstance(replay, DifferentiableThresholdTrial):
        raise TypeError(
            "Differentiable evaluator must return DifferentiableThresholdTrial."
        )
    amplitude, margin = replay.amplitude, replay.margin
    if not isinstance(amplitude, torch.Tensor) or not isinstance(margin, torch.Tensor):
        raise TypeError("Differentiable trial amplitude and margin must be tensors.")
    if replay.branch_id != root.center.branch_id:
        return ProtocolThresholdResult(
            root,
            None,
            None,
            ("differentiable_replay_branch_mismatch",),
        )
    if margin.numel() != 1 or not bool(torch.isfinite(margin.detach()).all()):
        raise ValueError("Differentiable evaluator must return a finite scalar margin.")
    replay_difference = abs(float(margin.detach()) - root.center.value)
    if replay_difference > max_replay_margin_difference:
        return ProtocolThresholdResult(
            root,
            None,
            None,
            ("differentiable_replay_mismatch",),
        )
    try:
        threshold, partial = implicit_amplitude_root_proxy(
            root,
            amplitude,
            margin,
            min_positive_margin_partial=min_positive_margin_partial,
            max_relative_amplitude_partial_error=max_relative_amplitude_partial_error,
        )
    except ValueError as error:
        return ProtocolThresholdResult(
            root,
            None,
            None,
            (f"implicit_gradient_unavailable: {error}",),
        )
    return ProtocolThresholdResult(root, threshold, partial, ())


def stack_valid_thresholds(results: Sequence[ProtocolThresholdResult]) -> torch.Tensor:
    """Stack protocol thresholds for a descriptor fit, rejecting invalid roots."""
    if not results:
        raise ValueError("Need at least one threshold result.")
    invalid = [index for index, result in enumerate(results) if not result.valid]
    if invalid:
        raise ValueError(f"Invalid threshold results at indices {invalid}.")
    return torch.stack([result.threshold.reshape(()) for result in results])
