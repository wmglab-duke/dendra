"""Independent threshold searches in a shared, caller-defined batch trial.

One hard protocol call evaluates all lanes at each bisection amplitude vector.
The differentiable replay is likewise one batched call. Nothing in this module
selects a model, stimulus waveform, observation window, or amplitude unit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import torch

from .threshold_event_root import MarginObservation, RootResult, RootValidation
from .threshold_protocol import ProtocolThresholdResult


@dataclass(frozen=True)
class BatchedThresholdTrial:
    """Hard decisions and matching signed margins for independent lanes."""

    decisions: torch.Tensor
    margins: torch.Tensor
    branch_ids: Sequence[Hashable | None] | None = None


@dataclass(frozen=True)
class DifferentiableBatchedThresholdTrial:
    """Autograd replay at the accepted amplitude vector."""

    amplitudes: torch.Tensor
    margins: torch.Tensor
    branch_ids: Sequence[Hashable | None] | None = None


@dataclass(frozen=True)
class BatchedProtocolThresholdResult:
    """One explicit scalar result per lane, including rejected lanes."""

    results: tuple[ProtocolThresholdResult, ...]

    @property
    def valid_mask(self) -> torch.Tensor:
        device = self.results[0].root.left.margin.device
        return torch.tensor(
            [result.valid for result in self.results], dtype=torch.bool, device=device
        )


def _branch_ids(
    values: Sequence[Hashable | None] | None,
    count: int,
) -> tuple[Hashable | None, ...]:
    if values is None:
        return (None,) * count
    if len(values) != count:
        raise ValueError("Expected one branch ID per lane.")
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError("Branch ID tensor must be one-dimensional.")
        return tuple(values.detach().tolist())
    return tuple(values)


def differentiable_batched_protocol_thresholds(
    evaluate_hard: Callable[[torch.Tensor], BatchedThresholdTrial],
    evaluate_differentiable: Callable[
        [torch.Tensor], DifferentiableBatchedThresholdTrial
    ],
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    max_amplitude_bracket: float,
    max_margin_jump: float,
    max_center_residual: float,
    max_replay_margin_difference: float,
    min_positive_margin_partial: float,
    max_cross_amplitude_partial: float,
    max_relative_amplitude_partial_error: float | None = None,
    max_iter: int = 23,
) -> BatchedProtocolThresholdResult:
    """Differentiate independent threshold lanes using vectorized hard trials.

    ``evaluate_hard`` must report the complete hard decision and a signed
    margin per lane. Every decision is checked against ``margin >= 0``. Each
    lane is bracketed and validated separately; a failed lane has no gradient.

    ``evaluate_differentiable`` reruns the whole batch at the vector of root
    amplitudes. Its amplitude tensor must be the vector actually used by the
    simulation, not a view created after the simulation. A lane's margin may
    depend on its own amplitude and shared
    model parameters, but dependence on another lane's amplitude is rejected.
    That independence is required for the per-lane implicit derivative and
    elementwise bisection. The coupling tolerance has margin/amplitude units.
    The optional relative partial check compares each replay amplitude slope
    with its hard bracket secant; use a bracket that resolves the hard margin
    before interpreting this diagnostic.

    This first implementation uses one reverse-mode sweep per valid lane to
    check amplitude independence. That cost can be substantial for large
    populations even though hard search simulations are batched.
    """
    if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
        raise TypeError(
            "Amplitude bounds must be tensors with the model's device and dtype."
        )
    if lower.ndim != 1 or upper.shape != lower.shape or lower.numel() == 0:
        raise ValueError("Amplitude bounds must be nonempty vectors of equal shape.")
    if not lower.is_floating_point() or not upper.is_floating_point():
        raise ValueError("Amplitude bounds must have floating-point dtype.")
    if lower.device != upper.device or lower.dtype != upper.dtype:
        raise ValueError("Amplitude bounds must share device and dtype.")
    if not bool(
        (torch.isfinite(lower) & torch.isfinite(upper) & (lower < upper)).all()
    ):
        raise ValueError("Need finite ordered amplitude bounds in every lane.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    positive = (
        max_amplitude_bracket,
        max_margin_jump,
        max_center_residual,
        min_positive_margin_partial,
        max_cross_amplitude_partial,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("Positive tolerances must be finite and positive.")
    if (
        not math.isfinite(max_replay_margin_difference)
        or max_replay_margin_difference < 0
    ):
        raise ValueError("Replay margin tolerance must be finite and nonnegative.")
    if max_relative_amplitude_partial_error is not None and (
        not math.isfinite(max_relative_amplitude_partial_error)
        or max_relative_amplitude_partial_error < 0
    ):
        raise ValueError(
            "Amplitude partial error tolerance must be finite and nonnegative."
        )

    count = lower.numel()
    lo = lower.detach().clone()
    hi = upper.detach().clone()

    def checked_hard(amplitudes: torch.Tensor):
        with torch.no_grad():
            trial = evaluate_hard(amplitudes)
        if not isinstance(trial, BatchedThresholdTrial):
            raise TypeError("Hard evaluator must return BatchedThresholdTrial.")
        if not isinstance(trial.decisions, torch.Tensor) or not isinstance(
            trial.margins, torch.Tensor
        ):
            raise TypeError("Batched hard decisions and margins must be tensors.")
        if trial.decisions.dtype != torch.bool:
            raise TypeError("Hard decisions must have boolean dtype.")
        if not trial.margins.is_floating_point():
            raise TypeError("Hard margins must have floating-point dtype.")
        if trial.decisions.device != lo.device or trial.margins.device != lo.device:
            raise ValueError(
                "Hard trial outputs must use the amplitude bounds' device."
            )
        decisions = trial.decisions.detach()
        margins = trial.margins.detach()
        if decisions.shape != lo.shape or margins.shape != lo.shape:
            raise ValueError("Expected one hard decision and margin per lane.")
        if not bool(torch.isfinite(margins).all()):
            raise ValueError("Hard margins must be finite.")
        if not bool(torch.equal(decisions, margins >= 0)):
            raise ValueError("Signed margins and hard decisions disagree.")
        return margins, _branch_ids(trial.branch_ids, count)

    left_margin, left_ids = checked_hard(lo)
    right_margin, right_ids = checked_hard(hi)
    if not bool(((left_margin < 0) & (right_margin >= 0)).all()):
        raise ValueError(
            "Every lane needs negative lower and nonnegative upper margins."
        )

    resolution_lost = torch.zeros(count, dtype=torch.bool, device=lo.device)
    iterations = 0
    for _ in range(max_iter):
        middle = (lo + hi) / 2
        stalled = (middle == lo) | (middle == hi)
        resolution_lost |= stalled
        if bool(stalled.all()):
            break
        middle_margin, middle_ids = checked_hard(middle)
        iterations += 1
        active = middle_margin >= 0
        hi = torch.where(active, middle, hi)
        lo = torch.where(active, lo, middle)
        right_margin = torch.where(active, middle_margin, right_margin)
        left_margin = torch.where(active, left_margin, middle_margin)
        active_list = active.tolist()
        right_ids = tuple(
            middle_ids[i] if active_list[i] else right_ids[i] for i in range(count)
        )
        left_ids = tuple(
            left_ids[i] if active_list[i] else middle_ids[i] for i in range(count)
        )

    centers = (lo + hi) / 2
    center_margin, center_ids = checked_hard(centers)
    roots = []
    for i in range(count):
        ids = (left_ids[i], right_ids[i], center_ids[i])
        if all(branch_id is None for branch_id in ids):
            branch_consistent = None
        else:
            branch_consistent = (
                all(branch_id is not None for branch_id in ids)
                and ids[0] == ids[1] == ids[2]
            )
        width = float(hi[i] - lo[i])
        span = float(right_margin[i] - left_margin[i])
        residual = abs(float(center_margin[i]))
        validation = RootValidation(
            iterations=iterations,
            amplitude_bracket_width=width,
            margin_span=span,
            absolute_center_residual=residual,
            secant_margin_partial=span / width if width > 0 else float("nan"),
            branch_consistent=branch_consistent,
            max_amplitude_bracket=max_amplitude_bracket,
            max_margin_jump=max_margin_jump,
            max_center_residual=max_center_residual,
        )
        reasons = []
        if bool(resolution_lost[i]) or width <= 0:
            reasons.append("amplitude_resolution_lost")
        if width > max_amplitude_bracket:
            reasons.append("amplitude_bracket_too_wide")
        if span > max_margin_jump:
            reasons.append("margin_jump")
        if branch_consistent is False:
            reasons.append(
                "branch_changed"
                if all(x is not None for x in ids)
                else "branch_metadata_incomplete"
            )
        if residual > max_center_residual:
            reasons.append("root_residual")
        roots.append(
            RootResult(
                float(centers[i]),
                float(lo[i]),
                float(hi[i]),
                MarginObservation(left_margin[i], left_ids[i]),
                MarginObservation(right_margin[i], right_ids[i]),
                MarginObservation(center_margin[i], center_ids[i]),
                not reasons,
                tuple(reasons),
                validation,
            )
        )

    if not any(root.valid for root in roots):
        return BatchedProtocolThresholdResult(
            tuple(
                ProtocolThresholdResult(root, None, None, root.rejection_reasons)
                for root in roots
            )
        )

    replay = evaluate_differentiable(centers)
    if not isinstance(replay, DifferentiableBatchedThresholdTrial):
        raise TypeError(
            "Differentiable evaluator must return DifferentiableBatchedThresholdTrial."
        )
    amplitudes, margins = replay.amplitudes, replay.margins
    if not isinstance(amplitudes, torch.Tensor) or not isinstance(
        margins, torch.Tensor
    ):
        raise TypeError("Differentiable amplitudes and margins must be tensors.")
    if amplitudes.shape != lo.shape or margins.shape != lo.shape:
        raise ValueError(
            "Differentiable trial must return one amplitude and margin per lane."
        )
    if not amplitudes.is_floating_point() or not margins.is_floating_point():
        raise ValueError(
            "Differentiable amplitudes and margins must be floating point."
        )
    replay_ids = _branch_ids(replay.branch_ids, count)
    results = []
    for i, root in enumerate(roots):
        if not root.valid:
            results.append(
                ProtocolThresholdResult(root, None, None, root.rejection_reasons)
            )
            continue
        reasons = []
        if replay_ids[i] != root.center.branch_id:
            reasons.append("differentiable_replay_branch_mismatch")
        if not bool(torch.isfinite(margins[i].detach())):
            reasons.append("differentiable_margin_nonfinite")
        elif (
            abs(float(margins[i].detach()) - root.center.value)
            > max_replay_margin_difference
        ):
            reasons.append("differentiable_replay_mismatch")
        if abs(float(amplitudes[i].detach()) - root.root) > max(
            8 * torch.finfo(amplitudes.dtype).eps * max(1.0, abs(root.root)),
            root.validation.amplitude_bracket_width / 2,
        ):
            reasons.append("differentiable_amplitude_mismatch")
        if reasons:
            results.append(ProtocolThresholdResult(root, None, None, tuple(reasons)))
            continue
        if not amplitudes.requires_grad or not margins[i].requires_grad:
            results.append(
                ProtocolThresholdResult(
                    root,
                    None,
                    None,
                    ("implicit_gradient_unavailable",),
                )
            )
            continue
        (row,) = torch.autograd.grad(
            margins[i],
            amplitudes,
            retain_graph=True,
            allow_unused=True,
        )
        if row is None:
            results.append(
                ProtocolThresholdResult(
                    root,
                    None,
                    None,
                    ("amplitude_autograd_connection_missing",),
                )
            )
            continue
        partial = float(row[i].detach())
        other = row.detach().clone()
        other[i] = 0
        if not bool(torch.isfinite(row).all()):
            reasons.append("amplitude_partial_nonfinite")
        elif bool((other.abs() > max_cross_amplitude_partial).any()):
            reasons.append("cross_amplitude_coupling")
        if not math.isfinite(partial) or partial <= min_positive_margin_partial:
            reasons.append("amplitude_partial_not_positive")
        if max_relative_amplitude_partial_error is not None:
            secant = root.validation.secant_margin_partial
            relative_error = abs(partial - secant) / max(
                abs(partial), abs(secant), min_positive_margin_partial
            )
            if (
                not math.isfinite(relative_error)
                or relative_error > max_relative_amplitude_partial_error
            ):
                reasons.append("replay_amplitude_partial_mismatch")
        if abs(float(margins[i].detach())) > max_center_residual:
            reasons.append("differentiable_root_residual")
        if reasons:
            results.append(ProtocolThresholdResult(root, None, None, tuple(reasons)))
            continue
        # Subtract the replay's direct amplitude tangent while keeping its
        # derivative through shared model parameters.
        parameter_margin = (
            margins[i]
            - margins[i].detach()
            - torch.dot(row.detach(), amplitudes - amplitudes.detach())
        )
        threshold = (
            amplitudes.new_tensor(root.root) - parameter_margin / row[i].detach()
        )
        results.append(ProtocolThresholdResult(root, threshold, partial, ()))

    return BatchedProtocolThresholdResult(tuple(results))
