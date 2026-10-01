"""Numerical checks for observed hard-descriptor transitions.

These checks concern the complete hard protocol supplied to
``discover_hard_transitions``. Agreement between two finite scans is useful
evidence, but cannot prove that arbitrarily narrow hidden transitions are
absent. Gaussian bounds account only for the observed root brackets; the
caller must bound unscanned stimulus tails separately.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .hard_transition_table import HardTransitionTable


@dataclass(frozen=True)
class TransitionScanComparison:
    """Observed agreement between a coarse and a finer scan."""

    consistent: bool
    reasons: tuple[str, ...]
    coarse_transitions: int
    fine_transitions: int
    largest_root_midpoint_shift: float | None


@dataclass(frozen=True)
class GaussianTransitionBracketBounds:
    """Expected-hard value and bounds due only to observed root brackets."""

    midpoint: float
    lower: float
    upper: float


@dataclass(frozen=True)
class TransitionChallengeProbe:
    """One independent hard rerun compared with the observed signed steps."""

    amplitude: float
    observed_value: float
    expected_value: float | None
    scan_interval_index: int
    classification: str


@dataclass(frozen=True)
class TransitionScanChallenge:
    """Finite-budget challenge; ``inconclusive`` never certifies completeness."""

    status: str
    probes: tuple[TransitionChallengeProbe, ...]
    endpoint_consistent: bool | None
    evaluations: int
    challenged_interval_indices: tuple[int, ...]

    @property
    def mismatches(self) -> tuple[TransitionChallengeProbe, ...]:
        return tuple(
            probe for probe in self.probes if probe.classification == "mismatch"
        )


def challenge_transition_scan(
    table: HardTransitionTable,
    evaluate: Callable[[float], float | torch.Tensor],
    *,
    max_evaluations: int,
    probe_fractions: Sequence[float] = (0.21132486540518713, 0.7886751345948129),
    interval_indices: Sequence[int] | None = None,
    probe_amplitudes: Sequence[float] = (),
    value_tolerance: float = 0.0,
    recheck_endpoints: bool = True,
) -> TransitionScanChallenge:
    r"""Challenge an observed transition table with independent hard probes.

    In each selected *original scan* interval, sample the interior at the
    specified non-dyadic fractions. Extra amplitudes can target suspicious
    intervals. Compare each complete hard rerun with the step function
    reconstructed from the table's signed jumps. A probe inside an observed
    root bracket is marked ambiguous, because either adjacent value is
    compatible with that bracket. Endpoint reruns optionally detect a changed
    or nondeterministic hard protocol. All calls use ``torch.no_grad()``.

    ``status`` is ``"scan_contradicted"`` if an unambiguous probe differs,
    ``"protocol_changed"`` if either endpoint changed on rerun, and
    ``"inconclusive"`` otherwise. Even two agreeing scans plus this challenge
    cannot exclude narrower transitions between finite probes. The budget is
    checked *before* any callback call and counts endpoint reruns. Select
    ``interval_indices`` or supply ``probe_amplitudes`` to focus the budget
    where tightly packed transitions are plausible. The evaluator must run
    the same complete hard protocol from the same initialized state.
    """
    if not isinstance(table, HardTransitionTable):
        raise TypeError("table must be a HardTransitionTable.")
    if not table.valid:
        raise ValueError("Cannot challenge an unresolved transition table.")
    if not callable(evaluate):
        raise TypeError("evaluate must be callable.")
    if not isinstance(max_evaluations, int) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer.")
    if not math.isfinite(value_tolerance) or value_tolerance < 0:
        raise ValueError("value_tolerance must be finite and nonnegative.")
    if len(table.scan_amplitudes) < 2 or len(table.scan_amplitudes) != len(
        table.scan_values
    ):
        raise ValueError(
            "Transition table must have paired scan amplitudes and values."
        )
    scan = table.scan_amplitudes
    if any(not math.isfinite(x) for x in scan) or any(
        b <= a for a, b in zip(scan, scan[1:])
    ):
        raise ValueError(
            "Transition table scan amplitudes must be strictly increasing."
        )

    try:
        fractions = tuple(float(f) for f in probe_fractions)
        selected = (
            tuple(range(len(scan) - 1))
            if interval_indices is None
            else tuple(interval_indices)
        )
        targeted = tuple(float(a) for a in probe_amplitudes)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Probe fractions, interval indices, and amplitudes must be valid."
        ) from error
    if any(not math.isfinite(f) or not 0 < f < 1 for f in fractions):
        raise ValueError("probe_fractions must be finite and strictly inside (0, 1).")
    if any(not isinstance(i, int) or i < 0 or i >= len(scan) - 1 for i in selected):
        raise ValueError("interval_indices must index original scan intervals.")
    if any(not math.isfinite(a) or not scan[0] < a < scan[-1] for a in targeted):
        raise ValueError("probe_amplitudes must be finite and inside the scan range.")

    amplitudes: set[float] = set(targeted)
    for i in selected:
        left, right = scan[i : i + 2]
        for fraction in fractions:
            amplitude = left + (right - left) * fraction
            if not left < amplitude < right:
                raise ValueError(
                    "An offset probe is not representably inside its interval."
                )
            amplitudes.add(amplitude)
    if not amplitudes:
        raise ValueError("Specify at least one offset or targeted probe.")
    probe_points = tuple(sorted(amplitudes))
    endpoint_points = (scan[0], scan[-1]) if recheck_endpoints else ()
    if len(probe_points) + len(endpoint_points) > max_evaluations:
        raise ValueError("Requested challenge exceeds max_evaluations before reruns.")

    def hard_value(amplitude: float) -> float:
        with torch.no_grad():
            output = evaluate(amplitude)
        if isinstance(output, torch.Tensor):
            if output.numel() != 1:
                raise ValueError("Hard evaluator must return a scalar.")
            value = float(output.detach())
        else:
            value = float(output)
        if not math.isfinite(value):
            raise ValueError("Hard evaluator returned a nonfinite value.")
        return value

    endpoint_consistent: bool | None = None
    if recheck_endpoints:
        endpoint_observations = tuple(
            hard_value(amplitude) for amplitude in endpoint_points
        )
        endpoint_consistent = all(
            abs(observed - expected) <= value_tolerance
            for observed, expected in zip(
                endpoint_observations, (table.scan_values[0], table.scan_values[-1])
            )
        )

    probes: list[TransitionChallengeProbe] = []
    challenged_intervals: set[int] = set()
    for amplitude in probe_points:
        interval = min(bisect_right(scan, amplitude) - 1, len(scan) - 2)
        challenged_intervals.add(interval)
        observed = hard_value(amplitude)
        if any(
            t.lower_amplitude <= amplitude <= t.upper_amplitude
            for t in table.transitions
        ):
            expected = None
            classification = "within_root_bracket"
        else:
            expected = table.scan_values[0] + sum(
                t.jump for t in table.transitions if t.upper_amplitude < amplitude
            )
            classification = (
                "match" if abs(observed - expected) <= value_tolerance else "mismatch"
            )
        probes.append(
            TransitionChallengeProbe(
                amplitude,
                observed,
                expected,
                interval,
                classification,
            )
        )

    status = (
        "protocol_changed"
        if endpoint_consistent is False
        else "scan_contradicted"
        if any(p.classification == "mismatch" for p in probes)
        else "inconclusive"
    )
    return TransitionScanChallenge(
        status,
        tuple(probes),
        endpoint_consistent,
        len(probe_points) + len(endpoint_points),
        tuple(sorted(challenged_intervals)),
    )


def compare_transition_scans(
    coarse: HardTransitionTable,
    fine: HardTransitionTable,
) -> TransitionScanComparison:
    """Check whether refining a hard scan preserved its observed jumps.

    The scans must cover identical endpoints. Every amplitude sampled by the
    coarse scan must occur in the fine scan with the same hard value. The
    observed jump sequences must have equal length, left/right values, and
    overlapping root brackets. A consistent result is a *finite-resolution*
    check, not certification of all transitions in the continuum.
    """
    if not isinstance(coarse, HardTransitionTable) or not isinstance(
        fine, HardTransitionTable
    ):
        raise TypeError("Both scans must be HardTransitionTable instances.")
    reasons: list[str] = []
    if not coarse.valid or not fine.valid:
        reasons.append("unresolved_transition")
    if (
        coarse.scan_amplitudes[0] != fine.scan_amplitudes[0]
        or coarse.scan_amplitudes[-1] != fine.scan_amplitudes[-1]
    ):
        reasons.append("scan_range_changed")
    if fine.scan_depth <= coarse.scan_depth:
        reasons.append("fine_scan_not_deeper")

    fine_samples = dict(zip(fine.scan_amplitudes, fine.scan_values))
    if any(
        fine_samples.get(amplitude) != value
        for amplitude, value in zip(coarse.scan_amplitudes, coarse.scan_values)
    ):
        reasons.append("shared_sample_changed_or_missing")

    n_coarse = len(coarse.transitions)
    n_fine = len(fine.transitions)
    largest_shift: float | None = None
    if n_coarse != n_fine:
        reasons.append("transition_count_changed")
    else:
        shifts = []
        for old, new in zip(coarse.transitions, fine.transitions):
            if old.left_value != new.left_value or old.right_value != new.right_value:
                reasons.append("jump_signature_changed")
            if (
                old.upper_amplitude < new.lower_amplitude
                or new.upper_amplitude < old.lower_amplitude
            ):
                reasons.append("root_brackets_disjoint")
            shifts.append(abs(old.midpoint - new.midpoint))
        if shifts:
            largest_shift = max(shifts)
    return TransitionScanComparison(
        not reasons,
        tuple(dict.fromkeys(reasons)),
        n_coarse,
        n_fine,
        largest_shift,
    )


def gaussian_expected_hard_bracket_bounds(
    table: HardTransitionTable,
    *,
    center: float,
    noise_std: float,
) -> GaussianTransitionBracketBounds:
    r"""Propagate signed, observed root brackets into a Gaussian expectation.

    For each signed jump ``Delta`` at a root bracket ``[lo, hi]``, bound
    ``Delta Phi((center-root)/noise_std)`` by evaluating both endpoints.
    Summing these bounds handles nonmonotone descriptors without assuming
    that jumps all have one sign. The midpoint uses each bracket midpoint.

    This is conditional on the observed transitions being complete throughout
    the Gaussian stimulus support. It does not bound omitted transitions or
    behavior outside the finite scan range. Use ``compare_transition_scans``
    and a caller-supplied tail bound before treating it as a numerical
    uncertainty interval for the full expected-hard protocol.
    """
    if not isinstance(table, HardTransitionTable):
        raise TypeError("table must be a HardTransitionTable.")
    if not table.valid:
        raise ValueError("Cannot bound an unresolved transition table.")
    if not math.isfinite(center) or not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("Center must be finite and noise_std finite and positive.")
    if not table.scan_values:
        raise ValueError("Transition table must contain hard scan values.")
    if not math.isclose(
        table.scan_values[0] + sum(item.jump for item in table.transitions),
        table.scan_values[-1],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Transition jumps do not reconstruct the final hard value.")

    def cdf(root: float) -> float:
        return 0.5 * math.erfc((root - center) / (noise_std * math.sqrt(2.0)))

    midpoint = lower = upper = table.scan_values[0]
    for transition in table.transitions:
        jump = transition.jump
        at_lower = jump * cdf(transition.lower_amplitude)
        at_upper = jump * cdf(transition.upper_amplitude)
        midpoint += jump * cdf(transition.midpoint)
        lower += min(at_lower, at_upper)
        upper += max(at_lower, at_upper)
    return GaussianTransitionBracketBounds(midpoint, lower, upper)
