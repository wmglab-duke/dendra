"""Bracket observed jumps of a complete hard descriptor along one stimulus axis.

The caller supplies the *entire* hard protocol as ``evaluate(amplitude)``.
This routine can refine transitions exposed by its scan; no finite set of
samples can certify that narrower even-numbered jumps are absent between
samples. Record the scan range and refinement level with every result.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardTransition:
    """One observed jump, bracketed in the caller's amplitude units."""

    lower_amplitude: float
    upper_amplitude: float
    left_value: float
    right_value: float

    @property
    def jump(self) -> float:
        return self.right_value - self.left_value

    @property
    def midpoint(self) -> float:
        return (self.lower_amplitude + self.upper_amplitude) / 2


@dataclass(frozen=True)
class HardTransitionTable:
    """Observed transition table and its finite scan resolution."""

    scan_amplitudes: tuple[float, ...]
    scan_values: tuple[float, ...]
    transitions: tuple[HardTransition, ...]
    unresolved: tuple[HardTransition, ...]
    tolerance: float
    scan_depth: int
    evaluations: int

    @property
    def valid(self) -> bool:
        return not self.unresolved


def discover_hard_transitions(
    evaluate: Callable[[float], float | torch.Tensor],
    amplitudes: Sequence[float],
    *,
    tolerance: float,
    scan_depth: int = 0,
    max_bisection_steps: int = 40,
    max_evaluations: int = 10000,
) -> HardTransitionTable:
    r"""Refine every *observed* descriptor jump with complete hard reruns.

    Initial amplitudes must be increasing. ``scan_depth`` adds dyadic probes
    to every initial interval before transition refinement; depth ``d`` uses
    ``2**d`` subintervals. Adjacent unequal hard values are bisected. If a
    midpoint has a third value, both halves are refined, retaining signed
    jumps. The returned table is useful for a signed-root Gaussian
    convolution only after the caller has checked scan-resolution stability
    over the relevant stimulus support. A valid table means all *observed*
    jumps reached tolerance, not that hidden transitions were ruled out.

    The evaluator runs under ``torch.no_grad()`` and must restore the same
    initialized state at every amplitude. The routine does no model-specific
    stimulus, callback, site, or noise selection.
    """
    if not callable(evaluate):
        raise TypeError("evaluate must be callable.")
    try:
        base = tuple(float(value) for value in amplitudes)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Amplitudes must be a finite increasing sequence.") from error
    if (
        len(base) < 2
        or any(not math.isfinite(value) for value in base)
        or any(b <= a for a, b in zip(base, base[1:]))
    ):
        raise ValueError(
            "Amplitudes must be a finite strictly increasing sequence of length >= 2."
        )
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive.")
    if not isinstance(scan_depth, int) or scan_depth < 0 or scan_depth > 20:
        raise ValueError("Scan depth must be an integer from 0 to 20.")
    if not isinstance(max_bisection_steps, int) or max_bisection_steps < 1:
        raise ValueError("max_bisection_steps must be positive.")
    if not isinstance(max_evaluations, int) or max_evaluations < 2:
        raise ValueError("max_evaluations must be at least 2.")

    cache: dict[float, float] = {}

    def hard_value(amplitude: float) -> float:
        if amplitude not in cache:
            if len(cache) >= max_evaluations:
                raise RuntimeError("Hard transition evaluation budget exceeded.")
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
            cache[amplitude] = value
        return cache[amplitude]

    scan = [base[0]]
    n_subintervals = 1 << scan_depth
    if (len(base) - 1) * n_subintervals + 1 > max_evaluations:
        raise ValueError("Requested scan exceeds max_evaluations.")
    for left, right in zip(base, base[1:]):
        scan.extend(
            left + (right - left) * index / n_subintervals
            for index in range(1, n_subintervals + 1)
        )
    if any(not math.isfinite(value) for value in scan) or any(
        right <= left for left, right in zip(scan, scan[1:])
    ):
        raise ValueError("Scan amplitudes are not representably increasing.")
    scan_values = [hard_value(amplitude) for amplitude in scan]
    transitions: list[HardTransition] = []
    unresolved: list[HardTransition] = []

    def refine(
        left: float, right: float, left_value: float, right_value: float, steps: int
    ) -> None:
        if left_value == right_value:
            return
        if right - left <= tolerance:
            transitions.append(HardTransition(left, right, left_value, right_value))
            return
        if steps >= max_bisection_steps:
            unresolved.append(HardTransition(left, right, left_value, right_value))
            return
        middle = (left + right) / 2
        if middle == left or middle == right:
            unresolved.append(HardTransition(left, right, left_value, right_value))
            return
        middle_value = hard_value(middle)
        refine(left, middle, left_value, middle_value, steps + 1)
        refine(middle, right, middle_value, right_value, steps + 1)

    for left, right, left_value, right_value in zip(
        scan, scan[1:], scan_values, scan_values[1:]
    ):
        refine(left, right, left_value, right_value, 0)
    transitions.sort(key=lambda item: item.midpoint)
    unresolved.sort(key=lambda item: item.midpoint)
    return HardTransitionTable(
        tuple(scan),
        tuple(scan_values),
        tuple(transitions),
        tuple(unresolved),
        tolerance,
        scan_depth,
        len(cache),
    )
