"""Finite-scan and signed Gaussian bracket checks."""

import math

import pytest
import torch

from dendra.models.analysis.hard_transition_table import discover_hard_transitions
from dendra.models.analysis.hard_transition_validation import (
    challenge_transition_scan,
    compare_transition_scans,
    gaussian_expected_hard_bracket_bounds,
)


def test_finer_scan_preserves_one_observed_transition():
    def hard(amplitude):
        return float(amplitude >= 0.43)

    coarse = discover_hard_transitions(hard, [0.0, 1.0], tolerance=1e-5)
    fine = discover_hard_transitions(hard, [0.0, 1.0], tolerance=1e-5, scan_depth=2)
    comparison = compare_transition_scans(coarse, fine)
    assert comparison.consistent
    assert comparison.coarse_transitions == comparison.fine_transitions == 1
    assert comparison.largest_root_midpoint_shift <= 1e-5


def test_finer_scan_finds_hidden_even_number_of_jumps():
    def hard(amplitude):
        return float(0.2 <= amplitude < 0.3)

    coarse = discover_hard_transitions(hard, [0.0, 1.0], tolerance=1e-5)
    fine = discover_hard_transitions(hard, [0.0, 1.0], tolerance=1e-5, scan_depth=3)
    comparison = compare_transition_scans(coarse, fine)
    assert not comparison.consistent
    assert comparison.coarse_transitions == 0
    assert comparison.fine_transitions == 2
    assert "transition_count_changed" in comparison.reasons


def test_signed_gaussian_bracket_bounds_contain_exact_expectation():
    def hard(amplitude):
        return 1.0 + float(amplitude >= 0.25) - 2.0 * float(amplitude >= 0.75)

    table = discover_hard_transitions(hard, [0.0, 0.5, 1.0], tolerance=1e-4)
    center, sigma = 0.53, 0.2
    bounds = gaussian_expected_hard_bracket_bounds(
        table,
        center=center,
        noise_std=sigma,
    )

    def cdf(root):
        return 0.5 * math.erfc((root - center) / (sigma * math.sqrt(2.0)))

    exact = 1.0 + cdf(0.25) - 2.0 * cdf(0.75)
    assert bounds.lower <= exact <= bounds.upper
    assert bounds.lower <= bounds.midpoint <= bounds.upper
    assert bounds.upper - bounds.lower < 0.002


def test_unresolved_table_cannot_produce_bracket_bounds():
    table = discover_hard_transitions(
        lambda amplitude: float(amplitude >= 0.4),
        [0.0, 1.0],
        tolerance=1e-12,
        max_bisection_steps=2,
    )
    assert not table.valid
    with pytest.raises(ValueError, match="unresolved"):
        gaussian_expected_hard_bracket_bounds(table, center=0.5, noise_std=0.1)


def test_independent_offsets_expose_even_pair_missed_by_two_nested_scans():
    # Both dyadic scans sample exactly zero, including the two points that
    # surround this narrow positive island. The offset challenge hits it.
    def hard(amplitude):
        return float(0.275 <= amplitude < 0.280)

    coarse = discover_hard_transitions(hard, [0.0, 1.0], tolerance=1e-6)
    fine = discover_hard_transitions(
        hard,
        [0.0, 1.0],
        tolerance=1e-6,
        scan_depth=3,
    )
    assert compare_transition_scans(coarse, fine).consistent
    assert coarse.transitions == fine.transitions == ()

    challenge = challenge_transition_scan(fine, hard, max_evaluations=18)
    assert challenge.status == "scan_contradicted"
    assert challenge.endpoint_consistent is True
    assert challenge.evaluations == 18
    assert challenge.mismatches
    witness = challenge.mismatches[0]
    assert 0.275 <= witness.amplitude < 0.280
    assert witness.observed_value == 1.0
    assert witness.expected_value == 0.0
    assert witness.scan_interval_index == 2


def test_signed_step_reconstruction_and_targeted_negative_island():
    # The observed table has both positive and negative jumps. A further
    # negative island preserves the values at every dyadic scan point.
    def hard(amplitude):
        return (
            1.0
            + 2.0 * float(amplitude >= 0.2)
            - float(amplitude >= 0.8)
            - float(0.525 <= amplitude < 0.55)
        )

    table = discover_hard_transitions(
        hard,
        [0.0, 1.0],
        tolerance=1e-6,
        scan_depth=3,
    )
    assert [t.jump for t in table.transitions] == [2.0, -1.0]
    challenge = challenge_transition_scan(
        table,
        hard,
        max_evaluations=4,
        probe_fractions=(),
        interval_indices=(),
        probe_amplitudes=(0.53, 0.7),
    )
    assert challenge.status == "scan_contradicted"
    assert challenge.evaluations == 4
    assert [
        (p.amplitude, p.expected_value, p.observed_value, p.classification)
        for p in challenge.probes
    ] == [
        (0.53, 3.0, 2.0, "mismatch"),
        (0.7, 3.0, 3.0, "match"),
    ]


def test_no_mismatch_is_inconclusive_and_root_bracket_is_ambiguous():
    def hard(amplitude):
        return float(amplitude >= 0.43)

    table = discover_hard_transitions(hard, [0.0, 1.0], tolerance=0.2)
    root_bracket = table.transitions[0]
    challenge = challenge_transition_scan(
        table,
        hard,
        max_evaluations=4,
        probe_fractions=(),
        interval_indices=(),
        probe_amplitudes=(root_bracket.midpoint, 0.9),
    )
    assert challenge.status == "inconclusive"
    assert challenge.mismatches == ()
    assert [p.classification for p in challenge.probes] == [
        "within_root_bracket",
        "match",
    ]


def test_challenge_preflights_budget_and_rechecks_both_endpoints():
    table = discover_hard_transitions(lambda _: 0.0, [0.0, 1.0], tolerance=0.01)
    calls = []

    def changed_hard(amplitude):
        calls.append((amplitude, torch.is_grad_enabled()))
        return torch.tensor(2.0 if amplitude == 0.0 else 0.0, requires_grad=True)

    with pytest.raises(ValueError, match="exceeds max_evaluations"):
        challenge_transition_scan(
            table,
            changed_hard,
            max_evaluations=2,
            probe_fractions=(),
            interval_indices=(),
            probe_amplitudes=(0.4,),
        )
    assert not calls
    challenge = challenge_transition_scan(
        table,
        changed_hard,
        max_evaluations=3,
        probe_fractions=(),
        interval_indices=(),
        probe_amplitudes=(0.4,),
    )
    assert challenge.status == "protocol_changed"
    assert challenge.endpoint_consistent is False
    assert len(calls) == challenge.evaluations == 3
    assert all(not enabled for _, enabled in calls)
