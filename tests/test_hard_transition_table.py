"""Complete-protocol transition bracketing, including signed jumps."""

import pytest
import torch

from dendra.models.analysis.hard_transition_table import discover_hard_transitions


def test_signed_nonmonotone_transitions_are_bracketed():
    def hard(amplitude):
        return 0.0 if amplitude < 1.0 or amplitude >= 2.0 else 1.0

    table = discover_hard_transitions(
        hard,
        [0.0, 1.5, 3.0],
        tolerance=1e-7,
    )
    assert table.valid
    assert len(table.transitions) == 2
    assert [transition.jump for transition in table.transitions] == [1.0, -1.0]
    for transition, root in zip(table.transitions, (1.0, 2.0)):
        assert transition.lower_amplitude <= root <= transition.upper_amplitude
        assert transition.upper_amplitude - transition.lower_amplitude <= 1e-7


def test_third_midpoint_value_splits_multiple_jumps():
    def hard(amplitude):
        return 0.0 if amplitude < 1.0 else (1.0 if amplitude < 2.0 else 2.0)

    table = discover_hard_transitions(hard, [0.0, 3.0], tolerance=1e-7)
    assert table.valid
    assert len(table.transitions) == 2
    assert [transition.jump for transition in table.transitions] == [1.0, 1.0]
    assert [transition.left_value for transition in table.transitions] == [0.0, 1.0]


def test_scan_depth_exposes_a_hidden_even_number_of_jumps():
    def hard(amplitude):
        return 1.0 if 1.0 <= amplitude < 2.0 else 0.0

    coarse = discover_hard_transitions(hard, [0.0, 3.0], tolerance=1e-7)
    assert coarse.valid and coarse.transitions == ()
    refined = discover_hard_transitions(
        hard,
        [0.0, 3.0],
        tolerance=1e-7,
        scan_depth=1,
    )
    assert [transition.jump for transition in refined.transitions] == [1.0, -1.0]


def test_hard_evaluator_runs_without_autograd_and_budget_is_bounded():
    seen = []

    def hard(amplitude):
        seen.append(torch.is_grad_enabled())
        return torch.tensor(float(amplitude >= 1.0), requires_grad=True)

    table = discover_hard_transitions(hard, [0.0, 2.0], tolerance=1e-5)
    assert table.valid and not any(seen)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        discover_hard_transitions(hard, [0.0, 2.0], tolerance=1e-8, max_evaluations=3)


def test_invalid_scan_and_unresolved_bracket_are_explicit():
    with pytest.raises(ValueError, match="strictly increasing"):
        discover_hard_transitions(lambda a: 0.0, [0.0, 0.0], tolerance=0.01)
    with pytest.raises(ValueError, match="positive"):
        discover_hard_transitions(lambda a: 0.0, [0.0, 1.0], tolerance=0.0)
    table = discover_hard_transitions(
        lambda a: float(a >= 1.0),
        [0.0, 2.0],
        tolerance=1e-9,
        max_bisection_steps=1,
    )
    assert not table.valid and len(table.unresolved) == 1
    assert (
        table.unresolved[0].upper_amplitude - table.unresolved[0].lower_amplitude > 1e-9
    )
