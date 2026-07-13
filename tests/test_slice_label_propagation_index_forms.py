"""Index-syntax stress tests for exact Slice-label propagation."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import dendra as dn
from dendra.models.slice import Slice

DTYPE = torch.float64


def _population() -> dn.Population:
    population = dn.Population(N=4, C=5, dtype=DTYPE)
    population.v.copy_(torch.arange(20, dtype=DTYPE).reshape(4, 5))
    return population


def _coordinates(population: dn.Population, selection) -> torch.Tensor:
    grid = torch.arange(population.v.numel(), device=population.device()).reshape(
        population.shape
    )
    return grid[selection.index]


def _assert_same_selection(left, right) -> None:
    assert tuple(left.shape) == tuple(right.shape)
    assert left.is_scalar is right.is_scalar
    assert torch.equal(left.v, right.v)
    assert torch.equal(
        _coordinates(left.root_model, left),
        _coordinates(right.root_model, right),
    )


@pytest.mark.parametrize(
    ("population_key", "label_key"),
    [
        pytest.param(Ellipsis, Ellipsis, id="ellipsis"),
        pytest.param(
            (slice(None), slice(None)),
            slice(None),
            id="redundant-full-compartment-slice",
        ),
        pytest.param((None, Ellipsis), (None, Ellipsis), id="leading-newaxis"),
        pytest.param(
            (slice(None), None, Ellipsis),
            (slice(None), None, Ellipsis),
            id="middle-newaxis-with-ellipsis",
        ),
        pytest.param(
            ([3, 1, 3], None, slice(None)),
            ([3, 1, 3], None),
            id="reordered-duplicate-rows-and-newaxis",
        ),
        pytest.param(
            ([3, 1, 3], Ellipsis, None),
            ([3, 1, 3], Ellipsis, None),
            id="reordered-rows-and-trailing-newaxis",
        ),
    ],
)
def test_scalar_compartment_label_propagates_across_equivalent_key_forms(
    population_key, label_key
):
    population = _population()
    population[:, 0].label("soma")

    _assert_same_selection(
        population[population_key].soma,
        population.soma[label_key],
    )


@settings(max_examples=40, deadline=None)
@given(
    rows=st.lists(st.integers(min_value=0, max_value=3), max_size=7),
    leading_newaxis=st.booleans(),
    trailing_newaxis=st.booleans(),
)
def test_redundant_full_slice_and_newaxes_preserve_reordered_row_semantics(
    rows, leading_newaxis, trailing_newaxis
):
    population = _population()
    population[:, 0].label("soma")

    if trailing_newaxis:
        population_key = (*((None,) if leading_newaxis else ()), rows, Ellipsis, None)
        label_key = population_key
    else:
        population_key = (
            *((None,) if leading_newaxis else ()),
            rows,
            slice(None),
        )
        label_key = (*((None,) if leading_newaxis else ()), rows)

    _assert_same_selection(
        population[population_key].soma,
        population.soma[label_key],
    )


@pytest.mark.parametrize(
    ("label_key", "expected_key"),
    [
        pytest.param(
            (slice(None), 0),
            (slice(None), None),
            id="trailing-axis-label",
        ),
        pytest.param(
            (0, slice(None)),
            (None, slice(None)),
            id="leading-axis-label",
        ),
    ],
)
def test_ambiguous_full_slice_removal_never_silently_changes_newaxis_position(
    label_key, expected_key
):
    population = _population()
    population[label_key].label("region")
    expected = population.region[expected_key]

    try:
        propagated = population[:, None, :].region
    except AttributeError:
        # Rejection is the safe result when exact axis correspondence cannot be
        # proved.  A future provenance-aware implementation may propagate it.
        return

    _assert_same_selection(propagated, expected)


def test_propagation_replays_multiple_batch_axes_and_paired_advanced_indices():
    population = _population()
    population[:, [4, 1]].label("landmarks")
    population.batch(2)
    population.batch(3)

    key = (
        torch.tensor([2, 0, 2]),
        slice(None),
        torch.tensor([3, 1, 3]),
        slice(None),
    )

    _assert_same_selection(population[key].landmarks, population.landmarks[key])


def test_retained_newaxis_and_duplicate_selection_rebases_before_propagation():
    population = _population()
    population[:, [4, 1]].label("landmarks")
    selected = population[None, [3, 1, 3], :]

    population.batch(2)
    population.batch(3)

    _assert_same_selection(
        selected.landmarks,
        population.landmarks[:, :, None, [3, 1, 3], :],
    )


def test_nested_exact_keys_preserve_label_scope_and_shape():
    population = _population()
    population[:, [4, 1]].label("landmarks")

    selected = population[[3, 1, 3], None, :][..., 0, :]
    expected = population.landmarks[[3, 1, 3], None, :][..., 0, :]

    _assert_same_selection(selected.landmarks, expected)


def test_deep_noop_slice_chain_has_bounded_label_replay(monkeypatch):
    population = _population()
    population[:, 0].label("soma")
    selected = population
    for _ in range(20):
        selected = selected[:]

    original_getitem = Slice.__getitem__
    replay_calls = 0

    def counted_getitem(self, key):
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls > 200:
            raise AssertionError(
                "Label propagation replayed more than 200 candidate keys for "
                "20 semantically identical no-op Slice ancestors."
            )
        return original_getitem(self, key)

    monkeypatch.setattr(Slice, "__getitem__", counted_getitem)

    propagated = selected.soma

    _assert_same_selection(propagated, population.soma)
    assert replay_calls <= 200


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(
            (slice(None), None, slice(2, None)),
            id="compartment-tail-after-newaxis",
        ),
        pytest.param(
            (torch.tensor([0, 2]), torch.tensor([1, 0])),
            id="paired-cell-and-compartment-indices",
        ),
    ],
)
def test_noncommuting_key_forms_require_explicit_intersection(key):
    population = _population()
    population[:, [4, 1]].label("landmarks")
    selected = population[key]

    with pytest.raises(AttributeError, match="does not commute exactly"):
        _ = selected.landmarks

    overlap = selected.intersect(population.landmarks)
    selected_coordinates = _coordinates(population, selected).reshape(-1)
    label_support = _coordinates(population, population.landmarks).reshape(-1)
    expected_coordinates = selected_coordinates[
        torch.isin(selected_coordinates, label_support)
    ]

    assert tuple(overlap.shape) == (int(expected_coordinates.numel()),)
    assert torch.equal(
        _coordinates(population, overlap).reshape(-1),
        expected_coordinates,
    )
