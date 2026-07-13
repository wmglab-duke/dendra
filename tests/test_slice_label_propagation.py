"""Contracts for label propagation and explicit physical Slice intersection."""

from __future__ import annotations

import copy

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import dendra as dn
from dendra.models.mod import pas

DTYPE = torch.float64


def _population(*, n: int = 3, c: int = 5) -> dn.Population:
    population = dn.Population(N=n, C=c, dtype=DTYPE)
    population.v.copy_(
        torch.arange(n * c, dtype=DTYPE, device=population.device()).reshape(n, c)
    )
    return population


def _flat_coordinates(population, selection) -> torch.Tensor:
    grid = torch.arange(
        population.v.numel(), device=population.device(), dtype=torch.long
    ).reshape(population.shape)
    return grid[selection.index].reshape(-1)


def _assert_same_selection(left, right) -> None:
    assert tuple(left.shape) == tuple(right.shape)
    assert left.is_scalar is right.is_scalar
    assert torch.equal(left.v, right.v)
    assert torch.equal(
        _flat_coordinates(left.root_model, left),
        _flat_coordinates(right.root_model, right),
    )


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(1, id="integer"),
        pytest.param(slice(1, None), id="slice"),
        pytest.param([2, 0], id="list"),
        pytest.param(torch.tensor([2, 0]), id="integer-tensor"),
        pytest.param(
            torch.tensor([True, False, True]),
            id="boolean-tensor",
        ),
        pytest.param([2, 0, 2], id="reordered-duplicates"),
    ],
)
def test_leading_index_propagates_population_label_exactly(key):
    population = _population()
    population[:, [3, 0]].label("ends")

    propagated = population[key].ends
    explicit = population.ends[key]

    _assert_same_selection(propagated, explicit)


@settings(max_examples=60, deadline=None)
@given(
    rows=st.lists(st.integers(min_value=0, max_value=3), max_size=8),
    compartments=st.lists(
        st.integers(min_value=0, max_value=4),
        min_size=1,
        max_size=5,
        unique=True,
    ),
)
def test_leading_advanced_index_propagation_matches_explicit_label(rows, compartments):
    population = _population(n=4)
    population[:, compartments].label("region")

    _assert_same_selection(population[rows].region, population.region[rows])


def test_population_label_propagation_preserves_scalar_and_empty_results():
    population = _population()
    population[:, 0].label("soma")

    scalar = population[1].soma
    empty = population[:0].soma

    _assert_same_selection(scalar, population.soma[1])
    _assert_same_selection(empty, population.soma[:0])
    assert scalar.is_scalar
    assert tuple(empty.shape) == (0,)
    assert empty.is_empty


def test_propagated_label_write_matches_explicit_label_indexing():
    population = _population()
    population[:, 0].label("soma")

    population[[2, 0]].soma.v = torch.tensor([-2.0, -1.0], dtype=DTYPE)

    assert population.soma[[2, 0]].v.tolist() == [-2.0, -1.0]
    assert population.v[:, 0].tolist() == [-1.0, 5.0, -2.0]


def test_propagated_label_injection_targets_the_explicit_label_index():
    population = _population()
    population[:, 0].label("soma")
    waveform = object()

    propagated = population[[2, 0]].soma
    explicit = population.soma[[2, 0]]
    propagated.inject(waveform)

    assert len(population.injections) == 1
    registered_waveform, registered_shape, registered_index = population.injections[0]
    assert registered_waveform is waveform
    assert tuple(registered_shape) == tuple(explicit.shape)
    grid = torch.arange(population.v.numel()).reshape(population.shape)
    assert torch.equal(
        grid[registered_index].reshape(-1),
        grid[explicit.index].reshape(-1),
    )


def test_nested_labels_propagate_only_through_their_owning_slice_descendants():
    population = _population()
    dendrites = population[:, 1:].label("dendrites")
    ends = dendrites[:, [3, 0]].label("ends")

    descendant = dendrites[[2, 0]]
    _assert_same_selection(descendant.ends, ends[[2, 0]])

    # A propagated parent label retains the same nested-label scope.
    propagated_parent = population[[2, 0]].dendrites
    _assert_same_selection(propagated_parent.ends, ends[[2, 0]])

    unrelated = population[:, 0]
    with pytest.raises(AttributeError, match="ends"):
        _ = unrelated.ends


def test_deeply_nested_descendant_replays_each_commuting_index():
    population = _population(n=4)
    dendrites = population[:, 1:].label("dendrites")
    dendrites[:, [3, 0]].label("ends")

    descendant = dendrites[[3, 1, 3]][1:]
    explicit = dendrites.ends[[3, 1, 3]][1:]

    _assert_same_selection(descendant.ends, explicit)


def test_noncommuting_compartment_index_does_not_silently_retarget_a_label():
    population = _population()
    population[:, [0, 3, 4]].label("landmarks")
    compartment_tail = population[:, 2:]

    # Replaying ``[:, 2:]`` on the label would select only compartment 4,
    # although the physical overlap also contains compartment 3.
    with pytest.raises(AttributeError, match="landmarks"):
        _ = compartment_tail.landmarks

    overlap = compartment_tail.intersect(population.landmarks)
    assert tuple(overlap.shape) == (6,)
    assert overlap.v.tolist() == [3.0, 4.0, 8.0, 9.0, 13.0, 14.0]


def test_full_rank_boolean_selection_requires_explicit_intersection():
    population = _population()
    population[:, 0].label("soma")
    mask = torch.zeros(population.shape, dtype=torch.bool)
    mask[[2, 0], 0] = True
    irregular = population[mask]

    with pytest.raises(AttributeError, match="soma"):
        _ = irregular.soma

    overlap = irregular.intersect(population.soma)
    assert tuple(overlap.shape) == (2,)
    assert overlap.v.tolist() == [0.0, 10.0]


def test_duplicate_bearing_label_requires_explicit_index_or_intersection():
    population = _population()
    population[:, [0, 0]].label("soma_twice")

    with pytest.raises(AttributeError, match="soma_twice"):
        _ = population[1].soma_twice

    assert population.soma_twice[1].v.tolist() == [5.0, 5.0]
    overlap = population[1].intersect(population.soma_twice)
    assert overlap.v.tolist() == [5.0]


def test_retained_selection_and_propagated_result_rebase_across_batching():
    population = _population()
    population[:, 0].label("soma")
    selected_cell = population[1]
    propagated = selected_cell.soma

    population.batch(2)
    _assert_same_selection(selected_cell.soma, population.soma[:, 1])
    _assert_same_selection(propagated, population.soma[:, 1])

    population.batch(3)
    _assert_same_selection(selected_cell.soma, population.soma[..., 1])
    _assert_same_selection(propagated, population.soma[..., 1])


def test_label_propagates_for_selection_created_after_batching():
    population = _population()
    population[:, 0].label("soma")
    population.batch(2)

    selected_cells = population[:, [2, 0]]

    _assert_same_selection(selected_cells.soma, population.soma[:, [2, 0]])


def test_label_created_after_batching_propagates_across_leading_axes():
    population = _population()
    population.batch(2)
    population[..., 0].label("soma")

    selected_cells = population[:, [2, 0]]

    _assert_same_selection(selected_cells.soma, population.soma[:, [2, 0]])


def test_fresh_lookup_observes_label_replacement_and_clear():
    population = _population()
    population[:, 0].label("region")
    selected_cell = population[1]
    retained = selected_cell.region

    replacement = population[:, 1].label("region", replace=True)
    fresh = selected_cell.region

    _assert_same_selection(fresh, replacement[1])
    assert retained.v.item() == 5.0
    assert fresh.v.item() == 6.0

    population.clear_labels()
    with pytest.raises(AttributeError, match="region"):
        _ = selected_cell.region
    assert retained.v.item() == 5.0


def test_deepcopy_retargets_propagation_to_the_copied_population():
    population = _population()
    population[:, 0].label("soma")

    cloned = copy.deepcopy(population)
    cloned[1].soma.v = -99.0

    assert cloned.v[1, 0].item() == -99.0
    assert population.v[1, 0].item() == 5.0


def test_nested_label_cannot_shadow_a_visible_population_label():
    population = _population()
    population[:, 0].label("soma")
    dendrites = population[:, 1:]
    child = dendrites[:, 0]

    with pytest.raises(ValueError, match="soma|shadow|visible"):
        child.label("soma")

    assert population.soma.v.tolist() == [0.0, 5.0, 10.0]
    assert dendrites._labels == {}


def test_labels_must_be_resolved_before_entering_a_mechanism_namespace():
    population = _population()
    population[:, 0].label("soma")
    population.insert(pas, g=0.1, e=-70.0)
    population.build()

    assert population[1].soma.mech.pas.g.item() == pytest.approx(0.1)
    with pytest.raises(AttributeError, match="soma"):
        _ = population[1].mech.pas.soma
    with pytest.raises(ValueError, match="population-backed"):
        population[1].mech.pas[:].label("mechanism_region")


def test_intersection_is_one_dimensional_and_preserves_left_order_and_duplicates():
    population = _population()
    left = population[
        torch.tensor([2, 0, 2, 1]),
        torch.tensor([4, 3, 4, 0]),
    ]
    right_with_duplicates = population[
        torch.tensor([2, 2, 0]),
        torch.tensor([4, 4, 3]),
    ]

    overlap = left.intersect(right_with_duplicates)

    assert tuple(overlap.shape) == (3,)
    assert overlap.v.tolist() == [14.0, 3.0, 14.0]
    assert _flat_coordinates(population, overlap).tolist() == [14, 3, 14]


@settings(max_examples=60, deadline=None)
@given(
    left_ids=st.lists(st.integers(min_value=0, max_value=14), max_size=10),
    right_ids=st.lists(st.integers(min_value=0, max_value=14), max_size=10),
)
def test_intersection_matches_left_ordered_membership_for_arbitrary_coordinates(
    left_ids, right_ids
):
    population = _population()

    def select(flat_ids):
        flat = torch.tensor(flat_ids, dtype=torch.long)
        return population[flat // 5, flat % 5]

    overlap = select(left_ids).intersect(select(right_ids))
    support = set(right_ids)
    expected = [float(index) for index in left_ids if index in support]

    assert tuple(overlap.shape) == (len(expected),)
    assert overlap.v.tolist() == expected


def test_intersection_handles_scalar_matches_misses_and_empty_slices():
    population = _population()
    scalar = population[1, 2]

    match = scalar.intersect(population[:, 2])
    miss = scalar.intersect(population[:, 0])
    empty = population[:0].intersect(population[:, 0])

    assert tuple(match.shape) == (1,)
    assert match.v.tolist() == [7.0]
    assert tuple(miss.shape) == (0,)
    assert miss.is_empty
    assert tuple(empty.shape) == (0,)
    assert empty.is_empty


def test_retained_intersection_rebases_over_later_batch_axes():
    population = _population()
    left = population[:, :2]
    right = population[:, 0]
    retained = left.intersect(right)

    assert tuple(retained.shape) == (3,)
    population.batch(2)

    # Like every retained Slice, an existing one-dimensional intersection gains
    # the new leading batch axis. A newly formed intersection still applies its
    # canonical one-dimensional boolean selection at creation time.
    fresh = left.intersect(right)
    assert tuple(retained.shape) == (2, 3)
    assert tuple(fresh.shape) == (6,)
    assert torch.equal(retained.v, fresh.v.reshape(2, 3))


def test_intersection_rejects_a_different_population():
    population = _population()
    other = _population()

    with pytest.raises(ValueError, match="same.*population|same root"):
        population[:, 0].intersect(other[:, 0])


def test_intersection_rejects_mechanism_backed_slices():
    population = _population()
    population.insert(pas, g=0.1, e=-70.0)
    population.build()

    with pytest.raises(ValueError, match="population-backed"):
        population[:, 0].mech.pas.intersect(population[:, 0])
    with pytest.raises(ValueError, match="population-backed"):
        population[:, 0].intersect(population[:, 0].mech.pas)


def test_intersection_rejects_non_slice_operands():
    population = _population()

    with pytest.raises(TypeError, match="Slice"):
        population[:, 0].intersect(torch.tensor([0]))
