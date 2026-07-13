"""Production contracts for Slice indexing, labels, and batching."""

from __future__ import annotations

import copy
import keyword

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.slice import parse_key

DTYPE = torch.float64


def _population(*, n: int = 3, c: int = 5) -> dn.Population:
    population = dn.Population(N=n, C=c, dtype=DTYPE)
    population.v.copy_(
        torch.arange(n * c, dtype=DTYPE, device=population.device()).reshape(n, c)
    )
    return population


def _assert_replicated_selection(selection, expected, batch_shape):
    expected_shape = tuple(batch_shape) + tuple(expected.shape)
    assert tuple(selection.shape) == expected_shape
    assert tuple(selection.v.shape) == expected_shape

    expanded = expected.reshape(*(1 for _ in batch_shape), *expected.shape).expand(
        expected_shape
    )
    assert torch.equal(selection.v, expanded)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(1, id="integer"),
        pytest.param([2, 0], id="list"),
        pytest.param(np.array([2, 0]), id="numpy-array"),
        pytest.param(torch.tensor([2, 0]), id="integer-tensor"),
        pytest.param(
            torch.tensor([True, False, True]),
            id="boolean-tensor",
        ),
        pytest.param(Ellipsis, id="ellipsis"),
        pytest.param(None, id="new-axis"),
        pytest.param((None, Ellipsis, slice(1, 4)), id="new-axis-and-ellipsis"),
    ],
)
def test_parse_key_stores_a_tuple_without_changing_torch_semantics(key):
    values = torch.arange(15).reshape(3, 5)

    spec = parse_key(key, values.shape)

    assert isinstance(spec.index, tuple)
    assert torch.equal(values[spec.index], values[key])
    assert tuple(spec.shape) == tuple(values[key].shape)
    assert spec.is_scalar is (values[key].ndim == 0)


@pytest.mark.parametrize("kind", ["tensor", "numpy", "list"])
def test_slice_owns_mutable_advanced_indices(kind):
    population = _population()
    if kind == "tensor":
        key = torch.tensor([2, 0])
    elif kind == "numpy":
        key = np.array([2, 0])
    else:
        key = [2, 0]

    selection = population[key]
    expected = selection.v.clone()
    key[0] = 1

    assert torch.equal(selection.v, expected)


def test_index_spec_projects_batched_selection_to_shared_parameter_keys():
    population = _population(n=2, c=3).batch(4)
    spec = parse_key((2, slice(None), 1), population.shape)

    assert spec.to_key(population).tolist() == [1, 4]


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(1, id="integer"),
        pytest.param((1, 2), id="scalar"),
        pytest.param((slice(None), slice(1, None, 2)), id="basic-slices"),
        pytest.param((Ellipsis, 2), id="ellipsis"),
        pytest.param((None, Ellipsis), id="new-axis"),
        pytest.param([2, 0], id="list"),
        pytest.param(np.array([2, 0]), id="numpy-array"),
        pytest.param(torch.tensor([2, 0]), id="integer-tensor"),
        pytest.param(
            torch.tensor([True, False, True]),
            id="row-boolean-mask",
        ),
        pytest.param(
            torch.tensor(
                [
                    [True, False, True, False, False],
                    [False, True, False, True, False],
                    [True, True, False, False, False],
                ]
            ),
            id="full-boolean-mask",
        ),
        pytest.param(
            (torch.tensor([0, 2]), torch.tensor([1, 4])),
            id="paired-advanced-indices",
        ),
        pytest.param(
            (torch.tensor([[0], [2]]), torch.tensor([[1, 4]])),
            id="cartesian-advanced-indices",
        ),
        pytest.param((slice(None), slice(0, 0)), id="empty"),
    ],
)
def test_direct_slice_indexing_matches_torch(key):
    population = _population()

    selection = population[key]
    expected = population.v[key]

    assert tuple(selection.shape) == tuple(expected.shape)
    assert selection.is_scalar is (expected.ndim == 0)
    assert selection.is_empty is (expected.numel() == 0)
    assert torch.equal(selection.v, expected)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            (slice(None), slice(1, None)),
            (slice(1, None), slice(None, None, 2)),
            id="basic-after-basic",
        ),
        pytest.param(
            (None, Ellipsis),
            (0, slice(None), torch.tensor([1, 4])),
            id="advanced-after-new-axis",
        ),
        pytest.param(
            (torch.tensor([0, 2]), slice(None)),
            (torch.tensor([1, 0]), torch.tensor([4, 1])),
            id="paired-after-advanced",
        ),
        pytest.param(
            (torch.tensor([[0], [2]]), torch.tensor([[1, 4]])),
            (slice(None), 1),
            id="rank-reduction-after-cartesian",
        ),
        pytest.param(
            (
                torch.tensor(
                    [
                        [True, False, True, False, False],
                        [False, True, False, True, False],
                        [True, True, False, False, False],
                    ]
                ),
            ),
            slice(1, 5, 2),
            id="slice-after-full-boolean-mask",
        ),
        pytest.param(
            (slice(None), slice(0, 0)),
            slice(None),
            id="empty-after-basic",
        ),
    ],
)
def test_nested_slice_indexing_matches_successive_torch_indexing(first, second):
    population = _population()

    selection = population[first][second]
    expected = population.v[first][second]

    assert tuple(selection.shape) == tuple(expected.shape)
    assert selection.is_scalar is (expected.ndim == 0)
    assert selection.is_empty is (expected.numel() == 0)
    assert torch.equal(selection.v, expected)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(slice(None), id="full"),
        pytest.param(slice(0, 0), id="empty-leading-axis"),
        pytest.param((slice(None), slice(0, 0)), id="empty-trailing-axis"),
        pytest.param((None, Ellipsis), id="new-axis"),
        pytest.param(torch.tensor([True, False, True]), id="boolean-mask"),
    ],
)
def test_slice_len_matches_the_first_result_dimension(key):
    population = _population()
    selection = population[key]

    assert len(selection) == population.v[key].shape[0]


def test_scalar_slice_has_no_len():
    population = _population()

    with pytest.raises(TypeError):
        len(population[1, 2])


def test_held_anonymous_slice_preserves_selection_across_repeated_batching():
    population = _population()
    selection = population[:, 1::2]
    expected = selection.v.clone()

    population.batch(2)
    _assert_replicated_selection(selection, expected, (2,))

    population.batch(3)
    _assert_replicated_selection(selection, expected, (3, 2))


def test_ellipsis_slice_absorbs_new_batch_axes_without_changing_core_selection():
    population = _population()
    selection = population[..., 2]
    expected = selection.v.clone()

    population.batch(2)
    _assert_replicated_selection(selection, expected, (2,))

    population.batch(3)
    _assert_replicated_selection(selection, expected, (3, 2))


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(1, id="integer"),
        pytest.param([2, 0], id="list"),
        pytest.param(np.array([2, 0]), id="numpy-array"),
        pytest.param(torch.tensor([2, 0]), id="integer-tensor"),
        pytest.param(
            torch.tensor([True, False, True]),
            id="boolean-tensor",
        ),
        pytest.param(None, id="new-axis"),
    ],
)
def test_held_non_tuple_key_preserves_selection_across_repeated_batching(key):
    population = _population()
    selection = population[key]
    expected = selection.v.clone()

    population.batch(2)
    _assert_replicated_selection(selection, expected, (2,))

    population.batch(3)
    _assert_replicated_selection(selection, expected, (3, 2))


def test_labeled_slice_preserves_identity_and_selection_across_batching():
    population = _population()
    selection = population[[2, 0]].label("selected_rows")
    expected = selection.v.clone()

    assert selection is population.selected_rows
    assert population._labels["selected_rows"] is selection

    population.batch(2)
    assert population.selected_rows is selection
    _assert_replicated_selection(selection, expected, (2,))

    population.batch(3)
    assert population.selected_rows is selection
    _assert_replicated_selection(selection, expected, (3, 2))


def test_duplicate_label_aliases_rebase_the_shared_slice_exactly_once():
    population = _population()
    selection = population[:, [0, 4]]
    expected = selection.v.clone()

    assert selection.label("ends") is selection
    assert selection.label("terminals") is selection
    assert population.ends is selection
    assert population.terminals is selection

    population.batch(2)
    _assert_replicated_selection(selection, expected, (2,))

    population.batch(3)
    _assert_replicated_selection(selection, expected, (3, 2))


def test_nested_label_preserves_identity_and_selection_across_batching():
    population = _population()
    parent = population[:, 1:].label("dendrites")
    child = parent[:, [0, 3]].label("ends")
    expected_parent = parent.v.clone()
    expected_child = child.v.clone()

    assert population.dendrites is parent
    assert parent.ends is child

    population.batch(2)
    assert population.dendrites is parent
    assert parent.ends is child
    _assert_replicated_selection(parent, expected_parent, (2,))
    _assert_replicated_selection(child, expected_child, (2,))

    population.batch(3)
    assert population.dendrites is parent
    assert parent.ends is child
    _assert_replicated_selection(parent, expected_parent, (3, 2))
    _assert_replicated_selection(child, expected_child, (3, 2))


@pytest.mark.parametrize("name", [None, 1, object()])
def test_label_rejects_non_string_names_atomically(name):
    population = _population()
    selection = population[:, 0]
    before_labels = dict(population._labels)

    with pytest.raises(TypeError):
        selection.label(name)

    assert population._labels == before_labels


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("", id="empty"),
        pytest.param("two words", id="spaces"),
        pytest.param("not-valid", id="punctuation"),
        pytest.param("1region", id="leading-digit"),
        pytest.param("class", id="keyword"),
        pytest.param("_private", id="private"),
    ],
)
def test_label_rejects_invalid_keyword_and_private_names_atomically(name):
    assert not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_")
    population = _population()
    selection = population[:, 0]
    before_labels = dict(population._labels)

    with pytest.raises(ValueError):
        selection.label(name)

    assert population._labels == before_labels
    assert name not in vars(population)


@pytest.mark.parametrize("name", ["v", "shape", "batch", "initialize"])
def test_label_rejects_population_attribute_collisions_atomically(name):
    population = _population()
    selection = population[:, 0]
    before_labels = dict(population._labels)
    before_attribute = getattr(population, name)

    with pytest.raises(ValueError):
        selection.label(name)

    assert population._labels == before_labels
    after_attribute = getattr(population, name)
    if torch.is_tensor(before_attribute):
        assert after_attribute is before_attribute
    else:
        assert type(after_attribute) is type(before_attribute)


def test_label_rejects_an_existing_label_collision_atomically():
    population = _population()
    original = population[:, 0].label("region")
    replacement = population[:, 1]

    with pytest.raises(ValueError):
        replacement.label("region")

    assert population.region is original
    assert population._labels == {"region": original}


def test_label_can_explicitly_replace_only_an_existing_label():
    population = _population()
    original = population[:, 0].label("region")
    replacement = population[:, 1].label("region", replace=True)

    assert population.region is replacement
    assert population._labels == {"region": replacement}
    assert original is not replacement


def test_label_rejects_an_existing_instance_attribute_atomically():
    population = _population()
    sentinel = object()
    population.user_metadata = sentinel

    with pytest.raises(ValueError):
        population[:, 0].label("user_metadata")

    assert population.user_metadata is sentinel
    assert "user_metadata" not in population._labels


@pytest.mark.parametrize(
    "name",
    [
        "model",
        "index_spec",
        "base_shape",
        "parent_slice",
        "shape",
        "v",
        "get",
        "set",
        "label",
        "inject",
    ],
)
def test_nested_label_rejects_slice_api_collisions_atomically(name):
    population = _population()
    parent = population[:, 1:]
    child = parent[:, 0]
    before_model = parent.model
    before_spec = parent.index_spec
    before_values = parent.v.clone()
    before_dict = dict(vars(parent))

    with pytest.raises(ValueError):
        child.label(name)

    assert parent.model is before_model
    assert parent.index_spec is before_spec
    assert torch.equal(parent.v, before_values)
    assert vars(parent) == before_dict


def test_population_label_access_is_only_supported_from_its_owner():
    population = _population()
    dendrites = population[:, 1:].label("dendrites")
    tip = dendrites[:, -1:].label("tip")

    assert population.dendrites is dendrites
    assert population.dendrites.tip is tip
    assert dendrites.tip is tip

    arbitrary = population[:, 0]
    with pytest.raises(AttributeError, match="dendrites"):
        _ = arbitrary.dendrites


@pytest.mark.parametrize("method", ["batch", "build", "initialize", "run"])
def test_model_wide_methods_require_explicit_model_access(method):
    population = _population()
    selection = population[:, 0]
    before_shape = population.shape

    with pytest.raises(AttributeError, match="model-wide method"):
        getattr(selection, method)

    assert population.shape == before_shape
    assert getattr(selection.model, method) is not None


def test_model_wide_metadata_requires_explicit_model_access():
    population = _population()
    selection = population[:, 0]

    with pytest.raises(AttributeError, match="model-wide attribute"):
        _ = selection.np

    assert selection.model.np == population.np


def test_deepcopy_retargets_labels_to_the_copied_population():
    population = _population()
    parent = population[:, 1:].label("dendrites")
    parent[:, -1].label("tip")

    cloned = copy.deepcopy(population)

    assert cloned.dendrites.root_model is cloned
    assert cloned.dendrites.tip.root_model is cloned
    cloned.dendrites.tip.v = -99.0
    assert torch.all(cloned.v[:, -1] == -99.0)
    assert torch.all(population.v[:, -1] != -99.0)


def test_incompatible_topology_shape_change_fails_instead_of_retargeting_slice():
    population = _population(n=2, c=3)
    selection = population[:, 1]
    population.v = torch.zeros(2, 4, dtype=DTYPE)

    with pytest.raises(RuntimeError, match="incompatible shape"):
        _ = selection.v
