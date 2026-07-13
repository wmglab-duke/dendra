"""Contracts for labels projected from MultiPopulation components."""

import pytest
import torch

import dendra as dn


def _population(n=1, c=3):
    return dn.Population(N=n, C=c, dtype=torch.float64)


def test_component_labels_are_registered_on_their_composite_owner_only():
    left = _population(n=2)
    right = _population()
    left[:, 1].label("tip")

    multi = dn.concat_models({"left": left, "right": right})

    assert multi.left._labels == {"tip": multi.left.tip}
    assert multi.left.tip.parent_slice is multi.left
    assert "tip" not in multi.right._labels
    assert "tip" not in vars(multi.right)


def test_component_label_propagates_through_its_descendants_without_sibling_leakage():
    left = _population(n=2)
    right = _population()
    left[:, 1].label("tip")
    multi = dn.concat_models({"left": left, "right": right})

    propagated = multi.left[0].tip
    explicit = multi.left.tip[0]

    assert propagated.shape == explicit.shape
    assert torch.equal(propagated.v, explicit.v)
    assert all(torch.equal(a, b) for a, b in zip(propagated.index, explicit.index))
    with pytest.raises(AttributeError, match="tip"):
        _ = multi.right[0].tip


def test_component_label_resynchronization_replaces_and_removes_registry_entries():
    left = _population()
    right = _population()
    left[:, 0].label("old")
    multi = dn.concat_models({"left": left, "right": right})
    old = multi.left.old

    left.clear_labels()
    left[:, 2].label("new")
    multi._sync_component_labels("left", left)

    assert "old" not in multi.left._labels
    assert "old" not in vars(multi.left)
    assert multi.left._labels == {"new": multi.left.new}
    assert multi.left.new is not old

    first = multi.left.new
    left[:, 1].label("new", replace=True)
    multi._sync_component_labels("left", left)

    assert multi.left._labels == {"new": multi.left.new}
    assert multi.left.new is not first
    assert multi.left.new.index[-1].tolist() == [[1]]


def test_component_label_sync_does_not_overwrite_a_user_owned_nested_label():
    left = _population()
    multi = dn.concat_models({"left": left, "right": _population()})
    manual = multi.left[:, :1].label("target")
    left[:, 2].label("target")

    with pytest.raises(ValueError, match="conflicts with a nested label"):
        multi._sync_component_labels("left", left)

    assert multi.left._labels == {"target": manual}
    assert multi.left.target is manual


def test_component_label_and_descendants_survive_repeated_batching():
    left = _population(n=2)
    left[:, 1].label("tip")
    multi = dn.concat_models({"left": left, "right": _population()})
    multi.v.copy_(
        torch.arange(multi.v.numel(), dtype=multi.dtype()).reshape(multi.shape)
    )
    retained_tip = multi.left.tip

    multi.batch(2)
    multi.batch(3)

    expected = torch.tensor([1.0, 4.0], dtype=multi.dtype()).expand(3, 2, 1, 2)
    assert multi.left.tip.shape == torch.Size([3, 2, 1, 2])
    assert retained_tip.shape == torch.Size([3, 2, 1, 2])
    assert torch.equal(multi.left.tip.v, expected)
    assert torch.equal(retained_tip.v, expected)

    propagated = multi.left[..., 0, :].tip
    explicit = multi.left.tip[..., 0, :]
    assert propagated.shape == explicit.shape == torch.Size([3, 2, 2])
    assert torch.equal(propagated.v, explicit.v)
    assert all(torch.equal(a, b) for a, b in zip(propagated.index, explicit.index))


def test_interleaved_duplicate_component_label_order_survives_repeated_batching():
    left = _population()
    left[
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 1, 0]),
    ].label("pattern")
    source_grid = torch.arange(left.v.numel()).reshape(left.shape)
    assert source_grid[left.pattern.index].reshape(-1).tolist() == [0, 1, 0]

    multi = dn.concat_models({"left": left, "right": _population()})
    retained_pattern = multi.left.pattern
    core_numel = multi.v.numel()
    initial_grid = torch.arange(core_numel).reshape(multi.shape)
    assert initial_grid[retained_pattern.index].reshape(-1).tolist() == [0, 1, 0]

    multi.batch(2)
    multi.batch(3)

    assert multi.left.pattern is retained_pattern
    assert retained_pattern.shape == torch.Size([3, 2, 1, 3])
    batched_grid = torch.arange(multi.v.numel()).reshape(multi.shape)
    expected = torch.arange(6).reshape(3, 2, 1, 1) * core_numel + torch.tensor(
        [0, 1, 0]
    ).reshape(1, 1, 1, 3)
    assert torch.equal(batched_grid[retained_pattern.index], expected)


def test_batch_specific_component_label_projection_is_rejected_atomically():
    left = _population(n=2)
    left[:, 1].label("tip")
    multi = dn.concat_models({"left": left, "right": _population()})
    multi.batch(2)

    composite_tip = multi.left.tip
    composite_registry = dict(multi.left._labels)
    multi_shape = tuple(multi.shape)
    left_shape = tuple(left.shape)

    # This replacement exists only in inner batch replica zero. Projecting it
    # as a structural component label would silently widen it to replica one.
    left[0, ..., 2].label("tip", replace=True)

    with pytest.raises(ValueError, match="every batch replica|same ordered"):
        multi.batch(3)

    assert tuple(multi.shape) == multi_shape
    assert tuple(left.shape) == left_shape
    assert multi.left.tip is composite_tip
    assert multi.left._labels == composite_registry
    assert vars(multi.left)["tip"] is composite_tip
