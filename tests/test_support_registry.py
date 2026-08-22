"""Runtime support-registry identity, storage, and lifecycle contracts."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace

import pytest
import torch

import dendra as dn  # noqa: F401 - configure Dendra before defining mechanisms
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._support import SupportKind, SupportSpec
from dendra.models.mechanisms._support_registry import SupportRegistry

CORE_SHAPE = (3, 6)
DTYPE = torch.float64


class _Probe(Mechanism):
    pass


def _probe(
    name,
    key,
    *,
    local_shape,
    is_composable=False,
    support_spec=None,
    preserves_multiplicity=False,
):
    field = torch.zeros(CORE_SHAPE, dtype=DTYPE)
    if support_spec is None:
        support_spec = SupportSpec.from_compiled(
            core_shape=CORE_SHAPE,
            key=key,
            is_composable=is_composable,
            local_shape=local_shape,
            preserves_multiplicity=preserves_multiplicity,
        )
    return _Probe(
        name,
        torch.full_like(field, 34.0),
        torch.ones_like(field),
        local_shape,
        local_shape,
        key=key,
        is_composable=is_composable,
        support_spec=support_spec,
        preserves_multiplicity=preserves_multiplicity,
    )


def _shared(name, columns=(1, 4), *, population_axis=False):
    key = torch.tensor(
        [row * CORE_SHAPE[1] + column for row in range(3) for column in columns]
    )
    spec = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=key,
        is_composable=False,
        local_shape=(key.numel(),),
    )
    local_shape = (key.numel(),)
    if population_axis:
        spec = spec.with_population_axis()
        local_shape = spec.runtime_local_shape
    return _probe(name, key, local_shape=local_shape, support_spec=spec)


def _packed(name, key, *, support_spec=None, preserves_multiplicity=False):
    key = torch.as_tensor(key, dtype=torch.long)
    return _probe(
        name,
        key,
        local_shape=(key.numel(),),
        support_spec=support_spec,
        preserves_multiplicity=preserves_multiplicity,
    )


def test_equivalent_structured_supports_share_id_map_and_spec_without_key_changes():
    first = _shared("first")
    second = _shared("second")
    first_key = first.key
    second_key = second.key
    first_state_names = tuple(first.state_dict())
    second_state_names = tuple(second.state_dict())

    registry = SupportRegistry((first, second))

    assert registry.support_id(first) == registry.support_id(second) == 0
    assert registry.active_ids == (0,)
    assert registry.entries_by_id[1] is None
    assert first.support_map is second.support_map is registry.entry(0).support_map
    assert first.support_spec is second.support_spec is registry.entry(0).spec
    assert first.key is first_key
    assert second.key is second_key
    assert tuple(first.state_dict()) == first_state_names
    assert tuple(second.state_dict()) == second_state_names


def test_ids_use_first_authored_ordinal_and_survive_an_earlier_group_split():
    key = torch.tensor([0, 3, 7, 14, 16, 17])
    first = _packed("first", key)
    second = _packed("second", key.clone())
    later = _packed("later", [0, 2, 7, 14, 16, 17])
    registry = SupportRegistry((first, second, later))

    assert registry.active_ids == (0, 2)
    assert registry.support_id(second) == 0
    assert registry.support_id(later) == 2

    second._support_key_values_valid = False
    registry.rebuild((first, second, later))

    assert registry.active_ids == (0, 1, 2)
    assert registry.support_id(first) == 0
    assert registry.support_id(second) == 1
    assert registry.support_id(later) == 2


def test_population_shaped_and_flat_local_layouts_are_distinct_supports():
    flat = _shared("flat")
    population_shaped = _shared("population_shaped", population_axis=True)

    registry = SupportRegistry((flat, population_shaped))

    assert registry.active_ids == (0, 1)
    assert registry.support_id(flat) == 0
    assert registry.support_id(population_shaped) == 1
    assert flat.support_spec.preserves_population_axis is False
    assert population_shaped.support_spec.preserves_population_axis is True


def test_packed_support_requires_exact_key_even_if_fingerprints_collide():
    left_key = torch.tensor([0, 3, 7, 14, 16, 17])
    right_key = torch.tensor([0, 2, 7, 14, 16, 17])
    left_spec = SupportSpec.from_compiled(
        CORE_SHAPE,
        left_key,
        is_composable=False,
        local_shape=(6,),
    )
    right_spec = SupportSpec.from_compiled(
        CORE_SHAPE,
        right_key,
        is_composable=False,
        local_shape=(6,),
    )
    assert left_spec.kind is right_spec.kind is SupportKind.PACKED_FLAT
    right_spec = replace(right_spec, flat_fingerprint=left_spec.flat_fingerprint)
    assert left_spec.ordered_signature == right_spec.ordered_signature

    left = _packed("left", left_key, support_spec=left_spec)
    same = _packed("same", left_key.clone(), support_spec=left_spec)
    different = _packed("different", right_key, support_spec=right_spec)
    registry = SupportRegistry((left, same, different))

    assert registry.active_ids == (0, 2)
    assert registry.support_id(left) == registry.support_id(same) == 0
    assert registry.support_id(different) == 2


def test_duplicate_packed_slot_order_is_part_of_identity():
    first = _packed(
        "first",
        [1, 1, 7, 13],
        preserves_multiplicity=True,
    )
    same = _packed(
        "same",
        [1, 1, 7, 13],
        preserves_multiplicity=True,
    )
    reordered = _packed(
        "reordered",
        [1, 7, 1, 13],
        preserves_multiplicity=True,
    )

    registry = SupportRegistry((first, same, reordered))

    assert registry.support_id(first) == registry.support_id(same) == 0
    assert registry.support_id(reordered) == 2
    assert not registry.entry(0).is_injective()


def test_selector_accounting_separates_serialized_keys_interning_and_compaction():
    first = _shared("first")
    second = _shared("second")
    third = _shared("third")
    packed = _packed("packed", [0, 3, 7, 14, 16, 17])

    accounting = SupportRegistry((first, second, third, packed)).accounting

    assert accounting.registered_mechanisms == 4
    assert accounting.unique_supports == 2
    # Three P*K compatibility keys plus one six-slot packed key.
    assert accounting.compatibility_indices == 24
    # One shared P*K selector plus the packed selector.
    assert accounting.unique_legacy_indices == 12
    # K shared columns plus the irreducible packed selector.
    assert accounting.compact_indices == 8
    assert accounting.interning_savings == 12
    assert accounting.compact_savings == 16


def test_partition_uses_global_sparse_ids_and_rejects_unregistered_mechanisms():
    first = _shared("first")
    same = _shared("same")
    later = _shared("later", columns=(0, 5))
    registry = SupportRegistry((first, same, later))

    representatives, plan, by_object = registry.partition((later, same))

    assert representatives == (later, first)
    assert plan == ((later, 2), (same, 0))
    assert by_object == {id(later): 2, id(same): 0}
    with pytest.raises(KeyError, match="not registered"):
        registry.partition((_shared("unregistered"),))


def test_failed_intern_is_atomic_and_does_not_consume_an_authored_id():
    registry = SupportRegistry()

    with pytest.raises(TypeError, match="must expose"):
        registry.intern(object())

    mechanism = _shared("valid")
    assert registry.registered_count == 0
    assert registry.intern(mechanism) == 0
    assert registry.active_ids == (0,)


def test_repeated_object_aliases_consume_authored_ordinals_without_new_supports():
    shared = _shared("shared")
    later = _shared("later", columns=(0, 5))

    registry = SupportRegistry((shared, shared, later))

    assert registry.registered_count == 3
    assert registry.active_ids == (0, 2)
    assert registry.support_id(shared) == 0
    assert registry.support_id(later) == 2
    assert registry.entries_by_id[1] is None
    assert registry.accounting.compatibility_indices == (
        2 * shared.support_spec.legacy_index_count
        + later.support_spec.legacy_index_count
    )


def test_handler_plans_share_one_global_registry_without_state_dict_changes():
    first = _shared("first")
    same = _shared("same")
    later = _shared("later", columns=(0, 5))
    handler = MechanismHandler(
        torch.full(CORE_SHAPE, 34.0, dtype=DTYPE),
        torch.ones(CORE_SHAPE, dtype=DTYPE),
        {"first": first, "same": same, "later": later},
    )
    state_before = {
        name: value.detach().clone() for name, value in handler.state_dict().items()
    }

    handler.make_maps()

    assert handler.support_registry.active_ids == (0, 2)
    assert handler._state_support_ids == (0, 2)
    assert [support_id for _, support_id in handler._state_advance_plan] == [
        0,
        0,
        2,
    ]
    assert first.support_map is same.support_map
    assert all("support_registry" not in name for name in handler.state_dict())
    assert tuple(handler.state_dict()) == tuple(state_before)
    for name, expected in state_before.items():
        torch.testing.assert_close(handler.state_dict()[name], expected)


def test_handler_deepcopy_rebuilds_registry_with_clone_owned_representatives():
    source = MechanismHandler(
        torch.full(CORE_SHAPE, 34.0, dtype=DTYPE),
        torch.ones(CORE_SHAPE, dtype=DTYPE),
        {"first": _shared("first"), "same": _shared("same")},
    )

    clone = copy.deepcopy(source)

    assert clone.support_registry.active_ids == (0,)
    assert clone.support_registry.entry(0).representative is clone.mechanisms["first"]
    assert (
        clone.support_registry.entry(0).representative is not source.mechanisms["first"]
    )
    assert clone.mechanisms["first"].support_map is clone.mechanisms["same"].support_map


def test_entry_operations_follow_the_representatives_live_key_buffer():
    mechanism = _packed("packed", [0, 3, 7, 14, 16, 17])
    registry = SupportRegistry((mechanism,))
    entry = registry.entry(0)
    field = torch.arange(18, dtype=DTYPE).reshape(CORE_SHAPE)

    original_key = mechanism.key
    mechanism.key = mechanism.key.flip(0).clone()
    mechanism.support_spec = SupportSpec.from_compiled(
        CORE_SHAPE,
        mechanism.key,
        is_composable=False,
        local_shape=(6,),
    )
    mechanism.support_map = type(entry.support_map)(mechanism.support_spec)

    assert mechanism.key is not original_key
    assert torch.equal(entry.gather(field), field.reshape(-1)[mechanism.key])


def test_entry_executes_canonical_map_without_mechanism_accessors(monkeypatch):
    mechanism = _packed("packed", [0, 3, 7, 14, 16, 17])
    entry = SupportRegistry((mechanism,)).entry(0)
    field = torch.arange(18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.linspace(1.0, 6.0, 6, dtype=DTYPE)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("handler execution re-entered a Mechanism mapper")

    for name in ("get", "add_", "add", "put"):
        monkeypatch.setattr(mechanism, name, forbidden)

    torch.testing.assert_close(entry.gather(field), field.reshape(-1)[mechanism.key])

    expected_add = torch.zeros_like(field)
    expected_add.reshape(-1).scatter_add_(0, mechanism.key, local)
    in_place = torch.zeros_like(field)
    assert entry.scatter_add_(in_place, local) is in_place
    torch.testing.assert_close(in_place, expected_add)

    functional_source = torch.zeros_like(field)
    functional = entry.scatter_add(functional_source, local)
    torch.testing.assert_close(functional, expected_add)
    torch.testing.assert_close(functional_source, torch.zeros_like(field))

    expected_set = torch.zeros_like(field)
    expected_set.reshape(-1).scatter_(0, mechanism.key, local)
    replaced = entry.scatter_set(
        local,
        torch.zeros_like(field),
        field,
    )
    torch.testing.assert_close(replaced, expected_set)


def test_structured_meta_supports_intern_but_packed_meta_supports_do_not():
    structured_left = _shared("structured_left").to(device="meta")
    structured_right = _shared("structured_right").to(device="meta")
    packed_key = [0, 3, 7, 14, 16, 17]
    packed_left = _packed("packed_left", packed_key).to(device="meta")
    packed_right = _packed("packed_right", packed_key).to(device="meta")

    registry = SupportRegistry(
        (structured_left, structured_right, packed_left, packed_right)
    )

    assert registry.support_id(structured_left) == 0
    assert registry.support_id(structured_right) == 0
    assert registry.support_id(packed_left) == 2
    assert registry.support_id(packed_right) == 3


def test_copy_and_pickle_drop_runtime_registry_metadata_for_explicit_rebuild():
    first = _shared("first")
    second = _shared("second")
    registry = SupportRegistry((first, second))

    copies = (
        copy.copy(registry),
        copy.deepcopy(registry),
        pickle.loads(pickle.dumps(registry)),
    )

    assert all(copy_registry.registered_count == 0 for copy_registry in copies)
    for copy_registry in copies:
        copy_registry.rebuild((first, second))
        assert copy_registry.active_ids == (0,)


def test_entry_rejects_boolean_sparse_and_out_of_range_ids():
    first = _shared("first")
    second = _shared("second")
    registry = SupportRegistry((first, second))

    with pytest.raises(TypeError, match="integer"):
        registry.entry(True)
    with pytest.raises(KeyError, match="not an active group"):
        registry.entry(1)
    with pytest.raises(KeyError, match="Unknown"):
        registry.entry(2)
