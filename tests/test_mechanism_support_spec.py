"""Contracts for structural mechanism-support metadata.

These tests compare every structural representation with the same row-major
physical support. Opt-in structured storage may change the local tensor rank,
but never its numerical result or slot ordering.
"""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch

import dendra as dn  # noqa: F401 - configure Dendra before defining mechanisms
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import (
    MechanismHandler,
    _same_current_support,
    _support_scatter_is_unambiguous,
)
from dendra.models.mechanisms._support import SupportKind, SupportSpec

CORE_SHAPE = (3, 6)
DTYPE = torch.float64


class _Probe(Mechanism):
    pass


@dataclass(frozen=True)
class _Case:
    name: str
    key: object
    is_composable: bool
    local_shape: tuple[int, ...]
    expected_kind: SupportKind
    expected_flat: tuple[int, ...]
    expected_local_core_shape: tuple[int, ...]
    preserves_multiplicity: bool = False
    has_duplicates: bool = False


CASES = (
    _Case(
        "dense",
        None,
        False,
        CORE_SHAPE,
        SupportKind.DENSE,
        tuple(range(18)),
        CORE_SHAPE,
    ),
    _Case(
        "rectangular",
        (slice(1, 3), slice(2, 5)),
        True,
        (2, 3),
        SupportKind.RECTANGULAR,
        (8, 9, 10, 14, 15, 16),
        (2, 3),
    ),
    _Case(
        "shared_columns",
        torch.tensor([1, 4, 7, 10, 13, 16]),
        False,
        (6,),
        SupportKind.SHARED_COLUMNS,
        (1, 4, 7, 10, 13, 16),
        (3, 2),
    ),
    _Case(
        "rowwise_deferred_to_packed",
        torch.tensor([0, 3, 7, 11, 14, 16]),
        False,
        (6,),
        # Phase 1 intentionally recognizes this shape but retains the packed
        # runtime representation.  Flip this to ROWWISE/(3, 2) only when the
        # rowwise encoding lands in the later rollout.
        SupportKind.PACKED_FLAT,
        (0, 3, 7, 11, 14, 16),
        (6,),
    ),
    _Case(
        "ragged",
        torch.tensor([0, 3, 7, 14, 16, 17]),
        False,
        (6,),
        SupportKind.PACKED_FLAT,
        (0, 3, 7, 14, 16, 17),
        (6,),
    ),
    _Case(
        "duplicate_multiset",
        torch.tensor([1, 7, 13, 1, 7, 13]),
        False,
        (6,),
        SupportKind.PACKED_FLAT,
        (1, 7, 13, 1, 7, 13),
        (6,),
        preserves_multiplicity=True,
        has_duplicates=True,
    ),
    _Case(
        "defensive_unflagged_duplicates",
        torch.tensor([1, 1, 7, 7, 13, 13]),
        False,
        (6,),
        SupportKind.PACKED_FLAT,
        (1, 1, 7, 7, 13, 13),
        (6,),
        has_duplicates=True,
    ),
)


def _spec(case: _Case) -> SupportSpec:
    key = case.key.clone() if torch.is_tensor(case.key) else case.key
    return SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=key,
        is_composable=case.is_composable,
        local_shape=case.local_shape,
        preserves_multiplicity=case.preserves_multiplicity,
    )


def _probe(case: _Case) -> _Probe:
    key = case.key.clone() if torch.is_tensor(case.key) else case.key
    field = torch.zeros(CORE_SHAPE, dtype=DTYPE)
    support_spec = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=key,
        is_composable=case.is_composable,
        local_shape=case.local_shape,
        preserves_multiplicity=case.preserves_multiplicity,
    )
    return _Probe(
        case.name,
        torch.full_like(field, 34.0),
        torch.ones_like(field),
        case.local_shape,
        case.local_shape,
        key=key,
        is_composable=case.is_composable,
        support_spec=support_spec,
        preserves_multiplicity=case.preserves_multiplicity,
    )


def _materialized(spec: SupportSpec, runtime_key=None) -> torch.Tensor:
    """Supply the live key only when a packed spec does not own indices."""

    if spec.kind is SupportKind.PACKED_FLAT:
        return spec.materialized_flat_indices(runtime_key=runtime_key)
    return spec.materialized_flat_indices()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_support_classifier_and_materialization_are_exact(case):
    spec = _spec(case)

    assert spec.kind is case.expected_kind
    assert spec.local_core_shape == case.expected_local_core_shape
    assert spec.preserves_multiplicity is case.preserves_multiplicity
    assert spec.has_duplicates is case.has_duplicates
    runtime_key = case.key if torch.is_tensor(case.key) else None
    assert tuple(_materialized(spec, runtime_key).tolist()) == case.expected_flat


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_structural_support_matches_legacy_batched_gather_and_scatter_add(case):
    """The classifier must be a semantic no-op during the metadata rollout."""

    mechanism = _probe(case)
    indices = _materialized(mechanism.support_spec, mechanism.key)
    field = torch.arange(2 * 18, dtype=DTYPE).reshape(2, *CORE_SHAPE)

    expected_gather = field.reshape(2, -1).index_select(-1, indices)
    actual_gather = mechanism.get(field)
    assert torch.equal(actual_gather.reshape(2, -1), expected_gather)

    local = torch.arange(actual_gather.numel(), dtype=DTYPE).reshape_as(actual_gather)
    expected_scatter = torch.zeros_like(field)
    expected_scatter.reshape(2, -1).scatter_add_(
        -1,
        indices.expand(2, -1),
        local.reshape(2, -1),
    )
    actual_scatter = torch.zeros_like(field)
    mechanism.add_(actual_scatter, local)
    assert torch.equal(actual_scatter, expected_scatter)


@pytest.mark.parametrize(
    "case",
    tuple(case for case in CASES if not case.has_duplicates),
    ids=lambda case: case.name,
)
def test_structural_support_matches_legacy_batched_put(case):
    mechanism = _probe(case)
    indices = _materialized(mechanism.support_spec, mechanism.key)
    voltage = torch.zeros((2, *CORE_SHAPE), dtype=DTYPE)
    initial = torch.full_like(voltage, -1.0)
    gathered = mechanism.get(voltage)
    updates = torch.arange(gathered.numel(), dtype=DTYPE).reshape_as(gathered)

    expected = initial.clone()
    expected.reshape(2, -1).scatter_(
        -1,
        indices.expand(2, -1),
        updates.reshape(2, -1),
    )
    actual = mechanism.put(updates, initial, voltage)
    assert torch.equal(actual, expected)


def test_explicit_multiplicity_forces_packed_flat_even_without_duplicates():
    spec = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=torch.tensor([1, 7, 13]),
        is_composable=False,
        local_shape=(3,),
        preserves_multiplicity=True,
    )

    assert spec.kind is SupportKind.PACKED_FLAT
    assert spec.preserves_multiplicity
    assert not spec.has_duplicates


def test_forced_flat_full_support_preserves_one_dimensional_local_layout():
    """Physical coverage alone must not erase a deliberately flat local ABI."""

    spec = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=torch.arange(18),
        is_composable=False,
        local_shape=(18,),
        force_packed=True,
    )

    assert spec.kind is SupportKind.PACKED_FLAT
    assert spec.local_core_shape == (18,)


def test_signature_is_owned_hashable_ordered_and_support_spec_is_immutable():
    key = torch.tensor([1, 4, 7, 10, 13, 16])
    first = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=key,
        is_composable=False,
        local_shape=(6,),
    )
    same = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=key.clone(),
        is_composable=False,
        local_shape=(6,),
    )
    reordered = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=key.flip(0),
        is_composable=False,
        local_shape=(6,),
    )
    expected = _materialized(first, key).clone()
    signature = first.ordered_signature

    assert first.ordered_signature == same.ordered_signature
    assert hash(first.ordered_signature) == hash(same.ordered_signature)
    assert first.ordered_signature != reordered.ordered_signature

    key.fill_(0)
    assert first.ordered_signature == signature
    # Compact shared-column specs own enough structural metadata to rematerialize
    # independently of the original caller's mutable key.
    assert torch.equal(_materialized(first), expected)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        first.kind = SupportKind.PACKED_FLAT


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_support_spec_pickle_uses_stable_named_schema(case):
    spec = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=case.key,
        is_composable=case.is_composable,
        local_shape=case.local_shape,
        preserves_multiplicity=case.name == "duplicate_multiset",
        force_packed=case.name == "forced_packed",
    )
    if spec.kind is SupportKind.SHARED_COLUMNS and spec.all_populations:
        spec = spec.with_population_axis()

    restored = pickle.loads(pickle.dumps(spec))

    assert restored == spec
    assert restored.checkpoint_identity() == spec.checkpoint_identity()


def test_meta_keys_still_group_by_precomputed_ordered_signature():
    shared = next(case for case in CASES if case.name == "shared_columns")
    left = _probe(shared).to(device="meta")
    right = _probe(shared).to(device="meta")

    different_case = _Case(
        "different",
        torch.tensor([0, 4, 6, 10, 12, 16]),
        False,
        (6,),
        SupportKind.SHARED_COLUMNS,
        (0, 4, 6, 10, 12, 16),
        (3, 2),
    )
    different = _probe(different_case).to(device="meta")

    assert left.key.device.type == right.key.device.type == "meta"
    assert _same_current_support(left, right)
    assert not _same_current_support(left, different)
    assert _support_scatter_is_unambiguous(left)
    representatives, plan, _ = MechanismHandler._partition_current_supports(
        (left, right, different)
    )
    assert len(representatives) == 2
    assert [support_index for _, support_index in plan] == [0, 0, 1]

    duplicate_case = next(case for case in CASES if case.name == "duplicate_multiset")
    duplicate = _probe(duplicate_case).to(device="meta")
    assert not _support_scatter_is_unambiguous(duplicate)


def test_state_dict_rejects_structural_selector_key_changes_before_regrouping():
    shared = next(case for case in CASES if case.name == "shared_columns")
    different = _Case(
        "different",
        torch.tensor([0, 4, 6, 10, 12, 16]),
        False,
        (6,),
        SupportKind.SHARED_COLUMNS,
        (0, 4, 6, 10, 12, 16),
        (3, 2),
    )

    def handler(second_case):
        field = torch.zeros(CORE_SHAPE, dtype=DTYPE)
        result = MechanismHandler(
            torch.full_like(field, 34.0),
            torch.ones_like(field),
            {"first": _probe(shared), "second": _probe(second_case)},
        )
        result.make_maps()
        return result

    target = handler(shared)
    source = handler(different)
    assert len(target._current_support_representatives) == 1
    assert len(source._current_support_representatives) == 2
    expected_signature = target.second.support_spec.ordered_signature
    expected_key = target.second.key.clone()

    with pytest.raises(ValueError, match="selector keys encode mechanism placement"):
        target.load_state_dict(source.state_dict())

    assert len(target._current_support_representatives) == 1
    assert target.second.support_spec.ordered_signature == expected_signature
    assert torch.equal(target.second.key, expected_key)


@pytest.mark.parametrize(
    ("name", "key", "preserves_multiplicity", "expected_kind"),
    [
        ("rectangular", (slice(1, 3), slice(2, 5)), False, SupportKind.RECTANGULAR),
        (
            "shared_columns",
            (slice(None), torch.tensor([1, 4])),
            False,
            SupportKind.SHARED_COLUMNS,
        ),
        (
            "rowwise_deferred",
            (torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 3, 1, 5, 2, 4])),
            False,
            SupportKind.PACKED_FLAT,
        ),
        (
            "ragged",
            (torch.tensor([0, 0, 1, 2, 2, 2]), torch.tensor([0, 3, 1, 2, 4, 5])),
            False,
            SupportKind.PACKED_FLAT,
        ),
        (
            "duplicates",
            (torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([1, 1, 1, 1, 1, 1])),
            True,
            SupportKind.PACKED_FLAT,
        ),
    ],
)
def test_population_compilation_attaches_the_expected_support_metadata(
    name, key, preserves_multiplicity, expected_kind
):
    population = dn.Population(N=3, C=6, dtype=DTYPE)
    population[key].insert(
        _Probe,
        preserve_multiplicity=preserves_multiplicity,
    )
    population.build()
    mechanism = getattr(population.mech, "_Probe")

    assert mechanism.support_spec.kind is expected_kind, name
    assert mechanism.support_spec.preserves_multiplicity is preserves_multiplicity
    legacy_flat = (
        mechanism.key
        if torch.is_tensor(mechanism.key)
        else torch.arange(18).reshape(CORE_SHAPE)[mechanism.key].reshape(-1)
    )
    assert torch.equal(
        _materialized(mechanism.support_spec, mechanism.key),
        legacy_flat,
    )


def test_everywhere_and_forced_flat_population_paths_are_distinguished():
    dense = dn.Population(N=3, C=6, dtype=DTYPE)
    dense.insert(_Probe)
    dense.build()
    assert dense.mech._Probe.support_spec.kind is SupportKind.DENSE

    cropped = dn.Population(N=3, C=6, dtype=DTYPE)
    cropped.insert(_Probe)
    cropped[:, 1].delete(_Probe)
    cropped.build()
    spec = cropped.mech._Probe.support_spec
    assert spec.kind is SupportKind.PACKED_FLAT
    assert spec.force_packed


@pytest.mark.parametrize(
    ("key", "local_shape", "message"),
    [
        (torch.tensor([0, 18]), (2,), "range"),
        (torch.tensor([0, 1, 2]), (2,), "local"),
    ],
)
def test_invalid_compiled_support_is_rejected(key, local_shape, message):
    with pytest.raises(ValueError, match=message):
        SupportSpec.from_compiled(
            core_shape=CORE_SHAPE,
            key=key,
            is_composable=False,
            local_shape=local_shape,
        )
