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
    _Case(
        "empty_rectangular",
        (slice(None), slice(2, 2)),
        True,
        (3, 0),
        SupportKind.RECTANGULAR,
        (),
        (3, 0),
    ),
    _Case(
        "empty_packed",
        torch.tensor([], dtype=torch.long),
        False,
        (0,),
        SupportKind.PACKED_FLAT,
        (),
        (0,),
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


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_out_of_place_scatter_add_is_pure_and_differentiable(case):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    indices = _materialized(support_map.spec, mechanism.key)
    local_shape = support_map.spec.runtime_local_shape

    destination = (
        torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE).requires_grad_()
    )
    local = (
        torch.linspace(0.25, 1.75, support_map.spec.slot_count, dtype=DTYPE)
        .reshape(local_shape)
        .requires_grad_()
    )
    original = destination.detach().clone()
    original_version = destination._version

    actual = support_map.scatter_add(destination, local, mechanism.key)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (destination, local))

    reference_destination = original.clone().requires_grad_()
    reference_local = local.detach().clone().requires_grad_()
    expected = torch.scatter_add(
        reference_destination.reshape(-1),
        0,
        indices,
        reference_local.reshape(-1),
    ).reshape(CORE_SHAPE)
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (reference_destination, reference_local),
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    torch.testing.assert_close(destination, original)
    assert destination._version == original_version
    assert actual is not destination


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_out_of_place_scatter_add_supports_parameter_only_vmap(case, lane_count):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.randn(
        (lane_count,) + support_map.spec.runtime_local_shape,
        dtype=DTYPE,
        requires_grad=True,
    )

    def scatter(one_local):
        return support_map.scatter_add(destination, one_local, mechanism.key)

    actual = torch.vmap(scatter)(local)
    if lane_count:
        expected = torch.stack([scatter(one_local) for one_local in local])
        torch.testing.assert_close(actual, expected)
    else:
        assert actual.shape == (0,) + CORE_SHAPE

    gradient = torch.autograd.grad(actual.square().sum(), local)[0]
    assert gradient.shape == local.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize(
    "case",
    tuple(
        case
        for case in CASES
        if case.expected_kind in {SupportKind.DENSE, SupportKind.RECTANGULAR}
        and case.expected_flat
    ),
    ids=lambda case: case.name,
)
@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_out_of_place_scatter_add_broadcasts_a_vmapped_scalar(case, lane_count):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.randn(lane_count, dtype=DTYPE, requires_grad=True)

    vmapped = torch.vmap(
        lambda scalar: support_map.scatter_add(
            destination,
            scalar,
            mechanism.key,
        )
    )
    expected = vmapped(local)
    torch.compiler.reset()
    compiled = torch.compile(vmapped, backend="eager", fullgraph=True)
    actual = compiled(local)

    assert actual.shape == (lane_count,) + CORE_SHAPE
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(expected.square().sum(), local)[0]
    assert gradient.shape == local.shape


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_gather_and_out_of_place_scatter_support_state_vmap(case, lane_count):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    voltages = torch.randn(
        (lane_count,) + CORE_SHAPE,
        dtype=DTYPE,
        requires_grad=True,
    )

    def round_trip(voltage):
        local = support_map.gather(voltage, mechanism.key)
        return support_map.scatter_add(voltage, local.square(), mechanism.key)

    actual = torch.vmap(round_trip)(voltages)
    assert actual.shape == voltages.shape
    if lane_count:
        expected = torch.stack([round_trip(voltage) for voltage in voltages])
        torch.testing.assert_close(actual, expected)

    gradient = torch.autograd.grad(actual.square().sum(), voltages)[0]
    assert gradient.shape == voltages.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_compiled_vmapped_out_of_place_scatter_add_is_fullgraph(case):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.randn(
        (2,) + support_map.spec.runtime_local_shape,
        dtype=DTYPE,
    )

    vmapped = torch.vmap(
        lambda one_local: support_map.scatter_add(
            destination,
            one_local,
            mechanism.key,
        )
    )
    torch.compiler.reset()
    compiled = torch.compile(vmapped, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(local), vmapped(local))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_out_of_place_scatter_add_supports_reverse_and_forward_mode(case):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.randn(support_map.spec.runtime_local_shape, dtype=DTYPE)

    def scatter(one_local):
        return support_map.scatter_add(destination, one_local, mechanism.key)

    reverse = torch.func.jacrev(scatter)(local)
    forward = torch.func.jacfwd(scatter)(local)

    assert reverse.shape == CORE_SHAPE + support_map.spec.runtime_local_shape
    torch.testing.assert_close(reverse, forward)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_support_map_handles_an_explicit_empty_batch_axis(case):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.empty((0,) + CORE_SHAPE, dtype=DTYPE, requires_grad=True)

    local = support_map.gather(destination, mechanism.key)
    actual = support_map.scatter_add(destination, local, mechanism.key)

    assert local.shape == (0,) + support_map.spec.runtime_local_shape
    assert actual.shape == destination.shape
    gradient = torch.autograd.grad(actual.sum(), destination)[0]
    assert gradient.shape == destination.shape


@pytest.mark.parametrize(
    "case",
    tuple(case for case in CASES if not case.has_duplicates),
    ids=lambda case: case.name,
)
def test_out_of_place_scatter_set_is_pure_and_differentiable(case):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    indices = _materialized(support_map.spec, mechanism.key)
    destination = (
        torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE).requires_grad_()
    )
    local = torch.randn(
        support_map.spec.runtime_local_shape,
        dtype=DTYPE,
        requires_grad=True,
    )
    original = destination.detach().clone()
    original_version = destination._version

    actual = support_map.scatter_set(
        local,
        destination,
        destination,
        mechanism.key,
    )
    actual_gradients = torch.autograd.grad(
        actual.square().sum(),
        (destination, local),
        allow_unused=True,
    )

    reference_destination = original.clone().requires_grad_()
    reference_local = local.detach().clone().requires_grad_()
    expected = torch.scatter(
        reference_destination.reshape(-1),
        0,
        indices,
        reference_local.reshape(-1),
    ).reshape(CORE_SHAPE)
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (reference_destination, reference_local),
        allow_unused=True,
    )

    actual_destination_gradient = (
        torch.zeros_like(destination)
        if actual_gradients[0] is None
        else actual_gradients[0]
    )
    expected_destination_gradient = (
        torch.zeros_like(reference_destination)
        if expected_gradients[0] is None
        else expected_gradients[0]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual_destination_gradient,
        expected_destination_gradient,
    )
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1])
    torch.testing.assert_close(destination, original)
    assert destination._version == original_version


@pytest.mark.parametrize(
    "case",
    tuple(case for case in CASES if not case.has_duplicates),
    ids=lambda case: case.name,
)
@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_out_of_place_scatter_set_supports_parameter_only_vmap(case, lane_count):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.randn(
        (lane_count,) + support_map.spec.runtime_local_shape,
        dtype=DTYPE,
        requires_grad=True,
    )

    def replace(one_local):
        return support_map.scatter_set(
            one_local,
            destination,
            destination,
            mechanism.key,
        )

    actual = torch.vmap(replace)(local)
    assert actual.shape == (lane_count,) + CORE_SHAPE
    gradient = torch.autograd.grad(actual.square().sum(), local)[0]
    assert gradient.shape == local.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("lane_count", (3, 0), ids=("nonempty", "empty"))
def test_rectangular_out_of_place_scatter_set_broadcasts_a_vmapped_scalar(
    lane_count,
):
    case = next(case for case in CASES if case.name == "rectangular")
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.randn(lane_count, dtype=DTYPE, requires_grad=True)

    vmapped = torch.vmap(
        lambda scalar: support_map.scatter_set(
            scalar,
            destination,
            destination,
            mechanism.key,
        )
    )
    expected = vmapped(local)
    torch.compiler.reset()
    compiled = torch.compile(vmapped, backend="eager", fullgraph=True)
    actual = compiled(local)

    assert actual.shape == (lane_count,) + CORE_SHAPE
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(expected.square().sum(), local)[0]
    assert gradient.shape == local.shape


@pytest.mark.parametrize(
    "case",
    tuple(case for case in CASES if not case.has_duplicates),
    ids=lambda case: case.name,
)
def test_compiled_vmapped_out_of_place_scatter_set_is_fullgraph(case):
    mechanism = _probe(case)
    support_map = mechanism.support_map
    destination = torch.linspace(-1.0, 2.0, 18, dtype=DTYPE).reshape(CORE_SHAPE)
    local = torch.randn(
        (2,) + support_map.spec.runtime_local_shape,
        dtype=DTYPE,
    )

    vmapped = torch.vmap(
        lambda one_local: support_map.scatter_set(
            one_local,
            destination,
            destination,
            mechanism.key,
        )
    )
    torch.compiler.reset()
    compiled = torch.compile(vmapped, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(local), vmapped(local))


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
