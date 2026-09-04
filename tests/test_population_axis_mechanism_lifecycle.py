"""Lifecycle regressions for opt-in population-shaped mechanism storage."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._support import SupportKind, SupportSpec

DTYPE = torch.float64
N_POPULATIONS = 3
N_COMPARTMENTS = 6
COLUMNS = torch.tensor([1, 4], dtype=torch.long)
N_COLUMNS = int(COLUMNS.numel())
BATCH_SIZE = 4


class _LifecycleState(State):
    State.STATE("x")
    State.RANGE(rate=0.1)
    State.DERIVATIVE("x' = -rate * x")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class _LifecycleProbe(Mechanism):
    Mechanism.RANGE(g=2.0e-4, e=-52.0)
    Mechanism.BATCH(scale=1.0)
    Mechanism.STATE_BUNDLE(_LifecycleState)
    Mechanism.CARRY("scratch")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.SAVE_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        del values
        return {"scratch": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del dt, values
        return {"scratch": 0.25 * v}

    def i(self, v):
        return self.scale * self.g * self.x * (v - self.e)


class _StructuredMetaProbe(_LifecycleProbe):
    pass


class _PackedMetaProbe(_LifecycleProbe):
    pass


class _PackedMetaTwinProbe(_LifecycleProbe):
    pass


class _LegacyPicklePopulation(dn.Population):
    """Emit a Population state from before the opt-in flag existed."""

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("preserve_mechanism_population_axis", None)
        state.pop("_preserve_mechanism_population_axis", None)
        return state


def _mechanism(population, cls=_LifecycleProbe):
    name = cls._name or cls.__name__
    return getattr(population.mech, name)


def _shared_columns(columns):
    columns = torch.as_tensor(columns, dtype=torch.long)
    rows = torch.arange(N_POPULATIONS, dtype=torch.long)
    return (rows[:, None] * N_COMPARTMENTS + columns[None, :]).reshape(-1)


def _compiled(layout: str):
    preserve = layout == "structured"
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=preserve,
    )
    if layout == "packed":
        rows = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
        columns = torch.tensor([0, 3, 2, 1, 4, 5], dtype=torch.long)
        population[rows, columns].insert(_LifecycleProbe)
    else:
        population[:, COLUMNS].insert(_LifecycleProbe)
    population.build()
    return population, _mechanism(population)


def _expected_local(field, key, shape):
    return field.reshape(-1).index_select(0, key).reshape(shape)


@pytest.mark.parametrize("layout", ("structured", "legacy", "packed"))
def test_deepcopied_mechanism_support_operations_are_bound_to_the_clone(layout):
    population, source = _compiled(layout)
    clone = copy.deepcopy(source)

    assert clone is not source
    assert clone.support_map is not source.support_map
    assert clone.support_map.spec is clone.support_spec
    assert clone.support_spec.ordered_signature == source.support_spec.ordered_signature

    if layout == "packed":
        clone_key = torch.tensor([0, 2, 7, 15, 16, 17], dtype=torch.long)
        source_key = torch.tensor([1, 3, 6, 8, 14, 17], dtype=torch.long)
    else:
        clone_key = _shared_columns([0, 5])
        source_key = _shared_columns([2, 3])

    # Mutate both selector owners after copying. Every clone operation must use
    # clone.key and clone.support_map, not accessors that retained ``source``.
    clone.key.copy_(clone_key)
    clone.refresh_support_spec()
    source.key.copy_(source_key)
    source.refresh_support_spec()

    if layout == "structured":
        expected_shape = (N_POPULATIONS, N_COLUMNS)
        assert clone.support_spec.preserves_population_axis
    else:
        expected_shape = (N_POPULATIONS * N_COLUMNS,)
        assert not clone.support_spec.preserves_population_axis
    assert clone.support_spec.runtime_local_shape == expected_shape

    field = torch.arange(N_POPULATIONS * N_COMPARTMENTS, dtype=DTYPE).reshape(
        N_POPULATIONS, N_COMPARTMENTS
    )
    expected_gather = _expected_local(field, clone_key, expected_shape)
    assert torch.equal(clone.get(field), expected_gather)

    local = 10.0 + torch.arange(expected_gather.numel(), dtype=DTYPE).reshape(
        expected_shape
    )
    expected_add = torch.zeros_like(field)
    expected_add.reshape(-1).scatter_add_(0, clone_key, local.reshape(-1))

    destination = torch.zeros_like(field)
    returned = clone.add_(destination, local)
    assert returned is destination
    assert torch.equal(destination, expected_add)
    assert torch.equal(clone.add(torch.zeros_like(field), local), expected_add)

    original = torch.full_like(field, -7.0)
    expected_put = original.clone()
    expected_put.reshape(-1).scatter_(0, clone_key, local.reshape(-1))
    assert torch.equal(clone.put(local, original, field), expected_put)
    assert torch.equal(original, torch.full_like(field, -7.0))

    # Keep the source live until the end so this proves selector independence
    # rather than merely surviving a dead closure target.
    source_expected = _expected_local(field, source_key, source.shape_f)
    assert torch.equal(source.get(field), source_expected)
    assert population.mech is not None


def test_handler_meta_roundtrip_preserves_support_identity_and_layout(monkeypatch):
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    population[:, COLUMNS].insert(_StructuredMetaProbe)
    rows = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    columns = torch.tensor([0, 3, 2, 1, 4, 5], dtype=torch.long)
    population[rows, columns].insert(_PackedMetaProbe)
    population.build()
    handler = population.mech
    handler.make_maps()

    original = {
        name: (
            mechanism.support_spec.ordered_signature,
            mechanism.support_spec.runtime_local_shape,
            mechanism.shape_p,
            mechanism.shape_f,
        )
        for name, mechanism in handler.mechanisms.items()
    }
    assert handler._StructuredMetaProbe.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert handler._StructuredMetaProbe.support_spec.preserves_population_axis
    assert handler._PackedMetaProbe.support_spec.kind is SupportKind.PACKED_FLAT

    def forbid_reclassification(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "meta/to_empty lifecycle must not infer support from selector values"
        )

    # CPU allocators can coincidentally recycle old key bytes into to_empty
    # storage, so explicitly forbid value-derived reclassification.
    monkeypatch.setattr(SupportSpec, "from_compiled", forbid_reclassification)
    handler.to(device="meta")
    handler.make_maps()
    assert all(
        mechanism.key.device.type == "meta" for mechanism in handler.mechanisms.values()
    )

    # to_empty intentionally does not recover tensor values. Support metadata
    # is immutable configuration, so make_maps must not infer it again.
    handler.to_empty(device="cpu")
    handler.make_maps()

    for name, mechanism in handler.mechanisms.items():
        signature, local_shape, shape_p, shape_f = original[name]
        assert mechanism.key.device.type == "cpu"
        assert mechanism.support_spec.ordered_signature == signature
        assert mechanism.support_spec.runtime_local_shape == local_shape
        assert mechanism.shape_p == shape_p
        assert mechanism.shape_f == shape_f
        assert mechanism.support_map.spec is mechanism.support_spec
    assert len(handler._current_support_representatives) == 2


def test_packed_meta_to_empty_defers_exact_grouping_until_key_restore(monkeypatch):
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
    )
    rows = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    columns = torch.tensor([0, 3, 2, 1, 4, 5], dtype=torch.long)
    population[rows, columns].insert(_PackedMetaProbe)
    population[rows, columns].insert(_PackedMetaTwinProbe)
    population.build()
    handler = population.mech
    handler.make_maps()

    assert all(
        mechanism.support_spec.kind is SupportKind.PACKED_FLAT
        for mechanism in handler.mechanisms.values()
    )
    assert len(handler._state_support_representatives) == 1
    assert len(handler._current_support_representatives) == 1
    saved_state = copy.deepcopy(handler.state_dict())
    signatures = {
        name: mechanism.support_spec.ordered_signature
        for name, mechanism in handler.mechanisms.items()
    }

    def forbid_key_comparison(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "meta/to_empty support planning must not inspect uninitialized keys"
        )

    def forbid_reclassification(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "meta/to_empty support planning must retain immutable SupportSpec metadata"
        )

    # Both meta selector tensors and the CPU tensors created by to_empty lack
    # runtime values. Equal SupportSpecs are not enough to prove exact packed
    # key equality, so planning must conservatively keep them separate.
    monkeypatch.setattr(torch, "equal", forbid_key_comparison)
    monkeypatch.setattr(SupportSpec, "from_compiled", forbid_reclassification)
    handler.to(device="meta")
    handler.make_maps()
    assert len(handler._state_support_representatives) == 2
    assert len(handler._current_support_representatives) == 2

    handler.to_empty(device="cpu")
    handler.make_maps()
    assert len(handler._state_support_representatives) == 2
    assert len(handler._current_support_representatives) == 2
    for name, mechanism in handler.mechanisms.items():
        assert mechanism.support_spec.ordered_signature == signatures[name]

    # A real state load restores selector bytes and reruns the normal support
    # refresh hooks. Exact grouping is valid again only after that boundary.
    monkeypatch.undo()
    handler.load_state_dict(saved_state)
    assert len(handler._state_support_representatives) == 1
    assert len(handler._current_support_representatives) == 1


def test_legacy_population_pickle_without_axis_flag_defaults_to_false():
    population = _LegacyPicklePopulation(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    state = population.__getstate__()
    assert "preserve_mechanism_population_axis" not in state
    restored = _LegacyPicklePopulation.__new__(_LegacyPicklePopulation)
    # Legacy serialized layouts remain legacy even when the current process is
    # opting new models into population-shaped storage.
    with dn.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=True):
        restored.__setstate__(state)
        assert restored.preserve_mechanism_population_axis is False
        restored[:, COLUMNS].insert(_LifecycleProbe)
        restored.build()

    mechanism = _mechanism(restored)
    assert mechanism.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert not mechanism.support_spec.preserves_population_axis
    assert mechanism.shape_p == (N_POPULATIONS * N_COLUMNS,)


def test_population_axis_snapshot_survives_state_roundtrip_independent_of_context():
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    state = population.__getstate__()
    restored = dn.Population.__new__(dn.Population)

    with dn.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=False):
        restored.__setstate__(state)
        restored[:, COLUMNS].insert(_LifecycleProbe)
        restored.build()

    assert restored.preserve_mechanism_population_axis is True
    mechanism = _mechanism(restored)
    assert mechanism.support_spec.preserves_population_axis
    assert mechanism.shape_p == (N_POPULATIONS, N_COLUMNS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_preserve_mechanism_population_axis", "yes"),
        ("preserve_mechanism_population_axis", 1),
    ],
)
def test_population_axis_snapshot_rejects_corrupt_serialized_values(field, value):
    population = dn.Population(N=1, C=1)
    state = population.__getstate__()
    state.pop("_preserve_mechanism_population_axis", None)
    state[field] = value
    restored = dn.Population.__new__(dn.Population)

    with pytest.raises(TypeError, match="layout must be a boolean"):
        restored.__setstate__(state)


def test_postbuild_batch_refreshes_structured_mechanism_shape_metadata_and_runtime():
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        v_init=-65.0,
        dtype=DTYPE,
        integrator=dn.euler(),
        preserve_mechanism_population_axis=True,
    )
    population[:, COLUMNS].insert(_LifecycleProbe)
    population.build()
    before = _mechanism(population)
    support_signature = before.support_spec.ordered_signature
    assert before.shape_p == (N_POPULATIONS, N_COLUMNS)
    assert before.shape_f == (N_POPULATIONS, N_COLUMNS)

    returned = population.batch(BATCH_SIZE)
    assert returned is population
    assert population.shape == (BATCH_SIZE, N_POPULATIONS, N_COMPARTMENTS)

    mechanism = _mechanism(population)
    state = mechanism.DE[_LifecycleState.__name__]
    expected_shape_p = (1, N_POPULATIONS, N_COLUMNS)
    expected_shape_f = (BATCH_SIZE, N_POPULATIONS, N_COLUMNS)
    assert mechanism.support_spec.ordered_signature == support_signature
    assert mechanism.support_spec.preserves_population_axis
    assert mechanism.shape_p == expected_shape_p
    assert mechanism.shape_f == expected_shape_f
    for value in (
        mechanism.g,
        mechanism.e,
        mechanism.x,
        state.rate,
    ):
        assert tuple(value.shape) == expected_shape_p
    assert tuple(mechanism.i_.shape) == expected_shape_f
    assert tuple(mechanism.scratch.shape) == expected_shape_f
    assert tuple(mechanism.scale.shape) == (1, N_POPULATIONS, 1)
    assert tuple(mechanism.get(population.v).shape) == expected_shape_f

    population.initialize()
    mechanism = _mechanism(population)
    population.step(dt=0.025)
    assert tuple(mechanism.x.shape) == expected_shape_f
    assert tuple(mechanism.i_.shape) == expected_shape_f
    assert tuple(mechanism.scratch.shape) == expected_shape_f
