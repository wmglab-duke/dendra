"""Contracts for removing distributed mechanisms from Population support."""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State

DTYPE = torch.float64


class _RangeProbe(Mechanism):
    Mechanism.RANGE(g=1.0, e=-65.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)


class _RetainedProbe(Mechanism):
    Mechanism.RANGE(g=2.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * v


class _BatchProbe(Mechanism):
    Mechanism.BATCH(scale=1.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.scale * v


class _GlobalProbe(Mechanism):
    Mechanism.GLOBAL(scale=1.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.scale * v


class _FrozenState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0 * x")


class _StatefulProbe(Mechanism):
    Mechanism.RANGE(g=1.0)
    Mechanism.STATE(_FrozenState)
    Mechanism.INIT(x=1.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * v


class _SodiumProbe(Mechanism):
    Mechanism.RANGE(g=0.01, e=50.0)
    Mechanism.USEION("na", write=["ina"])

    def ina(self, v):
        return self.g * (v - self.e)


_RenamedRangeProbe = _RangeProbe.rename("renamed_range_probe")


class _VectorOverride(torch.nn.Module):
    """A non-scalar graph parametrization that cannot be cropped generically."""

    def forward(self, value):
        return torch.arange(
            value.numel(), device=value.device, dtype=value.dtype
        ).reshape_as(value)


def _population(*, n=1, c=4):
    return dn.Population(N=n, C=c, v_init=-65.0, dtype=DTYPE)


def _mechanism(population, mechanism_class):
    name = mechanism_class._name or mechanism_class.__name__
    return population.mech.mechanisms[name]


def _compiled_support(population, mechanism_class):
    mechanism = _mechanism(population, mechanism_class)
    if mechanism.key is None:
        return torch.arange(
            math.prod(population.core_shape()),
            dtype=torch.long,
            device=population.device(),
        )
    if mechanism.is_composable:
        grid = torch.arange(
            math.prod(population.core_shape()),
            dtype=torch.long,
            device=population.device(),
        ).reshape(population.core_shape())
        return grid[mechanism.key].reshape(-1)
    return mechanism.key.reshape(-1)


def _physical_values(population, mechanism_class, name):
    mechanism = _mechanism(population, mechanism_class)
    support = _compiled_support(population, mechanism_class).tolist()
    values = getattr(mechanism, name).detach().reshape(-1).tolist()
    return dict(zip(support, values))


def test_population_delete_defaults_to_removing_the_class_everywhere():
    population = _population(c=3)
    population[:, :2].insert(_RangeProbe, alias="left", g=2.0)
    population[:, 2].insert(_RangeProbe, alias="right", g=3.0)
    population.insert(_RetainedProbe)
    population.initialize()
    old_integrator = population.integrator

    population[:, 1:].mech._RangeProbe.parametrize("g", 7.0, alias="persistent")
    assert population._slice_mechanism_parametrizations

    population.delete(_RangeProbe)

    assert _RangeProbe not in population._mech_data
    assert _RangeProbe not in population._mech_everywhere
    assert _RangeProbe not in population._mech_data_ic
    assert not population._slice_mechanism_parametrizations
    assert population._flag_rebuild
    assert not population.initialized
    assert not old_integrator.initialized

    population.initialize()
    assert "_RangeProbe" not in population.mech.mechanisms
    assert "_RetainedProbe" in population.mech.mechanisms


def test_legacy_population_state_migrates_sparse_global_configuration():
    population = _population(c=2)
    population[:, :].insert(_GlobalProbe, scale=3.0)
    state = population.__getstate__()
    alias, kwargs, key, preserve, copies = state["_mech_data"][_GlobalProbe][0]
    assert kwargs == {}
    state["_mech_data"][_GlobalProbe] = [(alias, {"scale": 3.0}, key, preserve, copies)]
    for name in (
        "_mech_exclusions",
        "_mech_data_ic",
        "_mech_data_base_kwargs",
    ):
        state.pop(name)

    restored = dn.Population.__new__(dn.Population)
    restored.__setstate__(state)

    assert restored._mech_exclusions == {}
    assert restored._mech_data_ic == {}
    assert restored._mech_data_base_kwargs == {_GlobalProbe: {"scale": 3.0}}
    assert restored._mech_data[_GlobalProbe][0][1] == {}
    restored.build()
    torch.testing.assert_close(
        _mechanism(restored, _GlobalProbe).scale,
        torch.tensor(3.0, dtype=DTYPE),
    )

    packed = dn.concat_models({"restored": restored})
    packed.initialize()
    torch.testing.assert_close(
        _mechanism(packed, _GlobalProbe).scale,
        torch.tensor(3.0, dtype=DTYPE),
    )


def test_sparse_global_configuration_accepts_equal_scalar_representations():
    population = _population(c=2)
    population[:, 0].insert(
        _GlobalProbe,
        alias="left",
        scale=torch.tensor(3.0, dtype=DTYPE),
    )
    population[:, 1].insert(_GlobalProbe, alias="right", scale=3.0)

    population.initialize()

    torch.testing.assert_close(
        _mechanism(population, _GlobalProbe).scale,
        torch.tensor(3.0, dtype=DTYPE),
    )


@pytest.mark.parametrize(
    "conflicting",
    (
        torch.tensor(1.0001, dtype=torch.float64),
        1.0 + 1.0j,
    ),
)
def test_sparse_global_configuration_rejects_distinct_typed_scalars(conflicting):
    population = _population(c=2)
    population[:, 0].insert(
        _GlobalProbe,
        alias="left",
        scale=torch.tensor(1.0, dtype=torch.float16),
    )

    with pytest.raises(ValueError, match="different class-wide GLOBAL value"):
        population[:, 1].insert(
            _GlobalProbe,
            alias="right",
            scale=conflicting,
        )

    assert len(population._mech_data[_GlobalProbe]) == 1


def test_partial_sparse_delete_projects_support_and_tensor_values():
    population = _population(c=4)
    population[:, :3].insert(
        _RangeProbe,
        alias="selected",
        g=torch.tensor([[1.0, 2.0, 3.0]], dtype=DTYPE),
    )
    population.build()

    population[:, 1].delete(_RangeProbe)
    population.build()

    assert _compiled_support(population, _RangeProbe).tolist() == [0, 2]
    assert _physical_values(population, _RangeProbe, "g") == {0: 1.0, 2: 3.0}


def test_partial_delete_of_last_support_removes_the_mechanism_class():
    population = _population(c=3)
    population[:, :2].insert(_RangeProbe, g=2.0)

    population[:, :2].delete(_RangeProbe)

    assert _RangeProbe not in population._mech_data
    assert _RangeProbe not in population._mech_data_ic
    assert _RangeProbe not in population._mech_data_base_kwargs
    population.initialize()
    assert "_RangeProbe" not in population.mech.mechanisms


def test_partial_delete_of_everywhere_insertion_preserves_constructor_values():
    population = _population(c=3)
    population.insert(_RangeProbe, g=0.25, e=-61.0)
    population.build()

    population[:, 1].delete(_RangeProbe)
    population.build()

    assert _compiled_support(population, _RangeProbe).tolist() == [0, 2]
    assert _physical_values(population, _RangeProbe, "g") == {0: 0.25, 2: 0.25}
    assert _physical_values(population, _RangeProbe, "e") == {0: -61.0, 2: -61.0}


def test_partial_delete_rejects_unsupported_compartments_atomically():
    population = _population(c=4)
    population[:, :2].insert(_RangeProbe, g=2.0)
    population.initialize()
    old_mechanism = _mechanism(population, _RangeProbe)
    old_integrator = population.integrator
    old_support = population._mechanism_support_flat(_RangeProbe).clone()

    with pytest.raises(ValueError, match="do not currently host"):
        population[:, [1, 3]].delete(_RangeProbe, strict=True)

    assert torch.equal(population._mechanism_support_flat(_RangeProbe), old_support)
    assert _mechanism(population, _RangeProbe) is old_mechanism
    assert population.integrator is old_integrator
    assert population.initialized
    assert not population._flag_rebuild


def test_delete_matches_the_exact_configured_mechanism_class():
    population = _population(c=2)
    population.insert(_RenamedRangeProbe)

    with pytest.raises(ValueError, match="is not inserted"):
        population.delete(_RangeProbe, strict=True)

    assert population._mechanism_support_flat(_RenamedRangeProbe).tolist() == [0, 1]


def test_non_strict_delete_subtracts_only_the_existing_support_intersection():
    population = _population(c=4)
    population[:, :2].insert(_RangeProbe, g=2.0)

    population[:, [1, 3]].delete(_RangeProbe)

    assert population._mechanism_support_flat(_RangeProbe).tolist() == [0]
    population.initialize()
    assert _compiled_support(population, _RangeProbe).tolist() == [0]


def test_non_strict_empty_intersection_is_a_true_lifecycle_no_op():
    population = _population(c=4)
    population[:, :2].insert(_RangeProbe, g=2.0)
    population.initialize()
    old_mechanism = _mechanism(population, _RangeProbe)
    old_integrator = population.integrator

    population[:, 3].delete(_RangeProbe)

    assert _mechanism(population, _RangeProbe) is old_mechanism
    assert population.integrator is old_integrator
    assert population.initialized
    assert not population._flag_rebuild


def test_absent_mechanism_is_no_op_non_strict_and_error_strict():
    population = _population(c=2)
    population.insert(_RetainedProbe)
    population.initialize()
    old_mechanism = _mechanism(population, _RetainedProbe)
    old_integrator = population.integrator

    population.delete(_RangeProbe)

    assert _mechanism(population, _RetainedProbe) is old_mechanism
    assert population.integrator is old_integrator
    assert population.initialized
    assert not population._flag_rebuild

    with pytest.raises(ValueError, match="is not inserted"):
        population.delete(_RangeProbe, strict=True)

    assert _mechanism(population, _RetainedProbe) is old_mechanism
    assert population.integrator is old_integrator
    assert population.initialized
    assert not population._flag_rebuild


def test_partial_delete_crops_every_overlapping_insertion_record():
    population = _population(c=4)
    population[:, :3].insert(_RangeProbe, alias="left", g=1.0)
    population[:, 2:].insert(_RangeProbe, alias="right", g=2.0)
    population.build()

    population[:, 2].delete(_RangeProbe)
    population.build()

    assert _compiled_support(population, _RangeProbe).tolist() == [0, 1, 3]
    assert _physical_values(population, _RangeProbe, "g") == {
        0: 1.0,
        1: 1.0,
        3: 2.0,
    }
    assert len(population._mech_data[_RangeProbe]) == 2


def test_batched_slice_delete_projects_to_the_shared_structural_core():
    population = _population(n=2, c=3)
    population.insert(_RangeProbe)
    population.batch(4)

    population[2, :, 1].delete(_RangeProbe)
    population.build()

    assert _compiled_support(population, _RangeProbe).tolist() == [0, 2, 3, 5]
    values = population[:, :, 1].inspect("g", mechanism="_RangeProbe")
    assert values.shape == (4, 2)
    assert torch.isnan(values).all()


def test_delete_removes_all_copied_slots_at_selected_physical_locations():
    population = _population(c=3)
    population[:, :2].insert(
        _RangeProbe,
        copies=3,
        g=torch.tensor([10.0, 20.0, 10.0, 20.0, 10.0, 20.0], dtype=DTYPE),
    )
    population.build()
    assert _compiled_support(population, _RangeProbe).tolist() == [0, 1, 0, 1, 0, 1]

    population[:, 0].delete(_RangeProbe)
    population.build()

    assert _compiled_support(population, _RangeProbe).tolist() == [1, 1, 1]
    torch.testing.assert_close(
        _mechanism(population, _RangeProbe).g,
        torch.full((3,), 20.0, dtype=DTYPE),
    )


def test_delete_projects_per_copy_range_broadcast_before_rebuild():
    population = _population(c=3)
    population[:, :2].insert(
        _RangeProbe,
        copies=3,
        g=torch.tensor([[10.0], [20.0], [30.0]], dtype=DTYPE),
    )
    population.build()

    torch.testing.assert_close(
        _mechanism(population, _RangeProbe).g,
        torch.tensor([10.0, 10.0, 20.0, 20.0, 30.0, 30.0], dtype=DTYPE),
    )

    population[:, 0].delete(_RangeProbe)
    population.build()

    assert _compiled_support(population, _RangeProbe).tolist() == [1, 1, 1]
    torch.testing.assert_close(
        _mechanism(population, _RangeProbe).g,
        torch.tensor([10.0, 20.0, 30.0], dtype=DTYPE),
    )


def test_partial_delete_projects_persistent_slice_parametrization():
    population = _population(c=3)
    population.insert(_RangeProbe, g=1.0)
    population.build()
    population[:, 1:].mech._RangeProbe.parametrize(
        "g",
        torch.tensor([[2.0, 3.0]], dtype=DTYPE),
        alias="tail",
    )

    population[:, 1].delete(_RangeProbe)
    population.initialize()

    assert _compiled_support(population, _RangeProbe).tolist() == [0, 2]
    assert _physical_values(population, _RangeProbe, "g") == {0: 1.0, 2: 3.0}
    assert len(population._slice_mechanism_parametrizations) == 1
    record = population._slice_mechanism_parametrizations[0]
    assert record["core_indices"].tolist() == [2]
    torch.testing.assert_close(record["value"], torch.tensor([3.0], dtype=DTYPE))


def test_partial_everywhere_delete_preserves_initial_conditions():
    population = _population(c=3)
    population.insert(_StatefulProbe, ic={"x": 7.0}, g=0.125)

    population[:, 1].delete(_StatefulProbe)
    population.initialize()

    assert _compiled_support(population, _StatefulProbe).tolist() == [0, 2]
    torch.testing.assert_close(
        _mechanism(population, _StatefulProbe).x,
        torch.full((2,), 7.0, dtype=DTYPE),
    )
    torch.testing.assert_close(
        _mechanism(population, _StatefulProbe).g,
        torch.full((2,), 0.125, dtype=DTYPE),
    )


def test_slice_delete_all_crops_every_present_class_and_preserves_outside_support():
    population = _population(c=4)
    population[:, :2].insert(_RangeProbe, g=1.0)
    population[:, 1:3].insert(_RetainedProbe, g=2.0)
    population[:, 1].insert(_RenamedRangeProbe, g=3.0)
    population[:, 3].insert(_SodiumProbe)

    population[:, 1].delete_all()

    assert population._mechanism_support_flat(_RangeProbe).tolist() == [0]
    assert population._mechanism_support_flat(_RetainedProbe).tolist() == [2]
    assert _RenamedRangeProbe not in population._mech_data
    assert _RenamedRangeProbe not in population._mech_data_base_kwargs
    assert population._mechanism_support_flat(_SodiumProbe).tolist() == [3]

    population.initialize()
    assert _compiled_support(population, _RangeProbe).tolist() == [0]
    assert _compiled_support(population, _RetainedProbe).tolist() == [2]
    assert _compiled_support(population, _SodiumProbe).tolist() == [3]
    assert "renamed_range_probe" not in population.mech.mechanisms


def test_delete_all_projects_persistent_parametrizations_for_every_class():
    population = _population(c=3)
    population.insert(_RangeProbe, g=1.0)
    population.insert(_RetainedProbe, g=4.0)
    population.build()
    population[:, 1:].mech._RangeProbe.parametrize(
        "g",
        torch.tensor([[2.0, 3.0]], dtype=DTYPE),
        alias="range_tail",
    )
    population[:, 1:].mech._RetainedProbe.parametrize(
        "g",
        torch.tensor([[5.0, 6.0]], dtype=DTYPE),
        alias="retained_tail",
    )

    population[:, 1].delete_all()

    records = {
        record["mechanism_class"]: record
        for record in population._slice_mechanism_parametrizations
    }
    assert records[_RangeProbe]["core_indices"].tolist() == [2]
    assert records[_RetainedProbe]["core_indices"].tolist() == [2]
    torch.testing.assert_close(
        records[_RangeProbe]["value"],
        torch.tensor([3.0], dtype=DTYPE),
    )
    torch.testing.assert_close(
        records[_RetainedProbe]["value"],
        torch.tensor([6.0], dtype=DTYPE),
    )
    population.initialize()
    assert _physical_values(population, _RangeProbe, "g") == {0: 1.0, 2: 3.0}
    assert _physical_values(population, _RetainedProbe, "g") == {
        0: 4.0,
        2: 6.0,
    }


def test_delete_all_invalidates_once_and_empty_intersection_is_a_no_op(monkeypatch):
    population = _population(c=3)
    population.insert(_RangeProbe)
    population.insert(_RetainedProbe)
    population.initialize()
    original_invalidate = population._invalidate_mechanism_structure
    invalidations = 0

    def counted_invalidation():
        nonlocal invalidations
        invalidations += 1
        original_invalidate()

    monkeypatch.setattr(
        population,
        "_invalidate_mechanism_structure",
        counted_invalidation,
    )

    population[:, 1].delete_all()
    assert invalidations == 1

    population[:, 1].delete_all()
    assert invalidations == 1


def test_population_delete_all_without_index_removes_every_mechanism_class():
    population = _population(c=3)
    population.insert(_RangeProbe)
    population[:, :2].insert(_RetainedProbe)
    population[:, 2].insert(_SodiumProbe)
    population.build()

    population.delete_all()

    assert population._mech_everywhere == {}
    assert population._mech_exclusions == {}
    assert population._mech_data == {}
    assert population._mech_data_ic == {}
    assert population._mech_data_base_kwargs == {}
    assert population._slice_mechanism_parametrizations == []

    population.initialize()
    assert not population.mech.mechanisms
    assert not population.mech.ions
    assert not population.mech.currents


def test_population_delete_apis_validate_indices_even_when_nothing_is_present():
    population = _population(c=2)

    with pytest.raises(IndexError):
        population.delete(_RangeProbe, index=(0, 9))

    with pytest.raises(IndexError):
        population.delete_all(index=(0, 9))

    population.initialize()
    old_integrator = population.integrator
    population.delete_all()
    assert population.integrator is old_integrator
    assert population.initialized
    assert not population._flag_rebuild


def test_batched_slice_delete_all_projects_to_shared_core_and_drops_empty_classes():
    population = _population(n=2, c=3)
    population.insert(_RangeProbe)
    population[:, 1].insert(_RetainedProbe)
    population[:, 2].insert(_SodiumProbe)
    population.batch(4)

    population[2, :, 1].delete_all()

    assert population._mechanism_support_flat(_RangeProbe).tolist() == [0, 2, 3, 5]
    assert _RetainedProbe not in population._mech_data
    assert population._mechanism_support_flat(_SodiumProbe).tolist() == [2, 5]

    population.initialize()
    assert _compiled_support(population, _RangeProbe).tolist() == [0, 2, 3, 5]
    assert "_RetainedProbe" not in population.mech.mechanisms
    assert _compiled_support(population, _SodiumProbe).tolist() == [2, 5]
    values = population[:, :, 1].inspect("g", mechanism="_RangeProbe")
    assert values.shape == (4, 2)
    assert torch.isnan(values).all()


def test_delete_all_rolls_back_every_class_if_one_projection_is_unsafe():
    population = _population(c=4)
    population[:, :3].insert(_RetainedProbe, g=2.0)
    module = _VectorOverride()
    population[:, :3].insert(_RangeProbe, alias="generated", g=module)
    population.initialize()
    retained_support = population._mechanism_support_flat(_RetainedProbe).clone()
    range_support = population._mechanism_support_flat(_RangeProbe).clone()
    mechanism_data = population._mech_data
    retained_records = population._mech_data[_RetainedProbe]
    range_records = population._mech_data[_RangeProbe]
    parametrizations = population._slice_mechanism_parametrizations
    range_record = population._mech_data[_RangeProbe][0]
    old_retained = _mechanism(population, _RetainedProbe)
    old_range = _mechanism(population, _RangeProbe)
    old_integrator = population.integrator

    with pytest.raises(ValueError, match="module-valued override"):
        population[:, 1].delete_all()

    assert torch.equal(
        population._mechanism_support_flat(_RetainedProbe), retained_support
    )
    assert torch.equal(population._mechanism_support_flat(_RangeProbe), range_support)
    assert population._mech_data is mechanism_data
    assert population._mech_data[_RetainedProbe] is retained_records
    assert population._mech_data[_RangeProbe] is range_records
    assert population._slice_mechanism_parametrizations is parametrizations
    assert population._mech_data[_RangeProbe][0][1]["g"] is module
    assert population._mech_data[_RangeProbe][0][2] is range_record[2]
    assert _mechanism(population, _RetainedProbe) is old_retained
    assert _mechanism(population, _RangeProbe) is old_range
    assert population.integrator is old_integrator
    assert population.initialized
    assert not population._flag_rebuild


def test_non_scalar_module_override_rejection_is_atomic():
    population = _population(c=4)
    module = _VectorOverride()
    population[:, :3].insert(_RangeProbe, alias="generated", g=module)
    before_support = population._mechanism_support_flat(_RangeProbe).clone()
    before_record = population._mech_data[_RangeProbe][0]

    with pytest.raises(ValueError, match="module-valued override"):
        population[:, 1].delete(_RangeProbe)

    after_record = population._mech_data[_RangeProbe][0]
    assert torch.equal(population._mechanism_support_flat(_RangeProbe), before_support)
    assert after_record[1]["g"] is module
    assert after_record[2] is before_record[2]
    assert not population._flag_rebuild


def test_non_scalar_batch_override_rejection_is_atomic():
    population = _population(n=2, c=3)
    value = torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    population[:, :].insert(_BatchProbe, scale=value)
    before_support = population._mechanism_support_flat(_BatchProbe).clone()

    with pytest.raises(ValueError, match="non-scalar BATCH override"):
        population[:, 1].delete(_BatchProbe)

    assert torch.equal(population._mechanism_support_flat(_BatchProbe), before_support)
    assert population._mech_data[_BatchProbe][0][1]["scale"] is value
    assert not population._flag_rebuild


def test_rebuild_after_whole_delete_clears_ion_and_current_registries():
    population = _population(c=2)
    population.insert(_SodiumProbe)
    population.insert(_RetainedProbe)
    population.build()

    assert "na" in population.mech.ions
    assert "ina" in population.mech.currents
    assert "na" in population._ion_write

    population.delete(_SodiumProbe)
    population.build()

    assert "_SodiumProbe" not in population.mech.mechanisms
    assert "_RetainedProbe" in population.mech.mechanisms
    assert "na" not in population.mech.ions
    assert "ina" not in population.mech.currents
    assert "na" not in population._ion_write
