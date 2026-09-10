"""Physical-support and lifecycle contracts for passive cable ends."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.core import passive_end_nodes_
from dendra.models.mod import hh, pas

DTYPE = torch.float64
DT = 0.01


@pytest.fixture(autouse=True)
def _eager_models():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0):
        yield


def _model(*, kind="cable", batches=()):
    initial = -72.0 + torch.arange(14, dtype=DTYPE).reshape(2, 7)
    # Explicit spatial parameters make public Slice reads meaningful even for
    # a mechanism whose default parameter storage is a broadcast scalar.
    rhoa = torch.full((2, 7), 80.0, dtype=DTYPE)
    cm = torch.full((2, 7), 2.5, dtype=DTYPE)
    if kind == "cable":
        model = dn.Cable(N=2, C=7, v_init=initial, rhoa=rhoa, cm=cm, dtype=DTYPE)
    elif kind == "axon":
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=6.0,
            dx=1.0,
            v_init=initial,
            rhoa=rhoa,
            cm=cm,
            dtype=DTYPE,
        )
    elif kind == "native":
        morphology = dn.Morphology(rhoa=120.0, cm=2.5)
        morphology.section("path", L=70.0, diam=2.0, nseg=7)
        model = dn.Cable.from_morphology(morphology, N=2, v_init=initial, dtype=DTYPE)
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(kind)
    # Existing passive membrane outside the selected ends must survive with
    # its original parameters even though it shares the new ends' class.
    model.insert(hh)
    model.insert(
        pas,
        g=torch.full((2, 7), 0.02, dtype=DTYPE),
        e=torch.full((2, 7), -42.0, dtype=DTYPE),
    )
    for count in batches:
        model.batch(count)
    model.initialize()
    return model


def _freeze(tree):
    return pytree.tree_map(
        lambda value: value.detach().clone()
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value),
        tree,
    )


def _assert_tree(actual, expected):
    actual_leaves, actual_spec = pytree.tree_flatten(actual)
    expected_leaves, expected_spec = pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for observed, reference in zip(actual_leaves, expected_leaves, strict=True):
        if isinstance(reference, torch.Tensor):
            torch.testing.assert_close(
                observed, reference, rtol=0, atol=0, equal_nan=True
            )
        else:
            assert observed == reference


def _snapshot(model):
    return {
        "flags": (model.is_built, model.initialized, model._flag_rebuild),
        "live_objects": (id(model.mech), id(model.integrator)),
        "tensors": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        },
        "initial": model.expanded_v_init().clone(),
        "configuration": _freeze(
            (model._mech_everywhere, model._mech_data, model._mech_exclusions)
        ),
    }


def _mechanism_values(model, mechanism, field):
    if mechanism not in model.mech.mechanisms:
        return torch.full(model.shape, torch.nan, dtype=model.dtype())
    instance = model.mech.mechanisms[mechanism]
    value = getattr(instance, field)
    if value.ndim == 0:
        # GLOBAL parameters have no Slice coordinates. Project the scalar onto
        # the mechanism's physical support to check removal without inventing
        # a spatial parameter for HH.
        result = torch.full(model.core_shape(), torch.nan, dtype=model.dtype())
        if instance.key is None:
            result.fill_(value)
        elif instance.is_composable:
            result[instance.key] = value
        else:
            result.reshape(-1)[instance.key.reshape(-1)] = value
        return result.expand(model.shape)
    return model[...].inspect(field, mechanism=mechanism)


def _configuration(model):
    return {
        "gnabar": _mechanism_values(model, "hh", "gnabar"),
        "g": _mechanism_values(model, "pas", "g"),
        "e": _mechanism_values(model, "pas", "e"),
        "rhoa": model[...].rhoa,
        "cm": model[...].cm,
    }


def _mask(model, columns, *, row=None):
    result = torch.zeros(model.core_shape(), dtype=torch.bool)
    if row is None:
        result[:, columns] = True
    else:
        result[row, columns] = True
    return result.expand(model.shape)


def _assert_configuration(model, selected, before, *, rhoa=1e10, cm=1.0, e=None):
    actual = _configuration(model)
    reversal = (
        model.expanded_v_init()
        if e is None
        else torch.broadcast_to(torch.as_tensor(e, dtype=DTYPE), model.shape)
    )
    assert torch.equal(torch.isnan(actual["gnabar"]), selected)
    torch.testing.assert_close(
        actual["gnabar"][~selected], before["gnabar"][~selected], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["g"][selected],
        torch.full_like(actual["g"][selected], 1e-4),
        rtol=1e-7,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual["e"][selected], reversal[selected], rtol=0, atol=0
    )
    for field in ("g", "e", "rhoa", "cm"):
        torch.testing.assert_close(
            actual[field][~selected], before[field][~selected], rtol=0, atol=0
        )
    for field, override in (("rhoa", rhoa), ("cm", cm)):
        wanted = (
            before[field][selected]
            if override is None
            else torch.full_like(actual[field][selected], override)
        )
        torch.testing.assert_close(
            actual[field][selected], wanted, rtol=1e-7, atol=1e-12
        )


def test_passive_end_nodes_is_exported_with_the_mutating_name():
    assert "passive_end_nodes_" in dn.__all__
    assert "passive_end_nodes" not in dn.__all__
    assert dn.passive_end_nodes_ is passive_end_nodes_


@pytest.mark.parametrize("sliced", [False, True])
def test_passive_ends_can_be_configured_before_first_initialization(sliced):
    model = dn.Cable(N=1, C=7, v_init=-71.0, dtype=DTYPE)
    model.insert(hh)
    passive_end_nodes_(model[:, 1:6] if sliced else model)
    model.initialize()
    selected = _mask(model, [1, 5] if sliced else [0, 6])
    assert torch.equal(torch.isnan(_mechanism_values(model, "hh", "gnabar")), selected)
    torch.testing.assert_close(
        _mechanism_values(model, "pas", "e")[selected],
        torch.full((2,), -71.0, dtype=DTYPE),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", ["cable", "axon"])
def test_passive_ends_replace_all_mechanisms_and_survive_rebuild_and_simulation(kind):
    model = _model(kind=kind)
    before = _configuration(model)
    initial = model.expanded_v_init().clone()
    model.v.add_(5.0)
    live_voltage = model.v.clone()

    assert passive_end_nodes_(model, 1, rhoa=1e10, cm=1.0) is None
    torch.testing.assert_close(model.v, live_voltage, rtol=0, atol=0)
    torch.testing.assert_close(model.expanded_v_init(), initial, rtol=0, atol=0)
    assert model._flag_rebuild and not model.initialized

    selected = _mask(model, [0, 6])
    first_run = None
    for force_rebuild in (False, False, True):
        if force_rebuild:
            model.build(force_rebuild=True)
        model.initialize()
        _assert_configuration(model, selected, before)
        torch.testing.assert_close(model.v, initial, rtol=0, atol=0)
        model.run(tstop=3 * DT, dt=DT)
        assert torch.isfinite(model.v).all()
        if first_run is None:
            first_run = model.v.clone()
            assert not torch.equal(first_run, initial)
        else:
            torch.testing.assert_close(model.v, first_run, rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 3, 20])
def test_slice_local_ends_union_overlap_and_preserve_all_other_compartments(n):
    model = _model()
    before = _configuration(model)
    target = model[:, 1:6]
    passive_end_nodes_(target, n, rhoa=1e10, cm=1.0)
    model.initialize()
    _assert_configuration(
        model, _mask(model, [1, 5] if n == 1 else [1, 2, 3, 4, 5]), before
    )


def test_reordered_duplicate_columns_keep_each_physical_initial_reversal():
    model = _model()
    before = _configuration(model)
    passive_end_nodes_(model[..., [5, 2, 5, 1]], 2, rhoa=1e10, cm=1.0)
    model.initialize()
    _assert_configuration(model, _mask(model, [1, 2, 5]), before)


def test_scalar_slice_configures_only_its_single_physical_compartment():
    model = _model()
    before = _configuration(model)
    passive_end_nodes_(model[1, 3], 4, rhoa=1e10, cm=1.0)
    model.initialize()
    _assert_configuration(model, _mask(model, [3], row=1), before)


@pytest.mark.parametrize("empty", [False, True])
def test_zero_count_and_empty_targets_are_complete_noops(empty):
    model = _model()
    before = _snapshot(model)
    target, count = (model[:, :0], 3) if empty else (model, 0)
    assert passive_end_nodes_(target, count) is None
    _assert_tree(_snapshot(model), before)


def test_multibatch_slice_configures_shared_physical_ends_for_every_replica():
    model = _model(batches=(2, 3))
    before = _configuration(model)
    assert tuple(model.shape) == (3, 2, 2, 7)
    # Restrict one logical replica, but structural mechanisms/materials are
    # shared. The local end columns are 1 and 5 in every physical replica.
    passive_end_nodes_(model[1, 0, :, 1:6], 1, rhoa=1e10, cm=1.0)
    model.initialize()
    _assert_configuration(model, _mask(model, [1, 5]), before)


def test_batch_initial_voltages_set_each_replica_even_from_a_restricted_slice():
    model = _model(batches=(2, 3))
    initial = model.expanded_v_init().clone()
    initial += torch.arange(6, dtype=DTYPE).reshape(3, 2, 1, 1)
    model.set_v_init(initial)
    before = _configuration(model)
    live_voltage = model.v.clone()
    passive_end_nodes_(model[1, 0], rhoa=1e10, cm=1.0)
    torch.testing.assert_close(model.v, live_voltage, rtol=0, atol=0)
    model.initialize()
    _assert_configuration(model, _mask(model, [0, 6]), before)


def test_batch_initial_voltage_differences_outside_selected_ends_are_allowed():
    model = _model(batches=(2,))
    initial = model.expanded_v_init().clone()
    initial[1, :, 3] += 4.0
    model.set_v_init(initial)
    before = _configuration(model)
    passive_end_nodes_(model[0, :, 1:6], 1, rhoa=1e10, cm=1.0)
    model.initialize()
    _assert_configuration(model, _mask(model, [1, 5]), before)


@pytest.mark.parametrize("n", [-1, True, 1.5, "1"])
def test_invalid_counts_are_rejected_before_any_mutation(n):
    model = _model()
    before = _snapshot(model)
    with pytest.raises((TypeError, ValueError), match="non-negative integer"):
        passive_end_nodes_(model, n)
    _assert_tree(_snapshot(model), before)


@pytest.mark.parametrize("kind", ["population", "population_slice", "mechanism_slice"])
def test_non_cable_and_wrapped_mechanism_targets_are_rejected(kind):
    if kind == "mechanism_slice":
        model = _model()
        target = model[:, 1:6].mech.hh
    else:
        model = dn.Population(N=2, C=7, dtype=DTYPE)
        model.insert(hh)
        model.initialize()
        target = model if kind == "population" else model[:, 1:6]
    before = _snapshot(model)
    with pytest.raises(TypeError, match="Cable"):
        passive_end_nodes_(target, 1)
    _assert_tree(_snapshot(model), before)


@pytest.mark.parametrize(
    "overrides", [{}, {"rhoa": None}, {"cm": None}, {"rhoa": None, "cm": None}]
)
def test_defaults_and_explicit_none_material_overrides(overrides):
    model = _model()
    before = _configuration(model)
    passive_end_nodes_(model, 1, **overrides)
    model.initialize()
    _assert_configuration(
        model,
        _mask(model, [0, 6]),
        before,
        rhoa=overrides.get("rhoa", 1e10),
        cm=overrides.get("cm", 1.0),
    )


def test_native_cable_default_resistivity_override_fails_before_mutation():
    model = _model(kind="native")
    before = _snapshot(model)
    with pytest.raises(ValueError, match="immutable|rhoa=None"):
        passive_end_nodes_(model[:, 1:6], 1)
    _assert_tree(_snapshot(model), before)


def test_native_cable_accepts_passive_ends_and_capacitance_with_rhoa_none():
    model = _model(kind="native")
    before = _configuration(model)
    resistance = model.edge_resistance_ohm.clone()
    passive_end_nodes_(model[:, 1:6], 1, rhoa=None, cm=1.0)
    model.initialize()
    _assert_configuration(model, _mask(model, [1, 5]), before, rhoa=None, cm=1.0)
    torch.testing.assert_close(model.edge_resistance_ohm, resistance, rtol=0, atol=0)
    model.run(tstop=2 * DT, dt=DT)
    assert torch.isfinite(model.v).all()


@pytest.mark.parametrize(
    "profile",
    ["scalar", "compartment", "cell", "spatial", "replica", "full"],
)
def test_custom_reversal_broadcasts_in_root_coordinates_for_reordered_slice(profile):
    model = _model(batches=(2, 3))
    values = {
        "scalar": -61.25,
        "compartment": torch.linspace(-78.0, -66.0, 7, dtype=DTYPE),
        "cell": torch.tensor([[-64.0], [-69.0]], dtype=DTYPE),
        "spatial": -88.0 + torch.arange(14, dtype=DTYPE).reshape(2, 7),
        "replica": -71.0 + torch.arange(6, dtype=DTYPE).reshape(3, 2, 1, 1),
        "full": -95.0 + 0.25 * torch.arange(84, dtype=DTYPE).reshape(3, 2, 2, 7),
    }
    e = values[profile]
    before = _configuration(model)
    initial = model.expanded_v_init().clone()
    passive_end_nodes_(model[1, 0, :, [5, 2, 5, 1]], 2, e=e)
    model.initialize()
    _assert_configuration(model, _mask(model, [1, 2, 5]), before, e=e)
    torch.testing.assert_close(model.expanded_v_init(), initial, rtol=0, atol=0)
    torch.testing.assert_close(model.v, initial, rtol=0, atol=0)


def test_explicit_equal_replica_reversals_retain_replica_storage():
    model = _model(batches=(2,))
    e = torch.full((2, 1, 1), -61.0, dtype=DTYPE)
    before = _configuration(model)
    passive_end_nodes_(model, e=e)
    model.initialize()
    _assert_configuration(model, _mask(model, [0, 6]), before, e=e)
    # Equal values must not collapse the authored batch axis into shared
    # storage: later parameter fitting may give the replicas distinct values.
    assert model.mech.mechanisms["pas"].e.numel() == model.v.numel()


@pytest.mark.parametrize("shape", [(2,), (2, 5), (4, 2, 2, 7)])
def test_invalid_reversal_shapes_are_rejected_before_any_mutation(shape):
    model = _model(batches=(2, 3))
    before = _snapshot(model)
    e = torch.full(shape, -61.0, dtype=DTYPE)
    # A Slice-local [N, 5] profile is deliberately invalid: e belongs to the
    # root Cable, whose compartment axis still has length seven.
    with pytest.raises(ValueError, match="broadcast|shape"):
        passive_end_nodes_(model[1, 0, :, 1:6], e=e)
    _assert_tree(_snapshot(model), before)


def test_reversal_is_keyword_only_and_invalid_positional_call_is_atomic():
    model = _model()
    before = _snapshot(model)
    with pytest.raises(TypeError):
        passive_end_nodes_(model, 1, None, None, -61.0)
    _assert_tree(_snapshot(model), before)


def test_replica_reversal_simulation_matches_independent_cables():
    model = _model(batches=(2,))
    e = torch.tensor([-76.0, -57.0], dtype=DTYPE).reshape(2, 1, 1)
    passive_end_nodes_(model, e=e)
    model.initialize()
    model.run(tstop=3 * DT, dt=DT)
    actual = model.v.clone()
    references = []
    for value in e.reshape(-1):
        single = _model()
        passive_end_nodes_(single, e=value)
        single.initialize()
        single.run(tstop=3 * DT, dt=DT)
        references.append(single.v.clone())
    torch.testing.assert_close(actual, torch.stack(references), rtol=1e-12, atol=1e-12)
    assert not torch.equal(actual[0], actual[1])


def test_replica_reversal_survives_repeated_edits_deletion_rebuild_and_added_batch():
    model = _model(batches=(2,))
    before = _configuration(model)
    e1 = -80.0 + 0.25 * torch.arange(28, dtype=DTYPE).reshape(2, 2, 7)
    e2 = -50.0 + 0.125 * torch.arange(28, dtype=DTYPE).reshape(2, 2, 7)
    outer = _mask(model, [0, 6])
    inner = _mask(model, [1, 5])
    expected = before["e"].clone()
    expected[outer] = e1[outer]
    expected[inner] = e2[inner]
    passive_end_nodes_(model, e=e1)
    model.initialize()
    passive_end_nodes_(model[0, :, 1:6], e=e2)
    model.initialize()
    _assert_configuration(model, outer | inner, before, e=expected)

    # Deleting one replica's logical location changes physical support in all
    # replicas, while retaining distinct e values on the surviving support.
    model[0, 0, 0].delete(pas)
    expected[:, 0, 0] = torch.nan
    model.initialize()
    torch.testing.assert_close(
        _mechanism_values(model, "pas", "e"), expected, rtol=0, atol=0, equal_nan=True
    )
    model.build(force_rebuild=True)
    model.initialize()
    torch.testing.assert_close(
        _mechanism_values(model, "pas", "e"), expected, rtol=0, atol=0, equal_nan=True
    )
    model.batch(3)
    model.initialize()
    torch.testing.assert_close(
        _mechanism_values(model, "pas", "e"),
        expected.unsqueeze(0).expand(model.shape),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
