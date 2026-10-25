"""Retained packets preserve native operands, ordering, and projection routes."""

from __future__ import annotations

import re

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_initialization_execution import _assert_snapshot, _snapshot
from tests.func.test_network_initialization_ordering import (
    DT,
    DTYPE,
    RAW_G,
    _native_voltage,
    _network,
    _packed_network,
    _voltage,
)
from tests.func.test_noise_workspaces import DT as NOISE_DT
from tests.func.test_noise_workspaces import _network as _noise_network

pytestmark = pytest.mark.cpu
_POPULATION_FIELDS = ("celsius", "rhoa_scale", "cm_scale", "area_scale", "cm", "rhoa")
_PAS = "integrator.mech.mechanisms.pas."
_INTEGRATOR_FIELDS = ("integrator.cmdt", "integrator.area")
_ORDINARY_PATHS = (*_POPULATION_FIELDS, _PAS + "g", _PAS + "e", *_INTEGRATOR_FIELDS)
_PACKED_PATHS = (
    *_POPULATION_FIELDS,
    "diam",
    "dx",
    "_component_area",
    _PAS + "g",
    _PAS + "e",
    *(f"populations.one.{name}" for name in _POPULATION_FIELDS),
    *(f"populations.two.{name}" for name in _POPULATION_FIELDS),
    *_INTEGRATOR_FIELDS,
)


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=True,
            IMEM=False,
            DEVICE="cpu",
            DTYPE=DTYPE,
            CHECK_PURITY=False,
        ),
    ):
        yield


def _key(kind, path):
    return f"materializations.{kind}.{path}"


@pytest.mark.parametrize("packed", (False, True), ids=("ordinary", "packed"))
def test_snapshot_packets_keep_value_active_baseline_order_and_native_projection(
    packed,
):
    source = _packed_network() if packed else _network()
    functional, tensors = dn.func.make_functional_network(source)
    owner = "packed" if packed else "left"
    paths = _PACKED_PATHS if packed else _ORDINARY_PATHS
    parameters = tensors.parameters["populations"][owner]
    constants = tensors.constants["populations"][owner]
    expected_keys = tuple(
        _key(kind, path) for kind in ("values", "active", "baseline") for path in paths
    )
    assert (
        tuple(key for key in constants if key.startswith("materializations."))
        == expected_keys
    )
    assert not any(key.endswith(".integrator.dt") for key in expected_keys)
    prepared = functional.populations[owner].prepare(parameters, constants)
    assert tuple(prepared.values["materializations"]) == paths
    torch.testing.assert_close(prepared.values["integrator"]["dt"], constants["dt"])
    for path in paths:
        value = constants[_key("values", path)]
        native = source.populations[owner].get_buffer(path)
        torch.testing.assert_close(value, native)
        assert value is not native and not torch._C._is_alias_of(value, native)
        assert constants[_key("active", path)].shape == ()
        assert constants[_key("active", path)].dtype == torch.bool
        assert not constants[_key("active", path)].item()
        torch.testing.assert_close(prepared.values["materializations"][path], native)
    if packed:
        for name in ("one", "two"):
            path = f"populations.{name}.cm"
            torch.testing.assert_close(
                prepared.values["materializations"][path],
                source.packed.populations[name].cm,
            )


@pytest.mark.parametrize("failure", ("value", "baseline", "active"))
def test_retained_packet_validation_keeps_first_operand_and_field_precedence(failure):
    functional, tensors = dn.func.make_functional_network(_network())
    plan = functional.populations["left"]
    parameters = tensors.parameters["populations"]["left"]
    constants = dict(tensors.constants["populations"]["left"])
    path, later = "celsius", "rhoa_scale"
    value, baseline, active = (
        _key(kind, path) for kind in ("values", "baseline", "active")
    )
    constants[active] = constants[active].to(DTYPE)
    constants[_key("values", later)] = constants[_key("values", later)].unsqueeze(0)
    if failure in ("value", "baseline"):
        constants[baseline] = constants[baseline].unsqueeze(0)
        selected = baseline
        if failure == "value":
            constants[value] = constants[value].unsqueeze(0)
            selected = value
        message = f"constant {selected!r} has incompatible retained schema"
    else:
        message = f"constant {active!r} must be a scalar bool on cpu"
    supplied = _snapshot((parameters, constants))
    with pytest.raises(ValueError, match=re.escape(message)):
        plan.prepare(parameters, constants)
    _assert_snapshot((parameters, constants), supplied)


def test_fresh_retention_replaces_old_graphs_without_losing_earlier_owner_values():
    functional, tensors = dn.func.make_functional_network(_network())
    constants = tensors.constants
    target = tensors.parameters["populations"]["left"][RAW_G]
    gradients = []
    for _ in range(2):
        incoming = _snapshot(constants)
        initialized = functional.initialize(
            tensors.parameters,
            constants,
            tensors.state,
            functional.initialization_inputs,
        )
        native = _network()
        native.initialize(DT)
        for name in ("left", "right"):
            values = initialized.constants["populations"][name]
            torch.testing.assert_close(
                values[_key("values", "cm_scale")], native.populations[name].cm_scale
            )
            assert values[_key("active", "cm_scale")].item()
            torch.testing.assert_close(
                values[_key("baseline", "cm_scale")],
                torch.zeros_like(values[_key("baseline", "cm_scale")]),
                rtol=0,
                atol=0,
            )
        assert not torch.equal(
            initialized.constants["populations"]["left"][_key("values", "cm_scale")],
            initialized.constants["populations"]["right"][_key("values", "cm_scale")],
        )
        final, _ = functional.prepare_and_step(*initialized)
        native.step()
        torch.testing.assert_close(_voltage(final), _native_voltage(native))
        actual = torch.autograd.grad(_voltage(final).square().sum(), target)[0]
        expected = torch.autograd.grad(
            _native_voltage(native).square().sum(), native.left.get_parameter(RAW_G)
        )[0]
        assert actual.abs().sum() > 0 and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
        gradients.append(actual)
        _assert_snapshot(constants, incoming)
        constants = initialized.constants
    torch.testing.assert_close(gradients[0], gradients[1])


def test_owned_cache_operand_is_detached_while_fresh_owner_keeps_native_gradient():
    source, native = _network(cached="left"), _network(cached="left")
    functional, tensors = dn.func.make_functional_network(source)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    path = _PAS + "g"
    cached = initialized.constants["populations"]["left"][f"cache.values.{path}"]
    assert not cached.requires_grad and cached.grad_fn is None
    retained = tuple(
        initialized.constants["populations"][name][_key("values", path)]
        for name in ("left", "right")
    )
    for name, value in zip(("left", "right"), retained, strict=True):
        torch.testing.assert_close(value, native.populations[name].get_buffer(path))
    actual = torch.autograd.grad(
        sum(value.sum() for value in retained),
        tensors.parameters["populations"]["left"][RAW_G],
    )[0]
    expected = torch.autograd.grad(
        native.left.mech.pas.g.sum() + native.right.mech.pas.g.sum(),
        native.left.get_parameter(RAW_G),
    )[0]
    torch.testing.assert_close(actual, expected)


def test_noise_timestep_replacement_updates_retained_values_flags_and_baselines():
    source, native = _noise_network(transforms=True), _noise_network(transforms=True)
    functional, tensors = dn.func.make_functional_network(source)
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    defaults = functional.initialization_inputs
    inputs = defaults._replace(
        populations={
            name: value._replace(random_draw=packets[name])
            for name, value in defaults.populations.items()
        }
    )
    initialized = functional.initialize(*tensors, inputs)
    native.initialize(NOISE_DT)
    for name in ("left", "right"):
        constants = initialized.constants["populations"][name]
        for suffix in ("coefficient", "DE._WorkspaceGate.coefficient"):
            path = f"integrator.mech.mechanisms.probe.{suffix}"
            value = constants[_key("values", path)]
            torch.testing.assert_close(value, native.populations[name].get_buffer(path))
            assert constants[_key("active", path)].item()
            torch.testing.assert_close(
                constants[_key("baseline", path)],
                torch.zeros_like(value),
                rtol=0,
                atol=0,
            )


def test_preparation_projection_prefers_prepared_then_non_none_extras_then_geometry():
    functional, tensors = dn.func.make_functional_network(_packed_network())
    plan = functional.populations["packed"]
    constants = tensors.constants["populations"]["packed"]
    prepared = plan.prepare(tensors.parameters["populations"]["packed"], constants)
    values = pytree.tree_map(lambda value: value, prepared.values)
    extras = dict(values["materializations"])
    selected = tuple(
        torch.full_like(value, number, requires_grad=True)
        for value, number in (
            (values["population"]["cm"], 2.0),
            (values["geometry"]["diam"], 3.0),
            (values["geometry"]["dx"], 4.0),
        )
    )
    ignored = tuple(
        torch.full_like(value, number, requires_grad=True)
        for value, number in (
            (selected[0], 5.0),
            (selected[0], 6.0),
            (selected[1], 7.0),
        )
    )
    values["population"]["cm"] = selected[0]
    extras["cm"] = ignored[0]
    values["geometry"]["cm"] = ignored[1]
    extras["diam"] = selected[1]
    values["geometry"]["diam"] = ignored[2]
    extras["dx"] = None
    values["geometry"]["dx"] = selected[2]
    projected = plan._finish_retained_preparation(values, constants, extras=extras)[
        "materializations"
    ]
    assert projected["cm"] is selected[0]
    assert projected["diam"] is selected[1]
    assert projected["dx"] is selected[2]
    derivatives = torch.autograd.grad(
        sum(projected[path].sum() for path in ("cm", "diam", "dx")),
        (*selected, *ignored),
        allow_unused=True,
    )
    for value, derivative in zip(selected, derivatives[:3], strict=True):
        torch.testing.assert_close(derivative, torch.ones_like(value))
    assert derivatives[3:] == (None, None, None)


@pytest.mark.parametrize("packed", (False, True), ids=("ordinary", "packed"))
def test_retained_preparation_fullgraph_keeps_voltage_vjp(packed):
    functional, tensors = dn.func.make_functional_network(
        _packed_network() if packed else _network()
    )
    owner = "packed" if packed else "left"
    plan = functional.populations[owner]
    parameters = tensors.parameters["populations"][owner]
    constants = tensors.constants["populations"][owner]
    previous = tensors.state["populations"][owner]
    voltage = previous["integrator"]["v"].detach().clone().requires_grad_()

    def advance(voltage):
        state = {
            **previous,
            "integrator": {**previous["integrator"], "v": voltage},
        }
        final, _ = plan.prepare_and_step(parameters, constants, state)
        return final["integrator"]["v"]

    expected = advance(voltage)
    compiled = torch.compile(advance, backend="aot_eager", fullgraph=True)
    actual = compiled(voltage)
    torch.testing.assert_close(actual, expected)
    actual_vjp = torch.autograd.grad(actual.square().sum(), voltage)[0]
    expected_vjp = torch.autograd.grad(expected.square().sum(), voltage)[0]
    assert torch.isfinite(actual_vjp).all() and torch.count_nonzero(actual_vjp)
    torch.testing.assert_close(actual_vjp, expected_vjp)
