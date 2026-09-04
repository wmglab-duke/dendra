"""Functional geometry contracts for regionally placed mechanisms."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._support import SupportKind
from dendra.models.mod import pas

DT = 0.01
DTYPE = torch.float64
LAYOUTS = ("rectangular", "shared_columns", "packed", "duplicates")
EXPECTED_KINDS = {
    "rectangular": SupportKind.RECTANGULAR,
    "shared_columns": SupportKind.SHARED_COLUMNS,
    "packed": SupportKind.PACKED_FLAT,
    "duplicates": SupportKind.PACKED_FLAT,
}

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _DiameterState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = diam")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _DiameterMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_DiameterState)


def _diameters() -> torch.Tensor:
    return torch.arange(1.0, 11.0, dtype=DTYPE).reshape(2, 5)


def _insert_regional_mechanism(model, layout: str) -> None:
    if layout == "rectangular":
        target = model[:, 1:4]
        preserve_multiplicity = False
    elif layout == "shared_columns":
        target = model[:, torch.tensor([1, 3])]
        preserve_multiplicity = False
    elif layout == "packed":
        target = model[
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 4, 2]),
        ]
        preserve_multiplicity = False
    elif layout == "duplicates":
        target = model[
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 1, 3, 3]),
        ]
        preserve_multiplicity = True
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unknown regional layout {layout!r}")

    target.insert(
        _DiameterMechanism,
        preserve_multiplicity=preserve_multiplicity,
    )


def _model(layout: str):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            _diameters(),
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        _insert_regional_mechanism(model, layout)
        model.initialize()
        model.train()

    mechanism = model.mech._DiameterMechanism
    assert mechanism.support_spec.kind is EXPECTED_KINDS[layout]
    return model, mechanism


def _case(layout: str):
    model, mechanism = _model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    return model, mechanism, functional, tensors


def _change_first_support_slot(mechanism) -> None:
    with torch.no_grad():
        mechanism.key[0] = (int(mechanism.key[0]) + 1) % 10


def _flat_support_indices(mechanism) -> torch.Tensor:
    return mechanism.support_spec.materialized_flat_indices(
        mechanism.key,
        device=mechanism.x.device,
    )


def _expected_local(diameter: torch.Tensor, mechanism) -> torch.Tensor:
    batch_shape = diameter.shape[:-2]
    flat = diameter.reshape(*batch_shape, -1)
    selected = flat.index_select(-1, _flat_support_indices(mechanism))
    return selected.reshape(*batch_shape, *mechanism.x.shape)


def _next_x(functional, tensors, diameter: torch.Tensor) -> torch.Tensor:
    constants = dict(tensors.constants)
    constants["diam"] = diameter
    next_state, _aux = functional.prepare_and_step(
        tensors.parameters,
        constants,
        tensors.state,
    )
    return next_state["mechanisms"]["_DiameterMechanism"]["x"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_regional_preparation_binds_exact_local_geometry_and_aliases(layout):
    _model, mechanism, functional, tensors = _case(layout)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert len(prepared.values["local_geometry"]) == 1
    local_diameter = prepared.values["local_geometry"][0]
    expected = _expected_local(tensors.constants["diam"], mechanism)
    assert local_diameter.shape == mechanism.x.shape
    torch.testing.assert_close(local_diameter, expected, rtol=0.0, atol=0.0)

    mapping, *_positionals = functional._mapping(
        tensors.parameters,
        prepared.values,
        tensors.state,
        isolate_read_only=False,
    )
    mechanism_path = "population.integrator.mech.mechanisms._DiameterMechanism.diam"
    state_path = (
        "population.integrator.mech.mechanisms._DiameterMechanism."
        "DE._DiameterState.diam"
    )
    assert mapping[mechanism_path] is local_diameter
    assert mapping[state_path] is local_diameter


@pytest.mark.parametrize("layout", LAYOUTS)
def test_regional_diameter_state_has_exact_step_and_jacobian_oracles(layout):
    _model, mechanism, functional, tensors = _case(layout)
    diameter = tensors.constants["diam"]

    expected_state = DT * _expected_local(diameter, mechanism)
    actual_state = _next_x(functional, tensors, diameter)
    torch.testing.assert_close(actual_state, expected_state, rtol=0.0, atol=0.0)

    def response(value):
        return _next_x(functional, tensors, value)

    reverse = torch.func.jacrev(response)(diameter)
    forward = torch.func.jacfwd(response)(diameter)

    selector = torch.eye(
        diameter.numel(),
        dtype=diameter.dtype,
        device=diameter.device,
    ).index_select(0, _flat_support_indices(mechanism))
    expected_jacobian = (DT * selector).reshape(
        *mechanism.x.shape,
        *diameter.shape,
    )
    torch.testing.assert_close(reverse, expected_jacobian, rtol=0.0, atol=0.0)
    torch.testing.assert_close(forward, expected_jacobian, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layout", ["shared_columns", "duplicates"])
def test_regional_diameter_state_vmap_includes_zero_lanes(layout):
    _model, mechanism, functional, tensors = _case(layout)
    diameter = tensors.constants["diam"]
    lanes = torch.stack((diameter * 0.5, diameter, diameter + 10.0))

    def response(value):
        return _next_x(functional, tensors, value)

    actual = torch.vmap(response)(lanes)
    expected = DT * _expected_local(lanes, mechanism)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    empty = torch.vmap(response)(diameter.new_empty((0, *diameter.shape)))
    assert empty.shape == (0, *mechanism.x.shape)


@pytest.mark.parametrize("layout", ["shared_columns", "duplicates"])
def test_compile_jacrev_of_regional_diameter_state_is_fullgraph_compatible(layout):
    _model, _mechanism, functional, tensors = _case(layout)
    diameter = tensors.constants["diam"]

    def response(value):
        return _next_x(functional, tensors, value)

    transformed = torch.func.jacrev(response)
    expected = transformed(diameter)

    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(diameter)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layout", ["shared_columns", "packed", "duplicates"])
def test_existing_plan_rejects_inplace_support_key_changes(layout):
    _model_value, mechanism, functional, tensors = _case(layout)
    _change_first_support_slot(mechanism)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"selector|support|placement",
    ):
        _next_x(functional, tensors, tensors.constants["diam"])


@pytest.mark.parametrize("layout", ["shared_columns", "packed", "duplicates"])
def test_existing_plan_accepts_same_value_support_key_rebinding(layout):
    _model_value, mechanism, functional, tensors = _case(layout)
    rebound_key = mechanism.key.clone()
    mechanism.key = rebound_key

    expected = DT * _expected_local(tensors.constants["diam"], mechanism)
    first = _next_x(functional, tensors, tensors.constants["diam"])
    second = _next_x(functional, tensors, tensors.constants["diam"])

    assert mechanism.key is rebound_key
    torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layout", ["shared_columns", "packed"])
def test_existing_plan_fails_closed_when_support_key_values_are_invalid(layout):
    _model_value, mechanism, functional, _tensors = _case(layout)
    mechanism._support_key_values_valid = False

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"selector values are unavailable or invalid",
    ):
        functional.extract()


def test_existing_plan_rejects_pending_population_rebuild():
    model, _mechanism, functional, _tensors = _case("shared_columns")
    model[:, 0].insert(pas)

    assert model._flag_rebuild
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"pending structural changes",
    ):
        functional.extract()


@pytest.mark.parametrize("layout", ["shared_columns", "packed", "duplicates"])
def test_pre_lowering_support_key_change_disagrees_with_compiled_placement(layout):
    model, mechanism = _model(layout)
    _change_first_support_slot(mechanism)
    compiled_keys = dict(zip(model._m_name, model._m_keys, strict=True))
    compiled_key = torch.as_tensor(
        compiled_keys[mechanism.name],
        dtype=mechanism.key.dtype,
        device=mechanism.key.device,
    )

    assert not torch.equal(mechanism.key, compiled_key)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"selector|support|placement",
    ):
        dn.func.FunctionalPopulation(model, dt=DT)
