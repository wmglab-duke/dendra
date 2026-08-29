"""Pure Population pre/post initialization-transform contracts."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh

DT = 0.01
Celsius = "celsius_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _Add(torch.nn.Module):
    def forward(self, value, increment):
        return (value + increment,)


class _ShiftScalarAndVoltage(torch.nn.Module):
    def forward(self, scalar, voltage, offset):
        return (scalar + offset, voltage + offset)


class _AdvanceClock(torch.nn.Module):
    def forward(self, time, remainder, amount):
        return (time + amount, remainder + amount)


class _AddClockToVoltage(torch.nn.Module):
    def forward(self, voltage, time):
        return (voltage + time,)


def _model(*, pre=(), post=()):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            v_init=torch.tensor([-67.0, -63.0, -59.0], dtype=torch.float64),
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(hh)
        for args, kwargs in pre:
            model.register_pre_initialize_transform(*args, **kwargs)
        for args, kwargs in post:
            model.register_post_initialize_transform(*args, **kwargs)
        model.initialize()
        model.train()
    return model


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_values, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_values, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(
        actual_values,
        expected_values,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
        )


def test_pre_transform_inputs_are_explicit_and_compose_in_registration_order():
    reference = "state.integrator.v"
    model = _model(
        pre=(
            (
                ("first", _Add()),
                {
                    "reads": (reference,),
                    "writes": (reference,),
                    "inputs": {"increment": torch.tensor(2.0, dtype=torch.float64)},
                },
            ),
            (
                ("second", _Add()),
                {
                    "reads": (reference,),
                    "writes": (reference,),
                    "inputs": {"increment": torch.tensor(3.0, dtype=torch.float64)},
                },
            ),
        )
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert tuple(tensors.initialization.transforms) == (
        "pre.first.increment",
        "pre.second.increment",
    )
    inputs = dict(tensors.initialization.transforms)
    inputs["pre.first.increment"] = torch.tensor(-1.0, dtype=model.dtype())
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(transforms=inputs),
    )

    expected = tensors.initialization.v_init + 2.0
    torch.testing.assert_close(
        initialized.state["integrator"]["v"],
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert initialized.initialization.transforms is inputs
    torch.testing.assert_close(
        tensors.initialization.transforms["pre.first.increment"],
        torch.tensor(2.0, dtype=model.dtype()),
    )


def test_post_parameter_transform_returns_updated_parameters_and_composes_with_step():
    parameter = f"parameters.{Celsius}"
    model = _model(
        post=(
            (
                ("raise_temperature", _Add()),
                {
                    "reads": (parameter,),
                    "writes": (parameter,),
                    "inputs": {"increment": torch.tensor(1.0, dtype=torch.float64)},
                },
            ),
        )
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    source_parameter = tensors.parameters[Celsius].detach().clone()

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    torch.testing.assert_close(
        initialized.parameters[Celsius],
        source_parameter + 1.0,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        tensors.parameters[Celsius],
        source_parameter,
        rtol=0.0,
        atol=0.0,
    )

    reference = copy.deepcopy(model)
    reference.initialize()
    expected_initialized = functional.extract(reference)
    _assert_tree_close(initialized.state, expected_initialized.state)
    torch.testing.assert_close(
        initialized.parameters[Celsius],
        expected_initialized.parameters[Celsius],
        rtol=0.0,
        atol=0.0,
    )

    actual_state, _auxiliary = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.StepInput(),
    )
    reference.step(dt=DT)
    _assert_tree_close(
        actual_state,
        functional.extract(reference).state,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_transform_input_supports_higher_order_autograd_vmap_and_compile():
    reference = "state.integrator.v"
    model = _model(
        post=(
            (
                ("shift_final_voltage", _Add()),
                {
                    "reads": (reference,),
                    "writes": (reference,),
                    "inputs": {"offset": torch.tensor(0.5, dtype=torch.float64)},
                },
            ),
        )
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    key = "post.shift_final_voltage.offset"

    def voltage(offset):
        transform_inputs = dict(tensors.initialization.transforms)
        transform_inputs[key] = offset
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization._replace(transforms=transform_inputs),
        )
        return initialized.state["integrator"]["v"]

    offset = torch.tensor(0.25, dtype=model.dtype())
    expected = tensors.initialization.v_init + offset
    torch.testing.assert_close(voltage(offset), expected, rtol=0.0, atol=0.0)

    jacobian = torch.func.jacrev(voltage)(offset)
    torch.testing.assert_close(jacobian, torch.ones_like(expected))

    def loss(value):
        return voltage(value).pow(3).sum()

    hessian = torch.func.hessian(loss)(offset)
    torch.testing.assert_close(hessian, 6.0 * expected.sum())

    lanes = torch.tensor([-1.0, 0.0, 2.0], dtype=model.dtype())
    actual_lanes = torch.vmap(voltage)(lanes)
    expected_lanes = torch.stack([voltage(value) for value in lanes])
    torch.testing.assert_close(actual_lanes, expected_lanes, rtol=0.0, atol=0.0)
    empty = torch.vmap(voltage)(lanes[:0])
    assert empty.shape == (0, *model.shape)

    transformed = torch.func.jacrev(voltage)
    compiled = torch.compile(
        transformed,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        actual_compiled = compiled(offset)
    torch.testing.assert_close(actual_compiled, jacobian, rtol=0.0, atol=0.0)


def test_mixed_rank_multiwrite_transform_preserves_each_output_schema():
    parameter = f"parameters.{Celsius}"
    voltage_reference = "state.integrator.v"
    model = _model(
        post=(
            (
                ("shift_scalar_and_voltage", _ShiftScalarAndVoltage()),
                {
                    "reads": (parameter, voltage_reference),
                    "writes": (parameter, voltage_reference),
                    "inputs": {"offset": torch.tensor(0.5, dtype=torch.float64)},
                },
            ),
        )
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    key = "post.shift_scalar_and_voltage.offset"

    def outputs(offset):
        transform_inputs = dict(tensors.initialization.transforms)
        transform_inputs[key] = offset
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization._replace(transforms=transform_inputs),
        )
        return (
            initialized.parameters[Celsius],
            initialized.state["integrator"]["v"],
        )

    scalar, voltage = outputs(torch.tensor(0.25, dtype=model.dtype()))
    assert scalar.shape == tensors.parameters[Celsius].shape == ()
    torch.testing.assert_close(scalar, tensors.parameters[Celsius] + 0.25)
    torch.testing.assert_close(voltage, tensors.initialization.v_init + 0.25)

    lanes = torch.tensor([-1.0, 2.0], dtype=model.dtype())
    scalar_lanes, voltage_lanes = torch.vmap(outputs)(lanes)
    torch.testing.assert_close(scalar_lanes, tensors.parameters[Celsius] + lanes)
    torch.testing.assert_close(
        voltage_lanes,
        tensors.initialization.v_init.unsqueeze(0) + lanes[:, None, None],
    )


def test_post_clock_writes_are_visible_to_later_actions_then_reset():
    time = "state.clock.t"
    remainder = "state.control.duration_remainder"
    voltage = "state.integrator.v"
    model = _model(
        post=(
            (
                ("advance_clock", _AdvanceClock()),
                {
                    "reads": (time, remainder),
                    "writes": (time, remainder),
                    "inputs": {"amount": torch.tensor(2.0, dtype=torch.float64)},
                },
            ),
            (
                ("use_clock", _AddClockToVoltage()),
                {
                    "reads": (voltage, time),
                    "writes": (voltage,),
                },
            ),
        )
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)

    torch.testing.assert_close(model.v, model.expanded_v_init(model.shape) + 2.0)
    torch.testing.assert_close(model.t, torch.zeros_like(model.t))
    torch.testing.assert_close(
        model._duration_remainder,
        torch.zeros_like(model._duration_remainder),
    )

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )

    torch.testing.assert_close(
        initialized.state["integrator"]["v"],
        tensors.initialization.v_init + 2.0,
    )
    torch.testing.assert_close(
        initialized.state["clock"]["t"],
        torch.zeros_like(initialized.state["clock"]["t"]),
    )
    torch.testing.assert_close(
        initialized.state["control"]["duration_remainder"],
        torch.zeros_like(initialized.state["control"]["duration_remainder"]),
    )


def test_functional_initialization_transform_runs_under_inference_mode():
    reference = "state.integrator.v"
    model = _model(
        pre=(
            (
                ("shift_voltage", _Add()),
                {
                    "reads": (reference,),
                    "writes": (reference,),
                    "inputs": {"increment": torch.tensor(1.0, dtype=torch.float64)},
                },
            ),
        )
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with torch.inference_mode():
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    torch.testing.assert_close(
        initialized.state["integrator"]["v"],
        tensors.initialization.v_init + 1.0,
    )


def test_legacy_population_hook_still_fails_closed_only_for_fresh_initialize():
    model = _model()
    model.register_post_initialize_hook(lambda _model: None)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="arbitrary Population pre/post-initialize hooks",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)
