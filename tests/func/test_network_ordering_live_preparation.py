"""Ordered initialization retains effective fields and live process recipes."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_custom_material_processes import DT, DTYPE, _network
from tests.func.test_network import _close

pytestmark = pytest.mark.cpu


class _AddScale(torch.nn.Module):
    def forward(self, value, increment):
        return (value + increment,)


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _loss(state):
    return sum(
        population["materials"]["pool"]["amount"].sum()
        for population in state["populations"].values()
    )


def _native_loss(network):
    return sum(
        population.mech.materials["pool"].amount.sum()
        for population in network.populations.values()
    )


@pytest.mark.parametrize("phase", (None, "pre", "post"))
def test_fresh_retained_coefficients_keep_direct_material_process_parameter_live(phase):
    def network():
        model = _network()
        if phase is not None:
            register = getattr(model.right, f"register_{phase}_initialize_transform")
            reference = "parameters.integrator.mech.material_processes.local.scale"
            register(
                "adjust_scale",
                _AddScale(),
                reads=(reference,),
                writes=(reference,),
                inputs={"increment": model.t.new_tensor(0.2)},
            )
        return model

    source, native = network(), network()
    functional, tensors = dn.func.make_functional_network(source)
    parameters = tensors.independent_parameters(
        ("populations.*.*material_processes.local.scale",)
    )
    initialized = functional.initialize(
        parameters, tensors.constants, tensors.state, functional.initialization_inputs
    )
    with torch.no_grad():
        native.initialize(DT)
    key = "integrator.mech.material_processes.local.scale"
    assert (
        initialized.parameters["populations"]["left"][key]
        is initialized.parameters["populations"]["right"][key]
    )
    state = initialized.state
    for _ in range(3):
        state, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, state
        )
        native.step()
        _close(state, functional.extract(native).state)
    actual = torch.autograd.grad(_loss(state), parameters["populations"]["left"][key])[
        0
    ]
    expected = torch.autograd.grad(
        _native_loss(native), native.left.mech.material_processes["local"].scale
    )[0]
    assert bool(expected.abs() > 0)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("cached", (False, True))
def test_fresh_retained_network_workspace_keeps_parametrized_geometry_gradient(cached):
    from tests.func.test_network_torch_parametrizations import _TemperatureScale
    from tests.func.test_torch_parametrizations import _population

    def network():
        model = _population("ub")
        model.v_init = torch.linspace(-65, -50, model.v.numel(), dtype=DTYPE).reshape(
            model.shape
        )
        transform = _TemperatureScale(torch.nn.Parameter(model.t.new_tensor(0.9)))
        model.register_parametrization("diam", transform)
        result = dn.Network({"cell": model})
        result.train()
        result.initialize(DT)
        if cached:
            with torch.no_grad():
                result.step()
                model.cache("_steady_state")
        return result

    source, native = network(), network()
    functional, tensors = dn.func.make_functional_network(source)
    key = "parametrizations.diam.0.factor"
    parameters = tensors.independent_parameters((f"populations.cell.{key}",))
    initialized = functional.initialize(
        parameters, tensors.constants, tensors.state, functional.initialization_inputs
    )
    native.initialize(DT)
    state = initialized.state
    for _ in range(2):
        state, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, state
        )
        native.step()
        _close(state, functional.extract(native).state)
    actual = torch.autograd.grad(
        state["populations"]["cell"]["integrator"]["v"].square().sum(),
        parameters["populations"]["cell"][key],
    )[0]
    expected = torch.autograd.grad(
        native.cell.v.square().sum(), native.cell.parametrizations.diam[0].factor
    )[0]
    assert bool(expected.abs() > 0)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)


class _SqrtFactor(torch.nn.Module):
    def forward(self, value, factor):
        return value * factor.sqrt()


def test_retained_shared_configuration_does_not_evaluate_an_inactive_singular_vjp():
    from tests.func.test_network_initialization_bindings import (
        _cross_population_binding,
    )

    source, key = _cross_population_binding()
    source.right.in_graph_parametrizations["cm"].clear()
    source.right.register_parametrization_in_graph(
        "cm", _SqrtFactor(), args=("factor",)
    )
    source.populations = torch.nn.ModuleDict(
        {"right": source.right, "left": source.left}
    )
    source._refresh_step_schedule()
    functional, tensors = dn.func.make_functional_network(source)
    parameters = tensors.independent_parameters((f"populations.left.{key}",))
    defaults = functional.initialization_inputs

    def run(increment):
        populations = dict(defaults.populations)
        owner = populations["left"]
        populations["left"] = owner._replace(
            transforms={
                name: value.new_full(value.shape, increment)
                for name, value in owner.transforms.items()
            }
        )
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            tensors.state,
            defaults._replace(populations=populations),
        )
        state, _ = functional.prepare_and_step(*initialized)
        return state["populations"]["right"]["integrator"]["v"]

    expected = run(0.0)
    expected_gradient = torch.autograd.grad(
        expected.square().sum(), parameters["populations"]["left"][key]
    )[0]
    actual = run(-1.0)
    actual_gradient = torch.autograd.grad(
        actual.square().sum(), parameters["populations"]["left"][key]
    )[0]
    torch.testing.assert_close(actual, expected)
    assert bool(
        expected_gradient.isfinite().all() and expected_gradient.abs().sum() > 0
    )
    torch.testing.assert_close(actual_gradient, expected_gradient)
