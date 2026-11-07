"""Canonical DF contracts, optional diagnostics, and tensor transforms."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.nn.utils import parametrize

import dendra as dn
from dendra.func._population import _resolve_functional_integrator_spec
from dendra.models.integrators.explicit import (
    _dufort_frankel,
    _dufort_frankel_homogeneous,
)
from tests.func.test_dufort_frankel import DT, VARIANTS, _native_step, _population
from tests.func.test_euler_contracts import _StencilGain

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "variant,child", [("df", "filter"), ("dfh-conv", "filter"), ("dfh-conv", "ssd")]
)
def test_dufort_parametrized_stencil_and_filter_preserve_raw_gradients(variant, child):
    population = _population(variant, beta=0.8, imem=False)
    population._df_history_valid.fill_(True)
    module = getattr(population.integrator, child)
    parametrize.register_parametrization(module, "weight", _StencilGain())
    native = copy.deepcopy(population)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    raw_name = f"integrator.{child}.parametrizations.weight.original"
    gain_name = f"integrator.{child}.parametrizations.weight.0.gain"
    targets = tuple(
        tensors.parameters[name].detach().clone().requires_grad_()
        for name in (raw_name, gain_name)
    )
    parameters = {**tensors.parameters, **dict(zip((raw_name, gain_name), targets))}
    actual, _ = functional.prepare_and_step(
        parameters, tensors.constants, tensors.state
    )
    gradients = torch.autograd.grad(actual["integrator"]["v"].square().sum(), targets)
    native_targets = tuple(
        dict(native.named_parameters())[name].requires_grad_()
        for name in (raw_name, gain_name)
    )
    _native_step(native)
    expected = torch.autograd.grad(native.v.square().sum(), native_targets)
    torch.testing.assert_close(actual, functional.extract(native).state)
    torch.testing.assert_close(gradients, expected)
    assert all(
        torch.isfinite(gradient).all() and gradient.abs().max() > 0
        for gradient in gradients
    )


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        yield


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("diagnostics", [False, True])
def test_dufort_canonical_family_and_optional_purity_diagnostics(variant, diagnostics):
    population = _population(variant, beta=0.8, imem=False)
    expected = _dufort_frankel if variant == "df" else _dufort_frankel_homogeneous
    spec = _resolve_functional_integrator_spec(
        population.integrator, "scalar_dufort_path"
    )
    assert spec is not None and spec.implementation is expected
    assert spec.workspace_schema and "_prepare_history" in spec.critical_methods
    with dn.ctx(CHECK_PURITY=diagnostics):
        functional, tensors = dn.func.make_functional(population, dt=DT)
        actual, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    assert torch.isfinite(actual["integrator"]["v"]).all()
    assert actual["integrator"]["_df_history_valid"].dtype == torch.bool


@pytest.mark.parametrize(
    "implementation,variant",
    [(_dufort_frankel, "df"), (_dufort_frankel_homogeneous, "dfh-conv")],
)
@pytest.mark.parametrize(
    "hook", ["_prepare_history", "_initial_voltage", "_functional_tensor_step"]
)
def test_dufort_rejects_replaced_numerical_class_hooks(
    implementation, variant, hook, monkeypatch
):
    population = _population(variant, imem=False)
    original = getattr(implementation, hook)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, hook, replacement)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical Euler"):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("mutation", ["beta", "conv", "smoothing", "stencil", "filter"])
def test_dufort_source_configuration_changes_require_lowering_again(mutation):
    population = _population("dfh-conv", beta=0.8, imem=False)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    integrator = population.integrator
    if mutation == "beta":
        integrator.beta = 0.9
    elif mutation == "conv":
        integrator.conv = False
    elif mutation == "smoothing":
        integrator.smoothing = False
    elif mutation == "stencil":
        integrator.ssd._reversed_padding_repeated_twice[0] = 0
    else:
        integrator.filter = torch.nn.Identity()
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)


def _voltage(functional, tensors, voltage, valid=None):
    state = {
        **tensors.state,
        "integrator": {
            **tensors.state["integrator"],
            "v": voltage,
            **({} if valid is None else {"_df_history_valid": valid}),
        },
    }
    output, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, state
    )
    return output["integrator"]["v"]


@pytest.mark.parametrize("variant", ["df", "dfh-conv"])
@pytest.mark.parametrize("lanes", [0, 3])
def test_dufort_mixed_startup_history_vmap_preserves_values_and_gradients(
    variant, lanes
):
    population = _population(variant, beta=0.8, imem=False)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    values = (
        tensors.state["integrator"]["v"]
        .expand(lanes, *population.shape)
        .clone()
        .requires_grad_()
    )
    valid = torch.arange(lanes).remainder(2).to(torch.bool)

    def evaluate(voltage, history_valid):
        return _voltage(functional, tensors, voltage, history_valid)

    actual = torch.vmap(evaluate)(values, valid)
    assert actual.shape == (lanes, *population.shape)
    if lanes:
        expected = torch.stack([evaluate(v, flag) for v, flag in zip(values, valid)])
        torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.square().sum(), values)[0]
    assert gradient.shape == values.shape and torch.isfinite(gradient).all()


@pytest.mark.parametrize("variant", ["df", "dfh-conv"])
def test_dufort_jvp_jacrev_preserve_history_dependent_voltage_derivatives(variant):
    population = _population(variant, beta=0.8, imem=False)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    voltage = tensors.state["integrator"]["v"].detach()
    direction = torch.linspace(-0.2, 0.3, voltage.numel()).reshape_as(voltage)

    def evaluate(value):
        return _voltage(functional, tensors, value, torch.tensor(True))

    _, tangent = torch.func.jvp(evaluate, (voltage,), (direction,))
    epsilon = 1.0e-4
    difference = (
        evaluate(voltage + epsilon * direction)
        - evaluate(voltage - epsilon * direction)
    ) / (2 * epsilon)
    torch.testing.assert_close(tangent, difference, rtol=2e-8, atol=2e-9)
    jacobian = torch.func.jacrev(evaluate)(voltage)
    expected = (jacobian.reshape(voltage.numel(), -1) @ direction.flatten()).reshape_as(
        voltage
    )
    torch.testing.assert_close(tangent, expected, rtol=2e-13, atol=2e-13)


def test_dufort_optional_purity_audit_checks_authored_current_python_state():
    from dendra.models.mechanisms import Mechanism

    class MutatingCurrent(Mechanism):
        Mechanism.RANGE(g=0.0001)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.EXPLICIT("i")
        calls = 0

        def i(self, v):
            self.calls += 1
            return self.g * (v + 65.0)

    population = _population("df", imem=False)
    population.insert(MutatingCurrent.rename("mutating"))
    population.initialize()
    before = population.mech.mutating.calls
    with dn.ctx(CHECK_PURITY=False):
        trusted, tensors = dn.func.make_functional(population, dt=DT)
        actual, _ = trusted.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    assert not trusted._check_purity
    assert torch.isfinite(actual["integrator"]["v"]).all()
    assert population.mech.mutating.calls == before
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(
            dn.func.FunctionalizationError, match="unregistered Python instance state"
        ):
            dn.func.make_functional(population, dt=DT)
    assert population.mech.mutating.calls == before


def _auto_current_delay_population(variant, training):
    from dendra.models.mechanisms import Mechanism

    class AutoDelayedCurrent(Mechanism):
        Mechanism.RANGE(g=0.001)
        Mechanism.CARRY("signal_delay_buffer", shape="deferred")
        Mechanism.CARRY("signal_delay_ptr", shape=(), dtype=torch.long)
        Mechanism.NONSPECIFIC_CURRENT("i")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_delayed_state(
                "signal", self.g.expand(self.shape_f), 1, mode="auto"
            )

        def initial_values(self, v, values):
            del v
            return self.initial_delay_values(values)

        def advance(self, v, dt, values):
            del dt
            _delayed, updates = self.delayed_state_values("signal", 0.001 * v, values)
            return updates

        def i(self, v):
            values = {
                "signal_delay_buffer": self.signal_delay_buffer,
                "signal_delay_ptr": self.signal_delay_ptr,
            }
            delayed, _updates = self.delayed_state_values("signal", 0.001 * v, values)
            return self.g * v + delayed

        def i_with_conductance(self, v):
            return self.i(v), self.g.expand_as(v)

    integrator = dn.df() if variant == "df" else dn.dfh(conv=True)
    population = dn.Unmyelinated(
        [2.0, 2.5],
        L=30.0,
        dx=10.0,
        integrator=integrator,
        v_init=-63.0,
        dtype=torch.float64,
    )
    population.insert(AutoDelayedCurrent)
    population.train(training)
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


@pytest.mark.parametrize("variant", ["df", "dfh-conv"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_dufort_current_delay_auto_policy_follows_runtime_grad_mode(
    variant, training, grad_enabled
):
    population = _auto_current_delay_population(variant, training)
    native = copy.deepcopy(population)
    # Lowering context does not fix the runtime policy of an auto delay. Current
    # evaluation is repeatable: it reads the pure delay helper's result while
    # accepted advances alone install history updates.
    with torch.set_grad_enabled(not grad_enabled):
        functional, tensors = dn.func.make_functional(population, dt=DT)
    name = next(name for name in tensors.parameters if name.endswith(".g_param"))
    target = tensors.parameters[name].detach().clone().requires_grad_()
    parameters = {**tensors.parameters, name: target}
    native_target = dict(native.named_parameters())[name].requires_grad_()
    with torch.set_grad_enabled(grad_enabled):
        native.initialize()
        prepared = functional.prepare(parameters, tensors.constants)
        state = tensors.state
        for _ in range(2):
            state, _auxiliary = functional.step(parameters, prepared, state)
            _native_step(native)
            torch.testing.assert_close(state, functional.extract(native).state)
        if grad_enabled:
            actual = torch.autograd.grad(
                state["integrator"]["v"].square().mean(), target
            )[0]
            expected = torch.autograd.grad(native.v.square().mean(), native_target)[0]
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
            assert torch.isfinite(actual).all() and actual.abs().max() > 0
