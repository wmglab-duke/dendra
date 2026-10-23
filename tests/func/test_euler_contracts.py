"""Capability, source-guard, and transform contracts for scalar Euler paths."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.nn.utils import parametrize

import dendra as dn
from dendra.func._population import _resolve_functional_integrator_spec
from dendra.models.integrators.explicit import SymmetricConv1D, _euler, _eulerv1
from dendra.models.mod import pas

DT = 0.0001
WEIGHT = "integrator.ssd.weight"
GAIN = "integrator.mech.mechanisms.pas.g_param"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        yield


def _model(factory=dn.euler, *, dtype=torch.float64, training=True):
    population = dn.Unmyelinated(
        [2.0],
        L=30.0,
        dx=10.0,
        integrator=(
            partial(factory, imem=False)
            if isinstance(factory, type) and issubclass(factory, _euler)
            else factory(imem=False)
        ),
        dtype=dtype,
    )
    population.insert(pas, g=0.001, e=-70.0)
    population.initialize()
    population.train(training)
    population.v = torch.linspace(
        -66.0, -62.0, population.v.numel(), dtype=dtype
    ).reshape(population.shape)
    if hasattr(population.integrator, "ssd"):
        population.integrator.ssd.weight.requires_grad_(True)
    return population


def _voltage(functional, tensors, *, weight=None, voltage=None, gain=None):
    parameters = {
        **tensors.parameters,
        **({WEIGHT: weight} if weight is not None else {}),
        **({GAIN: gain} if gain is not None else {}),
    }
    state = tensors.state
    if voltage is not None:
        state = {**state, "integrator": {**state["integrator"], "v": voltage}}
    prepared = functional.prepare(parameters, tensors.constants)
    next_state, _ = functional.step(parameters, prepared, state)
    return next_state["integrator"]["v"]


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize(
    ("factory", "implementation"),
    [(dn.euler, _euler), (dn.rk1, _euler), (dn.eulerv1, _eulerv1)],
)
def test_canonical_euler_families_support_optional_purity_diagnostics(
    factory, implementation, diagnostics
):
    population = _model(factory)
    spec = _resolve_functional_integrator_spec(
        population.integrator, "scalar_explicit_path"
    )
    assert spec is not None and spec.implementation is implementation
    with dn.ctx(CHECK_PURITY=diagnostics):
        functional, tensors = dn.func.make_functional(population, dt=DT)
        actual = _voltage(functional, tensors)
    assert actual.dtype == torch.float64
    assert torch.isfinite(actual).all()


class _CustomEuler(_euler):
    pass


@pytest.mark.parametrize("factory", [dn.rk2, dn.rk4, dn.df, dn.dfh, _CustomEuler])
def test_euler_admission_does_not_authorize_other_explicit_integrators(factory):
    population = _model(factory)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical Euler"):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("hook", ["FRK", "_explicit_slope", "_functional_step"])
def test_euler_rejects_instance_numerical_hook_overrides(hook):
    population = _model()
    setattr(population.integrator, hook, lambda *args, **kwargs: None)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical Euler"):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("hook", ["FRK", "_explicit_slope", "_functional_step"])
def test_euler_rejects_numerical_class_monkeypatch_before_lowering(monkeypatch, hook):
    population = _model()
    original = getattr(_euler, hook)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(_euler, hook, replacement)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical Euler"):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("hook", ["forward", "_conv_forward"])
def test_euler_rejects_stencil_class_monkeypatch_before_lowering(monkeypatch, hook):
    population = _model()
    monkeypatch.setattr(SymmetricConv1D, hook, lambda *args, **kwargs: None)
    with pytest.raises(
        dn.func.FunctionalizationError, match="canonical SymmetricConv1D"
    ):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("padding", [[0, 0], [1, 0], [1, 1, 1]])
def test_euler_rejects_inconsistent_stencil_padding_before_lowering(padding):
    population = _model()
    population.integrator.ssd._reversed_padding_repeated_twice = padding
    with pytest.raises(dn.func.FunctionalizationError, match="reflection stencil"):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize(
    "mutation", ["stride", "padding_mode", "reversed_padding", "forward", "child"]
)
def test_euler_rejects_stencil_structure_changes_after_lowering(mutation):
    population = _model()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    stencil = population.integrator.ssd
    if mutation == "stride":
        stencil.stride = (2,)
    elif mutation == "padding_mode":
        stencil.padding_mode = "replicate"
    elif mutation == "reversed_padding":
        stencil._reversed_padding_repeated_twice[0] += 1
    elif mutation == "forward":
        stencil.forward = lambda value: value[:, :1]
    else:
        population.integrator.ssd = torch.nn.Identity()
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("training", [False, True])
def test_euler_raw_stencil_updates_preserve_dtype_and_training_symmetry(
    dtype, training
):
    source = _model(dtype=dtype, training=training)
    native = _model(dtype=dtype, training=training)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    weight = torch.tensor([[[0.7, -2.0, 1.3], [1.2, -2.0, 0.8]]], dtype=dtype)
    native.integrator.ssd.weight = torch.nn.Parameter(weight.clone())
    native.integrator._initialize(native, DT, force=True)
    native.integrator.step(native, DT)
    actual = _voltage(functional, tensors, weight=weight)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, native.v, rtol=0.0, atol=0.0)
    assert actual.dtype == tensors.state["integrator"]["v"].dtype


class _StencilGain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float64))

    def forward(self, weight):
        return weight * self.gain


def test_euler_materializes_parametrized_stencil_and_preserves_both_raw_gradients():
    source = _model()
    parametrize.register_parametrization(
        source.integrator.ssd, "weight", _StencilGain()
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    raw_name = "integrator.ssd.parametrizations.weight.original"
    gain_name = "integrator.ssd.parametrizations.weight.0.gain"
    assert raw_name in tensors.parameters and gain_name in tensors.parameters
    raw = tensors.parameters[raw_name].detach().clone().requires_grad_()
    gain = tensors.parameters[gain_name].detach().clone().requires_grad_()
    parameters = {**tensors.parameters, raw_name: raw, gain_name: gain}
    prepared = functional.prepare(parameters, tensors.constants)
    state, _ = functional.step(parameters, prepared, tensors.state)
    actual_gradients = torch.autograd.grad(
        state["integrator"]["v"].square().sum(), (raw, gain)
    )
    for gradient in actual_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0


@pytest.mark.parametrize("lanes", [0, 3])
@pytest.mark.parametrize("mapped", ["voltage", "weight", "gain"])
def test_euler_supports_voltage_and_parameter_only_vmap_including_empty_lanes(
    lanes, mapped
):
    source = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    base = {
        "voltage": tensors.state["integrator"]["v"],
        "weight": tensors.parameters[WEIGHT],
        "gain": tensors.parameters[GAIN],
    }[mapped]
    values = base.detach().expand((lanes,) + base.shape).clone().requires_grad_()

    def evaluate(value):
        return _voltage(functional, tensors, **{mapped: value})

    actual = torch.vmap(evaluate)(values)
    assert actual.shape == (lanes, *source.shape)
    if lanes:
        expected = torch.stack(tuple(evaluate(value) for value in values))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    gradients = torch.autograd.grad(actual.square().sum(), values)[0]
    assert gradients.shape == values.shape
    assert torch.isfinite(gradients).all()


def test_euler_jvp_and_jacrev_match_centered_differences_for_voltage_and_stencil():
    source = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    voltage = tensors.state["integrator"]["v"].detach()
    weight = tensors.parameters[WEIGHT].detach()
    voltage_direction = torch.linspace(-0.2, 0.3, voltage.numel()).reshape_as(voltage)
    weight_direction = torch.linspace(-0.03, 0.05, weight.numel()).reshape_as(weight)

    def evaluate(v, w):
        return _voltage(functional, tensors, voltage=v, weight=w)

    actual, tangent = torch.func.jvp(
        evaluate, (voltage, weight), (voltage_direction, weight_direction)
    )
    torch.testing.assert_close(actual, evaluate(voltage, weight), rtol=0.0, atol=0.0)
    epsilon = 1.0e-4
    finite_difference = (
        evaluate(
            voltage + epsilon * voltage_direction, weight + epsilon * weight_direction
        )
        - evaluate(
            voltage - epsilon * voltage_direction, weight - epsilon * weight_direction
        )
    ) / (2 * epsilon)
    torch.testing.assert_close(tangent, finite_difference, rtol=2e-8, atol=2e-9)
    jac_v, jac_w = torch.func.jacrev(evaluate, argnums=(0, 1))(voltage, weight)
    expected_tangent = (
        jac_v.reshape(voltage.numel(), -1) @ voltage_direction.flatten()
        + jac_w.reshape(voltage.numel(), -1) @ weight_direction.flatten()
    ).reshape_as(voltage)
    torch.testing.assert_close(tangent, expected_tangent, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("lanes", [0, 2])
def test_euler_stencil_vmap_compiles_with_empty_or_nonempty_weight_batches(lanes):
    source = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    base = tensors.parameters[WEIGHT].detach()
    values = base.expand((lanes,) + base.shape).clone().requires_grad_()

    def evaluate(weight):
        state, _ = functional.prepare_and_step(
            {**tensors.parameters, WEIGHT: weight}, tensors.constants, tensors.state
        )
        return state["integrator"]["v"]

    eager = torch.vmap(evaluate)(values)
    compiled_values = values.detach().clone().requires_grad_()
    compiled = torch.compile(torch.vmap(evaluate), backend="aot_eager", fullgraph=True)(
        compiled_values
    )
    torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)
    eager_gradient = torch.autograd.grad(eager.square().sum(), values)[0]
    compiled_gradient = torch.autograd.grad(compiled.square().sum(), compiled_values)[0]
    torch.testing.assert_close(
        compiled_gradient, eager_gradient, rtol=2e-13, atol=2e-13
    )


def test_euler_stencil_vmap_preserves_nested_empty_weight_lanes():
    source = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    base = tensors.parameters[WEIGHT].detach()
    weights = base.expand((2, 0) + base.shape).clone().requires_grad_()

    def evaluate(weight):
        return _voltage(functional, tensors, weight=weight)

    actual = torch.vmap(torch.vmap(evaluate))(weights)
    assert actual.shape == (2, 0, *source.shape)
    gradient = torch.autograd.grad(actual.sum(), weights)[0]
    assert gradient.shape == weights.shape
