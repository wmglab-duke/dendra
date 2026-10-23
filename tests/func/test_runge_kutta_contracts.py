"""Audited RK stage kernels retain explicit tensor-transform contracts."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils import parametrize

import dendra as dn
from dendra.func._population import _resolve_functional_integrator_spec
from dendra.models.integrators.explicit import _rk2, _rk4
from tests.func.test_euler_contracts import (
    DT,
    GAIN,
    WEIGHT,
    _model,
    _StencilGain,
    _voltage,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        yield


@pytest.mark.parametrize("factory,implementation", [(dn.rk2, _rk2), (dn.rk4, _rk4)])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_runge_kutta_resolves_its_own_kernel_with_optional_purity_diagnostics(
    factory, implementation, diagnostics
):
    source = _model(factory)
    spec = _resolve_functional_integrator_spec(
        source.integrator, "scalar_explicit_path"
    )
    assert spec is not None and spec.implementation is implementation
    with dn.ctx(CHECK_PURITY=diagnostics):
        functional, tensors = dn.func.make_functional(source, dt=DT)
        voltage = _voltage(functional, tensors)
    assert torch.isfinite(voltage).all()


class _CustomRK2(_rk2):
    pass


class _CustomRK4(_rk4):
    pass


@pytest.mark.parametrize("factory", [_CustomRK2, _CustomRK4])
def test_runge_kutta_does_not_admit_unaudited_subclasses(factory):
    with pytest.raises(dn.func.FunctionalizationError, match="canonical Euler"):
        dn.func.make_functional(_model(factory), dt=DT)


@pytest.mark.parametrize("factory,implementation", [(dn.rk2, _rk2), (dn.rk4, _rk4)])
@pytest.mark.parametrize("hook", ["_step", "_explicit_slope", "_functional_step"])
def test_runge_kutta_rejects_numerical_class_overrides_before_lowering(
    factory, implementation, hook, monkeypatch
):
    source = _model(factory)
    original = getattr(implementation, hook)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, hook, replacement)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical Euler"):
        dn.func.make_functional(source, dt=DT)


@pytest.mark.parametrize("placement", ["class", "instance"])
def test_rk4_rejects_current_quadrature_overrides(placement, monkeypatch):
    source = _model(dn.rk4)
    owner = _rk4 if placement == "class" else source.integrator
    original = owner._combine_ion_current_frames

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "_combine_ion_current_frames", replacement)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical Euler"):
        dn.func.make_functional(source, dt=DT)


@pytest.mark.parametrize("factory", [dn.rk2, dn.rk4])
def test_runge_kutta_materializes_parametrized_stencils_and_shared_gradients(factory):
    source = _model(factory)
    parametrize.register_parametrization(
        source.integrator.ssd, "weight", _StencilGain()
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    raw = "integrator.ssd.parametrizations.weight.original"
    gain = "integrator.ssd.parametrizations.weight.0.gain"
    parameters = {
        **tensors.parameters,
        raw: tensors.parameters[raw].detach().clone().requires_grad_(),
        gain: tensors.parameters[gain].detach().clone().requires_grad_(),
    }
    prepared = functional.prepare(parameters, tensors.constants)
    state, _ = functional.step(parameters, prepared, tensors.state)
    gradients = torch.autograd.grad(
        state["integrator"]["v"].square().sum(), (parameters[raw], parameters[gain])
    )
    assert all(
        torch.isfinite(value).all() and value.abs().max() > 0 for value in gradients
    )


@pytest.mark.parametrize("factory", [dn.rk2, dn.rk4])
@pytest.mark.parametrize("lanes", [0, 3])
@pytest.mark.parametrize("mapped", ["voltage", "weight", "gain"])
def test_runge_kutta_vmap_preserves_state_and_parameter_only_batches(
    factory, lanes, mapped
):
    source = _model(factory)
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
        torch.testing.assert_close(
            actual, torch.stack([evaluate(value) for value in values])
        )
    gradient = torch.autograd.grad(actual.square().sum(), values)[0]
    assert gradient.shape == values.shape and torch.isfinite(gradient).all()


@pytest.mark.parametrize("factory", [dn.rk2, dn.rk4])
def test_runge_kutta_jvp_and_jacrev_agree_with_stage_finite_differences(factory):
    source = _model(factory)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    voltage = tensors.state["integrator"]["v"].detach()
    weight = tensors.parameters[WEIGHT].detach()
    direction_v = torch.linspace(-0.2, 0.3, voltage.numel()).reshape_as(voltage)
    direction_w = torch.linspace(-0.03, 0.05, weight.numel()).reshape_as(weight)

    def evaluate(v, w):
        return _voltage(functional, tensors, voltage=v, weight=w)

    _, tangent = torch.func.jvp(evaluate, (voltage, weight), (direction_v, direction_w))
    epsilon = 1.0e-4
    difference = (
        evaluate(voltage + epsilon * direction_v, weight + epsilon * direction_w)
        - evaluate(voltage - epsilon * direction_v, weight - epsilon * direction_w)
    ) / (2 * epsilon)
    torch.testing.assert_close(tangent, difference, rtol=2e-8, atol=2e-9)
    jac_v, jac_w = torch.func.jacrev(evaluate, argnums=(0, 1))(voltage, weight)
    expected = (
        jac_v.reshape(voltage.numel(), -1) @ direction_v.flatten()
        + jac_w.reshape(voltage.numel(), -1) @ direction_w.flatten()
    ).reshape_as(voltage)
    torch.testing.assert_close(tangent, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("factory", [dn.rk2, dn.rk4])
@pytest.mark.parametrize("lanes", [0, 2])
def test_runge_kutta_stencil_vmap_compiles_and_preserves_backward(factory, lanes):
    source = _model(factory)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    base = tensors.parameters[WEIGHT].detach()
    weights = base.expand((lanes,) + base.shape).clone().requires_grad_()

    def evaluate(weight):
        state, _ = functional.prepare_and_step(
            {**tensors.parameters, WEIGHT: weight}, tensors.constants, tensors.state
        )
        return state["integrator"]["v"]

    eager = torch.vmap(evaluate)(weights)
    compiled_weights = weights.detach().clone().requires_grad_()
    compiled = torch.compile(torch.vmap(evaluate), backend="aot_eager", fullgraph=True)(
        compiled_weights
    )
    torch.testing.assert_close(compiled, eager)
    eager_grad = torch.autograd.grad(eager.square().sum(), weights)[0]
    compiled_grad = torch.autograd.grad(compiled.square().sum(), compiled_weights)[0]
    torch.testing.assert_close(compiled_grad, eager_grad, rtol=2e-13, atol=2e-13)
