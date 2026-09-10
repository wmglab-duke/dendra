"""Replica-varying passive-end reversals remain live functional parameters."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

DT = 0.01
DTYPE = torch.float64
E_PARAMETER = "integrator.mech.mechanisms.pas.e_param_0"
E_BUFFER = "integrator.mech.mechanisms.pas.e"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=-60.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        ).batch(2)
        reversal = torch.tensor([-70.0, -65.0], dtype=DTYPE)[:, None, None]
        dn.passive_end_nodes_(model, e=reversal)
        model.initialize()
        model.train()
    return model


def _case():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert E_PARAMETER in tensors.parameters
    assert tensors.parameters[E_PARAMETER].shape[0] == 2
    return model, functional, tensors


def _parameters(tensors, reversal):
    parameters = dict(tensors.parameters)
    parameters[E_PARAMETER] = reversal
    return parameters


def _effective_e(functional, tensors, reversal):
    prepared = functional.prepare(_parameters(tensors, reversal), tensors.constants)
    return prepared.values["mechanisms"][E_BUFFER]


def _replica_values(values, tensor):
    return tensor.new_tensor(values).reshape(2, *((1,) * (tensor.ndim - 1)))


def test_passive_end_preparation_preserves_replicas_and_materialization_jvp():
    model, functional, tensors = _case()
    raw = tensors.parameters[E_PARAMETER]
    actual = _effective_e(functional, tensors, raw)
    assert actual.shape[0] == 2
    expected = _replica_values([-70.0, -65.0], actual).expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, model.mech.pas.e, rtol=0, atol=0)

    changed = raw + _replica_values([1.5, -2.5], raw)
    torch.testing.assert_close(
        _effective_e(functional, tensors, changed),
        expected + _replica_values([1.5, -2.5], expected),
        rtol=0,
        atol=0,
    )
    # Changing only one replica must leave every other replica unchanged,
    # including while differentiating pure parameter materialization.
    tangent = _replica_values([1.0, 0.0], raw).expand_as(raw)
    _, actual_tangent = torch.func.jvp(
        lambda value: _effective_e(functional, tensors, value),
        (raw,),
        (tangent,),
    )
    expected_tangent = _replica_values([1.0, 0.0], actual).expand_as(actual)
    torch.testing.assert_close(actual_tangent, expected_tangent, rtol=0, atol=0)
    # A functional parameter replacement must not edit the imperative source.
    torch.testing.assert_close(model.mech.pas.e, expected, rtol=0, atol=0)


def test_passive_end_step_matches_imperative_gradients_and_replica_jacobian():
    _source, functional, tensors = _case()
    imperative = _model()
    raw = tensors.parameters[E_PARAMETER].detach().clone().requires_grad_()

    def voltage(reversal):
        parameters = _parameters(tensors, reversal)
        prepared = functional.prepare(parameters, tensors.constants)
        final, _ = functional.step(parameters, prepared, tensors.state)
        return final["integrator"]["v"]

    actual = voltage(raw)
    dt = torch.as_tensor(DT, device=imperative.device(), dtype=imperative.dtype())
    imperative.integrator._initialize(
        imperative, dt, force=True, compile_scope="population"
    )
    imperative.integrator.step(imperative, dt, None, None)
    torch.testing.assert_close(actual, imperative.v, rtol=2e-12, atol=2e-12)
    assert not torch.equal(actual[0], actual[1])

    weights = torch.linspace(0.5, 1.5, actual.numel(), dtype=DTYPE).reshape_as(actual)
    expected_parameter = dict(imperative.named_parameters())[E_PARAMETER]
    actual_gradient = torch.autograd.grad((actual * weights).sum(), raw)[0]
    expected_gradient = torch.autograd.grad(
        (imperative.v * weights).sum(), expected_parameter
    )[0]
    assert torch.isfinite(expected_gradient).all()
    assert torch.count_nonzero(expected_gradient) == expected_gradient.numel()
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-10, atol=2e-12
    )

    jacobian = torch.func.jacrev(voltage)(raw)
    blocks = jacobian.reshape(2, actual[0].numel(), 2, raw[0].numel())
    assert torch.count_nonzero(blocks[0, :, 0, :]) > 0
    assert torch.count_nonzero(blocks[1, :, 1, :]) > 0
    assert torch.count_nonzero(blocks[0, :, 1, :]) == 0
    assert torch.count_nonzero(blocks[1, :, 0, :]) == 0
    contracted = (
        (jacobian.reshape(actual.numel(), raw.numel()) * weights.reshape(-1, 1))
        .sum(0)
        .reshape_as(raw)
    )
    torch.testing.assert_close(contracted, actual_gradient, rtol=2e-10, atol=2e-12)


def test_passive_end_atomic_compilation_preserves_live_reversals_and_gradients():
    _source, functional, tensors = _case()

    def voltage(reversal):
        final, _ = functional.prepare_and_step(
            _parameters(tensors, reversal), tensors.constants, tensors.state
        )
        return final["integrator"]["v"]

    compiled = torch.compile(voltage, backend="aot_eager", fullgraph=True)
    initial = tensors.parameters[E_PARAMETER].detach()
    for replacement in (initial, initial + _replica_values([2.0, -3.0], initial)):
        raw = replacement.clone().requires_grad_()
        expected = voltage(raw)
        expected_gradient = torch.autograd.grad(expected.square().sum(), raw)[0]
        with torch_compiler_warning_context():
            actual = compiled(raw)
            actual_gradient = torch.autograd.grad(actual.square().sum(), raw)[0]
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient) == actual_gradient.numel()
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-10, atol=2e-12
        )
