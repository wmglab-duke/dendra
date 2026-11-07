"""Implicit authored broadcasting through functional initialization and steps."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import hh

DT = 0.01
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
GATES = ("m", "h", "n")
RTOL, ATOL = 2.0e-10, 2.0e-11


class _OpenState(State):
    State.STATE("q")
    State.DERIVATIVE("q' = 0")

    def state_defaults(self, v, values):
        del values
        return {"q": torch.full_like(v, 0.25)}


class _ImplicitAffine(Mechanism):
    """No HH inheritance, rank padding, or explicitly expanded coefficients."""

    Mechanism.STATE_BUNDLE(_OpenState)
    Mechanism.GLOBAL(g=0.01, e=-67.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        conductance = self.g * self.q
        return conductance, conductance * self.e


def _functional(mechanism=hh, *, dtype=torch.float64):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, MEMBRANE_ASSEMBLY="auto"):
        model = dn.SingleCompartment(
            N=2,
            C=3,
            v_init=torch.tensor([-68.0, -62.0, -55.0], dtype=dtype),
            dtype=dtype,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(mechanism)
        model.initialize()
        model.train()
        functional, tensors = dn.func.make_functional(model, dt=DT)
    name = next(iter(model.mech.mechanisms))
    return model, functional, tensors, name


def _parameter_key(tensors, field):
    matches = [key for key in tensors.parameters if key.endswith(f".{field}_param")]
    assert len(matches) == 1
    return matches[0]


def _voltage_and_gates(state, name, gates=GATES):
    return torch.stack(
        (state["integrator"]["v"], *(state["mechanisms"][name][gate] for gate in gates))
    )


def _snapshot(model, tensors):
    values = [*model.parameters(), *model.buffers()]
    values.extend(
        value
        for value in torch.utils._pytree.tree_leaves(tensors)
        if isinstance(value, torch.Tensor)
    )
    return [(value, value._version, value.detach().clone()) for value in values]


def _assert_unchanged(snapshot):
    for value, version, expected in snapshot:
        assert value._version == version
        torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def _assert_scalar_schema(model, tensors, mechanism_name, fields):
    mechanism = model.mech.mechanisms[mechanism_name]
    for field in fields:
        assert tensors.parameters[_parameter_key(tensors, field)].shape == ()
        assert getattr(mechanism, field).shape == ()


def _parameter_rollout(functional, tensors, name, field, *, gates=GATES):
    key = _parameter_key(tensors, field)

    def lane(value):
        parameters = dict(tensors.parameters)
        parameters[key] = value
        initialized = functional.initialize(
            parameters, tensors.constants, tensors.initialization
        )
        final, _auxiliary = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            steps=2,
        )
        return _voltage_and_gates(final, name, gates)

    return lane


def _check_scalar_lanes(lane, base, batch, shape):
    rtol, atol = (2.0e-5, 2.0e-6) if base.dtype == torch.float32 else (RTOL, ATOL)
    lanes = base.detach() * torch.linspace(0.95, 1.05, batch, dtype=base.dtype)
    actual = torch.vmap(lane)(lanes)
    gradient = torch.func.grad(lambda value: lane(value).square().mean())
    actual_gradient = torch.vmap(gradient)(lanes)
    assert actual.shape == (batch, *shape)
    assert actual_gradient.shape == (batch,)
    if batch:
        expected = torch.stack([lane(value) for value in lanes])
        expected_gradient = torch.stack([gradient(value) for value in lanes])
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=rtol, atol=atol
        )
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient) == batch


@pytest.mark.parametrize("field", CHANNELS)
@pytest.mark.parametrize("batch", (0, 1, 4))
def test_scalar_channel_vmap_initialization_and_rollout_match_stacked_lanes(
    field, batch
):
    model, functional, tensors, name = _functional()
    _assert_scalar_schema(model, tensors, name, CHANNELS)
    snapshot = _snapshot(model, tensors)
    lane = _parameter_rollout(functional, tensors, name, field)
    base = tensors.parameters[_parameter_key(tensors, field)]

    _check_scalar_lanes(lane, base, batch, (4, *model.shape))

    _assert_scalar_schema(model, tensors, name, CHANNELS)
    _assert_unchanged(snapshot)


def test_float32_scalar_parameter_schema_and_empty_vmap_smoke():
    model, functional, tensors, name = _functional(dtype=torch.float32)
    snapshot = _snapshot(model, tensors)
    lane = _parameter_rollout(functional, tensors, name, "gnabar")
    base = tensors.parameters[_parameter_key(tensors, "gnabar")]
    for batch in (0, 4):
        _check_scalar_lanes(lane, base, batch, (4, *model.shape))
    _assert_scalar_schema(model, tensors, name, CHANNELS)
    _assert_unchanged(snapshot)


@pytest.mark.parametrize("batch", (0, 1, 4))
def test_full_shaped_mapped_gates_and_their_gradients_match_stacked_lanes(batch):
    model, functional, tensors, name = _functional()
    snapshot = _snapshot(model, tensors)

    def lane(m, h, n):
        state = dict(tensors.state)
        mechanisms = dict(state["mechanisms"])
        mechanisms[name] = dict(mechanisms[name], m=m, h=h, n=n)
        state["mechanisms"] = mechanisms
        final, _auxiliary = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )
        return _voltage_and_gates(final, name)

    scales = torch.linspace(0.95, 1.05, batch, dtype=model.dtype()).reshape(batch, 1, 1)
    mapped = tuple(
        tensors.state["mechanisms"][name][gate].unsqueeze(0) * scales for gate in GATES
    )
    actual = torch.vmap(lane)(*mapped)
    gradient = torch.func.grad(
        lambda *values: lane(*values).square().mean(), argnums=(0, 1, 2)
    )
    actual_gradients = torch.vmap(gradient)(*mapped)
    assert actual.shape == (batch, 4, *model.shape)
    for derivative in actual_gradients:
        assert derivative.shape == (batch, *model.shape)
    if batch:
        expected = torch.stack(
            [lane(*(value[index] for value in mapped)) for index in range(batch)]
        )
        expected_gradients = [
            gradient(*(value[index] for value in mapped)) for index in range(batch)
        ]
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        for index, derivative in enumerate(actual_gradients):
            expected_derivative = torch.stack(
                [value[index] for value in expected_gradients]
            )
            torch.testing.assert_close(
                derivative, expected_derivative, rtol=RTOL, atol=ATOL
            )
            assert torch.isfinite(derivative).all()
            assert torch.count_nonzero(derivative) > 0
    _assert_scalar_schema(model, tensors, name, CHANNELS)
    _assert_unchanged(snapshot)


@pytest.mark.parametrize("batch", (0, 4))
def test_implicit_affine_hook_support_is_generic_not_hh_specific(batch):
    model, functional, tensors, name = _functional(_ImplicitAffine)
    snapshot = _snapshot(model, tensors)
    for field in ("g", "e"):
        lane = _parameter_rollout(functional, tensors, name, field, gates=("q",))
        base = tensors.parameters[_parameter_key(tensors, field)]
        _check_scalar_lanes(lane, base, batch, (2, *model.shape))
    _assert_scalar_schema(model, tensors, name, ("g", "e"))
    _assert_unchanged(snapshot)


def test_fullgraph_compiled_vmap_forward_and_channel_gradient_smoke():
    model, functional, tensors, name = _functional()
    snapshot = _snapshot(model, tensors)
    lane = _parameter_rollout(functional, tensors, name, "gnabar")
    gradient = torch.func.grad(lambda value: lane(value).square().mean())

    def values_and_gradient(value):
        return lane(value), gradient(value)

    transformed = torch.vmap(values_and_gradient)
    base = tensors.parameters[_parameter_key(tensors, "gnabar")].detach()
    lanes = base * torch.linspace(0.95, 1.05, 4, dtype=model.dtype())
    expected_values, expected_gradient = transformed(lanes)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(
        transformed, backend=backend, fullgraph=True, dynamic=False
    )
    with torch_compiler_warning_context():
        actual_values, actual_gradient = compiled(lanes)
    assert graphs
    torch.testing.assert_close(actual_values, expected_values, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=RTOL, atol=ATOL)
    _assert_unchanged(snapshot)
