"""Eager schema validation must not invalidate compiled training kernels."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State

DT = 0.0125


class _AssignedState(State):
    State.STATE("a")
    State.ASSIGNED("rate")
    State.DERIVATIVE("a' = rate")

    def state_defaults(self, v, values):
        del values
        return {"a": torch.zeros_like(v)}

    def assigned_values(self, v, values):
        del values
        return {"rate": 0.001 * (v + 65.0)}


class _AdvanceState(State):
    State.STATE("b")
    State.DERIVATIVE("b' = 0.0 * b")

    def state_defaults(self, v, values):
        del values
        return {"b": torch.ones_like(v)}

    def advance(self, v, dt, values):
        return {"b": values["b"] + 0.002 * dt * (v + 65.0)}


class _ValidationMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_AssignedState, _AdvanceState)
    Mechanism.CARRY("memory")
    Mechanism.ASSIGNED("scale")
    Mechanism.GLOBAL(g=1.0e-4)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        del values
        return {"memory": torch.zeros_like(v)}

    def assigned_values(self, v, values):
        del v
        return {"scale": values["a"] + values["b"]}

    def advance(self, v, dt, values):
        del v
        return {"memory": values["memory"] + dt * values["scale"]}

    def i(self, v):
        return self.g * (v + 65.0)


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            v_init=torch.tensor([-64.0, -62.0, -60.0], dtype=torch.float64),
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(_ValidationMechanism)
        model.initialize()
        model.train()
    return model


def _clone(tree):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), tree)


def _evaluate(functional, tensors, chunk, *, checkpointed):
    parameters = _clone(tensors.parameters)
    constants = _clone(tensors.constants)
    state = _clone(tensors.state)
    voltage = state["integrator"]["v"].requires_grad_()
    conductance = parameters[
        "integrator.mech.mechanisms._ValidationMechanism.g_param"
    ].requires_grad_()
    intra = torch.linspace(-1.0e-9, 2.0e-9, 15, dtype=voltage.dtype)
    intra = intra.reshape(5, *voltage.shape).requires_grad_()
    prepared = functional.prepare(parameters, constants)
    bound = partial(chunk or functional.step, parameters, prepared)
    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.longrun
    final, _auxiliary = runner(
        functional,
        bound,
        state,
        5 * DT,
        3,
        dn.func.RolloutInput(intra=intra),
    )
    loss = final["integrator"]["v"].square().mean()
    memory = final["mechanism_buffers"]["_ValidationMechanism"]["memory"]
    loss = loss + memory.square().mean()
    gradients = torch.autograd.grad(loss, (conductance, voltage, intra))
    return _clone(final), loss.detach(), _clone(gradients)


def _assert_close(actual, expected):
    actual_values, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_values, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(
        actual_values, expected_values, strict=True
    ):
        torch.testing.assert_close(
            actual_value, expected_value, rtol=2.0e-10, atol=2.0e-11
        )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_eager_validation_preserves_compiled_training_cache(checkpointed):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    mechanism = functional._transition.population.mech.mechanisms[
        "_ValidationMechanism"
    ]
    cache_flags = (
        (mechanism, "_assigned_schema_validated"),
        (mechanism, "_advance_schema_validated"),
        (mechanism.DE["_AssignedState"], "_assigned_schema_validated"),
        (mechanism.DE["_AdvanceState"], "_advance_schema_validated"),
    )
    # Lowering can eagerly validate Mechanism ASSIGNED values while binding
    # initial workspaces. Exercise pending validation in every supported hook.
    for owner, name in cache_flags:
        setattr(owner, name, False)
    assert all(not getattr(owner, name) for owner, name in cache_flags)
    graph_count = 0
    aot_eager = lookup_backend("aot_eager")

    def counting_backend(graph, example_inputs):
        nonlocal graph_count
        graph_count += 1
        return aot_eager(graph, example_inputs)

    chunk = functional.compile_rollout_chunk(2, backend=counting_backend)
    with torch_compiler_warning_context():
        cold = _evaluate(functional, tensors, chunk, checkpointed=checkpointed)
        cold_graph_count = graph_count
        assert cold_graph_count > 0
        assert all(not getattr(owner, name) for owner, name in cache_flags)

        # Eager validation initializes four Python caches on the same plan.
        # Their values are irrelevant to the compiled numerical transition.
        expected = _evaluate(functional, tensors, None, checkpointed=False)
        assert all(getattr(owner, name) for owner, name in cache_flags)
        _assert_close(cold, expected)
        for gradient in expected[2]:
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0

        for _ in range(2):
            actual = _evaluate(functional, tensors, chunk, checkpointed=checkpointed)
            _assert_close(actual, expected)
            assert graph_count == cold_graph_count
