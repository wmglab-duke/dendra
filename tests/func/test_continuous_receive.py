"""Explicit continuous transactions preserve ordering, carry and gradients."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import ContinuousSynapse, State
from dendra.models.mechanisms._continuous_receive import reduce_continuous_values
from dendra.models.mod import graded_syn, pas

pytestmark = pytest.mark.cpu
DT = 0.05
DTYPE = torch.float64


class _Memory(State):
    State.STATE("a")
    State.CARRY("mark")
    State.DERIVATIVE("a' = 0.0 * a")

    def initial_values(self, v, values):
        return {"mark": torch.zeros_like(v)}


class _Remember(ContinuousSynapse):
    ContinuousSynapse.INPUT("x")
    ContinuousSynapse.INPUT("other", keep_old=False)
    ContinuousSynapse.CARRY("ticks")
    ContinuousSynapse.STATE_BUNDLE(_Memory)
    ContinuousSynapse.GLOBAL(scale=1.2)

    def continuous_receive_values(
        self, value, dt, values, *, input, reduce, mask, received
    ):
        return {
            "states": {"a": values["a"] + value * self.scale * dt},
            "carry": {
                input: reduce_continuous_values(
                    values[input], value, received, mask, reduce
                ),
                "ticks": values["ticks"] + 1,
            },
            "state_carry": {
                "_Memory": {
                    "mark": values["state_carry"]["_Memory"]["mark"] + values["x_old"]
                }
            },
        }


class _LegacyReceive(graded_syn):
    def continuous_receive(self, value, con=None, **kwargs):
        self.g_pre = self.g_pre + value


class _LegacyReset(graded_syn):
    def reset_continuous_inputs(self):
        self.g_pre = self.g_pre * 0


class _Mutating(graded_syn):
    def continuous_receive_values(self, value, dt, values, **kwargs):
        value.add_(1)
        return {"carry": {"g_pre": value}}


class _BadOutput(graded_syn):
    def continuous_receive_values(self, value, dt, values, **kwargs):
        return {"carry": {"missing": value}}


class _HiddenResetCount(graded_syn):
    def continuous_receive_values(self, value, dt, values, **kwargs):
        return {"carry": {"g_pre": value + self._continuous_reset_count}}


class _NoInputs(ContinuousSynapse):
    pass


def _model(kind=graded_syn, *, batch=2, regional=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        model = dn.Population(
            N=1, C=3, dtype=DTYPE, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas, g=0.001)
        target = model[:, torch.tensor([2, 0])] if regional else model
        target.insert(kind.rename("analog"))
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
    return model


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, spec = pytree.tree_flatten(actual)
    right, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize(
    "reduction", ("sum", "add", "set", "replace", "last", "min", "max")
)
@pytest.mark.parametrize("regional", (False, True))
def test_ordered_receive_native_parity_and_no_checkpoint_scratch(reduction, regional):
    model = _model(regional=regional)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    frozen = _clone(tensors.state)
    initial = (
        tensors.state["mechanism_buffers"]["analog"]["g_pre"]
        .clone()
        .add_(3)
        .requires_grad_()
    )
    state = _clone(tensors.state)
    state["mechanism_buffers"]["analog"]["g_pre"] = initial
    model.mech.analog.g_pre = initial
    state, scratch = functional.reset_continuous_inputs(state)
    model.mech.analog.reset_continuous_inputs()
    shape = initial.shape
    first = torch.zeros(shape, dtype=DTYPE, requires_grad=True)
    second = torch.full(shape, -2.0, dtype=DTYPE, requires_grad=True)
    mask = torch.arange(shape[-1]) % 2 == 0
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    for value, active in ((first, mask), (second, None)):
        before = _clone(scratch)
        next_state, next_scratch = functional.continuous_receive(
            tensors.parameters,
            prepared,
            state,
            scratch,
            target="analog",
            payload=value,
            reduce=reduction,
            mask=active,
        )
        _close(scratch, before)
        state, scratch = next_state, next_scratch
        model.mech.analog.continuous_receive(value, reduce=reduction, mask=active)
        torch.testing.assert_close(
            state["mechanism_buffers"]["analog"]["g_pre"], model.mech.analog.g_pre
        )
        torch.testing.assert_close(
            state["mechanism_buffers"]["analog"]["g_pre_old"], initial
        )
    assert set(state) == set(tensors.state)
    assert set(state["mechanism_buffers"]["analog"]) == {"g_pre", "g_pre_old"}
    actual_loss = state["mechanism_buffers"]["analog"]["g_pre"].square().sum()
    expected_loss = model.mech.analog.g_pre.square().sum()
    actual = torch.autograd.grad(actual_loss, (first, second), retain_graph=True)
    expected = torch.autograd.grad(expected_loss, (first, second))
    _close(actual, expected)
    old_gradient = torch.autograd.grad(
        state["mechanism_buffers"]["analog"]["g_pre_old"].sum(), initial
    )[0]
    torch.testing.assert_close(old_gradient, torch.ones_like(initial))
    _close(tensors.state, frozen)
    next_state, next_scratch = functional.reset_continuous_inputs(state)
    torch.testing.assert_close(
        next_state["mechanism_buffers"]["analog"]["g_pre_old"],
        state["mechanism_buffers"]["analog"]["g_pre"],
    )
    assert not next_scratch["analog"]["g_pre"].any()


def test_custom_hook_updates_declared_state_carry_and_nested_state_carry():
    model = _model(_Remember, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state = _clone(tensors.state)
    state["mechanism_buffers"]["analog"]["x"] += 2
    state, scratch = functional.reset_continuous_inputs(state)
    payload = torch.ones(1, 3, dtype=DTYPE, requires_grad=True)
    for _ in range(2):
        state, scratch = functional.prepare_and_continuous_receive(
            tensors.parameters,
            tensors.constants,
            state,
            scratch,
            target="analog",
            input="x",
            payload=payload,
        )
    torch.testing.assert_close(
        state["mechanisms"]["analog"]["a"], payload * (2 * DT * 1.2)
    )
    torch.testing.assert_close(
        state["mechanism_buffers"]["analog"]["ticks"], payload * 0 + 2
    )
    torch.testing.assert_close(
        state["state_buffers"]["analog"]["_Memory"]["mark"], payload * 0 + 4
    )
    assert not scratch["analog"]["other"].any()


def test_continuous_receiver_without_inputs_has_noop_reset():
    model = _model(_NoInputs, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state, scratch = functional.reset_continuous_inputs(tensors.state)
    _close(state, tensors.state)
    assert scratch == {"analog": {}}


def test_midstep_continuation_uses_explicit_delivery_scratch():
    model = _model(batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state, scratch = functional.reset_continuous_inputs(tensors.state)
    state, scratch = functional.prepare_and_continuous_receive(
        tensors.parameters,
        tensors.constants,
        state,
        scratch,
        target="analog",
        payload=torch.zeros(1, 3, dtype=DTYPE),
        reduce="min",
    )
    checkpoint = (_clone(state), _clone(scratch))
    options = dict(target="analog", payload=torch.ones(1, 3, dtype=DTYPE), reduce="min")
    expected = functional.prepare_and_continuous_receive(
        tensors.parameters,
        tensors.constants,
        state,
        scratch,
        **options,
    )
    actual = functional.prepare_and_continuous_receive(
        tensors.parameters,
        tensors.constants,
        *checkpoint,
        **options,
    )
    _close(actual, expected)
    assert not torch.count_nonzero(actual[0]["mechanism_buffers"]["analog"]["g_pre"])


@pytest.mark.parametrize("kind", (_LegacyReceive, _LegacyReset))
def test_legacy_imperative_overrides_can_lower_for_local_dynamics_but_not_receive(kind):
    model = _model(kind, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    functional.prepare_and_step(tensors.parameters, tensors.constants, tensors.state)
    with pytest.raises(
        dn.func.FunctionalizationError, match="canonical continuous adapters"
    ):
        functional.reset_continuous_inputs(tensors.state)
    with pytest.raises(
        dn.func.FunctionalizationError, match="canonical continuous adapters"
    ):
        functional.prepare_and_continuous_receive(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            {},
            target="analog",
            payload=torch.ones(1, 3, dtype=DTYPE),
        )


@pytest.mark.parametrize("kind", (_Mutating, _BadOutput, _HiddenResetCount))
def test_invalid_pure_receivers_reject_lowering_without_touching_source(kind):
    model = _model(kind, batch=None)
    original = model.mech.analog.g_pre.clone()
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional(model, dt=DT)
    torch.testing.assert_close(model.mech.analog.g_pre, original)


@pytest.mark.parametrize("invalid", ("payload", "mask", "scratch", "input", "reduce"))
def test_invalid_receive_preserves_caller_state_and_scratch(invalid):
    model = _model(batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state, scratch = functional.reset_continuous_inputs(tensors.state)
    original, original_scratch = _clone(state), _clone(scratch)
    payload = torch.ones(1, 3, dtype=DTYPE)
    options = {"input": None, "reduce": "sum", "mask": None}
    supplied_scratch = scratch
    if invalid == "payload":
        payload = payload.float()
    elif invalid == "mask":
        options["mask"] = torch.ones_like(payload)
    elif invalid == "scratch":
        supplied_scratch = {"analog": {"g_pre": payload}}
    elif invalid == "input":
        options["input"] = "missing"
    else:
        options["reduce"] = "median"
    with pytest.raises((TypeError, ValueError)):
        functional.prepare_and_continuous_receive(
            tensors.parameters,
            tensors.constants,
            state,
            supplied_scratch,
            target="analog",
            payload=payload,
            **options,
        )
    _close(state, original)
    _close(scratch, original_scratch)


def test_replaced_receiver_invalidates_existing_plan(monkeypatch):
    model = _model(batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state, scratch = functional.reset_continuous_inputs(tensors.state)
    monkeypatch.setattr(
        model.mech.analog, "continuous_receive_values", lambda *args, **kwargs: {}
    )
    with pytest.raises(dn.func.FunctionalizationError):
        functional.prepare_and_continuous_receive(
            tensors.parameters,
            tensors.constants,
            state,
            scratch,
            target="analog",
            payload=torch.ones(1, 3, dtype=DTYPE),
        )


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_reset_receive_gradients_and_caller_owned_scratch(backend):
    model = _model(batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def evaluate(parameters, constants, state, first, second):
        state, scratch = functional.reset_continuous_inputs(state)
        for payload in (first, second):
            state, scratch = functional.prepare_and_continuous_receive(
                parameters,
                constants,
                state,
                scratch,
                target="analog",
                payload=payload,
                reduce="min",
            )
        state, _ = functional.prepare_and_step(parameters, constants, state)
        return state, scratch

    first = torch.tensor([[0.0, 2.0, -1.0]], dtype=DTYPE, requires_grad=True)
    second = torch.tensor([[1.0, -2.0, 3.0]], dtype=DTYPE, requires_grad=True)
    with torch_compiler_warning_context():
        options = {"compile_threads": 1} if backend == "inductor" else None
        compiled = torch.compile(
            evaluate, fullgraph=True, backend=backend, options=options
        )
        actual = compiled(
            tensors.parameters, tensors.constants, tensors.state, first, second
        )
    expected = evaluate(
        tensors.parameters, tensors.constants, tensors.state, first, second
    )
    _close(actual, expected)
    actual_loss = (
        actual[0]["integrator"]["v"].sum()
        + actual[0]["mechanism_buffers"]["analog"]["g_pre"].sum()
    )
    expected_loss = (
        expected[0]["integrator"]["v"].sum()
        + expected[0]["mechanism_buffers"]["analog"]["g_pre"].sum()
    )
    _close(
        torch.autograd.grad(actual_loss, (first, second), retain_graph=True),
        torch.autograd.grad(expected_loss, (first, second)),
    )


@pytest.mark.parametrize("lanes", (0, 3))
def test_receive_vmap_maps_payload_without_mapping_checkpoint_state(lanes):
    model = _model(batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, scratch = functional.reset_continuous_inputs(tensors.state)
    payloads = torch.ones(lanes, 1, 3, dtype=DTYPE, requires_grad=True)

    def evaluate(payload):
        return functional.continuous_receive(
            tensors.parameters,
            prepared,
            state,
            scratch,
            target="analog",
            payload=payload,
        )

    result = torch.vmap(evaluate)(payloads)
    assert result[0]["mechanism_buffers"]["analog"]["g_pre"].shape == (lanes, 1, 3)
    gradient = torch.autograd.grad(
        result[0]["mechanism_buffers"]["analog"]["g_pre"].sum(), payloads
    )[0]
    torch.testing.assert_close(gradient, torch.ones_like(payloads))
