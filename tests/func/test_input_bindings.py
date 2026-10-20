"""Declared same-Population inputs retain phase timing, transforms, and carry."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State, VoltageProcess
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import _live_state

DT = 0.05
STEPS = 4
pytestmark = pytest.mark.cpu


class SourceValues(State):
    State.STATE("drive")
    State.GLOBAL(rate=0.7, baseline=0.3)
    State.DERIVATIVE("drive' = 0.0 * drive")

    def state_defaults(self, v, values):
        del values
        return {"drive": self.baseline + 0.1 * (v + 65)}

    def advance(self, v, dt, values):
        del v
        return {"drive": values["drive"] + dt * self.rate}


class _Source(Mechanism):
    Mechanism.STATE_BUNDLE(SourceValues)
    Mechanism.GLOBAL(gain=1.7)
    Mechanism.CARRY("bias")

    def initial_values(self, v, values):
        del values
        return {"bias": torch.ones_like(v) * 0.4}

    def advance(self, v, dt, values):
        del v
        return {"bias": values["bias"] + dt * 0.2}


class ReceiverValues(State):
    State.STATE("owned", "seen")
    State.ASSIGNED("rhs")
    State.INPUT("drive", "bias", "gain")
    State.INPUT("unbound", default=0.2)
    State.DERIVATIVE("owned' = 0.0 * owned", "seen' = 0.0 * seen")

    def state_defaults(self, v, values):
        del values
        return {"owned": v, "seen": torch.zeros_like(v)}

    def assigned_values(self, v, values):
        del v
        return {
            "rhs": values["gain"] * values["drive"]
            + values["bias"]
            + values["unbound"]
            + 0.1 * values["owned"]
        }

    def advance(self, v, dt, values):
        assigned = self._derive_assigned_values(v, values)
        return {
            "owned": values["owned"] + dt * assigned["rhs"],
            "seen": values["seen"] + dt * values["drive"],
        }


class _Receiver(VoltageProcess):
    VoltageProcess.STATE_BUNDLE(ReceiverValues)
    VoltageProcess.GLOBAL(threshold=-62.0, rest=-66.0)

    def voltage_values(self, v, dt, values):
        del v, dt
        owned = torch.where(
            values["owned"] >= self.threshold, self.rest, values["owned"]
        )
        return owned, {"owned": owned}


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _key(parameters, suffix):
    keys = [name for name in parameters if name.endswith(suffix)]
    assert len(keys) == 1, (suffix, keys)
    return keys[0]


def _model(*, layout="dense", batch=2, bind=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        model = dn.Population(
            N=1, C=4, dtype=torch.float64, integrator=dn.scnv(imem=False)
        )
        source = model[:, torch.tensor([3, 0, 2])] if layout == "reordered" else model
        receiver = model[:, torch.tensor([2, 0])] if layout != "dense" else model
        source.insert(_Source.rename("source"))
        receiver.insert(_Receiver.rename("receiver"))
        if bind:
            for name in ("drive", "bias", "gain"):
                model.bind_input(f"receiver.DE.ReceiverValues.{name}", f"source.{name}")
        if batch is not None:
            model.batch(batch)
        model.set_v_init(
            torch.linspace(-65.0, -61.0, model.v.numel()).reshape(model.shape).double()
        )
        model.train()
        model.initialize()
    return model


def _observe(state):
    return {
        "v": state["integrator"]["v"],
        "source.drive": state["mechanisms"]["source"]["drive"],
        "source.bias": state["mechanism_buffers"]["source"]["bias"],
        "receiver.owned": state["mechanisms"]["receiver"]["owned"],
        "receiver.seen": state["mechanisms"]["receiver"]["seen"],
    }


def _reference(model, parameters, constants, state, steps):
    """Independent ordered recurrence; provider advances before receiver visit."""
    values = _observe(state)
    frames = [values]
    gain = parameters[_key(parameters, ".source.gain_param")]
    rate = parameters[_key(parameters, ".source.DE.SourceValues.rate_param")]
    threshold = parameters[_key(parameters, ".receiver.threshold_param")]
    rest = parameters[_key(parameters, ".receiver.rest_param")]
    dt = constants["dt"]
    batch_shape = values["v"].shape[:-2]
    receiver_key = model.mech.receiver.key
    if receiver_key is not None:
        receiver_key = receiver_key.reshape(-1)
    source_key = model.mech.source.key
    if source_key is not None:
        source_key = source_key.reshape(-1)
    for _ in range(steps):
        owned = torch.where(
            values["receiver.owned"] >= threshold, rest, values["receiver.owned"]
        )
        voltage = (
            owned
            if receiver_key is None
            else values["v"]
            .reshape(*batch_shape, -1)
            .index_copy(-1, receiver_key, owned.reshape(*batch_shape, -1))
            .reshape_as(values["v"])
        )
        drive = values["source.drive"] + dt * rate
        bias = values["source.bias"] + dt * 0.2
        local_drive, local_bias = drive, bias
        if receiver_key is not None:
            gather = (
                receiver_key
                if source_key is None
                else torch.stack(
                    [
                        (source_key == item).nonzero().reshape(-1)[0]
                        for item in receiver_key
                    ]
                )
            )
            target_shape = owned.shape
            local_drive = (
                drive.reshape(*batch_shape, -1)
                .index_select(-1, gather)
                .reshape(target_shape)
            )
            local_bias = (
                bias.reshape(*batch_shape, -1)
                .index_select(-1, gather)
                .reshape(target_shape)
            )
        values = {
            "v": voltage,
            "source.drive": drive,
            "source.bias": bias,
            "receiver.owned": owned
            + dt * (gain * local_drive + local_bias + 0.2 + 0.1 * owned),
            "receiver.seen": values["receiver.seen"] + dt * local_drive,
        }
        frames.append(values)
    return {name: torch.stack([frame[name] for frame in frames]) for name in frames[0]}


def _runner(
    model,
    *,
    steps=STEPS,
    initialization="fresh",
    checkpoint_every=None,
    backend="eager",
):
    options = {}
    if backend == "inductor":
        options = {
            "fullgraph": True,
            "options": {"cpp_wrapper": True, "compile_threads": 1},
        }
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        initialization=initialization,
        execution="scan",
        inference=False,
        checkpoint_every=checkpoint_every,
        backend=backend,
        callbacks={
            "trace": dn.func.Recorder(
                tuple(_observe(dn.func.make_functional(model, dt=DT)[1].state))
            )
        },
        **options,
    )


@pytest.mark.parametrize("layout", ("dense", "regional", "reordered"))
def test_input_binding_trace_matches_independent_ordered_recurrence(layout):
    model = _model(layout=layout)
    snapshot = _module_snapshot(model)
    runner = _runner(model)
    parameters = _clone(runner.parameters)
    targets = [
        parameters[_key(parameters, suffix)].requires_grad_()
        for suffix in (
            ".source.gain_param",
            ".source.DE.SourceValues.rate_param",
        )
    ]
    state, auxiliary = runner(parameters=parameters)
    trace = auxiliary["callbacks"]["trace"]
    initialized = runner.functional.initialize(
        parameters, runner.constants, runner.initialization_inputs
    )
    reference = _reference(
        model, initialized.parameters, initialized.constants, initialized.state, STEPS
    )
    _assert_tree_close(dict(trace), reference, rtol=2e-9, atol=2e-10)
    loss = (
        trace["receiver.owned"].square().sum() + trace["receiver.seen"].square().sum()
    )
    expected_loss = (
        reference["receiver.owned"].square().sum()
        + reference["receiver.seen"].square().sum()
    )
    actual_gradients = torch.autograd.grad(loss, targets)
    expected_gradients = torch.autograd.grad(expected_loss, targets)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
        assert torch.count_nonzero(actual)
    leaves = {".".join(leaf.public_path) for leaf in runner.functional._state_layout}
    assert not any(name.endswith((".gain", ".unbound", ".rhs")) for name in leaves)
    assert not any(
        "receiver" in name
        and name.endswith(("drive_param", "bias_param", "gain_param", "unbound_param"))
        for name in runner.parameters
    )
    _assert_module_snapshot(model, snapshot)


def test_input_binding_snapshot_continuation_commit_and_fresh_initialization():
    model = _model()
    runner = _runner(model, steps=2, initialization="snapshot")
    snapshot = _module_snapshot(model)
    first, _ = runner()
    first = _clone(first)
    first["mechanisms"]["source"]["drive"] = (
        first["mechanisms"]["source"]["drive"] + 0.8
    ).requires_grad_()
    final, auxiliary = runner(first)
    expected = _reference(model, runner.parameters, runner.constants, first, 2)
    _assert_tree_close(
        dict(auxiliary["callbacks"]["trace"]), expected, rtol=2e-9, atol=2e-10
    )
    assert torch.count_nonzero(
        torch.autograd.grad(
            final["mechanisms"]["receiver"]["owned"].sum(),
            first["mechanisms"]["source"]["drive"],
        )[0]
    )
    _assert_module_snapshot(model, snapshot)
    runner.functional.commit_state_(model, final)
    _assert_tree_close(
        _live_state(runner.functional, model), final, rtol=2e-9, atol=2e-10
    )
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    model.integrator.step(model, dt)
    model.t = model.t + dt
    next_state, _ = runner.functional.prepare_and_step(
        runner.parameters, runner.constants, final
    )
    _assert_tree_close(
        _live_state(runner.functional, model), next_state, rtol=2e-9, atol=2e-10
    )
    fresh = _runner(model, steps=1)
    _, fresh_aux = fresh()
    assert not torch.equal(
        fresh_aux["callbacks"]["trace"]["source.drive"][0],
        final["mechanisms"]["source"]["drive"],
    )


def test_input_bindings_vmap_forward_reverse_ad_and_empty_lanes():
    model = _model(batch=None, layout="reordered")
    snapshot = _module_snapshot(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    key = _key(tensors.parameters, ".source.gain_param")
    output_width = model.mech.receiver.owned.numel()

    def evaluate(gain, initial_voltage):
        initialized = functional.initialize(
            {**tensors.parameters, key: gain},
            tensors.constants,
            tensors.initialization._replace(v_init=initial_voltage),
        )
        state, _ = functional.prepare_and_rollout(
            initialized.parameters, initialized.constants, initialized.state, steps=2
        )
        return state["mechanisms"]["receiver"]["owned"].reshape(output_width)

    inputs = (tensors.parameters[key], tensors.initialization.v_init)
    lanes = tuple(torch.stack((value, value + 0.1)) for value in inputs)
    torch.testing.assert_close(
        torch.vmap(evaluate)(*lanes),
        torch.stack(
            [evaluate(*(value[index] for value in lanes)) for index in range(2)]
        ),
    )
    assert torch.vmap(evaluate)(*(value[:0] for value in lanes)).shape[0] == 0
    for actual, expected in zip(
        torch.func.jacfwd(evaluate, argnums=(0, 1))(*inputs),
        torch.func.jacrev(evaluate, argnums=(0, 1))(*inputs),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize(
    "mutation",
    (
        "binding",
        "handler_binding",
        "plan",
        "inputs",
        "defaults",
        "indices",
        "index_rebind",
    ),
)
def test_input_binding_source_mutation_requires_new_lowering(mutation):
    model = _model(layout="regional")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    if mutation == "binding":
        from dendra.models.mechanisms._inputs import InputBinding

        model._input_bindings = (
            InputBinding("receiver.DE.ReceiverValues.drive", "source.bias"),
            *model._input_bindings[1:],
        )
    elif mutation == "handler_binding":
        model.mech._input_bindings = model.mech._input_bindings[:-1]
    elif mutation == "plan":
        model.mech._input_binding_plan = model.mech._input_binding_plan[:-1]
    elif mutation == "inputs":
        model.mech.receiver.DE["ReceiverValues"]._inputs = ("drive", "bias", "gain")
    elif mutation == "defaults":
        state = model.mech.receiver.DE["ReceiverValues"]
        state._input_defaults = {**state._input_defaults, "unbound": 0.7}
    else:
        name = next(
            name
            for name in model.mech._buffers
            if name.startswith("_input_binding_index_")
        )
        if mutation == "indices":
            model.mech._buffers[name].add_(1)
        else:
            model.mech._buffers[name] = model.mech._buffers[name].clone()
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="input-binding|structure"):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("operation", ("step", "rollout"))
def test_public_rebinding_invalidates_warmed_raw_compiled_atomic_methods(operation):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def evaluate(parameters, constants, state):
        if operation == "step":
            return functional.prepare_and_step(parameters, constants, state)
        return functional.prepare_and_rollout(parameters, constants, state, steps=2)

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    model.bind_input("receiver.DE.ReceiverValues.drive", "source.bias")
    snapshot = _module_snapshot(model)
    initial_state = _clone(tensors.state)
    # A guard miss retraces the callable. Dynamo reports the deliberate
    # FunctionalizationError as Unsupported when fullgraph tracing reaches it.
    with pytest.raises(
        (dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported),
        match="input bindings changed",
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
    _assert_tree_close(tensors.state, initial_state)
    _assert_module_snapshot(model, snapshot)


def test_public_rebinding_invalidates_eager_step_and_new_chunks():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    model.bind_input("receiver.DE.ReceiverValues.drive", "source.bias")
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="input-binding"):
        functional.step(tensors.parameters, prepared, tensors.state)
    with pytest.raises(dn.func.FunctionalizationError, match="input-binding"):
        functional.compile_rollout_chunk(2, backend="eager")
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("mutation", ("binding", "plan", "indices"))
def test_input_binding_invalid_retained_plan_is_rejected_before_lowering(mutation):
    model = _model(layout="regional")
    if mutation == "binding":
        model._input_bindings = model._input_bindings[:-1]
    elif mutation == "plan":
        model.mech._input_binding_plan = model.mech._input_binding_plan[:-1]
    else:
        name = next(
            name
            for name in model.mech._buffers
            if name.startswith("_input_binding_index_")
        )
        model.mech._buffers[name].add_(1)
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="input-binding"):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, snapshot)


def test_unbound_input_defaults_are_ephemeral_and_support_isolated_lowering():
    model = _model(bind=False)
    snapshot = _module_snapshot(model)
    runner = _runner(model, steps=1)
    state, auxiliary = runner()
    trace = auxiliary["callbacks"]["trace"]
    owned = torch.where(
        trace["receiver.owned"][0] >= -62.0, -66.0, trace["receiver.owned"][0]
    )
    torch.testing.assert_close(
        state["mechanisms"]["receiver"]["owned"], owned + DT * (0.2 + 0.1 * owned)
    )
    assert torch.count_nonzero(state["mechanisms"]["receiver"]["seen"]) == 0
    receiver = model.mech.receiver.DE["ReceiverValues"]
    assert not set(receiver._inputs) & set(receiver._buffers)
    _assert_module_snapshot(model, snapshot)


def test_input_bindings_do_not_admit_hidden_live_state_references():
    model = _model()
    model.mech.receiver.DE["ReceiverValues"].setreference(
        "hidden", lambda: model.mech.source.drive
    )
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="setreference|live"):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("lowered", (False, True))
def test_input_binding_rejects_poisoned_source_getter_without_evaluating_it(lowered):
    model = _model()
    if lowered:
        functional, tensors = dn.func.make_functional(model, dt=DT)
    calls = []
    model.mech.source.setreference("drive", lambda: calls.append("called"))
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError):
        if lowered:
            functional.prepare_and_step(
                tensors.parameters, tensors.constants, tensors.state
            )
        else:
            dn.func.make_functional(model, dt=DT)
    assert calls == []
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("adapter", ("frame", "resolver"))
def test_input_bindings_reject_replaced_canonical_adapters_before_calls(adapter):
    model = _model()
    calls = []

    def unsupported(*args):
        calls.append(args)
        return {}

    if adapter == "frame":
        model.mech.receiver.DE["ReceiverValues"]._input_value_frame = unsupported
    else:
        model.mech._resolve_state_inputs = unsupported
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical|standard"):
        dn.func.make_functional(model, dt=DT)
    assert not calls
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("mutation", ("in_place", "replace"))
@pytest.mark.parametrize("bound", (False, True))
@pytest.mark.parametrize("source_first", (False, True))
def test_input_bindings_reject_writes_inside_assigned_hooks(
    mutation, bound, source_first
):
    class BadInputValues(ReceiverValues):
        def assigned_values(self, v, values):
            if mutation == "in_place":
                values["drive"].add_(1.0)
            else:
                values["drive"] = values["drive"] + 1.0
            return super().assigned_values(v, values)

    class BadReceiver(VoltageProcess):
        VoltageProcess.STATE_BUNDLE(BadInputValues)
        VoltageProcess.GLOBAL(threshold=-62.0, rest=-66.0)
        voltage_values = _Receiver.voltage_values

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        model = dn.Population(
            N=1, C=2, dtype=torch.float64, integrator=dn.scnv(imem=False)
        )
        order = (_Source.rename("source"), BadReceiver.rename("receiver"))
        for mechanism in order if source_first else reversed(order):
            model.insert(mechanism)
        if bound:
            for name in ("drive", "bias", "gain"):
                model.bind_input(f"receiver.DE.BadInputValues.{name}", f"source.{name}")
        model.train()
        model.initialize()
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="read-only INPUT"):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, snapshot)


def test_input_binding_checkpointed_scan_preserves_trace_and_gradients():
    model = _model()
    runner = _runner(model, checkpoint_every=2)
    parameters = _clone(runner.parameters)
    gain = parameters[_key(parameters, ".source.gain_param")].requires_grad_()
    _, auxiliary = runner(parameters=parameters)
    initialized = runner.functional.initialize(
        parameters, runner.constants, runner.initialization_inputs
    )
    expected = _reference(
        model, initialized.parameters, initialized.constants, initialized.state, STEPS
    )
    _assert_tree_close(
        dict(auxiliary["callbacks"]["trace"]), expected, rtol=2e-9, atol=2e-10
    )
    actual = auxiliary["callbacks"]["trace"]["receiver.owned"].sum()
    torch.testing.assert_close(
        torch.autograd.grad(actual, gain)[0],
        torch.autograd.grad(expected["receiver.owned"].sum(), gain)[0],
    )


@pytest.mark.slow
def test_input_bindings_native_inductor_scan():
    with torch_compiler_warning_context():
        model = _model(batch=None, layout="reordered")
        runner = _runner(model, backend="inductor", checkpoint_every=2)
        parameters = _clone(runner.parameters)
        gain = parameters[_key(parameters, ".source.gain_param")].requires_grad_()
        state, auxiliary = runner(parameters=parameters)
        initialized = runner.functional.initialize(
            parameters, runner.constants, runner.initialization_inputs
        )
        expected = _reference(
            model,
            initialized.parameters,
            initialized.constants,
            initialized.state,
            STEPS,
        )
        _assert_tree_close(
            dict(auxiliary["callbacks"]["trace"]), expected, rtol=2e-9, atol=2e-10
        )
        torch.testing.assert_close(
            state["mechanisms"]["receiver"]["owned"], expected["receiver.owned"][-1]
        )
        actual_gradient = torch.autograd.grad(
            auxiliary["callbacks"]["trace"]["receiver.owned"].sum(), gain
        )[0]
        expected_gradient = torch.autograd.grad(expected["receiver.owned"].sum(), gain)[
            0
        ]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.count_nonzero(actual_gradient)
