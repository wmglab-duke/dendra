"""Pure event transactions retain ordering, carry, preparation, and gradients."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State, Synapse
from dendra.models.mod import exp2syn, expsyn, pas
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import _live_state


@pytest.fixture(autouse=True)
def _check_purity_diagnostics():
    """Exercise purity contracts explicitly; production defaults trust hooks."""
    with dn.ctx(CHECK_PURITY=True):
        yield


DT = 0.05
pytestmark = pytest.mark.cpu


class EventMemory(State):
    State.STATE("x")
    State.CARRY("mark")
    State.DERIVATIVE("x' = 0.0 * x")

    def state_defaults(self, v, values):
        return {"x": torch.zeros_like(v)}

    def initial_values(self, v, values):
        return {"mark": torch.zeros_like(v)}


class OtherMemory(State):
    State.STATE("y")
    State.CARRY("mark")
    State.DERIVATIVE("y' = 0.0 * y")

    def state_defaults(self, v, values):
        return {"y": torch.zeros_like(v)}

    def initial_values(self, v, values):
        return {"mark": torch.ones_like(v) * 2}


class _OrderedEvent(Synapse):
    Synapse.STATE_BUNDLE(EventMemory, OtherMemory)
    Synapse.CARRY("ticks", "accum")
    Synapse.GLOBAL(scale=1.2)
    Synapse.DERIVED_BUFFER("gain", shape=())

    def derive_buffers(self):
        return {"gain": self.scale.square()}

    def initial_values(self, v, values):
        return {"ticks": torch.zeros_like(v), "accum": torch.zeros_like(v)}

    def net_receive_values(self, weights, dt, values):
        gain = values["gain"].reshape((1,) * weights.ndim)
        return {
            "states": {"x": values["x"] + weights * gain + 0.1 * values["ticks"]},
            "carry": {
                "ticks": values["ticks"] + 1,
                "accum": 2 * values["accum"] + weights,
            },
            "state_carry": {
                "EventMemory": {
                    "mark": values["state_carry"]["EventMemory"]["mark"] + dt + weights
                }
            },
        }


class _LegacyEvent(Synapse):
    Synapse.STATE_BUNDLE(EventMemory)

    def net_receive(self, weights, netcon):
        self.x = self.x + weights


class CalciumState(State):
    State.STATE("cai")
    State.DERIVATIVE("cai' = 0.0 * cai")

    def state_defaults(self, v, values):
        return {"cai": torch.full_like(v, 0.4)}


class _SharedState(Mechanism):
    Mechanism.STATE_BUNDLE(CalciumState)
    Mechanism.USEION("ca", read=["cai"])


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _key(parameters, suffix):
    matches = [name for name in parameters if name.endswith(suffix)]
    assert len(matches) == 1, (suffix, matches)
    return matches[0]


def _model(synapse=expsyn, *, batch=2, regional=False, shared=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        model = dn.Population(
            N=1, C=3, dtype=torch.float64, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas, g=0.001, e=-65.0)
        target = model[:, torch.tensor([2, 0])] if regional else model
        target.insert(synapse.rename("event"))
        if shared:
            model.insert(_SharedState.rename("shared"))
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
    return model


def _factor(parameters):
    tau1 = parameters[_key(parameters, ".event.DE.A.tau1_param")]
    tau2 = parameters[_key(parameters, ".event.DE.B.tau2_param")]
    ratio = torch.clamp(tau1 / tau2, 1e-9, 0.9999)
    tau1_effective = ratio * tau2
    peak = (
        tau1_effective
        * tau2
        / (tau2 - tau1_effective)
        * torch.log(tau2 / tau1_effective)
    )
    return 1 / (torch.exp(-peak / tau2) - torch.exp(-peak / tau1_effective))


@pytest.mark.parametrize("synapse", (expsyn, exp2syn))
@pytest.mark.parametrize("regional", (False, True))
def test_receive_updates_only_synaptic_state_with_independent_weight_oracle(
    synapse, regional
):
    model = _model(synapse, regional=regional, shared=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initial = _clone(tensors.state)
    # Deliberately separate a mechanism's flattened shared State from its ion.
    # Receiving another mechanism must not synchronize either public leaf.
    initial["mechanisms"]["shared"]["cai"] = (
        initial["mechanisms"]["shared"]["cai"] + 0.7
    )
    incoming = _clone(initial)
    snapshot = _module_snapshot(model)
    shape = functional._receiver_shapes["event"]
    payload = (
        torch.linspace(
            -0.1, 0.4, torch.tensor(shape).prod().item(), dtype=torch.float64
        )
        .reshape(shape)
        .requires_grad_()
    )
    parameters = _clone(tensors.parameters)
    targets = [payload]
    if synapse is exp2syn:
        targets.extend(
            parameters[_key(parameters, suffix)].requires_grad_()
            for suffix in (".event.DE.A.tau1_param", ".event.DE.B.tau2_param")
        )
    prepared = functional.prepare(parameters, tensors.constants)
    result = functional.receive(
        parameters, prepared, initial, target="event", payload=payload
    )
    expected = _clone(incoming)
    if synapse is expsyn:
        expected["mechanisms"]["event"]["g"] = (
            initial["mechanisms"]["event"]["g"] + payload
        )
    else:
        factor = _factor(parameters)
        for name in ("A", "B"):
            expected["mechanisms"]["event"][name] = (
                initial["mechanisms"]["event"][name] + payload * factor
            )
    _assert_tree_close(result, expected, rtol=2e-9, atol=2e-10)
    actual_loss = sum(
        value.square().sum() for value in result["mechanisms"]["event"].values()
    )
    expected_loss = sum(
        value.square().sum() for value in expected["mechanisms"]["event"].values()
    )
    for actual, reference in zip(
        torch.autograd.grad(actual_loss, targets),
        torch.autograd.grad(expected_loss, targets),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference)
        assert torch.count_nonzero(actual)
    _assert_tree_close(initial, incoming)
    _assert_module_snapshot(model, snapshot)


def test_zero_payloads_and_separate_calls_preserve_ordered_namespaced_updates():
    model = _model(_OrderedEvent, regional=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)
    state = _clone(tensors.state)
    expected = _clone(state)
    scale = tensors.parameters[_key(tensors.parameters, ".event.scale_param")]
    for weight in (0.3, 0.0, -0.1):
        payload = model.v.new_tensor(weight)
        state = functional.prepare_and_receive(
            tensors.parameters,
            tensors.constants,
            state,
            target="event",
            payload=payload,
        )
        tick = expected["mechanism_buffers"]["event"]["ticks"]
        expected["mechanisms"]["event"]["x"] = (
            expected["mechanisms"]["event"]["x"] + weight * scale.square() + 0.1 * tick
        )
        expected["mechanism_buffers"]["event"]["accum"] = (
            2 * expected["mechanism_buffers"]["event"]["accum"] + weight
        )
        expected["mechanism_buffers"]["event"]["ticks"] = tick + 1
        expected["state_buffers"]["event"]["EventMemory"]["mark"] = (
            expected["state_buffers"]["event"]["EventMemory"]["mark"] + DT + weight
        )
    _assert_tree_close(state, expected)
    torch.testing.assert_close(
        state["state_buffers"]["event"]["OtherMemory"]["mark"],
        torch.full_like(state["state_buffers"]["event"]["OtherMemory"]["mark"], 2),
    )
    assert torch.all(state["mechanism_buffers"]["event"]["ticks"] == 3)
    _assert_module_snapshot(model, snapshot)
    functional.commit_state_(model, state)
    _assert_tree_close(_live_state(functional, model), state)


@pytest.mark.parametrize("synapse", (expsyn, exp2syn))
def test_receive_then_step_matches_imperative_without_advancing_clock_on_receive(
    synapse,
):
    model = _model(synapse)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    reference = _model(synapse)
    dt = reference.v.new_tensor(DT)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    state = tensors.state
    for weight in (0.2, 0.0, -0.03):
        payload = model.v.new_tensor(weight)
        before = state["clock"]["t"]
        state = functional.receive(
            tensors.parameters, prepared, state, target="event", payload=payload
        )
        torch.testing.assert_close(state["clock"]["t"], before)
        reference.mech.event.net_receive(payload, None)
        state, _ = functional.step(tensors.parameters, prepared, state)
        reference.integrator.step(reference, dt)
        reference.t = reference.t + dt
        _assert_tree_close(
            _live_state(functional, reference), state, rtol=2e-9, atol=2e-10
        )
    _assert_module_snapshot(model, snapshot)


def test_receive_requires_fresh_preparation_and_uses_explicit_carry():
    model = _model(_OrderedEvent)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone(tensors.parameters)
    prepared = functional.prepare(parameters, tensors.constants)
    key = _key(parameters, ".event.scale_param")
    parameters[key] = (parameters[key] + 0.4).requires_grad_()
    state = _clone(tensors.state)
    state["mechanisms"]["event"]["x"] = state["mechanisms"]["event"]["x"] + 4
    state["state_buffers"]["event"]["EventMemory"]["mark"] = (
        state["state_buffers"]["event"]["EventMemory"]["mark"] + 2
    )
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="parameters changed"):
        functional.receive(
            parameters, prepared, state, target="event", payload=model.v.new_tensor(0.2)
        )
    actual = functional.prepare_and_receive(
        parameters,
        tensors.constants,
        state,
        target="event",
        payload=model.v.new_tensor(0.2),
    )
    torch.testing.assert_close(
        actual["mechanisms"]["event"]["x"],
        state["mechanisms"]["event"]["x"] + 0.2 * parameters[key].square(),
    )
    torch.testing.assert_close(
        actual["state_buffers"]["event"]["EventMemory"]["mark"],
        state["state_buffers"]["event"]["EventMemory"]["mark"] + DT + 0.2,
    )
    _assert_module_snapshot(model, snapshot)


def test_receive_vmap_empty_lanes_and_forward_reverse_jacobians():
    model = _model(_OrderedEvent, batch=None, regional=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)
    key = _key(tensors.parameters, ".event.scale_param")
    width = model.mech.event.x.numel()

    def evaluate(scale, payload):
        state = functional.prepare_and_receive(
            {**tensors.parameters, key: scale},
            tensors.constants,
            tensors.state,
            target="event",
            payload=payload,
        )
        state = functional.prepare_and_receive(
            {**tensors.parameters, key: scale},
            tensors.constants,
            state,
            target="event",
            payload=torch.zeros_like(payload),
        )
        return state["mechanisms"]["event"]["x"].reshape(width)

    inputs = (tensors.parameters[key], model.v.new_tensor(0.2))
    lanes = tuple(torch.stack((value, value + 0.1)) for value in inputs)
    torch.testing.assert_close(
        torch.vmap(evaluate)(*lanes),
        torch.stack(
            [evaluate(*(value[index] for value in lanes)) for index in range(2)]
        ),
    )
    assert torch.vmap(evaluate)(*(value[:0] for value in lanes)).shape == (0, width)
    for actual, expected in zip(
        torch.func.jacfwd(evaluate, argnums=(0, 1))(*inputs),
        torch.func.jacrev(evaluate, argnums=(0, 1))(*inputs),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("regional", (False, True))
def test_receive_dt_vmap_empty_lanes_and_timestep_gradients(regional):
    model = _model(_OrderedEvent, batch=None, regional=regional)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)

    def evaluate(dt):
        state = functional.prepare_and_receive(
            tensors.parameters,
            {**tensors.constants, "dt": dt},
            tensors.state,
            target="event",
            payload=model.v.new_tensor(0.1),
        )
        return state["state_buffers"]["event"]["EventMemory"]["mark"]

    times = model.v.new_tensor([DT, DT * 2]).requires_grad_()
    actual = torch.vmap(evaluate)(times)
    expected = torch.stack([evaluate(dt) for dt in times])
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), times)[0]
    torch.testing.assert_close(gradient, torch.full_like(times, actual[0].numel()))
    empty = model.v.new_empty(0).requires_grad_()
    empty_result = torch.vmap(evaluate)(empty)
    assert empty_result.shape == (0, *expected.shape[1:])
    assert torch.autograd.grad(empty_result.sum(), empty)[0].shape == (0,)
    reverse = torch.func.jacrev(evaluate)(tensors.constants["dt"])
    forward = torch.func.jacfwd(evaluate)(tensors.constants["dt"])
    torch.testing.assert_close(forward, reverse)
    torch.testing.assert_close(forward, torch.ones_like(forward))
    _assert_module_snapshot(model, snapshot)


def test_receive_celsius_vmap_retains_empty_lanes_and_gradients():
    class TemperatureEvent(_OrderedEvent):
        def net_receive_values(self, weights, dt, values):
            return {"states": {"x": values["x"] + weights * values["celsius"]}}

    model = _model(TemperatureEvent, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)
    key = _key(tensors.parameters, "celsius_param")

    def evaluate(temperature):
        state = functional.prepare_and_receive(
            {**tensors.parameters, key: temperature},
            tensors.constants,
            tensors.state,
            target="event",
            payload=model.v.new_tensor(0.1),
        )
        return state["mechanisms"]["event"]["x"]

    temperatures = model.v.new_tensor([32.0, 37.0]).requires_grad_()
    actual = torch.vmap(evaluate)(temperatures)
    torch.testing.assert_close(
        actual, torch.stack([evaluate(value) for value in temperatures])
    )
    gradient = torch.autograd.grad(actual.sum(), temperatures)[0]
    torch.testing.assert_close(
        gradient, torch.full_like(gradient, 0.1 * actual[0].numel())
    )
    assert torch.vmap(evaluate)(model.v.new_empty(0)).shape == (0, *actual.shape[1:])
    torch.testing.assert_close(
        torch.func.jacfwd(evaluate)(tensors.parameters[key]),
        torch.func.jacrev(evaluate)(tensors.parameters[key]),
    )
    _assert_module_snapshot(model, snapshot)


def test_receive_dt_vmap_retains_scalar_structural_carry_schema():
    class ScalarTimeEvent(Synapse):
        Synapse.CARRY("elapsed", shape=())

        def initial_values(self, v, values):
            return {"elapsed": v.new_zeros(())}

        def net_receive_values(self, weights, dt, values):
            return {
                "carry": {
                    "elapsed": (values["elapsed"] + dt).reshape_as(values["elapsed"])
                }
            }

    model = _model(ScalarTimeEvent, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)

    def evaluate(dt):
        state = functional.prepare_and_receive(
            tensors.parameters,
            {**tensors.constants, "dt": dt},
            tensors.state,
            target="event",
            payload=model.v.new_tensor(0.1),
        )
        return state["mechanism_buffers"]["event"]["elapsed"]

    assert evaluate(tensors.constants["dt"]).shape == ()
    times = model.v.new_tensor([DT, 2 * DT]).requires_grad_()
    actual = torch.vmap(evaluate)(times)
    torch.testing.assert_close(actual, times)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), times)[0], torch.ones_like(times)
    )
    empty = model.v.new_empty(0).requires_grad_()
    result = torch.vmap(evaluate)(empty)
    assert result.shape == (0,)
    assert torch.autograd.grad(result.sum(), empty)[0].shape == (0,)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("synapse", (expsyn, exp2syn))
def test_builtin_receive_step_tau_and_payload_transforms(synapse):
    model = _model(synapse, batch=None, regional=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)
    suffix = ".event.DE.A.tau1_param" if synapse is exp2syn else ".event.DE.g.tau_param"
    key = _key(tensors.parameters, suffix)
    field = "A" if synapse is exp2syn else "g"
    local_width = model.mech.event._buffers[field].numel()
    full_width = model.v.numel()

    def evaluate(tau, payload):
        parameters = {**tensors.parameters, key: tau}
        state = functional.prepare_and_receive(
            parameters,
            tensors.constants,
            tensors.state,
            target="event",
            payload=payload,
        )
        state, _ = functional.prepare_and_step(parameters, tensors.constants, state)
        return torch.cat(
            (
                state["integrator"]["v"].reshape(full_width),
                state["mechanisms"]["event"][field].reshape(local_width),
            )
        )

    inputs = (
        tensors.parameters[key],
        model.v.new_full(functional._receiver_shapes["event"], 0.02),
    )
    lanes = tuple(torch.stack((value, value + 0.01)) for value in inputs)
    torch.testing.assert_close(
        torch.vmap(evaluate)(*lanes),
        torch.stack(
            [evaluate(*(value[index] for value in lanes)) for index in range(2)]
        ),
    )
    assert torch.vmap(evaluate)(*(value[:0] for value in lanes)).shape == (
        0,
        local_width + full_width,
    )
    for actual, reference in zip(
        torch.func.jacfwd(evaluate, argnums=(0, 1))(*inputs),
        torch.func.jacrev(evaluate, argnums=(0, 1))(*inputs),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, rtol=2e-8, atol=2e-10)
        assert torch.count_nonzero(actual)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize(
    "invalid", ("target", "not_synapse", "legacy", "shape", "dtype", "not_tensor")
)
def test_invalid_receive_calls_fail_without_source_or_input_changes(invalid):
    model = _model(_LegacyEvent if invalid == "legacy" else expsyn)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)
    initial = _clone(tensors.state)
    target = (
        "missing"
        if invalid == "target"
        else "pas"
        if invalid == "not_synapse"
        else "event"
    )
    payload = (
        model.v.new_ones(5)
        if invalid == "shape"
        else torch.ones((), dtype=torch.float32)
        if invalid == "dtype"
        else 0.2
        if invalid == "not_tensor"
        else model.v.new_tensor(0.2)
    )
    with pytest.raises((dn.func.FunctionalizationError, TypeError, ValueError)):
        functional.prepare_and_receive(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            target=target,
            payload=payload,
        )
    _assert_tree_close(tensors.state, initial)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize(
    "effect", ("payload", "values", "nested", "self", "python", "rng")
)
def test_receiver_purity_audit_rejects_effects_without_source_changes(effect):
    class Impure(_OrderedEvent):
        def net_receive_values(self, weights, dt, values):
            if effect == "payload":
                weights.add_(1)
            elif effect == "values":
                values["x"] = values["x"] + 1
            elif effect == "nested":
                values["state_carry"]["EventMemory"]["mark"].add_(1)
            elif effect == "self":
                self.x = self.x + 1
            elif effect == "python":
                self.counter = self.counter + 1
            elif effect == "rng":
                weights = weights + torch.rand_like(weights)
            return super().net_receive_values(weights, dt, values)

    model = _model(Impure)
    model.mech.event.counter = 0
    snapshot = _module_snapshot(model)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="event|receive"):
        dn.func.make_functional(model, dt=DT)
    assert model.mech.event.counter == 0
    assert torch.equal(torch.random.get_rng_state(), rng)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize(
    "bad", ("section", "state", "carry", "nested", "shape", "dtype")
)
def test_receiver_return_schemas_are_validated_before_updates(bad):
    class Invalid(_OrderedEvent):
        def net_receive_values(self, weights, dt, values):
            if bad == "section":
                return {"assigned": {"x": values["x"]}}
            if bad == "state":
                return {"states": {"missing": values["x"]}}
            if bad == "carry":
                return {"carry": {"gain": values["gain"]}}
            if bad == "nested":
                return {"state_carry": {"OtherMemory": {"x": values["x"]}}}
            if bad == "shape":
                return {"states": {"x": values["x"].sum()}}
            return {"states": {"x": values["x"].float()}}

    model = _model(Invalid)
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="event|receive"):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("access", ("values", "self", "buffers", "nested"))
def test_receive_rejects_ephemeral_assigned_dependencies(access):
    class AssignedMemory(EventMemory):
        State.ASSIGNED("local_scratch")

        def assigned_values(self, v, values):
            return {"local_scratch": 3 * values["x"]}

    class AssignedReceiver(Synapse):
        Synapse.STATE_BUNDLE(AssignedMemory)
        Synapse.ASSIGNED("scratch")

        def assigned_values(self, v, values):
            return {"scratch": 2 * values["x"]}

        def net_receive_values(self, weights, dt, values):
            if access == "values":
                scratch = values["scratch"]
            elif access == "self":
                scratch = self.scratch
            elif access == "buffers":
                scratch = self._buffers["scratch"]
            else:
                scratch = self.DE["AssignedMemory"].local_scratch
            return {"states": {"x": values["x"] + scratch + weights}}

    model = _model(AssignedReceiver)
    snapshot = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="receive|event"):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize("compiled", (False, True))
def test_receiver_hook_replacement_invalidates_lowered_and_warmed_compiled_calls(
    compiled,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def evaluate(payload):
        return functional.prepare_and_receive(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            target="event",
            payload=payload,
        )

    evaluate = (
        torch.compile(evaluate, backend="eager", fullgraph=True)
        if compiled
        else evaluate
    )
    evaluate(model.v.new_tensor(0.2))
    model.mech.event.net_receive_values = lambda weights, dt, values: {
        "states": {"g": values["g"] + 2 * weights}
    }
    snapshot = _module_snapshot(model)
    with pytest.raises(
        (dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported),
        match="changed|structure",
    ):
        evaluate(model.v.new_tensor(0.2))
    _assert_module_snapshot(model, snapshot)


@pytest.mark.slow
@pytest.mark.parametrize("synapse", (expsyn, exp2syn))
def test_native_inductor_receive_and_step_trace_and_gradients(synapse):
    with torch_compiler_warning_context():
        model = _model(synapse, batch=None, regional=True)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        snapshot = _module_snapshot(model)
        parameters = _clone(tensors.parameters)
        targets = []
        if synapse is exp2syn:
            targets.append(
                parameters[_key(parameters, ".event.DE.B.tau2_param")].requires_grad_()
            )

        def evaluate(payload):
            state = functional.prepare_and_receive(
                parameters,
                tensors.constants,
                tensors.state,
                target="event",
                payload=payload,
            )
            state, _ = functional.prepare_and_step(parameters, tensors.constants, state)
            return state["integrator"]["v"]

        payload = model.v.new_tensor(0.02).requires_grad_()
        targets.append(payload)
        compiled = torch.compile(
            evaluate,
            backend="inductor",
            fullgraph=True,
            options={"cpp_wrapper": True, "compile_threads": 1},
        )
        actual = compiled(payload)
        expected = evaluate(payload)
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)
        for value, reference in zip(
            torch.autograd.grad(actual.sum(), targets),
            torch.autograd.grad(expected.sum(), targets),
            strict=True,
        ):
            torch.testing.assert_close(value, reference, rtol=2e-8, atol=2e-10)
            assert torch.count_nonzero(value)
        _assert_module_snapshot(model, snapshot)


@pytest.mark.parametrize(
    "replacement",
    (
        "method",
        "staticmethod",
        "classmethod",
        "property",
        "descriptor",
        "equal_descriptor",
    ),
)
def test_repeated_compiled_receive_guards_raw_hook_descriptors_before_getters(
    replacement, monkeypatch
):
    model = _model(batch=None)
    owner = type(model.mech.event)
    original = owner.net_receive_values
    # Exercise replacement of an existing raw class entry, including a
    # staticmethod whose resolved function would still be the same object.
    monkeypatch.setattr(owner, "net_receive_values", original)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def evaluate(payload):
        state = functional.prepare_and_receive(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            target="event",
            payload=payload,
        )
        return functional.prepare_and_receive(
            tensors.parameters,
            tensors.constants,
            state,
            target="event",
            payload=payload,
        )

    compiled = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    payload = model.v.new_tensor(0.2)
    _assert_tree_close(compiled(payload), evaluate(payload))
    calls = []

    class Descriptor:
        def __get__(self, instance, owner):
            calls.append("class" if instance is None else "instance")
            return original

    class EqualDescriptor(Descriptor):
        def __eq__(self, other):
            calls.append("equality")
            return True

    if replacement == "method":

        def changed(self, weights, dt, values):
            return {"states": {"g": values["g"] + 2 * weights}}

        value = changed
    elif replacement == "staticmethod":
        value = staticmethod(original)
    elif replacement == "classmethod":
        value = classmethod(original)
    elif replacement == "property":
        value = property(lambda self: calls.append("property"))
    elif replacement == "equal_descriptor":
        value = EqualDescriptor()
    else:
        value = Descriptor()
    monkeypatch.setattr(owner, "net_receive_values", value)
    snapshot = _module_snapshot(model)
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        compiled(payload)
    assert calls == []
    _assert_module_snapshot(model, snapshot)


def test_repeated_compiled_receiver_accepts_overridden_none_base_hook():
    class AbstractReceiver(Synapse):
        net_receive_values = None

    class ConcreteReceiver(AbstractReceiver):
        Synapse.CARRY("calls")

        def net_receive_values(self, weights, dt, values):
            return {"carry": {"calls": values["calls"] + 1}}

    model = _model(ConcreteReceiver, batch=None)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    snapshot = _module_snapshot(model)

    def evaluate(payload):
        state = tensors.state
        for _ in range(2):
            state = functional.prepare_and_receive(
                tensors.parameters,
                tensors.constants,
                state,
                target="event",
                payload=payload,
            )
        return state

    payload = model.v.new_zeros(())
    actual = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(payload)
    _assert_tree_close(actual, evaluate(payload))
    torch.testing.assert_close(
        actual["mechanism_buffers"]["event"]["calls"],
        torch.full_like(model.mech.event.calls, 2),
    )
    _assert_module_snapshot(model, snapshot)
