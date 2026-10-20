"""Pure Synapse receive hooks preserve imperative transactions and timing."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import ContinuousSynapse, State, Synapse
from dendra.models.mod import exp2syn, expsyn


class LeftValues(State):
    State.STATE("left")
    State.CARRY("count", dtype=torch.int64)
    State.CARRY("flag", dtype=torch.bool)
    State.INPUT("external")
    State.DERIVATIVE("left' = -left")

    def state_defaults(self, v, values):
        return {"left": torch.zeros_like(v)}


class RightValues(State):
    State.STATE("right")
    State.CARRY("count", dtype=torch.int64)
    State.DERIVATIVE("right' = -right")

    def state_defaults(self, v, values):
        return {"right": torch.zeros_like(v)}


class _ReceivingSynapse(Synapse):
    Synapse.STATE_BUNDLE(LeftValues, RightValues)
    Synapse.CARRY("count", dtype=torch.int64)
    Synapse.CARRY("elapsed")
    Synapse.ASSIGNED("scratch")
    Synapse.DERIVED_BUFFER("factor")
    Synapse.GLOBAL(scale=1.75)

    def derive_buffers(self):
        return {"factor": torch.ones_like(self.diam) * self.scale}

    def assigned_values(self, v, values):
        return {"scratch": torch.zeros_like(v)}

    def net_receive_values(self, weights, dt, values):
        increment = weights * values["factor"]
        return {
            "states": {
                "left": values["left"] + increment,
                "right": values["right"] + 2 * increment,
            },
            "carry": {"count": values["count"] + 2, "elapsed": values["elapsed"] + dt},
            "state_carry": {
                "LeftValues": {
                    "count": values["state_carry"]["LeftValues"]["count"] + 1
                },
                "RightValues": {
                    "count": values["state_carry"]["RightValues"]["count"] + 3
                },
            },
        }


def _model(synapse=_ReceivingSynapse, *, batch=None, dtype=torch.float64):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Population(N=1, C=3, dtype=dtype)
        model.insert(synapse.rename("syn"))
        if batch is not None:
            model.batch(batch)
        model.initialize()
        model.mech.set_dt(torch.tensor(0.2, dtype=dtype))
    return model


def _snapshot(synapse):
    return {
        name: (value, value.detach().clone()) for name, value in synapse.named_buffers()
    }


def _unchanged(synapse, snapshot):
    assert dict(synapse.named_buffers()).keys() == snapshot.keys()
    for name, value in synapse.named_buffers():
        original, expected = snapshot[name]
        assert value is original, name
        torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("batch", (None, 2))
def test_namespaced_receive_updates_count_every_delivery_including_zero(batch):
    model = _model(batch=batch)
    synapse = model.mech.syn
    initial_flag = synapse.DE["LeftValues"].flag
    weights = torch.tensor([0.0, 0.4, -0.2], dtype=model.v.dtype)
    for payload in (weights, torch.zeros_like(weights), weights):
        synapse.net_receive(payload, netcon=None)
    torch.testing.assert_close(
        synapse.left, (2 * 1.75 * weights).expand_as(synapse.left)
    )
    torch.testing.assert_close(synapse.right, 2 * synapse.left)
    torch.testing.assert_close(synapse.count, torch.full_like(synapse.count, 6))
    torch.testing.assert_close(
        synapse.DE["LeftValues"].count,
        torch.full_like(synapse.DE["LeftValues"].count, 3),
    )
    torch.testing.assert_close(
        synapse.DE["RightValues"].count,
        torch.full_like(synapse.DE["RightValues"].count, 9),
    )
    torch.testing.assert_close(synapse.elapsed, torch.full_like(synapse.elapsed, 0.6))
    assert synapse.DE["LeftValues"].flag is initial_flag


def test_pure_hook_reads_explicit_state_and_prepared_values_without_mutation():
    synapse = _model().mech.syn
    snapshot = _snapshot(synapse)
    values = synapse._net_receive_value_frame()
    assert "scratch" not in values
    assert "factor" in values
    values["left"] = torch.full_like(synapse.left, 11.0)
    values["right"] = torch.full_like(synapse.right, -9.0)
    values["factor"] = torch.full_like(synapse.factor, 3.0)
    weights = torch.full_like(synapse.left, 0.5, requires_grad=True)
    before = {
        name: value.clone() for name, value in values.items() if torch.is_tensor(value)
    }
    result = synapse._derive_net_receive_values(weights, synapse.dt, values)
    torch.testing.assert_close(
        result["states"]["left"], torch.full_like(synapse.left, 12.5)
    )
    torch.testing.assert_close(
        result["states"]["right"], torch.full_like(synapse.right, -6.0)
    )
    for name, expected in before.items():
        torch.testing.assert_close(values[name], expected)
    _unchanged(synapse, snapshot)
    torch.testing.assert_close(
        torch.autograd.grad(result["states"]["left"].sum(), weights)[0],
        torch.full_like(weights, 3.0),
    )


def test_omitted_receive_sections_and_fields_preserve_buffer_identity():
    class EmptyReceive(_ReceivingSynapse):
        def net_receive_values(self, weights, dt, values):
            return {}

    synapse = _model(EmptyReceive).mech.syn
    snapshot = _snapshot(synapse)
    synapse.net_receive(torch.zeros_like(synapse.left), netcon=None)
    _unchanged(synapse, snapshot)


def test_receive_frame_does_not_reuse_ephemeral_assigned_scratch():
    class ScratchReceive(_ReceivingSynapse):
        def net_receive_values(self, weights, dt, values):
            return {"states": {"left": values["left"] + values["scratch"]}}

    synapse = _model(ScratchReceive).mech.syn
    synapse.scratch = torch.full_like(synapse.left, 12.0)
    snapshot = _snapshot(synapse)
    with pytest.raises(KeyError, match="scratch"):
        synapse.net_receive(torch.zeros_like(synapse.left), netcon=None)
    _unchanged(synapse, snapshot)


def test_receive_frame_exposes_saved_current_without_recomputing_it():
    class SavedReceive(expsyn):
        Synapse.SAVE_CURRENT("i")

        def net_receive_values(self, weights, dt, values):
            return {"states": {"g": values["g"] + weights + values["i_"]}}

    synapse = _model(SavedReceive).mech.syn
    synapse.i_ = torch.full_like(synapse.g, 3.0)
    saved_current = synapse.i_
    synapse.net_receive(torch.zeros_like(synapse.g), netcon=None)
    torch.testing.assert_close(synapse.g, torch.full_like(synapse.g, 3.0))
    assert synapse.i_ is saved_current


@pytest.mark.parametrize(
    "bad",
    (
        "outer_type",
        "unknown_section",
        "states_type",
        "undeclared_state",
        "assigned",
        "workspace",
        "input",
        "wrong_carry_role",
        "state_carry_type",
        "unknown_bundle",
        "nested_type",
        "nested_undeclared",
        "tensor_type",
        "shape",
        "dtype",
        "device",
        "nested_dtype",
    ),
)
def test_malformed_receive_updates_are_transactional_on_every_call(bad):
    class InvalidReceive(_ReceivingSynapse):
        invalid = False

        def net_receive_values(self, weights, dt, values):
            result = _ReceivingSynapse.net_receive_values(self, weights, dt, values)
            if not self.invalid:
                return result
            if bad == "outer_type":
                return []
            if bad == "unknown_section":
                result["wrong"] = {}
            elif bad == "states_type":
                result["states"] = []
            elif bad in {"undeclared_state", "assigned", "workspace", "input"}:
                name = {
                    "undeclared_state": "unknown",
                    "assigned": "scratch",
                    "workspace": "factor",
                    "input": "external",
                }[bad]
                result["states"][name] = values["left"]
            elif bad == "wrong_carry_role":
                result["carry"]["left"] = values["left"]
            elif bad == "state_carry_type":
                result["state_carry"] = []
            elif bad == "unknown_bundle":
                result["state_carry"]["Missing"] = {}
            elif bad == "nested_type":
                result["state_carry"]["LeftValues"] = []
            elif bad == "nested_undeclared":
                result["state_carry"]["LeftValues"]["left"] = values["left"]
            elif bad == "tensor_type":
                result["states"]["right"] = 3.0
            elif bad == "shape":
                result["states"]["right"] = values["right"].reshape(-1)
            elif bad == "dtype":
                result["states"]["right"] = values["right"].float()
            elif bad == "device":
                result["states"]["right"] = torch.empty_like(
                    values["right"], device="meta"
                )
            elif bad == "nested_dtype":
                result["state_carry"]["LeftValues"]["count"] = values["left"]
            return result

    synapse = _model(InvalidReceive).mech.syn
    weights = torch.ones_like(synapse.left)
    synapse.net_receive(weights, netcon=None)
    synapse.invalid = True
    snapshot = _snapshot(synapse)
    with pytest.raises((KeyError, TypeError, ValueError)):
        synapse.net_receive(weights, netcon=None)
    _unchanged(synapse, snapshot)


@pytest.mark.parametrize("bad", ("not_tensor", "shape", "dtype", "device"))
def test_invalid_payload_does_not_change_synapse(bad):
    synapse = _model().mech.syn
    weights = {
        "not_tensor": 1.0,
        "shape": torch.ones(4, dtype=torch.float64),
        "dtype": torch.ones(3, dtype=torch.float32),
        "device": torch.ones(3, dtype=torch.float64, device="meta"),
    }[bad]
    snapshot = _snapshot(synapse)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        synapse.net_receive(weights, netcon=None)
    _unchanged(synapse, snapshot)


def test_flat_state_carry_namespace_collision_is_rejected():
    class Conflicting(Synapse):
        Synapse.CARRY("state_carry")

        def net_receive_values(self, weights, dt, values):
            return {}

    synapse = _model(Conflicting).mech.syn
    with pytest.raises(ValueError, match="reserve 'state_carry'"):
        synapse.net_receive(torch.tensor(0.0, dtype=torch.float64), netcon=None)


def test_receive_cannot_replace_a_declared_state_that_aliases_a_shared_field():
    class CalciumValues(State):
        State.STATE("cai")
        State.DERIVATIVE("cai' = 0")

        def state_defaults(self, v, values):
            return {"cai": torch.full_like(v, 0.00005)}

    class SharedReceive(Synapse):
        Synapse.STATE_BUNDLE(CalciumValues)
        Synapse.USEION("ca", write=["cai"])

        def net_receive_values(self, weights, dt, values):
            return {"states": {"cai": values["cai"] + weights}}

    synapse = _model(SharedReceive).mech.syn
    snapshot = _snapshot(synapse)
    with pytest.raises(KeyError, match="undeclared"):
        synapse.net_receive(torch.zeros_like(synapse.cai), netcon=None)
    _unchanged(synapse, snapshot)


@pytest.mark.parametrize("field", ("incoming", "incoming_old"))
def test_receive_cannot_replace_continuous_input_carry(field):
    class Hybrid(Synapse, ContinuousSynapse):
        ContinuousSynapse.INPUT("incoming")

        def net_receive_values(self, weights, dt, values):
            return {"carry": {field: values[field] + weights}}

    synapse = _model(Hybrid).mech.syn
    snapshot = _snapshot(synapse)
    with pytest.raises(KeyError, match="undeclared"):
        synapse.net_receive(torch.ones_like(synapse.incoming), netcon=None)
    _unchanged(synapse, snapshot)


def test_compiled_receive_keeps_the_validated_namespaced_transaction():
    synapse = _model(batch=2).mech.syn

    def receive(weights):
        updates = synapse._derive_net_receive_values(
            weights, synapse.dt, synapse._net_receive_value_frame()
        )
        synapse._apply_net_receive_updates(updates)
        return synapse.left, synapse.DE["LeftValues"].count

    compiled = torch.compile(receive, fullgraph=True, backend="eager")
    for weight in (0.2, 0.0):
        result, count = compiled(torch.tensor(weight, dtype=torch.float64))
    torch.testing.assert_close(result, torch.full_like(result, 0.35))
    torch.testing.assert_close(count, torch.full_like(count, 2))


def test_compiled_receive_rejects_a_changed_invalid_schema_before_writes():
    class ChangedReceive(_ReceivingSynapse):
        invalid = False

        def net_receive_values(self, weights, dt, values):
            result = _ReceivingSynapse.net_receive_values(self, weights, dt, values)
            if self.invalid:
                result["unknown"] = {}
            return result

    synapse = _model(ChangedReceive).mech.syn

    def receive(weights):
        synapse.net_receive(weights, netcon=None)
        return synapse.left

    compiled = torch.compile(receive, fullgraph=True, backend="eager")
    weights = torch.ones_like(synapse.left)
    compiled(weights)
    synapse.invalid = True
    snapshot = _snapshot(synapse)
    with pytest.raises((KeyError, RuntimeError), match="unknown|exception"):
        compiled(weights)
    _unchanged(synapse, snapshot)


class _NormalizationProbe(torch.nn.Module):
    def __init__(self, synapse):
        super().__init__()
        self.synapse = synapse

    def forward(self):
        return self.synapse.derive_buffers()["factor"]


class _ReceiveTimingProbe(torch.nn.Module):
    def __init__(self, synapse):
        super().__init__()
        self.synapse = synapse

    def forward(self, dt):
        weights = self.synapse.diam.new_zeros(self.synapse._net_receive_payload_shape())
        return self.synapse._derive_net_receive_values(
            weights, dt, self.synapse._net_receive_value_frame()
        )["carry"]["elapsed"]


@pytest.mark.parametrize("scalar_carry", (False, True))
@pytest.mark.parametrize("field", ("dt", "celsius"))
@pytest.mark.parametrize("lanes", (0, 3))
def test_receive_timing_inputs_keep_gradients_under_empty_and_nonempty_vmap(
    scalar_carry, field, lanes
):
    class TimingReceive(Synapse):
        Synapse.CARRY("elapsed", shape=() if scalar_carry else "local")

        def net_receive_values(self, weights, dt, values):
            assert dt.shape == (1,)
            elapsed = values["elapsed"].reshape(-1) + dt * values["celsius"] / 37
            return {"carry": {"elapsed": elapsed.reshape(values["elapsed"].shape)}}

    synapse = _model(TimingReceive).mech.syn
    probe = _ReceiveTimingProbe(synapse)
    incoming = torch.linspace(
        0.1 if field == "dt" else 30.0,
        0.3 if field == "dt" else 40.0,
        lanes,
        dtype=torch.float64,
        requires_grad=True,
    )

    def event(value):
        if field == "dt":
            return probe(value)
        return torch.func.functional_call(
            probe, {"synapse.celsius": value}, (synapse.dt,)
        )

    actual = torch.vmap(event)(incoming)
    expected = incoming if field == "dt" else incoming * 0.2 / 37
    expected = expected.reshape((lanes,) + (1,) * synapse.elapsed.ndim)
    expected = expected.expand((lanes,) + synapse.elapsed.shape)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), incoming)[0]
    scale = 1.0 if field == "dt" else 0.2 / 37
    torch.testing.assert_close(
        gradient, torch.full_like(incoming, scale * synapse.elapsed.numel())
    )


def _normalization_function():
    probe = _NormalizationProbe(_model(exp2syn).mech.syn)

    def normalization(tau1, tau2):
        return torch.func.functional_call(
            probe,
            {"synapse.DE.A.tau1": tau1, "synapse.DE.B.tau2": tau2},
            (),
        )

    return normalization


def _normalization_oracle(tau1, tau2):
    peak = tau1 * tau2 / (tau2 - tau1) * torch.log(tau2 / tau1)
    return 1 / (torch.exp(-peak / tau2) - torch.exp(-peak / tau1))


@pytest.mark.parametrize("compiled", (False, True))
def test_exp2syn_normalization_supports_vmap_and_rejects_invalid_lanes(compiled):
    normalize = _normalization_function()
    mapped = torch.vmap(normalize)
    if compiled:
        mapped = torch.compile(mapped, fullgraph=True, backend="eager")
    tau1 = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)
    tau2 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    actual = mapped(tau1, tau2)
    expected = _normalization_oracle(tau1, tau2)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(
        actual.sum(), (tau1, tau2), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.sum(), (tau1, tau2))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    error = RuntimeError if compiled else ValueError
    with pytest.raises(error, match="positive and finite"):
        mapped(-tau1.detach(), tau2.detach())


@pytest.mark.slow
def test_native_exp2syn_normalization_keeps_tau_gradients_and_rejects_invalid_values():
    normalize = torch.compile(
        _normalization_function(), fullgraph=True, backend="inductor"
    )
    tau1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    tau2 = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    actual = normalize(tau1, tau2)
    expected = _normalization_oracle(tau1, tau2)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual, (tau1, tau2), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (tau1, tau2))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert actual_gradient.abs() > 0
    with pytest.raises(RuntimeError, match="positive and finite"):
        normalize(-tau1.detach(), tau2.detach())


def test_legacy_override_remains_imperative():
    class Legacy(Synapse):
        Synapse.CARRY("received")

        def net_receive(self, weights, netcon):
            self.received = self.received + weights + netcon

    synapse = _model(Legacy).mech.syn
    synapse.net_receive(2.0, netcon=3.0)
    torch.testing.assert_close(synapse.received, torch.full_like(synapse.received, 5.0))


@pytest.mark.parametrize("synapse_type", (expsyn, exp2syn))
def test_builtin_scalar_receive_preserves_weight_and_parameter_gradients(synapse_type):
    model = _model(synapse_type)
    synapse = model.mech.syn
    weight = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    synapse.net_receive(weight, netcon=None)
    if synapse_type is expsyn:
        actual = synapse.g
        expected = torch.ones_like(actual) * weight
    else:
        tau1, tau2 = synapse.DE["A"].tau1, synapse.DE["B"].tau2
        peak = (tau1 * tau2) / (tau2 - tau1) * torch.log(tau2 / tau1)
        factor = 1 / (torch.exp(-peak / tau2) - torch.exp(-peak / tau1))
        actual = synapse.A + synapse.B
        expected = torch.ones_like(actual) * 2 * weight * factor
    torch.testing.assert_close(actual, expected)
    targets = (
        (weight,)
        if synapse_type is expsyn
        else (weight, synapse.DE["A"].tau1_param, synapse.DE["B"].tau2_param)
    )
    actual_gradients = torch.autograd.grad(actual.sum(), targets, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), targets)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
