"""Imperative contracts for explicit same-Population State inputs."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State


class ProviderState(State):
    State.STATE("drive")
    State.DERIVATIVE("drive' = 1")

    def state_defaults(self, v, values):
        return {"drive": torch.zeros_like(v) + 2}


class _Provider(Mechanism):
    Mechanism.STATE_BUNDLE(ProviderState)
    Mechanism.GLOBAL(gain=3.0)


class ReceiverState(State):
    State.STATE("total")
    State.INPUT("incoming", default=0.25)
    State.DERIVATIVE("total' = incoming")

    def state_defaults(self, v, values):
        return {"total": torch.zeros_like(v) + values["incoming"]}


class _Receiver(Mechanism):
    Mechanism.STATE_BUNDLE(ReceiverState)


def _model(*, provider_first=True, regional=False, batch=None, bind=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Population(N=2, C=4, dtype=torch.float64)
        target = model[:, torch.tensor([3, 1])] if regional else model
        actions = (
            lambda: model.insert(_Provider.rename("provider")),
            lambda: target.insert(_Receiver.rename("receiver")),
        )
        for insert in actions if provider_first else reversed(actions):
            insert()
        if bind:
            model.bind_input("receiver.DE.ReceiverState.incoming", "provider.drive")
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
    return model


@pytest.mark.parametrize("provider_first", (True, False))
def test_input_direct_derivative_preserves_consumer_visit_order(provider_first):
    model = _model(provider_first=provider_first)
    initial = model.mech.receiver.total.clone()
    model.step(0.1)
    drive = 2.1 if provider_first else 2.0
    torch.testing.assert_close(model.mech.receiver.total, initial + 0.1 * drive)


@pytest.mark.parametrize("batch", (None, 2))
def test_input_support_projection_retains_batch_axes(batch):
    model = _model(regional=True, batch=batch)
    provider = model.mech.provider
    provider.drive = torch.arange(
        provider.drive.numel(), dtype=torch.float64
    ).reshape_as(provider.drive)
    expected = (
        provider.drive.reshape(*provider.drive.shape[:-2], -1).index_select(
            -1, model.mech.receiver.key
        )
        + 0.1
    )
    initial = model.mech.receiver.total.clone()
    model.step(0.1)
    torch.testing.assert_close(model.mech.receiver.total, initial + 0.1 * expected)


def test_unbound_input_defaults_are_initialization_inputs_and_have_no_buffers():
    model = _model(bind=False)
    receiver = model.mech.receiver
    torch.testing.assert_close(receiver.total, torch.full_like(receiver.total, 0.25))
    assert not hasattr(receiver.DE["ReceiverState"], "incoming")
    assert not any("incoming" in key for key in model.state_dict())
    model.step(0.1)
    torch.testing.assert_close(receiver.total, torch.full_like(receiver.total, 0.275))


def test_scalar_parameter_binding_broadcasts_and_preserves_gradient():
    model = _model(batch=2, regional=True)
    model.bind_input("receiver.DE.ReceiverState.incoming", "provider.gain")
    initial = model.mech.receiver.total.clone()
    model.step(0.1)
    torch.testing.assert_close(model.mech.receiver.total, initial + 0.3)
    gradient = torch.autograd.grad(
        model.mech.receiver.total.sum(), model.mech.provider.gain_param
    )[0]
    torch.testing.assert_close(
        gradient, torch.full_like(gradient, 0.1 * model.mech.receiver.total.numel())
    )


def test_input_bindings_survive_rebuild_deepcopy_and_state_dict():
    model = _model(regional=True)
    model.build(force_rebuild=True)
    model.initialize()
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict())
    model.step(0.1)
    clone.step(0.1)
    torch.testing.assert_close(clone.mech.receiver.total, model.mech.receiver.total)
    clone.mech.provider.drive = clone.mech.provider.drive + 10
    assert not torch.equal(
        clone.mech._resolve_state_inputs("receiver")["ReceiverState"]["incoming"],
        model.mech._resolve_state_inputs("receiver")["ReceiverState"]["incoming"],
    )


@pytest.mark.parametrize("built", (False, True))
def test_legacy_serialized_population_defaults_to_no_input_bindings(built):
    if built:
        model = _model(bind=False)
        del model.mech._input_bindings
        del model.mech._input_binding_plan
    else:
        with dn.ctx(JIT=0):
            model = dn.Population()
            model.insert(_Provider.rename("provider"))
            model.insert(_Receiver.rename("receiver"))
    # Match a pickle created before input-binding configuration existed.
    del model._input_bindings
    restored = copy.deepcopy(model)
    assert restored._input_bindings == ()
    restored.initialize()
    assert restored.mech._input_bindings == ()
    assert restored.mech._input_binding_plan == ()
    restored.bind_input("receiver.DE.ReceiverState.incoming", "provider.drive")
    initial = restored.mech.receiver.total.clone()
    restored.step(0.1)
    torch.testing.assert_close(restored.mech.receiver.total, initial + 0.21)


def test_serialized_handler_preserves_current_input_bindings():
    model = _model(regional=True)
    restored = copy.deepcopy(model).mech
    assert restored._input_bindings == model._input_bindings
    assert restored._input_binding_plan == model.mech._input_binding_plan
    actual = restored._resolve_state_inputs("receiver")["ReceiverState"]["incoming"]
    expected = model.mech._resolve_state_inputs("receiver")["ReceiverState"]["incoming"]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bound", (False, True))
@pytest.mark.parametrize("average_drift", (False, True))
def test_euler_heun_keeps_inputs_outside_assigned_output_schema(bound, average_drift):
    class HeunReceiverState(State):
        State.STATE("total")
        State.INPUT("incoming", default=0.25)
        State.ASSIGNED("rhs", "sigma")
        State.DERIVATIVE("total' = rhs + incoming")
        State.DIFFUSION("total = sigma * incoming")
        State.METHOD("euler_heun", average_drift=average_drift)

        def assigned_values(self, v, values):
            return {"rhs": 0.5 * values["total"], "sigma": 0.1 * values["total"]}

        def state_defaults(self, v, values):
            return {"total": torch.ones_like(v)}

    class HeunReceiver(Mechanism):
        Mechanism.STATE_BUNDLE(HeunReceiverState)

    with dn.ctx(JIT=0):
        model = dn.Population(dtype=torch.float64)
        model.insert(_Provider.rename("provider"))
        model.insert(HeunReceiver.rename("receiver"))
        if bound:
            model.bind_input("receiver.DE.HeunReceiverState.incoming", "provider.drive")
        model.initialize()
        state = model.mech.receiver.DE["HeunReceiverState"]
        state.total_dW_rng.randn = lambda shape, *, device, dtype: torch.ones(
            shape, device=device, dtype=dtype
        )
        model.step(0.1)

    incoming = 2.1 if bound else 0.25
    dt = torch.tensor(0.1, dtype=torch.float64)
    dw = dt.sqrt()
    drift_old = 0.5 + incoming
    diffusion_old = 0.1 * incoming
    predicted = 1 + drift_old * dt + diffusion_old * dw
    drift = (
        0.5 * (drift_old + 0.5 * predicted + incoming) if average_drift else drift_old
    )
    expected = 1 + drift * dt + 0.5 * (diffusion_old + 0.1 * predicted * incoming) * dw
    torch.testing.assert_close(
        model.mech.receiver.total, expected.expand_as(model.mech.receiver.total)
    )


def test_packed_input_projection_survives_meta_key_relocation(monkeypatch):
    from dendra.models.mechanisms._support import SupportKind, SupportSpec

    with dn.ctx(JIT=0):
        model = dn.Population(N=2, C=4, dtype=torch.float64)
        model[torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2, 1, 3])].insert(
            _Provider.rename("provider")
        )
        model[torch.tensor([0, 1]), torch.tensor([2, 1])].insert(
            _Receiver.rename("receiver")
        )
        model.bind_input("receiver.DE.ReceiverState.incoming", "provider.drive")
        model.initialize()
    model.mech.provider.drive = torch.tensor(
        [4.0, 7.0, 11.0, 16.0], dtype=torch.float64
    )
    checkpoint = copy.deepcopy(model.state_dict())
    expected_inputs = torch.tensor([7.0, 11.0], dtype=torch.float64)
    established_plan = model.mech._input_binding_plan

    def forbid_packed_materialization(*args, **kwargs):
        raise AssertionError("Relocation must not inspect packed selector values.")

    with monkeypatch.context() as patch:
        patch.setattr(
            SupportSpec, "materialized_flat_indices", forbid_packed_materialization
        )
        model.to(device="meta")
        model.to_empty(device="cpu")
        model.mech.make_maps()
        assert model.mech._input_binding_plan == established_plan
        assert all(
            mechanism.support_spec.kind is SupportKind.PACKED_FLAT
            and not mechanism._support_key_values_valid
            for mechanism in model.mech.mechanisms.values()
        )
        with pytest.raises(ValueError, match="support keys are unavailable"):
            model.bind_input("receiver.DE.ReceiverState.incoming", "provider.gain")

    model.load_state_dict(checkpoint)
    actual = model.mech._resolve_state_inputs("receiver")["ReceiverState"]["incoming"]
    torch.testing.assert_close(actual, expected_inputs, rtol=0, atol=0)
    model.step(0.1)
    torch.testing.assert_close(
        model.mech.receiver.total, 0.25 + 0.1 * (expected_inputs + 0.1)
    )


@pytest.mark.parametrize(
    "target, source",
    (
        ("receiver.incoming", "provider.drive"),
        ("receiver.DE.ReceiverState.missing", "provider.drive"),
        ("receiver.DE.ReceiverState.incoming", "missing.drive"),
        ("receiver.DE.ReceiverState.incoming", "provider.diam"),
        ("receiver.DE.ReceiverState.incoming", "provider._buffers.drive"),
    ),
)
def test_invalid_bindings_do_not_replace_existing_plan(target, source):
    model = _model()
    original = model._input_bindings
    with pytest.raises((ValueError, TypeError)):
        model.bind_input(target, source)
    assert model._input_bindings == original


def test_source_property_is_rejected_without_evaluation():
    model = _model(bind=False)
    calls = []
    model.mech.provider.setreference("drive", lambda: calls.append(True))
    with pytest.raises(ValueError, match="properties or callbacks"):
        model.bind_input("receiver.DE.ReceiverState.incoming", "provider.drive")
    assert calls == []


def test_uncovered_source_support_is_rejected():
    with dn.ctx(JIT=0):
        model = dn.Population(N=1, C=4)
        model[:, :2].insert(_Provider.rename("provider"))
        model.insert(_Receiver.rename("receiver"))
        model.bind_input("receiver.DE.ReceiverState.incoming", "provider.drive")
        with pytest.raises(ValueError, match="does not cover"):
            model.build()


def test_duplicate_source_support_is_rejected():
    with dn.ctx(JIT=0):
        model = dn.Population(N=1, C=4)
        model[:, 0].insert(_Provider.rename("provider"), copies=2)
        model[:, 0].insert(_Receiver.rename("receiver"))
        model.bind_input("receiver.DE.ReceiverState.incoming", "provider.drive")
        with pytest.raises(ValueError, match="ambiguous duplicate"):
            model.build()


@pytest.mark.parametrize("direct", (False, True))
def test_packed_binding_rejection_preserves_source_population(direct):
    from dendra.models.multi import MultiPopulation, concat_models

    model = _model()
    tensors = {name: value.detach().clone() for name, value in model.named_buffers()}
    mechanisms = model.mech
    with pytest.raises(ValueError, match="packed input-binding"):
        if direct:
            MultiPopulation(component=model)
        else:
            concat_models({"component": model})
    assert model.mech is mechanisms
    assert dict(model.named_buffers()).keys() == tensors.keys()
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, tensors[name])


@pytest.mark.parametrize("role", ("STATE", "CARRY", "ASSIGNED", "RANGE"))
def test_input_role_conflicts_are_rejected(role):
    with pytest.raises(ValueError, match="INPUT"):

        class Invalid(State):
            State.INPUT("input_value")
            if role == "RANGE":
                State.RANGE(input_value=1.0)
            else:
                getattr(State, role)("input_value")


def test_input_cannot_shadow_parent_mechanism_carry():
    class InputState(State):
        State.INPUT("shared")

    with pytest.raises(ValueError, match="INPUT"):

        class Invalid(Mechanism):
            Mechanism.STATE_BUNDLE(InputState)
            Mechanism.CARRY("shared")


@pytest.mark.parametrize("name", ("class", "_private", "bad.name", 3))
def test_input_names_must_be_public_identifiers(name):
    with pytest.raises(ValueError, match="identifiers"):

        class Invalid(State):
            State.INPUT(name)


def test_input_cannot_be_returned_as_a_state_update():
    class InvalidState(State):
        State.STATE("x")
        State.INPUT("incoming")
        State.DERIVATIVE("x' = 0")

        def state_defaults(self, v, values):
            return {"x": torch.zeros_like(v)}

        def advance(self, v, dt, values):
            return {"x": values["x"], "incoming": torch.ones_like(v)}

    class Invalid(Mechanism):
        Mechanism.STATE_BUNDLE(InvalidState)

    with dn.ctx(JIT=0):
        model = dn.Population()
        model.insert(Invalid)
        model.initialize()
        with pytest.raises(KeyError, match="undeclared"):
            model.step(0.1)
