"""Declared checkpoint slots retain ownership, order, and restoration policy."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population import _canonical_handler_keys
from dendra.models.mechanisms import MaterialProcess, Mechanism, State
from dendra.models.mechanisms import _handler as handler_api
from dendra.models.mechanisms._handler import MechanismHandler

pytestmark = pytest.mark.cpu
SHAPE = (1, 2)


class _FirstState(State):
    State.STATE("z", "a")
    State.CARRY("z_memory", "a_memory")
    State.DERIVATIVE("z' = 0.0 * z", "a' = 0.0 * a")


class _SecondState(State):
    State.STATE("b")
    State.CARRY("b_memory")
    State.DERIVATIVE("b' = 0.0 * b")


class _DeclaredMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_FirstState, _SecondState)
    Mechanism.CARRY("z_carry", "a_carry")
    Mechanism.NONSPECIFIC_CURRENT("j", "i")
    Mechanism.SAVE_CURRENT("j", "i")

    def i(self, v):
        return 0.0 * v

    def j(self, v):
        return 0.0 * v


class _EmptyProcess(MaterialProcess):
    material_fields = ()

    def prepare_materials(self, fields, dt):
        del fields, dt
        return ()

    def material_updates(self, fields, prepared):
        del prepared
        return fields


class _CarryHistory(Mechanism):
    Mechanism.CARRY("history", shape="deferred")


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64),
    ):
        yield


def _module(kind, name):
    celsius = torch.full(SHAPE, 34.0, dtype=torch.float64)
    return kind(name, celsius, torch.ones_like(celsius), SHAPE, SHAPE)


def _handler(*, process=True):
    mechanism = _module(_DeclaredMechanism, "declared")
    modules = {"declared": mechanism}
    if process:
        modules["empty"] = _module(_EmptyProcess, "empty")
    handler = MechanismHandler(
        mechanism.celsius, torch.ones_like(mechanism.celsius), modules
    )
    return handler, mechanism


def _registered_snapshot(module):
    return tuple(
        (owner, name, value, value.detach().clone())
        for owner in module.modules()
        for name, value in owner._buffers.items()
        if value is not None
    )


def _assert_unchanged(snapshot):
    for owner, name, original, expected in snapshot:
        assert owner._buffers[name] is original
        torch.testing.assert_close(owner._buffers[name], expected, rtol=0, atol=0)


def test_native_and_functional_slot_orders_keep_their_distinct_contracts():
    handler, mechanism = _handler()
    assert tuple(handler.mutable_state_dict()) == (
        "__dendra_mechanism_support_identity__",
        "declared.a",
        "declared.z",
        "declared.DE._FirstState.a_memory",
        "declared.DE._FirstState.z_memory",
        "declared.b",
        "declared.DE._SecondState.b_memory",
        "declared.z_carry",
        "declared.a_carry",
        "declared.i_",
        "declared.j_",
    )
    assert tuple(_canonical_handler_keys(handler).items()) == (
        ("declared.a", ("mechanisms", "declared", "a")),
        ("declared.z", ("mechanisms", "declared", "z")),
        ("declared.b", ("mechanisms", "declared", "b")),
        ("declared.a_carry", ("mechanism_buffers", "declared", "a_carry")),
        ("declared.z_carry", ("mechanism_buffers", "declared", "z_carry")),
        ("declared.i_", ("mechanism_buffers", "declared", "i_")),
        ("declared.j_", ("mechanism_buffers", "declared", "j_")),
        (
            "declared.DE._FirstState.a_memory",
            ("state_buffers", "declared", "_FirstState", "a_memory"),
        ),
        (
            "declared.DE._FirstState.z_memory",
            ("state_buffers", "declared", "_FirstState", "z_memory"),
        ),
        (
            "declared.DE._SecondState.b_memory",
            ("state_buffers", "declared", "_SecondState", "b_memory"),
        ),
    )
    first, second = mechanism.DE.values()
    assert tuple(
        (key, owner, name) for key, owner, name in handler._mutable_state_bindings()
    ) == (
        ("declared.a", mechanism, "a"),
        ("declared.z", mechanism, "z"),
        ("declared.DE._FirstState.a_memory", first, "a_memory"),
        ("declared.DE._FirstState.z_memory", first, "z_memory"),
        ("declared.b", mechanism, "b"),
        ("declared.DE._SecondState.b_memory", second, "b_memory"),
        ("declared.z_carry", mechanism, "z_carry"),
        ("declared.a_carry", mechanism, "a_carry"),
        ("declared.i_", mechanism, "i_"),
        ("declared.j_", mechanism, "j_"),
    )


class _ObservedNames(list):
    def __init__(self, values, trace, label):
        super().__init__(values)
        self.trace, self.label = trace, label

    def __iter__(self):
        self.trace.append(self.label)
        return super().__iter__()


@pytest.mark.parametrize("operation", ("snapshot", "preflight"))
def test_native_declaration_reads_retain_stochastic_phase_boundaries(
    operation, monkeypatch
):
    handler, mechanism = _handler(process=False)
    checkpoint = handler.mutable_state_dict()
    trace = []
    for state_name, state in mechanism.DE.items():
        monkeypatch.setattr(
            state, "_state", _ObservedNames(state._state, trace, f"values:{state_name}")
        )
        monkeypatch.setattr(
            state, "_carry", _ObservedNames(state._carry, trace, f"carry:{state_name}")
        )
    monkeypatch.setattr(
        mechanism, "_carry", _ObservedNames(mechanism._carry, trace, "carry:declared")
    )
    monkeypatch.setattr(
        mechanism, "_save", _ObservedNames(mechanism._save, trace, "save:declared")
    )
    original = handler_api._stochastic_rng_names

    def observed_rng_names(owner):
        label = "declared" if owner is mechanism else type(owner).__name__
        trace.append(f"stochastic:{label}")
        return original(owner)

    monkeypatch.setattr(handler_api, "_stochastic_rng_names", observed_rng_names)
    if operation == "snapshot":
        handler.mutable_state_dict()
    else:
        handler._preflight_mutable_state_dict(checkpoint)
    assert trace == [
        "stochastic:declared",
        "values:_FirstState",
        "carry:_FirstState",
        "stochastic:_FirstState",
        "values:_SecondState",
        "carry:_SecondState",
        "stochastic:_SecondState",
        "carry:declared",
        "save:declared",
    ]


def test_snapshot_and_restore_keep_cross_role_aliases_and_gradient_connectivity():
    handler, mechanism = _handler()
    shared = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    mechanism.a = shared
    mechanism.z_carry = shared
    mechanism.i_ = shared
    first = mechanism.DE["_FirstState"]
    first.a_memory = shared
    snapshot = handler.mutable_state_dict()
    keys = (
        "declared.a",
        "declared.z_carry",
        "declared.i_",
        "declared.DE._FirstState.a_memory",
    )
    assert all(snapshot[key] is snapshot[keys[0]] for key in keys)
    assert snapshot[keys[0]] is not shared
    handler.restore_mutable_state_dict(snapshot)
    restored = (mechanism.a, mechanism.z_carry, mechanism.i_, first.a_memory)
    assert all(value is restored[0] for value in restored)
    assert restored[0] is not snapshot[keys[0]]
    torch.testing.assert_close(
        torch.autograd.grad(sum(value.sum() for value in restored), shared)[0],
        torch.full_like(shared, 4.0),
    )


def test_legacy_checkpoint_can_omit_nested_state_carry_without_rebinding_it():
    handler, mechanism = _handler()
    snapshot = handler.mutable_state_dict()
    key = "declared.DE._FirstState.a_memory"
    snapshot.pop(key)
    first = mechanism.DE["_FirstState"]
    first.a_memory = torch.full_like(first.a_memory, 17.0)
    retained = first.a_memory
    mechanism.a = torch.full_like(mechanism.a, 23.0)
    handler.restore_mutable_state_dict(snapshot)
    assert first.a_memory is retained
    torch.testing.assert_close(first.a_memory, torch.full_like(retained, 17.0))
    torch.testing.assert_close(mechanism.a, snapshot["declared.a"])


@pytest.mark.parametrize("conflicting", (False, True))
def test_aliased_state_targets_keep_first_key_and_reject_conflicting_entries(
    conflicting,
):
    handler, mechanism = _handler()
    first = mechanism.DE["_FirstState"]
    mechanism.DE["alias"] = first
    snapshot = handler.mutable_state_dict()
    alias_key = "declared.DE.alias.a_memory"
    first_key = "declared.DE._FirstState.a_memory"
    assert snapshot[alias_key] is snapshot[first_key]
    before = _registered_snapshot(handler)
    if conflicting:
        snapshot[alias_key] = snapshot[alias_key] + 1.0
        with pytest.raises(ValueError) as caught:
            handler.restore_mutable_state_dict(snapshot)
        assert str(caught.value) == (
            "Mutable field 'a_memory' received conflicting snapshot entries."
        )
        _assert_unchanged(before)
    else:
        plan = handler._preflight_mutable_state_dict(snapshot)
        target = (id(first), "a_memory")
        assert plan["assignment_keys"][target] == first_key
        assert (
            sum(
                owner is first and name == "a_memory"
                for owner, name, _value in plan["assignments"]
            )
            == 1
        )
        _assert_unchanged(before)


def test_delay_storage_declared_as_carry_uses_saved_history_shape_on_restore():
    mechanism = _module(_CarryHistory, "history")
    like = torch.ones(SHAPE, dtype=torch.float64)
    mechanism.register_delayed_state(
        "signal", like, 1, mode="shift", buffer_name="history"
    )
    mechanism.delayed_state("signal", 2.0 * like, mode="shift")
    handler = MechanismHandler(mechanism.celsius, like, {"history": mechanism})
    snapshot = handler.mutable_state_dict()
    mechanism.register_delayed_state(
        "signal", like, 3, mode="shift", buffer_name="history"
    )
    assert mechanism.history.shape != snapshot["history.history"].shape
    handler.restore_mutable_state_dict(snapshot)
    torch.testing.assert_close(mechanism.history, snapshot["history.history"])
    assert mechanism._delayed_state_specs["signal"]["steps"] == 1
    before = _registered_snapshot(handler)
    invalid = copy.copy(snapshot)
    invalid["history.signal_delay_ptr"] = like.new_tensor(2, dtype=torch.long)
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        handler.restore_mutable_state_dict(invalid)
    _assert_unchanged(before)
