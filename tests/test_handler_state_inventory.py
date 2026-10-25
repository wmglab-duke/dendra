"""Live handler bindings share checkpoint ownership without snapshot work."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _population as func_api
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.rng import RNGModule
from tests.test_declared_state_inventory import (
    SHAPE,
    _assert_unchanged,
    _DeclaredMechanism,
)
from tests.test_declared_state_inventory import _handler as declared_handler
from tests.test_declared_state_inventory import (
    _module,
    _ObservedNames,
    _registered_snapshot,
)
from tests.test_shared_field_inventory import _handler as shared_handler

pytestmark = pytest.mark.cpu
DT = 0.01


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=True,
            DEVICE="cpu",
            DTYPE=torch.float64,
            CHECK_PURITY=False,
        ),
    ):
        yield


def _handler(kind):
    if kind == "shared":
        return shared_handler()
    handler, mechanism = declared_handler()
    if kind == "alias":
        mechanism.DE["alias"] = mechanism.DE["_FirstState"]
    return handler


def _legacy_bindings(handler):
    plan = handler._preflight_mutable_state_dict(handler.mutable_state_dict())
    return tuple(
        (plan["assignment_keys"][(id(owner), name)], owner, name)
        for owner, name, _value in plan["assignments"]
    )


def _assert_bindings(actual, expected):
    assert [(key, name) for key, _owner, name in actual] == [
        (key, name) for key, _owner, name in expected
    ]
    assert all(
        left[1] is right[1] for left, right in zip(actual, expected, strict=True)
    )


def _forbidden(*args, **kwargs):
    raise AssertionError(
        "live inventory entered a snapshot, conversion, or sampling path"
    )


@pytest.mark.parametrize("kind", ("declared", "alias", "shared"))
def test_live_bindings_match_checkpoint_preflight_owners_and_first_key_order(kind):
    handler = _handler(kind)
    checkpoint_keys = tuple(handler.mutable_state_dict())
    public = tuple(func_api._canonical_handler_keys(handler).items())
    expected = _legacy_bindings(handler)
    _assert_bindings(handler._mutable_state_bindings(), expected)
    assert tuple(handler.mutable_state_dict()) == checkpoint_keys
    assert tuple(func_api._canonical_handler_keys(handler).items()) == public
    assert checkpoint_keys[0] == "__dendra_mechanism_support_identity__"
    binding_keys = tuple(dict.fromkeys(key for key, _owner, _name in expected))
    assert set(binding_keys).issubset(checkpoint_keys)
    assert binding_keys == tuple(
        key for key in checkpoint_keys[1:] if key in binding_keys
    )


@pytest.mark.parametrize("kind", ("declared", "shared"))
def test_deterministic_lookup_and_public_projection_do_not_snapshot_preflight_or_clone(
    kind, monkeypatch
):
    handler = _handler(kind)
    expected = _legacy_bindings(handler)
    public = tuple(func_api._canonical_handler_keys(handler).items())
    before = _registered_snapshot(handler)
    global_rng = torch.get_rng_state().clone()
    with monkeypatch.context() as guard:
        guard.setattr(handler, "mutable_state_dict", _forbidden)
        guard.setattr(handler, "_preflight_mutable_state_dict", _forbidden)
        guard.setattr(torch.Tensor, "clone", _forbidden)
        guard.setattr(torch, "clone", _forbidden)
        actual = handler._mutable_state_bindings()
        actual_public = tuple(func_api._canonical_handler_keys(handler).items())
    _assert_bindings(actual, expected)
    assert actual_public == public
    _assert_unchanged(before)
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("detach", (False, True))
def test_live_aliases_snapshot_gradients_and_detached_restore_remain_distinct(detach):
    handler, mechanism = declared_handler()
    shared = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    first = mechanism.DE["_FirstState"]
    mechanism.a = mechanism.z_carry = mechanism.i_ = first.a_memory = shared
    keys = (
        "declared.a",
        "declared.z_carry",
        "declared.i_",
        "declared.DE._FirstState.a_memory",
    )
    bindings = {
        key: (owner, name) for key, owner, name in handler._mutable_state_bindings()
    }
    assert all(getattr(*bindings[key]) is shared for key in keys)
    snapshot = handler.mutable_state_dict()
    assert all(snapshot[key] is snapshot[keys[0]] for key in keys)
    assert snapshot[keys[0]] is not shared and snapshot[keys[0]].requires_grad
    if detach:
        memo = {}
        for key, value in snapshot.items():
            if torch.is_tensor(value):
                if id(value) not in memo:
                    memo[id(value)] = value.detach()
                snapshot[key] = memo[id(value)]
    handler.restore_mutable_state_dict(snapshot)
    restored = [getattr(*bindings[key]) for key in keys]
    assert all(value is restored[0] for value in restored)
    assert restored[0] is not snapshot[keys[0]]
    assert restored[0].requires_grad is (not detach)
    if not detach:
        gradient = torch.autograd.grad(sum(value.sum() for value in restored), shared)[
            0
        ]
        torch.testing.assert_close(gradient, torch.full_like(shared, 4.0))


@pytest.mark.parametrize("invalid", ("missing", "shape", "conflict"))
def test_live_admission_keeps_legacy_state_local_errors_without_mutating_owners(
    invalid,
):
    handler = shared_handler()
    state = handler.mechanisms["zeta"].DE["_LocalState"]
    if invalid == "missing":
        del state._buffers["read"]
    elif invalid == "shape":
        state.read = torch.zeros((1, 3), dtype=torch.float64)
    else:
        handler.mechanisms["alpha"].DE["_LocalState"] = state
    before = _registered_snapshot(handler)
    with pytest.raises((TypeError, ValueError)) as legacy:
        _legacy_bindings(handler)
    with pytest.raises(type(legacy.value)) as actual:
        handler._mutable_state_bindings()
    assert str(actual.value) == str(legacy.value)
    _assert_unchanged(before)


@pytest.mark.parametrize(
    "kind",
    (
        "meta",
        pytest.param(
            "quantized",
            marks=pytest.mark.filterwarnings(
                r"ignore:torch\.quantize_per_tensor, torch\.quantize_per_channel and other quantized tensor creation functions.*are deprecated:UserWarning"
            ),
        ),
    ),
)
def test_special_tensor_locals_keep_whole_legacy_conversion_errors(kind, monkeypatch):
    handler = shared_handler()
    owner = handler.mechanisms["zeta"]
    owner.read = (
        torch.empty(SHAPE, device="meta", dtype=torch.float64)
        if kind == "meta"
        else torch.quantize_per_tensor(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        )
    )
    assert type(owner.read) is torch.Tensor
    before = _registered_snapshot(handler)
    with pytest.raises((RuntimeError, TypeError)) as legacy:
        _legacy_bindings(handler)
    snapshot, preflight = (
        handler.mutable_state_dict,
        handler._preflight_mutable_state_dict,
    )
    calls = []

    def capture():
        calls.append("snapshot")
        return snapshot()

    def validate(values):
        calls.append("preflight")
        return preflight(values)

    monkeypatch.setattr(handler, "mutable_state_dict", capture)
    monkeypatch.setattr(handler, "_preflight_mutable_state_dict", validate)
    with pytest.raises(type(legacy.value)) as actual:
        handler._mutable_state_bindings()
    assert str(actual.value) == str(legacy.value)
    assert calls and calls[0] == "snapshot" and "preflight" in calls
    _assert_unchanged(before)


def test_ordinary_cross_dtype_mirrors_keep_native_binding_admission():
    handler = shared_handler()
    owner = handler.mechanisms["zeta"]
    owner.read = owner.read.float()
    mirror = owner.DE["_LocalState"]
    assert owner.read.dtype == torch.float32 and mirror.read.dtype == torch.float64
    before = _registered_snapshot(handler)
    _assert_bindings(handler._mutable_state_bindings(), _legacy_bindings(handler))
    _assert_unchanged(before)


def test_late_native_restore_validation_keeps_all_live_tensor_objects_and_values():
    handler, _mechanism = declared_handler()
    snapshot = handler.mutable_state_dict()
    snapshot["declared.a"] = snapshot["declared.a"] + 7
    snapshot["declared.j_"] = torch.zeros((1, 3), dtype=torch.float64)
    before = _registered_snapshot(handler)
    with pytest.raises(ValueError, match=r"declared\.j_.*shape"):
        handler.restore_mutable_state_dict(snapshot)
    _assert_unchanged(before)


class _Noise(Mechanism):
    Mechanism.CARRY("scratch")
    Mechanism.RANGENOISE("eta", distribution="normal", seed=137)


class _ClassRNG(Mechanism):
    Mechanism.CARRY("scratch")
    Mechanism.RNG(stream=139)


@pytest.mark.parametrize("kind", ("noise", "rng", "delay", "metadata"))
def test_dynamic_metadata_keeps_whole_handler_legacy_bindings_without_sampling(
    kind, monkeypatch
):
    handler, mechanism = declared_handler(process=False)
    if kind in ("noise", "rng"):
        dynamic = _module(_Noise if kind == "noise" else _ClassRNG, "dynamic")
        handler = MechanismHandler(
            mechanism.celsius,
            torch.ones(SHAPE, dtype=torch.float64),
            {"declared": mechanism, "dynamic": dynamic},
        )
    elif kind == "delay":
        mechanism.register_delayed_state(
            "signal", torch.zeros(SHAPE, dtype=torch.float64), 1, mode="shift"
        )
    else:
        mechanism._carry = _ObservedNames(mechanism._carry, [], "carry")
    expected = _legacy_bindings(handler)
    before = _registered_snapshot(handler)
    streams = [owner for owner in handler.modules() if isinstance(owner, RNGModule)]
    rng_frames = [
        (stream, stream._cpu_gen, stream._cpu_gen.get_state().clone())
        for stream in streams
    ]
    global_rng = torch.get_rng_state().clone()
    snapshot, preflight = (
        handler.mutable_state_dict,
        handler._preflight_mutable_state_dict,
    )
    calls = []

    def capture():
        calls.append("snapshot")
        return snapshot()

    def validate(values):
        calls.append("preflight")
        return preflight(values)

    monkeypatch.setattr(handler, "mutable_state_dict", capture)
    monkeypatch.setattr(handler, "_preflight_mutable_state_dict", validate)
    monkeypatch.setattr(RNGModule, "rand", _forbidden)
    monkeypatch.setattr(RNGModule, "randn", _forbidden)
    actual = handler._mutable_state_bindings()
    _assert_bindings(actual, expected)
    assert calls and calls[0] == "snapshot" and "preflight" in calls
    _assert_unchanged(before)
    assert torch.equal(torch.get_rng_state(), global_rng)
    for stream, generator, frame in rng_frames:
        assert stream._cpu_gen is generator
        assert torch.equal(generator.get_state(), frame)


@pytest.mark.parametrize("field", ("support_map", "support_spec"))
def test_support_descriptors_wait_for_legacy_snapshot_before_authored_access(
    field, monkeypatch
):
    handler, mechanism = declared_handler()
    if field == "support_spec":
        mechanism.support_map = None
    before = _registered_snapshot(handler)
    events = []
    error = RuntimeError("authored support descriptor")
    snapshot = handler.mutable_state_dict

    def support(owner):
        events.append("support")
        raise error

    def capture():
        events.append("snapshot")
        return snapshot()

    monkeypatch.setattr(type(mechanism), field, property(support), raising=False)
    with pytest.raises(RuntimeError) as legacy:
        snapshot()
    assert legacy.value is error
    events.clear()
    monkeypatch.setattr(handler, "mutable_state_dict", capture)
    with pytest.raises(RuntimeError) as actual:
        handler._mutable_state_bindings()
    assert actual.value is legacy.value
    assert events == ["snapshot", "support"]
    _assert_unchanged(before)


def test_population_layout_carry_gradients_and_native_commit_continue_together(
    monkeypatch,
):
    model = dn.SingleCompartment(
        N=1, C=2, dtype=torch.float64, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(_DeclaredMechanism.rename("declared"))
    model.initialize()
    model.train()
    expected = tuple(func_api._canonical_handler_keys(model.mech).items())
    with monkeypatch.context() as guard:
        guard.setattr(model.mech, "mutable_state_dict", _forbidden)
        guard.setattr(model.mech, "_preflight_mutable_state_dict", _forbidden)
        guard.setattr(torch.Tensor, "clone", _forbidden)
        guard.setattr(torch, "clone", _forbidden)
        layout = func_api._state_layout(model)
    projected = [leaf for leaf in layout if leaf.checkpoint_key in dict(expected)]
    assert [(leaf.checkpoint_key, leaf.public_path) for leaf in projected] == list(
        expected
    )
    nested = next(
        leaf
        for leaf in projected
        if leaf.checkpoint_key == "declared.DE._FirstState.a_memory"
    )
    assert nested.mapping_slots == (
        ("integrator.mech.mechanisms.declared.DE._FirstState", "a_memory"),
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    value = (
        tensors.state["mechanisms"]["declared"]["a"].detach().clone().requires_grad_()
    )
    tensors.state["mechanisms"]["declared"]["a"] = value
    tensors.state["mechanism_buffers"]["declared"]["z_carry"] = value
    tensors.state["state_buffers"]["declared"]["_FirstState"]["a_memory"] = value
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    final, _auxiliary = functional.step(tensors.parameters, prepared, tensors.state)
    loss = (
        final["mechanisms"]["declared"]["a"].sum()
        + final["mechanism_buffers"]["declared"]["z_carry"].sum()
        + final["state_buffers"]["declared"]["_FirstState"]["a_memory"].sum()
    )
    torch.testing.assert_close(
        torch.autograd.grad(loss, value)[0], torch.full_like(value, 3.0)
    )
    functional.commit_state_(model, final)
    continued, _auxiliary = functional.step(tensors.parameters, prepared, final)
    model.step(dt=DT)
    torch.testing.assert_close(model.v, continued["integrator"]["v"])
    torch.testing.assert_close(
        model.mech.declared.a, continued["mechanisms"]["declared"]["a"]
    )
