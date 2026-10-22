"""Shared event-cache restore preserves native history and gradient policy."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.networks.test_state_cache_translation import (
    EVENT_DELAYS,
    EVENT_WEIGHTS,
    OLD_DT,
    _build_event_connection,
    _run_event_prefix,
)

pytestmark = pytest.mark.cpu


def _equal(actual, expected):
    left, spec = pytree.tree_flatten(actual)
    right, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            assert value == reference


def _packet(
    backend,
    *,
    recording=True,
    training=False,
    train_backend="dense",
    diff_spiking=False,
    new_dt=0.05,
):
    source_net, source = _build_event_connection(backend, OLD_DT)
    if recording:
        source.enable_state_cache_recording()
    _run_event_prefix(source_net, source)
    cache = source.state_cache()
    target_net, target = _build_event_connection(
        backend,
        new_dt,
        weights=EVENT_WEIGHTS * 1.3,
        delays=EVENT_DELAYS * 0.8,
        training=training,
        train_backend=train_backend,
        diff_spiking=diff_spiking,
    )
    return target_net, target, cache


@pytest.mark.parametrize(
    ("backend", "recording", "training", "train_backend", "diff_spiking"),
    [
        ("dense", False, False, "dense", False),
        ("dense", True, False, "dense", False),
        ("dense", True, True, "source_history", False),
        ("dense", True, True, "source_history", True),
        ("bitpacked_history", True, False, "dense", False),
        ("bitpacked_history", True, True, "dense", False),
        ("bitpacked_history", True, True, "source_history", False),
        ("bitpacked_history", True, True, "source_history", True),
        ("sparse_calendar", False, False, "dense", False),
        ("sparse_calendar", True, False, "dense", False),
    ],
)
def test_functional_event_cache_restore_matches_native(
    backend, recording, training, train_backend, diff_spiking
):
    network, native, cache = _packet(
        backend,
        recording=recording,
        training=training,
        train_backend=train_backend,
        diff_spiking=diff_spiking,
    )
    plan, tensors = dn.func.make_functional_netcon(native, pending_capacity=64)
    normalized = plan.normalize_cache(cache)
    start = network.t.new_tensor(0.3)
    state = plan.initial_state(tensors.state, t=start)
    frozen = pytree.tree_map_only(torch.Tensor, torch.clone, (state, normalized))
    actual = plan.restore_cache(
        {"weights": tensors.weights, "delays": tensors.delays},
        state,
        normalized,
        t=start,
    )
    network.t = start
    native.initialize_from_state_cache(cache, rebuild_delays=False)
    expected = plan.extract(native).state
    _equal(actual, expected)
    _equal((state, normalized), frozen)


@pytest.mark.parametrize("diff_spiking", (False, True))
def test_source_history_cache_has_detached_gates_and_delivery_time_weight_gradients(
    diff_spiking,
):
    _, native, cache = _packet(
        "dense",
        training=True,
        train_backend="source_history",
        diff_spiking=diff_spiking,
    )
    cache["backend_state"]["gate_history"].requires_grad_()
    plan, tensors = dn.func.make_functional_netcon(native)
    normalized = plan.normalize_cache(cache)
    # Explicit normalized overrides retain the native detached-cache policy.
    normalized = normalized._replace(
        gate_history=normalized.gate_history.requires_grad_()
    )
    weight = tensors.weights.detach().clone().requires_grad_()
    state = plan.restore_cache(
        {"weights": weight, "delays": tensors.delays},
        plan.initial_state(tensors.state, t=0.0),
        normalized,
        t=0.0,
    )
    assert not state[plan._history_name].requires_grad
    total = weight.new_zeros(())
    for _ in range(native.max_delay_steps):
        state, due = plan.read_due(state, weights=weight, delays=tensors.delays)
        total = total + due.sum()
        state = plan.advance(
            weight, state, weight.new_zeros(plan.source_shape), delays=tensors.delays
        )
    gradient = torch.autograd.grad(total, weight)[0]
    assert gradient.count_nonzero() > 0
    assert normalized.gate_history.grad is None


def test_dense_gate_cache_materialization_detaches_current_weights():
    _, native, cache = _packet("dense")
    plan, tensors = dn.func.make_functional_netcon(native)
    weight = tensors.weights.detach().clone().requires_grad_()
    state = plan.restore_cache(
        {"weights": weight, "delays": None},
        plan.initial_state(tensors.state, t=0.0),
        plan.normalize_cache(cache),
        t=0.0,
    )
    assert state["delivery_buffer"].count_nonzero() > 0
    assert not state["delivery_buffer"].requires_grad


def test_legacy_tuple_cache_uses_detached_weighted_payloads():
    _, native, cache = _packet("dense", recording=False)
    legacy = (
        cache["dt"],
        torch.empty(0, dtype=torch.bool),
        torch.empty(0),
        cache["backend_state"]["delivery_buffer"].requires_grad_(),
    )
    plan, tensors = dn.func.make_functional_netcon(native)
    result = plan.restore_cache(
        {"weights": tensors.weights, "delays": None},
        plan.initial_state(tensors.state, t=0.0),
        plan.normalize_cache(legacy),
        t=0.0,
    )
    assert not result["delivery_buffer"].requires_grad
    assert result["delivery_buffer"].count_nonzero() > 0


@pytest.mark.parametrize("backend", ("dense", "sparse_calendar"))
def test_normalizing_an_event_packet_owns_detached_leaves_including_calendars(backend):
    _, native, raw = _packet(backend, recording=False)
    plan, _ = dn.func.make_functional_netcon(native)
    cache = plan.normalize_cache(raw)
    if cache.pending is None:
        incoming = cache.delivery_buffer.detach().clone().requires_grad_() * 1
        cache = cache._replace(delivery_buffer=incoming)
    else:
        incoming = cache.pending.values.detach().clone().requires_grad_() * 1
        cache = cache._replace(pending=cache.pending._replace(values=incoming))
    assert incoming.grad_fn is not None
    normalized = plan.normalize_cache(cache)
    _equal(normalized, cache)
    assert normalized is not cache
    for actual, original in zip(
        pytree.tree_leaves(normalized), pytree.tree_leaves(cache), strict=True
    ):
        if torch.is_tensor(original):
            assert actual is not original
            if original.numel():
                assert actual.data_ptr() != original.data_ptr()
            assert not actual.requires_grad
            assert actual.grad_fn is None
    saved = pytree.tree_map_only(torch.Tensor, torch.clone, normalized)
    with torch.no_grad():
        incoming.add_(7)
    _equal(normalized, saved)


def test_normalized_event_cache_restore_compiles_without_native_cloning():
    _, native, cache = _packet(
        "dense", training=True, train_backend="source_history", diff_spiking=True
    )
    plan, tensors = dn.func.make_functional_netcon(native)
    normalized = plan.normalize_cache(cache)
    initial = plan.initial_state(tensors.state, t=0.0)
    parameters = {"weights": tensors.weights, "delays": tensors.delays}
    compiled = torch.compile(plan.restore_cache, backend="aot_eager", fullgraph=True)
    actual = compiled(
        parameters,
        initial,
        normalized,
        t=initial["global_step"].new_zeros(()).to(torch.float64),
    )
    expected = plan.restore_cache(parameters, initial, normalized, t=0.0)
    _equal(actual, expected)


@pytest.mark.parametrize("lanes", (0, 3))
def test_normalized_cache_restore_accepts_mapped_weight_and_history_inputs(lanes):
    _, native, cache = _packet(
        "dense", training=True, train_backend="source_history", diff_spiking=True
    )
    plan, tensors = dn.func.make_functional_netcon(native)
    normalized = plan.normalize_cache(cache)
    initial = plan.initial_state(tensors.state, t=0.0)

    def restore(weight, history):
        return plan.restore_cache(
            {"weights": weight, "delays": tensors.delays},
            initial,
            normalized._replace(gate_history=history),
            t=0.0,
        )

    weights = tensors.weights.expand(lanes, -1)
    histories = torch.stack(
        (
            normalized.gate_history,
            normalized.gate_history * 0.5,
            normalized.gate_history * 2,
        )
    )[:lanes]
    actual = torch.vmap(restore)(weights, histories)
    for lane in range(lanes):
        _equal(
            {name: value[lane] for name, value in actual.items()},
            restore(weights[lane], histories[lane]),
        )


def test_gate_history_can_restore_into_bitpacked_inference():
    source_net, source = _build_event_connection(
        "bitpacked_history", OLD_DT, training=True, train_backend="dense"
    )
    source.enable_state_cache_recording()
    _run_event_prefix(source_net, source)
    cache = source.state_cache()
    assert "gate_history" in cache["backend_state"]
    _, target = _build_event_connection("bitpacked_history", OLD_DT)
    plan, tensors = dn.func.make_functional_netcon(target)
    normalized = plan.normalize_cache(cache)
    restored = plan.restore_cache(
        {"weights": tensors.weights, "delays": None},
        plan.initial_state(tensors.state, t=0.0),
        normalized,
        t=0.0,
    )
    target.initialize_from_state_cache(cache, rebuild_delays=False)
    _equal(restored, plan.extract(target).state)
    assert restored["spike_history_packed"].count_nonzero() > 0


@pytest.mark.parametrize("normalized", (False, True))
@pytest.mark.parametrize("nonbinary", (False, True))
def test_gate_to_hard_history_rejects_nonrepresentable_source_activity(
    nonbinary, normalized
):
    _, native, cache = _packet("dense", training=True, train_backend="source_history")
    plan, tensors = dn.func.make_functional_netcon(native)
    if normalized:
        cache = plan.normalize_cache(cache)
        gates = cache.gate_history.clone()
    else:
        gates = cache["backend_state"]["gate_history"].clone()
    gates[1, 0] = 0.5 if nonbinary else 1.0
    gates[1, 1] = 0.5 if nonbinary else 0.0
    if normalized:
        cache = cache._replace(gate_history=gates)
    else:
        cache["backend_state"]["gate_history"] = gates
    initial = plan.initial_state(tensors.state, t=0.0)
    parameters = {"weights": tensors.weights, "delays": tensors.delays}
    frozen = pytree.tree_map_only(
        torch.Tensor, torch.clone, (initial, cache, parameters)
    )
    with pytest.raises(ValueError, match="binary|same source"):
        plan.normalize_cache(cache)
    with pytest.raises(ValueError, match="binary|same source"):
        plan.restore_cache(parameters, initial, cache, t=0.0)
    _equal((initial, cache, parameters), frozen)


@pytest.mark.parametrize("channel", ("pending", "event_pending"))
@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("size", -1, "size"),
        ("size", 1000, "size"),
        ("overflow", True, "overflow"),
        ("slots", -1, "slot"),
        ("slots", 1000, "slot"),
        ("indices", -1, "index"),
        ("indices", 1000, "index"),
    ),
)
def test_normalized_pending_cache_rejects_invalid_live_prefix_atomically(
    channel, field, value, message
):
    from tests.func.test_netcon_pending_calendar import _netcon

    _, native = _netcon(scheduled=False)
    native.pre.gate.fill_(1.0)
    native.advance()
    plan, tensors = dn.func.make_functional_netcon(native, pending_capacity=64)
    cache = plan.normalize_cache(native.state_cache())
    calendar = getattr(cache, channel)
    assert calendar.size.item() > 0
    corrupt = getattr(calendar, field).clone()
    corrupt[0] = value
    cache = cache._replace(**{channel: calendar._replace(**{field: corrupt})})
    initial = plan.initial_state(tensors.state, t=0.0)
    parameters = {"weights": tensors.weights, "delays": tensors.delays}
    frozen = pytree.tree_map_only(
        torch.Tensor, torch.clone, (initial, cache, parameters)
    )
    native_before = pytree.tree_map_only(
        torch.Tensor, torch.clone, native.state_dict_for_checkpoint()
    )
    error = RuntimeError if field == "overflow" else ValueError
    with pytest.raises(error, match=message):
        plan.restore_cache(parameters, initial, cache, t=0.0)
    with pytest.raises(error, match=message):
        native.initialize_from_state_cache(cache, rebuild_delays=False)
    _equal((initial, cache, parameters), frozen)
    _equal(native.state_dict_for_checkpoint(), native_before)


def test_compiled_cache_restore_enforces_hard_source_representability():
    _, native, cache = _packet("dense", training=True, train_backend="source_history")
    plan, tensors = dn.func.make_functional_netcon(native)
    normalized = plan.normalize_cache(cache)
    parameters = {"weights": tensors.weights, "delays": tensors.delays}
    initial = plan.initial_state(tensors.state, t=0.0)
    compiled = torch.compile(plan.restore_cache, backend="aot_eager", fullgraph=True)
    actual = compiled(parameters, initial, normalized, t=0.0)
    _equal(actual, plan.restore_cache(parameters, initial, normalized, t=0.0))
    gates = normalized.gate_history.clone()
    gates[1, native.pre_idx == native.pre_idx[0]] = 0.5
    with pytest.raises(RuntimeError, match="binary"):
        compiled(parameters, initial, normalized._replace(gate_history=gates), t=0.0)


@pytest.mark.parametrize("invalid", ("size", "overflow"))
def test_mapped_pending_cache_preserves_invalid_status_for_host_checks(invalid):
    _, native, raw = _packet("sparse_calendar", recording=False)
    plan, tensors = dn.func.make_functional_netcon(native, pending_capacity=64)
    cache = plan.normalize_cache(raw)
    initial = plan.initial_state(tensors.state, t=0.0)
    parameters = {"weights": tensors.weights, "delays": tensors.delays}
    value = getattr(cache.pending, invalid)
    mapped = torch.stack(
        (value, torch.full_like(value, -1 if invalid == "size" else True))
    )

    def restore(replacement):
        return plan.restore_cache(
            parameters,
            initial,
            cache._replace(pending=cache.pending._replace(**{invalid: replacement})),
            t=0.0,
        )

    state = torch.vmap(restore)(mapped)
    assert state["pending_overflow"].flatten().tolist() == [False, True]
    assert state["pending_size"][1].item() == 0
    with pytest.raises(RuntimeError, match="overflow"):
        plan.check_overflow(state)


def test_native_event_cache_restore_detaches_buffers_and_supports_chaining():
    _, native, cache = _packet("dense")
    native.is_spiking.requires_grad_()
    assert native.initialize_from_state_cache(cache, rebuild_delays=False) is native
    assert not any(value.requires_grad for value in native.buffers())


def test_cache_restore_rejects_overridden_native_cache_adapter():
    _, native, cache = _packet("dense")
    plan, _ = dn.func.make_functional_netcon(native)
    native.initialize_from_state_cache = lambda *args, **kwargs: None
    with pytest.raises(
        dn.func.FunctionalizationError, match="canonical initialize_from_state_cache"
    ):
        plan.normalize_cache(cache)
