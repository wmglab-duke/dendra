"""Shared detector semantics and receive-before-observe projection phases."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn

pytestmark = pytest.mark.cpu
DT = 0.1
TAU = 0.12


def _netcon(
    *, mixed=False, training=False, tracked=False, fanout=False, batch=None, raw=False
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        compartments = 2 if fanout else 3
        pre = dn.Population(N=1, C=compartments, dtype=torch.float64)
        post = dn.Population(N=1, C=compartments, dtype=torch.float64)
        post.insert(expsyn.rename("syn"))
        net = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
            track_netcon_events=tracked,
        )
        threshold = (
            [-0.1, 0.2, -0.1, 0.2]
            if fanout
            else [0.0, float("nan") if mixed else 0.2, -0.1]
        )
        edges = 4 if fanout else 3
        connect = net.connect_dense if fanout else net.connect_one_to_one
        connect(
            pre[:],
            post[:],
            post.mech.syn,
            threshold=None if raw else torch.tensor(threshold, dtype=torch.float64),
            pre_var="v",
            weight=torch.arange(1, edges + 1, dtype=torch.float64),
            delay=torch.arange(edges, dtype=torch.float64) * DT,
        )
        if batch is not None:
            net.batch(batch)
        net.initialize(dt=DT)
        native = next(iter(net.synapses.values()))
        if training:
            native.set_diff_config(
                diff_weights=True,
                diff_delays=False,
                diff_spiking=not tracked,
                diff_scheduled_times=False,
                tau=TAU,
                train_delay_backend="dense",
            )
            native.train()
            native.initialize(reinit_weights=False, reinit_delays=False)
    return net, native


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, left_spec = pytree.tree_flatten(actual)
    right, right_spec = pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for value, reference in zip(left, right, strict=True):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize(
    "training,tracked", ((False, False), (False, True), (True, False), (True, True))
)
@pytest.mark.parametrize("mixed", (False, True))
@pytest.mark.parametrize("batch", (None, 2))
def test_threshold_tape_matches_native_states_and_surrogate_gradients(
    training, tracked, mixed, batch
):
    _, native = _netcon(mixed=mixed, training=training, tracked=tracked, batch=batch)
    projection, tensors = dn.func.make_functional_netcon(native)
    assert tensors.state["has_spiked"].dtype == torch.bool
    assert not tensors.state["has_spiked"].any()
    tape = torch.tensor(
        [
            [[0.0, 0.1, -0.2]],
            [[0.3, 0.2, 0.4]],
            [[0.4, 0.1, -0.2]],
            [[-0.1, 0.4, 0.0]],
            [[0.1, 0.4, 0.2]],
        ],
        dtype=native.pre_dtype,
    )
    tape = tape.reshape(5, *([1] * (len(projection.source_shape) - 1)), 3).expand(
        5, *projection.source_shape
    )
    functional_tape = tape.clone().requires_grad_()
    native_tape = tape.clone().requires_grad_()
    weights = tensors.weights.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native.weight.w = native_weights
    state = tensors.state
    functional_outputs = []
    native_outputs = []
    for values, native_values in zip(functional_tape, native_tape, strict=True):
        delivery_state, payload = projection.read_due(state)
        # Reading due traffic changes no detector or ring carry.
        for name in (
            "has_spiked",
            "delivery_buffer",
            "current_time_step",
            "global_step",
        ):
            torch.testing.assert_close(delivery_state[name], state[name])
        state = projection.advance(weights, delivery_state, values)
        native.pre.v = native_values
        if training:
            native.advance()
        else:
            with torch.no_grad():
                native.advance()
        _close(state, native.state_dict_for_checkpoint())
        functional_outputs.extend(
            (payload.flatten(), state["delivery_buffer"].flatten())
        )
        # Native expsyn receives today's full payload; pending ring validates all
        # individual enqueue gradients without depending on expsyn dynamics.
        native_outputs.append(native.delivery_buffer.flatten().clone())
    actual_loss = sum(value.sum() for value in functional_outputs[1::2])
    expected_loss = sum(value.sum() for value in native_outputs)
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_grad = torch.autograd.grad(
        actual_loss, (weights, functional_tape), allow_unused=True
    )
    if training:
        expected_grad = torch.autograd.grad(
            expected_loss, (native_weights, native_tape), allow_unused=True
        )
        for actual, expected in zip(actual_grad, expected_grad, strict=True):
            if actual is None or expected is None:
                assert actual is expected
            else:
                torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual_grad[0])
    if training and not tracked:
        assert torch.count_nonzero(actual_grad[1])


@pytest.mark.parametrize("raw", (False, True))
def test_step_composes_split_phases_without_mutating_caller_state(raw):
    _, native = _netcon(raw=raw, tracked=True)
    projection, tensors = dn.func.make_functional_netcon(native)
    state = _clone(tensors.state)
    state["delivery_buffer"] = state["delivery_buffer"] + 0.7
    state["event_queue"] = state["event_queue"] + 2
    original = _clone(state)
    values = native.delivery_buffer.new_full(projection.source_shape, 0.3)
    delivery_state, due = projection.read_due(state)
    assert torch.equal(delivery_state["events"], torch.full_like(state["events"], 2))
    next_state = projection.advance(tensors.weights, delivery_state, values)
    _close(projection.step(tensors.weights, state, values), (next_state, due))
    _close(state, original)
    assert ("has_spiked" in state) is not raw
    # A repeated read has no side effects; only advance consumes the current slot.
    _close(projection.read_due(state), (delivery_state, due))


def test_per_edge_thresholds_for_one_source_keep_independent_latches():
    _, native = _netcon(fanout=True)
    projection, tensors = dn.func.make_functional_netcon(native)
    values = native.delivery_buffer.new_full(projection.source_shape, 0.0)
    state, _ = projection.step(tensors.weights, tensors.state, values)
    selected = values.flatten().index_select(0, native.pre_idx)
    torch.testing.assert_close(state["has_spiked"], selected >= native.threshold)
    native.pre.v = values
    native.advance()
    _close(state, native.state_dict_for_checkpoint())
    assert native.pre_idx[0] == native.pre_idx[1]
    assert state["has_spiked"][0] != state["has_spiked"][1]
    state, _ = projection.step(tensors.weights, state, values + 0.3)
    native.pre.v = values + 0.3
    native.advance()
    _close(state, native.state_dict_for_checkpoint())


def test_surrogate_derivative_uses_current_value_and_detached_previous_latch():
    _, native = _netcon(training=True)
    projection, tensors = dn.func.make_functional_netcon(native)
    previous = torch.tensor([True, False, False])
    state = {**tensors.state, "has_spiked": previous}
    values = torch.tensor(
        [[0.05, 0.16, -0.08]], dtype=native.pre_dtype, requires_grad=True
    )
    updated = projection.advance(tensors.weights, state, values)
    gradient = torch.autograd.grad(updated["delivery_buffer"].sum(), values)[0]
    sigmoid = torch.sigmoid((values.detach().flatten() - native.threshold) / TAU)
    expected = tensors.weights * sigmoid * (1 - sigmoid) / TAU * (~previous)
    torch.testing.assert_close(gradient.flatten(), expected)
    assert gradient[0, 0] == 0
    assert not updated["has_spiked"].requires_grad


def test_frozen_threshold_metadata_does_not_capture_native_parameter_gradients():
    _, native = _netcon(training=True)
    native.threshold.requires_grad_()
    projection, tensors = dn.func.make_functional_netcon(native)
    weights = tensors.weights.detach().clone().requires_grad_()
    values = native.delivery_buffer.new_full(
        projection.source_shape, 0.15
    ).requires_grad_()
    updated = projection.advance(weights, tensors.state, values)
    weight_grad, value_grad, threshold_grad = torch.autograd.grad(
        updated["delivery_buffer"].sum(),
        (weights, values, native.threshold),
        allow_unused=True,
    )
    assert weight_grad is not None and value_grad is not None
    assert threshold_grad is None


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_split_threshold_projection_preserves_values_and_gradients(backend):
    with torch_compiler_warning_context():
        _, native = _netcon(training=True, mixed=True)
        projection, tensors = dn.func.make_functional_netcon(native)

        def evaluate(weights, state, values):
            state, payload = projection.read_due(state)
            state = projection.advance(weights, state, values)
            state, following = projection.read_due(state)
            return state, payload + following

        weights = tensors.weights.detach().clone().requires_grad_()
        values = native.delivery_buffer.new_tensor(
            [[0.03, 0.25, -0.08]]
        ).requires_grad_()
        kwargs = (
            {"options": {"cpp_wrapper": True, "compile_threads": 1}}
            if backend == "inductor"
            else {}
        )
        compiled = torch.compile(evaluate, backend=backend, fullgraph=True, **kwargs)
        actual = compiled(weights, tensors.state, values)
        expected = evaluate(weights, tensors.state, values)
        _close(actual, expected)
        actual_loss = actual[0]["delivery_buffer"].sum() + actual[1].sum()
        expected_loss = expected[0]["delivery_buffer"].sum() + expected[1].sum()
        _close(
            torch.autograd.grad(actual_loss, (weights, values)),
            torch.autograd.grad(expected_loss, (weights, values)),
        )


@pytest.mark.parametrize("lanes", (0, 2))
@pytest.mark.parametrize("mapped_state", (False, True))
def test_threshold_projection_vmap_including_empty_lanes(lanes, mapped_state):
    _, native = _netcon(training=True, mixed=True)
    projection, tensors = dn.func.make_functional_netcon(native)
    values = native.delivery_buffer.new_full(
        (lanes, *projection.source_shape), 0.15
    ).requires_grad_()
    weights = tensors.weights.repeat(lanes, 1).requires_grad_()
    state = (
        pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.unsqueeze(0).expand(lanes, *value.shape).clone(),
            tensors.state,
        )
        if mapped_state
        else tensors.state
    )

    def evaluate(weights, state, values):
        state, payload = projection.read_due(state)
        return projection.advance(weights, state, values), payload

    actual = torch.vmap(evaluate, in_dims=(0, 0 if mapped_state else None, 0))(
        weights, state, values
    )
    assert actual[0]["has_spiked"].shape == (lanes, 3)
    assert actual[0]["delivery_buffer"].shape == (
        lanes,
        *tensors.state["delivery_buffer"].shape,
    )
    if lanes:
        expected = [
            evaluate(
                weights[i],
                pytree.tree_map_only(torch.Tensor, lambda value: value[i], state)
                if mapped_state
                else state,
                values[i],
            )
            for i in range(lanes)
        ]
        _close(actual, pytree.tree_map(lambda *items: torch.stack(items), *expected))
    gradients = torch.autograd.grad(
        actual[0]["delivery_buffer"].sum(), (weights, values)
    )
    assert gradients[0].shape == weights.shape
    assert gradients[1].shape == values.shape


def test_threshold_projection_forward_and_reverse_jacobians_agree():
    _, native = _netcon(training=True, mixed=True)
    projection, tensors = dn.func.make_functional_netcon(native)

    def evaluate(weights, values, pending):
        state = {**tensors.state, "delivery_buffer": pending}
        state, payload = projection.read_due(state)
        state = projection.advance(weights, state, values)
        return torch.cat((payload.flatten(), state["delivery_buffer"].flatten()))

    inputs = (
        tensors.weights,
        native.delivery_buffer.new_full(projection.source_shape, 0.15),
        tensors.state["delivery_buffer"] + 0.4,
    )
    _close(
        torch.func.jacfwd(evaluate, argnums=(0, 1, 2))(*inputs),
        torch.func.jacrev(evaluate, argnums=(0, 1, 2))(*inputs),
    )


@pytest.mark.parametrize("training", (False, True))
def test_checkpoint_continuation_while_above_threshold_avoids_a_second_event(training):
    _, native = _netcon(training=training)
    projection, tensors = dn.func.make_functional_netcon(native)
    values = native.delivery_buffer.new_full(projection.source_shape, 0.4)
    state, _ = projection.step(tensors.weights, tensors.state, values)
    assert state["has_spiked"].all()
    projection.commit_state_(native, state)
    _close(projection.extract(native).state, state)
    native.pre.v = values
    for _ in range(4):
        state, _ = projection.step(tensors.weights, state, values)
        native.advance()
        _close(state, native.state_dict_for_checkpoint())
    assert not torch.count_nonzero(state["delivery_buffer"])


@pytest.mark.parametrize("metadata", ("threshold", "mask"))
def test_cold_transfer_rejects_changed_detector_but_hot_plan_keeps_its_snapshot(
    metadata,
):
    _, native = _netcon(mixed=True)
    projection, tensors = dn.func.make_functional_netcon(native)
    values = native.delivery_buffer.new_full(projection.source_shape, 0.15)
    expected = projection.step(tensors.weights, tensors.state, values)
    if metadata == "threshold":
        native.threshold.add_(2.0)
    else:
        native.thresh_is_nan.logical_not_()
    _close(projection.step(tensors.weights, tensors.state, values), expected)
    for action in (
        lambda: projection.extract(native),
        lambda: projection.commit_state_(native, tensors.state),
    ):
        with pytest.raises(dn.func.FunctionalizationError, match="detector"):
            action()


def test_threshold_state_requires_boolean_latch_and_phases_fail_without_effects():
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    snapshot = _clone(native.state_dict_for_checkpoint())
    for bad in (
        {name: value for name, value in tensors.state.items() if name != "has_spiked"},
        {**tensors.state, "has_spiked": tensors.state["has_spiked"].float()},
    ):
        for action in (
            lambda: projection.read_due(bad),
            lambda: projection.advance(tensors.weights, bad, native.pre.v),
        ):
            with pytest.raises(ValueError):
                action()
    _close(native.state_dict_for_checkpoint(), snapshot)


@pytest.mark.parametrize("training", (False, True))
def test_netstim_explicit_gates_bypass_default_finite_threshold(training):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        post = dn.Population(N=1, C=3, dtype=torch.float64)
        post.insert(expsyn.rename("syn"))
        net = dn.Network(
            {"post": post},
            netstim=dn.NetStim(N=3, dtype=torch.float64),
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        net.connect_one_to_one(net.netstim[:], post[:], post.mech.syn)
        net.initialize(dt=DT)
        native = next(iter(net.synapses.values()))
        if training:
            native.set_diff_config(
                diff_weights=True,
                diff_delays=False,
                diff_spiking=True,
                diff_scheduled_times=False,
                train_delay_backend="dense",
            )
            native.train()
            native.initialize(reinit_weights=False, reinit_delays=False)
    assert not native.skip_thresholding
    projection, tensors = dn.func.make_functional_netcon(native)
    gates = tensors.weights.new_tensor([-0.2, 0.5, 1.5] if training else [1, 0, 1])
    gates = gates.reshape(projection.source_shape).requires_grad_()
    state = tensors.state
    for _ in range(2):
        state, _ = projection.step(tensors.weights, state, gates)
        if training:
            native.pre.spike_gate = gates
        else:
            native.pre.spikes = gates.bool()
        native.advance()
        _close(state, native.state_dict_for_checkpoint())
        assert not state["has_spiked"].any()
    gradient = torch.autograd.grad(state["delivery_buffer"].sum(), gates)[0]
    torch.testing.assert_close(gradient.flatten(), tensors.weights)


@pytest.mark.parametrize("from_netstim", (False, True))
def test_cold_transfer_rejects_different_effective_detector_modes_without_effects(
    from_netstim,
):
    _, population_connection = _netcon()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        post = dn.Population(N=1, C=3, dtype=torch.float64)
        post.insert(expsyn.rename("syn"))
        network = dn.Network(
            {"post": post},
            netstim=dn.NetStim(
                N=3, interval=torch.full((1, 3), 10.0), dtype=torch.float64
            ),
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            network.netstim[:],
            post[:],
            post.mech.syn,
            pre_var="v",
            threshold=population_connection.threshold.clone(),
            weight=population_connection.weight.w.clone(),
            delay=population_connection.delay_ms.w.clone(),
        )
        network.initialize(dt=DT)
        netstim_connection = next(iter(network.synapses.values()))
    source, destination = (
        (netstim_connection, population_connection)
        if from_netstim
        else (population_connection, netstim_connection)
    )
    projection, tensors = dn.func.make_functional_netcon(source)
    assert projection.source_shape == (1, 3)
    assert source.skip_thresholding == destination.skip_thresholding
    torch.testing.assert_close(source.threshold, destination.threshold)
    original_source = _clone(source.state_dict_for_checkpoint())
    original_destination = _clone(destination.state_dict_for_checkpoint())
    for action in (
        lambda: projection.extract(destination),
        lambda: projection.commit_state_(destination, tensors.state),
    ):
        with pytest.raises(dn.func.FunctionalizationError, match="detector"):
            action()
        _close(source.state_dict_for_checkpoint(), original_source)
        _close(destination.state_dict_for_checkpoint(), original_destination)


@pytest.mark.parametrize(
    "tau", (torch.tensor(TAU), torch.tensor(TAU, requires_grad=True), "0.12")
)
def test_tensor_or_nonreal_tau_is_rejected_as_detector_metadata_without_effects(tau):
    _, native = _netcon(training=True)
    native.set_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=False,
        tau=tau,
        train_delay_backend="dense",
    )
    snapshot = _clone(native.state_dict_for_checkpoint())
    with pytest.raises(dn.func.FunctionalizationError, match="tau=float"):
        dn.func.make_functional_netcon(native)
    _close(native.state_dict_for_checkpoint(), snapshot)
    if torch.is_tensor(tau):
        assert tau.grad is None


@pytest.mark.parametrize("compiled", (False, True))
def test_warm_projection_rejects_tensor_tau_before_reading_or_comparing_its_value(
    compiled,
):
    _, native = _netcon(training=True)
    projection, tensors = dn.func.make_functional_netcon(native)
    values = native.delivery_buffer.new_full(projection.source_shape, 0.15)

    def evaluate(values):
        return projection.step(tensors.weights, tensors.state, values)

    if compiled:
        evaluate = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    evaluate(values)
    tau = torch.tensor(TAU, requires_grad=True)
    native.set_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=False,
        tau=tau,
        train_delay_backend="dense",
    )
    snapshot = _clone(native.state_dict_for_checkpoint())
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        evaluate(values)
    _close(native.state_dict_for_checkpoint(), snapshot)
    assert tau.grad is None
