"""Functional NetCons share transport, state transfer, and transformed gradients."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn

DT = 0.1
pytestmark = pytest.mark.cpu


def _netcon(*, tracked=True, batch=None, delay_backend="dense", threshold=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        pre = dn.Population(N=1, C=3, dtype=torch.float64)
        post = dn.Population(N=1, C=3, dtype=torch.float64)
        post.insert(expsyn.rename("syn"))
        net = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend=delay_backend,
            netcon_train_backend="dense",
            track_netcon_events=tracked,
        )
        net.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.syn,
            threshold=threshold,
            pre_var="v",
            weight=torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64),
            delay=torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64),
        )
        if batch is not None:
            net.batch(batch)
        net.build(DT)
        net.init_synapses()
    return net, next(iter(net.synapses.values()))


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, left_spec = pytree.tree_flatten(actual)
    right, right_spec = pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


@pytest.mark.parametrize("tracked", (False, True))
@pytest.mark.parametrize("batch", (None, 2))
def test_raw_signed_gates_pending_weights_and_checkpoint_transfer(tracked, batch):
    _, native = _netcon(tracked=tracked, batch=batch)
    projection, tensors = dn.func.make_functional_netcon(native)
    snapshot = _clone(native.state_dict_for_checkpoint())
    weights = (
        tensors.weights.new_tensor([1.0, -2.0, 0.0])
        .repeat(tensors.weights.numel() // 3)
        .requires_grad_()
    )
    gates = torch.tensor([0.5, -0.25, 1.5], dtype=native.dtype).expand(
        projection.source_shape
    )
    initial = _clone(tensors.state)
    state, payload = projection.step(weights, tensors.state, gates)
    torch.testing.assert_close(payload, torch.zeros_like(payload))
    # Weights are captured at enqueue time, so changing the next step's weights
    # cannot rescale these two already pending one-step deliveries.
    state, payload = projection.step(weights * 9, state, torch.zeros_like(gates))
    expected = torch.tensor([0.5, 0.5, 0.0], dtype=native.dtype).expand(
        projection.shape
    )
    torch.testing.assert_close(payload, expected)
    gradient = torch.autograd.grad(payload.sum(), weights)[0]
    torch.testing.assert_close(gradient, gates.reshape(-1) * (native.delay_steps <= 1))
    if tracked:
        counts = torch.tensor([1, 0, 0], dtype=torch.int32).repeat(
            math_prod(projection.shape) // 3
        )
        torch.testing.assert_close(state["events"], counts)
    _close(tensors.state, initial)
    _close(native.state_dict_for_checkpoint(), snapshot)
    assert projection.commit_state_(native, state) is native
    _close(projection.extract(native).state, state)
    for name, value in state.items():
        assert native._buffers[name] is not value
    # The remaining two-step event has zero weight but still has one event.
    following, payload = projection.step(weights, state, torch.zeros_like(gates))
    torch.testing.assert_close(payload, torch.zeros_like(payload))
    if tracked:
        assert (
            torch.count_nonzero(following["events"]) == math_prod(projection.shape) // 3
        )


def math_prod(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def test_extracted_weights_are_independent_leaves_with_source_gradient_convention():
    _, native = _netcon()
    native.weight.w.requires_grad_()
    projection, tensors = dn.func.make_functional_netcon(native)
    assert tensors.weights.is_leaf and tensors.weights.requires_grad
    assert tensors.weights.data_ptr() != native.weight.w.data_ptr()
    for name, value in tensors.state.items():
        assert value.data_ptr() != native._buffers[name].data_ptr()
    with torch.no_grad():
        native.weight.w.add_(0.4)
    torch.testing.assert_close(projection.extract().weights, native.weight.w)
    assert not torch.equal(tensors.weights, native.weight.w)


@pytest.mark.parametrize("mapped", ("gates", "weights", "both", "state"))
@pytest.mark.parametrize("lanes", (0, 2))
def test_vmap_with_unmapped_or_mapped_queue_and_empty_lanes(mapped, lanes):
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    gates = native.delivery_buffer.new_tensor([[0.2, -0.3, 0.7]])
    weights = tensors.weights
    state = tensors.state
    state["delivery_buffer"] = state["delivery_buffer"] + 0.1

    def evaluate(weights, state, gates):
        # Include a second transition so source/weight gradients affect payload.
        state, first = projection.step(weights, state, gates)
        state, second = projection.step(weights, state, torch.zeros_like(gates))
        return state, first + second

    batched_weights = weights.repeat(lanes, 1).requires_grad_()
    batched_gates = gates.repeat(lanes, 1, 1).requires_grad_()
    batched_state = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape).clone(),
        state,
    )
    dims = (
        0 if mapped in {"weights", "both"} else None,
        0 if mapped == "state" else None,
        0 if mapped in {"gates", "both", "state"} else None,
    )
    inputs = (
        batched_weights if dims[0] == 0 else weights,
        batched_state if dims[1] == 0 else state,
        batched_gates if dims[2] == 0 else gates,
    )
    actual = torch.vmap(evaluate, in_dims=dims)(*inputs)
    if lanes:
        expected = [
            evaluate(
                inputs[0][index] if dims[0] == 0 else inputs[0],
                pytree.tree_map_only(
                    torch.Tensor, lambda value: value[index], inputs[1]
                )
                if dims[1] == 0
                else inputs[1],
                inputs[2][index] if dims[2] == 0 else inputs[2],
            )
            for index in range(lanes)
        ]
        _close(actual, pytree.tree_map(lambda *values: torch.stack(values), *expected))
    else:
        assert actual[0]["delivery_buffer"].shape == (
            0,
            *state["delivery_buffer"].shape,
        )
        assert actual[1].shape == (0, *projection.shape)
    loss = actual[0]["delivery_buffer"].sum() + actual[1].sum()
    targets = tuple(
        value
        for value, dim in ((batched_weights, dims[0]), (batched_gates, dims[2]))
        if dim == 0
    )
    for gradient, target in zip(
        torch.autograd.grad(loss, targets), targets, strict=True
    ):
        assert gradient.shape == target.shape
        if lanes:
            assert torch.count_nonzero(gradient)


def test_forward_reverse_jacobians_of_source_gates_weights_and_pending_queue():
    _, native = _netcon(tracked=False)
    projection, tensors = dn.func.make_functional_netcon(native)

    def evaluate(weights, gates, pending):
        state = {**tensors.state, "delivery_buffer": pending}
        state, payload = projection.step(weights, state, gates)
        return torch.cat((payload.flatten(), state["delivery_buffer"].flatten()))

    inputs = (
        tensors.weights,
        native.delivery_buffer.new_full((1, 3), 0.3),
        tensors.state["delivery_buffer"] + 0.1,
    )
    forward = torch.func.jacfwd(evaluate, argnums=(0, 1, 2))(*inputs)
    reverse = torch.func.jacrev(evaluate, argnums=(0, 1, 2))(*inputs)
    _close(forward, reverse)
    assert all(torch.count_nonzero(value) for value in forward)


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_fullgraph_compiled_projection_keeps_weight_and_gate_gradients(backend):
    with torch_compiler_warning_context():
        _, native = _netcon(tracked=False)
        projection, tensors = dn.func.make_functional_netcon(native)

        def evaluate(weights, gates):
            state, first = projection.step(weights, tensors.state, gates)
            state, second = projection.step(weights, state, torch.zeros_like(gates))
            return first + second, state["delivery_buffer"]

        weights = tensors.weights.detach().clone().requires_grad_()
        gates = native.delivery_buffer.new_full((1, 3), 0.3).requires_grad_()
        kwargs = (
            {"options": {"cpp_wrapper": True, "compile_threads": 1}}
            if backend == "inductor"
            else {}
        )
        compiled = torch.compile(evaluate, backend=backend, fullgraph=True, **kwargs)
        actual = compiled(weights, gates)
        expected = evaluate(weights, gates)
        _close(actual, expected)
        actual_grad = torch.autograd.grad(
            sum(value.sum() for value in actual), (weights, gates)
        )
        reference_grad = torch.autograd.grad(
            sum(value.sum() for value in expected), (weights, gates)
        )
        _close(actual_grad, reference_grad)
        assert all(torch.count_nonzero(value) for value in actual_grad)


@pytest.mark.parametrize(
    "invalid", ("gates_shape", "weights_shape", "dtype", "state", "slot")
)
def test_invalid_explicit_inputs_do_not_change_source_or_caller_state(invalid):
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    state = _clone(tensors.state)
    weights = tensors.weights
    gates = native.delivery_buffer.new_ones(projection.source_shape)
    if invalid == "gates_shape":
        gates = gates.flatten()
    elif invalid == "weights_shape":
        weights = weights.reshape(1, -1)
    elif invalid == "dtype":
        gates = gates.float()
    elif invalid == "state":
        state.pop("global_step")
    elif invalid == "slot":
        state["current_time_step"] += 100
    snapshot = _clone(native.state_dict_for_checkpoint())
    caller = _clone(state)
    with pytest.raises((ValueError, TypeError, IndexError, RuntimeError)):
        projection.step(weights, state, gates)
    _close(state, caller)
    _close(native.state_dict_for_checkpoint(), snapshot)


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("change", ("initialize", "schedule", "mode"))
def test_public_configuration_changes_invalidate_source_bound_calls(compiled, change):
    _, native = _netcon(tracked=False)
    projection, tensors = dn.func.make_functional_netcon(native)

    def evaluate(gates):
        return projection.step(tensors.weights, tensors.state, gates)

    if compiled:
        evaluate = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    gates = native.delivery_buffer.new_ones(projection.source_shape)
    evaluate(gates)
    if change == "initialize":
        native.initialize(reinit_weights=False, reinit_delays=False)
    elif change == "schedule":
        native.schedule(con_indices=[0], times_ms=[DT])
    else:
        native.set_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=False,
            train_delay_backend="dense",
        )
        native.train()
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        evaluate(gates)


def test_cold_transfer_rejects_changed_topology_and_invalid_counter_atomically():
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    bad = {**tensors.state, "current_time_step": torch.tensor([100])}
    snapshot = _clone(native.state_dict_for_checkpoint())
    with pytest.raises(ValueError, match="outside"):
        projection.commit_state_(native, bad)
    _close(native.state_dict_for_checkpoint(), snapshot)
    native.post_idx[0] = 2
    for operation in (
        lambda: projection.extract(native),
        lambda: projection.commit_state_(native, tensors.state),
    ):
        with pytest.raises(dn.func.FunctionalizationError, match="topology"):
            operation()


def test_noncontiguous_checkpoint_commit_resumes_native_dense_transport():
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    queue = torch.arange(9, dtype=native.dtype).reshape(3, 3).transpose(0, 1)
    assert not queue.is_contiguous()
    state = {**tensors.state, "delivery_buffer": queue}
    projection.commit_state_(native, state)
    assert native.delivery_buffer.is_contiguous()
    gates = queue.new_zeros(projection.source_shape)
    native.pre.v = gates.clone()
    expected, payload = projection.step(tensors.weights, state, gates)
    with torch.no_grad():
        native.advance()
    _close(native.state_dict_for_checkpoint(), expected)
    torch.testing.assert_close(native.syn.g, payload)
    torch.testing.assert_close(queue, state["delivery_buffer"])


def test_inference_mode_commit_publishes_ordinary_native_mutable_storage():
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    with torch.inference_mode():
        projection.commit_state_(native, tensors.state)
    assert all(not native._buffers[name].is_inference() for name in tensors.state)
    native.pre.v = native.pre.v.new_zeros(projection.source_shape)
    native.advance()


@pytest.mark.parametrize(
    "unsupported", ("compact", "schedule", "delay", "uninitialized")
)
def test_unsupported_projection_layouts_fail_before_effects(unsupported):
    _, native = _netcon(
        delay_backend="sparse_calendar" if unsupported == "compact" else "dense",
        tracked=False,
    )
    if unsupported == "schedule":
        native.schedule(con_indices=[0], times_ms=[DT])
    elif unsupported == "delay":
        native.set_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=False,
            train_delay_backend="dense",
        )
        native.train()
        native.initialize(reinit_weights=False, reinit_delays=False)
    elif unsupported == "uninitialized":
        native._event_delivery_initialized = False
    snapshot = _clone(native.state_dict_for_checkpoint())
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_netcon(native)
    _close(native.state_dict_for_checkpoint(), snapshot)
