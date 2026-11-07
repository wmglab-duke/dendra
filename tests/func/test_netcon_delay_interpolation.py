"""Explicit differentiable delays retain native dense transport semantics."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn

pytestmark = pytest.mark.cpu
DT = 0.1


def _netcon(
    *,
    taps=2,
    batch=None,
    mixed=False,
    scheduled=False,
    diff_delays=True,
    physical_delays=(0.0, 0.15, 0.36),
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        pre = dn.Population(N=1, C=3, dtype=torch.float64)
        post = dn.Population(N=1, C=3, dtype=torch.float64)
        post.insert(expsyn.rename("syn"))
        network = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
            track_netcon_events=False,
        )
        network.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.syn,
            threshold=torch.tensor([0.0, float("nan"), 0.2]) if mixed else None,
            pre_var="v",
            weight=torch.tensor([0.7, -0.4, 0.9], dtype=torch.float64),
            delay=torch.tensor(physical_delays, dtype=torch.float64),
        )
        if batch is not None:
            network.batch(batch)
        network.initialize(dt=DT)
        native = next(iter(network.synapses.values()))
        native.set_diff_config(
            diff_weights=True,
            diff_delays=diff_delays,
            diff_spiking=mixed,
            taps=taps,
            sigma=0.43,
            tau=0.17,
            diff_scheduled_times=scheduled,
            sched_width=0.8,
            train_delay_backend="dense",
        )
        native.train()
        native.initialize(reinit_weights=False, reinit_delays=False)
        if scheduled:
            for offset in range(0, native.pre_idx.numel(), 3):
                native.schedule(
                    con_indices=[offset, offset + 1, offset + 1, offset + 2],
                    times_ms=[0.025, 0.17, 0.17, 0.27],
                    weight=torch.tensor([1.2, 0.0, -0.4, 0.8], dtype=native.dtype),
                )
            native.sched_time_ms = native.sched_time_ms.detach().requires_grad_()
            native.sched_weight = native.sched_weight.detach().requires_grad_()
    return network, native


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, spec = pytree.tree_flatten(actual)
    right, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("mixed,scheduled", ((False, False), (True, True)))
def test_native_values_and_gradients_include_delays_pending_queue_and_schedules(
    taps, batch, mixed, scheduled
):
    _, native = _netcon(taps=taps, batch=batch, mixed=mixed, scheduled=scheduled)
    projection, tensors = dn.func.make_functional_netcon(native)
    assert tensors.delays.is_leaf
    assert tensors.delays.requires_grad == native.delay_ms.w.requires_grad
    assert tensors.delays.data_ptr() != native.delay_ms.w.data_ptr()
    assert dn.func.NetConTensors(tensors.weights, tensors.state).delays is None
    initial = _clone(tensors.state)
    pending = (tensors.state["delivery_buffer"] + 0.04).requires_grad_()
    native_pending = pending.detach().clone().requires_grad_()
    native.delivery_buffer = native_pending
    weights = tensors.weights.detach().clone().requires_grad_()
    delays = tensors.delays.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native_delays = delays.detach().clone().requires_grad_()
    native.weight.w = native_weights
    native.delay_ms.w = native_delays
    tape = torch.tensor(
        [
            [[-0.1, -0.3, 0.1]],
            [[0.2, 0.4, 0.3]],
            [[0.3, -0.1, 0.4]],
            [[-0.2, 0.2, 0.0]],
            [[0.1, -0.4, 0.3]],
            [[-0.2, 0.0, 0.0]],
        ],
        dtype=native.dtype,
    )
    tape = (
        tape.reshape(6, *([1] * (len(projection.source_shape) - 1)), 3)
        .expand(6, *projection.source_shape)
        .clone()
        .requires_grad_()
    )
    native_tape = tape.detach().clone().requires_grad_()
    state = {**tensors.state, "delivery_buffer": pending}
    losses, native_losses = [], []
    for step, (values, native_values) in enumerate(zip(tape, native_tape, strict=True)):
        state, payload = projection.step(
            weights, state, values, delays=delays, schedule=tensors.schedule
        )
        native.pre.v = native_values
        previous_g = native.syn.g.detach().clone()
        native.advance()
        _close(state, native.state_dict_for_checkpoint())
        torch.testing.assert_close(payload, native.syn.g - previous_g)
        position = (
            state["delivery_buffer"]
            .new_tensor(range(1, state["delivery_buffer"].shape[0] + 1))
            .unsqueeze(1)
        )
        losses.append((step + 1) * (position * state["delivery_buffer"]).sum())
        native_losses.append((step + 1) * (position * native.delivery_buffer).sum())
    inputs = (weights, delays, tape, pending)
    native_inputs = (native_weights, native_delays, native_tape, native_pending)
    if scheduled:
        inputs += (tensors.schedule.times, tensors.schedule.amplitudes)
        native_inputs += (native.sched_time_ms, native.sched_weight)
    actual = torch.autograd.grad(sum(losses), inputs)
    expected = torch.autograd.grad(sum(native_losses), native_inputs)
    _close(actual, expected)
    assert all(torch.count_nonzero(value) for value in actual)
    _close(tensors.state, initial)


@pytest.mark.parametrize("taps", (2, 3))
def test_parameter_updates_keep_warm_plan_and_checkpoint_native_continuation(taps):
    _, native = _netcon(taps=taps)
    projection, tensors = dn.func.make_functional_netcon(native)
    state = tensors.state
    values = tensors.weights.new_tensor([[0.4, -0.2, 0.3]])
    for _ in range(2):
        state, _ = projection.step(
            tensors.weights, state, values, delays=tensors.delays
        )
    checkpoint = _clone(state)
    # Carry alone contains no parameter values; align them before native use.
    weights = tensors.weights.detach().clone().add_(0.2).requires_grad_()
    delays = tensors.delays.detach().clone().add_(0.035).requires_grad_()
    native.weight.w = weights.detach().clone().requires_grad_()
    native.delay_ms.w = delays.detach().clone().requires_grad_()
    projection.commit_state_(native, checkpoint)
    _close(projection.extract().state, checkpoint)
    torch.testing.assert_close(projection.extract().delays, delays)
    for step in range(6):
        signal = values if step == 1 else torch.zeros_like(values)
        state, payload = projection.step(weights, checkpoint, signal, delays=delays)
        native.pre.v = signal
        previous_g = native.syn.g.detach().clone()
        native.advance()
        _close(state, native.state_dict_for_checkpoint())
        torch.testing.assert_close(payload, native.syn.g - previous_g)
        checkpoint = state
    assert all(native._buffers[name] is not value for name, value in checkpoint.items())


@pytest.mark.parametrize("taps", (2, 3))
def test_cold_compatible_target_can_have_different_delay_values_and_quantization(taps):
    _, original = _netcon(taps=taps)
    projection, tensors = dn.func.make_functional_netcon(original)
    # A different owner's physical delays can produce different inference
    # metadata without changing interpolated routing or fixed ring capacity.
    _, target = _netcon(taps=taps, physical_delays=(0.0, 0.23, 0.36))
    assert target.max_delay_steps == projection._depth
    assert not torch.equal(target.inference_delay_steps, original.inference_delay_steps)
    extracted = projection.extract(target)
    torch.testing.assert_close(extracted.delays, target.delay_ms.w)
    projection.commit_state_(target, tensors.state)
    _close(target.state_dict_for_checkpoint(), tensors.state)


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_explicit_delay_updates_and_gradients(taps, backend):
    with torch_compiler_warning_context():
        _, native = _netcon(taps=taps)
        projection, tensors = dn.func.make_functional_netcon(native)

        def evaluate(weights, delays, gates, pending):
            state = {**tensors.state, "delivery_buffer": pending}
            state, first = projection.step(weights, state, gates, delays=delays)
            state, second = projection.step(weights, state, gates * 0.3, delays=delays)
            return state["delivery_buffer"], first + second

        inputs = (
            tensors.weights.detach().requires_grad_(),
            tensors.delays.detach().requires_grad_(),
            tensors.weights.new_tensor([[0.4, -0.2, 0.6]]).requires_grad_(),
            (tensors.state["delivery_buffer"] + 0.03).requires_grad_(),
        )
        kwargs = (
            {"options": {"cpp_wrapper": True, "compile_threads": 1}}
            if backend == "inductor"
            else {}
        )
        compiled = torch.compile(evaluate, backend=backend, fullgraph=True, **kwargs)
        for offset in (0.0, 0.045):
            current = (inputs[0], inputs[1] + offset, *inputs[2:])
            actual = compiled(*current)
            expected = evaluate(*current)
            _close(actual, expected)
            positions = torch.arange(actual[0].shape[0], dtype=native.dtype)[:, None]
            actual_loss = (actual[0] * positions).sum() + actual[1].sum()
            expected_loss = (expected[0] * positions).sum() + expected[1].sum()
            _close(
                torch.autograd.grad(actual_loss, inputs, retain_graph=True),
                torch.autograd.grad(expected_loss, inputs),
            )


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("lanes", (0, 2))
@pytest.mark.parametrize("mapped", ("delays", "all"))
def test_vmap_delays_with_mapped_or_unmapped_carry_including_empty(taps, lanes, mapped):
    _, native = _netcon(taps=taps)
    projection, tensors = dn.func.make_functional_netcon(native)
    gates = tensors.weights.new_tensor([[0.4, -0.2, 0.6]])

    def evaluate(weights, delays, state, gates):
        return projection.step(weights, state, gates, delays=delays)

    batched = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape).clone(),
        (tensors.weights, tensors.delays, tensors.state, gates),
    )
    batched = (batched[0], batched[1].detach().requires_grad_(), *batched[2:])
    dims = (None, 0, None, None) if mapped == "delays" else (0, 0, 0, 0)
    inputs = (
        (tensors.weights, batched[1], tensors.state, gates)
        if mapped == "delays"
        else batched
    )
    actual = torch.vmap(evaluate, in_dims=dims)(*inputs)
    if lanes:
        expected = [
            evaluate(
                *pytree.tree_map_only(torch.Tensor, lambda value: value[i], batched)
            )
            for i in range(lanes)
        ]
        _close(actual, pytree.tree_map(lambda *values: torch.stack(values), *expected))
    else:
        assert actual[0]["delivery_buffer"].shape == (0, *native.delivery_buffer.shape)
        assert actual[1].shape == (0, *projection.shape)
    positions = torch.arange(native.delivery_buffer.shape[0], dtype=native.dtype)
    loss = (actual[0]["delivery_buffer"] * positions[None, :, None]).sum()
    gradient = torch.autograd.grad(loss, batched[1])[0]
    assert gradient.shape == batched[1].shape
    if lanes:
        assert torch.count_nonzero(gradient)


@pytest.mark.parametrize("taps", (2, 3))
def test_forward_reverse_jacobians_of_delays_and_pending_queue(taps):
    _, native = _netcon(taps=taps)
    projection, tensors = dn.func.make_functional_netcon(native)
    gates = tensors.weights.new_tensor([[0.4, -0.2, 0.6]])

    def evaluate(delays, pending):
        state = {**tensors.state, "delivery_buffer": pending}
        state = projection.advance(tensors.weights, state, gates, delays=delays)
        return state["delivery_buffer"]

    inputs = (tensors.delays, tensors.state["delivery_buffer"] + 0.1)
    forward = torch.func.jacfwd(evaluate, argnums=(0, 1))(*inputs)
    reverse = torch.func.jacrev(evaluate, argnums=(0, 1))(*inputs)
    _close(forward, reverse)
    assert all(torch.count_nonzero(value) for value in forward)


@pytest.mark.parametrize("invalid", ("missing", "shape", "dtype", "not_tensor"))
def test_bad_explicit_delays_leave_source_and_carry_unchanged(invalid):
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    state = _clone(tensors.state)
    delays = tensors.delays
    if invalid == "missing":
        delays = None
    elif invalid == "shape":
        delays = delays.reshape(1, -1)
    elif invalid == "dtype":
        delays = delays.float()
    else:
        delays = delays.tolist()
    with pytest.raises((TypeError, ValueError), match="delays"):
        projection.advance(
            tensors.weights,
            state,
            torch.zeros(projection.source_shape, dtype=native.dtype),
            delays=delays,
        )
    _close(state, tensors.state)
    _close(native.state_dict_for_checkpoint(), tensors.state)


def test_fixed_delay_plan_rejects_explicit_delays():
    _, native = _netcon(diff_delays=False)
    projection, tensors = dn.func.make_functional_netcon(native)
    assert tensors.delays is None
    with pytest.raises(ValueError, match="Fixed-delay"):
        projection.step(
            tensors.weights,
            tensors.state,
            torch.zeros(projection.source_shape, dtype=native.dtype),
            delays=native.delay_ms.w,
        )


@pytest.mark.parametrize("invalid", ("tracked", "taps", "sigma"))
def test_corrupted_interpolation_metadata_rejects_lowering(invalid):
    _, native = _netcon(taps=3)
    flags = list(native.train_flags)
    if invalid == "tracked":
        native.track_events = True
    elif invalid == "taps":
        flags[3] = 4
    else:
        flags[4] = torch.tensor(0.4, requires_grad=True)
    native.train_flags = tuple(flags)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_netcon(native)
