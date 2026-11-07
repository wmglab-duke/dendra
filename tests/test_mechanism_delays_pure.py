"""Native and pure fixed-step history share timing and differentiation rules."""

import copy

import pytest
import torch
from torch.func import jacfwd, jacrev, jvp, vmap

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._delays import delay_history_values
from dendra.models.mechanisms._handler import MechanismHandler


def _mechanism(shape=(2, 3), dtype=torch.float64, cls=Mechanism):
    return cls(
        "delay",
        torch.full(shape, 34.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
        shape,
        shape,
    )


def _history_values(mechanism):
    return {
        name: mechanism._buffers[name].clone()
        for name in mechanism._delay_history_buffer_names
    }


def _assert_unchanged(mechanism, before):
    for name, value in mechanism.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["shift", "circular"])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize(
    "shape,axis",
    [((), 0), ((2,), 0), ((2,), 1), ((2, 3), 0), ((2, 3), 1), ((2, 3), 2)],
)
def test_single_history_pure_matches_native_and_exact_delay(
    dtype, mode, steps, shape, axis
):
    mechanism = _mechanism(shape or (1,), dtype)
    mechanism.register_delayed_state(
        "signal", torch.zeros(shape, dtype=dtype), steps, mode=mode, insert_axis=axis
    )
    state = _history_values(mechanism)
    source = copy.deepcopy(mechanism.state_dict())
    buffer_identity = mechanism.signal_delay_buffer
    pointer_identity = mechanism.signal_delay_ptr
    inputs = [torch.full(shape, k + 1.0, dtype=dtype) for k in range(7)]

    for k, value in enumerate(inputs):
        delayed, state = mechanism.delayed_state_values("signal", value, state)
        _assert_unchanged(mechanism, source)
        expected = inputs[k - steps] if k >= steps else torch.zeros_like(value)
        torch.testing.assert_close(delayed, expected, rtol=0, atol=0)
        native = mechanism.delayed_state("signal", value)
        torch.testing.assert_close(delayed, native, rtol=0, atol=0)
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, getattr(mechanism, name), rtol=0, atol=0)
        source = copy.deepcopy(mechanism.state_dict())

    if mode == "circular" or steps == 0:
        assert mechanism.signal_delay_buffer is buffer_identity
        assert mechanism.signal_delay_ptr is pointer_identity


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["shift", "circular"])
@pytest.mark.parametrize(
    "shape,stream_axis,delay_axis",
    [
        ((3, 2), 0, 0),
        ((3, 2), 0, 1),
        ((3, 2), 0, 2),
        ((2, 3), 1, 0),
        ((2, 3), 1, 1),
        ((2, 3), 1, 2),
        ((2, 3, 4), 1, 0),
        ((2, 3, 4), 1, 1),
        ((2, 3, 4), 1, 2),
        ((2, 3, 4), 1, 3),
    ],
)
def test_multistream_history_pure_matches_native_all_axis_orders(
    dtype, mode, shape, stream_axis, delay_axis
):
    mechanism = _mechanism(shape, dtype)
    mechanism.register_delayed_states(
        "signal",
        torch.zeros(shape, dtype=dtype),
        [0, 1, 3],
        mode=mode,
        stream_axis=stream_axis,
        delay_axis=delay_axis,
    )
    state = _history_values(mechanism)
    source = copy.deepcopy(mechanism.state_dict())
    buffer_identity = mechanism.signal_delay_buffer
    pointer_identity = mechanism.signal_delay_ptr
    inputs = [torch.full(shape, k + 1.0, dtype=dtype) for k in range(7)]

    for k, value in enumerate(inputs):
        delayed, state = mechanism.delayed_states_values("signal", value, state)
        _assert_unchanged(mechanism, source)
        expected = torch.zeros_like(value)
        for stream, steps in enumerate((0, 1, 3)):
            if k >= steps:
                expected.select(stream_axis, stream).copy_(
                    inputs[k - steps].select(stream_axis, stream)
                )
        torch.testing.assert_close(delayed, expected, rtol=0, atol=0)
        native = mechanism.delayed_states("signal", value)
        torch.testing.assert_close(delayed, native, rtol=0, atol=0)
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, getattr(mechanism, name), rtol=0, atol=0)
        source = copy.deepcopy(mechanism.state_dict())

    if mode == "circular":
        assert mechanism.signal_delay_buffer is buffer_identity
        assert mechanism.signal_delay_ptr is pointer_identity


@pytest.mark.parametrize("mode", ["shift", "circular"])
def test_all_zero_streams_do_not_advance_or_rebind_history(mode):
    mechanism = _mechanism()
    mechanism.register_delayed_states(
        "signal",
        torch.zeros(2, 3, dtype=torch.float64),
        [0, 0],
        stream_axis=0,
        mode=mode,
    )
    state = _history_values(mechanism)
    state["signal_delay_buffer"].fill_(7)
    value = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    delayed, updated = mechanism.delayed_states_values("signal", value, state)
    assert delayed is value
    for name, tensor in updated.items():
        assert tensor is state[name]
    torch.testing.assert_close(
        torch.autograd.grad(delayed.sum(), value)[0], torch.ones_like(value)
    )


@pytest.mark.parametrize("multi", [False, True])
def test_shift_history_routes_exact_temporal_gradients(multi):
    mechanism = _mechanism()
    if multi:
        mechanism.register_delayed_states(
            "signal",
            torch.zeros(2, 3, dtype=torch.float64),
            [0, 2],
            stream_axis=0,
            mode="shift",
        )
        update = mechanism.delayed_states_values
    else:
        mechanism.register_delayed_state(
            "signal", torch.zeros(2, 3, dtype=torch.float64), 2, mode="shift"
        )
        update = mechanism.delayed_state_values
    state = _history_values(mechanism)
    inputs = torch.randn(5, 2, 3, dtype=torch.float64, requires_grad=True)
    outputs = []
    for value in inputs.unbind():
        delayed, state = update("signal", value, state)
        outputs.append(delayed)
    weights = torch.arange(1, 6, dtype=inputs.dtype).reshape(5, 1, 1)
    loss = (torch.stack(outputs) * weights).sum()
    gradient = torch.autograd.grad(loss, inputs)[0]
    expected = torch.zeros_like(inputs)
    for source in range(5):
        if multi:
            expected[source, 0] = source + 1
            if source + 2 < 5:
                expected[source, 1] = source + 3
        elif source + 2 < 5:
            expected[source] = source + 3
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


def test_circular_history_detaches_queued_writes_and_keeps_zero_stream_gradient():
    mechanism = _mechanism()
    mechanism.register_delayed_states(
        "signal",
        torch.zeros(2, 3, dtype=torch.float64),
        [0, 1],
        stream_axis=0,
        mode="circular",
    )
    state = _history_values(mechanism)
    first = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    second = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    _, state = mechanism.delayed_states_values("signal", first, state)
    assert not state["signal_delay_buffer"].requires_grad
    output, state = mechanism.delayed_states_values("signal", second, state)
    torch.testing.assert_close(output[1], first[1])
    first_grad, second_grad = torch.autograd.grad(
        output.sum(), (first, second), allow_unused=True
    )
    assert first_grad is None
    expected = torch.zeros_like(second)
    expected[0] = 1
    torch.testing.assert_close(second_grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize("multi", [False, True])
def test_circular_history_preserves_native_incoming_queue_gradient_identity(multi):
    mechanism = _mechanism()
    if multi:
        mechanism.register_delayed_states(
            "signal",
            torch.zeros(2, 3, dtype=torch.float64),
            [0, 1],
            stream_axis=0,
            delay_axis=0,
            mode="circular",
        )
        pure_update = mechanism.delayed_states_values
        native_update = mechanism.delayed_states
    else:
        mechanism.register_delayed_state(
            "signal",
            torch.zeros(2, 3, dtype=torch.float64),
            1,
            insert_axis=0,
            mode="circular",
        )
        pure_update = mechanism.delayed_state_values
        native_update = mechanism.delayed_state
    initial = torch.randn_like(mechanism.signal_delay_buffer, requires_grad=True)
    payloads = [torch.full_like(mechanism.diam, k + 1) for k in range(5)]

    def execute(queue):
        state = {
            "signal_delay_buffer": queue,
            "signal_delay_ptr": torch.zeros((), dtype=torch.long),
        }
        outputs = []
        for payload in payloads:
            delayed, state = pure_update("signal", payload, state)
            outputs.append(delayed)
        return torch.stack(outputs)

    def execute_native(queue):
        mechanism.signal_delay_buffer = queue.clone()
        mechanism.signal_delay_ptr = torch.zeros((), dtype=torch.long)
        return torch.stack([native_update("signal", payload) for payload in payloads])

    native = execute_native(initial)
    actual = execute(initial)
    torch.testing.assert_close(actual, native, rtol=0, atol=0)
    native_gradient = torch.autograd.grad(native.square().sum(), initial)[0]
    pure_gradient = torch.autograd.grad(actual.square().sum(), initial)[0]
    torch.testing.assert_close(pure_gradient, native_gradient, rtol=0, atol=0)
    # Native no-grad writes retain the reverse-mode destination identity, while
    # forward AD updates tangents in overwritten slots. Preserve each native
    # rule independently, including reads after overwriting the same slot.
    tangent = torch.randn_like(initial)
    native_output, native_tangent = jvp(execute_native, (initial,), (tangent,))
    pure_output, pure_tangent = jvp(execute, (initial,), (tangent,))
    torch.testing.assert_close(pure_output, native_output, rtol=0, atol=0)
    torch.testing.assert_close(pure_tangent, native_tangent, rtol=0, atol=0)


def test_circular_history_replacement_keeps_exact_nonfinite_payload_values():
    mechanism = _mechanism((3,))
    mechanism.register_delayed_state(
        "signal", torch.zeros(3, dtype=torch.float64), 1, mode="circular"
    )
    state = _history_values(mechanism)
    payload = torch.tensor(
        [float("inf"), -float("inf"), float("nan")], dtype=torch.float64
    )
    _, state = mechanism.delayed_state_values("signal", payload, state)
    delayed, _ = mechanism.delayed_state_values(
        "signal", torch.zeros_like(payload), state
    )
    torch.testing.assert_close(delayed, payload, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("mode", ["shift", "circular"])
def test_pure_history_composes_with_fullgraph_compile_and_transforms(mode):
    steps = torch.tensor([0, 2])
    buffer = torch.zeros(3, 2, 3, dtype=torch.float64)
    pointer = torch.zeros((), dtype=torch.long)
    payload = torch.randn(2, 3, dtype=torch.float64)

    def execute(value):
        state, current = buffer, pointer
        outputs = []
        for k in range(4):
            delayed, state, current = delay_history_values(
                state,
                current,
                value + k,
                steps,
                axis=0,
                mode=mode,
                stream_axis=1,
                value_stream_axis=0,
            )
            outputs.append(delayed)
        return torch.stack(outputs)

    compiled = torch.compile(execute, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(payload), execute(payload), rtol=0, atol=0)
    torch.testing.assert_close(
        jacfwd(execute)(payload), jacrev(execute)(payload), rtol=0, atol=0
    )
    batch = torch.stack((payload, payload + 1))
    torch.testing.assert_close(
        vmap(execute)(batch), torch.stack([execute(x) for x in batch]), rtol=0, atol=0
    )
    assert vmap(execute)(batch[:0]).shape == (0, 4, 2, 3)


@pytest.mark.parametrize("multi", [False, True])
def test_vmap_continues_independent_circular_checkpoint_cursors(multi):
    steps = torch.tensor([0, 2]) if multi else 2

    def execute(history, pointer, payload):
        return delay_history_values(
            history,
            pointer,
            payload,
            steps,
            axis=0,
            mode="circular",
            stream_axis=1 if multi else None,
            value_stream_axis=0 if multi else None,
        )

    histories = torch.arange(54, dtype=torch.float64).reshape(3, 3, 2, 3)
    pointers = torch.tensor([0, 1, 2])
    payloads = torch.randn(3, 2, 3, dtype=torch.float64)
    actual = vmap(execute)(histories, pointers, payloads)
    expected = tuple(
        torch.stack(leaves)
        for leaves in zip(
            *(execute(*arguments) for arguments in zip(histories, pointers, payloads)),
            strict=True,
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _PureDelayedAdvance(Mechanism):
    Mechanism.CARRY("delivered")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_delayed_state(
            "signal", torch.zeros_like(self.diam), 1, mode="shift"
        )

    def advance(self, v, dt, values):
        del dt
        delayed, updates = self.delayed_state_values("signal", v, values)
        return {**updates, "delivered": delayed}


class _ResetDelayedAdvance(_PureDelayedAdvance):
    def initial_values(self, v, values):
        del v
        return self.initial_delay_values(values)


@pytest.mark.parametrize("reset", [False, True])
def test_cold_registration_exposes_native_hook_frames_and_initialization_schema(reset):
    cls = _ResetDelayedAdvance if reset else _PureDelayedAdvance
    mechanism = _mechanism(cls=cls)
    first = torch.full_like(mechanism.diam, 3)
    mechanism._advance_states(first, first.new_tensor(0.1))
    queue = mechanism.signal_delay_buffer.clone()
    mechanism._init_buffers_s(torch.zeros_like(first))
    expected_queue = torch.zeros_like(queue) if reset else queue
    torch.testing.assert_close(
        mechanism.signal_delay_buffer, expected_queue, rtol=0, atol=0
    )
    mechanism._advance_states(first + 1, first.new_tensor(0.1))
    expected = torch.zeros_like(first) if reset else first
    torch.testing.assert_close(mechanism.delivered, expected, rtol=0, atol=0)


def test_initial_delay_values_does_not_mutate_checkpoint_or_source():
    mechanism = _mechanism()
    mechanism.register_delayed_state("signal", torch.zeros_like(mechanism.diam), 2)
    state = _history_values(mechanism)
    state["signal_delay_buffer"].fill_(2)
    state["signal_delay_ptr"].fill_(1)
    source = copy.deepcopy(mechanism.state_dict())
    outputs = mechanism.initial_delay_values(state)
    assert state["signal_delay_buffer"].eq(2).all()
    assert state["signal_delay_ptr"].item() == 1
    for name, tensor in outputs.items():
        assert not tensor.requires_grad
        assert not torch.count_nonzero(tensor)
        assert tensor is not state[name]
    _assert_unchanged(mechanism, source)


class _DynamicHistoryAdvance(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.CARRY("delivered")
    Mechanism.NONSPECIFIC_CURRENT("i")

    def advance(self, v, dt, values):
        del dt
        delayed, updates = self.delayed_state_values(self.delay_name, v, values)
        return {**updates, "delivered": delayed}

    def i(self, v):
        return self.g * (v - self.e)


def _dynamic_history(name, *, cls=_DynamicHistoryAdvance, shape=(2, 3)):
    mechanism = _mechanism(shape, cls=cls)
    mechanism.delay_name = name
    mechanism.register_delayed_state(
        name, torch.zeros_like(mechanism.diam), 1, mode="shift", insert_axis=0
    )
    return mechanism


def test_generated_native_advance_keeps_each_instances_history_registration():
    first = _dynamic_history("first")
    advance = type(first)._advance_states
    second = _dynamic_history("second")
    assert type(first)._advance_states is advance
    first_value, second_value = (
        torch.ones_like(first.diam),
        torch.full_like(second.diam, 2),
    )
    dt = first_value.new_tensor(0.1)
    first._advance_states(first_value, dt)
    second._advance_states(second_value, dt)
    first._advance_states(first_value + 5, dt)
    second._advance_states(second_value + 7, dt)
    torch.testing.assert_close(first.delivered, first_value, rtol=0, atol=0)
    torch.testing.assert_close(second.delivered, second_value, rtol=0, atol=0)


def test_new_sibling_registration_preserves_existing_functional_plan():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        population = dn.SingleCompartment(N=2, C=1, v_init=-55.0)
        population.insert(_DynamicHistoryAdvance.rename("history"))
        population.initialize()
        mechanism = population.mech.history
        mechanism.delay_name = "first"
        mechanism.register_delayed_state(
            "first", torch.zeros_like(mechanism.diam), 1, mode="shift", insert_axis=0
        )
        population.initialize()
        functional, tensors = dn.func.make_functional(population, dt=0.1)
        before, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
        advance = type(mechanism)._advance_states
        sibling = _dynamic_history(
            "second",
            cls=_DynamicHistoryAdvance.rename("history"),
            shape=tuple(mechanism.shape_f),
        )
        sibling._advance_states(
            torch.ones_like(sibling.diam), sibling.diam.new_tensor(0.1)
        )
        assert type(mechanism)._advance_states is advance
        after, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_external_registration_keeps_compatible_population_hook_identity():
    def build():
        population = dn.SingleCompartment(N=2, C=1, v_init=-55.0)
        population.insert(_DynamicHistoryAdvance.rename("history"))
        population.initialize()
        mechanism = population.mech.history
        mechanism.delay_name = "signal"
        mechanism.register_delayed_state(
            "signal", torch.zeros_like(mechanism.diam), 1, mode="shift", insert_axis=0
        )
        return population

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        population, other = build(), build()
        first = population.mech.history._advance_states.__func__
        assert other.mech.history._advance_states.__func__ is first
        functional, tensors = dn.func.make_functional(population, dt=0.1)
        torch.testing.assert_close(functional.extract(other).state, tensors.state)


class _DeferredHistoryAdvance(_PureDelayedAdvance):
    Mechanism.CARRY("signal_delay_buffer", shape="deferred")
    Mechanism.CARRY("signal_delay_ptr", shape=(), dtype=torch.long)

    def initial_values(self, v, values):
        del v
        return self.initial_delay_values(values)


def _resized_deferred_history():
    mechanism = _mechanism(cls=_DeferredHistoryAdvance)
    handler = MechanismHandler(mechanism.celsius, mechanism.diam, {"delay": mechanism})
    value = torch.ones_like(mechanism.diam)
    mechanism.delayed_state("signal", value)
    snapshot = handler.mutable_state_dict()
    mechanism.delayed_state("signal", value + 1, delay_steps=3)
    return mechanism, handler, snapshot


def test_deferred_history_capacity_tracks_reregistration_and_checkpoint():
    mechanism, handler, snapshot = _resized_deferred_history()
    assert mechanism._carry_resolved_shapes["signal_delay_buffer"] == (2, 3, 4)
    mechanism._init_buffers_s(torch.zeros_like(mechanism.diam))
    handler.restore_mutable_state_dict(snapshot)
    assert mechanism._carry_resolved_shapes["signal_delay_buffer"] == (2, 3, 2)
    torch.testing.assert_close(
        mechanism.delayed_state("signal", torch.zeros_like(mechanism.diam)),
        torch.ones_like(mechanism.diam),
        rtol=0,
        atol=0,
    )
    mechanism._init_buffers_s(torch.zeros_like(mechanism.diam))
    assert not mechanism.signal_delay_buffer.count_nonzero()


def test_deferred_history_checkpoint_failure_restores_shape_and_tensor_identity(
    monkeypatch,
):
    mechanism, handler, snapshot = _resized_deferred_history()
    queue, pointer = mechanism.signal_delay_buffer, mechanism.signal_delay_ptr
    specs = mechanism._delayed_state_specs
    resolutions = mechanism._carry_resolved_shapes
    original_shapes = dict(resolutions)
    synchronize = mechanism._sync_delay_history_shapes

    def fail_after_synchronizing():
        synchronize()
        raise RuntimeError("late history restoration failure")

    monkeypatch.setattr(
        mechanism, "_sync_delay_history_shapes", fail_after_synchronizing
    )
    with pytest.raises(RuntimeError, match="late history restoration failure"):
        handler.restore_mutable_state_dict(snapshot)
    assert mechanism.signal_delay_buffer is queue
    assert mechanism.signal_delay_ptr is pointer
    assert mechanism._delayed_state_specs is specs
    assert mechanism._carry_resolved_shapes is resolutions
    assert resolutions == original_shapes


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["shift", "circular"])
@pytest.mark.parametrize("multi", [False, True])
def test_cuda_history_values_and_gradients_match_native(dtype, mode, multi):
    mechanism = _mechanism(dtype=dtype).to("cuda")
    like = torch.zeros_like(mechanism.diam)
    if multi:
        mechanism.register_delayed_states(
            "signal", like, [0, 2], stream_axis=0, delay_axis=0, mode=mode
        )
        pure_update = mechanism.delayed_states_values
        native_update = mechanism.delayed_states
    else:
        mechanism.register_delayed_state("signal", like, 2, insert_axis=0, mode=mode)
        pure_update = mechanism.delayed_state_values
        native_update = mechanism.delayed_state
    initial = torch.randn_like(mechanism.signal_delay_buffer, requires_grad=True)
    payload = torch.randn_like(like, requires_grad=True)
    state = {
        "signal_delay_buffer": initial,
        "signal_delay_ptr": torch.zeros((), dtype=torch.long, device="cuda"),
    }
    outputs = []
    for k in range(5):
        delayed, state = pure_update("signal", payload + k, state)
        outputs.append(delayed)
    actual = torch.stack(outputs)
    mechanism.signal_delay_buffer = initial.clone()
    expected = torch.stack([native_update("signal", payload + k) for k in range(5)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), (initial, payload), allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), (initial, payload), allow_unused=True
    )
    for observed, wanted in zip(actual_gradients, expected_gradients, strict=True):
        if observed is None or wanted is None:
            assert observed is wanted
        else:
            torch.testing.assert_close(observed, wanted)
