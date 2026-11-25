"""Public compiled Population contracts for a separate Newton forward."""

import gc
import weakref

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._derivimplicit_forward import _PROGRAMS

pytestmark = pytest.mark.cpu
DT = 0.1
DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu", IMEM=False),
    ):
        yield


class _DecayState(State):
    State.STATE("u", "w")
    State.GLOBAL(rate=1.25)
    State.DERIVATIVE("u' = -rate*u*w", "w' = rate*u*w")
    State.METHOD("derivimplicit")

    def state_defaults(self, v, values):
        return {"u": torch.full_like(v, 0.7), "w": torch.full_like(v, 0.3)}


class _Decay(Mechanism):
    Mechanism.STATE_BUNDLE(_DecayState)


class _DrivenState(State):
    State.STATE("u", "w")
    State.GLOBAL(rate=0.7)
    State.DERIVATIVE("u' = 0.2 - rate*u*w", "w' = rate*u*w - 0.2")
    State.METHOD("derivimplicit")

    def state_defaults(self, v, values):
        return {"u": torch.full_like(v, 0.6), "w": torch.full_like(v, 0.4)}


class _Driven(Mechanism):
    Mechanism.STATE_BUNDLE(_DrivenState)


def _model(*mechanisms):
    model = dn.SingleCompartment(N=2, C=1, dtype=DTYPE).train()
    for mechanism in mechanisms:
        model.insert(mechanism)
    model.initialize()
    return model


def _incoming(state, values):
    mechanisms = state["mechanisms"]
    return {
        **state,
        "mechanisms": {
            **mechanisms,
            **{
                name: {**mechanisms[name], "u": u, "w": w}
                for name, (u, w) in values.items()
            },
        },
    }


def _backward_euler_trace(initial, rate, source, steps):
    u_old, w_old = initial
    total = u_old + w_old
    scale = DT * rate
    linear = 1 + scale * total
    trace = [u_old]
    for _ in range(steps):
        forcing = trace[-1] + DT * source
        root = (linear.square() - 4 * scale * forcing).sqrt()
        trace.append(2 * forcing / (linear + root))
    u = torch.stack(trace)
    return u, total - u


def _state_pair(state, name):
    mechanism = state["mechanisms"][name]
    return mechanism["u"], mechanism["w"]


def _initial_pairs():
    return (
        (
            torch.tensor([[0.5], [0.65]], dtype=DTYPE, requires_grad=True),
            torch.tensor([[0.25], [0.48]], dtype=DTYPE, requires_grad=True),
        ),
        (
            torch.tensor([[0.4], [0.85]], dtype=DTYPE, requires_grad=True),
            torch.tensor([[0.65], [0.32]], dtype=DTYPE, requires_grad=True),
        ),
    )


def _exercise_direct_chunk(registry):
    previous_programs = set(registry)
    model = _model(_Decay, _Driven)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*rate_param")
    mechanisms = (_Decay.__name__, _Driven.__name__)
    rate_names = tuple(
        next(
            key
            for key in parameters
            if key.endswith(".rate_param") and f".{name}." in key
        )
        for name in mechanisms
    )
    steps = 3
    chunk = functional.compile_rollout_chunk(
        steps,
        execution="scan",
        backend="eager",
        derivimplicit_forward="separate",
    )
    for rates in ((1.25, 0.7), (0.8, 1.6)):
        initial = _initial_pairs()
        live = {
            **parameters,
            **{
                name: torch.full_like(parameters[name], value, requires_grad=True)
                for name, value in zip(rate_names, rates, strict=True)
            },
        }
        rate_tensors = tuple(live[name] for name in rate_names)
        incoming = _incoming(tensors.state, dict(zip(mechanisms, initial, strict=True)))
        prepared = functional.prepare(live, tensors.constants)
        state, _ = chunk(live, prepared, incoming)
        actual = tuple(
            value for name in mechanisms for value in _state_pair(state, name)
        )
        expected = tuple(
            trace[-1]
            for pair, rate, source in zip(
                initial, rate_tensors, (0.0, 0.2), strict=True
            )
            for trace in _backward_euler_trace(pair, rate, source, steps)
        )
        cotangents = (
            actual[0].new_tensor([[0.5], [-0.7]]),
            actual[1].new_tensor([[-0.3], [1.3]]),
            actual[2].new_tensor([[0.3], [0.8]]),
            actual[3].new_tensor([[0.4], [-1.0]]),
        )
        inputs = (*initial[0], *initial[1], *rate_tensors)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(
            torch.autograd.grad(actual, inputs, cotangents),
            torch.autograd.grad(expected, inputs, cotangents),
            rtol=1e-7,
            atol=1e-9,
        )
        assert chunk.execution_report().strategy == "compiled_scan"

    # At total u+w=0, this lane has a negative quadratic discriminant.
    bad_u, bad_w = (value.detach().clone() for value in initial[0])
    bad_u[1, 0], bad_w[1, 0] = 4.5, -4.5
    bad_inputs = (bad_u.requires_grad_(), bad_w.requires_grad_())
    frozen = tuple(value.detach().clone() for value in bad_inputs)
    bad_state = _incoming(incoming, {mechanisms[0]: bad_inputs})
    with pytest.raises(
        RuntimeError, match=r"_Decay.*chunk step 0.*batch index \(1, 0\)"
    ):
        chunk(live, prepared, bad_state)
    torch.testing.assert_close(
        _state_pair(bad_state, mechanisms[0]), frozen, rtol=0, atol=0
    )
    replay, _ = chunk(live, prepared, incoming)
    replay_values = tuple(
        value for name in mechanisms for value in _state_pair(replay, name)
    )
    torch.testing.assert_close(replay_values, actual, rtol=0, atol=0)

    captured_programs = set(registry) - previous_programs
    assert len(captured_programs) >= 2
    return captured_programs


def test_separate_direct_chunk_distinguishes_same_shape_mechanisms_and_live_rates():
    captured_programs = _exercise_direct_chunk(_PROGRAMS)

    gc.collect()
    assert captured_programs.isdisjoint(_PROGRAMS)


@pytest.mark.parametrize(
    "execution,inference,policy",
    [
        pytest.param("scan", False, "separate", id="separate-training-scan"),
        pytest.param("scan", True, "auto", id="inference-scan"),
        pytest.param("while", True, "auto", id="inference-while"),
    ],
)
def test_training_population_newton_forward_runs_without_gradients(
    execution, inference, policy
):
    model = _model(_Decay)
    name = _Decay.__name__
    steps = 4
    runner = dn.compile(
        model,
        dt=DT,
        steps=steps,
        execution=execution,
        inference=inference,
        trainable="*rate_param",
        backend="eager",
        derivimplicit_forward=policy,
    )
    initial, _ = _initial_pairs()
    (rate,) = runner.trainable_parameters.values()
    incoming = _incoming(runner.initial_state, {name: initial})
    if not inference:
        training_state, _ = runner(incoming)
        assert all(value.requires_grad for value in _state_pair(training_state, name))

    programs_before = set(_PROGRAMS)
    with torch.no_grad():
        state, _ = runner(incoming)
        expected = tuple(
            t[-1] for t in _backward_euler_trace(initial, rate, 0.0, steps)
        )

    actual = _state_pair(state, name)
    assert not any(value.requires_grad for value in actual)
    assert set(_PROGRAMS) <= programs_before
    strategy = "compiled_scan" if execution == "scan" else "compiled_while_loop"
    assert runner.execution_report().strategy == strategy
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize(
    "checkpoint_every", [None, 3], ids=["ordinary", "checkpointed"]
)
def test_separate_host_tail_and_callback_allow_delayed_backward(checkpoint_every):
    previous_programs = set(_PROGRAMS)
    model = _model(_Driven)
    name = _Driven.__name__
    fields = (f"{name}.u", f"{name}.w")
    steps = 5
    runner = dn.compile(
        model,
        dt=DT,
        steps=steps,
        chunk_steps=3,
        checkpoint_every=checkpoint_every,
        execution="scan",
        inference=False,
        trainable="*rate_param",
        callbacks={"trace": dn.func.Recorder(fields)},
        backend="eager",
        derivimplicit_forward="separate",
    )
    _, initial = _initial_pairs()
    (rate,) = runner.trainable_parameters.values()
    incoming = _incoming(runner.initial_state, {name: initial})
    final_state, auxiliary = runner(incoming)
    assert runner.execution_report().strategy == "compiled_scan"
    assert set(_PROGRAMS) - previous_programs
    trace = tuple(auxiliary["callbacks"]["trace"][field] for field in fields)
    final = _state_pair(final_state, name)
    expected = _backward_euler_trace(initial, rate, 0.2, steps)
    torch.testing.assert_close(trace, expected, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(
        final, tuple(t[-1] for t in expected), rtol=1e-7, atol=1e-9
    )

    runner_reference = weakref.ref(runner)
    del runner, final_state, auxiliary
    gc.collect()
    assert runner_reference() is None

    inputs = (*initial, rate)
    loss = sum(v.square().sum() for v in final) + sum(v.square().mean() for v in trace)
    expected_loss = sum(v[-1].square().sum() + v.square().mean() for v in expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, inputs),
        torch.autograd.grad(expected_loss, inputs),
        rtol=1e-7,
        atol=1e-9,
    )
