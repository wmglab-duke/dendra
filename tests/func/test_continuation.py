"""Host continuation captures actual operands without severing their graphs."""

from __future__ import annotations

import gc
import math
import weakref
from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        children = tuple(_plain(child) for child in value)
        return type(value)(*children) if hasattr(value, "_fields") else children
    if isinstance(value, list):
        return [_plain(child) for child in value]
    return value


def _snapshot(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            with torch.inference_mode(False):
                memo[id(value)] = value.detach().clone()
        return memo[id(value)]
    if isinstance(value, Mapping):
        return {key: _snapshot(child, memo) for key, child in value.items()}
    if isinstance(value, tuple):
        children = tuple(_snapshot(child, memo) for child in value)
        return type(value)(*children) if hasattr(value, "_fields") else children
    if isinstance(value, list):
        return [_snapshot(child, memo) for child in value]
    return value


def _close(actual, expected):
    torch.testing.assert_close(_plain(actual), _plain(expected))


def _callbacks(network):
    count = dn.func.APCount(node_check=[0], threshold=-59.0, dt=DT)
    return {
        "trace": dn.func.Recorder(
            ("left.v", "right.v", "t") if network else ("v", "t")
        ),
        "count": dn.func.PopulationCallback("left", count) if network else count,
    }


def _runner(network, *, callbacks=True, inference=False, steps=3, **options):
    return dn.compile(
        _network() if network else _population(),
        dt=DT,
        steps=steps,
        callbacks=_callbacks(network) if callbacks else None,
        inference=inference,
        trainable="*g_param" if not inference else (),
        backend=options.pop("backend", "eager"),
        disable=options.pop("disable", True),
        **options,
    )


def _drive(runner, network, steps, *, requires_grad=False):
    population = (
        runner.functional.populations["right"] if network else runner.functional
    )
    shape = population.shape
    current = torch.linspace(
        -2e-6,
        4e-6,
        steps * math.prod(shape),
        dtype=population.dtype,
        device=population.device,
    ).reshape(steps, *shape)
    current.requires_grad_(requires_grad)
    return current, {
        "right": dn.func.Inputs(intra=current)
    } if network else dn.func.Inputs(intra=current)


def _assign(parameters, network, value):
    if network:
        parameters["populations"]["left"][G] = value
        parameters["populations"]["right"][G] = value
    else:
        parameters[G] = value


def _joined(first, second):
    return {
        name: torch.cat((first["callbacks"]["trace"][name], value))
        for name, value in second["callbacks"]["trace"].items()
    }


def _failure(call):
    try:
        call()
    except Exception as error:
        return type(error), str(error)
    raise AssertionError("the malformed continuation must fail")


def test_continuation_snapshots_containers_and_retains_alias_views_and_gradients():
    upstream = torch.arange(6.0, requires_grad=True)
    view = (upstream * 2).reshape(2, 3)[:, ::2]
    packet = dn.func.StepInput(intra=view)
    state = {"integrator": {"v": view}, "packet": packet, "sequence": [view]}
    parameters = {"a": view, "b": view}
    constants = {"shared": view}
    continuation = dn.func.Continuation(state, parameters, constants)
    assert continuation.state is not state
    assert continuation.state["integrator"] is not state["integrator"]
    assert continuation.state["integrator"]["v"] is view
    assert type(continuation.state["packet"]) is type(packet)
    assert continuation.state["packet"].intra is view
    assert continuation.state["sequence"] is not state["sequence"]
    assert (
        continuation.parameters["a"]
        is continuation.parameters["b"]
        is continuation.constants["shared"]
        is view
    )
    state["integrator"]["v"] = None
    state["sequence"].clear()
    parameters["a"] = None
    constants.clear()
    assert continuation.state["integrator"]["v"] is view
    assert continuation.state["sequence"] == [view]
    assert continuation.parameters["a"] is view
    assert continuation.constants["shared"] is view
    gradient = torch.autograd.grad(
        continuation.state["integrator"]["v"].sum(), upstream
    )[0]
    torch.testing.assert_close(gradient, torch.tensor([2.0, 0.0, 2.0, 2.0, 0.0, 2.0]))
    with pytest.raises((AttributeError, TypeError)):
        continuation.state = {}


def test_continuation_container_capture_does_not_dispatch_or_copy_tensor_subclasses():
    class TrackedTensor(torch.Tensor):
        calls = []

        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            cls.calls.append(function)
            return super().__torch_function__(function, types, args, kwargs or {})

    value = torch.ones(2, requires_grad=True).as_subclass(TrackedTensor)
    TrackedTensor.calls.clear()
    continuation = dn.func.Continuation({"v": value}, {"a": value}, {"b": value})
    assert (
        continuation.state["v"]
        is continuation.parameters["a"]
        is continuation.constants["b"]
        is value
    )
    assert not TrackedTensor.calls


@pytest.mark.parametrize(
    "field", ("state", "parameters", "constants", "callback_state")
)
def test_continuation_rejects_missing_state_and_invalid_source_or_callback_types(field):
    values = dict(state={}, parameters={}, constants={}, callback_state=None)
    values[field] = object() if field == "callback_state" else None
    with pytest.raises((TypeError, ValueError), match=field):
        dn.func.Continuation(**values)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("callbacks", (False, True))
def test_runner_publishes_selected_sources_and_packet_resume_matches_legacy_keywords(
    network, callbacks
):
    runner = _runner(network, callbacks=callbacks)
    parameters, constants = _plain(runner.parameters), _plain(runner.constants)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = 1.3 * upstream
    _assign(parameters, network, replacement)
    first, auxiliary = runner(parameters=parameters, constants=constants)
    continuation = auxiliary["continuation"]
    assert isinstance(continuation, dn.func.Continuation)
    assert _target(continuation.parameters, network) is replacement
    assert _voltage(continuation.state, network).shape == _voltage(first, network).shape
    if network:
        assert (
            continuation.parameters["populations"]["left"][G]
            is continuation.parameters["populations"]["right"][G]
        )
    if callbacks:
        assert (
            continuation.callback_state.plan_token
            is auxiliary["callbacks"].state.plan_token
        )
    else:
        assert continuation.callback_state is None and "callbacks" not in auxiliary
    final, resumed = runner(continuation)
    expected, legacy = runner(
        first,
        parameters=parameters,
        constants=constants,
        callback_state=auxiliary["callbacks"].state if callbacks else None,
    )
    _close(final, expected)
    if callbacks:
        _close(resumed["callbacks"].outputs, legacy["callbacks"].outputs)
        assert resumed["callbacks"]["trace"]["left.v" if network else "v"].shape[0] == 3
    assert _target(resumed["continuation"].parameters, network) is replacement
    assert _target(runner.parameters, network) is not replacement
    restarted, _ = runner()
    assert not torch.allclose(_voltage(first, network), _voltage(restarted, network))


@pytest.mark.parametrize("network", (False, True))
def test_packet_continuation_preserves_prefix_parameter_and_input_gradients_through_checkpoint_replay(
    network,
):
    runner = _runner(network, backend="aot_eager", disable=False, checkpoint_every=2)
    parameters = _plain(runner.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, 1.2 * upstream)
    current, inputs = _drive(runner, network, 6, requires_grad=True)
    first_inputs = (
        {"right": dn.func.Inputs(intra=current[:3])}
        if network
        else dn.func.Inputs(intra=current[:3])
    )
    last_inputs = (
        {"right": dn.func.Inputs(intra=current[3:])}
        if network
        else dn.func.Inputs(intra=current[3:])
    )
    prefix, first = runner(parameters=parameters, inputs=first_inputs)
    actual, last = runner(first["continuation"], inputs=last_inputs)
    joined = _joined(first, last)
    loss = _voltage(actual, network).square().sum() + sum(
        value.square().mean() for name, value in joined.items() if name != "t"
    )
    actual_gradients = torch.autograd.grad(loss, (upstream, current), retain_graph=True)
    expected, whole = runner.functional.prepare_and_rollout(
        parameters,
        runner.constants,
        runner.initial_state,
        inputs,
        steps=6,
        callbacks=runner.callbacks,
    )
    reference_loss = _voltage(expected, network).square().sum() + sum(
        value.square().mean()
        for name, value in whole["callbacks"]["trace"].items()
        if name != "t"
    )
    reference_gradients = torch.autograd.grad(reference_loss, (upstream, current))
    _close(actual, expected)
    _close(joined, whole["callbacks"]["trace"])
    _close(actual_gradients, reference_gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual_gradients
    )
    assert current.grad is None
    assert joined["left.v" if network else "v"].shape[0] == 7
    assert first["continuation"].state["clock"]["t"] is prefix["clock"]["t"]


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize(
    "keyword", ("parameters", "constants", "callback_state", "initialization")
)
def test_packet_rejects_competing_execution_sources_before_advancing(network, keyword):
    runner = _runner(network)
    _, auxiliary = runner()
    continuation = auxiliary["continuation"]
    before = _snapshot(continuation.state)
    value = continuation.callback_state if keyword == "callback_state" else {}
    with pytest.raises((TypeError, ValueError), match=keyword):
        runner(continuation, **{keyword: value})
    _close(continuation.state, before)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("changed", ("state", "parameters", "source", "callback"))
def test_packet_preserves_existing_shape_source_alias_and_callback_plan_validation(
    network, changed
):
    runner = _runner(network)
    _, auxiliary = runner()
    current = auxiliary["continuation"]
    state, parameters = _plain(current.state), _plain(current.parameters)
    callback_state = current.callback_state
    if changed == "state":
        selected = state["populations"]["left"] if network else state
        selected["integrator"]["v"] = selected["integrator"]["v"].reshape(-1)
    elif changed == "parameters":
        _assign(parameters, network, _target(parameters, network).unsqueeze(0))
    elif changed == "source":
        runner.functional._source.eval() if network else runner.functional._population.eval()
    else:
        _, other = _runner(network)()
        callback_state = other["continuation"].callback_state
    packet = dn.func.Continuation(state, parameters, current.constants, callback_state)
    expected = _failure(
        lambda: runner(
            state,
            parameters=parameters,
            constants=current.constants,
            callback_state=callback_state,
        )
    )
    actual = _failure(lambda: runner(packet))
    assert actual == expected


@pytest.mark.parametrize("network", (False, True))
def test_packet_has_an_explicit_raw_binding_bridge_without_rebinding_existing_sources(
    network,
):
    runner = _runner(network)
    _, first = runner()
    continuation = first["continuation"]
    bound = runner.kernel.bind(
        continuation.parameters, constants=continuation.constants
    )
    expected, last = runner(continuation)
    actual, auxiliary = dn.func.run(
        bound,
        continuation.state,
        steps=3,
        callbacks=runner.callbacks,
        callback_state=continuation.callback_state,
    )
    _close(actual, expected)
    _close(auxiliary["callbacks"].outputs, last["callbacks"].outputs)
    returned = auxiliary["continuation"]
    _close(returned.state, actual)
    assert _target(returned.parameters, network) is _target(
        continuation.parameters, network
    )
    assert returned.callback_state.plan_token is continuation.callback_state.plan_token


def test_population_packet_resume_retains_existing_inference_cache_admission():
    runner = _runner(False, callbacks=False, inference=True)
    _, first = runner()
    cached = runner._prepared
    assert cached is not None
    runner(first["continuation"])
    assert runner._prepared is cached
    with torch.no_grad():
        _target(runner.parameters, False).add_(0.0002)
    runner(first["continuation"])
    assert runner._prepared is cached
    runner()
    assert runner._prepared is not cached


@pytest.mark.parametrize("network", (False, True))
def test_failed_continuation_publication_restores_the_last_successful_report(
    network, monkeypatch
):
    runner = _runner(network)
    _, first = runner()
    continuation = first["continuation"]
    previous = runner.execution_report()
    assert previous is not None
    before = _snapshot(continuation.state)
    source = _snapshot(runner.functional.extract().state)

    def reject(*args, **kwargs):
        raise ValueError("continuation publication rejected")

    monkeypatch.setattr(dn.func.Continuation, "__init__", reject)
    with pytest.raises(ValueError, match="continuation publication rejected"):
        runner(continuation)
    assert runner.execution_report() is previous
    _close(continuation.state, before)
    _close(runner.functional.extract().state, source)


@pytest.mark.parametrize("network", (False, True))
def test_packet_keeps_no_runner_or_recording_history_alive(network):
    runner = _runner(network, inference=True)
    _, auxiliary = runner()
    continuation = auxiliary["continuation"]
    runner_reference = weakref.ref(runner)
    trace_reference = weakref.ref(
        auxiliary["callbacks"]["trace"]["left.v" if network else "v"]
    )
    del runner, auxiliary
    gc.collect()
    assert runner_reference() is None
    assert trace_reference() is None
    assert torch.isfinite(_voltage(continuation.state, network)).all()


@pytest.mark.parametrize("network", (False, True))
def test_detached_portable_checkpoint_rebuilds_an_in_memory_packet_with_existing_callback_bridge(
    network, tmp_path
):
    runner = _runner(network, inference=True)
    _, first = runner()
    continuation = first["continuation"]
    path = tmp_path / "continuation.pt"
    torch.save(
        _snapshot(
            {
                "state": continuation.state,
                "parameters": continuation.parameters,
                "constants": continuation.constants,
                "callbacks": runner.callbacks.export_state(continuation.callback_state),
            }
        ),
        path,
    )
    saved = torch.load(path, weights_only=True)
    receiver = _runner(network, inference=True)
    restored = dn.func.Continuation(
        saved["state"],
        saved["parameters"],
        saved["constants"],
        receiver.callbacks.restore_state(saved["callbacks"]),
    )
    expected, reference = runner(continuation)
    actual, auxiliary = receiver(restored)
    _close(actual, expected)
    _close(auxiliary["callbacks"].outputs, reference["callbacks"].outputs)
    if network:
        assert (
            restored.parameters["populations"]["left"][G]
            is restored.parameters["populations"]["right"][G]
        )


@pytest.mark.parametrize("network", (False, True))
def test_runtime_extra_is_supplied_again_when_resuming_a_packet(network):
    if network:
        from tests.func.test_network_extra import DT as extra_dt
        from tests.func.test_network_extra import _network as extra_network
        from tests.func.test_network_runtime_extra import _field

        model = extra_network(batch=None)
        field = _field(model)
        paths = ("left.v", "right.v", "t")
    else:
        from tests.func.test_runtime_extra import DT as extra_dt
        from tests.func.test_runtime_extra import _field, _model

        model = _model(batch=None)
        field = _field(model)
        paths = ("v", "t")
    runner = dn.compile(
        model,
        dt=extra_dt,
        steps=2,
        extra=dn.VarInput(),
        callbacks={"trace": dn.func.Recorder(paths)},
        backend="eager",
        disable=True,
    )
    first_extra = (field, dn.constant(value=0.1))
    last_extra = [(field * -0.7, dn.mono_rect(amp=0.2, delay=0.0, pw=1.0))]
    if network:
        first_extra, last_extra = {"left": first_extra}, {"right": last_extra}
    _, auxiliary = runner(extra=first_extra)
    continuation = auxiliary["continuation"]
    with pytest.raises(TypeError, match="extra"):
        runner(continuation)
    actual, last = runner(continuation, extra=last_extra)
    expected, reference = runner(
        continuation.state,
        parameters=continuation.parameters,
        constants=continuation.constants,
        callback_state=continuation.callback_state,
        extra=last_extra,
    )
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
