"""Population execution selects useful captures without changing its results."""

from __future__ import annotations

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_runner_stimulation_clock import DT as PULSE_DT
from tests.func.test_runner_stimulation_clock import STEPS as PULSE_STEPS
from tests.func.test_runner_stimulation_clock import _case as _pulse_case
from tests.func.test_runner_stimulation_clock import _source_case as _native_pulse_case
from tests.func.test_runtime_extra import DT, _close
from tests.func.test_runtime_extra import _model as _passive

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def _case():
    functional, tensors = dn.func.make_functional(_passive(batch=()), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    return functional, tensors, prepared


def _observe_structured_captures(monkeypatch, functional):
    original = functional._capture_structured_step_graph
    captures = []

    def capture(*args):
        captures.append(args)
        return original(*args)

    monkeypatch.setattr(functional, "_capture_structured_step_graph", capture)
    return captures


def _activity():
    from torch._dynamo import utils

    return deepcopy(utils.counters), len(utils.get_compilation_metrics())


def _inputs(state, steps, *, requires_grad=False):
    voltage = state["integrator"]["v"]
    current = torch.linspace(
        -0.1e-9,
        0.3e-9,
        steps * voltage.numel(),
        dtype=voltage.dtype,
        device=voltage.device,
    ).reshape(steps, *voltage.shape)
    current.requires_grad_(requires_grad)
    return dn.func.Inputs(intra=current)


@pytest.mark.parametrize("steps", (1, 2, 3))
def test_short_default_inference_preserves_capture_and_reuses_compiled_tail(
    monkeypatch, steps
):
    functional, tensors, prepared = _case()
    captures = _observe_structured_captures(monkeypatch, functional)
    inputs = _inputs(tensors.state, steps)
    kernel = functional.compile_rollout_chunk(steps, backend="aot_eager")
    with torch.no_grad():
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, inputs, steps=steps
        )
        actual = kernel(tensors.parameters, prepared, tensors.state, inputs)
        _close(actual, expected)
        report = kernel.execution_report()
        assert report.strategy == "compiled_unrolled"
        assert report.reason == "The chunk is shorter than one inference loop block."
        assert report.steps == steps and report.grad_enabled is False
        assert captures == [(False, True)]
        assert len(functional._structured_step_graphs) == 1
        admitted = _activity()
        _close(kernel(tensors.parameters, prepared, tensors.state, inputs), expected)
        assert _activity() == admitted and captures == [(False, True)]


def test_disabled_default_inference_does_not_capture_a_compiler_only_graph(monkeypatch):
    functional, tensors, prepared = _case()
    captures = _observe_structured_captures(monkeypatch, functional)
    inputs = _inputs(tensors.state, 4)
    kernel = functional.compile_rollout_chunk(4, backend="aot_eager", disable=True)
    with torch.no_grad():
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, inputs, steps=4
        )
        actual = kernel(tensors.parameters, prepared, tensors.state, inputs)
    _close(actual, expected)
    report = kernel.execution_report()
    assert report.strategy == "eager_disabled"
    assert report.reason == "Compilation was explicitly disabled."
    assert report.grad_enabled is False
    assert captures == [] and not functional._structured_step_graphs


def test_eligible_default_inference_captures_once_and_reuses_its_drive_signature(
    monkeypatch,
):
    functional, tensors, prepared = _case()
    captures = _observe_structured_captures(monkeypatch, functional)
    inputs = _inputs(tensors.state, 4)
    kernel = functional.compile_rollout_chunk(4, backend="aot_eager")
    with torch.no_grad():
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, inputs, steps=4
        )
        actual = kernel(tensors.parameters, prepared, tensors.state, inputs)
        _close(actual, expected)
        assert captures == [(False, True)]
        report = kernel.execution_report()
        assert report.strategy == "compiled_while_loop" and report.reason is None
        admitted = _activity()
        _close(kernel(tensors.parameters, prepared, tensors.state, inputs), expected)
        assert captures == [(False, True)] and _activity() == admitted


def test_explicit_structured_prewarm_remains_available_for_a_short_default_chunk(
    monkeypatch,
):
    functional, tensors, prepared = _case()
    captures = _observe_structured_captures(monkeypatch, functional)
    inputs = _inputs(tensors.state, 2)
    assert functional.prewarm_structured_rollout(intra=True)
    assert functional.prewarm_structured_rollout(intra=True)
    assert captures == [(False, True)]
    kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
    with torch.no_grad():
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, inputs, steps=2
        )
        _close(kernel(tensors.parameters, prepared, tensors.state, inputs), expected)
    assert captures == [(False, True)]
    assert kernel.execution_report().strategy == "compiled_unrolled"
    assert kernel.execution_report().reason == (
        "The chunk is shorter than one inference loop block."
    )


@pytest.mark.parametrize("stimulus_kind", ("intra", "extra"))
def test_stepwise_bound_waveforms_skip_legacy_capture_and_keep_accepted_clock(
    monkeypatch, stimulus_kind
):
    functional, tensors, parameters, prepared, _amplitude = _pulse_case(stimulus_kind)
    captures = _observe_structured_captures(monkeypatch, functional)
    kernel = functional.compile_rollout_chunk(PULSE_STEPS, backend="aot_eager")
    with torch.no_grad():
        stepped = tensors.state
        for _ in range(PULSE_STEPS):
            stepped, _ = functional.step(parameters, prepared, stepped)
        native, extra, _waveform = _native_pulse_case(stimulus_kind)
        native.run(steps=PULSE_STEPS, dt=PULSE_DT, extra=extra)
        expected = functional.extract(native).state
        actual, _ = dn.func.run(
            functional,
            partial(kernel, parameters, prepared),
            tensors.state,
            tstop=PULSE_STEPS * PULSE_DT,
        )
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual["integrator"]["v"], stepped["integrator"]["v"])
    report = kernel.execution_report()
    assert report.strategy == "compiled_unrolled"
    assert report.reason == "Bound waveforms are sampled at each accepted timestep."
    assert captures == [] and not functional._structured_step_graphs


@pytest.mark.parametrize("tail", (2, 3))
def test_callback_free_while_tail_preserves_capture_and_restores_checkpoint(
    monkeypatch, tail
):
    program = dn.compile(
        _passive(batch=()),
        dt=DT,
        steps=4 + tail,
        chunk_steps=4,
        execution="while",
        backend="aot_eager",
    )
    functional = program.functional
    captures = _observe_structured_captures(monkeypatch, functional)
    inputs = _inputs(program.initial_state, 4 + tail)
    prefix, auxiliary = program(steps=4, inputs=dn.func.Inputs(intra=inputs.intra[:4]))
    assert program.execution_report().strategy == "compiled_while_loop"
    packet = auxiliary["continuation"]
    restored = program.restore_continuation(program.export_continuation(packet))
    for old, new in zip(
        pytree.tree_leaves(packet.parameters),
        pytree.tree_leaves(restored.parameters),
        strict=True,
    ):
        if torch.is_tensor(old):
            assert old is not new
    suffix_inputs = dn.func.Inputs(intra=inputs.intra[4:])
    actual, _ = program(restored, steps=tail, inputs=suffix_inputs)
    with torch.no_grad():
        expected, _ = functional.prepare_and_rollout(
            restored.parameters,
            restored.constants,
            program.initial_state,
            inputs=inputs,
            steps=4 + tail,
        )
    _close(actual, expected)
    report = program.execution_report()
    assert report.strategy == "compiled_unrolled" and report.steps == tail
    assert report.reason == "The chunk is shorter than one inference loop block."
    assert captures == [(False, True)]
    assert len(functional._structured_step_graphs) == 1
    admitted = _activity()
    _close(program(restored, steps=tail, inputs=suffix_inputs)[0], actual)
    assert _activity() == admitted and captures == [(False, True)]
    assert torch.all(actual["clock"]["t"] > prefix["clock"]["t"])


@pytest.mark.parametrize("disabled", (False, True), ids=("training", "disabled"))
def test_short_training_route_preserves_reason_priority_and_live_input_gradients(
    monkeypatch, disabled
):
    functional, tensors, _ = _case()
    captures = _observe_structured_captures(monkeypatch, functional)
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    prepared = functional.prepare(parameters, tensors.constants)
    state = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )
    voltage = state["integrator"]["v"].requires_grad_()
    inputs = _inputs(state, 2, requires_grad=True)
    kernel = functional.compile_rollout_chunk(2, backend="aot_eager", disable=disabled)
    with torch.enable_grad():
        actual = kernel(parameters, prepared, state, inputs)
        expected = functional.rollout(parameters, prepared, state, inputs, steps=2)
        _close(actual, expected)
        actual_gradients = torch.autograd.grad(
            actual[0]["integrator"]["v"].square().mean(), (voltage, inputs.intra)
        )
        expected_gradients = torch.autograd.grad(
            expected[0]["integrator"]["v"].square().mean(), (voltage, inputs.intra)
        )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient) > 0
        torch.testing.assert_close(actual_gradient, expected_gradient)
    report = kernel.execution_report()
    assert report.grad_enabled is True
    assert report.strategy == ("eager_disabled" if disabled else "compiled_unrolled")
    assert report.reason == (
        "Compilation was explicitly disabled."
        if disabled
        else "Gradient recording uses the unrolled recurrence."
    )
    assert captures == [] and not functional._structured_step_graphs


def test_short_rand_only_inference_preserves_owned_rng_and_continuation(monkeypatch):
    from tests.func.test_random_parameters import DT as random_dt
    from tests.func.test_random_parameters import _population, _step

    source, native = _population(), _population()
    functional, tensors = dn.func.make_functional(source, dt=random_dt)
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    prepared = functional.prepare(parameters, tensors.constants)
    captures = _observe_structured_captures(monkeypatch, functional)
    saved = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )
    saved_parameters = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), parameters
    )
    rng = torch.get_rng_state().clone()
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")

    def owned_random(result, *previous):
        leaves = pytree.tree_leaves(result["random"])
        sources = [
            value for state in previous for value in pytree.tree_leaves(state["random"])
        ]
        assert leaves and len(leaves) == len(pytree.tree_leaves(saved["random"]))
        assert all(torch.is_tensor(value) for value in (*leaves, *sources))
        for value in leaves:
            assert all(not torch._C._is_alias_of(value, other) for other in sources)

    with torch.no_grad():
        actual, _ = kernel(parameters, prepared, tensors.state)
        expected, _ = functional.rollout(parameters, prepared, tensors.state, steps=3)
        _close(actual, expected)
        for _ in range(3):
            _step(native)
        _close(actual, functional.extract(native).state)
        owned_random(actual, tensors.state)
        assert captures == [(False, False)]
        admitted = _activity()
        repeated, _ = kernel(parameters, prepared, tensors.state)
        _close(repeated, actual)
        owned_random(repeated, tensors.state, actual)
        assert _activity() == admitted

        continued, _ = kernel(parameters, prepared, actual)
        expected, _ = functional.rollout(parameters, prepared, tensors.state, steps=6)
        _close(continued, expected)
        for _ in range(3):
            _step(native)
        _close(continued, functional.extract(native).state)
        owned_random(continued, tensors.state, actual, repeated)
        assert captures == [(False, False)]
    # RAND samples and stream counters belong to initialization. No rollout
    # draws them again, and the caller's original state stays independent.
    torch.testing.assert_close(continued["random"], saved["random"], rtol=0, atol=0)
    torch.testing.assert_close(tensors.state, saved, rtol=0, atol=0)
    torch.testing.assert_close(parameters, saved_parameters, rtol=0, atol=0)
    torch.testing.assert_close(functional.extract().state, saved, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
