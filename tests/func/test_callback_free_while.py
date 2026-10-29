"""Explicit while inference does not require recording or callback carry."""

from __future__ import annotations

import copy
import io
import os
from contextlib import contextmanager

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population_while import _PopulationWhileExecutor
from tests.func.test_runtime_extra import DT, _close, _field
from tests.func.test_runtime_extra import _model as _passive
from tests.func.test_runtime_noise import DT as NOISE_DT
from tests.func.test_runtime_noise import _population as _noisy
from tests.func.test_stimulation_rng_sharing import _clone, _slice

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
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
        torch.set_num_threads(previous)


def _without_callbacks(result):
    assert "callbacks" not in result[1]
    assert result[1]["continuation"].callback_state is None
    assert not result[0]["integrator"]["v"].requires_grad


def _reference(program, *, steps, inputs=None, draws=None):
    with torch.no_grad():
        return program.functional.prepare_and_rollout(
            program.parameters,
            program.constants,
            program.initial_state,
            inputs=inputs,
            draws=draws,
            steps=steps,
        )


@pytest.mark.parametrize(
    "callback_options,backend",
    (({}, "aot_eager"), ({"callbacks": {}}, "inductor")),
    ids=("omitted-AOT", "empty-Inductor"),
)
def test_callback_free_while_matches_native_and_scan_without_recording(
    callback_options,
    backend,
):
    model = _passive(batch=())
    native = copy.deepcopy(model)
    original = _clone(model.state_dict())
    inputs = dn.func.Inputs(
        ve=torch.stack([_field(model) * (0.1 + index / 20) for index in range(8)])
    )
    program = dn.compile(
        model,
        dt=DT,
        steps=8,
        chunk_steps=4,
        execution="while",
        inference=True,
        backend=backend,
        **callback_options,
    )
    actual = program(inputs=inputs)
    _without_callbacks(actual)
    assert program.callbacks is None
    report = program.execution_report()
    assert report.strategy == "compiled_while_loop" and report.callbacks is False
    scan = dn.compile(model, dt=DT, steps=8, chunk_steps=4, backend="eager")
    expected = scan(inputs=inputs)
    _close(actual[0], expected[0])
    with torch.no_grad():
        for frame in inputs.ve:
            native.step(DT, ve=frame)
    _close(actual[0], program.functional.extract(native).state)
    _close(program(inputs=inputs)[0], actual[0])
    _close(model.state_dict(), original)


@contextmanager
def _observe_private_boundaries(monkeypatch, program):
    """Observe only host request/selection methods, leaving public owners exact."""
    draws, executors = [], []
    plan = program.functional._runtime_draw_plan
    sample = plan.sample
    kernel_for = _PopulationWhileExecutor.kernel_for

    def sampled(state, *, steps=None):
        draws.append(steps)
        return sample(state, steps=steps)

    def selected(executor, *args, **kwargs):
        if executor._functional is program.functional:
            executors.append(
                (
                    executor._steps,
                    id(executor),
                    len(executor._plans),
                )
            )
        return kernel_for(executor, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(plan, "sample", sampled)
        scoped.setattr(_PopulationWhileExecutor, "kernel_for", selected)
        yield draws, executors


def test_callback_free_bounded_noise_replays_and_restores_a_portable_continuation(
    monkeypatch,
):
    model = _noisy()
    source = dn.compile(
        model, dt=NOISE_DT, steps=9, chunk_steps=4, execution="while", backend="eager"
    )
    original = _clone(source.functional.extract())
    global_rng = torch.get_rng_state().clone()
    tape = source.functional.sample_runtime_draws(source.initial_state, steps=18)
    frozen = _clone(tape)
    with _observe_private_boundaries(monkeypatch, source) as (draws, executors):
        automatic = source()
    assert draws and all(0 < count <= 4 for count in draws)
    assert executors and all(count <= 4 and plans == 0 for count, _, plans in executors)
    with _observe_private_boundaries(monkeypatch, source) as (draws, _):
        explicit = source(draws=_slice(tape, 0, 9))
    assert draws == []
    _without_callbacks(automatic)
    _without_callbacks(explicit)
    _close(automatic[0], explicit[0])
    saved = io.BytesIO()
    torch.save(source.export_continuation(explicit[1]["continuation"]), saved)
    saved.seek(0)
    receiver = dn.compile(
        _noisy(),
        dt=NOISE_DT,
        steps=9,
        chunk_steps=5,
        execution="while",
        backend="eager",
        callbacks={},
    )
    restored = receiver.restore_continuation(torch.load(saved, weights_only=True))
    assert restored.callback_state is None
    resumed = receiver(restored)
    _without_callbacks(resumed)
    _close(resumed[0], _reference(source, steps=18, draws=tape)[0])
    _close(source.functional.extract(), original)
    _close(tape, frozen)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("stimulation", ("registered", "bound", "runtime"))
def test_callback_free_stimulation_uses_the_same_effective_executor_and_native_clock(
    monkeypatch,
    stimulation,
):
    model = _passive(batch=())
    recipe = (_field(model), dn.mono_rect(amp=0.2, delay=0.19, pw=0.025))
    if stimulation == "registered":
        model[:, 1].inject(dn.mono_rect(amp=0.1 * dn.units.nA, delay=0.19, pw=0.025))
        model.initialize()
        model.t.fill_(0.175)
    native = copy.deepcopy(model)
    original = _clone(model.state_dict())
    program = dn.compile(
        model,
        dt=DT,
        steps=5,
        chunk_steps=4,
        execution="while",
        backend="eager",
        extra=recipe
        if stimulation == "bound"
        else dn.VarInput()
        if stimulation == "runtime"
        else None,
    )
    with _observe_private_boundaries(monkeypatch, program) as (_, executors):
        actual = program(**({"extra": recipe} if stimulation == "runtime" else {}))
    _without_callbacks(actual)
    full = [identity for count, identity, _ in executors if count == 4]
    assert len(full) >= 2 and len(set(full)) == 1
    assert all(plans == 0 for _, _, plans in executors)
    with torch.no_grad():
        for _ in range(5):
            native.step(DT, extra=recipe if stimulation != "registered" else None)
    _close(actual[0], program.functional.extract(native).state)
    _close(model.state_dict(), original)


def test_callback_free_capture_failure_preserves_backend_and_last_report(monkeypatch):
    graphs = []

    def backend(graph, operands):
        graphs.append(graph)
        return graph.forward

    program = dn.compile(
        _passive(batch=()),
        dt=DT,
        steps=8,
        chunk_steps=4,
        execution="while",
        backend=backend,
    )

    def unavailable(*args, **kwargs):
        return None, "sentinel callback-free inference capture"

    with monkeypatch.context() as scoped:
        scoped.setattr(_PopulationWhileExecutor, "kernel_for", unavailable)
        with pytest.raises(dn.func.FunctionalizationError, match="sentinel"):
            program()
    assert graphs == [] and program.execution_report() is None
    _without_callbacks(program())
    assert graphs
    count, report = len(graphs), program.execution_report()
    with monkeypatch.context() as scoped:
        scoped.setattr(_PopulationWhileExecutor, "kernel_for", unavailable)
        with pytest.raises(dn.func.FunctionalizationError, match="sentinel"):
            program()
    assert len(graphs) == count
    assert program.execution_report() is report

    kernel_for = _PopulationWhileExecutor.kernel_for
    matching_calls = []

    def fail_during_execution(executor, *args, **kwargs):
        if executor._functional is program.functional:
            matching_calls.append(executor)
            if len(matching_calls) > 1:
                return None, "sentinel execution capture"
        return kernel_for(executor, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(_PopulationWhileExecutor, "kernel_for", fail_during_execution)
        with pytest.raises(dn.func.FunctionalizationError, match="sentinel execution"):
            program()
    assert len(matching_calls) >= 2
    assert len(graphs) == count
    assert program.execution_report() is report


def test_failed_while_admission_does_not_enable_model_route_for_a_raw_empty_packet(
    monkeypatch,
):
    program = dn.compile(
        _passive(batch=()), dt=DT, steps=4, execution="while", backend="eager"
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            _PopulationWhileExecutor,
            "kernel_for",
            lambda *args, **kwargs: (None, "sentinel failed admission"),
        )
        with pytest.raises(dn.func.FunctionalizationError, match="failed admission"):
            program()
    assert program.execution_report() is None

    def unexpected(*args, **kwargs):
        pytest.fail("raw empty packet selected a model-only while executor")

    monkeypatch.setattr(_PopulationWhileExecutor, "kernel_for", unexpected)
    functional = program.functional
    with torch.no_grad():
        prepared = functional.prepare(program.parameters, program.constants)
        packet = functional.sample_runtime_draws(program.initial_state, steps=4)
        actual, _ = program.kernel(
            program.parameters, prepared, program.initial_state, draws=packet
        )
        expected, _ = functional.prepare_and_rollout(
            program.parameters,
            program.constants,
            program.initial_state,
            steps=4,
            draws=packet,
        )
    _close(actual, expected)
    assert program.kernel.execution_report().strategy == "compiled_unrolled"


@pytest.mark.parametrize("callbacks", (None, {}), ids=("none", "empty"))
def test_callback_state_without_callbacks_is_rejected_before_capture(
    monkeypatch, callbacks
):
    program = dn.compile(
        _passive(batch=()),
        dt=DT,
        steps=4,
        execution="while",
        backend="eager",
        callbacks=callbacks,
    )

    def unexpected(*args, **kwargs):
        pytest.fail("callback_state without callbacks reached while capture")

    monkeypatch.setattr(_PopulationWhileExecutor, "kernel_for", unexpected)
    with pytest.raises(ValueError, match="callback_state requires callbacks"):
        program(program.initial_state, callback_state={})


@pytest.mark.parametrize(
    "invalid", ("training", "disabled", "short_horizon", "short_chunk", "custom")
)
def test_callback_free_admission_keeps_policies_and_accepts_pure_callbacks(invalid):
    class Custom(dn.func.FunctionalCallback):
        def initialize(self, state, auxiliary):
            return state["integrator"]["v"].new_zeros(()), None

        def update(self, carry, state, auxiliary):
            return carry + state["integrator"]["v"].sum(), None

        def finalize(self, carry, emissions):
            return carry

    choices = {
        "training": ({"inference": False}, "inference"),
        "disabled": ({"disable": True}, "enabled|disabled"),
        "short_horizon": ({"steps": 3}, "four|4"),
        "short_chunk": ({"chunk_steps": 3}, "four|4"),
        "custom": ({"callbacks": {"custom": Custom()}}, "callback|built-in"),
    }
    options = dict(dt=DT, steps=4, execution="while", backend="eager")
    keywords, match = choices[invalid]
    options.update(keywords)
    if invalid == "custom":
        program = dn.compile(_passive(batch=()), **options)
        actual, auxiliary = program()
        with torch.no_grad():
            expected, reference = program.functional.prepare_and_rollout(
                program.parameters,
                program.constants,
                program.initial_state,
                steps=4,
                callbacks=program.callbacks,
            )
        _close(actual, expected)
        _close(auxiliary["callbacks"]["custom"], reference["callbacks"]["custom"])
        assert program.execution_report().strategy == "compiled_while_loop"
    else:
        with pytest.raises((ValueError, dn.func.FunctionalizationError), match=match):
            dn.compile(_passive(batch=()), **options)()


@pytest.mark.mps
def test_mps_callback_free_while_has_no_recording_and_matches_eager_state():
    from tests.func.test_runtime_extra_backends import _environment as environment
    from tests.func.test_runtime_extra_backends import _field as field
    from tests.func.test_runtime_extra_backends import _model as model_factory

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with environment("mps"):
        model = model_factory("mps")
        program = dn.compile(
            model, dt=DT, steps=8, chunk_steps=4, execution="while", backend="aot_eager"
        )
        inputs = dn.func.Inputs(
            ve=torch.stack([field(model) * (0.1 + index / 20) for index in range(8)])
        )
        actual = program(inputs=inputs)
        expected = _reference(program, steps=8, inputs=inputs)
        _without_callbacks(actual)
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-5, atol=2e-5)
        assert program.execution_report().strategy == "compiled_while_loop"
        assert program.execution_report().callbacks is False
