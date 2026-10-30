"""Convenience runners share explicit scheduling horizons and continuation."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_lowering_api import (
    DT,
    _clone,
    _close,
    _network,
    _population,
    _target,
)
from tests.func.test_recorder_cadence import _inputs
from tests.func.test_runner_chunking import _kernel_cache, _stochastic_runner
from tests.func.test_stimulation_rng_sharing import DT as RANDOM_DT
from tests.func.test_stimulation_rng_sharing import _slice

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


def _model(network):
    return _network() if network else _population()


def _callbacks(network):
    return {
        "trace": dn.func.Recorder(
            ("left.v", "t") if network else ("v", "t"), dt=3.5 * DT
        )
    }


def _reference(program, state=None, **keywords):
    bound = program.functional.bind(program.parameters, constants=program.constants)
    return dn.func.run(
        bound,
        program.initial_state if state is None else state,
        callbacks=program.callbacks,
        host_span_steps=program.kernel.steps,
        **keywords,
    )


def _same(actual, expected):
    _close(actual[0], expected[0])
    _close(actual[1]["callbacks"], expected[1]["callbacks"])


def _pending(state, network):
    return state["clock" if network else "control"]["duration_remainder"].item()


def _voltage_loss(result, network):
    key = "left.v" if network else "v"
    return result[1]["callbacks"]["trace"][key].square().sum()


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_compile_horizon_errors_precede_lowering(monkeypatch, network):
    import dendra.func._convenience as population_module
    import dendra.func._network_compile as network_module

    def unexpected(*args, **kwargs):
        pytest.fail("invalid default horizon reached functional lowering")

    monkeypatch.setattr(population_module, "make_functional", unexpected)
    monkeypatch.setattr(network_module, "_lower_network", unexpected)
    factory = dn.func.CompiledNetworkRunner if network else dn.func.CompiledRunner
    for options in (
        {},
        {"steps": 4, "duration": DT},
        {"steps": True},
        {"steps": 0},
        {"steps": -1},
        {"steps": 1.5},
        {"duration": True},
        {"duration": -DT},
        {"duration": float("nan")},
        {"duration": float("inf")},
    ):
        with pytest.raises((TypeError, ValueError)):
            factory(object(), dt=DT, backend="eager", **options)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_call_horizon_errors_preserve_snapshot_and_execution_report(network):
    program = dn.compile(
        _model(network), dt=DT, steps=7, chunk_steps=4, backend="eager"
    )
    snapshot = _clone(program.initial_state)
    report = program.execution_report()
    for options in (
        {"steps": 1, "duration": DT},
        {"steps": True},
        {"steps": -1},
        {"steps": 1.5},
        {"duration": True},
        {"duration": -DT},
        {"duration": float("nan")},
        {"duration": float("inf")},
    ):
        with pytest.raises((TypeError, ValueError)):
            program(**options)
        _close(program.initial_state, snapshot)
        assert program.execution_report() is report


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_step_default_and_call_overrides_reuse_chunks_without_changing_default(
    network, execution
):
    program = dn.compile(
        _model(network),
        dt=DT,
        steps=7,
        chunk_steps=4,
        execution=execution,
        backend="eager",
        callbacks=_callbacks(network),
    )
    kernel = program.kernel
    with torch.no_grad():
        _same(program(), _reference(program, steps=7))
        _same(program(steps=13), _reference(program, steps=13))
        _same(program(duration=2.4 * DT), _reference(program, duration=2.4 * DT))
        cached = _kernel_cache(program, network)
        _same(program(steps=13), _reference(program, steps=13))
        assert _kernel_cache(program, network) == cached
        _same(program(), _reference(program, steps=7))
        zero = program(steps=0)
    assert program.kernel is kernel and kernel.steps == 4 and program._steps == 7
    _close(zero[0], program.initial_state)
    assert zero[1]["callbacks"]["trace"]["t"].shape == (1,)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_duration_defaults_accumulate_fractional_time_and_resume_callback_phase(
    network, execution
):
    program = dn.compile(
        _model(network),
        dt=DT,
        duration=0.6 * DT,
        chunk_steps=4,
        execution=execution,
        backend="eager",
        callbacks=_callbacks(network),
    )
    with torch.no_grad():
        first = program()
        expected = _reference(program, duration=0.6 * DT)
        _same(first, expected)
        assert _pending(first[0], network) == pytest.approx(0.6 * DT)
        assert first[1]["callbacks"]["trace"]["t"].shape == (1,)
        report = program.execution_report()
        empty = program(first[1]["continuation"], duration=0.0)
        _same(empty, _reference(program, expected[1]["continuation"], duration=0.0))
        assert program.execution_report() is report
        assert empty[1]["callbacks"]["trace"]["t"].shape == (0,)
        second = program(empty[1]["continuation"])
        expected = _reference(program, expected[1]["continuation"], duration=0.6 * DT)
        _same(second, expected)
        assert _pending(second[0], network) == pytest.approx(0.2 * DT)
        assert second[1]["callbacks"]["trace"]["t"].shape == (0,)
        third = program(second[1]["continuation"], steps=7)
        expected = _reference(program, expected[1]["continuation"], steps=7)
        _same(third, expected)
        assert _pending(third[0], network) == pytest.approx(0.2 * DT)
        restored = program.restore_continuation(
            program.export_continuation(third[1]["continuation"])
        )
        final = program(restored)
        expected = _reference(program, expected[1]["continuation"], duration=0.6 * DT)
        _same(final, expected)
        assert _pending(final[0], network) == pytest.approx(0.8 * DT)
        # An independent no-state call restarts the same duration and phase.
        _same(program(), first)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_zero_duration_default_retains_initial_frame_without_advancing(network):
    program = dn.compile(
        _model(network),
        dt=DT,
        duration=0.0,
        backend="eager",
        callbacks=_callbacks(network),
    )
    initial = _clone(program.initial_state)
    first = program()
    second = program(first[1]["continuation"])
    _close(first[0], initial)
    _close(second[0], initial)
    assert first[1]["callbacks"]["trace"]["t"].shape == (1,)
    assert second[1]["callbacks"]["trace"]["t"].shape == (0,)
    assert program.kernel.steps >= 1


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_callback_free_zero_horizons_complete_without_capture(network, execution):
    for initialization in ("snapshot", "fresh"):
        graphs = []

        def backend(graph, _examples):
            graphs.append(graph)
            return graph.forward

        program = dn.compile(
            _model(network),
            dt=DT,
            duration=0.0,
            execution=execution,
            initialization=initialization,
            backend=backend,
        )
        initial = _clone(program.initial_state)
        report = program.execution_report()
        assert graphs == []
        first = program()
        second = program(first[1]["continuation"], steps=0)
        for state, auxiliary in (first, second):
            _close(state, initial)
            assert "callbacks" not in auxiliary
            assert auxiliary["continuation"].callback_state is None
            _close(auxiliary["continuation"].state, state)
            if initialization == "fresh":
                assert "parameters" in auxiliary and "constants" in auxiliary
        assert program.execution_report() is report
        assert graphs == []


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_variable_checkpointed_horizons_preserve_input_and_parameter_vjps(network):
    model = _model(network)
    program = dn.compile(
        model,
        dt=DT,
        duration=4.4 * DT,
        chunk_steps=4,
        inference=False,
        checkpoint_every=5,
        trainable="*g_param",
        backend="aot_eager",
        callbacks=_callbacks(network),
    )
    shape = model.left.shape if network else model.shape
    tape = torch.linspace(1e-7, 4e-7, 9 * torch.Size(shape).numel()).reshape(9, *shape)
    tape.requires_grad_()
    targets = (_target(program.parameters, network), tape)
    reference = _reference(program, duration=9.2 * DT, inputs=_inputs(network, tape))
    expected = torch.autograd.grad(_voltage_loss(reference, network), targets)
    actual = program(duration=9.2 * DT, inputs=_inputs(network, tape))
    _same(actual, reference)
    observed = torch.autograd.grad(_voltage_loss(actual, network), targets)
    _close(observed, expected)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    prefix = program(steps=1, inputs=_inputs(network, tape[:1]))
    suffix = program(
        prefix[1]["continuation"], duration=8.2 * DT, inputs=_inputs(network, tape[1:])
    )
    _close(suffix[0], actual[0])
    key = "left.v" if network else "v"
    joined = torch.cat(
        (prefix[1]["callbacks"]["trace"][key], suffix[1]["callbacks"]["trace"][key])
    )
    _close(joined, actual[1]["callbacks"]["trace"][key])
    observed = torch.autograd.grad(joined.square().sum(), targets)
    _close(observed, expected)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_fresh_duration_initialization_and_continuation_preserve_vjps(network):
    program = dn.compile(
        _model(network),
        dt=DT,
        duration=4.4 * DT,
        chunk_steps=4,
        initialization="fresh",
        inference=False,
        checkpoint_every=3,
        trainable="*g_param",
        callbacks=_callbacks(network),
        backend="eager",
    )
    initialization = program.initialization_inputs
    selected = initialization.populations["left"] if network else initialization
    voltage = selected.v_init.detach().clone().requires_grad_()
    selected = selected._replace(v_init=voltage)
    initialization = (
        initialization._replace(
            populations={**initialization.populations, "left": selected}
        )
        if network
        else selected
    )
    tensors = program.functional.extract()._replace(
        parameters=dict(program.parameters),
        constants=dict(program.constants),
        state=program.initial_state,
    )
    initial = dn.func.initialize(
        program.functional, tensors, initialization=initialization
    )
    reference = dn.func.run(
        program.functional.bind(initial),
        initial,
        duration=9.2 * DT,
        callbacks=program.callbacks,
        host_span_steps=4,
    )
    prefix = program(initialization=initialization)
    actual = program(prefix[1]["continuation"], duration=4.8 * DT)
    _close(actual[0], reference[0])
    key = "left.v" if network else "v"
    joined = torch.cat(
        (prefix[1]["callbacks"]["trace"][key], actual[1]["callbacks"]["trace"][key])
    )
    _close(joined, reference[1]["callbacks"]["trace"][key])
    targets = (voltage, _target(program.parameters, network))
    wanted = torch.autograd.grad(_voltage_loss(reference, network), targets)
    observed = torch.autograd.grad(joined.square().sum(), targets)
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert _pending(actual[0], network) == pytest.approx(0.2 * DT)


def test_population_duration_uses_the_effective_prepared_timestep():
    program = dn.compile(
        _population(), dt=DT, duration=4.6 * DT, chunk_steps=4, backend="eager"
    )
    constants = dict(program.constants)
    constants["dt"] = constants["dt"].new_tensor(2 * DT)
    bound = program.functional.bind(program.parameters, constants=constants)
    with torch.no_grad():
        expected = dn.func.run(bound, program.initial_state, duration=4.6 * DT)
        actual = program(constants=constants)
        _close(actual[0], expected[0])
        torch.testing.assert_close(
            actual[0]["clock"]["t"], program.initial_state["clock"]["t"] + 4 * DT
        )
        assert _pending(actual[0], False) == pytest.approx(0.6 * DT)
        resumed = program(actual[1]["continuation"], duration=0.7 * DT)
        expected = dn.func.run(bound, expected[1]["continuation"], duration=0.7 * DT)
        _close(resumed[0], expected[0])
        assert _pending(resumed[0], False) == pytest.approx(1.3 * DT)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_variable_while_horizons_preserve_shared_runtime_draws_and_rng(network):
    program, pulses = _stochastic_runner(network, 4)
    global_rng = torch.get_rng_state().clone()
    native_rng = [pulse.generator.get_state().clone() for pulse in pulses]
    tape = program.functional.sample_runtime_draws(program.initial_state, steps=9)
    with torch.no_grad():
        reference = _reference(program, steps=9, draws=tape)
        automatic = program(steps=9)
        _same(automatic, reference)
        prefix = program(steps=3, draws=_slice(tape, 0, 3))
        suffix = program(
            prefix[1]["continuation"],
            duration=6 * RANDOM_DT,
            draws=_slice(tape, 3, 9),
        )
    _close(suffix[0], reference[0])
    for name, values in reference[1]["callbacks"]["trace"].items():
        _close(
            torch.cat(
                (
                    prefix[1]["callbacks"]["trace"][name],
                    suffix[1]["callbacks"]["trace"][name],
                )
            ),
            values,
        )
    report = program.execution_report()
    with pytest.raises((ValueError, dn.func.FunctionalizationError)):
        program(steps=5, draws=tape)
    assert program.execution_report() is report
    _close(torch.get_rng_state(), global_rng)
    for pulse, expected in zip(pulses, native_rng, strict=True):
        _close(pulse.generator.get_state(), expected)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_runtime_extra_uses_selected_horizon_and_preserves_gradients(network):
    from tests.func.test_runner_chunking import DT as drive_dt
    from tests.func.test_runner_chunking import _model as drive_model
    from tests.func.test_runtime_extra import _field

    model = drive_model(network)
    population = model.cell if network else model
    program = dn.compile(
        model,
        dt=drive_dt,
        duration=5.4 * drive_dt,
        chunk_steps=3,
        inference=False,
        checkpoint_every=5,
        extra=dn.VarInput(),
        callbacks={
            "trace": dn.func.Recorder(("cell.v", "t") if network else ("v", "t"))
        },
        backend="eager",
    )
    field = _field(population).requires_grad_()
    waveform = dn.sin(amp=0.2, freq=0.9, phase=0.3)
    extra = {"cell": (field, waveform)} if network else (field, waveform)
    targets = (field, waveform.amp)
    with torch.enable_grad():
        reference = _reference(program, duration=9.2 * drive_dt, extra=extra)
        actual = program(duration=9.2 * drive_dt, extra=extra)
        _same(actual, reference)
        key = "cell.v" if network else "v"
        wanted = torch.autograd.grad(
            reference[1]["callbacks"]["trace"][key].square().sum(), targets
        )
        observed = torch.autograd.grad(
            actual[1]["callbacks"]["trace"][key].square().sum(), targets
        )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert actual[1]["callbacks"]["trace"]["t"].shape == (10,)


@pytest.mark.mps
def test_mps_variable_duration_while_preserves_fractional_continuation():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        program = dn.compile(
            _population(),
            dt=DT,
            duration=4.6 * DT,
            chunk_steps=4,
            execution="while",
            backend="aot_eager",
            callbacks={"trace": dn.func.Recorder(("v", "t"), dt=2.5 * DT)},
        )
        with torch.no_grad():
            first = program()
            second = program(first[1]["continuation"], duration=3.6 * DT)
            reference = _reference(program, duration=8.2 * DT)
        _close(second[0], reference[0])
        for name, expected in reference[1]["callbacks"]["trace"].items():
            observed = torch.cat(
                (
                    first[1]["callbacks"]["trace"][name],
                    second[1]["callbacks"]["trace"][name],
                )
            )
            _close(observed, expected)
            assert observed.device.type == "mps"
