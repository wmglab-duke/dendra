"""Runner horizons, compiled widths and checkpoint spans have distinct ownership."""

from __future__ import annotations

import inspect
import io
import os
import sys
from contextlib import contextmanager

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._network import FunctionalNetwork
from dendra.func._population import FunctionalPopulation
from dendra.func._recorder_while import _RecorderWhileExecutor
from tests.func.test_runtime_extra import DT, _field
from tests.func.test_runtime_extra import _model as _passive
from tests.func.test_stimulation_rng_sharing import DT as RANDOM_DT
from tests.func.test_stimulation_rng_sharing import (
    _clone,
    _close,
    _network,
    _population,
    _slice,
)

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


def _model(network, *, registered=False):
    population = _passive(batch=())
    if registered:
        population[:, 1].inject(
            dn.mono_rect(amp=0.1 * dn.units.nA, delay=0.02, pw=0.035)
        )
    population.initialize()
    if not network:
        return population
    model = dn.Network({"cell": population})
    model.train()
    model.initialize(DT)
    return model


def _callbacks(network):
    return {"trace": dn.func.Recorder(("cell.v", "t") if network else ("v", "t"))}


def _inputs(network, packet):
    return {"cell": packet} if network else packet


def _trace(result):
    return dict(result[1]["callbacks"]["trace"])


def _loss(result):
    return sum(
        value.square().mean() for name, value in _trace(result).items() if name != "t"
    )


def _reference(runner, *, inputs=None, draws=None, steps=None):
    functional = runner.functional
    return functional.prepare_and_rollout(
        runner.parameters,
        runner.constants,
        runner.initial_state,
        inputs=inputs,
        steps=runner._steps if steps is None else steps,
        draws=draws,
        callbacks=runner.callbacks,
    )


def _kernel_cache(runner, network):
    if network:
        return {size: id(kernel) for size, kernel in runner.kernel._tails.items()}
    return {
        key: id(kernel)
        for key, kernel in runner.kernel._callback_chunks[runner.callbacks].items()
    }


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_default_widths_and_explicit_caps_preserve_each_runner_horizon(network):
    model = _model(network)
    for runtime, steps, width, expected in (
        (False, 73, None, 73),
        (True, 73, None, 64),
        (False, 5, 20, 5),
        (True, 5, 20, 5),
        (False, 11, 3, 3),
    ):
        program = dn.compile(
            model,
            dt=DT,
            steps=steps,
            chunk_steps=width,
            extra=dn.VarInput() if runtime else None,
            backend="eager",
        )
        assert program._steps == steps
        assert program.kernel.steps == expected


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("width", (True, 0, 2.5))
def test_invalid_ordinary_width_fails_before_lowering(monkeypatch, network, width):
    import dendra.func._convenience as population_module
    import dendra.func._network_compile as network_module

    def unexpected(*args, **kwargs):
        pytest.fail("invalid chunk_steps reached functional lowering")

    monkeypatch.setattr(population_module, "make_functional", unexpected)
    monkeypatch.setattr(network_module, "_lower_network", unexpected)
    factory = dn.func.CompiledNetworkRunner if network else dn.func.CompiledRunner
    with pytest.raises(ValueError, match="chunk_steps.*positive integer"):
        factory(object(), dt=DT, steps=11, chunk_steps=width, backend="eager")


@pytest.mark.parametrize(
    "network,width,checkpoint,drive,backend",
    (
        (False, 3, 7, "registered", "aot_eager"),
        (False, 5, 2, "bound_extra", "eager"),
        (True, 3, 7, "bound_extra", "eager"),
        (True, 5, 2, "registered", "aot_eager"),
        (False, 3, 7, "runtime_extra", "eager"),
        (True, 3, 7, "runtime_extra", "eager"),
    ),
    ids=(
        "population-H-over-C",
        "population-H-under-C",
        "network-H-over-C",
        "network-H-under-C",
        "population-runtime-minimum",
        "network-runtime-minimum",
    ),
)
def test_checkpoint_spans_keep_their_own_width_and_preserve_all_gradients(
    monkeypatch,
    network,
    width,
    checkpoint,
    drive,
    backend,
):
    import dendra.func._network_rollout as network_loops
    import dendra.func._runners as population_loops

    steps = 11
    model = _model(network, registered=drive == "registered")
    population = model.cell if network else model
    if drive == "bound_extra":
        extra = (_field(population), dn.sin(amp=0.2, freq=0.9, phase=0.3))
        if network:
            extra = {"cell": extra}
    else:
        extra = dn.VarInput() if drive == "runtime_extra" else None
    runner = dn.compile(
        model,
        dt=DT,
        steps=steps,
        chunk_steps=width,
        checkpoint_every=checkpoint,
        extra=extra,
        inference=False,
        backend=backend,
        callbacks=_callbacks(network),
        trainable=("*g_param", "*amp") if drive != "runtime_extra" else "*g_param",
    )
    tape = torch.linspace(
        0.1,
        0.3,
        steps * population.v.numel(),
        dtype=population.dtype(),
        device=population.device(),
    ).reshape(steps, *population.shape)
    tape = (tape if drive == "registered" else tape * dn.units.nA).requires_grad_()
    inputs = _inputs(
        network,
        dn.func.Inputs(
            **{
                "ve" if drive == "registered" else "intra": tape,
            }
        ),
    )
    targets = tuple(
        {id(value): value for value in runner.trainable_parameters.values()}.values()
    ) + (tape,)
    expected = _reference(runner, inputs=inputs)
    expected_gradients = torch.autograd.grad(_loss(expected), targets)
    before = _clone((runner.initial_state, inputs))
    module = network_loops if network else population_loops
    original = module._checkpoint_tensor_tree
    spans = []

    def checkpoint_tree(function, operands):
        parameters = inspect.signature(function).parameters
        name = "span_steps" if network else "chunk_steps"
        spans.append(parameters[name].default)
        return original(function, operands)

    monkeypatch.setattr(module, "_checkpoint_tensor_tree", checkpoint_tree)
    effective = min(checkpoint, width) if drive == "runtime_extra" else checkpoint
    expected_spans = [
        min(effective, steps - start) for start in range(0, steps, effective)
    ]
    cached = None
    for iteration in range(2):
        actual = runner(
            inputs=inputs, **({"extra": None} if drive == "runtime_extra" else {})
        )
        assert spans == expected_spans * (iteration + 1)
        _close(actual[0], expected[0])
        _close(_trace(actual), _trace(expected))
        assert all(value.shape[0] == steps + 1 for value in _trace(actual).values())
        gradients = torch.autograd.grad(_loss(actual), targets)
        _close(gradients, expected_gradients)
        assert all(
            torch.isfinite(value).all() and torch.count_nonzero(value)
            for value in gradients
        )
        current = _kernel_cache(runner, network)
        assert current
        if cached is not None:
            assert current == cached
        cached = current
    assert runner.kernel.steps == width
    _close((runner.initial_state, inputs), before)


def _stochastic_runner(network, width):
    if network:
        model, pulses = _network()
    else:
        model, _, pulse = _population()
        pulses = [pulse]
    model.eval()
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    runner = dn.compile(
        model,
        dt=RANDOM_DT,
        steps=9,
        chunk_steps=width,
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(paths)},
        backend="aot_eager",
    )
    return runner, pulses


@contextmanager
def _observe_while_boundaries():
    sampling_codes = {
        method.__code__
        for method in (
            FunctionalPopulation.sample_runtime_draws,
            FunctionalPopulation._sample_runtime_draws_source_validated,
            FunctionalPopulation.sample_noise_draw_tape,
            FunctionalPopulation._sample_noise_draw_tape_source_validated,
            FunctionalPopulation.sample_stimulation_draw_tape,
            FunctionalPopulation._sample_stimulation_draw_tape_source_validated,
            FunctionalNetwork.sample_runtime_draws,
        )
    }
    samples, loops = [], []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call":
            if frame.f_code in sampling_codes:
                samples.append(frame.f_locals.get("steps"))
            elif frame.f_code is _RecorderWhileExecutor.kernel_for.__code__:
                loops.append(frame.f_locals["self"]._steps)
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        yield samples, loops
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_bounded_while_shared_rng_draws_recording_and_portable_continuation(network):
    runner, pulses = _stochastic_runner(network, 4)
    source = _clone(runner.functional.extract())
    initial = _clone(runner.initial_state)
    source_rng = pulses[0].generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    tape = runner.functional.sample_runtime_draws(initial, steps=18)
    with _observe_while_boundaries() as (samples, loops):
        automatic = runner()
    assert samples and all(isinstance(size, int) and 0 < size <= 4 for size in samples)
    if not network:
        assert loops and loops[0] == 4 and max(loops) == 4
    with _observe_while_boundaries() as (samples, loops):
        explicit = runner(draws=_slice(tape, 0, 9))
    assert samples == []
    if not network:
        assert loops and max(loops) == 4
    _close(explicit[0], automatic[0])
    _close(_trace(explicit), _trace(automatic))
    saved = io.BytesIO()
    torch.save(runner.export_continuation(explicit[1]["continuation"]), saved)
    saved.seek(0)
    receiver, _ = _stochastic_runner(network, 5)
    continuation = receiver.restore_continuation(torch.load(saved, weights_only=True))
    resumed = receiver(continuation)
    with torch.no_grad():
        expected = _reference(runner, draws=tape, steps=18)
    _close(resumed[0], expected[0])
    for name, prefix in _trace(explicit).items():
        suffix = _trace(resumed)[name]
        assert prefix.shape[0] == 10 and suffix.shape[0] == 9
        _close(torch.cat((prefix, suffix)), _trace(expected)[name])
    _close(runner.initial_state, initial)
    _close(runner.functional.extract(), source)
    _close(pulses[0].generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


def test_population_while_keeps_minimum_chunk_and_callback_admission():
    model = _model(False)
    for width in (1, 2, 3):
        with pytest.raises(ValueError, match="chunk_steps.*four"):
            dn.compile(
                model,
                dt=DT,
                steps=9,
                chunk_steps=width,
                execution="while",
                callbacks=_callbacks(False),
                backend="eager",
            )
    with pytest.raises(ValueError, match="nonempty callbacks"):
        dn.compile(
            model, dt=DT, steps=9, chunk_steps=4, execution="while", backend="eager"
        )


def test_network_short_callback_free_while_chunks_keep_the_full_horizon():
    runner = dn.compile(
        _model(True),
        dt=DT,
        steps=5,
        chunk_steps=2,
        execution="while",
        inference=True,
        backend="aot_eager",
    )
    actual, auxiliary = runner()
    with torch.no_grad():
        expected, _ = runner.functional.prepare_and_rollout(
            runner.parameters, runner.constants, runner.initial_state, steps=5
        )
    _close(actual, expected)
    assert "callbacks" not in auxiliary
    assert runner.kernel.steps == 2
    assert actual["clock"]["step"].item() == 5


@pytest.mark.parametrize("packet", ("ve", "intra", "legacy_noise"))
def test_bounded_population_while_rejects_overlong_tapes_before_capture(packet):
    from tests.func.test_runtime_noise import DT as NOISE_DT
    from tests.func.test_runtime_noise import _population as noise_population

    graphs = []

    def backend(graph, operands):
        graphs.append(graph)
        return graph.forward

    runner = dn.compile(
        noise_population(),
        dt=NOISE_DT,
        steps=9,
        chunk_steps=4,
        execution="while",
        callbacks=_callbacks(False),
        backend=backend,
    )
    if packet == "legacy_noise":
        inputs = dn.func.RolloutInput(
            noise=runner.functional.sample_noise_draw_tape(runner.initial_state, 10)
        )
    else:
        inputs = dn.func.Inputs(
            **{
                packet: torch.zeros(
                    (10, *runner.functional.shape), dtype=runner.functional.dtype
                )
            }
        )
    with pytest.raises(ValueError):
        runner(inputs=inputs)
    assert graphs == []


def test_bounded_population_while_requalifies_a_replaced_sampler_before_capture(
    monkeypatch,
):
    graphs = []

    def backend(graph, operands):
        graphs.append(graph)
        return graph.forward

    runner = dn.compile(
        _model(False),
        dt=DT,
        steps=9,
        chunk_steps=4,
        execution="while",
        callbacks=_callbacks(False),
        backend=backend,
    )

    def mutate(state, steps):
        assert steps == 4
        runner.functional._population._flag_rebuild = True
        return None

    monkeypatch.setattr(runner.functional, "sample_noise_draw_tape", mutate)
    with pytest.raises(dn.func.FunctionalizationError, match="structur|rebuild"):
        runner()
    assert graphs == []


@pytest.mark.mps
def test_mps_ordinary_chunking_preserves_checkpointed_drive_gradients():
    from tests.func.test_runtime_extra_backends import _environment as environment
    from tests.func.test_runtime_extra_backends import _model as model_factory

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with environment("mps"):
        model = model_factory("mps")
        runner = dn.compile(
            model,
            dt=DT,
            steps=7,
            chunk_steps=3,
            checkpoint_every=5,
            inference=False,
            backend="aot_eager",
            trainable="*g_param",
            callbacks=_callbacks(False),
        )
        current = model.v.new_full(
            (7, *model.shape), 0.1 * dn.units.nA
        ).requires_grad_()
        inputs = dn.func.Inputs(intra=current)
        targets = (runner.get_parameter("g_param"), current)
        expected = _reference(runner, inputs=inputs)
        reference = torch.autograd.grad(_loss(expected), targets)
        actual = runner(inputs=inputs)
        observed = torch.autograd.grad(_loss(actual), targets)
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-5, atol=2e-5)
        torch.testing.assert_close(
            _trace(actual), _trace(expected), rtol=3e-5, atol=2e-5
        )
        for value, wanted in zip(observed, reference, strict=True):
            assert torch.isfinite(value).all() and torch.count_nonzero(value)
            scale = wanted.abs().amax()
            torch.testing.assert_close(
                value / scale, wanted / scale, rtol=4e-3, atol=4e-4
            )
        assert runner.kernel.steps == 3
