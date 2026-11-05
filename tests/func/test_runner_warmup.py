"""Warm-up retains real runner caches without publishing a simulation run."""

from __future__ import annotations

import gc
import sys
import weakref
from collections.abc import Mapping
from copy import deepcopy
from threading import Event, current_thread
from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.data import H5Reader
from dendra.func import _warm
from dendra.models.mod import pas
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_bitpacked_inference import _stimulus
from tests.func.test_runtime_noise import DT as NOISE_DT
from tests.func.test_runtime_noise import _population as _noisy_population
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu
DT = 0.025
STEPS = 4


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


def _population():
    model = dn.SingleCompartment(
        N=1,
        C=2,
        v_init=-62.0,
        dtype=torch.float64,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(pas, g=1e-4, e=-65.0)
    model.initialize()
    model.train()
    return model


def _model(network):
    return (_stimulus(noise=0.6), NETWORK_DT) if network else (_population(), DT)


def _fields(network):
    return ("cell.v", "t") if network else ("v", "t")


def _plain(tree):
    if isinstance(tree, Mapping):
        return {name: _plain(value) for name, value in tree.items()}
    if isinstance(tree, tuple):
        values = tuple(_plain(value) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    if isinstance(tree, list):
        return [_plain(value) for value in tree]
    return tree


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), _plain(tree)
    )


def _freeze(tree):
    leaves, spec = pytree.tree_flatten(_plain(tree))
    metadata = [
        (
            id(value),
            None if value.is_inference() else value._version,
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            value.requires_grad,
            value.is_leaf,
            type(value),
            value.is_inference(),
        )
        if torch.is_tensor(value)
        else value
        for value in leaves
    ]
    copies = [
        value.detach().clone() if torch.is_tensor(value) else value for value in leaves
    ]
    return spec, metadata, copies


def _unchanged(tree, frozen):
    observed = _freeze(tree)
    assert observed[:2] == frozen[:2]
    torch.testing.assert_close(observed[2], frozen[2], rtol=0, atol=0)


def _activity():
    from torch._dynamo import utils

    return deepcopy(utils.counters), len(utils.get_compilation_metrics())


def _sources(program):
    return program.parameters, program.constants, program.initial_state


def _eager(program, state=None, *, steps=STEPS, inputs=None):
    parameters, constants = program.parameters, program.constants
    if isinstance(state, dn.func.Continuation):
        parameters, constants = state.parameters, state.constants
    with torch.no_grad():
        return dn.func.run(
            program.functional.bind(parameters, constants=constants),
            program.initial_state if state is None else state,
            steps=steps,
            inputs=inputs,
            callbacks=program.callbacks,
        )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_warm_returns_runner_reuses_restart_and_preserves_sources(network, execution):
    model, dt = _model(network)
    program = dn.compile(
        model,
        dt=dt,
        steps=STEPS,
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.Recorder(_fields(network))},
    )
    frozen = _freeze(_sources(program))
    native = _clone(program.functional.extract().state)
    rng = torch.get_rng_state().clone()
    assert program.execution_report() is None
    assert program.warm() is program
    admitted = _activity()
    actual, auxiliary = program()
    assert _activity() == admitted
    expected, reference = _eager(program)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"],
        reference["callbacks"]["trace"],
        rtol=2e-10,
        atol=2e-11,
    )
    assert auxiliary["callbacks"]["trace"]["t"].shape[0] == STEPS + 1
    _unchanged(_sources(program), frozen)
    torch.testing.assert_close(
        program.functional.extract().state, native, rtol=0, atol=0
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_warm_discards_hdf5_publication_but_retains_storage_spans(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects imported fixture
    network,
):
    model, dt = _model(network)
    path = tmp_path / "warm.h5"
    fields = _fields(network)
    disk = dn.Recorder(fields, dt=2 * dt).set_hdf5(str(path), cache_every=4)
    memory = dn.Recorder(fields, dt=2 * dt)
    program = dn.compile(
        model,
        dt=dt,
        steps=8,
        chunk_steps=8,
        execution="while",
        backend="aot_eager",
        callbacks={"disk": disk, "memory": memory},
    )
    callbacks, storage = program.callbacks, program.callbacks._storage
    before = (tuple(local_hdf5_writer.queue.items), disk.run_number, disk.save_count)
    old_file = path.read_bytes() if path.exists() else None
    try:
        # A failed dry run must also release the warm-only storage context.
        shape = model.populations["cell"].v.shape if network else model.v.shape
        invalid = dn.func.Inputs(intra=torch.zeros(3, *shape))
        if network:
            invalid = {"cell": invalid}
        with pytest.raises(ValueError):
            program.warm(inputs=invalid)
        assert program.warm() is program
        assert program.callbacks is callbacks and callbacks._storage is storage
        assert (
            tuple(local_hdf5_writer.queue.items),
            disk.run_number,
            disk.save_count,
        ) == before
        assert (path.read_bytes() if path.exists() else None) == old_file
        assert disk._hdf5_session is None
        assert disk.i == 0 and all(not values for values in disk.rec.values())
        admitted = _activity()
        _, auxiliary = program()
        assert _activity() == admitted
        expected = _clone(auxiliary["callbacks"]["memory"])
        assert expected["t"].shape == (5,)
        assert disk.run_number == before[1] + 1
        assert any(isinstance(item, tuple) for item in local_hdf5_writer.queue.items)
        assert program.execution_report().steps == 4
    finally:
        disk.close()
    with H5Reader(path) as reader:
        for field in fields:
            persisted = torch.from_numpy(reader[field][0].data.compute()).clone()
            torch.testing.assert_close(persisted, expected[field], rtol=0, atol=0)


def test_training_warm_keeps_gradients_and_releases_discarded_graph(monkeypatch):
    program = dn.compile(
        _population(),
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable="*g_param",
        backend="aot_eager",
    )
    parameter = next(
        value for value in program.parameters.values() if value.requires_grad
    )
    parameter.grad = torch.full_like(parameter, 0.37)
    old_grad, old_grad_values = parameter.grad, parameter.grad.clone()
    frozen = _freeze(_sources(program))
    preparations, outputs = [], []
    prepare = program._prepare_source_validated
    call = type(program).__call__

    def observe_prepare(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        preparations.append(weakref.ref(prepared))
        return prepared

    def observe_call(self, *args, **kwargs):
        result = call(self, *args, **kwargs)
        outputs.append(weakref.ref(result[0]["integrator"]["v"]))
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(program, "_prepare_source_validated", observe_prepare)
        scoped.setattr(type(program), "__call__", observe_call)
        assert program.warm() is program
    gc.collect()
    assert preparations and all(reference() is None for reference in preparations)
    assert outputs and all(reference() is None for reference in outputs)
    assert parameter.grad is old_grad
    torch.testing.assert_close(parameter.grad, old_grad_values, rtol=0, atol=0)
    _unchanged(_sources(program), frozen)
    actual, _ = program()
    (gradient,) = torch.autograd.grad(
        actual["integrator"]["v"].square().sum(), parameter
    )
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    assert parameter.grad is old_grad
    torch.testing.assert_close(parameter.grad, old_grad_values, rtol=0, atol=0)


def test_warm_replays_explicit_draws_without_consuming_source_rng():
    program = dn.compile(
        _noisy_population(), dt=NOISE_DT, steps=STEPS, backend="aot_eager"
    )
    draws = program.functional.sample_runtime_draws(program.initial_state, steps=STEPS)
    frozen = _freeze((_sources(program), draws))
    native = _clone(program.functional.extract().state)
    rng = torch.get_rng_state().clone()
    assert program.warm(draws=draws) is program
    admitted = _activity()
    actual, _ = program(draws=draws)
    assert _activity() == admitted
    automatic, _ = program()
    torch.testing.assert_close(actual, automatic, rtol=2e-10, atol=2e-11)
    _unchanged((_sources(program), draws), frozen)
    torch.testing.assert_close(
        program.functional.extract().state, native, rtol=0, atol=0
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_warm_accepts_runtime_extra_examples_and_retains_validation():
    model = _population()
    program = dn.compile(
        model, dt=DT, steps=STEPS, extra=dn.VarInput(), backend="aot_eager"
    )
    with pytest.raises(TypeError, match="extra"):
        program.warm()
    field = torch.linspace(-1.0, 1.0, model.v.numel()).reshape(model.shape)
    waveform = dn.sin(amp=0.2, freq=0.8, phase=0.1)
    extra = [(field, waveform)]
    frozen = _freeze((field, tuple(waveform.parameters())))
    assert program.warm(extra=extra) is program
    admitted = _activity()
    first, _ = program(extra=extra)
    assert _activity() == admitted
    second, _ = program(extra=extra)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    _unchanged((field, tuple(waveform.parameters())), frozen)


def test_warm_explicit_continuation_preserves_owners_and_callback_phase():
    program = dn.compile(
        _population(),
        dt=DT,
        steps=STEPS,
        backend="aot_eager",
        callbacks={"trace": dn.Recorder(("v", "t"), dt=2 * DT)},
    )
    _, auxiliary = program()
    packet = auxiliary["continuation"]
    values = (
        packet.parameters,
        packet.constants,
        packet.state,
        packet.callback_state.carries,
    )
    frozen = _freeze(values)
    assert program.warm(packet, steps=3) is program
    admitted = _activity()
    actual, following = program(packet, steps=3)
    assert _activity() == admitted
    expected, reference = _eager(program, packet, steps=3)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        following["callbacks"]["trace"],
        reference["callbacks"]["trace"],
        rtol=2e-10,
        atol=2e-11,
    )
    assert torch.all(following["callbacks"]["trace"]["t"] > packet.state["clock"]["t"])
    _unchanged(values, frozen)


def test_warm_fresh_initialization_preserves_restart_defaults():
    model = _population()
    model.step(dt=DT)
    program = dn.compile(
        model, dt=DT, steps=STEPS, initialization="fresh", backend="aot_eager"
    )
    frozen = _freeze((_sources(program), program.initialization_inputs))
    native = _clone(program.functional.extract().state)
    assert program.warm() is program
    initialized = dn.func.initialize(program)
    expected, _ = _eager(program, initialized)
    actual, _ = program()
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    _unchanged((_sources(program), program.initialization_inputs), frozen)
    torch.testing.assert_close(
        program.functional.extract().state, native, rtol=0, atol=0
    )


def test_shared_warm_helper_forwards_exact_examples_and_is_quiet(capsys):
    state, inputs, parameters, extra = object(), object(), object(), object()
    calls = []

    def program(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    _cpu(program)
    assert (
        _warm._warm_runner(
            program, state, inputs=inputs, parameters=parameters, extra=extra
        )
        is program
    )
    assert calls == [
        ((state,), {"inputs": inputs, "parameters": parameters, "extra": extra})
    ]
    assert capsys.readouterr() == ("", "")


def test_verbose_warm_reports_actual_forward_completion(capsys):
    def program(_state):
        return None

    _cpu(program)
    assert _warm._warm_runner(program, verbose=True) is program
    output = capsys.readouterr()
    assert output.out == ""
    assert "Warming forward" in output.err and "Forward ready" in output.err
    assert "failed" not in output.err and "%" not in output.err


@pytest.mark.parametrize("fail", (False, True), ids=("success", "failure"))
def test_verbose_heartbeat_stops_before_completion_and_preserves_errors(
    monkeypatch, fail
):
    observed = Event()
    reporters, messages = [], []
    failure = RuntimeError("warm example failed")

    class Progress:
        def write(self, text):
            messages.append(text)
            if "elapsed" in text:
                reporters.append(current_thread())
                observed.set()
            return len(text)

        def flush(self):
            pass

    def program(_state):
        assert observed.wait(2.0)
        if fail:
            raise failure

    _cpu(program)
    monkeypatch.setattr(_warm, "_WARM_PROGRESS_INTERVAL", 0.001)
    monkeypatch.setattr(sys, "stderr", Progress())
    if fail:
        with pytest.raises(RuntimeError) as caught:
            _warm._warm_runner(program, verbose=True)
        assert caught.value is failure
    else:
        assert _warm._warm_runner(program, verbose=True) is program
    assert reporters and all(not reporter.is_alive() for reporter in reporters)
    output = "".join(messages)
    assert "elapsed" in output
    assert ("failed" in output) is fail
    assert ("Forward ready" in output) is not fail


@pytest.mark.parametrize("invalid", (1, "yes"))
def test_invalid_verbose_is_rejected_before_executing_examples(invalid):
    def unexpected(_state):
        pytest.fail("invalid verbosity executed the warm-up call")

    _cpu(unexpected)
    with pytest.raises(TypeError, match="verbose must be a bool"):
        _warm._warm_runner(unexpected, verbose=invalid)


def _cpu(function):
    function.functional = SimpleNamespace(device=torch.device("cpu"))
    return function


@pytest.mark.parametrize("device", ("cuda:1", "mps"))
def test_warm_waits_for_the_selected_device_before_reporting_ready(monkeypatch, device):
    events = []

    def program(_state):
        events.append("run")

    program.functional = SimpleNamespace(device=torch.device(device))

    def synchronize():
        events.append("synchronize")

    def stream(selected):
        assert selected == torch.device(device)
        events.append("selected stream")
        return SimpleNamespace(synchronize=synchronize)

    def unexpected(*_args, **_kwargs):
        pytest.fail("warm-up synchronized the wrong backend")

    def progress(message):
        if "ready" in message:
            events.append("ready")

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        stream if device.startswith("cuda") else unexpected,
    )
    monkeypatch.setattr(
        torch.mps, "synchronize", synchronize if device == "mps" else unexpected
    )
    monkeypatch.setattr(_warm, "_progress", progress)
    assert _warm._warm_runner(program, verbose=True) is program
    expected = (
        ["run", "selected stream", "synchronize", "ready"]
        if device.startswith("cuda")
        else ["run", "synchronize", "ready"]
    )
    assert events == expected


def test_warm_synchronization_failure_is_reported_and_preserved(monkeypatch, capsys):
    failure = RuntimeError("asynchronous warm-up execution failed")

    def program(_state):
        return None

    program.functional = SimpleNamespace(device=torch.device("cuda:1"))

    def synchronize():
        raise failure

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(synchronize=synchronize),
    )
    with pytest.raises(RuntimeError) as caught:
        _warm._warm_runner(program, verbose=True)
    assert caught.value is failure
    output = capsys.readouterr()
    assert "failed" in output.err and "ready" not in output.err
