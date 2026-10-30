"""Compiled programs share one host execution contract with ``dn.func.run``."""

from __future__ import annotations

import os
import typing

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _drive,
    _failure,
    _plain,
    _runner,
    _snapshot,
)
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_runner_chunking import _model as _extra_model
from tests.func.test_runner_initialization import _inputs as _initialization_inputs
from tests.func.test_runner_initialization import _program
from tests.func.test_runtime_extra import DT as EXTRA_DT
from tests.func.test_runtime_extra import _field
from tests.func.test_stimulation_rng_sharing import DT as RANDOM_DT
from tests.func.test_stimulation_rng_sharing import _network as _random_network
from tests.func.test_stimulation_rng_sharing import _population as _random_population
from tests.func.test_stimulation_rng_sharing import _slice

pytestmark = pytest.mark.cpu


def test_run_annotations_resolve_for_existing_bindings_and_compiled_programs():
    hints = typing.get_type_hints(dn.func.run)
    assert set(typing.get_args(hints["functional"])) == {
        dn.func.FunctionalPopulation,
        dn.func.BoundPopulation,
        dn.func.BoundNetwork,
        dn.func.CompiledRunner,
        dn.func.CompiledNetworkRunner,
    }


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


def _same(actual, expected):
    _close(actual[0], expected[0])
    _close(actual[1]["callbacks"], expected[1]["callbacks"])
    packet = actual[1]["continuation"]
    assert type(packet) is dn.func.Continuation
    _close(packet.state, actual[0])
    assert packet._lifecycle == "active"
    assert packet.callback_state.plan_token is actual[1]["callbacks"].state.plan_token


def _trace(result, network):
    return result[1]["callbacks"]["trace"]["left.v" if network else "v"]


def _left_state(state, network):
    return state["populations"]["left"] if network else state


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("form", ("positional", "state_keyword", "step_keyword"))
def test_program_state_forms_preserve_returned_sources_and_callback_carry(
    network, form
):
    program = _runner(network)
    initial = dn.func.initialize(program)
    before = _snapshot((program.parameters, program.constants, program.initial_state))
    if form == "positional":
        actual = dn.func.run(program, initial)
    elif form == "state_keyword":
        actual = dn.func.run(program, state=initial)
    else:
        actual = dn.func.run(program, step=initial)
    expected = program(initial)
    _same(actual, expected)
    assert _trace(actual, network).shape[0] == 4
    packet = actual[1]["continuation"]
    assert _target(packet.parameters, network) is _target(initial.parameters, network)
    if network:
        assert (
            _target(packet.parameters, True)
            is packet.parameters["populations"]["right"][G]
        )
    resumed = dn.func.run(program, packet, steps=1)
    _same(resumed, program(packet, steps=1))
    assert _trace(resumed, network).shape[0] == 1
    _close((program.parameters, program.constants, program.initial_state), before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("policy", ("snapshot", "fresh"))
def test_program_defaults_restart_with_the_configured_initialization_policy(
    network, policy
):
    program = _program(
        network,
        scaled=True,
        initialization=policy,
        inference=False,
        callbacks=_callbacks(network),
    )
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    first = dn.func.run(program)
    _same(first, program())
    selected = (
        program.initialization_inputs.populations["left"]
        if network
        else program.initialization_inputs
    )
    with torch.no_grad():
        selected.v_init.add_(5.0)
    zero = dn.func.run(program, steps=0)
    _same(zero, program(steps=0))
    original = _left_state(program.initial_state, network)["integrator"]["v"]
    _close(
        _left_state(zero[0], network)["integrator"]["v"],
        original + (5.0 if policy == "fresh" else 0.0),
    )
    assert _trace(zero, network).shape[0] == 1
    resumed = dn.func.run(program, first[1]["continuation"], steps=1)
    _same(resumed, program(first[1]["continuation"], steps=1))
    assert _target(resumed[1]["continuation"].parameters, network) is _target(
        first[1]["continuation"].parameters, network
    )
    _close((program.parameters, program.constants, program.initial_state), defaults)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_duration_defaults_keep_fractional_time_and_zero_step_callback_phase(
    network,
):
    program = dn.compile(
        _network() if network else _population(),
        dt=DT,
        duration=0.6 * DT,
        chunk_steps=4,
        callbacks=_callbacks(network),
        inference=True,
        backend="eager",
        disable=True,
    )
    kernel, default_duration = program.kernel, program._duration
    first = dn.func.run(program)
    _same(first, program())
    assert _trace(first, network).shape[0] == 1
    remainder = "clock" if network else "control"
    assert first[0][remainder]["duration_remainder"].item() == pytest.approx(0.6 * DT)
    empty = dn.func.run(program, first[1]["continuation"], duration=0.0)
    _same(empty, program(first[1]["continuation"], duration=0.0))
    assert _trace(empty, network).shape[0] == 0
    second = dn.func.run(program, state=empty[1]["continuation"])
    _same(second, program(empty[1]["continuation"]))
    assert _trace(second, network).shape[0] == 1
    assert second[0][remainder]["duration_remainder"].item() == pytest.approx(0.2 * DT)
    advanced = dn.func.run(program, second[1]["continuation"], steps=2)
    _same(advanced, program(second[1]["continuation"], steps=2))
    assert advanced[0][remainder]["duration_remainder"].item() == pytest.approx(
        0.2 * DT
    )
    _same(dn.func.run(program), first)
    assert program.kernel is kernel and program._duration == default_duration


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_source_and_initialization_overrides_preserve_aliases_and_vjps(network):
    program = _program(
        network,
        scaled=True,
        initialization="fresh",
        inference=False,
        callbacks=_callbacks(network),
        checkpoint_every=1,
    )
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    parameters, constants = _plain(program.parameters), _plain(program.constants)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = upstream * 1.2
    _assign(parameters, network, replacement)
    selected_constants = constants["populations"]["left"] if network else constants
    timestep = selected_constants["dt"].clone()
    selected_constants["dt"] = timestep
    voltage = (
        _left_state(program.initial_state, network)["integrator"]["v"]
        .detach()
        .clone()
        .requires_grad_()
    )
    factor = upstream.new_tensor(1.7, requires_grad=True)
    initialization = _initialization_inputs(
        program, network, voltage=voltage, factor=factor
    )
    current, inputs = _drive(program, network, 2, requires_grad=True)
    options = dict(
        parameters=parameters,
        constants=constants,
        initialization=initialization,
        inputs=inputs,
    )
    expected = program(**options)
    actual = dn.func.run(program, **options)
    _same(actual, expected)
    targets = (upstream, factor, voltage, current)
    wanted = torch.autograd.grad(
        _voltage(expected[0], network).square().sum(), targets, retain_graph=True
    )
    observed = torch.autograd.grad(_voltage(actual[0], network).square().sum(), targets)
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    packet = actual[1]["continuation"]
    assert _target(packet.parameters, network) is _target(
        actual[1]["parameters"], network
    )
    assert _target(packet.parameters, network) is not replacement
    if network:
        assert (
            _target(packet.parameters, network)
            is packet.parameters["populations"]["right"][G]
        )
    _close((program.parameters, program.constants, program.initial_state), defaults)


def test_program_population_keeps_inference_cache_admission_and_refreshes_live_sources():
    program = _runner(False, callbacks=False, inference=True)
    first = dn.func.run(program)
    cached = program._prepared
    assert cached is not None
    _close(dn.func.run(program)[0], program()[0])
    assert program._prepared is cached
    dn.func.run(program, first[1]["continuation"])
    assert program._prepared is cached
    with torch.no_grad():
        _target(program.parameters, False).add_(0.0002)
    dn.func.run(program, first[1]["continuation"])
    assert program._prepared is cached
    changed = dn.func.run(program)
    assert program._prepared is not cached
    _close(changed[0], program()[0])
    assert not torch.equal(changed[0]["integrator"]["v"], first[0]["integrator"]["v"])


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_explicit_and_automatic_runtime_draws_keep_rng_continuation(network):
    if network:
        model, pulses = _random_network()
    else:
        model, _, pulse = _random_population()
        pulses = [pulse]
    program = dn.compile(
        model,
        dt=RANDOM_DT,
        steps=2,
        chunk_steps=2,
        inference=True,
        callbacks={
            "trace": dn.func.Recorder(
                ("left.v", "right.v", "t") if network else ("v", "t")
            )
        },
        backend="eager",
        disable=True,
    )
    initial = _snapshot(program.initial_state)
    source_rng, global_rng = (
        pulses[0].generator.get_state().clone(),
        torch.get_rng_state().clone(),
    )
    tape = program.functional.sample_runtime_draws(initial, steps=4)
    automatic = dn.func.run(program)
    explicit = dn.func.run(program, draws=_slice(tape, 0, 2))
    _same(automatic, program())
    _same(explicit, program(draws=_slice(tape, 0, 2)))
    _close(automatic[0], explicit[0])
    resumed = dn.func.run(
        program, explicit[1]["continuation"], draws=_slice(tape, 2, 4)
    )
    _same(resumed, program(explicit[1]["continuation"], draws=_slice(tape, 2, 4)))
    _close(resumed[0], program(steps=4)[0])
    _close(program.initial_state, initial)
    _close(pulses[0].generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_runtime_extra_distinguishes_omission_and_explicit_none(network):
    model = _extra_model(network)
    program = dn.compile(
        model,
        dt=EXTRA_DT,
        steps=3,
        chunk_steps=2,
        extra=dn.VarInput(),
        inference=False,
        backend="eager",
        disable=True,
    )
    assert _failure(lambda: dn.func.run(program)) == _failure(lambda: program())
    _close(dn.func.run(program, extra=None)[0], program(extra=None)[0])
    population = model.cell if network else model
    field = _field(population).detach().requires_grad_()
    waveform = dn.sin(amp=0.15, freq=0.7, phase=0.3)
    recipe = {"cell": (field, waveform)} if network else (field, waveform)
    expected = program(extra=recipe)
    actual = dn.func.run(program, extra=recipe)
    _close(actual[0], expected[0])
    targets = (field, waveform.amp)
    expected_voltage = (
        expected[0]["populations"]["cell"]["integrator"]["v"]
        if network
        else expected[0]["integrator"]["v"]
    )
    actual_voltage = (
        actual[0]["populations"]["cell"]["integrator"]["v"]
        if network
        else actual[0]["integrator"]["v"]
    )
    wanted = torch.autograd.grad(
        expected_voltage.square().sum(), targets, retain_graph=True
    )
    observed = torch.autograd.grad(actual_voltage.square().sum(), targets)
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert _failure(lambda: dn.func.run(program, extra=object())) == _failure(
        lambda: program(extra=object())
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_owned_configuration_cannot_be_overridden_by_binding_schedule_options(
    network,
):
    program = _runner(network)
    program()
    report = program.execution_report()
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    for options in (
        {"callbacks": {}},
        {"checkpoint_every": 1},
        {"host_span_steps": 1},
        {"dt": DT},
        {"tstop": DT},
        {"_runtime_extra_slot": True},
    ):
        name = next(iter(options))
        with pytest.raises((TypeError, ValueError), match=name):
            dn.func.run(program, **options)
        assert program.execution_report() is report
        _close((program.parameters, program.constants, program.initial_state), defaults)
    _same(
        dn.func.run(
            program,
            callbacks=None,
            checkpoint_every=None,
            host_span_steps=None,
            dt=None,
            tstop=None,
            _runtime_extra_slot=False,
        ),
        program(),
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_duplicate_state_rejects_instead_of_accepting_positional_inputs(
    network,
):
    program = _runner(network)
    for second in (None, program.initial_state, dn.func.Inputs()):
        with pytest.raises(TypeError, match="state"):
            dn.func.run(program, program.initial_state, second)
    with pytest.raises(TypeError, match="state"):
        dn.func.run(program, step=None, state=None)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_validation_and_publication_errors_keep_report_and_defaults(
    network, monkeypatch
):
    program = _runner(network)
    _, first = program()
    packet = first["continuation"]
    state_before = _snapshot(packet.state)
    report = program.execution_report()
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    for options in (
        {"steps": 1, "duration": DT},
        {"steps": True},
        {"parameters": {}},
        {"constants": {}},
        {"inputs": object()},
        {"draws": object()},
        {"initialization": object()},
    ):
        assert _failure(lambda: dn.func.run(program, **options)) == _failure(
            lambda: program(**options)
        )
        assert program.execution_report() is report
        _close((program.parameters, program.constants, program.initial_state), defaults)

    def reject(*args, **kwargs):
        raise ValueError("continuation publication rejected")

    monkeypatch.setattr(dn.func.Continuation, "__init__", reject)
    with pytest.raises(ValueError, match="continuation publication rejected"):
        dn.func.run(program, packet)
    assert program.execution_report() is report
    _close(packet.state, state_before)
    _close((program.parameters, program.constants, program.initial_state), defaults)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("context", ("compile", "grad", "forward_ad"))
def test_program_execution_keeps_host_boundary_checks_before_source_access(
    network, context, monkeypatch
):
    program = _runner(network)

    def reject(*args, **kwargs):
        raise AssertionError("host guard must precede source validation")

    monkeypatch.setattr(program.functional, "_validate_source", reject)
    with pytest.raises(
        dn.func.FunctionalizationError, match="outside torch.compile|host-side"
    ):
        if context == "compile":
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            dn.func.run(program)
        elif context == "grad":
            torch.func.grad(
                lambda value: _voltage(dn.func.run(program)[0], network).sum() + value
            )(torch.tensor(0.0))
        else:
            with torch.autograd.forward_ad.dual_level():
                dn.func.run(program)
    assert program.execution_report() is None


@pytest.mark.parametrize(
    "kind", ("raw_population", "prepared_population", "raw_network", "legacy")
)
def test_runner_only_arguments_do_not_expand_existing_binding_and_legacy_forms(kind):
    network = kind == "raw_network"
    program = _runner(network, callbacks=False, inference=True)
    functional = program.functional
    if kind == "prepared_population":
        bound = functional.bind(
            program.parameters,
            functional.prepare(program.parameters, program.constants),
        )
    else:
        bound = functional.bind(program.parameters, constants=program.constants)
    if kind == "legacy":
        args, options = (functional, bound, program.initial_state), {"tstop": DT}
    else:
        args, options = (bound, program.initial_state), {"steps": 1}
    actual = dn.func.run(*args, **options)
    _close(actual[0], program(steps=1)[0])
    for name in ("parameters", "constants", "initialization"):
        with pytest.raises(TypeError, match=name):
            dn.func.run(*args, **options, **{name: None})


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_program_execution_uses_owned_core_without_instance_call_lookup(
    network, monkeypatch
):
    program = _runner(network)
    expected = program()

    def reject(*args, **kwargs):
        raise AssertionError("instance __call__ attributes are not Python call hooks")

    monkeypatch.setattr(program, "__call__", reject)
    _same(program(), expected)
    _same(dn.func.run(program), expected)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_direct_runner_subclasses_keep_call_behavior_without_new_canonical_admission(
    network,
):
    base = dn.func.CompiledNetworkRunner if network else dn.func.CompiledRunner

    class Inherited(base):
        pass

    inherited = Inherited(
        _network() if network else _population(),
        dt=DT,
        steps=2,
        inference=True,
        trainable=(),
        callbacks=_callbacks(network),
        backend="eager",
        disable=True,
    )
    _same(inherited(), _runner(network, inference=True, steps=2)())
    with pytest.raises(TypeError):
        dn.func.run(inherited)

    sentinel = object()

    class Overridden(base):
        def __call__(self, *args, **kwargs):
            return sentinel, args, kwargs

    overridden = object.__new__(Overridden)
    result = overridden("state", steps=7)
    assert result == (sentinel, ("state",), {"steps": 7})
    with pytest.raises(TypeError):
        dn.func.run(overridden)


def test_program_execution_does_not_duck_type_arbitrary_callables():
    class Callable:
        def __call__(self, *args, **kwargs):
            raise AssertionError("a generic callable is not a compiled runner")

    with pytest.raises(TypeError):
        dn.func.run(Callable())


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_captured_program_execution_resumes_initialized_and_portable_packets(
    network, execution, tmp_path
):
    program = _runner(
        network,
        steps=4,
        chunk_steps=4,
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        disable=False,
    )
    initial = dn.func.initialize(program)
    restored = dn.func.restore_continuation(
        program,
        _disk(dn.func.export_continuation(program, initial), tmp_path / "initial.pt"),
    )
    expected = program(restored)
    actual = dn.func.run(program, restored)
    _same(actual, expected)
    assert program.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    assert _trace(actual, network).shape[0] == 5
    active = dn.func.restore_continuation(
        program,
        _disk(
            dn.func.export_continuation(program, actual[1]["continuation"]),
            tmp_path / "active.pt",
        ),
    )
    resumed = dn.func.run(program, active)
    _same(resumed, program(active))
    assert _trace(resumed, network).shape[0] == 4
    if execution == "scan":
        gradients = torch.autograd.grad(
            _voltage(resumed[0], network).square().sum(),
            dn.func.inspect_sources(active).trainable_leaves,
            allow_unused=True,
        )
        assert any(value is not None and value.abs().sum() > 0 for value in gradients)
        assert all(value is None or torch.isfinite(value).all() for value in gradients)


@pytest.mark.mps
@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_mps_program_execution_matches_direct_while_checkpoint_continuation(
    network, tmp_path
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        program = dn.compile(
            _network() if network else _population(),
            dt=DT,
            steps=4,
            chunk_steps=4,
            execution="while",
            inference=True,
            trainable=(),
            callbacks={
                "trace": dn.func.Recorder(
                    ("left.v", "right.v", "t") if network else ("v", "t")
                )
            },
            backend="aot_eager",
        )
        initial = dn.func.initialize(program)
        actual = dn.func.run(program, initial)
        _same(actual, program(initial))
        assert _voltage(actual[0], network).device.type == "mps"
        assert _voltage(actual[0], network).dtype == torch.float32
        restored = dn.func.restore_continuation(
            program,
            _disk(
                dn.func.export_continuation(program, actual[1]["continuation"]),
                tmp_path / "mps.pt",
            ),
        )
        resumed = dn.func.run(program, restored)
        _same(resumed, program(actual[1]["continuation"]))
        assert _trace(resumed, network).shape[0] == 4


def test_runner_dispatch_does_not_invoke_unrelated_metaclass_equality():
    class Meta(type):
        def __eq__(self, other):
            raise AssertionError(
                "runner dispatch must compare target types by identity"
            )

    class Other(metaclass=Meta):
        pass

    with pytest.raises(ValueError, match="require run\\(bound"):
        dn.func.run(Other(), steps=1)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_plain_state_and_explicit_callback_carry_resume_without_initial_sample(network):
    program = _runner(network)
    state, prefix = program()
    carry = prefix["callbacks"].state
    state_before = _snapshot(state)
    actual = dn.func.run(program, state=state, callback_state=carry, steps=2)
    expected = program(state, callback_state=carry, steps=2)
    _same(actual, expected)
    assert _trace(actual, network).shape[0] == 2
    _close(state, state_before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_input_tapes_require_the_configured_or_explicit_program_horizon(network):
    program = _runner(network, steps=3)
    _, inputs = _drive(program, network, 2)
    assert _failure(lambda: dn.func.run(program, inputs=inputs)) == _failure(
        lambda: program(inputs=inputs)
    )
    actual = dn.func.run(program, steps=2, inputs=inputs)
    _same(actual, program(steps=2, inputs=inputs))
    assert _trace(actual, network).shape[0] == 3
    restarted = dn.func.run(program)
    _same(restarted, program())
    assert _trace(restarted, network).shape[0] == 4
