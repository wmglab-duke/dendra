"""Fresh trials retain a continuation's effective sources and RNG ownership."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _plain,
    _snapshot,
)
from tests.func.test_explicit_initialization import _frames
from tests.func.test_initialization_previous_state import PROBE
from tests.func.test_initialization_previous_state import _model as _previous_model
from tests.func.test_lowering_api import DT, G, _population, _target, _voltage
from tests.func.test_runner_initialization import _inputs, _program
from tests.func.test_stimulation_rng_sharing import DT as SHARED_DT
from tests.func.test_stimulation_rng_sharing import _network as _random_network
from tests.func.test_stimulation_rng_sharing import _one_stream
from tests.func.test_stimulation_rng_sharing import _population as _random_population
from tests.func.test_stimulation_rng_sharing import _step_native

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                CHECK_PURITY=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                IMEM=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _bundle(program, previous):
    if isinstance(program.functional, dn.func.FunctionalNetwork):
        return dn.func.NetworkTensors(
            previous.parameters, previous.constants, previous.state
        )
    return dn.func.PopulationTensors(
        previous.parameters,
        previous.constants,
        previous.state,
        program.initialization_inputs,
    )


def _workaround(program, previous, *, initialization=None, sample=False):
    initialize = dn.func.sample_initialization_draws if sample else dn.func.initialize
    return initialize(
        program.functional,
        _bundle(program, previous),
        initialization=(
            program.initialization_inputs if initialization is None else initialization
        ),
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_packet_initialization_matches_explicit_bundle_sources_and_gradient_history(
    network, runner, monkeypatch
):
    program = _program(
        network, scaled=True, inference=False, callbacks=_callbacks(network)
    )
    _, auxiliary = program()
    finished = auxiliary["continuation"]
    parameters, constants = _plain(finished.parameters), _plain(finished.constants)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = upstream * 1.2
    _assign(parameters, network, replacement)
    selected_constants = constants["populations"]["left"] if network else constants
    timestep = selected_constants["dt"].clone()
    selected_constants["dt"] = timestep
    previous = dn.func.Continuation(
        finished.state, parameters, constants, finished.callback_state
    )
    factor = upstream.new_tensor(1.7, requires_grad=True)
    defaults = program.initialization_inputs
    voltage = (defaults.populations["left"] if network else defaults).v_init
    inputs = _inputs(
        program, network, voltage=torch.full_like(voltage, -57.0), factor=factor
    )
    if network:
        inputs = inputs._replace(t=inputs.t.new_tensor(0.75))
    expected = _workaround(program, previous, initialization=inputs)
    before = _snapshot(
        (
            _frames(previous),
            previous.callback_state.carries,
            inputs,
            program.parameters,
            program.constants,
            program.initial_state,
        )
    )
    native = _snapshot(program.functional.extract())
    report = program.execution_report()
    prepared = None if network else program._prepared

    def reject(*args, **kwargs):
        raise AssertionError("fresh packet initialization must not enter execution")

    target = program if runner else program.functional
    with monkeypatch.context() as guarded:
        guarded.setattr(program.functional, "extract", reject)
        guarded.setattr(program.functional, "prepare", reject)
        guarded.setattr(program.kernel, "bind", reject)
        guarded.setattr(program.callbacks, "_initialize", reject)
        actual = dn.func.initialize(target, previous, initialization=inputs)
    assert actual.callback_state is None
    _close(_frames(actual), _frames(expected))
    effective = _target(actual.parameters, network)
    assert effective.grad_fn is not None and effective is not replacement
    torch.testing.assert_close(effective, 1.2 * upstream * factor)
    if network:
        assert effective is actual.parameters["populations"]["right"][G]
    selected = actual.constants["populations"]["left"] if network else actual.constants
    assert selected["dt"] is timestep
    assert program.execution_report() is report
    if not network:
        assert program._prepared is prepared
    _close(
        (
            _frames(previous),
            previous.callback_state.carries,
            inputs,
            program.parameters,
            program.constants,
            program.initial_state,
        ),
        before,
    )
    _close(program.functional.extract(), native)
    reference, _ = program(expected, steps=2)
    state, _ = program(actual, steps=2)
    observed = torch.autograd.grad(
        _voltage(state, network).square().mean(),
        (upstream, factor),
        retain_graph=True,
    )
    wanted = torch.autograd.grad(
        _voltage(reference, network).square().mean(), (upstream, factor)
    )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_reinitialized_optimizer_trials_keep_leaves_and_restart_callback_lifecycle(
    network, monkeypatch, tmp_path
):
    program = _program(
        network,
        inference=False,
        callbacks=_callbacks(network),
        checkpoint_every=1,
    )
    trial = dn.func.initialize(program)
    leaf = _target(trial.parameters, network)
    assert leaf.is_leaf and leaf.requires_grad
    optimizer = torch.optim.SGD(
        dn.func.inspect_sources(trial).trainable_leaves, lr=1e-10
    )
    calls = []
    initialize_callbacks = program.callbacks._initialize

    def counted(*args, **kwargs):
        calls.append(None)
        return initialize_callbacks(*args, **kwargs)

    monkeypatch.setattr(program.callbacks, "_initialize", counted)
    key = "left.v" if network else "v"
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        before = leaf.detach().clone()
        state, auxiliary = program(trial, steps=2)
        loss = (_voltage(state, network) + 59.0).square().mean()
        loss.backward()
        assert torch.isfinite(leaf.grad).all() and leaf.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(leaf, before)
        assert len(calls) == iteration + 1
        assert auxiliary["callbacks"]["trace"][key].shape[0] == 3
        finished = dn.func.detach_continuation(program, auxiliary["continuation"])
        assert _target(finished.parameters, network) is leaf
        trial = dn.func.initialize(program, finished)
        assert trial.callback_state is None
        assert _target(trial.parameters, network) is leaf
        assert len(calls) == iteration + 1
        if network:
            assert leaf is trial.parameters["populations"]["right"][G]

    # A restarted trial is an unstarted portable checkpoint; its first run
    # initializes callbacks once, and a restored active checkpoint only resumes.
    path = tmp_path / "trial.pt"
    torch.save(program.export_continuation(trial), path)
    restored = program.restore_continuation(torch.load(path, weights_only=True))
    assert restored.callback_state is None and len(calls) == 2
    with torch.no_grad():
        _, first = program(restored, steps=1)
        torch.save(program.export_continuation(first["continuation"]), path)
        active = program.restore_continuation(torch.load(path, weights_only=True))
        final, last = program(active, steps=1)
        expected, _ = program.functional.prepare_and_rollout(
            active.parameters,
            active.constants,
            active.state,
            steps=1,
            callbacks=program.callbacks,
            callback_state=active.callback_state,
        )
    assert len(calls) == 3
    _close(final, expected)
    assert first["callbacks"]["trace"][key].shape[0] == 2
    assert last["callbacks"]["trace"][key].shape[0] == 1


def _stochastic_program(network, policy):
    options = {
        "reseed": policy == "reseed",
        "retained": policy in ("retained", "cache_restore", "cache_ignore"),
    }
    if network:
        model, pulses = _random_network(**options)
    else:
        model, _, pulse = _random_population(**options)
        pulses = [pulse]
    if policy.startswith("cache_"):
        for _ in range(2):
            _step_native(model)
        owners = model.populations.values() if network else (model,)
        for population in owners:
            population.mech.probe.stream.ignore_rng_on_load(policy == "cache_ignore")
        if network:
            model.cache_state()
        else:
            model.cache("_steady_state")
        for _ in range(2):
            _step_native(model)
    return (
        dn.compile(
            model,
            dt=SHARED_DT,
            steps=2,
            inference=True,
            backend="eager",
            disable=True,
        ),
        pulses,
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize(
    "policy", ("auto", "reseed", "retained", "cache_restore", "cache_ignore")
)
def test_packet_rng_and_initialization_draw_replay_preserve_existing_sampling_policies(
    network, policy, monkeypatch
):
    program, pulses = _stochastic_program(network, policy)
    initial = dn.func.initialize(program)
    _, auxiliary = program(initial)
    previous = auxiliary["continuation"]
    before = _snapshot(
        (
            _frames(previous),
            program.parameters,
            program.constants,
            program.initial_state,
        )
    )
    native = _snapshot(program.functional.extract())
    global_rng = torch.get_rng_state().clone()
    source_rng = [pulse.generator.get_state().clone() for pulse in pulses]
    expected = _workaround(program, previous)
    expected_draws = _workaround(program, previous, sample=True)
    captured = dn.func.sample_initialization_draws(program, previous)
    _close(captured, expected_draws)
    for value in pytree.tree_leaves(captured):
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad and value.grad_fn is None
    actual = dn.func.initialize(program, previous)
    _close(_frames(actual), _frames(expected))
    options = {} if network else {"initialization": program.initialization_inputs}
    direct = dn.func.initialize(program.functional, previous, **options)
    _close(_frames(direct), _frames(expected))
    _close(
        dn.func.sample_initialization_draws(program.functional, previous, **options),
        expected_draws,
    )
    if policy == "auto":
        # Advancing from the packet differs from a no-packet restart, whose
        # starting stream remains the runner's compile-time snapshot.
        assert not torch.equal(_one_stream(actual.state), _one_stream(initial.state))

    def reject(*args, **kwargs):
        raise AssertionError("captured packet replay must not sample automatically")

    with monkeypatch.context() as guarded:
        for name in (
            "sample_initialization_draw",
            "sample_stimulation_initialization_draw",
            "sample_population_initialization_draws",
            "sample_population_stimulation_initialization_draws",
        ):
            if hasattr(program.functional, name):
                guarded.setattr(program.functional, name, reject)
        replayed = dn.func.initialize(program, previous, draws=captured)
    _close(_frames(replayed), _frames(expected))
    _close(
        (
            _frames(previous),
            program.parameters,
            program.constants,
            program.initial_state,
        ),
        before,
    )
    _close(program.functional.extract(), native)
    _close(torch.get_rng_state(), global_rng)
    for pulse, rng in zip(pulses, source_rng, strict=True):
        _close(pulse.generator.get_state(), rng)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("sample", (False, True), ids=("initialize", "sample"))
def test_packet_initialization_requires_explicit_ownership_and_population_recipe(
    network, sample, monkeypatch
):
    program = _program(network)
    packet = dn.func.initialize(program)
    initialize = dn.func.sample_initialization_draws if sample else dn.func.initialize
    expected = initialize(program, packet)
    actual = initialize(program, packet, parameters=None, constants=None)
    _close(
        actual if sample else _frames(actual), expected if sample else _frames(expected)
    )
    for keyword in ("parameters", "constants"):
        with pytest.raises(
            ValueError, match=f"^continuation cannot be combined with {keyword}$"
        ):
            initialize(program, packet, **{keyword: {}})
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{keyword}'"):
            initialize(program.functional, packet, **{keyword: None})
    with pytest.raises(
        ValueError, match="^continuation cannot be combined with parameters, constants$"
    ):
        initialize(program, packet, parameters={}, constants={})

    def reject(*args, **kwargs):
        raise AssertionError("initialization must not extract a native recipe or state")

    with monkeypatch.context() as guarded:
        guarded.setattr(program.functional, "extract", reject)
        if network:
            selected = initialize(program.functional, packet)
            _close(
                selected if sample else _frames(selected),
                expected if sample else _frames(expected),
            )
        else:
            with pytest.raises(TypeError, match="initialization"):
                initialize(program.functional, packet)
            selected = initialize(
                program.functional, packet, initialization=program.initialization_inputs
            )
            _close(
                selected if sample else _frames(selected),
                expected if sample else _frames(expected),
            )


def test_packet_initialization_keeps_previous_state_reads_distinct_from_insertion_values():
    program = dn.compile(
        _previous_model(), dt=DT, steps=1, backend="eager", disable=True
    )
    state = _plain(program.initial_state)
    previous_x = state["mechanisms"][PROBE]["x"].new_full((1, 2), 3.0).requires_grad_()
    previous_y = state["mechanisms"][PROBE]["y"].new_full((1, 2), 4.0).requires_grad_()
    state["mechanisms"][PROBE].update(x=previous_x, y=previous_y)
    packet = dn.func.Continuation(state, program.parameters, program.constants)
    actual = dn.func.initialize(program, packet)
    voltage = actual.state["integrator"]["v"]
    torch.testing.assert_close(voltage, torch.full_like(voltage, -44.0))
    for key, value in (("x", 0.4), ("y", 0.8)):
        initialized = actual.state["mechanisms"][PROBE][key]
        torch.testing.assert_close(initialized, torch.full_like(initialized, value))
    gradients = torch.autograd.grad(voltage.sum(), (previous_x, previous_y))
    for gradient, coefficient in zip(gradients, (1.0, 2.0), strict=True):
        torch.testing.assert_close(gradient, torch.full_like(gradient, coefficient))
    torch.testing.assert_close(previous_x, torch.full_like(previous_x, 3.0))
    torch.testing.assert_close(previous_y, torch.full_like(previous_y, 4.0))


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_fresh_initialization_discards_callback_carry_from_a_different_collection(
    network,
):
    program = _program(network, callbacks=_callbacks(network))
    other = _program(network, callbacks=_callbacks(network))
    _, auxiliary = other(steps=1)
    foreign = auxiliary["continuation"].callback_state
    packet = dn.func.Continuation(
        program.initial_state, program.parameters, program.constants, foreign
    )
    initial = dn.func.initialize(program, packet)
    assert initial.callback_state is None
    _, outputs = program(initial, steps=1)
    key = "left.v" if network else "v"
    assert outputs["callbacks"]["trace"][key].shape[0] == 2


@pytest.mark.mps
def test_mps_packet_initialization_preserves_caller_gradients_and_while_checkpoint(
    tmp_path,
):
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        program = dn.compile(
            _population(),
            dt=DT,
            steps=4,
            chunk_steps=4,
            execution="while",
            callbacks={"trace": dn.func.Recorder(("v", "t"))},
            backend="aot_eager",
        )
        _, first = program()
        previous = first["continuation"]
        voltage = program.initialization_inputs.v_init.detach().clone().requires_grad_()
        inputs = _inputs(program, False, voltage=voltage)
        expected = _workaround(program, previous, initialization=inputs)
        captured = dn.func.sample_initialization_draws(
            program, previous, initialization=inputs
        )
        actual = dn.func.initialize(
            program, previous, initialization=inputs, draws=captured
        )
        assert actual.callback_state is None
        _close(_frames(actual), _frames(expected))
        assert actual.state["integrator"]["v"].device.type == "mps"
        assert actual.state["integrator"]["v"].dtype == torch.float32
        assert not actual.state["integrator"]["v"].is_inference()
        derivative = torch.autograd.grad(
            actual.state["integrator"]["v"].sum(), voltage
        )[0]
        torch.testing.assert_close(derivative, torch.ones_like(voltage))
        with torch.no_grad():
            reference, outputs = program.functional.prepare_and_rollout(
                expected.parameters,
                expected.constants,
                expected.state,
                steps=4,
                callbacks=program.callbacks,
            )
        _, prefix = program(actual, steps=2)
        path = tmp_path / "restarted.pt"
        torch.save(program.export_continuation(prefix["continuation"]), path)
        restored = program.restore_continuation(torch.load(path, weights_only=True))
        final, suffix = program(restored, steps=2)
        _close(final, reference)
        for key, value in outputs["callbacks"]["trace"].items():
            _close(
                torch.cat(
                    (
                        prefix["callbacks"]["trace"][key],
                        suffix["callbacks"]["trace"][key],
                    )
                ),
                value,
            )
