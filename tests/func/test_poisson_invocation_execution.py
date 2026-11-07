"""Each randomized waveform invocation owns draws and publishes its final schedule."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.units import nA
from tests.func.test_mechanism_injection_execution import _InjectedLeak
from tests.func.test_poisson_stimulation import (
    DT,
    DTYPE,
    STEPS,
    _clone,
    _close,
    _population,
    _slice,
    _waveform,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _TwiceInjectedLeak(_InjectedLeak):
    def assigned_values(self, v, values):
        del values
        return {
            "i_inj": self.injection_values(v, t=self.t)
            + 0.25 * self.injection_values(v, t=self.t)
        }


def _expression(kind, role, generator, *, n=5):
    pulse = _waveform(
        generator=generator, amp=0.2 * nA if role == "intra" else 0.2, n=n
    )
    if kind == "repeated":
        expression = pulse + pulse
    else:
        expression = pulse.poisson(
            interval=0.085,
            n=2,
            start=-0.045,
            noise=0.6,
            randomize_every_call=True,
            generator=generator,
        )
    return expression, pulse


def _case(kind="repeated", role="intra", *, mechanism=_InjectedLeak, n=5):
    generator = torch.Generator().manual_seed(73)
    expression, pulse = _expression(kind, role, generator, n=n)
    population, _, extra = _population(role, waveform=expression, mechanism=mechanism)
    return population, extra, pulse, generator


def _native_state(population, extra=None):
    # A separately lowered extracellular binding captures the native extra
    # waveform's final schedule, rather than the source extra supplied to the
    # actual functional program.
    if isinstance(population, dn.Network):
        return dn.func.make_functional_network(population, extra=extra)[1].state
    return dn.func.make_functional(population, dt=DT, extra=extra)[1].state


@pytest.mark.parametrize("kind", ("repeated", "nested"))
@pytest.mark.parametrize("role", ("intra", "mechanism", "extra"))
def test_invocation_draw_tape_matches_native_calls_chunks_and_detached_gradients(
    kind, role
):
    source, extra, _, source_generator = _case(kind, role)
    native, native_extra, _, native_generator = _case(kind, role)
    source_rng = source_generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT, extra=extra)
    frozen = _clone(tensors)
    tape = functional.sample_stimulation_draw_tape(tensors.state, 2 * STEPS)
    _close(tape, functional.sample_stimulation_draw_tape(tensors.state, 2 * STEPS))
    assert isinstance(tape, dn.func.StimulationDraw)
    # Exact end-frame comparisons below prove invocation multiplicity and
    # ordering independently of the draw packet's reserved slot capacity.
    assert len(tape.rng) == 1
    tape = tape._replace(
        draws={
            name: value.detach().clone().requires_grad_()
            for name, value in tape.draws.items()
        }
    )
    inputs = dn.func.RolloutInput(stimulation=tape)
    frozen_inputs = _clone(inputs)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    whole, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, inputs, steps=2 * STEPS
    )
    prefix, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        _slice(inputs, 0, STEPS),
        steps=STEPS,
    )
    chunked, _ = functional.rollout(
        tensors.parameters,
        prepared,
        prefix,
        _slice(inputs, STEPS, 2 * STEPS),
        steps=STEPS,
    )
    _close(whole, chunked)
    stepped = tensors.state
    for index in range(2 * STEPS):
        packet = pytree.tree_map_only(torch.Tensor, lambda value: value[index], tape)
        stepped, _ = functional.step(
            tensors.parameters,
            prepared,
            stepped,
            dn.func.StepInput(stimulation=packet),
        )
        native.step(DT, extra=native_extra)
        _close(stepped["integrator"]["v"], native.v)
        _close(next(iter(packet.rng.values())), native_generator.get_state())
    _close(whole, stepped)
    _close(whole, _native_state(native, native_extra))
    gradients = torch.autograd.grad(
        whole["integrator"]["v"].square().mean(),
        tuple(tape.draws.values()),
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)
    _close(tensors, frozen)
    _close(inputs, frozen_inputs)
    _close(source_generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("kind", ("repeated", "nested"))
def test_invocation_runner_restarts_continues_records_and_restores_native_checkpoint(
    execution, kind
):
    source, _, _, generator = _case(kind, "mechanism")
    native, _, _, _ = _case(kind, "mechanism")
    source_rng = generator.get_state().clone()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract())
    prefix, first = runner()
    replay, replay_auxiliary = runner()
    _close(prefix, replay)
    _close(first["callbacks"]["trace"], replay_auxiliary["callbacks"]["trace"])
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    with torch.no_grad():
        expected = [native.v.clone()]
        for _ in range(2 * STEPS):
            native.step(DT)
            expected.append(native.v.clone())
    observed = torch.cat(
        (first["callbacks"]["trace"]["v"], second["callbacks"]["trace"]["v"])
    )
    _close(observed, torch.stack(expected))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    _close(generator.get_state(), source_rng)
    committed, _, _, _ = _case(kind, "mechanism")
    runner.functional.commit_state_(committed, final)
    restored, _, _, _ = _case(kind, "mechanism")
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("kind", ("repeated", "nested"))
@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
@pytest.mark.parametrize("mechanism", (_InjectedLeak, _TwiceInjectedLeak))
def test_shared_amplitude_gradients_match_native_multiple_hook_invocations(
    kind, initialization, mechanism
):
    source, _, _, generator = _case(kind, "mechanism", mechanism=mechanism)
    native, _, native_pulse, _ = _case(kind, "mechanism", mechanism=mechanism)
    before = generator.get_state().clone()
    runner = dn.compile(
        source,
        dt=DT,
        steps=2 * STEPS,
        initialization=initialization,
        inference=False,
        backend="aot_eager",
        checkpoint_every=2,
        trainable=("*waveform.amp",),
    )
    amplitudes = tuple(runner.trainable_parameters.values())
    assert len({id(value) for value in amplitudes}) == 1
    frozen = _clone(runner.functional.extract())
    final, _ = runner()
    replay, _ = runner()
    _close(final, replay)
    if initialization == "fresh":
        native.initialize()
    for _ in range(2 * STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().mean(), amplitudes[0]
    )[0]
    expected = torch.autograd.grad(native.v.square().mean(), native_pulse.waveform.amp)[
        0
    ]
    assert expected.abs().sum() > 0
    _close(actual, expected)
    assert all(
        not value.requires_grad for value in final["stimulation"]["schedules"].values()
    )
    _close(runner.functional.extract(), frozen)
    _close(generator.get_state(), before)


def _network_case(sharing, *, n=5):
    generator = torch.Generator().manual_seed(137)
    shared = _expression("repeated", "mechanism", generator, n=n)[0]
    populations = {}
    for name in ("left", "right"):
        expression = (
            shared
            if sharing == "waveform"
            else _expression("repeated", "mechanism", generator, n=n)[0]
        )
        populations[name] = _population(
            "mechanism", waveform=expression, mechanism=_TwiceInjectedLeak
        )[0]
    network = dn.Network(populations)
    network.train()
    network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network, generator


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("sharing", ("generator", "waveform"))
def test_network_invocations_keep_one_stream_native_population_order_and_continuation(
    execution, sharing
):
    # Keep event traffic and overlap through the native checkpoint suffix.
    source, source_generator = _network_case(sharing, n=4)
    native, native_generator = _network_case(sharing, n=4)
    before = source_generator.get_state().clone()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=execution == "while",
        execution=execution,
        initialization="fresh",
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract())
    prefix, _ = runner()
    final, _ = runner(prefix)
    native.initialize(DT)
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step()
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    _close(source_generator.get_state(), before)
    # One Network carry owns the physical stream; individual Population
    # checkpoints retain schedules without duplicated RNG state.
    assert len(final["stimulation_rng"]) == 1
    _close(next(iter(final["stimulation_rng"].values())), native_generator.get_state())
    for population in final["populations"].values():
        assert "rng" not in population["stimulation"]
    committed, _ = _network_case(sharing, n=4)
    runner.functional.commit_state_(committed, final)
    restored, _ = _network_case(sharing, n=4)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("kind", ("repeated", "nested"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_inductor_poisson_invocations_preserve_native_values_and_rng(kind, execution):
    # Two events retain repeated/nested calls and overlap in this short rollout.
    source, _, _, generator = _case(kind, "mechanism", n=2)
    native, _, native_pulse, _ = _case(kind, "mechanism", n=2)
    before = generator.get_state().clone()
    steps = STEPS if execution == "while" else 2
    runner = dn.compile(
        source,
        dt=DT,
        steps=steps,
        inference=execution == "while",
        execution=execution,
        backend="inductor",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        **(
            {"checkpoint_every": 1, "trainable": ("*waveform.amp",)}
            if execution == "scan"
            else {}
        ),
    )
    final, _ = runner()
    for _ in range(steps):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    if execution == "scan":
        amplitude = next(
            value
            for name, value in runner.named_parameters.items()
            if name.endswith("waveform.amp")
        )
        actual = torch.autograd.grad(
            final["integrator"]["v"].square().mean(), amplitude
        )[0]
        expected = torch.autograd.grad(
            native.v.square().mean(), native_pulse.waveform.amp
        )[0]
        assert expected.abs().sum() > 0
        _close(actual, expected)
    _close(generator.get_state(), before)


def test_explicit_initialization_draws_replay_multiple_mechanism_invocations():
    source, _, _, generator = _case(
        "repeated", "mechanism", mechanism=_TwiceInjectedLeak
    )
    native, _, _, native_generator = _case(
        "repeated", "mechanism", mechanism=_TwiceInjectedLeak
    )
    before = generator.get_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    packet = functional.sample_stimulation_initialization_draw(tensors.state)
    _close(packet, functional.sample_stimulation_initialization_draw(tensors.state))
    initialization = tensors.initialization._replace(stimulation=packet)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        initialization,
        state=tensors.state,
    )
    replay = functional.initialize(
        tensors.parameters,
        tensors.constants,
        initialization,
        state=tensors.state,
    )
    _close(initialized, replay)
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
    _close(next(iter(packet.rng.values())), native_generator.get_state())
    final, _ = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.StepInput(
            stimulation=functional.sample_stimulation_draw(initialized.state)
        ),
    )
    native.step(DT)
    _close(final, functional.extract(native).state)
    _close(tensors, frozen)
    _close(generator.get_state(), before)


def _ordered_network_case():
    generator = torch.Generator().manual_seed(197)
    populations, contacts = {}, {}
    for name, role in (("left", "intra"), ("right", "mechanism")):
        expression, _ = _expression("repeated", role, generator)
        population = _population(role, waveform=expression)[0]
        populations[name] = population
        field = torch.linspace(-1.0, 1.5, population.v.numel(), dtype=DTYPE).reshape(
            population.shape
        )
        contacts[name] = (field, _expression("repeated", "extra", generator)[0])
    network = dn.Network(populations)
    network.train()
    network.initialize(DT)
    for population in populations.values():
        population.detach()
    return network, {name: contacts[name] for name in ("right", "left")}, generator


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_network_invocations_preserve_extra_intra_mechanism_phase_order(execution):
    source, extra, generator = _ordered_network_case()
    native, native_extra, native_generator = _ordered_network_case()
    before = generator.get_state().clone()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        extra=extra,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract())
    prefix, _ = runner()
    final, _ = runner(prefix)
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(extra=native_extra)
    _close(final, _native_state(native, native_extra))
    assert len(final["stimulation_rng"]) == 1
    _close(next(iter(final["stimulation_rng"].values())), native_generator.get_state())
    _close(runner.functional.extract(), frozen)
    _close(generator.get_state(), before)


@pytest.mark.parametrize("outer", ("empty", "early_off", "single", "off_before_start"))
def test_nested_invocations_follow_raw_schedule_slots_including_empty_sentinel(outer):
    def make():
        generator = torch.Generator().manual_seed(257)
        pulse = _waveform(generator=generator, amp=0.2 * nA)
        n, noise, off = {
            "empty": (0, 0.7, torch.inf),
            "early_off": (5, 0.0, 0.15),
            "single": (1, 0.7, 0.15),
            "off_before_start": (5, 0.7, -0.01),
        }[outer]
        expression = pulse.poisson(
            interval=0.1,
            n=n,
            start=0.0,
            noise=noise,
            off=off,
            randomize_every_call=True,
            generator=generator,
        )
        return _population("intra", waveform=expression)[0], generator

    source, generator = make()
    native, native_generator = make()
    before = generator.get_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    for _ in range(STEPS):
        packet = functional.sample_stimulation_draw(state)
        state, _ = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(stimulation=packet),
        )
        native.step(DT)
        _close(state, functional.extract(native).state)
        _close(next(iter(packet.rng.values())), native_generator.get_state())
    _close(tensors, frozen)
    _close(generator.get_state(), before)


def test_draw_dependent_nested_invocation_count_is_rejected_without_consuming_rng():
    generator = torch.Generator().manual_seed(277)
    pulse = _waveform(generator=generator, amp=0.2 * nA)
    expression = pulse.poisson(
        interval=0.1,
        n=5,
        start=0.0,
        noise=0.7,
        off=0.15,
        randomize_every_call=True,
        generator=generator,
    )
    population = _population("intra", waveform=expression)[0]
    before = generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="[Nn]est|finite.*off|fixed.*invocation",
    ):
        dn.func.make_functional(population, dt=DT)
    _close(generator.get_state(), before)
    _close(torch.get_rng_state(), global_rng)


def test_invocation_draw_axis_is_validated_before_execution_or_rng_consumption():
    source, _, _, generator = _case(
        "repeated", "mechanism", mechanism=_TwiceInjectedLeak
    )
    before = generator.get_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = functional.sample_stimulation_draw(tensors.state)
    draws = dict(packet.draws)
    name = next(name for name, value in draws.items() if value.ndim == 2)
    draws[name] = draws[name][:-1]
    invalid = packet._replace(draws=draws)
    frozen = _clone(tensors)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    with pytest.raises(ValueError, match="schema|shape|invocation"):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(stimulation=invalid),
        )
    _close(tensors, frozen)
    _close(generator.get_state(), before)


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("zero", ("early", "last"))
def test_nested_parent_zero_draw_preserves_fixed_invocations_or_rejects(compiled, zero):
    source, _, _, generator = _case("nested", "intra")
    before = generator.get_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = functional.sample_stimulation_draw(tensors.state)
    draws = {name: value.clone() for name, value in packet.draws.items()}
    parent = next(name for name, value in draws.items() if value.shape == (2,))
    draws[parent][0 if zero == "early" else -1] = 0.0
    modified = packet._replace(draws=draws)
    frozen = _clone((tensors, packet, modified))
    evaluate = functional._poisson_plan.schedules
    if compiled:
        evaluate = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    expected = evaluate(packet)
    if zero == "early":
        with pytest.raises(
            (dn.func.FunctionalizationError, RuntimeError),
            match="fixed nested invocation count|draw-dependent invocation",
        ):
            evaluate(modified)
    else:
        # The interval following the final emitted event is unused, even when
        # that interval is infinite. Its draw does not change the child calls.
        _close(evaluate(modified), expected)
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        actual, _ = functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(stimulation=modified),
        )
        reference, _ = functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(stimulation=packet),
        )
        _close(actual, reference)
    _close((tensors, packet, modified), frozen)
    _close(generator.get_state(), before)
