"""Custom waveform arithmetic keeps its authored random invocation order."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.stim.waveform.core import Waveform
from dendra.units import nA
from tests.func.test_mechanism_injection_execution import _InjectedLeak
from tests.func.test_poisson_invocation_execution import (
    _native_state,
    _TwiceInjectedLeak,
)
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


class _SquaredGain(torch.nn.Module):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor(0.15))

    def forward(self, value):
        return value.square() + self.offset


class _OrderedContainer(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(gain=0.7)

    def __init__(self, first, second, unused, *, shared):
        super().__init__()
        self.nodes = torch.nn.ModuleDict(
            {"first": first, "second": second, "unused": unused}
        )
        self.aliases = torch.nn.ModuleList([first, first if shared else second])

    def fn(self, times):
        # Registration order differs from evaluation order. Both supported
        # child dispatch spellings participate in the same invocation recipe.
        second = self.nodes["second"](times)
        first = self.aliases[0].fn(times)
        repeated = self.nodes["second"].fn(times)
        last = self.aliases[1](times)
        return self.gain * (second + 0.25 * first + 0.5 * repeated + 0.125 * last)


class _UnmarkedContainer(_OrderedContainer):
    FUNCTIONAL_PURE = False


def _expression(
    role, *, generator, shared=False, parametrized=False, unused_generator=None
):
    unit = nA if role == "intra" else 1.0
    first = _waveform(generator=generator, amp=0.2 * unit)
    second = _waveform(generator=generator, amp=0.13 * unit)
    # Sharing a raw Parameter between different Poisson nodes must survive
    # their different registration paths and independently generated schedules.
    second.waveform.amp = first.waveform.amp
    unused_generator = (
        torch.Generator().manual_seed(251)
        if unused_generator is None
        else unused_generator
    )
    unused = _waveform(generator=unused_generator, amp=0.17 * unit)
    result = _OrderedContainer(first, second, unused, shared=shared)
    if parametrized:
        torch.nn.utils.parametrize.register_parametrization(
            result, "gain", _SquaredGain()
        )
    return result, unused_generator


def _case(role="intra", *, shared=False, parametrized=False, twice=False):
    generator = torch.Generator().manual_seed(149)
    expression, unused_generator = _expression(
        role, generator=generator, shared=shared, parametrized=parametrized
    )
    population, _, extra = _population(
        role,
        waveform=expression,
        mechanism=_TwiceInjectedLeak if twice else _InjectedLeak,
    )
    return population, extra, expression, generator, unused_generator


def _equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_custom_container_purity_marker_remains_an_optional_diagnostic():
    generator = torch.Generator().manual_seed(151)
    template, unused = _expression("intra", generator=generator, shared=True)
    expression = _UnmarkedContainer(
        template.nodes["first"],
        template.nodes["second"],
        template.nodes["unused"],
        shared=True,
    )
    source = _population("intra", waveform=expression)[0]
    before = generator.get_state().clone()
    unused_before = unused.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=False):
        functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    packet = functional.sample_stimulation_draw(tensors.state)
    functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(stimulation=packet),
    )
    with (
        dn.ctx(CHECK_PURITY=True),
        pytest.raises(dn.func.FunctionalizationError, match="FUNCTIONAL_PURE|pure"),
    ):
        dn.func.make_functional(source, dt=DT)
    _close(tensors, frozen)
    _equal(generator.get_state(), before)
    _equal(unused.get_state(), unused_before)
    _equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("role", ("intra", "extra", "mechanism"))
@pytest.mark.parametrize("shared", (False, True))
def test_custom_call_order_replays_native_tapes_chunks_and_keeps_unused_rng(
    role, shared
):
    source, extra, expression, generator, unused_generator = _case(role, shared=shared)
    native, native_extra, _, native_generator, native_unused = _case(
        role, shared=shared
    )
    source_rng = generator.get_state().clone()
    unused_rng = unused_generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=shared):
        functional, tensors = dn.func.make_functional(source, dt=DT, extra=extra)
    frozen = _clone(tensors)
    tape = functional.sample_stimulation_draw_tape(tensors.state, 2 * STEPS)
    _equal(tape, functional.sample_stimulation_draw_tape(tensors.state, 2 * STEPS))
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
    state = tensors.state
    for index in range(2 * STEPS):
        packet = pytree.tree_map_only(torch.Tensor, lambda value: value[index], tape)
        state, _ = functional.step(
            tensors.parameters, prepared, state, dn.func.StepInput(stimulation=packet)
        )
        native.step(DT, extra=native_extra)
        _close(state["integrator"]["v"], native.v)
        for entry in functional._poisson_plan.entries:
            expected = (
                native_unused.get_state()
                if entry.source is expression.nodes["unused"]
                else native_generator.get_state()
            )
            _equal(packet.rng[entry.stream], expected)
    _close(whole, state)
    _close(whole, _native_state(native, native_extra))
    gradients = torch.autograd.grad(
        whole["integrator"]["v"].square().mean(),
        tuple(tape.draws.values()),
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)
    assert (
        expression.nodes["first"].waveform.amp
        is expression.nodes["second"].waveform.amp
    )
    _equal(unused_rng, native_unused.get_state())
    _equal(generator.get_state(), source_rng)
    _equal(unused_generator.get_state(), unused_rng)
    _equal(torch.get_rng_state(), global_rng)
    _close(tensors, frozen)
    _close(inputs, frozen_inputs)


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
@pytest.mark.parametrize("parametrized", (False, True))
def test_custom_gain_and_shared_amplitude_gradients_match_native_initialization(
    initialization, parametrized
):
    source, _, _, generator, _ = _case(
        "mechanism", shared=True, parametrized=parametrized, twice=True
    )
    native, _, expression, _, _ = _case(
        "mechanism", shared=True, parametrized=parametrized, twice=True
    )
    before = generator.get_state().clone()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization=initialization,
        inference=False,
        backend="aot_eager",
        checkpoint_every=2,
        trainable=(
            "*injected_waveforms.0.parametrizations.gain.original"
            if parametrized
            else "*injected_waveforms.0.gain",
            "*waveform.amp",
        ),
    )
    frozen = _clone(runner.functional.extract())
    final, _ = runner()
    replay, _ = runner()
    _close(final, replay)
    if initialization == "fresh":
        native.initialize()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    amplitude = next(
        value
        for name, value in runner.trainable_parameters.items()
        if name.endswith("waveform.amp")
    )
    gain = next(
        value
        for name, value in runner.trainable_parameters.items()
        if name.endswith(".gain") or name.endswith("gain.original")
    )
    native_gain = (
        expression.parametrizations.gain.original if parametrized else expression.gain
    )
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().mean(), (amplitude, gain)
    )
    expected = torch.autograd.grad(
        native.v.square().mean(), (expression.nodes["first"].waveform.amp, native_gain)
    )
    assert all(value.abs().sum() > 0 for value in expected)
    _close(actual, expected)
    assert all(
        not value.requires_grad for value in final["stimulation"]["schedules"].values()
    )
    _close(runner.functional.extract(), frozen)
    _equal(generator.get_state(), before)


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("role", ("intra", "mechanism"))
def test_custom_runner_records_restarts_and_restores_native_checkpoint(role, execution):
    source, _, _, generator, _ = _case(role, shared=True)
    native, _, _, _, _ = _case(role, shared=True)
    before = generator.get_state().clone()
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
    committed = _case(role, shared=True)[0]
    runner.functional.commit_state_(committed, final)
    restored = _case(role, shared=True)[0]
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)
    _close(runner.functional.extract(), frozen)
    _equal(generator.get_state(), before)


def _network_case(sharing):
    generator = torch.Generator().manual_seed(227)
    shared = _expression(
        "mechanism", generator=generator, shared=True, unused_generator=generator
    )[0]
    populations, contacts = {}, {}
    for name in ("left", "right"):
        expression = (
            shared
            if sharing == "waveform"
            else _expression(
                "mechanism",
                generator=generator,
                shared=True,
                unused_generator=generator,
            )[0]
        )
        population = _population("mechanism", waveform=expression)[0]
        populations[name] = population
        field = torch.linspace(-1.0, 1.5, population.v.numel(), dtype=DTYPE).reshape(
            population.shape
        )
        contacts[name] = (
            field,
            _expression(
                "extra", generator=generator, shared=False, unused_generator=generator
            )[0],
        )
    network = dn.Network(populations)
    network.train()
    network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network, {name: contacts[name] for name in ("right", "left")}, generator


@pytest.mark.parametrize("sharing", ("generator", "waveform"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_custom_network_orders_extra_and_population_calls_with_one_stream(
    sharing, execution
):
    source, extra, generator = _network_case(sharing)
    native, native_extra, native_generator = _network_case(sharing)
    before = generator.get_state().clone()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        extra=extra,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract())
    prefix, first = runner()
    saved = _clone(prefix)
    final, _ = runner(saved, callback_state=first["callbacks"].state)
    native.initialize(DT)
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(extra=native_extra)
    _close(final, _native_state(native, native_extra))
    assert len(final["stimulation_rng"]) == 1
    _equal(next(iter(final["stimulation_rng"].values())), native_generator.get_state())
    for state in final["populations"].values():
        assert "rng" not in state["stimulation"]
    _close(prefix, saved)
    _close(runner.functional.extract(), frozen)
    _equal(generator.get_state(), before)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_custom_invocations_compile_with_inductor_forward_and_backward(execution):
    role = "mechanism" if execution == "scan" else "intra"
    source, _, _, generator, _ = _case(role, shared=True)
    native, _, expression, _, _ = _case(role, shared=True)
    before = generator.get_state().clone()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=execution == "while",
        backend="inductor",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        **(
            {"checkpoint_every": 2, "trainable": ("*waveform.amp",)}
            if execution == "scan"
            else {}
        ),
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    if execution == "scan":
        amplitude = next(
            value
            for name, value in runner.trainable_parameters.items()
            if name.endswith("waveform.amp")
        )
        actual = torch.autograd.grad(
            final["integrator"]["v"].square().mean(), amplitude
        )[0]
        expected = torch.autograd.grad(
            native.v.square().mean(), expression.nodes["first"].waveform.amp
        )[0]
        assert expected.abs().sum() > 0
        _close(actual, expected)
    _equal(generator.get_state(), before)


@pytest.mark.parametrize("randomized", (False, True))
def test_custom_container_inside_poisson_envelope_preserves_nested_call_order(
    randomized,
):
    def make():
        generator = torch.Generator().manual_seed(293)
        inner, unused = _expression("intra", generator=generator, shared=True)
        outer = inner.poisson(
            interval=0.1,
            n=2,
            start=-0.035,
            noise=0.7,
            randomize_every_call=randomized,
            generator=generator,
        )
        return _population("intra", waveform=outer)[0], generator, unused

    source, generator, unused = make()
    native, native_generator, native_unused = make()
    before = generator.get_state().clone()
    unused_before = unused.get_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    for _ in range(STEPS):
        packet = functional.sample_stimulation_draw(state)
        state, _ = functional.step(
            tensors.parameters, prepared, state, dn.func.StepInput(stimulation=packet)
        )
        native.step(DT)
        _close(state, functional.extract(native).state)
        assert any(
            torch.equal(frame, native_generator.get_state())
            for frame in packet.rng.values()
        )
    _equal(generator.get_state(), before)
    _equal(unused.get_state(), unused_before)
    _equal(native_unused.get_state(), unused_before)
    _close(tensors, frozen)
