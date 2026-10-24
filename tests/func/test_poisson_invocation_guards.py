"""Poisson packets must match each mechanism's numerical invocation profile."""

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.stim.waveform.core import Waveform
from dendra.units import nA
from tests.func.test_mechanism_injection_execution import (
    _InjectedLeak,
    _NamedInjectedLeak,
)
from tests.func.test_poisson_stimulation import (
    DT,
    DTYPE,
    _clone,
    _close,
    _population,
    _waveform,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        yield


class _RepeatedAssigned(_InjectedLeak):
    def assigned_values(self, v, values):
        del values
        return {
            "i_inj": self.injection_values(v, t=self.t)
            + self.injection_values(v, t=self.t)
        }


class _RepeatedInitializer(_InjectedLeak):
    Mechanism.CARRY("seen")

    def initial_values(self, v, values):
        del values
        return {
            "seen": self.injection_values(v, t=self.t)
            + self.injection_values(v, t=self.t)
        }


class _PhaseScoped(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.CARRY("seen")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def initial_values(self, v, values):
        del values
        return {"seen": torch.zeros_like(v)}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e


class _AdvanceOnly(_PhaseScoped):
    def advance(self, v, dt, values):
        del dt, values
        return {"seen": self.injection_values(v, t=self.t)}


class _InitializationOnly(_PhaseScoped):
    def initial_values(self, v, values):
        del values
        return {"seen": self.injection_values(v, t=self.t)}


def _case(mechanism, *, waveform=None):
    population = dn.SingleCompartment(
        N=1, C=2, dtype=DTYPE, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism.rename("owner"))
    waveform = _waveform() if waveform is None else waveform
    population[:, 0].inject(waveform)
    from dendra.models.stim.waveform.core import _poisson

    poisson_nodes = [node for node in waveform.modules() if isinstance(node, _poisson)]
    for node in poisson_nodes:
        node.randomize_every_call = False
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    for node in poisson_nodes:
        node.randomize_every_call = True
    return population, waveform


@pytest.mark.parametrize(
    "mechanism", (_RepeatedAssigned, _NamedInjectedLeak, _RepeatedInitializer)
)
@pytest.mark.parametrize("diagnostic", (False, True))
def test_repeated_poisson_evaluation_matches_native_without_mutating_source(
    mechanism, diagnostic
):
    population, waveform = _case(mechanism)
    native, _ = _case(mechanism)
    generator = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    voltage = population.v.detach().clone()
    schedule = waveform._spike_times.detach().clone()
    with dn.ctx(CHECK_PURITY=diagnostic):
        functional, tensors = dn.func.make_functional(population, dt=DT)
        frozen = _clone(tensors)
        parameters, constants, state = tensors[:3]
        if mechanism is not _RepeatedInitializer:
            initialization = tensors.initialization._replace(
                stimulation=functional.sample_stimulation_initialization_draw(state)
            )
            initialized = functional.initialize(
                parameters, constants, initialization, state=state
            )
            native.initialize()
            _close(initialized.state, functional.extract(native).state)
            parameters, constants, state = initialized[:3]
        # Native initial_values builders reject implicit RNG consumption. The
        # initializer-only case starts from its valid fixed-schedule snapshot;
        # its accepted steps can evaluate the randomized current normally.
        prepared = functional.prepare(parameters, constants)
        for _ in range(3):
            packet = functional.sample_stimulation_draw(state)
            state, _ = functional.step(
                parameters,
                prepared,
                state,
                dn.func.StepInput(stimulation=packet),
            )
            native.step(dt=DT)
            _close(state, functional.extract(native).state)
        _close(tensors, frozen)
    torch.testing.assert_close(population.v, voltage, rtol=0, atol=0)
    torch.testing.assert_close(waveform._spike_times, schedule, rtol=0, atol=0)
    torch.testing.assert_close(
        waveform.generator.get_state(), generator, rtol=0, atol=0
    )
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("diagnostic", (False, True))
def test_shared_poisson_children_intra_and_extra_keep_native_invocations_and_aliases(
    diagnostic,
):
    def case():
        generator = torch.Generator().manual_seed(105)
        pulse = _waveform(generator=generator, amp=0.2 * nA)
        expression = pulse + pulse
        population, _, _ = _population("intra", waveform=expression)
        field = torch.linspace(
            -1.0e6, 1.5e6, population.v.numel(), dtype=DTYPE
        ).reshape(population.shape)
        # Distinct outer expressions retain separate stimulus namespaces while
        # their randomized children share parameters, the stream, and carry.
        return population, (field, 0.7 * expression), generator

    source, extra, generator = case()
    native, native_extra, native_generator = case()
    before = generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=diagnostic):
        functional, tensors = dn.func.make_functional(source, dt=DT, extra=extra)
        frozen = _clone(tensors)
        amplitudes = [
            value
            for name, value in tensors.parameters.items()
            if name.endswith("waveform.amp")
        ]
        assert len(amplitudes) >= 2
        assert all(value is amplitudes[0] for value in amplitudes)
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        state = tensors.state
        for _ in range(3):
            packet = functional.sample_stimulation_draw(state)
            state, _ = functional.step(
                tensors.parameters,
                prepared,
                state,
                dn.func.StepInput(stimulation=packet),
            )
            native.step(DT, extra=native_extra)
            expected = dn.func.make_functional(native, dt=DT, extra=native_extra)[
                1
            ].state
            _close(state, expected)
            assert len(state["stimulation"]["rng"]) == 1
            _close(
                next(iter(state["stimulation"]["rng"].values())),
                native_generator.get_state(),
            )
            schedules = tuple(state["stimulation"]["schedules"].values())
            for schedule in schedules[1:]:
                _close(schedule, schedules[0])
        _close(tensors, frozen)
    _close(generator.get_state(), before)
    _close(torch.get_rng_state(), global_rng)


def test_batched_mechanism_invocations_keep_schedule_axes_and_native_gradients():
    def case():
        population, waveform = _case(_RepeatedAssigned)
        population.batch(2)
        population.train()
        population.initialize()
        population.integrator._initialize(
            population, population.t.new_tensor(DT), force=True
        )
        population.detach()
        return population, waveform

    source, waveform = case()
    native, native_waveform = case()
    before = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    tape = functional.sample_stimulation_draw_tape(tensors.state, 3)
    final, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.RolloutInput(stimulation=tape),
        steps=3,
    )
    for _ in range(3):
        native.step(dt=DT)
    assert tuple(final["integrator"]["v"].shape) == (2, 1, 2)
    _close(final, functional.extract(native).state)
    amplitude = next(
        value
        for name, value in tensors.parameters.items()
        if name.endswith("waveform.amp")
    )
    actual = torch.autograd.grad(final["integrator"]["v"].square().mean(), amplitude)[0]
    expected = torch.autograd.grad(
        native.v.square().mean(), native_waveform.waveform.amp
    )[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)
    _close(tensors, frozen)
    _close(waveform.generator.get_state(), before)
    _close(torch.get_rng_state(), global_rng)


def test_distinct_poisson_nodes_in_one_expression_remain_supported():
    waveform = (_waveform(seed=17) + _waveform(seed=19)) * 0.7
    population, _ = _case(_InjectedLeak, waveform=waveform)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    assert len(functional._poisson_plan.entries) == 2
    packet = functional.sample_stimulation_draw(tensors.state)
    functional.step(
        tensors.parameters,
        functional.prepare(tensors.parameters, tensors.constants),
        tensors.state,
        dn.func.StepInput(stimulation=packet),
    )


@pytest.mark.parametrize(
    "mechanism, runtime, initialization",
    ((_AdvanceOnly, True, False), (_InitializationOnly, False, True)),
)
def test_phase_scoped_poisson_consumes_only_its_native_invocations(
    mechanism, runtime, initialization
):
    population, _ = _case(mechanism)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    assert functional._poisson_plan.has_initialization is initialization
    packet = functional.sample_stimulation_draw(tensors.state)
    for stream, previous in tensors.state["stimulation"]["rng"].items():
        assert torch.equal(packet.rng[stream], previous) is not runtime
    initial = functional.sample_stimulation_initialization_draw(tensors.state)
    assert (initial is not None) is initialization


class _ReversedNamed(_NamedInjectedLeak):
    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        current_name = "i_a" if not self._injection_specs else "i_b"
        return self.register_waveform_injection(
            waveform,
            index=index,
            model_shape=model_shape,
            model=model,
            current_name=current_name,
        )

    def assigned_values(self, v, values):
        del values
        current_b = self.injection_values(v, t=self.t, current_name="i_b")
        current_a = self.injection_values(v, t=self.t, current_name="i_a")
        return {"i_a": current_a, "i_b": current_b}


def _reversed_case():
    generator = torch.Generator().manual_seed(71)
    population, _ = _case(_ReversedNamed, waveform=_waveform(generator=generator))
    population[:, 0].inject(_waveform(generator=generator, amp=0.7))
    population.initialize()
    return population


@pytest.mark.parametrize("initialization", (False, True))
def test_shared_generator_follows_authored_current_order(initialization):
    source, native = _reversed_case(), _reversed_case()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    if initialization:
        packet = functional.sample_stimulation_initialization_draw(tensors.state)
        actual = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization._replace(stimulation=packet),
            state=tensors.state,
        ).state
        native.initialize()
    else:
        packet = functional.sample_stimulation_draw(tensors.state)
        actual, _ = functional.step(
            tensors.parameters,
            functional.prepare(tensors.parameters, tensors.constants),
            tensors.state,
            dn.func.StepInput(stimulation=packet),
        )
        native.step(dt=DT)
    torch.testing.assert_close(actual["integrator"]["v"], native.v)
    for entry in functional._poisson_plan.entries:
        waveform = native.get_submodule(entry.parameter.rsplit(".", 1)[0])
        torch.testing.assert_close(
            actual["stimulation"]["schedules"][entry.key],
            waveform._spike_times.reshape(-1),
        )
        torch.testing.assert_close(
            actual["stimulation"]["rng"][entry.stream],
            waveform.generator.get_state(),
            rtol=0,
            atol=0,
        )


class _CustomOuter(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, first, second):
        super().__init__()
        self.first, self.second = first, second

    def fn(self, times):
        return self.second.fn(times) + self.first.fn(times)


class _CustomLeaf(Waveform):
    FUNCTIONAL_PURE = True

    def fn(self, times):
        return torch.exp(-times.square())


def test_custom_outer_poisson_composition_rejects_an_unknown_invocation_topology():
    waveform = _CustomOuter(_waveform(seed=3), _waveform(seed=5))
    population, _ = _case(_InjectedLeak, waveform=waveform)
    with pytest.raises(dn.func.FunctionalizationError, match="[Cc]ustom"):
        dn.func.make_functional(population, dt=DT)


def test_custom_leaf_inside_a_poisson_envelope_remains_supported():
    waveform = _CustomLeaf().poisson(
        interval=0.04,
        n=3,
        randomize_every_call=True,
        generator=torch.Generator().manual_seed(3),
    )
    population, _ = _case(_InjectedLeak, waveform=waveform)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    packet = functional.sample_stimulation_draw(tensors.state)
    functional.step(
        tensors.parameters,
        functional.prepare(tensors.parameters, tensors.constants),
        tensors.state,
        dn.func.StepInput(stimulation=packet),
    )
