"""Poisson packets must match each mechanism's numerical invocation profile."""

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.stim.waveform.core import Waveform
from tests.func.test_mechanism_injection_execution import (
    _InjectedLeak,
    _NamedInjectedLeak,
)
from tests.func.test_poisson_stimulation import DT, DTYPE, _waveform

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
def test_repeated_poisson_evaluation_rejects_without_mutating_source(
    mechanism, diagnostic
):
    population, waveform = _case(mechanism)
    generator = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    voltage = population.v.detach().clone()
    schedule = waveform._spike_times.detach().clone()
    with (
        dn.ctx(CHECK_PURITY=diagnostic),
        pytest.raises(
            dn.func.FunctionalizationError, match="Multiple evaluations.*Poisson"
        ),
    ):
        dn.func.make_functional(population, dt=DT)
    torch.testing.assert_close(population.v, voltage, rtol=0, atol=0)
    torch.testing.assert_close(waveform._spike_times, schedule, rtol=0, atol=0)
    torch.testing.assert_close(
        waveform.generator.get_state(), generator, rtol=0, atol=0
    )
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


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
