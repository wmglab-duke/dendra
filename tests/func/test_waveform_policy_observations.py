"""Pure adapters preserve public Poisson policy observations in authored code."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.stim.waveform.core import Waveform
from dendra.units import nA
from tests.func.test_poisson_stimulation import DT, DTYPE, _clone, _close, _population

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


class _ObservedPolicy(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, child):
        super().__init__()
        self.child = child

    def fn(self, times):
        first = self.child(times)
        if self.child.randomize_every_call:
            return first + 0.25 * self.child(times)
        return first


def _case(role, randomized):
    generator = torch.Generator().manual_seed(337)
    pulse = dn.mono_rect(amp=0.2 * nA if role == "intra" else 0.2, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=randomized,
        generator=generator,
    )
    return _population(role, waveform=_ObservedPolicy(pulse))[0], pulse, generator


@pytest.mark.parametrize("role", ("intra", "mechanism"))
@pytest.mark.parametrize("diagnostic", (False, True))
@pytest.mark.parametrize("randomized", (False, True))
def test_authored_policy_branch_preserves_native_values_draws_and_source_flags(
    role, diagnostic, randomized
):
    source, pulse, generator = _case(role, randomized)
    native, native_pulse, native_generator = _case(role, randomized)
    source_type = type(pulse)
    before_rng = generator.get_state().clone()
    before_schedule = pulse._spike_times.detach().clone()
    global_rng = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=diagnostic):
        functional, tensors = dn.func.make_functional(source, dt=DT)
        frozen = _clone(tensors)
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
            native.step(DT)
            _close(state, functional.extract(native).state)
            if randomized:
                for value in packet.rng.values():
                    _close(value, native_generator.get_state())
            else:
                assert packet is None
            assert native_pulse.randomize_every_call is randomized
        _close(tensors, frozen)
    assert pulse.randomize_every_call is randomized
    assert type(pulse) is source_type
    _close(generator.get_state(), before_rng)
    _close(pulse._spike_times, before_schedule)
    _close(torch.get_rng_state(), global_rng)


def test_policy_branch_is_preserved_inside_fullgraph_atomic_step():
    source, pulse, generator = _case("intra", True)
    native, _, native_generator = _case("intra", True)
    before_rng = generator.get_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = functional.sample_stimulation_draw(tensors.state)
    frozen = _clone(tensors)
    frozen_packet = _clone(packet)

    def atomic(parameters, constants, state, draws):
        return functional.prepare_and_step(
            parameters, constants, state, dn.func.StepInput(stimulation=draws)
        )[0]

    compiled = torch.compile(atomic, backend="aot_eager", fullgraph=True)
    actual = compiled(tensors.parameters, tensors.constants, tensors.state, packet)
    native.step(DT)
    _close(actual, functional.extract(native).state)
    for value in actual["stimulation"]["rng"].values():
        _close(value, native_generator.get_state())
    assert pulse.randomize_every_call is True
    _close(generator.get_state(), before_rng)
    _close(tensors, frozen)
    _close(packet, frozen_packet)
