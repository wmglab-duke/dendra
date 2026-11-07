"""Invocation metadata follows the frozen, optional purity diagnostic policy."""

import pytest
import torch

import dendra as dn
from dendra.func import _population as functional_population
from dendra.models.stim.waveform.core import Waveform
from dendra.units import nA
from tests.func.test_poisson_stimulation import DT, DTYPE, _clone, _close, _population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with dn.ctx(
            JIT=0,
            REQUIRE_GRAD=1,
            IMEM=False,
            DTYPE=DTYPE,
            DEVICE="cpu",
            CHECK_PURITY=False,
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _OpaqueLeaf(Waveform):
    FUNCTIONAL_PURE = True
    TOKEN = object()
    Waveform.PARAMETER(amp=0.2)

    def fn(self, times):
        return self.amp * torch.exp(-times.square())


class _OpaqueContainer(Waveform):
    FUNCTIONAL_PURE = True
    TOKEN = object()

    def __init__(self, child):
        super().__init__()
        self.child = child

    def fn(self, times):
        return self.child(times) + 0.25 * self.child.fn(times)


def _case(role, recipe):
    generator = torch.Generator().manual_seed(311)
    amplitude = 0.2 * nA if role == "intra" else 0.2
    leaf = (
        _OpaqueLeaf(amp=amplitude)
        if recipe == "leaf"
        else dn.mono_rect(amp=amplitude, pw=0.045)
    )
    pulse = leaf.poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    waveform = pulse if recipe == "leaf" else _OpaqueContainer(pulse)
    return _population(role, waveform=waveform)[0], pulse, generator


@pytest.mark.parametrize("role", ("intra", "mechanism"))
@pytest.mark.parametrize("recipe", ("leaf", "container"))
def test_trusted_profiles_accept_opaque_metadata_without_running_diagnostic_snapshots(
    monkeypatch, role, recipe
):
    source, pulse, generator = _case(role, recipe)
    native, _, _ = _case(role, recipe)
    metadata = (_OpaqueLeaf.TOKEN, _OpaqueContainer.TOKEN)
    before_rng = generator.get_state().clone()
    before_schedule = pulse._spike_times.detach().clone()
    global_rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        del args, kwargs
        pytest.fail("Trusted invocation profiling must not run diagnostic snapshots")

    monkeypatch.setattr(functional_population, "_class_state_snapshot", forbidden)
    monkeypatch.setattr(functional_population, "_global_rng_snapshot", forbidden)
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
    _close(tensors, frozen)
    _close(generator.get_state(), before_rng)
    _close(pulse._spike_times, before_schedule)
    _close(torch.get_rng_state(), global_rng)
    assert _OpaqueLeaf.TOKEN is metadata[0]
    assert _OpaqueContainer.TOKEN is metadata[1]


@pytest.mark.parametrize("role", ("intra", "mechanism"))
@pytest.mark.parametrize("recipe", ("leaf", "container"))
def test_enabled_diagnostics_still_reject_unauditable_waveform_class_metadata(
    role, recipe
):
    source, pulse, generator = _case(role, recipe)
    metadata = (_OpaqueLeaf.TOKEN, _OpaqueContainer.TOKEN)
    before_rng = generator.get_state().clone()
    before_schedule = pulse._spike_times.detach().clone()
    global_rng = torch.get_rng_state().clone()
    with (
        dn.ctx(CHECK_PURITY=True),
        pytest.raises(dn.func.FunctionalizationError) as error,
    ):
        dn.func.make_functional(source, dt=DT)
    assert "TOKEN" in str(error.value)
    _close(generator.get_state(), before_rng)
    _close(pulse._spike_times, before_schedule)
    _close(torch.get_rng_state(), global_rng)
    assert _OpaqueLeaf.TOKEN is metadata[0]
    assert _OpaqueContainer.TOKEN is metadata[1]
