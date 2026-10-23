"""Steady-state restoration publishes physical Poisson buffers to every alias."""

import pytest
import torch

import dendra as dn
from tests.func.test_poisson_invocation_guards import _AdvanceOnly, _case
from tests.func.test_poisson_stimulation import (
    DT,
    DTYPE,
    _close,
    _population,
    _waveform,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        yield


def test_population_cache_restores_shared_extra_and_mechanism_schedule():
    source, waveform, _ = _population("mechanism")
    source.step(DT)
    source.cache("_steady_state")
    cached = waveform._spike_times.detach().clone().reshape(-1)
    source.step(DT)
    original_schedule = waveform._spike_times.detach().clone()
    original_rng = waveform.generator.get_state().clone()
    assert not torch.equal(cached, original_schedule.reshape(-1))
    functional, tensors = dn.func.make_functional(
        source, dt=DT, extra=(torch.ones_like(source.v), waveform)
    )

    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )

    functional._poisson_plan.validate_state(initialized.state["stimulation"])
    assert {entry.kind for entry in functional._poisson_plan.entries} == {
        "extra",
        "mechanism",
    }
    for value in initialized.state["stimulation"]["schedules"].values():
        _close(value, cached)
    _close(waveform._spike_times, original_schedule)
    _close(waveform.generator.get_state(), original_rng)


@pytest.mark.parametrize("last_cached", (False, True))
def test_network_cache_schedule_order_includes_inactive_initialization(last_cached):
    def make():
        waveform = _waveform()
        populations = {
            name: _case(_AdvanceOnly, waveform=waveform)[0]
            for name in ("left", "right")
        }
        network = dn.Network(populations)
        network.train()
        network.initialize(DT)
        network.step()
        populations["left"].cache("_steady_state")
        network.step()
        if last_cached:
            populations["right"].cache("_steady_state")
        network.step()
        return network, waveform

    source, waveform = make()
    native, native_waveform = make()
    original_rng = waveform.generator.get_state().clone()
    functional, tensors = dn.func.make_functional_network(source)
    assert all(
        not plan._poisson_plan.has_initialization
        for plan in functional.populations.values()
    )
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        functional.initialization_inputs,
    )

    native.initialize(DT)
    _close(initialized.state, functional.extract(native).state)
    functional._stimulation_plan.validate_state(initialized.state)
    for population_state in initialized.state["populations"].values():
        for value in population_state["stimulation"]["schedules"].values():
            _close(value, native_waveform._spike_times.reshape(-1))
    for value in initialized.state["stimulation_rng"].values():
        _close(value, original_rng)
    _close(waveform.generator.get_state(), original_rng)
