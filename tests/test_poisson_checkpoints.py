"""Owned stochastic waveform schedules and physical RNG aliases checkpoint together."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas
from dendra.models.stim.waveform._poisson import (
    defer_poisson_generator_restore,
    poisson_generator_state,
    poisson_state_dict_for_checkpoint,
)
from dendra.units import nA
from tests.func.test_mechanism_injection_execution import _InjectedLeak

DTYPE = torch.float64
DT = 0.025
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    state = torch.get_rng_state().clone()
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=DTYPE, IMEM=False):
            yield
    finally:
        torch.set_rng_state(state)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _waveform(generator):
    return dn.mono_rect(amp=0.3 * nA, pw=0.07).poisson(
        interval=0.08,
        n=9,
        start=-0.03,
        off=0.31,
        noise=0.8,
        randomize_every_call=True,
        generator=generator,
    )


def _population(*, mechanism=False, shared=True, generator=None):
    population = dn.Unmyelinated(
        [2.0, 2.5],
        L=4.0,
        dx=1.0,
        v_init=-65.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    population.insert(_InjectedLeak.rename("owner") if mechanism else pas)
    generator = torch.Generator().manual_seed(713) if generator is None else generator
    first = _waveform(generator)
    second = _waveform(generator if shared else torch.Generator().manual_seed(151))
    population[:, 1].inject(first)
    population[:, 2].inject(second)
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    return population, (first, second)


@pytest.mark.parametrize("mechanism", (False, True))
@pytest.mark.parametrize("shared", (False, True))
def test_population_checkpoint_replays_stimulation_voltage_and_owned_schedules(
    mechanism, shared
):
    population, waveforms = _population(mechanism=mechanism, shared=shared)
    population.step(DT)
    saved = _clone(population.state_dict_for_checkpoint())
    saved_schedules = tuple(waveform._spike_times.clone() for waveform in waveforms)
    saved_rng = tuple(waveform.generator.get_state().clone() for waveform in waveforms)
    checkpoint = saved["stochastic"]["poisson"]
    assert len(checkpoint["modules"]) == 2
    assert len(checkpoint["streams"]) == (1 if shared else 2)
    assert not any(stream["pending"] for stream in checkpoint["streams"])
    expected = []
    for _ in range(3):
        population.step(DT)
        expected.append(population.v.detach().clone())
    population.restore_dict_from_checkpoint(saved)
    for waveform, schedule, rng in zip(waveforms, saved_schedules, saved_rng):
        torch.testing.assert_close(waveform._spike_times, schedule, rtol=0, atol=0)
        assert torch.equal(waveform.generator.get_state(), rng)
    for value in expected:
        population.step(DT)
        torch.testing.assert_close(population.v, value, rtol=0, atol=0)
    assert (waveforms[0].generator is waveforms[1].generator) is shared


def test_checkpoint_rejects_changed_rng_aliases_without_changing_either_stream():
    source, _ = _population(shared=True)
    target, waveforms = _population(shared=False)
    before = tuple(waveform.generator.get_state().clone() for waveform in waveforms)
    with pytest.raises(ValueError, match="topology"):
        target.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    for waveform, state in zip(waveforms, before):
        assert torch.equal(waveform.generator.get_state(), state)


def test_poisson_restore_preflights_all_payloads_before_applying_schedules_or_rng():
    population, waveforms = _population(shared=False)
    saved = _clone(population.state_dict_for_checkpoint())
    before_schedules = tuple(waveform._spike_times.clone() for waveform in waveforms)
    before_rng = tuple(waveform.generator.get_state().clone() for waveform in waveforms)
    poisson = saved["stochastic"]["poisson"]
    first = next(iter(poisson["modules"].values()))
    first["schedule"] = torch.full_like(first["schedule"], torch.inf)
    poisson["streams"][0]["state"] = torch.Generator().manual_seed(999).get_state()
    poisson["streams"][1]["state"] = torch.empty((0,), dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        population.restore_dict_from_checkpoint(saved)
    for waveform, schedule, rng in zip(waveforms, before_schedules, before_rng):
        torch.testing.assert_close(waveform._spike_times, schedule, rtol=0, atol=0)
        assert torch.equal(waveform.generator.get_state(), rng)


def test_legacy_checkpoint_without_poisson_section_retains_current_waveform_state():
    population, waveforms = _population()
    saved = _clone(population.state_dict_for_checkpoint())
    del saved["stochastic"]["poisson"]
    population.step(DT)
    before_schedules = tuple(waveform._spike_times.clone() for waveform in waveforms)
    before_rng = tuple(waveform.generator.get_state().clone() for waveform in waveforms)
    population.restore_dict_from_checkpoint(saved)
    for waveform, schedule, rng in zip(waveforms, before_schedules, before_rng):
        torch.testing.assert_close(waveform._spike_times, schedule, rtol=0, atol=0)
        assert torch.equal(waveform.generator.get_state(), rng)


def test_deferred_global_rng_frame_is_consumed_once_across_waveform_aliases():
    first, second = (
        _waveform(torch.default_generator),
        _waveform(torch.default_generator),
    )
    frame = torch.Generator().manual_seed(2468).get_state()
    ambient = torch.get_rng_state().clone()
    defer_poisson_generator_restore((first, second), frame)
    assert torch.equal(torch.get_rng_state(), ambient)
    assert torch.equal(poisson_generator_state(first), frame)
    generator = torch.Generator()
    generator.set_state(frame)
    expected_first, expected_second = _waveform(generator), _waveform(generator)
    # Construction samples once, matching the first regeneration after commit.
    first.regenerate_schedule_()
    second.regenerate_schedule_()
    torch.testing.assert_close(
        first._spike_times, expected_first._spike_times, rtol=0, atol=0
    )
    torch.testing.assert_close(
        second._spike_times, expected_second._spike_times, rtol=0, atol=0
    )
    assert torch.equal(torch.get_rng_state(), generator.get_state())
    assert first._functional_poisson_rng is second._functional_poisson_rng
    assert first._functional_poisson_rng.state is None


def test_pending_global_frame_checkpoint_round_trip_keeps_ambient_rng_untouched():
    population, waveforms = _population(generator=torch.default_generator)
    frame = torch.Generator().manual_seed(2468).get_state()
    defer_poisson_generator_restore(waveforms, frame)
    saved = _clone(population.state_dict_for_checkpoint())
    assert saved["stochastic"]["poisson"]["streams"][0]["pending"]
    torch.rand((7,))
    ambient = torch.get_rng_state().clone()
    population.restore_dict_from_checkpoint(saved)
    assert torch.equal(torch.get_rng_state(), ambient)
    assert torch.equal(poisson_generator_state(waveforms[0]), frame)
    assert waveforms[0]._functional_poisson_rng is waveforms[1]._functional_poisson_rng


def test_network_restore_failure_rolls_back_actual_global_rng_and_shared_pending_frame():
    left, left_waveforms = _population(generator=torch.default_generator)
    right, right_waveforms = _population(generator=torch.default_generator)
    network = dn.Network({"left": left, "right": right})
    network.train()
    network.initialize(DT)
    waveforms = left_waveforms + right_waveforms
    frame = torch.Generator().manual_seed(2468).get_state()
    defer_poisson_generator_restore(waveforms, frame)
    original_holder = waveforms[0]._functional_poisson_rng
    saved = _clone(network.state_dict_for_checkpoint())
    left_stream = saved["populations"]["left"]["stochastic"]["poisson"]["streams"][0]
    left_stream["pending"] = False
    left_stream["state"] = torch.Generator().manual_seed(999).get_state()
    saved["populations"]["right"]["stochastic"]["poisson"]["streams"][0]["state"] = (
        torch.empty((0,), dtype=torch.uint8)
    )
    ambient = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        network.restore_dict_from_checkpoint(saved)
    assert torch.equal(torch.get_rng_state(), ambient)
    for waveform in waveforms:
        assert waveform._functional_poisson_rng is original_holder
        assert torch.equal(poisson_generator_state(waveform), frame)
    assert len(poisson_state_dict_for_checkpoint(network)["streams"]) == 1


def test_network_checkpoint_merges_pending_holders_across_populations():
    generator = torch.Generator().manual_seed(842)
    left, left_waveforms = _population(generator=generator)
    right, right_waveforms = _population(generator=generator)
    network = dn.Network({"left": left, "right": right})
    network.train()
    network.initialize(DT)
    frame = torch.Generator().manual_seed(2468).get_state()
    # Population commits can produce separate holders for the same generator.
    defer_poisson_generator_restore(left_waveforms, frame)
    defer_poisson_generator_restore(right_waveforms, frame)
    checkpoint = _clone(network.state_dict_for_checkpoint())
    network.restore_dict_from_checkpoint(checkpoint)
    waveforms = left_waveforms + right_waveforms
    assert all(
        waveform._functional_poisson_rng is waveforms[0]._functional_poisson_rng
        for waveform in waveforms
    )
    expected_generator = torch.Generator()
    expected_generator.set_state(frame)
    for waveform in waveforms:
        expected = _waveform(expected_generator)
        waveform.regenerate_schedule_()
        torch.testing.assert_close(
            waveform._spike_times.reshape(-1), expected._spike_times, rtol=0, atol=0
        )
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_network_checkpoint_rejects_inconsistent_shared_stream_frames_before_publish():
    generator = torch.Generator().manual_seed(842)
    left, left_waveforms = _population(generator=generator)
    right, right_waveforms = _population(generator=generator)
    network = dn.Network({"left": left, "right": right})
    network.train()
    network.initialize(DT)
    checkpoint = _clone(network.state_dict_for_checkpoint())
    before_rng = generator.get_state().clone()
    before_schedules = tuple(
        waveform._spike_times.clone() for waveform in left_waveforms + right_waveforms
    )
    checkpoint["populations"]["right"]["stochastic"]["poisson"]["streams"][0][
        "state"
    ] = torch.Generator().manual_seed(999).get_state()
    with pytest.raises(ValueError, match="frames differ"):
        network.restore_dict_from_checkpoint(checkpoint)
    assert torch.equal(generator.get_state(), before_rng)
    for waveform, expected in zip(left_waveforms + right_waveforms, before_schedules):
        torch.testing.assert_close(waveform._spike_times, expected, rtol=0, atol=0)


def test_network_checkpoint_rejects_conflicting_shared_waveform_schedules_before_publish():
    generator = torch.Generator().manual_seed(842)
    left, left_waveforms = _population(generator=generator)
    right, _ = _population(generator=generator)
    # Reuse the same two physical waveforms through both Population registries.
    right.delete_injections()
    right[:, 1].inject(left_waveforms[0])
    right[:, 2].inject(left_waveforms[1])
    network = dn.Network({"left": left, "right": right})
    network.train()
    network.initialize(DT)
    checkpoint = _clone(network.state_dict_for_checkpoint())
    before = tuple(waveform._spike_times.clone() for waveform in left_waveforms)
    before_rng = generator.get_state().clone()
    modules = checkpoint["populations"]["left"]["stochastic"]["poisson"]["modules"]
    next(iter(modules.values()))["schedule"][0] -= 0.001
    with pytest.raises(ValueError, match="waveform schedules differ"):
        network.restore_dict_from_checkpoint(checkpoint)
    for waveform, expected in zip(left_waveforms, before):
        torch.testing.assert_close(waveform._spike_times, expected, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), before_rng)
