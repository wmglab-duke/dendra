"""Native Network steps replay physical RNG boundaries and NetStim draws."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_network import DT, _stim_network
from tests.func.test_network_rng_sharing import PREFIX, RAW, _assert_aliases
from tests.func.test_network_rng_sharing import _network as _shared_network
from tests.func.test_network_runtime_noise import _network as _noise_network
from tests.func.test_stimulation_rng_sharing import _network as _poisson_network
from tests.test_native_runtime_draws import (
    _base_draws,
    _changed_draws,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import (
    _forbid_owned_sampling,
    _parameters,
    _same_parameters,
)

pytestmark = pytest.mark.cpu


def _state_loss(state):
    return sum(
        local["integrator"]["v"].square().mean()
        for local in state["populations"].values()
    )


def _native_loss(model):
    return sum(cell.v.square().mean() for cell in model.populations.values())


def test_shared_network_draws_preserve_live_gradients_and_each_population_rng_boundary(
    monkeypatch,
):
    source, native = _shared_network(), _shared_network()
    functional, tensors = dn.func.make_functional_network(source)
    initialized = dn.func.initialize(functional, tensors)
    native.initialize(DT)
    _close(functional.extract(native).state, initialized.state)
    packet = _changed_draws(functional.sample_runtime_draws(initialized.state))
    saved_packet, parameters = _snapshot(packet), _parameters(native)
    global_rng = torch.get_rng_state().clone()
    streams = (native.left.mech.probe.primary, native.left.mech.probe.secondary)
    generators = tuple(stream.rng for stream in streams)
    boundaries = packet.rng["model"]["stream_0"]["cpu"]
    assert set(boundaries) == {"left", "right"}
    assert not torch.equal(boundaries["left"], boundaries["right"])
    observed = []
    right = native.right.mech.probe
    current = type(right).i_affine_terms

    def observe(self, voltage):
        if self is right:
            observed.append(generators[0].get_state().clone())
        return current(self, voltage)

    expected, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state, draws=packet
    )
    with monkeypatch.context() as guard:
        guard.setattr(
            type(right),
            "i_affine_terms",
            observe,
        )
        _forbid_owned_sampling(guard, native)
        assert dn.step_network(native, draws=packet) is native
        assert observed
        assert all(torch.equal(value, boundaries["left"]) for value in observed)
        _close(generators[0].get_state(), boundaries["right"])
    _close(functional.extract(native).state, expected)
    _assert_aliases(native)
    assert (native.left.mech.probe.primary, native.left.mech.probe.secondary) == streams
    assert all(
        stream.rng is generator
        for stream, generator in zip(streams, generators, strict=True)
    )
    names = (RAW, f"{PREFIX}DE._SharedGate.sigma_param")
    targets = (
        native.left.mech.probe.g_mu_param,
        native.left.mech.probe.DE._SharedGate.sigma_param,
    )
    references = tuple(
        initialized.parameters["populations"]["left"][name] for name in names
    )
    actual = torch.autograd.grad(
        _native_loss(native), (*targets, *_base_draws(packet)), allow_unused=True
    )
    wanted = torch.autograd.grad(
        _state_loss(expected), (*references, *_base_draws(packet)), allow_unused=True
    )
    _close(actual[:2], wanted[:2])
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual[:2])
    assert all(value is None or value.count_nonzero() == 0 for value in actual[2:])
    _same_parameters(native, parameters)
    _assert_packet_equal(packet, saved_packet)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize(
    "supplied_exp", (None, -0.25), ids=("changed_sample", "negative")
)
def test_network_netstim_draws_detach_bases_and_replay_the_native_clock_gradients(
    monkeypatch,
    supplied_exp,
):
    source, native, automatic = _stim_network(), _stim_network(), _stim_network()
    with torch.no_grad():
        source.netstim.next_stoch_time.zero_()
    for model in (native, automatic):
        model.restore_dict_from_checkpoint(
            _snapshot(source.state_dict_for_checkpoint())
        )
    functional, tensors = dn.func.make_functional_network(source)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
    if supplied_exp is not None:
        packet = packet._replace(
            draws={
                **packet.draws,
                "netstim": torch.full_like(
                    packet.draws["netstim"], supplied_exp
                ).requires_grad_(),
            }
        )
    saved_packet, parameters = _snapshot(packet), _parameters(native)
    generator, seeder = native.netstim._rng, native.netstim._seeder
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, generator)
        assert dn.step(native, draws=packet) is native
    automatic.step()
    _close(functional.extract(native).state, expected)
    assert not torch.equal(
        native.netstim.next_stoch_time, automatic.netstim.next_stoch_time
    )
    _close(generator.get_state(), packet.rng["netstim"])
    assert native.netstim._rng is generator and native.netstim._seeder is seeder
    native_interval = native.netstim.interval.rho
    reference_interval = tensors.parameters["netstim"]["parameters"]["interval.rho"]
    actual = torch.autograd.grad(
        native.netstim.next_stoch_time.sum(),
        (native_interval, *_base_draws(packet)),
        allow_unused=True,
    )
    wanted = torch.autograd.grad(
        expected["netstim"]["next_stoch_time"].sum(), reference_interval
    )[0]
    _close(actual[0], wanted)
    assert actual[0].isfinite().all() and actual[0].abs().sum() > 0
    assert all(value is None or value.count_nonzero() == 0 for value in actual[1:])
    _same_parameters(native, parameters)
    _assert_packet_equal(packet, saved_packet)


def test_network_explicit_then_automatic_draws_survive_native_checkpoint_continuation():
    source, native, restored = (_noise_network(stimulus=True) for _ in range(3))
    for model in (native, restored):
        model.restore_dict_from_checkpoint(
            _snapshot(source.state_dict_for_checkpoint())
        )
    functional, tensors = dn.func.make_functional_network(source)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    native.step(draws=packet)
    for index in range(2):
        expected, _ = dn.func.run(
            functional,
            dn.func.Continuation(expected, tensors.parameters, tensors.constants),
            steps=1,
        )
        if index == 0:
            native.step()
            restored.restore_dict_from_checkpoint(
                _snapshot(native.state_dict_for_checkpoint())
            )
        else:
            dn.step_network(restored)
    _close(functional.extract(restored).state, expected)
    assert restored.t.item() == pytest.approx(3 * DT)


@pytest.mark.parametrize("invalid", ("populations", "boundary", "netstim", "tape"))
def test_network_runtime_draw_rejection_keeps_clock_values_rng_and_parameters(
    invalid, monkeypatch
):
    model = _noise_network(stimulus=True)
    functional, tensors = dn.func.make_functional_network(model)
    packet = functional.sample_runtime_draws(tensors.state)
    if invalid == "populations":
        packet = packet._replace(draws={**packet.draws, "populations": {}})
    elif invalid == "boundary":
        rng = _snapshot(packet.rng)
        frame = next(iter(rng["model"].values()))
        leaf = next(name for name in frame if name not in {"seed", "negative"})
        frame[leaf] = {"unknown": next(iter(frame[leaf].values()))}
        packet = packet._replace(rng=rng)
    elif invalid == "netstim":
        packet = packet._replace(draws={**packet.draws, "netstim": None})
    else:
        packet = functional.sample_runtime_draws(tensors.state, steps=1)
    saved, parameters, global_rng = (
        _snapshot(model.state_dict_for_checkpoint()),
        _parameters(model),
        torch.get_rng_state().clone(),
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, model.netstim._rng)
        with pytest.raises((TypeError, ValueError, dn.func.FunctionalizationError)):
            model.step(draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _same_parameters(model, parameters)
    _close(torch.get_rng_state(), global_rng)


def test_network_randomized_poisson_rejects_before_sampling_and_clock_advancement(
    monkeypatch,
):
    model, pulses = _poisson_network()
    functional, tensors = dn.func.make_functional_network(model)
    packet = functional.sample_runtime_draws(tensors.state)
    saved = _snapshot(model.state_dict_for_checkpoint())
    states = tuple(pulse.generator.get_state().clone() for pulse in pulses)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported replay evaluated a Poisson waveform")

    for pulse in pulses:
        monkeypatch.setattr(pulse, "forward", forbidden)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="Poisson|poisson"
    ):
        model.step(draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    for pulse, state in zip(pulses, states, strict=True):
        _close(pulse.generator.get_state(), state)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_native_network_draws_keep_shared_streams_and_float32_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        source, native = _shared_network(), _shared_network()
        functional, tensors = dn.func.make_functional_network(source)
        packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
        saved = _snapshot(packet)
        stream = native.left.mech.probe.primary
        generators = {
            "cpu": stream._cpu_gen,
            **{
                str(device): generator
                for device, generator in stream._device_gens.items()
            },
        }
        expected, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )
        dn.step_network(native, draws=packet)
        for name, population in native.populations.items():
            torch.testing.assert_close(
                population.v,
                expected["populations"][name]["integrator"]["v"],
                rtol=2e-5,
                atol=2e-6,
            )
        sigma = native.left.mech.probe.DE._SharedGate.sigma_param
        reference = tensors.parameters["populations"]["left"][
            f"{PREFIX}DE._SharedGate.sigma_param"
        ]
        actual = torch.autograd.grad(_native_loss(native), sigma)[0]
        wanted = torch.autograd.grad(_state_loss(expected), reference)[0]
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)
        assert actual.isfinite().all() and actual.abs().sum() > 0
        _assert_aliases(native)
        frames = packet.rng["model"]["stream_0"]
        states = stream.get_extra_state()["rng_state"]
        assert set(states) == set(frames) - {"seed", "negative"}
        for device, state in states.items():
            _close(state, frames[device]["right"].cpu())
        assert stream._cpu_gen is generators["cpu"]
        assert set(stream._device_gens) == {
            torch.device(device) for device in generators if device != "cpu"
        }
        assert all(
            generator is generators[str(device)]
            for device, generator in stream._device_gens.items()
        )
        _assert_packet_equal(packet, saved)
