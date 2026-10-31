"""Native handoff resumes shared RAND/NOISE/SDE and Poisson streams."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_stimulation_rng_sharing import (
    DTYPE,
    _clone,
    _close,
    _lower,
    _network,
    _one_stream,
    _population,
    _step_native,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=DTYPE,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _case(network):
    if network:
        return _network()
    model, extra, pulse = _population()
    assert extra is None
    return model, [pulse]


def _populations(model):
    return model.populations.values() if isinstance(model, dn.Network) else (model,)


def _noise_and_state(model):
    return tuple(
        (
            cell.mech.probe.eta,
            cell.mech.probe.DE._SharedStochasticGate.forcing,
            cell.mech.probe.x,
            cell.mech.probe.offset,
        )
        for cell in _populations(model)
    )


def _assert_borrowed_streams(model, pulses, streams, generators):
    for cell, stream, generator in zip(
        _populations(model), streams, generators, strict=True
    ):
        owner = cell.mech.probe
        gate = owner.DE._SharedStochasticGate
        assert owner.stream is gate.stream is gate.x_dW_rng is stream
        assert stream.rng is generator
    assert all(pulse.generator is generators[0] for pulse in pulses)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_handoff_resumes_shared_stochastic_stream_without_sampling(
    network, monkeypatch
):
    source, source_pulses = _case(network)
    target, target_pulses = _case(network)
    oracle, oracle_pulses = _case(network)
    functional, tensors = _lower(source)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    source_rng = source_pulses[0].generator.get_state().clone()
    _, auxiliary = dn.func.run(functional, initial, steps=2)
    packet = auxiliary["continuation"]
    frozen = _clone((packet.state, packet.parameters, packet.constants))
    assert not torch.equal(_one_stream(packet.state), source_rng)
    for _ in range(2):
        _step_native(oracle)
    _close(functional.extract(oracle).state, packet.state)

    parameters = dict(target.named_parameters(remove_duplicate=False))
    streams = tuple(cell.mech.probe.stream for cell in _populations(target))
    generators = tuple(stream.rng for stream in streams)
    global_rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Native continuation handoff sampled random values")

    # All three stochastic recipes and the randomized Poisson child use these
    # primitives. Restoration may set generator bytes but must not draw values.
    with monkeypatch.context() as guard:
        for name in ("rand", "randn", "rand_like", "randn_like"):
            guard.setattr(torch, name, forbidden)
        assert dn.func.commit_continuation_(functional, target, packet) is target

    _assert_borrowed_streams(target, target_pulses, streams, generators)
    _close(generators[0].get_state(), _one_stream(packet.state))
    _close(functional.extract(target).state, packet.state)
    _close(source_pulses[0].generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)
    current_parameters = dict(target.named_parameters(remove_duplicate=False))
    assert current_parameters.keys() == parameters.keys()
    assert all(current_parameters[name] is value for name, value in parameters.items())

    continuation = packet
    for _ in range(3):
        expected, auxiliary = dn.func.run(functional, continuation, steps=1)
        continuation = auxiliary["continuation"]
        _step_native(target)
        _step_native(oracle)
        _close(functional.extract(target).state, expected)
        _close(functional.extract(oracle).state, expected)
        _close(_noise_and_state(target), _noise_and_state(oracle))
        _close(generators[0].get_state(), _one_stream(expected))
        _close(
            target_pulses[0].generator.get_state(),
            oracle_pulses[0].generator.get_state(),
        )
        _assert_borrowed_streams(target, target_pulses, streams, generators)

    _close((packet.state, packet.parameters, packet.constants), frozen)
    _close(source_pulses[0].generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)
