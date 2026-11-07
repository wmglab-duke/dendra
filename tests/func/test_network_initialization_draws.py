"""Canonical initialization draws retain Network phase and RNG ownership."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close as _mapping_close
from tests.func.test_continuation import _snapshot
from tests.func.test_netstim_parameter_initialization import _noise_cache_network
from tests.func.test_network import DT, DTYPE, _close
from tests.func.test_network_connection_sampling import (
    _check_native,
    _materialized_loss,
    _network_with_population_source_phase,
)
from tests.func.test_network_initialization_rng import _shared_noise_network
from tests.func.test_stimulation_rng_sharing import (
    _initialize_native,
    _lower,
)
from tests.func.test_stimulation_rng_sharing import _network as _shared_network
from tests.func.test_stimulation_rng_sharing import (
    _one_stream,
)
from tests.func.test_stochastic_initialization_ordering import (
    RAW,
    _native_close,
)
from tests.func.test_stochastic_initialization_ordering import (
    _network as _ordered_network,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)


def _capture(functional, tensors, inputs=None):
    captured = {}
    initialized = functional._initialize_host(
        *tensors,
        functional.initialization_inputs if inputs is None else inputs,
        _draws_capture=captured,
    )
    return initialized, dn.func.InitializationDraws(**captured)


def _legacy(inputs, draws):
    return inputs._replace(
        populations={
            name: packet._replace(
                random_draw=None if draws.random is None else draws.random[name],
                stimulation=None
                if draws.stimulation is None
                else draws.stimulation[name],
            )
            for name, packet in inputs.populations.items()
        },
        netstim_draw=draws.netstim,
        connection_draw=draws.connections,
    )


@pytest.mark.parametrize("reseed", (False, True))
def test_shared_rand_poisson_capture_and_replay_preserve_native_order_and_rng(reseed):
    source, pulses = _shared_network(reseed=reseed)
    native, _ = _shared_network(reseed=reseed)
    functional, tensors = _lower(source)
    frozen = _snapshot((tensors, functional.initialization_inputs))
    source_rng = [pulse.generator.get_state().clone() for pulse in pulses]
    global_rng = torch.get_rng_state().clone()
    initialized, captured = _capture(functional, tensors)
    sampled = dn.func.sample_initialization_draws(functional, tensors)
    _assert_packet_equal(sampled, captured)
    assert set(sampled.random) == set(sampled.stimulation) == {"left", "right"}
    assert all(value is not None for value in sampled.random.values())
    assert all(value is not None for value in sampled.stimulation.values())
    assert sampled.netstim is sampled.connections is None
    replay = functional.initialize(
        *tensors, functional.initialization_inputs, draws=sampled
    )
    legacy = functional.initialize(
        *tensors, _legacy(functional.initialization_inputs, sampled)
    )
    host = functional._initialize_host(
        *tensors, functional.initialization_inputs, draws=sampled
    )
    _close(replay, initialized)
    _close(legacy, initialized)
    _close(host, initialized)
    _close((tensors, functional.initialization_inputs), frozen)
    _close(torch.get_rng_state(), global_rng)
    for pulse, before in zip(pulses, source_rng, strict=True):
        _close(pulse.generator.get_state(), before)
    _initialize_native(native)
    _close(initialized.state, functional.extract(native).state)
    _one_stream(initialized.state)
    draws = functional.sample_runtime_draws(initialized.state, steps=2)
    final, _ = dn.func.run(
        functional,
        dn.func.Continuation(
            initialized.state, initialized.parameters, initialized.constants
        ),
        steps=2,
        draws=draws,
    )
    for _ in range(2):
        native.step()
    _close(final, functional.extract(native).state)


@pytest.mark.parametrize("cache_kind", (None, "population", "network"))
def test_ordered_population_capture_retains_aliases_gradients_and_cache_consumption(
    cache_kind,
):
    model = _ordered_network(cache_kind=cache_kind)
    native = _ordered_network(cache_kind=cache_kind)
    functional, tensors = dn.func.make_functional_network(model)
    initialized, draws = _capture(functional, tensors)
    replay = functional.initialize(
        *tensors, functional.initialization_inputs, draws=draws
    )
    _close(replay, initialized)
    native.initialize(DT)
    _native_close(replay.state, functional.extract(native).state)
    raw = tensors.parameters["populations"]["left"][RAW]
    assert (
        replay.parameters["populations"]["left"][RAW]
        is replay.parameters["populations"]["right"][RAW]
    )
    for name, plan in functional.populations.items():
        cached = (
            functional.initialization_inputs.cache is not None
            or plan._owned_cache_bound
        )
        assert (draws.random[name] is None) is cached
        assert draws.stimulation[name] is None
    if cache_kind != "network":
        path = "integrator.mech.mechanisms.leak.g"

        def materialized_loss(packet):
            # Saved realizations are detached checkpoint metadata. Preparation
            # reconnects each sample to its retained distribution-parameter frame.
            return sum(
                plan.prepare(
                    packet.parameters["populations"][name],
                    packet.constants["populations"][name],
                )
                .values["mechanisms"][path]
                .sum()
                for name, plan in functional.populations.items()
            )

        actual = torch.autograd.grad(
            materialized_loss(replay),
            raw,
            retain_graph=True,
        )[0]
        expected = torch.autograd.grad(materialized_loss(initialized), raw)[0]
        _close(actual, expected)
        _close(actual, torch.full_like(raw, 1.5 + 3.0))
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0


@pytest.mark.parametrize("ignore", (False, True))
def test_network_cache_capture_consumes_no_population_draws_and_retains_rng_load_policy(
    ignore,
):
    def build():
        model, _ = _shared_network(retained=True)
        for _ in range(2):
            model.step()
        for cell in model.populations.values():
            cell.mech.probe.stream.ignore_rng_on_load(ignore)
        model.cache_state()
        for _ in range(2):
            model.step()
        return model

    source, native = build(), build()
    functional, tensors = _lower(source)
    initialized, draws = _capture(functional, tensors)
    assert draws.random == draws.stimulation == {"left": None, "right": None}
    replay = functional.initialize(
        *tensors, functional.initialization_inputs, draws=draws
    )
    _close(replay, initialized)
    _initialize_native(native)
    # Native extraction and cache restoration publish equivalent mappings in
    # different insertion orders; execution/replay ordering is checked above.
    _mapping_close(replay.state, functional.extract(native).state)
    _one_stream(replay.state)


@pytest.mark.parametrize("initial_noise,increment", ((0.0, 0.6), (0.6, -0.6)))
def test_netstim_capture_observes_conditional_noise_after_population_transform(
    initial_noise, increment
):
    source, _ = _shared_noise_network(
        seed=812, initial_noise=initial_noise, increment=increment
    )
    native, _ = _shared_noise_network(
        seed=812, initial_noise=initial_noise, increment=increment
    )
    native.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(source)
    captured = dn.func.sample_initialization_draws(functional, tensors)
    assert captured.netstim is not None
    if initial_noise + increment == 0:
        assert not captured.netstim.exp_rand.any()
    else:
        assert (captured.netstim.exp_rand > 0).all()
    initial = dn.func.initialize(functional, tensors, draws=captured)
    native.initialize(DT)
    _close(initial.state, functional.extract(native).state)
    _close(initial.state["netstim"]["rng_state"], captured.netstim.rng_state)


@pytest.mark.parametrize("cached_noise", (0.0, 0.6))
def test_netstim_capture_follows_population_cache_restored_transform_buffers(
    cached_noise,
):
    source = _noise_cache_network("population", cached_noise)
    native = _noise_cache_network("population", cached_noise)
    native.restore_dict_from_checkpoint(_snapshot(source.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(source)
    initialized, draws = _capture(functional, tensors)
    assert draws.random == draws.stimulation == {"cell": None}
    assert bool(draws.netstim.exp_rand.any()) is bool(cached_noise)
    replay = functional.initialize(
        *tensors, functional.initialization_inputs, draws=draws
    )
    _close(replay, initialized)
    native.initialize(DT)
    _close(replay.state, functional.extract(native).state)


@pytest.mark.parametrize("mode", ("transform", "population_cache", "network_cache"))
def test_connection_capture_uses_post_population_sources_and_preserves_gradients(mode):
    source = _network_with_population_source_phase(mode)
    native = _network_with_population_source_phase(mode)
    functional, tensors = dn.func.make_functional_network(source)
    global_rng = torch.get_rng_state().clone()
    initialized, draws = _capture(functional, tensors)
    _close(torch.get_rng_state(), global_rng)
    assert draws.connections is not None
    sampled = dn.func.sample_initialization_draws(functional, tensors)
    _assert_packet_equal(draws, sampled)
    replay = functional.initialize(
        *tensors, functional.initialization_inputs, draws=sampled
    )
    _close(replay, initialized)
    native.initialize(DT)
    _check_native(functional, replay, native)
    _close(replay.state, functional.extract(native).state)
    raw = tensors.parameters["populations"]["cell"]["cm_param.rho"]
    actual_gradient = torch.autograd.grad(
        _materialized_loss(functional, replay), raw, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(
        _materialized_loss(functional, initialized), raw
    )[0]
    _close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0


def test_connection_capture_retains_disabled_resampling_flags_and_rng_position():
    model = _network_with_population_source_phase("transform")
    functional, tensors = dn.func.make_functional_network(model)
    inputs = functional.initialization_inputs._replace(
        reinit_weights=False, reinit_delays=False
    )
    initialized, draws = _capture(functional, tensors, inputs)
    for collection in draws.connections.values.values():
        for roles in collection.values():
            assert roles == {"weights": {}, "delays": {}}
    _close(draws.connections.rng, tensors.state["connection_sampling"]["rng"])
    replay = functional.initialize(*tensors, inputs, draws=draws)
    _close(replay, initialized)
    with pytest.raises(ValueError, match="resampling flags"):
        functional.initialize(*tensors, functional.initialization_inputs, draws=draws)


@pytest.mark.parametrize("family", ("random", "stimulation", "netstim", "connections"))
def test_network_canonical_draws_reject_legacy_conflicts_before_component_execution(
    family, monkeypatch
):
    model, _ = _shared_noise_network(seed=812, initial_noise=0.0, increment=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    draws = dn.func.sample_initialization_draws(functional, tensors)
    inputs = functional.initialization_inputs
    if family in {"random", "stimulation"}:
        name = next(iter(inputs.populations))
        inputs = inputs._replace(
            populations={
                **inputs.populations,
                name: inputs.populations[name]._replace(
                    **{"random_draw" if family == "random" else "stimulation": object()}
                ),
            }
        )
    else:
        inputs = inputs._replace(
            **{"netstim_draw" if family == "netstim" else "connection_draw": object()}
        )

    def fail(*args, **kwargs):
        raise AssertionError("component validation ran before draw conflict validation")

    monkeypatch.setattr(functional, "_validate_initialization_source", fail)
    with pytest.raises(ValueError, match="legacy initialization draw packets"):
        functional.initialize(*tensors, inputs, draws=draws)
    with pytest.raises(ValueError, match="legacy initialization draw packets"):
        functional._initialize_host(*tensors, inputs, draws=draws)


def test_explicit_initialization_draws_never_sample_missing_netstim_family(monkeypatch):
    model, _ = _shared_noise_network(seed=812, initial_noise=0.0, increment=0.6)
    functional, tensors = dn.func.make_functional_network(model)
    draws = dn.func.sample_initialization_draws(functional, tensors)

    def fail(*args, **kwargs):
        raise AssertionError("explicit replay sampled NetStim initialization")

    monkeypatch.setattr(functional, "sample_initialization_draw", fail)
    complete = functional._initialize_host(
        *tensors, functional.initialization_inputs, draws=draws
    )
    assert complete.state["netstim"]["rng_state"] is not None
    with pytest.raises(ValueError, match="requires an explicit draw"):
        functional._initialize_host(
            *tensors,
            functional.initialization_inputs,
            draws=draws._replace(netstim=None),
        )
    with pytest.raises(ValueError, match="legacy initialization draw packets"):
        functional.initialize(
            *tensors,
            functional.initialization_inputs,
            draws=draws,
            netstim_draw=draws.netstim,
        )
