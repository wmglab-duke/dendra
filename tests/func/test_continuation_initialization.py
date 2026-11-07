"""Continuation carries fresh/cache replacements and authoritative RNG state."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _assign, _close, _joined, _plain, _snapshot
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_membrane_current import DT as CACHE_DT
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network import _stim_network
from tests.func.test_network_cache_initialization import _cached_network
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_network_population_caches import _owned_stim_network
from tests.func.test_population_owned_cache import _cached as _cached_population
from tests.func.test_population_owned_cache import _step as _cached_step

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


class _Scale(torch.nn.Module):
    def forward(self, value, factor):
        return (value * factor,)


@pytest.mark.parametrize("network", (False, True))
def test_fresh_packet_captures_parameter_replacement_and_spans_initialization_gradients(
    network, monkeypatch
):
    model = _network() if network else _population()
    owner = model.left if network else model
    owner.register_pre_initialize_transform(
        "scale",
        _Scale(),
        reads=(f"parameters.{G}",),
        writes=(f"parameters.{G}",),
        inputs={"factor": owner.v.new_tensor(1.5)},
    )
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    runner = dn.compile(
        model,
        dt=DT,
        steps=2,
        initialization="fresh",
        inference=False,
        callbacks={"trace": dn.func.Recorder(paths)},
        backend="eager",
        disable=True,
    )
    parameters = _plain(runner.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, 1.2 * upstream)
    factor = owner.v.new_tensor(1.7, requires_grad=True)
    defaults = runner.initialization_inputs
    if network:
        populations = dict(defaults.populations)
        population = populations["left"]
        populations["left"] = population._replace(
            transforms={**population.transforms, "pre.scale.factor": factor}
        )
        initialization = defaults._replace(populations=populations)
        initialized = runner.functional.initialize(
            parameters, runner.constants, runner.initial_state, initialization
        )
    else:
        initialization = defaults._replace(
            transforms={**defaults.transforms, "pre.scale.factor": factor}
        )
        initialized = runner.functional.initialize(
            parameters, runner.constants, initialization
        )
    expected, reference = runner.functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        steps=4,
        callbacks=runner.callbacks,
    )
    host_name = "_initialize_host" if network else "initialize_host"
    initialize_host = getattr(runner.functional, host_name)
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return initialize_host(*args, **kwargs)

    monkeypatch.setattr(runner.functional, host_name, counted)
    first, auxiliary = runner(parameters=parameters, initialization=initialization)
    continuation = auxiliary["continuation"]
    effective = _target(continuation.parameters, network)
    assert effective is _target(auxiliary["parameters"], network)
    assert effective is not _target(parameters, network)
    assert effective.grad_fn is not None
    torch.testing.assert_close(effective, 1.2 * upstream * factor)
    if network:
        assert (
            continuation.parameters["populations"]["left"][G]
            is continuation.parameters["populations"]["right"][G]
        )
    with pytest.raises(ValueError, match="parameters="):
        runner(first)
    actual, last = runner(continuation)
    assert len(calls) == 1
    assert _target(last["continuation"].parameters, network) is effective
    _close(actual, expected)
    _close(_joined(auxiliary, last), reference["callbacks"]["trace"])
    actual_gradients = torch.autograd.grad(
        _voltage(actual, network).square().sum(), (upstream, factor), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        _voltage(expected, network).square().sum(), (upstream, factor)
    )
    _close(actual_gradients, expected_gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual_gradients
    )
    assert _target(runner.parameters, network) is not effective


@pytest.mark.parametrize("reader_first", (False, True))
def test_fresh_network_packet_preserves_parameter_constant_aliases_and_native_order(
    reader_first,
):
    model, key = _cross_population_binding(reader_first=reader_first)
    native, _ = _cross_population_binding(reader_first=reader_first)
    runner = dn.compile(
        model,
        dt=NETWORK_DT,
        steps=2,
        initialization="fresh",
        inference=False,
        backend="eager",
        disable=True,
    )
    prefix, first = runner()
    continuation = first["continuation"]
    effective = continuation.parameters["populations"]["left"][key]
    assert (
        effective
        is continuation.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert effective is first["parameters"]["populations"]["left"][key]
    assert (
        effective
        is first["constants"]["populations"]["right"]["parametrizations.factor"]
    )
    assert effective is not runner.parameters["populations"]["left"][key]
    final, _ = runner(continuation)
    expected, _ = runner(
        prefix, parameters=first["parameters"], constants=first["constants"]
    )
    _close(final, expected)
    native.initialize(NETWORK_DT)
    for _ in range(4):
        native.step()
    _close(final, runner.functional.extract(native).state)


def test_population_owned_cache_packet_carries_effective_parameters_and_active_constants():
    source, native = _cached_population("point"), _cached_population("point")
    runner = dn.compile(
        source,
        dt=CACHE_DT,
        steps=2,
        initialization="fresh",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="eager",
        disable=True,
    )
    frozen = _snapshot(runner.initial_state)
    prefix, first = runner()
    continuation = first["continuation"]
    assert continuation.constants["cache.active"] is first["constants"]["cache.active"]
    assert bool(continuation.constants["cache.active"])
    assert not bool(runner.constants["cache.active"])
    with pytest.raises(ValueError, match="parameters="):
        runner(prefix)
    with pytest.raises(ValueError, match="constants="):
        runner(prefix, parameters=first["parameters"])
    final, last = runner(continuation)
    expected, reference = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    _close(final, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    native.initialize()
    with torch.no_grad():
        for _ in range(4):
            _cached_step(native)
    _close(final, runner.functional.extract(native).state)
    _close(runner.initial_state, frozen)


@pytest.mark.parametrize("owner", ("network", "population"))
def test_cached_network_packet_keeps_current_materializations_callbacks_and_rng(owner):
    model = (
        _cached_network(training=False, stimulus=True)
        if owner == "network"
        else _owned_stim_network()
    )
    runner = dn.compile(
        model,
        dt=NETWORK_DT,
        steps=4,
        initialization="fresh",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="eager",
        disable=True,
    )
    source_rng = model.netstim._rng.get_state().clone()
    source_state = _snapshot(runner.functional.extract().state)
    prefix, first = runner()
    continuation = first["continuation"]
    _close(continuation.parameters, first["parameters"])
    _close(continuation.constants, first["constants"])
    actual, last = runner(continuation)
    expected, reference = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    repeated, _ = runner()
    _close(repeated, prefix)
    _close(runner.functional.extract().state, source_state)
    torch.testing.assert_close(model.netstim._rng.get_state(), source_rng)
    assert last["callbacks"]["trace"]["cell.v"].shape[0] == 4


def test_network_packet_continues_explicit_draws_and_complete_rng_without_retaining_tapes():
    model = _stim_network()
    runner = dn.compile(
        model,
        dt=NETWORK_DT,
        steps=3,
        inference=True,
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="eager",
        disable=True,
    )
    initial = _snapshot(runner.initial_state)
    native_rng = model.netstim._rng.get_state().clone()
    first_draws = runner.functional.sample_runtime_draws(runner.initial_state, steps=3)
    prefix, first = runner(draws=first_draws)
    continuation = first["continuation"]
    last_draws = runner.functional.sample_runtime_draws(continuation.state, steps=3)
    actual, last = runner(continuation, draws=last_draws)
    with torch.no_grad():
        expected, reference = runner.functional.run(
            runner.parameters,
            runner.constants,
            initial,
            steps=6,
            callbacks=runner.callbacks,
        )
    _close(actual, expected)
    _close(_joined(first, last), reference["callbacks"]["trace"])
    _close(runner.initial_state, initial)
    torch.testing.assert_close(model.netstim._rng.get_state(), native_rng)
    assert not torch.equal(
        prefix["netstim"]["rng_state"], initial["netstim"]["rng_state"]
    )
