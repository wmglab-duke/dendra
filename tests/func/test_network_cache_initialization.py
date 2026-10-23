"""Cached Network starts share native restoration and explicit continuation."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Mapping

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _mixed_network,
    _stim_network,
)
from tests.func.test_network_bitpacked_inference import _sync_stimulus
from tests.func.test_network_pending_calendar import _network as _pending_network

pytestmark = pytest.mark.cpu
CONDUCTANCE = "integrator.mech.mechanisms.pas.g_param"


@pytest.fixture(autouse=True)
def _compiler_environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


def _warm_cache(network):
    network._enable_synapse_state_cache_recording()
    with torch.no_grad():
        for _ in range(6):
            network.step()
        network.cache_state()
    network._disable_synapse_state_cache_recording(release=True)
    return network


def _change_connection_parameters(network):
    with torch.no_grad():
        for connection in network.synapses.values():
            for parameter in connection.weight.parameters():
                parameter.add_(0.35)
            for parameter in connection.delay_ms.parameters():
                parameter.add_(0.55)
        for connection in network.continuous_synapses.values():
            for parameter in connection.weight.parameters():
                parameter.sub_(0.25)
            for parameter in connection.delay_ms.parameters():
                parameter.add_(0.3)
            if isinstance(connection.transform, torch.nn.Module):
                for parameter in connection.transform.parameters():
                    parameter.add_(0.4)


def _cached_network(*, training=True, batch=None, changed=True, stimulus=False):
    network = _stim_network() if stimulus else _mixed_network(batch=batch)
    if not training:
        network.eval()
        network.initialize(DT, reinit_weights=False, reinit_delays=False)
    _warm_cache(network)
    if changed:
        _change_connection_parameters(network)
    # Lower against the current fixed delay topology. Saved source histories
    # still describe the warmup performed with the earlier parameter values.
    network.initialize(DT)
    return network


def _trace():
    return {"trace": dn.func.Recorder(("cell.v", "t"))}


def _plain(value):
    if isinstance(value, Mapping):
        return {name: _plain(child) for name, child in value.items()}
    if isinstance(value, tuple):
        values = tuple(_plain(child) for child in value)
        return type(value)(*values) if hasattr(value, "_fields") else values
    return value


def _cache_initialized_native_packet(functional, native):
    packet = functional.extract(native)
    # Ordinary extraction describes snapshot execution. Fresh cache restore
    # additionally selects its saved materialized Population buffers.
    for constants in packet.constants["populations"].values():
        if "cache.active" in constants:
            constants["cache.active"] = torch.ones_like(constants["cache.active"])
        # Fresh initialization publishes explicit owner materializations;
        # ordinary extraction keeps its live snapshot bindings inactive.
        for name, value in tuple(constants.items()):
            if name.startswith("materializations.active."):
                constants[name] = torch.ones_like(value)
            elif name.startswith("materializations.baseline."):
                constants[name] = torch.zeros_like(value)
    # Fresh connections use their current raw materialization; extraction
    # additionally retains the native initialized output as a snapshot offset.
    for collection in ("synapses", "continuous_synapses"):
        for constants in packet.constants[collection].values():
            for role in ("weights", "delays"):
                key = f"use_offset.{role}"
                constants[key] = torch.zeros_like(constants[key])
    return packet


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("training", (False, True))
def test_cache_initialization_matches_native_current_parameters_and_pending_traffic(
    batch, training
):
    source = _cached_network(training=training, batch=batch)
    native = _cached_network(training=training, batch=batch)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    defaults = functional.initialization_inputs
    initialized = functional.initialize(
        *tensors, defaults._replace(t=defaults.t.new_tensor(-0.25))
    )
    native.initialize(DT, t=-0.25)
    expected = _cache_initialized_native_packet(functional, native)
    _close(initialized, expected)
    assert any(
        value["delivery_buffer"].ne(0).any()
        for value in initialized.state["synapses"].values()
    )
    assert any(
        value["delivery_buffer"].ne(0).any()
        for value in initialized.state["continuous_synapses"].values()
    )
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)


def test_cached_population_values_override_inputs_and_preserve_native_parameter_identity_gradients():
    functional, tensors = dn.func.make_functional_network(_cached_network())
    parameters = _clone(tensors.parameters, detached=True)
    upstream = parameters["populations"]["cell"][CONDUCTANCE].requires_grad_()
    parameters["populations"]["cell"][CONDUCTANCE] = upstream * 3
    defaults = functional.initialization_inputs
    voltage = defaults.populations["cell"].v_init.detach().clone().requires_grad_()
    inputs = defaults._replace(
        populations={"cell": defaults.populations["cell"]._replace(v_init=voltage + 15)}
    )
    initialized = functional.initialize(
        parameters, tensors.constants, tensors.state, inputs
    )
    torch.testing.assert_close(
        initialized.parameters["populations"]["cell"][CONDUCTANCE],
        tensors.parameters["populations"]["cell"][CONDUCTANCE],
    )
    torch.testing.assert_close(
        initialized.state["populations"]["cell"]["integrator"]["v"],
        tensors.state["populations"]["cell"]["integrator"]["v"],
    )
    leaves = pytree.tree_leaves(initialized.state["populations"])
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in leaves
        if torch.is_tensor(value)
    )
    value = initialized.parameters["populations"]["cell"][CONDUCTANCE]
    parameter_gradient, voltage_gradient = torch.autograd.grad(
        value.sum(), (upstream, voltage), allow_unused=True
    )
    torch.testing.assert_close(parameter_gradient, torch.full_like(upstream, 3))
    assert voltage_gradient is None


def test_cached_start_skips_population_initialization_hooks_like_native_network():
    source = _cached_network()
    native = _cached_network()

    def forbidden(population):
        del population
        raise AssertionError("cached Network initialization must not run fresh hooks")

    for network in (source, native):
        network.cell.register_pre_initialize_hook(forbidden)
        network.cell.register_post_initialize_hook(forbidden)
    functional, tensors = dn.func.make_functional_network(source)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    _close(initialized, _cache_initialized_native_packet(functional, native))


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_compiled_cached_starts_restart_and_continue_like_native_and_checkpoint(
    execution,
):
    source = _cached_network()
    native = _cached_network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        callbacks=_trace(),
        backend="aot_eager",
    )
    frozen = _clone(runner.initial_state, detached=True)
    native.initialize(DT)
    frames = [native.cell.v.detach().clone()]
    for _ in range(8):
        native.step()
        frames.append(native.cell.v.detach().clone())
    prefix, first = runner()
    repeated, repeated_aux = runner()
    _close(prefix, repeated)
    _close(first["callbacks"].outputs, repeated_aux["callbacks"].outputs)
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    _close(final, runner.functional.extract(native).state)
    observed = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(observed, torch.stack(frames), rtol=2e-8, atol=2e-9)
    _close(runner.initial_state, frozen)
    if execution == "while":
        assert runner.execution_report().strategy == "compiled_while_loop"
    receiver = _cached_network()
    runner.functional.commit_state_(receiver, final)
    checkpoint = receiver.state_dict_for_checkpoint()
    resumed = _cached_network()
    resumed.restore_dict_from_checkpoint(checkpoint)
    resumed.step()
    continued, _ = runner.functional.prepare_and_step(
        first["parameters"], first["constants"], final
    )
    _close(continued, runner.functional.extract(resumed).state)


def test_cache_boundary_detaches_pending_history_but_later_steps_keep_weight_gradients():
    source = _cached_network()
    name = next(iter(source.synapses))
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        inference=False,
        trainable=(f"synapses.{name}.weight.weights.0.param.rho",),
        checkpoint_every=2,
        callbacks=_trace(),
        backend="aot_eager",
    )
    initialized = runner.functional.initialize(
        runner.parameters,
        runner.constants,
        runner.initial_state,
        runner.initialization_inputs,
    )
    assert (
        initialized.parameters["synapses"][name]["weight.weights.0.param.rho"]
        is runner.parameters["synapses"][name]["weight.weights.0.param.rho"]
    )
    assert all(
        value.grad_fn is None
        for value in initialized.state["synapses"][name].values()
        if torch.is_tensor(value)
    )
    prefix, first = runner()
    final, second = runner(
        prefix, parameters=first["parameters"], constants=first["constants"]
    )
    loss = final["populations"]["cell"]["mechanism_buffers"]["relay"]["ledger"].sum()
    gradient = torch.autograd.grad(
        loss, runner.get_parameter(f"synapses.{name}.weight.weights.0.param.rho")
    )[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert second["callbacks"].state["trace"] == ()


def test_cached_netstim_initialization_and_continuation_match_native_rng_order():
    source = _cached_network(stimulus=True)
    native = _cached_network(stimulus=True)
    _sync_stimulus(source, native)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        inference=False,
        backend="aot_eager",
    )
    original = runner.functional.extract().state
    native.initialize(DT)
    for _ in range(8):
        native.step()
    prefix, first = runner()
    final, _ = runner(
        prefix, parameters=first["parameters"], constants=first["constants"]
    )
    _close(final, runner.functional.extract(native).state)
    repeated, _ = runner()
    _close(repeated, prefix)
    _close(runner.functional.extract().state, original)


def test_cache_initialization_results_do_not_retain_previous_invocation_graphs():
    functional, tensors = dn.func.make_functional_network(_cached_network())
    first = functional.initialize(*tensors, functional.initialization_inputs)
    state, auxiliary = functional.prepare_and_step(*first)
    references = [
        weakref.ref(value)
        for value in pytree.tree_leaves(state)
        if torch.is_tensor(value) and value.grad_fn is not None
    ]
    assert references
    del first, state, auxiliary
    gc.collect()
    assert all(reference() is None for reference in references)


def test_captured_cache_defaults_are_independent_detached_and_explicitly_replaceable():
    source = _cached_network()
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    assert isinstance(defaults.cache, dn.func.NetworkInitializationCache)
    original = source._state_cache["cell"]["v"]
    captured = defaults.cache.populations["cell"]["v"]
    assert original.data_ptr() != captured.data_ptr()
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in pytree.tree_leaves(defaults.cache)
        if torch.is_tensor(value)
    )
    override = _clone(defaults.cache, detached=True)
    changed = override.populations["cell"]["v"].requires_grad_()
    override.populations["cell"]["v"] = changed + 2
    initialized = functional.initialize(*tensors, defaults._replace(cache=override))
    actual = initialized.state["populations"]["cell"]["integrator"]["v"]
    torch.testing.assert_close(actual, captured + 2)
    assert not actual.requires_grad
    torch.testing.assert_close(defaults.cache.populations["cell"]["v"], original)


def test_loaded_normalized_event_cache_is_captured_independently_without_its_graph():
    source = _cached_network(changed=False)
    first, _ = dn.func.make_functional_network(source)
    name = next(iter(source.synapses))
    cache = first.synapses[name].normalize_cache(source._syn_cache["event"][name])
    tracked = cache.gate_history.detach().clone().requires_grad_() * 1
    cache = cache._replace(gate_history=tracked)
    source.load_state_cache(
        syn_cache={
            **source._syn_cache,
            "event": {**source._syn_cache["event"], name: cache},
        }
    )
    # Native restoration accepts these same pre-normalized cache packets.
    source.initialize(DT)
    functional, tensors = dn.func.make_functional_network(source)
    captured = functional.initialization_inputs.cache.synapses[name]
    assert captured is not cache
    assert captured.gate_history.data_ptr() != tracked.data_ptr()
    assert not captured.gate_history.requires_grad
    assert captured.gate_history.grad_fn is None
    saved = _clone(captured, detached=True)
    expected = functional.initialize(*tensors, functional.initialization_inputs)
    with torch.no_grad():
        tracked.add_(1)
    _close(captured, saved)
    _close(functional.initialize(*tensors, functional.initialization_inputs), expected)


def test_cache_presence_is_frozen_and_synapse_cache_alone_follows_native_uncached_path():
    cached, tensors = dn.func.make_functional_network(_cached_network())
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="cache"):
        cached.initialize(*tensors, cached.initialization_inputs._replace(cache=None))
    source, native = _mixed_network(), _mixed_network()
    source._syn_cache = _cached_network()._syn_cache
    functional, packet = dn.func.make_functional_network(source)
    assert functional.initialization_inputs.cache is None
    initialized = functional.initialize(*packet, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="cache"):
        functional.initialize(
            *packet,
            functional.initialization_inputs._replace(
                cache=cached.initialization_inputs.cache
            ),
        )


def test_invalid_explicit_cache_is_atomic_for_runner_state_source_and_report():
    source = _cached_network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        inference=False,
        callbacks=_trace(),
        backend="aot_eager",
    )
    runner()
    report = runner.execution_report()
    initial = _clone(runner.initial_state, detached=True)
    parameters = _clone(_plain(runner.parameters), detached=True)
    original = runner.functional.extract().state
    bad = _clone(runner.initialization_inputs.cache, detached=True)
    bad.populations["cell"]["v"] = torch.empty(0, dtype=DTYPE)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="cache|shape|schema"
    ):
        runner(initialization=runner.initialization_inputs._replace(cache=bad))
    assert runner.execution_report() is report
    _close(runner.initial_state, initial)
    _close(_plain(runner.parameters), parameters)
    _close(runner.functional.extract().state, original)


def test_cached_geometry_is_returned_explicitly_and_continuation_requires_effective_packets():
    source = _cached_network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        inference=False,
        backend="aot_eager",
    )
    defaults = runner.initialization_inputs
    cache = _clone(defaults.cache, detached=True)
    cache.populations["cell"]["dx"].mul_(1.25)
    prefix, first = runner(initialization=defaults._replace(cache=cache))
    assert "constants" in first and "parameters" in first
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            pytree.tree_leaves(first["constants"]),
            pytree.tree_leaves(_plain(runner.constants)),
            strict=True,
        )
        if torch.is_tensor(left)
    )
    report = runner.execution_report()
    with pytest.raises(ValueError, match="constants"):
        runner(prefix, parameters=first["parameters"])
    assert runner.execution_report() is report
    final, _ = runner(
        prefix, parameters=first["parameters"], constants=first["constants"]
    )
    assert final["clock"]["step"].item() == 8


def test_cache_calendar_overflow_does_not_publish_a_partial_initialization():
    source = _pending_network("sparse_calendar", start=100)
    source.step()  # Two scheduled entries remain pending in the cache.
    source.cache_state()
    next(iter(source.synapses.values())).zero()
    source._reset_runtime_clock(0)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        pending_capacity=1,
        backend="aot_eager",
    )
    original = runner.functional.extract().state
    initial = _clone(runner.initial_state, detached=True)
    cache = _clone(runner.initialization_inputs.cache, detached=True)
    with pytest.raises(RuntimeError, match="overflow"):
        runner()
    assert runner.execution_report() is None
    _close(runner.functional.extract().state, original)
    _close(runner.initial_state, initial)
    _close(runner.initialization_inputs.cache, cache)
