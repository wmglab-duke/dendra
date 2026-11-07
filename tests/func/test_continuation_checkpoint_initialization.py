"""Portable continuation retains fresh/cache frames, shared sources and RNG."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _joined,
    _plain,
    _snapshot,
)
from tests.func.test_continuation_checkpoints import _disk, _plain_checkpoint
from tests.func.test_continuation_initialization import _Scale
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_membrane_current import DT as CACHE_DT
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network import _stim_network
from tests.func.test_network_cache_initialization import _cached_network
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_network_population_caches import _owned_stim_network
from tests.func.test_population_owned_cache import _cached as _cached_population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _fresh_program(network):
    model = _network() if network else _population()
    owner = model.left if network else model
    owner.register_pre_initialize_transform(
        "scale",
        _Scale(),
        reads=(f"parameters.{G}",),
        writes=(f"parameters.{G}",),
        inputs={"factor": owner.v.new_tensor(1.5)},
    )
    return dn.compile(
        model,
        dt=DT,
        steps=2,
        initialization="fresh",
        inference=False,
        callbacks=_callbacks(network),
        backend="eager",
        disable=True,
    )


@pytest.mark.parametrize("network", (False, True))
def test_portable_fresh_replacements_are_new_trainable_leaves_without_reinitialization(
    network, tmp_path, monkeypatch
):
    source, receiver = _fresh_program(network), _fresh_program(network)
    parameters = _plain(source.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, upstream * 1.2)
    _, first = source(parameters=parameters)
    current = first["continuation"]
    original = _target(current.parameters, network)
    assert original.grad_fn is not None and original.requires_grad
    saved = _disk(source.export_continuation(current), tmp_path / "fresh.pt")
    restored = receiver.restore_continuation(saved)
    effective = _target(restored.parameters, network)
    assert effective.is_leaf and effective.grad_fn is None and effective.requires_grad
    torch.testing.assert_close(effective, original)
    assert effective is not _target(receiver.parameters, network)
    selected = effective
    if network:
        assert effective is restored.parameters["populations"]["right"][G]
        key = "materializations.values.integrator.mech.mechanisms.leak.g"
        before = current.constants["populations"]["left"][key]
        selected = restored.constants["populations"]["left"][key]
        assert selected.requires_grad == before.requires_grad
        assert selected.requires_grad and selected.is_leaf and selected.grad_fn is None
        assert selected is not before and selected.data_ptr() != before.data_ptr()
        torch.testing.assert_close(selected, before)

    def reject(*args, **kwargs):
        raise AssertionError("restored continuation must bypass fresh initialization")

    monkeypatch.setattr(
        receiver.functional,
        "_initialize_host" if network else "initialize_host",
        reject,
    )
    actual, last = receiver(restored)
    expected, reference = source(current)
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    targets = (selected, effective, upstream) if network else (selected, upstream)
    gradient, *raw_gradients, prefix_gradient = torch.autograd.grad(
        _voltage(actual, network).square().sum(),
        targets,
        allow_unused=True,
    )
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    if network:
        # Active retained values are independent leaves at the portable boundary.
        (raw_gradient,) = raw_gradients
        assert raw_gradient is None or not torch.count_nonzero(raw_gradient)
        assert last["continuation"].constants["populations"]["left"][key] is selected
    assert prefix_gradient is None
    assert _target(last["continuation"].parameters, network) is effective


@pytest.mark.parametrize("reader_first", (False, True))
def test_portable_network_preserves_effective_cross_parameter_constant_aliases_and_flags(
    reader_first, tmp_path
):
    model, key = _cross_population_binding(reader_first=reader_first)
    receiver_model, _ = _cross_population_binding(reader_first=reader_first)
    options = dict(
        dt=NETWORK_DT,
        steps=2,
        initialization="fresh",
        inference=False,
        backend="eager",
        disable=True,
    )
    source, receiver = (
        dn.compile(model, **options),
        dn.compile(receiver_model, **options),
    )
    parameters = _plain(source.parameters)
    upstream = parameters["populations"]["left"][key].detach().clone().requires_grad_()
    parameters["populations"]["left"][key] = upstream
    _, first = source(parameters=parameters)
    current = first["continuation"]
    expected_source = current.parameters["populations"]["left"][key]
    assert (
        expected_source
        is current.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert expected_source.requires_grad
    restored = receiver.restore_continuation(
        _disk(source.export_continuation(current), tmp_path / "alias.pt")
    )
    selected = restored.parameters["populations"]["left"][key]
    assert (
        selected
        is restored.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert selected.is_leaf and selected.requires_grad and selected.grad_fn is None
    assert selected is not expected_source
    torch.testing.assert_close(selected, expected_source)
    actual, _ = receiver(restored)
    expected, _ = source(current)
    _close(actual, expected)


def test_portable_population_owned_cache_restores_active_effective_frames_without_mutating_defaults(
    tmp_path,
):
    source_model, receiver_model = (
        _cached_population("point"),
        _cached_population("point"),
    )
    options = dict(
        dt=CACHE_DT,
        steps=2,
        initialization="fresh",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="eager",
        disable=True,
    )
    source, receiver = (
        dn.compile(source_model, **options),
        dn.compile(receiver_model, **options),
    )
    defaults = _snapshot(
        (receiver.parameters, receiver.constants, receiver.initial_state)
    )
    _, first = source()
    current = first["continuation"]
    assert bool(current.constants["cache.active"])
    assert not bool(receiver.constants["cache.active"])
    restored = receiver.restore_continuation(
        _disk(source.export_continuation(current), tmp_path / "owned-cache.pt")
    )
    assert bool(restored.constants["cache.active"])
    actual, last = receiver(restored)
    expected, reference = source(current)
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    _close((receiver.parameters, receiver.constants, receiver.initial_state), defaults)


@pytest.mark.parametrize("owner", ("network", "population"))
def test_portable_cached_network_retains_rng_pending_delivery_and_callback_phase(
    owner, tmp_path
):
    make = (
        (lambda: _cached_network(training=False, stimulus=True))
        if owner == "network"
        else _owned_stim_network
    )
    model, receiver_model = make(), make()
    options = dict(
        dt=NETWORK_DT,
        steps=4,
        initialization="fresh",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="eager",
        disable=True,
    )
    source, receiver = (
        dn.compile(model, **options),
        dn.compile(receiver_model, **options),
    )
    source_rng = model.netstim._rng.get_state().clone()
    receiving_rng = receiver_model.netstim._rng.get_state().clone()
    native = _snapshot(receiver.functional.extract().state)
    _, first = source()
    current = first["continuation"]
    restored = receiver.restore_continuation(
        _disk(source.export_continuation(current), tmp_path / "network-cache.pt")
    )
    _close(restored.parameters, current.parameters)
    _close(restored.constants, current.constants)
    _close(restored.state, current.state)
    torch.testing.assert_close(model.netstim._rng.get_state(), source_rng)
    torch.testing.assert_close(receiver_model.netstim._rng.get_state(), receiving_rng)
    _close(receiver.functional.extract().state, native)
    actual, last = receiver(restored)
    expected, reference = source(current)
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    assert last["callbacks"]["trace"]["cell.v"].shape[0] == 4


def _scheduled_network():
    network = _stim_network()
    stimulus = network.netstim
    stimulus.schedule(
        [0, 0],
        torch.tensor([NETWORK_DT, 4 * NETWORK_DT]),
        weight=torch.tensor([0.4, 0.9]),
    )
    stimulus.sched_time_ms.requires_grad_()
    stimulus.sched_weight.requires_grad_()
    connection = next(iter(network.synapses.values()))
    connection.schedule(
        con_indices=[0, 0],
        times_ms=[2 * NETWORK_DT, 5 * NETWORK_DT],
        weight=torch.tensor([0.2, -0.1]),
    )
    connection.sched_time_ms.requires_grad_()
    connection.sched_weight.requires_grad_()
    return network


def test_weights_only_network_checkpoint_reconstructs_trusted_schedule_namedtuples_and_rng(
    tmp_path,
):
    model, receiver_model = _scheduled_network(), _scheduled_network()
    options = dict(
        dt=NETWORK_DT,
        steps=3,
        inference=True,
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        backend="eager",
        disable=True,
    )
    source, receiver = (
        dn.compile(model, **options),
        dn.compile(receiver_model, **options),
    )
    initial = _snapshot(source.initial_state)
    parameters = _plain(source.parameters)

    def trainable(schedule):
        return schedule._replace(
            times=schedule.times.detach().clone().requires_grad_(),
            amplitudes=schedule.amplitudes.detach().clone().requires_grad_(),
        )

    parameters["netstim"]["schedule"] = trainable(parameters["netstim"]["schedule"])
    for values in parameters["synapses"].values():
        values["schedule"] = trainable(values["schedule"])
    _, first = source(parameters=parameters)
    current = first["continuation"]
    packet = source.export_continuation(current)
    _plain_checkpoint(packet)
    restored = receiver.restore_continuation(_disk(packet, tmp_path / "schedules.pt"))
    schedules = [
        (
            current.parameters["netstim"]["schedule"],
            restored.parameters["netstim"]["schedule"],
        )
    ]
    schedules.extend(
        (values["schedule"], restored.parameters["synapses"][name]["schedule"])
        for name, values in current.parameters["synapses"].items()
    )
    for before, after in schedules:
        assert type(before) is type(after) is dn.func.TensorScheduleTensors
        _close(after, before)
        assert after.times.requires_grad == before.times.requires_grad
        assert after.amplitudes.requires_grad == before.amplitudes.requires_grad
        assert after.times.requires_grad and after.amplitudes.requires_grad
        assert after.times.is_leaf and after.amplitudes.is_leaf
    actual, last = receiver(restored)
    with torch.no_grad():
        expected, reference = source.functional.run(
            parameters, source.constants, initial, steps=6, callbacks=source.callbacks
        )
    _close(actual, expected)
    _close(_joined(first, last), reference["callbacks"]["trace"])
