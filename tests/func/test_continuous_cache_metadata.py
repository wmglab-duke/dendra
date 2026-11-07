"""Continuous receive diagnostics do not freeze warmed Population caches."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import graded_syn, pas
from tests.func.test_functional_initialization import (
    _assert_tensor_snapshot,
    _tensor_snapshot,
)
from tests.func.test_network import _clone, _close

DT = 0.001
COUNTER = "integrator.mech.mechanisms.analog._continuous_reset_count"
SUPPORT = "integrator.mech.mechanisms.pas.key"
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def _network(scope):
    pre = dn.SingleCompartment(N=1, C=1, v_init=-62.0)
    pre.insert(pas, g=0.0004, e=-70.0)
    post = dn.Unmyelinated(
        [2.0], L=30.0, dx=10.0, v_init=-65.0, integrator=dn.bwd_euler_ub(method="pcr")
    )
    post[:, [0, 2]].insert(pas, g=0.0004, e=-70.0)
    post.insert(graded_syn.rename("analog"), e=-10.0)
    network = dn.Network({"pre": pre, "post": post})
    network.connect_continuous_one_to_one(
        pre[:],
        post[:, 1],
        post.mech.analog,
        input="g_pre",
        transform=_release,
        weight=0.0002,
        delay=DT,
    )
    network.train()
    network.initialize(DT)
    network._enable_synapse_state_cache_recording()
    with torch.no_grad():
        for _ in range(2):
            network.step()
        if scope == "network":
            network.cache_state()
        else:
            post.cache("_steady_state")
        for _ in range(2):
            network.step()
    network._disable_synapse_state_cache_recording(release=True)
    saved = _cache(network, scope)
    assert saved[COUNTER].item() == 2
    assert post.mech.analog._continuous_reset_count.item() == 4
    return network


def _cache(network, scope):
    return (
        network._state_cache["post"]
        if scope == "network"
        else network.post._caches["_steady_state"]
    )


def _snapshot(network):
    return {
        (kind, name): _tensor_snapshot(value)
        for kind, entries in (
            ("parameters", network.named_parameters(remove_duplicate=False)),
            ("buffers", network.named_buffers(remove_duplicate=False)),
        )
        for name, value in entries
    }


def _assert_snapshot(network, snapshot):
    values = {
        (kind, name): value
        for kind, entries in (
            ("parameters", network.named_parameters(remove_duplicate=False)),
            ("buffers", network.named_buffers(remove_duplicate=False)),
        )
        for name, value in entries
    }
    assert values.keys() == snapshot.keys()
    for name, value in values.items():
        _assert_tensor_snapshot(value, snapshot[name])


@pytest.mark.parametrize("scope", ("population", "owned", "network"))
def test_stale_continuous_reset_metadata_restores_cache_without_changing_source(scope):
    source, native = _network(scope), _network(scope)
    snapshot = _snapshot(source)
    saved = _clone(_cache(source, scope), detached=True)
    if scope == "population":
        functional, tensors = dn.func.make_functional(source.post, dt=DT)
        frozen = _clone(tensors, detached=True)
        initialized = functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
        native.post.initialize()
        _close(initialized.state, functional.extract(native.post).state)
        actual = initialized.state
        for _ in range(2):
            actual, _ = functional.prepare_and_step(
                initialized.parameters, initialized.constants, actual
            )
            native.post.step(DT)
            _close(actual, functional.extract(native.post).state)
    else:
        functional, tensors = dn.func.make_functional_network(source)
        frozen = _clone(tensors, detached=True)
        defaults = functional.initialization_inputs
        frozen_defaults = _clone(defaults, detached=True)
        initialized = functional.initialize(*tensors, defaults)
        native.initialize(DT)
        _close(initialized.state, functional.extract(native).state)
        actual = initialized.state
        for _ in range(2):
            actual, _ = functional.prepare_and_step(
                initialized.parameters, initialized.constants, actual
            )
            native.step()
            _close(actual, functional.extract(native).state)
        _close(defaults, frozen_defaults)
    assert source.post.mech.analog._continuous_reset_count.item() == 4
    _close(tensors, frozen)
    _close(_cache(source, scope), saved)
    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("field", (SUPPORT, "x"))
def test_continuous_cache_keeps_support_indices_and_coordinates_frozen(field):
    source = _network("network")
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    cache = defaults.cache.populations["post"]
    cache[field].reshape(-1)[0].add_(1)
    snapshot = _snapshot(source)
    frozen = _clone((tensors, defaults), detached=True)
    with pytest.raises(dn.func.FunctionalizationError, match="frozen topology"):
        functional.initialize(*tensors, defaults)
    _close((tensors, defaults), frozen)
    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("invalid", ("shape", "dtype"))
def test_continuous_reset_metadata_keeps_its_explicit_tensor_schema(invalid):
    source = _network("network")
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    cache = defaults.cache.populations["post"]
    cache[COUNTER] = (
        cache[COUNTER].reshape(1)
        if invalid == "shape"
        else cache[COUNTER].to(torch.float32)
    )
    snapshot = _snapshot(source)
    frozen = _clone((tensors, defaults), detached=True)
    with pytest.raises(ValueError, match="_continuous_reset_count must match"):
        functional.initialize(*tensors, defaults)
    _close((tensors, defaults), frozen)
    _assert_snapshot(source, snapshot)
