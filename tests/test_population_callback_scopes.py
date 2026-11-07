"""Native scope forwarding without building or advancing a simulation."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from dendra.func import FunctionalCallback
from dendra.func import PopulationCallback as FunctionalScope
from dendra.models.callbacks import (
    APCount,
    Callback,
    PopulationCallback,
    Recorder,
    _bind_callback_dt,
)
from dendra.models.networks.net import Network

pytestmark = pytest.mark.cpu
_HOOKS = (
    "pre_loop_hook",
    "post_loop_hook",
    "pre_step_hook",
    "post_step_hook",
    "pre_chunk_hook",
    "post_chunk_hook",
)


def _network(**populations):
    # Scope dispatch only requires Network identity and its owner mapping.
    # Avoid unrelated initialization, RNG, geometry, and numerical setup.
    network = object.__new__(Network)
    torch.nn.Module.__init__(network)
    network.populations = populations
    return network


class _Spy(Callback):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.result = object()

    def pre_loop_hook(self, model):
        self.calls.append(("pre_loop_hook", model))
        return self.result

    def post_loop_hook(self, model):
        self.calls.append(("post_loop_hook", model))
        return self.result

    def pre_step_hook(self, model):
        self.calls.append(("pre_step_hook", model))
        return self.result

    def post_step_hook(self, model):
        self.calls.append(("post_step_hook", model))
        return self.result

    def pre_chunk_hook(self, model, timepoints=None):
        self.calls.append(("pre_chunk_hook", model, timepoints))
        return self.result

    def post_chunk_hook(self, model, timepoints=None):
        self.calls.append(("post_chunk_hook", model, timepoints))
        return self.result


class _Pure(FunctionalCallback):
    def initialize(self, state, auxiliary):
        raise AssertionError("native scope must not run pure hooks")

    def update(self, carry, state, auxiliary):
        raise AssertionError("native scope must not run pure hooks")

    def finalize(self, carry, emissions):
        raise AssertionError("native scope must not run pure hooks")


@pytest.mark.parametrize("hook", _HOOKS)
def test_scope_forwards_native_hooks_results_and_timepoint_identity(hook):
    left, right = object(), object()
    network = _network(left=left, right=right)
    callback = _Spy()
    scope = PopulationCallback("right", callback)
    times = object()
    args = (network, times) if "chunk" in hook else (network,)
    assert getattr(scope, hook)(*args) is callback.result
    expected = (hook, right, times) if "chunk" in hook else (hook, right)
    assert callback.calls == [expected]
    assert scope.callback is callback


@pytest.mark.parametrize("hook", ("pre_chunk_hook", "post_chunk_hook"))
def test_scope_preserves_optional_chunk_timepoints(hook):
    target = object()
    callback = _Spy()
    scope = PopulationCallback("left", callback)
    assert getattr(scope, hook)(_network(left=target)) is callback.result
    assert callback.calls == [(hook, target, None)]


@pytest.mark.parametrize(
    "population,error,message",
    (
        (None, TypeError, "must be a string"),
        (1, TypeError, "must be a string"),
        ("", ValueError, "must not be empty"),
    ),
)
def test_scope_validates_names_before_callback_configuration(
    population, error, message
):
    with pytest.raises(error, match=message):
        PopulationCallback(population, object())


def test_scope_validates_callback_configuration_and_preserves_name_identity():
    class Name(str):
        pass

    name = Name("left")
    scope = PopulationCallback(name, Callback())
    assert scope.population is name
    with pytest.raises(TypeError, match="native Callback or pure FunctionalCallback"):
        PopulationCallback("left", object())


@pytest.mark.parametrize("native", (False, True))
def test_scope_rejects_nested_canonical_and_functional_scopes(native):
    child = (
        PopulationCallback("left", Callback())
        if native
        else FunctionalScope("left", _Pure())
    )
    with pytest.raises(ValueError, match="scopes cannot be nested"):
        PopulationCallback("right", child)


def test_scope_resolves_current_top_level_owner_without_attribute_fallback():
    original, replacement = object(), object()
    network = _network(left=original)
    callback = _Spy()
    scope = PopulationCallback("left", callback)
    scope.pre_step_hook(network)
    network.populations["left"] = replacement
    scope.post_step_hook(network)
    assert callback.calls == [
        ("pre_step_hook", original),
        ("post_step_hook", replacement),
    ]
    network.other = object()
    scope.population = "other"
    with pytest.raises(
        ValueError,
        match="^Unknown Network Population 'other'; available names are \\('left',\\)$",
    ):
        scope.pre_loop_hook(network)
    assert len(callback.calls) == 2


@pytest.mark.parametrize(
    "population,error,message",
    (
        (None, TypeError, "must be a string"),
        ([], TypeError, "must be a string"),
        ("", ValueError, "must not be empty"),
    ),
)
def test_scope_revalidates_mutable_name_before_native_hooks(population, error, message):
    callback = _Spy()
    scope = PopulationCallback("left", callback)
    scope.population = population
    with pytest.raises(error, match=message):
        scope.pre_loop_hook(_network(left=object()))
    assert callback.calls == []


def test_scope_requires_network_without_invoking_inner_hooks():
    callback = _Spy()
    scope = PopulationCallback("left", callback)
    with pytest.raises(TypeError, match="must be used with a Network"):
        scope.pre_loop_hook(SimpleNamespace(populations={"left": object()}))
    assert callback.calls == []


@pytest.mark.parametrize("hook", _HOOKS)
def test_scope_accepts_pure_configuration_but_rejects_native_execution(hook):
    callback = _Pure()
    scope = PopulationCallback("left", callback)
    assert scope.callback is callback
    assert dict(scope.named_children()) == {}
    _bind_callback_dt(scope, 0.125)
    assert "dt" not in vars(scope) and "dt" not in vars(callback)
    with pytest.raises(TypeError, match="native Callback for native execution"):
        getattr(scope, hook)(_network(left=object()))


@pytest.mark.parametrize("configured_dt", (None, 0.125))
def test_scope_timestep_binding_preserves_authored_threshold_configuration(
    configured_dt,
):
    callback = APCount(
        node_check=[0], t_start_check=0.2, t_end_check=0.8, dt=configured_dt
    )
    scope = PopulationCallback("left", callback)
    _bind_callback_dt(scope, 0.1)
    assert callback.dt == 0.1 and (callback.ind_start, callback.ind_end) == (2, 8)
    assert callback._functional_threshold_configuration()["dt"] == configured_dt
    assert "dt" not in vars(scope)
    _bind_callback_dt(scope, 0.05)
    assert callback.dt == 0.05 and (callback.ind_start, callback.ind_end) == (4, 16)
    assert callback._functional_threshold_configuration()["dt"] == configured_dt


def test_scope_timestep_binding_reuses_recorder_cadence_and_custom_setters():
    recorder = Recorder("v", dt=0.3)
    scope = PopulationCallback("left", recorder)
    _bind_callback_dt(scope, 0.1)
    assert recorder.dt == 0.1 and recorder.save_every == int(0.3 / 0.1)
    assert recorder.save_dt == 0.3 and "dt" not in vars(scope)

    writes = []

    class Custom(Callback):
        @property
        def dt(self):
            return self._dt

        @dt.setter
        def dt(self, value):
            writes.append(value)
            self._dt = value
            if value == 0.05:
                raise RuntimeError("custom timestep setter")

    callback = Custom()
    scope = PopulationCallback("left", callback)
    _bind_callback_dt(scope, 0.1)
    with pytest.raises(RuntimeError, match="custom timestep setter"):
        _bind_callback_dt(scope, 0.05)
    assert writes == [0.1, 0.05] and callback.dt == 0.05
    assert "dt" not in vars(scope)


def test_scope_registers_only_inner_callback_parameters_and_buffers():
    callback = Callback()
    callback.register_parameter("weight", torch.nn.Parameter(torch.tensor(2.0)))
    callback.register_buffer("count", torch.tensor(3, dtype=torch.int32))
    scope = PopulationCallback("left", callback)
    assert dict(scope.named_children()) == {"callback": callback}
    assert dict(scope.named_parameters()) == {"callback.weight": callback.weight}
    assert dict(scope.named_buffers()) == {"callback.count": callback.count}
    assert tuple(scope.state_dict()) == ("callback.weight", "callback.count")
    scope.to(dtype=torch.float64)
    assert (
        callback.weight.dtype == torch.float64 and callback.count.dtype == torch.int32
    )


def test_scope_does_not_capture_or_register_network_after_dispatch():
    population = torch.nn.Module()
    network = _network(left=population)
    network_reference = weakref.ref(network)
    scope = PopulationCallback("left", Callback())
    scope.pre_loop_hook(network)
    scope.post_step_hook(network)
    assert list(scope.modules()) == [scope, scope.callback]
    assert all(value is not network for value in vars(scope).values())
    assert all(value is not population for value in vars(scope).values())
    del network
    gc.collect()
    assert network_reference() is None
