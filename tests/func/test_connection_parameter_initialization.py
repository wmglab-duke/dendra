"""Connection sources materialize after native Population initialization phases."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import graded_syn, pas
from dendra.models.parametric import PositiveParam
from tests.func.test_network import DT, DTYPE, _close
from tests.networks.test_functional_event_delivery import ThresholdRelay

pytestmark = pytest.mark.cpu


class _Add(torch.nn.Module):
    def forward(self, value, amount):
        return (value + amount,)


class _Scale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", scale)

    def forward(self, value):
        return value * self.scale


class _BufferedSource(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.rho = torch.nn.Parameter(torch.tensor(-1.5, dtype=DTYPE))
        self.register_buffer("scale", scale)

    def forward(self):
        return torch.nn.functional.softplus(self.rho) * self.scale


def _network(*, cache=None, transform=False, config=False, event_delay=DT):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        scale = torch.tensor(0.8, dtype=DTYPE)
        source = (
            _BufferedSource(scale)
            if config
            else PositiveParam(torch.tensor(0.2, dtype=DTYPE))
        )
        cell = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(ThresholdRelay.rename("relay"))
        cell.insert(graded_syn.rename("analog"), e=-10.0)
        if config:
            cell.register_parametrization_in_graph("cm", _Scale(scale))
        else:
            cell.material(
                "pool",
                fields={"amount": source},
                min_values={"amount": 0.0},
                conserved={"amount": False},
            )
        model = dn.Network(
            {"cell": cell},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        model.connect_one_to_one(
            cell[:],
            cell[:],
            cell.mech.relay,
            pre_var="mech.relay.level",
            weight=source,
            delay=event_delay,
            allow_autapses=True,
        )
        model.connect_continuous_one_to_one(
            cell[:],
            cell[:],
            cell.mech.analog,
            pre_var="mech.relay.level",
            input="g_pre",
            weight=source,
            delay=0.0,
            transform=torch.sigmoid,
            allow_autapses=True,
        )
        model.train()
        model.initialize(DT)
        model.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
        )
        if transform:
            raw_name = next(
                name for name, value in cell.named_parameters() if value is source.rho
            )
            cell.register_pre_initialize_transform(
                "adjust",
                _Add(),
                reads=(f"parameters.{raw_name}",),
                writes=(f"parameters.{raw_name}",),
                inputs={"amount": source.rho.new_tensor(0.35)},
            )
        if cache is not None:
            if cache == "network":
                model._enable_synapse_state_cache_recording()
            with torch.no_grad():
                for _ in range(3):
                    model.step()
                if cache == "network":
                    model.cache_state()
                else:
                    cell.cache("_steady_state")
                if config:
                    scale.add_(0.4)
                else:
                    source.rho.add_(0.6)
            if cache == "network":
                model._disable_synapse_state_cache_recording(release=True)
            saved_network = model._state_cache
            saved_population = cell._caches.pop("_steady_state", None)
            model._state_cache = {}
            model.initialize(DT)
            model._state_cache = saved_network
            if saved_population is not None:
                cell._caches["_steady_state"] = saved_population
    return model, source


def _materializations(functional, packet):
    return functional._materialize_connections(packet.parameters, packet.constants)


def _native_materializations(model):
    return {
        collection: {
            name: connection.weight()
            for name, connection in getattr(model, collection).items()
        }
        for collection in ("synapses", "continuous_synapses")
    }


def _check_weights(functional, packet, native):
    actual = _materializations(functional, packet)
    for collection, connections in _native_materializations(native).items():
        for name, value in connections.items():
            torch.testing.assert_close(actual[collection][name]["weights"], value)


def _replace_raw(packet, replacement):
    name = next(iter(packet.parameters["synapses"]))
    original = packet.parameters["synapses"][name]["weight.weights.0.param.rho"]
    parameters = pytree.tree_map_only(
        torch.Tensor,
        lambda value: replacement if value is original else value,
        packet.parameters,
    )
    return packet._replace(parameters=parameters)


@pytest.mark.parametrize("flag", ("reinit_weights", "reinit_delays"))
@pytest.mark.parametrize("value", (0, 1, None, torch.tensor(True)))
def test_connection_reinitialization_flags_require_plain_bools(flag, value):
    functional, packet = dn.func.make_functional_network(_network()[0])
    inputs = functional.initialization_inputs._replace(**{flag: value})
    with pytest.raises(TypeError, match=f"initialization.{flag} must be a bool"):
        functional.initialize(*packet, inputs)


@pytest.mark.parametrize("delay", (0.0, 0.2 * DT))
def test_fixed_event_delay_preserves_native_minimum_one_step_routing(delay):
    source, _ = _network(event_delay=delay)
    native, _ = _network(event_delay=delay)
    functional, packet = dn.func.make_functional_network(source)
    initialized = functional.initialize(*packet, functional.initialization_inputs)
    native.initialize(DT)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("reinit", (False, True))
def test_population_source_writes_precede_connection_materialization(reinit):
    source, _ = _network(transform=True)
    native, _ = _network(transform=True)
    functional, packet = dn.func.make_functional_network(source)
    initialized = functional.initialize(
        *packet, functional.initialization_inputs._replace(reinit_weights=reinit)
    )
    native.initialize(DT, reinit_weights=reinit)
    _check_weights(functional, initialized, native)
    _close(initialized.parameters, functional.extract(native).parameters)


def test_refreshed_materialization_is_retained_by_later_initialization():
    source, _ = _network()
    native, native_raw = _network()
    functional, packet = dn.func.make_functional_network(source)
    upstream = native_raw.rho.detach().clone().requires_grad_()
    first_raw = upstream + 0.4
    changed = _replace_raw(packet, first_raw)
    refreshed = functional.initialize(*changed, functional.initialization_inputs)
    with torch.no_grad():
        native_raw.rho.copy_(first_raw)
    native.initialize(DT)
    _check_weights(functional, refreshed, native)
    effects = _materializations(functional, refreshed)
    actual_loss = sum(
        value["weights"].sum()
        for connections in effects.values()
        for value in connections.values()
    )
    native_loss = sum(
        value.sum()
        for connections in _native_materializations(native).values()
        for value in connections.values()
    )
    observed = torch.autograd.grad(actual_loss, upstream)[0]
    expected = torch.autograd.grad(native_loss, native_raw.rho)[0]
    torch.testing.assert_close(observed, expected)
    later = _replace_raw(refreshed, first_raw + 0.6)
    retained = functional.initialize(
        *later, functional.initialization_inputs._replace(reinit_weights=False)
    )
    with torch.no_grad():
        native_raw.rho.add_(0.6)
    native.initialize(DT, reinit_weights=False)
    _check_weights(functional, retained, native)
    for collection in effects:
        for name in effects[collection]:
            torch.testing.assert_close(
                _materializations(functional, retained)[collection][name]["weights"],
                effects[collection][name]["weights"],
            )


@pytest.mark.parametrize("compatible", (False, True))
@pytest.mark.parametrize("refresh", (False, True))
def test_extraction_retains_current_native_materialization_and_source_baseline(
    compatible, refresh
):
    source, source_raw = _network()
    functional, _ = dn.func.make_functional_network(source)
    native, raw = _network() if compatible else (source, source_raw)
    with torch.no_grad():
        raw.rho.add_(0.4)
    if refresh:
        for collection in (native.synapses, native.continuous_synapses):
            for connection in collection.values():
                connection.weight.init(True)
    packet = functional.extract(native)
    _check_weights(functional, packet, native)
    retained = functional.initialize(
        *packet, functional.initialization_inputs._replace(reinit_weights=False)
    )
    _check_weights(functional, retained, native)


@pytest.mark.parametrize("cache", ("population", "network"))
@pytest.mark.parametrize("reinit", (False, True))
@pytest.mark.parametrize("config", (False, True))
def test_population_cache_replays_connection_raw_sources_before_refresh(
    cache, reinit, config
):
    source, _ = _network(cache=cache, config=config)
    native, _ = _network(cache=cache, config=config)
    functional, packet = dn.func.make_functional_network(source)
    initialized = functional.initialize(
        *packet, functional.initialization_inputs._replace(reinit_weights=reinit)
    )
    native.initialize(DT, reinit_weights=reinit)
    _check_weights(functional, initialized, native)
    _close(initialized.parameters, functional.extract(native).parameters)
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("cache", ("population", "network"))
def test_cache_restored_raw_parameters_keep_native_refresh_gradients(cache):
    source, _ = _network(cache=cache)
    native, native_source = _network(cache=cache)
    functional, packet = dn.func.make_functional_network(source)
    upstream = native_source.rho.detach().clone().requires_grad_()
    initialized = functional.initialize(
        *_replace_raw(packet, upstream), functional.initialization_inputs
    )
    native.initialize(DT)
    actual_loss = sum(
        value["weights"].sum()
        for connections in _materializations(functional, initialized).values()
        for value in connections.values()
    )
    native_loss = sum(
        value.sum()
        for connections in _native_materializations(native).values()
        for value in connections.values()
    )
    observed = torch.autograd.grad(actual_loss, upstream)[0]
    expected = torch.autograd.grad(native_loss, native_source.rho)[0]
    torch.testing.assert_close(observed, expected)
    assert observed.isfinite().all() and observed.abs().sum() > 0


@pytest.mark.parametrize("cache", ("population", "network"))
def test_cache_copy_into_native_effective_storage_is_retained_when_refresh_is_false(
    cache,
):
    def model():
        network, source = _network(config=True)
        cell = network.cell
        # Native retained initialized values can be views of buffers copied by
        # Population caches, independently of the raw parameter-source path.
        for collection in (network.synapses, network.continuous_synapses):
            for connection in collection.values():
                connection.weight.w = source.scale.reshape(1)
        with torch.no_grad():
            if cache == "network":
                network.cache_state()
            else:
                cell.cache("_steady_state")
            source.scale.add_(0.4)
        saved_population = cell._caches.pop("_steady_state", None)
        # Keep the native Population workspace coherent without breaking the
        # deliberately shared connection w storage through Network.initialize.
        cell.initialize()
        cell.integrator._initialize(
            cell, cell.v.new_tensor(DT), force=True, compile_scope="network_population"
        )
        if saved_population is not None:
            cell._caches["_steady_state"] = saved_population
        return network

    source, native = model(), model()
    functional, packet = dn.func.make_functional_network(source)
    initialized = functional.initialize(
        *packet, functional.initialization_inputs._replace(reinit_weights=False)
    )
    native.initialize(DT, reinit_weights=False)
    _check_weights(functional, initialized, native)
