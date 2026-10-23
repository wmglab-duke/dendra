"""NetStim source materialization follows Population initialization and caches."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import expsyn, pas
from dendra.models.parametric import PositiveParam
from tests.func.test_network import DT, DTYPE, _clone, _close

pytestmark = pytest.mark.cpu


class _Scale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", scale)

    def forward(self, original):
        return original * self.scale


class _Add(torch.nn.Module):
    def forward(self, value, increment):
        return (value + increment,)


class _OffsetScale(_Scale):
    def forward(self, original):
        return original * (1 + self.scale)


def _network(*, cache=None, config=False, transform=False, seed=812, runtime=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        stimulus = dn.NetStim(
            N=1,
            interval=0.5,
            start=0.0,
            noise=0.6,
            max_spikes=100,
            seed=812 if seed is None else seed,
        )
        if runtime:
            stimulus.interval = PositiveParam(torch.tensor(0.5, dtype=DTYPE))
        cell = dn.SingleCompartment(
            N=1,
            C=1,
            cm=stimulus.interval if runtime else 1.0,
            integrator=dn.bwd_euler_sc(imem=False),
            v_init=-65.0,
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        scale = torch.tensor(0.8, dtype=DTYPE)
        if config:
            register_parametrization(stimulus.interval, "rho", _Scale(scale))
            cell.register_parametrization_in_graph("cm", _Scale(scale))
        elif not runtime:
            cell.material(
                "pool",
                fields={"amount": stimulus.interval},
                min_values={"amount": 0.0},
                conserved={"amount": False},
            )
        model = dn.Network(
            {"cell": cell},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        model.connect_one_to_one(
            stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=DT
        )
        model.train()
        model.initialize(DT)
        model.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            train_delay_backend="dense",
        )
        raw = (
            stimulus.interval.parametrizations.rho.original
            if config
            else stimulus.interval.rho
        )
        if transform:
            raw_name = next(
                name for name, value in cell.named_parameters() if value is raw
            )
            cell.register_post_initialize_transform(
                "adjust_interval",
                _Add(),
                reads=(f"parameters.{raw_name}",),
                writes=(f"parameters.{raw_name}",),
                inputs={"increment": raw.new_tensor(0.4)},
            )
        if cache is not None:
            with torch.no_grad():
                for _ in range(3):
                    model.step()
                if cache == "network":
                    model.cache_state()
                else:
                    cell.cache("_steady_state")
                (scale if config else raw).add_(0.6)
            saved_network = model._state_cache
            saved_population = cell._caches.pop("_steady_state", None)
            model._state_cache = {}
            model.initialize(DT)
            model._state_cache = saved_network
            if saved_population is not None:
                cell._caches["_steady_state"] = saved_population
        # Warm-up caches have a reproducible trajectory in both fixtures;
        # fresh seed=None initialization still continues the captured RNG.
        stimulus.seed = seed
    return model


def _initialize(functional, packet):
    return functional._initialize_host(*packet, functional.initialization_inputs)


def _effective(functional, packet):
    return functional._materialize_netstim_parameters(
        packet.parameters, packet.constants
    )


@pytest.mark.parametrize("config", (False, True))
def test_fresh_initialization_preserves_shared_capacitance_and_netstim_sources(config):
    model = _network(config=config, runtime=True)
    native = _network(config=config, runtime=True)
    native.restore_dict_from_checkpoint(_clone(model.state_dict_for_checkpoint()))
    functional, packet = dn.func.make_functional_network(model)
    initialized = _initialize(functional, packet)
    native.initialize(DT)
    _close(initialized.parameters, functional.extract(native).parameters)
    _close(initialized.state, functional.extract(native).state)
    raw = "parametrizations.rho.original" if config else "rho"
    assert (
        initialized.parameters["populations"]["cell"][f"cm_param.{raw}"]
        is initialized.parameters["netstim"]["parameters"][f"interval.{raw}"]
    )
    actual, _ = functional.prepare_and_step(
        *initialized, netstim_draw=functional.sample_step_draw(initialized.state)
    )
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("seed", (None, 812))
def test_shared_raw_interval_initialization_transform_precedes_netstim_clock_and_rng(
    seed,
):
    model, native = (
        _network(transform=True, seed=seed),
        _network(transform=True, seed=seed),
    )
    native.restore_dict_from_checkpoint(_clone(model.state_dict_for_checkpoint()))
    functional, packet = dn.func.make_functional_network(model)
    before = _clone(packet)
    initialized = _initialize(functional, packet)
    native.initialize(DT)
    _close(initialized.parameters, functional.extract(native).parameters)
    _close(initialized.state, functional.extract(native).state)
    _close(packet, before)
    torch.testing.assert_close(
        _effective(functional, initialized)["interval"], native.netstim.interval()
    )
    draw = functional.sample_step_draw(initialized.state)
    actual, _ = functional.prepare_and_step(*initialized, netstim_draw=draw)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("cache", ("population", "network"))
@pytest.mark.parametrize("config", (False, True))
@pytest.mark.parametrize("seed", (None, 812))
def test_cache_restores_netstim_raw_parameter_and_transform_buffers_before_initialization(
    cache, config, seed
):
    model = _network(cache=cache, config=config, seed=seed)
    native = _network(cache=cache, config=config, seed=seed)
    native.restore_dict_from_checkpoint(_clone(model.state_dict_for_checkpoint()))
    functional, packet = dn.func.make_functional_network(model)
    initialized = _initialize(functional, packet)
    native.initialize(DT)
    expected = functional.extract(native)
    _close(initialized.parameters, expected.parameters)
    _close(initialized.constants["netstim"], expected.constants["netstim"])
    _close(initialized.state, expected.state)
    torch.testing.assert_close(
        _effective(functional, initialized)["interval"], native.netstim.interval()
    )
    actual, _ = functional.prepare_and_step(
        *initialized, netstim_draw=functional.sample_step_draw(initialized.state)
    )
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("cache", ("population", "network"))
def test_cache_restored_raw_interval_has_native_renewal_gradient(cache):
    model, native = _network(cache=cache), _network(cache=cache)
    functional, packet = dn.func.make_functional_network(model)
    original = packet.parameters["netstim"]["parameters"]["interval.rho"]
    upstream = original.detach().clone().requires_grad_()
    parameters = pytree.tree_map_only(
        torch.Tensor,
        lambda value: upstream if value is original else value,
        packet.parameters,
    )
    initialized = _initialize(functional, packet._replace(parameters=parameters))
    assert not initialized.state["netstim"]["next_stoch_time"].requires_grad
    # A late observation guarantees a renewal while retaining the native
    # detached initialization kick and its RNG advancement.
    state = dict(initialized.state)
    time = state["clock"]["t"].new_tensor(10.0)
    state["clock"] = {**state["clock"], "t": time, "origin": time}
    actual, _ = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        state,
        netstim_draw=functional.sample_step_draw(state),
    )
    native.initialize(DT)
    native.netstim(native.netstim.next_stoch_time.new_tensor(10.0), bptt=True, dt=DT)
    observed = torch.autograd.grad(
        actual["netstim"]["next_stoch_time"].sum(), upstream
    )[0]
    expected = torch.autograd.grad(
        native.netstim.next_stoch_time.sum(), native.netstim.interval.rho
    )[0]
    torch.testing.assert_close(observed, expected)
    assert observed.isfinite().all() and observed.abs().sum() > 0


def _noise_cache_network(cache, cached_noise):
    model = _network()
    scale = torch.tensor(cached_noise / 0.6, dtype=DTYPE)
    register_parametrization(model.netstim, "noise", _Scale(scale))
    model.cell.register_parametrization_in_graph("cm", _OffsetScale(scale))
    with torch.no_grad():
        if cache == "network":
            model.cache_state()
        else:
            model.cell.cache("_steady_state")
        scale.fill_(1.0 if cached_noise == 0.0 else 0.0)
    saved_network = model._state_cache
    saved_population = model.cell._caches.pop("_steady_state", None)
    model._state_cache = {}
    model.initialize(DT)
    model._state_cache = saved_network
    if saved_population is not None:
        model.cell._caches["_steady_state"] = saved_population
    return model


@pytest.mark.parametrize("cache", ("population", "network"))
@pytest.mark.parametrize("cached_noise", (0.0, 0.6))
def test_host_initialization_samples_after_cache_restored_noise_transform_buffer(
    cache, cached_noise
):
    model = _noise_cache_network(cache, cached_noise)
    native = _noise_cache_network(cache, cached_noise)
    native.restore_dict_from_checkpoint(_clone(model.state_dict_for_checkpoint()))
    functional, packet = dn.func.make_functional_network(model)
    initialized = _initialize(functional, packet)
    native.initialize(DT)
    _close(initialized.state, functional.extract(native).state)
    effective = _effective(functional, initialized)
    torch.testing.assert_close(
        effective["noise"], effective["noise"].new_full((1,), cached_noise)
    )
    if cached_noise == 0:
        torch.testing.assert_close(
            initialized.state["netstim"]["next_stoch_time"], effective["start"]
        )
