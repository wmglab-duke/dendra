import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.1


class HookCounter(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.pre_loop = 0
        self.pre_step = 0
        self.post_step = 0
        self.post_loop = 0

    def pre_loop_hook(self, model):
        self.pre_loop += 1

    def pre_step_hook(self, model):
        self.pre_step += 1

    def post_step_hook(self, model):
        self.post_step += 1

    def post_loop_hook(self, model):
        self.post_loop += 1


def _network(*, with_netstim=True):
    cell = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=torch.float64)
    cell.insert(pas, g=0.001, e=-70.0)
    stim = None
    if with_netstim:
        cell.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
        stim = dn.NetStim(
            N=1,
            interval=0.2,
            start=0.0,
            max_spikes=2,
            dtype=torch.float64,
        )
    net = dn.Network({"cell": cell}, netstim=stim, track_netcon_events=True)
    if stim is not None:
        net.connect_one_to_one(
            net.netstim[:],
            net.cell[:],
            net.cell.mech.syn,
            threshold=None,
            weight=0.5,
            delay=0.1,
        )
    return net


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return value


def test_run_invokes_complete_callback_lifecycle_each_step():
    net = _network(with_netstim=False)
    net.initialize(DT)
    callback = HookCounter()

    net.run(0.3, callbacks=[callback])

    assert callback.pre_loop == 1
    assert callback.pre_step == 3
    assert callback.post_step == 3
    assert callback.post_loop == 1
    assert callback.dt == DT


def test_whole_network_checkpoint_restores_runtime_state():
    net = _network()
    net.initialize(DT)
    net.step()
    net.step()
    checkpoint = _clone_nested(net.state_dict_for_checkpoint())

    expected_t = checkpoint["t"].clone()
    expected_v = checkpoint["populations"]["cell"]["integrator"]["v"].clone()
    synapse_name = next(iter(checkpoint["netcons"]["event"]))
    expected_delivery = checkpoint["netcons"]["event"][synapse_name][
        "delivery_buffer"
    ].clone()
    expected_spike_counts = checkpoint["netstim"]["spike_counts"].clone()

    net.step()
    net.restore_dict_from_checkpoint(checkpoint)

    assert torch.equal(net.t, expected_t)
    assert torch.equal(net.cell.v, expected_v)
    assert torch.equal(net.synapses[synapse_name].delivery_buffer, expected_delivery)
    assert torch.equal(net.netstim.spike_counts, expected_spike_counts)


def test_restore_accepts_legacy_flat_event_netcon_checkpoint():
    net = _network()
    net.initialize(DT)
    net.step()
    checkpoint = _clone_nested(net.state_dict_for_checkpoint())
    checkpoint["netcons"] = checkpoint["netcons"]["event"]

    net.step()
    net.restore_dict_from_checkpoint(checkpoint)

    assert net.t.item() == pytest.approx(DT)


def test_cache_state_is_detached_from_live_population_state():
    net = _network(with_netstim=False)
    net.initialize(DT)
    net.cache_state()
    cached_v = net._state_cache["cell"]["v"].clone()

    with torch.no_grad():
        net.cell.v.add_(10.0)

    assert torch.equal(net._state_cache["cell"]["v"], cached_v)


def test_state_cache_can_be_loaded_cleared_and_reinitialized():
    source = _network(with_netstim=False)
    source.initialize(DT)
    source.step()
    source.cache_state()

    restored = _network(with_netstim=False)
    restored.load_state_cache(source._state_cache, source._syn_cache)
    restored.initialize(DT)
    assert torch.allclose(restored.cell.v, source.cell.v)
    assert restored.cell.initializing_from_state_cache

    restored.clear_state_cache()
    assert restored._state_cache == {}
    assert restored._syn_cache == {}


def test_steady_state_populates_cache_and_restores_training_mode():
    net = _network(with_netstim=False)
    net.train()
    assert net.training

    returned = net.steady_state(tstop=0.2, dt=DT)

    assert returned is net
    assert net.training
    assert net.t.item() == pytest.approx(0.0)
    assert "cell" in net._state_cache
    assert set(net._syn_cache) == {"event", "continuous"}
