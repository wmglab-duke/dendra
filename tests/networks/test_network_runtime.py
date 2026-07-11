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


def test_network_clock_does_not_miss_exact_netstim_boundaries():
    dt = 0.025
    post = dn.SingleCompartment(N=2, C=1, v_init=-65.0, dtype=torch.float64)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(
        N=2,
        interval=100.0,
        start=[1.0, 100.0],
        noise=0.0,
        max_spikes=1,
        dtype=torch.float64,
    )
    # Exercise both the deterministic renewal clock and the explicit legacy
    # schedule at the same analytically exact boundary.
    stim.schedule(1, 1.0)
    net = dn.Network({"post": post}, netstim=stim, track_netcon_events=True)
    net.connect_one_to_one(
        net.netstim[:],
        net.post[:],
        net.post.mech.syn,
        threshold=None,
        weight=0.1,
        delay=dt,
    )
    net.initialize(dt)
    netcon = next(iter(net.synapses.values()))
    netcon.schedule(con_indices=[0], times_ms=[1.0], weight=2.0)

    spike_steps = []
    netcon_schedule_steps = []
    for step_index in range(42):
        assert net.t.item() == pytest.approx(step_index * dt, abs=1.0e-15)
        net.step()
        if bool(net.netstim.spikes.any()):
            spike_steps.append(step_index)
            assert torch.equal(net.netstim.spikes, torch.tensor([True, True]))
        if bool(netcon.sched_counts.any()):
            netcon_schedule_steps.append(step_index)

    assert spike_steps == [40]
    assert netcon_schedule_steps == [40]
    assert torch.equal(net.netstim.spike_counts, torch.tensor([1, 1]))
    assert net._clock_step.item() == 42
    assert net.t.item() == pytest.approx(42 * dt, abs=1.0e-15)
    assert all(
        pop.t.item() == pytest.approx(net.t.item()) for pop in net.populations.values()
    )
    assert torch.all(net.post.mech.syn.g > 0.0)


def test_nonzero_start_initializes_netcon_absolute_schedule_step():
    post = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=torch.float64)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(
        N=1,
        interval=100.0,
        start=100.0,
        max_spikes=1,
        dtype=torch.float64,
    )
    net = dn.Network({"post": post}, netstim=stim)
    net.connect_one_to_one(
        net.netstim[:],
        net.post[:],
        net.post.mech.syn,
        threshold=None,
        weight=0.1,
        delay=DT,
    )
    net.initialize(DT, t=5.0)
    netcon = next(iter(net.synapses.values()))
    assert netcon.global_step.item() == 50

    netcon.schedule(con_indices=[0], times_ms=[5.0])
    net.step()

    assert netcon.sched_counts.item() == 1
    assert net.t.item() == pytest.approx(5.1)
    assert net.post.t.item() == pytest.approx(5.1)


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


def test_checkpoint_preserves_coherent_clock_anchor_bit_for_bit():
    dt = 0.025
    net = _network()
    net.initialize(dt, t=0.1)
    net.run(157 * dt)
    checkpoint = _clone_nested(net.state_dict_for_checkpoint())

    expected_origin = checkpoint["clock_origin"].clone()
    net.step()
    expected_next_time = net.t.clone()

    net.restore_dict_from_checkpoint(checkpoint)

    assert torch.equal(net._clock_origin, expected_origin)
    net.step()
    assert torch.equal(net.t, expected_next_time)


def test_public_load_preserves_coherent_clock_anchor_bit_for_bit():
    dt = 0.025
    source = _network()
    source.initialize(dt, t=0.1)
    source.run(157 * dt)
    state = _clone_nested(source.state_dict())

    expected_origin = source._clock_origin.clone()
    source.step()
    expected_next_time = source.t.clone()

    restored = _network()
    restored.initialize(dt, t=0.1)
    restored.load(state)

    assert torch.equal(restored._clock_origin, expected_origin)
    restored.step()
    assert torch.equal(restored.t, expected_next_time)


def test_restore_accepts_legacy_flat_event_netcon_checkpoint():
    net = _network()
    net.initialize(DT)
    net.step()
    checkpoint = _clone_nested(net.state_dict_for_checkpoint())
    checkpoint["netcons"] = checkpoint["netcons"]["event"]
    checkpoint.pop("clock_origin")
    checkpoint.pop("clock_step")

    net.step()
    net.restore_dict_from_checkpoint(checkpoint)

    assert net.t.item() == pytest.approx(DT)
    net.step()
    assert net.t.item() == pytest.approx(2 * DT)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("clock_origin", torch.tensor(float("inf")), ValueError),
        ("clock_step", torch.tensor(-1, dtype=torch.long), ValueError),
        ("clock_step", torch.tensor(1.0), TypeError),
    ],
)
def test_checkpoint_clock_metadata_validation_is_atomic(field, value, error):
    net = _network()
    net.initialize(DT)
    net.step()
    before = _clone_nested(net.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)
    corrupt[field] = value

    with pytest.raises(error, match="clock"):
        net.restore_dict_from_checkpoint(corrupt)

    assert net.t.item() == pytest.approx(before["t"].item())
    assert torch.equal(net._clock_origin, before["clock_origin"])
    assert torch.equal(net._clock_step, before["clock_step"])


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
