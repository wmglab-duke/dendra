from __future__ import annotations

import copy
import math

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, graded_syn, pas, sigmoid_release

DT = 0.1
DTYPE = torch.float64
INFERENCE_BACKENDS = ("dense", "sparse_calendar", "bitpacked_history")


def _snapshot(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return copy.deepcopy(value)


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _build_replay_network(backend: str):
    cell = dn.SingleCompartment(N=3, C=1, v_init=-65.0, dtype=DTYPE)
    cell.insert(pas, g=0.001, e=-70.0)
    cell.insert(expsyn.rename("syn"), e=0.0, tau=0.8)
    stim = dn.NetStim(
        N=3,
        interval=torch.tensor([0.27, 0.41, 0.33], dtype=DTYPE),
        start=torch.tensor([0.0, 0.08, 0.14], dtype=DTYPE),
        noise=torch.tensor([0.55, 0.2, 0.75], dtype=DTYPE),
        max_spikes=torch.tensor([12, 10, 11]),
        seed=713,
        dtype=DTYPE,
    )
    # Heap schedules exercise consumed Python-side state in addition to the
    # stochastic renewal generator state.
    stim.schedule([0, 1, 2, 0], [0.2, 0.5, 0.9, 1.1])
    net = dn.Network(
        {"cell": cell},
        netstim=stim,
        track_netcon_events=False,
        netcon_delay_backend=backend,
    )
    net.connect_dense(
        stim[:],
        cell[:],
        cell.mech.syn,
        threshold=None,
        weight=torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=DTYPE),
        delay=torch.tensor([0.0, 0.1, 0.4, 0.2, 0.5, 0.1, 0.4, 0.3, 0.0], dtype=DTYPE),
    )
    # Duplicate source/target edges and a maximum-delay delivery make the
    # pending state impossible to reconstruct from only current synapse state.
    net.connect_one_to_one(
        stim[:],
        cell[:],
        cell.mech.syn,
        threshold=None,
        weight=torch.tensor([0.17, -0.11, 0.23], dtype=DTYPE),
        delay=torch.tensor([0.5, 0.0, 0.3], dtype=DTYPE),
    )
    net.initialize(DT)
    netcon = next(iter(net.synapses.values()))
    netcon.schedule(
        con_indices=[0, 7, 11, 2],
        times_ms=[0.4, 0.8, 1.2, 1.5],
        weight=torch.tensor([0.5, -0.25, 1.3, 0.7], dtype=DTYPE),
    )
    return net


def _runtime_trace(net, n_steps: int):
    netcon = next(iter(net.synapses.values()))
    trace = []
    for _ in range(n_steps):
        net.step()
        trace.append(
            {
                "t": net.t.detach().clone(),
                "v": net.cell.v.detach().clone(),
                "g": net.cell.mech.syn.g.detach().clone(),
                "spikes": net.netstim.spikes.detach().clone(),
                "spike_counts": net.netstim.spike_counts.detach().clone(),
                "next_stoch_time": net.netstim.next_stoch_time.detach().clone(),
                "current_time_step": netcon.current_time_step.detach().clone(),
                "global_step": netcon.global_step.detach().clone(),
            }
        )
    return trace


@pytest.mark.parametrize("backend", INFERENCE_BACKENDS)
def test_fresh_network_checkpoint_matches_uninterrupted_stochastic_replay(backend):
    split = 7
    continuation_steps = 13

    uninterrupted = _build_replay_network(backend)
    expected = _runtime_trace(uninterrupted, split + continuation_steps)[split:]

    source = _build_replay_network(backend)
    _runtime_trace(source, split)
    checkpoint = _snapshot(source.state_dict_for_checkpoint())

    resumed = _build_replay_network(backend)
    resumed.restore_dict_from_checkpoint(checkpoint)
    actual = _runtime_trace(resumed, continuation_steps)

    _assert_nested_equal(actual, expected)
    assert resumed.t.item() == pytest.approx((split + continuation_steps) * DT)


@pytest.mark.parametrize("backend", ["sparse_calendar", "bitpacked_history"])
def test_compact_backend_checkpoint_is_independent_of_later_calendar_mutation(backend):
    source = _build_replay_network(backend)
    _runtime_trace(source, 4)
    checkpoint = source.state_dict_for_checkpoint()
    frozen = _snapshot(checkpoint["netcons"])

    _runtime_trace(source, 5)

    _assert_nested_equal(checkpoint["netcons"], frozen)


def test_fresh_network_topology_mismatch_is_rejected_without_mutation():
    source = _build_replay_network("dense")
    _runtime_trace(source, 4)
    checkpoint = _snapshot(source.state_dict_for_checkpoint())

    incompatible = _build_replay_network("dense")
    # Prime a structure snapshot before changing topology. The digest must not
    # be memoized across a later buffer rebind.
    incompatible.state_dict_for_checkpoint()
    netcon = next(iter(incompatible.synapses.values()))
    # Keep every runtime-buffer shape unchanged so validation must examine the
    # actual source/target topology rather than merely tensor dimensions.
    netcon.post_idx = netcon.post_idx.roll(1)
    before = _snapshot(incompatible.state_dict_for_checkpoint())

    with pytest.raises(ValueError, match="topology or runtime mode"):
        incompatible.restore_dict_from_checkpoint(checkpoint)

    _assert_nested_equal(incompatible.state_dict_for_checkpoint(), before)


def _build_passive_network(dt):
    cell = dn.SingleCompartment(N=2, C=1, v_init=-55.0, dtype=DTYPE)
    cell.insert(pas, g=0.004, e=-70.0)
    net = dn.Network({"cell": cell})
    net.initialize(dt)
    return net


def test_fresh_network_timestep_mismatch_is_rejected_atomically():
    source = _build_passive_network(0.1)
    source.step()
    checkpoint = _snapshot(source.state_dict_for_checkpoint())

    incompatible = _build_passive_network(0.2)
    before = _snapshot(incompatible.state_dict_for_checkpoint())

    with pytest.raises(ValueError, match="topology or runtime mode"):
        incompatible.restore_dict_from_checkpoint(checkpoint)

    _assert_nested_equal(incompatible.state_dict_for_checkpoint(), before)


def test_fresh_network_training_mode_mismatch_is_rejected_atomically():
    source = _build_passive_network(0.1)
    source.train()
    checkpoint = _snapshot(source.state_dict_for_checkpoint())

    incompatible = _build_passive_network(0.1)
    incompatible.eval()
    before = _snapshot(incompatible.state_dict_for_checkpoint())

    with pytest.raises(ValueError, match="topology or runtime mode"):
        incompatible.restore_dict_from_checkpoint(checkpoint)

    _assert_nested_equal(incompatible.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("entrypoint", ["initialize", "build"])
@pytest.mark.parametrize("invalid", [0.0, -0.1, math.inf, math.nan, True])
def test_network_rejects_invalid_dt_before_mutation(entrypoint, invalid):
    net = _build_passive_network(0.1)
    net.step()
    before = _snapshot(net.state_dict_for_checkpoint())

    with pytest.raises((TypeError, ValueError), match="dt"):
        getattr(net, entrypoint)(invalid)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan, True])
def test_network_rejects_invalid_start_time_before_mutation(invalid):
    net = _build_passive_network(0.1)
    net.step()
    before = _snapshot(net.state_dict_for_checkpoint())

    with pytest.raises((TypeError, ValueError), match="t"):
        net.initialize(0.1, t=invalid)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("start", [-2.5, 5.0])
def test_network_nonzero_start_keeps_all_runtime_clocks_synchronized(start):
    first = dn.SingleCompartment(N=1, C=1, v_init=-60.0, dtype=DTYPE)
    second = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    first.insert(pas, g=0.001, e=-70.0)
    second.insert(pas, g=0.002, e=-70.0)
    stim = dn.NetStim(
        N=1,
        interval=100.0,
        start=100.0,
        noise=0.0,
        max_spikes=1,
        dtype=DTYPE,
    )
    net = dn.Network({"first": first, "second": second}, netstim=stim)
    net.initialize(0.1, t=start)

    assert net.t.item() == pytest.approx(start)
    assert all(pop.t.item() == pytest.approx(start) for pop in net.populations.values())

    net.step()

    assert net.t.item() == pytest.approx(start + 0.1)
    assert all(
        pop.t.item() == pytest.approx(start + 0.1) for pop in net.populations.values()
    )
    assert net.netstim.t_last.item() == pytest.approx(start)


@pytest.mark.parametrize("entrypoint", ["run", "checkpointed"])
@pytest.mark.parametrize("invalid", [-0.1, math.inf, math.nan, True])
def test_network_rejects_invalid_horizons_before_mutation(entrypoint, invalid):
    net = _build_passive_network(0.1)
    before = _snapshot(net.state_dict_for_checkpoint())

    with pytest.raises((TypeError, ValueError), match="tstop"):
        if entrypoint == "run":
            net.run(invalid)
        else:
            net.longrun_checkpointed(invalid, chunklength=2)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_network_checkpointed_run_rejects_invalid_chunklength_before_mutation(invalid):
    net = _build_passive_network(0.1)
    before = _snapshot(net.state_dict_for_checkpoint())

    with pytest.raises((TypeError, ValueError), match="chunklength"):
        net.longrun_checkpointed(0.2, chunklength=invalid)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("backend", ["sparse_calendar", "bitpacked_history"])
def test_malformed_compact_runtime_checkpoint_rolls_back_whole_network(backend):
    net = _build_replay_network(backend)
    _runtime_trace(net, 4)
    before = _snapshot(net.state_dict_for_checkpoint())
    corrupt = _snapshot(before)
    name = next(iter(corrupt["netcons"]["event"]))
    corrupt["populations"]["cell"]["integrator"]["v"].add_(9.0)
    backend_state = corrupt["netcons"]["event"][name]["backend_state"]
    if backend == "sparse_calendar":
        backend_state["sparse_calendar"] = {
            1: [(torch.tensor([0, 1]), torch.tensor([1.0], dtype=DTYPE))]
        }
        error = "equally sized"
    else:
        backend_state["spike_history_packed"] = torch.zeros((1, 1), dtype=torch.int64)
        error = "spike_history_packed.*shape"

    with pytest.raises(ValueError, match=error):
        net.restore_dict_from_checkpoint(corrupt)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


def test_nested_netstim_shape_is_preflighted_before_network_mutation():
    net = _build_replay_network("dense")
    _runtime_trace(net, 4)
    before = _snapshot(net.state_dict_for_checkpoint())
    corrupt = _snapshot(before)
    corrupt["populations"]["cell"]["integrator"]["v"].add_(9.0)
    # The flat size is unchanged, so heap-count validation alone cannot catch
    # this incompatible leading-dimension layout.
    corrupt["netstim"]["shape"] = (1, 3)

    with pytest.raises(ValueError, match="nested NetStim shape"):
        net.restore_dict_from_checkpoint(corrupt)

    assert net.netstim.shape == before["netstim"]["shape"]
    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


def _build_continuous_replay_network():
    pre = dn.SingleCompartment(N=2, C=1, v_init=-35.0, dtype=DTYPE)
    pre.insert(pas, g=0.003, e=-20.0)
    post = dn.SingleCompartment(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(pas, g=0.001, e=-70.0)
    post.insert(graded_syn, e=-10.0, g_scale=0.02)
    net = dn.Network({"pre": pre, "post": post})
    net.connect_continuous_one_to_one(
        pre[:],
        post[:],
        post.mech.graded_syn,
        pre_var="v",
        input="g_pre",
        weight=torch.tensor([0.4, 0.9], dtype=DTYPE),
        delay=torch.tensor([0.1, 0.4], dtype=DTYPE),
        transform=sigmoid_release(theta=-30.0, sigma=3.0),
    )
    net.initialize(DT)
    return net


def _continuous_trace(net, n_steps):
    trace = []
    for _ in range(n_steps):
        net.step()
        trace.append(
            torch.cat(
                [
                    net.pre.v.detach().reshape(-1),
                    net.post.v.detach().reshape(-1),
                    net.post.mech.graded_syn.g_pre.detach().reshape(-1),
                ]
            )
        )
    return torch.stack(trace)


def test_fresh_network_replay_restores_pending_continuous_deliveries():
    split = 3
    continuation_steps = 8
    uninterrupted = _build_continuous_replay_network()
    expected = _continuous_trace(uninterrupted, split + continuation_steps)[split:]

    source = _build_continuous_replay_network()
    _continuous_trace(source, split)
    checkpoint = _snapshot(source.state_dict_for_checkpoint())
    resumed = _build_continuous_replay_network()
    resumed.restore_dict_from_checkpoint(checkpoint)

    actual = _continuous_trace(resumed, continuation_steps)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def _build_training_network(train_backend: str):
    cell = dn.SingleCompartment(N=3, C=1, v_init=-65.0, dtype=DTYPE)
    cell.insert(pas, g=0.001, e=-70.0)
    cell.insert(expsyn.rename("syn"), e=0.0, tau=0.9)
    stim = dn.NetStim(
        N=3,
        interval=torch.tensor([0.3, 0.4, 0.5], dtype=DTYPE),
        start=torch.tensor([0.0, 0.1, 0.2], dtype=DTYPE),
        noise=0.0,
        max_spikes=20,
        seed=19,
        dtype=DTYPE,
    )
    stim.schedule([0, 1, 2], [0.7, 0.8, 1.0])
    net = dn.Network(
        {"cell": cell},
        netstim=stim,
        netcon_delay_backend="dense",
        netcon_train_backend=train_backend,
    )
    net.connect_one_to_one(
        stim[:],
        cell[:],
        cell.mech.syn,
        threshold=None,
        weight=torch.nn.Parameter(torch.tensor([0.45, 0.75, 1.1], dtype=DTYPE)),
        delay=torch.nn.Parameter(torch.tensor([0.14, 0.27, 0.43], dtype=DTYPE)),
    )
    net.train()
    net.initialize(DT)
    net.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=True,
        diff_spiking=False,
        taps=2,
        diff_scheduled_times=False,
        train_delay_backend=train_backend,
    )
    net.init_synapses(reinit_weights=False, reinit_delays=False)
    return net


def _training_trace(net, n_steps: int):
    values = []
    for _ in range(n_steps):
        net.step()
        values.append(
            torch.cat([net.cell.v.reshape(-1), net.cell.mech.syn.g.reshape(-1)], dim=0)
        )
    return torch.stack(values)


@pytest.mark.parametrize("train_backend", ["dense", "source_history"])
def test_fresh_training_replay_matches_truncated_reference_and_gradients(train_backend):
    split = 5
    continuation_steps = 9
    reference = _build_training_network(train_backend)
    _training_trace(reference, split)
    boundary = _snapshot(reference.state_dict_for_checkpoint())

    # Restore a detached boundary into both runs. This defines truncated-BPTT
    # semantics and ensures dense pending payloads do not retain an accidental
    # link to the reference model's prefix graph.
    reference.restore_dict_from_checkpoint(_snapshot(boundary))
    reference_netcon = next(iter(reference.synapses.values()))
    reference_netcon.w.retain_grad()
    reference_netcon.delay_ms.w.retain_grad()
    expected = _training_trace(reference, continuation_steps)

    resumed = _build_training_network(train_backend)
    resumed.restore_dict_from_checkpoint(_snapshot(boundary))
    resumed_netcon = next(iter(resumed.synapses.values()))
    resumed_netcon.w.retain_grad()
    resumed_netcon.delay_ms.w.retain_grad()
    actual = _training_trace(resumed, continuation_steps)

    coefficients = torch.linspace(-0.4, 1.2, actual.numel(), dtype=DTYPE).reshape_as(
        actual
    )
    expected_loss = (expected * coefficients).sum()
    actual_loss = (actual * coefficients).sum()
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(resumed_netcon.w.grad, reference_netcon.w.grad)
    torch.testing.assert_close(
        resumed_netcon.delay_ms.w.grad, reference_netcon.delay_ms.w.grad
    )
    assert resumed.t.item() == pytest.approx((split + continuation_steps) * DT)


@pytest.mark.parametrize("train_backend", ["dense", "source_history"])
def test_runtime_checkpoint_does_not_silently_overwrite_weights_or_delays(
    train_backend,
):
    source = _build_training_network(train_backend)
    _training_trace(source, 3)
    checkpoint = _snapshot(source.state_dict_for_checkpoint())

    resumed = _build_training_network(train_backend)
    netcon = next(iter(resumed.synapses.values()))
    with torch.no_grad():
        netcon.w.add_(2.0)
        netcon.delay_ms.w.add_(0.05)
    expected_weight = netcon.w.detach().clone()
    expected_delay = netcon.delay_ms.w.detach().clone()

    resumed.restore_dict_from_checkpoint(checkpoint)

    assert torch.equal(netcon.w, expected_weight)
    assert torch.equal(netcon.delay_ms.w, expected_delay)


class _PureReplayLoss(dn.callbacks.Callback):
    def post_step_hook(self, model):
        scale = 1.0 + model.t.to(dtype=DTYPE)
        return scale * (model.cell.mech.syn.g.sum() + 0.001 * model.cell.v.sum())


@pytest.mark.parametrize("train_backend", ["dense", "source_history"])
def test_checkpointed_training_matches_eager_and_restores_forward_state_after_backward(
    train_backend,
):
    n_steps = 11
    eager = _build_training_network(train_backend)
    eager_netcon = next(iter(eager.synapses.values()))
    eager_netcon.w.retain_grad()
    eager_netcon.delay_ms.w.retain_grad()
    eager_loss = torch.zeros((), dtype=DTYPE)
    eager_callback = _PureReplayLoss()
    for _ in range(n_steps):
        eager.step()
        eager_loss = eager_loss + eager_callback.post_step_hook(eager)
    eager_final = _snapshot(eager.state_dict_for_checkpoint())
    eager_loss.backward()

    checkpointed = _build_training_network(train_backend)
    checkpointed_netcon = next(iter(checkpointed.synapses.values()))
    checkpointed_netcon.w.retain_grad()
    checkpointed_netcon.delay_ms.w.retain_grad()
    loss, final_state = checkpointed.longrun_checkpointed(
        n_steps * DT,
        chunklength=4,
        callbacks=[_PureReplayLoss()],
        safe_checkpoint=True,
        restore_state_after_backward=True,
        return_final_state=True,
    )
    frozen_final = _snapshot(final_state)

    torch.testing.assert_close(loss, eager_loss, atol=2e-12, rtol=2e-12)
    # The private seeder is intentionally nondeterministic at construction and
    # is used only to seed a newly created device generator. The active RNG is
    # deterministically seeded here and is compared as part of the checkpoint.
    eager_final["netstim"].pop("seeder_state")
    comparable_final = _snapshot(frozen_final)
    comparable_final["netstim"].pop("seeder_state")
    _assert_nested_equal(comparable_final, eager_final)

    loss.backward()

    torch.testing.assert_close(checkpointed_netcon.w.grad, eager_netcon.w.grad)
    torch.testing.assert_close(
        checkpointed_netcon.delay_ms.w.grad, eager_netcon.delay_ms.w.grad
    )
    _assert_nested_equal(checkpointed.state_dict_for_checkpoint(), frozen_final)
