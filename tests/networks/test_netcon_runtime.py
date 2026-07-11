import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn

DT = 0.1


def _built_netcon(
    *,
    delays=(0.0, 0.1, 0.2),
    weights=(1.0, 2.0, 3.0),
    delay_backend="dense",
    track_events=True,
):
    post = dn.Population(N=1, C=3, v_init=-65.0, dtype=torch.float64)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(N=3)
    net = dn.Network(
        {"post": post},
        netstim=stim,
        track_netcon_events=track_events,
        netcon_delay_backend=delay_backend,
    )
    net.connect_one_to_one(
        net.netstim[:],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=torch.as_tensor(weights, dtype=torch.float64),
        delay=torch.as_tensor(delays, dtype=torch.float64),
    )
    net.build(DT)
    net.init_synapses()
    return net, next(iter(net.synapses.values()))


@pytest.mark.parametrize(
    "delays, expected",
    [
        ((0.0, 0.0, 0.0), "advance_non_diff_dense_uniform"),
        ((0.1, 0.1, 0.1), "advance_non_diff_dense_uniform"),
        ((0.1, 0.2, 0.3), "advance_non_diff_dense_mixed"),
        ((0.0, 0.1, 0.1), "advance_non_diff_dense_uniform"),
        ((0.0, 0.1, 0.2), "advance_non_diff_dense_mixed"),
    ],
)
def test_dense_backend_selects_delay_specialization(delays, expected):
    _, netcon = _built_netcon(delays=delays)
    assert netcon.advance.__name__ == expected
    assert netcon.delay_steps.tolist() == [round(delay / DT) for delay in delays]
    assert netcon.numel() == 3
    assert torch.allclose(netcon.w, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


def test_sparse_backend_uses_calendar_storage_and_exports_checkpoint_state():
    _, netcon = _built_netcon(delay_backend="sparse_calendar")
    assert netcon.advance.__name__ == "advance_non_diff_sparse_calendar"
    assert netcon.delivery_buffer.shape == (1, 3)

    state = netcon.state_dict_for_checkpoint()
    assert state["backend_state"]["kind"] == "sparse_calendar"
    assert state["backend_state"]["sparse_calendar"] == {}


def test_expand_pre_indices_handles_empty_missing_and_known_sources():
    _, netcon = _built_netcon()
    assert netcon._expand_pre_to_con([]).numel() == 0
    assert netcon._expand_pre_to_con([1]).tolist() == [1]
    assert netcon._expand_pre_to_con([0, 2]).tolist() == [0, 2]
    assert netcon._expand_pre_to_con([99]).numel() == 0


def test_value_schedule_filters_past_events_and_keeps_metadata_aligned():
    _, netcon = _built_netcon()
    netcon.global_step.fill_(2)
    netcon.schedule(
        con_indices=[0, 1, 2],
        times_ms=[0.1, 0.2, 0.3],
        weight=torch.tensor([10.0, 20.0, 30.0]),
    )

    assert netcon.sched_con_idx.tolist() == [1, 2]
    assert netcon.sched_abs_step.tolist() == [2, 3]
    assert netcon.sched_weight.tolist() == [20.0, 30.0]
    assert netcon.sched_time_ms.tolist() == pytest.approx([0.2, 0.3])
    assert netcon.sched_weight_idx.tolist() == [-1, -1]
    assert netcon.sched_time_idx.tolist() == [-1, -1]

    gate, counts = netcon._scheduled_gate_this_step(
        torch.tensor(2), use_tri_kernel=False
    )
    assert gate.tolist() == [0.0, 20.0, 0.0]
    assert counts.tolist() == [0, 1, 0]

    # An all-past append is a no-op.
    netcon.schedule(con_indices=[0], times_ms=[0.0], weight=99.0)
    assert netcon.sched_con_idx.tolist() == [1, 2]


def test_value_schedule_supports_pre_indices_scalars_and_validation():
    _, netcon = _built_netcon()
    netcon.schedule(pre_indices=[0, 2], times_ms=[0.1, 0.2], weight=2.5)
    assert netcon.sched_con_idx.tolist() == [0, 2]
    assert netcon.sched_weight.tolist() == [2.5, 2.5]

    with pytest.raises(ValueError, match="exactly one"):
        netcon.schedule(times_ms=[0.1])
    with pytest.raises(ValueError, match="exactly one"):
        netcon.schedule(con_indices=[0], pre_indices=[0], times_ms=[0.1])
    with pytest.raises(ValueError, match="times_ms is required"):
        netcon.schedule(con_indices=[0])
    with pytest.raises(ValueError, match="must match length"):
        netcon.schedule(con_indices=[0, 1], times_ms=[0.1])
    with pytest.raises(IndexError, match="out of range"):
        netcon.schedule(con_indices=[3], times_ms=[0.1])
    with pytest.raises(ValueError, match="weight tensor length"):
        netcon.schedule(con_indices=[0, 1], times_ms=[0.1, 0.2], weight=torch.ones(3))


def test_reference_weight_schedule_reads_bound_source_and_preserves_alignment():
    _, netcon = _built_netcon()
    with pytest.raises(RuntimeError, match="bind_weight_source"):
        netcon.schedule_ref(con_indices=[0], times_ms=[0.1], weight_idx=[0])

    source = torch.tensor([4.0, 7.0], dtype=netcon.dtype, requires_grad=True)
    netcon.bind_weight_source(source)
    netcon.schedule_ref(con_indices=[0, 1], times_ms=[0.1, 0.2], weight_idx=[0, 1])

    assert netcon.sched_weight_idx.tolist() == [0, 1]
    assert netcon.sched_time_idx.tolist() == [-1, -1]
    assert netcon.sched_time_ms.tolist() == pytest.approx([0.1, 0.2])
    gate, _ = netcon._scheduled_gate_this_step(torch.tensor(1), use_tri_kernel=False)
    assert gate.tolist() == [4.0, 0.0, 0.0]
    gate.sum().backward()
    assert source.grad.tolist() == [1.0, 0.0]


def test_reference_weight_schedule_validates_indices_and_lengths():
    _, netcon = _built_netcon()
    netcon.bind_weight_source(torch.ones(2, dtype=netcon.dtype))
    with pytest.raises(ValueError, match="exactly one"):
        netcon.schedule_ref(times_ms=[0.1], weight_idx=[0])
    with pytest.raises(ValueError, match="required"):
        netcon.schedule_ref(con_indices=[0], times_ms=[0.1])
    with pytest.raises(ValueError, match="same length"):
        netcon.schedule_ref(con_indices=[0, 1], times_ms=[0.1], weight_idx=[0, 1])
    with pytest.raises(IndexError, match="connection index"):
        netcon.schedule_ref(con_indices=[4], times_ms=[0.1], weight_idx=[0])
    with pytest.raises(IndexError, match="weight_idx"):
        netcon.schedule_ref(con_indices=[0], times_ms=[0.1], weight_idx=[2])


def test_reference_time_schedule_reads_bound_source_and_filters_past():
    _, netcon = _built_netcon()
    with pytest.raises(RuntimeError, match="bind_time_source"):
        netcon.schedule_time_ref(con_indices=[0], time_idx=[0])

    source = torch.tensor([0.1, 0.3], dtype=netcon.dtype, requires_grad=True)
    netcon.bind_time_source(source)
    netcon.global_step.fill_(2)
    netcon.schedule_time_ref(
        con_indices=[0, 2], time_idx=[0, 1], weight=torch.tensor([5.0, 9.0])
    )

    assert netcon.sched_con_idx.tolist() == [2]
    assert netcon.sched_time_idx.tolist() == [1]
    assert netcon.sched_weight.tolist() == [9.0]
    assert netcon.sched_weight_idx.tolist() == [-1]
    gate, _ = netcon._scheduled_gate_this_step(torch.tensor(3), use_tri_kernel=False)
    assert gate.tolist() == [0.0, 0.0, 9.0]


def test_reference_time_schedule_validates_indices_lengths_and_weights():
    _, netcon = _built_netcon()
    netcon.bind_time_source(torch.tensor([0.1, 0.2], dtype=netcon.dtype))
    with pytest.raises(ValueError, match="exactly one"):
        netcon.schedule_time_ref(time_idx=[0])
    with pytest.raises(ValueError, match="time_idx is required"):
        netcon.schedule_time_ref(con_indices=[0])
    with pytest.raises(ValueError, match="match length"):
        netcon.schedule_time_ref(con_indices=[0, 1], time_idx=[0])
    with pytest.raises(IndexError, match="connection index"):
        netcon.schedule_time_ref(con_indices=[3], time_idx=[0])
    with pytest.raises(IndexError, match="time_idx"):
        netcon.schedule_time_ref(con_indices=[0], time_idx=[2])
    with pytest.raises(ValueError, match="weight must be scalar"):
        netcon.schedule_time_ref(
            con_indices=[0, 1], time_idx=[0, 1], weight=torch.ones(3)
        )


def test_clear_schedule_resets_all_event_metadata_but_keeps_sources():
    _, netcon = _built_netcon()
    w_source = torch.ones(1, dtype=netcon.dtype)
    t_source = torch.full((1,), 0.1, dtype=netcon.dtype)
    netcon.bind_weight_source(w_source)
    netcon.bind_time_source(t_source)
    netcon.schedule(con_indices=[0], times_ms=[0.1])
    netcon.sched_wsum.fill_(1.0)
    netcon.sched_counts.fill_(1)

    netcon.clear_schedule()

    for name in (
        "sched_con_idx",
        "sched_abs_step",
        "sched_weight",
        "sched_weight_idx",
        "sched_time_ms",
        "sched_time_idx",
    ):
        assert getattr(netcon, name).numel() == 0
    assert torch.count_nonzero(netcon.sched_wsum) == 0
    assert torch.count_nonzero(netcon.sched_counts) == 0
    assert netcon._sched_w_source is w_source
    assert netcon._sched_t_source is t_source


def test_dense_state_cache_normalizes_ring_and_restores_tracking_state():
    _, netcon = _built_netcon(track_events=True)
    values = torch.arange(
        netcon.delivery_buffer.numel(), dtype=netcon.dtype
    ).reshape_as(netcon.delivery_buffer)
    netcon.delivery_buffer.copy_(values)
    netcon.event_queue.copy_(
        torch.arange(netcon.event_queue.numel(), dtype=torch.int32).reshape_as(
            netcon.event_queue
        )
    )
    netcon.current_time_step.fill_(2)

    cache = netcon.state_cache()
    expected_delivery = torch.roll(values, -2, dims=0)
    assert cache["param_invariant"] is False
    assert torch.equal(cache["backend_state"]["delivery_buffer"], expected_delivery)

    netcon.delivery_buffer.zero_()
    netcon.event_queue.zero_()
    netcon.initialize_from_state_cache(cache, rebuild_delays=False)
    assert torch.equal(netcon.delivery_buffer, expected_delivery)
    assert torch.equal(netcon.event_queue, cache["backend_state"]["event_queue"])
    assert netcon.current_time_step.item() == 0


def test_dense_state_cache_validates_type_backend_and_shape():
    _, dense = _built_netcon()
    _, sparse = _built_netcon(delay_backend="sparse_calendar")
    with pytest.raises(TypeError, match="dict or legacy tuple"):
        dense.initialize_from_state_cache("bad")
    with pytest.raises(ValueError, match="backend"):
        dense.initialize_from_state_cache(sparse.state_cache())

    cache = dense.state_cache()
    cache["backend_state"]["delivery_buffer"] = torch.zeros((2, 99))
    with pytest.raises(ValueError, match="incompatible shape"):
        dense.initialize_from_state_cache(cache, rebuild_delays=False)


def test_sparse_state_cache_clones_and_restores_relative_calendar():
    _, netcon = _built_netcon(delay_backend="sparse_calendar", track_events=True)
    idx = torch.tensor([1], dtype=torch.long)
    value = torch.tensor([2.5], dtype=netcon.dtype)
    netcon._sparse_calendar = {2: [(idx, value)]}
    netcon._sparse_event_calendar = {2: [(idx, torch.tensor([1], dtype=torch.int32))]}
    netcon.current_time_step.fill_(1)

    cache = netcon.state_cache()
    idx.fill_(0)
    value.zero_()
    assert cache["backend_state"]["sparse_calendar"][1][0][0].item() == 1
    assert cache["backend_state"]["sparse_calendar"][1][0][1].item() == 2.5

    netcon._sparse_calendar.clear()
    netcon._sparse_event_calendar.clear()
    netcon.initialize_from_state_cache(cache, rebuild_delays=False)
    assert netcon._sparse_calendar[1][0][0].item() == 1
    assert netcon._sparse_event_calendar[1][0][1].item() == 1


def test_gate_history_cache_is_parameter_invariant_and_releasable():
    _, netcon = _built_netcon()
    netcon.enable_state_cache_recording(horizon_steps=4)
    assert netcon.state_cache_gate_history.shape == (4, 3)
    netcon.current_time_step.fill_(2)
    netcon._record_gate_for_state_cache(torch.tensor([1.0, 2.0, 3.0]))

    cache = netcon.state_cache()
    assert cache["param_invariant"] is True
    assert cache["backend_state"]["history_layout"] == "age"
    assert cache["backend_state"]["gate_history"][0].tolist() == [1.0, 2.0, 3.0]

    netcon.disable_state_cache_recording(release=True)
    assert netcon.state_cache_gate_history.shape == (0, 0)


def test_dense_checkpoint_state_rebinds_runtime_tensors():
    _, netcon = _built_netcon()
    state = netcon.state_dict_for_checkpoint()
    replacement = {name: tensor.clone() + 1 for name, tensor in state.items()}
    returned = netcon.restore_dict_from_checkpoint(replacement)
    assert returned is netcon
    assert netcon.delivery_buffer is replacement["delivery_buffer"]
    assert netcon.current_time_step is replacement["current_time_step"]
    assert netcon.global_step is replacement["global_step"]


def test_diff_config_and_mode_switching_select_training_runtime():
    _, netcon = _built_netcon()
    with pytest.raises(ValueError, match="train_delay_backend"):
        netcon.set_diff_config(train_delay_backend="bad")

    netcon.train()
    netcon.set_diff_config(
        diff_weights=True,
        diff_delays=True,
        diff_spiking=False,
        taps=2,
        diff_scheduled_times=True,
        train_delay_backend="dense",
    )
    assert netcon.advance.__name__ == "advance_diff"
    netcon.initialize(reinit_weights=False, reinit_delays=False)
    assert netcon.advance.__name__ == "advance_diff"

    assert netcon.eval() is netcon
    netcon.initialize(reinit_weights=False, reinit_delays=False)
    assert netcon.advance.__name__.startswith("advance_non_diff_dense")


def test_spiking_from_netstim_supports_hard_and_soft_gates():
    _, netcon = _built_netcon()
    netcon.pre.spikes = torch.tensor([True, False, True])
    netcon.pre.spike_gate = torch.tensor([0.2, 0.4, 0.6], dtype=netcon.pre_dtype)

    netcon.determine_spiking_ns(netcon.pre, diff_spiking=False)
    assert netcon.is_spiking.tolist() == [1.0, 0.0, 1.0]
    netcon.determine_spiking_ns(netcon.pre, diff_spiking=True)
    assert netcon.is_spiking.tolist() == pytest.approx([0.2, 0.4, 0.6])


def test_zero_can_preserve_or_clear_dense_delivery_state():
    _, netcon = _built_netcon()
    netcon.delivery_buffer.fill_(2.0)
    netcon.current_time_step.fill_(1)
    netcon.zero(clear_delivery_buffers=False)
    assert netcon.current_time_step.item() == 0
    assert torch.all(netcon.delivery_buffer == 2.0)

    netcon.zero(clear_delivery_buffers=True)
    assert torch.count_nonzero(netcon.delivery_buffer) == 0
    assert torch.count_nonzero(netcon.events) == 0
    assert torch.count_nonzero(netcon.event_queue) == 0


def test_dense_and_sparse_backends_deliver_identical_delayed_spikes():
    traces = {}
    event_traces = {}

    for backend in ("dense", "sparse_calendar"):
        net, netcon = _built_netcon(delay_backend=backend, track_events=True)
        net.netstim.spikes.fill_(True)
        trace = []
        events = []

        for _ in range(4):
            netcon.advance()
            trace.append(net.populations["post"].mech.syn.g.detach().clone())
            events.append(netcon.events.detach().clone())
            net.netstim.spikes.zero_()

        traces[backend] = trace
        event_traces[backend] = events

    for dense, sparse in zip(traces["dense"], traces["sparse_calendar"]):
        assert torch.allclose(dense, sparse)
    for dense, sparse in zip(event_traces["dense"], event_traces["sparse_calendar"]):
        assert torch.equal(dense, sparse)

    expected_g = (
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )
    expected_events = (
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    )
    for actual, expected in zip(traces["dense"], expected_g):
        assert actual.flatten().tolist() == pytest.approx(expected)
    for actual, expected in zip(event_traces["dense"], expected_events):
        assert actual.tolist() == expected


def test_dense_and_sparse_backends_deliver_identical_scheduled_events():
    delivered = {}
    for backend in ("dense", "sparse_calendar"):
        net, netcon = _built_netcon(delay_backend=backend)
        net.netstim.spikes.zero_()
        netcon.schedule(con_indices=[1], times_ms=[0.0], weight=2.0)

        netcon.advance()
        assert torch.count_nonzero(net.populations["post"].mech.syn.g) == 0
        netcon.advance()
        delivered[backend] = net.populations["post"].mech.syn.g.detach().clone()

    assert torch.allclose(delivered["dense"], delivered["sparse_calendar"])
    assert delivered["dense"].flatten().tolist() == pytest.approx([0.0, 4.0, 0.0])


@pytest.mark.parametrize("backend", ["dense", "sparse_calendar"])
def test_state_cache_resumes_an_in_flight_delivery(backend):
    original, original_netcon = _built_netcon(delay_backend=backend, track_events=True)
    original.netstim.spikes.fill_(True)
    original_netcon.advance()
    cache = original_netcon.state_cache()

    resumed, resumed_netcon = _built_netcon(delay_backend=backend, track_events=True)
    resumed_netcon.initialize_from_state_cache(cache, rebuild_delays=False)

    original.netstim.spikes.zero_()
    resumed.netstim.spikes.zero_()
    for _ in range(3):
        original_netcon.advance()
        resumed_netcon.advance()
        assert torch.allclose(
            original.populations["post"].mech.syn.g,
            resumed.populations["post"].mech.syn.g,
        )
        assert torch.equal(original_netcon.events, resumed_netcon.events)


def test_dense_checkpoint_resumes_an_in_flight_delivery():
    original, original_netcon = _built_netcon(track_events=False)
    original.netstim.spikes.fill_(True)
    original_netcon.advance()
    state = {
        name: tensor.detach().clone()
        for name, tensor in original_netcon.state_dict_for_checkpoint().items()
    }

    resumed, resumed_netcon = _built_netcon(track_events=False)
    resumed_netcon.restore_dict_from_checkpoint(state)
    original.netstim.spikes.zero_()
    resumed.netstim.spikes.zero_()

    for _ in range(3):
        original_netcon.advance()
        resumed_netcon.advance()
        assert torch.allclose(
            original.populations["post"].mech.syn.g,
            resumed.populations["post"].mech.syn.g,
        )
