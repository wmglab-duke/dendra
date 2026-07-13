from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, graded_syn
from dendra.models.networks.netcon import (
    _dilate_history_age_rows,
    _resample_continuous_history_age_rows,
    _resample_continuous_time_rows,
)

DTYPE = torch.float64
OLD_DT = 0.1
MAX_DELAY_MS = 0.8
EVENT_PREFIX = (
    (True, False),
    (False, True),
    (True, True),
    (False, False),
    (True, False),
)
EVENT_WEIGHTS = torch.tensor([0.7, 1.1, 1.5, 1.9], dtype=DTYPE)
EVENT_DELAYS = torch.tensor([0.30, 0.45, 0.60, 0.25], dtype=DTYPE)
CONTINUOUS_PREFIX = (
    (0.25, 1.25),
    (0.50, 1.00),
    (0.75, 0.75),
    (1.00, 0.50),
    (1.25, 0.25),
)
CONTINUOUS_WEIGHTS = torch.tensor([0.4, 0.8, 1.2, 1.6], dtype=DTYPE)
CONTINUOUS_DELAYS = torch.tensor([0.30, 0.45, 0.60, 0.25], dtype=DTYPE)


def _build_event_connection(
    backend: str,
    dt: float,
    *,
    weights=EVENT_WEIGHTS,
    delays=EVENT_DELAYS,
    train_backend: str = "auto",
    training: bool = False,
    diff_spiking: bool = False,
):
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(expsyn.rename("cache_syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(N=2, dtype=DTYPE)
    net = dn.Network(
        {"post": post},
        netstim=stim,
        track_netcon_events=False,
        netcon_delay_backend=backend,
        netcon_train_backend=train_backend,
    )
    net.connect_dense(
        stim[:],
        post[:],
        post.mech.cache_syn,
        threshold=None,
        weight=weights,
        delay=delays,
    )
    net.build(dt, max_delay_ms=MAX_DELAY_MS)
    net.init_synapses()
    connection = next(iter(net.synapses.values()))
    if training:
        connection.train()
        connection.set_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=diff_spiking,
            diff_scheduled_times=False,
            train_delay_backend=train_backend,
        )
        connection.initialize(reinit_weights=False, reinit_delays=False)
    return net, connection


def _run_event_prefix(net, connection):
    with torch.no_grad():
        for spikes in EVENT_PREFIX:
            net.netstim.spikes.copy_(torch.tensor(spikes, dtype=torch.bool))
            connection.syn.g.zero_()
            connection.advance()


def _expected_event_age_history(connection, depth):
    history = torch.zeros((depth, connection._n_conn), dtype=DTYPE)
    for step, spikes in enumerate(EVENT_PREFIX):
        age = len(EVENT_PREFIX) - step
        gate = torch.tensor(spikes, dtype=DTYPE).index_select(
            0, connection.pre_idx.cpu()
        )
        history[age].add_(gate)
    return history


def _event_pending_oracle(connection, *, old_dt, new_dt):
    old_depth = int(MAX_DELAY_MS / old_dt) + 1
    old_history = _expected_event_age_history(connection, old_depth)
    new_depth = connection.max_delay_steps
    rebinned = torch.zeros((new_depth, connection._n_conn), dtype=DTYPE)
    for old_age in range(old_depth):
        new_age = math.floor(old_age * old_dt / new_dt + 0.5)
        new_age = min(max(new_age, 0), new_depth - 1)
        rebinned[new_age].add_(old_history[old_age])

    expected = torch.zeros((new_depth, connection._syn_numel), dtype=DTYPE)
    weights = connection.weight().detach().cpu()
    delays = connection.inference_delay_steps.detach().cpu()
    post_idx = connection.post_idx.detach().cpu()
    for age in range(1, new_depth):
        for edge in range(connection._n_conn):
            remaining = int(delays[edge]) - age
            if remaining >= 0:
                expected[remaining, post_idx[edge]] += (
                    weights[edge] * rebinned[age, edge]
                )
    return expected


def _drain_event_payloads(net, connection):
    trace = []
    with torch.no_grad():
        net.netstim.spikes.zero_()
        net.netstim.spike_gate.zero_()
        for _ in range(connection.max_delay_steps):
            connection.syn.g.zero_()
            connection.advance()
            trace.append(connection.syn.g.detach().flatten().clone())
    return torch.stack(trace)


@pytest.mark.parametrize("backend", ["dense", "sparse_calendar"])
@pytest.mark.parametrize("new_dt", [0.05, 0.2])
def test_event_gate_cache_rebuilds_changed_weights_delays_and_dt(backend, new_dt):
    source_net, source = _build_event_connection(backend, OLD_DT)
    source.enable_state_cache_recording()
    _run_event_prefix(source_net, source)
    assert source.current_time_step.item() == len(EVENT_PREFIX)

    cache = source.state_cache()
    assert cache["param_invariant"] is True
    torch.testing.assert_close(
        cache["backend_state"]["gate_history"],
        _expected_event_age_history(source, source.max_delay_steps),
        atol=0.0,
        rtol=0.0,
    )

    new_weights = EVENT_WEIGHTS * torch.tensor([1.3, 0.6, 1.7, 0.9], dtype=DTYPE)
    new_delays = torch.tensor([0.40, 0.20, 0.70, 0.30], dtype=DTYPE)
    resumed_net, resumed = _build_event_connection(
        backend, new_dt, weights=new_weights, delays=new_delays
    )
    expected = _event_pending_oracle(resumed, old_dt=OLD_DT, new_dt=new_dt)
    resumed.initialize_from_state_cache(cache, dt=new_dt, rebuild_delays=False)

    torch.testing.assert_close(
        _drain_event_payloads(resumed_net, resumed), expected, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("diff_spiking", [False, True])
def test_dense_gate_cache_translates_to_source_history_training_runtime(
    diff_spiking,
):
    source_net, source = _build_event_connection("dense", OLD_DT)
    source.enable_state_cache_recording()
    _run_event_prefix(source_net, source)
    cache = source.state_cache()

    new_dt = 0.05
    new_weights = EVENT_WEIGHTS * 1.25
    new_delays = torch.tensor([0.35, 0.20, 0.65, 0.30], dtype=DTYPE)
    resumed_net, resumed = _build_event_connection(
        "dense",
        new_dt,
        weights=new_weights,
        delays=new_delays,
        train_backend="source_history",
        training=True,
        diff_spiking=diff_spiking,
    )
    expected = _event_pending_oracle(resumed, old_dt=OLD_DT, new_dt=new_dt)
    resumed.initialize_from_state_cache(cache, dt=new_dt, rebuild_delays=False)

    assert resumed.advance.__name__ == "advance_diff_source_history"
    torch.testing.assert_close(
        _drain_event_payloads(resumed_net, resumed), expected, atol=0.0, rtol=0.0
    )


def _build_threshold_fanout(
    backend="dense", *, training_backend=None, diff_spiking=False
):
    pre = dn.Population(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(expsyn.rename("threshold_cache_syn"), e=0.0, tau=1.0)
    net = dn.Network(
        {"pre": pre, "post": post},
        netcon_delay_backend=backend,
        netcon_train_backend=training_backend or "source_history",
    )
    net.connect_dense(
        pre[:],
        post[:],
        post.mech.threshold_cache_syn,
        threshold=torch.tensor([-30.0, -30.0, -20.0, -20.0], dtype=DTYPE),
        weight=torch.tensor([0.5, 0.75, 1.0, 1.25], dtype=DTYPE),
        delay=OLD_DT,
    )
    net.build(OLD_DT, max_delay_ms=MAX_DELAY_MS)
    net.init_synapses()
    connection = next(iter(net.synapses.values()))
    if training_backend is not None:
        connection.train()
        connection.set_diff_config(
            diff_weights=False,
            diff_delays=False,
            diff_spiking=diff_spiking,
            diff_scheduled_times=False,
            train_delay_backend=training_backend,
        )
        connection.initialize(reinit_weights=False, reinit_delays=False)
    return net, connection


@pytest.mark.parametrize("diff_spiking", [False, True])
@pytest.mark.parametrize(
    ("backend", "training_backend"),
    [
        ("dense", "source_history"),
        ("bitpacked_history", "source_history"),
        ("bitpacked_history", "dense"),
    ],
)
def test_cache_restores_shared_source_threshold_latch_without_retrigger(
    backend, training_backend, diff_spiking
):
    source_net, source = _build_threshold_fanout(backend)
    source.enable_state_cache_recording()
    source_net.pre.v.fill_(-10.0)
    source.syn.g.zero_()
    source.advance()
    if backend == "dense":
        assert source.has_spiked.tolist() == [True, True, True, True]
    else:
        assert source.bitpack_source_has_spiked.tolist() == [True, True]
    cache = source.state_cache()

    resumed_net, resumed = _build_threshold_fanout(
        backend, training_backend=training_backend, diff_spiking=diff_spiking
    )
    resumed.initialize_from_state_cache(cache, rebuild_delays=False)
    assert resumed.bitpack_source_has_spiked.tolist() == [True, True]
    if training_backend == "dense":
        assert resumed.has_spiked.tolist() == [True, True, True, True]

    with torch.no_grad():
        # The cached crossing arrives once.  Merely remaining above threshold
        # must not synthesize another crossing on either fan-out edge.
        resumed_net.pre.v.fill_(-10.0)
        resumed.syn.g.zero_()
        resumed.advance()
        assert torch.count_nonzero(resumed.syn.g) == 2
        resumed.syn.g.zero_()
        resumed.advance()
        assert torch.count_nonzero(resumed.syn.g) == 0

        # Re-arming below threshold and crossing again produces exactly one
        # subsequent delivery per shared source.
        resumed_net.pre.v.fill_(-65.0)
        resumed.advance()
        resumed_net.pre.v.fill_(-10.0)
        resumed.advance()
        resumed.syn.g.zero_()
        resumed.advance()
        assert torch.count_nonzero(resumed.syn.g) == 2


@pytest.mark.parametrize(
    "runtime", ["eval", "dense", "source_history", "source_history_diff"]
)
def test_bitpacked_cache_translates_to_current_runtime_with_new_parameters(runtime):
    source_net, source = _build_event_connection("bitpacked_history", OLD_DT)
    _run_event_prefix(source_net, source)
    cache = source.state_cache()
    assert cache["param_invariant"] is True
    assert cache["backend_state"]["history_layout"] == "age"

    new_dt = 0.05
    new_weights = EVENT_WEIGHTS * torch.tensor([0.8, 1.4, 1.1, 1.6], dtype=DTYPE)
    new_delays = torch.tensor([0.35, 0.25, 0.70, 0.30], dtype=DTYPE)
    training = runtime != "eval"
    train_backend = "source_history" if runtime == "source_history_diff" else runtime
    if not training:
        train_backend = "auto"
    resumed_net, resumed = _build_event_connection(
        "bitpacked_history",
        new_dt,
        weights=new_weights,
        delays=new_delays,
        train_backend=train_backend,
        training=training,
        diff_spiking=runtime == "source_history_diff",
    )
    expected = _event_pending_oracle(resumed, old_dt=OLD_DT, new_dt=new_dt)
    resumed.initialize_from_state_cache(cache, dt=new_dt, rebuild_delays=False)

    torch.testing.assert_close(
        _drain_event_payloads(resumed_net, resumed), expected, atol=0.0, rtol=0.0
    )


def _build_continuous_connection(
    dt: float,
    *,
    weights=CONTINUOUS_WEIGHTS,
    delays=CONTINUOUS_DELAYS,
    transform=None,
    max_delay_ms=MAX_DELAY_MS,
):
    pre = dn.Population(N=2, C=1, v_init=0.0, dtype=DTYPE)
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(graded_syn.rename("cache_graded"), e=-10.0, g_scale=0.02)
    net = dn.Network({"pre": pre, "post": post})
    net.connect_continuous(
        pre[:],
        post[:],
        post.mech.cache_graded,
        conn_spec="all_to_all",
        pre_var="v",
        input="g_pre",
        weight=weights,
        delay=delays,
        transform=transform,
    )
    net.build(dt, max_delay_ms=max_delay_ms)
    net.init_synapses()
    return net, next(iter(net.continuous_synapses.values()))


def _run_continuous_prefix(net, connection):
    with torch.no_grad():
        for values in CONTINUOUS_PREFIX:
            net.pre.v.copy_(torch.tensor(values, dtype=DTYPE).view_as(net.pre.v))
            connection.syn.reset_continuous_inputs()
            connection.advance()


def _expected_continuous_age_history(connection, depth):
    history = torch.zeros((depth, connection._n_conn), dtype=DTYPE)
    for step, values in enumerate(CONTINUOUS_PREFIX):
        age = len(CONTINUOUS_PREFIX) - step
        selected = torch.tensor(values, dtype=DTYPE).index_select(
            0, connection.pre_idx.cpu()
        )
        history[age].copy_(selected)
    return history


def _continuous_pending_oracle(connection, *, old_dt, new_dt):
    old_depth = int(MAX_DELAY_MS / old_dt) + 1
    old_history = _expected_continuous_age_history(connection, old_depth)
    new_depth = connection.max_delay_steps
    resampled = torch.zeros((new_depth, connection._n_conn), dtype=DTYPE)
    for new_age in range(1, new_depth):
        old_age = math.floor(new_age * new_dt / old_dt + 0.5)
        if old_age >= old_depth:
            continue
        old_age = max(1, old_age)
        resampled[new_age].copy_(old_history[old_age])

    expected = torch.zeros((new_depth, connection._syn_numel), dtype=DTYPE)
    weights = connection.weight().detach().cpu()
    delays = connection.delay_steps.detach().cpu()
    post_idx = connection.post_idx.detach().cpu()
    for age in range(1, new_depth):
        for edge in range(connection._n_conn):
            remaining = int(delays[edge]) - age
            if remaining >= 0:
                expected[remaining, post_idx[edge]] += (
                    weights[edge] * resampled[age, edge]
                )
    return expected


def _drain_continuous_payloads(net, connection):
    trace = []
    with torch.no_grad():
        net.pre.v.zero_()
        for _ in range(connection.max_delay_steps):
            connection.syn.reset_continuous_inputs()
            connection.advance()
            trace.append(connection.syn.g_pre.detach().flatten().clone())
    return torch.stack(trace)


@pytest.mark.parametrize("new_dt", [0.05, 0.2])
def test_continuous_cache_resamples_history_and_uses_current_parameters(new_dt):
    source_net, source = _build_continuous_connection(OLD_DT)
    source.enable_state_cache_recording()
    _run_continuous_prefix(source_net, source)
    assert source.current_time_step.item() == len(CONTINUOUS_PREFIX)
    cache = source.state_cache()
    assert cache["param_invariant"] is True
    torch.testing.assert_close(
        cache["pre_value_history"],
        _expected_continuous_age_history(source, source.max_delay_steps),
        atol=0.0,
        rtol=0.0,
    )

    new_weights = CONTINUOUS_WEIGHTS * torch.tensor([1.5, 0.5, 1.25, 0.75], dtype=DTYPE)
    new_delays = torch.tensor([0.40, 0.20, 0.70, 0.30], dtype=DTYPE)
    resumed_net, resumed = _build_continuous_connection(
        new_dt, weights=new_weights, delays=new_delays
    )
    expected = _continuous_pending_oracle(resumed, old_dt=OLD_DT, new_dt=new_dt)
    resumed.initialize_from_state_cache(cache, dt=new_dt, rebuild_delays=False)

    torch.testing.assert_close(
        _drain_continuous_payloads(resumed_net, resumed),
        expected,
        atol=1e-15,
        rtol=0.0,
    )


@pytest.mark.parametrize("new_dt", [0.05, 0.2])
def test_continuous_fallback_cache_preserves_a_constant_pending_signal(new_dt):
    source_net, source = _build_continuous_connection(OLD_DT)
    source.delivery_buffer.fill_(2.5)
    source.delivery_mask.fill_(True)
    source.current_time_step.fill_(5)
    cache = source.state_cache()
    assert cache["param_invariant"] is False

    resumed_net, resumed = _build_continuous_connection(new_dt)
    resumed.initialize_from_state_cache(cache, dt=new_dt, rebuild_delays=False)

    expected = torch.full_like(resumed.delivery_buffer, 2.5)
    torch.testing.assert_close(resumed.delivery_buffer, expected, atol=0.0, rtol=0.0)
    assert resumed.delivery_mask.all()


def test_continuous_cache_preserves_zero_payload_presence_and_loads_legacy_cache():
    source_net, source = _build_continuous_connection(OLD_DT)
    source.delivery_buffer.zero_()
    source.delivery_mask.zero_()
    source.delivery_mask[2, 0] = True
    source.delivery_buffer[3, 1] = 4.0
    source.delivery_mask[3, 1] = True
    cache = source.state_cache()

    resumed_net, resumed = _build_continuous_connection(OLD_DT)
    resumed.initialize_from_state_cache(cache, rebuild_delays=False)
    assert torch.equal(resumed.delivery_mask, cache["delivery_mask"])
    assert resumed.delivery_mask[2, 0]
    assert resumed.delivery_buffer[2, 0] == 0.0

    legacy = dict(cache)
    legacy.pop("delivery_mask")
    legacy_net, legacy_resumed = _build_continuous_connection(OLD_DT)
    legacy_resumed.initialize_from_state_cache(legacy, rebuild_delays=False)
    assert torch.equal(
        legacy_resumed.delivery_mask, legacy_resumed.delivery_buffer != 0
    )


def test_parameter_invariant_cache_rebuilds_masks_for_zero_source_values():
    source_net, source = _build_continuous_connection(OLD_DT)
    source.enable_state_cache_recording()
    _run_continuous_prefix(source_net, source)
    cache = source.state_cache()
    cache["pre_value_history"].zero_()

    resumed_net, resumed = _build_continuous_connection(OLD_DT)
    resumed.initialize_from_state_cache(cache, rebuild_delays=False)

    assert not resumed.delivery_buffer.any()
    assert resumed.delivery_mask.any()


class _Scale(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value, dtype=DTYPE))

    def forward(self, value):
        return value * self.value


def test_continuous_cache_reapplies_current_trainable_transform():
    source_net, source = _build_continuous_connection(OLD_DT, transform=_Scale(2.0))
    source.enable_state_cache_recording()
    _run_continuous_prefix(source_net, source)
    cache = source.state_cache()
    assert cache["version"] == 4
    assert cache["value_layout"] == "raw"
    # A parameter-invariant cache contains raw source values, not values made
    # stale by the transform that happened to be active during recording.
    torch.testing.assert_close(
        cache["pre_value_history"],
        _expected_continuous_age_history(source, source.max_delay_steps),
        atol=0.0,
        rtol=0.0,
    )

    resumed_net, resumed = _build_continuous_connection(OLD_DT, transform=_Scale(3.0))
    base_expected = _continuous_pending_oracle(resumed, old_dt=OLD_DT, new_dt=OLD_DT)
    expected = base_expected * 3.0
    resumed.initialize_from_state_cache(cache, rebuild_delays=False)

    torch.testing.assert_close(
        _drain_continuous_payloads(resumed_net, resumed),
        expected,
        atol=1e-15,
        rtol=0.0,
    )

    # Version 2 stored already-transformed values. Its absence of the raw-value
    # marker must suppress current-transform replay and avoid double application.
    legacy = dict(cache)
    legacy["version"] = 2
    legacy.pop("value_layout")
    legacy["pre_value_history"] = cache["pre_value_history"] * 2.0
    legacy_net, legacy_resumed = _build_continuous_connection(
        OLD_DT, transform=_Scale(3.0)
    )
    legacy_resumed.initialize_from_state_cache(legacy, rebuild_delays=False)
    torch.testing.assert_close(
        _drain_continuous_payloads(legacy_net, legacy_resumed),
        base_expected * 2.0,
        atol=1e-15,
        rtol=0.0,
    )


def test_continuous_cache_holds_endpoint_across_quantized_horizon_change():
    old_dt = 0.125
    new_dt = 0.1875
    delays = torch.full((4,), 0.15, dtype=DTYPE)
    weights = torch.ones(4, dtype=DTYPE)
    source_net, source = _build_continuous_connection(
        old_dt,
        weights=weights,
        delays=delays,
        max_delay_ms=None,
    )
    source.enable_state_cache_recording()
    with torch.no_grad():
        for _ in range(2):
            source_net.pre.v.fill_(2.5)
            source.syn.reset_continuous_inputs()
            source.advance()
    cache = source.state_cache()
    assert cache["pre_value_history"].shape[0] == 2

    resumed_net, resumed = _build_continuous_connection(
        new_dt,
        weights=weights,
        delays=delays,
        max_delay_ms=None,
    )
    resumed.initialize_from_state_cache(cache, dt=new_dt, rebuild_delays=False)
    # The one available historical analog sample is held at the slightly longer
    # quantized age; it must not become an artificial all-zero discontinuity.
    assert resumed.delivery_buffer[0].tolist() == pytest.approx([5.0, 5.0])


def test_cache_time_mapping_uses_decimal_half_up_ties():
    age_rows = torch.tensor([[0.0], [10.0], [20.0]], dtype=DTYPE)
    resampled_age = _resample_continuous_history_age_rows(age_rows, 0.2, 0.3, n_limit=2)
    assert resampled_age[:, 0].tolist() == [0.0, 20.0]

    future_rows = torch.tensor([[10.0], [20.0], [30.0]], dtype=DTYPE)
    resampled_future = _resample_continuous_time_rows(future_rows, 0.2, 0.3, n_limit=2)
    assert resampled_future[:, 0].tolist() == [10.0, 30.0]

    event_rows = torch.tensor([[0.0], [7.0]], dtype=DTYPE)
    rebinned_events = _dilate_history_age_rows(event_rows, 0.3, 0.2, n_limit=3)
    assert rebinned_events[:, 0].tolist() == [0.0, 0.0, 7.0]


def test_network_routes_event_cache_after_parameter_and_dt_changes():
    source_net, source = _build_event_connection("dense", OLD_DT)
    source_net.initialize(OLD_DT)
    source.enable_state_cache_recording()
    _run_event_prefix(source_net, source)
    source_net.cache_state()

    new_dt = 0.05
    resumed_net, resumed = _build_event_connection(
        "dense",
        new_dt,
        weights=EVENT_WEIGHTS * 1.4,
        delays=torch.tensor([0.35, 0.20, 0.65, 0.30], dtype=DTYPE),
    )
    expected = _event_pending_oracle(resumed, old_dt=OLD_DT, new_dt=new_dt)
    resumed_net.load_state_cache(source_net._state_cache, source_net._syn_cache)
    resumed_net.initialize(new_dt)

    torch.testing.assert_close(
        _drain_event_payloads(resumed_net, resumed), expected, atol=0.0, rtol=0.0
    )


def test_network_routes_continuous_cache_after_parameter_and_dt_changes():
    source_net, source = _build_continuous_connection(OLD_DT)
    source_net.initialize(OLD_DT)
    source.enable_state_cache_recording()
    _run_continuous_prefix(source_net, source)
    source_net.cache_state()

    new_dt = 0.2
    resumed_net, resumed = _build_continuous_connection(
        new_dt,
        weights=CONTINUOUS_WEIGHTS * 0.75,
        delays=torch.tensor([0.40, 0.20, 0.60, 0.20], dtype=DTYPE),
    )
    expected = _continuous_pending_oracle(resumed, old_dt=OLD_DT, new_dt=new_dt)
    resumed_net.load_state_cache(source_net._state_cache, source_net._syn_cache)
    resumed_net.initialize(new_dt)

    torch.testing.assert_close(
        _drain_continuous_payloads(resumed_net, resumed),
        expected,
        atol=1e-15,
        rtol=0.0,
    )
