from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import dendra as dn
from dendra.models.mod import expsyn

DT = 0.1
DTYPE = torch.float64
INFERENCE_BACKENDS = ("dense", "sparse_calendar", "bitpacked_history")

PRE_IDX = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 1, 2], dtype=torch.long)
POST_IDX = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
WEIGHTS = torch.tensor(
    [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        0.25,
        1.25,
        2.25,
    ],
    dtype=DTYPE,
)
DELAY_STEPS = torch.tensor(
    [0, 1, 4, 2, 0, 3, 4, 1, 0, 3, 4, 2, 4, 0, 1], dtype=torch.long
)
SCHEDULE = (
    (0, 0, 2.0),
    (0, 0, -0.5),
    (4, 2, 1.5),
    (14, 2, -0.25),
    (3, 6, 0.0),
    (11, 7, 2.0),
    (6, 10, -1.0),
)
SPIKE_TAPE = (
    (True, False, True, False),
    (False, False, False, False),
    (False, True, True, True),
    (True, True, False, False),
    (False, False, False, False),
    (False, False, False, True),
    (True, True, True, True),
    (False, False, False, False),
    (True, False, False, False),
    (False, False, False, False),
    (False, True, False, True),
    (False, False, False, False),
    (False, False, False, False),
    (False, False, False, False),
    (False, False, False, False),
    (False, False, False, False),
)


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


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


def _build_adversarial_net(backend: str, *, schedule: bool = True):
    post = dn.Population(N=3, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(N=4, dtype=DTYPE)
    net = dn.Network(
        {"post": post},
        netstim=stim,
        track_netcon_events=False,
        netcon_delay_backend=backend,
    )
    net.connect_dense(
        stim[:],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=WEIGHTS[:12],
        delay=DELAY_STEPS[:12].to(DTYPE) * DT,
    )
    # These edges duplicate three source/target pairs from the dense fan-out.
    net.connect_one_to_one(
        stim[:3],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=WEIGHTS[12:],
        delay=DELAY_STEPS[12:].to(DTYPE) * DT,
    )
    net.build(DT)
    net.init_synapses()
    netcon = next(iter(net.synapses.values()))
    if schedule:
        # Empty scheduling is explicitly a no-op for every backend.
        netcon.schedule(con_indices=[], times_ms=[])
        netcon.schedule(
            con_indices=[event[0] for event in SCHEDULE],
            times_ms=[event[1] * DT for event in SCHEDULE],
            weight=torch.tensor([event[2] for event in SCHEDULE], dtype=DTYPE),
        )
    return net, netcon


def _run_spike_tape(net, netcon, tape):
    trace = []
    for spikes in tape:
        net.netstim.spikes.copy_(torch.tensor(spikes, dtype=torch.bool))
        netcon.advance()
        trace.append(netcon.syn.g.detach().flatten().clone())
    return torch.stack(trace)


def _discrete_event_oracle(tape, *, schedule=SCHEDULE):
    """Independent queue model of the public one-step-minimum delay contract."""
    pending = {}
    state = torch.zeros(3, dtype=DTYPE)
    trace = []
    effective_delays = DELAY_STEPS.clamp_min(1)

    for step, source_spikes in enumerate(tape):
        state = state + pending.pop(step, torch.zeros_like(state))
        trace.append(state.clone())

        gate = torch.tensor(source_spikes, dtype=DTYPE).index_select(0, PRE_IDX)
        for con_idx, event_step, multiplier in schedule:
            if event_step == step:
                gate[con_idx] += multiplier

        payloads = WEIGHTS * gate
        for edge, payload in enumerate(payloads):
            due = step + int(effective_delays[edge])
            contribution = F.one_hot(POST_IDX[edge], num_classes=3).to(DTYPE) * payload
            pending[due] = pending.get(due, torch.zeros_like(state)) + contribution

    return torch.stack(trace)


def test_all_inference_backends_match_adversarial_event_queue_oracle():
    expected = _discrete_event_oracle(SPIKE_TAPE)
    traces = {}

    for backend in INFERENCE_BACKENDS:
        net, netcon = _build_adversarial_net(backend)
        traces[backend] = _run_spike_tape(net, netcon, SPIKE_TAPE)

        torch.testing.assert_close(traces[backend], expected, atol=2e-5, rtol=1e-6)
        assert netcon.current_time_step.item() == len(SPIKE_TAPE) % 5

    for backend in INFERENCE_BACKENDS[1:]:
        torch.testing.assert_close(traces[backend], traces["dense"])


THRESHOLDS = torch.tensor([-30.0, -30.0, -20.0, -20.0, -10.0, -10.0])
THRESHOLD_WEIGHTS = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=DTYPE)
THRESHOLD_DELAYS = torch.tensor([0, 4, 2, 1, 3, 0], dtype=torch.long)
VOLTAGE_TAPE = (
    (-65.0, -65.0, -65.0),
    (-25.0, -25.0, -5.0),
    (-20.0, -15.0, -5.0),
    (-65.0, -15.0, -20.0),
    (-30.0, -30.0, -10.0),
    (-29.0, -20.0, -11.0),
    (-65.0, -65.0, -65.0),
    (-65.0, -65.0, -65.0),
    (-65.0, -65.0, -65.0),
    (-65.0, -65.0, -65.0),
    (-65.0, -65.0, -65.0),
)


def _build_threshold_net(backend):
    pre = dn.Population(N=3, C=1, v_init=-65.0, dtype=DTYPE)
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    net = dn.Network(
        {"pre": pre, "post": post},
        track_netcon_events=False,
        netcon_delay_backend=backend,
    )
    net.connect_dense(
        pre[:],
        post[:],
        post.mech.syn,
        threshold=THRESHOLDS,
        weight=THRESHOLD_WEIGHTS,
        delay=THRESHOLD_DELAYS.to(DTYPE) * DT,
    )
    net.build(DT)
    net.init_synapses()
    return net, next(iter(net.synapses.values()))


def _threshold_oracle():
    pre_idx = torch.tensor([0, 0, 1, 1, 2, 2])
    post_idx = torch.tensor([0, 1, 0, 1, 0, 1])
    source_thresholds = torch.tensor([-30.0, -20.0, -10.0])
    was_above = torch.zeros(3, dtype=torch.bool)
    pending = {}
    state = torch.zeros(2, dtype=DTYPE)
    trace = []

    for step, voltage in enumerate(VOLTAGE_TAPE):
        state = state + pending.pop(step, torch.zeros_like(state))
        trace.append(state.clone())
        above = torch.tensor(voltage) >= source_thresholds
        spikes = above & ~was_above
        was_above = above
        payloads = THRESHOLD_WEIGHTS * spikes.index_select(0, pre_idx)
        for edge, payload in enumerate(payloads):
            due = step + int(THRESHOLD_DELAYS[edge].clamp_min(1))
            contribution = F.one_hot(post_idx[edge], num_classes=2).to(DTYPE) * payload
            pending[due] = pending.get(due, torch.zeros_like(state)) + contribution

    return torch.stack(trace)


def test_threshold_crossing_backends_match_independent_rising_edge_oracle():
    expected = _threshold_oracle()
    for backend in INFERENCE_BACKENDS:
        net, netcon = _build_threshold_net(backend)
        trace = []
        for voltage in VOLTAGE_TAPE:
            net.pre.v.copy_(torch.tensor(voltage, dtype=DTYPE).view(3, 1))
            netcon.advance()
            trace.append(netcon.syn.g.detach().flatten().clone())
        torch.testing.assert_close(torch.stack(trace), expected, atol=2e-5, rtol=1e-6)


@pytest.mark.parametrize("backend", INFERENCE_BACKENDS)
def test_backend_state_cache_resumes_duplicate_edges_and_max_delay_traffic(backend):
    split = 5
    expected = _discrete_event_oracle(SPIKE_TAPE, schedule=())
    original, original_netcon = _build_adversarial_net(backend, schedule=False)
    prefix = _run_spike_tape(original, original_netcon, SPIKE_TAPE[:split])
    cache = original_netcon.state_cache()

    resumed, resumed_netcon = _build_adversarial_net(backend, schedule=False)
    resumed_netcon.initialize_from_state_cache(cache, rebuild_delays=False)
    resumed_netcon.syn.g.copy_(original_netcon.syn.g)

    continuation = _run_spike_tape(resumed, resumed_netcon, SPIKE_TAPE[split:])
    torch.testing.assert_close(prefix, expected[:split], atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(continuation, expected[split:], atol=2e-5, rtol=1e-6)


def test_dense_checkpoint_resumes_future_scheduled_and_intrinsic_deliveries():
    split = 6
    expected = _discrete_event_oracle(SPIKE_TAPE)
    original, original_netcon = _build_adversarial_net("dense")
    _run_spike_tape(original, original_netcon, SPIKE_TAPE[:split])
    checkpoint = {
        name: value.detach().clone()
        for name, value in original_netcon.state_dict_for_checkpoint().items()
    }

    resumed, resumed_netcon = _build_adversarial_net("dense")
    resumed_netcon.restore_dict_from_checkpoint(checkpoint)
    resumed_netcon.syn.g.copy_(original_netcon.syn.g)
    continuation = _run_spike_tape(resumed, resumed_netcon, SPIKE_TAPE[split:])

    torch.testing.assert_close(continuation, expected[split:], atol=2e-5, rtol=1e-6)


def test_network_checkpoint_restore_rolls_back_after_late_netcon_failure():
    net, _ = _build_adversarial_net("dense", schedule=False)
    before = _clone_nested(net.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)
    corrupt["populations"]["post"]["integrator"]["v"].add_(17.0)
    synapse_name = next(iter(corrupt["netcons"]["event"]))
    corrupt["netcons"]["event"][synapse_name].pop("global_step")

    with pytest.raises(KeyError, match="global_step"):
        net.restore_dict_from_checkpoint(corrupt)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


def test_network_checkpoint_validates_complete_structure_before_mutation():
    net, _ = _build_adversarial_net("dense", schedule=False)
    before = _clone_nested(net.state_dict_for_checkpoint())

    missing_population = _clone_nested(before)
    missing_population["populations"].pop("post")
    with pytest.raises(KeyError, match="population names"):
        net.restore_dict_from_checkpoint(missing_population)

    missing_event = _clone_nested(before)
    missing_event["netcons"]["event"].clear()
    with pytest.raises(KeyError, match="event NetCon names"):
        net.restore_dict_from_checkpoint(missing_event)

    missing_netstim = _clone_nested(before)
    missing_netstim["netstim"] = None
    with pytest.raises(ValueError, match="NetStim presence"):
        net.restore_dict_from_checkpoint(missing_netstim)

    malformed_time = _clone_nested(before)
    malformed_time["populations"]["post"]["integrator"]["v"].add_(13.0)
    malformed_time["t"] = torch.zeros(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="time has shape"):
        net.restore_dict_from_checkpoint(malformed_time)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


def test_network_checkpoint_time_is_cast_cloned_and_not_aliased():
    net, _ = _build_adversarial_net("dense", schedule=False)
    expected_dtype = net.t.dtype
    checkpoint = _clone_nested(net.state_dict_for_checkpoint())
    checkpoint["t"] = torch.tensor(2.5, dtype=torch.float64)

    net.restore_dict_from_checkpoint(checkpoint)
    checkpoint["t"].add_(7.0)

    assert net.t.dtype == expected_dtype
    assert net.t.item() == pytest.approx(2.5)


def test_network_load_rolls_back_tensors_when_rng_extra_state_is_corrupt():
    net, _ = _build_adversarial_net("dense", schedule=False)
    before = _clone_nested(net.state_dict())
    corrupt = _clone_nested(before)
    corrupt["t"].add_(9.0)
    corrupt["_extra_state"]["rng_state"]["cpu"] = torch.zeros(1, dtype=torch.uint8)

    with pytest.raises(RuntimeError):
        net.load(corrupt)

    _assert_nested_equal(net.state_dict(), before)


def test_network_load_accepts_pathlike_state_dict(tmp_path):
    net, _ = _build_adversarial_net("dense", schedule=False)
    state = _clone_nested(net.state_dict())
    state["t"].fill_(3.25)
    path = tmp_path / "network-state.pt"
    torch.save(state, path)

    net.load(path)

    assert net.t.item() == pytest.approx(3.25)


TRAIN_TAPE = (
    (True, False, True),
    (False, True, False),
    (True, True, False),
    (False, False, False),
    (False, False, True),
    (False, False, False),
    (False, False, False),
    (False, False, False),
)


def _build_training_net(train_backend, *, diff_delays=True):
    post = dn.Population(N=3, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(N=3, dtype=DTYPE)
    net = dn.Network(
        {"post": post},
        netstim=stim,
        netcon_delay_backend="dense",
        netcon_train_backend=train_backend,
    )
    net.connect_one_to_one(
        stim[:],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=torch.nn.Parameter(torch.tensor([0.75, 1.25, 2.0], dtype=DTYPE)),
        delay=torch.nn.Parameter(torch.tensor([0.05, 0.16, 0.28], dtype=DTYPE)),
    )
    net.build(DT)
    net.init_synapses()
    netcon = next(iter(net.synapses.values()))
    netcon.train()
    netcon.set_diff_config(
        diff_weights=True,
        diff_delays=diff_delays,
        diff_spiking=False,
        taps=2,
        diff_scheduled_times=False,
        train_delay_backend=train_backend,
    )
    netcon.initialize(reinit_weights=False, reinit_delays=False)
    return net, netcon


def _interpolated_training_oracle(tape, weights, delays):
    pending = {}
    state = torch.zeros(3, dtype=DTYPE)
    trace = []
    delay_in_steps = (delays / DT).clamp_min(1.0)
    lower = torch.floor(delay_in_steps)
    fraction = delay_in_steps - lower

    for step, spikes in enumerate(tape):
        state = state + pending.pop(step, torch.zeros_like(state))
        trace.append(state)
        gates = torch.tensor(spikes, dtype=DTYPE)
        for edge in range(3):
            payload = weights[edge] * gates[edge]
            due_lower = step + int(lower[edge].detach())
            due_upper = due_lower + 1
            basis = F.one_hot(torch.tensor(edge), num_classes=3).to(DTYPE)
            pending[due_lower] = pending.get(
                due_lower, torch.zeros_like(state)
            ) + basis * payload * (1.0 - fraction[edge])
            pending[due_upper] = (
                pending.get(due_upper, torch.zeros_like(state))
                + basis * payload * fraction[edge]
            )

    return torch.stack(trace)


def _hard_training_oracle(tape, weights, delays):
    pending = {}
    state = torch.zeros(3, dtype=DTYPE)
    trace = []
    delay_steps = torch.round(delays / DT).to(torch.long).clamp_min(1)

    for step, spikes in enumerate(tape):
        state = state + pending.pop(step, torch.zeros_like(state))
        trace.append(state)
        gates = torch.tensor(spikes, dtype=DTYPE)
        for edge in range(3):
            due = step + int(delay_steps[edge])
            basis = F.one_hot(torch.tensor(edge), num_classes=3).to(DTYPE)
            contribution = basis * weights[edge] * gates[edge]
            pending[due] = pending.get(due, torch.zeros_like(state)) + contribution

    return torch.stack(trace)


@pytest.mark.parametrize("train_backend", ["dense", "source_history"])
def test_training_backends_clamp_hard_round_to_zero_delays(train_backend):
    net, netcon = _build_training_net(train_backend, diff_delays=False)
    netcon.w.retain_grad()
    trace = []
    for spikes in TRAIN_TAPE:
        net.netstim.spikes.copy_(torch.tensor(spikes, dtype=torch.bool))
        netcon.advance()
        trace.append(netcon.syn.g.flatten().clone())
    actual = torch.stack(trace)

    oracle_weights = netcon.w.detach().clone().requires_grad_()
    expected = _hard_training_oracle(
        TRAIN_TAPE, oracle_weights, netcon.delay_ms.w.detach()
    )
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(netcon.w.grad, oracle_weights.grad)


@pytest.mark.parametrize("train_backend", ["dense", "source_history"])
def test_training_backends_match_interpolated_oracle_and_parameter_gradients(
    train_backend,
):
    net, netcon = _build_training_net(train_backend)
    netcon.w.retain_grad()
    netcon.delay_ms.w.retain_grad()

    trace = []
    for spikes in TRAIN_TAPE:
        net.netstim.spikes.copy_(torch.tensor(spikes, dtype=torch.bool))
        netcon.advance()
        trace.append(netcon.syn.g.flatten().clone())
    actual = torch.stack(trace)

    oracle_weights = netcon.w.detach().clone().requires_grad_()
    oracle_delays = netcon.delay_ms.w.detach().clone().requires_grad_()
    expected = _interpolated_training_oracle(TRAIN_TAPE, oracle_weights, oracle_delays)
    coefficients = torch.linspace(-0.75, 1.25, actual.numel(), dtype=DTYPE).reshape_as(
        actual
    )
    actual_loss = (actual * coefficients).sum()
    expected_loss = (expected * coefficients).sum()
    actual_loss.backward()
    expected_loss.backward()

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(netcon.w.grad, oracle_weights.grad)
    torch.testing.assert_close(netcon.delay_ms.w.grad, oracle_delays.grad)
