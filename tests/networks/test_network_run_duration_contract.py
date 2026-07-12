"""Contracts for carrying fractional fixed-step durations across Network.run calls."""

import math

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn

DTYPE = torch.float64


class _StepTrace(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []
        self.conductance = []
        self.events = []

    def post_step_hook(self, model):
        netcon = next(iter(model.synapses.values()))
        self.times.append(model.t.detach().clone())
        self.conductance.append(model.post.mech.syn.g.detach().clone())
        self.events.append(netcon.events.detach().clone())


def _built_network(dt):
    post = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(
        N=1,
        interval=100.0,
        start=100.0,
        noise=0.0,
        max_spikes=1,
        seed=1729,
        dtype=DTYPE,
    )
    net = dn.Network({"post": post}, netstim=stim, track_netcon_events=True)
    net.connect_one_to_one(
        stim[:],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=2.0,
        delay=dt,
    )
    net.initialize(dt)
    return net, next(iter(net.synapses.values()))


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


def _duration_remainder(net):
    checkpoint = net.state_dict_for_checkpoint()
    assert "duration_remainder" in checkpoint
    remainder = checkpoint["duration_remainder"]
    assert torch.is_tensor(remainder)
    assert remainder.shape == ()
    return remainder.item()


def _advance_with_entrypoint(net, entrypoint, duration):
    if entrypoint == "run":
        net.run(duration)
    elif entrypoint == "longrun":
        net.longrun(duration, chunklength=2)
    elif entrypoint == "checkpointed":
        net.longrun_checkpointed(duration, chunklength=2)
    else:  # pragma: no cover - local test helper contract
        raise ValueError(entrypoint)


def _run_scheduled_chunks(durations):
    dt = 0.1
    net, netcon = _built_network(dt)
    netcon.schedule(con_indices=[0], times_ms=[dt], weight=3.0)
    trace = _StepTrace()
    for duration in durations:
        net.run(duration, callbacks=[trace])
    return net, trace


def test_fractional_run_chunks_match_one_complete_run_with_scheduled_delivery():
    """Two 0.15-ms calls must retain the half-step needed to reach 0.3 ms."""
    whole, whole_trace = _run_scheduled_chunks([0.3])
    chunked, chunked_trace = _run_scheduled_chunks([0.15, 0.15])

    assert [time.item() for time in whole_trace.times] == pytest.approx([0.1, 0.2, 0.3])
    assert len(chunked_trace.times) == len(whole_trace.times) == 3
    for actual, expected in zip(chunked_trace.times, whole_trace.times):
        assert torch.equal(actual, expected)
    for actual, expected in zip(chunked_trace.conductance, whole_trace.conductance):
        assert torch.equal(actual, expected)
    for actual, expected in zip(chunked_trace.events, whole_trace.events):
        assert torch.equal(actual, expected)

    # The event is recognized at absolute step 1 and delivered at step 2.  Its
    # delivery therefore crosses the boundary between the two run() calls.
    assert [events.item() for events in chunked_trace.events] == [0, 0, 1]
    expected_g = 6.0 * math.exp(-0.1)
    assert chunked.post.mech.syn.g.item() == pytest.approx(expected_g)
    chunked_state = _clone_nested(chunked.state_dict_for_checkpoint())
    whole_state = _clone_nested(whole.state_dict_for_checkpoint())
    # The private seeder only initializes a newly created device RNG. The active
    # RNG itself is deterministically seeded and is part of the replay contract.
    chunked_state["netstim"].pop("seeder_state")
    whole_state["netstim"].pop("seeder_state")
    _assert_nested_equal(chunked_state, whole_state)


def test_multiple_tiny_run_calls_accumulate_complete_steps_without_loss():
    net, _ = _built_network(0.1)
    trace = _StepTrace()

    for _ in range(10):
        net.run(0.01, callbacks=[trace])

    assert len(trace.times) == 1
    assert net.t.item() == pytest.approx(0.1)
    assert net._clock_step.item() == 1


@pytest.mark.parametrize(
    ("relation", "expected_steps"),
    [("before", 0), ("exact", 1), ("after", 1)],
)
def test_run_duration_distinguishes_adjacent_binary_grid_boundaries(
    relation, expected_steps
):
    # A binary-exact dt isolates genuine adjacent values from decimal 0.1
    # representation effects.  A duration one ULP below dt is a real partial
    # step and must not be promoted to a complete step.
    dt = 0.125
    if relation == "before":
        duration = math.nextafter(dt, -math.inf)
    elif relation == "after":
        duration = math.nextafter(dt, math.inf)
    else:
        duration = dt

    net, _ = _built_network(dt)
    trace = _StepTrace()
    net.run(duration, callbacks=[trace])

    assert len(trace.times) == expected_steps
    assert net.t.item() == pytest.approx(expected_steps * dt)

    if relation == "before":
        # The missing ULP completes the retained partial duration exactly.
        net.run(dt - duration, callbacks=[trace])
        assert len(trace.times) == 1
        assert net.t.item() == pytest.approx(dt)


def test_initialize_discards_old_fractional_duration_and_starts_a_fresh_carry():
    dt = 0.1
    net, _ = _built_network(dt)
    net.run(0.15)
    assert net.t.item() == pytest.approx(dt)

    # Reinitialization is an episode boundary: the retained 0.05 ms from the
    # previous episode must not contribute to this new one.
    net.initialize(dt)
    trace = _StepTrace()
    net.run(0.05, callbacks=[trace])
    assert trace.times == []
    assert net.t.item() == 0.0

    # The new episode develops its own carry normally.
    net.run(0.05, callbacks=[trace])
    assert len(trace.times) == 1
    assert net.t.item() == pytest.approx(dt)


def test_run_longrun_and_checkpointed_execution_share_one_fractional_carry():
    net, _ = _built_network(0.1)

    net.run(0.04)
    assert net.t.item() == 0.0
    assert _duration_remainder(net) == pytest.approx(0.04)

    net.longrun(0.03, chunklength=2)
    assert net.t.item() == 0.0
    assert _duration_remainder(net) == pytest.approx(0.07)

    net.longrun_checkpointed(0.03, chunklength=2)
    assert net.t.item() == pytest.approx(0.1)
    assert net._clock_step.item() == 1
    assert _duration_remainder(net) == pytest.approx(0.0, abs=1.0e-15)


def test_runtime_checkpoint_restores_fractional_carry_for_exact_continuation():
    source, _ = _built_network(0.1)
    source.run(0.15)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    assert checkpoint["duration_remainder"].item() == pytest.approx(0.05)

    source.run(0.05)
    expected = _clone_nested(source.state_dict_for_checkpoint())

    resumed, _ = _built_network(0.1)
    resumed.restore_dict_from_checkpoint(checkpoint)
    assert _duration_remainder(resumed) == pytest.approx(0.05)
    resumed.run(0.05)

    _assert_nested_equal(resumed.state_dict_for_checkpoint(), expected)


def test_legacy_runtime_checkpoint_without_fractional_carry_restores_zero():
    source, _ = _built_network(0.1)
    source.run(0.1)
    legacy = _clone_nested(source.state_dict_for_checkpoint())
    legacy.pop("duration_remainder")

    resumed, _ = _built_network(0.1)
    resumed.run(0.15)
    assert _duration_remainder(resumed) == pytest.approx(0.05)

    resumed.restore_dict_from_checkpoint(legacy)
    assert _duration_remainder(resumed) == 0.0

    # With the receiver's old carry cleared, the first half-step remains pending.
    resumed.run(0.05)
    assert resumed.t.item() == pytest.approx(0.1)
    assert _duration_remainder(resumed) == pytest.approx(0.05)
    resumed.run(0.05)
    assert resumed.t.item() == pytest.approx(0.2)


def test_legacy_normal_state_without_fractional_carry_clears_receiver():
    source, _ = _built_network(0.1)
    legacy = _clone_nested(source.state_dict())
    legacy.pop("_duration_remainder")

    resumed, _ = _built_network(0.1)
    resumed.run(0.05)
    assert _duration_remainder(resumed) == pytest.approx(0.05)

    resumed.load(legacy)

    assert _duration_remainder(resumed) == 0.0


@pytest.mark.parametrize(
    ("bad_value", "error"),
    [
        (1.0, TypeError),
        (torch.zeros(1), ValueError),
        (torch.tensor(0), TypeError),
        (torch.tensor(float("nan")), ValueError),
        (torch.tensor(float("inf")), ValueError),
        (torch.tensor(-0.1), ValueError),
    ],
)
def test_runtime_checkpoint_validates_fractional_carry_atomically(bad_value, error):
    net, _ = _built_network(0.1)
    net.run(0.05)
    before = _clone_nested(net.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)
    corrupt["duration_remainder"] = bad_value

    with pytest.raises(error, match="duration remainder"):
        net.restore_dict_from_checkpoint(corrupt)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("entrypoint", ["run", "longrun", "checkpointed"])
@pytest.mark.parametrize("invalid", [-0.1, math.inf, math.nan, True])
def test_invalid_duration_is_atomic_with_a_nonzero_fractional_carry(
    entrypoint, invalid
):
    net, _ = _built_network(0.1)
    net.run(0.05)
    before = _clone_nested(net.state_dict_for_checkpoint())
    assert before["duration_remainder"].item() == pytest.approx(0.05)

    with pytest.raises((TypeError, ValueError), match="tstop"):
        _advance_with_entrypoint(net, entrypoint, invalid)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


def test_explicit_step_preserves_fractional_carry():
    net, _ = _built_network(0.1)
    net.run(0.05)
    assert _duration_remainder(net) == pytest.approx(0.05)

    net.step()
    assert net.t.item() == pytest.approx(0.1)
    assert net._clock_step.item() == 1
    assert _duration_remainder(net) == pytest.approx(0.05)

    net.run(0.05)
    assert net.t.item() == pytest.approx(0.2)
    assert net._clock_step.item() == 2
    assert _duration_remainder(net) == pytest.approx(0.0, abs=1.0e-15)


def test_new_timestep_build_preserves_physical_carry_but_initialize_clears_it():
    net, _ = _built_network(0.1)
    net.run(0.05)
    assert _duration_remainder(net) == pytest.approx(0.05)

    # Rebuilding changes the grid but not the amount of requested physical time
    # that has not yet formed a complete step.  The retained 0.05 ms combines
    # with 0.15 ms on the new grid to execute one 0.2-ms step.
    net.build(0.2)
    net.init_synapses()
    assert _duration_remainder(net) == pytest.approx(0.05)
    net.run(0.15)
    assert net.t.item() == pytest.approx(0.2)
    assert _duration_remainder(net) == pytest.approx(0.0, abs=1.0e-15)

    net.run(0.05)
    assert _duration_remainder(net) == pytest.approx(0.05)

    # initialize() starts a new simulation episode and therefore discards carry.
    net.initialize(0.2)
    assert _duration_remainder(net) == 0.0
    net.run(0.15)
    assert net.t.item() == 0.0
    assert _duration_remainder(net) == pytest.approx(0.15)


def test_huge_duration_with_tiny_timestep_raises_clean_value_error():
    dt = 1.0e-10
    post = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    net = dn.Network({"post": post})
    net.initialize(dt)
    before = _clone_nested(net.state_dict_for_checkpoint())

    with pytest.raises(ValueError, match="duration|steps|timestep"):
        net.run(1.0e308)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)
