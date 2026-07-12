"""Hypothesis-powered lifecycle contracts for complete CPU Networks."""

from __future__ import annotations

import copy

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import dendra as dn
from dendra.models.mod import expsyn, pas

DTYPE = torch.float64
_Leak = pas.rename("hypothesis_lifecycle_leak")
_Synapse = expsyn.rename("hypothesis_lifecycle_synapse")


@st.composite
def _network_configs(draw):
    return {
        "n": draw(st.integers(min_value=1, max_value=3)),
        "dt": draw(st.sampled_from((0.025, 0.05, 0.1))),
        "start_steps": draw(st.integers(min_value=0, max_value=2)),
        "interval_steps": draw(st.integers(min_value=2, max_value=5)),
        "delay_steps": draw(st.integers(min_value=0, max_value=3)),
        "max_spikes": draw(st.integers(min_value=2, max_value=6)),
        "weight": draw(st.integers(min_value=2, max_value=12)) / 16.0,
        "leak_g": draw(st.integers(min_value=1, max_value=5)) / 1024.0,
    }


def _build_network(
    config,
    *,
    training=False,
    leak_g=None,
    trainable_weight=False,
    track_events=True,
    delay_backend="dense",
    train_backend="dense",
):
    n = config["n"]
    dt = config["dt"]
    leak_g = config["leak_g"] if leak_g is None else leak_g

    post = dn.SingleCompartment(N=n, C=1, v_init=-60.0, dtype=DTYPE)
    post.insert(_Leak, g=torch.tensor(leak_g, dtype=DTYPE), e=-70.0)
    post.insert(_Synapse, e=0.0, tau=0.4)
    stim = dn.NetStim(
        N=n,
        interval=config["interval_steps"] * dt,
        start=config["start_steps"] * dt,
        noise=0.0,
        max_spikes=config["max_spikes"],
        seed=1701,
        dtype=DTYPE,
    )
    net = dn.Network(
        {"post": post},
        netstim=stim,
        track_netcon_events=track_events,
        netcon_delay_backend=delay_backend,
        netcon_train_backend=train_backend,
    )
    weight = torch.full((n,), config["weight"], dtype=DTYPE)
    if trainable_weight:
        weight = torch.nn.Parameter(weight)
    delay_steps = torch.as_tensor(config["delay_steps"], dtype=DTYPE).reshape(-1)
    if delay_steps.numel() == 1:
        delay_steps = delay_steps.expand(n)
    assert delay_steps.numel() == n
    net.connect_one_to_one(
        net.netstim[:],
        net.post[:],
        net.post.mech.hypothesis_lifecycle_synapse,
        threshold=None,
        weight=weight,
        delay=delay_steps * dt,
    )
    net.train(training)
    # Materialize NetCons first so their differentiability policy can be fixed
    # before initialization or any pending traffic exists.  A later mode switch
    # can then select a compatible runtime without clearing histories.
    net.build(dt)
    _configure_differentiation(net, diff_weights=trainable_weight)
    net.initialize(dt)
    return net


def _configure_differentiation(net, *, diff_weights=False):
    """Configure hard-event training semantics before an episode starts."""
    net.set_synaptic_diff_config(
        diff_weights=diff_weights,
        diff_delays=False,
        diff_spiking=False,
        taps=2,
        diff_scheduled_times=False,
        train_delay_backend=net.netcon_train_backend,
    )


def _leak(net):
    return net.post.mech.hypothesis_lifecycle_leak


def _synapse(net):
    return net.post.mech.hypothesis_lifecycle_synapse


def _netcon(net):
    return next(iter(net.synapses.values()))


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        cloned = value.__class__(
            (name, _clone_nested(item)) for name, item in value.items()
        )
        if hasattr(value, "_metadata"):
            cloned._metadata = copy.deepcopy(value._metadata)
        return cloned
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return copy.deepcopy(value)


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            _assert_nested_equal(actual[name], expected[name])
        return
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _comparable_checkpoint(net):
    return _comparable_checkpoint_value(net.state_dict_for_checkpoint())


def _comparable_checkpoint_value(value):
    checkpoint = _clone_nested(value)
    # The private seeder is intentionally nondeterministic and is used only to
    # seed a newly created generator. The active deterministic RNG is retained.
    checkpoint["netstim"].pop("seeder_state")
    return checkpoint


def _comparable_runtime(net):
    checkpoint = _comparable_checkpoint(net)
    # Mode and immutable differentiation policy live in structure. Removing
    # only that fingerprint lets a switched candidate be compared step-by-step
    # with a never-switched physical reference while every mutable owned tensor
    # (including event_queue/events and ring counters) remains under assertion.
    checkpoint.pop("structure")
    return checkpoint


def _assert_guarded_runtime_matches_checkpoint(net, checkpoint):
    """Compare live owned state while Network checkpointing is fail-closed."""
    _assert_nested_equal(
        net.populations_state_dict_for_checkpoint(), checkpoint["populations"]
    )
    _assert_nested_equal(net.netstim_state_dict_for_checkpoint(), checkpoint["netstim"])
    assert net.t == checkpoint["t"]
    _assert_nested_equal(net._clock_origin, checkpoint["clock_origin"])
    _assert_nested_equal(net._clock_step, checkpoint["clock_step"])
    _assert_nested_equal(net._duration_remainder, checkpoint["duration_remainder"])

    live = _netcon(net)
    saved = checkpoint["netcons"]["event"][next(iter(net.synapses))]
    for name in (
        "delivery_buffer",
        "current_time_step",
        "global_step",
        "has_spiked",
        "events",
        "event_queue",
    ):
        if name in saved:
            assert torch.equal(getattr(live, name), saved[name])


def _assert_mode_propagated(net):
    assert all(pop.training is net.training for pop in net.populations.values())
    assert all(syn.training is net.training for syn in net.synapses.values())
    assert net.netstim.training is net.training
    for syn in net.synapses.values():
        if net.training:
            assert syn.advance.__name__ == "advance_diff"
        else:
            assert syn.advance.__name__.startswith("advance_non_diff_dense")


def _assert_network_equal(actual, expected):
    _assert_nested_equal(
        _comparable_checkpoint(actual), _comparable_checkpoint(expected)
    )
    torch.testing.assert_close(_leak(actual).g, _leak(expected).g, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        _leak(actual).g_param, _leak(expected).g_param, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        _netcon(actual).w, _netcon(expected).w, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        _netcon(actual).delay_ms.w,
        _netcon(expected).delay_ms.w,
        rtol=0.0,
        atol=0.0,
    )
    assert actual.training is expected.training
    _assert_mode_propagated(actual)
    _assert_mode_propagated(expected)


_OPERATIONS = st.lists(
    st.sampled_from(
        (
            "step_vs_run",
            "longrun_vs_run",
            "initialize",
            "train",
            "eval",
            "update_parameter",
            "checkpoint",
            "restore",
        )
    ),
    min_size=1,
    max_size=10,
)


@given(config=_network_configs(), operations=_OPERATIONS)
@settings(max_examples=60, deadline=None)
def test_generated_network_lifecycle_sequences_preserve_owned_state(config, operations):
    actual = _build_network(config)
    expected = _build_network(config)
    actual_checkpoint = expected_checkpoint = None
    checkpoint_mode = None
    current_leak = config["leak_g"]

    for operation in operations:
        if operation == "step_vs_run":
            actual.step()
            expected.run(config["dt"])
        elif operation == "longrun_vs_run":
            actual.longrun(2 * config["dt"], chunklength=1)
            expected.run(2 * config["dt"])
        elif operation == "initialize":
            actual.initialize(config["dt"])
            # Reinitialization is an episode reset. Compare it to a genuinely
            # fresh object instead of applying the same implementation to both
            # sides and allowing a shared stale-state bug to pass.
            expected = _build_network(
                config,
                training=actual.training,
                leak_g=current_leak,
            )
        elif operation == "train":
            actual.train()
            expected.train()
        elif operation == "eval":
            # These public spellings must be lifecycle-equivalent.
            actual.train(False)
            expected.eval()
        elif operation == "update_parameter":
            current_leak = (
                config["leak_g"] * 1.75
                if current_leak == config["leak_g"]
                else config["leak_g"]
            )
            for net in (actual, expected):
                _leak(net).parameter_set_(g_param=current_leak)
                net.initialize(config["dt"])
            # Runtime checkpoints deliberately exclude model parameters.
            actual_checkpoint = expected_checkpoint = None
            checkpoint_mode = None
            fresh = _build_network(
                config, training=actual.training, leak_g=current_leak
            )
            _assert_network_equal(actual, fresh)
        elif operation == "checkpoint":
            actual_checkpoint = _clone_nested(actual.state_dict_for_checkpoint())
            expected_checkpoint = _clone_nested(expected.state_dict_for_checkpoint())
            checkpoint_mode = actual.training
        elif operation == "restore" and actual_checkpoint is not None:
            if actual.training is checkpoint_mode:
                actual.restore_dict_from_checkpoint(_clone_nested(actual_checkpoint))
                expected.restore_dict_from_checkpoint(
                    _clone_nested(expected_checkpoint)
                )
                _assert_nested_equal(
                    _comparable_checkpoint(actual),
                    _comparable_checkpoint_value(actual_checkpoint),
                )
                _assert_nested_equal(
                    _comparable_checkpoint(expected),
                    _comparable_checkpoint_value(expected_checkpoint),
                )
            else:
                before_actual = _clone_nested(actual.state_dict_for_checkpoint())
                before_expected = _clone_nested(expected.state_dict_for_checkpoint())
                with pytest.raises(ValueError, match="topology or runtime mode"):
                    actual.restore_dict_from_checkpoint(
                        _clone_nested(actual_checkpoint)
                    )
                with pytest.raises(ValueError, match="topology or runtime mode"):
                    expected.restore_dict_from_checkpoint(
                        _clone_nested(expected_checkpoint)
                    )
                _assert_nested_equal(actual.state_dict_for_checkpoint(), before_actual)
                _assert_nested_equal(
                    expected.state_dict_for_checkpoint(), before_expected
                )

        _assert_network_equal(actual, expected)
        assert torch.isfinite(actual.post.v).all()
        assert torch.isfinite(_synapse(actual).g).all()


@given(
    config=_network_configs(),
    total_steps=st.integers(min_value=1, max_value=10),
    split_fraction=st.integers(min_value=0, max_value=10),
    chunklength=st.integers(min_value=1, max_value=4),
    training=st.booleans(),
)
@settings(max_examples=25, deadline=None)
def test_generated_checkpoint_replay_and_chunking_match_uninterrupted(
    config, total_steps, split_fraction, chunklength, training
):
    split = min(total_steps, split_fraction)
    duration = total_steps * config["dt"]

    uninterrupted = _build_network(config, training=training)
    uninterrupted.run(duration)

    source = _build_network(config, training=training)
    source.run(split * config["dt"])
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())
    resumed = _build_network(config, training=training)
    resumed.restore_dict_from_checkpoint(checkpoint)
    resumed.run((total_steps - split) * config["dt"])

    chunked = _build_network(config, training=training)
    chunked.longrun(duration, chunklength=chunklength)
    checkpointed = _build_network(config, training=training)
    checkpointed.longrun_checkpointed(duration, chunklength=chunklength)

    for candidate in (resumed, chunked, checkpointed):
        _assert_network_equal(candidate, uninterrupted)


@pytest.mark.parametrize("delay_backend", ("dense", "sparse_calendar"))
def test_tracked_event_checkpoint_preserves_current_and_pending_counts(delay_backend):
    config = {
        "n": 3,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": (1, 2, 3),
        "max_spikes": 5,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    source = _build_network(config, delay_backend=delay_backend)
    for _ in range(8):
        source.step()
        netcon = _netcon(source)
        if hasattr(netcon, "event_queue"):
            has_pending_events = bool(torch.count_nonzero(netcon.event_queue))
        else:
            has_pending_events = any(netcon._sparse_event_calendar.values())
        if torch.count_nonzero(netcon.events) and has_pending_events:
            break

    assert torch.count_nonzero(_netcon(source).events) > 0
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())
    saved = checkpoint["netcons"]["event"][next(iter(source.synapses))]
    assert torch.equal(saved["events"], _netcon(source).events)
    if delay_backend == "dense":
        assert torch.count_nonzero(_netcon(source).event_queue) > 0
        assert torch.equal(saved["event_queue"], _netcon(source).event_queue)
    else:
        assert saved["backend_state"]["sparse_event_calendar"]

    resumed = _build_network(config, delay_backend=delay_backend)
    resumed.restore_dict_from_checkpoint(checkpoint)
    assert torch.equal(_netcon(resumed).events, _netcon(source).events)
    _assert_nested_equal(
        _comparable_checkpoint(resumed), _comparable_checkpoint(source)
    )

    for _ in range(8):
        source.step()
        resumed.step()
        _assert_nested_equal(
            _comparable_checkpoint(source), _comparable_checkpoint(resumed)
        )


def _network_loss(net, steps):
    loss = torch.zeros((), dtype=DTYPE)
    for step_index in range(steps):
        net.step()
        scale = float(step_index + 1)
        loss = loss + scale * (_synapse(net).g.sum() + 0.001 * net.post.v.sum())
    return loss


@given(
    n=st.integers(min_value=1, max_value=3),
    delay_steps=st.integers(min_value=0, max_value=2),
    weight_index=st.integers(min_value=2, max_value=8),
    steps=st.integers(min_value=5, max_value=9),
)
@settings(max_examples=20, deadline=None)
def test_generated_network_weight_gradient_matches_central_difference(
    n, delay_steps, weight_index, steps
):
    weight = weight_index / 10.0
    config = {
        "n": n,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": delay_steps,
        "max_spikes": 8,
        "weight": weight,
        "leak_g": 1.0 / 1024.0,
    }
    network = _build_network(config, training=True, trainable_weight=True)
    netcon = _netcon(network)
    loss = _network_loss(network, steps)
    gradient = torch.autograd.grad(loss, netcon.w)[0].sum()

    eps = 1.0e-5
    plus_config = dict(config, weight=weight + eps)
    minus_config = dict(config, weight=weight - eps)
    plus = _network_loss(_build_network(plus_config), steps)
    minus = _network_loss(_build_network(minus_config), steps)
    finite_difference = (plus - minus) / (2 * eps)

    assert abs(float(gradient)) > 1.0e-3
    torch.testing.assert_close(gradient, finite_difference, rtol=3.0e-6, atol=2.0e-6)


@pytest.mark.parametrize("component", ("population", "netstim", "netcon", "clock"))
@given(config=_network_configs())
@settings(max_examples=6, deadline=None)
def test_generated_malformed_owned_checkpoint_is_rejected_without_mutation(
    config, component
):
    net = _build_network(config)
    net.run(2 * config["dt"])
    before = _clone_nested(net.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)

    if component == "population":
        voltage = corrupt["populations"]["post"]["integrator"]["v"]
        corrupt["populations"]["post"]["integrator"]["v"] = torch.cat(
            (voltage, voltage[:1]), dim=0
        )
    elif component == "netstim":
        corrupt["netstim"]["shape"] = (config["n"] + 1,)
    elif component == "netcon":
        name = next(iter(corrupt["netcons"]["event"]))
        delivery = corrupt["netcons"]["event"][name]["delivery_buffer"]
        corrupt["netcons"]["event"][name]["delivery_buffer"] = torch.cat(
            (delivery, delivery[..., :1]), dim=-1
        )
    else:
        corrupt["clock_step"] = torch.tensor(-1, dtype=torch.long)

    with pytest.raises((KeyError, TypeError, ValueError, RuntimeError)):
        net.restore_dict_from_checkpoint(corrupt)
    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("failure_site", ("population", "netcon", "netstim", "clock"))
def test_network_checkpoint_restore_rolls_back_after_late_failure(
    monkeypatch, failure_site
):
    config = {
        "n": 2,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": 3,
        "max_spikes": 5,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(config)
    net.run(3 * config["dt"])
    before = _clone_nested(net.state_dict_for_checkpoint())
    candidate = _clone_nested(before)
    # This valid, shape-preserving change is applied before each injected late
    # failure. Without the Network transaction wrapper, partial restoration is
    # therefore observable rather than accidentally equal to the live state.
    candidate["populations"]["post"]["integrator"]["v"].add_(7.0)

    calls = 0
    if failure_site == "population":
        owner = net.post
        method_name = "restore_dict_from_checkpoint"
    elif failure_site == "netcon":
        owner = _netcon(net)
        method_name = "restore_dict_from_checkpoint"
    elif failure_site == "netstim":
        owner = net.netstim
        method_name = "restore_dict_from_checkpoint"
    else:
        owner = net
        method_name = "_sync_runtime_clock"
    original = getattr(owner, method_name)

    def fail_once(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        if calls == 0:
            calls += 1
            raise RuntimeError(f"injected late {failure_site} restore failure")
        return result

    monkeypatch.setattr(owner, method_name, fail_once)
    with pytest.raises(RuntimeError, match=f"late {failure_site} restore failure"):
        net.restore_dict_from_checkpoint(candidate)

    assert calls == 1
    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


def test_checkpoint_from_other_mode_is_rejected_atomically():
    config = {
        "n": 1,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": 2,
        "max_spikes": 3,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(config)
    checkpoint = _clone_nested(net.state_dict_for_checkpoint())
    net.train()
    before = _clone_nested(net.state_dict_for_checkpoint())

    with pytest.raises(ValueError, match="topology or runtime mode"):
        net.restore_dict_from_checkpoint(checkpoint)

    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


def test_dense_mode_switches_preserve_pending_traffic_and_physical_suffix():
    config = {
        "n": 3,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": (1, 2, 3),
        "max_spikes": 5,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    switched = _build_network(config)
    reference = _build_network(config)

    for _ in range(2):
        switched.step()
        reference.step()
    assert torch.count_nonzero(_netcon(switched).delivery_buffer) > 0
    pending = _netcon(switched).delivery_buffer.detach().clone()
    slot = _netcon(switched).current_time_step.detach().clone()

    switched.train()
    torch.testing.assert_close(_netcon(switched).delivery_buffer, pending)
    torch.testing.assert_close(_netcon(switched).current_time_step, slot)
    _assert_mode_propagated(switched)
    _assert_nested_equal(_comparable_runtime(switched), _comparable_runtime(reference))

    # Exercise the differentiable hard-event runtime while the reference stays
    # in inference. With frozen weights/delays/spikes, every physical and event-
    # introspection tensor must remain identical at every transient step.
    for _ in range(4):
        switched.step()
        reference.step()
        _assert_nested_equal(
            _comparable_runtime(switched), _comparable_runtime(reference)
        )
    switched.eval()
    _assert_mode_propagated(switched)
    for _ in range(6):
        switched.step()
        reference.step()
        _assert_nested_equal(
            _comparable_runtime(switched), _comparable_runtime(reference)
        )

    _assert_network_equal(switched, reference)


@pytest.mark.parametrize(
    "ambiguous",
    (
        {"diff_delays": True},
        {"diff_spiking": True},
        {"diff_scheduled_times": True},
    ),
)
def test_tracked_training_events_reject_ambiguous_surrogate_counts(ambiguous):
    config = {
        "n": 1,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": 2,
        "max_spikes": 3,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(config)
    netcon = _netcon(net)
    before_flags = tuple(netcon.train_flags)
    kwargs = {
        "diff_weights": False,
        "diff_delays": False,
        "diff_spiking": False,
        "diff_scheduled_times": False,
        "train_delay_backend": "dense",
    }
    kwargs.update(ambiguous)

    with pytest.raises(ValueError, match="hard event semantics"):
        netcon.set_diff_config(**kwargs)

    assert tuple(netcon.train_flags) == before_flags


@pytest.mark.parametrize("entrypoint", ("step", "run", "longrun", "checkpointed"))
def test_compact_mode_switch_requires_reinitialize_without_mutation(entrypoint):
    config = {
        "n": 2,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": (2, 3),
        "max_spikes": 4,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(
        config,
        track_events=False,
        train_backend="source_history",
    )
    net.run(2 * config["dt"])
    pending = _clone_nested(net.state_dict_for_checkpoint())
    net.train()

    with pytest.raises(RuntimeError, match="incompatible NetCon runtime layouts"):
        net.state_dict_for_checkpoint()
    with pytest.raises(RuntimeError, match="incompatible NetCon runtime layouts"):
        net.checkpoint_structure()

    with pytest.raises(RuntimeError, match="incompatible NetCon runtime layouts"):
        if entrypoint == "step":
            net.step()
        elif entrypoint == "run":
            net.run(config["dt"])
        elif entrypoint == "longrun":
            net.longrun(config["dt"], chunklength=1)
        else:
            net.longrun_checkpointed(config["dt"], chunklength=1)

    _assert_guarded_runtime_matches_checkpoint(net, pending)
    # The mode toggle itself must not have destroyed the inference traffic.
    torch.testing.assert_close(
        _netcon(net).delivery_buffer,
        pending["netcons"]["event"][next(iter(pending["netcons"]["event"]))][
            "delivery_buffer"
        ],
    )
    if entrypoint == "step":
        # Rebuilding connection modules alone must not bypass the episode-level
        # initialization requirement after an incompatible mode transition.
        net.build(config["dt"], force_rebuild=True)
        with pytest.raises(RuntimeError, match="incompatible NetCon runtime layouts"):
            net.step()
    _configure_differentiation(net)
    net.initialize(config["dt"])
    assert not _netcon(net)._mode_requires_initialize
    net.step()


def test_diff_reconfiguration_cannot_bypass_compact_mode_guard():
    config = {
        "n": 2,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": (1, 2),
        "max_spikes": 4,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(
        config,
        track_events=False,
        train_backend="source_history",
    )
    net.run(2 * config["dt"])
    pending = _netcon(net).delivery_buffer.detach().clone()
    net.train()

    _configure_differentiation(net)

    netcon = _netcon(net)
    assert netcon.advance.__name__ == "_advance_requires_mode_initialize"
    torch.testing.assert_close(netcon.delivery_buffer, pending)
    with pytest.raises(RuntimeError, match="incompatible runtime layouts"):
        netcon.advance()


def test_invalid_strict_source_history_config_is_atomic_in_eval():
    config = {
        "n": 2,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": (1, 2),
        "max_spikes": 4,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(config, track_events=True, train_backend="dense")
    netcon = _netcon(net)
    before = _clone_nested(net.state_dict_for_checkpoint())
    before_flags = tuple(netcon.train_flags)
    before_advance = netcon.advance.__name__

    with pytest.raises(ValueError, match="source_history.*not exact"):
        netcon.set_diff_config(
            diff_weights=False,
            diff_delays=False,
            diff_spiking=False,
            diff_scheduled_times=False,
            train_delay_backend="source_history",
        )

    assert netcon.train_delay_backend == "dense"
    assert tuple(netcon.train_flags) == before_flags
    assert netcon.advance.__name__ == before_advance
    _assert_nested_equal(net.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("caller", ("network", "netcon"))
def test_rejected_mode_transition_is_atomic_after_schedule_added(caller):
    config = {
        "n": 2,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": (1, 2),
        "max_spikes": 4,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(
        config,
        track_events=False,
        train_backend="source_history",
    )
    netcon = _netcon(net)
    netcon.schedule(con_indices=[0], times_ms=[net.t + 3 * config["dt"]])
    before = _clone_nested(net.state_dict_for_checkpoint())
    before_advance = netcon.advance.__name__
    before_step = net._step.__name__

    with pytest.raises(RuntimeError, match="intrinsic source-level events only"):
        if caller == "network":
            net.train()
        else:
            netcon.train()

    assert net.training is False
    assert netcon.advance.__name__ == before_advance
    assert net._step.__name__ == before_step
    assert not net._mode_requires_initialize
    assert not netcon._mode_requires_initialize
    _assert_mode_propagated(net)
    _assert_nested_equal(net.state_dict_for_checkpoint(), before)
    # The rejection leaves the existing inference episode runnable.
    net.step()


def test_train_false_selects_eval_step_and_propagates_mode():
    config = {
        "n": 1,
        "dt": 0.05,
        "start_steps": 0,
        "interval_steps": 2,
        "delay_steps": 1,
        "max_spikes": 2,
        "weight": 0.5,
        "leak_g": 1.0 / 1024.0,
    }
    net = _build_network(config)

    def train_step(*args, **kwargs):  # pragma: no cover - selection sentinel
        raise AssertionError("training sentinel should not execute")

    def eval_step(*args, **kwargs):  # pragma: no cover - selection sentinel
        raise AssertionError("evaluation sentinel should not execute")

    net._step_train = train_step
    net._step_eval = eval_step
    net.train(False)
    assert net._step is eval_step
    _assert_mode_propagated(net)
    net.train(True)
    assert net._step is train_step
    _assert_mode_propagated(net)
