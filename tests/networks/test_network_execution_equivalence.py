"""Whole-network execution entrypoints must preserve event-driven state."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, hh, pas
from dendra.units import nA

DT = 0.025
DTYPE = torch.float64
TOTAL_STEPS = 157
CHUNK_STEPS = 7
PULSE_START_STEP = CHUNK_STEPS
PULSE_STEPS = 6
DELAY_STEPS = 5
EXTRA_STEPS = 50
EXTRA_START_STEP = CHUNK_STEPS - 1
EXTRA_PULSE_STEPS = CHUNK_STEPS + 2


class _NetworkTrace(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.pre_loop_calls = 0
        self.post_loop_calls = 0
        self.pre_chunks = []
        self.post_chunks = []
        self.records = []

    def pre_loop_hook(self, model):
        self.pre_loop_calls += 1

    def post_loop_hook(self, model):
        self.post_loop_calls += 1

    def pre_chunk_hook(self, model, timepoints=None):
        self.pre_chunks.append(timepoints.detach().clone())

    def post_chunk_hook(self, model, timepoints=None):
        self.post_chunks.append(timepoints.detach().clone())

    def post_step_hook(self, model):
        netcon = next(iter(model.synapses.values()))
        self.records.append(
            {
                "t": model.t.detach().clone(),
                "pre_v": model.pre.v.detach().clone(),
                "post_v": model.post.v.detach().clone(),
                "syn_g": model.post.mech.syn.g.detach().clone(),
                "events": netcon.events.detach().clone(),
                "delivery_buffer": netcon.delivery_buffer.detach().clone(),
                "has_spiked": netcon.has_spiked.detach().clone(),
                "is_spiking": netcon.is_spiking.detach().clone(),
                "global_step": netcon.global_step.detach().clone(),
            }
        )


class _CableTrace(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.time = []
        self.voltage = []

    def post_step_hook(self, model):
        self.time.append(model.t.detach().clone())
        self.voltage.append(model.cable.v.detach().clone())


class _EmptyCheckpointLifecycle(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(3.0, dtype=DTYPE))
        self.pre_loop_calls = 0
        self.post_loop_calls = 0
        self.pre_chunk_calls = 0
        self.post_chunk_calls = 0
        self.post_step_calls = 0

    def pre_loop_hook(self, model):
        self.pre_loop_calls += 1

    def post_loop_hook(self, model):
        self.post_loop_calls += 1
        return self.weight.square()

    def pre_chunk_hook(self, model, timepoints=None):
        self.pre_chunk_calls += 1

    def post_chunk_hook(self, model, timepoints=None):
        self.post_chunk_calls += 1

    def post_step_hook(self, model):
        self.post_step_calls += 1


def _pulse():
    return dn.mono_rect(
        amp=2.0 * nA,
        delay=PULSE_START_STEP * DT,
        pw=PULSE_STEPS * DT,
    )


def _build_network(*, with_injection=True):
    pre = dn.SingleCompartment(
        N=1,
        C=1,
        celsius=6.3,
        v_init=-65.0,
        dtype=DTYPE,
    )
    post = dn.SingleCompartment(
        N=1,
        C=1,
        celsius=6.3,
        v_init=-65.0,
        dtype=DTYPE,
    )
    # Use a compact, independently explicit membrane area so the injected
    # current produces one robust HH spike without relying on Population's
    # deliberately large generic geometry defaults.
    pre.diam.fill_(20.0)
    pre.dx.fill_(20.0)
    post.diam.fill_(20.0)
    post.dx.fill_(20.0)

    pre.insert(hh)
    if with_injection:
        pre[:].inject(_pulse())
    post.insert(pas, g=0.001, e=-65.0)
    post.insert(expsyn.rename("syn"), e=0.0, tau=0.8)

    network = dn.Network(
        {"pre": pre, "post": post},
        track_netcon_events=True,
        netcon_delay_backend="dense",
    )
    network.connect_one_to_one(
        pre[:],
        post[:],
        post.mech.syn,
        threshold=0.0,
        weight=0.05,
        delay=DELAY_STEPS * DT,
    )
    network.initialize(DT)
    return network


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


def _execute(mode, *, split_step=None, late_injection=False):
    network = _build_network(with_injection=not late_injection)
    if late_injection:
        network.pre[:].inject(_pulse())
    trace = _NetworkTrace()
    duration = TOTAL_STEPS * DT

    if mode == "run":
        network.run(duration, callbacks=[trace])
    elif mode == "longrun":
        network.longrun(duration, chunklength=CHUNK_STEPS, callbacks=[trace])
    elif mode == "checkpointed":
        network.longrun_checkpointed(
            duration,
            chunklength=CHUNK_STEPS,
            callbacks=[trace],
        )
    elif mode == "manual":
        for _ in range(TOTAL_STEPS):
            network.step(callbacks=[trace])
    elif mode == "split":
        assert split_step is not None
        network.run(split_step * DT, callbacks=[trace])
        network.run((TOTAL_STEPS - split_step) * DT, callbacks=[trace])
    else:  # pragma: no cover - local test helper contract
        raise ValueError(mode)

    return trace, _snapshot(network.state_dict_for_checkpoint())


def _stack(trace, name):
    return torch.stack([record[name] for record in trace.records])


def _build_extracellular_cable_network():
    cable = dn.Unmyelinated(
        diameters=[2.0],
        L=100.0,
        dx=20.0,
        v_init=-70.0,
        dtype=DTYPE,
    )
    cable.insert(pas, g=0.001, e=-70.0)
    network = dn.Network({"cable": cable})
    network.initialize(DT)

    spatial = torch.tensor(
        [[-8.0, -3.0, 0.0, 4.0, 11.0]],
        dtype=DTYPE,
    )
    temporal = dn.mono_rect(
        amp=1.0,
        delay=EXTRA_START_STEP * DT,
        pw=EXTRA_PULSE_STEPS * DT,
    )
    return network, {"cable": (spatial, temporal)}


def test_network_execution_entrypoints_preserve_spikes_delays_and_state():
    reference, reference_state = _execute("run")
    pre_v = _stack(reference, "pre_v").flatten()
    post_v = _stack(reference, "post_v").flatten()
    syn_g = _stack(reference, "syn_g").flatten()
    events = _stack(reference, "events").flatten()
    source_events = _stack(reference, "is_spiking").flatten()

    assert pre_v.max() > 20.0
    assert (pre_v > 0.0).logical_and(torch.roll(pre_v <= 0.0, 1)).sum() == 1
    assert syn_g.max() > 0.01
    assert post_v.max() - post_v[0] > 1.0
    assert source_events.sum() == 1
    assert events.sum() == 1

    source_event_step = int(torch.nonzero(source_events, as_tuple=False)[0])
    event_step = int(torch.nonzero(events, as_tuple=False)[0])
    assert event_step > PULSE_START_STEP + PULSE_STEPS
    assert event_step - source_event_step == DELAY_STEPS
    assert PULSE_START_STEP % CHUNK_STEPS == 0
    assert source_event_step // CHUNK_STEPS < event_step // CHUNK_STEPS
    assert TOTAL_STEPS % CHUNK_STEPS != 0

    # Stop immediately before the first delivery. The delayed event must remain
    # pending across the public run() boundary and arrive on the first resumed
    # step.
    split_step = event_step
    assert _stack(reference, "delivery_buffer")[split_step - 1].any()

    mode_traces = {}
    for mode in ("longrun", "checkpointed", "manual", "split"):
        actual, actual_state = _execute(mode, split_step=split_step)
        mode_traces[mode] = actual
        assert len(actual.records) == len(reference.records) == TOTAL_STEPS
        for actual_record, expected_record in zip(actual.records, reference.records):
            _assert_nested_equal(actual_record, expected_record)
        _assert_nested_equal(actual_state, reference_state)

    assert reference.pre_loop_calls == reference.post_loop_calls == 1
    assert mode_traces["manual"].pre_loop_calls == 0
    assert mode_traces["manual"].post_loop_calls == 0
    assert mode_traces["split"].pre_loop_calls == 2
    assert mode_traces["split"].post_loop_calls == 2

    longrun = mode_traces["longrun"]
    expected_chunks = (TOTAL_STEPS + CHUNK_STEPS - 1) // CHUNK_STEPS
    assert longrun.pre_loop_calls == longrun.post_loop_calls == 1
    assert len(longrun.pre_chunks) == len(longrun.post_chunks) == expected_chunks
    assert [len(chunk) for chunk in longrun.pre_chunks[:-1]] == [CHUNK_STEPS] * (
        expected_chunks - 1
    )
    assert len(longrun.pre_chunks[-1]) == TOTAL_STEPS % CHUNK_STEPS
    chunk_time = torch.cat(longrun.pre_chunks)
    expected_time = torch.arange(TOTAL_STEPS, dtype=chunk_time.dtype) * DT
    torch.testing.assert_close(chunk_time, expected_time)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_network_longrun_rejects_invalid_chunklength(invalid):
    network = _build_network()
    before = _snapshot(network.state_dict_for_checkpoint())

    with pytest.raises(ValueError, match="chunklength must be a positive integer"):
        network.longrun(DT, chunklength=invalid)

    _assert_nested_equal(network.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("invalid", [-DT, float("inf"), float("nan"), True])
def test_network_longrun_rejects_invalid_duration_without_mutation(invalid):
    network = _build_network()
    before = _snapshot(network.state_dict_for_checkpoint())

    with pytest.raises((TypeError, ValueError), match="tstop"):
        network.longrun(invalid, chunklength=CHUNK_STEPS)

    _assert_nested_equal(network.state_dict_for_checkpoint(), before)


def test_network_longrun_substep_duration_has_one_empty_loop_lifecycle():
    network = _build_network()
    before = _snapshot(network.state_dict_for_checkpoint())
    expected = _snapshot(before)
    expected["duration_remainder"] = torch.as_tensor(
        DT / 2.0,
        device=before["duration_remainder"].device,
        dtype=before["duration_remainder"].dtype,
    )
    trace = _NetworkTrace()

    network.longrun(DT / 2.0, chunklength=CHUNK_STEPS, callbacks=[trace])

    assert trace.pre_loop_calls == trace.post_loop_calls == 1
    assert trace.pre_chunks == trace.post_chunks == []
    assert trace.records == []
    _assert_nested_equal(network.state_dict_for_checkpoint(), expected)


def test_network_loops_honor_injection_added_after_initialize():
    reference, reference_state = _execute("manual", late_injection=True)
    assert _stack(reference, "pre_v").max() > 20.0
    assert _stack(reference, "events").sum() == 1
    reference_post = _stack(reference, "post_v").flatten()
    assert reference_post.max() - reference_post[0] > 1.0

    for mode in ("run", "longrun", "checkpointed"):
        actual, actual_state = _execute(mode, late_injection=True)
        assert len(actual.records) == len(reference.records) == TOTAL_STEPS
        for actual_record, expected_record in zip(actual.records, reference.records):
            _assert_nested_equal(actual_record, expected_record)
        _assert_nested_equal(actual_state, reference_state)


@pytest.mark.parametrize(("step_ratio", "expected_steps"), [(0.5, 0), (1.5, 1)])
def test_network_checkpointed_partial_horizons_match_run(step_ratio, expected_steps):
    duration = step_ratio * DT
    run_network = _build_network(with_injection=False)
    run_trace = _NetworkTrace()
    run_network.run(duration, callbacks=[run_trace])

    checkpointed = _build_network(with_injection=False)
    checkpointed_trace = _NetworkTrace()
    checkpointed.longrun_checkpointed(
        duration,
        chunklength=CHUNK_STEPS,
        callbacks=[checkpointed_trace],
    )

    assert len(run_trace.records) == len(checkpointed_trace.records) == expected_steps
    for actual_record, expected_record in zip(
        checkpointed_trace.records, run_trace.records
    ):
        _assert_nested_equal(actual_record, expected_record)
    _assert_nested_equal(
        checkpointed.state_dict_for_checkpoint(),
        run_network.state_dict_for_checkpoint(),
    )


def test_network_checkpointed_substep_runs_empty_loop_loss_lifecycle():
    network = _build_network(with_injection=False)
    network.train()
    before = _snapshot(network.state_dict_for_checkpoint())
    expected = _snapshot(before)
    expected["duration_remainder"] = torch.as_tensor(
        DT / 2.0,
        device=before["duration_remainder"].device,
        dtype=before["duration_remainder"].dtype,
    )
    callback = _EmptyCheckpointLifecycle()

    loss, final_state = network.longrun_checkpointed(
        DT / 2.0,
        chunklength=CHUNK_STEPS,
        callbacks=[callback],
        return_final_state=True,
    )

    assert callback.pre_loop_calls == callback.post_loop_calls == 1
    assert callback.pre_chunk_calls == callback.post_chunk_calls == 0
    assert callback.post_step_calls == 0
    torch.testing.assert_close(loss, callback.weight.square())
    _assert_nested_equal(final_state, expected)
    _assert_nested_equal(network.state_dict_for_checkpoint(), expected)

    loss.backward()
    torch.testing.assert_close(
        callback.weight.grad,
        torch.tensor(6.0, dtype=DTYPE),
    )


def test_network_checkpointed_rejects_stale_wiring_without_mutation():
    network = _build_network(with_injection=False)
    network.connect_one_to_one(
        network.pre[:],
        network.post[:],
        network.post.mech.syn,
        threshold=0.0,
        weight=0.01,
        delay=DT,
        allow_multapses=True,
    )
    assert not network.built
    before = _snapshot(network.state_dict_for_checkpoint())

    with pytest.raises(RuntimeError, match="wiring has changed"):
        network.longrun_checkpointed(DT, chunklength=CHUNK_STEPS)

    _assert_nested_equal(network.state_dict_for_checkpoint(), before)


def test_network_longrun_matches_run_for_chunk_crossing_extracellular_field():
    run_network, run_extra = _build_extracellular_cable_network()
    run_trace = _CableTrace()
    initial_voltage = run_network.cable.v.detach().clone()
    run_network.run(EXTRA_STEPS * DT, extra=run_extra, callbacks=[run_trace])

    long_network, long_extra = _build_extracellular_cable_network()
    long_trace = _CableTrace()
    long_network.longrun(
        EXTRA_STEPS * DT,
        chunklength=CHUNK_STEPS,
        extra=long_extra,
        callbacks=[long_trace],
    )

    run_voltage = torch.stack(run_trace.voltage)
    long_voltage = torch.stack(long_trace.voltage)
    assert (run_voltage - initial_voltage).abs().max() > 0.1
    assert EXTRA_START_STEP < CHUNK_STEPS < 2 * CHUNK_STEPS
    assert 2 * CHUNK_STEPS < EXTRA_START_STEP + EXTRA_PULSE_STEPS
    torch.testing.assert_close(
        torch.stack(long_trace.time), torch.stack(run_trace.time), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(long_voltage, run_voltage, rtol=0.0, atol=0.0)
    _assert_nested_equal(
        long_network.state_dict_for_checkpoint(),
        run_network.state_dict_for_checkpoint(),
    )
