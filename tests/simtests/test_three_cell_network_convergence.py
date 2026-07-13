"""NEURON convergence oracle for a three-cell, multi-event HH network.

The network contains both a feed-forward chain (A -> B -> C) and a branch
(A -> C).  Two A -> C NetCons share one ExpSyn, have equal delays, and
therefore exercise deterministic summation of coincident events.  Cell A is
driven twice, so every edge also sees separated events.

NEURON and Dendra do not promise the same fixed-step event phase.  NEURON
reports the source event at its sampled crossing boundary, whereas Dendra
observes that completed population state on the following network iteration
and then uses its integer delay line.  This oracle therefore checks each
simulator's delivery phase explicitly, measures delivery-time convergence on
the absolute clock, and compares synaptic kinetics after aligning independently
observed arrivals.  It does not hide an event-phase difference inside a voltage
tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, hh
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

RESTING_VOLTAGE = -65.0
CELSIUS = 6.3
LENGTH = 100.0
DIAMETER = 100.0
RHOA = 35.4
CM = 1.0
SPIKE_THRESHOLD = 0.0
TSTOP = 26.0

STIMULI = ((1.0, 1.0, 5.0), (15.0, 1.0, 5.0))
TAU = 2.0
REVERSAL = 0.0

AB_DELAY = 0.5
AB_WEIGHT = 0.08
AC_DELAY = 1.5
AC_WEIGHTS = (0.02, 0.02)
BC_DELAY = 0.5
BC_WEIGHT = 0.05

DENDRA_DTS = (0.025, 0.0125, 0.00625)
NEURON_DT = 0.00078125
DTYPE = torch.float64


@dataclass(frozen=True)
class _Trace:
    dt: float
    time: np.ndarray
    a_v: np.ndarray
    b_v: np.ndarray
    c_v: np.ndarray
    ab_g: np.ndarray
    ac_g: np.ndarray
    bc_g: np.ndarray
    ab_i: np.ndarray
    ac_i: np.ndarray
    bc_i: np.ndarray
    a_events: np.ndarray
    b_events: np.ndarray


def _new_neuron_cell(name):
    section = h.Section(name=name)
    section.L = LENGTH
    section.diam = DIAMETER
    section.Ra = RHOA
    section.cm = CM
    section.nseg = 1
    section.insert("hh")
    return section


def _run_neuron():
    a = _new_neuron_cell("convergence_a")
    b = _new_neuron_cell("convergence_b")
    c = _new_neuron_cell("convergence_c")

    clamps = []
    for delay, duration, amplitude in STIMULI:
        clamp = h.IClamp(a(0.5))
        clamp.delay = delay
        clamp.dur = duration
        clamp.amp = amplitude
        clamps.append(clamp)

    ab_syn = h.ExpSyn(b(0.5))
    ac_syn = h.ExpSyn(c(0.5))
    bc_syn = h.ExpSyn(c(0.5))
    for synapse in (ab_syn, ac_syn, bc_syn):
        synapse.tau = TAU
        synapse.e = REVERSAL

    def connect(source, target, delay, weight):
        netcon = h.NetCon(source(0.5)._ref_v, target, sec=source)
        netcon.threshold = SPIKE_THRESHOLD
        netcon.delay = delay
        netcon.weight[0] = weight
        return netcon

    ab = connect(a, ab_syn, AB_DELAY, AB_WEIGHT)
    ac_1 = connect(a, ac_syn, AC_DELAY, AC_WEIGHTS[0])
    ac_2 = connect(a, ac_syn, AC_DELAY, AC_WEIGHTS[1])
    bc = connect(b, bc_syn, BC_DELAY, BC_WEIGHT)
    connections = (ab, ac_1, ac_2, bc)

    recorded = {
        "time": h.Vector(),
        "a_v": h.Vector(),
        "b_v": h.Vector(),
        "c_v": h.Vector(),
        "ab_g": h.Vector(),
        "ac_g": h.Vector(),
        "bc_g": h.Vector(),
    }
    recorded["time"].record(h._ref_t)
    recorded["a_v"].record(a(0.5)._ref_v)
    recorded["b_v"].record(b(0.5)._ref_v)
    recorded["c_v"].record(c(0.5)._ref_v)
    recorded["ab_g"].record(ab_syn._ref_g)
    recorded["ac_g"].record(ac_syn._ref_g)
    recorded["bc_g"].record(bc_syn._ref_g)
    a_events = h.Vector()
    b_events = h.Vector()
    ab.record(a_events)
    bc.record(b_events)

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = NEURON_DT
    assert h.usetable_hh == 0
    h.finitialize(RESTING_VOLTAGE)
    while h.t < TSTOP - NEURON_DT / 2:
        h.fadvance()

    arrays = {name: np.asarray(values).copy() for name, values in recorded.items()}
    arrays["ab_i"] = arrays["ab_g"] * (arrays["b_v"] - REVERSAL)
    arrays["ac_i"] = arrays["ac_g"] * (arrays["c_v"] - REVERSAL)
    arrays["bc_i"] = arrays["bc_g"] * (arrays["c_v"] - REVERSAL)
    expected_steps = int(round(TSTOP / NEURON_DT))
    assert arrays["time"].shape == (expected_steps + 1,)
    np.testing.assert_allclose(
        arrays["time"], np.arange(expected_steps + 1) * NEURON_DT, atol=1.0e-12
    )
    # Retain these locals until after integration; NEURON objects are otherwise
    # eligible for Python-side cleanup while the run is in progress.
    assert len(clamps) == 2 and len(connections) == 4
    return _Trace(
        dt=NEURON_DT,
        a_events=np.asarray(a_events).copy(),
        b_events=np.asarray(b_events).copy(),
        **arrays,
    )


def _new_dendra_cell():
    cell = dn.SingleCompartment(
        N=1,
        C=1,
        celsius=CELSIUS,
        v_init=RESTING_VOLTAGE,
        cm=CM,
        rhoa=RHOA,
        dtype=DTYPE,
    )
    cell.diam.fill_(DIAMETER)
    cell.dx.fill_(LENGTH)
    cell.insert(hh)
    return cell


def _run_dendra(dt):
    a = _new_dendra_cell()
    b = _new_dendra_cell()
    c = _new_dendra_cell()
    b.insert(expsyn.rename("ab_syn"), e=REVERSAL, tau=TAU)
    c.insert(expsyn.rename("ac_syn"), e=REVERSAL, tau=TAU)
    c.insert(expsyn.rename("bc_syn"), e=REVERSAL, tau=TAU)
    for delay, duration, amplitude in STIMULI:
        a[:, 0].inject(dn.mono_rect(amp=amplitude * nA, delay=delay, pw=duration))

    network = dn.Network({"a": a, "b": b, "c": c}, track_netcon_events=True)
    network.connect_one_to_one(
        network.a[:],
        network.b[:],
        network.b.mech.ab_syn,
        threshold=SPIKE_THRESHOLD,
        weight=AB_WEIGHT,
        delay=AB_DELAY,
    )
    # These are deliberately two distinct multapses onto one state variable.
    # Their same-source, same-delay events must collide and sum, not overwrite.
    for weight in AC_WEIGHTS:
        network.connect_one_to_one(
            network.a[:],
            network.c[:],
            network.c.mech.ac_syn,
            threshold=SPIKE_THRESHOLD,
            weight=weight,
            delay=AC_DELAY,
            allow_multapses=True,
        )
    network.connect_one_to_one(
        network.b[:],
        network.c[:],
        network.c.mech.bc_syn,
        threshold=SPIKE_THRESHOLD,
        weight=BC_WEIGHT,
        delay=BC_DELAY,
    )

    ab_syn = network.b.mech.ab_syn
    ac_syn = network.c.mech.ac_syn
    bc_syn = network.c.mech.bc_syn
    recorder = dn.callbacks.RecorderLambda(
        {
            "a_v": lambda net: net.a.v.detach().clone(),
            "b_v": lambda net: net.b.v.detach().clone(),
            "c_v": lambda net: net.c.v.detach().clone(),
            "ab_g": lambda _net: ab_syn.g.detach().clone(),
            "ac_g": lambda _net: ac_syn.g.detach().clone(),
            "bc_g": lambda _net: bc_syn.g.detach().clone(),
            "ab_i": lambda net: ab_syn.i(net.b.v).detach().clone(),
            "ac_i": lambda net: ac_syn.i(net.c.v).detach().clone(),
            "bc_i": lambda net: bc_syn.i(net.c.v).detach().clone(),
        }
    )
    network.initialize(dt)
    network.run(TSTOP, callbacks=[recorder])

    n_steps = int(round(TSTOP / dt))
    arrays = {
        name: recorder.numpy(name).reshape(n_steps + 1)
        for name in (
            "a_v",
            "b_v",
            "c_v",
            "ab_g",
            "ac_g",
            "bc_g",
            "ab_i",
            "ac_i",
            "bc_i",
        )
    }
    a_crossings = _crossing_indices(arrays["a_v"])
    b_crossings = _crossing_indices(arrays["b_v"])
    return _Trace(
        dt=dt,
        time=np.arange(n_steps + 1) * dt,
        a_events=a_crossings.astype(float) * dt,
        b_events=b_crossings.astype(float) * dt,
        **arrays,
    )


def _crossing_indices(values):
    return (
        np.flatnonzero(
            (values[:-1] < SPIKE_THRESHOLD) & (values[1:] >= SPIKE_THRESHOLD)
        )
        + 1
    )


def _crossing_times(trace):
    result = {}
    for cell in ("a", "b", "c"):
        voltage = getattr(trace, f"{cell}_v")
        indices = _crossing_indices(voltage)
        before = voltage[indices - 1]
        after = voltage[indices]
        fraction = (SPIKE_THRESHOLD - before) / (after - before)
        result[cell] = trace.time[indices - 1] + fraction * trace.dt
    return result


def _delivery_payload(g, dt):
    """Recover ExpSyn event payloads at the population post-step phase."""
    decay = np.exp(-dt / TAU)
    payload = np.zeros_like(g)
    payload[1:] = g[1:] / decay - g[:-1]
    return payload


def _positive_payload_indices(g, dt, tolerance=1.0e-10):
    return np.flatnonzero(_delivery_payload(g, dt) > tolerance)


def _dendra_expected_delivery_indices(spike_indices, delay, dt):
    delay_steps = int(round(delay / dt))
    return spike_indices + max(delay_steps, 1) + 1


def _sample_reference(reference, dt):
    stride = int(round(dt / reference.dt))
    assert stride * reference.dt == pytest.approx(dt)
    return np.arange(0, reference.time.size, stride)


def _rmse(actual, expected):
    difference = actual - expected
    return float(np.sqrt(np.mean(difference * difference)))


def _assert_roughly_first_order(label, errors):
    assert errors[2] < errors[1] < errors[0], f"{label}: {errors}"
    for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
        assert 1.7 < ratio < 2.8, f"{label}: errors={errors}, ratio={ratio:.6g}"


def test_three_cell_multi_event_network_converges_to_neuron():
    reference = _run_neuron()
    traces = [_run_dendra(dt) for dt in DENDRA_DTS]

    reference_crossings = _crossing_times(reference)
    assert reference.a_events.size == reference_crossings["a"].size == 2
    assert reference.b_events.size == reference_crossings["b"].size == 2
    assert np.all(np.diff(reference.a_events) > 4.0)
    assert np.all(np.diff(reference.b_events) > 4.0)

    # NEURON's two direct A -> C NetCons share one source/threshold and use the
    # same delay, so each pair arrives coincidently by construction.
    np.testing.assert_allclose(
        reference.a_events,
        reference_crossings["a"],
        atol=NEURON_DT,
    )
    np.testing.assert_allclose(
        reference.b_events,
        reference_crossings["b"],
        atol=NEURON_DT,
    )
    neuron_ab_arrivals = reference.a_events + AB_DELAY
    neuron_ac_arrivals = reference.a_events + AC_DELAY
    neuron_bc_arrivals = reference.b_events + BC_DELAY
    assert np.all(np.diff(neuron_ac_arrivals) > 4.0)

    arrival_errors = {edge: [] for edge in ("ab", "ac", "bc")}
    for trace in traces:
        for field in (
            "a_v",
            "b_v",
            "c_v",
            "ab_g",
            "ac_g",
            "bc_g",
            "ab_i",
            "ac_i",
            "bc_i",
        ):
            values = getattr(trace, field)
            assert values.shape == trace.time.shape
            assert np.isfinite(values).all()

        crossings = _crossing_times(trace)
        assert {cell: values.size for cell, values in crossings.items()} == {
            "a": 2,
            "b": 2,
            "c": 2,
        }
        a_spikes = _crossing_indices(trace.a_v)
        b_spikes = _crossing_indices(trace.b_v)

        ab_deliveries = _positive_payload_indices(trace.ab_g, trace.dt)
        ac_deliveries = _positive_payload_indices(trace.ac_g, trace.dt)
        bc_deliveries = _positive_payload_indices(trace.bc_g, trace.dt)
        np.testing.assert_array_equal(
            ab_deliveries,
            _dendra_expected_delivery_indices(a_spikes, AB_DELAY, trace.dt),
        )
        np.testing.assert_array_equal(
            ac_deliveries,
            _dendra_expected_delivery_indices(a_spikes, AC_DELAY, trace.dt),
        )
        np.testing.assert_array_equal(
            bc_deliveries,
            _dendra_expected_delivery_indices(b_spikes, BC_DELAY, trace.dt),
        )
        np.testing.assert_allclose(
            _delivery_payload(trace.ab_g, trace.dt)[ab_deliveries],
            AB_WEIGHT,
            rtol=2.0e-12,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            _delivery_payload(trace.ac_g, trace.dt)[ac_deliveries],
            sum(AC_WEIGHTS),
            rtol=2.0e-12,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            _delivery_payload(trace.bc_g, trace.dt)[bc_deliveries],
            BC_WEIGHT,
            rtol=2.0e-12,
            atol=2.0e-14,
        )

        # The public Dendra phase is explicit: detect the source's completed
        # threshold-crossing sample on the following network iteration, then
        # apply the quantized physical delay.  Do not equate this to NEURON's
        # fixed-step source/event phase.
        dendra_ab_arrivals = trace.time[ab_deliveries]
        dendra_ac_arrivals = trace.time[ac_deliveries]
        dendra_bc_arrivals = trace.time[bc_deliveries]
        a_crossing_error = np.max(np.abs(crossings["a"] - reference_crossings["a"]))
        b_crossing_error = np.max(np.abs(crossings["b"] - reference_crossings["b"]))
        ab_arrival_error = float(
            np.max(np.abs(dendra_ab_arrivals - neuron_ab_arrivals))
        )
        ac_arrival_error = float(
            np.max(np.abs(dendra_ac_arrivals - neuron_ac_arrivals))
        )
        bc_arrival_error = float(
            np.max(np.abs(dendra_bc_arrivals - neuron_bc_arrivals))
        )
        arrival_errors["ab"].append(ab_arrival_error)
        arrival_errors["ac"].append(ac_arrival_error)
        arrival_errors["bc"].append(bc_arrival_error)
        # From the linearly interpolated crossing to Dendra's completed source
        # sample costs at most one dt; next-iteration detection costs one more.
        # Separating that phase budget from the source solver's own spike-time
        # error makes the contract diagnostic rather than merely permissive.
        assert ab_arrival_error <= a_crossing_error + 2 * trace.dt + NEURON_DT
        assert ac_arrival_error <= a_crossing_error + 2 * trace.dt + NEURON_DT
        assert bc_arrival_error <= b_crossing_error + 2 * trace.dt + NEURON_DT

    for edge, errors in arrival_errors.items():
        _assert_roughly_first_order(f"{edge} delivery time", errors)

    # Complete-model voltage and spike timing must approach one much finer
    # NEURON run as Dendra's fixed step is halved.  The source cell has no
    # network input and should agree especially closely.
    voltage_errors = {name: [] for name in ("a_v", "b_v", "c_v")}
    spike_errors = {cell: [] for cell in ("a", "b", "c")}
    for trace in traces:
        sample = _sample_reference(reference, trace.dt)
        for name in voltage_errors:
            voltage_errors[name].append(
                _rmse(getattr(trace, name), getattr(reference, name)[sample])
            )
        crossings = _crossing_times(trace)
        for cell in spike_errors:
            spike_errors[cell].append(
                float(np.max(np.abs(crossings[cell] - reference_crossings[cell])))
            )

    for name, errors in voltage_errors.items():
        _assert_roughly_first_order(name, errors)
    for cell, errors in spike_errors.items():
        _assert_roughly_first_order(f"{cell} spike time", errors)
        assert errors[-1] < 6 * DENDRA_DTS[-1]

    # Absolute-trace RMSE includes the steep action-potential flank, so a
    # small spike-time displacement contributes much more than the local
    # voltage error.  Keep explicit finest-grid guards alongside the much
    # tighter spike-time bounds above, and require roughly first-order decay
    # across the nested grid rather than disguising phase as amplitude error.
    assert voltage_errors["a_v"][-1] < 0.4
    assert voltage_errors["b_v"][-1] < 0.8
    assert voltage_errors["c_v"][-1] < 1.3

    # Synaptic state and current are discontinuous at delivery. Compare each
    # response on its simulator's independently observed arrival rather than
    # counting a one-sample phase difference as a kinetics error.  Windows end
    # before the next event, so every comparison begins with a known payload.
    synapse_specs = (
        ("ab", neuron_ab_arrivals[0]),
        ("ac", neuron_ac_arrivals[0]),
        ("bc", neuron_bc_arrivals[0]),
    )
    aligned_errors = {
        f"{prefix}_{quantity}": []
        for prefix, _ in synapse_specs
        for quantity in ("g", "i")
    }
    for trace in traces:
        for prefix, reference_arrival in synapse_specs:
            actual_g = getattr(trace, f"{prefix}_g")
            actual_arrival = _positive_payload_indices(actual_g, trace.dt)[0]
            width = min(
                int(round(1.0 / trace.dt)),
                actual_g.size - actual_arrival,
            )
            # Dendra receives before advancing the population, so the state at
            # its first visible post-step sample has synaptic age ``dt``.
            # Interpolate the fine NEURON oracle at the same ages relative to
            # NEURON's independently recorded delivery.  This removes only the
            # documented event phase; it does not shift either voltage trace to
            # optimize agreement.
            ages = np.arange(1, width + 1) * trace.dt
            for quantity in ("g", "i"):
                actual = getattr(trace, f"{prefix}_{quantity}")[
                    actual_arrival : actual_arrival + width
                ]
                expected = np.interp(
                    reference_arrival + ages,
                    reference.time,
                    getattr(reference, f"{prefix}_{quantity}"),
                )
                aligned_errors[f"{prefix}_{quantity}"].append(_rmse(actual, expected))

    # Both simulators use the analytic ExpSyn exponential, so once the event
    # age is aligned, conductance agrees to interpolation precision rather than
    # exhibiting a meaningful dt trend.  Current additionally contains the
    # evolving postsynaptic voltage and must converge as that solve is refined.
    for prefix, _ in synapse_specs:
        conductance_errors = aligned_errors[f"{prefix}_g"]
        assert max(conductance_errors) < 1.0e-9, f"{prefix}_g: {conductance_errors}"
        current_errors = aligned_errors[f"{prefix}_i"]
        assert current_errors[2] < current_errors[1] < current_errors[0], (
            f"{prefix}_i: {current_errors}"
        )
        assert current_errors[-1] < 3.0e-3
