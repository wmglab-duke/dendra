"""NEURON oracle for a two-cell active-HH event network."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.mod import exp2syn, expsyn, hh
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DT = 0.025
TSTOP = 12.0
N_STEPS = int(TSTOP / DT)
RESTING_VOLTAGE = -65.0
CELSIUS = 6.3
LENGTH = 100.0
DIAMETER = 100.0
RHOA = 35.4
CM = 1.0
STIM_DELAY = 1.0
STIM_DURATION = 1.0
STIM_AMPLITUDE = 5.0
SPIKE_THRESHOLD = 0.0
SYNAPTIC_DELAY = 1.0
SYNAPTIC_WEIGHT = 0.01
SYNAPTIC_TAU = 2.0
SYNAPTIC_TAU1 = 0.2
SYNAPTIC_TAU2 = 2.0
SYNAPTIC_REVERSAL = 0.0
DTYPE = torch.float64


def _new_neuron_cell(name):
    section = h.Section(name=name)
    section.L = LENGTH
    section.diam = DIAMETER
    section.Ra = RHOA
    section.cm = CM
    section.nseg = 1
    section.insert("hh")
    assert h.area(0.5, sec=section) == pytest.approx(np.pi * DIAMETER * LENGTH)
    segment = section(0.5)
    assert segment.hh.gnabar == pytest.approx(0.12)
    assert segment.hh.gkbar == pytest.approx(0.036)
    assert segment.hh.gl == pytest.approx(0.0003)
    assert segment.hh.el == pytest.approx(-54.3)
    assert segment.ena == pytest.approx(50.0)
    assert segment.ek == pytest.approx(-77.0)
    return section


def _exp2_taus(synapse_kind):
    if synapse_kind in ("exp2syn", "exp2syn_hookless"):
        return SYNAPTIC_TAU1, SYNAPTIC_TAU2
    if synapse_kind == "exp2syn_equal":
        return SYNAPTIC_TAU2, SYNAPTIC_TAU2
    raise ValueError(synapse_kind)


def _effective_exp2_tau1(tau1, tau2):
    return tau2 * min(max(tau1 / tau2, 1.0e-9), 0.9999)


def _run_neuron(synapse_kind):
    pre = _new_neuron_cell("network_pre")
    post = _new_neuron_cell("network_post")

    stimulus = h.IClamp(pre(0.5))
    stimulus.delay = STIM_DELAY
    stimulus.dur = STIM_DURATION
    stimulus.amp = STIM_AMPLITUDE

    if synapse_kind == "expsyn":
        synapse = h.ExpSyn(post(0.5))
        synapse.tau = SYNAPTIC_TAU
    elif synapse_kind.startswith("exp2syn"):
        tau1, tau2 = _exp2_taus(synapse_kind)
        synapse = h.Exp2Syn(post(0.5))
        synapse.tau1 = tau1
        synapse.tau2 = tau2
    else:  # pragma: no cover - guarded by parametrization
        raise ValueError(synapse_kind)
    synapse.e = SYNAPTIC_REVERSAL
    connection = h.NetCon(pre(0.5)._ref_v, synapse, sec=pre)
    connection.threshold = SPIKE_THRESHOLD
    connection.delay = SYNAPTIC_DELAY
    connection.weight[0] = SYNAPTIC_WEIGHT
    source_event_times = h.Vector()
    connection.record(source_event_times)

    recorded = {
        "time": h.Vector(),
        "pre_v": h.Vector(),
        "post_v": h.Vector(),
    }
    recorded["time"].record(h._ref_t)
    recorded["pre_v"].record(pre(0.5)._ref_v)
    recorded["post_v"].record(post(0.5)._ref_v)
    if synapse_kind == "expsyn":
        recorded["g"] = h.Vector()
        recorded["g"].record(synapse._ref_g)
    else:
        # Exp2Syn's RANGE g is evaluated in BREAKPOINT before its A/B states
        # are advanced.  After fadvance(), _ref_g therefore describes an
        # earlier solver phase than the simultaneously recorded voltage and
        # states.  Record the states and derive the physical conductance below.
        recorded["A"] = h.Vector()
        recorded["B"] = h.Vector()
        recorded["A"].record(synapse._ref_A)
        recorded["B"].record(synapse._ref_B)

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = DT
    h.finitialize(RESTING_VOLTAGE)
    if synapse_kind.startswith("exp2syn"):
        tau1, tau2 = _exp2_taus(synapse_kind)
        assert synapse.tau1 == pytest.approx(_effective_exp2_tau1(tau1, tau2))
    while h.t < TSTOP - DT / 2:
        h.fadvance()

    sampled = {name: np.asarray(values).copy() for name, values in recorded.items()}
    if synapse_kind.startswith("exp2syn"):
        physical_g = sampled["B"] - sampled["A"]
    else:
        physical_g = sampled["g"]
    result = {
        "time": sampled["time"],
        "pre_v": sampled["pre_v"],
        "post_v": sampled["post_v"],
        "g": physical_g,
    }
    # Point-process current RANGE variables retain NEURON's earlier
    # BREAKPOINT evaluation after fadvance().  For Exp2Syn the same applies to
    # its published g RANGE variable, so g is derived from simultaneous A/B
    # states above. Reconstruct current from simultaneous conductance and
    # voltage so both simulators are compared at one explicit post-step phase.
    result["i"] = result["g"] * (result["post_v"] - SYNAPTIC_REVERSAL)
    assert result["time"].shape == (N_STEPS + 1,)
    np.testing.assert_allclose(result["time"], np.arange(N_STEPS + 1) * DT)
    return result, np.asarray(source_event_times).copy()


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
    torch.testing.assert_close(cell.diam, torch.full_like(cell.diam, DIAMETER))
    torch.testing.assert_close(cell.dx, torch.full_like(cell.dx, LENGTH))
    torch.testing.assert_close(
        cell.area,
        torch.full_like(cell.area, np.pi * DIAMETER * LENGTH * 1.0e-8),
    )
    cell.insert(hh)
    return cell


def _run_dendra(synapse_kind):
    pre = _new_dendra_cell()
    post = _new_dendra_cell()
    if synapse_kind == "expsyn":
        post.insert(
            expsyn.rename("oracle_expsyn"),
            e=SYNAPTIC_REVERSAL,
            tau=SYNAPTIC_TAU,
        )
    elif synapse_kind.startswith("exp2syn"):
        tau1, tau2 = _exp2_taus(synapse_kind)
        mechanism = exp2syn.rename(f"oracle_{synapse_kind}")
        if synapse_kind == "exp2syn_hookless":
            delattr(mechanism, "i_with_conductance")
        post.insert(
            mechanism,
            e=SYNAPTIC_REVERSAL,
            tau1=tau1,
            tau2=tau2,
        )
    else:  # pragma: no cover - guarded by parametrization
        raise ValueError(synapse_kind)
    pre[:, 0].inject(
        dn.mono_rect(
            amp=STIM_AMPLITUDE * nA,
            delay=STIM_DELAY,
            pw=STIM_DURATION,
        )
    )

    network = dn.Network({"pre": pre, "post": post}, track_netcon_events=True)
    network.connect_one_to_one(
        network.pre[:],
        network.post[:],
        getattr(network.post.mech, f"oracle_{synapse_kind}"),
        threshold=SPIKE_THRESHOLD,
        weight=SYNAPTIC_WEIGHT,
        delay=SYNAPTIC_DELAY,
    )

    synapse = getattr(network.post.mech, f"oracle_{synapse_kind}")

    def conductance(_network):
        if synapse_kind == "expsyn":
            return synapse.g.detach().clone()
        return (synapse.B - synapse.A).detach().clone()

    recorders = {
        "pre_v": lambda net: net.pre.v.detach().clone(),
        "post_v": lambda net: net.post.v.detach().clone(),
        "g": conductance,
        "i": lambda net: synapse.i(net.post.v).detach().clone(),
    }
    if synapse_kind.startswith("exp2syn"):
        recorders.update(
            {
                "A": lambda _net: synapse.A.detach().clone(),
                "B": lambda _net: synapse.B.detach().clone(),
            }
        )
    if synapse_kind == "exp2syn_hookless":
        recorders.update(
            {
                "symbolic_i": lambda net: synapse.i_with_g(net.post.v)[0]
                .detach()
                .clone(),
                "symbolic_g": lambda net: synapse.i_with_g(net.post.v)[1]
                .detach()
                .clone(),
            }
        )
    recorder = dn.callbacks.RecorderLambda(recorders)
    network.initialize(DT)
    if synapse_kind.startswith("exp2syn"):
        tau1, tau2 = _exp2_taus(synapse_kind)
        effective_tau1 = float(synapse.DE["A"].tau1.item())
        assert effective_tau1 == pytest.approx(
            _effective_exp2_tau1(tau1, tau2), rel=2.0e-7
        )
    if synapse_kind == "exp2syn_hookless":
        assert type(synapse)._dendra_symbolic_source_class is exp2syn
        assert synapse._current_conductance_mode == {"i": "symbolic"}
        assert synapse._current_conductance_fallback_reason == {"i": None}
    network.run(TSTOP, callbacks=[recorder])
    result = {
        name: recorder.numpy(name).reshape(N_STEPS + 1)
        for name in ("pre_v", "post_v", "g", "i")
    }
    if synapse_kind == "exp2syn_hookless":
        state_a = recorder.numpy("A").reshape(N_STEPS + 1)
        state_b = recorder.numpy("B").reshape(N_STEPS + 1)
        assert np.isfinite(state_a).all()
        assert np.isfinite(state_b).all()
        assert np.any(state_a > 0.0)
        assert np.any(state_b > 0.0)
        np.testing.assert_allclose(result["g"], state_b - state_a, atol=1.0e-15)
        np.testing.assert_allclose(
            recorder.numpy("symbolic_g").reshape(N_STEPS + 1),
            result["g"],
            rtol=2.0e-15,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            recorder.numpy("symbolic_i").reshape(N_STEPS + 1),
            result["i"],
            rtol=2.0e-15,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            result["i"],
            result["g"] * (result["post_v"] - SYNAPTIC_REVERSAL),
            rtol=2.0e-15,
            atol=1.0e-15,
        )
    result["time"] = np.arange(N_STEPS + 1) * DT
    return result


def _first_upward_crossing(values, threshold):
    crossings = np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold))
    assert crossings.size == 1
    return int(crossings[0] + 1)


def _expected_first_conductance(synapse_kind):
    if synapse_kind == "expsyn":
        return SYNAPTIC_WEIGHT * np.exp(-DT / SYNAPTIC_TAU)
    tau1, tau2 = _exp2_taus(synapse_kind)
    tau1 = _effective_exp2_tau1(tau1, tau2)
    peak_time = tau1 * tau2 / (tau2 - tau1) * np.log(tau2 / tau1)
    factor = 1.0 / (-np.exp(-peak_time / tau1) + np.exp(-peak_time / tau2))
    return SYNAPTIC_WEIGHT * factor * (np.exp(-DT / tau2) - np.exp(-DT / tau1))


@pytest.mark.parametrize(
    "synapse_kind",
    ["expsyn", "exp2syn", "exp2syn_equal", "exp2syn_hookless"],
)
def test_two_cell_hh_event_network_matches_neuron(synapse_kind):
    expected, expected_source_events = _run_neuron(synapse_kind)
    actual = _run_dendra(synapse_kind)

    assert set(actual) == set(expected)
    for trace in (*actual.values(), *expected.values()):
        assert trace.shape == (N_STEPS + 1,)
        assert np.isfinite(trace).all()

    expected_spike = _first_upward_crossing(expected["pre_v"], SPIKE_THRESHOLD)
    actual_spike = _first_upward_crossing(actual["pre_v"], SPIKE_THRESHOLD)
    assert abs(actual_spike - expected_spike) <= 1
    assert expected["pre_v"].max() > 20.0
    assert actual["pre_v"].max() > 20.0

    pre_difference = actual["pre_v"] - expected["pre_v"]
    pre_max_abs = float(np.max(np.abs(pre_difference)))
    pre_rmse = float(np.sqrt(np.mean(pre_difference * pre_difference)))
    pre_diagnostics = (
        f"Presynaptic HH mismatch: max_abs={pre_max_abs:.6g} mV, rmse={pre_rmse:.6g} mV"
    )
    # With NEURON's HH rate table disabled, the presynaptic voltage solve uses
    # the same analytic rates as Dendra.  Retain platform headroom around the
    # observed tens-of-nanovolts-to-microvolts agreement rather than masking a
    # future active-membrane regression behind spike-scale bounds.
    assert pre_max_abs < 1.0e-3, pre_diagnostics
    assert pre_rmse < 2.0e-4, pre_diagnostics
    expected_peak = int(np.argmax(expected["pre_v"]))
    actual_peak = int(np.argmax(actual["pre_v"]))
    assert abs(actual_peak - expected_peak) <= 1, pre_diagnostics
    assert abs(float(actual["pre_v"].max() - expected["pre_v"].max())) < 1.0e-4

    expected_delivery = int(np.flatnonzero(expected["g"] > 0.0)[0])
    actual_delivery = int(np.flatnonzero(actual["g"] > 0.0)[0])
    assert expected_source_events.shape == (1,)
    expected_source_event = float(expected_source_events[0])
    # NetCon.record reports the source event time directly. Relate it to the
    # sampled voltage-crossing bracket (allowing NEURON's tiny event-queue
    # epsilon at the upper boundary), rather than inferring it from conductance.
    crossing_t0 = float(expected["time"][expected_spike - 1])
    crossing_t1 = float(expected["time"][expected_spike])
    event_time_tolerance = 1.0e-8
    assert crossing_t0 < expected_source_event
    assert expected_source_event <= crossing_t1 + event_time_tolerance

    expected_arrival = expected_source_event + SYNAPTIC_DELAY
    expected_delivery_time = float(expected["time"][expected_delivery])
    # The phase-consistent A/B-derived Exp2Syn conductance, like ExpSyn's state,
    # is first sampled on the fixed-step boundary following the queued arrival.
    assert expected_arrival - event_time_tolerance <= expected_delivery_time
    assert expected_delivery_time <= expected_arrival + DT + event_time_tolerance
    assert not np.any(expected["g"][:expected_delivery])
    assert expected["g"][expected_delivery] == pytest.approx(
        _expected_first_conductance(synapse_kind), rel=2.0e-6
    )

    # Dendra detects a threshold crossing from the source state on the next
    # network iteration, then delivers after its integer delay line. Keep this
    # public quantized timing contract explicit instead of requiring NEURON to
    # choose the same absolute event phase within a fixed-step interval.
    delay_steps = int(round(SYNAPTIC_DELAY / DT))
    assert actual_delivery == actual_spike + delay_steps + 1
    assert not np.any(actual["g"][:actual_delivery])
    assert actual["g"][actual_delivery] == pytest.approx(
        _expected_first_conductance(synapse_kind), rel=2.0e-6
    )

    expected_post_excursion = expected["post_v"] - expected["post_v"][0]
    actual_post_excursion = actual["post_v"] - actual["post_v"][0]
    assert expected_post_excursion.max() > 1.0
    assert actual_post_excursion.max() > 1.0
    assert expected["post_v"].max() < SPIKE_THRESHOLD
    assert actual["post_v"].max() < SPIKE_THRESHOLD
    assert expected["i"].min() < -0.5
    assert actual["i"].min() < -0.5

    common_before = min(expected_delivery, actual_delivery)
    np.testing.assert_allclose(
        actual["post_v"][:common_before],
        expected["post_v"][:common_before],
        rtol=1.0e-8,
        atol=1.0e-5,
    )

    # Align on the independently observed first physical responses. In
    # particular, the NEURON Exp2Syn boundary is determined from A/B-derived g,
    # not its one-BREAKPOINT-phase-stale published RANGE variable. This compares
    # response kinetics without conflating simulator-specific event phases.
    n_after = min(
        len(expected["g"]) - expected_delivery,
        len(actual["g"]) - actual_delivery,
    )
    for name, atol in (("post_v", 0.04), ("g", 1.0e-8), ("i", 5.0e-4)):
        expected_aligned = expected[name][
            expected_delivery : expected_delivery + n_after
        ]
        actual_aligned = actual[name][actual_delivery : actual_delivery + n_after]
        difference = actual_aligned - expected_aligned
        max_abs = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(difference * difference)))
        assert np.allclose(actual_aligned, expected_aligned, rtol=2.0e-4, atol=atol), (
            f"Two-cell {synapse_kind} {name} mismatch: "
            f"max_abs={max_abs:.6g}, rmse={rmse:.6g}"
        )
