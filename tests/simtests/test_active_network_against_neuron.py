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
    if synapse_kind == "exp2syn":
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

    recorded = {
        "time": h.Vector(),
        "pre_v": h.Vector(),
        "post_v": h.Vector(),
        "g": h.Vector(),
        "i": h.Vector(),
    }
    recorded["time"].record(h._ref_t)
    recorded["pre_v"].record(pre(0.5)._ref_v)
    recorded["post_v"].record(post(0.5)._ref_v)
    recorded["g"].record(synapse._ref_g)
    recorded["i"].record(synapse._ref_i)

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

    result = {name: np.asarray(values).copy() for name, values in recorded.items()}
    assert result["time"].shape == (N_STEPS + 1,)
    np.testing.assert_allclose(result["time"], np.arange(N_STEPS + 1) * DT)
    return result


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
        post.insert(
            exp2syn.rename(f"oracle_{synapse_kind}"),
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

    recorder = dn.callbacks.RecorderLambda(
        {
            "pre_v": lambda net: net.pre.v.detach().clone(),
            "post_v": lambda net: net.post.v.detach().clone(),
            "g": conductance,
            "i": lambda net: synapse.i(net.post.v).detach().clone(),
        }
    )
    network.initialize(DT)
    if synapse_kind.startswith("exp2syn"):
        tau1, tau2 = _exp2_taus(synapse_kind)
        effective_tau1 = float(synapse.DE["A"].tau1.item())
        assert effective_tau1 == pytest.approx(
            _effective_exp2_tau1(tau1, tau2), rel=2.0e-7
        )
    network.run(TSTOP, callbacks=[recorder])
    result = {
        name: recorder.numpy(name).reshape(N_STEPS + 1)
        for name in ("pre_v", "post_v", "g", "i")
    }
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


@pytest.mark.parametrize("synapse_kind", ["expsyn", "exp2syn", "exp2syn_equal"])
def test_two_cell_hh_event_network_matches_neuron(synapse_kind):
    expected = _run_neuron(synapse_kind)
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
    assert pre_max_abs < 0.4, pre_diagnostics
    assert pre_rmse < 0.06, pre_diagnostics
    expected_peak = int(np.argmax(expected["pre_v"]))
    actual_peak = int(np.argmax(actual["pre_v"]))
    assert abs(actual_peak - expected_peak) <= 1, pre_diagnostics
    assert abs(float(actual["pre_v"].max() - expected["pre_v"].max())) < 0.01

    expected_delivery = int(np.flatnonzero(expected["g"] > 0.0)[0])
    actual_delivery = int(np.flatnonzero(actual["g"] > 0.0)[0])
    expected_delay = (expected_delivery - expected_spike) * DT
    # The first observable Exp2Syn conductance follows delivery by one sample
    # because equal A/B jumps initially cancel. NEURON may also locate the
    # source crossing within the preceding fixed step, so this is deliberately
    # a bounded oracle sanity check rather than a cross-simulator phase claim.
    assert SYNAPTIC_DELAY <= expected_delay <= SYNAPTIC_DELAY + 2 * DT + 1.0e-12

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
        rtol=2.0e-4,
        atol=3.0e-3,
    )

    # Align on the independently observed arrivals. This compares the entire
    # synaptic conductance/current and postsynaptic voltage response without
    # conflating simulator-specific within-step event phases with kinetics.
    n_after = min(
        len(expected["g"]) - expected_delivery,
        len(actual["g"]) - actual_delivery,
    )
    for name, atol in (("post_v", 0.04), ("g", 1.0e-8), ("i", 0.01)):
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
