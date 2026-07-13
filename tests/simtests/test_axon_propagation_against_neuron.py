"""NEURON oracles for active intracellular propagation in Axon models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.mod import hh
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DTYPE = torch.float64
RESTING_VOLTAGE = -65.0
CELSIUS = 6.3
CM = 1.0
BASE_RHOA = 35.4
STIM_DELAY = 1.0


@dataclass(frozen=True)
class _AxonCase:
    name: str
    n_comp: int
    membrane_diameter: float
    compartment_length: float
    effective_rhoa: float
    coordinate_spacing: float
    stimulus_node: int
    record_nodes: tuple[int, int, int]
    stimulus_amp_na: float
    stimulus_duration: float
    tstop: float


@dataclass(frozen=True)
class _Simulation:
    time: np.ndarray
    voltage: np.ndarray


CASES = {
    "unmyelinated": _AxonCase(
        name="unmyelinated",
        n_comp=41,
        membrane_diameter=2.0,
        compartment_length=20.0,
        effective_rhoa=BASE_RHOA,
        coordinate_spacing=20.0,
        stimulus_node=5,
        record_nodes=(5, 20, 35),
        stimulus_amp_na=0.8,
        stimulus_duration=0.5,
        tstop=9.0,
    ),
    "myelinated": _AxonCase(
        name="myelinated",
        n_comp=13,
        # The default Myelinated polynomials give both axon and node diameters
        # of 0.7 * fiber diameter for a 10 um fiber.
        membrane_diameter=7.0,
        compartment_length=2.0,
        # 35.4 * (node_d / axon_d)^2 * (internodal spacing / node length)
        effective_rhoa=BASE_RHOA * (7.0 / 7.0) ** 2 * (1000.0 / 2.0),
        coordinate_spacing=1000.0,
        stimulus_node=2,
        record_nodes=(2, 6, 10),
        stimulus_amp_na=0.8,
        stimulus_duration=0.3,
        tstop=9.0,
    ),
}


def _build_dendra(case: _AxonCase):
    if case.name == "unmyelinated":
        model = dn.Unmyelinated(
            diameters=[case.membrane_diameter],
            L=case.n_comp * case.compartment_length,
            dx=case.compartment_length,
            celsius=CELSIUS,
            v_init=RESTING_VOLTAGE,
            rhoa=BASE_RHOA,
            cm=CM,
            dtype=DTYPE,
        )
    else:
        model = dn.Myelinated(
            diameters=[10.0],
            n_node=case.n_comp,
            node_length=case.compartment_length,
            celsius=CELSIUS,
            v_init=RESTING_VOLTAGE,
            rhoa=BASE_RHOA,
            cm=CM,
            dtype=DTYPE,
        )

    model.insert(hh)
    model[..., case.stimulus_node].inject(
        dn.mono_rect(
            amp=case.stimulus_amp_na * nA,
            delay=STIM_DELAY,
            pw=case.stimulus_duration,
        )
    )
    model.eval()
    model.initialize()
    return model


def _assert_dendra_geometry(model, case: _AxonCase):
    assert model.shape == (1, case.n_comp)
    for name, expected in (
        ("diam", case.membrane_diameter),
        ("dx", case.compartment_length),
        ("rhoa", case.effective_rhoa),
        ("cm", CM),
    ):
        value = getattr(model, name)
        torch.testing.assert_close(value, torch.full_like(value, expected))

    expected_x = (
        torch.arange(case.n_comp, dtype=DTYPE) - (case.n_comp - 1) / 2.0
    ) * case.coordinate_spacing
    torch.testing.assert_close(model.x[0], expected_x)


def _build_neuron(case: _AxonCase):
    section = h.Section(name=f"intracellular_{case.name}")
    section.L = case.n_comp * case.compartment_length
    section.diam = case.membrane_diameter
    section.Ra = case.effective_rhoa
    section.cm = CM
    section.nseg = case.n_comp
    section.insert("hh")
    assert section.nseg == case.n_comp
    assert section.L == pytest.approx(case.n_comp * case.compartment_length)
    assert section.diam == pytest.approx(case.membrane_diameter)
    assert section.Ra == pytest.approx(case.effective_rhoa)
    assert section.cm == pytest.approx(CM)
    segments = list(section)
    assert len(segments) == case.n_comp

    stimulus = h.IClamp(segments[case.stimulus_node])
    stimulus.delay = STIM_DELAY
    stimulus.dur = case.stimulus_duration
    stimulus.amp = case.stimulus_amp_na
    return section, segments, stimulus


def _run_neuron(case: _AxonCase, dt: float):
    section, segments, stimulus = _build_neuron(case)
    # Retain point-process and section references until the fixed-step run ends.
    assert stimulus.get_segment().sec == section

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = dt
    h.finitialize(RESTING_VOLTAGE)

    recorded = [segments[node] for node in case.record_nodes]
    times = [float(h.t)]
    traces = [[float(segment.v) for segment in recorded]]
    while h.t < case.tstop - dt / 2.0:
        h.fadvance()
        times.append(float(h.t))
        traces.append([float(segment.v) for segment in recorded])
    return _Simulation(np.asarray(times), np.asarray(traces))


def _run_dendra(case: _AxonCase, dt: float):
    model = _build_dendra(case)
    _assert_dendra_geometry(model, case)
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=case.record_nodes)
    model.run(tstop=case.tstop, dt=dt, callbacks=[recorder])
    voltage = recorder.numpy("v")[:, 0, :]
    time = np.arange(voltage.shape[0], dtype=np.float64) * dt
    return _Simulation(time, voltage)


def _assert_genuine_propagation(trace: np.ndarray, case: _AxonCase, dt: float):
    assert trace.shape[1] == 3
    peak_indices = np.argmax(trace, axis=0)
    peaks = np.max(trace, axis=0)
    assert np.all(peaks > 20.0), (
        f"{case.name} failed to produce robust source-to-distal spikes: peaks={peaks!r}"
    )
    crossing_indices = np.asarray(
        [np.flatnonzero(trace[:, site] > 0.0)[0] for site in range(trace.shape[1])]
    )
    assert np.all(np.diff(crossing_indices) > 0), (
        f"{case.name} spike thresholds were not crossed source-to-distal: "
        f"crossing times={crossing_indices * dt!r}"
    )
    assert peak_indices[0] < peak_indices[1] < peak_indices[2], (
        f"{case.name} did not propagate source-to-distal: "
        f"peak times={peak_indices * dt!r}"
    )


def _assert_time_aligned(actual: _Simulation, expected: _Simulation, dt: float):
    assert actual.voltage.shape == expected.voltage.shape
    # NEURON accumulates h.t by repeated floating-point addition; a very fine
    # reference therefore carries a few extra picoseconds of roundoff by tstop.
    np.testing.assert_allclose(
        actual.time,
        expected.time,
        rtol=0.0,
        atol=max(dt * 1.0e-10, 1.0e-11),
    )


def _rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = actual - expected
    return float(np.sqrt(np.mean(difference * difference)))


def _spike_feature_times(simulation: _Simulation):
    crossing = np.asarray(
        [
            simulation.time[np.flatnonzero(simulation.voltage[:, site] > 0.0)[0]]
            for site in range(simulation.voltage.shape[1])
        ]
    )
    peak = simulation.time[np.argmax(simulation.voltage, axis=0)]
    return crossing, peak


@pytest.mark.parametrize("case_name", CASES)
def test_active_axon_propagation_matches_neuron(case_name):
    case = CASES[case_name]
    dt = 0.00625
    expected = _run_neuron(case, dt)
    actual = _run_dendra(case, dt)

    _assert_time_aligned(actual, expected, dt)
    _assert_genuine_propagation(expected.voltage, case, dt)
    _assert_genuine_propagation(actual.voltage, case, dt)

    difference = actual.voltage - expected.voltage
    max_abs = float(np.max(np.abs(difference)))
    rmse = _rmse(actual.voltage, expected.voltage)
    max_abs_bound, rmse_bound = {
        # With analytic HH rates on both sides, the conventional cable agrees
        # to well below a microvolt across the full propagated waveform.
        "unmyelinated": (1.0e-3, 2.0e-4),
        # The reduced node-only cable has a very steep spike upstroke; its
        # pointwise error is dominated by a sub-step phase offset. The RMSE
        # guards the complete trajectory more tightly. This residual is not
        # caused by NEURON's HH lookup table, so retain its established bound.
        "myelinated": (5.0, 0.25),
    }[case.name]
    assert max_abs < max_abs_bound and rmse < rmse_bound, (
        f"Active {case.name} mismatch: max_abs={max_abs:.6g} mV, rmse={rmse:.6g} mV"
    )


def test_reduced_myelinated_active_propagation_converges_to_fine_neuron_reference():
    """The challenging node-only cable converges toward one fine oracle."""
    case = CASES["myelinated"]
    dts = (0.025, 0.0125, 0.00625)
    dendra_runs = [_run_dendra(case, dt) for dt in dts]
    reference_dt = dts[-1] / 8.0
    reference = _run_neuron(case, reference_dt)
    _assert_genuine_propagation(reference.voltage, case, reference_dt)

    sampled_references = []
    for dt, actual in zip(dts, dendra_runs):
        stride = round(dt / reference_dt)
        assert dt == pytest.approx(stride * reference_dt, abs=1.0e-15)
        sampled = _Simulation(
            reference.time[::stride],
            reference.voltage[::stride],
        )
        sampled_references.append(sampled)
        _assert_time_aligned(actual, sampled, dt)
        _assert_genuine_propagation(actual.voltage, case, dt)

    # Compare every Dendra resolution to the same substantially finer NEURON
    # trajectory at exact nested sample times.  This measures convergence to a
    # common physical reference rather than agreement between two same-dt,
    # oppositely ordered first-order splittings.
    trace_errors = [
        _rmse(actual.voltage, expected.voltage)
        for actual, expected in zip(dendra_runs, sampled_references)
    ]
    assert trace_errors[1] < 0.65 * trace_errors[0]
    assert trace_errors[2] < 0.65 * trace_errors[1]
    assert trace_errors[2] < 0.2

    reference_crossing, reference_peak = _spike_feature_times(reference)
    feature_errors = []
    for actual in dendra_runs:
        actual_crossing, actual_peak = _spike_feature_times(actual)
        feature_errors.append(
            max(
                float(np.max(np.abs(actual_crossing - reference_crossing))),
                float(np.max(np.abs(actual_peak - reference_peak))),
            )
        )
    assert feature_errors[2] < feature_errors[1] < feature_errors[0]
    assert feature_errors[2] <= 2.0 * dts[2] + 1.0e-12
