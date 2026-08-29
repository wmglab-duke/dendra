"""Temperature-stressed active propagation oracles against fine NEURON runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mod import hh
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DTYPE = torch.float64
DTS = (0.0125, 0.00625, 0.003125)
REFERENCE_DT = DTS[-1] / 8.0
TSTOP = 4.0
V_INIT = -65.0
RHOA = 35.4
CM = 1.0
STIM_DELAY = 0.25
STIM_DURATION = 0.25
SPIKE_THRESHOLD = 0.0
STATE_NAMES = ("m", "h", "n")
CURRENT_NAMES = ("ina", "ik", "il")
TRACE_NAMES = ("v", *STATE_NAMES, *CURRENT_NAMES)


@dataclass(frozen=True)
class _AxonGeometry:
    name: str
    n_comp: int
    diameter: float
    compartment_length: float
    effective_rhoa: float
    coordinate_spacing: float
    stimulus_node: int
    record_nodes: tuple[int, int, int]
    stimulus_amplitude_na: float


@dataclass
class _PreparedCase:
    name: str
    node_to_segment: dict[int, object]
    record_nodes: tuple[int, ...]
    stimulus_node: int
    stimulus_amplitude_na: float
    build_dendra: object
    propagation_distances_um: np.ndarray | None


AXON_CASES = {
    "unmyelinated": _AxonGeometry(
        name="unmyelinated",
        n_comp=21,
        diameter=6.0,
        compartment_length=20.0,
        effective_rhoa=RHOA,
        coordinate_spacing=20.0,
        stimulus_node=3,
        record_nodes=(3, 10, 17),
        stimulus_amplitude_na=14.0,
    ),
    "myelinated": _AxonGeometry(
        name="myelinated",
        n_comp=9,
        diameter=7.0,
        compartment_length=2.0,
        effective_rhoa=RHOA * 500.0,
        coordinate_spacing=1000.0,
        stimulus_node=1,
        record_nodes=(1, 4, 7),
        stimulus_amplitude_na=0.8,
    ),
}


def _canonical_currents(v, m, h_gate, n):
    return {
        "ina": 0.12 * m**3 * h_gate * (v - 50.0),
        "ik": 0.036 * n**4 * (v + 77.0),
        "il": 0.0003 * (v + 54.3),
    }


def _configure_neuron_hh(sections):
    for section in sections:
        section.insert("hh")
        for segment in section:
            assert segment.hh.gnabar == pytest.approx(0.12)
            assert segment.hh.gkbar == pytest.approx(0.036)
            assert segment.hh.gl == pytest.approx(0.0003)
            assert segment.hh.el == pytest.approx(-54.3)


def _prepare_axon(case_name, celsius):
    geometry = AXON_CASES[case_name]
    section = h.Section(name=f"hot_{case_name}_{celsius:g}")
    section.L = geometry.n_comp * geometry.compartment_length
    section.diam = geometry.diameter
    section.Ra = geometry.effective_rhoa
    section.cm = CM
    section.nseg = geometry.n_comp
    _configure_neuron_hh((section,))
    segments = list(section)

    def build_dendra():
        if geometry.name == "unmyelinated":
            model = dn.Unmyelinated(
                diameters=[geometry.diameter],
                L=geometry.n_comp * geometry.compartment_length,
                dx=geometry.compartment_length,
                celsius=celsius,
                v_init=V_INIT,
                rhoa=RHOA,
                cm=CM,
                dtype=DTYPE,
            )
        else:
            model = dn.Myelinated(
                diameters=[10.0],
                n_node=geometry.n_comp,
                node_length=geometry.compartment_length,
                celsius=celsius,
                v_init=V_INIT,
                rhoa=RHOA,
                cm=CM,
                dtype=DTYPE,
            )
        model.insert(hh)
        model[..., geometry.stimulus_node].inject(
            dn.mono_rect(
                amp=geometry.stimulus_amplitude_na * nA,
                delay=STIM_DELAY,
                pw=STIM_DURATION,
            )
        )
        model.eval()
        model.initialize()

        expected_x = (
            torch.arange(geometry.n_comp, dtype=DTYPE) - (geometry.n_comp - 1) / 2.0
        ) * geometry.coordinate_spacing
        torch.testing.assert_close(model.x[0], expected_x)
        for name, expected in (
            ("diam", geometry.diameter),
            ("dx", geometry.compartment_length),
            ("rhoa", geometry.effective_rhoa),
            ("cm", CM),
        ):
            value = getattr(model, name)
            torch.testing.assert_close(value, torch.full_like(value, expected))
        return model

    distances = np.asarray(
        [
            abs(node - geometry.stimulus_node) * geometry.coordinate_spacing
            for node in geometry.record_nodes
        ],
        dtype=np.float64,
    )
    return _PreparedCase(
        geometry.name,
        dict(enumerate(segments)),
        geometry.record_nodes,
        geometry.stimulus_node,
        geometry.stimulus_amplitude_na,
        build_dendra,
        distances,
    )


def _pt3d_section(name, points, *, ra, cm, nseg):
    section = h.Section(name=name)
    section.Ra = float(ra)
    section.cm = float(cm)
    section.nseg = int(nseg)
    h.pt3dclear(sec=section)
    for point in points:
        h.pt3dadd(*point, sec=section)
    return section


def _prepare_tree(celsius):
    soma = _pt3d_section(
        f"hot_tree_soma_{celsius:g}",
        [(-20.0, 0.0, 0.0, 18.0), (20.0, 0.0, 0.0, 12.0)],
        ra=80.0,
        cm=0.9,
        nseg=3,
    )
    trunk = _pt3d_section(
        f"hot_tree_trunk_{celsius:g}",
        [(0.0, 0.0, 0.0, 6.0), (0.0, 90.0, 5.0, 2.5)],
        ra=120.0,
        cm=1.2,
        nseg=4,
    )
    branch_a = _pt3d_section(
        f"hot_tree_a_{celsius:g}",
        [(0.0, 35.0, 2.0, 4.0), (75.0, 85.0, 15.0, 1.3)],
        ra=95.0,
        cm=1.35,
        nseg=4,
    )
    branch_b = _pt3d_section(
        f"hot_tree_b_{celsius:g}",
        [(0.0, 75.0, 4.0, 3.2), (-65.0, 135.0, -12.0, 1.0)],
        ra=155.0,
        cm=1.05,
        nseg=4,
    )
    trunk.connect(soma(0.5), 0.0)
    branch_a.connect(trunk(0.375), 0.0)
    branch_b.connect(trunk(0.875), 0.0)
    sections = (soma, trunk, branch_a, branch_b)
    _configure_neuron_hh(sections)
    graph, node_to_segment = neuron_to_dendra_graph(soma)
    assert not any(
        str(attrs.get("name", "")).startswith("branchpoint.")
        for _, attrs in graph.nodes(data=True)
    )

    def node_for(section, segment_index):
        target = list(section)[segment_index]
        matches = [
            node
            for node, segment in node_to_segment.items()
            if segment.sec == target.sec
            and float(segment.x) == pytest.approx(float(target.x))
        ]
        assert len(matches) == 1
        return matches[0]

    stimulus_node = node_for(soma, 1)
    record_nodes = (
        stimulus_node,
        node_for(branch_a, -1),
        node_for(branch_b, -1),
    )
    stimulus_amplitude_na = 8.0

    def build_dendra():
        model = dn.Tree.from_graph(
            graph,
            N=1,
            celsius=celsius,
            v_init=V_INIT,
            dtype=DTYPE,
        )
        model.insert(hh)
        model[..., stimulus_node].inject(
            dn.mono_rect(
                amp=stimulus_amplitude_na * nA,
                delay=STIM_DELAY,
                pw=STIM_DURATION,
            )
        )
        model.eval()
        model.initialize()
        return model

    return _PreparedCase(
        "tree",
        node_to_segment,
        record_nodes,
        stimulus_node,
        stimulus_amplitude_na,
        build_dendra,
        None,
    )


def _run_neuron(case, celsius, dt):
    assert h.usetable_hh == 0
    n_steps = round(TSTOP / dt)
    assert TSTOP == pytest.approx(n_steps * dt, abs=1.0e-14)
    stimulus_segment = case.node_to_segment[case.stimulus_node]
    stimulus = h.IClamp(stimulus_segment)
    stimulus.delay = STIM_DELAY
    stimulus.dur = STIM_DURATION
    stimulus.amp = case.stimulus_amplitude_na
    assert stimulus.get_segment().sec == stimulus_segment.sec

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = celsius
    h.dt = dt
    h.finitialize(V_INIT)
    recorded = [case.node_to_segment[node] for node in case.record_nodes]

    def sample():
        return np.asarray(
            [
                [float(segment.v) for segment in recorded],
                [float(segment.hh.m) for segment in recorded],
                [float(segment.hh.h) for segment in recorded],
                [float(segment.hh.n) for segment in recorded],
            ]
        )

    rows = [sample()]
    for _ in range(n_steps):
        h.fadvance()
        rows.append(sample())
    values = np.asarray(rows)
    result = {
        name: values[:, index, :] for index, name in enumerate(("v", "m", "h", "n"))
    }
    # Compare currents at the simultaneously reported physical state, not
    # NEURON's stale BREAKPOINT current fields.
    result.update(
        _canonical_currents(result["v"], result["m"], result["h"], result["n"])
    )
    return result


def _run_dendra(case, dt):
    model = case.build_dendra()
    mechanism = model.mech.hh
    expected_q10 = 3.0 ** ((float(model.celsius) - 6.3) / 10.0)
    q10 = mechanism.DE["mhn"].q10
    torch.testing.assert_close(
        q10,
        torch.full_like(q10, expected_q10),
        rtol=2.0e-15,
        atol=0.0,
    )
    nodes = torch.as_tensor(case.record_nodes, dtype=torch.long)
    recorder = dn.callbacks.RecorderLambda(
        {
            "v": lambda current: current.v.index_select(-1, nodes),
            "m": lambda _current: mechanism.m.index_select(-1, nodes),
            "h": lambda _current: mechanism.h.index_select(-1, nodes),
            "n": lambda _current: mechanism.n.index_select(-1, nodes),
            "ina": lambda current: mechanism.ina(current.v).index_select(-1, nodes),
            "ik": lambda current: mechanism.ik(current.v).index_select(-1, nodes),
            "il": lambda current: mechanism.il(current.v).index_select(-1, nodes),
        }
    )
    model.run(tstop=TSTOP, dt=dt, callbacks=[recorder])
    n_samples = round(TSTOP / dt) + 1
    return {
        name: recorder.numpy(name).reshape(n_samples, len(case.record_nodes))
        for name in TRACE_NAMES
    }


def _features(voltage, dt):
    crossing = []
    for site in range(voltage.shape[1]):
        values = voltage[:, site]
        indices = np.flatnonzero(
            (values[:-1] < SPIKE_THRESHOLD) & (values[1:] >= SPIKE_THRESHOLD)
        )
        assert indices.size >= 1, f"site {site} never crossed {SPIKE_THRESHOLD:g} mV"
        left = int(indices[0])
        fraction = (SPIKE_THRESHOLD - values[left]) / (values[left + 1] - values[left])
        crossing.append((left + float(fraction)) * dt)
    crossing = np.asarray(crossing)

    peak_index = np.argmax(voltage, axis=0)
    peak_time = []
    peak_voltage = []
    for site, index in enumerate(peak_index):
        index = int(index)
        center = float(voltage[index, site])
        if index == 0 or index == voltage.shape[0] - 1:
            offset = 0.0
            vertex = center
        else:
            left = float(voltage[index - 1, site])
            right = float(voltage[index + 1, site])
            denominator = left - 2.0 * center + right
            offset = 0.0 if denominator == 0.0 else 0.5 * (left - right) / denominator
            offset = float(np.clip(offset, -1.0, 1.0))
            vertex = center - 0.25 * (left - right) * offset
        peak_time.append((index + offset) * dt)
        peak_voltage.append(vertex)
    peak_time = np.asarray(peak_time)
    peak_voltage = np.asarray(peak_voltage)
    ahp = np.asarray(
        [np.min(voltage[peak_index[site] :, site]) for site in range(voltage.shape[1])]
    )
    return crossing, peak_time, peak_voltage, ahp


def _assert_convergence(case, celsius, reference, runs):
    for name in TRACE_NAMES:
        errors = []
        max_errors = []
        for dt, actual in zip(DTS, runs):
            stride = round(dt / REFERENCE_DT)
            expected = reference[name][::stride]
            assert actual[name].shape == expected.shape
            difference = actual[name] - expected
            assert np.isfinite(difference).all()
            errors.append(float(np.sqrt(np.mean(difference * difference))))
            max_errors.append(float(np.max(np.abs(difference))))

        assert errors[0] > errors[1] > errors[2] > 0.0, (
            f"{case.name} {celsius:g} C {name} errors not monotone: "
            f"rmse={errors!r}, max={max_errors!r}"
        )
        for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
            assert 1.8 < ratio < 2.35, (
                f"{case.name} {celsius:g} C {name} not first order: "
                f"rmse={errors!r}, ratio={ratio:.6g}"
            )

        reference_scale = float(np.max(reference[name]) - np.min(reference[name]))
        reference_scale = max(reference_scale, float(np.max(np.abs(reference[name]))))
        # The current worst case is below 0.7%; retain more than twofold
        # cross-platform headroom without allowing a materially degraded
        # trajectory to hide behind otherwise monotone convergence.
        assert errors[-1] < 0.015 * reference_scale, (
            f"{case.name} {celsius:g} C {name} finest normalized RMSE too large: "
            f"rmse={errors[-1]:.6g}, scale={reference_scale:.6g}"
        )

    reference_features = _features(reference["v"], REFERENCE_DT)
    actual_features = [_features(run["v"], dt) for run, dt in zip(runs, DTS)]
    if case.name == "tree":
        assert np.all(reference_features[0][1:] > reference_features[0][0])
        for features in actual_features:
            assert np.all(features[0][1:] > features[0][0])
    for feature_index, label in ((0, "crossing"), (1, "peak time")):
        errors = [
            float(
                np.max(
                    np.abs(features[feature_index] - reference_features[feature_index])
                )
            )
            for features in actual_features
        ]
        assert errors[2] <= errors[1] <= errors[0], (
            f"{case.name} {celsius:g} C {label} errors not refining: {errors!r}"
        )
        assert errors[-1] <= 2.0 * DTS[-1]

    for feature_index, label in ((2, "peak voltage"), (3, "AHP")):
        errors = [
            float(
                np.max(
                    np.abs(features[feature_index] - reference_features[feature_index])
                )
            )
            for features in actual_features
        ]
        assert errors[2] < errors[1] < errors[0], (
            f"{case.name} {celsius:g} C {label} errors not refining: {errors!r}"
        )
        assert errors[-1] < 2.0

    if case.propagation_distances_um is not None:
        reference_crossing = reference_features[0]
        reference_delays = reference_crossing[1:] - reference_crossing[0]
        assert np.all(reference_delays > 0.0)
        reference_velocity = case.propagation_distances_um[1:] / reference_delays
        velocity_errors = []
        for features in actual_features:
            delays = features[0][1:] - features[0][0]
            assert np.all(delays > 0.0)
            velocity = case.propagation_distances_um[1:] / delays
            velocity_errors.append(
                float(
                    np.max(np.abs(velocity - reference_velocity) / reference_velocity)
                )
            )
        # Velocity is a quotient of two interpolated crossing-time differences;
        # tiny cancellation can make the two coarser maxima swap order even
        # while every trajectory and crossing time refines. Require the finest
        # result to improve on both coarser estimates instead.
        assert velocity_errors[2] < min(velocity_errors[0], velocity_errors[1])
        assert velocity_errors[-1] < 0.03


@pytest.mark.parametrize("celsius", [22.0, 34.0])
@pytest.mark.parametrize("case_name", ["unmyelinated", "myelinated", "tree"])
def test_high_temperature_active_propagation_converges_to_neuron(case_name, celsius):
    case = (
        _prepare_tree(celsius)
        if case_name == "tree"
        else _prepare_axon(case_name, celsius)
    )
    reference = _run_neuron(case, celsius, REFERENCE_DT)
    runs = [_run_dendra(case, dt) for dt in DTS]

    assert np.all(np.max(reference["v"], axis=0) > SPIKE_THRESHOLD)
    reference_crossing = _features(reference["v"], REFERENCE_DT)[0]
    assert reference_crossing[0] >= STIM_DELAY
    _assert_convergence(case, celsius, reference, runs)
