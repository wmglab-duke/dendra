"""NEURON oracles for the generic native ``Cable`` fast path.

The morphology is authored independently in Dendra and NEURON.  Dendra's
``Tree`` implementation is also run from the same native declaration so these
tests distinguish native graph/compiler errors from Cable-specific path
packing or solver errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.mod import hh, pas
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DTYPE = torch.float64
V_INIT = -65.0
CELSIUS = 22.0
PASSIVE_DT = 0.025
PASSIVE_TSTOP = 5.0
EXTRACELLULAR_DT = 0.025
EXTRACELLULAR_STEPS = 160
ACTIVE_DTS = (0.025, 0.0125, 0.00625)
ACTIVE_REFERENCE_DT = ACTIVE_DTS[-1] / 8.0
ACTIVE_TSTOP = 4.0
STIMULUS_DELAY = 0.25
STIMULUS_DURATION = 0.25


@dataclass(frozen=True)
class _SectionSpec:
    name: str
    points: tuple[tuple[float, float, float, float], ...]
    nseg: int
    rhoa: float
    cm: float
    labels: frozenset[str]
    parent: str | None = None
    child_end: int = 0


@dataclass(frozen=True)
class _OracleCable:
    morphology: dn.Morphology
    neuron_sections: dict[str, object]
    record_segments: tuple[object, ...]


SECTION_SPECS = (
    _SectionSpec(
        "source",
        ((-15.0, 0.0, 0.0, 18.0), (15.0, 0.0, 0.0, 12.0)),
        1,
        82.0,
        0.9,
        frozenset({"membrane", "stimulus_site"}),
    ),
    _SectionSpec(
        "forward_path",
        ((15.0, 0.0, 0.0, 6.0), (18.0, 72.0, 8.0, 2.4)),
        5,
        126.0,
        1.25,
        frozenset({"membrane", "neurite"}),
        parent="source",
    ),
    # Authored from the distal tip toward its physical attachment.  Connecting
    # endpoint 1 makes solver traversal run toward decreasing authored x.
    _SectionSpec(
        "reversed_path",
        ((88.0, 155.0, 17.0, 0.8), (18.0, 72.0, 8.0, 2.8)),
        5,
        171.0,
        1.05,
        frozenset({"membrane", "neurite", "distal_path"}),
        parent="forward_path",
        child_end=1,
    ),
)

RECORDING_LOCATIONS = (
    ("source", 0.5),
    ("forward_path", 0.9),
    # The far tip of an endpoint-1 child is near authored x=0.
    ("reversed_path", 0.1),
)


def _build_oracle_cable() -> _OracleCable:
    morphology = dn.Morphology()
    native_sections = {}
    neuron_sections = {}

    for spec in SECTION_SPECS:
        native_sections[spec.name] = morphology.section(
            spec.name,
            points=spec.points,
            nseg=spec.nseg,
            rhoa=spec.rhoa,
            cm=spec.cm,
            labels=spec.labels,
        )

        section = h.Section(name=f"native_cable_oracle_{spec.name}")
        section.Ra = spec.rhoa
        section.cm = spec.cm
        section.nseg = spec.nseg
        h.pt3dclear(sec=section)
        # For an orientation-1 NEURON Section, reflect the stored point order
        # so logical Section.x is the native declaration's stable authored x.
        points = reversed(spec.points) if spec.child_end == 1 else spec.points
        for point in points:
            h.pt3dadd(*point, sec=section)
        neuron_sections[spec.name] = section

    for spec in SECTION_SPECS:
        if spec.parent is None:
            continue
        native_sections[spec.name].connect(
            native_sections[spec.parent].at(1.0),
            child_end=spec.child_end,
        )
        neuron_sections[spec.name].connect(
            neuron_sections[spec.parent](1.0),
            float(spec.child_end),
        )

    record_segments = tuple(
        neuron_sections[name](authored_x) for name, authored_x in RECORDING_LOCATIONS
    )
    return _OracleCable(morphology, neuron_sections, record_segments)


def _configure_neuron_membrane(cable: _OracleCable, mechanism: str) -> None:
    for section in cable.neuron_sections.values():
        section.insert(mechanism)
        if mechanism == "pas":
            for segment in section:
                segment.pas.g = 3.0e-4
                segment.pas.e = -70.0
        else:
            for segment in section:
                assert segment.hh.gnabar == pytest.approx(0.12)
                assert segment.hh.gkbar == pytest.approx(0.036)
                assert segment.hh.gl == pytest.approx(0.0003)
                assert segment.hh.el == pytest.approx(-54.3)


def _configure_neuron_extracellular(cable: _OracleCable) -> None:
    """Install an ideal prescribed extracellular-potential boundary."""
    for section in cable.neuron_sections.values():
        section.insert("extracellular")
        for segment in section:
            for layer in range(int(h.nlayer_extracellular())):
                segment.xraxial[layer] = 1.0e9
                segment.xg[layer] = 1.0e9
                segment.xc[layer] = 0.0


def _run_neuron(
    cable: _OracleCable,
    *,
    mechanism: str,
    dt: float,
    tstop: float,
    amplitude_na: float,
) -> np.ndarray:
    assert h.usetable_hh == 0
    _configure_neuron_membrane(cable, mechanism)
    clamp = h.IClamp(cable.neuron_sections["source"](0.5))
    clamp.delay = STIMULUS_DELAY
    clamp.dur = STIMULUS_DURATION
    clamp.amp = amplitude_na

    n_steps = round(tstop / dt)
    assert tstop == pytest.approx(n_steps * dt, abs=1.0e-14)
    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = dt
    h.finitialize(V_INIT)

    rows = [[float(segment.v) for segment in cable.record_segments]]
    for _ in range(n_steps):
        h.fadvance()
        rows.append([float(segment.v) for segment in cable.record_segments])
    return np.asarray(rows)


def _nodes_for_locations(graph, locations=RECORDING_LOCATIONS) -> tuple[int, ...]:
    nodes = []
    for section_name, authored_x in locations:
        matches = [
            node
            for node, (source, x) in enumerate(
                zip(graph.metadata.section_name, graph.metadata.section_x)
            )
            if source == section_name and x == pytest.approx(authored_x)
        ]
        assert len(matches) == 1
        nodes.append(matches[0])
    return tuple(nodes)


def _segments_in_graph_order(cable: _OracleCable, graph) -> tuple[object, ...]:
    """Map every canonical native compartment back to its NEURON segment."""
    segments = tuple(
        cable.neuron_sections[section_name](float(section_x))
        for section_name, section_x in zip(
            graph.metadata.section_name,
            graph.metadata.section_x,
        )
    )
    assert len(segments) == graph.n_compartments
    return segments


def _extracellular_spatial_profile(model) -> torch.Tensor:
    """Return a curved, gauge-centred potential over native pt3d centres."""
    coordinate = 0.8 * model.x - 0.45 * model.y + 0.3 * model.z
    coordinate = coordinate - coordinate.mean(dim=-1, keepdim=True)
    scale = coordinate.abs().amax(dim=-1, keepdim=True)
    assert torch.all(scale > 0.0)
    normalized = coordinate / scale
    profile = 0.65 * normalized + 0.35 * torch.cos(torch.pi * (normalized - 0.2))
    profile = profile - profile.mean(dim=-1, keepdim=True)
    profile = 7.5 * profile / profile.abs().amax(dim=-1, keepdim=True)
    assert torch.unique(profile).numel() == model.nc
    assert torch.any(profile > 0.0)
    assert torch.any(profile < 0.0)
    return profile


def _extracellular_temporal_profile() -> torch.Tensor:
    """Return a smooth, polarity-reversing fixed-step field waveform."""
    time = torch.arange(EXTRACELLULAR_STEPS, dtype=DTYPE) * EXTRACELLULAR_DT
    profile = 0.65 * torch.sin(2.0 * torch.pi * 0.31 * time)
    profile += 0.35 * torch.sin(2.0 * torch.pi * 0.83 * time + 0.4)
    assert torch.any(profile > 0.0)
    assert torch.any(profile < 0.0)
    return profile.unsqueeze(0)


def _run_neuron_extracellular(
    cable: _OracleCable,
    segments: tuple[object, ...],
    spatial: torch.Tensor,
    temporal: torch.Tensor,
) -> np.ndarray:
    assert h.usetable_hh == 0
    _configure_neuron_membrane(cable, "pas")
    _configure_neuron_extracellular(cable)

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = EXTRACELLULAR_DT
    for segment in segments:
        segment.e_extracellular = 0.0
    h.finitialize(V_INIT)

    trace = [[float(segment.v) for segment in segments]]
    assignments = tuple(zip(segments, spatial[0].tolist()))
    for scale in temporal[0].tolist():
        for segment, field_value in assignments:
            segment.e_extracellular = field_value * scale
        h.fadvance()
        trace.append([float(segment.v) for segment in segments])
    return np.asarray(trace)


def _run_dendra(
    model_type,
    cable: _OracleCable,
    *,
    mechanism,
    dt: float,
    tstop: float,
    amplitude_na: float,
) -> tuple[object, np.ndarray]:
    model = model_type.from_morphology(
        cable.morphology,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    graph = model.compartment_graph
    record_nodes = _nodes_for_locations(graph)
    material_nodes = [
        node for node, kind in enumerate(graph.metadata.kind) if kind == "compartment"
    ]
    assert len(material_nodes) == graph.n_compartments
    assert sorted(model.membrane.index[-1].tolist()) == material_nodes
    assert model.stimulus_site.index[-1].tolist() == [record_nodes[0]]

    # Both operations intentionally use labels propagated from native Sections.
    if mechanism is pas:
        model.membrane.insert(pas, g=3.0e-4, e=-70.0)
    else:
        model.membrane.insert(hh)
    model.stimulus_site.inject(
        dn.mono_rect(
            amp=amplitude_na * nA,
            delay=STIMULUS_DELAY,
            pw=STIMULUS_DURATION,
        )
    )
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=record_nodes)
    model.eval()
    model.initialize()
    model.run(tstop=tstop, dt=dt, callbacks=[recorder])
    return model, recorder.numpy("v")[:, 0, :]


def _run_dendra_extracellular(
    model_type,
    cable: _OracleCable,
    *,
    spatial: torch.Tensor,
    temporal: torch.Tensor,
) -> tuple[object, np.ndarray]:
    model = model_type.from_morphology(
        cable.morphology,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    model.membrane.insert(pas, g=3.0e-4, e=-70.0)
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=list(range(model.nc)))
    model.eval()
    model.initialize()
    model.run(
        tstop=EXTRACELLULAR_STEPS * EXTRACELLULAR_DT,
        dt=EXTRACELLULAR_DT,
        extra=(spatial, temporal),
        callbacks=[recorder],
    )
    return model, recorder.numpy("v")[:, 0, :]


def _rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = actual - expected
    assert np.isfinite(difference).all()
    return float(np.sqrt(np.mean(difference * difference)))


def _first_crossing_times(voltage: np.ndarray, dt: float) -> np.ndarray:
    crossing_times = []
    for site in range(voltage.shape[1]):
        values = voltage[:, site]
        crossings = np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0))
        assert crossings.size >= 1, f"Recording site {site} did not spike."
        left = int(crossings[0])
        fraction = -values[left] / (values[left + 1] - values[left])
        crossing_times.append((left + float(fraction)) * dt)
    return np.asarray(crossing_times)


def _assert_cable_path_contract(cable_model) -> None:
    graph = cable_model.compartment_graph
    assert graph.topology.parent_index == (-1, *range(graph.n_compartments - 1))

    # The Cable snapshot is allowed to be reordered relative to the compiler's
    # original IDs, but metadata and labels must stay aligned.  The reversed
    # section remains publicly ordered by increasing authored Section.x.
    reversed_nodes = cable_model.reversed_path.index[-1].tolist()
    assert [graph.metadata.section_x[node] for node in reversed_nodes] == pytest.approx(
        [0.1, 0.3, 0.5, 0.7, 0.9]
    )


def test_native_tapered_heterogeneous_cable_passive_matches_neuron_and_tree():
    cable = _build_oracle_cable()
    expected = _run_neuron(
        cable,
        mechanism="pas",
        dt=PASSIVE_DT,
        tstop=PASSIVE_TSTOP,
        amplitude_na=0.25,
    )
    cable_model, actual = _run_dendra(
        dn.Cable,
        cable,
        mechanism=pas,
        dt=PASSIVE_DT,
        tstop=PASSIVE_TSTOP,
        amplitude_na=0.25,
    )
    _, tree = _run_dendra(
        dn.Tree,
        cable,
        mechanism=pas,
        dt=PASSIVE_DT,
        tstop=PASSIVE_TSTOP,
        amplitude_na=0.25,
    )

    _assert_cable_path_contract(cable_model)
    assert actual.shape == tree.shape == expected.shape
    excursion = np.max(np.abs(expected - expected[0]), axis=0)
    assert np.all(excursion > 0.02), f"Passive response was trivial: {excursion!r}"
    np.testing.assert_allclose(actual, tree, rtol=2.0e-12, atol=2.0e-10)

    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = _rmse(actual, expected)
    assert np.allclose(actual, expected, rtol=2.0e-6, atol=3.0e-4), (
        f"Native Cable passive mismatch: max_abs={max_abs:.6g}, rmse={rmse:.6g} mV"
    )


def test_native_tapered_cable_extracellular_field_matches_neuron_and_tree():
    cable = _build_oracle_cable()
    template = dn.Cable.from_morphology(
        cable.morphology,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    spatial = _extracellular_spatial_profile(template)
    temporal = _extracellular_temporal_profile()
    segments = _segments_in_graph_order(cable, template.compartment_graph)

    baseline = _run_neuron_extracellular(
        cable,
        segments,
        spatial,
        torch.zeros_like(temporal),
    )
    expected = _run_neuron_extracellular(cable, segments, spatial, temporal)
    cable_model, actual = _run_dendra_extracellular(
        dn.Cable,
        cable,
        spatial=spatial,
        temporal=temporal,
    )
    tree_model, tree = _run_dendra_extracellular(
        dn.Tree,
        cable,
        spatial=spatial,
        temporal=temporal,
    )

    _assert_cable_path_contract(cable_model)
    assert cable_model.compartment_graph == tree_model.compartment_graph
    assert actual.shape == tree.shape == expected.shape == baseline.shape
    assert actual.shape == (EXTRACELLULAR_STEPS + 1, template.nc)
    np.testing.assert_allclose(actual, tree, rtol=2.0e-12, atol=2.0e-10)

    effect = np.max(np.abs(expected - baseline), axis=0)
    record_nodes = _nodes_for_locations(template.compartment_graph)
    assert np.all(effect[list(record_nodes)] > 0.02), (
        "The prescribed field must materially affect every strategic site: "
        f"{effect[list(record_nodes)]!r}"
    )
    assert float(np.max(effect)) > 0.1
    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = _rmse(actual, expected)
    assert np.allclose(actual, expected, rtol=2.0e-6, atol=3.0e-4), (
        "Native Cable extracellular mismatch: "
        f"max_abs={max_abs:.6g}, rmse={rmse:.6g} mV"
    )


def test_native_tapered_heterogeneous_cable_hh_converges_to_neuron_and_tree():
    cable = _build_oracle_cable()
    reference = _run_neuron(
        cable,
        mechanism="hh",
        dt=ACTIVE_REFERENCE_DT,
        tstop=ACTIVE_TSTOP,
        amplitude_na=8.0,
    )
    cable_runs = []
    for dt in ACTIVE_DTS:
        cable_model, actual = _run_dendra(
            dn.Cable,
            cable,
            mechanism=hh,
            dt=dt,
            tstop=ACTIVE_TSTOP,
            amplitude_na=8.0,
        )
        _, tree = _run_dendra(
            dn.Tree,
            cable,
            mechanism=hh,
            dt=dt,
            tstop=ACTIVE_TSTOP,
            amplitude_na=8.0,
        )
        _assert_cable_path_contract(cable_model)
        np.testing.assert_allclose(actual, tree, rtol=2.0e-12, atol=2.0e-10)
        cable_runs.append(actual)

    assert np.all(np.max(reference, axis=0) > 0.0), (
        "The active oracle must propagate a spike along the complete Cable."
    )
    errors = []
    per_site_errors = []
    max_errors = []
    for dt, actual in zip(ACTIVE_DTS, cable_runs):
        stride = round(dt / ACTIVE_REFERENCE_DT)
        assert dt == pytest.approx(stride * ACTIVE_REFERENCE_DT, abs=1.0e-14)
        expected = reference[::stride]
        assert actual.shape == expected.shape
        errors.append(_rmse(actual, expected))
        per_site_errors.append(
            [
                _rmse(actual[:, site], expected[:, site])
                for site in range(actual.shape[1])
            ]
        )
        max_errors.append(float(np.max(np.abs(actual - expected))))

    assert errors[0] > errors[1] > errors[2] > 0.0
    for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
        assert 1.7 < ratio < 2.4, (
            "Native Cable HH voltage did not refine at first order: "
            f"rmse={errors!r}, max={max_errors!r}, ratio={ratio:.6g}"
        )

    for site in range(reference.shape[1]):
        site_errors = [errors_for_dt[site] for errors_for_dt in per_site_errors]
        assert site_errors[0] > site_errors[1] > site_errors[2] > 0.0
        for ratio in (
            site_errors[0] / site_errors[1],
            site_errors[1] / site_errors[2],
        ):
            assert 1.7 < ratio < 2.4, (
                f"Native Cable HH site {site} did not refine at first order: "
                f"errors={site_errors!r}, ratio={ratio:.6g}"
            )

    reference_crossings = _first_crossing_times(reference, ACTIVE_REFERENCE_DT)
    finest_crossings = _first_crossing_times(cable_runs[-1], ACTIVE_DTS[-1])
    assert np.all(reference_crossings[1:] > reference_crossings[:-1])
    assert np.all(finest_crossings[1:] > finest_crossings[:-1])
    assert np.max(np.abs(finest_crossings - reference_crossings)) < 2 * ACTIVE_DTS[-1]

    voltage_scale = float(np.max(reference) - np.min(reference))
    assert errors[-1] < 0.01 * voltage_scale
