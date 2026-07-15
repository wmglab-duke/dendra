"""Full-model NEURON oracles for simulated native Dendra morphologies.

These tests deliberately author the same cable twice: once through Dendra's
native ``Morphology`` API and once through NEURON Sections.  They therefore
exercise the native compiler rather than importing the oracle's graph back
into Dendra.
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
PASSIVE_TSTOP = 6.0
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
    parent_x: float | None = None
    child_end: int = 0


@dataclass(frozen=True)
class _OracleCell:
    morphology: dn.Morphology
    neuron_sections: dict[str, object]
    record_nodes: tuple[int, ...]
    record_segments: tuple[object, ...]


SECTION_SPECS = (
    _SectionSpec(
        "soma",
        ((-15.0, 0.0, 0.0, 18.0), (15.0, 0.0, 0.0, 12.0)),
        1,
        82.0,
        0.9,
        frozenset({"membrane", "stimulus_site"}),
    ),
    _SectionSpec(
        "trunk",
        ((0.0, 0.0, 0.0, 6.0), (0.0, 100.0, 5.0, 2.5)),
        5,
        120.0,
        1.15,
        frozenset({"membrane", "dendrite"}),
        parent="soma",
        parent_x=0.5,
    ),
    _SectionSpec(
        "forward_branch",
        ((0.0, 30.0, 1.5, 4.2), (72.0, 88.0, 14.0, 1.2)),
        5,
        96.0,
        1.35,
        frozenset({"membrane", "dendrite", "readout_branch"}),
        parent="trunk",
        parent_x=0.3,
    ),
    # Authored from its distal tip toward the physical trunk attachment.  The
    # endpoint-1 connection is an important native-coordinate contract: graph
    # traversal is toward decreasing authored x, while labels remain ordered
    # in increasing authored x.
    _SectionSpec(
        "reversed_branch",
        ((-65.0, 142.0, -11.0, 1.0), (0.0, 70.0, 3.5, 3.4)),
        5,
        154.0,
        1.05,
        frozenset({"membrane", "dendrite", "readout_branch"}),
        parent="trunk",
        parent_x=0.7,
        child_end=1,
    ),
)


def _build_oracle_cell() -> _OracleCell:
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

        section = h.Section(name=f"native_oracle_{spec.name}")
        section.Ra = spec.rhoa
        section.cm = spec.cm
        section.nseg = spec.nseg
        h.pt3dclear(sec=section)
        # NEURON reflects logical Section.x relative to stored pt3d arclength
        # for an orientation-1 child.  Reversing those raw points preserves the
        # native API's stable authored x coordinate and the same physical cable.
        points = reversed(spec.points) if spec.child_end == 1 else spec.points
        for point in points:
            h.pt3dadd(*point, sec=section)
        neuron_sections[spec.name] = section

    for spec in SECTION_SPECS:
        if spec.parent is None:
            continue
        native_sections[spec.name].connect(
            native_sections[spec.parent].at(spec.parent_x),
            child_end=spec.child_end,
        )
        neuron_sections[spec.name].connect(
            neuron_sections[spec.parent](spec.parent_x),
            float(spec.child_end),
        )

    graph = morphology.compile()

    def native_node(section_name: str, authored_x: float) -> int:
        matches = [
            node
            for node, (source, x) in enumerate(
                zip(graph.metadata.section_name, graph.metadata.section_x)
            )
            if source == section_name and x == pytest.approx(authored_x)
        ]
        assert len(matches) == 1
        return matches[0]

    # Record the injection site, distal trunk, and both branch tips.  For the
    # reversed branch the distal tip is near authored x=0, not x=1.
    recording_locations = (
        ("soma", 0.5),
        ("trunk", 0.9),
        ("forward_branch", 0.9),
        ("reversed_branch", 0.1),
    )
    record_nodes = tuple(native_node(*location) for location in recording_locations)
    record_segments = tuple(
        neuron_sections[name](authored_x) for name, authored_x in recording_locations
    )
    return _OracleCell(
        morphology,
        neuron_sections,
        record_nodes,
        record_segments,
    )


def _configure_neuron_membrane(cell: _OracleCell, mechanism: str) -> None:
    for section in cell.neuron_sections.values():
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


def _run_neuron(
    cell: _OracleCell,
    *,
    mechanism: str,
    dt: float,
    tstop: float,
    amplitude_na: float,
) -> np.ndarray:
    assert h.usetable_hh == 0
    _configure_neuron_membrane(cell, mechanism)
    clamp = h.IClamp(cell.neuron_sections["soma"](0.5))
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

    rows = [[float(segment.v) for segment in cell.record_segments]]
    for _ in range(n_steps):
        h.fadvance()
        rows.append([float(segment.v) for segment in cell.record_segments])
    return np.asarray(rows)


def _run_dendra(
    cell: _OracleCell,
    *,
    mechanism,
    dt: float,
    tstop: float,
    amplitude_na: float,
) -> np.ndarray:
    model = dn.Tree.from_morphology(
        cell.morphology,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    material_nodes = [
        node
        for node, kind in enumerate(model.compartment_graph.metadata.kind)
        if kind == "compartment"
    ]
    # A multi-Section label is deliberately ordered by declaration and
    # increasing authored Section.x, rather than by solver storage order.
    # It must nevertheless cover every material compartment exactly once.
    assert sorted(model.membrane.index[-1].tolist()) == material_nodes
    assert model.stimulus_site.index[-1].tolist() == [cell.record_nodes[0]]
    # This insertion and injection go through labels produced by the native
    # Section declaration, rather than through manually reconstructed indices.
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
    recorder = dn.callbacks.Recorder(
        states=["v"],
        node_indices=cell.record_nodes,
    )
    model.eval()
    model.initialize()
    model.run(tstop=tstop, dt=dt, callbacks=[recorder])
    return recorder.numpy("v")[:, 0, :]


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


def test_native_tapered_branched_passive_current_injection_matches_neuron():
    cell = _build_oracle_cell()
    expected = _run_neuron(
        cell,
        mechanism="pas",
        dt=PASSIVE_DT,
        tstop=PASSIVE_TSTOP,
        amplitude_na=0.25,
    )
    actual = _run_dendra(
        cell,
        mechanism=pas,
        dt=PASSIVE_DT,
        tstop=PASSIVE_TSTOP,
        amplitude_na=0.25,
    )

    assert actual.shape == expected.shape
    excursion = np.max(np.abs(expected - expected[0]), axis=0)
    assert np.all(excursion > 0.02), (
        f"Passive oracle response was trivial: {excursion!r}"
    )
    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = _rmse(actual, expected)
    assert np.allclose(actual, expected, rtol=2.0e-6, atol=3.0e-4), (
        f"Native passive Tree mismatch: max_abs={max_abs:.6g} mV, rmse={rmse:.6g} mV"
    )


def test_native_tapered_branched_hh_converges_to_neuron():
    cell = _build_oracle_cell()
    reference = _run_neuron(
        cell,
        mechanism="hh",
        dt=ACTIVE_REFERENCE_DT,
        tstop=ACTIVE_TSTOP,
        amplitude_na=8.0,
    )
    actual_by_dt = [
        _run_dendra(
            cell,
            mechanism=hh,
            dt=dt,
            tstop=ACTIVE_TSTOP,
            amplitude_na=8.0,
        )
        for dt in ACTIVE_DTS
    ]

    assert np.all(np.max(reference, axis=0) > 0.0), (
        "The active oracle must propagate a spike to every strategic site."
    )
    errors = []
    per_site_errors = []
    max_errors = []
    for dt, actual in zip(ACTIVE_DTS, actual_by_dt):
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

    # Dendra and NEURON use first-order Lie splitting with opposite substep
    # order.  Convergence to a much finer NEURON trajectory should therefore
    # be monotone and approximately first order under timestep halving.
    assert errors[0] > errors[1] > errors[2] > 0.0, (
        f"Native HH voltage errors did not refine: {errors!r}"
    )
    for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
        assert 1.7 < ratio < 2.4, (
            "Native HH voltage did not show first-order refinement: "
            f"rmse={errors!r}, max={max_errors!r}, ratio={ratio:.6g}"
        )

    # Aggregate refinement cannot hide a regression confined to one branch.
    # Require each independently recorded path to exhibit the same order.
    for site in range(reference.shape[1]):
        site_errors = [errors_for_dt[site] for errors_for_dt in per_site_errors]
        assert site_errors[0] > site_errors[1] > site_errors[2] > 0.0
        for ratio in (
            site_errors[0] / site_errors[1],
            site_errors[1] / site_errors[2],
        ):
            assert 1.7 < ratio < 2.4, (
                f"Native HH site {site} did not refine at first order: "
                f"errors={site_errors!r}, ratio={ratio:.6g}"
            )

    reference_crossings = _first_crossing_times(reference, ACTIVE_REFERENCE_DT)
    finest_crossings = _first_crossing_times(actual_by_dt[-1], ACTIVE_DTS[-1])
    assert np.all(reference_crossings[1:] > reference_crossings[0])
    assert np.all(finest_crossings[1:] > finest_crossings[0])
    assert np.max(np.abs(finest_crossings - reference_crossings)) < 2 * ACTIVE_DTS[-1]

    voltage_scale = float(np.max(reference) - np.min(reference))
    assert errors[-1] < 0.01 * voltage_scale, (
        "Native HH finest-step error is too large despite refinement: "
        f"rmse={errors[-1]:.6g}, scale={voltage_scale:.6g}"
    )
