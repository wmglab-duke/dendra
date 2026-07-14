"""Generated passive NEURON oracles for native Dendra morphologies.

The cases in this module are generated from fixed seeds.  This gives us broad
geometry and topology variation without making the simulator-backed lane
flaky, difficult to reproduce, or unexpectedly expensive.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DT = 0.025
N_STEPS = 160
V_INIT = -67.0
CELSIUS = 32.0
DTYPE = torch.float64


@dataclass(frozen=True)
class _SectionSpec:
    name: str
    points: tuple[tuple[float, float, float, float], ...]
    nseg: int
    rhoa: float
    cm: float
    g_pas: float
    e_pas: float
    parent: str | None = None
    parent_x: float | None = None
    child_end: int = 0


@dataclass(frozen=True)
class _GeneratedCase:
    seed: int
    sections: tuple[_SectionSpec, ...]
    inject_section: str
    inject_segment: int
    amplitude_nA: float


def _oriented_points(junction, outward, connection_diam, distal_diam, child_end):
    """Author a cable so ``child_end`` is physically at ``junction``."""
    junction_point = (*junction, connection_diam)
    outward_point = (*outward, distal_diam)
    if child_end == 0:
        return (junction_point, outward_point)
    return (outward_point, junction_point)


def _point_at(points, x):
    start, end = points
    return tuple(start[axis] + x * (end[axis] - start[axis]) for axis in range(3))


def _generate_case(seed: int) -> _GeneratedCase:
    rng = random.Random(seed)

    def biophysical_values():
        return (
            rng.uniform(65.0, 185.0),
            rng.uniform(0.65, 1.65),
            rng.uniform(1.5e-4, 7.5e-4),
            rng.uniform(-78.0, -63.0),
        )

    root_nseg = rng.choice((3, 5))
    root_length = rng.uniform(24.0, 62.0)
    root_diameters = (rng.uniform(8.0, 17.0), rng.uniform(6.0, 14.0))
    root = _SectionSpec(
        "root",
        (
            (0.0, 0.0, 0.0, root_diameters[0]),
            (0.0, root_length, 0.0, root_diameters[1]),
        ),
        root_nseg,
        *biophysical_values(),
    )

    junction = (0.0, root_length, 0.0)
    branch_specs = []
    for name, direction in (("branch_a", 1.0), ("branch_b", -1.0)):
        child_end = rng.choice((0, 1))
        length = rng.uniform(38.0, 105.0)
        outward = (
            direction * rng.uniform(20.0, 55.0),
            root_length + length,
            rng.uniform(-14.0, 18.0),
        )
        branch_specs.append(
            _SectionSpec(
                name,
                _oriented_points(
                    junction,
                    outward,
                    rng.uniform(2.5, 6.0),
                    rng.uniform(0.8, 2.4),
                    child_end,
                ),
                rng.choice((1, 3, 5)),
                *biophysical_values(),
                parent="root",
                parent_x=1.0,
                child_end=child_end,
            )
        )

    branch_a = branch_specs[0]
    # A compartment-centre parent location has unambiguous host semantics in
    # both APIs and exercises an interior attachment without discretization
    # rounding at a segment boundary.
    host_segment = rng.randrange(branch_a.nseg)
    parent_x = (host_segment + 0.5) / branch_a.nseg
    twig_junction = _point_at(branch_a.points, parent_x)
    twig_child_end = rng.choice((0, 1))
    twig_outward = (
        twig_junction[0] + rng.uniform(16.0, 40.0),
        twig_junction[1] + rng.uniform(14.0, 42.0),
        twig_junction[2] + rng.uniform(8.0, 25.0),
    )
    twig = _SectionSpec(
        "twig",
        _oriented_points(
            twig_junction,
            twig_outward,
            rng.uniform(1.2, 3.0),
            rng.uniform(0.45, 1.1),
            twig_child_end,
        ),
        rng.choice((1, 3)),
        *biophysical_values(),
        parent="branch_a",
        parent_x=parent_x,
        child_end=twig_child_end,
    )

    sections = (root, *branch_specs, twig)
    injection_candidates = sections[1:]
    injection = rng.choice(injection_candidates)
    return _GeneratedCase(
        seed=seed,
        sections=sections,
        inject_section=injection.name,
        inject_segment=rng.randrange(injection.nseg),
        amplitude_nA=rng.uniform(0.12, 0.28),
    )


def _build_native(case: _GeneratedCase):
    morphology = dn.Morphology()
    sections = {
        spec.name: morphology.section(
            spec.name,
            points=spec.points,
            nseg=spec.nseg,
            rhoa=spec.rhoa,
            cm=spec.cm,
        )
        for spec in case.sections
    }
    for spec in case.sections:
        if spec.parent is not None:
            sections[spec.name].connect(
                sections[spec.parent].at(spec.parent_x),
                child_end=spec.child_end,
            )
    return morphology


def _build_neuron(case: _GeneratedCase):
    sections = {}
    for spec in case.sections:
        section = h.Section(name=f"generated_{case.seed}_{spec.name}")
        section.Ra = spec.rhoa
        section.cm = spec.cm
        section.nseg = spec.nseg
        h.pt3dclear(sec=section)

        # Native points always describe authored x=0 -> 1.  For child_end=1,
        # NEURON gives logical section x the opposite orientation from its raw
        # pt3d storage, so reversing storage preserves the authored convention.
        points = reversed(spec.points) if spec.child_end == 1 else spec.points
        for point in points:
            h.pt3dadd(*point, sec=section)
        section.insert("pas")
        for segment in section:
            segment.pas.g = spec.g_pas
            segment.pas.e = spec.e_pas
        sections[spec.name] = section

    for spec in case.sections:
        if spec.parent is not None:
            sections[spec.name].connect(
                sections[spec.parent](spec.parent_x), float(spec.child_end)
            )
    return sections


def _material_nodes(graph):
    return [
        node for node, kind in enumerate(graph.metadata.kind) if kind == "compartment"
    ]


def _node_for_section_segment(graph, section_name, segment_index):
    matches = [
        node
        for node in _material_nodes(graph)
        if graph.metadata.section_name[node] == section_name
        and graph.metadata.segment_index[node] == segment_index
    ]
    assert len(matches) == 1
    return matches[0]


def _neuron_segment_for_node(graph, sections, node):
    section_name = graph.metadata.section_name[node]
    segment_index = graph.metadata.segment_index[node]
    assert section_name is not None
    assert segment_index is not None
    return list(sections[section_name])[segment_index]


def _run_neuron(case, graph, sections, record_nodes, *, amplitude_nA):
    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = DT

    injection_node = _node_for_section_segment(
        graph, case.inject_section, case.inject_segment
    )
    clamp = h.IClamp(_neuron_segment_for_node(graph, sections, injection_node))
    clamp.delay = 0.5
    clamp.dur = 0.75
    clamp.amp = amplitude_nA

    recorded = [
        _neuron_segment_for_node(graph, sections, node) for node in record_nodes
    ]
    h.finitialize(V_INIT)
    trace = [[float(segment.v) for segment in recorded]]
    for _ in range(N_STEPS):
        h.fadvance()
        trace.append([float(segment.v) for segment in recorded])
    return np.asarray(trace)


def _run_dendra(case, morphology, record_nodes):
    graph = morphology.compile()
    model = dn.Tree.from_morphology(
        morphology,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    specs = {spec.name: spec for spec in case.sections}
    g_pas = torch.full((1, model.nc), 0.001, dtype=DTYPE)
    e_pas = torch.full((1, model.nc), -70.0, dtype=DTYPE)
    for node in _material_nodes(graph):
        spec = specs[graph.metadata.section_name[node]]
        g_pas[0, node] = spec.g_pas
        e_pas[0, node] = spec.e_pas
    model.insert(pas, g=g_pas, e=e_pas)

    injection_node = _node_for_section_segment(
        graph, case.inject_section, case.inject_segment
    )
    model[:, injection_node].inject(
        dn.mono_rect(
            amp=case.amplitude_nA * nA,
            delay=0.5,
            pw=0.75,
        )
    )
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=record_nodes)
    model.eval()
    model.initialize()
    model.run(
        tstop=N_STEPS * DT,
        dt=DT,
        callbacks=[recorder],
    )
    return recorder.numpy("v")[:, 0, :]


@pytest.mark.parametrize("seed", [1729, 314159, 8675309])
def test_generated_passive_native_tree_matches_neuron(seed):
    case = _generate_case(seed)
    morphology = _build_native(case)
    graph = morphology.compile()
    neuron_sections = _build_neuron(case)
    record_nodes = _material_nodes(graph)

    baseline = _run_neuron(
        case,
        graph,
        neuron_sections,
        record_nodes,
        amplitude_nA=0.0,
    )
    expected = _run_neuron(
        case,
        graph,
        neuron_sections,
        record_nodes,
        amplitude_nA=case.amplitude_nA,
    )
    actual = _run_dendra(case, morphology, record_nodes)

    assert actual.shape == expected.shape == (N_STEPS + 1, len(record_nodes))
    injection_node = _node_for_section_segment(
        graph, case.inject_section, case.inject_segment
    )
    injection_column = record_nodes.index(injection_node)
    stimulus_effect = np.max(
        np.abs(expected[:, injection_column] - baseline[:, injection_column])
    )
    assert stimulus_effect > 0.02, (
        f"Seed {seed} produced a trivial current-injection oracle: "
        f"{stimulus_effect:.6g} mV"
    )

    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    assert np.allclose(actual, expected, rtol=2.0e-6, atol=4.0e-4), (
        f"Generated native morphology seed {seed} mismatch: "
        f"max_abs={max_abs:.6g} mV, rmse={rmse:.6g} mV"
    )
