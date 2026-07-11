"""Holistic NEURON oracle for a heterogeneous, tapered resistor tree."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mod import pas
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DT = 0.025
N_STEPS = 240
V_INIT = -68.0
CELSIUS = 34.0
DTYPE = torch.float64


def _section(name, *, points, ra, cm, nseg):
    section = h.Section(name=name)
    section.Ra = float(ra)
    section.cm = float(cm)
    section.nseg = int(nseg)
    h.pt3dclear(sec=section)
    for x, y, z, diameter in points:
        h.pt3dadd(x, y, z, diameter, sec=section)
    return section


def _build_neuron_tree():
    """Create a deliberately nonuniform tree without using Dendra geometry."""
    soma = _section(
        "soma_tapered",
        points=[(-15.0, 0.0, 0.0, 18.0), (0.0, 0.0, 0.0, 15.0), (15.0, 0.0, 0.0, 12.0)],
        ra=82.0,
        cm=0.75,
        nseg=3,
    )
    trunk = _section(
        "apic_tapered",
        points=[(0.0, 0.0, 0.0, 5.5), (4.0, 55.0, 3.0, 3.2), (0.0, 120.0, 8.0, 1.7)],
        ra=143.0,
        cm=1.25,
        nseg=5,
    )
    branch_a = _section(
        "dend_tapered_a",
        points=[(0.0, 36.0, 2.0, 4.2), (35.0, 63.0, 9.0, 2.4), (82.0, 91.0, 14.0, 0.9)],
        ra=108.0,
        cm=1.55,
        nseg=5,
    )
    branch_b = _section(
        "dend_tapered_b",
        points=[
            (2.0, 84.0, 6.0, 3.6),
            (-31.0, 112.0, 2.0, 2.0),
            (-72.0, 139.0, -7.0, 1.1),
        ],
        ra=191.0,
        cm=0.92,
        nseg=5,
    )
    twig = _section(
        "dend_tapered_twig",
        points=[
            (38.0, 64.0, 9.0, 2.2),
            (61.0, 45.0, 15.0, 1.4),
            (91.0, 31.0, 20.0, 0.65),
        ],
        ra=69.0,
        cm=1.8,
        nseg=3,
    )

    # Interior connections coincide with compartment centres.  This exercises
    # a genuinely branched resistor tree while avoiding artificial zero-area
    # junction compartments in either simulator.
    trunk.connect(soma(0.5), 0.0)
    branch_a.connect(trunk(0.3), 0.0)
    branch_b.connect(trunk(0.7), 0.0)
    twig.connect(branch_a(0.5), 0.0)
    sections = (soma, trunk, branch_a, branch_b, twig)

    for section_index, section in enumerate(sections):
        section.insert("pas")
        section.insert("extracellular")
        for segment_index, segment in enumerate(section):
            # Segment-level variation makes cm and the passive membrane
            # heterogeneous even within each already-distinct section.
            position = segment_index / max(1, int(section.nseg) - 1)
            segment.cm = float(section.cm) * (0.88 + 0.24 * position)
            segment.pas.g = 2.5e-4 + 7.5e-5 * section_index + 4.0e-5 * segment_index
            segment.pas.e = -75.0 + 1.7 * section_index + 0.55 * segment_index
            for layer in range(int(h.nlayer_extracellular())):
                segment.xraxial[layer] = 1.0e9
                segment.xg[layer] = 1.0e9
                segment.xc[layer] = 0.0

    return sections


def _node_for_segment(node_to_segment, section, segment_index):
    target = list(section)[segment_index]
    matches = [
        node
        for node, segment in node_to_segment.items()
        if segment.sec == target.sec
        and float(segment.x) == pytest.approx(float(target.x))
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_nonuniform_oracle(graph, model):
    assert any(degree >= 2 for _, degree in graph.out_degree())
    assert not any(
        str(attrs.get("name", "")).startswith("branchpoint")
        for _, attrs in graph.nodes(data=True)
    )

    material = model.internal_nodes.index_spec.index
    diameters = model.diam[material]
    rhoa = model.rhoa[material]
    cm = model.cm[material]
    assert torch.unique(diameters).numel() >= 10
    assert float(diameters.max() / diameters.min()) > 10.0
    assert torch.unique(rhoa).numel() == 5
    assert torch.unique(cm).numel() >= 15
    assert float(cm.max() / cm.min()) > 2.0


def _spatial_profile(model):
    coordinate = 0.7 * model.x - 0.45 * model.y + 0.3 * model.z
    coordinate = coordinate - coordinate.mean(dim=-1, keepdim=True)
    scale = coordinate.abs().amax(dim=-1, keepdim=True)
    assert torch.all(scale > 0.0)
    return 7.5 * coordinate / scale


def _temporal_profile():
    time = torch.arange(N_STEPS, dtype=DTYPE) * DT
    profile = 0.65 * torch.sin(2.0 * torch.pi * 0.31 * time)
    profile += 0.35 * torch.sin(2.0 * torch.pi * 0.83 * time + 0.4)
    return profile.unsqueeze(0)


def _run_neuron(
    node_to_segment,
    record_indices,
    *,
    inject_segment,
    spatial,
    temporal,
    drive,
):
    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = DT

    clamp = h.IClamp(inject_segment)
    clamp.delay = 0.5
    clamp.dur = 0.75
    clamp.amp = 0.18 if drive == "intracellular" else 0.0

    assignments = [
        (node_to_segment[node], float(spatial[0, node]))
        for node in range(spatial.shape[-1])
    ]
    recorded = [node_to_segment[node] for node in record_indices]
    for segment, _ in assignments:
        segment.e_extracellular = 0.0

    h.finitialize(V_INIT)
    trace = [[float(segment.v) for segment in recorded]]
    for scale in temporal[0].tolist():
        for segment, field_value in assignments:
            segment.e_extracellular = (
                field_value * scale if drive == "extracellular" else 0.0
            )
        h.fadvance()
        trace.append([float(segment.v) for segment in recorded])
    return np.asarray(trace)


def _run_dendra(
    graph,
    node_to_segment,
    record_indices,
    *,
    inject_node,
    spatial,
    temporal,
    drive,
):
    model = dn.Tree.from_graph(
        graph,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    g = torch.tensor(
        [[float(node_to_segment[node].pas.g) for node in range(model.nc)]],
        dtype=DTYPE,
    )
    e = torch.tensor(
        [[float(node_to_segment[node].pas.e) for node in range(model.nc)]],
        dtype=DTYPE,
    )
    model.insert(pas, g=g, e=e)
    if drive == "intracellular":
        model[:, inject_node].inject(dn.mono_rect(amp=0.18 * nA, delay=0.5, pw=0.75))

    recorder = dn.callbacks.Recorder(states=["v"], node_indices=record_indices)
    model.eval()
    model.initialize()
    run_kwargs = {}
    if drive == "extracellular":
        run_kwargs["extra"] = (spatial, temporal)
    model.run(
        tstop=N_STEPS * DT,
        dt=DT,
        callbacks=[recorder],
        **run_kwargs,
    )
    return model, recorder.numpy("v")[:, 0, :]


@pytest.mark.parametrize("drive", ["intracellular", "extracellular"])
def test_tapered_heterogeneous_branched_tree_matches_neuron(drive):
    sections = _build_neuron_tree()
    soma, trunk, branch_a, branch_b, twig = sections
    graph, node_to_segment = neuron_to_dendra_graph(soma)

    inject_node = _node_for_segment(node_to_segment, soma, 1)
    record_indices = [
        inject_node,
        _node_for_segment(node_to_segment, trunk, 4),
        _node_for_segment(node_to_segment, branch_b, 4),
        _node_for_segment(node_to_segment, twig, 2),
    ]
    spatial = _spatial_profile(
        dn.Tree.from_graph(graph, N=1, dtype=DTYPE, v_init=V_INIT)
    )
    temporal = _temporal_profile()

    baseline = _run_neuron(
        node_to_segment,
        record_indices,
        inject_segment=node_to_segment[inject_node],
        spatial=spatial,
        temporal=temporal,
        drive="baseline",
    )
    expected = _run_neuron(
        node_to_segment,
        record_indices,
        inject_segment=node_to_segment[inject_node],
        spatial=spatial,
        temporal=temporal,
        drive=drive,
    )
    model, actual = _run_dendra(
        graph,
        node_to_segment,
        record_indices,
        inject_node=inject_node,
        spatial=spatial,
        temporal=temporal,
        drive=drive,
    )

    _assert_nonuniform_oracle(graph, model)
    assert actual.shape == expected.shape == (N_STEPS + 1, len(record_indices))
    excursion = np.max(np.abs(expected - expected[0]), axis=0)
    assert np.all(excursion > 0.02), (
        f"{drive} oracle response was trivial at one or more sites: {excursion!r}"
    )
    drive_effect = np.max(np.abs(expected - baseline), axis=0)
    assert np.all(drive_effect > 0.02), (
        f"{drive} did not materially affect every strategic recording site: "
        f"{drive_effect!r}"
    )

    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    assert np.allclose(actual, expected, rtol=2.0e-6, atol=3.0e-4), (
        f"Tapered heterogeneous Tree ({drive}) mismatch: "
        f"max_abs={max_abs:.6g} mV, rmse={rmse:.6g} mV"
    )
