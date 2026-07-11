"""NEURON oracles for prescribed extracellular stimulation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mod import pas

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DT = 0.025
N_STEPS = 120
RESTING_VOLTAGE = -70.0
LEAK_CONDUCTANCE = 0.001
RHOA = 35.4
CM = 1.0
DTYPE = torch.float64


def _insert_neuron_membrane(sections):
    """Install the passive membrane and ideal prescribed-vext boundary."""
    for section in sections:
        section.insert("pas")
        section.insert("extracellular")
        for segment in section:
            segment.pas.g = LEAK_CONDUCTANCE
            segment.pas.e = RESTING_VOLTAGE
            for layer in range(int(h.nlayer_extracellular())):
                segment.xraxial[layer] = 1.0e9
                segment.xg[layer] = 1.0e9
                segment.xc[layer] = 0.0


def _new_stylized_section(name, *, length, diameter, nseg):
    section = h.Section(name=name)
    section.L = float(length)
    section.diam = float(diameter)
    section.Ra = RHOA
    section.cm = CM
    section.nseg = int(nseg)
    return section


def _set_3d_path(section, start, end):
    h.pt3dclear(sec=section)
    h.pt3dadd(*start, float(section.diam), sec=section)
    h.pt3dadd(*end, float(section.diam), sec=section)


def _initialize_dendra(model):
    model.insert(pas, g=LEAK_CONDUCTANCE, e=RESTING_VOLTAGE)
    model.eval()
    model.initialize()
    return model


def _branched_tree_case():
    """Build Dendra from the exact resistor tree used by NEURON."""
    soma = _new_stylized_section("ext_soma", length=100.0, diameter=3.0, nseg=5)
    dend_a = _new_stylized_section("ext_dend_a", length=60.0, diameter=2.0, nseg=3)
    dend_b = _new_stylized_section("ext_dend_b", length=60.0, diameter=1.5, nseg=3)

    _set_3d_path(soma, (0.0, 0.0, 0.0), (100.0, 0.0, 0.0))
    _set_3d_path(dend_a, (30.0, 0.0, 0.0), (90.0, 0.0, 0.0))
    _set_3d_path(dend_b, (70.0, 0.0, 0.0), (70.0, 60.0, 0.0))
    dend_a.connect(soma(0.3), 0.0)
    dend_b.connect(soma(0.7), 0.0)

    sections = (soma, dend_a, dend_b)
    graph, id_to_segment = neuron_to_dendra_graph(soma)
    assert set(id_to_segment) == set(graph)
    assert not any(
        str(attrs["name"]).startswith("branchpoint.")
        for _, attrs in graph.nodes(data=True)
    )

    model = dn.Tree.from_graph(
        graph,
        N=1,
        celsius=37.0,
        v_init=RESTING_VOLTAGE,
        dtype=DTYPE,
    )
    _initialize_dendra(model)
    _insert_neuron_membrane(sections)
    indices = list(range(model.nc))
    return model, id_to_segment, indices


def _axon_case(kind):
    if kind == "unmyelinated":
        model = dn.Unmyelinated(
            diameters=[2.0],
            L=100.0,
            dx=20.0,
            celsius=37.0,
            v_init=RESTING_VOLTAGE,
            rhoa=RHOA,
            cm=CM,
            dtype=DTYPE,
        )
    elif kind == "myelinated":
        model = dn.Myelinated(
            diameters=[10.0],
            n_node=5,
            node_length=2.0,
            celsius=37.0,
            v_init=RESTING_VOLTAGE,
            rhoa=RHOA,
            cm=CM,
            dtype=DTYPE,
        )
    else:  # pragma: no cover - guarded by the parametrization below
        raise ValueError(kind)

    _initialize_dendra(model)

    if kind == "unmyelinated":
        expected_geometry = {"dx": 20.0, "diam": 2.0, "rhoa": RHOA, "cm": CM}
        expected_x = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0], dtype=DTYPE)
    else:
        expected_geometry = {"dx": 2.0, "diam": 7.0, "rhoa": 17700.0, "cm": CM}
        expected_x = torch.tensor([-2000.0, -1000.0, 0.0, 1000.0, 2000.0], dtype=DTYPE)
    assert model.nc == 5
    torch.testing.assert_close(model.x[0], expected_x)
    for name, expected in expected_geometry.items():
        value = getattr(model, name)
        torch.testing.assert_close(value, torch.full_like(value, expected))

    # Myelinated is deliberately a reduced node-only cable: its short membrane
    # compartments retain physical node area while an effective Ra represents
    # the long, narrow internodal axial path. Reading these tensors after
    # initialize() is essential because in-graph parametrizations are now applied.
    section = _new_stylized_section(
        f"ext_{kind}",
        length=float(model.dx[0].sum()),
        diameter=float(model.diam[0, 0]),
        nseg=model.nc,
    )
    section.Ra = float(model.rhoa[0, 0])
    section.cm = float(model.cm[0, 0])
    _insert_neuron_membrane((section,))
    segments = list(section)
    assert len(segments) == model.nc
    return model, dict(enumerate(segments)), list(range(model.nc))


def _spatial_profile(model):
    """A nonuniform, gauge-centered extracellular potential in mV."""
    coordinate = model.x + 0.6 * model.y - 0.25 * model.z
    coordinate = coordinate - coordinate.mean(dim=-1, keepdim=True)
    magnitude = coordinate.abs().amax(dim=-1, keepdim=True)
    assert torch.all(magnitude > 0)
    return 8.0 * coordinate / magnitude


def _temporal_profile():
    """A smooth polarity-reversing prescribed-field scale."""
    time = torch.arange(N_STEPS, dtype=DTYPE) * DT
    return torch.sin(2.0 * torch.pi * 0.25 * time).unsqueeze(0)


def _run_neuron(node_to_segment, record_indices, spatial, temporal):
    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = 37.0
    h.dt = DT

    assignments = [
        (node_to_segment[node], float(spatial[0, node]))
        for node in range(spatial.shape[-1])
    ]
    recorded_segments = [node_to_segment[node] for node in record_indices]

    for segment, _ in assignments:
        segment.e_extracellular = 0.0
    h.finitialize(RESTING_VOLTAGE)
    trace = [[float(segment.v) for segment in recorded_segments]]

    for scale in temporal[0].tolist():
        for segment, field_value in assignments:
            segment.e_extracellular = field_value * scale
        h.fadvance()
        trace.append([float(segment.v) for segment in recorded_segments])
    return np.asarray(trace)


def _run_dendra(model, record_indices, spatial, temporal):
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=record_indices)
    model.run(
        tstop=N_STEPS * DT,
        dt=DT,
        extra=(spatial, temporal),
        callbacks=[recorder],
    )
    return recorder.numpy("v")[:, 0, :]


def _assert_simulators_close(actual, expected, *, case):
    assert actual.shape == expected.shape == (N_STEPS + 1, expected.shape[1])
    excursion = float(np.max(np.abs(expected - expected[0])))
    assert excursion > 0.05, f"{case} oracle response was trivially small"

    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    assert np.allclose(actual, expected, rtol=1.0e-7, atol=2.0e-4), (
        f"Extracellular {case} mismatch: max_abs={max_abs:.6g} mV, rmse={rmse:.6g} mV"
    )


@pytest.mark.parametrize("case", ["tree", "unmyelinated", "myelinated"])
def test_prescribed_extracellular_stimulation_matches_neuron(case):
    if case == "tree":
        model, node_to_segment, record_indices = _branched_tree_case()
    else:
        model, node_to_segment, record_indices = _axon_case(case)

    spatial = _spatial_profile(model)
    temporal = _temporal_profile()
    expected = _run_neuron(node_to_segment, record_indices, spatial, temporal)
    actual = _run_dendra(model, record_indices, spatial, temporal)

    _assert_simulators_close(actual, expected, case=case)
