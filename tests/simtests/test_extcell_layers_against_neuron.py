"""NEURON oracles for Dendra's finite two-layer extracellular models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.extcell import ExtCellAxon, ExtCellTree
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mod import pas

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

# NEURON 9's Python wrapper aliases indexed ``vext`` RANGE references to
# ``vext[0]``.  HOC access remains layer-correct, so sample through these tiny
# helpers until the upstream wrapper exposes distinct handles again.
h("func dendra_extcell_test_vext0() { return vext[0]($1) }")
h("func dendra_extcell_test_vext1() { return vext[1]($1) }")

DTYPE = torch.float64
DT = 0.025
N_STEPS = 160
V_INIT = -70.0
E_PAS = -70.0
G_PAS = 8.0e-4  # S/cm2
RHOA = 90.0  # ohm cm
CM = 1.1  # uF/cm2

# NEURON's extracellular RANGE units are MOhm/cm, uF/cm2, and S/cm2.
# The large but finite longitudinal resistances make both extracellular
# voltages spatially nonuniform without reducing either layer to an ideal
# prescribed-voltage boundary.
XRAXIAL = (8.0e4, 1.3e5)
XC = (0.18, 0.07)
XG = (3.5e-4, 6.5e-4)


def _new_section(name, *, length, diameter, nseg):
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


def _configure_neuron_membrane(sections):
    assert int(h.nlayer_extracellular()) == 2
    for section in sections:
        section.insert("pas")
        section.insert("extracellular")
        for segment in section:
            segment.pas.g = G_PAS
            segment.pas.e = E_PAS
            for layer in range(2):
                segment.xraxial[layer] = XRAXIAL[layer]
                segment.xc[layer] = XC[layer]
                segment.xg[layer] = XG[layer]


def _bath_timecourse():
    time = torch.arange(N_STEPS, dtype=DTYPE) * DT
    envelope = ((time >= 0.25) & (time < 3.25)).to(DTYPE)
    carrier = 0.7 + 0.3 * torch.sin(2.0 * torch.pi * 0.55 * time)
    return (envelope * carrier).unsqueeze(0)


def _gauge_centered_profile(coordinate):
    coordinate = coordinate.to(dtype=DTYPE)
    centered = coordinate - coordinate.mean(dim=-1, keepdim=True)
    scale = centered.abs().amax(dim=-1, keepdim=True)
    assert torch.all(scale > 0.0)
    profile = 12.0 * centered / scale
    assert torch.any(profile > 0.0)
    assert torch.any(profile < 0.0)
    return profile


def _run_neuron(node_to_segment, record_nodes, spatial, temporal, *, bath_nodes=None):
    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = 34.0
    h.dt = DT

    if bath_nodes is None:
        bath_nodes = range(spatial.shape[-1])
    assignments = [
        (node_to_segment[node], float(spatial[0, node]))
        for node in bath_nodes
        if node in node_to_segment
    ]
    recorded = [node_to_segment[node] for node in record_nodes]
    for segment, _ in assignments:
        segment.e_extracellular = 0.0

    h.finitialize(V_INIT)

    def sample():
        return [
            [float(segment.v) for segment in recorded],
            [
                float(h.dendra_extcell_test_vext0(segment.x, sec=segment.sec))
                for segment in recorded
            ],
            [
                float(h.dendra_extcell_test_vext1(segment.x, sec=segment.sec))
                for segment in recorded
            ],
            [
                # NEURON exposes a current density (mA/cm2); Dendra's public
                # ``i_membrane`` is absolute current in mA.
                float(segment.i_membrane) * float(segment.area()) * 1.0e-8
                for segment in recorded
            ],
        ]

    trace = [sample()]
    for scale in temporal[0].tolist():
        for segment, bath in assignments:
            segment.e_extracellular = bath * scale
        h.fadvance()
        trace.append(sample())
    values = np.asarray(trace)
    return {
        "v": values[:, 0, :],
        "vext0": values[:, 1, :],
        "vext1": values[:, 2, :],
        "imem": values[:, 3, :],
    }


def _run_dendra(model, record_nodes, spatial, temporal):
    node_index = torch.as_tensor(record_nodes, dtype=torch.long)
    recorder = dn.callbacks.RecorderLambda(
        {
            "v": lambda current: current.v.index_select(-1, node_index),
            "vext0": lambda current: current.vc[..., 1].index_select(-1, node_index),
            "vext1": lambda current: current.vc[..., 2].index_select(-1, node_index),
            "imem": lambda current: current.i_membrane.index_select(-1, node_index),
        }
    )
    model.run(
        tstop=N_STEPS * DT,
        dt=DT,
        extra=(spatial, temporal),
        callbacks=[recorder],
    )
    return {
        name: recorder.numpy(name).reshape(N_STEPS + 1, len(record_nodes))
        for name in ("v", "vext0", "vext1", "imem")
    }


def _assert_close(actual, expected, *, case):
    assert set(actual) == set(expected) == {"v", "vext0", "vext1", "imem"}
    for name in actual:
        assert actual[name].shape == expected[name].shape
        assert np.isfinite(actual[name]).all()
        assert np.isfinite(expected[name]).all()

        expected_excursion = float(np.max(np.abs(expected[name] - expected[name][0])))
        if name == "v":
            minimum_excursion = 0.01
        elif name == "imem":
            minimum_excursion = 1.0e-12
        else:
            minimum_excursion = 0.1
        assert expected_excursion > minimum_excursion, (
            f"{case} {name} NEURON response was trivial: {expected_excursion:.6g}"
        )

        difference = actual[name] - expected[name]
        max_abs = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(difference * difference)))
        atol = 2.0e-14 if name == "imem" else 3.0e-4
        assert np.allclose(actual[name], expected[name], rtol=2.0e-6, atol=atol), (
            f"Two-layer extracellular {case} {name} mismatch: "
            f"max_abs={max_abs:.6g}, rmse={rmse:.6g}"
        )

    assert float(np.max(np.abs(expected["vext0"] - expected["vext1"]))) > 0.1, (
        f"{case} oracle did not resolve two distinct extracellular layers"
    )


def test_extcell_axon_two_finite_layers_match_neuron():
    n_comp = 7
    dx = 20.0
    diameter = 6.0
    section = _new_section(
        "extcell_axon", length=n_comp * dx, diameter=diameter, nseg=n_comp
    )
    _configure_neuron_membrane((section,))
    segments = list(section)
    node_to_segment = dict(enumerate(segments))

    with dn.ctx(IMEM=1):
        model = ExtCellAxon(
            diameters=[diameter],
            n_comp=n_comp,
            celsius=34.0,
            v_init=V_INIT,
            dtype=DTYPE,
            rhoa=RHOA,
            cm=CM,
        )
    model.dx.fill_(dx)
    model.x.copy_(model._x())
    for layer in range(2):
        model.xraxial[..., layer].fill_(XRAXIAL[layer])
        model.xc[..., layer].fill_(XC[layer])
        model.xg[..., layer].fill_(XG[layer])
    model.insert(pas, g=G_PAS, e=E_PAS)
    model.eval()
    model.initialize()

    spatial = _gauge_centered_profile(model.x)
    temporal = _bath_timecourse()
    record_nodes = [0, n_comp // 2, n_comp - 1]
    expected = _run_neuron(node_to_segment, record_nodes, spatial, temporal)
    actual = _run_dendra(model, record_nodes, spatial, temporal)

    torch.testing.assert_close(model.v, model.vc[..., 0] - model.vc[..., 1])
    _assert_close(actual, expected, case="axon")


def test_extcell_branched_tree_two_finite_layers_match_neuron():
    trunk = _new_section("extcell_trunk", length=60.0, diameter=8.0, nseg=3)
    branch_a = _new_section("extcell_branch_a", length=80.0, diameter=5.0, nseg=3)
    branch_b = _new_section("extcell_branch_b", length=100.0, diameter=3.5, nseg=3)
    branch_a.connect(trunk(1.0), 0.0)
    branch_b.connect(trunk(1.0), 0.0)
    _set_3d_path(trunk, (0.0, 0.0, 0.0), (60.0, 0.0, 0.0))
    _set_3d_path(branch_a, (60.0, 0.0, 0.0), (60.0, 80.0, 0.0))
    _set_3d_path(branch_b, (60.0, 0.0, 0.0), (60.0, -100.0, 0.0))
    sections = (trunk, branch_a, branch_b)
    _configure_neuron_membrane(sections)

    graph, node_to_segment = neuron_to_dendra_graph(trunk, extcell=2)
    with dn.ctx(IMEM=1):
        model = ExtCellTree.from_graph(
            graph,
            N=1,
            celsius=34.0,
            v_init=V_INIT,
            dtype=DTYPE,
        )
    model.insert(pas, g=G_PAS, e=E_PAS)
    model.eval()
    model.initialize()

    branchpoints = [
        node
        for node, attrs in graph.nodes(data=True)
        if str(attrs["name"]).startswith("branchpoint.")
    ]
    assert len(branchpoints) == 1
    assert float(graph.nodes[branchpoints[0]]["area"]) == 0.0

    def node_for(section, segment_index):
        target = list(section)[segment_index]
        return next(
            node
            for node, segment in node_to_segment.items()
            if node not in branchpoints
            and segment.sec == target.sec
            and abs(float(segment.x) - float(target.x)) < 1.0e-12
        )

    record_nodes = [
        node_for(trunk, 0),
        node_for(branch_a, -1),
        node_for(branch_b, -1),
    ]
    assert len(record_nodes) == 3

    coordinate = model.x + 0.45 * model.y - 0.2 * model.z
    if not torch.any(coordinate != coordinate[..., :1]):
        coordinate = torch.arange(model.nc, dtype=DTYPE).unsqueeze(0)
    spatial = _gauge_centered_profile(coordinate)
    temporal = _bath_timecourse()
    expected = _run_neuron(
        node_to_segment,
        record_nodes,
        spatial,
        temporal,
        # The retained branchpoint represents a zero-area circuit junction,
        # not a membrane compartment with a radial bath battery. Assigning a
        # RANGE value through its endpoint proxy would overwrite an adjacent
        # NEURON segment even though the corresponding Dendra radial terms are
        # correctly zero.
        bath_nodes=[node for node in graph if node not in branchpoints],
    )
    actual = _run_dendra(model, record_nodes, spatial, temporal)

    torch.testing.assert_close(model.v, model.vc[..., 0] - model.vc[..., 1])
    _assert_close(actual, expected, case="branched tree")
