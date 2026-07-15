# ruff: noqa: E402
"""Regression tests for NEURON -> Dendra morphology topology conversion."""

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import networkx as nx
import numpy as np
import pytest

neuron = pytest.importorskip("neuron")
from neuron import h

from dendra.models.io import neuron_to_dendra_graph, r_ohm

_CREATED = []
_COUNTER = 0


@pytest.fixture(autouse=True)
def _delete_created_sections_after_test():
    """Keep this unit module from leaking HOC sections into simtest oracles."""
    existing = {section.name() for section in h.allsec()}
    try:
        yield
    finally:
        for section in list(h.allsec()):
            if section.name() not in existing:
                h.delete_section(sec=section)
        _CREATED.clear()


def _section(prefix, *, L, diam, Ra, nseg):
    global _COUNTER
    _COUNTER += 1
    sec = h.Section(name=f"{prefix}_{_COUNTER}")
    sec.L = L
    sec.diam = diam
    sec.Ra = Ra
    sec.cm = 1.0
    sec.nseg = nseg
    _CREATED.append(sec)
    return sec


def _center(sec, idx):
    return sec((idx + 0.5) / int(sec.nseg))


def _node_for_segment(id2seg, sec, idx):
    target_x = (idx + 0.5) / int(sec.nseg)
    hits = [
        node
        for node, seg in id2seg.items()
        if seg.sec is sec and math.isclose(float(seg.x), target_x, abs_tol=1e-12)
    ]
    assert len(hits) == 1, (sec, idx, hits)
    return hits[0]


def _branchpoints(graph):
    return [
        node
        for node, attrs in graph.nodes(data=True)
        if "branchpoint" in str(attrs.get("name", "")).lower()
    ]


def _assert_valid_tree(graph):
    assert nx.is_arborescence(graph)
    assert graph.number_of_edges() == graph.number_of_nodes() - 1
    for _, _, attrs in graph.edges(data=True):
        resistance = float(attrs["R_ohm"])
        assert math.isfinite(resistance)
        assert 0.0 < resistance < 1e30
        assert math.isfinite(float(attrs["diff_geom_um"]))
        assert float(attrs["diff_geom_um"]) > 0.0


def test_root_x0_single_child_collapses_exact_series_resistance():
    root = _section("root_x0", L=20.0, diam=20.0, Ra=150.0, nseg=3)
    child = _section("child_x0", L=350.0, diam=2.5, Ra=150.0, nseg=5)
    child.connect(root(0.0), 0.0)

    graph, id2seg = neuron_to_dendra_graph(root)
    _assert_valid_tree(graph)
    assert not _branchpoints(graph)

    root_node = _node_for_segment(id2seg, root, 0)
    child_node = _node_for_segment(id2seg, child, 0)
    assert graph.has_edge(root_node, child_node)

    expected = (_center(root, 0).ri() + _center(child, 0).ri()) * 1e6
    assert graph.edges[root_node, child_node]["R_ohm"] == pytest.approx(
        expected, rel=1e-12
    )
    assert r_ohm(_center(root, 0), _center(child, 0)) == pytest.approx(
        expected, rel=1e-12
    )


def test_root_x0_single_compartment_does_not_use_endpoint_sentinel():
    root = _section("root_one", L=20.0, diam=20.0, Ra=150.0, nseg=1)
    child = _section("child_one", L=100.0, diam=2.0, Ra=150.0, nseg=1)
    child.connect(root(0.0), 0.0)

    graph, id2seg = neuron_to_dendra_graph(root)
    _assert_valid_tree(graph)
    root_node = _node_for_segment(id2seg, root, 0)
    child_node = _node_for_segment(id2seg, child, 0)
    expected = (_center(root, 0).ri() + _center(child, 0).ri()) * 1e6
    assert graph.edges[root_node, child_node]["R_ohm"] == pytest.approx(
        expected, rel=1e-12
    )


def test_multiple_children_at_endpoint_retain_shared_branchpoint():
    root = _section("root_multi", L=20.0, diam=20.0, Ra=150.0, nseg=3)
    child_a = _section("child_a", L=80.0, diam=2.0, Ra=120.0, nseg=3)
    child_b = _section("child_b", L=60.0, diam=3.0, Ra=110.0, nseg=3)
    child_a.connect(root(0.0), 0.0)
    child_b.connect(root(0.0), 0.0)

    graph, id2seg = neuron_to_dendra_graph(root)
    _assert_valid_tree(graph)
    branchpoints = _branchpoints(graph)
    assert len(branchpoints) == 1
    bp = branchpoints[0]
    assert graph.nodes[bp]["area"] == 0.0
    assert graph.nodes[bp]["volume"] == 0.0

    root_node = _node_for_segment(id2seg, root, 0)
    child_a_node = _node_for_segment(id2seg, child_a, 0)
    child_b_node = _node_for_segment(id2seg, child_b, 0)
    assert set(graph.to_undirected().neighbors(bp)) == {
        root_node,
        child_a_node,
        child_b_node,
    }
    assert graph.edges[root_node, bp]["R_ohm"] == pytest.approx(
        _center(root, 0).ri() * 1e6, rel=1e-12
    )
    assert graph.edges[bp, child_a_node]["R_ohm"] == pytest.approx(
        _center(child_a, 0).ri() * 1e6, rel=1e-12
    )
    assert graph.edges[bp, child_b_node]["R_ohm"] == pytest.approx(
        _center(child_b, 0).ri() * 1e6, rel=1e-12
    )


def test_reversed_child_orientation_is_oriented_away_from_root():
    root = _section("root_reverse", L=20.0, diam=10.0, Ra=100.0, nseg=1)
    child = _section("child_reverse", L=100.0, diam=2.0, Ra=120.0, nseg=5)
    child.connect(root(1.0), 1.0)

    graph, id2seg = neuron_to_dendra_graph(root)
    _assert_valid_tree(graph)
    root_node = _node_for_segment(id2seg, root, 0)
    child_distal_idx = int(child.nseg) - 1
    child_node = _node_for_segment(id2seg, child, child_distal_idx)
    assert graph.has_edge(root_node, child_node)

    expected = (root(1.0).ri() + _center(child, child_distal_idx).ri()) * 1e6
    assert graph.edges[root_node, child_node]["R_ohm"] == pytest.approx(
        expected, rel=1e-12
    )

    # Away from the parent-facing x=1 end, graph direction must descend in x.
    for idx in range(child_distal_idx, 0, -1):
        near = _node_for_segment(id2seg, child, idx)
        far = _node_for_segment(id2seg, child, idx - 1)
        assert graph.has_edge(near, far)


def test_reversed_taper_geometry_tracks_neuron_logical_orientation():
    """Electrical and extracted pt3d geometry must describe the same path."""
    root = _section("root_reverse_taper", L=20.0, diam=10.0, Ra=100.0, nseg=1)
    child = h.Section(name="child_reverse_taper")
    child.pt3dclear()
    child.pt3dadd(0.0, 0.0, 0.0, 1.0)
    child.pt3dadd(100.0, 0.0, 0.0, 4.0)
    child.Ra = 120.0
    child.cm = 1.0
    child.nseg = 2
    child.connect(root(1.0), 1.0)
    _CREATED.append(child)

    graph, id2seg = neuron_to_dendra_graph(root)
    root_node = _node_for_segment(id2seg, root, 0)
    child_node = _node_for_segment(id2seg, child, 1)

    # With orientation=1, logical x runs opposite stored pt3d arclength.
    # NEURON's ri() already follows logical x; Dendra's diffusion geometry
    # must reverse raw pt3d integration to describe that same cable path.
    expected = (root(1.0).ri() + child(0.75).ri()) * 1e6
    actual = graph.edges[root_node, child_node]["R_ohm"]
    assert actual == pytest.approx(expected, rel=1e-12)
    inverse_area = 1.0 / graph.edges[root_node, child_node]["diff_geom_um"]
    parent_inverse_area = root(1.0).ri() * 1e6 / (float(root.Ra) * 1e4)
    child_inverse_area = child(0.75).ri() * 1e6 / (float(child.Ra) * 1e4)
    assert inverse_area == pytest.approx(
        parent_inverse_area + child_inverse_area, rel=3e-7
    )

    child_attrs = graph.nodes[child_node]
    assert child_attrs["x"] == pytest.approx(25.0, rel=1e-12)
    # The logical x=0.75 segment occupies raw pt3d arclength [0, 0.5].
    expected_volume = math.pi * 50.0 * (0.5**2 + 0.5 * 1.25 + 1.25**2) / 3.0
    assert child_attrs["volume"] == pytest.approx(expected_volume, rel=1e-12)


def test_nonroot_proximal_endpoint_branch_has_one_physical_junction():
    root = _section("root_nonroot", L=20.0, diam=20.0, Ra=150.0, nseg=3)
    trunk = _section("trunk_nonroot", L=100.0, diam=4.0, Ra=100.0, nseg=3)
    branch = _section("branch_nonroot", L=60.0, diam=2.0, Ra=120.0, nseg=3)
    trunk.connect(root(1.0), 0.0)
    branch.connect(trunk(0.0), 0.0)

    graph, id2seg = neuron_to_dendra_graph(root)
    _assert_valid_tree(graph)
    branchpoints = _branchpoints(graph)
    assert len(branchpoints) == 1
    bp = branchpoints[0]

    expected_neighbors = {
        _node_for_segment(id2seg, root, int(root.nseg) - 1),
        _node_for_segment(id2seg, trunk, 0),
        _node_for_segment(id2seg, branch, 0),
    }
    assert set(graph.to_undirected().neighbors(bp)) == expected_neighbors


def test_interior_parent_connection_uses_containing_compartment_node():
    root = _section("root_interior", L=30.0, diam=3.0, Ra=100.0, nseg=3)
    child = _section("child_interior", L=30.0, diam=2.0, Ra=120.0, nseg=3)
    child.connect(root(0.4), 0.0)

    graph, id2seg = neuron_to_dendra_graph(root)
    _assert_valid_tree(graph)
    assert not _branchpoints(graph)

    parent_node = _node_for_segment(id2seg, root, 1)  # root(0.5)
    child_node = _node_for_segment(id2seg, child, 0)
    assert graph.has_edge(parent_node, child_node)
    # NEURON attaches an interior connection to the containing segment node;
    # only the child's proximal half segment contributes to ri().
    assert graph.edges[parent_node, child_node]["R_ohm"] == pytest.approx(
        _center(child, 0).ri() * 1e6, rel=1e-12
    )
    assert graph.edges[parent_node, child_node]["L"] == pytest.approx(
        float(child.L) / (2.0 * int(child.nseg)), rel=1e-12
    )


def test_from_neuron_builds_tree_population_without_invalid_edges():
    from dendra.models.tree import Tree

    root = _section("soma_tree", L=20.0, diam=20.0, Ra=150.0, nseg=3)
    child = _section("dend_tree", L=350.0, diam=2.5, Ra=150.0, nseg=5)
    child.connect(root(0.0), 0.0)

    tree = Tree.from_NEURON(root, N=2)
    assert tuple(tree.shape) == (2, 8)
    _assert_valid_tree(tree.graph)
    assert not _branchpoints(tree.graph)


def test_passive_dc_solution_matches_neuron_for_endpoint_junctions():
    """The imported resistor graph should reproduce NEURON to roundoff."""
    h.load_file("stdrun.hoc")

    root = _section("soma_dc", L=30.0, diam=12.0, Ra=130.0, nseg=3)
    child_a = _section("dend_dc_a", L=120.0, diam=2.0, Ra=90.0, nseg=5)
    child_b = _section("dend_dc_b", L=90.0, diam=3.0, Ra=170.0, nseg=3)
    child_c = _section("dend_dc_c", L=75.0, diam=2.5, Ra=110.0, nseg=3)
    child_a.connect(root(0.0), 0.0)
    child_b.connect(root(0.0), 1.0)  # reversed child orientation
    child_c.connect(root(1.0), 0.0)

    sections = [root, child_a, child_b, child_c]
    g_pas = 1.0e-4
    e_pas = -65.0
    for sec in sections:
        sec.insert("pas")
        for seg in sec:
            seg.pas.g = g_pas
            seg.pas.e = e_pas

    graph, id2seg = neuron_to_dendra_graph(root)
    _assert_valid_tree(graph)

    n = graph.number_of_nodes()
    conductance = np.zeros((n, n), dtype=float)
    current = np.zeros(n, dtype=float)
    for node in range(n):
        area_cm2 = float(graph.nodes[node].get("area", 0.0)) * 1.0e-8
        conductance[node, node] += g_pas * area_cm2
    for parent, child, attrs in graph.edges(data=True):
        g_axial = 1.0 / float(attrs["R_ohm"])
        conductance[parent, parent] += g_axial
        conductance[child, child] += g_axial
        conductance[parent, child] -= g_axial
        conductance[child, parent] -= g_axial

    inject_node = _node_for_segment(id2seg, root, 1)
    current[inject_node] = 0.1e-9  # 0.1 nA in amperes
    predicted_mV = e_pas + np.linalg.solve(conductance, current) * 1.0e3

    clamp = h.IClamp(root(0.5))
    clamp.delay = 0.0
    clamp.dur = 1.0e9
    clamp.amp = 0.1
    h.dt = 0.025
    h.finitialize(e_pas)
    h.continuerun(2000.0)

    errors = []
    for node, seg in id2seg.items():
        if "branchpoint" in str(graph.nodes[node].get("name", "")):
            continue
        errors.append(abs(float(seg.v) - float(predicted_mV[node])))
    assert max(errors) < 1.0e-7
