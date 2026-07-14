import math
from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.morphology import CompartmentGraph, Morphology

pytestmark = pytest.mark.cpu


def test_stylized_section_compiles_deterministic_binary64_cylinder_geometry():
    morphology = Morphology(rhoa=100.0, cm=1.25)
    morphology.section("soma", L=30.0, diam=2.0, nseg=3, labels="cell_body")

    graph = morphology.compile()

    assert graph == morphology.compile()
    assert graph.topology.parent_index == (-1, 0, 1)
    assert graph.geometry.length_um == (10.0, 10.0, 10.0)
    assert graph.geometry.diameter_um == (2.0, 2.0, 2.0)
    assert graph.geometry.z_um == (5.0, 15.0, 25.0)
    assert graph.geometry.area_um2 == pytest.approx((20 * math.pi,) * 3)
    assert graph.geometry.volume_um3 == pytest.approx((10 * math.pi,) * 3)
    assert graph.geometry.volume_i_um3 == graph.geometry.volume_um3
    assert graph.geometry.volume_o_um3 == (0.0, 0.0, 0.0)
    assert graph.geometry.edge_length_um == pytest.approx((0.0, 10.0, 10.0))
    assert graph.geometry.edge_resistance_ohm[1:] == pytest.approx(
        (1e7 / math.pi, 1e7 / math.pi)
    )
    assert graph.geometry.edge_diff_geom_um[1:] == pytest.approx(
        (math.pi / 10, math.pi / 10)
    )
    assert graph.geometry.array("volume_um3").dtype == np.float64
    assert graph.nodes_with_label("soma") == (0, 1, 2)
    assert graph.nodes_with_label("cell_body") == (0, 1, 2)


def test_pt3d_section_uses_frustum_geometry_and_centerline_coordinates():
    morphology = Morphology()
    morphology.section(
        "apic",
        points=[(0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0)],
        nseg=1,
    )

    graph = morphology.compile()

    assert graph.geometry.length_um == (10.0,)
    assert graph.geometry.diameter_um == (3.0,)
    assert graph.geometry.z_um == (5.0,)
    assert graph.geometry.area_um2[0] == pytest.approx(
        math.pi * 3.0 * math.hypot(10.0, 1.0)
    )
    assert graph.geometry.volume_um3[0] == pytest.approx(
        math.pi * 10.0 * (1.0 + 2.0 + 4.0) / 3.0
    )


def test_endpoint_branch_retains_unlabelled_zero_volume_junction():
    morphology = Morphology()
    soma = morphology.section("soma", L=10.0, diam=10.0)
    dend = morphology.section("dend", L=20.0, diam=2.0)
    apic = morphology.section("apic", L=30.0, diam=3.0)
    dend.connect(soma.at(1.0), child_end=0)
    morphology.connect(soma.at(1.0), apic.at(0.0))

    graph = morphology.compile()

    assert graph.topology.parent_index == (-1, 0, 1, 1)
    junction = graph.metadata.kind.index("junction")
    assert junction == 1
    assert graph.metadata.name[junction] == "branchpoint.0"
    assert graph.geometry.length_um[junction] == 0.0
    assert graph.geometry.area_um2[junction] == 0.0
    assert graph.geometry.volume_um3[junction] == 0.0
    assert graph.metadata.labels[junction] == frozenset()
    assert graph.nodes_with_label("soma") == (0,)
    assert graph.nodes_with_label("dend") == (2,)
    assert graph.nodes_with_label("apic") == (3,)
    assert set(graph.topology.edges) == {(0, 1), (1, 2), (1, 3)}


def test_degree_two_endpoint_junction_is_collapsed_as_series_resistance():
    morphology = Morphology(rhoa=80.0)
    soma = morphology.section("soma", L=10.0, diam=4.0)
    dend = morphology.section("dend", L=20.0, diam=2.0)
    dend.connect(soma.at(1.0), child_end=0)

    graph = morphology.compile()

    assert graph.topology.parent_index == (-1, 0)
    assert graph.metadata.kind == ("compartment", "compartment")
    assert graph.geometry.edge_length_um == (0.0, 15.0)
    expected_inv_area = 5.0 / (4.0 * math.pi) + 10.0 / math.pi
    assert graph.geometry.edge_resistance_ohm[1] == pytest.approx(
        80.0 * 1e4 * expected_inv_area
    )
    assert graph.geometry.edge_diff_geom_um[1] == pytest.approx(1 / expected_inv_area)


def test_interior_connection_attaches_to_containing_parent_compartment():
    morphology = Morphology(rhoa=100.0)
    trunk = morphology.section("trunk", L=30.0, diam=2.0, nseg=3)
    branch = morphology.section("branch", L=10.0, diam=2.0)
    branch.connect(trunk.at(0.5), child_end=0)

    graph = morphology.compile()

    assert graph.topology.parent_index == (-1, 0, 1, 1)
    branch_node = graph.metadata.section_name.index("branch")
    assert graph.topology.parent_index[branch_node] == 1
    # NEURON-compatible interior attachment contributes no parent half segment:
    # the child endpoint is identified directly with the containing parent node.
    assert graph.geometry.edge_length_um[branch_node] == 5.0
    assert graph.geometry.edge_resistance_ohm[branch_node] == pytest.approx(
        100.0 * 1e4 * 5.0 / math.pi
    )


def test_child_end_one_reverses_topological_traversal_without_losing_provenance():
    morphology = Morphology()
    soma = morphology.section("soma", L=10.0, diam=4.0)
    axon = morphology.section("axon", L=20.0, diam=2.0, nseg=2)
    axon.connect(soma.at(1.0), child_end=1)

    graph = morphology.compile()

    assert graph.metadata.section_name == ("soma", "axon", "axon")
    assert graph.metadata.segment_index == (0, 1, 0)
    assert graph.metadata.section_x == (0.5, 0.75, 0.25)
    assert graph.topology.parent_index == (-1, 0, 1)


def test_networkx_adapter_round_trips_canonical_data_and_accepts_other_node_ids():
    morphology = Morphology(rhoa=120.0, cm=0.8)
    soma = morphology.section("soma", L=12.0, diam=8.0)
    dend = morphology.section("dend", L=24.0, diam=2.0, nseg=2)
    dend.connect(soma.at(1.0), child_end=0)
    canonical = morphology.compile()

    nx_graph = canonical.to_networkx()
    relabelled = nx.relabel_nodes(nx_graph, {0: "root", 1: "mid", 2: "tip"})
    restored = CompartmentGraph.from_networkx(relabelled)

    assert restored == canonical
    assert list(restored.to_networkx().nodes) == [0, 1, 2]
    assert restored.to_networkx().graph["dendra_compartment_schema"] == 1


def test_networkx_adapter_preserves_storage_order_and_material_domains():
    graph = nx.DiGraph()
    graph.add_node(
        "child",
        L=5.0,
        diam=2.0,
        Ra=100.0,
        cm=1.0,
        area=10.0,
        volume=8.0,
        volume_i=3.0,
        volume_o=5.0,
    )
    graph.add_node(
        "root",
        L=7.0,
        diam=3.0,
        Ra=120.0,
        cm=1.2,
        area=20.0,
        volume=11.0,
        volume_i=9.0,
        volume_o=2.0,
    )
    graph.add_edge("root", "child", L=6.0, R_ohm=40.0, diff_geom_um=2.0)

    canonical = CompartmentGraph.from_networkx(graph)

    # Storage/mechanism order follows graph insertion order even though the
    # solver will later construct its own topological permutation.
    assert canonical.metadata.name == ("child", "root")
    assert canonical.topology.parent_index == (1, -1)
    assert canonical.geometry.volume_i_um3 == (3.0, 9.0)
    assert canonical.geometry.volume_o_um3 == (5.0, 2.0)
    restored = canonical.to_networkx()
    assert restored.nodes[0]["volume_i"] == 3.0
    assert restored.nodes[0]["volume_o"] == 5.0


def test_tree_from_morphology_retains_canonical_graph_and_ordered_slice_labels():
    morphology = dn.Morphology(rhoa=90.0, cm=1.4)
    soma = morphology.section("soma", L=12.0, diam=8.0, labels="cell_body")
    axon = morphology.section("axon_branch", L=30.0, diam=2.0, nseg=3, labels="neurite")
    axon.connect(soma.at(1.0), child_end=1)

    model = dn.Tree.from_morphology(morphology, N=2, dtype=torch.float64)

    assert isinstance(model.compartment_graph, dn.CompartmentGraph)
    assert model.compartment_graph == morphology.compile()
    assert model.shape == (2, 4)
    assert model.cell_body.shape == (2, 1)
    assert model.neurite.shape == (2, 3)
    # A child attached through end 1 traverses in decreasing section x, while
    # its public Section label deliberately retains increasing x order.
    labelled_x = [
        model.compartment_graph.metadata.section_x[index]
        for index in model.neurite.index[-1].tolist()
    ]
    assert labelled_x == sorted(labelled_x)
    assert model.compartment_graph.metadata.segment_index[1:] == (2, 1, 0)
    np.testing.assert_allclose(
        model.volume_i.detach().cpu().numpy(),
        np.broadcast_to(model.compartment_graph.geometry.volume_i_um3, (2, 4)),
    )


def test_tree_from_graph_canonical_snapshot_aligns_with_storage_order():
    graph = nx.DiGraph()
    for node, length in ((0, 5.0), (1, 7.0)):
        graph.add_node(
            node,
            name=f"node{node}",
            L=length,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=2.0 * math.pi * length,
        )
    graph.add_edge(1, 0, L=6.0, R_ohm=40.0, diff_geom_um=2.0)

    model = dn.Tree.from_graph(graph, dtype=torch.float64)

    assert model.compartment_graph.topology.parent_index == (1, -1)
    assert model.compartment_graph.metadata.name == tuple(model.names)
    assert model.dx.tolist() == [[5.0, 7.0]]


def test_networkx_adapter_computes_stylized_axial_defaults():
    graph = nx.DiGraph()
    common = {"L": 10.0, "diam": 2.0, "Ra": 100.0, "cm": 1.0}
    graph.add_node("parent", name="soma", **common)
    graph.add_node("child", name="dend", **common)
    graph.add_edge("parent", "child")

    canonical = CompartmentGraph.from_networkx(graph)

    assert canonical.topology.parent_index == (-1, 0)
    assert canonical.geometry.edge_length_um[1] == 10.0
    assert canonical.geometry.edge_resistance_ohm[1] == pytest.approx(1e7 / math.pi)
    assert canonical.geometry.edge_diff_geom_um[1] == pytest.approx(math.pi / 10)


def test_builder_rejects_ambiguous_geometry_invalid_connections_and_forests():
    morphology = Morphology()
    with pytest.raises(ValueError, match="cannot also specify"):
        morphology.section("mixed", L=10.0, points=[(0, 0, 0, 1), (0, 0, 10, 1)])
    with pytest.raises(TypeError, match="positive integer"):
        morphology.section("bad_nseg", L=10.0, diam=1.0, nseg=1.5)

    first = morphology.section("first", L=10.0, diam=1.0)
    second = morphology.section("second", L=10.0, diam=1.0)
    with pytest.raises(ValueError, match="exactly one root"):
        morphology.compile()
    with pytest.raises(ValueError, match="endpoint"):
        morphology.connect(first.at(0.5), second.at(0.5))
    second.connect(first.at(1.0), child_end=0)
    with pytest.raises(ValueError, match="cycle"):
        first.connect(second.at(1.0), child_end=0)


def test_compiled_contract_is_immutable():
    morphology = Morphology()
    morphology.section("soma", L=10.0, diam=2.0)
    graph = morphology.compile()

    with pytest.raises(FrozenInstanceError):
        graph.schema_version = 2
    with pytest.raises(TypeError):
        graph.geometry.length_um[0] = 20.0


@pytest.mark.parametrize(
    "invalid_graph, message",
    [
        (nx.DiGraph(), "at least one"),
        (nx.path_graph(2), "simple DiGraph"),
    ],
)
def test_networkx_adapter_rejects_invalid_graph_kinds(invalid_graph, message):
    with pytest.raises((TypeError, ValueError), match=message):
        CompartmentGraph.from_networkx(invalid_graph)
