"""Independent contract tests for native scalar morphologies and Tree integration."""

from __future__ import annotations

import math
from dataclasses import replace

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mod import pas

pytestmark = pytest.mark.cpu


def _section_nodes(graph, section_name):
    return [
        node
        for node, source in enumerate(graph.metadata.section_name)
        if source == section_name
    ]


def _graph_node(graph, node, *, length, diameter=2.0, name=None):
    graph.add_node(
        node,
        name=str(node) if name is None else name,
        L=float(length),
        diam=float(diameter),
        Ra=100.0,
        cm=1.0,
        area=math.pi * float(diameter) * float(length),
        volume=math.pi * (0.5 * float(diameter)) ** 2 * float(length),
    )


def test_native_compilation_is_repeatable_parent_first_and_field_aligned():
    morphology = dn.Morphology(rhoa=93.0, cm=1.1)
    # Declaration order is deliberately unrelated to tree traversal order.
    left = morphology.section("left", L=20.0, diam=2.0, nseg=2)
    right = morphology.section("right", L=30.0, diam=3.0, nseg=3)
    root = morphology.section("root", L=12.0, diam=8.0, nseg=2)
    left.connect(root.at(1.0), child_end=0)
    right.connect(root.at(1.0), child_end=1)

    first = morphology.compile()
    second = morphology.compile()

    assert first == second
    assert first.topology.root == 0
    assert first.n_compartments == 2 + 2 + 3 + 1  # material nodes + junction
    assert all(
        parent == -1 or parent < child
        for child, parent in enumerate(first.topology.parent_index)
    )
    assert set(first.topology.edges) == set(first.to_networkx().edges)

    for field_name in first.geometry.__dataclass_fields__:
        assert len(getattr(first.geometry, field_name)) == first.n_compartments
    for field_name in first.metadata.__dataclass_fields__:
        assert len(getattr(first.metadata, field_name)) == first.n_compartments

    root_index = first.topology.root
    for edge_field in (
        first.geometry.edge_length_um,
        first.geometry.edge_resistance_ohm,
        first.geometry.edge_diff_geom_um,
    ):
        assert edge_field[root_index] == 0.0
        assert all(value > 0.0 for node, value in enumerate(edge_field) if node != 0)
    assert first.metadata.kind.count("junction") == 1
    assert first.metadata.kind.count("compartment") == 7


def test_section_name_is_an_automatic_label_covering_the_whole_section():
    morphology = dn.Morphology()
    soma = morphology.section("soma", L=30.0, diam=10.0, nseg=3, labels="cell_body")

    assert soma.labels == frozenset({"soma", "cell_body"})

    graph = morphology.compile()
    soma_nodes = _section_nodes(graph, "soma")
    assert len(soma_nodes) == soma.nseg
    assert graph.metadata.section_name == ("soma",) * soma.nseg
    assert graph.metadata.name == tuple(
        f"soma({(index + 0.5) / soma.nseg:.12g})" for index in range(soma.nseg)
    )
    assert all("soma" in graph.metadata.labels[node] for node in soma_nodes)
    assert all("cell_body" in graph.metadata.labels[node] for node in soma_nodes)

    model = dn.Tree.from_morphology(morphology)
    assert model.soma.index[-1].tolist() == soma_nodes
    assert model.cell_body.index[-1].tolist() == soma_nodes


@pytest.mark.parametrize("name", (None, "", "   "))
def test_section_name_must_be_a_nonempty_string_atomically(name):
    morphology = dn.Morphology()

    with pytest.raises(ValueError, match="non-empty strings"):
        morphology.section(name, L=10.0, diam=2.0)

    assert morphology.sections == ()


def test_section_names_are_unique_atomically():
    morphology = dn.Morphology()
    soma = morphology.section("soma", L=10.0, diam=8.0)

    with pytest.raises(ValueError, match="already exists"):
        morphology.section("soma", L=20.0, diam=2.0)

    assert morphology.sections == (soma,)


@pytest.mark.parametrize("model_cls", (dn.Tree, dn.Cable), ids=("tree", "cable"))
def test_shared_labels_union_whole_sections_in_declaration_and_authored_x_order(
    model_cls,
):
    morphology = dn.Morphology()
    # Declare the child first and connect through its authored x=1 end. Graph
    # traversal therefore disagrees with both declaration and authored-x order.
    child = morphology.section("child", L=40.0, diam=2.0, nseg=4, labels="excitable")
    root = morphology.section("root", L=20.0, diam=8.0, nseg=2, labels="excitable")
    child.connect(root.at(1.0), child_end=1)

    model = model_cls.from_morphology(morphology)
    graph = model.compartment_graph
    selected = model.excitable.index[-1].tolist()
    provenance = [
        (graph.metadata.section_name[node], graph.metadata.section_x[node])
        for node in selected
    ]

    assert provenance == [
        ("child", pytest.approx(0.125)),
        ("child", pytest.approx(0.375)),
        ("child", pytest.approx(0.625)),
        ("child", pytest.approx(0.875)),
        ("root", pytest.approx(0.25)),
        ("root", pytest.approx(0.75)),
    ]
    assert len(selected) == child.nseg + root.nseg
    assert set(selected) == set(graph.nodes_with_label("excitable"))


def test_explicit_label_cannot_reuse_an_existing_section_name_atomically():
    morphology = dn.Morphology()
    soma = morphology.section("soma", L=10.0, diam=8.0)
    before_sections = morphology.sections
    before_graph = morphology.compile()

    with pytest.raises(ValueError, match=r"(?i)section.*label|label.*section"):
        morphology.section("dend", L=20.0, diam=2.0, labels="soma")

    assert morphology.sections == before_sections
    assert morphology.compile() == before_graph

    # A rejected declaration must not reserve its name or otherwise poison the
    # builder; a corrected declaration remains possible.
    dend = morphology.section("dend", L=20.0, diam=2.0, labels="neurite")
    dend.connect(soma.at(1.0), child_end=0)
    assert morphology.compile().n_compartments == 2


def test_section_name_cannot_reuse_an_existing_explicit_label_atomically():
    morphology = dn.Morphology()
    dend = morphology.section("dend", L=20.0, diam=2.0, labels={"neurite", "soma"})
    before_sections = morphology.sections
    before_graph = morphology.compile()

    with pytest.raises(ValueError, match=r"(?i)section.*label|label.*section"):
        morphology.section("soma", L=10.0, diam=8.0)

    assert morphology.sections == before_sections
    assert morphology.compile() == before_graph

    # The failed name declaration has no partial effects: an unrelated valid
    # Section can still be added and connected normally.
    axon = morphology.section("axon", L=30.0, diam=1.0)
    axon.connect(dend.at(1.0), child_end=0)
    assert morphology.compile().n_compartments == 2


def test_builder_validation_failures_are_atomic_and_repairable():
    morphology = dn.Morphology()

    with pytest.raises(ValueError, match="positive"):
        morphology.section("root", L=0.0, diam=2.0)
    root = morphology.section("root", L=10.0, diam=2.0)
    child = morphology.section("child", L=10.0, diam=2.0)
    grandchild = morphology.section("grandchild", L=10.0, diam=2.0)
    child.connect(root.at(1.0), child_end=0)
    grandchild.connect(child.at(1.0), child_end=0)
    before = morphology.compile()

    with pytest.raises(ValueError, match="cycle"):
        root.connect(grandchild.at(1.0), child_end=0)
    with pytest.raises(ValueError, match="already has a parent"):
        morphology.connect(root.at(0.0), grandchild, child_end=1)
    assert morphology.compile() == before

    new_branch = morphology.section("new_branch", L=8.0, diam=1.0)
    with pytest.raises(ValueError, match="endpoint"):
        morphology.connect(root.at(0.5), new_branch.at(0.5))
    # The rejected connection did not partially assign a parent.
    with pytest.raises(ValueError, match="exactly one root"):
        morphology.compile()
    new_branch.connect(root.at(0.5), child_end=0)
    assert morphology.compile().n_compartments == 4


def test_canonical_value_objects_reject_invalid_domain_and_provenance_values():
    morphology = dn.Morphology()
    morphology.section("root", L=10.0, diam=2.0)
    graph = morphology.compile()

    with pytest.raises(ValueError, match="volume_i_um3.*non-negative"):
        replace(graph.geometry, volume_i_um3=(-1.0,))
    with pytest.raises(ValueError, match="volume_o_um3.*non-negative"):
        replace(graph.geometry, volume_o_um3=(-1.0,))
    with pytest.raises(TypeError, match="segment_index.*integers"):
        replace(graph.metadata, segment_index=(0.5,))

    # A single label string is one label, never an iterable of characters.
    metadata = replace(graph.metadata, labels=("whole_cell",))
    assert metadata.labels == (frozenset({"whole_cell"}),)


def test_canonical_junction_rejects_nonzero_material_domain_volume():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=4.0)
    left = morphology.section("left", L=10.0, diam=2.0)
    right = morphology.section("right", L=10.0, diam=2.0)
    left.connect(root.at(1.0), child_end=0)
    right.connect(root.at(1.0), child_end=0)
    graph = morphology.compile()
    junction = graph.metadata.kind.index("junction")
    volume_i = list(graph.geometry.volume_i_um3)
    volume_i[junction] = 1.0
    invalid_geometry = replace(graph.geometry, volume_i_um3=tuple(volume_i))

    with pytest.raises(ValueError, match="Junction.*material volumes"):
        replace(graph, geometry=invalid_geometry)


@pytest.mark.parametrize("boundary_index", range(1, 7))
def test_exact_arbitrary_interior_boundaries_choose_x_increasing_segment(
    boundary_index,
):
    nseg = 7
    morphology = dn.Morphology()
    trunk = morphology.section("trunk", L=70.0, diam=2.0, nseg=nseg)
    branch = morphology.section("branch", L=8.0, diam=1.0)
    branch.connect(trunk.at(boundary_index / nseg), child_end=0)

    graph = morphology.compile()
    branch_node = _section_nodes(graph, "branch")[0]
    host_node = graph.topology.parent_index[branch_node]

    assert graph.metadata.section_name[host_node] == "trunk"
    assert graph.metadata.segment_index[host_node] == boundary_index
    assert graph.geometry.edge_length_um[branch_node] == pytest.approx(4.0)


def test_values_immediately_below_boundary_select_lower_segment():
    nseg = 7
    exact = 3.0 / nseg
    parent_x = math.nextafter(exact, 0.0)
    morphology = dn.Morphology()
    trunk = morphology.section("trunk", L=70.0, diam=2.0, nseg=nseg)
    branch = morphology.section("branch", L=8.0, diam=1.0)
    branch.connect(trunk.at(parent_x), child_end=0)

    graph = morphology.compile()
    branch_node = _section_nodes(graph, "branch")[0]
    host_node = graph.topology.parent_index[branch_node]

    assert graph.metadata.segment_index[host_node] == 2


def test_reversed_pt3d_child_preserves_authored_coordinates_and_label_order():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=8.0)
    cable = morphology.section(
        "cable",
        points=[(0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 40.0, 6.0)],
        nseg=4,
        labels="ordered_region",
    )
    cable.connect(root.at(1.0), child_end=1)

    graph = morphology.compile()
    traversal_nodes = _section_nodes(graph, "cable")
    assert [graph.metadata.segment_index[node] for node in traversal_nodes] == [
        3,
        2,
        1,
        0,
    ]
    assert [
        graph.metadata.section_x[node] for node in traversal_nodes
    ] == pytest.approx([0.875, 0.625, 0.375, 0.125])
    assert [graph.geometry.z_um[node] for node in traversal_nodes] == pytest.approx(
        [35.0, 25.0, 15.0, 5.0]
    )
    assert [
        graph.geometry.diameter_um[node] for node in traversal_nodes
    ] == pytest.approx([5.5, 4.5, 3.5, 2.5])

    model = dn.Tree.from_morphology(morphology, dtype=torch.float64)
    labelled_nodes = model.ordered_region.index[-1].tolist()
    assert [graph.metadata.segment_index[node] for node in labelled_nodes] == [
        0,
        1,
        2,
        3,
    ]
    assert model.ordered_region.z.tolist() == [[5.0, 15.0, 25.0, 35.0]]


def test_endpoint_junction_collapse_and_retention_preserve_exact_series_paths():
    collapsed = dn.Morphology()
    root = collapsed.section("root", L=10.0, diam=4.0, rhoa=80.0)
    child = collapsed.section("child", L=20.0, diam=2.0, rhoa=120.0)
    child.connect(root.at(1.0), child_end=0)
    collapsed_graph = collapsed.compile()

    inv_root = 5.0 / (4.0 * math.pi)
    inv_child = 10.0 / math.pi
    assert collapsed_graph.metadata.kind == ("compartment", "compartment")
    assert collapsed_graph.geometry.edge_resistance_ohm[1] == pytest.approx(
        1e4 * (80.0 * inv_root + 120.0 * inv_child)
    )
    assert collapsed_graph.geometry.edge_diff_geom_um[1] == pytest.approx(
        1.0 / (inv_root + inv_child)
    )

    retained = dn.Morphology()
    root = retained.section("root", L=10.0, diam=4.0, rhoa=80.0)
    left = retained.section("left", L=20.0, diam=2.0, rhoa=120.0)
    right = retained.section("right", L=30.0, diam=3.0, rhoa=90.0)
    left.connect(root.at(1.0), child_end=0)
    right.connect(root.at(1.0), child_end=0)
    retained_graph = retained.compile()
    junction = retained_graph.metadata.kind.index("junction")

    assert retained_graph.topology.parent_index[junction] == 0
    assert set(retained_graph.topology.parent_index).issuperset({0, junction})
    assert retained_graph.geometry.volume_um3[junction] == 0.0
    assert retained_graph.geometry.volume_i_um3[junction] == 0.0
    assert retained_graph.geometry.volume_o_um3[junction] == 0.0
    assert retained_graph.metadata.labels[junction] == frozenset()
    assert retained_graph.geometry.edge_resistance_ohm[junction] == pytest.approx(
        1e4 * 80.0 * inv_root
    )


def test_dense_networkx_ids_define_storage_order_even_if_inserted_out_of_order():
    graph = nx.DiGraph()
    _graph_node(graph, 2, length=30.0, name="node_2")
    _graph_node(graph, 0, length=10.0, name="node_0")
    _graph_node(graph, 1, length=20.0, name="node_1")
    graph.add_edge(2, 0, L=20.0, R_ohm=20.0, diff_geom_um=2.0)
    graph.add_edge(0, 1, L=15.0, R_ohm=15.0, diff_geom_um=1.5)

    canonical = dn.CompartmentGraph.from_networkx(graph)
    model = dn.Tree.from_graph(graph, dtype=torch.float64)

    assert canonical.metadata.name == ("node_0", "node_1", "node_2")
    assert canonical.topology.parent_index == (2, 0, -1)
    assert canonical == dn.CompartmentGraph.from_networkx(canonical.to_networkx())
    assert model.compartment_graph == canonical
    assert model.names == list(canonical.metadata.name)
    assert model.dx.tolist() == [[10.0, 20.0, 30.0]]


def test_arbitrary_networkx_ids_use_insertion_order_then_round_trip_stably():
    graph = nx.DiGraph()
    _graph_node(graph, "tip", length=6.0)
    _graph_node(graph, "root", length=10.0)
    _graph_node(graph, "middle", length=8.0)
    graph.add_edge("root", "middle", L=9.0, R_ohm=9.0, diff_geom_um=0.9)
    graph.add_edge("middle", "tip", L=7.0, R_ohm=7.0, diff_geom_um=0.7)

    canonical = dn.CompartmentGraph.from_networkx(graph)
    model = dn.Tree.from_graph(graph, dtype=torch.float64)

    assert canonical.metadata.name == ("tip", "root", "middle")
    assert canonical.topology.parent_index == (2, -1, 1)
    assert canonical == dn.CompartmentGraph.from_networkx(canonical.to_networkx())
    assert model.compartment_graph == canonical
    assert model.dx.tolist() == [[6.0, 10.0, 8.0]]


def test_safe_labels_register_without_overwriting_attributes_or_internal_nodes():
    morphology = dn.Morphology()
    root = morphology.section(
        "root",
        L=10.0,
        diam=4.0,
        labels={
            "safe_region",
            "shape",
            "bad-label",
            "_private",
            "class",
            "internal_nodes",
        },
    )
    left = morphology.section("left", L=10.0, diam=2.0)
    right = morphology.section("right", L=10.0, diam=2.0)
    left.connect(root.at(1.0), child_end=0)
    right.connect(root.at(1.0), child_end=0)

    model = dn.Tree.from_morphology(morphology)
    graph = model.compartment_graph
    material_nodes = [
        node for node, kind in enumerate(graph.metadata.kind) if kind == "compartment"
    ]

    assert model.safe_region.index[-1].tolist() == [0]
    assert model.shape == (1, graph.n_compartments)
    assert "shape" not in model._labels
    assert "bad-label" not in model._labels
    assert "_private" not in model._labels
    assert "class" not in model._labels
    assert model.internal_nodes.index[-1].tolist() == material_nodes
    assert "internal_nodes" in graph.metadata.labels[0]


def test_internal_nodes_uses_canonical_kind_not_junction_name_heuristics():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=4.0)
    left = morphology.section("left", L=10.0, diam=2.0)
    right = morphology.section("right", L=10.0, diam=2.0)
    left.connect(root.at(1.0), child_end=0)
    right.connect(root.at(1.0), child_end=0)
    networkx_graph = morphology.compile().to_networkx()
    junction = next(
        node
        for node, attrs in networkx_graph.nodes(data=True)
        if attrs["kind"] == "junction"
    )
    networkx_graph.nodes[junction]["name"] = "electrical_join"
    canonical = dn.CompartmentGraph.from_networkx(networkx_graph)

    model = dn.Tree.from_compartment_graph(canonical)
    material_nodes = [
        node
        for node, kind in enumerate(canonical.metadata.kind)
        if kind == "compartment"
    ]

    assert junction not in material_nodes
    assert model.internal_nodes.index[-1].tolist() == material_nodes


def test_native_tree_population_and_explicit_batch_run_share_canonical_template():
    morphology = dn.Morphology()
    morphology.section("cable", L=30.0, diam=2.0, nseg=3, labels="active")
    model = dn.Tree.from_morphology(morphology, N=2, dtype=torch.float64)
    template = model.compartment_graph
    model.active.insert(pas, g=0.001, e=-65.0)

    model.batch(3)

    assert model.shape == (3, 2, 3)
    assert model.active.shape == (3, 2, 3)
    assert model.compartment_graph is template
    assert model.compartment_graph == morphology.compile()
    assert model.x.shape == (3, 2, 3)
    assert model.dx.shape == (2, 3)  # geometry broadcasts over explicit batch axes
    assert model.dx.tolist() == [[10.0, 10.0, 10.0]] * 2

    model.initialize()
    model.run(tstop=0.05, dt=0.025)
    assert model.v.shape == (3, 2, 3)
    assert torch.isfinite(model.v).all()
    torch.testing.assert_close(model.v, model.v[0].expand_as(model.v))


def test_custom_graph_subclasses_must_explicitly_adapt_canonical_morphology():
    morphology = dn.Morphology()
    morphology.section("cable", L=10.0, diam=2.0)

    with pytest.raises(NotImplementedError, match="custom from_graph semantics"):
        dn.ExtCellTree.from_morphology(morphology)
