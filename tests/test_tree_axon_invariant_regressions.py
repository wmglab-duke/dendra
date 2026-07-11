import math

import networkx as nx
import pytest
import torch

from dendra.models.core import Unmyelinated
from dendra.models.tree import Tree

DTYPE = torch.float64


def _tree_graph(labels=tuple(range(5)), edges=None):
    labels = tuple(labels)
    if edges is None:
        edges = [(labels[0], label) for label in labels[1:]]

    graph = nx.DiGraph()
    for position, label in enumerate(labels):
        region = "soma" if position == 0 else "dend"
        graph.add_node(
            label,
            name=f"Cell.{region}[{position}](0.5)",
            L=2.0 + position,
            diam=1.0 + 0.1 * position,
            Ra=80.0 + position,
            cm=1.0 + 0.05 * position,
            area=10.0 + position,
            x=float(position),
            y=float(position % 2),
            z=2.0 * position,
        )
    for parent, child in edges:
        graph.add_edge(parent, child, L=1.0, diff_geom_um=0.5)
    return graph


def test_tree_and_axon_direct_construction_honor_device_and_dtype():
    tree = Tree.from_graph(_tree_graph(), N=2, device="meta", dtype=DTYPE)
    tree_buffers = dict(tree.named_buffers())
    assert tree_buffers
    assert all(value.device.type == "meta" for value in tree_buffers.values())
    assert all(
        not value.is_floating_point() or value.dtype == DTYPE
        for value in tree_buffers.values()
    )
    assert all(value.device.type == "meta" for value in tree.parameters())
    for name in (
        "v",
        "diam",
        "dx",
        "x",
        "y",
        "z",
        "volume",
        "volume_i",
        "volume_o",
        "diff_geom_um",
        "diff_parent_index",
    ):
        assert tree_buffers[name].device.type == "meta"
    for name in (
        "v",
        "diam",
        "dx",
        "x",
        "y",
        "z",
        "volume",
        "volume_i",
        "volume_o",
        "diff_geom_um",
    ):
        assert tree_buffers[name].dtype == DTYPE
    assert tree_buffers["diff_parent_index"].dtype == torch.long

    diameters = torch.tensor([8.0, 12.0], dtype=torch.float32)
    axon = Unmyelinated(
        diameters=diameters,
        L=40.0,
        dx=10.0,
        device="meta",
        dtype=DTYPE,
    )
    axon_buffers = dict(axon.named_buffers())
    assert all(value.device.type == "meta" for value in axon_buffers.values())
    assert all(
        not value.is_floating_point() or value.dtype == DTYPE
        for value in axon_buffers.values()
    )
    assert all(value.device.type == "meta" for value in axon.parameters())
    for name in ("v", "diam", "dx", "x", "y", "z", "diameters"):
        value = axon_buffers[name]
        assert value.device.type == "meta"
        assert value.dtype == DTYPE


def test_batched_tree_rotations_match_unbatched_five_compartment_reference():
    graph = _tree_graph()
    reference = Tree.from_graph(graph, N=2, dtype=DTYPE)
    batched = Tree.from_graph(graph, N=2, dtype=DTYPE).batch(3)
    target = torch.tensor([[1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]], dtype=DTYPE)
    target = torch.nn.functional.normalize(target, dim=1)
    angles = torch.tensor([35.0, -20.0], dtype=DTYPE)

    reference.rotate_into_direction(target, origin=0)
    reference.rotate_azimuthal(angles, origin=0)
    batched.rotate_into_direction(target, origin=0)
    batched.rotate_azimuthal(angles, origin=0)

    for coordinate in ("x", "y", "z"):
        expected = getattr(reference, coordinate).expand(3, -1, -1)
        torch.testing.assert_close(getattr(batched, coordinate), expected)
    torch.testing.assert_close(batched.directions, reference.directions)
    torch.testing.assert_close(
        batched.azimuthal_rotations, reference.azimuthal_rotations
    )

    reference.reset_rotations(origin=0)
    batched.reset_rotations(origin=0)
    for coordinate in ("x", "y", "z"):
        expected = getattr(reference, coordinate).expand(3, -1, -1)
        torch.testing.assert_close(getattr(batched, coordinate), expected)


def test_tree_from_graph_relabels_noninteger_nodes_in_insertion_order():
    labels = ("tip", "root", ("branch", 1))
    graph = _tree_graph(
        labels,
        edges=[("root", "tip"), ("root", ("branch", 1))],
    )
    original_nodes = list(graph.nodes)
    original_edges = list(graph.edges)

    tree = Tree.from_graph(graph, N=1, dtype=DTYPE)

    assert list(tree.graph.nodes) == [0, 1, 2]
    assert set(tree.graph.edges) == {(1, 0), (1, 2)}
    assert tree.names == [
        "Cell.soma[0](0.5)",
        "Cell.dend[1](0.5)",
        "Cell.dend[2](0.5)",
    ]
    assert tree.terminal_indices() == [0, 2]
    assert list(graph.nodes) == original_nodes
    assert list(graph.edges) == original_edges


@pytest.mark.parametrize(
    "graph,error,match",
    [
        (None, TypeError, "NetworkX graph"),
        (nx.Graph([(0, 1)]), ValueError, "directed graph"),
        (nx.MultiDiGraph([(0, 1)]), ValueError, "simple directed graph"),
        (nx.DiGraph(), ValueError, "at least one node"),
    ],
)
def test_tree_from_graph_rejects_unsupported_graph_kinds(graph, error, match):
    with pytest.raises(error, match=match):
        Tree.from_graph(graph, N=1)


@pytest.mark.parametrize(
    "graph,match",
    [
        (
            _tree_graph((0, 1), edges=[(0, 0), (0, 1)]),
            "self-loops",
        ),
        (
            _tree_graph((0, 1, 2), edges=[(0, 1), (1, 2), (2, 0)]),
            "acyclic",
        ),
        (
            _tree_graph((0, 1, 2), edges=[(0, 1)]),
            "connected",
        ),
        (
            _tree_graph((0, 1, 2), edges=[(0, 2), (1, 2)]),
            "at most one parent",
        ),
    ],
)
def test_tree_from_graph_rejects_invalid_tree_topologies_before_mutation(graph, match):
    nodes_before = list(graph.nodes(data=True))
    edges_before = list(graph.edges(data=True))

    with pytest.raises(ValueError, match=match):
        Tree.from_graph(graph, N=1)

    assert list(graph.nodes(data=True)) == nodes_before
    assert list(graph.edges(data=True)) == edges_before


def test_single_compartment_axon_graph_has_finite_canonical_metadata():
    axon = Unmyelinated([8.0, 12.0], L=1.0, dx=10.0, dtype=DTYPE)
    assert axon.n_comp == 1

    graphs = axon.assemble_graphs()

    assert len(graphs) == 2
    for graph in graphs:
        assert list(graph.nodes) == [0]
        assert graph.number_of_edges() == 0
        attrs = graph.nodes[0]
        assert attrs["name"].endswith("(0.50)")
        for name in ("x", "y", "z", "diam", "L", "Ra", "Cm"):
            assert math.isfinite(float(attrs[name]))
