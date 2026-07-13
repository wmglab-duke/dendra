"""Dtype contracts for graph-derived Tree geometry."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn


def _precise_graph():
    graph = nx.DiGraph()
    bases = (1.000000123456789, 2.000000234567891)
    for node, base in enumerate(bases):
        graph.add_node(
            node,
            name="soma.precise" if node == 0 else "dend.precise",
            L=base,
            diam=base + 1.0,
            x=base + 3.0,
            y=base + 4.0,
            z=base + 5.0,
            volume=base + 20.0,
            volume_um3=base + 20.0,
            volume_i=base + 18.0,
            volume_o=base + 2.0,
            Ra=base + 100.0,
            cm=base + 0.5,
            area=base + 10.0,
        )
    graph.add_edge(
        0,
        1,
        R_ohm=123456.789123456,
        diff_geom_um=0.123456789123456,
        L=2.000000345678912,
    )
    return graph


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_tree_from_graph_preserves_source_precision_until_requested_downcast(dtype):
    graph = _precise_graph()
    tree = dn.Tree.from_graph(graph, N=2, dtype=dtype)

    node_fields = {
        "dx": "L",
        "diam": "diam",
        "x": "x",
        "y": "y",
        "z": "z",
        "volume": "volume",
        "volume_um3": "volume_um3",
        "volume_i": "volume_i",
        "volume_o": "volume_o",
        "rhoa": "Ra",
        "cm": "cm",
    }
    for buffer_name, graph_name in node_fields.items():
        expected = torch.tensor(
            [graph.nodes[node][graph_name] for node in range(2)], dtype=dtype
        ).expand(2, -1)
        actual = getattr(tree, buffer_name)
        assert actual.dtype == dtype
        assert torch.equal(actual, expected), buffer_name

    expected_diffusion = torch.tensor(
        [[0.0, graph.edges[0, 1]["diff_geom_um"]]], dtype=dtype
    ).expand(2, -1)
    expected_edge_diffusion = torch.tensor(
        [[graph.edges[0, 1]["diff_geom_um"]]], dtype=dtype
    ).expand(2, -1)
    assert torch.equal(tree.diff_geom_um, expected_diffusion)
    assert torch.equal(tree.diff_edge_geom_um, expected_edge_diffusion)

    expected_area = (
        (
            torch.tensor(
                [[graph.nodes[node]["area"] for node in range(2)]],
                dtype=torch.float64,
            )
            * 1e-8
        )
        .to(dtype=dtype)
        .expand(2, -1)
    )
    assert torch.equal(tree.area, expected_area)

    # Topology is integer metadata and must not be coupled to the model's
    # floating dtype conversion.
    assert tree.diff_parent_index.dtype == torch.long
    assert tree.diff_edge_parent.dtype == torch.long
    assert tree.diff_edge_child.dtype == torch.long
    assert torch.equal(tree.diff_parent_index, torch.tensor([-1, 0]))
    assert torch.equal(tree.diff_edge_parent, torch.tensor([0]))
    assert torch.equal(tree.diff_edge_child, torch.tensor([1]))
