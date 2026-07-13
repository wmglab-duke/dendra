"""Reusable helpers for model morphology graphs."""

from collections.abc import Iterable

import networkx as nx
import torch

__all__ = ["distance", "undirected_weighted_lengths"]


def distance(cell, origin, targets):
    """Return undirected, length-weighted distances between cell compartments.

    Parameters
    ----------
    cell : object
        An object whose ``graph`` attribute is a NetworkX graph. Edges are
        weighted by their ``L`` attribute.
    origin : node label or slice
        Source node. A slice is resolved against the graph's node iteration
        order; for compatibility, the first selected node is used.
    targets : node label, iterable of node labels, or slice
        Destination node or nodes. A slice is resolved against the graph's
        node iteration order.

    Returns
    -------
    torch.Tensor
        One distance per requested target, in target order. Unreachable
        targets have distance ``inf``. A scalar target produces a one-element
        tensor.

    Raises
    ------
    ValueError
        If the cell has no graph or an origin slice selects no nodes.
    """
    graph = cell.graph
    if graph is None:
        raise ValueError("Graph is not defined for this cell.")

    nodes = list(graph.nodes)
    if isinstance(origin, slice):
        selected = nodes[origin]
        if not selected:
            raise ValueError("The origin slice selects no graph nodes.")
        origin = selected[0]
    if isinstance(targets, slice):
        targets = nodes[targets]
    return undirected_weighted_lengths(graph, origin, targets)


def undirected_weighted_lengths(G: nx.DiGraph, origin, targets, weight_attr="L"):
    """Return shortest-path distances while ignoring edge direction.

    Parameters
    ----------
    G : nx.Graph
        Graph whose edges carry the numeric attribute named by
        ``weight_attr``. Directed graphs are viewed as undirected.
    origin : node label
        The single source node.
    targets : node label or iterable of node labels
        Node or nodes for which to return distances. A node label that is
        itself iterable, such as a tuple, is treated as a scalar when it is
        present in ``G``.
    weight_attr : str, optional
        Edge attribute storing length or weight. Defaults to ``"L"``.

    Returns
    -------
    torch.Tensor
        A one-dimensional CPU tensor containing one distance per target.
        Unreachable targets are represented by ``inf``.
    """
    UG = G.to_undirected(as_view=True)
    lengths = nx.single_source_dijkstra_path_length(
        UG, source=origin, weight=weight_attr
    )

    # Node labels can themselves be iterable (most notably tuple labels), so
    # graph membership is the most reliable scalar check. Unhashable values
    # simply fall through to normal iterable handling.
    try:
        scalar_target = targets in G
    except TypeError:
        scalar_target = False
    if (
        scalar_target
        or isinstance(targets, (str, bytes))
        or not isinstance(targets, Iterable)
    ):
        targets = (targets,)

    return torch.tensor([lengths.get(t, float("inf")) for t in targets])
