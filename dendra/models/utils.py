"""Reusable helpers for model morphology graphs."""

from collections.abc import Iterable

import networkx as nx
import torch

__all__ = ["distance", "undirected_weighted_lengths"]


def _slice_resolution_nodes(cell, graph):
    """Return the node order represented by population compartment slices."""
    compartment_graph = getattr(cell, "compartment_graph", None)
    if compartment_graph is None:
        return list(graph.nodes)

    n_compartments = compartment_graph.n_compartments
    storage_nodes = range(n_compartments)
    if graph.number_of_nodes() != n_compartments or set(graph.nodes) != set(
        storage_nodes
    ):
        raise ValueError(
            "The cell's NetworkX graph is not aligned with its canonical "
            "compartment graph; expected node IDs 0..n_compartments-1."
        )
    return storage_nodes


def distance(cell, origin, targets, *, origin_offset=0.0):
    """Return undirected, length-weighted distances between cell compartments.

    Parameters
    ----------
    cell : object
        An object whose ``graph`` attribute is a NetworkX graph. Edges are
        weighted by their ``L`` attribute.
    origin : node label or slice
        Source node. For a canonical Dendra cell, a slice is resolved against
        compartment storage order. For a generic graph-bearing object, it is
        resolved against graph node iteration order. The first selected node
        is used.
    targets : node label, iterable of node labels, or slice
        Destination node or nodes. Slice resolution follows the same canonical
        storage-order or generic graph-order rule as ``origin``.
    origin_offset : float or scalar tensor, optional
        Additional path length added to every result. This makes endpoint
        origins explicit when the graph stores compartment-center distances;
        for example, a one-segment soma origin at ``soma(0)`` may require half
        the soma length. Defaults to zero.

    Returns
    -------
    torch.Tensor
        One distance per requested target, in target order. Unreachable
        targets have distance ``inf``. A scalar target produces a one-element
        tensor.

    Raises
    ------
    ValueError
        If the graph is unavailable or misaligned with canonical storage, or
        if an origin, target, or offset selector is invalid.
    """
    graph = cell.graph
    if graph is None:
        raise ValueError("Graph is not defined for this cell.")

    nodes = _slice_resolution_nodes(cell, graph)
    if isinstance(origin, slice):
        selected = nodes[origin]
        if not selected:
            raise ValueError("The origin slice selects no graph nodes.")
        origin = selected[0]
    if torch.is_tensor(origin):
        if origin.numel() != 1:
            raise ValueError("origin must identify exactly one graph node.")
        origin = origin.detach().cpu().item()
    if isinstance(targets, slice):
        targets = nodes[targets]
    if torch.is_tensor(targets):
        if targets.ndim > 1:
            raise ValueError("targets must be a scalar or one-dimensional node list.")
        targets = targets.detach().cpu()
        targets = targets.item() if targets.ndim == 0 else targets.tolist()
    lengths = undirected_weighted_lengths(graph, origin, targets)
    offset = torch.as_tensor(origin_offset, dtype=lengths.dtype, device=lengths.device)
    if offset.numel() != 1:
        raise ValueError("origin_offset must be a scalar path length.")
    return lengths + offset.reshape(())


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
