"""Graph utilities for analysing Dendra morphologies."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Tuple, Union

import networkx as nx
import torch

GraphLike = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


def get_area_from_graph(G: nx.DiGraph) -> torch.Tensor:
    """
    Extracts the area from the graph's nodes if available.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph representing the tree structure.

    Returns
    -------
    torch.Tensor or None
        A tensor containing the area in µm² for each node, or None if not available.
    """
    areas = []
    for n in range(len(G.nodes)):
        area = G.nodes[n].get("area", None)
        if area is not None:
            areas.append(area)
        else:
            return None  # If any node lacks area, return None

    # area in µm² -> convert to cm²
    # 1 cm² = 1e8 µm²
    # Preserve Python/graph binary64 values until the owning model performs an
    # explicit dtype conversion.  Otherwise a requested float64 Tree inherits
    # geometry that was already rounded through PyTorch's float32 default.
    return 1e-8 * torch.tensor(areas, dtype=torch.float64)


def _edge_signature(G: GraphLike):
    """Generate a hashable signature for a graph's edge structure.

    Parameters
    ----------
    G : GraphLike
        Input graph.

    Returns
    -------
    collections.Counter or frozenset
        Signature capturing edge multiplicity and direction appropriate for
        the graph type.
    """
    if G.is_multigraph():
        c = Counter()
        if G.is_directed():
            for u, v in G.edges():
                c[(u, v)] += 1
        else:
            for u, v in G.edges():
                c[frozenset((u, v))] += 1
        return c
    else:
        if G.is_directed():
            return frozenset(G.edges())
        else:
            return frozenset(frozenset((u, v)) for u, v in G.edges())


def share_topology_labeled(graphs: Iterable[GraphLike]) -> Tuple[bool, str]:
    """Check whether graphs share an identical labeled topology.

    Parameters
    ----------
    graphs : Iterable[GraphLike]
        Collection of graphs to compare.

    Returns
    -------
    tuple of (bool, str)
        Tuple containing a success flag and diagnostic message.
    """
    graphs = list(graphs)
    if not graphs:
        return False, "No graphs provided."

    g0 = graphs[0]
    kind0 = (g0.is_directed(), g0.is_multigraph())
    nodes0 = set(g0.nodes())
    sig0 = _edge_signature(g0)

    for i, g in enumerate(graphs[1:], 1):
        kind = (g.is_directed(), g.is_multigraph())
        if kind != kind0:
            return (
                False,
                f"Type mismatch at index {i}: directed/multigraph flags differ.",
            )
        if set(g.nodes()) != nodes0:
            return False, f"Node set mismatch at index {i}."
        if _edge_signature(g) != sig0:
            return False, f"Edge set mismatch at index {i}."
    return True, "All graphs share the same labeled topology."


def _is_isomorphic_unlabeled(g1: GraphLike, g2: GraphLike) -> bool:
    """Check unlabeled isomorphism ignoring node identities.

    Parameters
    ----------
    g1 : GraphLike
        First graph to compare.
    g2 : GraphLike
        Second graph to compare.

    Returns
    -------
    bool
        ``True`` if the graphs are isomorphic after relabeling, ``False`` otherwise.
    """
    directed = g1.is_directed()
    multi = g1.is_multigraph()
    if (g2.is_directed() != directed) or (g2.is_multigraph() != multi):
        return False

    if multi and directed:
        GM = nx.algorithms.isomorphism.MultiDiGraphMatcher
    elif multi and not directed:
        GM = nx.algorithms.isomorphism.MultiGraphMatcher
    elif not multi and directed:
        GM = nx.algorithms.isomorphism.DiGraphMatcher
    else:
        GM = nx.algorithms.isomorphism.GraphMatcher

    return GM(g1, g2).is_isomorphic()


def share_topology_isomorphic(graphs: Iterable[GraphLike]) -> Tuple[bool, str]:
    """Check whether graphs are mutually isomorphic.

    Parameters
    ----------
    graphs : Iterable[GraphLike]
        Collection of graphs to compare.

    Returns
    -------
    tuple of (bool, str)
        Tuple containing a success flag and diagnostic message.
    """
    graphs = list(graphs)
    if not graphs:
        return False, "No graphs provided."

    g0 = graphs[0]
    for i, g in enumerate(graphs[1:], 1):
        if not _is_isomorphic_unlabeled(g0, g):
            return False, f"Not isomorphic to graph 0 at index {i}."
    return True, "All graphs are isomorphic (same unlabeled topology)."
