import networkx as nx
import torch


def distance(cell, origin, targets):
    """
    Calculate the distance between two nodes in a cell.

    Parameters
    ----------
    cell : Cell
        The cell object containing the graph.
    idx1 : int
        The index of the first node.
    idx2 : int
        The index of the second node.

    Returns
    -------
    float
        The distance between the two nodes.
    """
    graph = cell.graph
    if graph is None:
        raise ValueError("Graph is not defined for this cell.")
    if isinstance(origin, slice):
        origin = list(range(len(graph.nodes)))[origin][0]
    if isinstance(targets, slice):
        targets = list(range(len(graph.nodes)))[targets]
    return undirected_weighted_lengths(graph, origin, targets)


def undirected_weighted_lengths(G: nx.DiGraph, origin, targets, weight_attr="L"):
    """
    Parameters
    ----------
    G : nx.DiGraph
        Directed graph whose edges carry an attribute `weight_attr`
        (e.g. "L").
    origin : node label
        The single source node.
    targets : iterable
        An iterable of node labels for which you want distances.
    weight_attr : str, optional
        Name of the edge attribute that stores the length/weight. Default: "L".

    Returns
    -------
    dict
        {target_node: shortest-path length (sum of `weight_attr`), ...}.
        If a target is unreachable it is omitted (or you can map it to
        `float("inf")`, see example).
    """
    # Treat the digraph as *undirected* without copying edge data
    UG = G.to_undirected(as_view=True)

    # One Dijkstra run from the origin gives all lengths in O((V+E) log V)
    lengths = nx.single_source_dijkstra_path_length(
        UG, source=origin, weight=weight_attr
    )

    # Return only the distances you asked for
    return torch.tensor([lengths.get(t, float("inf")) for t in targets])
