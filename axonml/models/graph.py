import networkx as nx
import torch


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
        area = G.nodes[n].get('area', None)
        if area is not None:
            areas.append(area)
        else:
            return None  # If any node lacks area, return None

    return 1e-8 * torch.tensor(areas)