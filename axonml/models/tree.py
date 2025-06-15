import torch

from axonml.models.declarations import PARAMETER
from axonml.models.integrators import dhs

from .core import Population


def gather_morphology(graph):
    # iterate through nodes and gather morphology data
    L, diam, rhoa, cm, x, y, z = [], [], [], [], [], [], []
    for node, attrs in graph.nodes(data=True):
        L       .append(attrs.get('L'))
        diam    .append(attrs.get('diam'))
        rhoa    .append(attrs.get('Ra'))
        cm      .append(attrs.get('cm'))
        x       .append(attrs.get('x', 0.0))
        y       .append(attrs.get('y', 0.0))
        z       .append(attrs.get('z', 0.0))
    return {
        'dx':   torch.tensor(L,     dtype=torch.float32).unsqueeze(0),
        'diam': torch.tensor(diam,  dtype=torch.float32).unsqueeze(0),
        'rhoa': torch.tensor(rhoa,  dtype=torch.float32).unsqueeze(0),
        'cm':   torch.tensor(cm,    dtype=torch.float32).unsqueeze(0),
        'x':    torch.tensor(x,     dtype=torch.float32).unsqueeze(0),
        'y':    torch.tensor(y,     dtype=torch.float32).unsqueeze(0),
        'z':    torch.tensor(z,     dtype=torch.float32).unsqueeze(0),
    }


class Tree(Population):
    """
    Base class for tree-like structures in axonal models.

    This class serves as a foundation for creating tree structures that can
    represent branching axons or dendrites in neural models. It inherits from
    the Population class, allowing it to utilize population-level features.

    Parameters
    ----------
    name : str
        Name of the tree structure.
    nodes : int
        Number of nodes in the tree.
    """

    PARAMETER(celsius=37.0)
    
    def __init__(self, N, C, graph=None, integrator=None, **kwargs):
        if integrator is None:
            integrator = dhs()
        super().__init__(N, C, integrator=integrator, **kwargs)
        self._graph = graph

    @property
    def graph(self):
        """
        Returns the graph structure of the tree.

        Returns
        -------
        networkx.DiGraph
            The directed graph representing the tree structure.
        """
        return self._graph
        
    @classmethod
    def from_graph(cls, graph, N=1, integrator=None, **kwargs):
        """
        Create a Tree instance from a graph structure.

        Parameters
        ----------
        graph : networkx.DiGraph
            A graph representing the tree structure.
        integrator : Integrator, optional
            The integrator to use for the model. Defaults to None.

        Returns
        -------
        Tree
            An instance of the Tree class.
        """
        C = len(graph.nodes)
        data = gather_morphology(graph)
        tree = cls(N, C, graph, integrator, **kwargs)
        for key, value in data.items():
            tree.register_buffer(key, value.expand(N, -1))
        return tree

    def recentre(self, x=0.0, y=0.0, z=0.0):
        """
        Recenters the tree structure so soma is at the origin.
        Parameters
        ----------
        x : float, optional
            X-coordinate of the new center. Default is 0.0.
        y : float, optional
            Y-coordinate of the new center. Default is 0.0.
        z : float, optional
            Z-coordinate of the new center. Default is 0.0.
        """ 
        current_centre_x = self.x[:, 0]
        current_centre_y = self.y[:, 0]
        current_centre_z = self.z[:, 0]

        offsets = [
            x - current_centre_x,
            y - current_centre_y,
            z - current_centre_z
        ]
        
        self.x += offsets[0]
        self.y += offsets[1]
        self.z += offsets[2]

    def move(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Moves the tree structure by specified offsets.

        Parameters
        ----------
        dx : float, optional
            Offset in the x-direction. Default is 0.0.
        dy : float, optional
            Offset in the y-direction. Default is 0.0.
        dz : float, optional
            Offset in the z-direction. Default is 0.0.
        """
        self.x += dx
        self.y += dy
        self.z += dz