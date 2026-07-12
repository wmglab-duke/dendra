"""Extended extracellular coupling models.

The two supported extracellular layers follow NEURON's ``extracellular``
mechanism convention. ``xraxial`` is longitudinal resistance in MOhm/cm,
``xc`` is radial capacitance density in uF/cm2, and ``xg`` is radial
conductance density in S/cm2.  A prescribed ``extra``/``ve`` value is the bath
battery ``e_extracellular`` in mV outside the outermost modeled layer.

The block solvers store absolute circuit-node potentials as
``vc[..., 0] = vi``, ``vc[..., 1] = vext[0]``, and
``vc[..., 2] = vext[1]``.  Public membrane voltage therefore follows NEURON's
sign convention, ``v = vi - vext[0]``.
"""

import torch

from .core import Axon
from .integrators import bwd_euler_bt, dhs_bt
from .tree import Tree, gather_diffusion_edges, gather_membrane, gather_morphology


class ExtCellAxon(Axon):
    """Axon model with two-layer extracellular coupling.

    Parameters
    ----------
    diameters : Sequence[float], optional
        Compartment diameters in micrometers.
    n_comp : int, optional
        Number of compartments per axon.
    celsius : float, optional
        Simulation temperature in degrees Celsius.
    v_init : float, optional
        Initial membrane voltage in millivolts.
    n_layers : int, optional
        Number of extracellular layers. Only ``2`` is currently supported.
    integrator : callable, optional
        Integrator factory used to create the simulation solver.
    **kwargs
        Additional population parameters such as ``dtype``, ``device``,
        ``rhoa``, and ``cm`` forwarded to :class:`~dendra.models.core.Axon`.
    """

    def __init__(
        self,
        diameters=[10.0],
        n_comp=101,
        celsius=37.0,
        v_init=-80.0,
        n_layers=2,
        integrator=None,
        **kwargs,
    ):
        if n_layers != 2:
            raise ValueError("Only 2 layers are currently supported.")
        if integrator is None:
            integrator = bwd_euler_bt()
        super().__init__(diameters, n_comp, celsius, v_init, integrator, **kwargs)
        self.n_layers = n_layers
        self._register_buffers()
        self.dx[:] = 10.0
        self.x[:] = self._x()

    def _register_buffers(self):
        """Initialise extracellular parameter buffers."""
        options = {"device": self.device(), "dtype": self.dtype()}
        self.register_buffer(
            "xraxial",
            torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9, **options),
        )
        self.register_buffer(
            "xc", torch.full((self.n_ax, self.n_comp, self.n_layers), 0.0, **options)
        )
        self.register_buffer(
            "xg", torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9, **options)
        )

    def _x(self):
        """Compute compartment midpoints centered along the axon."""
        dtype = self.dx.dtype
        node_l = torch.atleast_2d(self.dx.squeeze().to(torch.double))
        x = node_l.cumsum(dim=1) - node_l / 2
        x = x - torch.sum(node_l, dim=1, keepdim=True) / 2
        return x.to(dtype)

    def assemble_graphs(self):
        """Export the axon morphology and parameters as a graph.

        Returns
        -------
        networkx.Graph
            Morphology graph with extracellular attributes.
        """
        graphs = super().assemble_graphs()
        for i in range(self.n_ax):
            for node in graphs[i].nodes:
                graphs[i].nodes[node]["xraxial"] = self.xraxial[i, node].tolist()
                graphs[i].nodes[node]["xc"] = self.xc[i, node].tolist()
                graphs[i].nodes[node]["xg"] = self.xg[i, node].tolist()
        return graphs


def gather_extcell(graph, n_layers=2):
    """Collect extracellular parameters from a morphology graph.

    Parameters
    ----------
    graph : networkx.Graph
        Morphology graph whose nodes contain extracellular attributes.
    n_layers : int, optional
        Number of extracellular layers to gather. Defaults to ``2``.

    Returns
    -------
    dict
        Mapping of parameter names to tensors shaped ``(1, n_comp, n_layers)``.
    """
    xraxial, xc, xg = [], [], []
    for i in range(len(graph.nodes)):
        attrs = graph.nodes[i]
        xraxial.append(attrs.get("xraxial", [1e9] * n_layers))
        xc.append(attrs.get("xc", [0.0] * n_layers))
        xg.append(attrs.get("xg", [1e9] * n_layers))
    return {
        "xraxial": torch.tensor(xraxial).unsqueeze(0),
        "xc": torch.tensor(xc).unsqueeze(0),
        "xg": torch.tensor(xg).unsqueeze(0),
    }


class ExtCellTree(Tree):
    """Tree model supporting extracellular coupling layers.

    Parameters
    ----------
    N : int
        Number of tree instances (populations).
    C : int
        Number of compartments per tree.
    graph : networkx.Graph
        Morphology graph describing compartment connections.
    n_layers : int, optional
        Number of extracellular layers. Only ``2`` is currently supported.
    integrator : callable, optional
        Integrator factory used to create the solver.
    **kwargs
        Additional parameters forwarded to :class:`Tree`.
    """

    def __init__(self, N, C, graph, n_layers=2, integrator=None, **kwargs):
        if n_layers != 2:
            raise ValueError("Only 2 layers are currently supported.")
        if integrator is None:
            integrator = dhs_bt()
        super().__init__(N, C, graph, integrator, **kwargs)
        self.n_layers = n_layers
        self._register_buffers()

    def _register_buffers(self):
        """Initialise extracellular buffers for the tree morphology."""
        options = {"device": self.device(), "dtype": self.dtype()}
        self.register_buffer(
            "xraxial",
            torch.full((self.np, self.nc, self.n_layers), 1e9, **options),
        )
        self.register_buffer(
            "xc", torch.full((self.np, self.nc, self.n_layers), 0.0, **options)
        )
        self.register_buffer(
            "xg", torch.full((self.np, self.nc, self.n_layers), 1e9, **options)
        )

    def load_extcell(self, extcell):
        """Load extracellular parameters into buffers.

        Parameters
        ----------
        extcell : dict[str, torch.Tensor]
            Mapping of extracellular parameter names to tensors shaped
            ``(1, n_comp, n_layers)``.
        """
        for key, value in extcell.items():
            getattr(self, key).copy_(value.expand(self.np, -1, -1))

    def load_morphology(self, morphology):
        """Register morphology tensors as buffers.

        Parameters
        ----------
        morphology : dict[str, torch.Tensor]
            Mapping of morphology parameter names to tensors shaped
            ``(1, n_comp)``.
        """
        for key, value in morphology.items():
            expanded = value.expand(self.np, -1).clone()
            self.register_buffer(
                key, expanded.to(device=self.device(), dtype=self.dtype())
            )

    def load_diffusion_edges(self, diffusion_edges):
        """Register child-indexed and compact tree-diffusion metadata."""
        for key, value in diffusion_edges.items():
            if value.dtype.is_floating_point:
                if value.ndim == 2:
                    value = value.expand(self.np, -1).clone()
                else:
                    value = value.clone()
                value = value.to(device=self.device(), dtype=self.dtype())
            else:
                value = value.clone().to(device=self.device())
            self.register_buffer(key, value)

    @classmethod
    def from_graph(cls, graph, N=1, n_layers=2, integrator=None, **kwargs):
        """Construct an extracellular tree model from a morphology graph.

        Parameters
        ----------
        graph : networkx.Graph
            Morphology graph with extracellular attributes.
        N : int, optional
            Number of population instances. Defaults to ``1``.
        n_layers : int, optional
            Number of extracellular layers. Supports only ``2`` at present.
        integrator : callable, optional
            Integrator factory. Defaults to backward Euler branching tree.
        **kwargs
            Additional membrane parameters forwarded to :class:`Tree`.

        Returns
        -------
        ExtCellTree
            Configured tree population with extracellular coupling.
        """
        C = len(graph.nodes)
        morphology = gather_morphology(graph)
        diffusion_edges = gather_diffusion_edges(graph)
        membrane = gather_membrane(graph)
        membrane.update(kwargs)
        extcell = gather_extcell(graph, n_layers=n_layers)
        tree = cls(N, C, graph, n_layers=n_layers, integrator=integrator, **membrane)
        tree.load_morphology(morphology)
        tree.load_diffusion_edges(diffusion_edges)
        tree.load_extcell(extcell)
        tree.slice("soma").label("soma")
        tree.slice("axon").label("axon")
        tree.slice("dend").label("dend")
        tree.slice("apic").label("apic")
        return tree

    @classmethod
    def from_NEURON(cls, root_sec=None, N=1, n_layers=2, integrator=None, **kwargs):
        """Construct an extracellular tree from a NEURON section.

        Parameters
        ----------
        root_sec : neuron.h.Section, optional
            Root section from a NEURON model.
        N : int, optional
            Number of population instances. Defaults to ``1``.
        n_layers : int, optional
            Number of extracellular layers. Supports only ``2`` at present.
        integrator : callable, optional
            Integrator factory to use for the tree.
        **kwargs
            Additional keyword arguments forwarded to :meth:`from_graph`.

        Returns
        -------
        ExtCellTree
            Constructed extracellular tree instance.
        """
        from dendra.models.io import neuron_to_dendra_graph

        graph, _ = neuron_to_dendra_graph(root_sec, extcell=n_layers)
        return cls.from_graph(
            graph, N=N, n_layers=n_layers, integrator=integrator, **kwargs
        )

    def assemble_graphs(self):
        return [self.graph]
