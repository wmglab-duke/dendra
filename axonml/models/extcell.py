import torch

from .core import Axon
from .integrators import bwd_euler_bt, dhs_bt
from .tree import Tree, gather_morphology


class ExtCell(Axon):
    def __init__(
        self,
        diameters=[10.0],
        n_comp=101,
        celsius=37.0,
        v_init=-80.0,
        n_layers=2,
        integrator=None,
    ):
        if integrator is None:
            integrator = bwd_euler_bt()
        super().__init__(diameters, n_comp, celsius, v_init, integrator)
        self.n_layers = n_layers
        self._register_buffers()
        self.dx[:] = 10.0
        self.x[:] = self._x()

    def _register_buffers(self):
        self.register_buffer(
            "xraxial", torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9)
        )
        self.register_buffer(
            "xc", torch.full((self.n_ax, self.n_comp, self.n_layers), 0.0)
        )
        self.register_buffer(
            "xg", torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9)
        )

    def _x(self):
        node_l = torch.atleast_2d(self.dx.squeeze())
        x = node_l.cumsum(dim=1) - node_l / 2
        return x - torch.sum(node_l, dim=1, keepdim=True) / 2


def gather_extcell(graph, nlayers=2):
    xraxial, xc, xg = [], [], []
    for i in range(len(graph.nodes)):
        attrs = graph.nodes[i]
        xraxial.append(attrs.get("xraxial", [1e9] * nlayers))
        xc.append(attrs.get("xc", [0.0] * nlayers))
        xg.append(attrs.get("xg", [1e9] * nlayers))
    return {
        "xraxial": torch.tensor(xraxial).unsqueeze(0),
        "xc": torch.tensor(xc).unsqueeze(0),
        "xg": torch.tensor(xg).unsqueeze(0),
    }


class ExtCellTree(Tree):
    def __init__(self, N, C, graph, n_layers=2, integrator=None, **kwargs):
        if integrator is None:
            integrator = dhs_bt()
        super().__init__(N, C, graph, integrator, **kwargs)
        if n_layers != 2:
            raise ValueError("Only 2 layers are currently supported.")
        self.n_layers = n_layers
        self._register_buffers()

    def _register_buffers(self):
        self.register_buffer(
            "xraxial", torch.full((self.np, self.nc, self.n_layers), 1e9)
        )
        self.register_buffer("xc", torch.full((self.np, self.nc, self.n_layers), 0.0))
        self.register_buffer("xg", torch.full((self.np, self.nc, self.n_layers), 1e9))

    def load_extcell(self, extcell):
        for key, value in extcell.items():
            getattr(self, key).copy_(value.expand(self.np, -1))

    def load_morphology(self, morphology):
        for key, value in morphology.items():
            self.register_buffer(key, value.expand(self.np, -1))

    @classmethod
    def from_graph(cls, graph, N=1, n_layers=2, integrator=None, **kwargs):
        C = len(graph.nodes)
        morphology = gather_morphology(graph)
        extcell = gather_extcell(graph, n_layers=n_layers)
        tree = cls(N, C, graph, n_layers=n_layers, integrator=integrator, **kwargs)
        tree.load_morphology(morphology)
        tree.load_extcell(extcell)
        tree[:, tree.find("soma")].label("soma")
        tree[:, tree.find("axon")].label("axon")
        tree[:, tree.find("dend")].label("dend")
        tree[:, tree.find("apic")].label("apic")
        return tree

    @classmethod
    def from_NEURON(cls, root_sec=None, N=1, n_layers=2, integrator=None, **kwargs):
        from axonml.models.io import neuron_to_axonml_graph

        graph, _ = neuron_to_axonml_graph(root_sec, extcell=n_layers)
        return cls.from_graph(
            graph, N=N, n_layers=n_layers, integrator=integrator, **kwargs
        )
