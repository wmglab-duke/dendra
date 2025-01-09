import torch

from axonml.models.backend import Backend as A


class IntraStim:
    def __init__(self, n_axons, n_nodes):
        self.stim_vec = []
        self.stim_callable = []
        self.stim_synapse = []

        self.dt = A.dt

        self.n_nodes = n_nodes
        self.n_axons = n_axons
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def init(self, model):
        self.dt = model.dt
        for s in self.stim_synapse:
            s[2].init(model.n_axons, model.n_nodes, model.dt, self.device, self.dtype)

    def float(self):
        self.dtype = torch.float32
        return self

    def double(self):
        self.dtype = torch.float64
        return self

    def cuda(self):
        self.device = torch.device("cuda")
        return self

    def cpu(self):
        self.device = torch.device("cpu")
        return self

    def render_nodes(self, indices):
        if indices is None:
            return slice(0, self.n_nodes)
        return indices

    def render_axons(self, indices):
        if indices is None:
            return slice(0, self.n_axons)
        return indices

    def add_from_vec(self, intra, vec, i):
        axons = vec[0]
        nodes = vec[1]
        val = vec[2][i]
        intra[axons, nodes] += val

    def add_from_callable(self, intra, func, t):
        axons = func[0]
        nodes = func[1]
        val = func[2](t)
        intra[axons, nodes] += val

    def add_from_synapse(self, intra, synapse, t, v):
        axons = synapse[0]
        nodes = synapse[1]
        val = synapse[2](t, v)
        intra[axons, nodes] -= val

    def insert_func(self, axons: None, nodes: None, func):
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(nodes)
        self.stim_callable.append((axon_inds, node_inds, func))

    def insert_vec(self, axons: None, nodes: None, vec):
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(nodes)
        vec = torch.tensor(vec, device=self.device, dtype=self.dtype)
        self.stim_vec.append((axon_inds, node_inds, vec))

    def insert_synapse(self, axons: None, nodes: None, synapse):
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(nodes)
        self.stim_synapse.append((axon_inds, node_inds, synapse))

    def __getitem__(self, idx, v):
        t = self.dt * idx
        
        intra = torch.zeros(
            self.n_axons, self.n_nodes, device=self.device, dtype=self.dtype
        )

        for v in self.stim_vec:
            self.add_from_vec(intra, v, idx)
        for c in self.stim_callable:
            self.add_from_callable(intra, c, t)
        for s in self.stim_synapse:
            self.add_from_synapse(intra, s, t, v)
        
        return intra.unsqueeze(1)
