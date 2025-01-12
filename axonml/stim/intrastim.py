import torch

from axonml.models.backend import Backend as A


def avoid_smart_indexing(node_indices):
    if node_indices is not None:
        if hasattr(node_indices, "__len__"):
            if len(node_indices) == 1:
                return node_indices[0]
    return node_indices


class IntraStim:
    def __init__(self, model):
        self.stim_vec = []
        self.stim_callable = []
        self.stim_synapse = []

        self.dt = model.dt

        self.n_nodes = model.n_node
        self.n_axons = model.n_ax
        self.device = model.device()
        self.dtype = model.dtype()

    def init(self, model):
        self.dt = model.dt
        self.device = model.device()
        self.dtype = model.dtype()
        self.n_axons = model.n_ax
        self.n_nodes = model.n_node

        for ax, node, synapse in self.stim_synapse:
            n_ax = n(ax)
            n_node = n(node)
            synapse.init(n_ax, n_node, self.dt, self.device, self.dtype)

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
        node_inds = self.render_nodes(avoid_smart_indexing(nodes))
        self.stim_callable.append((axon_inds, node_inds, func))

    def insert_vec(self, axons: None, nodes: None, vec):
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(avoid_smart_indexing(nodes))
        vec = torch.tensor(vec, device=self.device, dtype=self.dtype)
        self.stim_vec.append((axon_inds, node_inds, vec))

    def insert_synapse(self, axons: None, nodes: None, synapse):
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(avoid_smart_indexing(nodes))
        self.stim_synapse.append((axon_inds, node_inds, synapse))

    def __call__(self, idx: int, vm):
        t = self.dt * idx

        intra = torch.zeros(
            self.n_axons, self.n_nodes, device=self.device, dtype=self.dtype
        )

        for v in self.stim_vec:
            self.add_from_vec(intra, v, idx)
        for c in self.stim_callable:
            self.add_from_callable(intra, c, t)
        for s in self.stim_synapse:
            self.add_from_synapse(intra, s, t, vm)

        return intra.unsqueeze(1)


def n(obj):
    """
    Return the 'length' of obj:
      - if obj is an int, return 1
      - if obj is a list, return len(obj)
      - if obj is a slice, compute how many indices it would produce

    Raises ValueError for unsupported types.
    """
    import sys

    # 1) If the object is an integer, length = 1
    if isinstance(obj, int):
        return 1

    # 2) If the object is a list, length = len(obj)
    elif isinstance(obj, list):
        return len(obj)

    # 3) If the object is a slice, compute the length
    elif isinstance(obj, slice):
        # Extract start, stop, step with Python's defaults
        start = obj.start if obj.start is not None else 0
        step = obj.step if obj.step is not None else 1
        if obj.stop is None:
            raise ValueError("Unbounded slice not supported.")
        stop = obj.stop

        # If stop is None and we try to interpret an "unbounded" slice,
        # we must pick some convention. Here we use sys.maxsize (or -sys.maxsize).
        # You might choose to raise an error instead.
        if step > 0:
            length = max(0, (stop - start + step - 1) // step)
        else:
            # step < 0
            length = max(0, (start - stop - step - 1) // abs(step))

        return length

    # If none of the above, raise an error for unsupported types
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")
