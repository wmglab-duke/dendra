import torch

from axonml.models.backend import Backend as A


from .synapse import Synapse


def avoid_smart_indexing(node_indices):
    if node_indices is not None:
        if hasattr(node_indices, "__len__"):
            if len(node_indices) == 1:
                return node_indices[0]
    return node_indices


class IntraStim:
    """
    Intracellular stimulation handler for axon models.

    This class manages intracellular current injections into axon models during simulation.
    It supports three types of stimulation:
    1. Vector-based: pre-defined current values for each time step
    2. Callable-based: functions that compute current values based on time
    3. Synapse-based: synaptic mechanisms that compute current values based on time and voltage

    Parameters
    ----------
    model : Axon
        The axon model to which this stimulation will be applied.

    Attributes
    ----------
    stim_vec : list
        List of vector-based stimulations as (axons, nodes, current_values) tuples.
    stim_callable : list
        List of callable-based stimulations as (axons, nodes, function) tuples.
    stim_synapse : list
        List of synapse-based stimulations as (axons, nodes, synapse) tuples.
    dt : float
        Time step size in milliseconds.
    n_comps : int
        Number of nodes in the model.
    n_axons : int
        Number of axons in the model.
    device : torch.device
        Computation device (CPU or CUDA).
    dtype : torch.dtype
        Data type for computations.

    Methods
    -------
    init(model)
        Reinitialize parameters from the model and initialize synapses.
    float()
        Set data type to single precision (float32).
    double()
        Set data type to double precision (float64).
    cuda()
        Set computation device to CUDA.
    cpu()
        Set computation device to CPU.
    insert(obj, axons=None, nodes=None)
        Insert a stimulation object (vector, callable, or synapse).
    insert_func(func, axons=None, nodes=None)
        Insert a callable function for stimulation.
    insert_vec(vec, axons=None, nodes=None)
        Insert a vector of pre-defined stimulation values.
    insert_synapse(synapse, axons=None, nodes=None)
        Insert a synaptic mechanism for stimulation.
    __call__(idx, vm)
        Compute total intracellular current at the given time index.

    Notes
    -----
    The class provides a flexible framework for defining complex stimulation patterns
    by combining multiple stimulation sources. When used in a simulation, the model
    calls this object to get the total intracellular current at each time step.
    """

    def __init__(self, model):
        """
        Initialize intracellular stimulation handler.

        Parameters
        ----------
        model : Axon
            The axon model to which this stimulation will be applied.
        """
        self.stim_vec = []
        self.stim_callable = []
        self.stim_synapse = []

        self.dt = model.dt

        self.n_comps = model.n_comp
        self.n_axons = model.n_ax
        self.device = model.device()
        self.dtype = model.dtype()

    def init(self, model):
        """
        Reinitialize parameters from the model and initialize synapses.

        This method is called before simulation to update internal parameters
        and initialize all synapse objects with the correct dimensions and properties.

        Parameters
        ----------
        model : Axon
            The axon model to update parameters from.
        """
        self.dt = model.dt
        self.device = model.device()
        self.dtype = model.dtype()
        self.n_axons = model.n_ax
        self.n_comps = model.n_comp

        for ax, node, synapse in self.stim_synapse:
            n_ax = n(ax)
            n_comp = n(node)
            synapse.init(n_ax, n_comp, self.dt, self.device, self.dtype)

    def float(self):
        """
        Set data type to single precision (float32).

        Returns
        -------
        self : IntraStim
            Returns self for method chaining.
        """
        self.dtype = torch.float32
        return self

    def double(self):
        """
        Set data type to double precision (float64).

        Returns
        -------
        self : IntraStim
            Returns self for method chaining.
        """
        self.dtype = torch.float64
        return self

    def cuda(self):
        """
        Set computation device to CUDA.

        Returns
        -------
        self : IntraStim
            Returns self for method chaining.
        """
        self.device = torch.device("cuda")
        return self

    def cpu(self):
        """
        Set computation device to CPU.

        Returns
        -------
        self : IntraStim
            Returns self for method chaining.
        """
        self.device = torch.device("cpu")
        return self

    def render_nodes(self, indices):
        """
        Process node indices for stimulation targeting.

        Parameters
        ----------
        indices : int, list, or slice, optional
            Indices of nodes to target. If None, targets all nodes.

        Returns
        -------
        int, list, or slice
            Processed node indices.
        """
        if indices is None:
            return slice(0, self.n_comps)
        return indices

    def render_axons(self, indices):
        """
        Process axon indices for stimulation targeting.

        Parameters
        ----------
        indices : int, list, or slice, optional
            Indices of axons to target. If None, targets all axons.

        Returns
        -------
        int, list, or slice
            Processed axon indices.
        """
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
        val = func[2](t).squeeze()
        intra[axons, nodes] += val

    def add_from_synapse(self, intra, synapse, t, v):
        axons = synapse[0]
        nodes = synapse[1]
        val = synapse[2](t, v)
        intra[axons, nodes] -= val

    def insert(self, obj, axons=None, nodes=None):
        """
        Insert a stimulation object (vector, callable, or synapse).

        This is a general-purpose method that detects the object type
        and calls the appropriate specialized insert method.

        Parameters
        ----------
        obj : array_like, callable, or Synapse
            The stimulation object to insert.
        axons : int, list, or slice, optional
            Indices of axons to target. If None, targets all axons.
        nodes : int, list, or slice, optional
            Indices of nodes to target. If None, targets all nodes.
        """
        if isinstance(obj, Synapse):
            self.insert_synapse(obj, axons, nodes)
        elif callable(obj):
            self.insert_func(obj, axons, nodes)
        else:
            self.insert_vec(obj, axons, nodes)

    def insert_func(self, func, axons=None, nodes=None):
        """
        Insert a callable function for stimulation.

        Parameters
        ----------
        func : callable
            A function that takes time (in ms) as input and returns
            a current value or array of current values.
        axons : int, list, or slice, optional
            Indices of axons to target. If None, targets all axons.
        nodes : int, list, or slice, optional
            Indices of nodes to target. If None, targets all nodes.
        """
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(avoid_smart_indexing(nodes))
        self.stim_callable.append((axon_inds, node_inds, func))

    def insert_vec(self, vec, axons=None, nodes=None):
        """
        Insert a vector of pre-defined stimulation values.

        Parameters
        ----------
        vec : array_like
            A vector or array of current values for each time step.
        axons : int, list, or slice, optional
            Indices of axons to target. If None, targets all axons.
        nodes : int, list, or slice, optional
            Indices of nodes to target. If None, targets all nodes.
        """
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(avoid_smart_indexing(nodes))
        vec = torch.as_tensor(vec, device=self.device, dtype=self.dtype).squeeze()
        self.stim_vec.append((axon_inds, node_inds, vec))

    def insert_synapse(self, synapse, axons=None, nodes=None):
        """
        Insert a synaptic mechanism for stimulation.

        Parameters
        ----------
        synapse : Synapse
            A synapse object that computes current based on time and voltage.
        axons : int, list, or slice, optional
            Indices of axons to target. If None, targets all axons.
        nodes : int, list, or slice, optional
            Indices of nodes to target. If None, targets all nodes.
        """
        axon_inds = self.render_axons(axons)
        node_inds = self.render_nodes(avoid_smart_indexing(nodes))
        self.stim_synapse.append((axon_inds, node_inds, synapse))

    def __call__(self, idx: int, vm):
        """
        Compute total intracellular current at the given time index.

        This method is called by the model during simulation to get
        the total intracellular current for the current time step.

        Parameters
        ----------
        idx : int
            Current time index in the simulation.
        vm : torch.Tensor
            Current membrane potential values.

        Returns
        -------
        torch.Tensor
            Tensor of intracellular current values with shape [n_axons, 1, n_comps].
        """
        t = self.dt * idx

        intra = torch.zeros(
            self.n_axons, self.n_comps, device=self.device, dtype=self.dtype
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
