import torch

from .waveform import Waveform


def avoid_smart_indexing(node_indices):
    if node_indices is not None:
        if hasattr(node_indices, "__len__"):
            if len(node_indices) == 1:
                return node_indices[0]
    return node_indices


def _canonicalize_index_for_index_put(idx, shape, device, linear_index=None):
    """
    Turn any valid tensor ``__getitem__`` index into a tuple of long tensors
    acceptable to ``Tensor.index_put_``. Handles ints, slices, lists, long
    tensors, 1-D and N-D boolean masks, Ellipsis, and None.

    Returns
    -------
    indices_tuple : tuple[torch.Tensor, ...]
        One tensor per dimension of 'shape'; all broadcastable to a common shape.
    """
    shape = tuple(int(size) for size in shape)
    if not shape:
        raise ValueError("Intracellular stimulation requires a non-scalar model.")

    def move_tensor_indices(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(move_tensor_indices(item) for item in value)
        return value

    idx = move_tensor_indices(idx)

    # Apply the user's index to a linear coordinate grid.  This delegates the
    # subtle mix of basic, advanced, boolean, Ellipsis, and None semantics to
    # PyTorch itself, then converts the selected linear locations into the
    # coordinate tuple required by index_put_.  Keeping the selected tensor's
    # shape also preserves Cartesian slice dimensions for value broadcasting.
    if linear_index is None:
        numel = 1
        for size in shape:
            numel *= size
        linear_index = torch.arange(numel, device=device, dtype=torch.long).reshape(
            shape
        )
    elif tuple(linear_index.shape) != shape:
        raise ValueError(
            f"linear_index has shape {tuple(linear_index.shape)}, expected {shape}."
        )
    selected = linear_index[idx]
    return tuple(torch.unravel_index(selected, shape))


class Intra(torch.nn.Module):
    def __init__(self, model, stims):
        """
        Initialize intracellular stimulation handler.

        Parameters
        ----------
        model : Axon
            The axon model to which this stimulation will be applied.
        """
        super(Intra, self).__init__()
        self.shape = model.v.shape
        self.dtype = model.dtype()
        self.device = model.device()

        self.indices = []
        self.stims = []

        numel = 1
        for size in self.shape:
            numel *= int(size)
        linear_index = torch.arange(
            numel, device=self.device, dtype=torch.long
        ).reshape(self.shape)

        for stim, _, idx in stims:
            if isinstance(stim, Waveform):
                stim = stim.to(device=self.device, dtype=self.dtype)
            else:
                raise TypeError(
                    f"Unsupported stimulation type: {type(stim)}. Expected Waveform."
                )
            self.indices.append(
                _canonicalize_index_for_index_put(
                    idx,
                    self.shape,
                    device=self.device,
                    linear_index=linear_index,
                )
            )
            self.stims.append(stim)

    def init(self, t):
        t = torch.as_tensor(t).to(device=self.device, dtype=self.dtype)
        wavs = [stim(t).to(self.dtype) for stim in self.stims]
        inds = self.indices
        return wavs, inds

    def __call__(self, stims, inds):
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
            Tensor of intracellular current values with shape [n_cells, n_comps].
        """
        intra = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        for stim, idx in zip(stims, inds):
            bshape = ()
            if len(idx) > 0:
                bshape = torch.broadcast_shapes(*[t.shape for t in idx])
                stim = stim.expand(bshape)
            intra.index_put_(idx, stim, accumulate=True)
        return intra


def n(obj):
    """
    Return the 'length' of obj:
      - if obj is an int, return 1
      - if obj is a list, return len(obj)
      - if obj is a slice, compute how many indices it would produce

    Raises ValueError for unsupported types.
    """
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
