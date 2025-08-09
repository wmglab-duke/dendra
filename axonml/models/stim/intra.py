import torch

from .waveform import Waveform


def avoid_smart_indexing(node_indices):
    if node_indices is not None:
        if hasattr(node_indices, "__len__"):
            if len(node_indices) == 1:
                return node_indices[0]
    return node_indices


def _as_index_tensor(indices, n):
    """
    Convert indices to a tensor of indices, ensuring they are within bounds.

    Parameters
    ----------
    indices : int, list, slice, or torch.Tensor
        Indices to convert.
    n : int
        The upper bound for the indices.

    Returns
    -------
    torch.Tensor
        A tensor of indices.
    """
    if isinstance(indices, int):
        return torch.tensor([indices], dtype=torch.long)
    elif isinstance(indices, list):
        return torch.tensor(indices, dtype=torch.long)
    elif isinstance(indices, slice):
        return torch.arange(
            start=indices.start or 0,
            end=indices.stop or n,
            step=indices.step or 1,
            dtype=torch.long,
        )
    elif isinstance(indices, torch.Tensor):
        return indices.to(torch.long)
    else:
        raise TypeError(f"Unsupported index type: {type(indices)}")


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

        for stim, shape, idx in stims:
            if isinstance(stim, Waveform):
                stim = stim.to(device=self.device, dtype=self.dtype)
                stim = stim.expand(shape).reshape_for_intra()
            else:
                raise TypeError(
                    f"Unsupported stimulation type: {type(stim)}. Expected Waveform."
                )
            self.indices.append(idx)
            self.stims.append(stim)

    def init(self, t):
        t = torch.as_tensor(t).to(device=self.device, dtype=self.dtype)
        for _ in range(len(self.shape)):
            t = t.unsqueeze(-1)
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
        intra = torch.zeros(
            self.shape[0], self.shape[1], device=self.device, dtype=self.dtype
        )
        for stim, idx in zip(stims, inds):
            intra[idx] += stim.squeeze()
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
