import torch

from .waveform import Waveform


def avoid_smart_indexing(node_indices):
    if node_indices is not None:
        if hasattr(node_indices, "__len__"):
            if len(node_indices) == 1:
                return node_indices[0]
    return node_indices


def _ensure_tuple(idx):
    return idx if isinstance(idx, tuple) else (idx,)


def _expand_ellipsis(idx_tuple, ndim):
    if Ellipsis not in idx_tuple:
        # pad with full slices to ndim once we drop Nones
        used = sum(1 for x in idx_tuple if x is not None)
        return idx_tuple + (slice(None),) * (ndim - used)
    out = []
    # count entries that actually consume a dimension (ignore None, Ellipsis)
    used = sum(1 for x in idx_tuple if (x is not None and x is not Ellipsis))
    need = max(0, ndim - used)
    for x in idx_tuple:
        if x is Ellipsis:
            out.extend([slice(None)] * need)
        else:
            out.append(x)
    return tuple(out)


def _drop_none(idx_tuple):
    # None (newaxis) affects result/view shape in __getitem__, but for assignment
    # via index_put_ we only need coordinates; broadcasting of values covers shape.
    return tuple(x for x in idx_tuple if x is not None)


def _canonicalize_index_for_index_put(idx, shape, device):
    """
    Turn any valid __getitem__ index into a tuple of long/bool tensors acceptable
    to Tensor.index_put_. Handles: ints, slices (incl. negative step), lists,
    long tensors, 1-D and N-D boolean masks, Ellipsis, None.

    Returns
    -------
    indices_tuple : tuple[torch.Tensor, ...]
        One tensor per dimension of 'shape'; all broadcastable to a common shape.
    """
    ndim = len(shape)
    idx = _ensure_tuple(idx)
    idx = _expand_ellipsis(idx, ndim)
    idx = _drop_none(idx)

    # if shorter than ndim, pad with full slices
    if len(idx) < ndim:
        idx = idx + (slice(None),) * (ndim - len(idx))

    out_indices = []
    dim_ptr = 0

    def arange_dim(sz):
        return torch.arange(sz, device=device, dtype=torch.long)

    while dim_ptr < ndim:
        sel = idx[dim_ptr]

        if isinstance(sel, slice):
            start, stop, step = sel.indices(shape[dim_ptr])
            out_indices.append(
                torch.arange(start, stop, step, device=device, dtype=torch.long)
            )
            dim_ptr += 1
            continue

        if isinstance(sel, int):
            # normalize negative and wrap into length-1 long tensor
            i = sel if sel >= 0 else shape[dim_ptr] + sel
            out_indices.append(torch.tensor([i], device=device, dtype=torch.long))
            dim_ptr += 1
            continue

        if isinstance(sel, list):
            out_indices.append(torch.as_tensor(sel, device=device, dtype=torch.long))
            dim_ptr += 1
            continue

        if isinstance(sel, torch.Tensor):
            if sel.dtype == torch.bool:
                # Boolean mask may cover 1 or more dims starting at dim_ptr.
                m = sel.ndim
                # Validate that mask fits the upcoming dimensions
                if m == 0:
                    # scalar bool -> either select entire dim (True) or select none (False)
                    if bool(sel.item()):
                        out_indices.append(arange_dim(shape[dim_ptr]))
                    else:
                        out_indices.append(
                            torch.empty(0, device=device, dtype=torch.long)
                        )
                    dim_ptr += 1
                else:
                    # Ensure shapes match the next m dims
                    expected = tuple(shape[dim_ptr : dim_ptr + m])
                    if tuple(sel.shape) != expected:
                        raise IndexError(
                            f"Boolean mask of shape {tuple(sel.shape)} does not match "
                            f"indexed dims {expected} starting at dim {dim_ptr}"
                        )
                    # Replace these m dims with per-dim index tensors from nonzero
                    nz = torch.nonzero(sel, as_tuple=True)
                    for t in nz:
                        out_indices.append(t.to(device=device))
                    dim_ptr += m
                continue
            else:
                # integer-like tensor (long/int)
                t = sel.to(device=device, dtype=torch.long)
                out_indices.append(t)
                dim_ptr += 1
                continue

        if sel is Ellipsis:
            # already expanded
            raise RuntimeError("Internal error: Ellipsis should have been expanded.")
        if sel is None:
            # already dropped
            dim_ptr += 0
            continue

        # Fallback: full selection on this dim
        out_indices.append(arange_dim(shape[dim_ptr]))
        dim_ptr += 1

    # Broadcast all index tensors to a common shape (advanced indexing rule)
    if out_indices:
        bshape = torch.broadcast_shapes(*[t.shape for t in out_indices])
        out_indices = [t.expand(bshape) for t in out_indices]
    else:
        out_indices = ()

    return tuple(out_indices)


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

        for stim, _, idx in stims:
            if isinstance(stim, Waveform):
                stim = stim.to(device=self.device, dtype=self.dtype)
            else:
                raise TypeError(
                    f"Unsupported stimulation type: {type(stim)}. Expected Waveform."
                )
            self.indices.append(
                _canonicalize_index_for_index_put(idx, self.shape, device=self.device)
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
