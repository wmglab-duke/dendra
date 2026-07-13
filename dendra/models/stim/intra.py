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


def _retained_leading_batch_shape(indices, batch_shape):
    """Return batch axes proven to remain a full leading Cartesian prefix.

    Basic full-batch selections produced by ``Population`` and ``Network``
    retain this layout.  Integer, ``None``, boolean, or advanced indexing can
    remove or reorder axes, so shape alone is not enough to identify them.
    Conservatively disabling the batch-only fallback in those cases prevents a
    neuron/compartment axis from being silently mistaken for a batch axis.
    """
    batch_shape = tuple(int(size) for size in batch_shape)
    if not batch_shape or not indices:
        return ()

    selection_shape = tuple(torch.broadcast_shapes(*[item.shape for item in indices]))
    batch_rank = len(batch_shape)
    if len(selection_shape) < batch_rank or selection_shape[:batch_rank] != batch_shape:
        return ()

    for axis, size in enumerate(batch_shape):
        view_shape = [1] * len(selection_shape)
        view_shape[axis] = size
        expected = torch.arange(
            size, device=indices[axis].device, dtype=indices[axis].dtype
        ).reshape(view_shape)
        if not torch.equal(indices[axis], expected.expand(selection_shape)):
            return ()

    # The spatial selection must be shared across the retained batch grid.
    # Advanced indexing can otherwise pair a different compartment with each
    # batch coordinate while leaving an apparently valid leading dimension.
    for spatial_index in indices[batch_rank:]:
        for axis, size in enumerate(batch_shape):
            if size <= 1:
                continue
            first = spatial_index.select(axis, 0).unsqueeze(axis)
            if not torch.equal(spatial_index, first.expand(selection_shape)):
                return ()
    return batch_shape


def _expand_stimulus_to_selection(stim, selection_shape, batch_shape=()):
    """Broadcast one waveform sample to its indexed model selection.

    Ordinary PyTorch trailing broadcasting is tried first, preserving inputs
    such as ``[C]`` for a ``[B, N, C]`` selection.  If that fails, a low-rank
    value may broadcast to the model's explicit batch shape and is then padded
    with singleton selection axes.  Thus ``[B]`` works for a soma selection of
    shape ``[B, 1, 1]``.  Explicit singleton axes disambiguate intent when batch
    and spatial dimensions happen to have the same size.
    """
    selection_shape = tuple(int(size) for size in selection_shape)
    stimulus_shape = tuple(stim.shape)
    batch_shape = tuple(int(size) for size in batch_shape)

    if any(size == 0 for size in selection_shape):
        # There is nothing to write.  Returning an empty value with the exact
        # selection shape also keeps index_put_ happy for arbitrary empty masks.
        return stim.new_empty(selection_shape)

    try:
        return torch.broadcast_to(stim, selection_shape)
    except RuntimeError as exc:
        trailing_error = exc

    batch_rank = len(batch_shape)
    if batch_rank and 0 < stim.ndim <= batch_rank:
        # Standard Population/Network selections retain the explicit batch axes
        # at the front.  Right-align within that prefix so an inner sweep [B]
        # survives a later outer batch(), becoming [outer, B, ...].
        selected_batch_shape = selection_shape[:batch_rank]
        if len(selected_batch_shape) == batch_rank:
            try:
                batch_value = torch.broadcast_to(stim, selected_batch_shape)
                candidate = batch_value.reshape(
                    *selected_batch_shape,
                    *(1,) * (len(selection_shape) - batch_rank),
                )
                return candidate.expand(selection_shape)
            except RuntimeError:
                pass

    raise ValueError(
        "Intracellular waveform sample shape "
        f"{stimulus_shape} cannot broadcast to selected model shape "
        f"{selection_shape}. Ordinary trailing PyTorch broadcasting is tried "
        "first; a batch-only fallback accepts values broadcastable to explicit "
        f"batch shape {batch_shape}. Use singleton axes to disambiguate batch "
        "and spatial intent."
    ) from trailing_error


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
        self.batch_shape = tuple(model.v.shape[:-2])
        self.dtype = model.dtype()
        self.device = model.device()

        self.indices = []
        self.selected_batch_shapes = []
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
            indices = _canonicalize_index_for_index_put(
                idx,
                self.shape,
                device=self.device,
                linear_index=linear_index,
            )
            self.indices.append(indices)
            self.selected_batch_shapes.append(
                _retained_leading_batch_shape(indices, self.batch_shape)
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
            Tensor of intracellular current values with the full model shape
            ``[*batch, n_cells, n_comps]``.  Each per-step waveform value may be
            scalar, exactly match its indexed selection, use ordinary trailing
            broadcasting, or (when that fails) broadcast over explicit batch
            axes (``[B]`` to ``[B, 1, 1]``).  Use explicit singleton axes when
            equal-sized batch and spatial axes would otherwise be ambiguous.
        """
        intra = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        for stim, idx, selected_batch_shape in zip(
            stims, inds, self.selected_batch_shapes
        ):
            selection_shape = ()
            if len(idx) > 0:
                selection_shape = torch.broadcast_shapes(*[t.shape for t in idx])
                stim = _expand_stimulus_to_selection(
                    stim, selection_shape, selected_batch_shape
                )
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
