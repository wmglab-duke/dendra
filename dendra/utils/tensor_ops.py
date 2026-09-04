"""Tensor-shape helpers for parameter grids and model broadcasting."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

__all__ = ["cartesian_product", "add_dims_as_necessary"]

ArrayLike1D = Union[
    torch.Tensor,
    np.ndarray,
    list,
    tuple,
]


def _logical_tensor_bytes(value: torch.Tensor) -> torch.Tensor:
    """Materialize a tensor's logical value as dense CPU bytes.

    PyTorch may regard a size-one expanded tensor as contiguous even when its
    final stride is zero. Calling ``contiguous()`` is then a no-op, and viewing
    the result as bytes fails because dtype-changing views require a final
    stride of one. An explicit destination allocation guarantees a genuinely
    dense layout for audit fingerprints and bitwise comparisons.
    """

    detached = value.detach()
    if detached.layout != torch.strided:
        detached = detached.to_dense()
    if detached.numel() == 0:
        return torch.empty(0, device="cpu", dtype=torch.uint8)
    dense = torch.empty(
        tuple(detached.shape),
        device="cpu",
        dtype=detached.dtype,
    )
    dense.copy_(detached)
    return dense.reshape(-1).view(torch.uint8)


def cartesian_product(
    *seqs: ArrayLike1D,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
    require_1d: bool = True,
    return_stacked: bool = False,
) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
    """
    Compute a Cartesian product over 1D numerical sequences and return 1D expanded tensors.

    This function takes an arbitrary number of 1D sequences (Python lists/tuples,
    NumPy arrays, or PyTorch tensors) and returns the Cartesian product “grid”
    as *separate 1D tensors* (one per input), each of length:

        N = prod_i len(seqs[i])

    The returned tensors correspond to the flattened coordinates of the full
    Cartesian product, without constructing an intermediate k-dimensional meshgrid.

    Parameters
    ----------
    *seqs : array-like
        Arbitrarily many 1D numerical sequences. Each element may be a list, tuple,
        NumPy array, or torch.Tensor.

        If `require_1d=True`, each sequence must be 1D (or scalar). Scalars are treated
        as length-1 sequences.

    dtype : torch.dtype, optional
        Output dtype. If None, a common promoted dtype is inferred from inputs using
        PyTorch type promotion rules.

        Note: integer + float mixes will promote to float; float32 + float64 mixes will
        promote to float64, etc.

    device : torch.device or str, optional
        Output device. If None:
          - if any input is a torch.Tensor, the device of the first torch.Tensor is used
            (and all torch.Tensor inputs must already be on the same device), otherwise
          - CPU is used.

    require_1d : bool, default=True
        If True, enforce that each input is 1D (or scalar). If False, each input is
        flattened via `.reshape(-1)`.

    return_stacked : bool, default=False
        If False (default), return a tuple of k tensors, each shape (N,).
        If True, return a single tensor of shape (N, k) equivalent to
        `torch.stack(outputs, dim=-1)`.

    Returns
    -------
    outputs : tuple[torch.Tensor, ...] or torch.Tensor
        If `return_stacked=False`:
            A tuple `(x0, x1, ..., x{k-1})` where each `xi` is a 1D tensor of shape (N,)
            representing the i-th coordinate of the Cartesian product.

        If `return_stacked=True`:
            A tensor of shape (N, k), where column i is the i-th coordinate tensor.

    Notes
    -----
    Ordering
    --------
    The ordering matches the standard “nested loops” with the *last* sequence varying fastest:

    >>> for a0 in seq0:
    ...   for a1 in seq1:
    ...     ...
    ...       for a{k-1} in seq{k-1}:
    ...         emit (a0, a1, ..., a{k-1})

    Memory
    ------
    This function materializes k outputs of length N, which is unavoidable if you want
    the full Cartesian product explicitly. It avoids constructing an intermediate
    k-dimensional meshgrid (which can be significantly larger).

    Autograd
    --------
    If any input is a torch.Tensor requiring gradients, the outputs remain differentiable
    w.r.t. those inputs. Repeated elements will accumulate gradients accordingly.

    Examples
    --------
    Basic usage:

    >>> x, y = cartesian_product([1, 2], [10, 20, 30])
    >>> x
    tensor([1, 1, 1, 2, 2, 2])
    >>> y
    tensor([10, 20, 30, 10, 20, 30])

    Three sequences:

    >>> a, b, c = cartesian_product(torch.tensor([0., 1.]), [5, 6], np.array([9, 10, 11]))
    >>> a.shape, b.shape, c.shape
    (torch.Size([12]), torch.Size([12]), torch.Size([12]))

    Stacked output:

    >>> grid = cartesian_product([1, 2], [10, 20], return_stacked=True)
    >>> grid
    tensor([[ 1, 10],
            [ 1, 20],
            [ 2, 10],
            [ 2, 20]])

    Empty sequence yields empty outputs:

    >>> x, y = cartesian_product([], [1, 2, 3])
    >>> x.shape, y.shape
    (torch.Size([0]), torch.Size([0]))
    """
    if len(seqs) == 0:
        raise ValueError("cartesian_product requires at least one input sequence.")

    # Convert inputs to tensors; keep torch tensors as-is to preserve autograd.
    tensors = []
    torch_devices = []
    for s in seqs:
        if isinstance(s, torch.Tensor):
            t = s
            torch_devices.append(t.device)
        else:
            # torch.as_tensor works for list/tuple/np.ndarray scalars and arrays
            t = torch.as_tensor(s)
        # Enforce / normalize shape
        if t.ndim == 0:
            t = t.reshape(1)
        elif require_1d and t.ndim != 1:
            raise ValueError(
                f"Expected 1D inputs when require_1d=True, but got shape {tuple(t.shape)}."
            )
        elif not require_1d and t.ndim != 1:
            t = t.reshape(-1)
        tensors.append(t)

    # Determine device
    if device is None:
        if torch_devices:
            # require all existing torch tensors on same device
            first = torch_devices[0]
            if any(d != first for d in torch_devices[1:]):
                raise ValueError(
                    f"All torch.Tensor inputs must be on the same device when device=None. "
                    f"Got devices: {sorted({str(d) for d in torch_devices})}"
                )
            device = first
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Determine dtype (promote across inputs) if not provided
    if dtype is None:
        # torch.result_type promotes across tensors; start from first
        dt = tensors[0].dtype
        for t in tensors[1:]:
            dt = torch.result_type(
                torch.empty((), dtype=dt), torch.empty((), dtype=t.dtype)
            )
        dtype = dt

    # Cast/move tensors
    tensors = [t.to(device=device, dtype=dtype) for t in tensors]

    sizes = [int(t.numel()) for t in tensors]
    k = len(tensors)

    # Precompute suffix products for repeat patterns
    # repeat_each[i] = product_{j>i} sizes[j]
    # tile[i]        = product_{j<i} sizes[j]
    repeat_each = [1] * k
    tile = [1] * k
    prod_suffix = 1
    for i in range(k - 1, -1, -1):
        repeat_each[i] = prod_suffix
        prod_suffix *= sizes[i]
    prod_prefix = 1
    for i in range(k):
        tile[i] = prod_prefix
        prod_prefix *= sizes[i]
    N = prod_prefix  # total length

    outs = []
    for i, t in enumerate(tensors):
        # Pattern: each element repeats repeat_each times, then the whole block repeats tile times
        # Example sizes [n0,n1,n2]:
        #  i=0: repeat_each=n1*n2, tile=1
        #  i=1: repeat_each=n2,    tile=n0
        #  i=2: repeat_each=1,     tile=n0*n1
        o = t.repeat_interleave(repeat_each[i])
        if tile[i] != 1:
            o = o.repeat(tile[i])
        # Safety check
        if o.numel() != N:
            raise RuntimeError(
                f"Internal error: expected output length {N} but got {o.numel()} for dim {i}."
            )
        outs.append(o)

    if return_stacked:
        return torch.stack(outs, dim=-1)
    return tuple(outs)


def add_dims_as_necessary(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reshape a scalar or vector so that it broadcasts against ``b``.

    Scalars are returned unchanged. For a vector, the first dimension of
    ``b`` whose size matches the vector length is selected and singleton
    dimensions are inserted around it. The returned tensor is a
    broadcast-ready *view*; it is not expanded to the shape of ``b``.

    Parameters
    ----------
    a : torch.Tensor
        A scalar or one-dimensional tensor.
    b : torch.Tensor
        Tensor whose shape determines the desired alignment.

    Returns
    -------
    torch.Tensor
        ``a`` itself when scalar, or a reshaped view of ``a`` when it is a
        vector. Device, dtype, and autograd history are preserved.

    Raises
    ------
    ValueError
        If ``a`` has more than one dimension or no dimension of ``b`` matches
        the vector length.
    """

    if a.ndim == 0:
        return a
    if a.ndim == 1:
        for i, dim in enumerate(b.shape):
            if dim == a.shape[0]:
                return a.reshape((1,) * i + (-1,) + (1,) * (b.ndim - i - 1))
        raise ValueError(
            f"Cannot broadcast 1D tensor of length {a.shape[0]} to shape {b.shape}."
        )
    raise ValueError(f"Input tensor must be 0D or 1D, but got shape {a.shape}.")
