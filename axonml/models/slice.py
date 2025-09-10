from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def expand_into_shape(src, index, shape, fill_value=torch.nan):
    # out is just temporary storage → no grad required
    out = torch.full(shape, fill_value, dtype=src.dtype, device=src.device)
    out[index] = src
    return out


# Anything NumPy or PyTorch accepts in __getitem__
IndexElement = Union[int, slice, np.ndarray, list, tuple]


@dataclass(slots=True)
class IndexSpec:
    """
    Canonical description of one indexing request.

    Attributes
    ----------
    index : Tuple[IndexElement, ...]
        Exactly `ndim` elements, no Ellipsis, no `None` (new-axis) —
        safe to pass straight to `array.__getitem__`.
    is_scalar : bool
        True if the result collapses to a scalar.
    out_shape : Tuple[int, ...]
        What `array[key]` would return, **including** new axes.
    """

    index: Tuple[IndexElement, ...]
    is_scalar: bool
    shape: Tuple[int, ...]


def parse_key(key: Any, shape: Sequence[int], device=None) -> IndexSpec:
    """
    Turn *any* valid key plus `shape` into a reusable IndexSpec.
    Works for NumPy **and** PyTorch rules (they're identical here).

    Examples
    --------
    >>> shape = (4, 5, 6)
    >>> spec = parse_key((Ellipsis, 2, None), shape)
    >>> spec
    IndexSpec(index=(slice(None), slice(None), 2), new_axes=(3,), ...)
    """
    out = torch.empty(shape, device=device)[key]  # type: ignore

    return IndexSpec(
        index=key,
        is_scalar=out.ndim == 0,
        shape=out.shape,
    )


class Slice:
    def __init__(
        self, model, index_spec: IndexSpec, base_shape=None, parent_slice=None
    ):
        self.model: torch.nn.Module = model
        self.base_shape = base_shape
        if base_shape is None:
            self.base_shape = model.shape
        self.index_spec = index_spec
        self.parent_slice = parent_slice

    @property
    def index(self) -> Tuple[IndexElement, ...]:
        """
        Return the index of the slice.
        """
        return self.index_spec.index

    @property
    def shape(self):
        """
        Return the shape of the slice.
        """
        return self.index_spec.shape

    @property
    def is_scalar(self) -> bool:
        """
        Return True if the slice is scalar.
        """
        return self.index_spec.is_scalar

    def numel(self) -> int:
        """
        Return the number of elements in the slice.
        """
        return int(np.prod(self.index_spec.shape))

    @property
    def name(self) -> str:
        return self.model.name

    def inspect(self, var: str, mechanism: Optional[str] = None) -> Any:
        """
        Inspect the variable in the model or a specific mechanism.
        Note: Most of the time, this will involve a memory allocation.
        """
        if mechanism is not None:
            mech = self.model.mech.mechanisms[mechanism]
            if mech.key is None:
                return getattr(mech, var)[self.index_spec.index]
            dummy = torch.tensor(
                torch.nan, device=self.model.device(), dtype=self.model.dtype()
            )
            dummy = mech.put(getattr(mech, var), dummy, self.model.v)
            return dummy[self.index_spec.index]
        return getattr(self.model, var)[self.index_spec.index]

    def get(self, var: str, mechanism: Optional[str] = None) -> torch.Tensor:
        return self.inspect(var, mechanism)

    def set(self, var: str, value: torch.Tensor, mechanism: Optional[str] = None):
        """
        Set the variable in the model or a specific mechanism.
        The value must match the shape of the slice.
        """
        if mechanism is not None:
            mech = self.model.mech.mechanisms[mechanism]
            if mech.key is None:
                getattr(mech, var)[self.index_spec.index] = value
                getattr(mech, var).detach_()
                return
            dummy = torch.tensor(
                torch.nan, device=self.model.device(), dtype=self.model.dtype()
            )
            dummy = mech.put(getattr(mech, var), dummy, self.model.v)
            dummy[self.index_spec.index] = value
            getattr(mech, var).copy_(mech.get(dummy))
            getattr(mech, var).detach_()
            return
        getattr(self.model, var)[self.index_spec.index] = value
        setattr(self.model, var, self.model.getattr(self.model, var).detach())

    def inject(self, waveform):
        self.model.injections.append(
            (waveform, self.index_spec.shape, self.index_spec.index)
        )

    def insert(self, mechanism, alias=None, ic=None, **kwargs):
        self.model.insert(mechanism, alias=alias, index_spec=self.index_spec, **kwargs)

    def label(self, name: str):
        if self.parent_slice is not None:
            setattr(self.parent_slice, name, self)
            return
        setattr(self.model, name, self)
        self.model._labels[name] = self

    def __getitem__(self, key):
        idx = compose_indices(
            self.model.shape, self.index, key, device=self.model.device()
        )
        idx = parse_key(idx, self.model.shape, device=self.model.device())
        return Slice(self.model, idx, parent_slice=self)

    def __repr__(self):
        return f"Slice(index={self.index_spec.index}, shape={self.index_spec.shape}, is_scalar={self.index_spec.is_scalar})"

    def _batch(self):
        current_index = self.index_spec.index
        if current_index[0] is Ellipsis:
            new_index = current_index
        else:
            new_index = (slice(None),) + current_index

        self.index_spec.index = new_index

        test = torch.empty(
            self.model.shape, device=self.model.device(), dtype=self.model.dtype()
        )
        test = test[new_index]

        self.index_spec.is_scalar = test.ndim == 0
        self.index_spec.shape = test.shape


def compose_indices(shape, idx1, idx2, *, device="cpu"):
    """
    Return idx3 (a tuple of index tensors) such that for any tensor
    t with the given shape:  t[idx1][idx2] == t[idx3].

    Works with ints, slices, ellipsis, None (newaxis), boolean masks,
    and long/bool tensor indices.
    """
    # 1) Build a flat index map shaped like `shape`
    numel = math.prod(shape)
    base = torch.arange(numel, device=device).reshape(shape)

    # 2) Apply the two-stage indexing to the map
    flat = base[idx1][idx2]  # same shape as t[idx1][idx2]

    # 3) Convert selected flat positions back to per-dimension indices
    idx3 = torch.unravel_index(flat, shape)  # tuple of tensors

    return idx3  # use as t[idx3]


class Sliceable:
    def __getitem__(self, key):
        index = parse_key(key, self.shape)
        return Slice(self, index)
