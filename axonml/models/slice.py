from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Union, Optional
import numpy as np

import torch
import torch.nn as nn


def expand_into_shape(src, index, shape, fill_value=torch.nan):
    # out is just temporary storage → no grad required
    out = torch.full(shape,
                     fill_value,
                     dtype=src.dtype,
                     device=src.device)
    out[index] = src
    return out


# Anything NumPy or PyTorch accepts in __getitem__
IndexElement = Union[int, slice, np.ndarray, list, tuple]


@dataclass(frozen=True)
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


def parse_key(key: Any, shape: Sequence[int], device) -> IndexSpec:
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

    def __init__(self, model, index_spec: IndexSpec, base_shape=None):
        self.model : torch.nn.Module = model
        self.base_shape = base_shape
        if base_shape is None:
            self.base_shape = model.shape
        self.index_spec = index_spec


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


    def inspect(self, var: str, mechanism: Optional[str] = None) -> Any:
        """
        Inspect the variable in the model or a specific mechanism.
        Note: Most of the time, this will involve a memory allocation.
        """
        if mechanism is not None:
            mech = self.model.mech.mechanisms[mechanism]
            if not mech.key:
                return getattr(mech, var)[self.index_spec.index]
            dummy = torch.tensor(torch.nan, device=self.model.device(), dtype=self.model.dtype())
            dummy = mech.put(getattr(mech, var), dummy, self.model.v)
            return dummy[self.index_spec.index]
        return getattr(self.model, var)[self.index_spec.index]

    def get(self, var: str, mechanism: Optional[str] = None) -> torch.Tensor:
        """
        Get the variable from the model or a specific mechanism.
        Returns a tensor with the shape of the slice.
        """
        if mechanism is not None:
            mech = self.model.mech.mechanisms[mechanism]
            if not mech.key:
                return getattr(mech, var)[self.index_spec.index]
            dummy = torch.tensor(torch.nan, device=self.model.device(), dtype=self.model.dtype())
            dummy = mech.put(getattr(mech, var), dummy, self.model.v)
            return dummy[self.index_spec.index]
        return getattr(self.model, var)[self.index_spec.index]

    def set(self, var: str, value: torch.Tensor, mechanism: Optional[str] = None):
        """
        Set the variable in the model or a specific mechanism.
        The value must match the shape of the slice.
        """
        if mechanism is not None:
            mech = self.model.mech.mechanisms[mechanism]
            if not mech.key:
                getattr(mech, var)[self.index_spec.index] = value
                getattr(mech, var).detach_()
                return
            dummy = torch.tensor(torch.nan, device=self.model.device(), dtype=self.model.dtype())
            dummy = mech.put(getattr(mech, var), dummy, self.model.v)
            dummy[self.index_spec.index] = value
            getattr(mech, var).copy_(mech.get(dummy))
            getattr(mech, var).detach_()
            return
        getattr(self.model, var)[self.index_spec.index] = value
        getattr(self.model, var).detach_()

    def inject(self, waveform):
        self.model.injections.append((waveform, self.index_spec.shape, self.index_spec.index))

    def insert(self, mechanism, alias=None, ic=None, **kwargs):
        self.model.insert(mechanism, alias=alias, index_spec=self.index_spec, **kwargs)

    def label(self, name: str):
        setattr(self.model, name, self)
        self.model._labels[name] = self

    def __repr__(self):
        return f"Slice(index={self.index_spec.index}, shape={self.index_spec.shape}, is_scalar={self.index_spec.is_scalar})"
