from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Union, Optional
import numpy as np

import torch
import torch.nn as nn


def find_parent_module(
    model: nn.Module, 
    target_tensor: torch.Tensor
) -> Optional[Tuple[nn.Module, str]]:
    """
    Finds the parent module of a given parameter or buffer.

    Args:
        model (nn.Module): The top-level model to search within.
        target_tensor (torch.Tensor): The parameter or buffer to find.

    Returns:
        A tuple containing (parent_module, tensor_name) if found, otherwise None.
        - parent_module (nn.Module): The module that directly holds the tensor.
        - tensor_name (str): The name of the attribute on the parent module.
    """
    for module in model.modules():
        # Check direct parameters of the current module
        for param_name, param in module._parameters.items():
            if param is target_tensor:
                return module, param_name
        
        # Check direct buffers of the current module
        for buffer_name, buffer in module._buffers.items():
            if buffer is target_tensor:
                return module, buffer_name
    
    return None


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


    def shape(self):
        """
        Return the shape of the slice.
        """
        return self.index_spec.shape


    def index(self):
        """
        Return the index of the slice.
        """
        return self.index_spec.index

    
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


    def __getattr__(self, name: str) -> Any:
        """
        Allow access to the model's attributes directly.
        """
        t = None
        if name in self.model._buffers:
            t = self.model._buffers[name]
        elif name in self.model._parameters:
            t = self.model._parameters[name]
        if t is not None:
            if (k := self.model.key) is not None:
                return expand_into_shape(
                    t,
                    k,
                    self.base_shape
                )[self.index_spec.index]
            return t[self.index_spec.index]
        else:
            return Slice(getattr(self.model, name), self.index_spec, self.base_shape)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!s}")

    def inject(self, waveform):
        self.model.stimuli.append((waveform, self.index_spec.shape, self.index_spec.index))

    def insert(self, mechanism, alias=None, ic=None, **kwargs):
        self.model.insert(mechanism, alias=alias, index_spec=self.index_spec, **kwargs)

    def label(self, name: str):
        setattr(self.model, name, self)
        self.model._labels[name] = self
