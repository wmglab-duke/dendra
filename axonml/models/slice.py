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


def parse_key(key: Any, shape: Sequence[int]) -> IndexSpec:
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
    out = np.empty(shape, dtype=np.float32)[key]  # type: ignore

    return IndexSpec(
        index=key,
        is_scalar=out.ndim == 0,
        shape=out.shape,
    )


class Slice:

    def __init__(self, model, index_spec: IndexSpec):
        self.model : torch.nn.Module = model
        self.index_spec = index_spec

    def __getattr__(self, name: str) -> Any:
        """
        Allow access to the model's attributes directly.
        """
        t = getattr(self.model, name, None)
        if t is not None:
            module, attr_name = find_parent_module(self.model, t)
            if (k := module.key) is not None:
                return expand_into_shape(
                    t,
                    k,
                    self.model.shape
                ).reshape(self.model.shape)[self.index_spec.index]
            return t[self.index_spec.index]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!s}")

    def inject(self, waveform):
        self.model.stimuli.append((waveform, self.index_spec.out_shape, self.index_spec.index))

    def insert(self, mechanism, ic=None, **kwargs):
        self.model.insert(mechanism, ic, self.index_spec, **kwargs)
