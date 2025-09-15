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
    _RESERVED = ("model", "index_spec", "base_shape", "parent_slice")

    def __init__(self, model, index_spec: IndexSpec, base_shape=None, parent_slice=None):
        # Bypass interception for internal fields
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "index_spec", index_spec)
        object.__setattr__(self, "parent_slice", parent_slice)
        if base_shape is None:
            base_shape = model.shape
        object.__setattr__(self, "base_shape", base_shape)

    # -------------------------
    # Simple, safe properties
    # -------------------------
    @property
    def index(self) -> Tuple[IndexElement, ...]:
        return object.__getattribute__(self, "index_spec").index

    @property
    def shape(self):
        return object.__getattribute__(self, "index_spec").shape

    @property
    def is_scalar(self) -> bool:
        return object.__getattribute__(self, "index_spec").is_scalar

    def numel(self) -> int:
        return int(np.prod(object.__getattribute__(self, "index_spec").shape))

    @property
    def name(self) -> str:
        return object.__getattribute__(self, "model").name

    # -------------------------
    # Public API
    # -------------------------
    def inspect(self, var: str, mechanism: Optional[str] = None) -> Any:
        model = object.__getattribute__(self, "model")
        idx = object.__getattribute__(self, "index_spec").index

        if mechanism is not None:
            mech = model.mech.mechanisms[mechanism]
            if mech.key is None:
                return getattr(mech, var)[idx]
            dummy = torch.tensor(torch.nan, device=model.device(), dtype=model.dtype())
            dummy = mech.put(getattr(mech, var), dummy, model.v)
            return dummy[idx]

        return getattr(model, var)[idx]

    def _inspect(self, var: str):
        model = object.__getattribute__(self, "model")
        base_shape = object.__getattribute__(self, "base_shape")
        idx = object.__getattribute__(self, "index_spec").index

        if getattr(model, "key", None) is not None:
            v = getattr(model, var)
            dummy = torch.tensor(torch.nan, device=v.device, dtype=v.dtype)
            dummy = model.put(v, dummy, torch.empty(base_shape, device=v.device, dtype=v.dtype))
            return dummy[idx]
        return getattr(model, var)[idx]

    def get(self, var: str, mechanism: Optional[str] = None) -> torch.Tensor:
        return self.inspect(var, mechanism)

    def set(self, var: str, value: torch.Tensor, mechanism: Optional[str] = None):
        model = object.__getattribute__(self, "model")
        idx = object.__getattribute__(self, "index_spec").index

        if mechanism is not None:
            mech = model.mech.mechanisms[mechanism]
            if mech.key is None:
                with torch.no_grad():
                    getattr(mech, var)[idx] = value
                    getattr(mech, var).detach_()
                return

            dummy = torch.tensor(torch.nan, device=model.device(), dtype=model.dtype())
            dummy = mech.put(getattr(mech, var), dummy, model.v)
            with torch.no_grad():
                dummy[idx] = value
                getattr(mech, var).copy_(mech.get(dummy))
                getattr(mech, var).detach_()
            return

        with torch.no_grad():
            getattr(model, var)[idx] = value
            getattr(model, var).detach_()  # keep identity, drop history

    def inject(self, waveform):
        model = object.__getattribute__(self, "model")
        index_spec = object.__getattribute__(self, "index_spec")
        model.injections.append((waveform, index_spec.shape, index_spec.index))

    def insert(self, mechanism, alias=None, ic=None, **kwargs):
        object.__getattribute__(self, "model").insert(
            mechanism, alias=alias, index_spec=object.__getattribute__(self, "index_spec"), **kwargs
        )

    def label(self, name: str):
        parent_slice = object.__getattribute__(self, "parent_slice")
        if parent_slice is not None:
            # Attach label to the *wrapper* safely (avoid buffer interception)
            object.__setattr__(parent_slice, name, self)
            return
        model = object.__getattribute__(self, "model")
        setattr(model, name, self)
        model._labels[name] = self

    def __getitem__(self, key):
        model = object.__getattribute__(self, "model")
        idx = compose_indices(model.shape, object.__getattribute__(self, "index"), key, device=model.device())
        idx = parse_key(idx, model.shape, device=model.device())
        return type(self)(model, idx, parent_slice=self, base_shape=object.__getattribute__(self, "base_shape"))

    # -------------------------
    # Interceptors
    # -------------------------
    def __setattr__(self, name, value):
        # Always allow internal fields
        if name in Slice._RESERVED:
            object.__setattr__(self, name, value)
            return

        # Safely fetch model without triggering our __getattr__
        model = object.__getattribute__(self, "model")

        # Intercept writes to model buffers
        buffers = model._buffers  # nn.Module guarantee
        if name in buffers:
            buf = buffers[name]
            idx = object.__getattribute__(self, "index_spec").index
            with torch.no_grad():
                buf[idx] = value
                buf.detach_()  # drop history but keep identity
            return

        # Otherwise set on this wrapper
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Only runs if normal lookup failed
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

        # Buffers: return sliced/inspected view
        if name in model._buffers:
            return self._inspect(name)

        # Submodules: return a wrapped Slice
        if name in model._modules:
            sub = model._modules[name]
            return type(self)(
                sub,
                object.__getattribute__(self, "index_spec"),
                base_shape=object.__getattribute__(self, "base_shape"),
            )

        # Parameters (optional): often handy to read through
        if name in model._parameters:
            return self._inspect(name)

        # Fallback: delegate to the wrapped model (methods, attrs, etc.)
        return getattr(model, name)

    # -------------------------
    # Misc
    # -------------------------
    def __repr__(self):
        spec = object.__getattribute__(self, "index_spec")
        return f"Slice(index={spec.index}, shape={spec.shape}, is_scalar={spec.is_scalar})"

    def _batch(self):
        model = object.__getattribute__(self, "model")
        index_spec = object.__getattribute__(self, "index_spec")

        current_index = index_spec.index
        new_index = current_index if (current_index and current_index[0] is Ellipsis) else (slice(None),) + current_index
        index_spec.index = new_index

        test = torch.empty(model.shape, device=model.device(), dtype=model.dtype())[new_index]
        index_spec.is_scalar = test.ndim == 0
        index_spec.shape = test.shape



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
