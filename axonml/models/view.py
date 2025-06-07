from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Union
import numpy as np

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
    new_axes : Tuple[int, ...]
        Positions (in the *original* key) where `None`/`np.newaxis` appeared.
    advanced_axes : Tuple[int, ...]
        Axes that use fancy / boolean / integer array indexing.
    is_scalar : bool
        True if the result collapses to a scalar.
    out_shape : Tuple[int, ...]
        What `array[key]` would return, **including** new axes.
    """
    index: Tuple[IndexElement, ...]
    new_axes: Tuple[int, ...]
    advanced_axes: Tuple[int, ...]
    is_scalar: bool
    out_shape: Tuple[int, ...]


def parse_key(key: Any, shape: Sequence[int]) -> IndexSpec:
    """
    Turn *any* valid key plus `shape` into a reusable IndexSpec.
    Works for NumPy **and** PyTorch rules (they’re identical here).

    Examples
    --------
    >>> shape = (4, 5, 6)
    >>> spec = parse_key((Ellipsis, 2, None), shape)
    >>> spec
    IndexSpec(index=(slice(None), slice(None), 2), new_axes=(3,), ...)
    """
    ndim = len(shape)

    # 1. Make key a tuple so we can iterate uniformly.
    if not isinstance(key, tuple):
        key = (key,)

    # 2. Walk through the tuple and build a cleaned index.
    cleaned: List[IndexElement] = []
    new_axes: List[int] = []
    saw_ellipsis = False
    i_key = 0  # position in original key (needed for new_axes record)

    for obj in key:
        if obj is Ellipsis:
            if saw_ellipsis:
                raise IndexError("only one ellipsis ('...') allowed")
            saw_ellipsis = True
            # How many slices does '...' stand for?
            n_missing = ndim - (len(key) - i_key - 1)
            cleaned.extend([slice(None)] * n_missing)
        elif obj is None:
            new_axes.append(len(cleaned) + len(new_axes))  # after current expansions
            # newaxis does **not** consume a dimension of the data
        else:
            cleaned.append(obj)
        i_key += 1

    # If no ellipsis, pad with full slices at the end.
    if not saw_ellipsis:
        cleaned.extend([slice(None)] * (ndim - len(cleaned)))

    if len(cleaned) != ndim:
        raise IndexError(
            f"resulting index has {len(cleaned)} dims, "
            f"but data has {ndim} dims"
        )

    index_tuple = tuple(cleaned)

    # 3. Identify advanced-index axes (bool/int/ndarray/list)
    advanced_axes = tuple(
        ax for ax, obj in enumerate(index_tuple)
        if isinstance(obj, (np.ndarray, list)) or
           (isinstance(obj, slice) and obj.step is not None and obj.step == 0)  # PyTorch 0-step quirk
    )

    # 4. Compute result shape & scalar flag by hitting a dummy NumPy array.
    dummy = np.empty(shape)
    try:
        out = dummy[index_tuple]
    except Exception as e:
        raise IndexError(f"invalid index {key!r} for shape {shape}: {e}") from None

    # Now re-insert new axes (they were dropped for the dummy slice)
    for axis in new_axes:
        out = np.expand_dims(out, axis=axis)

    return IndexSpec(
        index=index_tuple,
        new_axes=tuple(new_axes),
        advanced_axes=advanced_axes,
        is_scalar=out.ndim == 0,
        out_shape=out.shape,
    )


class View:

    def __init__(self, model, index_spec: IndexSpec):
        self.model = model
        self.index_spec = index_spec

    def inject(self, waveform):
        pass