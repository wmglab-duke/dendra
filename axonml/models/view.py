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


def _compose_slices(parent: slice, child: slice, dim_len: int) -> slice:
    """
    Combine arr[parent][child]  into a single slice relative to the *original*
    dimension length `dim_len`.

    Works for any direction / step, using slice.indices to normalise.
    """
    # Expand both slices into (start, stop, step)
    p_start, p_stop, p_step = parent.indices(dim_len)
    # length after first slice:
    if p_step > 0:
        p_len = max(0, (p_stop - p_start + p_step - 1) // p_step)
    else:                              # negative step
        p_len = max(0, (p_start - p_stop - p_step - 1) // (-p_step))

    c_start, c_stop, c_step = child.indices(p_len)

    # Map child coords back into the original coordinate space
    new_step  = p_step * c_step
    new_start = p_start + p_step * c_start
    new_stop  = p_start + p_step * c_stop
    return slice(new_start, new_stop, new_step)


def compose(spec1: IndexSpec, key2: Any) -> IndexSpec:
    """
    Return an IndexSpec that is equivalent to  arr[spec1_key][key2].

    Limitations:
      * axis-wise advanced / boolean indexing is allowed in *either* step,
        but the composition falls back to "evaluate & re-slice" if *both*
        steps use advanced indexing on **the same axis** (NumPy can't
        express that as a single basic index anyway).
      * Works for NumPy and PyTorch rules.
    """
    # 1.  Parse key2 **relative to the shape after spec1 is applied**
    spec2 = parse_key(key2, spec1.out_shape)

    # 2.  Expand spec1.index so it already contains its new-axis positions
    full1: List[IndexElement] = []
    j = 0
    for ax in range(len(spec1.out_shape)):
        if ax in spec1.new_axes:
            full1.append(None)                 # new axis placeholder
        else:
            full1.append(spec1.index[j])
            j += 1
    assert len(full1) == len(spec1.out_shape)

    # 3.  Walk through axes of the *intermediate* result and merge
    out_index: List[IndexElement] = []
    out_new_axes: List[int] = []
    advanced_axes: List[int] = []

    orig_shape = spec1.out_shape
    composed_shape: List[int] = list(orig_shape)  # will shrink/grow as we go

    axis_orig = 0   # pointer in original (after first index)
    axis_out  = 0   # pointer in final composed result

    for ax2 in range(len(orig_shape) + len(spec1.new_axes)):
        # Insert new axes coming from key2 (np.newaxis)
        while ax2 in spec2.new_axes:
            out_index.append(None)
            out_new_axes.append(len(out_index) - 1)
            axis_out += 1
            ax2 += 1

        # Pull current elements
        idx1 = full1[ax2]
        idx2 = spec2.index[ax2 if ax2 < len(spec2.index) else -1]

        # Case A: first round inserted np.newaxis
        if idx1 is None:
            # idx1 created a size-1 axis.  idx2 can only be:
            #   * slice(None)  -> keep the axis
            #   * integer 0    -> drop it
            #   * newaxis      -> handled above
            if isinstance(idx2, slice):
                out_index.append(None)                       # keep the axis
                axis_orig += 1
                axis_out  += 1
            elif np.issubdtype(type(idx2), np.integer) or isinstance(idx2, int):
                # Picking the single element collapses axis
                axis_orig += 1
                # axis_out not incremented
            else:   # boolean / fancy on a length-1 axis → just evaluate
                raise NotImplementedError("advanced indexing on newaxis")
            continue

        # Case B: either side has advanced indexing → fallback
        if ax2 in spec1.advanced_axes or ax2 in spec2.advanced_axes:
            advanced_axes.append(len(out_index))
            out_index.append(idx2)              # second advanced indexing wins
            axis_orig += 1
            axis_out  += 1
            continue

        # Case C: both sides are basic (slice / int)
        if isinstance(idx1, slice) and isinstance(idx2, slice):
            out_index.append(
                _compose_slices(idx1, idx2, dim_len=orig_shape[ax2])
            )
            axis_orig += 1
            axis_out  += 1
        elif isinstance(idx1, slice):           # slice then integer
            # convert integer into absolute index
            start, stop, step = idx1.indices(orig_shape[ax2])
            abs_idx = start + step * idx2
            out_index.append(int(abs_idx))
            axis_orig += 1
            # integer drops axis_out
        elif isinstance(idx1, int):             # integer then anything
            if isinstance(idx2, slice):
                raise IndexError("cannot slice a collapsed axis")
            out_index.append(idx1)              # already collapsed; idx2 ignored
            axis_orig += 1
            # axis_out unchanged
        else:
            raise NotImplementedError("unhandled combination")

    # Remove leading / trailing None’s in index construction
    index_tuple = tuple(x for x in out_index if x is not Ellipsis)

    # Compute resulting shape by probing a dummy array
    dummy = np.empty(spec1.out_shape)
    out = dummy[key2]
    out_shape = out.shape
    is_scalar = out.ndim == 0

    return IndexSpec(
        index=index_tuple,
        new_axes=tuple(out_new_axes),
        advanced_axes=tuple(advanced_axes),
        is_scalar=is_scalar,
        out_shape=out_shape,
    )


class View:

    def __init__(self, model, index_spec: IndexSpec):
        self.model = model
        self.index_spec = index_spec

    def __getitem__(self, key: Any) -> View:
        """
        Return a new View that applies the given key to the current View.
        The key can be any valid NumPy or PyTorch indexing key.
        """
        new_spec = compose(self.index_spec, key)
        return View(self.model, new_spec)

    def inject(self, waveform):
        self.model.stimuli.append((waveform, self.index_spec.out_shape, self.index_spec.index))