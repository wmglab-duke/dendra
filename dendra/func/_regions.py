"""Owned physical-region indexing for functional recording callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import torch

from dendra.models.slice import Slice

from ._types import FunctionalizationError

if TYPE_CHECKING:
    from ._population import FunctionalPopulation


def _owned_region_index(index, *, ndim, device):
    """Freeze a Slice key and replace its ellipsis with physical axes.

    Boolean masks are resolved once, outside the compiled loop. Their owned
    integer coordinates have static shape and avoid data-dependent ``nonzero``
    operations during recording. A scalar Boolean inserts an advanced axis
    without consuming a physical dimension, like PyTorch's indexing semantics.
    """
    owned = []
    with torch.inference_mode(False):
        for item in index:
            if isinstance(item, (list, tuple)):
                item = torch.as_tensor(
                    item, device=device, dtype=None if item else torch.long
                )
            if torch.is_tensor(item):
                item = item.detach().to(device=device).clone()
                if item.dtype == torch.bool:
                    if item.ndim == 0:
                        owned.append(bool(item.item()))
                    else:
                        owned.extend(item.nonzero(as_tuple=True))
                else:
                    owned.append(item)
            elif item is None or item is Ellipsis or isinstance(item, slice):
                owned.append(item)
            elif isinstance(item, bool):
                owned.append(item)
            elif isinstance(item, Integral):
                owned.append(int(item))
            else:  # Slice.parse_key has already rejected invalid tensor keys.
                raise FunctionalizationError(
                    "functional Recorder region contains an unsupported "
                    f"index selector {type(item).__name__}"
                )

    consumed = sum(item is not None and not isinstance(item, bool) for item in owned)
    consumed -= sum(item is Ellipsis for item in owned)
    missing = ndim - consumed
    if missing < 0 or sum(item is Ellipsis for item in owned) > 1:
        raise FunctionalizationError(
            "functional Recorder region has an invalid physical indexing rank"
        )
    normalized = []
    for item in owned:
        if item is Ellipsis:
            normalized.extend((slice(None),) * missing)
        else:
            normalized.append(item)
    if not any(item is Ellipsis for item in owned):
        normalized.extend((slice(None),) * missing)
    return tuple(normalized)


@dataclass(frozen=True)
class _RecorderRegion:
    """An immutable snapshot of a source Population's physical Slice.

    No source object or mutable Slice is retained by the bound callback. The
    exact logical selection preserves explicit batch axes, advanced-index
    ordering, multiplicity, empty regions, and dimensions inserted with None.
    """

    index: tuple
    physical_shape: tuple[int, ...]
    selected_shape: tuple[int, ...]

    @classmethod
    def bind(cls, functional: FunctionalPopulation, region):
        if not isinstance(region, Slice):
            raise FunctionalizationError(
                "functional Recorder region must be a Population Slice, such "
                "as population.cell_a.soma"
            )
        if object.__getattribute__(region, "root_model") is not functional._population:
            raise FunctionalizationError(
                "functional Recorder region must belong to the exact source "
                "Population used to create this functional plan"
            )
        if object.__getattribute__(region, "module_path"):
            raise FunctionalizationError(
                "functional Recorder region must select physical Population "
                "coordinates, not a mechanism submodule"
            )
        physical_shape = tuple(functional.shape)
        if tuple(region.base_shape) != physical_shape:
            raise FunctionalizationError(
                "functional Recorder region source shape no longer matches "
                "the immutable functional plan"
            )
        return cls(
            _owned_region_index(
                region.index, ndim=len(physical_shape), device=functional.device
            ),
            physical_shape,
            tuple(region.shape),
        )

    def select(self, value: torch.Tensor, payload_ndim: int = 0):
        """Select physical axes while retaining optional trailing payload axes."""
        if (
            payload_ndim < 0
            or value.ndim != len(self.physical_shape) + payload_ndim
            or tuple(value.shape[: len(self.physical_shape)]) != self.physical_shape
        ):
            raise FunctionalizationError(
                "functional Recorder region requires a full-coordinate "
                "Population state with its declared trailing payload axes"
            )
        return value[self.index + (slice(None),) * payload_ndim]
