"""Pure current scatter shared by native and functional stimulation.

Only consecutive, mutually disjoint injections share a scatter. Overlapping
injections retain separate ordered additions; duplicate destinations within an
injection retain that injection's original scatter.
"""

from dataclasses import dataclass
from math import prod

import torch


def _current_groups(indices, shape):
    """Prepare immutable scatter coordinates outside numerical execution.

    Return singleton groups for noncanonical coordinates. This is an optimization
    of already valid indexing, not an additional validation stage.
    """
    singletons = tuple((1, index) for index in indices)
    if len(indices) < 2:
        return singletons
    destinations = []
    for index in indices:
        if len(index) != len(shape) or any(
            type(item) is not torch.Tensor
            or item.dtype != torch.long
            or item.layout != torch.strided
            for item in index
        ):
            return singletons
        coordinates = [item.detach().to(device="cpu") for item in index]
        try:
            coordinates = torch.broadcast_tensors(*coordinates)
        except RuntimeError:
            return singletons
        flat = torch.zeros_like(coordinates[0])
        for coordinate, size in zip(coordinates, shape, strict=True):
            if torch.any(coordinate < 0) or torch.any(coordinate >= size):
                return singletons
            flat = flat * size + coordinate
        locations = flat.reshape(-1).tolist()
        # Keep an injection with duplicates (or no writes) unchanged, including
        # its scatter shape and backend reduction behavior.
        destinations.append(
            set(locations)
            if locations and len(set(locations)) == len(locations)
            else None
        )

    groups = []
    start = 0
    occupied = set()

    def append(stop):
        group = indices[start:stop]
        if len(group) == 1:
            groups.append((1, group[0]))
        else:
            flattened = []
            for index in group:
                selection = torch.broadcast_shapes(*(item.shape for item in index))
                flattened.append(
                    tuple(item.expand(selection).reshape(-1) for item in index)
                )
            groups.append(
                (
                    len(group),
                    tuple(torch.cat(axis) for axis in zip(*flattened, strict=True)),
                )
            )

    for position, locations in enumerate(destinations):
        if locations is None or occupied.intersection(locations):
            if position > start:
                append(position)
            start = position
            occupied = set()
        if locations is None:
            append(position + 1)
            start = position + 1
        else:
            occupied.update(locations)
    if start < len(indices):
        append(len(indices))
    return tuple(groups)


def _assemble_currents(shape, dtype, device, samples, groups, *, n_steps=None):
    """Scatter already broadcast samples, optionally with a leading time axis.

    ``samples`` is consumed in registration order so caller-specific broadcast
    diagnostics remain ordered. No waveform evaluation or RNG access occurs here.
    """
    result = torch.zeros(
        shape if n_steps is None else (n_steps, *shape), dtype=dtype, device=device
    )
    samples = iter(samples)
    for count, indices in groups:
        if count == 1:
            value = next(samples)
        else:
            values = [next(samples) for _ in range(count)]
            value = torch.cat(
                [
                    item.reshape(-1)
                    if n_steps is None
                    else item.reshape(n_steps, prod(item.shape[1:]))
                    for item in values
                ],
                dim=-1,
            )
        if n_steps is not None:
            selection = torch.broadcast_shapes(*(item.shape for item in indices))
            time_index = torch.arange(n_steps, device=device, dtype=torch.long)
            indices = (
                time_index.reshape(n_steps, *(1,) * len(selection)).expand(
                    n_steps, *selection
                ),
                *(item.unsqueeze(0).expand(n_steps, *selection) for item in indices),
            )
        if n_steps is None:
            # Native assembly owns this fresh output, as in its original loop.
            # Avoid copying a dense model-sized tensor for every overlap group.
            result.index_put_(indices, value, accumulate=True)
        else:
            # Functional time-first assembly also supports vmap.
            result = result.index_put(indices, value, accumulate=True)
    return result


@dataclass(frozen=True)
class _PreparedCurrentIndices:
    """One eagerly checked native call's indices and disjoint scatter groups."""

    indices: tuple
    groups: tuple
    selection_shapes: tuple
