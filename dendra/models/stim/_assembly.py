"""Pure current scatter shared by native and functional stimulation.

Only consecutive, mutually disjoint injections share a scatter. Overlapping
injections retain separate ordered additions; duplicate destinations within an
injection retain that injection's original scatter.
"""

from dataclasses import dataclass
from math import prod

import torch

_TAPE_METHODS = tuple(
    (name, getattr(torch.Tensor, name))
    for name in ("data_ptr", "stride", "storage_offset", "as_strided", "detach")
)


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


@dataclass(frozen=True)
class _IndexedCurrentSamples:
    """Read one sample from captured time-last tapes inside a compiled step."""

    tapes: tuple
    index: torch.Tensor


def _indexed_current_tapes(tapes, n_steps, *, device, dtype):
    """Capture live storage with the metadata of the original unbind views.

    Only inference uses whole-tape gathers: their per-step backward would
    otherwise allocate a full tape gradient for every accepted sample. The
    caller has already performed canonical unbinds before authored hooks.
    """
    if torch.is_grad_enabled() or n_steps <= 0:
        return None
    try:
        if (
            type(tapes) not in (list, tuple)
            or not tapes
            or any(
                type(tape) not in (torch.Tensor, torch.nn.Parameter)
                or type(object.__getattribute__(tape, "__dict__")) is not dict
                or any(
                    name in object.__getattribute__(tape, "__dict__")
                    or getattr(type(tape), name) is not method
                    for name, method in _TAPE_METHODS
                )
                or tape.is_nested
                or tape.layout != torch.strided
                or tape.device != device
                or tape.dtype != dtype
                or tape.ndim == 0
                or tape.shape[-1] < n_steps
                or tape.data_ptr() % 16
                for tape in tapes
            )
        ):
            return None
        if not any(tape.requires_grad for tape in tapes) and (
            n_steps <= 1
            or not any(tape.stride(-1) * dtype.itemsize % 16 for tape in tapes)
        ):
            # These slices already meet the compiler's alignment contract.
            # Grad-requiring aliases retain detached capture: rebinding their
            # storage can otherwise invalidate FakeTensor's no-grad views.
            return None
        # no_grad does not disable forward AD. Such tapes must retain the
        # original route, including tangents hidden by ambient inference mode.
        with torch.inference_mode(False), torch.no_grad():
            if any(
                torch.autograd.forward_ad.unpack_dual(tape).tangent is not None
                for tape in tapes
            ):
                return None
        views = tuple(
            tape.detach().as_strided(tape.shape, tape.stride(), tape.storage_offset())
            for tape in tapes
        )
    except RuntimeError:
        # Unsupported metadata/view access keeps the original sample path and
        # its per-step diagnostics, rather than adding a preparation failure.
        return None
    for view in views:
        if view.shape[-1] > 1:
            # Time extent is private input bookkeeping, independent of the
            # fixed neuron shape. Varying run/chunk lengths can reuse a graph.
            torch._dynamo.mark_dynamic(view, view.ndim - 1)
    return views


def _aligned_current_indices(n_steps, *, device):
    """Framework-owned int64 index views with 16-byte-spaced sample starts."""
    values = torch.arange(n_steps, dtype=torch.int64, device=device)
    return values[:, None].expand(-1, 2).contiguous()[:, :1].unbind(0)
