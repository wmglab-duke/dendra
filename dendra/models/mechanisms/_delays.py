"""Shared fixed-step history operations for mechanism-owned delay lines.

The tensor operations in this module do not read or modify module state. Native
mechanisms retain an in-place circular write adapter; functional execution
materializes the same write plan as replacement history tensors.
"""

import torch


@torch.library.custom_op("dendra_mechanism::overwrite_delay_history", mutates_args=())
def _raw_overwrite_history(
    current: torch.Tensor, updated: torch.Tensor
) -> torch.Tensor:
    """Keep native destination gradients intact through no-grad FX capture."""
    return updated.clone(memory_format=torch.preserve_format)


@_raw_overwrite_history.register_fake
def _fake_overwrite_history(current, updated):
    return torch.empty_like(updated, memory_format=torch.preserve_format)


def _vmap_overwrite_history(info, in_dims, current, updated):
    def align(value, dimension):
        if dimension is not None:
            return value.movedim(dimension, 0)
        return value.unsqueeze(0).expand(info.batch_size, *value.shape)

    return (
        _raw_overwrite_history(align(current, in_dims[0]), align(updated, in_dims[1])),
        0,
    )


class _OverwriteHistory(torch.autograd.Function):
    """Preserve the destination gradient identity of native no-grad writes."""

    generate_vmap_rule = True

    @staticmethod
    def forward(current, updated):
        return _raw_overwrite_history(current, updated)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None

    @staticmethod
    def jvp(ctx, current_tangent, updated_tangent):
        # torch.no_grad suppresses reverse recording, but forward AD still
        # applies the detached index_copy write to the destination tangent.
        return updated_tangent.clone()


# make_fx erases Python autograd.Function wrappers from a scan's step graph.
# The surviving opaque node retains the same VJP and batching rule; the outer
# Function additionally supplies the native forward-mode tangent contract.
_raw_overwrite_history.register_autograd(
    _OverwriteHistory.backward, setup_context=_OverwriteHistory.setup_context
)
_raw_overwrite_history.register_vmap(_vmap_overwrite_history)


@torch.compiler.allow_in_graph
def _overwrite_history(current, updated):
    return _OverwriteHistory.apply(current, updated)


def resolve_delay_mode(mode, *, training, batched=False):
    """Resolve the native training/evaluation policy without tensor reads."""
    selected = str(mode).lower()
    aliases = {"functional": "shift", "queue": "shift", "shift_queue": "shift"}
    selected = aliases.get(selected, selected)
    if selected == "auto":
        selected = "shift" if (training or torch.is_grad_enabled()) else "circular"
    if selected not in {"shift", "circular"}:
        prefix = "batched " if batched else ""
        raise ValueError(
            f"{prefix}delayed-state mode must be 'auto', 'shift', or 'circular'; "
            f"got {selected!r}."
        )
    return selected


def shifted_history_values(buffer, value, *, axis):
    """Prepend one payload to a fixed-capacity, newest-first history."""
    slices = [slice(None)] * buffer.ndim
    slices[axis] = slice(0, buffer.shape[axis] - 1)
    return torch.cat((value.unsqueeze(axis), buffer[tuple(slices)]), dim=axis)


def gather_history_values(buffer, indices, *, axis, stream_axis=None):
    """Read a scalar delay or a distinct delay for each stream."""
    if stream_axis is None:
        if isinstance(indices, int):
            return buffer.select(axis, indices)
        return buffer.index_select(axis, indices.reshape(1)).squeeze(axis)

    index_shape = [1] * buffer.ndim
    index_shape[stream_axis] = buffer.shape[stream_axis]
    expand_shape = list(buffer.shape)
    expand_shape[axis] = 1
    index = indices.reshape(index_shape).expand(expand_shape)
    return torch.gather(buffer, dim=axis, index=index).squeeze(axis)


def circular_history_plan(
    buffer,
    pointer,
    value,
    steps,
    *,
    axis,
    stream_axis=None,
    value_stream_axis=None,
):
    """Read before writing and return the detached circular-write payload.

    The returned tuple contains the delayed output, write index, detached
    payload and next pointer. A zero-delay stream returns the current input,
    including its gradient; only queued writes are detached.
    """
    depth = buffer.shape[axis]
    read_index = torch.remainder(pointer - steps, depth)
    # gather's backward saves its input tensor. Native circular writes must not
    # invalidate that saved version when an incoming history has a gradient.
    # Ordinary evaluation histories need no additional allocation.
    read_buffer = (
        buffer.clone() if stream_axis is not None and buffer.requires_grad else buffer
    )
    delayed = gather_history_values(
        read_buffer, read_index, axis=axis, stream_axis=stream_axis
    )
    if stream_axis is not None:
        mask_shape = [1] * value.ndim
        mask_shape[value_stream_axis] = value.shape[value_stream_axis]
        delayed = torch.where((steps == 0).reshape(mask_shape), value, delayed)
    return (
        delayed,
        pointer.reshape(1),
        value.detach().unsqueeze(axis),
        torch.remainder(pointer + 1, depth),
    )


def delay_history_values(
    buffer,
    pointer,
    value,
    steps,
    *,
    axis,
    mode,
    stream_axis=None,
    value_stream_axis=None,
    all_zero=False,
):
    """Return delayed output and replacement queue/pointer without mutation."""
    if all_zero:
        return value, buffer, pointer
    if mode == "shift":
        updated = shifted_history_values(buffer, value, axis=axis)
        delayed = gather_history_values(
            updated, steps, axis=axis, stream_axis=stream_axis
        )
        return delayed, updated, pointer

    delayed, index, payload, next_pointer = circular_history_plan(
        buffer,
        pointer,
        value,
        steps,
        axis=axis,
        stream_axis=stream_axis,
        value_stream_axis=value_stream_axis,
    )
    # Native writes under no-grad into existing storage, retaining any incoming
    # destination gradient identity while detaching newly queued payloads.
    # Represent the same identity without mutating explicit history inputs.
    updated = buffer.index_copy(axis, index, payload)
    return delayed, _overwrite_history(buffer, updated), next_pointer


def initial_delay_history_values(buffer, pointer):
    """Return fresh detached zero history without modifying an old checkpoint."""
    return torch.zeros_like(buffer).detach(), torch.zeros_like(pointer).detach()
