"""Shared pure mechanism-local waveform padding."""

import math

import torch


def expand_injection_value(
    value,
    mask,
    out,
    *,
    selected_index=None,
    n_local_selected=None,
):
    """Pad a waveform payload into a local state shape without changing inputs.

    Registered injections supply the static selected count and index. The
    optional discovery path supports direct compatibility-helper calls outside
    graph capture.
    """
    value = torch.as_tensor(value, device=out.device, dtype=out.dtype)
    local_mask = mask.to(device=out.device, dtype=torch.bool)
    if n_local_selected is None and selected_index is not None:
        n_local_selected = selected_index.numel()
    mask = local_mask
    while mask.ndim < out.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(out)

    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(()) * mask.to(out.dtype)
    if tuple(value.shape) == tuple(out.shape):
        return value * mask.to(out.dtype)

    if value.ndim <= out.ndim:
        expanded = value
        while expanded.ndim < out.ndim:
            expanded = expanded.unsqueeze(0)
        if all(a == b or a == 1 for a, b in zip(expanded.shape, out.shape)):
            return expanded.expand_as(out) * mask.to(out.dtype)

    # A selected-compartment vector can be shared over every batch, or contain
    # independent vectors on an explicit batch grid. Shape/count information is
    # structural; the graph path does not discover indices from a live mask.
    if local_mask.ndim <= out.ndim and tuple(
        out.shape[out.ndim - local_mask.ndim :]
    ) == tuple(local_mask.shape):
        batch_shape = tuple(out.shape[: out.ndim - local_mask.ndim])
        if n_local_selected is None:
            n_local_selected = int(local_mask.reshape(-1).sum().item())
        selected_shape = batch_shape + (n_local_selected,)
        selected = None
        if value.numel() == n_local_selected:
            selected = value.reshape(
                (1,) * len(batch_shape) + (n_local_selected,)
            ).expand(selected_shape)
        else:
            try:
                selected = value.expand(selected_shape)
            except RuntimeError:
                pass
        if selected is not None:
            if selected_index is None:
                selected_index = torch.nonzero(
                    local_mask.reshape(-1), as_tuple=False
                ).reshape(-1)
            batch_count = math.prod(batch_shape)
            padded = torch.zeros_like(out).reshape(batch_count, local_mask.numel())
            padded = padded.index_copy(
                1,
                selected_index.to(device=out.device),
                selected.reshape(batch_count, n_local_selected),
            )
            return padded.reshape_as(out)

    # A broadcast local mask can also accept one flattened vector over every
    # selected output location. Derive its indices from the registered local
    # index and static shapes, preserving the native flattened-mask ordering.
    if selected_index is not None and n_local_selected is not None:
        local_shape = (1,) * (out.ndim - local_mask.ndim) + tuple(local_mask.shape)
        repetitions = 1
        for local_size, size in zip(local_shape, out.shape):
            if local_size == 1:
                repetitions *= size
        n_selected = n_local_selected * repetitions
        if value.numel() == n_selected:
            index = selected_index.to(device=out.device)
            if n_selected:
                base = torch.zeros_like(index)
                offsets = index.new_zeros(1)
                for axis, (local_size, size) in enumerate(zip(local_shape, out.shape)):
                    stride = math.prod(out.shape[axis + 1 :])
                    if local_size == 1:
                        offsets = (
                            offsets.unsqueeze(-1)
                            + torch.arange(size, device=out.device) * stride
                        ).reshape(-1)
                    else:
                        local_stride = math.prod(local_shape[axis + 1 :])
                        base = base + ((index // local_stride) % local_size) * stride
                index = (offsets.unsqueeze(-1) + base.unsqueeze(0)).reshape(-1)
                index = index.sort().values
            else:
                index = index.new_empty(0)
            return (
                torch.zeros_like(out)
                .reshape(-1)
                .index_copy(0, index, value.reshape(-1))
                .reshape_as(out)
            )
    elif selected_index is None:
        # Direct compatibility-helper calls can discover their mask indices.
        n_selected = int(mask.reshape(-1).sum().item())
        if value.numel() == n_selected:
            index = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
            return (
                torch.zeros_like(out)
                .reshape(-1)
                .index_copy(0, index, value.reshape(-1))
                .reshape_as(out)
            )

    try:
        return value.expand_as(out) * mask.to(out.dtype)
    except RuntimeError as exc:
        raise ValueError(
            f"Waveform output shape {tuple(value.shape)} cannot be broadcast "
            f"or padded into mechanism-local shape {tuple(out.shape)}."
        ) from exc
