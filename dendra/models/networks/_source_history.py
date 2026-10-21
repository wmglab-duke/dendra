"""Shared pure source-history transport for NetCon adapters.

Read and deliver from history before observing the source and recording its
current gate. Weights and physical delays belong to the read phase: changing
them changes deliveries reconstructed from already recorded source activity.
"""

import torch

from ._event_delivery import delay_interpolation
from .spiking import threshold_transition


def source_history_gates(
    source_values,
    has_spiked,
    *,
    source_idx,
    mode,
    threshold=None,
    diff_spiking=False,
    tau=0.1,
    dtype=None,
    binary=False,
):
    """Observe one value per unique source, preserving native gate semantics.

    ``binary=True`` returns hard booleans without floating event intermediates.
    Already generated NetStim events and boolean raw fields bypass detector
    dtype conversion, matching native packed inference. Surrogate gates cannot
    use this binary path.
    """
    if binary and diff_spiking:
        raise ValueError("binary source history cannot use surrogate spiking")
    selected = source_values.reshape(source_values.numel()).index_select(0, source_idx)
    if binary and mode == "netstim":
        return has_spiked, selected.to(torch.bool)
    if binary and mode == "pre_var" and selected.dtype == torch.bool:
        return has_spiked, selected
    dtype = source_values.dtype if dtype is None else dtype
    selected = selected.to(dtype=dtype)
    if mode == "netstim":
        return has_spiked, selected
    if mode == "pre_var":
        if diff_spiking:
            return has_spiked, selected
        gates = selected > 0
        return has_spiked, gates if binary else gates.to(dtype)
    if mode == "threshold":
        return threshold_transition(
            selected,
            has_spiked,
            threshold,
            diff_spiking=diff_spiking,
            tau=tau,
        )
    raise ValueError("source-history mode must be netstim, pre_var, or threshold")


def read_float_history(history, rows, source_pos):
    """Read floating gates in connection order at one or several tap rows."""
    if rows.ndim == 1:
        return history[rows.reshape(rows.numel()), source_pos]
    selected = source_pos.unsqueeze(-1).expand_as(rows)
    return history[
        rows.reshape(rows.numel()), selected.reshape(selected.numel())
    ].reshape(rows.shape)


def read_packed_history(history, rows, word_idx, bit_mask, *, dtype):
    """Read hard gates from the native signed-int64 packed word layout."""
    if rows.ndim == 1:
        words = history[rows.reshape(rows.numel()), word_idx]
        return (torch.bitwise_and(words, bit_mask) != 0).to(dtype=dtype)
    selected = word_idx.unsqueeze(-1).expand_as(rows)
    mask = bit_mask.unsqueeze(-1).expand_as(rows)
    words = history[
        rows.reshape(rows.numel()), selected.reshape(selected.numel())
    ].reshape(rows.shape)
    return (torch.bitwise_and(words, mask) != 0).to(dtype=dtype)


def source_history_project(
    history,
    slot,
    weights,
    *,
    post_idx,
    depth,
    width,
    source_pos=None,
    word_idx=None,
    bit_mask=None,
    delay_steps=None,
    delays=None,
    dt=None,
    taps=2,
    sigma=0.35,
    _owned_output=False,
    _scratch=None,
):
    """Reconstruct today's payload using current weights and delays.

    Supply ``source_pos`` for floating history or packed word/mask metadata.
    Fixed reads use ``delay_steps``. Differentiable physical delays use the
    native two-tap linear or three-tap soft interpolation within fixed capacity.
    No history, parameter, or pointer is changed by default. The private
    owned-output path accumulates into its freshly allocated payload. Native
    inference can instead donate ``_scratch`` to clear and reuse its existing
    dense output row, including after a partially failed optimized launch.
    """
    out = weights.new_zeros((width,)) if _scratch is None else _scratch.zero_()
    if weights.numel() == 0:
        return out

    def read(rows):
        if source_pos is not None:
            return read_float_history(history, rows, source_pos)
        return read_packed_history(
            history, rows, word_idx, bit_mask, dtype=weights.dtype
        )

    if delays is not None:
        offsets, interpolation = delay_interpolation(
            delays, dt, dtype=weights.dtype, taps=taps, sigma=sigma
        )
        if taps == 2:
            alpha = interpolation
            row0 = (slot - offsets).remainder(depth)
            row1 = (row0 - 1).remainder(depth)
            gate0, gate1 = (
                read(row0.reshape(row0.numel())),
                read(row1.reshape(row1.numel())),
            )
            values = weights * (gate0 * (1.0 - alpha) + gate1 * alpha)
        else:
            rows = (slot - offsets).remainder(depth)
            values = weights * (read(rows) * interpolation).sum(dim=-1)
    else:
        rows = (slot - delay_steps).remainder(depth)
        values = weights * read(rows.reshape(rows.numel()))
    # index_add preserves native accumulation precision on low-precision CPU
    # collisions. Out-of-place accumulation also admits independently mapped
    # history, slot, weights, and delays, including empty mapped lanes.
    return (
        out.index_add_(0, post_idx, values)
        if _owned_output or _scratch is not None
        else out.index_add(0, post_idx, values)
    )


def source_history_record(
    history,
    slot,
    gates,
    *,
    source_word_idx=None,
    source_bit_mask=None,
    _inplace=False,
    _clone_once=False,
):
    """Overwrite the current source row without altering the input by default.

    Floating gates retain gradients. Packed recording stores each nonzero
    source gate in its native bit and preserves the packed word layout.
    Private ownership flags let native adapters donate an integer history or
    update one owned floating clone; transformed callers use the pure default.
    """
    if source_word_idx is None:
        row = gates.reshape(1, gates.numel())
    else:
        values = source_bit_mask * gates.to(torch.bool).to(torch.int64)
        packed = history.new_zeros((history.shape[1],)).scatter_add(
            0, source_word_idx, values
        )
        row = packed.reshape(1, history.shape[1])
    if _clone_once:
        history = history.clone(memory_format=torch.contiguous_format)
    if _inplace or _clone_once:
        return history.index_copy_(0, slot.reshape(-1), row)
    rows = torch.arange(history.shape[0], device=history.device).reshape(-1, 1)
    return torch.where(rows == slot.reshape(1, 1), row, history)
