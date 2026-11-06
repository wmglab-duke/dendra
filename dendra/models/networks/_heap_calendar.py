"""Compact consuming calendars shared by native and functional NetStim.

Packing and validation are cold operations. Heads and consumption only read
tensors and return tensors; calendar topology and the search depth stay fixed
during a transformed run. Runtime times use Float64 where supported and
the simulation dtype on MPS. Adapters retain original Python-float precision in
cold heap layouts; comparisons cast selected times to the simulation dtype.
"""

import math
import struct

import torch


def heap_calendar_dtype(device, simulation_dtype=torch.float32):
    """Choose a device-supported runtime representation for legacy times."""
    return simulation_dtype if torch.device(device).type == "mps" else torch.float64


def pack_heap_calendar(heaps, *, device, dtype=None, simulation_dtype=torch.float32):
    """Pack ordered rows from legacy heap records, retaining duplicate times."""
    dtype = heap_calendar_dtype(device, simulation_dtype) if dtype is None else dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Heap calendar times must use a supported floating dtype.")
    values, offsets = [], [0]
    longest = 0
    for heap in heaps:
        row = sorted(float(value) for value in heap)
        if any(math.isnan(value) for value in row):
            raise ValueError("NaN heap calendars require native heapq execution.")
        values.extend(row)
        offsets.append(len(values))
        longest = max(longest, len(row))
    times = torch.tensor(values + [float("inf")], device=device, dtype=dtype)
    row_offsets = torch.tensor(offsets, device=device, dtype=torch.long)
    return times, row_offsets, row_offsets[:-1].clone(), longest.bit_length()


def heap_calendar_matches_layout(times, row_offsets, layout):
    """Match a saved representation independently of the receiving CPU's casts.

    PyTorch half constructors use direct rounding on some backends and an
    intermediate float32 on others. Accept either complete encoding, without
    tolerances or mixing the two encodings between individual events.
    """
    layout_times, layout_offsets, _, _ = pack_heap_calendar(
        layout, device="cpu", dtype=times.dtype
    )
    if torch.equal(layout_times, times):
        return torch.equal(layout_offsets, row_offsets)
    if times.dtype != torch.float16 or not torch.equal(layout_offsets, row_offsets):
        return False

    values = [value for row in layout for value in sorted(row)] + [float("inf")]

    def rounded(value, precision):
        format_code = "<" + precision
        try:
            return struct.unpack(format_code, struct.pack(format_code, value))[0]
        except OverflowError:
            return math.copysign(float("inf"), value)

    for via_float32 in (False, True):
        encoded = [
            rounded(rounded(value, "f") if via_float32 else value, "e")
            for value in values
        ]
        # Every value is already exactly representable in half, so the local
        # Torch constructor cannot introduce another rounding difference.
        candidate = torch.tensor(encoded, device="cpu", dtype=torch.float16)
        if torch.equal(candidate, times):
            return True
    return False


def validate_heap_calendar(
    times,
    row_offsets,
    cursor,
    *,
    rows,
    device,
    dtype=None,
    simulation_dtype=torch.float32,
):
    """Validate a cold calendar/checkpoint before any owned state is changed."""
    device = torch.device(device)
    time_dtype = (
        heap_calendar_dtype(device, simulation_dtype) if dtype is None else dtype
    )
    if time_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Heap calendar times must use a supported floating dtype.")
    for name, value, dtype, shape in (
        ("heap_times", times, time_dtype, None),
        ("heap_row_offsets", row_offsets, torch.long, (rows + 1,)),
        ("heap_cursor", cursor, torch.long, (rows,)),
    ):
        if (
            not torch.is_tensor(value)
            or value.device != device
            or value.dtype != dtype
            or value.ndim != 1
            or (shape is not None and tuple(value.shape) != shape)
            or value.requires_grad
        ):
            raise ValueError(f"Invalid {name} tensor for NetStim heap calendar.")
    if times.numel() < 1 or not bool(torch.isposinf(times[-1])):
        raise ValueError("Heap calendar requires a final positive-infinity sentinel.")
    if bool(times.isnan().any()):
        raise ValueError("NaN heap calendars require native heapq execution.")
    if (
        int(row_offsets[0]) != 0
        or int(row_offsets[-1]) != times.numel() - 1
        or bool((row_offsets[1:] < row_offsets[:-1]).any())
        or bool((cursor < row_offsets[:-1]).any())
        or bool((cursor > row_offsets[1:]).any())
    ):
        raise ValueError("Invalid NetStim heap calendar offsets or cursor.")
    # Ignore adjacency across row boundaries, which need not be ordered.
    for left, right in zip(row_offsets[:-1].tolist(), row_offsets[1:].tolist()):
        if right - left > 1 and bool(
            (times[left + 1 : right] < times[left : right - 1]).any()
        ):
            raise ValueError("NetStim heap calendar rows must be sorted.")


def heap_calendar_head(times, row_end, cursor, *, dtype):
    """Return one simulation-dtype head per row without changing the calendar."""
    selected = times.index_select(0, cursor).to(dtype=dtype)
    return torch.where(cursor < row_end, selected, float("inf"))


def heap_calendar_consume(times, row_end, cursor, t, fires, *, depth, dtype):
    """Consume the complete overdue prefix only for rows that actually fired.

    Search starts at the current cursor, so repeated or decreasing call times
    never rewind consumption. Fixed-depth upper-bound search uses O(rows log K)
    work and O(rows) temporary storage, where K is the longest calendar row.
    Empty rows safely gather an adjacent row or the appended sentinel.
    """
    lower, upper = cursor, row_end
    for _ in range(depth):
        active = lower < upper
        middle = (lower + upper) // 2
        value = times.index_select(0, middle).to(dtype=dtype)
        due = value <= t
        lower = torch.where(active & due, middle + 1, lower)
        upper = torch.where(active & ~due, middle, upper)
    return torch.where(fires, lower, cursor)
