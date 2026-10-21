"""Pure bounded calendars for weighted pending NetCon deliveries.

Entries retain enqueue order and already contain the enqueue-time parameters.
The valid prefix is compact, so storage depends on pending contributions rather
than the product of delay depth and postsynaptic width. Overflow is explicit:
an append either publishes every selected entry or leaves the calendar intact.
"""

from collections.abc import MutableMapping
from copy import deepcopy
from typing import NamedTuple

import torch


class CalendarState(NamedTuple):
    slots: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    size: torch.Tensor
    overflow: torch.Tensor


def pending_calendar_empty(capacity, *, device, dtype):
    """Allocate an empty fixed-capacity calendar outside transformed execution."""
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("Pending calendar capacity must be a nonnegative integer.")
    return CalendarState(
        torch.zeros(capacity, device=device, dtype=torch.long),
        torch.zeros(capacity, device=device, dtype=torch.long),
        torch.zeros(capacity, device=device, dtype=dtype),
        torch.zeros(1, device=device, dtype=torch.long),
        torch.zeros(1, device=device, dtype=torch.bool),
    )


def pending_calendar_append(state, slots, indices, values, active=None):
    """Append selected contributions in order, without changing any input.

    Candidate shapes and capacity are fixed during transforms. An inactive
    candidate may contain NaN or an invalid index: it is masked before storage.
    Overflow is sticky and prevents partial writes, including later appends.
    """
    capacity = state.values.numel()
    active = torch.ones_like(slots, dtype=torch.bool) if active is None else active
    ranks = active.to(torch.long).cumsum(0)
    count = active.sum().reshape(1)
    overflow = state.overflow | (state.size + count > capacity)
    accepted = active & ~overflow
    destination = torch.where(accepted, state.size + ranks - 1, capacity)

    def write(old, candidate):
        # Duplicate writes only touch the discarded sentinel, never valid data.
        padded = torch.cat((old, old.new_zeros(1)))
        candidate = torch.where(accepted, candidate.to(dtype=old.dtype), 0)
        return padded.scatter(0, destination, candidate)[:-1]

    return CalendarState(
        write(state.slots, slots),
        write(state.indices, indices),
        write(state.values, values),
        torch.where(overflow, state.size, state.size + count),
        overflow,
    )


def pending_calendar_pop(state, slot, *, width, base=None, _scratch=None):
    """Reduce due entries in enqueue order and compact all surviving entries.

    ``base`` adds deliveries to an existing pure payload (for packed history).
    The private native adapter may donate ``_scratch`` as that existing payload;
    callers own clearing it when an initially empty payload is required.
    """
    capacity = state.values.numel()
    positions = torch.arange(capacity, device=state.slots.device)
    valid = positions < state.size
    slot = slot.reshape(1) if torch.is_tensor(slot) else slot
    due = valid & (state.slots == slot)
    values = torch.where(due, state.values, 0)
    if _scratch is not None:
        # Native scratch is owned by its adapter. Pure callers use the padded
        # output sentinel so masked entries never touch a real postsynaptic bin.
        payload = _scratch
        if width:
            payload.index_add_(0, torch.where(due, state.indices, 0), values)
    else:
        out = state.values.new_zeros(width) if base is None else base
        padded = torch.cat((out, out.new_zeros(1)))
        indices = torch.where(due, state.indices, width)
        # scatter_add admits mapped indices even for zero lanes. CPU
        # Float16/BFloat16 scatter_add promotes collision accumulation; native
        # index_put retains the stepwise rounding of native index_add instead.
        if values.device.type == "cpu" and values.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            payload = padded.index_put((indices,), values, accumulate=True)[:-1]
        else:
            payload = padded.scatter_add(0, indices, values)[:-1]
    keep = valid & ~due
    ranks = keep.to(torch.long).cumsum(0)
    destination = torch.where(keep, ranks - 1, capacity)

    def compact(old):
        clean = old.new_zeros(capacity + 1)
        return clean.scatter(0, destination, torch.where(keep, old, 0))[:-1]

    return (
        CalendarState(
            compact(state.slots),
            compact(state.indices),
            compact(state.values),
            keep.sum().reshape(1),
            state.overflow,
        ),
        payload,
    )


def pending_calendar_from_legacy(chunks, *, capacity=None, device, dtype):
    """Import legacy slot/chunk dictionaries without detaching floating values."""
    slots, indices, values = [], [], []
    for slot, bucket in chunks.items():
        for idx, value in bucket:
            idx = idx.to(device=device, dtype=torch.long).reshape(idx.numel())
            value = value.to(device=device, dtype=dtype).reshape(value.numel())
            if idx.numel() != value.numel():
                raise ValueError("Pending calendar indices and values must align.")
            slots.append(torch.full_like(idx, int(slot)))
            indices.append(idx)
            values.append(value)
    count = sum(value.numel() for value in values)
    capacity = count if capacity is None else capacity
    state = pending_calendar_empty(capacity, device=device, dtype=dtype)
    if count > capacity:
        raise ValueError(
            f"Pending calendar capacity {capacity} cannot hold {count} entries."
        )
    if count == 0:
        return state
    return pending_calendar_append(
        state, torch.cat(slots), torch.cat(indices), torch.cat(values)
    )


def pending_calendar_to_legacy(state):
    """Export legal legacy chunks in bucket order, retaining value gradients."""
    if bool(state.overflow.any()):
        raise RuntimeError("Pending calendar overflow; increase pending_capacity.")
    size = int(state.size.item())
    buckets = {}
    # First occurrence defines dictionary insertion order. It matters when a
    # state cache rebins several physical slots into the same new delay slot.
    for position, slot in enumerate(state.slots[:size].detach().cpu().tolist()):
        buckets.setdefault(int(slot), []).append(position)
    return {
        slot: [
            (
                state.indices.index_select(
                    0, torch.tensor(positions, device=state.indices.device)
                ),
                state.values.index_select(
                    0, torch.tensor(positions, device=state.values.device)
                ),
            )
        ]
        for slot, positions in buckets.items()
    }


class PendingCalendarMapping(MutableMapping):
    """Cold legacy dictionary view over an owner's persistent tensor calendar.

    Native hot paths access the tensor state directly. Item assignment, clear,
    movement, checkpoint helpers, and older private inspection code can retain
    the slot/chunk dictionary representation without a second runtime engine.
    """

    def __init__(self, owner, attribute):
        self.owner, self.attribute = owner, attribute

    @property
    def state(self):
        return getattr(self.owner, self.attribute)

    @property
    def tensor_state(self):
        """Expose the synchronized tensor state to functional lowering adapters."""
        return self.state

    def replace_tensor_state(self, state):
        """Bind a validated functional state without regrouping legacy buckets."""
        if not isinstance(state, CalendarState):
            raise TypeError("Pending calendar state must be a CalendarState.")
        previous = self.state
        capacity = state.values.numel()
        if (
            any(value.device != previous.values.device for value in state)
            or state.values.dtype != previous.values.dtype
            or state.slots.dtype != torch.long
            or state.indices.dtype != torch.long
            or state.size.dtype != torch.long
            or state.overflow.dtype != torch.bool
            or state.slots.shape != (capacity,)
            or state.indices.shape != (capacity,)
            or state.values.shape != (capacity,)
            or state.size.shape != (1,)
            or state.overflow.shape != (1,)
            or bool(state.overflow.any())
            or not 0 <= int(state.size.item()) <= capacity
        ):
            raise ValueError("Invalid pending calendar tensor state.")
        setattr(self.owner, self.attribute, state)
        setattr(self.owner, f"{self.attribute}_size", int(state.size.item()))

    def _replace(self, chunks):
        count = sum(value.numel() for bucket in chunks.values() for _, value in bucket)
        state = self.state
        setattr(
            self.owner,
            self.attribute,
            pending_calendar_from_legacy(
                chunks,
                capacity=max(state.values.numel(), count, 16),
                device=state.values.device,
                dtype=state.values.dtype,
            ),
        )
        setattr(self.owner, f"{self.attribute}_size", count)

    def __getitem__(self, key):
        return pending_calendar_to_legacy(self.state)[key]

    def __setitem__(self, key, value):
        chunks = pending_calendar_to_legacy(self.state)
        chunks[int(key)] = value
        self._replace(chunks)

    def __delitem__(self, key):
        chunks = pending_calendar_to_legacy(self.state)
        del chunks[key]
        self._replace(chunks)

    def __iter__(self):
        return iter(pending_calendar_to_legacy(self.state))

    def __len__(self):
        return len(pending_calendar_to_legacy(self.state))

    def __bool__(self):
        return bool(getattr(self.owner, f"{self.attribute}_size"))

    def items(self):
        return pending_calendar_to_legacy(self.state).items()

    def values(self):
        return pending_calendar_to_legacy(self.state).values()

    def clear(self):
        state = self.state
        setattr(
            self.owner,
            self.attribute,
            pending_calendar_empty(
                state.values.numel(),
                device=state.values.device,
                dtype=state.values.dtype,
            ),
        )
        setattr(self.owner, f"{self.attribute}_size", 0)

    def __deepcopy__(self, memo):
        return deepcopy(pending_calendar_to_legacy(self.state), memo)
