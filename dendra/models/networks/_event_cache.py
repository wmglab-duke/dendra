"""Detached event-cache recipes and shared pure restoration of NetCon carry."""

from typing import NamedTuple

import torch

from ._connection_initialization import event_initial_step
from ._pending_calendar import (
    CalendarState,
    pending_calendar_append,
    pending_calendar_from_legacy,
)


class EventCache(NamedTuple):
    packed_history: torch.Tensor | None = None
    gate_history: torch.Tensor | None = None
    delivery_buffer: torch.Tensor | None = None
    event_queue: torch.Tensor | None = None
    has_spiked: torch.Tensor | None = None
    is_spiking: torch.Tensor | None = None
    source_has_spiked: torch.Tensor | None = None
    pending: CalendarState | None = None
    event_pending: CalendarState | None = None


def clone_event_cache(cache):
    """Own normalized cache leaves without retaining their warm-up graphs."""
    return EventCache(
        *(
            value.detach().clone()
            if torch.is_tensor(value)
            else (
                CalendarState(*(leaf.detach().clone() for leaf in value))
                if isinstance(value, CalendarState)
                else value
            )
            for value in cache
        )
    )


def _require_cache_value(condition, message, *, overflow=False):
    # Functorch has no batching rule for assertion operators. Validate concrete
    # packets before mapping; transformed restoration keeps its tensor-only path.
    if torch._C._are_functorch_transforms_active():
        return
    condition = condition.all()
    if torch.compiler.is_compiling():
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise (RuntimeError if overflow else ValueError)(message)


def validate_event_cache(
    cache,
    *,
    runtime,
    depth,
    delivery_depth,
    width,
    connections,
    device,
    dtype,
    source_device,
    source_dtype,
    metadata,
    binary_history=False,
):
    """Check normalized schema and concrete cache representability.

    Compiled restoration asserts value invariants in the graph. Concrete
    packets must be validated before functorch transforms, whose assertion
    operators do not support mapped values.
    """
    if not isinstance(cache, EventCache):
        raise TypeError("NetCon cache must be normalized before restoration")
    histories = cache.packed_history is not None, cache.gate_history is not None
    if all(histories) or (
        any(histories)
        and (cache.delivery_buffer is not None or cache.event_queue is not None)
    ):
        raise ValueError("NetCon cache history and delivery formats are exclusive")
    if cache.event_queue is not None and cache.delivery_buffer is None:
        raise ValueError("NetCon cached event_queue requires delivery_buffer")
    if not any(histories) and cache.delivery_buffer is None and cache.pending is None:
        raise ValueError("NetCon cache requires history or pending delivery traffic")

    def tensor(value, name, shape, expected_dtype=dtype, expected_device=device):
        if not torch.is_tensor(value):
            raise TypeError(f"NetCon cache {name} must be a Tensor")
        if (
            tuple(value.shape) != shape
            or value.dtype != expected_dtype
            or value.device != expected_device
        ):
            raise ValueError(
                f"NetCon cache {name} must match {expected_device}/{expected_dtype}/{shape}"
            )

    if cache.packed_history is not None:
        words = max(1, (metadata["bitpack_source_pre_idx"].numel() + 62) // 63)
        tensor(cache.packed_history, "packed_history", (depth, words), torch.int64)
    if cache.gate_history is not None:
        gates = cache.gate_history
        tensor(gates, "gate_history", (depth, connections))
        if runtime in {"source_history", "bitpacked_history"}:
            source = _source_gates(gates, metadata)
            _require_cache_value(
                source.index_select(1, metadata["source_pos"]) == gates,
                "Cached gates disagree across connections from the same source",
            )
            if binary_history:
                _require_cache_value(
                    (gates == 0) | (gates == 1),
                    "Cached gates cannot represent hard binary source history",
                )
    for name, value, shape, expected_dtype in (
        ("delivery_buffer", cache.delivery_buffer, (delivery_depth, width), dtype),
        ("event_queue", cache.event_queue, (delivery_depth, connections), torch.int32),
    ):
        if value is not None:
            tensor(value, name, shape, expected_dtype)
    for name, value, expected_dtype, count in (
        ("has_spiked", cache.has_spiked, torch.bool, connections),
        ("is_spiking", cache.is_spiking, source_dtype, connections),
        (
            "source_has_spiked",
            cache.source_has_spiked,
            torch.bool,
            metadata["bitpack_source_pre_idx"].numel(),
        ),
    ):
        if value is not None:
            if not torch.is_tensor(value):
                raise TypeError(f"NetCon cache {name} must be a Tensor")
            if value.numel() not in (0, count):
                raise ValueError(f"NetCon cache {name} has incompatible shape")
            tensor(value, name, tuple(value.shape), expected_dtype, source_device)
    for name, calendar, calendar_width, value_dtype in (
        ("pending", cache.pending, width, dtype),
        ("event_pending", cache.event_pending, connections, torch.int32),
    ):
        if calendar is None:
            continue
        if not isinstance(calendar, CalendarState):
            raise TypeError(f"NetCon cache {name} must be a CalendarState")
        if not torch.is_tensor(calendar.values) or calendar.values.ndim != 1:
            raise ValueError(f"NetCon cache {name}.values must be a vector")
        capacity = calendar.values.numel()
        for field, value, shape, expected_dtype in (
            ("slots", calendar.slots, (capacity,), torch.long),
            ("indices", calendar.indices, (capacity,), torch.long),
            ("values", calendar.values, (capacity,), value_dtype),
            ("size", calendar.size, (1,), torch.long),
            ("overflow", calendar.overflow, (1,), torch.bool),
        ):
            tensor(value, f"{name}.{field}", shape, expected_dtype)
        _require_cache_value(
            (calendar.size >= 0) & (calendar.size <= capacity),
            f"Cached {name} calendar size is outside its capacity",
        )
        _require_cache_value(
            ~calendar.overflow,
            f"Cached {name} calendar has overflowed; restore an earlier valid cache",
            overflow=True,
        )
        active = torch.arange(capacity, device=device) < calendar.size
        _require_cache_value(
            ~active | ((calendar.slots >= 0) & (calendar.slots < depth)),
            f"Cached {name} calendar slot is outside its delay ring",
        )
        _require_cache_value(
            ~active | ((calendar.indices >= 0) & (calendar.indices < calendar_width)),
            f"Cached {name} calendar index is outside its topology",
        )


def normalize_event_cache(
    cache,
    *,
    backend,
    runtime,
    depth,
    delivery_depth,
    width,
    dt,
    device,
    dtype,
    source_device,
    source_dtype,
    track_events,
    metadata=None,
    binary_history=False,
):
    """Freeze and translate a native dict/legacy tuple outside transforms.

    The cache is a detached steady-state recipe. Numeric histories and weighted
    pending traffic use the native timestep translation and bucket ordering.
    Runtime parameters are deliberately absent from this cold packet.
    """
    from .netcon import (
        _dilate_history_age_rows,
        _dilate_packed_history_age_rows,
        _dilate_packed_time_rows,
        _dilate_time_rows,
        _restore_calendar_chunks,
        _ring_rows_to_age_rows,
    )

    if isinstance(cache, EventCache):
        validate_event_cache(
            cache,
            runtime=runtime,
            depth=depth,
            delivery_depth=delivery_depth,
            width=width,
            connections=metadata["source_pos"].numel(),
            device=device,
            dtype=dtype,
            source_device=source_device,
            source_dtype=source_dtype,
            metadata=metadata,
            binary_history=binary_history,
        )
        return clone_event_cache(cache)
    if isinstance(cache, tuple):
        old_dt, has_spiked, is_spiking, delivery = cache
        cache = {
            "backend": "dense",
            "dt": float(old_dt),
            "has_spiked": has_spiked,
            "is_spiking": is_spiking,
            "backend_state": {"delivery_buffer": delivery},
        }
    if not isinstance(cache, dict):
        raise TypeError("NetCon state cache must be a dict or legacy tuple")
    if cache.get("backend", "dense") != backend:
        raise ValueError(
            f"Cannot restore NetCon state cache from backend {cache.get('backend', 'dense')!r} into backend {backend!r}."
        )
    old_dt = float(cache.get("dt", dt))
    values = cache.get("backend_state", {})

    def tensor(value, *, target_device=device, target_dtype=dtype):
        return (
            None
            if value is None
            else value.detach().to(device=target_device, dtype=target_dtype).clone()
        )

    def calendar(name, target_dtype):
        chunks = _restore_calendar_chunks(
            values.get(name, {}),
            old_dt=old_dt,
            new_dt=dt,
            depth=depth,
            idx_device=device,
            idx_dtype=torch.long,
            val_device=device,
            val_dtype=target_dtype,
        )
        return pending_calendar_from_legacy(chunks, device=device, dtype=target_dtype)

    packed = gates = delivery = counts = pending = event_pending = None
    if "source_spike_history_packed" in values or "spike_history_packed" in values:
        packed = tensor(
            values.get(
                "source_spike_history_packed", values.get("spike_history_packed")
            ),
            target_dtype=torch.int64,
        )
        if packed is None:
            raise KeyError("Bitpacked NetCon cache is missing source spike history")
        if values.get("history_layout", "ring_zero") == "age":
            packed = _dilate_packed_history_age_rows(packed, old_dt, dt, n_limit=depth)
        else:
            packed = _ring_rows_to_age_rows(
                _dilate_packed_time_rows(packed, old_dt, dt, n_limit=depth), 0
            )
        if runtime == "bitpacked_history":
            pending = calendar("sparse_calendar", dtype)
    elif "gate_history" in values:
        gates = _dilate_history_age_rows(
            tensor(values["gate_history"]), old_dt, dt, n_limit=depth
        )
    elif runtime == "sparse_calendar":
        pending = calendar("sparse_calendar", dtype)
        if track_events:
            event_pending = calendar("sparse_event_calendar", torch.int32)
    else:
        delivery = tensor(values.get("delivery_buffer"))
        if delivery is None:
            raise KeyError("Dense NetCon cache is missing delivery_buffer")
        if delivery.ndim != 2 or delivery.shape[1] != width:
            raise ValueError(
                f"Cached NetCon delivery_buffer has incompatible shape: {tuple(delivery.shape)} vs (*, {width})."
            )
        delivery = _dilate_time_rows(delivery, old_dt, dt, n_limit=delivery_depth)
        if track_events and values.get("event_queue") is not None:
            counts = _dilate_time_rows(
                tensor(values["event_queue"], target_dtype=torch.int32),
                old_dt,
                dt,
                n_limit=delivery_depth,
            )
    normalized = EventCache(
        packed,
        gates,
        delivery,
        counts,
        tensor(
            cache.get("has_spiked"),
            target_device=source_device,
            target_dtype=torch.bool,
        ),
        tensor(
            cache.get("is_spiking"),
            target_device=source_device,
            target_dtype=source_dtype,
        ),
        tensor(
            values.get("bitpack_source_has_spiked"),
            target_device=source_device,
            target_dtype=torch.bool,
        ),
        pending,
        event_pending,
    )
    validate_event_cache(
        normalized,
        runtime=runtime,
        depth=depth,
        delivery_depth=delivery_depth,
        width=width,
        connections=metadata["source_pos"].numel(),
        device=device,
        dtype=dtype,
        source_device=source_device,
        source_dtype=source_dtype,
        metadata=metadata,
        binary_history=binary_history,
    )
    return normalized


def event_cache_metadata(connection, *, frozen=None, copy=True):
    """Copy only immutable routing needed to restore an event cache."""
    names = (
        "bitpack_source_pre_idx",
        "bitpack_source_word_idx",
        "bitpack_source_bit_mask",
        "bitpack_conn_word_idx",
        "bitpack_conn_bit_mask",
    )
    frozen = {} if frozen is None else frozen
    result = {
        name: (
            frozen[name]
            if name in frozen
            else (
                getattr(connection, name).detach().clone()
                if copy
                else getattr(connection, name)
            )
        )
        for name in names
    }
    source = result["bitpack_source_pre_idx"].to(
        device=connection.device, dtype=torch.long
    )
    pre = connection.pre_idx.to(device=connection.device, dtype=torch.long)
    known_positions = frozen.get(
        "source_history_conn_source_pos", connection.source_history_conn_source_pos
    )
    if known_positions.numel() == pre.numel():
        positions = (
            known_positions.detach().clone()
            if copy and "source_history_conn_source_pos" not in frozen
            else known_positions
        )
    else:
        positions = (
            torch.searchsorted(source, pre) if source.numel() else torch.zeros_like(pre)
        )
    first = torch.full(
        (source.numel(),), pre.numel(), device=connection.device, dtype=torch.long
    )
    if source.numel():
        first = first.scatter_reduce(
            0,
            positions,
            torch.arange(pre.numel(), device=connection.device),
            reduce="amin",
            include_self=True,
        )
    result.update(source_pos=positions, first_connection=first)
    return result


def event_cache_pending_count(cache, delay_steps, metadata, depth):
    """Return the native adapter's exact required payload capacity cold."""
    count = 0 if cache.pending is None else int(cache.pending.size.item())
    history = (
        cache.packed_history if cache.packed_history is not None else cache.gate_history
    )
    if history is not None:
        for age in range(1, min(depth, history.shape[0])):
            gate = (
                cache.gate_history[age]
                if cache.gate_history is not None
                else (
                    torch.bitwise_and(
                        history[age].index_select(0, metadata["bitpack_conn_word_idx"]),
                        metadata["bitpack_conn_bit_mask"],
                    )
                    != 0
                )
            )
            count += int(((delay_steps >= age) & (gate != 0)).sum().item())
    return count


def _ring(age, template):
    count = min(age.shape[0], template.shape[0])
    indices = (-torch.arange(count, device=template.device)).remainder(
        template.shape[0]
    )
    return torch.zeros_like(template).index_copy(0, indices, age[:count])


def _source_gates(gates, metadata):
    first = metadata["first_connection"]
    if gates.shape[1] == 0:
        return gates.new_zeros((gates.shape[0], first.numel()))
    valid = first < gates.shape[1]
    return torch.where(
        valid.unsqueeze(0),
        gates.index_select(1, first.clamp_max(gates.shape[1] - 1)),
        0,
    )


def _unpack(packed, metadata, dtype):
    words = packed.index_select(1, metadata["bitpack_source_word_idx"])
    return (
        torch.bitwise_and(words, metadata["bitpack_source_bit_mask"].unsqueeze(0)) != 0
    ).to(dtype)


def _pack(gates, template, metadata):
    rows = []
    for age in range(min(gates.shape[0], template.shape[0])):
        values = metadata["bitpack_source_bit_mask"] * (gates[age] != 0).to(torch.int64)
        rows.append(
            template.new_zeros(template.shape[1]).scatter_add(
                0, metadata["bitpack_source_word_idx"], values
            )
        )
    age_rows = torch.stack(rows) if rows else template.new_zeros((0, template.shape[1]))
    return _ring(age_rows, template)


def _calendar(state, prefix):
    return CalendarState(*(state[prefix + field] for field in CalendarState._fields))


def _put_calendar(state, prefix, calendar):
    state.update(
        (prefix + field, value)
        for field, value in zip(CalendarState._fields, calendar, strict=True)
    )


def _import_calendar(state, prefix, cached, *, depth, width):
    if prefix + "size" not in state:
        if cached is not None and cached.values.numel():
            raise ValueError("This NetCon runtime has no pending-calendar storage")
        return
    current = CalendarState(
        *(torch.zeros_like(value) for value in _calendar(state, prefix))
    )
    if cached is not None:
        active = (
            torch.arange(cached.values.numel(), device=cached.values.device)
            < cached.size
        )
        # Host/compiled validation rejects malformed packets. Mapped restoration
        # cannot assert, so retain invalidity as explicit sticky overflow rather
        # than silently truncating a prefix or clearing its status.
        invalid = (
            cached.overflow
            | (cached.size < 0)
            | (cached.size > cached.values.numel())
            | (active & ((cached.slots < 0) | (cached.slots >= depth))).any().reshape(1)
            | (active & ((cached.indices < 0) | (cached.indices >= width)))
            .any()
            .reshape(1)
        )
        current = current._replace(overflow=invalid)
        current = pending_calendar_append(
            current, cached.slots, cached.indices, cached.values, active
        )
    _put_calendar(state, prefix, current)


def restore_event_cache(
    state,
    cache,
    weights,
    *,
    delay_steps,
    post_idx,
    metadata,
    runtime,
    depth,
    width,
    t,
    dt,
    binary_history=False,
):
    """Restore explicit carry without mutating cache, state, or parameters.

    Cached traffic is detached. Materialized future payloads use detached
    current weights and fixed quantization; compact histories retain no weight
    dependency until their delivery-time projection reads current parameters.
    """
    validate_event_cache(
        cache,
        runtime=runtime,
        depth=depth,
        delivery_depth=state["delivery_buffer"].shape[0],
        width=width,
        connections=weights.numel(),
        device=weights.device,
        dtype=weights.dtype,
        source_device=metadata["bitpack_source_pre_idx"].device,
        source_dtype=(
            cache.is_spiking.dtype if cache.is_spiking is not None else weights.dtype
        ),
        metadata=metadata,
        binary_history=binary_history,
    )
    cache = EventCache(
        *(
            (
                value.detach()
                if torch.is_tensor(value)
                else (
                    CalendarState(*(leaf.detach() for leaf in value))
                    if isinstance(value, CalendarState)
                    else value
                )
            )
            for value in cache
        )
    )
    result = dict(state)
    _import_calendar(result, "pending_", cache.pending, depth=depth, width=width)
    _import_calendar(
        result,
        "event_pending_",
        cache.event_pending,
        depth=depth,
        width=weights.numel(),
    )
    packed, gates = cache.packed_history, cache.gate_history
    if packed is not None:
        expected = max(1, (metadata["bitpack_source_pre_idx"].numel() + 62) // 63)
        if packed.ndim != 2 or packed.shape[1] != expected:
            raise ValueError(
                "Cached bitpacked spike history has incompatible word count"
            )
        if "spike_history_packed" in result and result["spike_history_packed"].numel():
            result["spike_history_packed"] = _ring(
                packed, result["spike_history_packed"]
            )
    if gates is not None and (gates.ndim != 2 or gates.shape[1] != weights.numel()):
        raise ValueError("Cached NetCon gate history has incompatible shape")
    if packed is not None or gates is not None:
        result["delivery_buffer"] = torch.zeros_like(state["delivery_buffer"])
        if runtime in {"source_history", "bitpacked_history"}:
            if (
                "source_gate_history" in result
                and result["source_gate_history"].numel()
            ):
                source = (
                    _unpack(packed, metadata, weights.dtype)
                    if packed is not None
                    else _source_gates(gates, metadata)
                )
                result["source_gate_history"] = _ring(
                    source, state["source_gate_history"]
                )
            elif packed is None:
                result["spike_history_packed"] = _pack(
                    _source_gates(gates, metadata),
                    state["spike_history_packed"],
                    metadata,
                )
        else:
            detached_weights = weights.detach()
            for age in range(
                1, min(depth, (packed if packed is not None else gates).shape[0])
            ):
                gate = (
                    gates[age]
                    if packed is None
                    else (
                        torch.bitwise_and(
                            packed[age].index_select(
                                0, metadata["bitpack_conn_word_idx"]
                            ),
                            metadata["bitpack_conn_bit_mask"],
                        )
                        != 0
                    ).to(weights.dtype)
                )
                remaining = delay_steps - age
                active = (remaining >= 0) & (gate != 0)
                values = torch.where(active, detached_weights * gate, 0)
                if runtime == "sparse_calendar":
                    slots = remaining.remainder(depth)
                    order = torch.argsort(
                        torch.where(active, slots, depth), stable=True
                    )
                    pending = pending_calendar_append(
                        _calendar(result, "pending_"),
                        slots.index_select(0, order),
                        post_idx.index_select(0, order),
                        values.index_select(0, order),
                        active.index_select(0, order),
                    )
                    _put_calendar(result, "pending_", pending)
                else:
                    flat = result["delivery_buffer"].reshape(-1)
                    indices = torch.where(
                        active, remaining * width + post_idx, flat.numel()
                    )
                    padded = torch.cat((flat, flat.new_zeros(1)))
                    updated = (
                        padded.index_put((indices,), values, accumulate=True)
                        if values.device.type == "cpu"
                        and values.dtype in (torch.float16, torch.bfloat16)
                        else padded.scatter_add(0, indices, values)
                    )
                    result["delivery_buffer"] = updated[:-1].reshape_as(
                        state["delivery_buffer"]
                    )
    elif cache.delivery_buffer is not None:
        result["delivery_buffer"] = cache.delivery_buffer.clone()
        if cache.event_queue is not None and "event_queue" in result:
            result["event_queue"] = cache.event_queue.clone()
    if (
        cache.is_spiking is not None
        and "is_spiking" in result
        and result["is_spiking"].numel()
    ):
        result["is_spiking"] = (
            cache.is_spiking.reshape_as(result["is_spiking"]).clone()
            if cache.is_spiking.numel() == result["is_spiking"].numel()
            else cache.is_spiking.clone()
        )
    if cache.has_spiked is not None:
        hs = cache.has_spiked
        if "has_spiked" in result and result["has_spiked"].numel():
            result["has_spiked"] = (
                hs.reshape_as(result["has_spiked"]).clone()
                if hs.numel() == result["has_spiked"].numel()
                else hs.clone()
            )
        if (
            runtime == "source_history"
            and "bitpack_source_has_spiked" in result
            and hs.numel() == weights.numel()
        ):
            first = metadata["first_connection"].to(hs.device)
            if hs.numel():
                result["bitpack_source_has_spiked"] = torch.where(
                    first < hs.numel(),
                    hs.index_select(0, first.clamp_max(hs.numel() - 1)),
                    False,
                )
    if (
        cache.source_has_spiked is not None
        and "bitpack_source_has_spiked" in result
        and result["bitpack_source_has_spiked"].numel()
    ):
        source_hs = cache.source_has_spiked.reshape_as(
            result["bitpack_source_has_spiked"]
        )
        result["bitpack_source_has_spiked"] = source_hs.clone()
        if (
            "has_spiked" in result
            and result["has_spiked"].numel() == weights.numel()
            and source_hs.numel()
        ):
            result["has_spiked"] = source_hs.index_select(
                0, metadata["source_pos"].to(source_hs.device)
            )
    result["current_time_step"] = torch.zeros_like(state["current_time_step"])
    result["global_step"] = event_initial_step(t, dt, state["global_step"])
    return result
