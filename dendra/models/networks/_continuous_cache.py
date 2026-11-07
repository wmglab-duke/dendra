"""Pure detached restoration of continuous steady-state delay traffic."""

import math
from collections.abc import Mapping
from typing import NamedTuple

import torch

from ._cache_translation import _round_half_up_time_indices
from ._connection_initialization import continuous_initial_state


def _resample_continuous_time_rows(rows, old_dt, new_dt, *, n_limit):
    """Nearest-neighbor resample future samples, holding both endpoints."""
    if rows.ndim != 2:
        raise ValueError("cached continuous time rows must be 2D")
    n_limit = int(n_limit)
    count, width = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, width))
    if count == 0:
        return rows.new_zeros((n_limit, width))
    if float(old_dt) == float(new_dt) and count == n_limit:
        return rows.detach().clone()
    indices = _round_half_up_time_indices(
        n_limit, new_dt, old_dt, device=rows.device
    ).clamp(0, count - 1)
    return rows.index_select(0, indices).detach().clone()


def _resample_continuous_history_age_rows(rows, old_dt, new_dt, *, n_limit):
    """Resample historical ages; age zero is not an observed source sample."""
    if rows.ndim != 2:
        raise ValueError("cached continuous age rows must be 2D")
    n_limit = int(n_limit)
    count, width = rows.shape
    if n_limit <= 0:
        return rows.new_zeros((0, width))
    if count <= 1:
        return rows.new_zeros((n_limit, width))
    if float(old_dt) == float(new_dt) and count == n_limit:
        return rows.detach().clone()
    result = rows.new_zeros((n_limit, width))
    if n_limit <= 1:
        return result
    indices = _round_half_up_time_indices(n_limit, new_dt, old_dt, device=rows.device)[
        1:
    ].clamp(1, count - 1)
    ages = torch.arange(1, n_limit, device=rows.device, dtype=torch.long)
    return result.index_copy(0, ages, rows.index_select(0, indices)).detach()


class ContinuousCache(NamedTuple):
    """Detached target-timestep rows; None leaves select the replay format."""

    raw_history: torch.Tensor | None
    transformed_history: torch.Tensor | None
    delivery_buffer: torch.Tensor | None
    delivery_mask: torch.Tensor | None


@torch.compiler.assume_constant_result
def _has_empty_cache_vmap_axis(value):
    """Read only hidden batch shape metadata for empty initialization lanes."""
    if value is None:
        return False
    functorch = torch._C._functorch
    while functorch.is_functorch_wrapped_tensor(value):
        batched = functorch.is_batchedtensor(value)
        axis = functorch.maybe_get_bdim(value)
        value = functorch.get_unwrapped(value)
        if batched and value.shape[axis] == 0:
            return True
    return False


def validate_continuous_cache(cache, *, device, dtype, connections, depth, width):
    """Validate a frozen Tensor/None cache packet without reading tensor data."""
    if not isinstance(cache, ContinuousCache):
        raise TypeError("ContinuousCon cache must be normalized before restoration")
    histories = cache.raw_history is not None, cache.transformed_history is not None
    if any(histories):
        if (
            all(histories)
            or cache.delivery_buffer is not None
            or cache.delivery_mask is not None
        ):
            raise ValueError(
                "ContinuousCon cache history and delivery formats are exclusive"
            )
    elif cache.delivery_buffer is None or cache.delivery_mask is None:
        raise ValueError(
            "ContinuousCon cache requires history or payload/presence rows"
        )
    for name, value in zip(cache._fields, cache):
        if value is None:
            continue
        expected_width = connections if name.endswith("history") else width
        expected_dtype = torch.bool if name == "delivery_mask" else dtype
        if not torch.is_tensor(value):
            raise TypeError(f"ContinuousCon cache {name} must be a Tensor")
        if (
            tuple(value.shape) != (depth, expected_width)
            or value.dtype != expected_dtype
            or value.device != device
        ):
            raise ValueError(
                f"ContinuousCon cache {name} must match {device}/{expected_dtype}/{(depth, expected_width)}"
            )


def normalize_continuous_cache(
    cache, *, dt, device, dtype, connections, depth, width, resample_dt=None
):
    """Own and pre-resample native caches to a Tensor/None-only frozen schema."""
    if isinstance(cache, ContinuousCache):
        validate_continuous_cache(
            cache,
            device=device,
            dtype=dtype,
            connections=connections,
            depth=depth,
            width=width,
        )
        return ContinuousCache(
            *(None if value is None else value.detach().clone() for value in cache)
        )
    if not isinstance(cache, Mapping):
        raise TypeError("ContinuousCon state cache must be a mapping")
    old_dt = float(cache.get("dt", dt))
    new_dt = float(dt) if resample_dt is None else float(resample_dt)
    if (
        not math.isfinite(old_dt)
        or old_dt <= 0
        or not math.isfinite(new_dt)
        or new_dt <= 0
    ):
        raise ValueError("ContinuousCon cache timesteps must be finite and positive")
    history = "pre_value_history" in cache
    source = cache["pre_value_history"] if history else cache["delivery_buffer"]
    if not torch.is_tensor(source):
        raise TypeError("ContinuousCon cached values must be a tensor")
    expected = connections if history else width
    if source.ndim != 2 or source.shape[1] != expected:
        name = "pre-value history" if history else "delivery_buffer"
        raise ValueError(
            f"Cached ContinuousCon {name} has incompatible shape: "
            f"{tuple(source.shape)} vs (*, {expected})."
        )
    values = source.detach().to(device=device, dtype=dtype)
    if history:
        values = _resample_continuous_history_age_rows(
            values, old_dt, new_dt, n_limit=depth
        )
        raw = cache.get("value_layout") == "raw" or int(cache.get("version", 2)) >= 3
        return ContinuousCache(
            values if raw else None, None if raw else values, None, None
        )
    mask = cache.get("delivery_mask")
    if mask is None:
        # Legacy payload-only caches cannot recover genuinely scheduled zeros.
        mask = values != 0
    else:
        if not torch.is_tensor(mask):
            raise TypeError("Cached ContinuousCon delivery_mask must be a tensor")
        if tuple(mask.shape) != tuple(source.shape):
            raise ValueError(
                "Cached ContinuousCon delivery_mask has incompatible shape: "
                f"{tuple(mask.shape)} vs {tuple(source.shape)}."
            )
        mask = mask.detach().to(device=device, dtype=torch.bool)
    return ContinuousCache(
        None,
        None,
        _resample_continuous_time_rows(values, old_dt, new_dt, n_limit=depth),
        _resample_continuous_time_rows(mask, old_dt, new_dt, n_limit=depth),
    )


class _ContinuousCacheRoutes(NamedTuple):
    delay_steps: torch.Tensor
    post_idx: torch.Tensor
    last_age: int


def continuous_cache_routes(delay_steps, post_idx, *, depth, width):
    """Retain compact frozen topology and a static pending-history bound."""
    del width
    last_age = int(delay_steps.detach().max().cpu()) if delay_steps.numel() else 0
    return _ContinuousCacheRoutes(delay_steps, post_idx, min(int(depth) - 1, last_age))


def continuous_history_delivery(
    history,
    weights,
    *,
    depth,
    width,
    delay_steps=None,
    post_idx=None,
    routes=None,
    transform=None,
):
    """Replay complete source rows, then accumulate current weighted traffic."""
    history = history.detach()
    weights = weights.detach()
    if history.ndim != 2 or history.shape[1] != weights.numel():
        raise ValueError(
            "Cached ContinuousCon pre-value history has incompatible shape: "
            f"{tuple(history.shape)} vs (*, {weights.numel()})."
        )
    flat = history.new_zeros((depth * width,))
    presence = torch.zeros((depth * width,), device=history.device, dtype=torch.bool)
    if routes is None:
        routes = continuous_cache_routes(
            delay_steps, post_idx, depth=depth, width=width
        )
    max_age = min(history.shape[0], depth, routes.last_age + 1)
    for age in range(1, max_age):
        remaining = routes.delay_steps - age
        active = remaining >= 0
        # The leading sentinel absorbs inactive edges. Shift real indices so
        # out-of-capacity active destinations still fail rather than disappear.
        destination = torch.where(active, remaining * width + routes.post_idx + 1, 0)
        row = history[age]
        if transform is not None:
            # Apply a memoryless transform to the complete edge row before
            # selecting pending edges, matching ordinary native restoration.
            row = transform(row).detach()
        values = torch.where(active, row * weights, 0)
        flat = torch.cat((flat.new_zeros((1,)), flat)).index_add(
            0, destination, values
        )[1:]
        presence = torch.cat((presence.new_zeros((1,)), presence)).scatter_add(
            0, destination, active
        )[1:]
    return flat.reshape(depth, width).detach(), presence.reshape(depth, width)


def restore_continuous_cache(state, cache, weights, *, dt, t, routes, transform=None):
    """Restore normalized cache data as detached initialization carry."""
    queue = state["delivery_buffer"]
    depth, width = queue.shape
    result = continuous_initial_state(state, t=t, dt=dt, dtype=queue.dtype)
    history = (
        cache.raw_history
        if cache.raw_history is not None
        else cache.transformed_history
    )
    if history is not None:
        delivery, mask = continuous_history_delivery(
            history,
            weights,
            depth=depth,
            width=width,
            routes=routes,
            transform=transform if cache.raw_history is not None else None,
        )
    else:
        delivery, mask = (
            cache.delivery_buffer.detach().clone(),
            cache.delivery_mask.detach().clone(),
        )
    result["delivery_buffer"], result["delivery_mask"] = delivery, mask
    return {name: value.detach() for name, value in result.items()}
