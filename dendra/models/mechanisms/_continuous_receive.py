"""Pure continuous-input resets and ordered per-element reductions."""

import torch


def reset_continuous_values(values, inputs, old_names):
    """Keep each previous input and create this step's empty delivery masks."""
    updates = {}
    received = {}
    for name in inputs:
        current = values[name]
        old_name = old_names.get(name)
        if old_name is not None:
            updates[old_name] = current
        updates[name] = torch.zeros_like(current)
        received[name] = torch.zeros_like(current, dtype=torch.bool)
    return updates, received


def reduce_continuous_values(current, value, received, mask, reduction):
    """Reduce actual deliveries, distinguishing first values from reset zeros."""
    if reduction in ("sum", "add"):
        result = current + torch.where(mask, value, torch.zeros_like(value))
    elif reduction in ("set", "replace", "last"):
        result = torch.where(mask, value, current)
    elif reduction in ("min", "max"):
        combined = (
            torch.minimum(current, value)
            if reduction == "min"
            else torch.maximum(current, value)
        )
        result = torch.where(mask, torch.where(received, combined, value), current)
    else:
        raise ValueError(f"Unsupported continuous reduction mode: {reduction!r}.")
    return result
