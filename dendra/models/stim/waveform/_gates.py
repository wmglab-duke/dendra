"""Shared hard waveform gates with sigmoid surrogate derivatives."""

import torch


def _as_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
    """
    Convert python/numpy scalars to a tensor on ref.device/ref.dtype.
    Leave torch.Tensors (incl. nn.Parameter) untouched to preserve grads.
    """
    if torch.is_tensor(x):
        return x
    return ref.new_tensor(x)


def _time_broadcast_param(x, t: torch.Tensor) -> torch.Tensor:
    """
    Ensure x broadcasts against t (shape [T]) with time as the LAST dim.

    Rules:
      - scalars (0-dim) are fine as-is
      - scalar-time evaluation preserves the parameter shape
      - if the last dim is already 1, keep that explicit broadcast axis
      - otherwise append a trailing singleton time dim, e.g. [B] -> [B,1]
        and [B,C] -> [B,C,1]

    Parameter axes are never inferred to be time merely because their length
    happens to equal the number of evaluation points. The old heuristic made
    output rank depend on ``T`` and silently confused population/compartment
    axes with time.
    """
    x = _as_tensor_like(x, t)
    if x.ndim == 0 or t.ndim == 0:
        return x
    if x.shape[-1] == 1:
        return x
    return x.unsqueeze(-1)


def _gate_sigmoid(distance, tau):
    """Evaluate an edge's sigmoid, including its infinite-distance limits."""
    infinite = torch.isinf(distance)
    # Mask before division: selecting a saturated sigmoid afterwards would
    # still let its backward pass multiply zero by an infinite tau derivative.
    safe_distance = torch.where(infinite, 0.0, distance)
    soft = torch.sigmoid(safe_distance / tau)
    return torch.where(infinite, (distance > 0).to(soft.dtype), soft)


def _rect_gate(t, start, stop, tau, inclusive_stop=False):
    # Canonicalize to broadcast across time
    start = _time_broadcast_param(start, t)
    stop = _time_broadcast_param(stop, t)
    tau = _time_broadcast_param(tau, t)

    tau = torch.clamp(tau, min=1e-6)
    soft = _gate_sigmoid(t - start, tau) * _gate_sigmoid(stop - t, tau)

    if inclusive_stop:
        hard = ((t >= start) & (t <= stop)).to(soft.dtype)
    else:
        hard = ((t >= start) & (t < stop)).to(soft.dtype)

    # Straight-through gate: hard in forward, soft for gradients.
    return hard + (soft - soft.detach())
