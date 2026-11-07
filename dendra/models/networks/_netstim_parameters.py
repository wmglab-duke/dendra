"""Shared deterministic source evaluation and shape policy for NetStim."""

import torch


def broadcast_netstim_parameter(value, shape, *, device, dtype, name):
    """Cast a renewal input and retain native per-batch broadcasting."""
    value = torch.as_tensor(value, device=device, dtype=dtype)
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as direct_exc:
        if value.ndim > 0:
            try:
                return torch.broadcast_to(value.unsqueeze(-1), shape)
            except RuntimeError:
                pass
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} cannot be broadcast "
            f"to NetStim state shape {shape}."
        ) from direct_exc


def evaluate_netstim_interval(interval):
    """Read raw interval sources without retaining canonical inference caches."""
    # The adapter is shared with connection source evaluation. Import lazily:
    # Network owns NetStim, and its module imports this materializer in turn.
    from .net import _connection_source_can_compute_uncached

    if _connection_source_can_compute_uncached(interval):
        return interval._compute()
    return interval()


def materialize_netstim_parameters(
    interval, start, noise, tau, shape, *, device, dtype
):
    """Materialize the four effective fields consumed by renewal kernels."""
    return {
        name: broadcast_netstim_parameter(
            value, shape, device=device, dtype=dtype, name=name
        )
        for name, value in (
            ("interval", interval),
            ("start", start),
            ("noise", noise),
            ("tau", tau),
        )
    }
