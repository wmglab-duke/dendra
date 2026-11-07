"""Explicit standard-normal inputs shared by generated State SDE solvers."""

from collections.abc import Mapping

import torch

from dendra.models.rng import RNGModule


def sample_state_sde_draw(rng, shape, *, device, dtype):
    """Sample a detached standard normal without using the distribution registry."""
    if isinstance(rng, RNGModule):
        rng.init(device)
    with torch.no_grad():
        return rng.randn(tuple(shape), device=device, dtype=dtype).detach()


def state_sde_dt(dt, like):
    """Match the State dtype while keeping scalar timestep arithmetic field-shaped."""
    value = torch.as_tensor(dt, dtype=like.dtype, device=like.device)
    # A logical scalar with an empty vmap batch can make PyTorch select lane
    # zero when combined with a field. Singleton visible axes preserve scalar
    # broadcasting and gradients without indexing the hidden batch dimension.
    return value.reshape((1,) * like.ndim) if value.ndim == 0 else value


def state_sde_draw(draws, name, like):
    """Return a detached draw with the exact current State field layout."""
    value = draws[name]
    if not torch.is_tensor(value):
        raise TypeError(f"State SDE draw {name!r} must be a Tensor")
    if value.shape != like.shape:
        raise ValueError(
            f"State SDE draw {name!r} has shape {tuple(value.shape)}; "
            f"expected {tuple(like.shape)}"
        )
    if value.dtype != like.dtype or value.device != like.device:
        raise ValueError(
            f"State SDE draw {name!r} must have dtype {like.dtype} "
            f"and device {like.device}"
        )
    return value.detach()


def validate_state_sde_draws(draws, names, values):
    """Validate the complete ordered solver draw frame without consuming RNG."""
    if not isinstance(draws, Mapping):
        raise TypeError("State SDE draws must be a mapping keyed by state name")
    if set(draws) != set(names):
        raise ValueError(
            f"State SDE draw names must match {names}; received {tuple(draws)}"
        )
    for name in names:
        state_sde_draw(draws, name, values[name])
