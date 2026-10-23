"""Checkpoint metadata for explicitly carried connection sampling streams."""

from collections.abc import Mapping

import torch


def clone_connection_sampling_rng(values):
    """Validate and own generator checkpoints without touching global streams."""
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError("Connection sampling RNG must be a device-to-state mapping.")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str):
            raise TypeError("Connection sampling RNG device names must be strings.")
        device = torch.device(name)
        if (
            device.type not in {"cpu", "cuda"}
            or str(device) != name
            or (device.type == "cpu" and name != "cpu")
            or (device.type == "cuda" and device.index is None)
        ):
            raise ValueError(
                "Connection sampling RNG needs canonical CPU/CUDA devices."
            )
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.uint8
            or value.device.type != "cpu"
            or value.ndim != 1
        ):
            raise ValueError("Connection sampling RNG states must be CPU byte vectors.")
        # set_state validates the generator-specific payload as well as shape.
        # The temporary generator owns its state; global Torch RNG is unchanged.
        generator = torch.Generator(device=device)
        generator.set_state(value.detach().clone())
        result[name] = generator.get_state().clone()
    return result
