"""Shared ownership of raw sources and current simulation state."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

from ._network_bindings import AliasSnapshots, _map_containers

_PRESERVE_TRAINABILITY = object()


def _snapshot_frames(functional, tensors, *, trainable=_PRESERVE_TRAINABILITY):
    """Own continuation frames or preserve the established runner policy.

    Capture retains current trainability and Tensor identity ties across all
    three frames. Runners explicitly select optimizer leaves and retain their
    existing model-specific constant bindings and independent state copies.
    """
    with torch.inference_mode(False), torch.no_grad():
        from ._network import FunctionalNetwork

        parameters = (
            tensors.parameters
            if trainable is _PRESERVE_TRAINABILITY
            else tensors.independent_parameters(trainable=trainable)
        )
        constants = (
            functional._bind_parameter_constants(
                parameters, _map_containers(tensors.constants, dict)
            )
            if isinstance(functional, FunctionalNetwork)
            else dict(tensors.constants)
        )
        if trainable is _PRESERVE_TRAINABILITY:
            return AliasSnapshots().tree(None, (parameters, constants, tensors.state))
        state = pytree.tree_map_only(
            torch.Tensor, lambda value: value.detach().clone(), tensors.state
        )
        return parameters, constants, state


__all__ = []
