"""Shared ownership of raw sources and current simulation state."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

from dendra.utils.tensor_ops import _logical_tensor_bytes_equal

from ._network_bindings import AliasSnapshots, _map_containers

_PRESERVE_TRAINABILITY = object()


def _snapshot_frames(
    functional,
    tensors,
    *,
    trainable=_PRESERVE_TRAINABILITY,
    snapshots=None,
    defer_state=False,
):
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
            snapshots = AliasSnapshots() if snapshots is None else snapshots
            return snapshots.tree(None, (parameters, constants, tensors.state))
        state = (
            _share_restart_state(
                tensors.state, functional._initialization_state_template
            )
            if defer_state
            else _copy_runner_state(tensors.state)
        )
        return parameters, constants, state


def _copy_runner_state(state):
    with torch.inference_mode(False), torch.no_grad():
        return pytree.tree_map_only(
            torch.Tensor, lambda value: value.detach().clone(), state
        )


def _share_restart_state(state, initialization):
    """Back an unexposed restart with byte-identical private state leaves."""
    initial_leaves = dict(pytree.tree_leaves_with_path(initialization))

    def shared_leaf(path, value):
        initial = initial_leaves.get(path)
        if (
            type(initial) is torch.Tensor
            and type(value) is torch.Tensor
            and initial.layout == value.layout == torch.strided
            and not value.is_quantized
            and initial.shape == value.shape
            and initial.stride() == value.stride()
            and initial.dtype == value.dtype
            and initial.device == value.device
            and _logical_tensor_bytes_equal(value, initial)
        ):
            return initial
        # Extraction already owns the setup-time value of this leaf.
        return value

    return pytree.tree_map_with_path(shared_leaf, state)


__all__ = []
