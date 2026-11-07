"""Owned Population cache values shared by native and pure initialization."""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch


class _RestoreParameter(torch.autograd.Function):
    """A state-dict copy keeps the destination Parameter's gradient identity."""

    generate_vmap_rule = True

    @staticmethod
    def forward(current, cached):
        return cached.detach().to(current).clone(memory_format=torch.preserve_format)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None

    @staticmethod
    def jvp(ctx, current_tangent, cached_tangent):
        return current_tangent.clone()


@torch.compiler.allow_in_graph
def _restore_parameter(current, cached):
    # The audited operation owns its output and mutates no input. Keep its
    # custom JVP visible to AOTAutograd rather than asking Dynamo to inline an
    # autograd.Function with a custom forward-mode rule.
    return _RestoreParameter.apply(current, cached)


def restore_population_cache_values(current, cached, *, parameter_names=()):
    """Restore detached cache values without mutating either mapping.

    Native ``load_state_dict`` copies into existing Parameters under no-grad;
    their subsequent gradients still belong to those destination Parameters.
    The pure adapter represents that identity explicitly, including when an
    incoming parameter is itself computed from other differentiable inputs.
    Buffers and cached inputs never retain the warm-up graph.
    """
    parameter_names = frozenset(parameter_names)
    result = OrderedDict()
    for name, value in cached.items():
        reference = current.get(name)
        if torch.is_tensor(value):
            if torch.is_tensor(reference):
                restored = (
                    _restore_parameter(reference, value)
                    if name in parameter_names
                    else value.detach()
                    .to(reference)
                    .clone(memory_format=torch.preserve_format)
                )
            else:
                restored = value.detach().clone(memory_format=torch.preserve_format)
        else:
            restored = copy.deepcopy(value)
        result[name] = restored
    if hasattr(cached, "_metadata"):
        result._metadata = copy.deepcopy(cached._metadata)
    return result


def replay_population_cache_storage(
    writes, write_indices, output_indices, output_shapes, *, size
):
    """Replay native cache copies across partial views of one storage frame.

    Index metadata is fixed during lowering. Non-accumulating copies preserve
    native last-write ordering and avoid arithmetic on cached values.
    """
    storage = writes[0].new_zeros((size,))
    for value, indices in zip(writes, write_indices, strict=True):
        storage = storage.index_copy(
            0, indices, value.detach().reshape(indices.numel())
        )
    return tuple(
        storage.index_select(0, indices).reshape(shape)
        for indices, shape in zip(output_indices, output_shapes, strict=True)
    )


__all__ = ["restore_population_cache_values", "replay_population_cache_storage"]
