"""Canonical explicit runtime draws and their physical RNG timelines."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.utils import _pytree as pytree


class RuntimeDraws(NamedTuple):
    """Explicit runtime stochastic values and their resulting RNG states.

    Obtain one frame with :func:`sample_runtime_draws` without a horizon, or a
    time-leading tape with explicit ``steps=`` or ``duration=``. Native initialized
    models use their current boundary; functional targets also require state or a
    :class:`Continuation`. Sampling leaves the source boundary unchanged.

    Pass ``draws=packet`` to native or functional execution. ``step`` consumes one
    frame; run/longrun methods consume a tape matching their accepted timestep
    count. The same duration must be used for sampling and execution so fractional
    carry is updated once. Families cover declared NOISE/State SDE, NetStim renewal
    and supported randomized Poisson waveform evaluations. Shared RNG ordering is
    preserved across these families.

    Base draws are detached; model/distribution parameters retain their normal
    gradient paths. Execution publishes the supplied RNG timeline, allowing later
    automatic sampling to continue from that point. Keep packet leaves unchanged
    until activation-checkpoint backward completes. Native checkpointed training
    requires versioned tensors; clone externally created inference tensors outside
    inference mode before using them there.

    Use :class:`InitializationDraws` for initialization rather than runtime draws.
    The older component-specific sampling methods retain their packet layouts;
    the shared sampler supplies runtime-only Poisson invocation rows.
    """

    draws: dict[str, object]
    rng: dict[str, object]


def _normalize_runtime_draws(packet):
    """Materialize authored Mappings before source qualification or capture."""
    if not isinstance(packet, RuntimeDraws):
        raise TypeError("draws must be a RuntimeDraws packet or None")
    from ._network_bindings import _canonicalize_mapping_tree

    return _canonicalize_mapping_tree(packet)[0]


def _with_runtime_draws(inputs, draws, *, step=False):
    """Keep deterministic inputs and canonical draw authority separate."""
    from ._inputs import _population_inputs

    if draws is None:
        return _population_inputs(inputs, step=step)

    packet = _normalize_runtime_draws(draws)
    inputs = _population_inputs(inputs, step=step)
    if (
        inputs.noise is not None
        or inputs.stimulation is not None
        or inputs.draws is not None
    ):
        raise ValueError(
            "draws cannot be combined with legacy noise or stimulation packets"
        )
    return inputs._replace(draws=packet)


def _has_runtime_stimulation(stimulation=None, draws=None):
    """Retain the existing atomic-versus-accepted-step waveform clocks."""
    return stimulation is not None or (
        draws is not None and draws.draws.get("stimulation") is not None
    )


def _runtime_draws_binding(packet):
    """Capture replay freshness without copying or inspecting draw values."""
    from ._tensor_contracts import _has_inference_tensor_component, _tensor_tree_binding

    leaves, spec = pytree.tree_flatten(packet)
    ordinary, inference, schema = [], [], []
    for index, value in enumerate(leaves):
        if value is None:
            schema.append(None)
            continue
        if not torch.is_tensor(value):
            raise TypeError("runtime draw packets must contain Tensor leaves")
        schema.append((tuple(value.shape), value.dtype, value.device))
        if _has_inference_tensor_component(value):
            inference.append((index, id(value), value.requires_grad))
        else:
            ordinary.append(value)
    return spec, tuple(schema), _tensor_tree_binding(tuple(ordinary)), tuple(inference)


__all__ = ["RuntimeDraws"]
