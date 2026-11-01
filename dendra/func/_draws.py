"""Canonical explicit runtime draws and their physical RNG timelines."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.utils import _pytree as pytree


class RuntimeDraws(NamedTuple):
    """Named runtime draw values and authoritative RNG publication timelines.

    Obtain a frame with ``dn.func.sample_runtime_draws(model, dt=dt)`` or a
    time-leading tape with ``dn.func.sample_runtime_draws(model, dt=dt, steps=N)``.
    Native Networks use their initialized timestep. Lowered plans, bindings,
    compiled chunks and runners require an explicit state or ``Continuation``
    as the sampler's second argument. Sampling an initialized native model
    does not require functional lowering or initialize a fresh model.
    Pass it through ``draws=`` in functional execution or native Population/
    Network execution. ``step`` and its public helpers consume one frame;
    ``run``, ``longrun`` and ``longrun_checkpointed`` consume a time-leading
    tape matching their accepted timestep count. These packets cover
    NOISE/State SDE, optional Network NetStim and randomized Poisson waveform
    injections with finite n and fixed invocation counts, including those
    owned by mechanisms and solver-owned intra/call-local extra banks.
    A Population with replayed Poisson banks evaluates its solver-level banks
    once per accepted step, independently of native execution chunk size.
    Populations without these banks retain native whole-run/per-chunk waveform
    evaluation. Omitted draws retain automatic bank preparation and sampling.
    Mechanism injections consume their authored invocation recipe in the
    numerical phases that evaluate them, including repeated and unused calls.
    Network execution retains global invocation order and shared waveform/
    generator ownership.
    Numerical draw families and stream routes follow the model's declarations.
    Shared RNG bytes occur once at each native publication boundary, including
    intermediate Population boundaries when a Network exposes them.

    Base draws and RNG carry are detached from their samplers. Parameters of
    the model and its distributions retain their ordinary gradient paths.
    Existing initialization draw packets and runtime input tuples keep their
    separate public formats.

    The shared sampler allocates Poisson invocation rows for runtime calls
    only, independently of fresh-initialization availability. Existing plan
    sampling methods retain their formats, including reserved initializer rows;
    runtime execution accepts those compatible layouts as well.

    Native checkpointed training requires versioned tensor leaves. Clone
    inference tensors outside ``torch.inference_mode()`` before passing them
    to that execution path; ordinary replay and inference retain their support.

    Sampling leaves source generators and incoming state unchanged. Native
    execution publishes the supplied RNG timeline at its existing boundaries;
    later steps with omitted draws continue from that published position. Keep
    tapes unchanged until checkpoint backward finishes; recomputation reuses
    their original frames and rejects changed tensor leaves.
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
