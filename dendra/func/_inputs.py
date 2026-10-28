"""Shared deterministic inputs and invocation-local runtime frames."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.utils import _pytree as pytree

from ._draws import RuntimeDraws


class Inputs(NamedTuple):
    """Explicit extracellular voltage and intracellular current inputs.

    ``ve`` and ``intra`` contain deterministic Tensor values or ``None``. A
    step consumes one frame; a rollout, compiled chunk or host run consumes
    time-leading tapes. Their existing broadcasting rules apply. Tensor
    references, views and gradients remain caller-owned.

    Networks accept a mapping from Population names to these same packets.
    Supply stochastic values and their RNG timelines separately through
    ``draws=``. Existing ``StepInput``, ``RolloutInput`` and
    ``NetworkRolloutInput`` packets remain supported.
    """

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None


class _RuntimeInputs(NamedTuple):
    """Common step/tape operands, including isolated compatibility families."""

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None
    noise: object = None
    stimulation: object = None
    draws: RuntimeDraws | None = None


def _population_inputs(inputs, *, step=False):
    """Adapt public packets once without copying their Tensor leaves."""
    if isinstance(inputs, _RuntimeInputs):
        if isinstance(inputs.noise, RuntimeDraws):
            if inputs.draws is not None:
                raise ValueError(
                    "draws cannot be combined with legacy noise or stimulation packets"
                )
            return inputs._replace(noise=None, draws=inputs.noise)
        return inputs
    if inputs is None:
        return _RuntimeInputs()
    if isinstance(inputs, Inputs):
        return _RuntimeInputs(inputs.ve, inputs.intra)
    from ._types import RolloutInput, StepInput

    expected = StepInput if step else RolloutInput
    if not isinstance(inputs, expected):
        raise TypeError(f"inputs must be a {expected.__name__} or None")
    noise = inputs.noise
    if isinstance(noise, RuntimeDraws):
        return _RuntimeInputs(inputs.ve, inputs.intra, None, inputs.stimulation, noise)
    return _RuntimeInputs(inputs.ve, inputs.intra, noise, inputs.stimulation)


def _slice_runtime_inputs(inputs, start, stop):
    """Slice a whole runtime frame, retaining leaf views and optional values."""
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


def _index_runtime_inputs(ve, intra, noise, stimulation, index, draws=None):
    """Select one frame in deterministic, legacy, then canonical packet order."""
    return _RuntimeInputs(
        None if ve is None else ve[index],
        None if intra is None else intra[index],
        None if noise is None else pytree.tree_map(lambda value: value[index], noise),
        None
        if stimulation is None
        else pytree.tree_map(lambda value: value[index], stimulation),
        None
        if draws is None
        else pytree.tree_map_only(torch.Tensor, lambda value: value[index], draws),
    )


def _legacy_step_input(inputs):
    """Retain the authored callable contract at the legacy scheduler boundary."""
    from ._types import StepInput

    return StepInput(
        inputs.ve,
        inputs.intra,
        inputs.noise if inputs.draws is None else inputs.draws,
        inputs.stimulation,
    )


__all__ = ["Inputs"]
