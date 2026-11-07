"""Tensor recurrence shared by atomic and host Population callbacks."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

from ._draws import _has_runtime_stimulation
from ._inputs import _index_runtime_inputs, _RuntimeInputs, _slice_runtime_inputs


def _transition_sample_values(
    functional, parameters, prepared, state, sample, *, stepwise_stimulation=False
):
    """Consume one frame after its caller has selected the waveform clock.

    Atomic deterministic tapes have already been assembled for the horizon.
    Host steps and explicit Poisson requests assemble at the accepted clock.
    """
    if stepwise_stimulation or _has_runtime_stimulation(
        sample.stimulation, sample.draws
    ):
        return functional._step_values(parameters, prepared, state, sample)
    return functional._transition_rollout_values(
        parameters,
        prepared,
        state,
        None if sample.ve is None else sample.ve.unsqueeze(0),
        None if sample.intra is None else sample.intra.unsqueeze(0),
        1,
        None
        if sample.noise is None
        else pytree.tree_map_only(
            torch.Tensor, lambda value: value.unsqueeze(0), sample.noise
        ),
        None,
        None
        if sample.draws is None
        else pytree.tree_map_only(
            torch.Tensor, lambda value: value.unsqueeze(0), sample.draws
        ),
    )


def _execute_callback_steps(
    functional,
    parameters,
    prepared,
    state,
    ve,
    intra,
    steps,
    carries,
    update,
    stack,
    noise=None,
    stimulation=None,
    draws=None,
    *,
    stepwise_stimulation=False,
    sampling_times=None,
):
    """Execute a tensor recurrence with caller-owned callback carry wrappers."""
    stochastic_stimulation = _has_runtime_stimulation(stimulation, draws)
    if not stepwise_stimulation and not stochastic_stimulation:
        ve, intra = functional._assemble_bound_inputs(
            parameters,
            state,
            ve,
            intra,
            steps,
            prepared["integrator"]["dt"],
            sampling_times=sampling_times,
        )
    emissions = []
    auxiliary = {"v": state["integrator"]["v"]}
    if steps == 0:
        state, auxiliary = functional._transition_rollout_values(
            parameters, prepared, state, ve, intra, 0
        )
    for index in range(steps):
        if stepwise_stimulation:
            state, auxiliary = functional._step_values(
                parameters,
                prepared,
                state,
                _index_runtime_inputs(
                    ve, intra, noise, stimulation, index, draws, sampling_times
                ),
            )
        else:
            # Atomic callbacks consume tape slices, retaining their Tensor
            # dispatch and the legacy packet's tensor-only leaf selection.
            sample = _slice_runtime_inputs(
                _RuntimeInputs(ve, intra, noise, stimulation, draws, sampling_times),
                index,
                index + 1,
            )
            execute = (
                functional._execute_rollout_values
                if stochastic_stimulation
                else functional._transition_rollout_values
            )
            kwargs = (
                {"sampling_times": sample.sampling_time}
                if stochastic_stimulation
                else {}
            )
            state, auxiliary = execute(
                parameters,
                prepared,
                state,
                sample.ve,
                sample.intra,
                1,
                sample.noise,
                sample.stimulation,
                sample.draws,
                **kwargs,
            )
        carries, emitted = update(carries, state, auxiliary)
        if emitted is not None:
            emissions.append(emitted)
    return state, auxiliary, carries, stack(emissions)


__all__ = []
