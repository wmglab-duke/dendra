"""Opt-in, fixed-horizon compiled scan kernels.

The authored tensor step is captured independently of the horizon. Live model,
preparation, input and callback carry tensors remain explicit graph operands.
This module is imported only when a caller requests scan execution.
"""

from __future__ import annotations

from functools import wraps

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from ..helpers import MEMBRANE_ASSEMBLY
from ._callback_carry import _CallbackCarryPartition
from ._callbacks import _update_callback_values
from ._hop_inputs import (
    _input_usage,
    _isolate_dense_lifted_inputs,
    _LiftedViewIsolation,
)
from ._scan_compat import install_scan_compatibility
from ._scan_inductor import _compile_scan
from ._scan_tree import _capture_tensor_step, _CompiledTensorTreeScan
from ._types import StepInput

# Bound the inference graph independently of the horizon, while amortizing the
# scan loop boundary over several accepted timesteps. Keep training's audited
# one-step partition/save policy. This is private so experiments can compare
# the small supported widths without adding an execution-policy public API.
_FORWARD_SCAN_BLOCK = 4


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.clone(memory_format=torch.contiguous_format),
        tree,
    )


def _detach(tree):
    return pytree.tree_map_only(torch.Tensor, lambda value: value.detach(), tree)


def _stack_steps(steps):
    return pytree.tree_map(lambda *values: torch.stack(values, dim=0), *steps)


class _CompiledScanPopulationChunk:
    """Lazy scan capture and compilation for an immutable execution plan."""

    def __init__(
        self,
        functional,
        *,
        steps,
        compile_options,
        stepwise_stimulation=False,
        plans=None,
    ):
        install_scan_compatibility()
        self._functional = functional
        self._steps = steps
        self._options = compile_options
        self._stepwise = stepwise_stimulation or plans is not None
        self._plans = plans
        # Only static graphs/callables belong here, never example operands.
        self._kernels = {}
        self._training = None
        self._assemblies = {}

    def _training_values(
        self, parameters, prepared, state, ve, intra, carries, noise, stimulation
    ):
        functional = self._functional
        plans = self._plans
        # Population parameters are named operands. Stable key order keeps
        # the shared engine's PyTree schema independent of caller dict order.
        parameters = {name: parameters[name] for name in sorted(parameters)}
        stepwise = self._stepwise or stimulation is not None
        if not stepwise and (functional.intra.enabled or functional.extra.enabled):
            signature = (ve is not None, intra is not None, MEMBRANE_ASSEMBLY.value)
            if signature not in self._assemblies:
                steps = self._steps

                def assemble(parameters, prepared, state, ve, intra):
                    return functional._assemble_bound_inputs(
                        parameters,
                        state,
                        ve,
                        intra,
                        steps,
                        prepared["integrator"]["dt"],
                    )

                with torch.inference_mode(False), torch.no_grad():
                    examples = _detach(
                        (
                            parameters,
                            prepared,
                            functional._clone_state_tree(state),
                            ve,
                            intra,
                        )
                    )
                    self._assemblies[signature] = make_fx(assemble, pre_dispatch=True)(
                        *examples
                    )
            ve, intra = self._assemblies[signature](
                parameters, prepared, state, ve, intra
            )

        if self._training is None:
            stepwise_stimulation = self._stepwise

            def body(parameters, prepared, state, carries, sample):
                if stepwise_stimulation or sample["stimulation"] is not None:
                    next_state, auxiliary = functional._step_values(
                        parameters,
                        prepared,
                        state,
                        StepInput(
                            sample["ve"],
                            sample["intra"],
                            sample["noise"],
                            sample["stimulation"],
                        ),
                    )
                else:
                    next_state, auxiliary = functional._transition_rollout_values(
                        parameters,
                        prepared,
                        state,
                        None if sample["ve"] is None else sample["ve"].unsqueeze(0),
                        None
                        if sample["intra"] is None
                        else sample["intra"].unsqueeze(0),
                        1,
                        None
                        if sample["noise"] is None
                        else pytree.tree_map(
                            lambda value: value.unsqueeze(0), sample["noise"]
                        ),
                    )
                emitted = None
                if plans is not None:
                    carries, emitted = _update_callback_values(
                        plans, carries, next_state, auxiliary
                    )
                return (next_state, carries), emitted or {}

            self._training = _CompiledTensorTreeScan(
                body,
                steps=self._steps,
                compile_options=self._options,
                device=functional.device,
                plans=plans,
                static_state_paths=(("control", "duration_remainder"),),
                clone_state=functional._clone_state_tree,
                capture_examples=_detach,
            )
        final, updated, emitted = self._training(
            parameters,
            prepared,
            state,
            carries,
            {"ve": ve, "intra": intra, "noise": noise, "stimulation": stimulation},
        )
        auxiliary = {"v": final["integrator"]["v"]}
        if plans is None:
            return final, auxiliary
        return (
            final,
            auxiliary,
            updated,
            {plan.name: emitted.get(plan.name) for plan in plans},
        )

    def _capture(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        carries,
        noise,
        stimulation=None,
        *,
        block=1,
    ):
        """Capture the specialized inference scan and its blocked emissions."""
        from torch._higher_order_ops.scan import scan

        functional = self._functional
        plans = self._plans
        steps = self._steps
        stepwise = self._stepwise or stimulation is not None
        phase_emissions = (
            block > 1 and plans is not None and functional.device.type == "cpu"
        )
        state = functional._clone_state_tree(state)
        parameters, prepared, state, ve, intra, carries, noise, stimulation = _detach(
            (parameters, prepared, state, ve, intra, carries, noise, stimulation)
        )
        callback_carry = _CallbackCarryPartition.from_plans(plans)
        carries, invariant_carries = callback_carry.split(carries)

        # Atomic rollout samples a bound waveform on start + arange * dt;
        # host runners sample it at each accepted, repeatedly added clock.
        # Preserve both established contracts instead of changing pulse edges.
        assembly = None
        if not stepwise and (functional.intra.enabled or functional.extra.enabled):

            def assemble(parameters, prepared, state, ve, intra):
                return functional._assemble_bound_inputs(
                    parameters,
                    state,
                    ve,
                    intra,
                    steps,
                    prepared["integrator"]["dt"],
                )

            assembly = make_fx(assemble, pre_dispatch=False)(
                parameters, prepared, state, ve, intra
            )
            ve, intra = assembly(parameters, prepared, state, ve, intra)

        sample = {}
        if ve is not None:
            sample["ve"] = ve[0]
        if intra is not None:
            sample["intra"] = intra[0]
        if noise is not None:
            sample["noise"] = pytree.tree_map(lambda value: value[0], noise)
        if stimulation is not None:
            sample["stimulation"] = pytree.tree_map(lambda value: value[0], stimulation)
        if not sample:
            # Scan requires a leading iteration axis even without input drives.
            sample["index"] = torch.zeros(
                (), dtype=torch.int64, device=functional.device
            )

        def authored(parameters, prepared, state, carries, invariant_carries, sample):
            if stepwise:
                next_state, auxiliary = functional._step_values(
                    parameters,
                    prepared,
                    state,
                    StepInput(
                        sample.get("ve"),
                        sample.get("intra"),
                        sample.get("noise"),
                        sample.get("stimulation"),
                    ),
                )
            else:
                next_state, auxiliary = functional._transition_rollout_values(
                    parameters,
                    prepared,
                    state,
                    None if "ve" not in sample else sample["ve"].unsqueeze(0),
                    None if "intra" not in sample else sample["intra"].unsqueeze(0),
                    1,
                    None
                    if "noise" not in sample
                    else pytree.tree_map(
                        lambda value: value.unsqueeze(0), sample["noise"]
                    ),
                )
            emitted = None
            if plans is not None:
                updated, emitted = _update_callback_values(
                    plans,
                    callback_carry.merge(carries, invariant_carries),
                    next_state,
                    auxiliary,
                )
                carries, _ = callback_carry.split(updated)
            if block == 1:
                # Keep the established single-step body/partition boundaries.
                # Unchanged clocks, reducers and Recorder views must not alias
                # the incoming carry or another output of the scan body.
                return _clone((next_state, carries)), _clone(emitted or {})
            # Intermediate inference steps are ordinary pure transitions;
            # scan's non-aliasing contract applies at the block boundary.
            return (next_state, carries), emitted or {}

        # Inference does not differentiate this captured graph. Training's
        # authored AD boundaries belong to the shared tensor-tree engine.
        graph = _capture_tensor_step(
            authored,
            (parameters, prepared, state, carries, invariant_carries, sample),
            grad_enabled=False,
            tracer=make_fx,
        )
        parameter_use, prepared_use, _, _, invariant_use, sample_use = _input_usage(
            graph, (parameters, prepared, state, carries, invariant_carries, sample)
        )
        sample_use = pytree.tree_map(lambda _value: True, sample_use)
        isolate_views = _LiftedViewIsolation.from_usage(
            (
                parameter_use,
                prepared_use,
                invariant_use,
                sample_use.get("ve", False),
                sample_use.get("intra", False),
                sample_use.get("noise", False),
                sample_use.get("stimulation", False),
            )
        )

        def execute(
            parameters, prepared, state, ve, intra, carries, noise, stimulation=None
        ):
            carries, invariant_carries = callback_carry.split(carries)
            parameters, prepared, invariant_carries, ve, intra, noise, stimulation = (
                _isolate_dense_lifted_inputs(
                    (
                        parameters,
                        prepared,
                        invariant_carries,
                        ve,
                        intra,
                        noise,
                        stimulation,
                    )
                )
            )
            samples = {}
            if ve is not None:
                samples["ve"] = ve
            if intra is not None:
                samples["intra"] = intra
            if noise is not None:
                samples["noise"] = noise
            if stimulation is not None:
                samples["stimulation"] = stimulation
            if not samples:
                samples["index"] = torch.arange(
                    steps, dtype=torch.int64, device=functional.device
                )

            if block == 1:

                def combine(carry, sample):
                    state, carries = carry
                    return graph(
                        parameters, prepared, state, carries, invariant_carries, sample
                    )

                (final, updated), emitted = scan(
                    combine,
                    (functional._clone_state_tree(state), _clone(carries)),
                    samples,
                )
            else:
                full_steps = (steps // block) * block
                carry = (functional._clone_state_tree(state), _clone(carries))

                def combine(carry, samples):
                    emissions = []
                    for offset in range(block):
                        sample = pytree.tree_map(lambda value: value[offset], samples)
                        carry, emission = graph(
                            parameters, prepared, *carry, invariant_carries, sample
                        )
                        emissions.append(emission)
                    # Each stack owns its storage, including when callbacks
                    # emit state views or the same Tensor in several leaves.
                    # Carry clones also separate unchanged reducers/clocks.
                    if phase_emissions:
                        # The extra singleton step axis separates recording
                        # output metadata from identical final-carry values.
                        # Copy each accepted frame, not a four-frame block.
                        phases = tuple(
                            pytree.tree_map(
                                lambda value: value.unsqueeze(0).clone(
                                    memory_format=torch.contiguous_format
                                ),
                                emission,
                            )
                            for emission in emissions
                        )
                        return _clone(carry), phases
                    return _clone(carry), _stack_steps(emissions)

                blocked_samples = pytree.tree_map(
                    lambda value: value[:full_steps].reshape(
                        full_steps // block, block, *value.shape[1:]
                    ),
                    samples,
                )
                carry, emitted = scan(combine, carry, blocked_samples)
                if not phase_emissions:
                    emitted = pytree.tree_map(
                        lambda value: value.flatten(0, 1), emitted
                    )

                # At most block-1 authored steps. Short horizons reduce the
                # effective block width before capture, retaining a real scan.
                tail = []
                for index in range(full_steps, steps):
                    sample = pytree.tree_map(lambda value: value[index], samples)
                    carry, emission = graph(
                        parameters, prepared, *carry, invariant_carries, sample
                    )
                    tail.append(emission)
                if tail:
                    tail = _stack_steps(tail)
                    if not phase_emissions:
                        emitted = pytree.tree_map(
                            lambda prefix, suffix: torch.cat((prefix, suffix), dim=0),
                            emitted,
                            tail,
                        )
                    carry = _clone(carry)
                elif phase_emissions:
                    tail = None
                final, updated = carry
            # Scan's AD wrapper marks every floating output differentiable.
            # Timesteps never change host scheduling carry, which must remain
            # gradient-free for duration validation and exact resumed runs.
            final["control"]["duration_remainder"] = state["control"][
                "duration_remainder"
            ].clone()
            auxiliary = {"v": final["integrator"]["v"]}
            if phase_emissions:
                return final, updated, emitted, tail
            if plans is None:
                return final, auxiliary
            return (
                final,
                auxiliary,
                updated,
                {plan.name: emitted.get(plan.name) for plan in plans},
            )

        compiled = _compile_scan(
            execute,
            self._options,
            prefer_cpp_wrapper=functional.device.type == "cpu",
        )

        @wraps(compiled)
        def materialize(
            parameters, prepared, state, ve, intra, carries, noise, stimulation=None
        ):
            if assembly is not None:
                ve, intra = assembly(parameters, prepared, state, ve, intra)
            recurrent, invariant = callback_carry.split(carries)
            parameters, prepared, invariant, ve, intra, noise, stimulation = (
                isolate_views(
                    (parameters, prepared, invariant, ve, intra, noise, stimulation)
                )
            )
            result = compiled(
                parameters,
                prepared,
                state,
                ve,
                intra,
                callback_carry.merge(recurrent, invariant),
                noise,
                stimulation,
            )
            if not phase_emissions and not callback_carry.invariant_names:
                return result
            if phase_emissions:
                final, updated, phases, tail = result
                if callback_carry.invariant_names:
                    updated = _clone(
                        callback_carry.merge(updated, callback_carry.split(carries)[1])
                    )
                # Each phase history is (blocks, 1, *sample_shape). Interleave
                # once outside the hot compiled loop, in accepted-clock order.
                emitted = pytree.tree_map(
                    lambda *values: torch.cat(values, dim=1).flatten(0, 1), *phases
                )
                if tail is not None:
                    emitted = pytree.tree_map(
                        lambda prefix, suffix: torch.cat((prefix, suffix), dim=0),
                        emitted,
                        tail,
                    )
                return (
                    final,
                    {"v": final["integrator"]["v"]},
                    updated,
                    {plan.name: emitted.get(plan.name) for plan in plans},
                )
            final, auxiliary, updated, emitted = result
            updated = _clone(
                callback_carry.merge(updated, callback_carry.split(carries)[1])
            )
            return final, auxiliary, updated, emitted

        return materialize

    def __call__(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        carries=None,
        noise=None,
        stimulation=None,
    ):
        carries = {} if carries is None else carries
        if torch.is_grad_enabled():
            result = self._training_values(
                parameters, prepared, state, ve, intra, carries, noise, stimulation
            )
        else:
            # Preserve the measured CPU inference block/phase-recording path.
            block = _FORWARD_SCAN_BLOCK if self._functional.device.type == "cpu" else 1
            if type(block) is not int or block not in (1, 2, 4):
                raise ValueError("the private forward scan block must be 1, 2, or 4")
            block = min(block, self._steps)
            signature = (
                ve is not None,
                intra is not None,
                noise is not None,
                stimulation is not None,
                MEMBRANE_ASSEMBLY.value,
                block,
            )
            if signature not in self._kernels:
                with torch.inference_mode(False), torch.no_grad():
                    self._kernels[signature] = self._capture(
                        parameters,
                        prepared,
                        state,
                        ve,
                        intra,
                        carries,
                        noise,
                        stimulation,
                        block=block,
                    )
            result = self._kernels[signature](
                parameters, prepared, state, ve, intra, carries, noise, stimulation
            )
        # Scan's AD wrapper marks floating carry outputs differentiable even
        # when their authored update is detached. Runtime NOISE retains its
        # native stop-gradient boundary when exposed as public carry. Poisson
        # event times and generator frames retain the same detached contract.
        for name in ("runtime_noise", "stimulation"):
            if name in result[0]:
                result[0][name] = _detach(result[0][name])
        return result
