"""Opt-in, fixed-horizon compiled scan kernels.

The authored tensor step is captured independently of the horizon. Live model,
preparation, input and callback carry tensors remain explicit graph operands.
This module is imported only when a caller requests scan execution.
"""

from __future__ import annotations

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from ..helpers import MEMBRANE_ASSEMBLY
from ._callbacks import _update_callback_values
from ._scan_autograd import guard_scan_outputs
from ._scan_compat import install_scan_compatibility
from ._scan_inductor import _compile_scan
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

    def _capture(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        carries,
        *,
        block=1,
        grad_enabled=True,
    ):
        from torch._higher_order_ops.scan import scan

        functional = self._functional
        plans = self._plans
        steps = self._steps
        stepwise = self._stepwise
        state = functional._clone_state_tree(state)
        parameters, prepared, state, ve, intra, carries = _detach(
            (parameters, prepared, state, ve, intra, carries)
        )

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

            assembly = make_fx(assemble)(parameters, prepared, state, ve, intra)
            ve, intra = assembly(parameters, prepared, state, ve, intra)

        sample = {}
        if ve is not None:
            sample["ve"] = ve[0]
        if intra is not None:
            sample["intra"] = intra[0]
        if not sample:
            # Scan requires a leading iteration axis even without input drives.
            sample["index"] = torch.zeros(
                (), dtype=torch.int64, device=functional.device
            )

        def authored(parameters, prepared, state, carries, sample):
            if stepwise:
                next_state, auxiliary = functional._step_values(
                    parameters,
                    prepared,
                    state,
                    StepInput(sample.get("ve"), sample.get("intra")),
                )
            else:
                next_state, auxiliary = functional._transition_rollout_values(
                    parameters,
                    prepared,
                    state,
                    None if "ve" not in sample else sample["ve"].unsqueeze(0),
                    None if "intra" not in sample else sample["intra"].unsqueeze(0),
                    1,
                )
            emitted = None
            if plans is not None:
                carries, emitted = _update_callback_values(
                    plans, carries, next_state, auxiliary
                )
            if block == 1:
                # Keep training's established body and partition boundaries.
                # Unchanged clocks, reducers and Recorder views must not alias
                # the incoming carry or another output of the scan body.
                return _clone((next_state, carries)), _clone(emitted or {})
            # Intermediate steps within a block are ordinary pure transitions;
            # scan's non-aliasing contract applies at the block boundary.
            return (next_state, carries), emitted or {}

        graph = make_fx(authored)(parameters, prepared, state, carries, sample)

        def execute(parameters, prepared, state, ve, intra, carries):
            if assembly is not None:
                ve, intra = assembly(parameters, prepared, state, ve, intra)
            samples = {}
            if ve is not None:
                samples["ve"] = ve
            if intra is not None:
                samples["intra"] = intra
            if not samples:
                samples["index"] = torch.arange(
                    steps, dtype=torch.int64, device=functional.device
                )

            if block == 1:

                def combine(carry, sample):
                    state, carries = carry
                    return graph(parameters, prepared, state, carries, sample)

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
                        carry, emission = graph(parameters, prepared, *carry, sample)
                        emissions.append(emission)
                    # Each stack owns its storage, including when callbacks
                    # emit state views or the same Tensor in several leaves.
                    # Carry clones also separate unchanged reducers/clocks.
                    return _clone(carry), _stack_steps(emissions)

                blocked_samples = pytree.tree_map(
                    lambda value: value[:full_steps].reshape(
                        full_steps // block, block, *value.shape[1:]
                    ),
                    samples,
                )
                carry, emitted = scan(combine, carry, blocked_samples)
                emitted = pytree.tree_map(lambda value: value.flatten(0, 1), emitted)

                # At most block-1 authored steps. Short horizons reduce the
                # effective block width before capture, retaining a real scan.
                tail = []
                for index in range(full_steps, steps):
                    sample = pytree.tree_map(lambda value: value[index], samples)
                    carry, emission = graph(parameters, prepared, *carry, sample)
                    tail.append(emission)
                if tail:
                    tail = _stack_steps(tail)
                    emitted = pytree.tree_map(
                        lambda prefix, suffix: torch.cat((prefix, suffix), dim=0),
                        emitted,
                        tail,
                    )
                    carry = _clone(carry)
                final, updated = carry
            # Scan's AD wrapper marks every floating output differentiable.
            # Timesteps never change host scheduling carry, which must remain
            # gradient-free for duration validation and exact resumed runs.
            final["control"]["duration_remainder"] = state["control"][
                "duration_remainder"
            ].clone()
            auxiliary = {"v": final["integrator"]["v"]}
            if plans is None:
                return final, auxiliary
            return (
                final,
                auxiliary,
                updated,
                {plan.name: emitted.get(plan.name) for plan in plans},
            )

        return _compile_scan(
            execute,
            self._options,
            prefer_cpp_wrapper=not grad_enabled and functional.device.type == "cpu",
        )

    def __call__(self, parameters, prepared, state, ve, intra, carries=None):
        grad_enabled = torch.is_grad_enabled()
        # Apply the measured block policy to CPU forward execution. Accelerator
        # scan and training retain their established one-step body.
        block = (
            _FORWARD_SCAN_BLOCK
            if not grad_enabled and self._functional.device.type == "cpu"
            else 1
        )
        if type(block) is not int or block not in (1, 2, 4):
            raise ValueError("the private forward scan block must be 1, 2, or 4")
        block = min(block, self._steps)
        signature = (
            ve is not None,
            intra is not None,
            MEMBRANE_ASSEMBLY.value,
            grad_enabled,
            block,
        )
        if signature not in self._kernels:
            # Capture a differentiable tensor program using disposable detached
            # examples, even when first called under inference_mode/no_grad.
            with torch.inference_mode(False), torch.no_grad():
                self._kernels[signature] = self._capture(
                    parameters,
                    prepared,
                    state,
                    ve,
                    intra,
                    {} if carries is None else carries,
                    block=block,
                    # Capture runs under no_grad even for a training kernel.
                    # Preserve the caller's mode when choosing its compiler.
                    grad_enabled=grad_enabled,
                )
        result = self._kernels[signature](
            parameters,
            prepared,
            state,
            ve,
            intra,
            {} if carries is None else carries,
        )
        return guard_scan_outputs(result)
