"""Opt-in, fixed-horizon compiled scan kernels.

The authored tensor step is captured independently of the horizon. Live model,
preparation, input and callback carry tensors remain explicit graph operands.
This module is imported only when a caller requests scan execution.
"""

from __future__ import annotations

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from ._callbacks import _update_callback_values
from ._scan_autograd import guard_scan_outputs
from ._scan_compat import install_scan_compatibility
from ._types import StepInput


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.clone(memory_format=torch.contiguous_format),
        tree,
    )


def _detach(tree):
    return pytree.tree_map_only(torch.Tensor, lambda value: value.detach(), tree)


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

    def _capture(self, parameters, prepared, state, ve, intra, carries):
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
            # Unchanged clocks, reducers and Recorder views must not alias the
            # incoming carry or another output of the scan body.
            return _clone((next_state, carries)), _clone(emitted or {})

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

            def combine(carry, sample):
                state, carries = carry
                return graph(parameters, prepared, state, carries, sample)

            (final, updated), emitted = scan(
                combine,
                (functional._clone_state_tree(state), _clone(carries)),
                samples,
            )
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

        return torch.compile(execute, **self._options)

    def __call__(self, parameters, prepared, state, ve, intra, carries=None):
        signature = (ve is not None, intra is not None)
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
