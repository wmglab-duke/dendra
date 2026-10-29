"""Population adapter over the shared tensor-tree inference while engine.

Bound-input assembly and accepted-step waveform clocks remain Population
policy. Pure callback updates share the model transition; the engine owns
packed recurrent carry and invocation-local emission buffers. Captured kernels
retain schemas and graphs rather than invocation values or preparation graphs.
"""

from __future__ import annotations

from functools import wraps

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from dendra.helpers import MEMBRANE_ASSEMBLY

from ._callback_recurrence import _transition_sample_values
from ._callbacks import _update_callback_values
from ._draws import _has_runtime_stimulation
from ._inputs import _RuntimeInputs
from ._tensor_tree_while import _CompiledTensorTreeWhile, _schema
from ._while_inductor import _compile_while


class _PopulationWhileExecutor:
    """Capture one transition with explicit caller-owned waveform timing."""

    def __init__(
        self,
        functional,
        *,
        steps,
        plans,
        compile_options,
        stepwise_stimulation=True,
    ):
        self._functional = functional
        self._steps = steps
        self._plans = plans
        self._options = compile_options
        self._stepwise_stimulation = stepwise_stimulation
        self._kernels = {}
        self._capture_errors = {}

    def kernel_for(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        carries,
        noise=None,
        stimulation=None,
        draws=None,
    ):
        from ._population import _STRUCTURED_ROLLOUT_CHUNK

        if self._steps < _STRUCTURED_ROLLOUT_CHUNK:
            return None, "The chunk is shorter than one inference loop block."
        # Captured-tensor mutation is part of the project's Torch>=2.14
        # baseline. Compile the selected route rather than choosing behavior
        # from a patch-version label.
        signature = (
            _schema(
                (
                    parameters,
                    prepared,
                    state,
                    ve,
                    intra,
                    carries,
                    noise,
                    stimulation,
                    draws,
                )
            ),
            MEMBRANE_ASSEMBLY.value,
        )
        if signature in self._capture_errors:
            return None, self._capture_errors[signature]
        if signature not in self._kernels:
            try:
                # Example operands must remain ordinary tensors even when the
                # invocation itself uses inference_mode. They are never kept.
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
                        draws,
                    )
            except Exception as error:
                reason = (
                    "No captured callback inference step graph is available: "
                    f"{type(error).__name__}: {error}"
                )
                self._capture_errors[signature] = reason
                return None, reason
        return self._kernels[signature], None

    def _capture(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        carries,
        noise,
        stimulation,
        draws=None,
    ):
        functional, plans = self._functional, self._plans
        stepwise = self._stepwise_stimulation or _has_runtime_stimulation(
            stimulation, draws
        )
        assembly = None
        if not stepwise and (functional.intra.enabled or functional.extra.enabled):

            def assemble(parameters, prepared, state, ve, intra):
                return functional._assemble_bound_inputs(
                    parameters,
                    state,
                    ve,
                    intra,
                    self._steps,
                    prepared["integrator"]["dt"],
                )

            examples = pytree.tree_map_only(
                torch.Tensor,
                lambda value: value.detach(),
                (parameters, prepared, functional._clone_state_tree(state), ve, intra),
            )
            assembly = make_fx(assemble, pre_dispatch=False)(*examples)
            ve, intra = assembly(parameters, prepared, state, ve, intra)

        def body(parameters, prepared, current, current_carries, sample):
            next_state, auxiliary = _transition_sample_values(
                functional,
                parameters,
                prepared,
                current,
                _RuntimeInputs(*sample),
                stepwise_stimulation=stepwise,
            )
            if plans:
                updated, emitted = _update_callback_values(
                    plans, current_carries, next_state, auxiliary
                )
            else:
                updated, emitted = current_carries, {}
            return (next_state, updated), emitted or {}

        engine = _CompiledTensorTreeWhile(
            body,
            steps=self._steps,
            compile_options=self._options,
            device=functional.device,
            plans=plans,
            clone_state=functional._clone_state_tree,
            tracer=make_fx,
            compiler=_compile_while,
        )
        kernel = engine.kernel_for(
            parameters, prepared, state, carries, (ve, intra, noise, stimulation, draws)
        )

        @wraps(kernel)
        def materialize(
            parameters,
            prepared,
            state,
            ve,
            intra,
            carries,
            noise=None,
            stimulation=None,
            draws=None,
        ):
            if assembly is not None:
                ve, intra = assembly(parameters, prepared, state, ve, intra)
            final, final_carries, observed = kernel(
                parameters,
                prepared,
                state,
                carries,
                (ve, intra, noise, stimulation, draws),
            )
            emitted = {plan.name: observed.get(plan.name) for plan in plans}
            return final, {"v": final["integrator"]["v"]}, final_carries, emitted

        return materialize


__all__ = []
