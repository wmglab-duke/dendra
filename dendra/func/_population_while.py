"""Population adapter over the shared tensor-tree inference while engine.

Bound-input assembly and accepted-step waveform clocks remain Population
policy. Pure callback updates share the model transition; the engine owns
packed recurrent carry and invocation-local emission buffers. Captured kernels
retain schemas and graphs rather than invocation values or preparation graphs.
"""

from __future__ import annotations

from collections import namedtuple
from functools import wraps
from types import MethodType

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from dendra.helpers import MEMBRANE_ASSEMBLY

from . import _tensor_tree_while
from ._callback_recurrence import _transition_sample_values
from ._callbacks import _update_callback_values
from ._draws import RuntimeDraws, _has_runtime_stimulation
from ._inputs import _RuntimeInputs
from ._state_projection import _plain_state_tree
from ._tensor_tree_while import _CompiledTensorTreeWhile, _schema
from ._types import RuntimeNoiseDraw, StimulationDraw
from ._while_inductor import _compile_while


def _ordinary_tape_spec(spec):
    if spec.is_leaf():
        return True
    if spec.type is dict and not all(
        type(key) is str or type(key) is int for key in spec.context
    ):
        return False
    if not any(spec.type is kind for kind in (dict, tuple, list)) and not (
        spec.type is namedtuple
        and any(
            spec.context is kind
            for kind in (RuntimeDraws, RuntimeNoiseDraw, StimulationDraw)
        )
    ):
        return False
    return all(_ordinary_tape_spec(child) for child in spec.children())


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
        self._stepwise_engine = None
        self._kernels = {}
        self._capture_errors = {}

    def _success_signature(self, operands, signature):
        # The shared engine owns ordinary tape views before capture. Match that
        # admission contract without packing or copying a live frame here.
        # Errors retain their original raw signature independently of successes.
        capture = self._capture
        if (
            type(self) is not _PopulationWhileExecutor
            or type(capture) is not MethodType
            or capture.__func__ is not _STANDARD_CAPTURE
            or capture.__func__.__code__ is not _STANDARD_CAPTURE_CODE
            or capture.__func__.__defaults__ is not _STANDARD_CAPTURE_DEFAULTS
            or make_fx is not _STANDARD_TRACER
            or _compile_while is not _STANDARD_COMPILER
        ):
            return signature
        parameters, prepared, state, ve, intra, carries, noise, stimulation, draws = (
            operands
        )
        spec, metadata = signature[0]
        children = spec.children()
        tape_fields = (3, 4, 6, 7, 8)
        if not all(_ordinary_tape_spec(child) for child in children):
            return signature
        leaves = spec.flatten_up_to(operands)
        if any(
            value is not None
            and type(value) is not torch.Tensor
            and type(value) is not torch.nn.Parameter
            for value in leaves
        ):
            return signature
        tape_positions = []
        offset = 0
        for index, child in enumerate(children):
            stop = offset + child.num_leaves
            if index in tape_fields:
                tape_positions.extend(range(offset, stop))
            offset = stop
        samples = tuple(
            leaves[index] for index in tape_positions if leaves[index] is not None
        )
        if any(type(value) is not torch.Tensor for value in samples):
            return signature
        if not _tensor_tree_while._can_own_samples(samples) or not _plain_state_tree(
            (parameters, prepared, state, carries), allow_parameters=True
        ):
            return signature
        if not (
            self._stepwise_stimulation or _has_runtime_stimulation(stimulation, draws)
        ):
            return signature
        normalized = list(metadata)
        for position in tape_positions:
            if leaves[position] is not None:
                shape, dtype, device = metadata[position][:3]
                if not shape or shape[0] != self._steps:
                    return signature
                normalized[position] = ("owned_sample", shape, dtype, device)
        return (spec, tuple(normalized)), signature[1]

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
        operands = (
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
        signature = (_schema(operands), MEMBRANE_ASSEMBLY.value)
        if signature in self._capture_errors:
            return None, self._capture_errors[signature]
        success_signature = self._success_signature(operands, signature)
        if success_signature not in self._kernels:
            try:
                # Example operands must remain ordinary tensors even when the
                # invocation itself uses inference_mode. They are never kept.
                with torch.inference_mode(False), torch.no_grad():
                    self._kernels[success_signature] = self._capture(
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
        return self._kernels[success_signature], None

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

        # Stepwise assembly is part of the pure body, so one engine can select
        # its owned-input specializations across public state/view changes.
        # Whole-tape assembly retains its existing per-signature capture.
        engine = self._stepwise_engine if stepwise else None
        if engine is None:

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
                project_state=getattr(functional, "_project_state_tree", None),
                tracer=make_fx,
                compiler=_compile_while,
            )
            if stepwise:
                self._stepwise_engine = engine
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

_STANDARD_CAPTURE = _PopulationWhileExecutor._capture
_STANDARD_CAPTURE_CODE = _STANDARD_CAPTURE.__code__
_STANDARD_CAPTURE_DEFAULTS = _STANDARD_CAPTURE.__defaults__
_STANDARD_TRACER = make_fx
_STANDARD_COMPILER = _compile_while
