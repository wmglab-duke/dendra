"""Bounded inference-only built-in callbacks with invocation-local buffers.

Recorder remains a pure per-step emitter. Only this private executor mutates
fresh buffers allocated inside its compiled invocation; no public input,
callback object, or persistent workspace is mutated. The buffers are captured
inference outputs, not while-loop carries, so their indexed writes need not
clone the complete recording every timestep.
Reducer state, crossing latches, and callback phase are explicit loop carry.
"""

from __future__ import annotations

from functools import wraps

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from dendra.helpers import MEMBRANE_ASSEMBLY

from ._callbacks import (
    AnomalyDetector,
    _BoundActiveAL,
    _BoundActiveALCount,
    _BoundAPCount,
    _BoundLFP,
    _BoundRaster,
    _BoundRecorder,
    _update_callback_values,
)
from ._tensor_tree_while import _CompiledTensorTreeWhile, _schema
from ._types import StepInput
from ._while_inductor import _compile_while


class _RecorderWhileExecutor:
    """Capture one accepted-clock transition, then reuse a bounded loop body."""

    def __init__(self, functional, *, steps, plans, compile_options):
        self._functional = functional
        self._steps = steps
        self._plans = plans
        self._options = compile_options
        self._kernels = {}
        self._capture_errors = {}

    def kernel_for(self, parameters, prepared, state, ve, intra, carries):
        from ._population import _STRUCTURED_ROLLOUT_CHUNK

        if self._steps < _STRUCTURED_ROLLOUT_CHUNK:
            return None, "The chunk is shorter than one inference loop block."
        supported = {
            _BoundRecorder,
            AnomalyDetector,
            _BoundRaster,
            _BoundAPCount,
            _BoundLFP,
            _BoundActiveAL,
            _BoundActiveALCount,
        }
        if not self._plans or not all(
            type(plan.callback) in supported for plan in self._plans
        ):
            return None, (
                "Inference callback loops require supported built-in functional "
                "callbacks; use scan for custom callbacks."
            )
        # Captured-tensor mutation is part of the project's Torch>=2.14
        # baseline. Compile the selected route rather than choosing behavior
        # from a patch-version label.
        signature = (
            _schema((parameters, prepared, state, ve, intra, carries)),
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
                        parameters, prepared, state, ve, intra, carries
                    )
            except Exception as error:
                reason = (
                    "No captured callback inference step graph is available: "
                    f"{type(error).__name__}: {error}"
                )
                self._capture_errors[signature] = reason
                return None, reason
        return self._kernels[signature], None

    def _capture(self, parameters, prepared, state, ve, intra, carries):
        functional, plans = self._functional, self._plans

        def body(parameters, prepared, current, current_carries, sample):
            # Accepted-clock sampling retains native waveform boundaries.
            next_state, auxiliary = functional._step_values(
                parameters, prepared, current, StepInput(*sample)
            )
            updated, emitted = _update_callback_values(
                plans, current_carries, next_state, auxiliary
            )
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
        kernel = engine.kernel_for(parameters, prepared, state, carries, (ve, intra))

        @wraps(kernel)
        def materialize(parameters, prepared, state, ve, intra, carries):
            final, final_carries, observed = kernel(
                parameters, prepared, state, carries, (ve, intra)
            )
            emitted = {plan.name: observed.get(plan.name) for plan in plans}
            return final, {"v": final["integrator"]["v"]}, final_carries, emitted

        return materialize


__all__ = []
