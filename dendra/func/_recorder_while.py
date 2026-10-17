"""Bounded inference-only built-in callbacks with invocation-local buffers.

Recorder remains a pure per-step emitter. Only this private executor mutates
fresh buffers allocated inside its compiled invocation; no public input,
callback object, or persistent workspace is mutated. The buffers are captured
inference outputs, not while-loop carries, so their indexed writes need not
clone the complete recording every timestep.
Reducer state, crossing latches, and callback phase are explicit loop carry.
"""

from __future__ import annotations

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from dendra._bootstrap import isolate_compilation_callable
from dendra.helpers import MEMBRANE_ASSEMBLY

from ._callbacks import (
    AnomalyDetector,
    _BoundActiveAL,
    _BoundActiveALCount,
    _BoundAPCount,
    _BoundRaster,
    _BoundRecorder,
    _update_callback_values,
)
from ._types import StepInput


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
        # Captured-tensor mutation is an inference-only public while_loop
        # capability on the verified 2.14 series. Older supported Dendra/Torch
        # combinations keep their existing pure unrolled callback executor.
        version = str(torch.__version__).split("+", 1)[0].split(".")
        try:
            supports_local_buffers = tuple(map(int, version[:2])) >= (2, 14)
        except ValueError:
            supports_local_buffers = False
        if not supports_local_buffers:
            return (
                None,
                "Built-in inference callback loops require Torch 2.14 or newer.",
            )
        signature = (ve is not None, intra is not None, MEMBRANE_ASSEMBLY.value)
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
        from ._population import _STRUCTURED_ROLLOUT_CHUNK

        functional, plans, steps = self._functional, self._plans, self._steps
        block = _STRUCTURED_ROLLOUT_CHUNK
        full_steps = (steps // block) * block
        examples = pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.detach().clone(),
            (parameters, prepared, state, ve, intra, carries),
        )
        parameters, prepared, state, ve, intra, carries = examples

        def authored(parameters, prepared, state, ve, intra, carries):
            # Deliberately use the accepted-clock step, not vectorized waveform
            # assembly: start + arange*dt can round pulse boundaries differently.
            next_state, auxiliary = functional._step_values(
                parameters, prepared, state, StepInput(ve, intra)
            )
            updated, emitted = _update_callback_values(
                plans, carries, next_state, auxiliary
            )
            updated = pytree.tree_map_only(
                torch.Tensor, lambda value: value.clone(), updated
            )
            return functional._clone_state_tree(next_state), updated, emitted or {}

        graph = make_fx(authored)(
            parameters,
            prepared,
            functional._clone_state_tree(state),
            None if ve is None else ve[0],
            None if intra is None else intra[0],
            carries,
        )
        # Raster emits only after a transition, so its emission schema is not
        # present at initialize(). Built-ins have value-independent emission
        # structure: observe one pure callback update on cloned example state
        # to discover shapes without an extra numerical model transition.
        _, sample_emitted = _update_callback_values(plans, carries, state, None)
        samples, emission_spec = pytree.tree_flatten(sample_emitted or {})
        shapes = [(tuple(value.shape), value.dtype, value.device) for value in samples]

        def execute(parameters, prepared, state, ve, intra, carries):
            buffers = tuple(
                torch.empty((steps, *shape), dtype=dtype, device=device)
                for shape, dtype, device in shapes
            )
            initial = functional._clone_state_tree(state)
            initial_carries = pytree.tree_map_only(
                torch.Tensor, lambda value: value.clone(), carries
            )

            def apply_step(index, current, current_carries):
                selector = index.reshape(1)
                next_state, next_carries, emitted = graph(
                    parameters,
                    prepared,
                    current,
                    None
                    if ve is None
                    else torch.index_select(ve, 0, selector).squeeze(0),
                    None
                    if intra is None
                    else torch.index_select(intra, 0, selector).squeeze(0),
                    current_carries,
                )
                samples, _ = pytree.tree_flatten(emitted)
                for buffer, sample in zip(buffers, samples, strict=True):
                    buffer.index_copy_(0, selector, sample.unsqueeze(0))
                return next_state, next_carries

            def condition(index, current, current_carries):
                return index < full_steps

            def body(index, current, current_carries):
                for offset in range(block):
                    current, current_carries = apply_step(
                        index + offset, current, current_carries
                    )
                return index + block, current, current_carries

            index = torch.zeros((), dtype=torch.int64, device=functional.device)
            index, final, final_carries = torch.while_loop(
                condition, body, (index, initial, initial_carries)
            )
            # The tail is bounded by block-1, independent of the chunk horizon.
            for offset in range(steps - full_steps):
                final, final_carries = apply_step(index + offset, final, final_carries)
            observed = pytree.tree_unflatten(list(buffers), emission_spec)
            emitted = {plan.name: observed.get(plan.name) for plan in plans}
            return final, {"v": final["integrator"]["v"]}, final_carries, emitted

        return torch.compile(isolate_compilation_callable(execute), **self._options)


__all__ = []
