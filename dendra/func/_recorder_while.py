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
from torch.utils import _pytree as pytree

from dendra.helpers import MEMBRANE_ASSEMBLY

from ._callback_carry import _CallbackCarryPartition
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
from ._carry_packing import _clone_tree_like, _PackedCarryLayout
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
        state = functional._clone_state_tree(state)
        callback_carry = _CallbackCarryPartition.from_plans(plans)
        carries, invariant_carries = callback_carry.split(carries)
        layout = _PackedCarryLayout.from_tree((state, carries))

        def authored(
            parameters, prepared, state, ve, intra, carries, invariant_carries
        ):
            # Deliberately use the accepted-clock step, not vectorized waveform
            # assembly: start + arange*dt can round pulse boundaries differently.
            next_state, auxiliary = functional._step_values(
                parameters, prepared, state, StepInput(ve, intra)
            )
            updated, emitted = _update_callback_values(
                plans,
                callback_carry.merge(carries, invariant_carries),
                next_state,
                auxiliary,
            )
            updated, _ = callback_carry.split(updated)
            # This pure transition is composed several times inside one loop
            # body. Intermediate views and unchanged leaves need no ownership
            # copy; pack only the actual higher-order loop's returned carry.
            return next_state, updated, emitted or {}

        graph = make_fx(authored)(
            parameters,
            prepared,
            state,
            None if ve is None else ve[0],
            None if intra is None else intra[0],
            carries,
            invariant_carries,
        )
        # Raster emits only after a transition, so its emission schema is not
        # present at initialize(). Built-ins have value-independent emission
        # structure: observe one pure callback update on cloned example state
        # to discover shapes without an extra numerical model transition.
        _, sample_emitted = _update_callback_values(
            plans, callback_carry.merge(carries, invariant_carries), state, None
        )
        samples, emission_spec = pytree.tree_flatten(sample_emitted or {})
        shapes = [(tuple(value.shape), value.dtype, value.device) for value in samples]

        def execute(parameters, prepared, state, ve, intra, carries):
            carries, invariant_carries = callback_carry.split(carries)
            buffers = tuple(
                torch.empty((steps, *shape), dtype=dtype, device=device)
                for shape, dtype, device in shapes
            )
            initial = layout.pack((functional._clone_state_tree(state), carries))

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
                    invariant_carries,
                )
                samples, _ = pytree.tree_flatten(emitted)
                for buffer, sample in zip(buffers, samples, strict=True):
                    buffer.index_copy_(0, selector, sample.unsqueeze(0))
                return next_state, next_carries

            def condition(index, packed):
                return index < full_steps

            def body(index, packed):
                current, current_carries = layout.unpack(packed)
                for offset in range(block):
                    current, current_carries = apply_step(
                        index + offset, current, current_carries
                    )
                # The HOP sees only one independently owned Tensor per
                # dtype/device. Duplicate logical leaves occupy separate
                # regions of a fresh pack, not duplicate output handles.
                return index + block, layout.pack((current, current_carries))

            index = torch.zeros((), dtype=torch.int64, device=functional.device)
            index, packed = torch.while_loop(condition, body, (index, initial))
            # The tail is bounded by block-1, independent of the chunk horizon.
            if steps != full_steps:
                final, final_carries = layout.unpack(packed)
                for offset in range(steps - full_steps):
                    final, final_carries = apply_step(
                        index + offset, final, final_carries
                    )
                packed = layout.pack((final, final_carries))
            observed = pytree.tree_unflatten(list(buffers), emission_spec)
            emitted = {plan.name: observed.get(plan.name) for plan in plans}
            return packed, emitted

        compiled = _compile_while(execute, self._options)

        @wraps(compiled)
        def materialize(parameters, prepared, state, ve, intra, carries):
            packed, emitted = compiled(parameters, prepared, state, ve, intra, carries)
            final, final_carries = layout.unpack(packed)
            # Public leaves must not share the private pack or each other.
            # Keep ownership materialization outside compilation: identical
            # clones must not be CSE'd into duplicate C++ HOP output handles.
            final = functional._clone_state_tree(final)
            final_carries = callback_carry.merge(
                final_carries, callback_carry.split(carries)[1]
            )
            final_carries = _clone_tree_like(final_carries, carries)
            return final, {"v": final["integrator"]["v"]}, final_carries, emitted

        return materialize


__all__ = []
