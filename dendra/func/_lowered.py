"""Internal tensor-only execution boundaries for functional Populations.

These helpers deliberately remain private while the functional lowering is
under development.  Validation belongs to :meth:`_LoweredPopulationChunk.bind`;
the callable itself contains only tensor operations so PyTorch transforms can
be applied before compilation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from ._callbacks import _stack_callback_emissions, _update_callback_values
from ._types import RolloutInput, StepInput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._population import FunctionalPopulation, _PreparedPopulation


class _PopulationChunkOperands(NamedTuple):
    """Validated tensor operands accepted by one lowered chunk."""

    parameters: Mapping[str, torch.Tensor]
    prepared: Mapping[str, object]
    state: Mapping[str, object]
    ve: torch.Tensor | None
    intra: torch.Tensor | None


class _LoweredPopulationChunk:
    """An uncompiled, fixed-length, tensor-only Population transition.

    ``bind`` is the eager Python boundary: it validates the source plan,
    parameters, opaque preparation, state, and optional inputs, then unwraps
    the preparation into explicit tensor operands.  ``__call__`` intentionally
    performs none of those checks.  This split permits compositions such as
    ``torch.compile(torch.func.jacrev(projected_chunk))`` without tracing
    opaque plans or Python validation through Dynamo.

    The class is private because both the operand tree and lowering boundary
    remain implementation details during the functional campaign.
    """

    __slots__ = ("_functional", "_steps", "_stepwise_stimulation")

    def __init__(
        self,
        functional: FunctionalPopulation,
        *,
        steps: int,
        stepwise_stimulation: bool = False,
    ):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        functional._validate_source()
        self._functional = functional
        self._steps = steps
        self._stepwise_stimulation = stepwise_stimulation

    @property
    def steps(self) -> int:
        return self._steps

    def bind(
        self,
        parameters,
        prepared: _PreparedPopulation,
        state,
        inputs: RolloutInput | None = None,
    ) -> _PopulationChunkOperands:
        """Validate and unwrap one call without entering the tensor kernel."""
        functional = self._functional
        functional._validate_source()
        functional._validate_parameters(parameters)
        functional._validate_state(state)
        prepared_values = functional._validate_prepared(prepared, parameters)
        ve, intra, _resolved_steps = functional._resolve_rollout_inputs(
            inputs,
            self._steps,
        )
        return _PopulationChunkOperands(
            parameters,
            prepared_values,
            state,
            ve,
            intra,
        )

    def __call__(self, parameters, prepared, state, ve, intra):
        """Execute the pure tensor transition for exactly ``steps`` steps."""
        if self._stepwise_stimulation:
            # A host step samples bound Waveforms at its accepted clock. A
            # vector start + arange * dt can round differently from repeated
            # clock additions and miss narrow pulses. Preserve that recurrence
            # inside the fixed compiled chunk; compiler size is bounded by C.
            for index in range(self._steps):
                state, auxiliary = self._functional._step_values(
                    parameters,
                    prepared,
                    state,
                    StepInput(
                        ve=None if ve is None else ve[index],
                        intra=None if intra is None else intra[index],
                    ),
                )
            return state, auxiliary
        return self._functional._execute_rollout_values(
            parameters,
            prepared,
            state,
            ve,
            intra,
            self._steps,
        )


class _LoweredCallbackPopulationChunk:
    """A fixed tensor recurrence including every per-step callback update.

    Keep only the callback plans, not their collection: the compiled wrapper
    caches these kernels with weak collection keys. No initialized carry or
    differentiable preparation belongs to this static callable.
    """

    __slots__ = ("_functional", "_steps", "_plans")

    def __init__(self, functional, *, steps, plans):
        self._functional = functional
        self._steps = steps
        self._plans = plans

    def __call__(self, parameters, prepared, state, ve, intra, carries):
        emissions = []
        for index in range(self._steps):
            # Sample bound waveforms at the accepted clock each time, matching
            # the host recurrence even at floating-point pulse boundaries.
            state, auxiliary = self._functional._step_values(
                parameters,
                prepared,
                state,
                StepInput(
                    ve=None if ve is None else ve[index],
                    intra=None if intra is None else intra[index],
                ),
            )
            carries, emitted = _update_callback_values(
                self._plans, carries, state, auxiliary
            )
            if emitted is not None:
                emissions.append(emitted)
        return (
            state,
            auxiliary,
            carries,
            _stack_callback_emissions(self._plans, emissions),
        )


__all__ = []
