"""Internal tensor-only execution boundaries for functional Populations.

These helpers deliberately remain private while the functional lowering is
under development.  Validation belongs to :meth:`_LoweredPopulationChunk.bind`;
the callable itself contains only tensor operations so PyTorch transforms can
be applied before compilation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from ._types import RolloutInput

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

    __slots__ = ("_functional", "_steps")

    def __init__(self, functional: FunctionalPopulation, *, steps: int):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        functional._validate_source()
        self._functional = functional
        self._steps = steps

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
        return self._functional._execute_rollout_values(
            parameters,
            prepared,
            state,
            ve,
            intra,
            self._steps,
        )


__all__ = []
