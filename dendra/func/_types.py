"""Public tensor containers for Dendra's functional execution API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, NamedTuple

import torch

from ._parameter_queries import ParameterQuery, matching_names
from ._parameter_queries import parameter_name as _parameter_name
from ._parameter_queries import select_names

TensorTree = dict[str, object]
_ParameterSubset = Literal["all", "model", "intra", "extra"]

_INTRA_PREFIX = "stimulation.intra."
_EXTRA_PREFIX = "stimulation.extra."


class RandomParameterDraw(NamedTuple):
    """RAND/NOISE initialization draws and resulting shared stream states."""

    draws: dict[str, torch.Tensor]
    rng: dict[str, dict[str, torch.Tensor]]


class RuntimeNoiseDraw(NamedTuple):
    """Detached NOISE/State SDE draws and resulting shared stream states."""

    draws: dict[str, torch.Tensor]
    rng: dict[str, dict[str, torch.Tensor]]


class InitializationInput(NamedTuple):
    """Explicit tensor inputs consumed only while constructing initial state.

    ``v_init`` is the fully expanded initial membrane voltage. ``states``
    contains only insertion-time ``ic`` leaves, keyed by their canonical
    functional state names. Their
    shapes, dtypes, and devices must match the corresponding support-local
    state leaves. ``transforms`` contains the explicitly registered tensor
    inputs of pure Population initialization transforms. Keeping
    initialization-only values separate from runtime constants makes batching
    and differentiation explicit without invalidating prepared transition
    workspaces.

    ``cache`` contains a Population-owned steady-state snapshot when one was
    bound during lowering. Cached initialization restores this mapping,
    skips fresh initialization hooks, and returns effective constants with
    the initialized parameter and state packet. ``random_draw`` supplies RAND
    and NOISE base draws and resulting stream state from
    ``sample_initialization_draw``.
    Host runners sample an independent packet when this field is omitted.
    """

    v_init: torch.Tensor
    states: dict[str, torch.Tensor] | None = None
    transforms: dict[str, torch.Tensor] | None = None
    cache: Mapping[str, object] | None = None
    random_draw: RandomParameterDraw | None = None


class PopulationTensors(NamedTuple):
    """Explicit tensors extracted from or initialized for one Population."""

    parameters: dict[str, torch.Tensor]
    constants: dict[str, torch.Tensor]
    state: TensorTree
    initialization: InitializationInput

    @property
    def trainable_parameters(self) -> dict[str, torch.Tensor]:
        """Named parameter leaves with live ``requires_grad=True`` flags.

        Shared aliases remain visible. Deduplicate values by identity before
        passing them to an optimizer.
        """
        return {
            name: value
            for name, value in self.parameters.items()
            if value.requires_grad
        }

    def _parameter_subset(
        self,
        within: _ParameterSubset,
    ) -> dict[str, torch.Tensor]:
        if within == "all":
            return dict(self.parameters)
        if within == "model":
            return {
                name: value
                for name, value in self.parameters.items()
                if not name.startswith("stimulation.")
            }
        if within == "intra":
            prefix = _INTRA_PREFIX
        elif within == "extra":
            prefix = _EXTRA_PREFIX
        else:
            raise ValueError(
                "within must be one of 'all', 'model', 'intra', or 'extra'"
            )
        return {
            name: value
            for name, value in self.parameters.items()
            if name.startswith(prefix)
        }

    @property
    def model_parameters(self) -> dict[str, torch.Tensor]:
        """Model-owned raw parameters, excluding stimulation leaves."""

        return self._parameter_subset("model")

    @property
    def intra_parameters(self) -> dict[str, torch.Tensor]:
        """Replaceable leaves owned by registered intracellular stimulation."""

        return self._parameter_subset("intra")

    @property
    def extra_parameters(self) -> dict[str, torch.Tensor]:
        """Replaceable leaves owned by bound extracellular stimulation."""

        return self._parameter_subset("extra")

    def find_parameters(
        self,
        query: ParameterQuery,
        *,
        within: _ParameterSubset = "all",
    ) -> dict[str, torch.Tensor]:
        """Find parameters by substring, full-name glob, or regular expression.

        Matching is case-sensitive and preserves canonical extraction order.
        Use ``re:expression`` or a compiled string regex for regex search.
        Use :meth:`parameter_name` when exactly one match is required.
        """

        candidates = self._parameter_subset(within)
        return {name: candidates[name] for name in matching_names(candidates, query)}

    def parameter_name(
        self,
        query: ParameterQuery,
        *,
        within: _ParameterSubset = "all",
    ) -> str:
        """Resolve one exact, partial, glob, or regex parameter-name match."""

        candidates = self._parameter_subset(within)
        return _parameter_name(candidates, query, within=within)

    def get_parameter(
        self,
        query: ParameterQuery,
        *,
        within: _ParameterSubset = "all",
    ) -> torch.Tensor:
        """Return the Tensor selected by one exact, partial, glob, or regex match."""

        return self.parameters[self.parameter_name(query, within=within)]

    def independent_parameters(
        self,
        trainable: ParameterQuery | Iterable[ParameterQuery] = (),
        *,
        within: _ParameterSubset = "all",
    ) -> dict[str, torch.Tensor]:
        """Clone the complete parameter tree independently of its Population.

        Exact or unique partial names select one leaf; full-name globs and
        ``re:`` or compiled regex selectors can select multiple leaves. Each
        selector must match. Selected leaves are returned as
        :class:`torch.nn.Parameter` objects. Every other leaf is a
        detached ordinary Tensor. The returned mapping retains canonical names
        and can be passed directly to functional preparation and transitions.
        """

        # Validate the scope even when no trainable queries were requested.
        candidates = self._parameter_subset(within)
        trainable_names = select_names(candidates, trainable, within=within)
        non_differentiable = sorted(
            name
            for name in trainable_names
            if not (
                torch.is_floating_point(self.parameters[name])
                or torch.is_complex(self.parameters[name])
            )
        )
        if non_differentiable:
            raise ValueError(
                "trainable parameter leaves must use a floating-point or complex "
                f"dtype; invalid={non_differentiable}"
            )
        # Functional optimizers should never share storage or autograd history
        # with the source Population. Also materialize ordinary versioned
        # tensors when this helper is called from inference mode.
        with torch.inference_mode(False):
            independent = {
                name: value.detach().clone(memory_format=torch.preserve_format)
                for name, value in self.parameters.items()
            }
            for name in trainable_names:
                independent[name] = torch.nn.Parameter(independent[name])
        return independent


class StepInput(NamedTuple):
    """Assembled voltage/current inputs and an explicit single-step NOISE packet."""

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None
    noise: RuntimeNoiseDraw | None = None


class RolloutInput(NamedTuple):
    """Time-first voltage/current inputs and an explicit NOISE draw/RNG tape."""

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None
    noise: RuntimeNoiseDraw | None = None


class FunctionalizationError(RuntimeError):
    """Raised when a Population cannot yet be lowered without semantic loss."""


__all__ = [
    "FunctionalizationError",
    "InitializationInput",
    "RandomParameterDraw",
    "RuntimeNoiseDraw",
    "PopulationTensors",
    "RolloutInput",
    "StepInput",
]
