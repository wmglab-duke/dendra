"""Public tensor containers for Dendra's functional execution API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NamedTuple

import torch

TensorTree = dict[str, object]
_ParameterSubset = Literal["all", "model", "intra", "extra"]

_INTRA_PREFIX = "stimulation.intra."
_EXTRA_PREFIX = "stimulation.extra."


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("parameter query must be a string")
    if not query:
        raise ValueError("parameter query must be a non-empty string")
    return query


def _normalize_queries(queries: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(queries, str):
        queries = (queries,)
    else:
        queries = tuple(queries)
    return tuple(_validate_query(query) for query in queries)


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
    """

    v_init: torch.Tensor
    states: dict[str, torch.Tensor] | None = None
    transforms: dict[str, torch.Tensor] | None = None


class PopulationTensors(NamedTuple):
    """Explicit tensors extracted from or initialized for one Population."""

    parameters: dict[str, torch.Tensor]
    constants: dict[str, torch.Tensor]
    state: TensorTree
    initialization: InitializationInput

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
        query: str,
        *,
        within: _ParameterSubset = "all",
    ) -> dict[str, torch.Tensor]:
        """Return every parameter whose canonical name contains ``query``.

        Matching is case-sensitive and preserves canonical extraction order.
        Use :meth:`parameter_name` when exactly one match is required.
        """

        query = _validate_query(query)
        return {
            name: value
            for name, value in self._parameter_subset(within).items()
            if query in name
        }

    def parameter_name(
        self,
        query: str,
        *,
        within: _ParameterSubset = "all",
    ) -> str:
        """Resolve an exact name or unique partial parameter-name match."""

        query = _validate_query(query)
        candidates = self._parameter_subset(within)
        if query in candidates:
            return query
        matches = tuple(name for name in candidates if query in name)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(
                f"no parameter matching {query!r} was found within {within!r}"
            )
        raise KeyError(
            f"parameter query {query!r} is ambiguous within {within!r}; "
            f"matches={list(matches)}"
        )

    def get_parameter(
        self,
        query: str,
        *,
        within: _ParameterSubset = "all",
    ) -> torch.Tensor:
        """Return the Tensor selected by an exact or unique partial name."""

        return self.parameters[self.parameter_name(query, within=within)]

    def independent_parameters(
        self,
        trainable: str | Iterable[str] = (),
        *,
        within: _ParameterSubset = "all",
    ) -> dict[str, torch.Tensor]:
        """Clone the complete parameter tree independently of its Population.

        Entries selected by exact or unique partial names in ``trainable`` are
        returned as :class:`torch.nn.Parameter` objects. Every other leaf is a
        detached ordinary Tensor. The returned mapping retains canonical names
        and can be passed directly to functional preparation and transitions.
        """

        # Validate the scope even when no trainable queries were requested.
        self._parameter_subset(within)
        trainable_names = {
            self.parameter_name(query, within=within)
            for query in _normalize_queries(trainable)
        }
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
    """Already assembled inputs for one functional Population step."""

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None


class RolloutInput(NamedTuple):
    """Time-first, already assembled inputs for a functional rollout."""

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None


class FunctionalizationError(RuntimeError):
    """Raised when a Population cannot yet be lowered without semantic loss."""


__all__ = [
    "FunctionalizationError",
    "InitializationInput",
    "PopulationTensors",
    "RolloutInput",
    "StepInput",
]
