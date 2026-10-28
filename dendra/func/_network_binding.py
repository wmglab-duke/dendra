"""Explicit raw-input bindings for functional Network execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree

from ._inputs import Inputs
from ._network_bindings import _canonicalize_mapping_tree, _map_containers
from ._types import FunctionalizationError

if TYPE_CHECKING:
    from ._network import FunctionalNetwork
    from ._network_rollout import CompiledNetworkChunk, NetworkRolloutInput


@dataclass(frozen=True, eq=False, init=False)
class BoundNetwork:
    """A Network transition bound to explicit parameters and raw constants.

    Construct with ``functional.bind(parameters, constants=constants)`` for one
    step, or ``kernel.bind(parameters, constants=constants)`` for a fixed chunk.
    The positional ``bind(parameters, constants)`` form is also supported.
    Mapping containers are snapshotted; tensor leaves retain their identities,
    aliases, and autograd history. Replacing a caller mapping entry requires a
    new binding. In-place parameter changes remain live.

    Network bindings retain raw constants rather than prepared coefficients.
    Each invocation derives fresh coefficients, preserving the graph lifetime
    of connection and NetStim parameter materialization. A reusable compiled
    chunk never retains the binding or its preparation graphs.

    An eager direct call accepts Population-name mappings of one-step
    :class:`Inputs` and a one-step ``draws=`` packet. Compiled bindings consume
    mappings with exactly ``steps`` time-leading samples. Legacy
    :class:`NetworkRolloutInput` packets and ``inputs=None`` retain their tape
    convention; use ``inputs={}`` for a step with only draws.
    Host ``run(bound, state, ...)``
    schedules arbitrary horizons and samples missing tapes in bounded spans.

    Direct calls accept ``callbacks`` from ``functional.make_callbacks(...)``
    and return results in ``auxiliary['callbacks']``. Supply its ``state`` as
    ``callback_state`` alongside the returned Network state to continue
    reducers and recording without repeating the boundary sample.
    """

    _functional: FunctionalNetwork
    _parameters: dict
    _constants: dict
    _compiled_chunk: CompiledNetworkChunk | None
    _eager_rollout: object

    def __init__(self, functional, parameters, constants, *, compiled_chunk=None):
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "BoundNetwork is an eager binding; use prepare_and_step() "
                "or prepare_and_rollout() inside torch.compile"
            )
        # Public Mapping methods can execute authored Python. Finish both
        # snapshots before validating the source or resolving owned methods.
        parameters = _map_containers(parameters, dict)
        constants = _map_containers(constants, dict)
        eager_rollout = self._validate_owner(functional, compiled_chunk)
        functional._validate_source()
        functional._validate_parameters(parameters)
        from ._network import _validate_tree

        _validate_tree(constants, functional._constant_schema, "constants")
        object.__setattr__(self, "_functional", functional)
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_constants", constants)
        object.__setattr__(self, "_compiled_chunk", compiled_chunk)
        object.__setattr__(
            self, "_eager_rollout", eager_rollout if compiled_chunk is None else None
        )

    @staticmethod
    def _validate_owner(functional, compiled_chunk):
        from ._network import FunctionalNetwork
        from ._network_rollout import CompiledNetworkChunk

        if type(functional) is not FunctionalNetwork:
            raise TypeError("functional must be an exact FunctionalNetwork")
        eager_rollout = functional.prepare_and_rollout
        if compiled_chunk is None and (
            getattr(eager_rollout, "__self__", None) is not functional
            or getattr(eager_rollout, "__func__", None)
            is not FunctionalNetwork.prepare_and_rollout
        ):
            raise FunctionalizationError(
                "eager binding requires the owned FunctionalNetwork.prepare_and_rollout"
            )
        if compiled_chunk is not None and (
            type(compiled_chunk) is not CompiledNetworkChunk
            or compiled_chunk._functional is not functional
        ):
            raise FunctionalizationError(
                "compiled chunk belongs to a different FunctionalNetwork plan"
            )
        return eager_rollout

    @property
    def functional(self) -> FunctionalNetwork:
        return self._functional

    @property
    def parameters(self):
        """Read-only nested parameter mapping with live tensor leaves."""
        return _map_containers(self._parameters, MappingProxyType)

    @property
    def constants(self):
        """Read-only nested raw constants with live tensor leaves."""
        return _map_containers(self._constants, MappingProxyType)

    @property
    def compiled_chunk(self) -> CompiledNetworkChunk | None:
        return self._compiled_chunk

    @property
    def steps(self) -> int:
        """Exact timestep count consumed by a direct invocation."""
        return 1 if self._compiled_chunk is None else self._compiled_chunk.steps

    def __call__(
        self,
        state,
        inputs: Mapping[str, Inputs] | NetworkRolloutInput | None = None,
        *,
        callbacks=None,
        callback_state=None,
        draws=None,
    ):
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "BoundNetwork is an eager binding; use prepare_and_step() "
                "or prepare_and_rollout() inside torch.compile"
            )
        if draws is not None:
            from ._draws import _normalize_runtime_draws

            draws = _normalize_runtime_draws(draws)
        options = {} if draws is None else {"draws": draws}
        if callbacks is not None:
            options["callbacks"] = callbacks
        if callback_state is not None:
            options["callback_state"] = callback_state
        if self._compiled_chunk is None:
            if isinstance(inputs, Mapping):
                inputs, _ = _canonicalize_mapping_tree(inputs)
                if all(isinstance(value, Inputs) for value in inputs.values()):
                    inputs, draws = pytree.tree_map_only(
                        torch.Tensor,
                        lambda value: value.unsqueeze(0),
                        (inputs, draws),
                    )
                    if draws is not None:
                        options["draws"] = draws
            return self._eager_rollout(
                self._parameters, self._constants, state, inputs, steps=1, **options
            )
        prepared = self._functional.prepare(self._parameters, self._constants)
        return self._compiled_chunk(
            self._parameters, prepared, state, inputs, **options
        )


__all__ = ["BoundNetwork"]
