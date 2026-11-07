"""Explicit raw or prepared bindings for functional Population execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from ._binding_sources import _BINDING_UNSET, _binding_packet_sources
from ._inputs import Inputs
from ._types import FunctionalizationError, RolloutInput, StepInput

if TYPE_CHECKING:
    from ._compiled import CompiledPopulationChunk
    from ._population import FunctionalPopulation


@dataclass(frozen=True, eq=False, init=False)
class BoundPopulation:
    """A transition bound to explicit parameters and preparation inputs.

    ``functional.bind(parameters, constants=constants)`` binds one eager step;
    the same keyword form on a compiled chunk binds its fixed-length rollout.
    Raw mapping containers are snapshotted while tensor identities, aliases,
    and autograd history remain live. Each complete invocation owns its
    preparation graph. Safe inference preparation can be reused while its
    parameter and constant bindings remain fresh.

    ``bind(tensors)`` selects raw sources from an exact ``PopulationTensors``
    bundle; ``bind(continuation)`` selects its effective sources. Packet forms
    cannot be combined with ``prepared`` or ``constants``. They read only
    parameters/constants and retain no packet, state, initialization recipe or
    callback carry. Supply state or the continuation explicitly to ``run``.

    ``bind(parameters, prepared)`` retains explicit preparation for advanced
    use. Recreate that prepared binding for each independent backward graph;
    in-place preparation input changes retain their existing freshness errors.
    Replacing an entry in a caller mapping requires a new binding in either
    form. Reusable kernels retain neither bindings nor preparation graphs.
    Call ``run(bound, state, ...)`` to schedule arbitrary horizons.

    Raw-bound host runs return ``auxiliary['continuation']``. Resume with
    ``run(bound, packet, ...)`` when its sources match this binding's Tensor
    objects. Different sources require a new raw binding. Prepared bindings
    and direct calls retain their explicit state/callback-carry contract.

    Direct eager calls consume one :class:`Inputs` frame; compiled calls
    consume exactly ``steps`` time-leading samples. Supply ``callbacks`` from
    ``functional.make_callbacks(...)`` to return results in
    ``auxiliary['callbacks']``. Continue their carry with
    ``callback_state=auxiliary['callbacks'].state`` alongside the returned
    simulation state; continuation records only new samples.
    """

    _functional: FunctionalPopulation
    _parameters: dict[str, torch.Tensor]
    _prepared: object
    _compiled_chunk: CompiledPopulationChunk | None
    _eager_step: object
    _constants: object
    _inference_prepared: object | None
    _inference_prepared_context: tuple | None

    def __init__(
        self,
        functional,
        parameters,
        prepared=_BINDING_UNSET,
        *,
        constants=_BINDING_UNSET,
        compiled_chunk=None,
    ):
        sources = _binding_packet_sources(
            parameters, population=True, prepared=prepared, constants=constants
        )
        if sources is None:
            if prepared is _BINDING_UNSET and constants is _BINDING_UNSET:
                raise TypeError("bind requires prepared or constants")
            if prepared is not _BINDING_UNSET and constants is not _BINDING_UNSET:
                raise TypeError("prepared and constants are mutually exclusive")
        else:
            parameters, constants = sources
        if sources is not None or constants is not _BINDING_UNSET:
            from ._network_bindings import _map_containers

            # Public Mapping traversal can execute authored Python. Finish
            # both snapshots before qualifying source or prepared freshness.
            parameters = _map_containers(parameters, dict)
            constants = _map_containers(constants, dict)
            eager_step = self._validate_owner(functional, compiled_chunk)
            functional._validate_source()
            functional._validate_parameters(parameters)
            functional._validate_constants(constants)
            object.__setattr__(self, "_functional", functional)
            object.__setattr__(self, "_parameters", parameters)
            object.__setattr__(self, "_prepared", None)
            object.__setattr__(self, "_constants", constants)
            object.__setattr__(self, "_inference_prepared", None)
            object.__setattr__(self, "_inference_prepared_context", None)
            object.__setattr__(self, "_compiled_chunk", compiled_chunk)
            object.__setattr__(
                self, "_eager_step", eager_step if compiled_chunk is None else None
            )
            return
        eager_step = self._validate_owner(functional, compiled_chunk)
        functional._validate_source()
        self._initialize(functional, parameters, prepared, compiled_chunk, eager_step)

    @classmethod
    def _from_source_validated(
        cls, functional, parameters, prepared, *, compiled_chunk
    ):
        """Bind inside an owned invocation, never retain a validation token."""
        eager_step = cls._validate_owner(functional, compiled_chunk)
        bound = object.__new__(cls)
        bound._initialize(functional, parameters, prepared, compiled_chunk, eager_step)
        return bound

    @staticmethod
    def _validate_owner(functional, compiled_chunk):
        from ._compiled import CompiledPopulationChunk
        from ._population import FunctionalPopulation

        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "BoundPopulation is an eager binding; use prepare_and_step() "
                "or prepare_and_rollout() inside torch.compile"
            )
        if type(functional) is not FunctionalPopulation:
            raise TypeError("functional must be an exact FunctionalPopulation")
        eager_step = functional.step
        if compiled_chunk is None and (
            getattr(eager_step, "__self__", None) is not functional
            or getattr(eager_step, "__func__", None) is not FunctionalPopulation.step
        ):
            raise FunctionalizationError(
                "eager binding requires the owned FunctionalPopulation.step; "
                "use a generic one-step callable with the legacy runner for adapters"
            )
        if compiled_chunk is not None:
            if type(compiled_chunk) is not CompiledPopulationChunk:
                raise TypeError(
                    "compiled_chunk must be an exact CompiledPopulationChunk"
                )
            if compiled_chunk._functional is not functional:
                raise FunctionalizationError(
                    "compiled chunk belongs to a different FunctionalPopulation plan"
                )
        return eager_step

    def _initialize(self, functional, parameters, prepared, compiled_chunk, eager_step):
        functional._validate_parameters(parameters)
        snapshot = dict(parameters)
        functional._validate_prepared(prepared, snapshot)
        object.__setattr__(self, "_functional", functional)
        object.__setattr__(self, "_parameters", snapshot)
        object.__setattr__(self, "_prepared", prepared)
        object.__setattr__(self, "_constants", None)
        object.__setattr__(self, "_inference_prepared", None)
        object.__setattr__(self, "_inference_prepared_context", None)
        object.__setattr__(self, "_compiled_chunk", compiled_chunk)
        object.__setattr__(
            self, "_eager_step", eager_step if compiled_chunk is None else None
        )

    @property
    def functional(self) -> FunctionalPopulation:
        """The lowered model plan that owns this binding."""
        return self._functional

    @property
    def parameters(self) -> Mapping[str, torch.Tensor]:
        """Read-only parameter mapping; its tensor leaves are not copied."""
        if self._is_raw:
            from ._network_bindings import _map_containers

            return _map_containers(self._parameters, MappingProxyType)
        return MappingProxyType(self._parameters)

    @property
    def constants(self):
        """Read-only raw constants, or ``None`` for an explicit prepared binding."""
        if not self._is_raw:
            return None
        from ._network_bindings import _map_containers

        return _map_containers(self._constants, MappingProxyType)

    @property
    def prepared(self):
        """The explicitly supplied preparation, or ``None`` for raw bindings."""
        return self._prepared

    @property
    def _is_raw(self):
        return self._constants is not None

    def _prepared_binding(self, *, source_validated=False):
        """Own a prepared frame only for this complete forward invocation."""
        if not self._is_raw:
            return self
        from ._population import _OWNED_POPULATION_PREPARE, _binding_capture_modes
        from ._preparation import (
            _cacheable_preparation_inputs,
            _cacheable_preparation_values,
            _select_population_preparation,
        )

        functional = self._functional
        public_prepare = functional.prepare
        owned = (
            getattr(public_prepare, "__self__", None) is functional
            and getattr(public_prepare, "__func__", None) is _OWNED_POPULATION_PREPARE
        )
        if not source_validated or not owned:
            functional._validate_source()
        modes = _binding_capture_modes() if owned else None
        context = None if modes is None else (modes, torch.get_default_dtype())
        cache = (
            owned
            and modes is not None
            and _cacheable_preparation_inputs(self._parameters, self._constants)
        )
        prepared = _select_population_preparation(
            functional,
            self._parameters,
            self._constants,
            cached=(
                self._inference_prepared
                if context == self._inference_prepared_context
                else None
            ),
            cache=cache,
            prepare=(functional._prepare_source_validated if owned else public_prepare),
        )
        if not owned:
            functional._validate_source()
        bound = object.__new__(type(self))
        bound._initialize(
            functional,
            self._parameters,
            prepared,
            self._compiled_chunk,
            self._eager_step,
        )
        if cache:
            retained = (
                prepared
                if context == (_binding_capture_modes(), torch.get_default_dtype())
                and _cacheable_preparation_values(prepared)
                else None
            )
            object.__setattr__(self, "_inference_prepared", retained)
            object.__setattr__(
                self,
                "_inference_prepared_context",
                context if retained is not None else None,
            )
        return bound

    @property
    def compiled_chunk(self) -> CompiledPopulationChunk | None:
        """The bound fixed-width kernel, or None for an eager binding."""
        return self._compiled_chunk

    @property
    def steps(self) -> int:
        """Exact timestep count consumed by a direct invocation."""
        return 1 if self._compiled_chunk is None else self._compiled_chunk.steps

    def __call__(
        self,
        state,
        inputs: Inputs | RolloutInput | StepInput | None = None,
        *,
        callbacks=None,
        callback_state=None,
        draws=None,
    ):
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "BoundPopulation is an eager binding; use prepare_and_step() "
                "or prepare_and_rollout() inside torch.compile"
            )
        if self._is_raw:
            from ._draws import _with_runtime_draws
            from ._network_bindings import _canonicalize_mapping_tree

            legacy_step = self._compiled_chunk is not None and isinstance(
                inputs, StepInput
            )
            if draws is not None:
                inputs = _with_runtime_draws(
                    inputs,
                    draws,
                    step=self._compiled_chunk is None or isinstance(inputs, StepInput),
                )
            (state, inputs), _ = _canonicalize_mapping_tree((state, inputs))
            if legacy_step and draws is not None:
                from ._inputs import _legacy_step_input

                inputs = _legacy_step_input(inputs)
            options = {}
            if callbacks is not None:
                options["callbacks"] = callbacks
            if callback_state is not None:
                options["callback_state"] = callback_state
            return self._prepared_binding()(state, inputs, **options)
        # MappingProxyType is an opaque leaf to PyTorch's PyTree machinery.
        # Only the owned ordinary dict may cross a tensor execution boundary.
        options = {} if draws is None else {"draws": draws}
        if callbacks is not None:
            options["callbacks"] = callbacks
        if callback_state is not None:
            options["callback_state"] = callback_state
        if self._compiled_chunk is None:
            return self._eager_step(
                self._parameters, self._prepared, state, inputs, **options
            )
        return self._compiled_chunk(
            self._parameters, self._prepared, state, inputs, **options
        )


__all__ = ["BoundPopulation"]
