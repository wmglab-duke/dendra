"""Pure callback reducers for functional Population execution.

Functional callbacks are a parallel API to Dendra's mutable callback classes.
They own explicit tensor carry, return optional tensor emissions, and finalize a
public tensor result. Host runners can compile callback updates together with
fixed-length model chunks while keeping the model-only transition API unchanged.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ._types import FunctionalizationError

if TYPE_CHECKING:
    from ._population import FunctionalPopulation


def _get_path(tree: Mapping[str, object], path: tuple[str, ...]):
    value: Any = tree
    for part in path:
        value = value[part]
    return value


def _put_path(tree: dict[str, object], path: tuple[str, ...], value) -> None:
    cursor = tree
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


class FunctionalCallback(ABC):
    """Base contract for a pure functional callback.

    Callback objects are immutable configuration. All values that evolve
    during execution must be returned in ``carry``. Carry, emissions, and the
    finalized result must be tensor PyTrees with stable structure, leaf shape,
    dtype, and device. An emission may be ``None`` when a callback is a
    constant-memory reducer.

    ``initialize`` runs exactly once for a fresh callback execution. Its
    ``auxiliary`` argument is ``None`` because no transition has occurred.
    ``update`` runs after each successful model step and receives that step's
    auxiliary mapping. Dendra stacks non-``None`` emissions on a leading
    sample axis before calling ``finalize`` once for each public runner result.
    If a segment has no samples and no initial emission schema, ``finalize``
    receives ``None`` and must still return its normal result PyTree (for
    example, an explicitly shaped empty recording).

    Implementations must be deterministic and side-effect-free: do not mutate
    callback objects, perform I/O, consume hidden randomness, call ``.item()``,
    or branch in Python on tensor values. These constraints make callbacks
    safe for autograd, ``torch.func``, compilation, and checkpoint replay.
    """

    @abstractmethod
    def initialize(self, state, auxiliary):
        """Return ``(carry, initial_emission_or_none)`` for a fresh run."""

    @abstractmethod
    def update(self, carry, state, auxiliary):
        """Return ``(next_carry, emission_or_none)`` after one model step."""

    @abstractmethod
    def finalize(self, carry, emissions):
        """Return the public tensor result for this execution segment."""

    def _bind(self, functional: FunctionalPopulation):
        """Resolve static plan-specific configuration for a built-in callback."""

        del functional
        return self


class Recorder(FunctionalCallback):
    """Record explicit functional state at the initial boundary and each step.

    The constructor mirrors the ordinary :class:`dendra.callbacks.Recorder`
    where the pure in-memory semantics overlap. ``dt`` and
    ``sliding_window`` are reserved but not yet supported because their phase
    must remain explicit across segmented and transformed executions.
    """

    def __init__(
        self,
        states,
        max_only=False,
        node_indices=None,
        dt=None,
        sliding_window=None,
        partition=None,
    ) -> None:
        if isinstance(states, str):
            states = (states,)
        else:
            states = tuple(states)
        if not states:
            raise ValueError("functional Recorder states must not be empty")
        if any(not isinstance(state, str) or not state for state in states):
            raise TypeError("functional Recorder states must be non-empty strings")
        if len(states) != len(set(states)):
            raise ValueError("functional Recorder state names must be unique")
        self.states = states
        self.max_only = bool(max_only)
        self.node_indices = node_indices
        self.dt = dt
        self.sliding_window = sliding_window
        self.partition = partition

    def _bind(self, functional: FunctionalPopulation):
        if self.dt is not None:
            raise FunctionalizationError(
                "dn.func.Recorder does not yet support dt; it records the "
                "initial boundary and every post-step value"
            )
        if self.sliding_window is not None:
            raise FunctionalizationError(
                "dn.func.Recorder does not yet support sliding_window; apply a "
                "pure tensor reduction to the finalized recording"
            )

        available = {}
        for leaf in functional._state_layout:
            if leaf.public_path[0] == "integrator":
                available.setdefault(leaf.public_path[-1], leaf.public_path)
            if leaf.public_path == ("clock", "t"):
                available.setdefault("t", leaf.public_path)
            if leaf.checkpoint_key is not None:
                available.setdefault(leaf.checkpoint_key, leaf.public_path)
        missing = [state for state in self.states if state not in available]
        if missing:
            raise FunctionalizationError(
                "dn.func.Recorder can only observe explicit transition state; "
                f"unsupported selection(s): {missing}"
            )
        paths = tuple(available[state] for state in self.states)
        selected_schemas = []
        for state_name, path in zip(self.states, paths, strict=True):
            schema = functional._state_schema
            for part in path:
                schema = schema[part]
            if (self.max_only or self.node_indices is not None) and not schema.shape:
                raise FunctionalizationError(
                    "dn.func.Recorder node selection and maximum reduction "
                    f"require a non-scalar state; {state_name!r} is scalar"
                )
            selected_schemas.append(schema)

        node_indices = None
        if not self.max_only and self.node_indices is not None:
            # A plan may be authored under inference_mode and later used with
            # autograd. Materialize an ordinary versioned index tensor.
            with torch.inference_mode(False):
                node_indices = (
                    torch.as_tensor(
                        self.node_indices,
                        device=functional.device,
                        dtype=torch.long,
                    )
                    .detach()
                    .clone()
                )
            if node_indices.ndim != 1:
                raise FunctionalizationError(
                    "dn.func.Recorder node_indices must be one-dimensional"
                )

        partition = None
        if self.max_only and self.partition is not None:
            partition = tuple(int(value) for value in self.partition)
            if not partition or any(value <= 0 for value in partition):
                raise FunctionalizationError(
                    "dn.func.Recorder partition entries must be positive"
                )
            for state_name, schema in zip(
                self.states,
                selected_schemas,
                strict=True,
            ):
                if sum(partition) != schema.shape[-1]:
                    raise FunctionalizationError(
                        "dn.func.Recorder partition must cover the complete "
                        f"final axis for {state_name!r}; got {sum(partition)} "
                        f"and {schema.shape[-1]}"
                    )

        return _BoundRecorder(
            states=self.states,
            paths=paths,
            max_only=self.max_only,
            node_indices=node_indices,
            partition=partition,
        )

    def initialize(self, state, auxiliary):  # pragma: no cover - guarded by bind
        del state, auxiliary
        raise FunctionalizationError(
            "dn.func.Recorder must be bound with functional.make_callbacks(...)"
        )

    def update(self, carry, state, auxiliary):  # pragma: no cover - guarded
        del carry, state, auxiliary
        raise FunctionalizationError(
            "dn.func.Recorder must be bound with functional.make_callbacks(...)"
        )

    def finalize(self, carry, emissions):  # pragma: no cover - guarded by bind
        del carry, emissions
        raise FunctionalizationError(
            "dn.func.Recorder must be bound with functional.make_callbacks(...)"
        )


@dataclass(frozen=True)
class _BoundRecorder(FunctionalCallback):
    states: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...]
    max_only: bool
    node_indices: torch.Tensor | None
    partition: tuple[int, ...] | None

    def _observe(self, state: Mapping[str, object]) -> dict[str, torch.Tensor]:
        output = {}
        for name, path in zip(self.states, self.paths, strict=True):
            value = _get_path(state, path)
            if not torch.is_tensor(value):  # pragma: no cover - plan owns schema
                raise FunctionalizationError(
                    f"functional Recorder state {name!r} is no longer a Tensor"
                )
            if self.max_only:
                if self.partition is None:
                    value = torch.amax(value, dim=-1, keepdim=True)
                else:
                    segments = torch.split(value, self.partition, dim=-1)
                    value = torch.stack(
                        [torch.amax(segment, dim=-1) for segment in segments],
                        dim=-1,
                    )
            elif self.node_indices is not None:
                value = torch.index_select(value, -1, self.node_indices)
            output[name] = value
        return output

    def initialize(self, state, auxiliary):
        del auxiliary
        observed = self._observe(state)
        return (), observed

    def update(self, carry, state, auxiliary):
        del auxiliary
        return carry, self._observe(state)

    def finalize(self, carry, emissions):
        del carry
        return emissions


class AnomalyDetector(FunctionalCallback):
    """Return a cumulative mask of non-finite post-step voltages.

    This mirrors the ordinary callback's public result semantics while keeping
    only a Boolean mask in carry; it emits no trajectory and therefore uses
    constant callback memory.
    """

    @staticmethod
    def _voltage(state):
        value = state["integrator"]["v"]
        if not torch.is_tensor(value):  # pragma: no cover - state is validated
            raise FunctionalizationError("functional voltage state is not a Tensor")
        return value

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = self._voltage(state)
        return torch.zeros_like(voltage[..., 0], dtype=torch.bool), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = self._voltage(state)
        return carry | ~torch.isfinite(voltage).all(dim=-1), None

    def finalize(self, carry, emissions):
        if emissions is not None:  # pragma: no cover - collection owns schema
            raise FunctionalizationError("AnomalyDetector does not emit samples")
        return carry


def _bind_threshold_configuration(
    functional: FunctionalPopulation,
    *,
    threshold,
    t_start_check,
    t_end_check,
    node_check,
    dt,
):
    """Resolve one threshold callback against an immutable functional plan."""

    voltage_schema = functional._state_schema["integrator"]["v"]
    if len(voltage_schema.shape) < 2:  # pragma: no cover - lowering owns shape
        raise FunctionalizationError(
            "functional threshold callbacks require voltage shaped (*batch, N, C)"
        )

    with torch.inference_mode(False):
        try:
            node_indices = (
                torch.as_tensor(
                    node_check,
                    device=functional.device,
                    dtype=torch.long,
                )
                .reshape(-1)
                .detach()
                .clone()
            )
        except (TypeError, ValueError) as exc:
            raise FunctionalizationError(
                "functional threshold callback node_check must be convertible "
                "to compartment indices"
            ) from exc

    compartments = voltage_schema.shape[-1]
    invalid = (node_indices < -compartments) | (node_indices >= compartments)
    if torch.any(invalid):
        invalid_values = node_indices[invalid].detach().cpu().tolist()
        raise IndexError(
            "node_check entries must be valid compartment indices in "
            f"[-{compartments}, {compartments - 1}]; got {invalid_values}"
        )
    node_indices = torch.where(
        node_indices < 0,
        node_indices + compartments,
        node_indices,
    )

    try:
        threshold = float(threshold)
        callback_dt = functional.dt if dt is None else float(dt)
        t_start_check = float(t_start_check)
        t_end_check = None if t_end_check is None else float(t_end_check)
    except (TypeError, ValueError) as exc:
        raise FunctionalizationError(
            "functional threshold, timing, and dt values must be real scalars"
        ) from exc
    if not math.isfinite(callback_dt) or not callback_dt > 0:
        raise FunctionalizationError(
            "functional threshold callback dt must be positive"
        )
    if not math.isfinite(t_start_check) or (
        t_end_check is not None and not math.isfinite(t_end_check)
    ):
        raise FunctionalizationError(
            "functional threshold callback time bounds must be finite"
        )
    if callback_dt != functional.dt:
        raise FunctionalizationError(
            "functional threshold callback dt must match the FunctionalPopulation "
            f"timestep ({functional.dt}); rebuild the functional plan for another dt"
        )

    start_step = int(t_start_check / callback_dt)
    end_step = (
        torch.iinfo(torch.int64).max
        if t_end_check is None
        else int(t_end_check / callback_dt)
    )
    return node_indices, threshold, start_step, end_step, callback_dt


class _ThresholdCallback(FunctionalCallback):
    """Static user-facing configuration shared by threshold built-ins."""

    _bound_type = None

    def __init__(
        self,
        threshold=0.0,
        t_start_check=0.0,
        t_end_check=None,
        node_check=(5, -5),
        dt=None,
    ) -> None:
        self.threshold = threshold
        self.t_start_check = t_start_check
        self.t_end_check = t_end_check
        self.node_check = node_check
        self.dt = dt

    def _bind(self, functional: FunctionalPopulation):
        configuration = _bind_threshold_configuration(
            functional,
            threshold=self.threshold,
            t_start_check=self.t_start_check,
            t_end_check=self.t_end_check,
            node_check=self.node_check,
            dt=self.dt,
        )
        return self._bound_type(*configuration)

    def initialize(self, state, auxiliary):  # pragma: no cover - guarded by bind
        del state, auxiliary
        raise FunctionalizationError(
            "functional threshold callbacks must be bound with "
            "functional.make_callbacks(...)"
        )

    def update(self, carry, state, auxiliary):  # pragma: no cover - guarded
        del carry, state, auxiliary
        raise FunctionalizationError(
            "functional threshold callbacks must be bound with "
            "functional.make_callbacks(...)"
        )

    def finalize(self, carry, emissions):  # pragma: no cover - guarded by bind
        del carry, emissions
        raise FunctionalizationError(
            "functional threshold callbacks must be bound with "
            "functional.make_callbacks(...)"
        )


def _selected_voltage(state, node_indices):
    return torch.index_select(
        state["integrator"]["v"],
        -1,
        node_indices,
    )


def _threshold_crossing(
    state,
    node_indices,
    threshold,
    below,
    step,
    start_step,
    end_step,
):
    selected = _selected_voltage(state, node_indices)
    above = selected >= threshold
    within_window = (step >= start_step) & (step < end_step)
    spikes = above & below & within_window
    next_below = torch.where(within_window, ~above, below)
    return next_below, spikes


@dataclass(frozen=True)
class _BoundRaster(FunctionalCallback):
    node_indices: torch.Tensor
    threshold: float
    start_step: int
    end_step: int
    callback_dt: float

    def initialize(self, state, auxiliary):
        del auxiliary
        selected = _selected_voltage(state, self.node_indices)
        carry = (
            torch.ones_like(selected, dtype=torch.bool),
            selected.new_zeros((), dtype=torch.int64),
        )
        return carry, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        below, step = carry
        next_below, spikes = _threshold_crossing(
            state,
            self.node_indices,
            self.threshold,
            below,
            step,
            self.start_step,
            self.end_step,
        )
        return (next_below, step + 1), spikes

    def finalize(self, carry, emissions):
        below, _step = carry
        if emissions is None:
            return below.new_empty((0, *below.shape))
        return emissions


class Raster(_ThresholdCallback):
    """Record post-step upward threshold crossings as a Boolean trajectory.

    The constructor mirrors :class:`dendra.callbacks.Raster`. The result has
    shape ``(steps, *batch, N, K)`` for ``K`` selected compartments. Unlike the
    mutable callback, the functional result retains every simulated step and
    masks events outside ``[t_start_check, t_end_check)`` to preserve a static,
    transform-friendly time axis.
    """

    _bound_type = _BoundRaster


@dataclass(frozen=True)
class _BoundAPCount(FunctionalCallback):
    node_indices: torch.Tensor
    threshold: float
    start_step: int
    end_step: int
    callback_dt: float

    def initialize(self, state, auxiliary):
        del auxiliary
        selected = _selected_voltage(state, self.node_indices)
        carry = (
            torch.ones_like(selected, dtype=torch.bool),
            torch.zeros_like(selected, dtype=torch.int64),
            selected.new_zeros((), dtype=torch.int64),
        )
        return carry, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        below, count, step = carry
        next_below, spikes = _threshold_crossing(
            state,
            self.node_indices,
            self.threshold,
            below,
            step,
            self.start_step,
            self.end_step,
        )
        return (next_below, count + spikes.to(count.dtype), step + 1), None

    def finalize(self, carry, emissions):
        if emissions is not None:  # pragma: no cover - collection owns schema
            raise FunctionalizationError("APCount does not emit samples")
        _below, count, _step = carry
        return count


class APCount(_ThresholdCallback):
    """Count upward voltage-threshold crossings in constant callback memory.

    The returned integer tensor has shape ``(*batch, N, K)`` for ``K``
    selected compartments. Crossing state, counts, and timestep phase all live
    in explicit carry, so segmented and checkpointed execution resume exactly.
    """

    _bound_type = _BoundAPCount


@dataclass(frozen=True)
class FunctionalCallbackResults(Mapping[str, object]):
    """Immutable finalized result envelope for one callback execution segment.

    Functional runners place it under ``auxiliary["callbacks"]``; the
    imperative checkpoint bridge returns it directly. ``state`` is the
    explicit callback carry to pass when continuing. Mapping access forwards
    to finalized ``outputs``.
    """

    state: FunctionalCallbackState
    outputs: Mapping[str, object]

    def __getitem__(self, key: str):
        return self.outputs[key]

    def __iter__(self):
        return iter(self.outputs)

    def __len__(self):
        return len(self.outputs)


@dataclass(frozen=True)
class FunctionalCallbackState(Mapping[str, object]):
    """Plan-tagged explicit carry for one functional callback collection.

    This is an output/carry type, not a user-construction API. Pass the state
    returned by Dendra so its static plan identity remains attached to the
    dynamic tensor PyTrees under ``torch.func`` and compilation.
    """

    carries: Mapping[str, object]
    _token: object
    _emission_schemas: tuple[_TensorTreeSchema | None, ...]

    @property
    def plan_token(self) -> object:
        return self._token

    @property
    def emission_schemas(self):
        return self._emission_schemas

    def __getitem__(self, key: str):
        return self.carries[key]

    def __iter__(self):
        return iter(self.carries)

    def __len__(self):
        return len(self.carries)


def _flatten_callback_state(
    state: FunctionalCallbackState,
) -> tuple[list[object], object]:
    return [state.carries], (state.plan_token, state.emission_schemas)


def _unflatten_callback_state(
    children: Sequence[object],
    context: object,
) -> FunctionalCallbackState:
    (carries,) = children
    plan_token, emission_schemas = context
    return FunctionalCallbackState(
        carries=carries,
        _token=plan_token,
        _emission_schemas=emission_schemas,
    )


def _flatten_callback_results(
    results: FunctionalCallbackResults,
) -> tuple[list[object], object]:
    return [results.state, results.outputs], None


def _unflatten_callback_results(
    children: Sequence[object],
    _context: object,
) -> FunctionalCallbackResults:
    state, outputs = children
    return FunctionalCallbackResults(state=state, outputs=outputs)


torch.utils._pytree.register_pytree_node(
    FunctionalCallbackState,
    _flatten_callback_state,
    _unflatten_callback_state,
)
torch.utils._pytree.register_pytree_node(
    FunctionalCallbackResults,
    _flatten_callback_results,
    _unflatten_callback_results,
)


@dataclass(frozen=True)
class _TensorLeafSchema:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _TensorTreeSchema:
    spec: object
    leaves: tuple[_TensorLeafSchema, ...]


def _materialize_inference_tree(value):
    """Make callback carry safe to resume outside inference mode."""

    if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
        return value
    leaves, spec = torch.utils._pytree.tree_flatten(value)
    if not any(torch.is_tensor(leaf) and torch.is_inference(leaf) for leaf in leaves):
        return value
    with torch.inference_mode(False):
        materialized = [
            leaf.detach().clone()
            if torch.is_tensor(leaf) and torch.is_inference(leaf)
            else leaf
            for leaf in leaves
        ]
    return torch.utils._pytree.tree_unflatten(materialized, spec)


def _tree_schema(value, *, label: str, allow_empty: bool) -> _TensorTreeSchema:
    leaves, spec = torch.utils._pytree.tree_flatten(value)
    if not leaves and not allow_empty:
        raise FunctionalizationError(f"{label} must contain at least one Tensor")
    schemas = []
    for index, leaf in enumerate(leaves):
        if not torch.is_tensor(leaf):
            raise FunctionalizationError(
                f"{label} leaf {index} must be a Tensor, got {type(leaf).__name__}"
            )
        if (
            not torch.compiler.is_compiling()
            and not torch._C._are_functorch_transforms_active()
            and torch.is_grad_enabled()
            and torch.is_inference(leaf)
        ):
            raise FunctionalizationError(
                f"{label} leaf {index} is an inference Tensor and cannot be "
                "resumed through autograd; return ordinary tensor carry"
            )
        schemas.append(
            _TensorLeafSchema(
                shape=tuple(leaf.shape),
                dtype=leaf.dtype,
                device=leaf.device,
            )
        )
    return _TensorTreeSchema(spec=spec, leaves=tuple(schemas))


def _validate_tree(value, schema: _TensorTreeSchema, *, label: str) -> None:
    leaves, spec = torch.utils._pytree.tree_flatten(value)
    if spec != schema.spec:
        raise FunctionalizationError(f"{label} changed its tensor PyTree structure")
    for index, (leaf, expected) in enumerate(zip(leaves, schema.leaves, strict=True)):
        if not torch.is_tensor(leaf):
            raise FunctionalizationError(
                f"{label} leaf {index} must be a Tensor, got {type(leaf).__name__}"
            )
        if (
            not torch.compiler.is_compiling()
            and not torch._C._are_functorch_transforms_active()
            and torch.is_grad_enabled()
            and torch.is_inference(leaf)
        ):
            raise FunctionalizationError(
                f"{label} leaf {index} is an inference Tensor and cannot be "
                "used through autograd; return ordinary tensor carry"
            )
        if tuple(leaf.shape) != expected.shape:
            raise FunctionalizationError(
                f"{label} leaf {index} changed shape from {expected.shape} to "
                f"{tuple(leaf.shape)}"
            )
        if leaf.dtype != expected.dtype or leaf.device != expected.device:
            raise FunctionalizationError(
                f"{label} leaf {index} must use "
                f"{expected.device}/{expected.dtype}, got "
                f"{leaf.device}/{leaf.dtype}"
            )


def _empty_emissions(schema: _TensorTreeSchema):
    return torch.utils._pytree.tree_unflatten(
        [
            torch.empty(
                (0, *leaf.shape),
                dtype=leaf.dtype,
                device=leaf.device,
            )
            for leaf in schema.leaves
        ],
        schema.spec,
    )


def _stack_emissions(values, schema, *, label: str):
    if not values:
        return None if schema is None else _empty_emissions(schema)
    if schema is None:
        schema = _tree_schema(values[0], label=label, allow_empty=False)
    for value in values:
        _validate_tree(value, schema, label=label)
    flattened = [torch.utils._pytree.tree_flatten(value)[0] for value in values]
    stacked = [
        torch.stack([leaves[index] for leaves in flattened], dim=0)
        for index in range(len(schema.leaves))
    ]
    return torch.utils._pytree.tree_unflatten(stacked, schema.spec)


def _concatenate_emissions(parts, schema, *, label: str):
    if not parts:
        return None if schema is None else _empty_emissions(schema)
    parts = [part for part in parts if part is not None]
    if not parts:
        return None if schema is None else _empty_emissions(schema)
    if schema is None:
        first_leaves, spec = torch.utils._pytree.tree_flatten(parts[0])
        if not first_leaves:
            raise FunctionalizationError(f"{label} must contain at least one Tensor")
        expected_leaves = tuple(
            _TensorLeafSchema(
                shape=tuple(leaf.shape),
                dtype=leaf.dtype,
                device=leaf.device,
            )
            for leaf in first_leaves
        )
    else:
        spec = schema.spec
        expected_leaves = tuple(
            _TensorLeafSchema(
                shape=(0, *expected.shape),
                dtype=expected.dtype,
                device=expected.device,
            )
            for expected in schema.leaves
        )
    flattened = []
    for part in parts:
        leaves, part_spec = torch.utils._pytree.tree_flatten(part)
        if part_spec != spec:
            raise FunctionalizationError(
                f"{label} changed its stacked tensor PyTree structure"
            )
        for index, (leaf, expected) in enumerate(
            zip(leaves, expected_leaves, strict=True)
        ):
            if not torch.is_tensor(leaf):
                raise FunctionalizationError(f"{label} leaf {index} must be a Tensor")
            if (
                leaf.dtype != expected.dtype
                or leaf.device != expected.device
                or leaf.ndim != len(expected.shape)
                or tuple(leaf.shape[1:]) != expected.shape[1:]
            ):
                raise FunctionalizationError(
                    f"{label} changed a stacked leaf's dtype, device, rank, "
                    "or trailing shape"
                )
        flattened.append(leaves)
    concatenated = [
        torch.cat([leaves[index] for leaves in flattened], dim=0)
        for index in range(len(expected_leaves))
    ]
    return torch.utils._pytree.tree_unflatten(concatenated, spec)


@dataclass(frozen=True)
class _CallbackPlan:
    name: str
    callback: FunctionalCallback
    carry_schema: _TensorTreeSchema
    emission_schema: _TensorTreeSchema | None
    result_schema: _TensorTreeSchema
    initial_emits: bool
    validate_updates: bool


def _update_callback_values(plans, carries, state, auxiliary, emission_schemas=None):
    """Advance explicit tensor carry without the host's plan-token envelope."""
    updated = {}
    emissions = {}
    for index, plan in enumerate(plans):
        carry, emission = FunctionalCallbacks._pair(
            plan.callback.update(carries[plan.name], state, auxiliary),
            label=f"functional callback {plan.name!r}.update()",
        )
        schema = (
            plan.emission_schema
            if emission_schemas is None
            else emission_schemas[index]
        )
        if plan.validate_updates:
            _validate_tree(
                carry,
                plan.carry_schema,
                label=f"functional callback {plan.name!r} carry",
            )
            if emission is not None and schema is not None:
                _validate_tree(
                    emission,
                    schema,
                    label=f"functional callback {plan.name!r} emission",
                )
        updated[plan.name] = carry
        if emission is not None:
            emissions[plan.name] = emission
    return updated, emissions or None


def _stack_callback_emissions(plans, emissions):
    """Stack each callback's per-step tensor emissions inside a fixed chunk."""
    return {
        plan.name: _stack_emissions(
            [
                emission[plan.name]
                for emission in emissions
                if emission is not None and plan.name in emission
            ],
            plan.emission_schema,
            label=f"functional callback {plan.name!r} emission",
        )
        for plan in plans
    }


@dataclass(frozen=True)
class _ImperativeCallbackBinding:
    callbacks: FunctionalCallbacks
    readers: tuple[tuple[tuple[str, ...], torch.nn.Module, str], ...]

    def state(self):
        state = {}
        for public_path, owner, name in self.readers:
            value = (
                owner._buffers[name] if name in owner._buffers else getattr(owner, name)
            )
            _put_path(state, public_path, value)
        return state


class FunctionalCallbacks:
    """An immutable named collection of native functional callbacks.

    Construct a collection with :meth:`FunctionalPopulation.make_callbacks`.
    Callback objects supply static configuration only and are never mutated.
    """

    def __init__(
        self,
        functional: FunctionalPopulation,
        callbacks: Mapping[str, FunctionalCallback],
    ) -> None:
        if not isinstance(callbacks, Mapping):
            raise TypeError(
                "callbacks must be a mapping of names to FunctionalCallback objects"
            )
        if not callbacks:
            raise ValueError("callbacks must contain at least one named callback")

        self._functional = functional
        self._state_token = object()
        with torch.inference_mode(False):
            example_state = functional.extract().state
        plans = []
        for name, callback in callbacks.items():
            if not isinstance(name, str) or not name:
                raise ValueError("functional callback names must be non-empty strings")
            callback = self._coerce_callback(callback)
            bound = callback._bind(functional)
            if not isinstance(bound, FunctionalCallback):  # pragma: no cover
                raise FunctionalizationError(
                    f"functional callback {name!r} returned an invalid bound object"
                )
            plans.append(self._probe(name, bound, example_state))
        self._plans = tuple(plans)

    @staticmethod
    def _coerce_callback(callback) -> FunctionalCallback:
        if not isinstance(callback, FunctionalCallback):
            raise TypeError(
                "functional callback values must be FunctionalCallback objects"
            )
        return callback

    @staticmethod
    def _pair(value, *, label: str):
        if not isinstance(value, tuple) or len(value) != 2:
            raise FunctionalizationError(
                f"{label} must return exactly (carry, emission_or_none)"
            )
        return value

    @classmethod
    def _probe(cls, name, callback, state) -> _CallbackPlan:
        label = f"functional callback {name!r}"
        with torch.inference_mode(False), torch.no_grad():
            initial_carry, initial_emission = cls._pair(
                callback.initialize(state, None),
                label=f"{label}.initialize()",
            )
            initial_carry = _materialize_inference_tree(initial_carry)
            carry_schema = _tree_schema(
                initial_carry,
                label=f"{label} carry",
                allow_empty=True,
            )
            initial_emits = initial_emission is not None
            emission_schema = None
            if initial_emission is not None:
                emission_schema = _tree_schema(
                    initial_emission,
                    label=f"{label} emission",
                    allow_empty=False,
                )
                if initial_emits:
                    _validate_tree(
                        initial_emission,
                        emission_schema,
                        label=f"{label} initial emission",
                    )

            empty = (
                None if emission_schema is None else _empty_emissions(emission_schema)
            )
            empty_result = callback.finalize(initial_carry, empty)
            result_schema = _tree_schema(
                empty_result,
                label=f"{label} finalized result",
                allow_empty=False,
            )

        return _CallbackPlan(
            name=name,
            callback=callback,
            carry_schema=carry_schema,
            emission_schema=emission_schema,
            result_schema=result_schema,
            initial_emits=initial_emits,
            validate_updates=type(callback)
            not in {
                _BoundRecorder,
                AnomalyDetector,
                _BoundRaster,
                _BoundAPCount,
            },
        )

    def _validate_owner(self, functional: FunctionalPopulation) -> None:
        if functional is not self._functional:
            raise FunctionalizationError(
                "callbacks belong to a different FunctionalPopulation plan"
            )

    def _validate_runtime_dt(self, dt: float) -> None:
        for plan in self._plans:
            callback = plan.callback
            if isinstance(callback, (_BoundRaster, _BoundAPCount)) and (
                dt != callback.callback_dt
            ):
                raise FunctionalizationError(
                    "functional threshold callback dt does not match the "
                    f"runtime timestep: expected {callback.callback_dt}, got {dt}"
                )

    def _validate_runtime_dt_tensor(self, dt: torch.Tensor) -> None:
        for plan in self._plans:
            callback = plan.callback
            if not isinstance(callback, (_BoundRaster, _BoundAPCount)):
                continue
            expected = dt.new_tensor(callback.callback_dt)
            message = (
                "functional threshold callback dt does not match the prepared "
                f"timestep {callback.callback_dt}"
            )
            if (
                torch.compiler.is_compiling()
                or torch._C._are_functorch_transforms_active()
            ):
                torch._assert_async(dt == expected, message)
            elif not torch.equal(dt, expected):
                raise FunctionalizationError(message)

    def _bind_imperative(self, population):
        if population is not self._functional._population:
            raise FunctionalizationError(
                "functional callbacks must be bound to the exact source "
                "Population used by make_functional()"
            )
        self._functional._validate_source()
        readers = []
        for leaf in self._functional._state_layout:
            owner = (
                population
                if not leaf.module_path
                else population.get_submodule(leaf.module_path)
            )
            if leaf.buffer_name not in owner._buffers and not hasattr(
                owner,
                leaf.buffer_name,
            ):
                raise FunctionalizationError(
                    "functional callback state binding lost runtime tensor "
                    f"{leaf.module_path}.{leaf.buffer_name}"
                )
            readers.append((leaf.public_path, owner, leaf.buffer_name))
        return _ImperativeCallbackBinding(self, tuple(readers))

    def _initialize(self, state):
        carries = {}
        emissions = {}
        for plan in self._plans:
            carry, emission = self._pair(
                plan.callback.initialize(state, None),
                label=f"functional callback {plan.name!r}.initialize()",
            )
            carry = _materialize_inference_tree(carry)
            _validate_tree(
                carry,
                plan.carry_schema,
                label=f"functional callback {plan.name!r} initial carry",
            )
            if (emission is not None) is not plan.initial_emits:
                raise FunctionalizationError(
                    f"functional callback {plan.name!r}.initialize() changed "
                    "whether it emits"
                )
            carries[plan.name] = carry
            if emission is not None:
                emissions[plan.name] = emission
        callback_state = FunctionalCallbackState(
            carries=carries,
            _token=self._state_token,
            _emission_schemas=tuple(plan.emission_schema for plan in self._plans),
        )
        return callback_state, emissions or None

    def _validate_state(self, state) -> FunctionalCallbackState:
        if not isinstance(state, FunctionalCallbackState):
            raise TypeError(
                "callback_state must be the FunctionalCallbackState returned "
                "by a callback-enabled runner"
            )
        if state.plan_token is not self._state_token:
            raise FunctionalizationError(
                "callback_state belongs to a different callback plan"
            )
        expected = {plan.name for plan in self._plans}
        if set(state) != expected:
            raise ValueError(
                "callback_state keys do not match the callback plan: expected "
                f"{sorted(expected)}, got {sorted(state)}"
            )
        if len(state.emission_schemas) != len(self._plans):
            raise ValueError(
                "callback_state emission schema count does not match the plan"
            )
        for plan in self._plans:
            _validate_tree(
                state[plan.name],
                plan.carry_schema,
                label=f"callback_state[{plan.name!r}]",
            )
        for plan, runtime_schema in zip(
            self._plans,
            state.emission_schemas,
            strict=True,
        ):
            if plan.emission_schema is not None:
                if runtime_schema != plan.emission_schema:
                    raise FunctionalizationError(
                        f"callback_state[{plan.name!r}] changed its bound "
                        "emission schema"
                    )
            elif runtime_schema is not None and not isinstance(
                runtime_schema,
                _TensorTreeSchema,
            ):
                raise TypeError(
                    f"callback_state[{plan.name!r}] has an invalid emission schema"
                )
        return state

    def _update(self, callback_state, state, auxiliary):
        carries, emissions = _update_callback_values(
            self._plans,
            callback_state.carries,
            state,
            auxiliary,
            callback_state.emission_schemas,
        )
        next_state = FunctionalCallbackState(
            carries=carries,
            _token=self._state_token,
            _emission_schemas=callback_state.emission_schemas,
        )
        return next_state, emissions

    def _stack(self, emissions):
        return _stack_callback_emissions(self._plans, emissions)

    def _wrap_compiled_update(self, previous, carries, emissions):
        """Restore host metadata and validate schemas discovered on earlier runs.

        Compiled kernels consume plain tensor carry, so discovering the first
        post-step emission does not change their input metadata on resume.
        Per-step carry/bound emission checks and within-chunk emission checks
        remain in the lowered transition. Previously discovered emission
        schemas are checked here before accepting a compiled chunk's output.
        """
        state = FunctionalCallbackState(
            carries=carries,
            _token=self._state_token,
            _emission_schemas=previous.emission_schemas,
        )
        self._validate_state(state)
        for plan, schema in zip(self._plans, previous.emission_schemas, strict=True):
            part = emissions[plan.name]
            if schema is None or part is None:
                continue
            leaves, spec = torch.utils._pytree.tree_flatten(part)
            if spec != schema.spec:
                raise FunctionalizationError(
                    f"functional callback {plan.name!r} emission changed its "
                    "tensor PyTree structure"
                )
            for leaf, expected in zip(leaves, schema.leaves, strict=True):
                if (
                    not torch.is_tensor(leaf)
                    or leaf.ndim != len(expected.shape) + 1
                    or tuple(leaf.shape[1:]) != expected.shape
                    or leaf.dtype != expected.dtype
                    or leaf.device != expected.device
                ):
                    raise FunctionalizationError(
                        f"functional callback {plan.name!r} emission changed a "
                        "leaf's shape, dtype, or device"
                    )
        return state

    def _concatenate(self, parts):
        return {
            plan.name: _concatenate_emissions(
                [part[plan.name] for part in parts],
                plan.emission_schema,
                label=f"functional callback {plan.name!r} emission",
            )
            for plan in self._plans
        }

    @staticmethod
    def _has_emissions(part) -> bool:
        return any(value is not None for value in part.values())

    def _materialize_state(self, callback_state):
        carries = _materialize_inference_tree(callback_state.carries)
        if carries is callback_state.carries:
            return callback_state
        return FunctionalCallbackState(
            carries=carries,
            _token=self._state_token,
            _emission_schemas=callback_state.emission_schemas,
        )

    def _capture_emission_schemas(self, callback_state, emissions):
        schemas = list(callback_state.emission_schemas)
        changed = False
        for index, plan in enumerate(self._plans):
            value = emissions[plan.name]
            if schemas[index] is not None or value is None:
                continue
            leaves, spec = torch.utils._pytree.tree_flatten(value)
            if not leaves:
                raise FunctionalizationError(
                    f"functional callback {plan.name!r} emission must contain "
                    "at least one Tensor"
                )
            leaf_schemas = []
            for leaf_index, leaf in enumerate(leaves):
                if not torch.is_tensor(leaf) or leaf.ndim < 1:
                    raise FunctionalizationError(
                        f"functional callback {plan.name!r} stacked emission "
                        f"leaf {leaf_index} must be a Tensor with a sample axis"
                    )
                leaf_schemas.append(
                    _TensorLeafSchema(
                        shape=tuple(leaf.shape[1:]),
                        dtype=leaf.dtype,
                        device=leaf.device,
                    )
                )
            schemas[index] = _TensorTreeSchema(
                spec=spec,
                leaves=tuple(leaf_schemas),
            )
            changed = True
        if not changed:
            return callback_state
        return FunctionalCallbackState(
            carries=callback_state.carries,
            _token=self._state_token,
            _emission_schemas=tuple(schemas),
        )

    def _finalize(self, callback_state, emissions):
        self._validate_state(callback_state)
        outputs = {
            plan.name: plan.callback.finalize(
                callback_state[plan.name],
                emissions[plan.name],
            )
            for plan in self._plans
        }
        for plan in self._plans:
            leaves, spec = torch.utils._pytree.tree_flatten(outputs[plan.name])
            if spec != plan.result_schema.spec:
                raise FunctionalizationError(
                    f"functional callback {plan.name!r}.finalize() changed its "
                    "tensor PyTree structure"
                )
            for index, (leaf, expected) in enumerate(
                zip(leaves, plan.result_schema.leaves, strict=True)
            ):
                if not torch.is_tensor(leaf):
                    raise FunctionalizationError(
                        f"functional callback {plan.name!r} finalized result "
                        f"leaf {index} must be a Tensor"
                    )
                shape = tuple(leaf.shape)
                if emissions[plan.name] is None:
                    shape_matches = shape == expected.shape
                else:
                    shape_matches = (
                        leaf.ndim == len(expected.shape)
                        and shape[1:] == expected.shape[1:]
                    )
                if (
                    leaf.dtype != expected.dtype
                    or leaf.device != expected.device
                    or not shape_matches
                ):
                    raise FunctionalizationError(
                        f"functional callback {plan.name!r}.finalize() changed a "
                        "result leaf's dtype, device, rank, or trailing shape"
                    )
        return outputs


__all__ = [
    "APCount",
    "AnomalyDetector",
    "FunctionalCallback",
    "FunctionalCallbackResults",
    "FunctionalCallbackState",
    "FunctionalCallbacks",
    "Raster",
    "Recorder",
]
