"""Explicit projections over ContinuousCon's shared analog transport phases."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import torch

from dendra.models.core import Population
from dendra.models.networks._connection_initialization import continuous_initial_state
from dendra.models.networks._continuous_cache import (
    ContinuousCache,
    _has_empty_cache_vmap_axis,
    continuous_cache_routes,
    normalize_continuous_cache,
    restore_continuous_cache,
    validate_continuous_cache,
)
from dendra.models.networks._continuous_delivery import (
    continuous_clear,
    continuous_counters,
    continuous_due,
    continuous_enqueue,
    continuous_project,
    gather_continuous_source,
)
from dendra.models.networks.netcon import ContinuousCon
from dendra.models.networks.utils import make_getattr

from ._continuous_transform import ContinuousTransformPlan
from ._netcon import _effective_tensor, _snapshot, _source_tensor
from ._types import FunctionalizationError


class ContinuousConTensors(NamedTuple):
    """Independent effective weights, runtime carry, and transform parameters."""

    weights: torch.Tensor
    state: dict[str, torch.Tensor]
    transform_parameters: dict[str, torch.Tensor] | None = None


_SOURCE_METHODS = {
    name: getattr(ContinuousCon, name)
    for name in ("_pre_value", "_weighted_pre_value", "_deliver")
}
_INITIALIZATION_METHODS = {
    name: getattr(ContinuousCon, name)
    for name in ("initialize", "zero", "detach", "_configure_advance_impl")
}
_CACHE_METHODS = {
    name: getattr(ContinuousCon, name)
    for name in (
        "initialize_from_state_cache",
        "_materialize_delivery_from_pre_value_history",
    )
}


def _eligible(con):
    kind = type(con)
    proxy = (
        kind.__module__ == "dendra.models.parametric"
        and kind.__bases__ == (ContinuousCon,)
        and set(vars(kind)) <= {"t", "__module__", "__doc__"}
    )
    if kind is not ContinuousCon and not proxy:
        raise FunctionalizationError("Expected a canonical ContinuousCon.")
    if not getattr(con, "_continuous_delivery_initialized", False):
        raise FunctionalizationError("Initialize ContinuousCon before lowering it.")
    if not isinstance(con.pre, Population):
        raise FunctionalizationError("ContinuousCon requires a Population source.")
    allowed = (
        ContinuousCon._advance_empty,
        ContinuousCon._advance_all_immediate,
        ContinuousCon._advance_all_delayed_uniform,
        ContinuousCon._advance_all_delayed_mixed,
        ContinuousCon._advance_mixed_uniform,
        ContinuousCon._advance_mixed,
    )
    if (
        getattr(con.advance, "__self__", None) is not con
        or getattr(con.advance, "__func__", None) not in allowed
    ):
        raise FunctionalizationError(
            "ContinuousCon needs its canonical advance adapter."
        )
    for name, expected in _SOURCE_METHODS.items():
        method = getattr(con, name)
        if getattr(method, "__func__", None) is not expected:
            raise FunctionalizationError(
                "ContinuousCon needs canonical source/delivery hooks."
            )
    getter = con.get_pre_var
    expected = make_getattr(con.pre_var)
    if (
        not hasattr(getter, "__code__")
        or getter.__code__.co_code != expected.__code__.co_code
        or getter.__code__.co_names != expected.__code__.co_names
        or getter.__code__.co_consts != expected.__code__.co_consts
        or getter.__closure__
        or getter.__defaults__
    ):
        raise FunctionalizationError(
            "ContinuousCon needs its canonical stored-Tensor getter."
        )


class FunctionalContinuousCon:
    """Frozen zero/integer-delay routing with explicit values and presence masks.

    ``advance(weights, state, source_values)`` returns ``(next_state, payload,
    mask)``. Source values have the complete presynaptic stored tensor shape;
    weights have one target-dtype value per edge. Transform values are explicit
    in ``transform_parameters`` using ordinary module parameter/buffer names.
    Stateless functions and modules use their same native callable; static
    Python configuration is frozen, while explicit tensor values may change.

    Native observation order is represented by ``receive_before_source``. When
    true, call ``read_due(state)``, deliver that payload/mask, then observe the
    source and call ``advance``; do not deliver its returned payload a second
    time. Otherwise observe the source first, call ``advance``, and deliver the
    combined immediate/due payload once. Empty plans have ``has_connections``
    false and do not call the receiver. Reset each target once before visiting
    its projections in native order. No method advances a Population clock.

    Delay quantization, capacity, routing, input, reduction, and source schemas
    are fixed. Native reinitialization or configuration changes require lowering
    again. Runtime queue snapshots preserve enqueue-time values and gradients;
    extraction creates independent detached carry. ``commit_state_`` transfers
    runtime carry only; align weights/transform parameters separately before
    native continuation. Detached steady-state caches remain a separate API.
    """

    def __init__(self, con):
        _eligible(con)
        source = _source_tensor(con)
        self._source = con
        self._source_type = type(con)
        self._pre, self._post, self._syn = con.pre, con.post, con.syn
        self._generation = con._continuous_delivery_generation
        self._advance = con.advance.__func__
        self._training = bool(con.training)
        self._pre_var = con.pre_var
        self._getter = con.get_pre_var
        self.source_shape = tuple(source.shape)
        self.source_device = torch.device(con.pre_device)
        self.source_dtype = con.pre_dtype
        self.shape = tuple(con.syn.shape_f)
        self.device, self.dtype = torch.device(con.device), con.dtype
        self.input, self.reduce = con.input, con.reduce
        self.has_connections = bool(con._has_connections)
        self.receive_before_source = bool(con._all_nonzero_delay)
        self._all_immediate = bool(con._all_zero_delay)
        self._depth, self._width = int(con.max_delay_steps), math.prod(self.shape)
        self._connections = int(con._n_conn)
        self._pre_idx, self._post_idx = _snapshot(con.pre_idx), _snapshot(con.post_idx)
        self._delay_steps, self._dt = _snapshot(con.delay_steps), _snapshot(con.dt)
        self._cache_dt = float(con.dt.detach().cpu())
        self._cache_routes = continuous_cache_routes(
            self._delay_steps, self._post_idx, depth=self._depth, width=self._width
        )
        self._zero_idx = _snapshot(con.zero_con_idx)
        self._nonzero_idx = _snapshot(con.nonzero_con_idx)
        self._post_nonzero = _snapshot(con.post_idx_nz)
        self._delay_nonzero = _snapshot(con.delay_steps_nz)
        self._post_zero = _snapshot(con.post_idx_zero)
        self._zero_mask = _snapshot(con.post_zero_delivery_mask)
        self._post_mask = _snapshot(con.post_delivery_mask)
        self._transform = con.transform
        self._transform_plan = ContinuousTransformPlan(
            con.transform, size=self._connections, device=self.device, dtype=self.dtype
        )
        checkpoint = con.state_dict_for_checkpoint()
        self._state_schema = {
            name: (tuple(value.shape), value.dtype, value.device)
            for name, value in checkpoint.items()
        }
        self._validate_compatible(con)

    @staticmethod
    def _validate_tensor(value, *, shape, dtype, device, label):
        if not torch.is_tensor(value):
            raise TypeError(f"{label} must be a Tensor")
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != device
        ):
            raise ValueError(f"{label} must match {device}/{dtype}/{shape}")

    def _validate_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self._state_schema):
            raise ValueError("ContinuousCon state must match its checkpoint schema")
        for name, (shape, dtype, device) in self._state_schema.items():
            self._validate_tensor(
                state[name],
                shape=shape,
                dtype=dtype,
                device=device,
                label=f"state.{name}",
            )

    def _validate_source(self):
        con = self._source
        if (
            type(con) is not self._source_type
            or not con._continuous_delivery_initialized
            or con._continuous_delivery_generation != self._generation
            or con.pre is not self._pre
            or con.post is not self._post
            or con.syn is not self._syn
            or con.pre_var != self._pre_var
            or con.get_pre_var is not self._getter
            or con.input != self.input
            or con.reduce != self.reduce
            or con.device != self.device
            or con.dtype != self.dtype
            or con.pre_device != self.source_device
            or con.pre_dtype != self.source_dtype
            or bool(con.training) != self._training
            or con.max_delay_steps != self._depth
            or tuple(con.syn.shape_f) != self.shape
            or con._n_conn != self._connections
            or getattr(con.advance, "__func__", None) is not self._advance
            or con.transform is not self._transform
        ):
            raise FunctionalizationError(
                "The source ContinuousCon configuration changed; initialize and lower again."
            )
        for name, expected in _SOURCE_METHODS.items():
            if name in con.__dict__ or getattr(type(con), name) is not expected:
                raise FunctionalizationError(
                    "ContinuousCon source/delivery hooks changed."
                )

        self._transform_plan.validate_source(con.transform)

    def _transform_parameters(self, con):
        return self._transform_plan.extract(con.transform)

    def _validate_transform_parameters(self, parameters):
        self._transform_plan.validate(parameters)

    def _validate_compatible(self, con):
        _eligible(con)
        source = _source_tensor(con)
        if (
            bool(con.training) != self._training
            or tuple(source.shape) != self.source_shape
            or con.pre_var != self._pre_var
            or tuple(con.syn.shape_f) != self.shape
            or con.device != self.device
            or con.dtype != self.dtype
            or con.pre_device != self.source_device
            or con.pre_dtype != self.source_dtype
            or con.input != self.input
            or con.reduce != self.reduce
            or con.max_delay_steps != self._depth
            or con._n_conn != self._connections
            or not torch.equal(con.pre_idx, self._pre_idx)
            or not torch.equal(con.post_idx, self._post_idx)
            or not torch.equal(con.delay_steps, self._delay_steps)
            or not torch.equal(con.dt, self._dt)
            or not torch.equal(
                (_effective_tensor(con.delay_ms, "delay") / con.dt).round().long(),
                self._delay_steps,
            )
        ):
            raise FunctionalizationError(
                "ContinuousCon topology does not match this plan."
            )
        self._validate_tensor(
            _effective_tensor(con.weight, "weight"),
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="effective weights",
        )
        self._validate_tensor(
            _effective_tensor(con.delay_ms, "delay"),
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="effective delays",
        )
        self._transform_plan.validate_compatible(con.transform)
        self._validate_state(con.state_dict_for_checkpoint())
        if (
            self._depth < 1
            or bool((self._delay_steps < 0).any())
            or bool(((self._pre_idx < 0) | (self._pre_idx >= source.numel())).any())
            or bool(((self._post_idx < 0) | (self._post_idx >= self._width)).any())
        ):
            raise FunctionalizationError("ContinuousCon has invalid routing or delays.")

    def extract(self, con=None):
        """Extract independent parameter leaves and detached runtime carry."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "Extract ContinuousCon tensors outside torch.compile."
            )
        con = self._source if con is None else con
        self._validate_compatible(con)
        return ContinuousConTensors(
            _snapshot(_effective_tensor(con.weight, "weight"), parameter=True),
            {
                name: _snapshot(value)
                for name, value in con.state_dict_for_checkpoint().items()
            },
            self._transform_parameters(con),
        )

    def initial_state(self, state, *, t):
        """Clear payload/presence rings at finite scalar starting time ``t``.

        The absolute counter retains native target-dtype rounding. Reset carry
        is detached; transform parameters, weights, and routing are unchanged.
        """
        self._validate_source()
        self._validate_state(state)
        for name, expected in _INITIALIZATION_METHODS.items():
            if (
                name in self._source.__dict__
                or getattr(type(self._source), name) is not expected
            ):
                raise FunctionalizationError(
                    f"Fresh ContinuousCon initialization requires its canonical {name} adapter."
                )
        return continuous_initial_state(state, t=t, dt=self._dt, dtype=self.dtype)

    def normalize_cache(self, cache):
        """Own and resample native cache rows to this frozen routing/timestep.

        The returned Tensor/None packet keeps raw and legacy transformed
        histories distinct. Cache construction is a cold operation; pass the
        normalized packet into transformed or compiled initialization.
        """
        self._validate_source()
        return normalize_continuous_cache(
            cache,
            dt=self._cache_dt,
            device=self.device,
            dtype=self.dtype,
            connections=self._connections,
            depth=self._depth,
            width=self._width,
        )

    def restore_cache(self, parameters, state, cache, *, t):
        """Restore detached traffic with current explicit weights/transform.

        ``parameters`` contains ``weights`` and ``transform_parameters`` as in
        the low-level ContinuousCon projection API. Historical observations and
        materialized runtime carry are detached, matching native steady-state
        restoration.
        """
        self._validate_source()
        self._validate_state(state)
        for name, expected in {**_INITIALIZATION_METHODS, **_CACHE_METHODS}.items():
            if (
                name in self._source.__dict__
                or getattr(type(self._source), name) is not expected
            ):
                raise FunctionalizationError(
                    f"ContinuousCon cache initialization requires its canonical {name} adapter."
                )
        if not isinstance(parameters, Mapping) or set(parameters) != {
            "weights",
            "transform_parameters",
        }:
            raise ValueError(
                "ContinuousCon cache parameters require weights and transform_parameters"
            )
        weights, transform = parameters["weights"], parameters["transform_parameters"]
        self._validate_tensor(
            weights,
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="weights",
        )
        self._validate_transform_parameters(transform)
        if not isinstance(cache, ContinuousCache):
            cache = self.normalize_cache(cache)
        validate_continuous_cache(
            cache,
            device=self.device,
            dtype=self.dtype,
            connections=self._connections,
            depth=self._depth,
            width=self._width,
        )
        # No historical values exist in an empty mapped invocation. Avoid
        # evaluating arbitrary transforms whose scalar reductions trigger
        # PyTorch's empty-batch scalar broadcasting path.
        empty = any(
            _has_empty_cache_vmap_axis(value)
            for value in (
                weights,
                cache.raw_history,
                cache.transformed_history,
                *(() if transform is None else transform.values()),
            )
        )
        return restore_continuous_cache(
            state,
            cache,
            weights,
            dt=self._dt,
            t=t,
            routes=self._cache_routes,
            transform=(
                None
                if empty
                else lambda row: self._transform_plan.apply(row, transform)
            ),
        )

    def read_due(self, state):
        """Read queued payload/presence without mutation or counter advancement."""
        self._validate_source()
        self._validate_state(state)
        return self._read_due_values(state)

    def _read_due_values(self, state):
        due = continuous_due(
            state["delivery_buffer"], state["delivery_mask"], state["current_time_step"]
        )
        return (
            dict(state),
            due.payload.reshape(self.shape),
            due.mask.reshape(self.shape),
        )

    def advance(self, weights, state, source_values, *, transform_parameters=None):
        """Project explicit source values and return next carry, payload, and mask."""
        self._validate_source()
        self._validate_state(state)
        self._validate_tensor(
            weights,
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="weights",
        )
        self._validate_tensor(
            source_values,
            shape=self.source_shape,
            dtype=self.source_dtype,
            device=self.source_device,
            label="source_values",
        )
        self._validate_transform_parameters(transform_parameters)
        return self._advance_values(
            weights, state, source_values, transform_parameters=transform_parameters
        )

    def _advance_values(
        self, weights, state, source_values, *, transform_parameters=None
    ):
        """Execute validated projection tensors without inspecting live source modules."""
        queue, presence = state["delivery_buffer"], state["delivery_mask"]
        slot, global_step = state["current_time_step"], state["global_step"]
        if not self.has_connections:
            payload = queue.new_zeros((self._width,))
            mask = presence.new_zeros((self._width,))
        else:
            values = gather_continuous_source(source_values, self._pre_idx).to(
                device=self.device, dtype=self.dtype
            )
            values = self._transform_plan.apply(values, transform_parameters)
            weighted = weights * values
            if self._all_immediate:
                payload = continuous_project(
                    weighted, self._post_idx, width=self._width
                )
                mask = self._post_mask
            else:
                due = continuous_due(queue, presence, slot)
                queue, presence = continuous_clear(queue, presence, slot)
                payload, mask = due.payload, due.mask
                if not self.receive_before_source:
                    payload = payload + continuous_project(
                        weighted.index_select(0, self._zero_idx),
                        self._post_zero,
                        width=self._width,
                    )
                    mask = mask | self._zero_mask
                queue, presence = continuous_enqueue(
                    queue,
                    presence,
                    slot,
                    weighted,
                    post_idx=self._post_nonzero,
                    delay_steps=self._delay_nonzero,
                    depth=self._depth,
                    width=self._width,
                    connection_idx=self._nonzero_idx,
                )
        slot, global_step = continuous_counters(
            slot, global_step, depth=self._depth, all_immediate=self._all_immediate
        )
        return (
            {
                "delivery_buffer": queue,
                "delivery_mask": presence,
                "current_time_step": slot,
                "global_step": global_step,
            },
            payload.reshape(self.shape),
            mask.reshape(self.shape),
        )

    def commit_state_(self, con, state):
        """Restore independent runtime carry without changing model parameters."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "Commit ContinuousCon state outside torch.compile."
            )
        self._validate_compatible(con)
        self._validate_state(state)
        with torch.inference_mode(False):
            copied = {
                name: value.clone(memory_format=torch.contiguous_format)
                for name, value in state.items()
            }
            con.restore_dict_from_checkpoint(copied)
        return con


def make_functional_continuous_con(con):
    """Lower initialized ContinuousCon routing and extract independent tensors."""
    plan = FunctionalContinuousCon(con)
    return plan, plan.extract()


__all__ = [
    "ContinuousConTensors",
    "FunctionalContinuousCon",
    "make_functional_continuous_con",
]
