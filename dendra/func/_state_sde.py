"""Explicit State SDE draw bindings over the shared generated solvers."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from types import MethodType

import torch

from dendra.models.mechanisms._sde import (
    sample_state_sde_draw,
    state_sde_draw,
    state_sde_dt,
)
from dendra.models.mechanisms._state import State
from dendra.models.random_parameters import DistributionSpec

from ._types import FunctionalizationError

_SDE_METHODS = ("euler_maruyama", "euler_heun")
_CANONICAL_HOOKS = {
    name: inspect.getattr_static(State, name)
    for name in ("advance", "_derive_advance_values", "_sde_randn_like")
}


def state_sde_owners(population):
    """Yield bundles in the handler's accepted pre/post-current phase order."""
    handler = getattr(population, "mech", None)
    if handler is None:
        return ()
    owners = {id(mechanism): name for name, mechanism in handler.mechanisms.items()}
    schedule = (
        (*handler._pre_state_advance_plan, *handler._post_state_advance_plan)
        if handler._ion_current_reads
        else handler._state_advance_plan
    )
    if len(schedule) != len(owners) or {
        id(mechanism) for mechanism, _support in schedule
    } != set(owners):
        raise FunctionalizationError(
            "State SDE execution requires the canonical handler phase schedule"
        )
    return tuple(
        (
            f"integrator.mech.mechanisms.{owners[id(mechanism)]}.DE.{bundle}",
            state,
            mechanism,
        )
        for mechanism, _support in schedule
        for bundle, state in mechanism.DE.items()
        if state.method in _SDE_METHODS or getattr(state, "_diffusion", ())
    )


def _solver_draw_names(solve):
    source = getattr(solve, "__source__", "")
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return None
    return tuple(
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "state_sde_draw"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    )


def state_sde_failures(state):
    """Reject custom implicit sampling/advance adapters before lowering."""
    if not isinstance(state, State):
        return ("State SDE execution requires a canonical State bundle",)
    if state.method not in _SDE_METHODS:
        return ("State DIFFUSION requires Euler-Maruyama or Euler-Heun",)
    failures = []
    for name, canonical in _CANONICAL_HOOKS.items():
        if (
            name in state.__dict__
            or inspect.getattr_static(type(state), name) is not canonical
        ):
            failures.append(
                f"State SDE execution requires canonical {name}; custom adapters are currently unsupported"
            )
    if (
        _CANONICAL_HOOKS["_sde_randn_like"].__globals__.get("sample_state_sde_draw")
        is not sample_state_sde_draw
    ):
        failures.append(
            "State SDE execution requires its canonical Gaussian sampler binding"
        )
    draw_names = getattr(state, "_sde_draw_state_names", None)
    rng_names = getattr(state, "_sde_rng_names", None)
    if (
        not isinstance(draw_names, tuple)
        or len(set(draw_names)) != len(draw_names)
        or tuple(name for name in state._state if name in draw_names) != draw_names
        or rng_names != tuple(f"{name}_dW_rng" for name in state._state)
    ):
        failures.append("State SDE ordered draw/RNG declarations are not canonical")
    solve = getattr(getattr(state, "_solve", None), "__func__", None)
    filename = getattr(solve, "__generated_filename__", "")
    code = getattr(solve, "__code__", None)
    if (
        code is None
        or not filename.startswith(f"<dendra.{state.method}.solve:")
        or filename != code.co_filename
        or _solver_draw_names(solve) != draw_names
        or solve.__globals__.get("state_sde_draw") is not state_sde_draw
        or solve.__globals__.get("state_sde_dt") is not state_sde_dt
    ):
        failures.append("State SDE execution requires its canonical generated solver")
    return tuple(failures)


def _normal_sample(parameters, rng, shape, *, device, dtype):
    del parameters
    return sample_state_sde_draw(rng, shape, device=device, dtype=dtype)


def _normal_value(parameters, draw):
    del parameters
    return draw


# State diffusion always uses a standard Gaussian, independent of user changes
# to the RAND/NOISE distribution registry.
_STANDARD_NORMAL = DistributionSpec(
    "state_sde_standard_normal",
    {},
    {},
    _normal_sample,
    draw_sampler=sample_state_sde_draw,
    transform=_normal_value,
)


@dataclass(frozen=True)
class _SDEEntry:
    key: str
    path: str
    name: str
    stream: str
    shape: tuple
    dtype: torch.dtype
    device: torch.device
    buffer_name: str
    leaves: tuple
    draw_names: tuple
    spec: object = _STANDARD_NORMAL

    @property
    def mapping_slot(self):
        return f"population.{self.path}.{self.buffer_name}"


def _explicit_sde_advance(self, v, dt, values):
    draws = {name: self._buffers[slot] for name, slot in self._functional_sde_slots}
    return self.advance(v, dt, values, draws=draws)


def _state_contract(state):
    return (
        state.method,
        tuple(state._state),
        tuple(state._diffusion),
        tuple(state._sde_rng_names),
        tuple(state._sde_draw_state_names),
        tuple(sorted(state.method_kwargs.items())),
        state._solve.__func__.__source__,
    )


class StateSDEPlan:
    """Frozen ordered Gaussian schemas; all values and RNG suffixes are inputs."""

    def __init__(self, population, stream_ids):
        from ._runtime_noise import _validate_rng_adapter

        self.population = population
        self.states = state_sde_owners(population)
        self.contracts, self.bindings, entries = {}, {}, []
        for path, state, parent in self.states:
            failures = state_sde_failures(state)
            if failures:
                raise FunctionalizationError("; ".join(failures))
            self.contracts[path] = _state_contract(state)
            self.bindings[path] = state._solve.__func__
            for name in state._sde_rng_names:
                _validate_rng_adapter(getattr(state, name))
            for name in state._sde_draw_state_names:
                value = parent._buffers.get(name)
                if not torch.is_tensor(value):
                    raise FunctionalizationError(
                        f"State SDE field {path}.{name} needs registered mechanism state"
                    )
                rng = getattr(state, f"{name}_dW_rng")
                key = f"sde.draws.{path}.{name}"
                entries.append(
                    _SDEEntry(
                        key,
                        path,
                        name,
                        stream_ids[id(rng)],
                        tuple(value.shape),
                        value.dtype,
                        value.device,
                        f"_functional_sde_draw_{name}",
                        (torch.ones_like(value).detach(),),
                        (key,),
                    )
                )
        self.entries = tuple(entries)

    def validate_population(self, population, *, source=False):
        from ._runtime_noise import _validate_rng_adapter

        states = state_sde_owners(population)
        if tuple(path for path, *_ in states) != tuple(self.contracts):
            raise FunctionalizationError("State SDE declarations changed; lower again")
        by_key = {(entry.path, entry.name): entry for entry in self.entries}
        for path, state, parent in states:
            failures = state_sde_failures(state)
            if failures or _state_contract(state) != self.contracts[path]:
                raise FunctionalizationError(
                    "State SDE solver/declarations changed; lower again"
                )
            if source and state._solve.__func__ is not self.bindings[path]:
                raise FunctionalizationError(
                    "State SDE solver binding changed; lower again"
                )
            for name in state._sde_rng_names:
                _validate_rng_adapter(getattr(state, name))
            for name in state._sde_draw_state_names:
                value = parent._buffers[name]
                entry = by_key[path, name]
                if (tuple(value.shape), value.dtype, value.device) != (
                    entry.shape,
                    entry.dtype,
                    entry.device,
                ):
                    raise FunctionalizationError(
                        "State SDE draw layout changed; lower again"
                    )

    def install(self, population):
        """Bind only an exclusively owned execution clone, after layout capture."""
        by_path = {}
        for entry in self.entries:
            by_path.setdefault(entry.path, []).append(entry)
        for path, state, _parent in state_sde_owners(population):
            slots = []
            for entry in by_path.get(path, ()):
                if hasattr(state, entry.buffer_name):
                    raise FunctionalizationError(
                        "State SDE temporary draw slot conflicts with model state"
                    )
                state.register_buffer(
                    entry.buffer_name,
                    torch.ones(entry.shape, dtype=entry.dtype, device=entry.device),
                    persistent=False,
                )
                slots.append((entry.name, entry.buffer_name))
            state._functional_sde_slots = tuple(slots)
            state._derive_advance_values = MethodType(_explicit_sde_advance, state)

    def values(self, packet):
        return {
            entry.mapping_slot: packet.draws[entry.key].detach().clone()
            for entry in self.entries
        }


__all__ = []
