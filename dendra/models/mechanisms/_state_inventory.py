"""Lazy checkpoint phases and tensor admission for handler state consumers."""

from dataclasses import dataclass

import torch

from ._state_slots import (
    _StateSlot,
    module_carry_slots,
    saved_buffer_slots,
    state_carry_slots,
    state_value_slots,
)


@dataclass(frozen=True)
class _TensorStateBlock:
    role: str
    prefix: str
    slots: tuple[_StateSlot, ...]
    state_key: str | None = None


@dataclass(frozen=True)
class _StochasticStateBoundary:
    owner: object
    prefix: str
    legacy_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class _DelayedStateBoundary:
    owner: object
    prefix: str


def checkpoint_module_blocks(module, prefix):
    """Yield native checkpoint phases without discovering later state early."""
    yield _StochasticStateBoundary(module, prefix, (prefix,))
    for state_key, state in module.DE.items():
        values = tuple(state_value_slots(module, state))
        yield _TensorStateBlock("state", prefix, values)
        yield _TensorStateBlock(
            "state_carry",
            prefix,
            tuple(
                state_carry_slots(state_key, state, tuple(getattr(state, "_carry", ())))
            ),
            state_key,
        )
        yield _StochasticStateBoundary(
            state,
            f"{prefix}.DE.{state_key}",
            tuple(f"{prefix}.{slot.checkpoint_suffix}" for slot in values),
        )
    yield _TensorStateBlock(
        "carry",
        prefix,
        tuple(module_carry_slots(module, tuple(getattr(module, "_carry", ())))),
    )
    yield _TensorStateBlock("saved", prefix, tuple(saved_buffer_slots(module)))
    yield _DelayedStateBoundary(module, prefix)


def functional_module_blocks(module, prefix):
    """Yield declared functional roles in their existing projection order."""
    for state in module.DE.values():
        yield _TensorStateBlock(
            "state", prefix, tuple(state_value_slots(module, state))
        )
    yield _TensorStateBlock(
        "carry", prefix, tuple(module_carry_slots(module, sorted(module._carry)))
    )
    yield _DelayedStateBoundary(module, prefix)
    yield _TensorStateBlock("saved", prefix, tuple(saved_buffer_slots(module)))
    for state_key, state in module.DE.items():
        yield _TensorStateBlock(
            "state_carry",
            prefix,
            tuple(state_carry_slots(state_key, state, state._carry)),
            state_key,
        )


class _MutableTensorPlan:
    """Validate targets and prepare each distinct candidate frame once."""

    def __init__(self, state_dict, prepare):
        self._state_dict = state_dict
        self._prepare = prepare
        self._tensor_memo = {}
        self._missing = object()
        self.assignment_map = {}
        self.assignment_keys = {}

    def add(
        self,
        owner,
        name,
        key,
        *,
        expected_shape=None,
        optional=False,
        reference=None,
    ):
        if key not in self._state_dict:
            if optional:
                return
            raise KeyError(f"MechanismHandler mutable state is missing {key!r}.")
        candidate = self._state_dict[key]
        if not torch.is_tensor(candidate):
            raise TypeError(f"MechanismHandler mutable state {key!r} must be a tensor.")

        live = getattr(owner, name, self._missing) if reference is None else reference
        if live is self._missing or not torch.is_tensor(live):
            raise TypeError(
                f"Live mutable field {key!r} must be a tensor before restore."
            )
        shape = tuple(live.shape) if expected_shape is None else tuple(expected_shape)
        if tuple(candidate.shape) != shape:
            raise ValueError(
                f"MechanismHandler mutable state {key!r} has shape "
                f"{tuple(candidate.shape)}, expected {shape}."
            )
        if candidate.layout != live.layout:
            raise ValueError(
                f"MechanismHandler mutable state {key!r} has layout "
                f"{candidate.layout}, expected {live.layout}."
            )

        memo_key = (id(candidate), live.device, live.dtype, live.layout)
        if memo_key not in self._tensor_memo:
            self._tensor_memo[memo_key] = self._prepare(candidate, live)
        prepared = self._tensor_memo[memo_key]
        target = (id(owner), name)
        existing = self.assignment_map.get(target)
        if existing is not None and existing[2] is not prepared:
            raise ValueError(
                f"Mutable field {name!r} received conflicting snapshot entries."
            )
        self.assignment_map[target] = (owner, name, prepared)
        self.assignment_keys.setdefault(target, key)
