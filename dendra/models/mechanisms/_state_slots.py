"""Declared deterministic state slots shared by checkpoints and lowering.

Lazy descriptions retain ownership and serialized names. Consumers choose
phase order, admitted declaration names, tensor access, and restoration policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _StateSlot:
    owner: object
    name: str
    checkpoint_suffix: str


def state_value_slots(module, state):
    for name in sorted(state._state):
        yield _StateSlot(module, name, name)


def state_carry_slots(state_key, state, names):
    for name in sorted(names):
        yield _StateSlot(state, name, f"DE.{state_key}.{name}")


def module_carry_slots(module, names):
    # Admission and ordering belong to the caller: native checkpoints retain
    # declaration order and legacy fallbacks; functional paths sort names.
    for name in names:
        yield _StateSlot(module, name, name)


def saved_buffer_slots(module):
    for name in sorted(module._save):
        buffer_name = f"{name}_"
        yield _StateSlot(module, buffer_name, buffer_name)
