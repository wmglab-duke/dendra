"""Runtime state slots shared by checkpoints and lowering.

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


def shared_field_slots(owner, checkpoint_prefix, names):
    for name in names:
        yield _StateSlot(owner, name, f"{checkpoint_prefix}.{name}")


@dataclass(frozen=True)
class _LocalBinding:
    mechanism_name: str
    name: str
    checkpoint_key: str


def local_binding_groups(mechanism_maps):
    """Keep groups intact so native owners resolve before names are read."""
    for mechanism_map in mechanism_maps:
        yield from mechanism_map.items()


def _local_bindings(groups):
    for mechanism_name, names in groups:
        for name in names:
            yield _LocalBinding(mechanism_name, name, f"{mechanism_name}.{name}")


def local_field_bindings(mechanism_maps):
    return _local_bindings(local_binding_groups(mechanism_maps))


def local_source_bindings(mechanism_maps):
    return _local_bindings(
        (mechanism_name, fields.values())
        for mechanism_name, fields in local_binding_groups(mechanism_maps)
    )


def read_local_bindings(handler):
    yield from local_field_bindings(handler.read_ion.values())
    yield from local_field_bindings(handler.read_material.values())


def material_local_buffer_names(handler):
    """Project native material locals in first-owner, sorted-name order."""
    names = {}
    for material_map in (handler.read_material, handler.write_material):
        for mechanism_name, fields in local_binding_groups(material_map.values()):
            names.setdefault(mechanism_name, set()).update(fields)
    for mechanism_name, fields in local_binding_groups(
        handler.source_material.values()
    ):
        names.setdefault(mechanism_name, set()).update(fields.values())
    return {
        mechanism_name: tuple(sorted(buffer_names))
        for mechanism_name, buffer_names in names.items()
    }
