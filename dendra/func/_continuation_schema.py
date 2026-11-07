"""Receiving continuation frames projected from lowered tensor contracts.

These templates describe only shape, dtype, device and container structure.
They neither extract native values nor prepare or initialize a receiving model.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ._continuation_codec import _NAMED_TUPLES, _TensorDescriptor
from ._types import FunctionalizationError


def _descriptor(shape, dtype, device):
    return _TensorDescriptor(tuple(shape), dtype, device)


def _schema_frame(schema, leaf_type):
    if isinstance(schema, leaf_type):
        return _descriptor(schema.shape, schema.dtype, schema.device)
    if schema is None:
        return None
    if isinstance(schema, Mapping):
        return {name: _schema_frame(child, leaf_type) for name, child in schema.items()}
    if type(schema) in (list, tuple, *_NAMED_TUPLES):
        children = [_schema_frame(child, leaf_type) for child in schema]
        return (
            type(schema)(*children)
            if type(schema) in _NAMED_TUPLES
            else type(schema)(children)
        )
    raise FunctionalizationError(
        "continuation checkpoint receiving frames contain an unsupported container type"
    )


def _population_constants(population):
    if population._topology_kind in {"multi_point", "multi_scalar", "multi_block"}:
        constants = {
            name: _descriptor(
                shape,
                population._multi_public_constant_dtypes[name],
                population._multi_public_constant_devices[name],
            )
            for name, shape in population._multi_public_constant_shapes.items()
        }
    else:
        constants = {
            name: _descriptor(
                population._geometry_constant_shapes[name],
                population.dtype,
                population.device,
            )
            for name in population._geometry_constant_names
        }
    constants.update(
        {
            entry.key: _descriptor(entry.shape, entry.dtype, entry.device)
            for entry in population._parametrization_constants
        }
    )
    constants["dt"] = _descriptor((), population.dtype, population.device)
    constants.update(
        {
            name: _descriptor(value.shape, value.dtype, value.device)
            for name, value in population._noise_workspace_constants.items()
        }
    )
    if population._cache_value_specs:
        constants.update(
            {
                name: _descriptor(shape, dtype, device)
                for name, _path, shape, dtype, device in population._cache_value_specs
            }
        )
        constants["cache.active"] = _descriptor((), torch.bool, population.device)
    for slot in population._retained_materializations.slots:
        constants[slot.value_key] = _descriptor(slot.shape, slot.dtype, slot.device)
        constants[slot.active_key] = _descriptor((), torch.bool, population.device)
        constants[slot.baseline_key] = _descriptor(slot.shape, slot.dtype, slot.device)
    if population._random_plan is not None:
        constants.update(
            {
                name: _descriptor(shape, dtype, device)
                for name, (
                    shape,
                    dtype,
                    device,
                ) in population._random_plan.schema.items()
            }
        )
    return constants


def _receiving_frames(functional):
    """Project frozen receiving schemas without accessing numerical values."""
    from ._network import FunctionalNetwork, _TensorSchema
    from ._population import FunctionalPopulation, _StateLeaf

    if type(functional) is FunctionalNetwork:
        return {
            "state": _schema_frame(functional._state_schema, _TensorSchema),
            "parameters": _schema_frame(functional._parameter_schema, _TensorSchema),
            "constants": _schema_frame(functional._constant_schema, _TensorSchema),
        }
    if type(functional) is FunctionalPopulation:
        return {
            "state": _schema_frame(functional._state_schema, _StateLeaf),
            "parameters": {
                name: _descriptor(
                    shape,
                    functional._parameter_dtypes[name],
                    functional._parameter_devices[name],
                )
                for name, shape in functional._parameter_shapes.items()
            },
            "constants": _population_constants(functional),
        }
    raise TypeError("functional must be a FunctionalPopulation or FunctionalNetwork")


__all__ = []
