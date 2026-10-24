"""Owned inspection snapshots and freshness checks for raw materializers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._tensor_contracts import _tensor_binding, _tensor_tree_binding
from ._types import FunctionalizationError


@dataclass(frozen=True)
class _PreparedValues:
    token: object
    parameter_binding: tuple
    constant_binding: tuple
    constants: dict
    value_binding: tuple
    values: dict


def _capture_prepared_values(
    token, parameters, constants, materialized, *, parameter_names=None
):
    """Snapshot materialized values without retaining their autograd graph.

    Materializers recompute their numerical values on consumption, so an
    inspection packet can serve independent backward passes and chunks.
    """
    with torch.inference_mode(False):
        values = {
            name: value.detach().clone(memory_format=torch.preserve_format)
            for name, value in materialized.items()
        }
    selected_parameters = (
        parameters
        if parameter_names is None
        else {name: parameters[name] for name in parameter_names}
    )
    return _PreparedValues(
        token,
        _tensor_binding(selected_parameters),
        _tensor_binding(constants),
        constants,
        _tensor_tree_binding(values),
        values,
    )


def _validate_prepared_owner(prepared, token, *, message):
    """Reject foreign packets before inspecting their values or operands."""
    if not isinstance(prepared, _PreparedValues) or prepared.token is not token:
        raise FunctionalizationError(message)


def _validate_prepared_freshness(prepared, parameters, *, message):
    """Validate the raw operands and detached inspection snapshot in order."""
    if (
        prepared.parameter_binding != _tensor_binding(parameters)
        or prepared.constant_binding != _tensor_binding(prepared.constants)
        or prepared.value_binding != _tensor_tree_binding(prepared.values)
    ):
        raise FunctionalizationError(message)
