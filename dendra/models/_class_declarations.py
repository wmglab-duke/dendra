"""Class-body-local storage for Dendra's declarative model APIs."""

from __future__ import annotations

import inspect
from collections.abc import MutableSequence
from typing import Any

import torch

_DECLARATIONS_KEY = "__dendra_class_declarations__"

_LOCAL_TIMESTEP_BUFFER_SHAPE = "local"
_DEFERRED_BUFFER_SHAPE = "deferred"


def normalize_buffer_schema(
    dtype: torch.dtype | None,
    shape: Any,
    *,
    declaration: str,
) -> tuple[torch.dtype | None, str | tuple[int, ...]]:
    """Normalize one dtype/shape schema for persistent declared storage."""
    label = str(declaration)
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError(f"{label} dtype must be a torch.dtype or None.")
    if isinstance(shape, str):
        if shape not in {"local", _DEFERRED_BUFFER_SHAPE}:
            raise ValueError(
                f"{label} shape must be 'local', 'deferred', or a tuple of "
                "non-negative Python integers."
            )
        return dtype, shape
    if not isinstance(shape, tuple):
        raise TypeError(
            f"{label} shape must be 'local', 'deferred', or a tuple of "
            "non-negative Python integers."
        )
    for dimension in shape:
        if type(dimension) is not int:
            raise TypeError(
                f"{label} shape dimensions must be non-negative Python "
                "integers (bool is not accepted)."
            )
        if dimension < 0:
            raise ValueError(
                f"{label} shape dimensions must be non-negative; got {shape!r}."
            )
    return dtype, tuple(shape)


def merge_buffer_schemas(
    target: dict[Any, tuple[torch.dtype | None, str | tuple[int, ...]]],
    declarations,
    *,
    owner: type,
    declaration: str,
) -> None:
    """Merge dtype/shape declarations while rejecting ambiguous MRO schemas."""
    label = str(declaration)
    for name, raw_schema in declarations.items():
        schema = normalize_buffer_schema(
            *raw_schema,
            declaration=label,
        )
        previous = target.get(name)
        if previous is not None and previous != schema:
            raise ValueError(
                f"{owner.__name__} inherits or redeclares {label} {name!r} "
                f"with conflicting schemas {previous!r} and {schema!r}."
            )
        target[name] = schema


def normalize_timestep_buffer_shape(shape: Any) -> str | tuple[int, ...]:
    """Return one canonical ``TIMESTEP_BUFFER`` shape declaration.

    ``"local"`` follows the owning Parameterized module's ``shape_p``.  An
    explicit tuple is structural: it is independent of the Population's local
    and batch shapes.  Keep this normalization in the shared declaration layer
    so Mechanism and State apply exactly the same class-definition contract.
    """

    if isinstance(shape, str):
        if shape != _LOCAL_TIMESTEP_BUFFER_SHAPE:
            raise ValueError(
                "TIMESTEP_BUFFER shape must be 'local' or a tuple of "
                "non-negative Python integers."
            )
        return shape
    if not isinstance(shape, tuple):
        raise TypeError(
            "TIMESTEP_BUFFER shape must be 'local' or a tuple of "
            "non-negative Python integers."
        )
    for dimension in shape:
        # bool is an int subclass, but accepting True/False as structural
        # dimensions makes authored schemas needlessly ambiguous.
        if type(dimension) is not int:
            raise TypeError(
                "TIMESTEP_BUFFER shape dimensions must be non-negative "
                "Python integers (bool is not accepted)."
            )
        if dimension < 0:
            raise ValueError(
                f"TIMESTEP_BUFFER shape dimensions must be non-negative; got {shape!r}."
            )
    return tuple(shape)


def merge_timestep_buffer_shapes(
    target: dict[Any, str | tuple[int, ...]],
    declarations,
    *,
    owner: type,
) -> None:
    """Merge declarations while rejecting ambiguous MRO redeclarations."""

    for name, raw_shape in declarations.items():
        shape = normalize_timestep_buffer_shape(raw_shape)
        previous = target.get(name)
        if previous is not None and previous != shape:
            raise ValueError(
                f"{owner.__name__} inherits or redeclares TIMESTEP_BUFFER "
                f"{name!r} with conflicting shapes {previous!r} and {shape!r}. "
                "A timestep-buffer name must have one shape throughout its MRO."
            )
        target[name] = shape


def declare_class_value(
    channel: str,
    value: Any,
    legacy_queue: MutableSequence[Any],
) -> None:
    """Record a declaration in its active class namespace when possible.

    Class namespaces are discarded automatically if execution of the class body
    raises. Keeping declarations there therefore prevents a failed definition
    from contaminating the next class. Calls made outside a class body retain
    the historical queue-to-next-class behavior through ``legacy_queue``.
    """
    frame = inspect.currentframe()
    try:
        caller = None if frame is None else frame.f_back
        while caller is not None:
            namespace = caller.f_locals
            if "__module__" in namespace and "__qualname__" in namespace:
                declarations = namespace.setdefault(_DECLARATIONS_KEY, {})
                declarations.setdefault(str(channel), []).append(value)
                return
            caller = caller.f_back
    finally:
        del frame

    legacy_queue.append(value)


def consume_class_values(
    cls: type,
    channel: str,
    legacy_queue: MutableSequence[Any],
) -> list[Any]:
    """Return declarations owned by ``cls`` plus legacy queued declarations."""
    declarations = cls.__dict__.get(_DECLARATIONS_KEY, {})
    values = []
    if legacy_queue:
        values.extend(legacy_queue)
        legacy_queue.clear()
    # Historically, declarations made before a class body appeared earlier in
    # the shared queue, so class-body declarations won on ordered merge/update.
    values.extend(declarations.get(str(channel), ()))
    return values
