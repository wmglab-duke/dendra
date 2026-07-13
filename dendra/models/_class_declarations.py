"""Class-body-local storage for Dendra's declarative model APIs."""

from __future__ import annotations

import inspect
from collections.abc import MutableSequence
from typing import Any

_DECLARATIONS_KEY = "__dendra_class_declarations__"


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
