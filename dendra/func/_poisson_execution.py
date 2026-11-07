"""Cold isolation of Waveform registration occurrences for explicit execution."""

from __future__ import annotations

from collections import OrderedDict
from types import MemberDescriptorType, MethodType

from dendra.models.stim.waveform.core import Waveform

from ._types import FunctionalizationError


def _copy_occurrence(module):
    """Copy a module's registration ownership without copying its tensors."""

    def bind(value, result):
        if value is module:
            return result
        if isinstance(value, MethodType) and value.__self__ is module:
            return MethodType(value.__func__, result)
        return value

    try:
        result = object.__new__(type(module))
        result.__dict__ = module.__dict__.copy()
        for owner_type in type(module).__mro__:
            for descriptor in owner_type.__dict__.values():
                if isinstance(descriptor, MemberDescriptorType):
                    try:
                        value = descriptor.__get__(module, type(module))
                    except AttributeError:
                        continue
                    descriptor.__set__(result, bind(value, result))
    except (AttributeError, TypeError) as error:
        raise FunctionalizationError(
            "Waveform module registrations could not be isolated for execution."
        ) from error
    # These containers belong to one occurrence. Their registered tensors and
    # callbacks retain their authored identities and sharing relationships.
    for name, value in tuple(result.__dict__.items()):
        if name in {"_modules", "_parameters", "_buffers"} or (
            name.startswith("_") and "hook" in name
        ):
            if isinstance(value, dict):
                values = ((key, bind(item, result)) for key, item in value.items())
                result.__dict__[name] = (
                    OrderedDict(values)
                    if isinstance(value, OrderedDict)
                    else dict(values)
                )
        elif name == "_non_persistent_buffers_set":
            result.__dict__[name] = set.copy(value)
        else:
            result.__dict__[name] = bind(value, result)
    return result


def isolate_waveform_aliases(module):
    """Give each private Waveform registration its own module occurrences.

    ``functional_call`` swaps registered tensors by path. Multiple paths to
    the same module would swap the same registration more than once and can
    leave invocation operands installed after restoration. Distinct module
    occurrences avoid that ambiguity while retaining every registered tensor
    identity, including shared trainable parameters.

    The caller supplies an isolated execution container. Only its Waveform
    subtrees are replaced; ordinary owners keep their lifecycle and identity.
    This cold operation does not invoke user constructors or copy hooks.
    """
    active = set()
    visited_owners = set()

    def expand(owner, *, copy):
        identity = id(owner)
        if identity in active:
            raise FunctionalizationError(
                "Waveform module registrations cannot contain cycles."
            )
        active.add(identity)
        result = _copy_occurrence(owner) if copy else owner
        result._modules = {
            name: None if child is None else expand(child, copy=True)
            for name, child in owner._modules.items()
        }
        active.remove(identity)
        return result

    def visit(owner):
        if id(owner) in visited_owners:
            return
        visited_owners.add(id(owner))
        for name, child in tuple(owner._modules.items()):
            if isinstance(child, Waveform):
                owner._modules[name] = expand(child, copy=True)
            elif child is not None:
                visit(child)

    if isinstance(module, Waveform):
        expand(module, copy=False)
    else:
        visit(module)
    return module
