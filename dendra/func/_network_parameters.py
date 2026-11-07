"""Canonical Network parameter names and identity-preserving independent leaves."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ._parameter_queries import matching_names
from ._parameter_queries import parameter_name as _parameter_name
from ._parameter_queries import select_names


def _parameter_paths(tree, path=()):
    if torch.is_tensor(tree):
        yield ".".join(path), tree
    elif isinstance(tree, Mapping):
        for name, value in tree.items():
            yield from _parameter_paths(value, (*path, str(name)))
    elif isinstance(tree, tuple):
        names = getattr(tree, "_fields", tuple(map(str, range(len(tree)))))
        for name, value in zip(names, tree, strict=True):
            yield from _parameter_paths(value, (*path, name))
    elif tree is not None:
        raise TypeError(
            "Network parameters contain only mappings, tuples, Tensors, and None"
        )


def named_parameters(parameters):
    """Return every explicit Tensor by canonical dotted path in extraction order.

    Population parameter keys already contain their module-qualified names.
    Network collection/component names prefix those keys; schedule NamedTuple
    fields retain names such as ``schedule.times``. Repeated identities remain
    visible under every alias so query selection does not hide shared leaves.
    """
    named = {}
    for name, value in _parameter_paths(parameters):
        if name in named:
            raise ValueError(
                f"Network parameter paths collide at canonical name {name!r}; "
                "choose unambiguous component names"
            )
        named[name] = value
    return named


def find_parameters(parameters, query):
    """Find canonical names by substring, full-name glob, or regex search."""
    candidates = named_parameters(parameters)
    return {name: candidates[name] for name in matching_names(candidates, query)}


def parameter_name(parameters, query):
    """Resolve one exact, partial, glob, or regex parameter-name match."""
    candidates = named_parameters(parameters)
    return _parameter_name(candidates, query, label="Network parameter")


def get_parameter(parameters, query):
    """Return the explicit Tensor selected by one exact, partial, glob, or regex match."""
    return named_parameters(parameters)[parameter_name(parameters, query)]


def independent_parameters(parameters, trainable=()):
    """Clone a nested packet once per Tensor identity, retaining parameter ties.

    Exact or unique partial names select one leaf; full-name globs and ``re:``
    or compiled regex selectors can select multiple. Each selector must match.
    Selecting any alias makes the shared independent leaf a single trainable
    ``nn.Parameter``. All other leaves are detached ordinary tensors, including
    registered buffers extracted in continuous-transform parameter packets.
    Neither storage nor autograd history is shared with the input packet.
    """
    named = named_parameters(parameters)
    selected = select_names(named, trainable, label="Network parameter")
    non_differentiable = sorted(
        name
        for name in selected
        if not (torch.is_floating_point(named[name]) or torch.is_complex(named[name]))
    )
    if non_differentiable:
        raise ValueError(
            "trainable parameter leaves must use a floating-point or complex dtype; "
            f"invalid={non_differentiable}"
        )
    trainable_ids = {id(named[name]) for name in selected}
    copies = {}

    def clone(tree):
        if torch.is_tensor(tree):
            identity = id(tree)
            if identity not in copies:
                value = tree.detach().clone(memory_format=torch.preserve_format)
                copies[identity] = (
                    torch.nn.Parameter(value) if identity in trainable_ids else value
                )
            return copies[identity]
        if isinstance(tree, Mapping):
            return {name: clone(value) for name, value in tree.items()}
        if isinstance(tree, tuple):
            values = tuple(map(clone, tree))
            return type(tree)(*values) if hasattr(tree, "_fields") else values
        return None

    # Independent optimizer leaves require version counters even when the
    # caller constructs the runner under inference_mode().
    with torch.inference_mode(False):
        return clone(parameters)


__all__ = []
