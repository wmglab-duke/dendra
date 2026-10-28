"""Read-only metadata for runner defaults and effective continuation sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from ._network_parameters import named_parameters
from ._types import FunctionalizationError


@dataclass(frozen=True, eq=False)
class SourceEntry:
    """One named parameter or constant Tensor in an inspected source frame.

    ``tensor`` is the original live object; inspection never copies or detaches
    it. ``requires_grad`` and ``is_leaf`` capture its flags at inspection time.
    ``aliases`` lists the other paths referring to that exact object, including
    ties between parameters and constants. Distinct Tensor views are distinct
    sources even when their underlying storage is shared.
    """

    path: str
    domain: Literal["parameters", "constants"]
    tensor: torch.Tensor = field(repr=False)
    requires_grad: bool
    is_leaf: bool
    _paths: tuple[str, ...] = field(default=(), init=False, repr=False)

    @property
    def aliases(self) -> tuple[str, ...]:
        """Other identity-alias paths, in their original encounter order."""
        return tuple(path for path in self._paths if path != self.path)


@dataclass(frozen=True, eq=False, repr=False)
class SourceReport:
    """An immutable source inventory with live references and captured metadata.

    ``entries`` preserves every Tensor path, visiting parameters before
    constants in mapping/tuple order. ``trainable_leaves`` contains each Tensor
    identity once, at its first path, when both captured flags are true. This
    tuple can be supplied explicitly to a PyTorch optimizer. Flagged nonleaf
    values remain visible in ``entries`` but are omitted from this leaf view.

    Reinspect after replacing bindings or changing flags. A report does not
    prove that a Tensor influences an output or recover external upstream
    optimizer leaves. State, callback carry, initialization-only inputs and
    call-local stimulation sources are outside this inventory.

    Printing the report shows counts and a compact parameter/constant path
    tree with shapes, dtypes, devices, flags and aliases, without Tensor values.
    Large displays prioritize gradient-bearing paths and indicate omissions;
    ``entries`` always retains the complete inventory.
    """

    entries: tuple[SourceEntry, ...]
    trainable_leaves: tuple[torch.Tensor, ...] = field(repr=False)

    def __repr__(self) -> str:
        return _format_report(self)

    __str__ = __repr__


def _count(value, singular, plural=None):
    return f"{value} {singular if value == 1 else plural or singular + 's'}"


def _format_report(report):
    groups = {}
    for entry in report.entries:
        groups.setdefault(id(entry.tensor), []).append(entry)
    canonical = {identity: entries[0].path for identity, entries in groups.items()}
    alias_groups = sum(len(entries) > 1 for entries in groups.values())
    nonleaf = sum(
        entries[0].requires_grad and not entries[0].is_leaf
        for entries in groups.values()
    )
    lines = [
        "SourceReport",
        "  Sources: "
        + "; ".join(
            (
                _count(len(report.entries), "named path"),
                _count(len(groups), "unique tensor"),
                _count(alias_groups, "alias group"),
            )
        ),
        "  Gradients: "
        + _count(len(report.trainable_leaves), "trainable leaf", "trainable leaves")
        + "; "
        + _count(nonleaf, "nonleaf tensor")
        + " requiring gradients",
    ]
    for domain in ("parameters", "constants"):
        entries = [entry for entry in report.entries if entry.domain == domain]
        lines.extend(("", f"{domain} ({_count(len(entries), 'path')})"))
        if not entries:
            lines.append("  (none)")
            continue
        # Keep large reports bounded while favoring sources relevant to
        # gradients. Selected paths retain their original traversal order.
        selected = [
            index for index, entry in enumerate(entries) if entry.requires_grad
        ][:12]
        selected = set(selected)
        for index in range(len(entries)):
            if len(selected) == 12:
                break
            selected.add(index)
        tree = {}
        for index, entry in enumerate(entries):
            if index not in selected:
                continue
            node = tree
            for part in entry.path.removeprefix(domain + ".").split("."):
                node = node.setdefault(part, {})
            node[None] = entry
        lines.extend(_format_source_tree(tree, canonical, groups))
        omitted = len(entries) - len(selected)
        if omitted:
            lines.append(
                f"  ... {_count(omitted, 'path')} omitted; "
                "see report.entries for the complete inventory."
            )
    return "\n".join(lines)


def _format_source_tree(tree, canonical, groups, prefix="  "):
    children = [(name, node) for name, node in tree.items() if name is not None]
    for index, (name, node) in enumerate(children):
        while None not in node and len(node) == 1:
            child_name, node = next(iter(node.items()))
            name += "." + child_name
        last = index == len(children) - 1
        line = prefix + ("└─ " if last else "├─ ") + name
        entry = node.get(None)
        if entry is not None:
            tensor = entry.tensor
            status = (
                "trainable leaf"
                if entry.requires_grad and entry.is_leaf
                else "nonleaf, requires_grad=True"
                if entry.requires_grad
                else "requires_grad=False"
            )
            line += (
                f"  [shape={tuple(tensor.shape)}, "
                f"{str(tensor.dtype).removeprefix('torch.')}, {tensor.device}; {status}]"
            )
            identity = id(tensor)
            if canonical[identity] != entry.path:
                line += f" -> {canonical[identity]}"
            elif len(groups[identity]) > 1:
                line += f" ({_count(len(groups[identity]) - 1, 'alias', 'aliases')})"
        yield line
        yield from _format_source_tree(
            node, canonical, groups, prefix + ("   " if last else "│  ")
        )


def inspect_sources(subject) -> SourceReport:
    """Inspect a compiled runner's defaults or a Continuation's effective frame.

    Both Population and Network runners returned by ``dn.compile`` are
    accepted. To inspect overrides or fresh/cache replacements from a particular
    call, pass ``auxiliary['continuation']`` instead of the runner. Existing
    runner parameter selectors continue to describe their defaults only.

    Paths start with ``parameters.`` or ``constants.``. Mapping order and named
    tuple fields are preserved, including all Tensor identity aliases. The
    shared parameter-path walker rejects ambiguous dotted paths or unsupported
    source containers. This is metadata inspection, not execution validation.

    Inspection performs no Tensor arithmetic, cloning, graph traversal,
    initialization, preparation, RNG sampling or source lifecycle validation.
    It is a host operation outside ``torch.compile``. Reports retain Tensor
    references, so discard them when their source graphs are no longer needed.
    """
    if torch.compiler.is_compiling():
        raise FunctionalizationError(
            "inspect_sources is a host inspection API; call it outside torch.compile"
        )
    from ._continuation import Continuation
    from ._convenience import CompiledRunner
    from ._network_compile import CompiledNetworkRunner

    if not isinstance(subject, (CompiledRunner, CompiledNetworkRunner, Continuation)):
        raise TypeError(
            "inspect_sources requires a compiled Population/Network runner "
            "or Continuation"
        )
    frames = (("parameters", subject.parameters), ("constants", subject.constants))
    # Mapping accessors can execute authored Python. Complete traversal before
    # capturing each unique Tensor's flags, without calling any model hooks.
    rows = [
        (f"{domain}.{path}", domain, tensor)
        for domain, frame in frames
        for path, tensor in named_parameters(frame).items()
    ]
    paths = {}
    for path, _domain, tensor in rows:
        paths.setdefault(id(tensor), []).append(path)
    paths = {identity: tuple(group) for identity, group in paths.items()}
    flags = {}
    leaves = []
    for _path, _domain, tensor in rows:
        identity = id(tensor)
        if identity not in flags:
            flags[identity] = (tensor.requires_grad, tensor.is_leaf)
            if all(flags[identity]):
                leaves.append(tensor)
    entries = []
    for path, domain, tensor in rows:
        identity = id(tensor)
        entry = SourceEntry(path, domain, tensor, *flags[identity])
        object.__setattr__(entry, "_paths", paths[identity])
        entries.append(entry)
    return SourceReport(entries=tuple(entries), trainable_leaves=tuple(leaves))


__all__ = ["SourceEntry", "SourceReport", "inspect_sources"]
