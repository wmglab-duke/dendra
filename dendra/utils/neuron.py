"""Helpers for navigating trees of NEURON sections."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal, Protocol, TypeAlias, overload

from neuron import h

__all__ = ["path_sections", "path_via"]


class _SectionLike(Protocol):
    """The part of NEURON's ``Section`` interface used in this module."""

    def name(self) -> str: ...

    def __hash__(self) -> int: ...


class _SectionListLike(Protocol):
    """Structural type for a NEURON ``SectionList``."""

    def __iter__(self) -> Iterator[_SectionLike]: ...

    def append(self, *, sec: _SectionLike) -> None: ...


_SectionPath: TypeAlias = list[_SectionLike]


def _parent_section(sec: _SectionLike) -> _SectionLike | None:
    """Return the parent of ``sec``, or ``None`` when ``sec`` is a root.

    Modern NEURON versions expose :meth:`Section.parentseg`, whose result is
    either the parent segment or ``None``.  The ``SectionRef`` branch preserves
    compatibility with older Section implementations; ``has_parent`` must be
    checked before accessing ``parent`` because HOC otherwise raises an error.
    """
    if hasattr(sec, "parentseg"):
        parent_segment = sec.parentseg()
        return None if parent_segment is None else parent_segment.sec

    section_ref = h.SectionRef(sec=sec)
    return section_ref.parent if section_ref.has_parent() else None


def _chain_to_root(sec: _SectionLike) -> _SectionPath:
    """Return ``sec`` and each ancestor in order, ending at its tree root."""
    chain = []
    current: _SectionLike | None = sec
    while current is not None:
        chain.append(current)
        current = _parent_section(current)
    return chain


def _as_section_list(sections: Iterable[_SectionLike]) -> _SectionListLike:
    """Build a NEURON ``SectionList`` in the order of ``sections``.

    Appending sections explicitly works across NEURON releases; passing a
    Python iterable to the HOC constructor is not supported consistently.
    """
    section_list = h.SectionList()
    for section in sections:
        section_list.append(sec=section)
    return section_list


def path_sections(a: _SectionLike, b: _SectionLike) -> _SectionPath | None:
    """Return the unique simple section path from ``a`` to ``b``.

    The returned Python list includes both endpoints and is ordered from
    ``a`` to ``b``.  If the sections belong to disconnected trees, return
    ``None``.
    """
    a_to_root = _chain_to_root(a)
    b_to_root = _chain_to_root(b)

    # Locate the lowest common ancestor by walking outward from ``a``.
    # Section names are not unique in NEURON, but Section identity is hashable
    # and is preserved when following ``parentseg().sec``.
    b_positions = {section: i for i, section in enumerate(b_to_root)}
    lca_a_index = lca_b_index = None
    for a_index, section in enumerate(a_to_root):
        b_index = b_positions.get(section)
        if b_index is not None:
            lca_a_index, lca_b_index = a_index, b_index
            break

    if lca_a_index is None or lca_b_index is None:
        return None

    # Ascend from a through the LCA, then descend from the LCA to b without
    # repeating the LCA.
    ascending = a_to_root[: lca_a_index + 1]
    descending = list(reversed(b_to_root[:lca_b_index]))
    return ascending + descending


@overload
def path_via(
    a: _SectionLike,
    b: _SectionLike,
    c: _SectionLike,
    *,
    return_sectionlist: Literal[False] = False,
) -> _SectionPath | None: ...


@overload
def path_via(
    a: _SectionLike,
    b: _SectionLike,
    c: _SectionLike,
    *,
    return_sectionlist: Literal[True],
) -> _SectionListLike | None: ...


def path_via(
    a: _SectionLike,
    b: _SectionLike,
    c: _SectionLike,
    *,
    return_sectionlist: bool = False,
) -> _SectionPath | _SectionListLike | None:
    """Return the simple path from ``a`` to ``b`` when it passes through ``c``.

    Because NEURON sections form a tree, there is exactly one simple path
    between connected sections.  Return ``None`` when ``a`` and ``b`` are
    disconnected or when their path does not contain ``c``.

    Parameters
    ----------
    a, b
        Endpoints of the requested path.
    c
        Section that the path must contain.
    return_sectionlist
        When true, return a NEURON ``h.SectionList`` instead of a Python list.
        Ordering and endpoint inclusion are otherwise unchanged.
    """
    path = path_sections(a, b)
    if path is None:
        return None

    if c not in path:
        return None

    return _as_section_list(path) if return_sectionlist else path
