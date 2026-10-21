"""Shared host-side selection over canonical parameter names."""

from __future__ import annotations

import re
from collections.abc import Iterable
from fnmatch import fnmatchcase

ParameterQuery = str | re.Pattern[str]


def validate_query(query: ParameterQuery) -> ParameterQuery:
    """Accept nonempty strings or compiled regular expressions over strings."""
    if isinstance(query, str):
        if not query:
            raise ValueError("parameter query must be a non-empty string")
        return query
    if isinstance(query, re.Pattern) and isinstance(query.pattern, str):
        if not query.pattern:
            raise ValueError("parameter regex must be a non-empty expression")
        return query
    raise TypeError("parameter query must be a string or a compiled string regex")


def normalize_queries(
    queries: ParameterQuery | Iterable[ParameterQuery],
) -> tuple[ParameterQuery, ...]:
    """Treat a string or compiled regex as one selector rather than an iterable."""
    if isinstance(queries, (str, re.Pattern)):
        queries = (queries,)
    return tuple(validate_query(query) for query in queries)


def _parse(query):
    query = validate_query(query)
    if isinstance(query, re.Pattern):
        return "regex", query
    if query.startswith("re:"):
        expression = query[3:]
        if not expression:
            raise ValueError("parameter regex must be a non-empty expression")
        try:
            return "regex", re.compile(expression)
        except re.error as error:
            raise ValueError(
                f"invalid parameter regex {expression!r}: {error}"
            ) from error
    if query.startswith("glob:"):
        expression = query[5:]
        if not expression:
            raise ValueError("parameter glob must be a non-empty expression")
        return "glob", expression
    if any(character in query for character in "*?["):
        return "glob", query
    return "substring", query


def matching_names(
    names, query: ParameterQuery, *, prefer_exact=False
) -> tuple[str, ...]:
    """Match in canonical order; plain find queries retain substring semantics."""
    query = validate_query(query)
    names = tuple(names)
    if prefer_exact and isinstance(query, str) and query in names:
        return (query,)
    kind, expression = _parse(query)
    if kind == "regex":
        return tuple(name for name in names if expression.search(name) is not None)
    if kind == "glob":
        return tuple(name for name in names if fnmatchcase(name, expression))
    return tuple(name for name in names if expression in name)


def _no_match(query, *, label, within):
    scope = "" if within is None else f" within {within!r}"
    return KeyError(f"no {label} matching {query!r} was found{scope}")


def _ambiguous(query, matches, *, label, within):
    scope = "" if within is None else f" within {within!r}"
    return KeyError(
        f"{label} query {query!r} is ambiguous{scope}; matches={list(matches)}"
    )


def parameter_name(
    names, query: ParameterQuery, *, label="parameter", within=None
) -> str:
    """Resolve exactly one canonical name, including for glob and regex queries."""
    matches = matching_names(names, query, prefer_exact=True)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise _no_match(query, label=label, within=within)
    raise _ambiguous(query, matches, label=label, within=within)


def select_names(
    names,
    queries: ParameterQuery | Iterable[ParameterQuery],
    *,
    label="parameter",
    within=None,
) -> tuple[str, ...]:
    """Union selectors in canonical order, rejecting misses and ambiguous substrings."""
    names = tuple(names)
    selected = set()
    for query in normalize_queries(queries):
        matches = matching_names(names, query, prefer_exact=True)
        if not matches:
            raise _no_match(query, label=label, within=within)
        if len(matches) > 1 and _parse(query)[0] == "substring":
            raise _ambiguous(query, matches, label=label, within=within)
        selected.update(matches)
    return tuple(name for name in names if name in selected)
