"""Loss-aware file import helpers for native :mod:`dendra.models.morphology`.

SWC is deliberately parsed without NEURON so authored samples and raw type IDs
remain available to the native declaration. Neurolucida ASC uses NEURON's
mature Import3d compatibility behavior and is implemented separately below.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _SwcNode:
    node_id: int
    type_id: int
    xyz: tuple[float, float, float]
    radius: float
    parent_id: int
    source_order: int
    line_number: int

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


_SWC_TYPE_BASE = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
    5: "fork_point",
    6: "end_point",
    7: "custom",
}
_SWC_TYPE_LABELS = {
    0: ("undefined",),
    1: ("soma",),
    2: ("axon",),
    3: ("dendrite", "basal_dendrite"),
    4: ("dendrite", "apical_dendrite"),
    5: ("fork_point",),
    6: ("end_point",),
    7: ("custom",),
}


def _swc_integer(token: str, *, field: str, line_number: int) -> int:
    try:
        value = Decimal(token)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(
            f"SWC line {line_number} {field} must be an integer, got {token!r}."
        ) from error
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(
            f"SWC line {line_number} {field} must be a finite integer, got {token!r}."
        )
    return int(value)


def _swc_float(token: str, *, field: str, line_number: int) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(
            f"SWC line {line_number} {field} must be a real number, got {token!r}."
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"SWC line {line_number} {field} must be finite, got {token!r}."
        )
    return value


def _parse_swc(file_path) -> tuple[dict[int, _SwcNode], _SwcNode]:
    # utf-8-sig is identical to UTF-8 for ordinary files and also consumes the
    # optional BOM emitted by some Windows morphology pipelines.
    text = Path(file_path).read_text(encoding="utf-8-sig")
    nodes: dict[int, _SwcNode] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        fields = content.split()
        if len(fields) != 7:
            raise ValueError(
                f"SWC line {line_number} must contain exactly seven columns; "
                f"found {len(fields)}."
            )
        node_id = _swc_integer(fields[0], field="node id", line_number=line_number)
        type_id = _swc_integer(fields[1], field="type id", line_number=line_number)
        parent_id = _swc_integer(fields[6], field="parent id", line_number=line_number)
        if node_id < 0:
            raise ValueError(f"SWC line {line_number} node id must be non-negative.")
        if type_id < 0:
            raise ValueError(f"SWC line {line_number} type id must be non-negative.")
        if parent_id < -1:
            raise ValueError(
                f"SWC line {line_number} parent id must be -1 or non-negative."
            )
        if node_id in nodes:
            previous = nodes[node_id]
            raise ValueError(
                f"Duplicate SWC node id {node_id} on lines "
                f"{previous.line_number} and {line_number}."
            )
        xyz = tuple(
            _swc_float(fields[index], field="xyz"[index - 2], line_number=line_number)
            for index in range(2, 5)
        )
        radius = _swc_float(fields[5], field="radius", line_number=line_number)
        if radius <= 0.0 or not math.isfinite(2.0 * radius):
            raise ValueError(
                f"SWC line {line_number} radius must be positive and its "
                "diameter must be representable."
            )
        nodes[node_id] = _SwcNode(
            node_id=node_id,
            type_id=type_id,
            xyz=xyz,
            radius=radius,
            parent_id=parent_id,
            source_order=len(nodes),
            line_number=line_number,
        )
    if not nodes:
        raise ValueError("SWC file contains no morphology samples.")

    roots = [node for node in nodes.values() if node.parent_id == -1]
    if len(roots) != 1:
        raise ValueError(
            f"SWC must contain exactly one root sample; found {len(roots)}."
        )
    root = roots[0]
    for node in nodes.values():
        if node.parent_id == -1:
            continue
        if node.parent_id == node.node_id:
            raise ValueError(f"SWC node {node.node_id} cannot be its own parent.")
        if node.parent_id not in nodes:
            raise ValueError(
                f"SWC node {node.node_id} references unknown parent {node.parent_id}."
            )
        parent = nodes[node.parent_id]
        distance = math.dist(parent.xyz, node.xyz)
        if distance == 0.0:
            raise ValueError(
                f"SWC edge {parent.node_id}->{node.node_id} has zero spatial "
                "length, which a cable Section cannot represent."
            )
        if not math.isfinite(distance):
            raise ValueError(
                f"SWC edge {parent.node_id}->{node.node_id} has a length beyond "
                "the representable morphology range."
            )

    for start in nodes.values():
        seen: set[int] = set()
        current = start
        while current.parent_id != -1:
            if current.node_id in seen:
                raise ValueError("SWC parent references contain a cycle.")
            seen.add(current.node_id)
            current = nodes[current.parent_id]
        if current.node_id != root.node_id:
            raise ValueError("Every SWC sample must be reachable from the root.")
    return nodes, root


def _normalize_type_labels(type_labels) -> dict[int, tuple[str, ...]]:
    if type_labels is None:
        return {}
    if not isinstance(type_labels, Mapping):
        raise TypeError("type_labels must be a mapping from SWC type ids to labels.")
    normalized = {}
    for raw_type, raw_labels in type_labels.items():
        if isinstance(raw_type, bool) or not isinstance(raw_type, Integral):
            raise TypeError("type_labels keys must be non-negative integer type ids.")
        if raw_type < 0:
            raise ValueError("type_labels keys must be non-negative integer type ids.")
        if isinstance(raw_labels, str):
            labels = (raw_labels,)
        else:
            if not isinstance(raw_labels, Iterable):
                raise TypeError("Each type_labels value must be a string or iterable.")
            labels = tuple(raw_labels)
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("Imported SWC labels must be non-empty strings.")
        normalized[int(raw_type)] = tuple(labels)
    return normalized


def morphology_from_swc(
    morphology_cls,
    file_path,
    *,
    rhoa: float,
    cm: float,
    nseg: int,
    single_point_soma: str,
    type_labels,
):
    """Parse classic SWC into a native Morphology declaration."""
    from .morphology import _positive_integer

    nseg = _positive_integer(nseg, name="imported Section nseg")
    if single_point_soma not in {"sphere", "error"}:
        raise ValueError("single_point_soma must be exactly 'sphere' or 'error'.")
    extra_labels = _normalize_type_labels(type_labels)
    nodes, root = _parse_swc(file_path)
    morphology = morphology_cls(rhoa=rhoa, cm=cm)

    children: dict[int, list[_SwcNode]] = {node_id: [] for node_id in nodes}
    for node in sorted(nodes.values(), key=lambda item: item.source_order):
        if node.parent_id != -1:
            children[node.parent_id].append(node)

    counters: dict[str, int] = {}
    section_types: dict[str, int] = {}

    def section_identity(type_id: int) -> tuple[str, tuple[str, ...]]:
        base = _SWC_TYPE_BASE.get(type_id, f"type_{type_id}")
        index = counters.get(base, 0)
        counters[base] = index + 1
        name = f"{base}_{index}"
        labels = set(_SWC_TYPE_LABELS.get(type_id, ()))
        labels.add(f"swc_type_{type_id}")
        labels.update(extra_labels.get(type_id, ()))
        return name, tuple(sorted(labels))

    def follow(parent: _SwcNode, child: _SwcNode):
        type_id = child.type_id
        path = [parent, child]
        current = child
        while len(children[current.node_id]) == 1:
            following = children[current.node_id][0]
            if following.type_id != type_id:
                break
            path.append(following)
            current = following
        return type_id, path

    def path_points(path: list[_SwcNode]):
        return [(*node.xyz, node.diameter) for node in path]

    def path_positions(path: list[_SwcNode]):
        arc = [0.0]
        for first, second in zip(path, path[1:]):
            arc.append(arc[-1] + math.dist(first.xyz, second.xyz))
        return [value / arc[-1] for value in arc]

    consumed_children: set[int] = set()
    node_locations = {}
    root_path: list[_SwcNode]

    matching = [
        child for child in children[root.node_id] if child.type_id == root.type_id
    ]
    if matching:
        root_type, root_path = follow(root, matching[0])
        consumed_children.update(node.node_id for node in root_path[1:])
        name, labels = section_identity(root_type)
        root_section = morphology.section(
            name,
            points=path_points(root_path),
            nseg=nseg,
            labels=labels,
        )
        for node, position in zip(root_path, path_positions(root_path)):
            node_locations[node.node_id] = root_section.at(position)
    elif root.type_id == 1:
        if single_point_soma == "error":
            raise ValueError(
                "SWC has a one-point soma root with no soma continuation; pass "
                "single_point_soma='sphere' to use the documented cable surrogate."
            )
        x, y, z = root.xyz
        radius = root.radius
        name, labels = section_identity(root.type_id)
        root_section = morphology.section(
            name,
            points=(
                (x - radius, y, z, root.diameter),
                (x, y, z, root.diameter),
                (x + radius, y, z, root.diameter),
            ),
            nseg=nseg,
            labels=labels,
        )
        root_path = [root]
        node_locations[root.node_id] = root_section.at(0.5)
    elif children[root.node_id]:
        child_types = sorted({child.type_id for child in children[root.node_id]})
        raise ValueError(
            f"SWC root sample {root.node_id} has type {root.type_id}, but none "
            f"of its children has that type (child types: {child_types!r}). A "
            "positive-length Section cannot preserve this isolated non-soma "
            "root type without inventing geometry. Correct the root type or "
            "add a same-type continuation before importing."
        )
    else:
        raise ValueError(
            "An isolated non-soma SWC sample cannot define a positive-length "
            "cable Section."
        )
    section_types[root_section.name] = (
        root.type_id if len(root_path) == 1 else root_path[1].type_id
    )

    pending = deque()

    def schedule(path):
        for parent in path:
            for child in children[parent.node_id]:
                if child.node_id not in consumed_children:
                    pending.append((parent, child))

    schedule(root_path)
    while pending:
        parent, child = pending.popleft()
        if child.node_id in consumed_children:
            continue
        type_id, path = follow(parent, child)
        consumed_children.update(node.node_id for node in path[1:])
        name, labels = section_identity(type_id)
        section = morphology.section(
            name,
            points=path_points(path),
            nseg=nseg,
            labels=labels,
        )
        parent_location = node_locations[parent.node_id]
        section.connect(parent_location, child_end=0)
        positions = path_positions(path)
        for node, position in zip(path[1:], positions[1:]):
            node_locations[node.node_id] = section.at(position)
        section_types[section.name] = type_id
        schedule(path[1:])

    if len(consumed_children) != len(nodes) - 1:
        raise RuntimeError("Internal SWC sectionization failed to consume every edge.")
    morphology._swc_section_types = section_types
    return morphology


def _neuron_section_orientation(section) -> int:
    try:
        orientation = float(section.orientation())
    except Exception:
        from neuron import h

        orientation = float(h.section_orientation(sec=section))
    return 0 if orientation < 0.5 else 1


def _local_neuron_section_name(section) -> str:
    return str(section.name()).rsplit(".", 1)[-1]


def _coalesced_neuron_points(section, *, section_name: str):
    """Snapshot pt3d controls while retaining NEURON diameter steps.

    Fully identical adjacent controls are redundant and are coalesced. Equal
    xyz with different diameters is not duplication: it is the zero-length
    degenerate frustum that carries annular membrane area in NEURON geometry.
    """
    count = int(section.n3d())
    points = [
        (
            float(section.x3d(index)),
            float(section.y3d(index)),
            float(section.z3d(index)),
            float(section.diam3d(index)),
        )
        for index in range(count)
    ]
    if _neuron_section_orientation(section) == 1:
        points.reverse()
    coalesced = []
    for index, point in enumerate(points):
        if not all(math.isfinite(value) for value in point):
            raise ValueError(
                f"ASC Section {section_name!r} pt3d sample {index} contains "
                "a non-finite coordinate or diameter."
            )
        if point[3] <= 0.0:
            raise ValueError(
                f"ASC Section {section_name!r} pt3d sample {index} has a "
                "non-positive diameter."
            )
        if coalesced and point[:3] == coalesced[-1][:3]:
            if point[3] == coalesced[-1][3]:
                # Fully identical controls are redundant. Different diameters
                # at one coordinate are meaningful NEURON geometry: a
                # zero-length frustum carrying annular membrane area.
                continue
        coalesced.append(point)
    return tuple(coalesced)


def morphology_from_asc(
    morphology_cls,
    file_path,
    *,
    root,
    rhoa: float,
    cm: float,
    nseg: int,
):
    """Import NEURON's normalized Neurolucida cable interpretation.

    Exact duplicate controls are coalesced, while repeated coordinates with
    different diameters are preserved as abrupt diameter steps. Every retained
    Section must nevertheless have positive total centerline length.
    """
    from .morphology import _positive_integer

    nseg = _positive_integer(nseg, name="imported Section nseg")
    if root is not None and not isinstance(root, str):
        raise TypeError("root must be an exact imported Section name or None.")
    # Validate electrical defaults transactionally before touching NEURON's
    # global Section registry.
    morphology = morphology_cls(rhoa=rhoa, cm=cm)
    path = Path(file_path)
    with path.open("rb") as stream:
        if not stream.read(1):
            raise ValueError(f"ASC file {str(path)!r} is empty.")

    try:
        import neuron
        from neuron import h
    except ImportError as error:
        raise ImportError("Morphology.from_asc requires the NEURON package.") from error

    before_sec_db = set(getattr(neuron, "_sec_db", {}))
    cell = None
    reader = None
    importer = None
    try:
        h.load_file("import3d.hoc")
        reader = h.Import3d_Neurolucida3()
        reader.quiet = 1
        try:
            reader.input(str(path))
        except Exception as error:
            raise ValueError(
                f"Could not parse Neurolucida ASC file {str(path)!r}."
            ) from error
        if bool(reader.err):
            diagnostics = []
            repair_messages = getattr(reader, "b2serr", None)
            if repair_messages is not None:
                try:
                    diagnostics = [
                        " ".join(str(repair_messages.object(index).s).split())
                        for index in range(int(repair_messages.count()))
                    ]
                except Exception:
                    diagnostics = []
            if not diagnostics:
                raise ValueError(
                    f"Invalid Neurolucida ASC morphology in {str(path)!r}."
                )
            warnings.warn(
                f"NEURON repaired Neurolucida ASC morphology {str(path)!r}: "
                + "; ".join(diagnostics),
                RuntimeWarning,
                stacklevel=3,
            )
        importer = h.Import3d_GUI(reader, 0)

        class _ImportedNeurolucidaCell:
            def __init__(self):
                importer.instantiate(self)

        try:
            cell = _ImportedNeurolucidaCell()
        except Exception as error:
            raise ValueError(
                f"Could not instantiate Neurolucida ASC morphology {str(path)!r}."
            ) from error
        all_sections = list(getattr(cell, "all", ()))
        if not all_sections:
            raise ValueError(f"ASC file {str(path)!r} produced no cable Sections.")

        full_names = [str(section.name()) for section in all_sections]
        by_full_name = dict(zip(full_names, all_sections))
        local_names = {
            full_name: _local_neuron_section_name(section)
            for full_name, section in zip(full_names, all_sections)
        }
        if len(set(local_names.values())) != len(local_names):
            raise ValueError("ASC import produced colliding local Section names.")

        children: dict[str, list[str]] = {name: [] for name in full_names}
        roots = []
        for full_name, section in zip(full_names, all_sections):
            parent_segment = section.parentseg()
            if parent_segment is None:
                roots.append(full_name)
            else:
                parent_name = str(parent_segment.sec.name())
                if parent_name not in children:
                    raise ValueError(
                        f"ASC Section {local_names[full_name]!r} references an "
                        "unavailable parent Section."
                    )
                children[parent_name].append(full_name)

        def subtree_names(root_name: str) -> list[str]:
            ordered = []
            stack = [root_name]
            while stack:
                name = stack.pop()
                ordered.append(name)
                stack.extend(reversed(children[name]))
            return ordered

        if root is None:
            if len(roots) != 1:
                candidates = ", ".join(
                    f"{local_names[name]!r} ({len(subtree_names(name))} Sections)"
                    for name in roots
                )
                raise ValueError(
                    "ASC import produced disconnected roots; select one with "
                    f"root=<exact name>. Candidates: {candidates}."
                )
            selected_root = roots[0]
        else:
            matches = [name for name in roots if local_names[name] == root]
            if not matches:
                candidates = ", ".join(repr(local_names[name]) for name in roots)
                raise ValueError(
                    f"Unknown ASC root {root!r}; available roots: {candidates}."
                )
            selected_root = matches[0]
        selected_names = subtree_names(selected_root)

        declarations = {}
        for full_name in selected_names:
            section = by_full_name[full_name]
            name = local_names[full_name]
            region = name.split("[", 1)[0]
            points = _coalesced_neuron_points(section, section_name=name)
            if len(points) < 2 or not any(
                point[:3] != points[0][:3] for point in points[1:]
            ):
                raise ValueError(
                    f"ASC Section {name!r} has fewer than two distinct pt3d "
                    "coordinates after normalization and cannot define a cable."
                )
            declaration = morphology.section(
                name,
                points=points,
                nseg=nseg,
                labels=(region,),
            )
            declarations[full_name] = declaration

        selected_set = set(selected_names)
        for full_name in selected_names:
            if full_name == selected_root:
                continue
            section = by_full_name[full_name]
            parent_segment = section.parentseg()
            if parent_segment is None:
                raise ValueError(
                    f"ASC Section {local_names[full_name]!r} lost its parent."
                )
            parent_name = str(parent_segment.sec.name())
            if parent_name not in selected_set:
                raise ValueError(
                    f"ASC Section {local_names[full_name]!r} connects outside "
                    "the selected root component."
                )
            child_end = _neuron_section_orientation(section)
            declarations[full_name].connect(
                declarations[parent_name].at(float(parent_segment.x)),
                child_end=child_end,
            )
        return morphology
    finally:
        # Imported Sections are temporary compatibility objects. Drop the cell
        # and best-effort remove only private registry entries created here;
        # the returned Morphology retains no NEURON objects.
        cell = None
        importer = None
        reader = None
        sec_db = getattr(neuron, "_sec_db", None) if "neuron" in locals() else None
        if isinstance(sec_db, dict):
            for key in set(sec_db).difference(before_sec_db):
                sec_db.pop(key, None)


__all__ = ["morphology_from_asc", "morphology_from_swc"]
