"""Native morphology declarations and a canonical scalar compartment graph.

This module deliberately separates two representations:

``Morphology`` / ``Section``
    A small, mutable builder for authoring section trees. Lengths and coordinates
    are expressed in micrometers, axial resistivity in ohm-centimeters, and
    specific membrane capacitance in microfarads per square centimeter.

``CompartmentGraph``
    An immutable, solver-facing resistor tree. All node and child-indexed edge
    arrays use deterministic integer compartment IDs. NetworkX is supported as
    an interoperability view, but is not the semantic source of truth.

The first implementation uses explicit ``nseg`` values. A d-lambda
discretization policy belongs in a later layer and is intentionally not inferred.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import networkx as nx
import numpy as np

_SCHEMA_VERSION = 1
_DIAMETER_EPS = 1e-12


def _real(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    try:
        out = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}.")
    return out


def _positive(value, *, name: str) -> float:
    out = _real(value, name=name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive, got {out!r}.")
    return out


def _nonnegative(value, *, name: str) -> float:
    out = _real(value, name=name)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative, got {out!r}.")
    return out


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {out!r}.")
    return out


def _float_tuple(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    return tuple(
        _real(value, name=f"{name}[{index}]") for index, value in enumerate(values)
    )


def _same_length(reference: int, values: tuple[object, ...], *, name: str) -> None:
    if len(values) != reference:
        raise ValueError(
            f"{name} has length {len(values)}, but the compartment graph has "
            f"{reference} nodes."
        )


@dataclass(frozen=True, slots=True)
class CompartmentTopology:
    """Immutable rooted-tree topology using child-indexed parent IDs."""

    parent_index: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            isinstance(parent, (bool, np.bool_))
            or not isinstance(parent, (int, np.integer))
            for parent in self.parent_index
        ):
            raise TypeError("parent_index must contain integers.")
        parents = tuple(int(parent) for parent in self.parent_index)
        if not parents:
            raise ValueError("A compartment topology must contain at least one node.")
        object.__setattr__(self, "parent_index", parents)

        roots = [node for node, parent in enumerate(parents) if parent == -1]
        if len(roots) != 1:
            raise ValueError(
                "A scalar compartment topology must have exactly one root; "
                f"found {len(roots)}."
            )
        size = len(parents)
        for node, parent in enumerate(parents):
            if parent < -1 or parent >= size:
                raise ValueError(
                    f"parent_index[{node}]={parent} is outside [-1, {size - 1}]."
                )
            if parent == node:
                raise ValueError(f"Node {node} cannot be its own parent.")

        root = roots[0]
        for start in range(size):
            seen: set[int] = set()
            node = start
            while node != root:
                if node == -1:
                    raise ValueError(
                        f"Node {start} does not lead to the unique root {root}."
                    )
                if node in seen:
                    raise ValueError("parent_index contains a cycle.")
                seen.add(node)
                node = parents[node]

    @property
    def root(self) -> int:
        return self.parent_index.index(-1)

    @property
    def n_compartments(self) -> int:
        return len(self.parent_index)

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (parent, child)
            for child, parent in enumerate(self.parent_index)
            if parent != -1
        )


@dataclass(frozen=True, slots=True)
class CompartmentGeometry:
    """Immutable binary64 node geometry and child-indexed axial edge data."""

    length_um: tuple[float, ...]
    diameter_um: tuple[float, ...]
    area_um2: tuple[float, ...]
    volume_um3: tuple[float, ...]
    volume_i_um3: tuple[float, ...]
    volume_o_um3: tuple[float, ...]
    x_um: tuple[float, ...]
    y_um: tuple[float, ...]
    z_um: tuple[float, ...]
    rhoa_ohm_cm: tuple[float, ...]
    cm_uF_cm2: tuple[float, ...]
    edge_length_um: tuple[float, ...]
    edge_resistance_ohm: tuple[float, ...]
    edge_diff_geom_um: tuple[float, ...]

    def __post_init__(self) -> None:
        field_names = (
            "length_um",
            "diameter_um",
            "area_um2",
            "volume_um3",
            "volume_i_um3",
            "volume_o_um3",
            "x_um",
            "y_um",
            "z_um",
            "rhoa_ohm_cm",
            "cm_uF_cm2",
            "edge_length_um",
            "edge_resistance_ohm",
            "edge_diff_geom_um",
        )
        for name in field_names:
            object.__setattr__(
                self,
                name,
                _float_tuple(getattr(self, name), name=name),
            )

        size = len(self.length_um)
        if size == 0:
            raise ValueError("Compartment geometry must contain at least one node.")
        for name in field_names[1:]:
            _same_length(size, getattr(self, name), name=name)

        for node in range(size):
            _nonnegative(self.length_um[node], name=f"length_um[{node}]")
            _positive(self.diameter_um[node], name=f"diameter_um[{node}]")
            _nonnegative(self.area_um2[node], name=f"area_um2[{node}]")
            _nonnegative(self.volume_um3[node], name=f"volume_um3[{node}]")
            _nonnegative(self.volume_i_um3[node], name=f"volume_i_um3[{node}]")
            _nonnegative(self.volume_o_um3[node], name=f"volume_o_um3[{node}]")
            _positive(self.rhoa_ohm_cm[node], name=f"rhoa_ohm_cm[{node}]")
            _positive(self.cm_uF_cm2[node], name=f"cm_uF_cm2[{node}]")
            _nonnegative(self.edge_length_um[node], name=f"edge_length_um[{node}]")
            _nonnegative(
                self.edge_resistance_ohm[node],
                name=f"edge_resistance_ohm[{node}]",
            )
            _nonnegative(
                self.edge_diff_geom_um[node], name=f"edge_diff_geom_um[{node}]"
            )

    @property
    def n_compartments(self) -> int:
        return len(self.length_um)

    def array(self, name: str) -> np.ndarray:
        """Return one geometry field as a fresh NumPy ``float64`` array."""
        if name not in self.__dataclass_fields__:
            raise KeyError(f"Unknown compartment geometry field {name!r}.")
        return np.asarray(getattr(self, name), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CompartmentMetadata:
    """Immutable provenance and structural labels for graph nodes."""

    name: tuple[str, ...]
    kind: tuple[Literal["compartment", "junction"], ...]
    section_name: tuple[str | None, ...]
    segment_index: tuple[int | None, ...]
    section_x: tuple[float | None, ...]
    labels: tuple[frozenset[str], ...]

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.name)
        kinds = tuple(str(value) for value in self.kind)
        sections = tuple(
            None if value is None else str(value) for value in self.section_name
        )
        segments = []
        for index, value in enumerate(self.segment_index):
            if value is None:
                segments.append(None)
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"segment_index[{index}] must contain integers.")
            segments.append(int(value))
        segments = tuple(segments)
        xs = tuple(
            None if value is None else _real(value, name=f"section_x[{index}]")
            for index, value in enumerate(self.section_x)
        )
        normalized_labels = []
        for values in self.labels:
            if isinstance(values, str):
                values = (values,)
            try:
                normalized_labels.append(frozenset(map(str, values)))
            except TypeError as error:
                raise TypeError(
                    "Each labels entry must be a string or an iterable of labels."
                ) from error
        labels = tuple(normalized_labels)
        object.__setattr__(self, "name", names)
        object.__setattr__(self, "kind", kinds)
        object.__setattr__(self, "section_name", sections)
        object.__setattr__(self, "segment_index", segments)
        object.__setattr__(self, "section_x", xs)
        object.__setattr__(self, "labels", labels)

        size = len(names)
        if size == 0:
            raise ValueError("Compartment metadata must contain at least one node.")
        for field_name in (
            "kind",
            "section_name",
            "segment_index",
            "section_x",
            "labels",
        ):
            _same_length(size, getattr(self, field_name), name=field_name)
        for node, kind in enumerate(kinds):
            if kind not in {"compartment", "junction"}:
                raise ValueError(
                    f"kind[{node}] must be 'compartment' or 'junction', got {kind!r}."
                )
            x = xs[node]
            if x is not None and not 0.0 <= x <= 1.0:
                raise ValueError(f"section_x[{node}] must lie in [0, 1].")
            index = segments[node]
            if index is not None and index < 0:
                raise ValueError(f"segment_index[{node}] must be non-negative.")

    @property
    def n_compartments(self) -> int:
        return len(self.name)


@dataclass(frozen=True, slots=True)
class CompartmentGraph:
    """Immutable canonical scalar compartment resistor tree."""

    topology: CompartmentTopology
    geometry: CompartmentGeometry
    metadata: CompartmentMetadata
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported compartment graph schema {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}."
            )
        size = self.topology.n_compartments
        if self.geometry.n_compartments != size:
            raise ValueError("Topology and geometry node counts do not match.")
        if self.metadata.n_compartments != size:
            raise ValueError("Topology and metadata node counts do not match.")

        root = self.topology.root
        for field_name in (
            "edge_length_um",
            "edge_resistance_ohm",
            "edge_diff_geom_um",
        ):
            if getattr(self.geometry, field_name)[root] != 0.0:
                raise ValueError(f"Root {field_name} must be zero.")
        for node, parent in enumerate(self.topology.parent_index):
            kind = self.metadata.kind[node]
            if kind == "junction":
                if any(
                    value != 0.0
                    for value in (
                        self.geometry.length_um[node],
                        self.geometry.area_um2[node],
                        self.geometry.volume_um3[node],
                        self.geometry.volume_i_um3[node],
                        self.geometry.volume_o_um3[node],
                    )
                ):
                    raise ValueError(
                        f"Junction node {node} must have zero length, area, and "
                        "material volumes."
                    )
            elif self.geometry.length_um[node] <= 0.0:
                raise ValueError(
                    f"Material compartment {node} must have positive length."
                )
            elif (
                self.geometry.area_um2[node] <= 0.0
                or self.geometry.volume_um3[node] <= 0.0
            ):
                raise ValueError(
                    f"Material compartment {node} must have positive area and volume."
                )
            if parent != -1:
                if self.geometry.edge_length_um[node] <= 0.0:
                    raise ValueError(f"Edge to node {node} must have positive length.")
                if self.geometry.edge_resistance_ohm[node] <= 0.0:
                    raise ValueError(
                        f"Edge to node {node} must have positive resistance."
                    )
                if self.geometry.edge_diff_geom_um[node] <= 0.0:
                    raise ValueError(
                        f"Edge to node {node} must have positive diffusion geometry."
                    )

    @property
    def n_compartments(self) -> int:
        return self.topology.n_compartments

    def nodes_with_label(self, label: str) -> tuple[int, ...]:
        """Return deterministic compartment IDs carrying ``label``."""
        label = str(label)
        return tuple(
            node for node, labels in enumerate(self.metadata.labels) if label in labels
        )

    def to_networkx(self) -> nx.DiGraph:
        """Return a Tree-compatible NetworkX interoperability view."""
        graph = nx.DiGraph()
        graph.graph["dendra_compartment_schema"] = self.schema_version
        geometry = self.geometry
        metadata = self.metadata
        for node in range(self.n_compartments):
            attrs = {
                "L": geometry.length_um[node],
                "diam": geometry.diameter_um[node],
                "area": geometry.area_um2[node],
                "volume": geometry.volume_um3[node],
                "volume_um3": geometry.volume_um3[node],
                "volume_i": geometry.volume_i_um3[node],
                "volume_o": geometry.volume_o_um3[node],
                "Ra": geometry.rhoa_ohm_cm[node],
                "cm": geometry.cm_uF_cm2[node],
                "x": geometry.x_um[node],
                "y": geometry.y_um[node],
                "z": geometry.z_um[node],
                "name": metadata.name[node],
                "kind": metadata.kind[node],
                "labels": tuple(sorted(metadata.labels[node])),
            }
            if metadata.section_name[node] is not None:
                attrs["section_name"] = metadata.section_name[node]
            if metadata.segment_index[node] is not None:
                attrs["segment_index"] = metadata.segment_index[node]
            if metadata.section_x[node] is not None:
                attrs["section_x"] = metadata.section_x[node]
            graph.add_node(node, **attrs)

        for child, parent in enumerate(self.topology.parent_index):
            if parent == -1:
                continue
            graph.add_edge(
                parent,
                child,
                L=geometry.edge_length_um[child],
                R_ohm=geometry.edge_resistance_ohm[child],
                diff_geom_um=geometry.edge_diff_geom_um[child],
            )
        return graph

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph) -> CompartmentGraph:
        """Validate and canonicalize a Tree-compatible NetworkX graph."""
        if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
            raise TypeError("CompartmentGraph.from_networkx requires a simple DiGraph.")
        if graph.number_of_nodes() == 0:
            raise ValueError("A compartment graph must contain at least one node.")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("A compartment graph must be acyclic.")
        if not nx.is_weakly_connected(graph):
            raise ValueError("A scalar compartment graph must be connected.")
        roots = [node for node, degree in graph.in_degree if degree == 0]
        if len(roots) != 1 or any(degree > 1 for _, degree in graph.in_degree):
            raise ValueError("A scalar compartment graph must be a single rooted tree.")

        # Dense integer node IDs already are Dendra storage indices, regardless
        # of NetworkX insertion order. Arbitrary labels have no such intrinsic
        # storage meaning and are deterministically mapped in insertion order.
        source_nodes = list(graph.nodes)
        if set(source_nodes) == set(range(len(source_nodes))):
            order = list(range(len(source_nodes)))
        else:
            order = source_nodes
        canonical_id = {node: index for index, node in enumerate(order)}
        parent_index: list[int] = []
        fields: dict[str, list[float]] = {
            "length_um": [],
            "diameter_um": [],
            "area_um2": [],
            "volume_um3": [],
            "volume_i_um3": [],
            "volume_o_um3": [],
            "x_um": [],
            "y_um": [],
            "z_um": [],
            "rhoa_ohm_cm": [],
            "cm_uF_cm2": [],
            "edge_length_um": [],
            "edge_resistance_ohm": [],
            "edge_diff_geom_um": [],
        }
        names: list[str] = []
        kinds: list[Literal["compartment", "junction"]] = []
        section_names: list[str | None] = []
        segment_indices: list[int | None] = []
        section_xs: list[float | None] = []
        node_labels: list[frozenset[str]] = []

        for old_node in order:
            attrs = graph.nodes[old_node]
            predecessors = list(graph.predecessors(old_node))
            parent_index.append(
                -1 if not predecessors else canonical_id[predecessors[0]]
            )
            length = _nonnegative(attrs.get("L"), name=f"node {old_node!r} L")
            diameter = _positive(attrs.get("diam"), name=f"node {old_node!r} diam")
            rhoa = _positive(attrs.get("Ra"), name=f"node {old_node!r} Ra")
            capacitance = _positive(attrs.get("cm"), name=f"node {old_node!r} cm")
            area = attrs.get("area")
            if area is None:
                area = math.pi * diameter * length
            area = _nonnegative(area, name=f"node {old_node!r} area")
            volume = attrs.get("volume", attrs.get("volume_um3"))
            if volume is None:
                volume = math.pi * (0.5 * diameter) ** 2 * length
            volume = _nonnegative(volume, name=f"node {old_node!r} volume")
            volume_i = _nonnegative(
                attrs.get("volume_i", volume),
                name=f"node {old_node!r} volume_i",
            )
            volume_o = _nonnegative(
                attrs.get("volume_o", 0.0),
                name=f"node {old_node!r} volume_o",
            )
            fields["length_um"].append(length)
            fields["diameter_um"].append(diameter)
            fields["area_um2"].append(area)
            fields["volume_um3"].append(volume)
            fields["volume_i_um3"].append(volume_i)
            fields["volume_o_um3"].append(volume_o)
            fields["x_um"].append(_real(attrs.get("x", 0.0), name="x"))
            fields["y_um"].append(_real(attrs.get("y", 0.0), name="y"))
            fields["z_um"].append(_real(attrs.get("z", 0.0), name="z"))
            fields["rhoa_ohm_cm"].append(rhoa)
            fields["cm_uF_cm2"].append(capacitance)

            raw_kind = attrs.get("kind")
            if raw_kind is None:
                raw_kind = (
                    "junction" if length == area == volume == 0.0 else "compartment"
                )
            kinds.append(raw_kind)
            names.append(str(attrs.get("name", old_node)))
            section_names.append(attrs.get("section_name", attrs.get("section")))
            segment_indices.append(attrs.get("segment_index"))
            section_xs.append(attrs.get("section_x"))
            raw_labels = attrs.get("labels", ())
            if isinstance(raw_labels, str):
                raw_labels = (raw_labels,)
            node_labels.append(frozenset(map(str, raw_labels)))

            if not predecessors:
                fields["edge_length_um"].append(0.0)
                fields["edge_resistance_ohm"].append(0.0)
                fields["edge_diff_geom_um"].append(0.0)
                continue

            parent_old = predecessors[0]
            parent_attrs = graph.nodes[parent_old]
            edge = graph.edges[parent_old, old_node]
            parent_length = _nonnegative(
                parent_attrs.get("L"), name=f"node {parent_old!r} L"
            )
            parent_diameter = _positive(
                parent_attrs.get("diam"), name=f"node {parent_old!r} diam"
            )
            parent_rhoa = _positive(
                parent_attrs.get("Ra"), name=f"node {parent_old!r} Ra"
            )
            path_length = edge.get("L", 0.5 * (parent_length + length))
            path_length = _positive(path_length, name=f"edge to node {old_node!r} L")

            resistance = edge.get("R_ohm")
            parent_area = math.pi * (0.5 * parent_diameter) ** 2
            child_area = math.pi * (0.5 * diameter) ** 2
            inv_area = 0.5 * parent_length / parent_area + 0.5 * length / child_area
            if resistance is None:
                resistance = 1e4 * (
                    0.5 * parent_rhoa * parent_length / parent_area
                    + 0.5 * rhoa * length / child_area
                )
            resistance = _positive(resistance, name=f"edge to node {old_node!r} R_ohm")
            diff_geom = edge.get("diff_geom_um")
            if diff_geom is None:
                if inv_area > 0.0:
                    diff_geom = 1.0 / inv_area
                else:
                    # A zero-volume junction needs explicit edge geometry for
                    # exact diffusion. This resistance-derived value is the
                    # same compatibility fallback used by Tree.
                    diff_geom = 0.5 * (parent_rhoa + rhoa) * 1e4 / resistance
            fields["edge_length_um"].append(path_length)
            fields["edge_resistance_ohm"].append(resistance)
            fields["edge_diff_geom_um"].append(
                _positive(diff_geom, name=f"edge to node {old_node!r} diff_geom_um")
            )

        return cls(
            topology=CompartmentTopology(tuple(parent_index)),
            geometry=CompartmentGeometry(
                **{name: tuple(values) for name, values in fields.items()}
            ),
            metadata=CompartmentMetadata(
                name=tuple(names),
                kind=tuple(kinds),
                section_name=tuple(section_names),
                segment_index=tuple(segment_indices),
                section_x=tuple(section_xs),
                labels=tuple(node_labels),
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class Section:
    """One named cable section in a native :class:`Morphology`."""

    name: str
    nseg: int
    L: float
    diam: float | None
    points: tuple[tuple[float, float, float, float], ...]
    rhoa: float
    cm: float
    labels: frozenset[str]
    _owner: Morphology = field(repr=False, compare=False)

    @property
    def is_pt3d(self) -> bool:
        return len(self.points) > 2 or self.diam is None

    def at(self, x: float) -> SectionLocation:
        """Return an immutable normalized location on this section."""
        x = _real(x, name=f"location on section {self.name!r}")
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                "Section locations must lie in the closed interval [0, 1]."
            )
        return SectionLocation(self, x)

    def connect(self, parent: SectionLocation, *, child_end: int = 0) -> Section:
        """Connect one endpoint of this section to a parent location."""
        self._owner.connect(parent, self, child_end=child_end)
        return self


@dataclass(frozen=True, slots=True)
class SectionLocation:
    """A normalized location on a native section."""

    section: Section
    x: float

    def __post_init__(self) -> None:
        x = _real(self.x, name=f"location on section {self.section.name!r}")
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                "Section locations must lie in the closed interval [0, 1]."
            )
        object.__setattr__(self, "x", x)


@dataclass(frozen=True, slots=True)
class _Connection:
    parent_name: str
    parent_x: float
    child_name: str
    child_end: int


class Morphology:
    """Mutable section-tree declaration compiled into a :class:`CompartmentGraph`.

    Parameters
    ----------
    rhoa : float, optional
        Default intracellular resistivity in Ω·cm.
    cm : float, optional
        Default specific membrane capacitance in µF/cm².
    """

    def __init__(self, *, rhoa: float = 100.0, cm: float = 1.0):
        self.rhoa = _positive(rhoa, name="default rhoa")
        self.cm = _positive(cm, name="default cm")
        self._sections: dict[str, Section] = {}
        self._connections: dict[str, _Connection] = {}

    @property
    def sections(self) -> tuple[Section, ...]:
        return tuple(self._sections.values())

    def section(
        self,
        name: str,
        *,
        L: float | None = None,
        diam: float | None = None,
        points: Sequence[Sequence[float]] | None = None,
        nseg: int = 1,
        rhoa: float | None = None,
        cm: float | None = None,
        labels: str | Iterable[str] = (),
    ) -> Section:
        """Declare a stylized cylinder or a pt3d tapered section.

        A stylized section requires ``L`` and scalar ``diam``. A pt3d section
        requires at least two ``(x, y, z, diameter)`` points and derives its
        length from the centerline. The two geometry forms cannot be mixed.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Section names must be non-empty strings.")
        if name in self._sections:
            raise ValueError(f"A section named {name!r} already exists.")
        nseg = _positive_integer(nseg, name=f"section {name!r} nseg")
        if isinstance(labels, str):
            labels = (labels,)
        normalized_labels = frozenset({name, *(str(label) for label in labels)})

        if points is None:
            if L is None or diam is None:
                raise ValueError("Stylized sections require both L and diam.")
            length = _positive(L, name=f"section {name!r} L")
            diameter = _positive(diam, name=f"section {name!r} diam")
            normalized_points = (
                (0.0, 0.0, 0.0, diameter),
                (0.0, 0.0, length, diameter),
            )
        else:
            if L is not None or diam is not None:
                raise ValueError("pt3d sections cannot also specify L or diam.")
            normalized_points = _normalize_points(points, section_name=name)
            length = _polyline_arclength(normalized_points)[-1]
            diameter = None

        section = Section(
            name=name,
            nseg=nseg,
            L=length,
            diam=diameter,
            points=normalized_points,
            rhoa=self.rhoa if rhoa is None else _positive(rhoa, name=f"{name} rhoa"),
            cm=self.cm if cm is None else _positive(cm, name=f"{name} cm"),
            labels=normalized_labels,
            _owner=self,
        )
        self._sections[name] = section
        return section

    def connect(
        self,
        parent: SectionLocation,
        child: Section | SectionLocation,
        *,
        child_end: int | None = None,
    ) -> None:
        """Connect a child endpoint to a parent location.

        ``parent`` must be produced by ``parent_section.at(x)``. ``child`` may
        be a Section plus an explicit ``child_end``, or ``child.at(0/1)``.
        Interior child locations are rejected because they do not define a
        section-tree orientation.
        """
        if not isinstance(parent, SectionLocation):
            raise TypeError(
                "parent must be a SectionLocation returned by Section.at()."
            )
        if isinstance(child, SectionLocation):
            if child_end is not None:
                raise ValueError(
                    "Do not pass child_end when child is a SectionLocation."
                )
            if child.x not in (0.0, 1.0):
                raise ValueError("A child connection must use section endpoint 0 or 1.")
            child_section = child.section
            child_end = int(child.x)
        elif isinstance(child, Section):
            child_section = child
            if child_end is None:
                raise ValueError("child_end must be 0 or 1 when child is a Section.")
        else:
            raise TypeError("child must be a Section or SectionLocation.")
        if isinstance(child_end, (bool, np.bool_)) or child_end not in (0, 1):
            raise ValueError("child_end must be exactly 0 or 1.")
        if parent.section._owner is not self or child_section._owner is not self:
            raise ValueError("Both connected sections must belong to this Morphology.")
        if parent.section is child_section:
            raise ValueError("A section cannot be connected to itself.")
        if child_section.name in self._connections:
            raise ValueError(f"Section {child_section.name!r} already has a parent.")

        # Reject cycles at mutation time while retaining full validation at compile.
        ancestor = parent.section.name
        while ancestor in self._connections:
            if ancestor == child_section.name:
                raise ValueError("Section connections must not contain a cycle.")
            ancestor = self._connections[ancestor].parent_name
        if ancestor == child_section.name:
            raise ValueError("Section connections must not contain a cycle.")
        self._connections[child_section.name] = _Connection(
            parent_name=parent.section.name,
            parent_x=parent.x,
            child_name=child_section.name,
            child_end=int(child_end),
        )

    def compile(self) -> CompartmentGraph:
        """Validate and discretize the declared section tree in binary64."""
        if not self._sections:
            raise ValueError("Cannot compile an empty Morphology.")
        roots = [name for name in self._sections if name not in self._connections]
        if len(roots) != 1:
            raise ValueError(
                "A scalar Morphology must have exactly one root section; "
                f"found {len(roots)} ({roots!r})."
            )
        if len(self._connections) != len(self._sections) - 1:
            raise ValueError("Every non-root section must have exactly one parent.")

        root_name = roots[0]
        section_children: dict[str, list[str]] = {name: [] for name in self._sections}
        for connection in self._connections.values():
            section_children[connection.parent_name].append(connection.child_name)
        visited: set[str] = set()
        stack = [root_name]
        while stack:
            name = stack.pop()
            if name in visited:
                raise ValueError("Section connections contain a cycle.")
            visited.add(name)
            stack.extend(reversed(section_children[name]))
        if visited != set(self._sections):
            raise ValueError("All sections must be reachable from the root section.")

        return _compile_morphology(self, root_name)


def _normalize_points(
    points: Sequence[Sequence[float]], *, section_name: str
) -> tuple[tuple[float, float, float, float], ...]:
    if len(points) < 2:
        raise ValueError("A pt3d section requires at least two points.")
    normalized: list[tuple[float, float, float, float]] = []
    for index, point in enumerate(points):
        if len(point) != 4:
            raise ValueError(
                f"pt3d point {index} on {section_name!r} must be (x, y, z, diameter)."
            )
        x, y, z = (
            _real(point[axis], name=f"{section_name} point {index} coordinate")
            for axis in range(3)
        )
        diameter = _positive(point[3], name=f"{section_name} point {index} diameter")
        normalized.append((x, y, z, diameter))
    for index, (first, second) in enumerate(zip(normalized, normalized[1:])):
        distance = math.dist(first[:3], second[:3])
        if distance <= 0.0:
            raise ValueError(
                f"Consecutive pt3d points {index} and {index + 1} on "
                f"{section_name!r} must have distinct coordinates."
            )
    return tuple(normalized)


def _polyline_arclength(
    points: tuple[tuple[float, float, float, float], ...],
) -> np.ndarray:
    arc = np.zeros(len(points), dtype=np.float64)
    for index in range(1, len(points)):
        arc[index] = arc[index - 1] + math.dist(
            points[index - 1][:3], points[index][:3]
        )
    return arc


def _interpolate_point(
    section: Section, s_um: float
) -> tuple[float, float, float, float]:
    arc = _polyline_arclength(section.points)
    s_um = min(max(float(s_um), 0.0), float(arc[-1]))
    index = min(int(np.searchsorted(arc, s_um, side="right")) - 1, len(arc) - 2)
    index = max(index, 0)
    span = float(arc[index + 1] - arc[index])
    fraction = 0.0 if span == 0.0 else (s_um - float(arc[index])) / span
    first = section.points[index]
    second = section.points[index + 1]
    return tuple(
        float(first[axis] + fraction * (second[axis] - first[axis]))
        for axis in range(4)
    )


def _section_integrals(
    section: Section, s0: float, s1: float
) -> tuple[float, float, float]:
    """Return lateral area, volume, and inverse-area integral for an interval."""
    if s1 < s0:
        s0, s1 = s1, s0
    s0 = max(0.0, min(float(s0), section.L))
    s1 = max(0.0, min(float(s1), section.L))
    if s1 <= s0:
        return 0.0, 0.0, 0.0
    arc = _polyline_arclength(section.points)
    boundaries = [s0]
    boundaries.extend(float(value) for value in arc[1:-1] if s0 < value < s1)
    boundaries.append(s1)
    area = 0.0
    volume = 0.0
    inv_area = 0.0
    for lo, hi in zip(boundaries, boundaries[1:]):
        d0 = max(_interpolate_point(section, lo)[3], _DIAMETER_EPS)
        d1 = max(_interpolate_point(section, hi)[3], _DIAMETER_EPS)
        length = hi - lo
        r0 = 0.5 * d0
        r1 = 0.5 * d1
        area += math.pi * (r0 + r1) * math.hypot(length, r1 - r0)
        volume += math.pi * length * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
        inv_area += 4.0 * length / (math.pi * d0 * d1)
    return float(area), float(volume), float(inv_area)


@dataclass(slots=True)
class _MutableNode:
    length_um: float
    diameter_um: float
    area_um2: float
    volume_um3: float
    x_um: float
    y_um: float
    z_um: float
    rhoa: float
    cm: float
    name: str
    kind: Literal["compartment", "junction"]
    section_name: str | None
    segment_index: int | None
    section_x: float | None
    labels: frozenset[str]


def _edge_attrs(section: Section, x0: float, x1: float) -> dict[str, float]:
    _, _, inv_area = _section_integrals(section, x0 * section.L, x1 * section.L)
    path_length = abs(x1 - x0) * section.L
    return {
        "L": path_length,
        "R_ohm": section.rhoa * 1e4 * inv_area,
        "inv_area": inv_area,
    }


def _combine_edges(
    first: dict[str, float], second: dict[str, float]
) -> dict[str, float]:
    return {
        "L": first["L"] + second["L"],
        "R_ohm": first["R_ohm"] + second["R_ohm"],
        "inv_area": first["inv_area"] + second["inv_area"],
    }


def _compile_morphology(morphology: Morphology, root_name: str) -> CompartmentGraph:
    sections = list(morphology.sections)
    nodes: dict[int, _MutableNode] = {}
    material_id: dict[tuple[str, int], int] = {}
    next_node = 0
    for section in sections:
        for index in range(section.nseg):
            x0 = index / section.nseg
            x1 = (index + 1) / section.nseg
            x_center = (index + 0.5) / section.nseg
            center = _interpolate_point(section, x_center * section.L)
            area, volume, _ = _section_integrals(
                section, x0 * section.L, x1 * section.L
            )
            material_id[(section.name, index)] = next_node
            nodes[next_node] = _MutableNode(
                length_um=section.L / section.nseg,
                diameter_um=center[3],
                area_um2=area,
                volume_um3=volume,
                x_um=center[0],
                y_um=center[1],
                z_um=center[2],
                rhoa=section.rhoa,
                cm=section.cm,
                name=f"{section.name}({x_center:.12g})",
                kind="compartment",
                section_name=section.name,
                segment_index=index,
                section_x=x_center,
                labels=section.labels,
            )
            next_node += 1

    endpoint_tokens = [
        ("endpoint", section.name, endpoint)
        for section in sections
        for endpoint in (0, 1)
    ]
    union_find = nx.utils.UnionFind(endpoint_tokens + list(nodes))
    for connection in morphology._connections.values():
        child_token = ("endpoint", connection.child_name, connection.child_end)
        parent_section = morphology._sections[connection.parent_name]
        if connection.parent_x in (0.0, 1.0):
            parent_target: object = (
                "endpoint",
                connection.parent_name,
                int(connection.parent_x),
            )
        else:
            parent_index = min(
                int(connection.parent_x * parent_section.nseg),
                parent_section.nseg - 1,
            )
            parent_target = material_id[(connection.parent_name, parent_index)]
        union_find.union(child_token, parent_target)

    groups: dict[object, list[object]] = {}
    for element in [*endpoint_tokens, *nodes]:
        groups.setdefault(union_find[element], []).append(element)

    endpoint_node: dict[tuple[str, str, int], int] = {}
    provisional_junctions: list[int] = []
    for members in groups.values():
        centers = [member for member in members if isinstance(member, int)]
        endpoints = [member for member in members if not isinstance(member, int)]
        if centers:
            graph_node = centers[0]
        else:
            graph_node = next_node
            next_node += 1
            provisional_junctions.append(graph_node)
            _, section_name, endpoint = endpoints[0]
            section = morphology._sections[section_name]
            point = _interpolate_point(section, endpoint * section.L)
            nodes[graph_node] = _MutableNode(
                length_um=0.0,
                diameter_um=point[3],
                area_um2=0.0,
                volume_um3=0.0,
                x_um=point[0],
                y_um=point[1],
                z_um=point[2],
                rhoa=section.rhoa,
                cm=section.cm,
                name="branchpoint.pending",
                kind="junction",
                section_name=None,
                segment_index=None,
                section_x=None,
                # A retained electrical junction belongs to no material
                # Section. Including it in a Section label would cause Slice
                # operations to insert mechanisms or materials onto a
                # zero-area node unexpectedly.
                labels=frozenset(),
            )
        for endpoint in endpoints:
            endpoint_node[endpoint] = graph_node

    resistor = nx.Graph()
    resistor.add_nodes_from(nodes)

    def add_edge(first: int, second: int, attrs: dict[str, float]) -> None:
        if first == second or resistor.has_edge(first, second):
            raise ValueError(
                "Section connections collapsed into a loop or parallel edge."
            )
        if attrs["L"] <= 0.0 or attrs["R_ohm"] <= 0.0 or attrs["inv_area"] <= 0.0:
            raise ValueError("Compiled axial edges must have positive finite geometry.")
        resistor.add_edge(first, second, **attrs)

    for section in sections:
        for left in range(section.nseg - 1):
            right = left + 1
            add_edge(
                material_id[(section.name, left)],
                material_id[(section.name, right)],
                _edge_attrs(
                    section,
                    (left + 0.5) / section.nseg,
                    (right + 0.5) / section.nseg,
                ),
            )
        for endpoint in (0, 1):
            index = 0 if endpoint == 0 else section.nseg - 1
            center_x = (index + 0.5) / section.nseg
            add_edge(
                material_id[(section.name, index)],
                endpoint_node[("endpoint", section.name, endpoint)],
                _edge_attrs(section, center_x, float(endpoint)),
            )

    for junction in provisional_junctions:
        if junction not in resistor:
            continue
        degree = resistor.degree(junction)
        if degree <= 1:
            resistor.remove_node(junction)
            nodes.pop(junction)
        elif degree == 2:
            first, second = list(resistor.neighbors(junction))
            attrs = _combine_edges(
                dict(resistor.edges[first, junction]),
                dict(resistor.edges[junction, second]),
            )
            resistor.remove_node(junction)
            nodes.pop(junction)
            add_edge(first, second, attrs)

    if not nx.is_tree(resistor):
        raise ValueError("Compiled morphology is not one resistor tree.")
    root_old = material_id[(root_name, 0)]

    # Stable breadth-first IDs make every parent precede its children and avoid
    # depending on NetworkX's hash ordering.
    bfs_order: list[int] = []
    parents_old: dict[int, int | None] = {root_old: None}
    queue: deque[int] = deque([root_old])
    while queue:
        parent = queue.popleft()
        bfs_order.append(parent)
        for child in sorted(resistor.neighbors(parent)):
            if child == parents_old[parent]:
                continue
            parents_old[child] = parent
            queue.append(child)
    canonical = {old: new for new, old in enumerate(bfs_order)}

    retained_junctions = [node for node in provisional_junctions if node in resistor]
    for index, old_node in enumerate(retained_junctions):
        nodes[old_node].name = f"branchpoint.{index}"

    parent_index: list[int] = []
    geometry: dict[str, list[float]] = {
        "length_um": [],
        "diameter_um": [],
        "area_um2": [],
        "volume_um3": [],
        "volume_i_um3": [],
        "volume_o_um3": [],
        "x_um": [],
        "y_um": [],
        "z_um": [],
        "rhoa_ohm_cm": [],
        "cm_uF_cm2": [],
        "edge_length_um": [],
        "edge_resistance_ohm": [],
        "edge_diff_geom_um": [],
    }
    names: list[str] = []
    kinds: list[Literal["compartment", "junction"]] = []
    section_names: list[str | None] = []
    segment_indices: list[int | None] = []
    section_xs: list[float | None] = []
    labels: list[frozenset[str]] = []
    for old_node in bfs_order:
        node = nodes[old_node]
        parent_old = parents_old[old_node]
        parent_index.append(-1 if parent_old is None else canonical[parent_old])
        geometry["length_um"].append(node.length_um)
        geometry["diameter_um"].append(node.diameter_um)
        geometry["area_um2"].append(node.area_um2)
        geometry["volume_um3"].append(node.volume_um3)
        geometry["volume_i_um3"].append(node.volume_um3)
        geometry["volume_o_um3"].append(0.0)
        geometry["x_um"].append(node.x_um)
        geometry["y_um"].append(node.y_um)
        geometry["z_um"].append(node.z_um)
        geometry["rhoa_ohm_cm"].append(node.rhoa)
        geometry["cm_uF_cm2"].append(node.cm)
        if parent_old is None:
            geometry["edge_length_um"].append(0.0)
            geometry["edge_resistance_ohm"].append(0.0)
            geometry["edge_diff_geom_um"].append(0.0)
        else:
            edge = resistor.edges[parent_old, old_node]
            geometry["edge_length_um"].append(edge["L"])
            geometry["edge_resistance_ohm"].append(edge["R_ohm"])
            geometry["edge_diff_geom_um"].append(1.0 / edge["inv_area"])
        names.append(node.name)
        kinds.append(node.kind)
        section_names.append(node.section_name)
        segment_indices.append(node.segment_index)
        section_xs.append(node.section_x)
        labels.append(node.labels)

    return CompartmentGraph(
        topology=CompartmentTopology(tuple(parent_index)),
        geometry=CompartmentGeometry(
            **{name: tuple(values) for name, values in geometry.items()}
        ),
        metadata=CompartmentMetadata(
            name=tuple(names),
            kind=tuple(kinds),
            section_name=tuple(section_names),
            segment_index=tuple(segment_indices),
            section_x=tuple(section_xs),
            labels=tuple(labels),
        ),
    )


__all__ = [
    "CompartmentGeometry",
    "CompartmentGraph",
    "CompartmentMetadata",
    "CompartmentTopology",
    "Morphology",
    "Section",
    "SectionLocation",
]
