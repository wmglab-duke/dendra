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
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType
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
    """Immutable binary64 node geometry and child-indexed axial edge data.

    For geometry produced by :meth:`Morphology.compile`, ``diameter_um`` is
    the arclength-mean diameter of each material compartment, matching
    NEURON's segment-diameter convention. It is descriptive metadata rather
    than a cylindrical approximation: ``area_um2``, ``volume_um3``, and the
    axial edge fields retain the exact compiled tapered-cable integrals.
    """

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
    """Immutable provenance and structural labels for graph nodes.

    ``name`` is the generated per-node display/search name, for example
    ``"soma(0.5)"``. ``section_name`` is the unique authored
    :class:`Section` identity from which a material compartment was compiled.
    ``labels`` contains reusable structural region tags; for native Sections it
    also includes that Section's name. Retained junction nodes have no source
    Section and therefore use ``None`` provenance and an empty label set.
    """

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

    def path_ordered(self) -> CompartmentGraph:
        """Return an equivalent graph ordered from one cable end to the other.

        The generic unbranched :class:`dendra.Cable` backend requires its final
        tensor dimension to follow physical path order.  A canonical rooted
        tree need not already have that ordering: a valid unbranched morphology
        can have its rooted node in the middle of the physical cable.  This
        method validates the stronger material-only path contract and returns a
        new immutable snapshot whose parent array is ``(-1, 0, 1, ...)``.

        Orientation is deterministic.  If the original root is a physical end
        it remains the first node; otherwise the lower-numbered end is first.
        Provenance, authored Section coordinates, labels, and exact edge
        geometry are retained while node IDs are remapped to storage order.

        Raises
        ------
        ValueError
            If any node is an algebraic junction or the underlying undirected
            graph is not one simple path.
        """
        size = self.n_compartments
        junctions = [
            node
            for node, kind in enumerate(self.metadata.kind)
            if kind != "compartment"
        ]
        if junctions:
            raise ValueError(
                "An unbranched Cable may contain only material compartments; "
                f"found junction node(s) {junctions}."
            )

        adjacency: list[list[int]] = [[] for _ in range(size)]
        edge_source: dict[frozenset[int], int] = {}
        for child, parent in enumerate(self.topology.parent_index):
            if parent == -1:
                continue
            adjacency[parent].append(child)
            adjacency[child].append(parent)
            edge_source[frozenset((parent, child))] = child

        if size == 1:
            order = [0]
        else:
            degrees = [len(neighbors) for neighbors in adjacency]
            if any(degree > 2 for degree in degrees):
                raise ValueError(
                    "A Cable morphology must be unbranched; its compartment "
                    "graph contains a node with degree greater than two."
                )
            ends = [node for node, degree in enumerate(degrees) if degree == 1]
            if len(ends) != 2:
                raise ValueError(
                    "A Cable morphology must be one connected simple path with "
                    "exactly two ends."
                )
            root = self.topology.root
            start = root if root in ends else min(ends)
            order = []
            previous = -1
            current = start
            while current != -1:
                order.append(current)
                following = [node for node in adjacency[current] if node != previous]
                if len(following) > 1:
                    raise ValueError("A Cable morphology cannot branch.")
                previous, current = current, following[0] if following else -1
            if len(order) != size:
                raise ValueError("A Cable morphology must be one connected path.")

        node_geometry_fields = (
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
        )
        geometry = {
            name: tuple(getattr(self.geometry, name)[node] for node in order)
            for name in node_geometry_fields
        }
        edge_fields = (
            "edge_length_um",
            "edge_resistance_ohm",
            "edge_diff_geom_um",
        )
        for name in edge_fields:
            values = [0.0]
            for left, right in zip(order, order[1:]):
                source = edge_source[frozenset((left, right))]
                values.append(getattr(self.geometry, name)[source])
            geometry[name] = tuple(values)

        metadata_fields = (
            "name",
            "kind",
            "section_name",
            "segment_index",
            "section_x",
            "labels",
        )
        metadata = {
            name: tuple(getattr(self.metadata, name)[node] for node in order)
            for name in metadata_fields
        }
        return CompartmentGraph(
            topology=CompartmentTopology(tuple([-1, *range(size - 1)])),
            geometry=CompartmentGeometry(**geometry),
            metadata=CompartmentMetadata(**metadata),
            schema_version=self.schema_version,
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
    """One named cable section in a native :class:`Morphology`.

    ``name`` is the Section's unique authored identity. ``labels`` is its
    read-only set of structural region tags and always includes ``name``.
    Attribute assignment is disabled; :meth:`update` performs a validated,
    transactional edit through the owning Morphology while preserving this
    object's identity.
    """

    name: str
    nseg: int
    L: float
    diam: float | None
    points: tuple[tuple[float, float, float, float], ...]
    rhoa: float
    cm: float
    labels: frozenset[str]
    _owner: Morphology = field(repr=False, compare=False)
    _location_controls: tuple[tuple[float, int], ...] = field(
        default=(), repr=False, compare=False
    )

    @property
    def is_pt3d(self) -> bool:
        return len(self.points) > 2 or self.diam is None

    def at(self, x: float) -> SectionLocation:
        """Return an immutable normalized location on this section."""
        self._owner._resolve_section(self)
        x = _real(x, name=f"location on section {self.name!r}")
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                "Section locations must lie in the closed interval [0, 1]."
            )
        return SectionLocation(self, x)

    def update(
        self,
        *,
        L: float | None = None,
        diam: float | None = None,
        points: Sequence[Sequence[float]] | None = None,
        nseg: int | None = None,
        rhoa: float | None = None,
        cm: float | None = None,
        labels: str | Iterable[str] | None = None,
    ) -> Section:
        """Transactionally update this authored Section and return it.

        ``rhoa``, ``cm``, and ``nseg`` are whole-Section properties. Stylized
        Sections may update ``L`` and ``diam``; pt3d Sections replace geometry
        through ``points`` because their length and diameter profile are
        derived. Use ``section.at(x).update(diam=...)`` for one pt3d diameter
        control point. Supplying ``points`` can promote a stylized Section to
        pt3d. ``labels`` replaces the explicit set, with ``name`` re-added.

        Existing compiled graphs and instantiated models are independent
        snapshots. Compile or construct a new model to observe an update.
        ``None`` means unchanged for every argument.
        """
        return self._owner.update_section(
            self,
            L=L,
            diam=diam,
            points=points,
            nseg=nseg,
            rhoa=rhoa,
            cm=cm,
            labels=labels,
        )

    def connect(self, parent: SectionLocation, *, child_end: int = 0) -> Section:
        """Connect one endpoint of this section to a parent location."""
        self._owner.connect(parent, self, child_end=child_end)
        return self


@dataclass(frozen=True, slots=True)
class SectionLocation:
    """An immutable normalized selector on a native Section's current geometry."""

    section: Section
    x: float

    def __post_init__(self) -> None:
        x = _real(self.x, name=f"location on section {self.section.name!r}")
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                "Section locations must lie in the closed interval [0, 1]."
            )
        object.__setattr__(self, "x", x)

    def update(self, *, diam: float) -> Section:
        """Edit or insert a pt3d diameter control point at this location.

        Diameter remains linearly interpolated between neighboring authored
        samples. The centerline path, ``nseg``, and electrical properties are
        unchanged; an inserted point may re-express the same path length within
        binary64 roundoff. If coordinate precision cannot represent a distinct
        control point, the update is rejected. A location containing an abrupt
        diameter discontinuity has two authored diameter limits and is therefore
        ambiguous; replace the complete Section ``points`` to edit either side.
        Location-scoped ``rhoa`` and ``cm`` are not defined by the current
        whole-Section electrical contract.

        Returns
        -------
        Section
            The canonical owning Section after a successful update.
        """
        return self.section._owner._update_section_location(self, diam=diam)


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
        self.rhoa = rhoa
        self.cm = cm
        self._sections: dict[str, Section] = {}
        self._connections: dict[str, _Connection] = {}
        self._swc_section_types: dict[str, int] = {}

    @property
    def rhoa(self) -> float:
        """Validated default axial resistivity for subsequently declared Sections."""
        return self._rhoa

    @rhoa.setter
    def rhoa(self, value: float) -> None:
        self._rhoa = _positive(value, name="default rhoa")

    @property
    def cm(self) -> float:
        """Validated specific-capacitance default for later Sections."""
        return self._cm

    @cm.setter
    def cm(self, value: float) -> None:
        self._cm = _positive(value, name="default cm")

    @property
    def sections(self) -> tuple[Section, ...]:
        return tuple(self._sections.values())

    @property
    def swc_section_types(self) -> Mapping[str, int]:
        """Read-only imported SWC type provenance keyed by generated Section.

        Manually authored Sections and non-SWC imports have no entry. Pass this
        mapping explicitly to :meth:`to_swc` or :meth:`write_swc` to retain raw
        type IDs; export never guesses types from names or labels.
        """
        return MappingProxyType(self._swc_section_types)

    def _resolve_section(self, section: str | Section) -> Section:
        if isinstance(section, str):
            try:
                return self._sections[section]
            except KeyError as error:
                raise KeyError(f"Unknown Section name {section!r}.") from error
        if not isinstance(section, Section):
            raise TypeError("section must be a Section or an exact Section name.")
        if section._owner is not self:
            raise ValueError("The Section belongs to a different Morphology.")
        if self._sections.get(section.name) is not section:
            raise ValueError(
                f"Section {section.name!r} is not the canonical Section registered "
                "with this Morphology."
            )
        return section

    def update_section(
        self,
        section: str | Section,
        *,
        L: float | None = None,
        diam: float | None = None,
        points: Sequence[Sequence[float]] | None = None,
        nseg: int | None = None,
        rhoa: float | None = None,
        cm: float | None = None,
        labels: str | Iterable[str] | None = None,
    ) -> Section:
        """Transactionally update a named Section while preserving its identity.

        Parameters use the same units and validation as :meth:`section`.
        ``rhoa``, ``cm``, and ``nseg`` apply to the complete Section. Stylized
        Sections may update ``L`` and ``diam``. A complete pt3d centerline or
        diameter-profile replacement uses ``points``; use
        :meth:`SectionLocation.update` for one local diameter control point.
        Passing ``points`` for a stylized Section promotes it to pt3d; the
        reverse pt3d-to-stylized conversion is unsupported by this update API.
        ``labels`` replaces the explicit labels, while the Section name remains
        automatic. ``None`` means unchanged for every optional argument.

        The canonical Section object and all existing :class:`SectionLocation`
        references remain valid. Connections and declaration order are
        preserved. Previously compiled graphs and instantiated models are
        immutable snapshots and are not changed by this authoring edit.

        Returns
        -------
        Section
            The same canonical Section object after a successful update.
        """
        current = self._resolve_section(section)

        if points is not None and (L is not None or diam is not None):
            raise ValueError("Do not combine points with L or diam in an update.")
        if points is not None:
            new_points = _normalize_points(points, section_name=current.name)
            new_length = _positive(
                _polyline_arclength(new_points)[-1],
                name=f"section {current.name!r} derived L",
            )
            new_diameter = None
            new_location_controls: tuple[tuple[float, int], ...] = ()
        elif current.is_pt3d:
            if L is not None or diam is not None:
                raise ValueError(
                    "A pt3d Section derives L and its diameter profile from "
                    "points; replace points to update its geometry."
                )
            new_points = current.points
            new_length = current.L
            new_diameter = None
            new_location_controls = current._location_controls
        else:
            new_length = (
                current.L
                if L is None
                else _positive(L, name=f"section {current.name!r} L")
            )
            new_diameter = (
                current.diam
                if diam is None
                else _positive(diam, name=f"section {current.name!r} diam")
            )
            new_points = (
                (0.0, 0.0, 0.0, new_diameter),
                (0.0, 0.0, new_length, new_diameter),
            )
            new_location_controls = ()

        new_nseg = (
            current.nseg
            if nseg is None
            else _positive_integer(nseg, name=f"section {current.name!r} nseg")
        )
        new_rhoa = (
            current.rhoa
            if rhoa is None
            else _positive(rhoa, name=f"section {current.name!r} rhoa")
        )
        new_cm = (
            current.cm
            if cm is None
            else _positive(cm, name=f"section {current.name!r} cm")
        )

        if labels is None:
            new_labels = current.labels
        else:
            if isinstance(labels, str):
                labels = (labels,)
            explicit_labels = frozenset(str(label) for label in labels)
            conflicting_names = sorted(
                explicit_labels.intersection(self._sections).difference({current.name})
            )
            if conflicting_names:
                conflicts = ", ".join(repr(value) for value in conflicting_names)
                raise ValueError(
                    f"Section {current.name!r} labels conflict with Section "
                    f"name(s): {conflicts}. Section names are reserved and "
                    "cannot be explicit labels on another Section."
                )
            new_labels = explicit_labels | {current.name}

        replacements = {
            "nseg": new_nseg,
            "L": new_length,
            "diam": new_diameter,
            "points": new_points,
            "rhoa": new_rhoa,
            "cm": new_cm,
            "labels": new_labels,
            "_location_controls": new_location_controls,
        }
        for field_name, value in replacements.items():
            object.__setattr__(current, field_name, value)
        return current

    def _update_section_location(
        self, location: SectionLocation, *, diam: float
    ) -> Section:
        if not isinstance(location, SectionLocation):
            raise TypeError("location must be returned by Section.at().")
        section = self._resolve_section(location.section)
        if not section.is_pt3d:
            raise ValueError(
                "Localized diameter updates require a pt3d Section; replace "
                "points explicitly to promote a stylized Section."
            )
        diameter = _positive(diam, name=f"diameter at {section.name!r}({location.x!r})")

        controls = dict(section._location_controls)
        controlled_index = controls.get(location.x)
        if controlled_index is not None:
            if not 0 <= controlled_index < len(section.points):
                raise RuntimeError(
                    f"Section {section.name!r} has invalid local-control provenance."
                )
            new_points = list(section.points)
            existing = new_points[controlled_index]
            new_points[controlled_index] = (*existing[:3], diameter)
            object.__setattr__(section, "points", tuple(new_points))
            return section

        arc = _polyline_arclength(section.points)
        sample_x = [float(value / arc[-1]) for value in arc]
        sample_x[0] = 0.0
        sample_x[-1] = 1.0
        exact_samples = [
            index for index, value in enumerate(sample_x) if value == location.x
        ]
        exact_sample_set = set(exact_samples)
        if any(
            index in exact_sample_set
            and index + 1 in exact_sample_set
            and section.points[index][:3] == section.points[index + 1][:3]
            for index in range(len(section.points) - 1)
        ):
            raise ValueError(
                f"Location {section.name!r}({location.x!r}) contains an abrupt "
                "diameter discontinuity with multiple authored controls; a "
                "localized diameter update is ambiguous. Replace the complete "
                "Section points instead."
            )
        if exact_samples:
            matched = exact_samples[-1]
            new_points = list(section.points)
            existing = new_points[matched]
            new_points[matched] = (*existing[:3], diameter)
            controls[location.x] = matched
            object.__setattr__(section, "points", tuple(new_points))
            object.__setattr__(section, "_location_controls", tuple(controls.items()))
            return section

        point = _interpolate_point(section, location.x * section.L)
        insertion = int(np.searchsorted(sample_x, location.x, side="right"))
        segment = min(max(insertion - 1, 0), len(section.points) - 2)
        tolerance = _location_representation_tolerance(section, segment, point)

        # Interpolation may round a requested position onto an existing sample.
        # Alias it only when that sample also represents the requested normalized
        # arclength within the documented binary64 precision bound.
        adjacent = {
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(section.points)
        }
        matching = [
            index for index in adjacent if section.points[index][:3] == point[:3]
        ]
        if matching:
            matched = min(matching, key=lambda index: abs(sample_x[index] - location.x))
            if abs(sample_x[matched] - location.x) > tolerance:
                raise _unrepresentable_location(section, location.x)
            new_points = list(section.points)
            existing = new_points[matched]
            new_points[matched] = (*existing[:3], diameter)
            controls[location.x] = matched
            object.__setattr__(section, "points", tuple(new_points))
            object.__setattr__(section, "_location_controls", tuple(controls.items()))
            return section

        candidate = list(section.points)
        candidate.insert(insertion, (*point[:3], diameter))
        try:
            normalized_points = _normalize_points(candidate, section_name=section.name)
        except ValueError as error:
            if "must have distinct coordinates" not in str(error):
                raise
            raise _unrepresentable_location(section, location.x) from None

        candidate_arc = _polyline_arclength(normalized_points)
        candidate_length = _positive(
            candidate_arc[-1], name=f"section {section.name!r} derived L"
        )
        if not (
            candidate_arc[insertion - 1]
            < candidate_arc[insertion]
            < candidate_arc[insertion + 1]
        ):
            raise _unrepresentable_location(section, location.x)
        realized_x = float(candidate_arc[insertion] / candidate_length)
        if abs(realized_x - location.x) > tolerance:
            raise _unrepresentable_location(section, location.x)

        shifted_controls = {
            x: index + (index >= insertion) for x, index in controls.items()
        }
        shifted_controls[location.x] = insertion

        # Commit only after geometry and representability validation succeeds.
        object.__setattr__(section, "points", normalized_points)
        object.__setattr__(section, "L", candidate_length)
        object.__setattr__(section, "diam", None)
        object.__setattr__(
            section, "_location_controls", tuple(shifted_controls.items())
        )
        return section

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

        Parameters
        ----------
        name : str
            Unique, non-empty authored Section identity. The exact name is used
            for connection bookkeeping and compiled ``section_name`` provenance,
            and is automatically included among the Section's labels. Section
            names are reserved: an explicit label on another Section cannot use
            the same string.
        L : float, optional
            Stylized-cylinder length in µm. Required with ``diam`` and mutually
            exclusive with ``points``.
        diam : float, optional
            Stylized-cylinder diameter in µm. Required with ``L`` and mutually
            exclusive with ``points``.
        points : sequence of (x, y, z, diameter), optional
            At least two pt3d samples in µm. Length is derived from centerline
            arclength. Consecutive samples may share coordinates only when their
            diameters differ; this represents an abrupt, zero-arclength diameter
            step with an annular membrane surface but no length, volume, or
            axial resistance. The complete Section must retain positive
            centerline length.
        nseg : int, optional
            Positive number of computational compartments. Default is 1.
        rhoa : float, optional
            Section intracellular resistivity in Ω·cm. Defaults to the owning
            Morphology's value.
        cm : float, optional
            Section specific membrane capacitance in µF/cm². Defaults to the
            owning Morphology's value.
        labels : str or iterable of str, optional
            Reusable structural region tags applied to every material
            compartment compiled from this Section. Labels may be shared by
            multiple Sections to form union selections. The Section ``name`` is
            added automatically and need not be supplied here.

        Returns
        -------
        Section
            The declaration owned by this Morphology. Its fields are read-only;
            use :meth:`Section.update` for validated authoring edits.

        Notes
        -----
        After constructing a :class:`~dendra.models.tree.Tree` or
        :class:`~dendra.models.core.Cable`, collision-free labels that are safe
        Python identifiers become population-owned Slice attributes. All labels
        remain available in the compiled graph's metadata even when an
        attribute cannot be installed.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Section names must be non-empty strings.")
        if name in self._sections:
            raise ValueError(f"A section named {name!r} already exists.")
        nseg = _positive_integer(nseg, name=f"section {name!r} nseg")
        if isinstance(labels, str):
            labels = (labels,)
        explicit_labels = frozenset(str(label) for label in labels)

        existing_names = self._sections.keys()
        conflicting_names = sorted(explicit_labels.intersection(existing_names))
        if conflicting_names:
            conflicts = ", ".join(repr(value) for value in conflicting_names)
            raise ValueError(
                f"Section {name!r} labels conflict with existing Section "
                f"name(s): {conflicts}. Section names are reserved and cannot "
                "be explicit labels on another Section."
            )

        conflicting_sections = sorted(
            section.name
            for section in self._sections.values()
            if name != section.name and name in section.labels
        )
        if conflicting_sections:
            owners = ", ".join(repr(value) for value in conflicting_sections)
            raise ValueError(
                f"Section name {name!r} conflicts with an explicit label on "
                f"existing Section(s): {owners}. Section names are reserved "
                "and cannot be reused as another Section's label."
            )

        normalized_labels = explicit_labels | {name}

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
            length = _positive(
                _polyline_arclength(normalized_points)[-1],
                name=f"section {name!r} derived L",
            )
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
        parent_section = self._resolve_section(parent.section)
        child_section = self._resolve_section(child_section)
        if parent_section is child_section:
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
        """Validate and discretize the declared Section tree in binary64.

        Pt3d compartments retain exact tapered-frustum membrane area, volume,
        and axial inverse-area integrals. An abrupt repeated-coordinate
        diameter step contributes its NEURON-compatible annular membrane area
        but no centerline length, volume, or axial resistance; a step exactly
        on a compartment boundary belongs to the lower-x compartment. Material
        ``diameter_um`` metadata is the arclength mean over the compartment,
        not a point sample used to reconstruct these exact integrals.
        """
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

    @classmethod
    def from_swc(
        cls,
        file_path: str | PathLike[str],
        *,
        rhoa: float = 100.0,
        cm: float = 1.0,
        nseg: int = 1,
        single_point_soma: Literal["sphere", "error"] = "sphere",
        type_labels: Mapping[int, str | Iterable[str]] | None = None,
    ) -> Morphology:
        """Load a classic SWC node tree as a native Morphology declaration.

        The pure-Python loader preserves finite xyz samples, radii, rooted
        connectivity, and raw structure type IDs without applying NEURON's
        d-lambda discretization or repair heuristics. Maximal root-away paths of
        one downstream SWC type become pt3d Sections; branches and type changes
        start a new Section at the shared parent sample.

        Parameters
        ----------
        file_path : path-like
            UTF-8 SWC file containing the seven classic columns ``id type x y
            z radius parent``. Blank lines, full-line comments, and inline
            ``#`` comments are accepted. Rows may appear in any order.
        rhoa : float, optional
            Positive default axial resistivity in Ω·cm. SWC does not carry this
            electrical property. Default is ``100``.
        cm : float, optional
            Positive default specific capacitance in µF/cm². SWC does not carry
            this electrical property. Default is ``1``.
        nseg : int, optional
            Positive compartment count assigned independently to every
            generated Section. SWC samples describe geometry, not numerical
            compartments. Default is ``1``.
        single_point_soma : {'sphere', 'error'}, optional
            A common type-1 root has no positive-length cable of its own.
            ``"sphere"`` creates the NEURON-compatible three-point x-axis cable
            surrogate centered on the sample with length and diameter ``2r``;
            its lateral area equals a sphere's surface area, though its volume
            remains cylindrical. ``"error"`` rejects this ambiguity. Default
            is ``"sphere"``.
        type_labels : mapping, optional
            Additional shared labels for selected integer SWC type IDs. Every
            imported Section always receives ``swc_type_<id>`` plus Dendra's
            conventional semantic labels where defined.

        Returns
        -------
        Morphology
            Native, editable pt3d Sections with deterministic collision-free
            names such as ``soma_0`` and ``basal_dendrite_0``.

        Notes
        -----
        Exactly one root is required. Duplicate IDs, unknown parents, cycles,
        non-positive radii, non-finite values, and zero-length edges are
        rejected rather than repaired silently. Structural integer IDs are
        parsed exactly rather than through binary64. A non-soma root must have a
        same-type child so its type can belong to a positive-length Section;
        an isolated root-type discontinuity is rejected instead of being
        silently retyped. SWC cannot preserve original Section
        names/boundaries, labels, ``nseg``, ``rhoa``, or ``cm``.
        :attr:`swc_section_types` retains raw type provenance for an explicit
        re-export via ``section_types=morphology.swc_section_types``.
        """
        from .morphology_io import morphology_from_swc

        return morphology_from_swc(
            cls,
            file_path,
            rhoa=rhoa,
            cm=cm,
            nseg=nseg,
            single_point_soma=single_point_soma,
            type_labels=type_labels,
        )

    @classmethod
    def from_asc(
        cls,
        file_path: str | PathLike[str],
        *,
        root: str | None = None,
        rhoa: float = 100.0,
        cm: float = 1.0,
        nseg: int = 1,
    ) -> Morphology:
        """Load NEURON's normalized interpretation of a Neurolucida ASC file.

        Neurolucida V3 text includes contours, nested cable trees, spines,
        markers, colors, arbitrary properties, and repair conventions. Dendra
        therefore delegates this compatibility grammar to NEURON's mature
        ``Import3d_Neurolucida3`` reader, then immediately snapshots the chosen
        cable tree into editable native pt3d Sections. The returned object holds
        no NEURON references.

        Parameters
        ----------
        file_path : path-like
            Neurolucida V3 ASCII morphology.
        root : str, optional
            Exact generated root Section name, such as ``"soma[0]"``. A file
            with several disconnected cable trees is rejected unless one root
            is selected explicitly; candidates are listed in the error.
        rhoa : float, optional
            Positive axial resistivity in Ω·cm for every imported Section.
            ASC does not define this electrical value. Default is ``100``.
        cm : float, optional
            Positive specific capacitance in µF/cm² for every imported Section.
            ASC does not define this electrical value. Default is ``1``.
        nseg : int, optional
            Positive compartment count assigned to every imported Section.
            No d-lambda policy is inferred from geometry. Default is ``1``.

        Returns
        -------
        Morphology
            A native snapshot with NEURON-generated names such as ``soma[0]``,
            ``axon[3]``, ``dend[8]``, and ``apic[1]`` and corresponding shared
            structural labels.

        Notes
        -----
        Centerlines, diameters, topology, logical attachment locations, and the
        selected cable component are preserved after NEURON normalization.
        Exact duplicate pt3d samples are coalesced. Consecutive samples at the
        same coordinate with different diameters are preserved as abrupt
        zero-length diameter steps, including their annular membrane area. A
        Section with no positive centerline length after coalescing is rejected.
        Source formatting, comments, colors, markers, properties,
        spines, original trace identifiers, and soma contour boundaries are not
        retained. NEURON may repair or approximate source geometry and stores
        pt3d values at its own precision. When NEURON makes its recoverable
        logical connection from an outlying main branch to the nearest soma,
        Dendra emits one :class:`RuntimeWarning` containing NEURON's repair
        diagnostics before snapshotting that topology. This method intentionally
        complements the pure-Python, sample-preserving :meth:`from_swc` loader.
        """
        from .morphology_io import morphology_from_asc

        return morphology_from_asc(
            cls,
            file_path,
            root=root,
            rhoa=rhoa,
            cm=cm,
            nseg=nseg,
        )

    def to_swc(
        self,
        *,
        section_types: Mapping[str, int] | None = None,
        default_type: int = 0,
        connection_tolerance_um: float = 1e-9,
    ) -> str:
        """Serialize the authored centerline tree in SWC format.

        Parameters
        ----------
        section_types : mapping of str to int, optional
            SWC structure type for each exactly named Section. Unlisted
            Sections use ``default_type``. No type is inferred from Section
            names or labels.
        default_type : int, optional
            Non-negative SWC structure type for unlisted Sections. The default
            is ``0`` (undefined).
        connection_tolerance_um : float, optional
            Maximum spatial separation, in micrometers, allowed between a
            parent attachment and its connected child endpoint. SWC combines
            topology and geometry, so larger discrepancies are rejected. The
            default is ``1e-9``.

        Returns
        -------
        str
            Deterministic SWC text ending in a newline.

        Notes
        -----
        SWC export describes the authored Section centerlines, not the
        compartment graph produced by :meth:`compile`. It therefore does not
        encode ``nseg``, ``rhoa``, ``cm``, Section names, or labels. Importing
        the result may choose a different electrical discretization.

        A shared connection is represented by one SWC sample carrying the
        parent's type and radius. Classic SWC cannot retain distinct radii on
        the two sides of that single junction. It also cannot represent the
        annular membrane surface of a repeated-coordinate diameter step, so
        export rejects a Morphology containing such a discontinuity rather
        than silently discarding geometry or writing a zero-length SWC edge.
        """
        return _morphology_to_swc(
            self,
            section_types=section_types,
            default_type=default_type,
            connection_tolerance_um=connection_tolerance_um,
        )

    def write_swc(
        self,
        file_path: str | PathLike[str],
        *,
        section_types: Mapping[str, int] | None = None,
        default_type: int = 0,
        connection_tolerance_um: float = 1e-9,
    ) -> None:
        """Validate and write the authored centerline tree as UTF-8 SWC.

        The complete document is serialized before the destination is opened,
        so a morphology validation error cannot truncate an existing file.
        See :meth:`to_swc` for the export contract and keyword arguments.
        """
        text = self.to_swc(
            section_types=section_types,
            default_type=default_type,
            connection_tolerance_um=connection_tolerance_um,
        )
        Path(file_path).write_text(text, encoding="utf-8", newline="\n")

    def plot(
        self,
        *,
        view: Literal["x", "y", "z"] = "y",
        color_by: str = "section",
        highlight: str | Iterable[str] | None = None,
        show_points: bool = True,
        show_connections: bool = True,
        show_compartments: bool = False,
        show_orientation: bool = True,
        annotate_sections: bool = False,
        annotate_connections: bool = False,
        connection_tolerance_um: float = 1e-9,
        diameter_scale: float = 0.35,
        min_linewidth: float = 0.75,
        legend: bool = True,
        cmap: str = "viridis",
        ax: object | None = None,
        figsize: tuple[float, float] = (8, 8),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Plot an orthographic view of the authored morphology.

        The plot is a read-only view of the current :class:`Section`
        declarations. It shows authored centerlines and optional authoring
        diagnostics without compiling a :class:`CompartmentGraph` or
        constructing a simulation model.

        Parameters
        ----------
        view : {"x", "y", "z"}, optional
            Coordinate axis to look along (and therefore omit from the plot).
            For example, ``view="y"`` displays x against z. Default is
            ``"y"``.
        color_by : str, optional
            Attribute used to color Section paths. ``"section"`` assigns a
            deterministic categorical color to each exact Section identity.
            Scalar morphology fields such as ``"length"``, ``"diameter"``,
            ``"nseg"``, ``"rhoa"``, and ``"cm"`` use a continuous color
            scale. Default is ``"section"``.
        highlight : str or iterable of str, optional
            Exact Section name or shared structural label, or several of
            either. Matching Sections remain prominent and other Sections are
            muted. ``None`` highlights the complete morphology.
        show_points : bool, optional
            Show authored pt3d control points. Stylized Sections expose their
            two canonical local endpoints. Default is ``True``.
        show_connections : bool, optional
            Show electrical attachment markers and spatial-gap diagnostics.
            These glyphs describe connectivity; a line between spatially
            separated endpoints is not an authored physical cable. Default is
            ``True``.
        show_compartments : bool, optional
            Overlay the compartment centers implied by each Section's
            ``nseg`` declaration. This is an authoring preview and does not
            compile the solver graph. Default is ``False``.
        show_orientation : bool, optional
            Draw increasing authored Section x along a local authored span and
            mark the ``0`` or ``1`` child endpoint facing each parent. A child
            attached through ``child_end=1`` retains its authored x direction.
            Default is ``True``.
        annotate_sections : bool, optional
            Annotate paths with exact Section names. Default is ``False``.
        annotate_connections : bool, optional
            Annotate attachments with parent x and child endpoint values.
            Default is ``False``.
        connection_tolerance_um : float, optional
            Spatial separation, in µm, at or below which connected attachment
            points are classified as within visualization tolerance rather
            than as a true spatial gap. Nonzero separation remains visible.
            This neither changes electrical connectivity nor applies the
            stricter SWC export contract. Default is ``1e-9``.
        diameter_scale : float, optional
            Scale converting authored diameter in µm to relative screen-space
            line width. Rendered thickness is an expressive encoding, not a
            geometrically to-scale tube radius. Default is ``0.35``.
        min_linewidth : float, optional
            Minimum visible centerline width in display points. Default is
            ``0.75``.
        legend : bool, optional
            Show a categorical legend or numeric color key as appropriate.
            Default is ``True``.
        cmap : str, optional
            Matplotlib colormap name used for continuous coloring. Default is
            ``"viridis"``.
        ax : object, optional
            Existing Matplotlib axes to draw into. A new figure and axes are
            created when omitted.
        figsize : tuple of float, optional
            Width and height in inches for a newly created figure. Ignored
            when ``ax`` is supplied. Default is ``(8, 8)``.
        dpi : int, optional
            Resolution of a newly created figure. Ignored when ``ax`` is
            supplied. Default is ``150``.
        title : str, optional
            Figure title. ``None`` uses the renderer's descriptive default.

        Returns
        -------
        object
            The Matplotlib ``(Figure, Axes)`` pair returned by the renderer.

        Notes
        -----
        Relative geometry is displayed exactly as authored. In particular,
        :meth:`connect` does not translate or rotate a child, and stylized
        Sections use local canonical coordinates, so electrically connected
        paths may overlap or remain spatially separated. At extreme binary64
        magnitudes, a numerical origin or scale may be applied to the display
        coordinates; every such transform is printed on the axes. Tapered
        authored spans may be subdivided visually to interpolate width/color,
        but this does not alter the Morphology. A repeated-coordinate diameter
        step is retained in the scene data, but a centerline drawing cannot
        display its annular surface; use :meth:`plot_shape` or
        :meth:`plot_shape_3d` to see that shoulder. The method never calls
        ``show``; display, save, or further customize the returned figure
        explicitly.
        """
        from .morphology_visualization import plot_morphology

        return plot_morphology(
            self,
            view=view,
            color_by=color_by,
            highlight=highlight,
            show_points=show_points,
            show_connections=show_connections,
            show_compartments=show_compartments,
            show_orientation=show_orientation,
            annotate_sections=annotate_sections,
            annotate_connections=annotate_connections,
            connection_tolerance_um=connection_tolerance_um,
            diameter_scale=diameter_scale,
            min_linewidth=min_linewidth,
            legend=legend,
            cmap=cmap,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

    def plot_3d(
        self,
        *,
        color_by: str = "section",
        highlight: str | Iterable[str] | None = None,
        show_points: bool = True,
        show_connections: bool = True,
        show_compartments: bool = False,
        show_orientation: bool = True,
        annotate_sections: bool = False,
        annotate_connections: bool = False,
        connection_tolerance_um: float = 1e-9,
        diameter_scale: float = 0.35,
        min_linewidth: float = 0.75,
        legend: bool = True,
        cmap: str = "viridis",
        ax: object | None = None,
        figsize: tuple[float, float] = (9, 8),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Plot the authored morphology in three spatial dimensions.

        This method has the same authoring-layer semantics as :meth:`plot`,
        but preserves x, y, and z. It is useful for checking pt3d placement,
        taper, orientation, and whether electrically attached points are also
        spatially coherent.

        Parameters
        ----------
        color_by : str, optional
            Attribute used to color Section paths. ``"section"`` is
            categorical; ``"length"``, ``"diameter"``, ``"nseg"``,
            ``"rhoa"``, and ``"cm"`` provide continuous encodings. Default
            is ``"section"``.
        highlight : str or iterable of str, optional
            Exact Section name or shared label, or several of either, to keep
            prominent while muting non-matches. Default is ``None``.
        show_points : bool, optional
            Show authored pt3d controls or stylized canonical endpoints.
            Default is ``True``.
        show_connections : bool, optional
            Show electrical attachment markers and spatial-gap diagnostics.
            Diagnostic connectors are not physical centerline segments.
            Default is ``True``.
        show_compartments : bool, optional
            Overlay centers implied by authored ``nseg`` values without
            compiling a solver graph. Default is ``False``.
        show_orientation : bool, optional
            Draw increasing authored Section x on a local span and mark each
            connected child endpoint as ``0`` or ``1``. Default is ``True``.
        annotate_sections : bool, optional
            Annotate paths with exact Section names. Default is ``False``.
        annotate_connections : bool, optional
            Annotate electrical attachments with their normalized locations.
            Default is ``False``.
        connection_tolerance_um : float, optional
            Visual coincidence tolerance for connected coordinates, in µm.
            It does not alter connectivity, compilation, or SWC validation.
            Default is ``1e-9``.
        diameter_scale : float, optional
            Conversion from diameter in µm to relative display width. The
            result is not a metrically exact solid tube. Default is ``0.35``.
        min_linewidth : float, optional
            Minimum centerline width in display points. Default is ``0.75``.
        legend : bool, optional
            Show the Section legend or numeric color key. Default is ``True``.
        cmap : str, optional
            Matplotlib colormap name for numeric coloring. Default is
            ``"viridis"``.
        ax : object, optional
            Existing three-dimensional Matplotlib axes. A new 3D axes is
            created when omitted.
        figsize : tuple of float, optional
            New figure size in inches. Ignored when ``ax`` is supplied.
            Default is ``(9, 8)``.
        dpi : int, optional
            New figure resolution. Ignored when ``ax`` is supplied. Default is
            ``150``.
        title : str, optional
            Figure title. ``None`` uses the renderer's descriptive default.

        Returns
        -------
        object
            The Matplotlib ``(Figure, 3D Axes)`` pair from the renderer.

        Notes
        -----
        The current declaration is read without mutation or compilation.
        Existing :class:`CompartmentGraph`, :class:`~dendra.models.tree.Tree`,
        and :class:`~dendra.models.core.Cable` snapshots are not consulted.
        Extreme-coordinate display origins/scales and extreme color-value
        divisors are disclosed on the figure. As in :meth:`plot`, a diagnostic
        centerline cannot show the annular surface of a zero-length diameter
        step; the shape views render it as a shoulder. No display call is made
        implicitly.
        """
        from .morphology_visualization import plot_morphology_3d

        return plot_morphology_3d(
            self,
            color_by=color_by,
            highlight=highlight,
            show_points=show_points,
            show_connections=show_connections,
            show_compartments=show_compartments,
            show_orientation=show_orientation,
            annotate_sections=annotate_sections,
            annotate_connections=annotate_connections,
            connection_tolerance_um=connection_tolerance_um,
            diameter_scale=diameter_scale,
            min_linewidth=min_linewidth,
            legend=legend,
            cmap=cmap,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

    def plot_shape(
        self,
        *,
        view: Literal["x", "y", "z"] = "y",
        diameter_scale: float = 1.0,
        radial_segments: int = 8,
        connection_tolerance_um: float = 1e-9,
        legend: bool | Literal["auto"] = "auto",
        show_axes: bool = False,
        ax: object | None = None,
        figsize: tuple[float, float] = (8, 8),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Draw a quiet orthographic view of the physical cable envelope.

        Unlike :meth:`plot`, this renderer uses authored diameters as radii in
        morphology coordinate units rather than as screen-space line widths.
        It constructs closed, linearly tapered tubes around the authored pt3d
        centerlines, colors them by structural morphology family, and omits
        diagnostic points, arrows, connection glyphs, annotations, and titles
        by default. Indexed Sections such as ``dend[0]`` and ``dend[1]`` share
        the canonical ``dend`` color used by :func:`~dendra.models.visualization.vis_2d`.

        Parameters
        ----------
        view : {'x', 'y', 'z'}, optional
            Axis viewed along. ``"y"`` produces an x-z projection. Default is
            ``"y"``.
        diameter_scale : float, optional
            Positive multiplier for every physical diameter. ``1`` preserves
            authored proportions; larger values intentionally exaggerate thin
            cables for display. Default is ``1``.
        radial_segments : int, optional
            Number of sides in each circular tube ring. Larger values make a
            smoother surface at greater rendering cost. Must be at least 3;
            default is ``8``.
        connection_tolerance_um : float, optional
            Spatial-gap threshold in µm. A single warning is emitted when an
            electrical connection exceeds it; no artificial bridge is drawn.
            Default is ``1e-9``.
        legend : bool or {'auto'}, optional
            Show structural color families. ``"auto"`` shows the legend for
            at most 16 families, avoiding an unusable key while retaining a
            compact legend for large imported cells. Default is ``"auto"``.
        show_axes : bool, optional
            Show physical coordinate axes. Default is ``False``.
        ax : object, optional
            Existing two-dimensional Matplotlib axes. A new figure and axes are
            created when omitted.
        figsize : tuple of float, optional
            New figure size in inches. Ignored when ``ax`` is supplied.
        dpi : int, optional
            New figure resolution. Ignored when ``ax`` is supplied.
        title : str, optional
            Optional explicit title. The default adds no title.

        Returns
        -------
        object
            The Matplotlib ``(Figure, Axes)`` pair.

        Notes
        -----
        The surface is a faithful tapered-cable envelope of the authored
        Section data, not a histological reconstruction or a boolean union at
        branches. It does not use ``nseg`` or compile the Morphology. Connected
        stylized Sections retain their canonical local z-axis and may overlap;
        use coherent pt3d coordinates for a meaningful whole-cell shape. Two
        coincident controls with different diameters are rendered as concentric
        rings and the corresponding annular shoulder, without inventing cable
        length. Exact Section provenance remains available on the rendered
        collection even when several Sections share one family color. The
        method never mutates the declaration or calls ``show``.
        """
        from .morphology_visualization import plot_morphology_shape

        return plot_morphology_shape(
            self,
            view=view,
            diameter_scale=diameter_scale,
            radial_segments=radial_segments,
            connection_tolerance_um=connection_tolerance_um,
            legend=legend,
            show_axes=show_axes,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

    def plot_shape_3d(
        self,
        *,
        diameter_scale: float = 1.0,
        radial_segments: int = 8,
        connection_tolerance_um: float = 1e-9,
        legend: bool | Literal["auto"] = "auto",
        show_axes: bool = False,
        ax: object | None = None,
        figsize: tuple[float, float] = (9, 8),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Draw the authored physical cable envelope as a quiet 3-D surface.

        This is the rotatable three-dimensional counterpart of
        :meth:`plot_shape`. Diameters remain in morphology coordinate units;
        its structural family colors and compact legend are identical to the
        2-D view. Diagnostic metadata and connection glyphs belong to
        :meth:`plot_3d` and :meth:`inspect` instead.

        Parameters are the same as :meth:`plot_shape`, except that no projection
        ``view`` is required and ``figsize`` defaults to ``(9, 8)``. A supplied
        axes must be a Matplotlib 3-D axes. The method returns ``(Figure, Axes)``,
        does not mutate the Morphology, and never calls ``show``.
        """
        from .morphology_visualization import plot_morphology_shape_3d

        return plot_morphology_shape_3d(
            self,
            diameter_scale=diameter_scale,
            radial_segments=radial_segments,
            connection_tolerance_um=connection_tolerance_um,
            legend=legend,
            show_axes=show_axes,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

    def plot_topology(
        self,
        *,
        highlight: str | Iterable[str] | None = None,
        interactive: bool = False,
        node_scale: float = 8.0,
        min_node_size: float = 12.0,
        max_node_size: float | None = 240.0,
        branchpoint_scale: float = 1.35,
        min_branchpoint_size: float = 24.0,
        max_branchpoint_size: float | None = 300.0,
        junction_size: float = 52.0,
        edge_scale: float = 0.12,
        min_edge_width: float = 0.35,
        max_edge_width: float | None = 3.0,
        node_size: float | None = None,
        branchpoint_size: float | None = None,
        edge_width: float | None = None,
        legend: bool = True,
        ax: object | None = None,
        figsize: tuple[float, float] = (10, 7),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Draw the flat, coordinate-independent compartment connectivity graph.

        Every material compartment is a node colored by its exact authored
        Section name. Retained zero-area algebraic junctions are separate
        neutral nodes. Edges are the canonical axial connections after
        endpoint removal, degree-two junction collapse, and
        interior-attachment resolution. The graph contains no per-node text;
        optional hover cards expose the underlying node metadata. Layout depth
        and spacing are schematic rather than morphological distances.

        Parameters
        ----------
        highlight : str or iterable of str, optional
            Exact Section name or shared label, or several of either, to keep
            prominent while muting non-matching material compartments.
            Unlabelled algebraic junctions remain neutral structural context.
            Default is ``None``.
        interactive : bool, optional
            Enable Matplotlib motion-event hover cards without adding a Dendra
            runtime dependency. Hovering over a material node reports its
            generated name, Section, segment, x, labels, length,
            arclength-mean diameter, integrated area/volume, ``rhoa``, ``cm``,
            and spatial center. Select an event-capable backend before creating
            the figure:
            for example, a GUI backend or ``%matplotlib widget`` with the
            optional ``ipympl`` package (available through
            ``dendra[jupyter]``). Restart the complete Jupyter server after
            installation, not only its kernel. A separately installed kernel
            and server both need compatible ipympl components. Static inline
            and saved raster output cannot react to pointer motion; an inline
            notebook call emits an actionable warning. Default is ``False``.
        node_scale : float, optional
            Scale from material-compartment arclength-mean diameter in µm to
            marker area in display points². Default is ``8``.
        min_node_size, max_node_size : float or None, optional
            Visible bounds for material-compartment marker areas. The default
            bounds are ``12`` and ``240`` points²; ``None`` removes the upper
            bound.
        branchpoint_scale : float, optional
            Additional marker-area factor for a material forking compartment.
            Default is ``1.35``.
        min_branchpoint_size, max_branchpoint_size : float or None, optional
            Visible bounds for material-fork marker areas after scaling. The
            defaults are ``24`` and ``300`` points²; ``None`` removes the
            upper bound.
        junction_size : float, optional
            Fixed marker area for a zero-area algebraic junction. Junctions
            have no physical diameter to encode. Default is ``52`` points².
        edge_scale : float, optional
            Scale from the mean of the material endpoints' arclength-mean
            diameters in µm to edge width in display points. A junction
            endpoint is excluded because its stored geometry is not material.
            Default is ``0.12``.
        min_edge_width, max_edge_width : float or None, optional
            Visible bounds for scaled edge widths. The defaults are ``0.35``
            and ``3`` points; ``None`` removes the upper bound.
        node_size, branchpoint_size, edge_width : float or None, optional
            Explicit fixed-size compatibility overrides. ``node_size`` fixes
            ordinary material markers; ``branchpoint_size`` fixes both
            material-fork and algebraic-junction markers; ``edge_width`` fixes
            every edge. ``None`` (the default) uses diameter scaling.
        legend : bool, optional
            Show exact Section-name colors and structural glyphs. Default is
            ``True``.
        ax : object, optional
            Existing Matplotlib axes. A new figure and axes are created when
            omitted.
        figsize : tuple of float, optional
            New figure size in inches. Ignored when ``ax`` is supplied.
            Default is ``(10, 7)``.
        dpi : int, optional
            New figure resolution. Ignored when ``ax`` is supplied. Default is
            ``150``.
        title : str, optional
            Figure title. ``None`` uses the renderer's descriptive default.

        Returns
        -------
        object
            The Matplotlib ``(Figure, Axes)`` pair from the renderer.

        Notes
        -----
        A complete Morphology uses the exact topology produced by
        :meth:`compile`. A partially authored forest is compiled one connected
        component at a time, retaining exact per-component compartment
        semantics without requiring a single root.

        Any node with at least three neighbors in the undirected compartment
        graph is a graph branchpoint. A Section-colored diamond distinguishes
        a material forking compartment; a neutral ``X`` distinguishes a
        retained zero-area algebraic junction. This invariant definition does
        not misclassify an arbitrary degree-two solver root. Junction hover
        cards deliberately omit copied storage fields that are not material
        properties; their marker size and incident-edge scaling likewise never
        use those copied fields. Scaling changes display geometry only, while
        highlighting changes alpha only. The title's branchpoint count includes
        both glyph classes. The method does not mutate the declaration or call
        ``show``.
        """
        from .morphology_visualization import plot_morphology_topology

        return plot_morphology_topology(
            self,
            highlight=highlight,
            interactive=interactive,
            node_scale=node_scale,
            min_node_size=min_node_size,
            max_node_size=max_node_size,
            branchpoint_scale=branchpoint_scale,
            min_branchpoint_size=min_branchpoint_size,
            max_branchpoint_size=max_branchpoint_size,
            junction_size=junction_size,
            edge_scale=edge_scale,
            min_edge_width=min_edge_width,
            max_edge_width=max_edge_width,
            node_size=node_size,
            branchpoint_size=branchpoint_size,
            edge_width=edge_width,
            legend=legend,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

    def plot_section_topology(
        self,
        *,
        color_by: str = "section",
        highlight: str | Iterable[str] | None = None,
        show_parameters: bool = True,
        show_labels: bool = True,
        show_compartments: bool = True,
        annotate_connections: bool = True,
        connection_tolerance_um: float = 1e-9,
        legend: bool = True,
        cmap: str = "viridis",
        ax: object | None = None,
        figsize: tuple[float, float] = (10, 7),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Draw the authored Section tree/forest as a labelled schematic.

        This authoring-level companion to :meth:`plot_topology` uses one box per
        Section and can display authoring parameters, structural labels,
        ``nseg`` rulers, connection locations, and spatial-gap diagnostics.
        It does not compile the compartment graph and therefore describes the
        declaration rather than solver-node connectivity.

        Parameters
        ----------
        color_by : str, optional
            ``"section"``, ``"length"``, ``"diameter"``, ``"nseg"``,
            ``"rhoa"``, or ``"cm"``. Default is ``"section"``.
        highlight : str or iterable of str, optional
            Exact Section name or shared label, or several of either, to keep
            prominent while muting non-matches. Default is ``None``.
        show_parameters : bool, optional
            Include geometry, length, ``rhoa``, and ``cm`` in each Section box.
            Default is ``True``.
        show_labels : bool, optional
            Include shared structural labels in each Section box. Default is
            ``True``.
        show_compartments : bool, optional
            Draw the declared ``nseg`` ruler and count. This is an authoring
            preview, not the compiled graph. Default is ``True``.
        annotate_connections : bool, optional
            Include attachment locations, child endpoints, and spatial-gap
            status on graph edges. Default is ``True``.
        connection_tolerance_um : float, optional
            Non-negative visual tolerance in µm for classifying connection
            coordinates as coincident. Default is ``1e-9``.
        legend : bool, optional
            Show the Section-name legend or numeric color scale. Default is
            ``True``.
        cmap : str, optional
            Matplotlib colormap for numeric ``color_by`` modes. Default is
            ``"viridis"``.
        ax : object, optional
            Existing two-dimensional Matplotlib axes. A new figure and axes
            are created when omitted.
        figsize : tuple of float, optional
            New figure size in inches. Ignored when ``ax`` is supplied.
            Default is ``(10, 7)``.
        dpi : int, optional
            New figure resolution. Ignored when ``ax`` is supplied. Default is
            ``150``.
        title : str, optional
            Figure title. ``None`` uses the renderer's descriptive default.

        Returns
        -------
        object
            The Matplotlib ``(Figure, Axes)`` pair. The method never calls
            ``show``.
        """
        from .morphology_visualization import plot_morphology_section_topology

        return plot_morphology_section_topology(
            self,
            color_by=color_by,
            highlight=highlight,
            show_parameters=show_parameters,
            show_labels=show_labels,
            show_compartments=show_compartments,
            annotate_connections=annotate_connections,
            connection_tolerance_um=connection_tolerance_um,
            legend=legend,
            cmap=cmap,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

    def plot_diameter_profile(
        self,
        *,
        sections: str | Section | Iterable[str | Section] | None = None,
        highlight: str | Iterable[str] | None = None,
        x_axis: Literal["normalized", "distance"] = "normalized",
        show_points: bool = True,
        show_compartments: bool = True,
        show_connections: bool = True,
        legend: bool = True,
        ax: object | None = None,
        figsize: tuple[float, float] = (9, 5),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Plot authored diameter as a function of position along Sections.

        Stylized cylinders appear as constant profiles. Pt3d Sections use
        their piecewise-linear diameter controls, making this view especially
        useful after :meth:`SectionLocation.update` inserts or edits a local
        control point. Repeated-coordinate controls have the same horizontal
        position and draw an abrupt diameter step as a vertical segment.

        Parameters
        ----------
        sections : Section, str, or iterable, optional
            Exact Section object or name, or several of either, to include.
            ``None`` includes all Sections in declaration order.
        highlight : str or iterable of str, optional
            Exact Section name or shared structural label, or several of
            either, to emphasize while muting non-matches. It does not filter
            the selected ``sections``. Default is ``None``.
        x_axis : {"normalized", "distance"}, optional
            Horizontal coordinate. ``"normalized"`` uses each Section's
            authored x in ``[0, 1]``; ``"distance"`` uses centerline distance
            in µm from authored x=0. Default is ``"normalized"``.
        show_points : bool, optional
            Mark authored diameter control points. Default is ``True``.
        show_compartments : bool, optional
            Mark centers or intervals implied by authored ``nseg`` values,
            without compiling a solver graph. Default is ``True``.
        show_connections : bool, optional
            Mark normalized parent attachment positions and child endpoints on
            included profiles. These marks describe electrical connectivity,
            not a continuous global path-distance coordinate. Default is
            ``True``.
        legend : bool, optional
            Show the Section legend. Default is ``True``.
        ax : object, optional
            Existing Matplotlib axes. A new figure and axes are created when
            omitted.
        figsize : tuple of float, optional
            New figure size in inches. Ignored when ``ax`` is supplied.
            Default is ``(9, 5)``.
        dpi : int, optional
            New figure resolution. Ignored when ``ax`` is supplied. Default is
            ``150``.
        title : str, optional
            Figure title. ``None`` uses the renderer's descriptive default.

        Returns
        -------
        object
            The Matplotlib ``(Figure, Axes)`` pair from the renderer.

        Notes
        -----
        Each Section retains its own authored coordinate and orientation;
        connected profiles are not concatenated into a global cable axis. The
        current declaration is read without mutation or compilation, and no
        display call is made implicitly.
        """
        from .morphology_visualization import plot_morphology_diameter_profile

        return plot_morphology_diameter_profile(
            self,
            sections=sections,
            highlight=highlight,
            x_axis=x_axis,
            show_points=show_points,
            show_compartments=show_compartments,
            show_connections=show_connections,
            legend=legend,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

    def inspect(
        self,
        *,
        highlight: str | Iterable[str] | None = None,
        show_points: bool = True,
        show_compartments: bool = True,
        show_orientation: bool = True,
        annotate_sections: bool = False,
        annotate_connections: bool = True,
        connection_tolerance_um: float = 1e-9,
        figsize: tuple[float, float] = (16, 9),
        dpi: int = 150,
        title: str | None = None,
    ) -> object:
        """Build a coordinated dashboard for human morphology validation.

        The dashboard combines complementary spatial projections with exact
        compartment topology and diameter/discretization diagnostics. All
        panels describe one read-only snapshot of the current authored
        declaration.

        Parameters
        ----------
        highlight : str or iterable of str, optional
            Exact Section name or shared structural label, or several of
            either, to emphasize consistently across panels. Default is
            ``None``.
        show_points : bool, optional
            Show authored pt3d controls and stylized endpoints. Default is
            ``True``.
        show_compartments : bool, optional
            Show centers or counts implied by authored ``nseg`` declarations
            in the spatial and diameter-profile panels. These are authoring
            previews; the topology panel always shows the canonical compiled
            graph regardless of this option. Default is ``True``.
        show_orientation : bool, optional
            Mark increasing authored x and the connected child endpoint.
            Default is ``True``.
        annotate_sections : bool, optional
            Label spatial paths with exact Section names. Default is ``False``.
        annotate_connections : bool, optional
            Label electrical attachments with parent x and child endpoint
            values once, on the z-projection panel. Default is ``True``.
        connection_tolerance_um : float, optional
            Visual threshold, in µm, for distinguishing coincident attachment
            coordinates from spatial gaps. It does not change electrical
            connectivity or validate SWC export. Default is ``1e-9``.
        figsize : tuple of float, optional
            Dashboard width and height in inches. Default is ``(16, 9)``.
        dpi : int, optional
            Dashboard resolution. Default is ``150``.
        title : str, optional
            Dashboard title. ``None`` uses the renderer's descriptive default.

        Returns
        -------
        object
            The Matplotlib figure and coordinated axes returned by the
            dashboard renderer.

        Notes
        -----
        Authored centerline coordinates and electrical connection semantics
        are deliberately shown as distinct information. Connected stylized
        Sections may overlap because :meth:`connect` does not place them in a
        shared coordinate frame. The topology panel canonically compiles each
        connected component of the detached declaration snapshot; this gives
        a complete Morphology the same graph as :meth:`compile` while keeping
        an incomplete forest inspectable. The dashboard does not mutate the
        Morphology, consult an existing model snapshot, or display itself
        implicitly.
        """
        from .morphology_visualization import inspect_morphology

        return inspect_morphology(
            self,
            highlight=highlight,
            show_points=show_points,
            show_compartments=show_compartments,
            show_orientation=show_orientation,
            annotate_sections=annotate_sections,
            annotate_connections=annotate_connections,
            connection_tolerance_um=connection_tolerance_um,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )


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
        if distance == 0.0 and first[3] == second[3]:
            raise ValueError(
                f"Consecutive pt3d points {index} and {index + 1} on "
                f"{section_name!r} must have distinct coordinates or different "
                "diameters. Exact duplicate controls are redundant."
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
    # Endpoint locations retain endpoint identity even when a tiny terminal
    # span is lost while accumulating a much larger binary64 arclength. The
    # generic interval lookup cannot distinguish that last control from the
    # preceding one when both accumulated coordinates round to arc[-1].
    if s_um <= 0.0:
        return section.points[0]
    if s_um >= float(arc[-1]):
        return section.points[-1]
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


def _location_representation_tolerance(
    section: Section,
    segment: int,
    point: tuple[float, float, float, float],
) -> float:
    """Bound normalized-location error from binary64 coordinate resolution."""
    first = section.points[segment]
    second = section.points[segment + 1]
    coordinate_resolution = math.hypot(
        *(
            max(
                math.ulp(first[axis]),
                math.ulp(second[axis]),
                math.ulp(point[axis]),
            )
            for axis in range(3)
        )
    )
    normalized_resolution = coordinate_resolution / section.L
    baseline = 1024.0 * np.finfo(np.float64).eps
    return min(1e-9, max(baseline, 16.0 * normalized_resolution))


def _unrepresentable_location(section: Section, x: float) -> ValueError:
    return ValueError(
        f"Location {section.name!r}({x!r}) cannot be represented as a distinct "
        "pt3d control point at the current coordinate precision. Re-center or "
        "rescale the authored coordinates."
    )


def _swc_type(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer.")
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative, got {out!r}.")
    return out


def _format_swc_float(value: float) -> str:
    if value == 0.0:
        return "0"
    return np.format_float_positional(float(value), unique=True, trim="-")


def _morphology_to_swc(
    morphology: Morphology,
    *,
    section_types: Mapping[str, int] | None,
    default_type: int,
    connection_tolerance_um: float,
) -> str:
    default_type = _swc_type(default_type, name="default_type")
    tolerance = _nonnegative(connection_tolerance_um, name="connection_tolerance_um")
    if section_types is None:
        section_types = {}
    elif not isinstance(section_types, Mapping):
        raise TypeError("section_types must be a mapping from Section names to ints.")

    normalized_types: dict[str, int] = {}
    for section_name, type_id in section_types.items():
        if not isinstance(section_name, str):
            raise TypeError("section_types keys must be exact Section name strings.")
        normalized_types[section_name] = _swc_type(
            type_id, name=f"section_types[{section_name!r}]"
        )
    unknown_names = sorted(set(normalized_types).difference(morphology._sections))
    if unknown_names:
        raise ValueError(
            "section_types contains unknown Section name(s): "
            + ", ".join(repr(name) for name in unknown_names)
            + "."
        )

    discontinuous_sections = [
        section.name
        for section in morphology.sections
        if any(
            first[:3] == second[:3] and first[3] != second[3]
            for first, second in zip(section.points, section.points[1:])
        )
    ]
    if discontinuous_sections:
        names = ", ".join(repr(name) for name in discontinuous_sections)
        raise ValueError(
            "Cannot export abrupt zero-length diameter discontinuities to SWC "
            f"without losing geometry (Section(s): {names}). Dendra's SWC "
            "contract requires positive-length edges and cannot encode the "
            "annular membrane surface of a repeated-coordinate diameter step."
        )

    # This performs the complete connected-tree validation before any output is
    # produced. The graph itself is intentionally not used for morphometric
    # serialization: its nodes are computational compartment centers.
    morphology.compile()

    root_name = next(
        name for name in morphology._sections if name not in morphology._connections
    )
    children: dict[str, list[str]] = {name: [] for name in morphology._sections}
    for child_name in morphology._sections:
        connection = morphology._connections.get(child_name)
        if connection is not None:
            children[connection.parent_name].append(child_name)

    for connection in morphology._connections.values():
        parent = morphology._sections[connection.parent_name]
        child = morphology._sections[connection.child_name]
        parent_point = _interpolate_point(parent, connection.parent_x * parent.L)
        child_point = _interpolate_point(child, connection.child_end * child.L)
        separation = math.dist(parent_point[:3], child_point[:3])
        if separation > tolerance:
            raise ValueError(
                f"Cannot export connection from Section {parent.name!r} at "
                f"x={connection.parent_x!r} to Section {child.name!r} endpoint "
                f"x={connection.child_end}: their coordinates are separated by "
                f"{separation!r} µm, exceeding connection_tolerance_um="
                f"{tolerance!r}. SWC edges encode both topology and geometry; "
                "author spatially coherent pt3d points before exporting."
            )

    # An authored control is identified by its source index as well as its
    # normalized position. Distinct controls can share the same binary64 x
    # when a tiny span follows a very large accumulated arclength; a mapping or
    # set keyed only by x would silently discard one of them.
    samples: dict[str, tuple[tuple[float, int | None], ...]] = {}
    location_samples: dict[str, dict[float, tuple[float, int | None]]] = {}
    for section in morphology.sections:
        arc = _polyline_arclength(section.points)
        sample_xs = [float(value / arc[-1]) for value in arc]
        sample_xs[0] = 0.0
        sample_xs[-1] = 1.0
        section_samples: list[tuple[float, int | None]] = [
            (section_x, point_index) for point_index, section_x in enumerate(sample_xs)
        ]
        authored_xs = set(sample_xs)
        connection_xs = {
            morphology._connections[child_name].parent_x
            for child_name in children[section.name]
        }
        section_samples.extend(
            (section_x, None)
            for section_x in connection_xs
            if section_x not in authored_xs
        )
        section_samples.sort(
            key=lambda sample: (
                sample[0],
                len(section.points) if sample[1] is None else sample[1],
            )
        )
        samples[section.name] = tuple(section_samples)

        by_location = {sample[0]: sample for sample in section_samples}
        # Endpoint locations select the actual first/last authored controls.
        # At a colliding interior x, _interpolate_point's right-sided lookup
        # likewise selects the last control at that accumulated arclength.
        by_location[0.0] = section_samples[0]
        by_location[1.0] = section_samples[-1]
        location_samples[section.name] = by_location

    lines = [
        "# Dendra Morphology SWC export",
        "# id type x y z radius parent",
        "# Coordinates and radii are in micrometers.",
    ]
    node_ids: dict[tuple[str, float], int] = {}
    next_node_id = 1
    stack = [root_name]
    while stack:
        section_name = stack.pop()
        section = morphology._sections[section_name]
        connection = morphology._connections.get(section_name)
        section_samples = samples[section_name]
        if connection is None:
            oriented_samples = section_samples
            parent_id = -1
            attached_sample = None
        else:
            attached_x = float(connection.child_end)
            oriented_samples = (
                section_samples
                if connection.child_end == 0
                else tuple(reversed(section_samples))
            )
            parent_id = node_ids[(connection.parent_name, connection.parent_x)]
            node_ids[(section_name, attached_x)] = parent_id
            attached_sample = location_samples[section_name][attached_x]

        swc_type = normalized_types.get(section_name, default_type)
        for sample in oriented_samples:
            if sample == attached_sample:
                continue
            section_x, point_index = sample
            if point_index is None:
                # Parent attachments may introduce a sample between authored
                # controls. Authored controls themselves are emitted verbatim
                # so normalization and interpolation cannot perturb source
                # coordinates or diameters during a round trip.
                point = _interpolate_point(section, section_x * section.L)
            else:
                point = section.points[point_index]
            node_id = next_node_id
            next_node_id += 1
            if sample == location_samples[section_name][section_x]:
                node_ids[(section_name, section_x)] = node_id
            lines.append(
                " ".join(
                    (
                        str(node_id),
                        str(swc_type),
                        _format_swc_float(point[0]),
                        _format_swc_float(point[1]),
                        _format_swc_float(point[2]),
                        _format_swc_float(0.5 * point[3]),
                        str(parent_id),
                    )
                )
            )
            parent_id = node_id

        stack.extend(reversed(children[section_name]))

    return "\n".join(lines) + "\n"


def _section_interval_limits(
    section: Section, s0: float, s1: float
) -> tuple[float, float]:
    if s1 < s0:
        s0, s1 = s1, s0
    s0 = max(0.0, min(float(s0), section.L))
    s1 = max(0.0, min(float(s1), section.L))
    return s0, s1


def _arclengths_equal_within_roundoff(first: float, second: float) -> bool:
    """Return whether two computed arclengths differ by only a few ULPs."""
    if first == second:
        return True
    tolerance = 8.0 * max(math.ulp(first), math.ulp(second))
    return abs(first - second) <= tolerance


def _snap_arclength_to_span_endpoint(
    value: float, first: float, second: float
) -> float:
    """Snap a computed interval boundary to the nearest coincident control."""
    candidates = [
        endpoint
        for endpoint in (first, second)
        if _arclengths_equal_within_roundoff(value, endpoint)
    ]
    if not candidates:
        return value
    return min(candidates, key=lambda endpoint: (abs(value - endpoint), endpoint))


def _section_interval_spans(section: Section, s0: float, s1: float):
    """Yield clipped pt3d spans, retaining zero-length diameter steps.

    The final boolean distinguishes an abrupt same-coordinate diameter change
    from an ordinary positive-arclength truncated-cone span. NEURON assigns a
    discontinuity exactly on a compartment boundary to the interval ending at
    that boundary; a discontinuity at Section x=0 belongs to the first interval.
    """
    s0, s1 = _section_interval_limits(section, s0, s1)
    if s1 <= s0:
        return
    arc = _polyline_arclength(section.points)
    for index, (first, second) in enumerate(zip(section.points, section.points[1:])):
        a0 = float(arc[index])
        a1 = float(arc[index + 1])
        if first[:3] == second[:3]:
            lower_match = _arclengths_equal_within_roundoff(a0, s0)
            upper_match = _arclengths_equal_within_roundoff(a0, s1)
            if lower_match or upper_match:
                # A shared boundary belongs to the interval ending there. If
                # an exceptionally short interval is within tolerance of both
                # ends, select the nearer boundary (and the upper one on a tie).
                lower_distance = abs(a0 - s0) if lower_match else math.inf
                upper_distance = abs(a0 - s1) if upper_match else math.inf
                include = upper_distance <= lower_distance
            else:
                include = s0 < a0 < s1
            # The first authored control is exactly at zero and its step has no
            # upstream interval, so it belongs to the first compartment.
            if (s0 == 0.0 and a0 == 0.0) or include:
                yield 0.0, first[3], second[3], True
            continue

        snapped_s0 = _snap_arclength_to_span_endpoint(s0, a0, a1)
        snapped_s1 = _snap_arclength_to_span_endpoint(s1, a0, a1)
        lo = max(snapped_s0, a0)
        hi = min(snapped_s1, a1)
        if hi <= lo:
            continue
        span = a1 - a0
        if span <= 0.0:
            # A positive coordinate displacement can be lost when accumulated
            # after an enormous arclength. The Section's binary64 L likewise
            # cannot resolve that span, so it contributes no representable
            # normalized interval here.
            continue
        lo_fraction = (lo - a0) / span
        hi_fraction = (hi - a0) / span
        d0 = first[3] + lo_fraction * (second[3] - first[3])
        d1 = first[3] + hi_fraction * (second[3] - first[3])
        yield hi - lo, float(d0), float(d1), False


def _section_integrals(
    section: Section, s0: float, s1: float
) -> tuple[float, float, float]:
    """Return lateral area, volume, and inverse-area integral for an interval."""
    area = 0.0
    volume = 0.0
    inv_area = 0.0
    for length, d0_raw, d1_raw, discontinuity in _section_interval_spans(
        section, s0, s1
    ):
        d0 = max(d0_raw, _DIAMETER_EPS)
        d1 = max(d1_raw, _DIAMETER_EPS)
        r0 = 0.5 * d0
        r1 = 0.5 * d1
        if discontinuity:
            # A zero-arclength truncated cone is the annular membrane surface
            # used by NEURON for an abrupt pt3d diameter step. It carries no
            # material volume and no axial resistance.
            area += math.pi * (r0 + r1) * abs(r1 - r0)
            continue
        area += math.pi * (r0 + r1) * math.hypot(length, r1 - r0)
        volume += math.pi * length * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
        inv_area += 4.0 * length / (math.pi * d0 * d1)
    return float(area), float(volume), float(inv_area)


def _section_mean_diameter(section: Section, s0: float, s1: float) -> float:
    """Return NEURON-compatible arclength-mean diameter for one interval."""
    s0, s1 = _section_interval_limits(section, s0, s1)
    interval_length = s1 - s0
    if interval_length <= 0.0:
        raise ValueError("A compartment diameter interval must have positive length.")
    diameter_integral = math.fsum(
        0.5 * (d0 + d1) * length
        for length, d0, d1, discontinuity in _section_interval_spans(section, s0, s1)
        if not discontinuity
    )
    return _positive(
        diameter_integral / interval_length,
        name=f"section {section.name!r} compartment mean diameter",
    )


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
            mean_diameter = _section_mean_diameter(
                section, x0 * section.L, x1 * section.L
            )
            area, volume, _ = _section_integrals(
                section, x0 * section.L, x1 * section.L
            )
            material_id[(section.name, index)] = next_node
            nodes[next_node] = _MutableNode(
                length_um=section.L / section.nseg,
                diameter_um=mean_diameter,
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
