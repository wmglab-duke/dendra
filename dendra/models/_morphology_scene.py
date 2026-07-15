"""Renderer-neutral snapshots of native :mod:`dendra` morphologies.

The plotting backends need a common, numerically faithful description of an
authored :class:`~dendra.models.morphology.Morphology`.  This module first
extracts an immutable authoring snapshot.  Spatial and Section-schematic
renderers consume that snapshot directly; the compartment-topology renderer
canonically compiles each connected component of it so the displayed graph is
the actual electrical graph rather than an ``nseg`` preview.  Empty and
partially connected morphologies therefore remain inspectable while they are
being authored.

All locations are sampled through the same binary64 arclength helpers used by
native morphology compilation and SWC export.  Coordinates and diameters are
in micrometers, ``rhoa`` is in ohm-centimeters, and ``cm`` is in
microfarads per square centimeter.  The returned dataclasses contain only
immutable Python values and remain independent snapshots after later edits to
the source Morphology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .morphology import Morphology, Section, _interpolate_point, _polyline_arclength

_Point = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SectionScene:
    """One authored Section and its exact discretization sample locations.

    ``points`` preserves the authored control-point order.  ``sample_x`` gives
    the corresponding normalized cumulative-arclength coordinates.  The
    ``boundary_*`` and ``center_*`` fields describe the Section's ``nseg``
    equal-arclength compartments; every point is ``(x, y, z, diameter)``.

    ``away_direction`` is ``1`` when increasing authored Section coordinate
    runs away from the parent and ``-1`` when decreasing coordinate does.  A
    root Section has no parent orientation, so its deterministic display
    convention is ``1``.
    """

    name: str
    labels: frozenset[str]
    is_pt3d: bool
    L: float
    nseg: int
    rhoa: float
    cm: float
    points: tuple[_Point, ...]
    sample_x: tuple[float, ...]
    boundary_x: tuple[float, ...]
    boundary_points: tuple[_Point, ...]
    center_x: tuple[float, ...]
    center_points: tuple[_Point, ...]
    parent_name: str | None
    parent_x: float | None
    child_end: int | None
    away_direction: int


@dataclass(frozen=True, slots=True)
class ConnectionScene:
    """One logical Section connection with both authored spatial endpoints.

    Dendra's electrical connectivity does not require ``parent_point`` and
    ``child_point`` to coincide.  ``gap_um`` reports their Euclidean separation
    rather than hiding or repairing a spatial discontinuity.  It is positive
    infinity when finite endpoint coordinates are farther apart than binary64
    can represent; renderers preserve that state as an explicit diagnostic.

    ``parent_host_segment`` follows Dendra's normalized-coordinate selection:
    an exact internal boundary belongs to the segment on its increasing-x
    side, while ``x=1`` is clamped to the final segment.  At a Section endpoint
    it identifies the adjacent material segment; compilation may additionally
    collapse or retain an algebraic junction.
    """

    parent_name: str
    parent_x: float
    parent_point: _Point
    child_name: str
    child_end: int
    child_point: _Point
    gap_um: float
    parent_host_segment: int


@dataclass(frozen=True, slots=True)
class MorphologyScene:
    """Immutable renderer-neutral snapshot of a Morphology declaration.

    ``sections`` and ``connections`` follow Section declaration order.
    ``roots`` contains every currently unparented Section, so it may be empty
    or contain more than one entry while a morphology is being built.
    ``children`` contains one ``(parent_name, child_names)`` entry for every
    declared Section, also in declaration order.
    """

    sections: tuple[SectionScene, ...]
    connections: tuple[ConnectionScene, ...]
    roots: tuple[str, ...]
    children: tuple[tuple[str, tuple[str, ...]], ...]

    def section(self, name: str) -> SectionScene:
        """Return the exactly named Section scene."""
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(f"Unknown Section scene {name!r}.")

    def children_of(self, name: str) -> tuple[str, ...]:
        """Return child Section names in declaration order."""
        for parent_name, child_names in self.children:
            if parent_name == name:
                return child_names
        raise KeyError(f"Unknown Section scene {name!r}.")


@dataclass(frozen=True, slots=True)
class CompartmentNodeScene:
    """One exact node from a compiled component of a Morphology forest."""

    node_id: int
    component_index: int
    local_node_id: int
    parent_id: int | None
    name: str
    kind: str
    section_name: str | None
    segment_index: int | None
    section_x: float | None
    labels: frozenset[str]
    degree: int
    length_um: float
    diameter_um: float
    area_um2: float
    volume_um3: float
    x_um: float
    y_um: float
    z_um: float
    rhoa_ohm_cm: float
    cm_uF_cm2: float
    edge_length_um: float
    edge_resistance_ohm: float
    edge_diff_geom_um: float

    @property
    def is_branchpoint(self) -> bool:
        """Whether this node has at least three undirected cable neighbors."""
        return self.degree >= 3


@dataclass(frozen=True, slots=True)
class CompartmentTopologyScene:
    """Renderer-neutral compartment topology for one Section forest.

    Each connected Section component is compiled independently through
    Dendra's canonical scalar compiler. This preserves exact compartment,
    endpoint-collapse, interior-attachment, and retained-junction semantics
    while still allowing an incomplete multi-root authoring forest to be
    inspected.
    """

    nodes: tuple[CompartmentNodeScene, ...]
    edges: tuple[tuple[int, int], ...]
    roots: tuple[int, ...]
    children: tuple[tuple[int, tuple[int, ...]], ...]

    def node(self, node_id: int) -> CompartmentNodeScene:
        """Return one globally numbered topology node."""
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            raise TypeError("node_id must be an integer.")
        if node_id < 0 or node_id >= len(self.nodes):
            raise KeyError(f"Unknown compartment topology node {node_id!r}.")
        return self.nodes[node_id]

    def children_of(self, node_id: int) -> tuple[int, ...]:
        """Return child IDs in canonical component order."""
        self.node(node_id)
        return self.children[node_id][1]


def _sample_x(section: Section) -> tuple[float, ...]:
    """Return normalized authored point positions from cumulative arclength."""
    arc = _polyline_arclength(section.points)
    if len(arc) != len(section.points) or len(arc) < 2:
        raise RuntimeError(
            f"Section {section.name!r} has invalid canonical point geometry."
        )
    total = float(arc[-1])
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            f"Section {section.name!r} has invalid canonical centerline length."
        )
    normalized = [float(value / total) for value in arc]
    normalized[0] = 0.0
    normalized[-1] = 1.0
    return tuple(normalized)


def _sample_points(
    section: Section, locations: tuple[float, ...]
) -> tuple[_Point, ...]:
    """Sample normalized Section locations using canonical interpolation."""
    return tuple(
        _interpolate_point(section, location * section.L) for location in locations
    )


def _canonical_sections(morphology: Morphology) -> tuple[Section, ...]:
    """Return declaration-ordered Sections after checking private invariants."""
    sections = morphology.sections
    names = tuple(section.name for section in sections)
    if len(set(names)) != len(names):
        raise RuntimeError("Morphology contains duplicate canonical Section names.")
    if tuple(morphology._sections) != names or any(
        morphology._sections.get(section.name) is not section for section in sections
    ):
        raise RuntimeError(
            "Morphology Section declaration order or canonical identity is corrupt."
        )
    return sections


def build_morphology_scene(morphology: Morphology) -> MorphologyScene:
    """Create an immutable visualization snapshot without compiling.

    Parameters
    ----------
    morphology : Morphology
        The native authoring declaration to inspect.  It may be empty,
        disconnected, or otherwise incomplete as a scalar solver tree.

    Returns
    -------
    MorphologyScene
        Declaration-ordered geometry, discretization samples, connectivity,
        and spatial-gap measurements containing no live mutable views.

    Notes
    -----
    This function checks only canonical internal ownership and connection
    assumptions.  It deliberately does not call :meth:`Morphology.compile`,
    because visualization is also useful while constructing an incomplete
    forest.
    """
    if not isinstance(morphology, Morphology):
        raise TypeError("morphology must be a native Morphology.")

    sections = _canonical_sections(morphology)
    section_by_name = {section.name: section for section in sections}
    section_names = tuple(section_by_name)
    known_names = set(section_names)

    connection_by_child = dict(morphology._connections)
    unknown_connection_keys = set(connection_by_child).difference(known_names)
    if unknown_connection_keys:
        names = ", ".join(repr(name) for name in sorted(unknown_connection_keys))
        raise RuntimeError(f"Morphology has connections for unknown Sections: {names}.")

    child_names: dict[str, list[str]] = {name: [] for name in section_names}
    connections: list[ConnectionScene] = []
    connection_details: dict[str, tuple[str, float, int]] = {}

    # Iterate over Sections, not the connection dictionary, so all scene
    # ordering follows declaration order rather than connection-call order.
    for child in sections:
        connection = connection_by_child.get(child.name)
        if connection is None:
            continue
        if connection.child_name != child.name:
            raise RuntimeError(
                f"Connection key {child.name!r} disagrees with its child name "
                f"{connection.child_name!r}."
            )
        if connection.parent_name not in known_names:
            raise RuntimeError(
                f"Section {child.name!r} has unknown parent {connection.parent_name!r}."
            )
        if connection.parent_name == child.name:
            raise RuntimeError(f"Section {child.name!r} is connected to itself.")
        if type(connection.child_end) is not int or connection.child_end not in (0, 1):
            raise RuntimeError(
                f"Section {child.name!r} has invalid connected child endpoint "
                f"{connection.child_end!r}."
            )
        parent_x = float(connection.parent_x)
        if not math.isfinite(parent_x) or not 0.0 <= parent_x <= 1.0:
            raise RuntimeError(
                f"Section {child.name!r} has invalid parent location {parent_x!r}."
            )

        parent = section_by_name[connection.parent_name]
        parent_point = _interpolate_point(parent, parent_x * parent.L)
        child_point = _interpolate_point(child, connection.child_end * child.L)
        gap_um = math.dist(parent_point[:3], child_point[:3])
        if math.isnan(gap_um):
            raise RuntimeError(
                f"Connection to Section {child.name!r} has an invalid spatial gap."
            )
        parent_host_segment = min(int(parent_x * parent.nseg), parent.nseg - 1)

        child_names[parent.name].append(child.name)
        connection_details[child.name] = (
            parent.name,
            parent_x,
            int(connection.child_end),
        )
        connections.append(
            ConnectionScene(
                parent_name=parent.name,
                parent_x=parent_x,
                parent_point=parent_point,
                child_name=child.name,
                child_end=int(connection.child_end),
                child_point=child_point,
                gap_um=float(gap_um),
                parent_host_segment=parent_host_segment,
            )
        )

    # Public mutation rejects cycles.  Rechecking the private snapshot here
    # prevents a corrupted parent chain from reaching schematic renderers that
    # reasonably assume they received a forest.
    for section_name in section_names:
        lineage: set[str] = set()
        current_name = section_name
        while current_name in connection_by_child:
            if current_name in lineage:
                raise RuntimeError("Morphology connections contain a cycle.")
            lineage.add(current_name)
            current_name = connection_by_child[current_name].parent_name

    section_scenes: list[SectionScene] = []
    for section in sections:
        boundary_x = tuple(index / section.nseg for index in range(section.nseg + 1))
        center_x = tuple((index + 0.5) / section.nseg for index in range(section.nseg))
        connection = connection_details.get(section.name)
        if connection is None:
            parent_name = None
            parent_x = None
            child_end = None
            away_direction = 1
        else:
            parent_name, parent_x, child_end = connection
            away_direction = 1 if child_end == 0 else -1

        section_scenes.append(
            SectionScene(
                name=section.name,
                labels=frozenset(section.labels),
                is_pt3d=section.is_pt3d,
                L=float(section.L),
                nseg=int(section.nseg),
                rhoa=float(section.rhoa),
                cm=float(section.cm),
                points=tuple(tuple(point) for point in section.points),
                sample_x=_sample_x(section),
                boundary_x=boundary_x,
                boundary_points=_sample_points(section, boundary_x),
                center_x=center_x,
                center_points=_sample_points(section, center_x),
                parent_name=parent_name,
                parent_x=parent_x,
                child_end=child_end,
                away_direction=away_direction,
            )
        )

    roots = tuple(
        section.name for section in sections if section.name not in connection_by_child
    )
    children = tuple(
        (section.name, tuple(child_names[section.name])) for section in sections
    )
    return MorphologyScene(
        sections=tuple(section_scenes),
        connections=tuple(connections),
        roots=roots,
        children=children,
    )


def _component_names(
    root_name: str, children: dict[str, tuple[str, ...]]
) -> frozenset[str]:
    names: set[str] = set()
    stack = [root_name]
    while stack:
        name = stack.pop()
        if name in names:
            raise RuntimeError("Morphology scene connections contain a cycle.")
        names.add(name)
        stack.extend(reversed(children[name]))
    return frozenset(names)


def _compile_scene_component(scene: MorphologyScene, section_names: frozenset[str]):
    """Reconstruct and canonically compile one immutable scene component."""
    component = Morphology()
    sections = [section for section in scene.sections if section.name in section_names]
    cloned = {}
    for section in sections:
        geometry = (
            {"points": section.points}
            if section.is_pt3d
            else {"L": section.L, "diam": section.points[0][3]}
        )
        cloned[section.name] = component.section(
            section.name,
            **geometry,
            nseg=section.nseg,
            rhoa=section.rhoa,
            cm=section.cm,
            labels=tuple(sorted(section.labels.difference({section.name}))),
        )
    for section in sections:
        if section.parent_name is None:
            continue
        assert section.parent_x is not None and section.child_end is not None
        cloned[section.name].connect(
            cloned[section.parent_name].at(section.parent_x),
            child_end=section.child_end,
        )
    return component.compile()


def build_compartment_topology_scene(
    morphology: Morphology,
    *,
    morphology_scene: MorphologyScene | None = None,
) -> CompartmentTopologyScene:
    """Compile a read-only compartment topology for every declared component.

    A complete scalar Morphology produces exactly the same canonical topology
    and metadata as :meth:`Morphology.compile`. A partially authored forest is
    split at its current roots and each connected component is compiled
    independently, allowing the visualization layer to remain useful before
    the final single-root contract is satisfied.
    """
    if not isinstance(morphology, Morphology):
        raise TypeError("morphology must be a native Morphology.")
    scene = (
        build_morphology_scene(morphology)
        if morphology_scene is None
        else morphology_scene
    )
    if not scene.sections:
        return CompartmentTopologyScene(nodes=(), edges=(), roots=(), children=())

    child_sections = dict(scene.children)
    covered: set[str] = set()
    nodes: list[CompartmentNodeScene] = []
    edges: list[tuple[int, int]] = []
    roots: list[int] = []
    child_nodes: list[list[int]] = []

    for component_index, root_name in enumerate(scene.roots):
        section_names = _component_names(root_name, child_sections)
        if covered.intersection(section_names):
            raise RuntimeError("Morphology scene components overlap.")
        covered.update(section_names)
        graph = _compile_scene_component(scene, section_names)
        offset = len(nodes)
        roots.append(offset + graph.topology.root)

        degree = [0] * graph.n_compartments
        for parent, child in graph.topology.edges:
            degree[parent] += 1
            degree[child] += 1
            edges.append((offset + parent, offset + child))

        for local_node in range(graph.n_compartments):
            parent = graph.topology.parent_index[local_node]
            global_node = offset + local_node
            metadata = graph.metadata
            geometry = graph.geometry
            nodes.append(
                CompartmentNodeScene(
                    node_id=global_node,
                    component_index=component_index,
                    local_node_id=local_node,
                    parent_id=None if parent == -1 else offset + parent,
                    name=metadata.name[local_node],
                    kind=metadata.kind[local_node],
                    section_name=metadata.section_name[local_node],
                    segment_index=metadata.segment_index[local_node],
                    section_x=metadata.section_x[local_node],
                    labels=metadata.labels[local_node],
                    degree=degree[local_node],
                    length_um=geometry.length_um[local_node],
                    diameter_um=geometry.diameter_um[local_node],
                    area_um2=geometry.area_um2[local_node],
                    volume_um3=geometry.volume_um3[local_node],
                    x_um=geometry.x_um[local_node],
                    y_um=geometry.y_um[local_node],
                    z_um=geometry.z_um[local_node],
                    rhoa_ohm_cm=geometry.rhoa_ohm_cm[local_node],
                    cm_uF_cm2=geometry.cm_uF_cm2[local_node],
                    edge_length_um=geometry.edge_length_um[local_node],
                    edge_resistance_ohm=geometry.edge_resistance_ohm[local_node],
                    edge_diff_geom_um=geometry.edge_diff_geom_um[local_node],
                )
            )
            child_nodes.append([])
        for parent, child in graph.topology.edges:
            child_nodes[offset + parent].append(offset + child)

    expected = {section.name for section in scene.sections}
    if covered != expected:
        missing = ", ".join(repr(name) for name in sorted(expected - covered))
        raise RuntimeError(f"Morphology scene has unreachable Sections: {missing}.")

    children = tuple(
        (node_id, tuple(values)) for node_id, values in enumerate(child_nodes)
    )
    return CompartmentTopologyScene(
        nodes=tuple(nodes),
        edges=tuple(edges),
        roots=tuple(roots),
        children=children,
    )


__all__ = [
    "CompartmentNodeScene",
    "CompartmentTopologyScene",
    "ConnectionScene",
    "MorphologyScene",
    "SectionScene",
    "build_compartment_topology_scene",
    "build_morphology_scene",
]
