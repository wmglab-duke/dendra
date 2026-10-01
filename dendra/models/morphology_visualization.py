"""Matplotlib renderers for native :class:`~dendra.Morphology` declarations.

The renderers consume immutable snapshots from :mod:`._morphology_scene` and
never mutate the source Morphology.  Spatial plots, the Section schematic, and
the diameter profile preserve the authored declaration.  The compartment
topology deliberately compiles the snapshot through Dendra's canonical scalar
compiler, making its nodes and edges solver-exact without confusing them with
physical centerline segments.

Matplotlib is imported only when this module is requested through a plotting
method.  Importing :mod:`dendra` therefore does not select a GUI backend, and
none of the functions below calls ``show()`` implicitly.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ._morphology_scene import (
    CompartmentNodeScene,
    CompartmentTopologyScene,
    ConnectionScene,
    MorphologyScene,
    SectionScene,
    build_compartment_topology_scene,
    build_morphology_scene,
)
from .morphology import Morphology, Section

_COLOR_FIELDS = frozenset({"section", "diameter", "length", "nseg", "rhoa", "cm"})
_NUMERIC_COLOR_LABELS = {
    "diameter": "Diameter (µm)",
    "length": "Section length (µm)",
    "nseg": "Section nseg",
    "rhoa": "Axial resistivity (Ω·cm)",
    "cm": "Specific capacitance (µF/cm²)",
}
_VIEW_AXES = {
    "x": ((1, 2), ("y (µm)", "z (µm)")),
    "y": ((0, 2), ("x (µm)", "z (µm)")),
    "z": ((0, 1), ("x (µm)", "y (µm)")),
}
_MUTED_ALPHA = 0.30
_MUTED_TOPOLOGY_EDGE_ALPHA = 0.24

# ``vis_2d`` established a useful visual language for runtime Population
# graphs: vivid, evenly separated HSV colors and geometry whose display weight
# follows cable diameter.  Native Morphology views keep those colors stable as
# Sections are appended by using a fixed, progressively refined hue order
# instead of ``vis_2d``'s count-dependent shuffle.
_SECTION_HUES = (
    0.0,
    2.0 / 3.0,
    1.0 / 3.0,
    1.0 / 6.0,
    5.0 / 6.0,
    1.0 / 2.0,
    1.0 / 12.0,
    3.0 / 4.0,
    5.0 / 12.0,
    1.0 / 4.0,
    11.0 / 12.0,
    7.0 / 12.0,
)

# ``Tree.from_morphology`` registers these structural regions in the same
# fixed slots used by ``vis_2d``.  Pinning the resulting ``palette_hsv(5)``
# colors gives a Morphology and its generated Tree the same canonical visual
# language without importing the substantially heavier runtime visualization
# module here.  The first palette slot belongs to the all/internal selection
# and is intentionally absent because it is not a physical morphology family.
_CANONICAL_SHAPE_FAMILY_COLORS = {
    "soma": "#c8e650",
    "axon": "#e65050",
    "dend": "#c850e6",
    "apic": "#508ce6",
}
_CANONICAL_SHAPE_FAMILY_ALIASES = {
    "soma": "soma",
    "cell_body": "soma",
    "axon": "axon",
    "dend": "dend",
    "dendrite": "dend",
    "basal": "dend",
    "basal_dendrite": "dend",
    "apic": "apic",
    "apical": "apic",
    "apical_dendrite": "apic",
}
_GENERATED_SECTION_SUFFIX = re.compile(r"(?:\[\d+\]|_\d+)$")


@dataclass(frozen=True, slots=True)
class _ColorContext:
    section_colors: Mapping[str, tuple[float, float, float, float]]
    color_by: str
    cmap: mpl.colors.Colormap | None = None
    norm: mpl.colors.Normalize | None = None
    display_scale: float = 1.0

    def _continuous_color(self, value: float):
        assert self.cmap is not None and self.norm is not None
        return self.cmap(self.norm(float(value) / self.display_scale))

    def section_color(self, section: SectionScene) -> tuple[float, float, float, float]:
        if self.color_by == "section":
            return self.section_colors[section.name]
        if self.color_by == "diameter":
            value = float(np.mean([point[3] for point in section.points]))
        else:
            value = _section_scalar(section, self.color_by)
        return self._continuous_color(value)

    def segment_colors(
        self, section: SectionScene, diameters: Sequence[float] | None = None
    ) -> list[tuple[float, float, float, float]]:
        if self.color_by != "diameter":
            count = len(section.points) - 1 if diameters is None else len(diameters)
            return [self.section_color(section)] * count
        if diameters is not None:
            return self.diameter_colors(diameters)
        return [
            self._continuous_color(0.5 * (first[3] + second[3]))
            for first, second in zip(section.points, section.points[1:])
        ]

    def diameter_colors(self, diameters) -> list[tuple[float, float, float, float]]:
        """Return local colors for diameter-encoded spatial glyphs."""
        if self.color_by != "diameter":
            raise ValueError("Local diameter colors require color_by='diameter'.")
        return [self._continuous_color(value) for value in diameters]


@dataclass(frozen=True, slots=True)
class _ShapeColorContext:
    colors: _ColorContext
    section_families: Mapping[str, str]
    family_colors: Mapping[str, tuple[float, float, float, float]]


def _finite(value, *, name: str, positive: bool = False, nonnegative: bool = False):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    try:
        out = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}.")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive, got {out!r}.")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be non-negative, got {out!r}.")
    return out


def _validate_dpi(dpi: int) -> int:
    if isinstance(dpi, (bool, np.bool_)) or not isinstance(dpi, (int, np.integer)):
        raise TypeError("dpi must be a positive integer.")
    dpi = int(dpi)
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    return dpi


def _validate_figsize(figsize) -> tuple[float, float]:
    try:
        width, height = figsize
    except (TypeError, ValueError) as error:
        raise TypeError("figsize must contain exactly two positive numbers.") from error
    return (
        _finite(width, name="figsize width", positive=True),
        _finite(height, name="figsize height", positive=True),
    )


def _new_2d_axes(ax, *, figsize, dpi):
    if ax is None:
        fig, ax = plt.subplots(
            figsize=_validate_figsize(figsize), dpi=_validate_dpi(dpi)
        )
    else:
        if getattr(ax, "name", "rectilinear") == "3d":
            raise ValueError("A two-dimensional Matplotlib axes is required.")
        fig = ax.figure
    return fig, ax


def _new_3d_axes(ax, *, figsize, dpi):
    if ax is None:
        fig = plt.figure(figsize=_validate_figsize(figsize), dpi=_validate_dpi(dpi))
        ax = fig.add_subplot(111, projection="3d")
    else:
        if getattr(ax, "name", None) != "3d":
            raise ValueError("A three-dimensional Matplotlib axes is required.")
        fig = ax.figure
    return fig, ax


def _selectors(value, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as error:
            raise TypeError(
                f"{name} must be a string or iterable of strings."
            ) from error
    if not values:
        return ()
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{name} must contain only exact name or label strings.")
    return values


def _highlight_names(
    scene: MorphologyScene, highlight: str | Iterable[str] | None
) -> frozenset[str] | None:
    selectors = _selectors(highlight, name="highlight")
    if selectors is None:
        return None
    available = (
        set().union(*(section.labels for section in scene.sections))
        if scene.sections
        else set()
    )
    unknown = sorted(set(selectors).difference(available))
    if unknown:
        names = ", ".join(repr(item) for item in unknown)
        raise ValueError(f"Unknown Section name or label in highlight: {names}.")
    return frozenset(
        section.name
        for section in scene.sections
        if set(selectors).intersection(section.labels)
    )


def _is_highlighted(section: SectionScene, names: frozenset[str] | None) -> bool:
    return names is None or section.name in names


def _progressive_categorical_color(index: int) -> tuple[float, float, float, float]:
    """Return one append-stable color in the native vivid HSV sequence."""
    if index < len(_SECTION_HUES):
        hue = _SECTION_HUES[index]
    else:
        extra = index - len(_SECTION_HUES)
        hue = (0.037 + (extra + 0.5) * 0.6180339887498949) % 1.0
    rgb = mpl.colors.hsv_to_rgb((hue, 0.65, 0.90))
    return *map(float, rgb), 1.0


def _categorical_colors(scene: MorphologyScene):
    # Match ``vis_2d``'s saturated HSV appearance while addressing colors by a
    # fixed index.  Unlike ``palette_hsv(len(sections))``, appending a Section
    # therefore cannot recolor an existing one.
    return {
        section.name: _progressive_categorical_color(index)
        for index, section in enumerate(scene.sections)
    }


def _normalized_shape_family_name(name: str) -> str:
    normalized = _GENERATED_SECTION_SUFFIX.sub("", name)
    return normalized or name


def _canonical_shape_family(value: str) -> str | None:
    return _CANONICAL_SHAPE_FAMILY_ALIASES.get(value.casefold())


def _shape_family(section: SectionScene) -> str:
    """Return one unambiguous quiet-shape color family for a Section."""
    normalized_name = _normalized_shape_family_name(section.name)
    name_family = _canonical_shape_family(normalized_name)
    if name_family is not None:
        # Exact authored/generated identity outranks broader structural labels;
        # for example, ``apic[0]`` remains apic even when also labelled dendrite.
        return name_family

    label_families = {
        family
        for label in section.labels
        if (family := _canonical_shape_family(label)) is not None
    }
    if len(label_families) == 1:
        return label_families.pop()
    if label_families == {"apic", "dend"}:
        # Apical dendrite labels commonly include both the broad dendrite class
        # and its more specific apical identity. This is a hierarchy, not a
        # conflict, and mirrors Tree/vis_2d where apic overrides dend.
        return "apic"
    # Unknown or conflicting labels have no primary-family semantics. Generated
    # numeric suffixes still collapse, while other names retain exact identity.
    return normalized_name


def _stable_shape_fallback_color(
    family: str,
) -> tuple[float, float, float, float]:
    """Return a process-, order-, and append-stable vivid fallback color."""
    digest = hashlib.blake2b(
        family.encode("utf-8"),
        digest_size=8,
        person=b"dendra-shape",
    ).digest()
    hue = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    rgb = mpl.colors.hsv_to_rgb((hue, 0.65, 0.90))
    return *map(float, rgb), 1.0


def _shape_color_context(scene: MorphologyScene) -> _ShapeColorContext:
    section_families = {
        section.name: _shape_family(section) for section in scene.sections
    }
    ordered_families = tuple(dict.fromkeys(section_families.values()))
    family_colors: dict[str, tuple[float, float, float, float]] = {}
    for family in ordered_families:
        canonical_color = _CANONICAL_SHAPE_FAMILY_COLORS.get(family)
        if canonical_color is not None:
            family_colors[family] = mpl.colors.to_rgba(canonical_color)
        else:
            family_colors[family] = _stable_shape_fallback_color(family)
    section_colors = {
        section_name: family_colors[family]
        for section_name, family in section_families.items()
    }
    return _ShapeColorContext(
        colors=_ColorContext(section_colors, "section"),
        section_families=section_families,
        family_colors=family_colors,
    )


def _section_scalar(section: SectionScene, color_by: str) -> float:
    if color_by == "length":
        return section.L
    if color_by == "nseg":
        return float(section.nseg)
    if color_by == "rhoa":
        return section.rhoa
    if color_by == "cm":
        return section.cm
    raise ValueError(f"Unsupported Section scalar {color_by!r}.")


def _color_context(scene: MorphologyScene, color_by: str, cmap: str) -> _ColorContext:
    if color_by not in _COLOR_FIELDS:
        allowed = ", ".join(repr(value) for value in sorted(_COLOR_FIELDS))
        raise ValueError(f"color_by must be one of {allowed}; got {color_by!r}.")
    categorical = _categorical_colors(scene)
    if color_by == "section":
        return _ColorContext(categorical, color_by)
    try:
        continuous = mpl.colormaps[cmap]
    except KeyError as error:
        raise ValueError(f"Unknown Matplotlib colormap {cmap!r}.") from error
    if color_by == "diameter":
        values = [point[3] for section in scene.sections for point in section.points]
    else:
        values = [_section_scalar(section, color_by) for section in scene.sections]
    if not values:
        display_scale = 1.0
        vmin, vmax = 0.0, 1.0
    else:
        maximum = max(abs(float(value)) for value in values)
        # Matplotlib's tick locator cannot safely perform arithmetic close to
        # binary64's largest exponent.  Normalize only those extreme display
        # values and disclose the divisor on the color key; ordinary values
        # retain their exact scale and labels.
        display_scale = maximum if maximum > 1e150 else 1.0
        display_values = [float(value) / display_scale for value in values]
        vmin, vmax = min(display_values), max(display_values)
        if vmin == vmax:
            pad = max(0.05 * abs(vmin), 0.5 if color_by == "nseg" else 1e-12)
            lower = vmin - pad
            upper = vmax + pad
            vmin = lower if math.isfinite(lower) else vmin
            vmax = upper if math.isfinite(upper) else vmax
            if vmin == vmax:
                vmin = float(np.nextafter(vmin, -math.inf))
                vmax = float(np.nextafter(vmax, math.inf))
    return _ColorContext(
        categorical,
        color_by,
        continuous,
        mpl.colors.Normalize(vmin, vmax),
        display_scale,
    )


def _with_alpha(color, alpha: float):
    red, green, blue, base_alpha = mpl.colors.to_rgba(color)
    return red, green, blue, base_alpha * alpha


def _blend_with_white(color, color_fraction: float):
    red, green, blue, _ = mpl.colors.to_rgba(color)
    return tuple(1.0 - color_fraction * (1.0 - value) for value in (red, green, blue))


def _section_alpha(section: SectionScene, highlight_names) -> float:
    return 1.0 if _is_highlighted(section, highlight_names) else _MUTED_ALPHA


def _format_scientific(value: float) -> str:
    """Format a finite scale compactly and deterministically."""
    return np.format_float_scientific(float(value), precision=6, unique=True, trim="-")


def _add_color_key(fig, ax, context: _ColorContext, *, legend: bool) -> None:
    if not legend or context.color_by == "section":
        return
    assert context.cmap is not None and context.norm is not None
    mappable = mpl.cm.ScalarMappable(norm=context.norm, cmap=context.cmap)
    mappable.set_array([])
    is_3d = getattr(ax, "name", "rectilinear") == "3d"
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        pad=0.1 if is_3d else 0.02,
        shrink=0.72 if is_3d else 0.82,
    )
    label = _NUMERIC_COLOR_LABELS[context.color_by]
    if context.display_scale != 1.0:
        label += f" · values ÷ {_format_scientific(context.display_scale)}"
    colorbar.set_label(label)


def _section_legend(ax, scene, context, highlight_names, *, legend: bool):
    if not legend or context.color_by != "section" or not scene.sections:
        return
    handles = [
        Line2D(
            [0],
            [0],
            color=context.section_color(section),
            alpha=_section_alpha(section, highlight_names),
            linewidth=3,
            label=section.name,
        )
        for section in scene.sections
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        title="Sections",
    )


def _shape_family_legend(
    ax,
    shape_context: _ShapeColorContext,
    *,
    legend: bool,
) -> None:
    if not legend or not shape_context.family_colors:
        return
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=3,
            label=family,
        )
        for family, color in shape_context.family_colors.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        title="Regions",
    )


def _project(
    point,
    view: str,
    origin: np.ndarray | tuple[float, float, float] | None = None,
    scale: np.ndarray | tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    try:
        axes, _ = _VIEW_AXES[view]
    except KeyError as error:
        raise ValueError("view must be exactly 'x', 'y', or 'z'.") from error
    if origin is None:
        origin = (0.0, 0.0, 0.0)
    if scale is None:
        scale = (1.0, 1.0, 1.0)
    return (
        float((point[axes[0]] - origin[axes[0]]) / scale[axes[0]]),
        float((point[axes[1]] - origin[axes[1]]) / scale[axes[1]]),
    )


def _sample_section(section: SectionScene, x: float):
    x = min(max(float(x), 0.0), 1.0)
    index = min(
        int(np.searchsorted(section.sample_x, x, side="right")) - 1,
        len(section.points) - 2,
    )
    index = max(index, 0)
    first_x = section.sample_x[index]
    second_x = section.sample_x[index + 1]
    fraction = 0.0 if second_x == first_x else (x - first_x) / (second_x - first_x)
    first = section.points[index]
    second = section.points[index + 1]
    return tuple(
        float(first[axis] + fraction * (second[axis] - first[axis]))
        for axis in range(4)
    )


def _render_spans(section: SectionScene) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide taper only for rendering, preserving each authored chord.

    A single sparse pt3d span can carry a large linear diameter change.  One
    LineCollection segment has only one width/color, so it would conceal that
    taper.  Bounded straight-line subdivision gives the display enough samples
    without adding controls to the Morphology or changing the scene snapshot.
    """
    segments = []
    diameters = []
    for first, second in zip(section.points, section.points[1:]):
        ratio = max(first[3], second[3]) / min(first[3], second[3])
        subdivisions = (
            1
            if ratio <= 1.000000000001
            else min(16, max(4, math.ceil(6.0 * abs(math.log(ratio)))))
        )
        first_xyz = np.asarray(first[:3], dtype=float)
        second_xyz = np.asarray(second[:3], dtype=float)
        delta = second_xyz - first_xyz
        for index in range(subdivisions):
            lo = index / subdivisions
            hi = (index + 1) / subdivisions
            segments.append((first_xyz + lo * delta, first_xyz + hi * delta))
            midpoint = (index + 0.5) / subdivisions
            diameters.append(first[3] + midpoint * (second[3] - first[3]))
    return np.asarray(segments, dtype=float), np.asarray(diameters, dtype=float)


def _safe_unit(vector: np.ndarray) -> np.ndarray | None:
    """Normalize a finite vector without overflowing at binary64 extremes."""
    vector = np.asarray(vector, dtype=float)
    scale = float(np.max(np.abs(vector)))
    if not math.isfinite(scale) or scale == 0.0:
        return None
    scaled = vector / scale
    norm = math.sqrt(float(np.dot(scaled, scaled)))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return scaled / norm


def _fallback_normal(tangent: np.ndarray) -> np.ndarray:
    """Choose a deterministic unit normal to one unit tangent."""
    axis = np.zeros(3, dtype=float)
    axis[int(np.argmin(np.abs(tangent)))] = 1.0
    normal = _safe_unit(np.cross(tangent, axis))
    if normal is None:  # defensive: a finite unit tangent always has a normal
        raise RuntimeError("Could not construct a finite morphology tube frame.")
    return normal


def _tube_rings(
    points: np.ndarray,
    diameters: np.ndarray,
    *,
    diameter_scale: float,
    radial_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build parallel-transport rings for one tapered cable centerline."""
    # A repeated centerline coordinate with a different diameter is an
    # intentional pt3d diameter step.  NEURON models that pair as a
    # zero-length degenerate frustum: its two concentric rings form an annular
    # shoulder with membrane area but no cable length.  Such a span has no
    # tangent of its own, so borrow the nearest incoming/outgoing cable
    # directions instead of fabricating a geometric displacement.
    chord_tangents: list[np.ndarray | None] = []
    for first, second in zip(points, points[1:]):
        chord_tangents.append(_safe_unit(second - first))
    if not any(tangent is not None for tangent in chord_tangents):
        raise ValueError(
            "A morphology shape Section must have positive total centerline length."
        )

    incoming: list[np.ndarray | None] = []
    nearest = None
    for index in range(len(points)):
        if index and chord_tangents[index - 1] is not None:
            nearest = chord_tangents[index - 1]
        incoming.append(nearest)

    outgoing: list[np.ndarray | None] = [None] * len(points)
    nearest = None
    for index in range(len(points) - 1, -1, -1):
        if index < len(chord_tangents) and chord_tangents[index] is not None:
            nearest = chord_tangents[index]
        outgoing[index] = nearest

    tangents = []
    for before, after in zip(incoming, outgoing):
        if before is None:
            tangent = after
        elif after is None:
            tangent = before
        else:
            tangent = _safe_unit(before + after)
            if tangent is None:
                # An exact reversal has no unique bisector. Continue in the
                # root-away chord direction; circular rings make twist
                # irrelevant. Repeated-coordinate controls at that vertex all
                # receive the same direction and therefore the same ring frame.
                tangent = after
        if tangent is None:  # guarded by the positive-total-length check above
            raise RuntimeError("Could not infer a morphology tube tangent.")
        tangents.append(tangent)
    tangents = np.asarray(tangents, dtype=float)

    normals = [_fallback_normal(tangents[0])]
    for tangent in tangents[1:]:
        transported = normals[-1] - np.dot(normals[-1], tangent) * tangent
        normal = _safe_unit(transported)
        normals.append(_fallback_normal(tangent) if normal is None else normal)
    normals = np.asarray(normals, dtype=float)
    binormals = np.asarray(
        [
            _safe_unit(np.cross(tangent, normal))
            for tangent, normal in zip(tangents, normals)
        ]
    )
    if binormals.dtype == object or not np.all(np.isfinite(binormals)):
        raise RuntimeError("Could not construct finite morphology tube binormals.")

    angles = np.arange(radial_segments, dtype=float) * (2.0 * math.pi / radial_segments)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    # Multiply the authored diameter by its user scale before halving. This
    # preserves a representable product when ``diameter_scale`` is the least
    # positive subnormal (halving that factor first would round it to zero).
    radii = 0.5 * (diameters * diameter_scale)
    rings = points[:, None, :] + radii[:, None, None] * (
        cosines[None, :, None] * normals[:, None, :]
        + sines[None, :, None] * binormals[:, None, :]
    )
    return rings, tangents


def _tube_faces(
    section: SectionScene,
    *,
    origin: np.ndarray,
    display_scale: float,
    post_scale: float,
    diameter_scale: float,
    radial_segments: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return closed tapered-tube faces and outward normals for one Section."""
    points = (
        np.asarray([point[:3] for point in section.points], dtype=float) - origin
    ) / display_scale
    points /= post_scale
    diameters = (
        np.asarray([point[3] for point in section.points], dtype=float) / display_scale
    )
    diameters /= post_scale
    rings, tangents = _tube_rings(
        points,
        diameters,
        diameter_scale=diameter_scale,
        radial_segments=radial_segments,
    )
    faces: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for first, second in zip(rings, rings[1:]):
        for index in range(radial_segments):
            following = (index + 1) % radial_segments
            face = np.asarray(
                (first[index], first[following], second[following], second[index]),
                dtype=float,
            )
            normal = _safe_unit(np.cross(face[1] - face[0], face[-1] - face[0]))
            faces.append(face)
            normals.append(_fallback_normal(tangents[0]) if normal is None else normal)
    faces.extend((rings[0][::-1].copy(), rings[-1].copy()))
    normals.extend((-tangents[0], tangents[-1]))
    return faces, normals


def _shape_display_transform(
    scene: MorphologyScene, *, diameter_scale: float
) -> tuple[np.ndarray, float, float]:
    """Return sequential uniform divisors preserving physical aspect.

    The two divisors are applied in sequence because their mathematical
    product may exceed binary64 even when every authored value and the final
    normalized surface are finite. Ordinary morphologies retain divisors of
    one and therefore exactly the historical display coordinates.
    """
    origin = _display_origin(scene)
    display_scale = float(np.max(_display_scale(scene, origin)))
    max_diameter = max(
        point[3] for section in scene.sections for point in section.points
    )
    safe_magnitude = 1e150

    # Test the product by division so a finite diameter and finite display
    # multiplier never overflow merely while selecting a display scale.
    if max_diameter > (2.0 * safe_magnitude) / diameter_scale:
        display_scale = max(display_scale, max_diameter)

    scaled_max_diameter = max_diameter / display_scale
    scaled_max_radius = 0.5 * (scaled_max_diameter * diameter_scale)
    post_scale = scaled_max_radius if scaled_max_radius > safe_magnitude else 1.0
    return origin, display_scale, post_scale


def _shape_axis_label(
    name: str, origin: float, display_scale: float, post_scale: float
) -> str:
    """Format a shape coordinate label without overflowing a scale product."""
    coordinate = name if origin == 0.0 else f"{name} − {name}₀"
    if display_scale == 1.0 and post_scale == 1.0:
        return f"{coordinate} (µm)"
    if display_scale <= np.finfo(np.float64).max / post_scale:
        scale_text = _format_scientific(display_scale * post_scale)
    else:
        logarithm = math.log10(display_scale) + math.log10(post_scale)
        exponent = math.floor(logarithm)
        mantissa = 10.0 ** (logarithm - exponent)
        scale_text = f"{mantissa:.6g}e{exponent:+d}"
    return f"{coordinate} (× {scale_text} µm)"


def _shape_radial_segments(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("radial_segments must be an integer of at least 3.")
    value = int(value)
    if value < 3:
        raise ValueError("radial_segments must be an integer of at least 3.")
    return value


def _shape_legend(legend, family_count: int) -> bool:
    if isinstance(legend, (bool, np.bool_)):
        return bool(legend)
    if legend == "auto":
        return family_count <= 16
    raise ValueError("legend must be a boolean or exactly 'auto'.")


class _ShapeAxisIndicator(Artist):
    """Screen-fixed xyz triad projected through an owning 3-D Axes camera."""

    _AXES = (
        ("x", "#d62728"),
        ("y", "#2ca02c"),
        ("z", "#1f77b4"),
    )
    _FALLBACK_DIRECTIONS = np.asarray(
        ((1.0, -0.25), (-0.45, 0.85), (0.25, 1.0)), dtype=float
    )

    def __init__(self, ax) -> None:
        super().__init__()
        self.axes = ax
        self.set_figure(ax.figure)
        self.set_gid("morphology-shape-axis-indicator")
        self.set_in_layout(False)
        self.set_clip_on(False)
        self.set_zorder(30)
        self.base = np.asarray((0.075, 0.075), dtype=float)
        self.radius = 0.055
        self.projected_vectors = np.zeros((3, 2), dtype=float)
        self.projected_endpoints = np.repeat(self.base[None, :], 3, axis=0)
        self.arrows = []
        self.end_on_markers = []
        self.labels = []
        for name, color in self._AXES:
            arrow = FancyArrowPatch(
                tuple(self.base),
                tuple(self.base),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=8.0,
                linewidth=1.35,
                color=color,
                shrinkA=0.0,
                shrinkB=0.0,
                clip_on=False,
                zorder=30,
            )
            arrow.set_figure(ax.figure)
            arrow.axes = ax
            arrow.set_gid(f"morphology-shape-axis:{name}")
            marker = Line2D(
                [self.base[0]],
                [self.base[1]],
                transform=ax.transAxes,
                marker="o",
                markersize=3.4,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.1,
                linestyle="none",
                clip_on=False,
                zorder=30,
            )
            marker.set_figure(ax.figure)
            marker.axes = ax
            marker.set_gid(f"morphology-shape-axis-end-on:{name}")
            marker.set_visible(False)
            label = Text(
                x=self.base[0],
                y=self.base[1],
                text=name,
                color=color,
                fontsize=7.5,
                fontweight="semibold",
                ha="center",
                va="center",
                transform=ax.transAxes,
                clip_on=False,
                zorder=30,
            )
            label.set_figure(ax.figure)
            label.axes = ax
            label.set_gid(f"morphology-shape-axis-label:{name}")
            self.arrows.append(arrow)
            self.end_on_markers.append(marker)
            self.labels.append(label)

    def _camera_vectors(self) -> np.ndarray:
        limits = np.asarray(
            (self.axes.get_xlim3d(), self.axes.get_ylim3d(), self.axes.get_zlim3d()),
            dtype=float,
        )
        center = limits.mean(axis=1)
        half_spans = 0.25 * np.abs(limits[:, 1] - limits[:, 0])
        half_spans = np.where(half_spans > 0.0, half_spans, 1.0)
        points = np.vstack((center, center + np.diag(half_spans)))
        projected = np.column_stack(
            proj3d.proj_transform(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                self.axes.get_proj(),
            )
        )
        display = self.axes.transData.transform(projected[:, :2])
        axes_coordinates = self.axes.transAxes.inverted().transform(display)
        return np.asarray(axes_coordinates[1:] - axes_coordinates[0], dtype=float)

    def _update_children(self) -> None:
        vectors = self._camera_vectors()
        if not np.isfinite(vectors).all():
            vectors = self._FALLBACK_DIRECTIONS.copy()
        endpoints = []
        unit_vectors = []
        for index, vector in enumerate(vectors):
            magnitude = float(np.linalg.norm(vector))
            if magnitude <= 1e-10:
                unit = np.zeros(2, dtype=float)
                endpoint = self.base.copy()
                self.arrows[index].set_visible(False)
                self.end_on_markers[index].set_visible(True)
                label_direction = self._FALLBACK_DIRECTIONS[index]
                label_direction /= np.linalg.norm(label_direction)
            else:
                unit = vector / magnitude
                endpoint = self.base + self.radius * unit
                self.arrows[index].set_visible(True)
                self.arrows[index].set_positions(tuple(self.base), tuple(endpoint))
                self.end_on_markers[index].set_visible(False)
                label_direction = unit
            self.labels[index].set_position(tuple(endpoint + 0.014 * label_direction))
            endpoints.append(endpoint)
            unit_vectors.append(unit)
        self.projected_vectors = np.asarray(unit_vectors, dtype=float)
        self.projected_endpoints = np.asarray(endpoints, dtype=float)

    def draw(self, renderer) -> None:
        if not self.get_visible():
            return
        self._update_children()
        for arrow, marker, label in zip(self.arrows, self.end_on_markers, self.labels):
            if arrow.get_visible():
                arrow.draw(renderer)
            if marker.get_visible():
                marker.draw(renderer)
            label.draw(renderer)
        self.stale = False

    def get_children(self):
        return tuple(self.arrows + self.end_on_markers + self.labels)


class _Shape3DInteraction:
    """Small addition to Axes3D navigation: scroll-wheel centered zoom."""

    def __init__(self, fig, ax) -> None:
        self.fig = fig
        self.ax = ax
        self.connection_id = fig.canvas.mpl_connect("scroll_event", self)

    def disconnect(self) -> None:
        self.fig.canvas.mpl_disconnect(self.connection_id)

    def __call__(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        try:
            step = float(event.step)
        except (AttributeError, TypeError, ValueError):
            return
        if not math.isfinite(step) or step == 0.0:
            return
        factor = 0.9 ** float(np.clip(step, -20.0, 20.0))
        for getter, setter in (
            (self.ax.get_xlim3d, self.ax.set_xlim3d),
            (self.ax.get_ylim3d, self.ax.set_ylim3d),
            (self.ax.get_zlim3d, self.ax.set_zlim3d),
        ):
            first, second = getter()
            center = 0.5 * (first + second)
            half_span = 0.5 * (second - first) * factor
            setter(center - half_span, center + half_span)
        self.fig.canvas.draw_idle()


def _replace_shape_axis_indicator(ax, *, show: bool) -> None:
    previous = getattr(ax, "_dendra_shape_axis_indicator", None)
    if previous is not None:
        previous.remove()
    ax._dendra_shape_axis_indicator = None
    if show:
        indicator = _ShapeAxisIndicator(ax)
        ax.add_artist(indicator)
        ax._dendra_shape_axis_indicator = indicator


def _configure_shape_3d_interaction(fig, ax, *, interactive: bool) -> None:
    previous = getattr(ax, "_dendra_shape_interaction", None)
    if previous is not None:
        previous.disconnect()
    ax._dendra_shape_interaction = None
    if not interactive:
        return

    backend = str(mpl.get_backend())
    normalized = backend.casefold()
    static_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if "matplotlib_inline" in normalized or normalized in static_backends:
        warnings.warn(
            f"interactive=True requested while Matplotlib is using the static "
            f"{backend!r} backend; the shape will render, but rotation and zoom "
            "events cannot be delivered. In Jupyter, install the optional "
            "backend with `python -m pip install ipympl`. If the kernel and "
            "Jupyter server use separate "
            "environments, install a compatible ipympl in both. Stop and "
            "restart the entire Jupyter server—not only the kernel—then refresh "
            "the page and run `%matplotlib widget` before creating the figure. "
            "A desktop GUI backend also works.",
            RuntimeWarning,
            stacklevel=4,
        )
    ax.set_navigate(True)
    ax.mouse_init()
    ax._dendra_shape_interaction = _Shape3DInteraction(fig, ax)


def _warn_shape_connection_gaps(
    scene: MorphologyScene, *, connection_tolerance_um: float
) -> None:
    gaps = [
        connection.gap_um
        for connection in scene.connections
        if connection.gap_um > connection_tolerance_um
    ]
    if not gaps:
        return
    largest = max(gaps)
    largest_text = (
        "beyond binary64 range" if math.isinf(largest) else f"{largest:.6g} µm"
    )
    warnings.warn(
        f"Morphology shape has {len(gaps)} spatially incoherent electrical "
        f"connection{'s' if len(gaps) != 1 else ''} (largest gap: "
        f"{largest_text}). The shape renderer preserves authored coordinates "
        "and does not draw artificial bridges; use Morphology.plot() or "
        "Morphology.inspect() to diagnose the connections.",
        RuntimeWarning,
        stacklevel=3,
    )


def _shaded_section_color(color, normal: np.ndarray):
    """Apply deterministic ambient/diffuse lighting without losing identity."""
    light = np.asarray((0.35, -0.45, 0.82), dtype=float)
    light /= np.linalg.norm(light)
    diffuse = max(0.0, float(np.dot(normal, light)))
    intensity = 0.72 + 0.28 * diffuse
    red, green, blue, alpha = mpl.colors.to_rgba(color)
    return (
        min(1.0, red * intensity),
        min(1.0, green * intensity),
        min(1.0, blue * intensity),
        alpha,
    )


def _shape_faces(
    scene: MorphologyScene,
    *,
    origin: np.ndarray,
    display_scale: float,
    post_scale: float,
    diameter_scale: float,
    radial_segments: int,
    context: _ColorContext,
):
    faces = []
    colors = []
    sections = []
    for section in scene.sections:
        section_faces, normals = _tube_faces(
            section,
            origin=origin,
            display_scale=display_scale,
            post_scale=post_scale,
            diameter_scale=diameter_scale,
            radial_segments=radial_segments,
        )
        base_color = context.section_color(section)
        faces.extend(section_faces)
        colors.extend(_shaded_section_color(base_color, normal) for normal in normals)
        sections.extend([section.name] * len(section_faces))
    return faces, colors, sections


def _orientation_span_index(section: SectionScene, points: np.ndarray) -> int:
    """Choose a visible local authored span, preferring the parent-facing end."""
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    if section.parent_name is None:
        return int(np.argmax(lengths))
    indexes = (
        range(len(lengths))
        if section.away_direction == 1
        else range(len(lengths) - 1, -1, -1)
    )
    for index in indexes:
        if lengths[index] > 0.0:
            return index
    return 0


def _geometry_title(scene: MorphologyScene, *, dimensionality: str) -> str:
    compartments = sum(section.nseg for section in scene.sections)
    suffix = "" if len(scene.roots) == 1 else f" · {len(scene.roots)} roots"
    return (
        f"Authored morphology ({dimensionality}) · {len(scene.sections)} Sections · "
        f"{compartments} declared compartments{suffix}"
    )


def _empty_axes(ax, *, title: str | None, text: str = "Empty Morphology"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center")
    ax.set_title(text if title is None else title)


def _format_location(value: float) -> str:
    return np.format_float_positional(float(value), precision=5, trim="-")


def _connection_style(connection: ConnectionScene, tolerance: float):
    if math.isinf(connection.gap_um):
        return "#d62728", "--", "beyond binary64 range"
    if connection.gap_um == 0.0:
        return "#2a9d55", "-", "coincident"
    if connection.gap_um <= tolerance:
        return "#d08a00", ":", "within tolerance"
    return "#d62728", "--", "spatial gap"


def _format_gap_diagnostic(
    connection: ConnectionScene,
    status: str,
    *,
    precision: int,
    include_status: bool = True,
) -> str:
    if math.isinf(connection.gap_um):
        return "Δ beyond binary64 range"
    text = f"Δ={connection.gap_um:.{precision}g} µm"
    return f"{text} · {status}" if include_status else text


def _collect_xyz(scene: MorphologyScene) -> np.ndarray:
    points = [point[:3] for section in scene.sections for point in section.points]
    points.extend(connection.parent_point[:3] for connection in scene.connections)
    points.extend(connection.child_point[:3] for connection in scene.connections)
    return np.asarray(points, dtype=float) if points else np.empty((0, 3), dtype=float)


def _display_origin(scene: MorphologyScene) -> np.ndarray:
    """Choose a disclosed numerical origin where offsets hide local spans.

    Binary64 cannot add ordinary plotting padding to, for example, a constant
    coordinate of ``1e100``.  Subtracting a common authored origin is an exact
    display transform at the precision already present in the declaration and
    prevents Matplotlib from producing a singular transform.  Axes that do
    not have an extreme offset retain an origin of zero.
    """
    xyz = _collect_xyz(scene)
    origin = np.zeros(3, dtype=float)
    if xyz.size == 0:
        return origin
    first = np.asarray(scene.sections[0].points[0][:3], dtype=float)
    for axis in range(3):
        values = xyz[:, axis]
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = hi - lo
        scale = max(abs(lo), abs(hi))
        if (
            math.isfinite(span)
            and scale > 1e12 * max(abs(span), 1.0)
            and first[axis] != 0.0
        ):
            origin[axis] = first[axis]
    return origin


def _display_scale(scene: MorphologyScene, origin: np.ndarray) -> np.ndarray:
    """Scale only axes whose finite magnitude exceeds safe plotting arithmetic."""
    xyz = _collect_xyz(scene)
    scale = np.ones(3, dtype=float)
    if xyz.size == 0:
        return scale
    for axis in range(3):
        shifted = xyz[:, axis] - origin[axis]
        maximum = float(np.max(np.abs(shifted)))
        if maximum > 1e150:
            scale[axis] = maximum
    return scale


def _shift_xyz(
    xyz: np.ndarray, origin: np.ndarray, scale: np.ndarray | None = None
) -> np.ndarray:
    shifted = np.asarray(xyz, dtype=float) - np.asarray(origin, dtype=float)
    return shifted if scale is None else shifted / np.asarray(scale, dtype=float)


def _axis_label(name: str, origin: float, scale: float) -> str:
    coordinate = name if origin == 0.0 else f"{name} − {name}₀"
    if scale == 1.0:
        return f"{coordinate} (µm)"
    return f"{coordinate} (× {_format_scientific(scale)} µm)"


def _annotate_display_transform(
    ax, origin: np.ndarray, scale: np.ndarray, axes: tuple[int, ...]
) -> None:
    shifted = [axis for axis in axes if origin[axis] != 0.0]
    scaled = [axis for axis in axes if scale[axis] != 1.0]
    if not shifted and not scaled:
        return
    names = "xyz"
    details = []
    if shifted:
        values = ", ".join(
            f"{names[axis]}₀={_format_scientific(origin[axis])} µm" for axis in shifted
        )
        details.append(f"origin subtracted: {values}")
    if scaled:
        values = ", ".join(
            f"{names[axis]}÷{_format_scientific(scale[axis])} µm" for axis in scaled
        )
        details.append(f"scale applied: {values}")
    text_value = "Display " + "; ".join(details)
    if getattr(ax, "name", "rectilinear") == "3d":
        annotation = ax.text2D(
            0.01,
            0.01,
            text_value,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="0.35",
        )
    else:
        annotation = ax.text(
            0.01,
            0.01,
            text_value,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="0.35",
        )
    annotation.set_gid("display-transform")


def _set_2d_limits(ax, projected: np.ndarray) -> None:
    if projected.size == 0:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        return
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    spans = maxs - mins
    reference = max(float(spans.max()), 1.0)
    pads = np.where(spans > 0.0, 0.05 * spans, 0.05 * reference)
    ax.set_xlim(float(mins[0] - pads[0]), float(maxs[0] + pads[0]))
    ax.set_ylim(float(mins[1] - pads[1]), float(maxs[1] + pads[1]))


def _set_3d_limits(ax, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_box_aspect((1, 1, 1))
        return
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    spans = maxs - mins
    reference = max(float(spans.max()), 1.0)
    pads = np.where(spans > 0.0, 0.05 * spans, 0.05 * reference)
    limits = [
        (float(lo - pad), float(hi + pad)) for lo, hi, pad in zip(mins, maxs, pads)
    ]
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect(tuple(max(float(span), 0.05 * reference) for span in spans))


def _draw_2d_section(
    ax,
    section: SectionScene,
    *,
    view: str,
    origin: np.ndarray,
    scale: np.ndarray,
    context: _ColorContext,
    highlight_names,
    show_points: bool,
    show_compartments: bool,
    show_orientation: bool,
    annotate_sections: bool,
    diameter_scale: float,
    min_linewidth: float,
) -> None:
    alpha = _section_alpha(section, highlight_names)
    projected = np.asarray(
        [_project(point, view, origin, scale) for point in section.points]
    )
    render_segments, render_diameters = _render_spans(section)
    axes, _ = _VIEW_AXES[view]
    render_segments = (render_segments - origin) / scale
    segments = render_segments[:, :, axes]
    widths = np.maximum(min_linewidth, render_diameters * diameter_scale)
    colors = context.segment_colors(section, render_diameters)
    centerline = LineCollection(
        segments,
        colors=colors,
        linewidths=widths,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    centerline.set_gid(f"section:{section.name}")
    ax.add_collection(centerline)

    section_color = context.section_color(section)
    if show_points:
        sizes = np.maximum(18.0, 8.0 * np.sqrt([point[3] for point in section.points]))
        point_colors = (
            context.diameter_colors([point[3] for point in section.points])
            if context.color_by == "diameter"
            else [section_color]
        )
        points = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            s=sizes,
            facecolors=point_colors,
            edgecolors="white" if section.is_pt3d else "black",
            linewidths=0.65,
            marker="o" if section.is_pt3d else "s",
            alpha=alpha,
            zorder=5,
        )
        points.set_gid(f"points:{section.name}")

    if show_compartments:
        centers = np.asarray(
            [_project(point, view, origin, scale) for point in section.center_points]
        )
        boundaries = np.asarray(
            [_project(point, view, origin, scale) for point in section.boundary_points]
        )
        center_colors = (
            context.diameter_colors([point[3] for point in section.center_points])
            if context.color_by == "diameter"
            else [section_color]
        )
        boundary_colors = (
            context.diameter_colors([point[3] for point in section.boundary_points])
            if context.color_by == "diameter"
            else [section_color]
        )
        compartments = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=20,
            facecolors="white",
            edgecolors=center_colors,
            linewidths=1.0,
            marker="o",
            alpha=alpha,
            zorder=6,
        )
        compartments.set_gid(f"compartments:{section.name}")
        ax.scatter(
            boundaries[:, 0],
            boundaries[:, 1],
            s=26,
            color=boundary_colors,
            marker="|",
            linewidths=0.8,
            alpha=alpha,
            zorder=4,
        )

    if show_orientation:
        span_index = _orientation_span_index(section, projected)
        span_start = projected[span_index]
        span_end = projected[span_index + 1]
        start = span_start + 0.25 * (span_end - span_start)
        end = span_start + 0.75 * (span_end - span_start)
        if np.array_equal(start, end):
            orientation = ax.scatter(
                [start[0]],
                [start[1]],
                s=46,
                color=[section_color],
                marker="$+x$",
                linewidths=0.8,
                alpha=alpha,
                zorder=7,
            )
        else:
            orientation = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.1,
                color=section_color,
                alpha=alpha,
                zorder=7,
            )
            ax.add_patch(orientation)
        orientation.set_gid(f"orientation:{section.name}")
        if section.parent_name is not None:
            assert section.child_end is not None
            parent_end = projected[section.child_end * (len(projected) - 1)]
            marker = ax.scatter(
                [parent_end[0]],
                [parent_end[1]],
                s=76,
                color=[section_color],
                marker=f"${section.child_end}$",
                linewidths=0.9,
                alpha=alpha,
                zorder=8,
            )
            marker.set_gid(f"parent-end:{section.name}")

    if annotate_sections:
        position = _project(_sample_section(section, 0.5), view, origin, scale)
        annotation = ax.annotate(
            section.name,
            position,
            xytext=(4, 4),
            textcoords="offset points",
            color=section_color,
            alpha=max(alpha, 0.35),
            fontsize=9,
            weight="semibold",
            zorder=8,
        )
        annotation.set_gid(f"annotation-section:{section.name}")


def _draw_2d_connections(
    ax,
    scene: MorphologyScene,
    *,
    view: str,
    origin: np.ndarray,
    scale: np.ndarray,
    tolerance: float,
    annotate: bool,
) -> None:
    for connection in scene.connections:
        parent = _project(connection.parent_point, view, origin, scale)
        child = _project(connection.child_point, view, origin, scale)
        color, linestyle, status = _connection_style(connection, tolerance)
        if connection.gap_um == 0.0:
            marker = ax.scatter(
                [parent[0]],
                [parent[1]],
                s=44,
                facecolors="none",
                edgecolors=color,
                marker="D",
                linewidths=1.2,
                zorder=9,
            )
            marker.set_gid(f"connection:{connection.child_name}")
        else:
            (connector,) = ax.plot(
                [parent[0], child[0]],
                [parent[1], child[1]],
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
                marker="",
                alpha=0.9,
                zorder=3,
            )
            connector.set_gid(f"connection:{connection.child_name}")
            ax.scatter(
                [parent[0]],
                [parent[1]],
                s=38,
                facecolors="none",
                edgecolors=color,
                marker="o",
                linewidths=1.1,
                zorder=9,
            )
            ax.scatter(
                [child[0]],
                [child[1]],
                s=38,
                color=color,
                marker="x",
                linewidths=1.1,
                zorder=9,
            )

        if annotate:
            midpoint = (0.5 * (parent[0] + child[0]), 0.5 * (parent[1] + child[1]))
            text = (
                f"{connection.parent_name}(x={_format_location(connection.parent_x)}) "
                f"→ {connection.child_name}(end {connection.child_end})\n"
                f"{_format_gap_diagnostic(connection, status, precision=5)}"
            )
            annotation = ax.annotate(
                text,
                midpoint,
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                zorder=10,
            )
            annotation.set_gid(f"annotation-connection:{connection.child_name}")


def plot_morphology(
    morphology: Morphology,
    *,
    view: str = "y",
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
    ax=None,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 150,
    title: str | None = None,
    _scene: MorphologyScene | None = None,
):
    """Render an authored Morphology in one orthographic projection."""
    if view not in _VIEW_AXES:
        raise ValueError("view must be exactly 'x', 'y', or 'z'.")
    tolerance = _finite(
        connection_tolerance_um,
        name="connection_tolerance_um",
        nonnegative=True,
    )
    diameter_scale = _finite(diameter_scale, name="diameter_scale", positive=True)
    min_linewidth = _finite(min_linewidth, name="min_linewidth", positive=True)
    scene = build_morphology_scene(morphology) if _scene is None else _scene
    highlight_names = _highlight_names(scene, highlight)
    context = _color_context(scene, color_by, cmap)
    fig, ax = _new_2d_axes(ax, figsize=figsize, dpi=dpi)

    if not scene.sections:
        _empty_axes(ax, title=title)
    else:
        origin = _display_origin(scene)
        scale = _display_scale(scene, origin)
        for section in scene.sections:
            _draw_2d_section(
                ax,
                section,
                view=view,
                origin=origin,
                scale=scale,
                context=context,
                highlight_names=highlight_names,
                show_points=show_points,
                show_compartments=show_compartments,
                show_orientation=show_orientation,
                annotate_sections=annotate_sections,
                diameter_scale=diameter_scale,
                min_linewidth=min_linewidth,
            )
        if show_connections:
            _draw_2d_connections(
                ax,
                scene,
                view=view,
                origin=origin,
                scale=scale,
                tolerance=tolerance,
                annotate=annotate_connections,
            )
        xyz = _collect_xyz(scene)
        axes, labels = _VIEW_AXES[view]
        shifted = _shift_xyz(xyz, origin, scale)
        _set_2d_limits(ax, shifted[:, axes])
        ax.set_xlabel(_axis_label(labels[0][0], origin[axes[0]], scale[axes[0]]))
        ax.set_ylabel(_axis_label(labels[1][0], origin[axes[1]], scale[axes[1]]))
        _annotate_display_transform(ax, origin, scale, axes)
        ax.set_aspect("equal")
        ax.set_title(
            _geometry_title(scene, dimensionality=f"view along {view}")
            if title is None
            else title
        )
        _section_legend(ax, scene, context, highlight_names, legend=legend)
        _add_color_key(fig, ax, context, legend=legend)
    return fig, ax


def _draw_3d_section(
    ax,
    section: SectionScene,
    *,
    origin: np.ndarray,
    scale: np.ndarray,
    context: _ColorContext,
    highlight_names,
    show_points: bool,
    show_compartments: bool,
    show_orientation: bool,
    annotate_sections: bool,
    diameter_scale: float,
    min_linewidth: float,
) -> None:
    alpha = _section_alpha(section, highlight_names)
    xyz = _shift_xyz(
        np.asarray([point[:3] for point in section.points], dtype=float),
        origin,
        scale,
    )
    render_segments, render_diameters = _render_spans(section)
    segments = (render_segments - origin) / scale
    widths = np.maximum(min_linewidth, render_diameters * diameter_scale)
    centerline = Line3DCollection(
        segments,
        colors=context.segment_colors(section, render_diameters),
        linewidths=widths,
        alpha=alpha,
        zorder=2,
    )
    centerline.set_gid(f"section:{section.name}")
    ax.add_collection3d(centerline)

    section_color = context.section_color(section)
    if show_points:
        sizes = np.maximum(18.0, 8.0 * np.sqrt([point[3] for point in section.points]))
        point_colors = (
            context.diameter_colors([point[3] for point in section.points])
            if context.color_by == "diameter"
            else [section_color]
        )
        points = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=sizes,
            color=point_colors,
            marker="o" if section.is_pt3d else "s",
            edgecolors="white" if section.is_pt3d else "black",
            linewidths=0.6,
            alpha=alpha,
            depthshade=False,
            zorder=5,
        )
        points.set_gid(f"points:{section.name}")

    if show_compartments:
        centers = _shift_xyz(
            np.asarray([point[:3] for point in section.center_points]), origin, scale
        )
        center_colors = (
            context.diameter_colors([point[3] for point in section.center_points])
            if context.color_by == "diameter"
            else [section_color]
        )
        compartments = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            s=20,
            facecolors="white",
            edgecolors=center_colors,
            marker="o",
            linewidths=0.9,
            alpha=alpha,
            depthshade=False,
            zorder=6,
        )
        compartments.set_gid(f"compartments:{section.name}")

    if show_orientation:
        span_index = _orientation_span_index(section, xyz)
        span_start = xyz[span_index]
        span_end = xyz[span_index + 1]
        start = span_start + 0.25 * (span_end - span_start)
        end = span_start + 0.75 * (span_end - span_start)
        delta = end - start
        orientation = ax.quiver(
            start[0],
            start[1],
            start[2],
            delta[0],
            delta[1],
            delta[2],
            color=section_color,
            alpha=alpha,
            arrow_length_ratio=0.25,
            linewidth=1.2,
            normalize=False,
            zorder=7,
        )
        orientation.set_gid(f"orientation:{section.name}")
        if section.parent_name is not None:
            assert section.child_end is not None
            parent_end = xyz[section.child_end * (len(xyz) - 1)]
            marker = ax.scatter(
                [parent_end[0]],
                [parent_end[1]],
                [parent_end[2]],
                s=76,
                color=[section_color],
                marker=f"${section.child_end}$",
                linewidths=0.9,
                alpha=alpha,
                depthshade=False,
                zorder=8,
            )
            marker.set_gid(f"parent-end:{section.name}")

    if annotate_sections:
        position = (
            np.asarray(_sample_section(section, 0.5)[:3], dtype=float) - origin
        ) / scale
        annotation = ax.text(
            position[0],
            position[1],
            position[2],
            f" {section.name}",
            color=section_color,
            alpha=max(alpha, 0.35),
            fontsize=9,
            weight="semibold",
            zorder=8,
        )
        annotation.set_gid(f"annotation-section:{section.name}")


def _draw_3d_connections(
    ax,
    scene: MorphologyScene,
    *,
    origin: np.ndarray,
    scale: np.ndarray,
    tolerance: float,
    annotate: bool,
) -> None:
    for connection in scene.connections:
        parent = (np.asarray(connection.parent_point[:3], dtype=float) - origin) / scale
        child = (np.asarray(connection.child_point[:3], dtype=float) - origin) / scale
        color, linestyle, status = _connection_style(connection, tolerance)
        if connection.gap_um == 0.0:
            marker = ax.scatter(
                [parent[0]],
                [parent[1]],
                [parent[2]],
                s=44,
                facecolors="none",
                edgecolors=color,
                marker="D",
                linewidths=1.2,
                depthshade=False,
                zorder=9,
            )
            marker.set_gid(f"connection:{connection.child_name}")
        else:
            (connector,) = ax.plot(
                [parent[0], child[0]],
                [parent[1], child[1]],
                [parent[2], child[2]],
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
                alpha=0.9,
                zorder=3,
            )
            connector.set_gid(f"connection:{connection.child_name}")
            ax.scatter(
                [parent[0]],
                [parent[1]],
                [parent[2]],
                s=38,
                facecolors="none",
                edgecolors=color,
                marker="o",
                linewidths=1.1,
                depthshade=False,
                zorder=9,
            )
            ax.scatter(
                [child[0]],
                [child[1]],
                [child[2]],
                s=38,
                color=color,
                marker="x",
                linewidths=1.1,
                depthshade=False,
                zorder=9,
            )
        if annotate:
            midpoint = tuple(0.5 * (parent[axis] + child[axis]) for axis in range(3))
            annotation = ax.text(
                *midpoint,
                (
                    f"{connection.parent_name}({_format_location(connection.parent_x)})"
                    f"→{connection.child_name}[{connection.child_end}]\n"
                    f"{_format_gap_diagnostic(connection, status, precision=4)}"
                ),
                color=color,
                fontsize=7.5,
                zorder=10,
            )
            annotation.set_gid(f"annotation-connection:{connection.child_name}")


def plot_morphology_3d(
    morphology: Morphology,
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
    ax=None,
    figsize: tuple[float, float] = (9, 8),
    dpi: int = 150,
    title: str | None = None,
):
    """Render the authored centerlines in three dimensions."""
    tolerance = _finite(
        connection_tolerance_um,
        name="connection_tolerance_um",
        nonnegative=True,
    )
    diameter_scale = _finite(diameter_scale, name="diameter_scale", positive=True)
    min_linewidth = _finite(min_linewidth, name="min_linewidth", positive=True)
    scene = build_morphology_scene(morphology)
    highlight_names = _highlight_names(scene, highlight)
    context = _color_context(scene, color_by, cmap)
    fig, ax = _new_3d_axes(ax, figsize=figsize, dpi=dpi)

    if not scene.sections:
        ax.text2D(0.5, 0.5, "Empty Morphology", transform=ax.transAxes, ha="center")
        ax.set_title("Empty Morphology" if title is None else title)
    else:
        origin = _display_origin(scene)
        scale = _display_scale(scene, origin)
        for section in scene.sections:
            _draw_3d_section(
                ax,
                section,
                origin=origin,
                scale=scale,
                context=context,
                highlight_names=highlight_names,
                show_points=show_points,
                show_compartments=show_compartments,
                show_orientation=show_orientation,
                annotate_sections=annotate_sections,
                diameter_scale=diameter_scale,
                min_linewidth=min_linewidth,
            )
        if show_connections:
            _draw_3d_connections(
                ax,
                scene,
                origin=origin,
                scale=scale,
                tolerance=tolerance,
                annotate=annotate_connections,
            )
        _set_3d_limits(ax, _shift_xyz(_collect_xyz(scene), origin, scale))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        ax.set_xlabel(_axis_label("x", origin[0], scale[0]))
        ax.set_ylabel(_axis_label("y", origin[1], scale[1]))
        ax.set_zlabel(_axis_label("z", origin[2], scale[2]))
        _annotate_display_transform(ax, origin, scale, (0, 1, 2))
        ax.set_title(
            _geometry_title(scene, dimensionality="3D") if title is None else title
        )
        _section_legend(ax, scene, context, highlight_names, legend=legend)
        _add_color_key(fig, ax, context, legend=legend)
    return fig, ax


def plot_morphology_shape(
    morphology: Morphology,
    *,
    view: str = "y",
    diameter_scale: float = 1.0,
    radial_segments: int = 8,
    connection_tolerance_um: float = 1e-9,
    legend: bool | str = "auto",
    show_axes: bool = False,
    ax=None,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 150,
    title: str | None = None,
    _scene: MorphologyScene | None = None,
):
    """Render a quiet orthographic view of the authored tapered-cable shape."""
    if view not in _VIEW_AXES:
        raise ValueError("view must be exactly 'x', 'y', or 'z'.")
    diameter_scale = _finite(diameter_scale, name="diameter_scale", positive=True)
    radial_segments = _shape_radial_segments(radial_segments)
    tolerance = _finite(
        connection_tolerance_um,
        name="connection_tolerance_um",
        nonnegative=True,
    )
    if not isinstance(show_axes, (bool, np.bool_)):
        raise TypeError("show_axes must be a boolean.")
    scene = build_morphology_scene(morphology) if _scene is None else _scene
    shape_context = _shape_color_context(scene)
    show_legend = _shape_legend(legend, len(shape_context.family_colors))
    context = shape_context.colors
    fig, ax = _new_2d_axes(ax, figsize=figsize, dpi=dpi)

    if not scene.sections:
        ax.text(
            0.5,
            0.5,
            "Empty Morphology",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        if title is not None:
            ax.set_title(title)
        if not show_axes:
            ax.set_axis_off()
        else:
            ax.set_axis_on()
        return fig, ax

    _warn_shape_connection_gaps(scene, connection_tolerance_um=tolerance)
    origin, display_scale, post_scale = _shape_display_transform(
        scene, diameter_scale=diameter_scale
    )
    faces, colors, section_names = _shape_faces(
        scene,
        origin=origin,
        display_scale=display_scale,
        post_scale=post_scale,
        diameter_scale=diameter_scale,
        radial_segments=radial_segments,
        context=context,
    )
    axes, labels = _VIEW_AXES[view]
    omitted_axis = ({0, 1, 2} - set(axes)).pop()
    depth = np.asarray(
        [float(np.mean(face[:, omitted_axis])) for face in faces], dtype=float
    )
    order = np.argsort(depth, kind="stable")
    projected_faces = [faces[index][:, axes] for index in order]
    ordered_colors = [colors[index] for index in order]
    ordered_sections = tuple(section_names[index] for index in order)
    collection = PolyCollection(
        projected_faces,
        facecolors=ordered_colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
        closed=True,
        zorder=2,
    )
    collection.set_gid("morphology-shape")
    collection._dendra_face_sections = ordered_sections
    collection._dendra_section_colors = {
        section.name: context.section_color(section) for section in scene.sections
    }
    collection._dendra_section_color_groups = dict(shape_context.section_families)
    collection._dendra_family_colors = dict(shape_context.family_colors)
    collection._dendra_surface_vertices = np.concatenate(faces, axis=0)
    ax.add_collection(collection)

    projected_vertices = collection._dendra_surface_vertices[:, axes]
    _set_2d_limits(ax, projected_vertices)
    ax.set_aspect("equal")
    if show_axes:
        ax.set_axis_on()
        ax.set_xlabel(
            _shape_axis_label(labels[0][0], origin[axes[0]], display_scale, post_scale)
        )
        ax.set_ylabel(
            _shape_axis_label(labels[1][0], origin[axes[1]], display_scale, post_scale)
        )
    else:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    _shape_family_legend(ax, shape_context, legend=show_legend)
    return fig, ax


def plot_morphology_shape_3d(
    morphology: Morphology,
    *,
    diameter_scale: float = 1.0,
    radial_segments: int = 8,
    connection_tolerance_um: float = 1e-9,
    legend: bool | str = "auto",
    show_axes: bool = False,
    show_axis_indicator: bool = True,
    interactive: bool = False,
    ax=None,
    figsize: tuple[float, float] = (9, 8),
    dpi: int = 150,
    title: str | None = None,
    _scene: MorphologyScene | None = None,
):
    """Render the authored tapered-cable shape as a quiet 3-D surface."""
    diameter_scale = _finite(diameter_scale, name="diameter_scale", positive=True)
    radial_segments = _shape_radial_segments(radial_segments)
    tolerance = _finite(
        connection_tolerance_um,
        name="connection_tolerance_um",
        nonnegative=True,
    )
    if not isinstance(show_axes, (bool, np.bool_)):
        raise TypeError("show_axes must be a boolean.")
    if not isinstance(show_axis_indicator, (bool, np.bool_)):
        raise TypeError("show_axis_indicator must be a boolean.")
    if not isinstance(interactive, (bool, np.bool_)):
        raise TypeError("interactive must be a boolean.")
    scene = build_morphology_scene(morphology) if _scene is None else _scene
    shape_context = _shape_color_context(scene)
    show_legend = _shape_legend(legend, len(shape_context.family_colors))
    context = shape_context.colors
    fig, ax = _new_3d_axes(ax, figsize=figsize, dpi=dpi)
    _replace_shape_axis_indicator(ax, show=bool(show_axis_indicator))
    _configure_shape_3d_interaction(fig, ax, interactive=bool(interactive))

    if not scene.sections:
        ax.text2D(0.5, 0.5, "Empty Morphology", transform=ax.transAxes, ha="center")
        if title is not None:
            ax.set_title(title)
        if not show_axes:
            ax.set_axis_off()
        else:
            ax.set_axis_on()
        return fig, ax

    _warn_shape_connection_gaps(scene, connection_tolerance_um=tolerance)
    origin, display_scale, post_scale = _shape_display_transform(
        scene, diameter_scale=diameter_scale
    )
    faces, colors, section_names = _shape_faces(
        scene,
        origin=origin,
        display_scale=display_scale,
        post_scale=post_scale,
        diameter_scale=diameter_scale,
        radial_segments=radial_segments,
        context=context,
    )
    collection = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
        zsort="average",
    )
    collection.set_gid("morphology-shape")
    collection._dendra_face_sections = tuple(section_names)
    collection._dendra_section_colors = {
        section.name: context.section_color(section) for section in scene.sections
    }
    collection._dendra_section_color_groups = dict(shape_context.section_families)
    collection._dendra_family_colors = dict(shape_context.family_colors)
    collection._dendra_surface_vertices = np.concatenate(faces, axis=0)
    ax.add_collection3d(collection)
    _set_3d_limits(ax, collection._dendra_surface_vertices)
    ax.set_proj_type("ortho")
    if show_axes:
        ax.set_axis_on()
        ax.set_xlabel(_shape_axis_label("x", origin[0], display_scale, post_scale))
        ax.set_ylabel(_shape_axis_label("y", origin[1], display_scale, post_scale))
        ax.set_zlabel(_shape_axis_label("z", origin[2], display_scale, post_scale))
    else:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    _shape_family_legend(ax, shape_context, legend=show_legend)
    return fig, ax


def _compartment_topology_positions(scene: CompartmentTopologyScene) -> np.ndarray:
    """Lay out a deterministic rooted forest with topology depth on x."""
    positions = np.zeros((len(scene.nodes), 2), dtype=float)
    children = dict(scene.children)
    next_leaf = 0.0
    for root in scene.roots:
        stack = [(root, 0, False)]
        while stack:
            node_id, depth, expanded = stack.pop()
            descendants = children[node_id]
            if descendants and not expanded:
                stack.append((node_id, depth, True))
                stack.extend(
                    (child, depth + 1, False) for child in reversed(descendants)
                )
                continue
            if descendants:
                y = float(np.mean([positions[child, 1] for child in descendants]))
            else:
                y = next_leaf
                next_leaf += 1.0
            positions[node_id] = (float(depth), y)
        next_leaf += 1.0
    if len(positions):
        positions[:, 1] -= 0.5 * (
            float(positions[:, 1].min()) + float(positions[:, 1].max())
        )
    return positions


def _compartment_hover_text(node: CompartmentNodeScene) -> str:
    """Return a compact, unit-explicit hover card for one graph node."""
    header = f"Node {node.node_id} · {node.name}"
    connectivity = (
        f"kind: {node.kind} · degree: {node.degree} · "
        f"parent: {node.parent_id if node.parent_id is not None else 'root'}"
    )
    if node.kind == "junction":
        return (
            f"{header}\n{connectivity}\n"
            "zero-area algebraic junction\n"
            "diameter, cm, rhoa, and xyz are not material properties here"
        )
    labels = ", ".join(sorted(node.labels))
    return (
        f"{header}\n"
        f"Section: {node.section_name} · segment: {node.segment_index} · "
        f"x={node.section_x:.6g}\n"
        f"L={node.length_um:.6g} µm · mean diam={node.diameter_um:.6g} µm\n"
        f"cm={node.cm_uF_cm2:.6g} µF/cm² · "
        f"rhoa={node.rhoa_ohm_cm:.6g} Ω·cm\n"
        f"area={node.area_um2:.6g} µm² · volume={node.volume_um3:.6g} µm³\n"
        f"center xyz=({node.x_um:.6g}, {node.y_um:.6g}, {node.z_um:.6g}) µm\n"
        f"{connectivity}\nlabels: {labels}"
    )


class _CompartmentTopologyHover:
    """Dependency-free Matplotlib hover controller retained by its Axes."""

    def __init__(
        self,
        fig,
        ax,
        scene: CompartmentTopologyScene,
        positions: np.ndarray,
        collections,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.scene = scene
        self.positions = positions
        self.collections = tuple(collections)
        self.annotation = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.35",
                "alpha": 0.96,
            },
            arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.8},
            zorder=20,
        )
        self.annotation.set_gid("topology:hover")
        self.annotation.set_visible(False)
        self.connection_id = fig.canvas.mpl_connect("motion_notify_event", self)

    def show_node(self, node_id: int) -> None:
        """Show one node card; useful for programmatic inspection and tests."""
        node = self.scene.node(node_id)
        self.annotation.xy = tuple(self.positions[node_id])
        self.annotation.set_text(_compartment_hover_text(node))
        self.annotation.set_visible(True)
        self.fig.canvas.draw_idle()

    def hide(self) -> None:
        if self.annotation.get_visible():
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()

    def disconnect(self, *, remove_annotation: bool = False) -> None:
        """Disconnect the canvas callback and optionally remove its annotation."""
        self.fig.canvas.mpl_disconnect(self.connection_id)
        self.hide()
        if remove_annotation:
            self.annotation.remove()

    def __call__(self, event) -> None:
        if event.inaxes is not self.ax:
            self.hide()
            return
        for collection in self.collections:
            contains, details = collection.contains(event)
            indexes = details.get("ind", ()) if contains else ()
            if len(indexes):
                node_ids = collection._dendra_topology_node_ids
                self.show_node(int(node_ids[int(indexes[0])]))
                return
        self.hide()


def _clear_compartment_topology_artists(ax) -> None:
    """Remove only artists/controllers owned by an earlier topology render."""
    previous_hover = getattr(ax, "_dendra_topology_hover", None)
    if previous_hover is not None:
        previous_hover.disconnect(remove_annotation=True)
        del ax._dendra_topology_hover
    for artist in tuple(ax.get_children()):
        gid = artist.get_gid()
        if isinstance(gid, str) and gid.startswith("topology:"):
            artist.remove()


def _topology_scatter(
    ax,
    scene: CompartmentTopologyScene,
    positions: np.ndarray,
    node_ids: list[int],
    *,
    colors,
    marker: str,
    size,
    gid: str,
    edgecolors,
    linewidths: float,
):
    if not node_ids:
        return None
    offsets = positions[node_ids]
    collection = ax.scatter(
        offsets[:, 0],
        offsets[:, 1],
        s=size,
        marker=marker,
        facecolors=colors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=5,
    )
    collection.set_gid(gid)
    collection.set_picker(5.0)
    collection._dendra_topology_node_ids = tuple(node_ids)
    collection._dendra_topology_scene = scene
    return collection


def _bounded_display_value(
    value: float,
    *,
    minimum: float,
    maximum: float | None,
) -> float:
    """Clamp one finite display value without changing its physical source."""
    value = max(float(value), minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _material_node_size(
    node: CompartmentNodeScene,
    *,
    node_scale: float,
    min_node_size: float,
    max_node_size: float | None,
) -> float:
    """Map material diameter in µm to Matplotlib marker area in points²."""
    return _bounded_display_value(
        node.diameter_um * node_scale,
        minimum=min_node_size,
        maximum=max_node_size,
    )


def _material_branchpoint_size(
    node: CompartmentNodeScene,
    *,
    node_scale: float,
    min_node_size: float,
    max_node_size: float | None,
    branchpoint_scale: float,
    min_branchpoint_size: float,
    max_branchpoint_size: float | None,
) -> float:
    """Scale a material fork from its already bounded material-node area."""
    ordinary_size = _material_node_size(
        node,
        node_scale=node_scale,
        min_node_size=min_node_size,
        max_node_size=max_node_size,
    )
    return _bounded_display_value(
        ordinary_size * branchpoint_scale,
        minimum=min_branchpoint_size,
        maximum=max_branchpoint_size,
    )


def _topology_edge_width(
    first: CompartmentNodeScene,
    second: CompartmentNodeScene,
    *,
    edge_scale: float,
    min_edge_width: float,
    max_edge_width: float | None,
) -> float:
    """Scale an edge from material endpoints, ignoring algebraic storage."""
    material_diameters = [
        node.diameter_um for node in (first, second) if node.kind == "compartment"
    ]
    if material_diameters:
        diameter = float(np.mean(material_diameters))
        value = diameter * edge_scale
    else:
        # A valid compiled graph should not contain a junction--junction edge,
        # but the visible floor is a safe and truthful structural fallback.
        value = min_edge_width
    return _bounded_display_value(
        value,
        minimum=min_edge_width,
        maximum=max_edge_width,
    )


def _compartment_topology_legend(
    ax,
    section_scene: MorphologyScene,
    context: _ColorContext,
    highlight_names,
    topology_scene: CompartmentTopologyScene,
    *,
    legend: bool,
) -> None:
    if not legend or not section_scene.sections:
        return
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=context.section_color(section),
            markeredgecolor="black",
            markeredgewidth=0.25,
            alpha=_section_alpha(section, highlight_names),
            label=section.name,
        )
        for section in section_scene.sections
    ]
    if any(
        node.is_branchpoint and node.kind == "compartment"
        for node in topology_scene.nodes
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="D",
                markersize=7,
                markerfacecolor="#9a9a9a",
                markeredgecolor="black",
                label="forking compartment (Section-colored)",
            )
        )
    if any(node.kind == "junction" for node in topology_scene.nodes):
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="X",
                markersize=7,
                markerfacecolor="#303030",
                markeredgecolor="white",
                label="algebraic junction",
            )
        )
    topology_legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        title="Sections and node kinds",
    )
    topology_legend.set_gid("topology:legend")


def plot_morphology_topology(
    morphology: Morphology,
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
    ax=None,
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 150,
    title: str | None = None,
    _scene: MorphologyScene | None = None,
    _topology_scene: CompartmentTopologyScene | None = None,
):
    """Draw the exact flat compartment connectivity graph."""
    if not isinstance(interactive, (bool, np.bool_)):
        raise TypeError("interactive must be a boolean.")
    node_scale = _finite(node_scale, name="node_scale", positive=True)
    min_node_size = _finite(min_node_size, name="min_node_size", positive=True)
    if max_node_size is not None:
        max_node_size = _finite(max_node_size, name="max_node_size", positive=True)
        if max_node_size < min_node_size:
            raise ValueError(
                "max_node_size must be greater than or equal to min_node_size."
            )
    branchpoint_scale = _finite(
        branchpoint_scale, name="branchpoint_scale", positive=True
    )
    min_branchpoint_size = _finite(
        min_branchpoint_size, name="min_branchpoint_size", positive=True
    )
    if max_branchpoint_size is not None:
        max_branchpoint_size = _finite(
            max_branchpoint_size, name="max_branchpoint_size", positive=True
        )
        if max_branchpoint_size < min_branchpoint_size:
            raise ValueError(
                "max_branchpoint_size must be greater than or equal to "
                "min_branchpoint_size."
            )
    junction_size = _finite(junction_size, name="junction_size", positive=True)
    edge_scale = _finite(edge_scale, name="edge_scale", positive=True)
    min_edge_width = _finite(min_edge_width, name="min_edge_width", positive=True)
    if max_edge_width is not None:
        max_edge_width = _finite(max_edge_width, name="max_edge_width", positive=True)
        if max_edge_width < min_edge_width:
            raise ValueError(
                "max_edge_width must be greater than or equal to min_edge_width."
            )
    if node_size is not None:
        node_size = _finite(node_size, name="node_size", positive=True)
    if branchpoint_size is not None:
        branchpoint_size = _finite(
            branchpoint_size, name="branchpoint_size", positive=True
        )
    if edge_width is not None:
        edge_width = _finite(edge_width, name="edge_width", positive=True)
    section_scene = build_morphology_scene(morphology) if _scene is None else _scene
    topology_scene = (
        build_compartment_topology_scene(
            morphology,
            morphology_scene=section_scene,
        )
        if _topology_scene is None
        else _topology_scene
    )
    highlight_names = _highlight_names(section_scene, highlight)
    context = _color_context(section_scene, "section", "viridis")
    fig, ax = _new_2d_axes(ax, figsize=figsize, dpi=dpi)
    _clear_compartment_topology_artists(ax)

    if not topology_scene.nodes:
        ax._dendra_topology_scene = topology_scene
        ax._dendra_topology_positions = np.empty((0, 2), dtype=float)
        _empty_axes(ax, title=title)
        ax.texts[-1].set_gid("topology:empty")
        ax.set_axis_off()
        return fig, ax

    positions = _compartment_topology_positions(topology_scene)

    neighbors: list[list[int]] = [[] for _ in topology_scene.nodes]
    for parent, child in topology_scene.edges:
        neighbors[parent].append(child)
        neighbors[child].append(parent)

    def selected_material(node: CompartmentNodeScene) -> bool:
        return highlight_names is None or (
            node.section_name is not None and node.section_name in highlight_names
        )

    def highlighted(node: CompartmentNodeScene) -> bool:
        if selected_material(node):
            return True
        return node.kind == "junction" and any(
            selected_material(topology_scene.nodes[neighbor])
            for neighbor in neighbors[node.node_id]
        )

    edge_segments = [
        positions[[parent, child]] for parent, child in topology_scene.edges
    ]
    edge_colors = []
    edge_widths = []
    for parent, child in topology_scene.edges:
        strong = selected_material(topology_scene.nodes[parent]) or selected_material(
            topology_scene.nodes[child]
        )
        alpha = (
            0.85 if highlight_names is None or strong else _MUTED_TOPOLOGY_EDGE_ALPHA
        )
        edge_colors.append((0.0, 0.0, 0.0, alpha))
        edge_widths.append(
            edge_width
            if edge_width is not None
            else _topology_edge_width(
                topology_scene.nodes[parent],
                topology_scene.nodes[child],
                edge_scale=edge_scale,
                min_edge_width=min_edge_width,
                max_edge_width=max_edge_width,
            )
        )
    if edge_segments:
        edges = LineCollection(
            edge_segments,
            colors=edge_colors,
            linewidths=edge_widths,
            capstyle="round",
            zorder=1,
        )
        edges.set_gid("topology:edges")
        edges._dendra_topology_edges = topology_scene.edges
        ax.add_collection(edges)

    section_colors = {
        section.name: context.section_color(section)
        for section in section_scene.sections
    }

    def node_color(node: CompartmentNodeScene):
        if node.kind == "junction":
            alpha = 0.92 if highlighted(node) else _MUTED_ALPHA
            return (0.18, 0.18, 0.18, alpha)
        alpha = 1.0 if highlighted(node) else _MUTED_ALPHA
        return _with_alpha(section_colors[node.section_name], alpha)

    ordinary = [
        node.node_id
        for node in topology_scene.nodes
        if node.kind == "compartment" and not node.is_branchpoint
    ]
    material_forks = [
        node.node_id
        for node in topology_scene.nodes
        if node.kind == "compartment" and node.is_branchpoint
    ]
    junctions = [
        node.node_id for node in topology_scene.nodes if node.kind == "junction"
    ]

    ordinary_sizes = (
        node_size
        if node_size is not None
        else [
            _material_node_size(
                topology_scene.nodes[node_id],
                node_scale=node_scale,
                min_node_size=min_node_size,
                max_node_size=max_node_size,
            )
            for node_id in ordinary
        ]
    )
    fork_sizes = (
        branchpoint_size
        if branchpoint_size is not None
        else [
            _material_branchpoint_size(
                topology_scene.nodes[node_id],
                node_scale=node_scale,
                min_node_size=min_node_size,
                max_node_size=max_node_size,
                branchpoint_scale=branchpoint_scale,
                min_branchpoint_size=min_branchpoint_size,
                max_branchpoint_size=max_branchpoint_size,
            )
            for node_id in material_forks
        ]
    )
    junction_marker_size = (
        branchpoint_size if branchpoint_size is not None else junction_size
    )

    collections = []
    ordinary_collection = _topology_scatter(
        ax,
        topology_scene,
        positions,
        ordinary,
        colors=[node_color(topology_scene.nodes[node_id]) for node_id in ordinary],
        marker="o",
        size=ordinary_sizes,
        gid="topology:compartments",
        edgecolors="black",
        linewidths=0.25,
    )
    if ordinary_collection is not None:
        collections.append(ordinary_collection)
    fork_collection = _topology_scatter(
        ax,
        topology_scene,
        positions,
        material_forks,
        colors=[
            node_color(topology_scene.nodes[node_id]) for node_id in material_forks
        ],
        marker="D",
        size=fork_sizes,
        gid="topology:branchpoints",
        edgecolors="black",
        linewidths=1.1,
    )
    if fork_collection is not None:
        collections.insert(0, fork_collection)
    junction_collection = _topology_scatter(
        ax,
        topology_scene,
        positions,
        junctions,
        colors=[node_color(topology_scene.nodes[node_id]) for node_id in junctions],
        marker="X",
        size=junction_marker_size,
        gid="topology:junctions",
        edgecolors="white",
        linewidths=0.7,
    )
    if junction_collection is not None:
        collections.insert(0, junction_collection)

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    spans = maxs - mins
    pads = np.where(spans > 0.0, 0.08 * spans, 0.6)
    ax.set_xlim(float(mins[0] - pads[0]), float(maxs[0] + pads[0]))
    ax.set_ylim(float(mins[1] - pads[1]), float(maxs[1] + pads[1]))
    # The coordinates encode only graph depth and leaf order. Let each axis
    # fill its available display extent so broad trees do not collapse depth
    # into a few pixels merely to preserve a meaningless 1:1 data aspect.
    ax.set_aspect("auto")
    ax.set_axis_off()
    branchpoint_count = sum(node.is_branchpoint for node in topology_scene.nodes)
    default_title = (
        f"Compartment topology · {len(topology_scene.nodes)} nodes · "
        f"{branchpoint_count} branchpoint"
        f"{'s' if branchpoint_count != 1 else ''}"
    )
    ax.set_title(default_title if title is None else title)
    _compartment_topology_legend(
        ax,
        section_scene,
        context,
        highlight_names,
        topology_scene,
        legend=legend,
    )
    ax._dendra_topology_scene = topology_scene
    ax._dendra_topology_positions = positions.copy()
    if interactive:
        backend = str(mpl.get_backend()).casefold()
        if "matplotlib_inline" in backend or backend == "inline":
            warnings.warn(
                "interactive=True requested while Matplotlib is using its "
                "static inline backend; the topology will render, but hover "
                "events cannot be delivered. Install the optional backend "
                "with `python -m pip install ipympl`. If the kernel and "
                "Jupyter server use separate "
                "environments, install a compatible ipympl in both. Stop and "
                "restart the entire Jupyter server—not only the kernel—then "
                "refresh the page and run `%matplotlib widget` before creating "
                "the figure; otherwise use interactive=False.",
                RuntimeWarning,
                stacklevel=2,
            )
        controller = _CompartmentTopologyHover(
            fig,
            ax,
            topology_scene,
            positions,
            collections,
        )
        ax._dendra_topology_hover = controller
    return fig, ax


def _topology_positions(scene: MorphologyScene, *, x_spacing: float, y_spacing: float):
    """Lay out an ordered forest without depending on authored coordinates."""
    children = dict(scene.children)
    positions: dict[str, tuple[float, float]] = {}
    next_leaf = 0.0
    for root in scene.roots:
        stack = [(root, 0, False)]
        while stack:
            name, depth, expanded = stack.pop()
            descendants = children[name]
            if not expanded and descendants:
                stack.append((name, depth, True))
                stack.extend(
                    (child, depth + 1, False) for child in reversed(descendants)
                )
                continue
            if descendants:
                x = float(np.mean([positions[child][0] for child in descendants]))
            else:
                x = next_leaf
                next_leaf += x_spacing
            positions[name] = (x, -float(depth) * y_spacing)
        next_leaf += 0.75 * x_spacing

    # A corrupted cycle can have no roots; scene extraction already rejects it.
    # Retain a deterministic fallback for an empty root tuple with no Sections.
    return positions


def _section_topology_node_text(
    section: SectionScene,
    *,
    show_parameters: bool,
    show_labels: bool,
    show_compartments: bool,
) -> str:
    lines = [section.name]
    if show_parameters:
        geometry = "pt3d" if section.is_pt3d else "stylized"
        line = f"{geometry} · L={section.L:.5g} µm"
        if show_compartments:
            line += f" · nseg={section.nseg}"
        lines.append(line)
        lines.append(f"rhoa={section.rhoa:.5g} Ω·cm · cm={section.cm:.5g} µF/cm²")
    elif show_compartments:
        lines.append(f"nseg={section.nseg}")
    if show_labels:
        labels = sorted(section.labels.difference({section.name}))
        if labels:
            lines.append("labels: " + ", ".join(labels))
    return "\n".join(lines)


def _draw_topology_compartments(
    ax,
    section: SectionScene,
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    color,
    alpha: float,
) -> None:
    x, y = center
    left = x - 0.42 * width
    right = x + 0.42 * width
    baseline = y - 0.37 * height
    segments = [[(left, baseline), (right, baseline)]]
    if section.nseg <= 32:
        for index in range(section.nseg + 1):
            tick_x = left + (right - left) * index / section.nseg
            segments.append(
                [
                    (tick_x, baseline - 0.035 * height),
                    (tick_x, baseline + 0.035 * height),
                ]
            )
    collection = LineCollection(
        segments,
        colors=[color],
        linewidths=0.8,
        alpha=alpha,
        zorder=5,
    )
    collection.set_gid(f"compartments:{section.name}")
    ax.add_collection(collection)


def plot_morphology_section_topology(
    morphology: Morphology,
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
    ax=None,
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 150,
    title: str | None = None,
    _scene: MorphologyScene | None = None,
):
    """Draw the legacy deterministic Section-tree/forest schematic."""
    tolerance = _finite(
        connection_tolerance_um,
        name="connection_tolerance_um",
        nonnegative=True,
    )
    scene = build_morphology_scene(morphology) if _scene is None else _scene
    highlight_names = _highlight_names(scene, highlight)
    context = _color_context(scene, color_by, cmap)
    fig, ax = _new_2d_axes(ax, figsize=figsize, dpi=dpi)

    if not scene.sections:
        _empty_axes(ax, title=title)
        ax.set_axis_off()
        return fig, ax

    node_text = {
        section.name: _section_topology_node_text(
            section,
            show_parameters=show_parameters,
            show_labels=show_labels,
            show_compartments=show_compartments,
        )
        for section in scene.sections
    }
    line_count = max(text.count("\n") + 1 for text in node_text.values())
    longest_line = max(
        len(line) for text in node_text.values() for line in text.splitlines()
    )
    minimum_width = 1.65 if show_parameters or show_labels else 1.15
    width = max(minimum_width, 0.072 * longest_line + 0.45)
    height = 0.38 + 0.22 * line_count
    positions = _topology_positions(
        scene,
        x_spacing=width + 0.5,
        y_spacing=height + 0.65,
    )

    for connection in scene.connections:
        parent = positions[connection.parent_name]
        child = positions[connection.child_name]
        color, linestyle, status = _connection_style(connection, tolerance)
        arrow = FancyArrowPatch(
            (parent[0], parent[1] - 0.5 * height),
            (child[0], child[1] + 0.5 * height),
            arrowstyle="-|>",
            mutation_scale=10,
            connectionstyle="arc3,rad=0.0",
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
            zorder=1,
        )
        arrow.set_gid(f"connection:{connection.child_name}")
        ax.add_patch(arrow)
        if annotate_connections:
            midpoint = (
                0.5 * (parent[0] + child[0]),
                0.5 * (parent[1] + child[1]),
            )
            gap = (
                "\n"
                + _format_gap_diagnostic(
                    connection,
                    status,
                    precision=4,
                    include_status=False,
                )
                if connection.gap_um
                else ""
            )
            annotation = ax.text(
                midpoint[0],
                midpoint[1],
                (
                    f"x={_format_location(connection.parent_x)} → "
                    f"end {connection.child_end}\nhost seg "
                    f"{connection.parent_host_segment}{gap}"
                ),
                ha="center",
                va="center",
                fontsize=7.0,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                    "pad": 1.0,
                },
                zorder=2,
            )
            annotation.set_gid(f"annotation-connection:{connection.child_name}")

    for section in scene.sections:
        x, y = positions[section.name]
        alpha = _section_alpha(section, highlight_names)
        color = context.section_color(section)
        node = FancyBboxPatch(
            (x - 0.5 * width, y - 0.5 * height),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=_blend_with_white(color, 0.34 if alpha == 1.0 else 0.14),
            edgecolor=_with_alpha(color, max(alpha, 0.45)),
            linewidth=1.5,
            zorder=3,
        )
        node.set_gid(f"section:{section.name}")
        ax.add_patch(node)
        annotation = ax.text(
            x,
            y + (0.055 * height if show_compartments else 0.0),
            node_text[section.name],
            ha="center",
            va="center",
            fontsize=8.2,
            color="black",
            alpha=max(alpha, 0.55),
            zorder=4,
        )
        annotation.set_gid(f"annotation-section:{section.name}")
        if show_compartments:
            _draw_topology_compartments(
                ax,
                section,
                center=(x, y),
                width=width,
                height=height,
                color=color,
                alpha=max(alpha, 0.45),
            )

    values = np.asarray(list(positions.values()), dtype=float)
    x_pad = 0.65 * width
    y_pad = 0.75 * height
    ax.set_xlim(float(values[:, 0].min() - x_pad), float(values[:, 0].max() + x_pad))
    ax.set_ylim(float(values[:, 1].min() - y_pad), float(values[:, 1].max() + y_pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    default_title = (
        f"Section topology · {len(scene.sections)} Sections · "
        f"{len(scene.roots)} root{'s' if len(scene.roots) != 1 else ''}"
    )
    ax.set_title(default_title if title is None else title)
    _section_legend(ax, scene, context, highlight_names, legend=legend)
    _add_color_key(fig, ax, context, legend=legend)
    return fig, ax


def _selected_section_names(morphology: Morphology, sections) -> tuple[str, ...] | None:
    if sections is None:
        return None
    if isinstance(sections, (str, Section)):
        values = (sections,)
    else:
        try:
            values = tuple(sections)
        except TypeError as error:
            raise TypeError(
                "sections must be a Section, exact name, or iterable of either."
            ) from error
    names = []
    for value in values:
        if isinstance(value, (str, Section)):
            names.append(morphology._resolve_section(value).name)
        else:
            raise TypeError(
                "sections must contain only canonical Sections or exact names."
            )
    if len(set(names)) != len(names):
        raise ValueError("sections must not contain duplicate Section selections.")
    return tuple(names)


def plot_morphology_diameter_profile(
    morphology: Morphology,
    *,
    sections: str | Section | Iterable[str | Section] | None = None,
    highlight: str | Iterable[str] | None = None,
    x_axis: str = "normalized",
    show_points: bool = True,
    show_compartments: bool = True,
    show_connections: bool = True,
    legend: bool = True,
    ax=None,
    figsize: tuple[float, float] = (9, 5),
    dpi: int = 150,
    title: str | None = None,
    _scene: MorphologyScene | None = None,
):
    """Plot each selected Section's authored diameter profile."""
    if x_axis not in ("normalized", "distance"):
        raise ValueError("x_axis must be exactly 'normalized' or 'distance'.")
    selected_names = _selected_section_names(morphology, sections)
    scene = build_morphology_scene(morphology) if _scene is None else _scene
    highlight_names = _highlight_names(scene, highlight)
    selected = tuple(
        section
        for section in scene.sections
        if selected_names is None or section.name in selected_names
    )
    context = _color_context(scene, "section", "viridis")
    fig, ax = _new_2d_axes(ax, figsize=figsize, dpi=dpi)

    if not selected:
        _empty_axes(ax, title=title, text="No selected Sections")
        ax.set_xlabel(
            "Normalized Section position x"
            if x_axis == "normalized"
            else "Distance from authored x=0 (µm)"
        )
        ax.set_ylabel("Diameter (µm)")
        return fig, ax

    for section in selected:
        alpha = _section_alpha(section, highlight_names)
        color = context.section_color(section)
        scale = 1.0 if x_axis == "normalized" else section.L
        x_values = np.asarray(section.sample_x) * scale
        diameters = np.asarray([point[3] for point in section.points])
        (profile,) = ax.plot(
            x_values,
            diameters,
            color=color,
            linewidth=2.0,
            alpha=alpha,
            label=section.name,
            zorder=3,
        )
        profile.set_gid(f"diameter-profile:{section.name}")
        if show_points:
            ax.scatter(
                x_values,
                diameters,
                s=28,
                facecolors=[color],
                edgecolors="white",
                linewidths=0.65,
                alpha=alpha,
                zorder=5,
            )
        if show_compartments:
            center_x = np.asarray(section.center_x) * scale
            center_diameter = [point[3] for point in section.center_points]
            ax.scatter(
                center_x,
                center_diameter,
                s=22,
                facecolors="white",
                edgecolors=[color],
                linewidths=0.9,
                alpha=alpha,
                zorder=6,
            )
            boundary_x = np.asarray(section.boundary_x) * scale
            boundary_diameter = [point[3] for point in section.boundary_points]
            ax.scatter(
                boundary_x,
                boundary_diameter,
                s=30,
                color=[color],
                marker="|",
                linewidths=0.8,
                alpha=alpha,
                zorder=4,
            )

    if show_connections:
        selected_set = {section.name for section in selected}
        for connection in scene.connections:
            if connection.parent_name in selected_set:
                parent = scene.section(connection.parent_name)
                scale = 1.0 if x_axis == "normalized" else parent.L
                ax.scatter(
                    [connection.parent_x * scale],
                    [connection.parent_point[3]],
                    s=42,
                    facecolors="none",
                    edgecolors="#333333",
                    marker="D",
                    linewidths=1.0,
                    zorder=8,
                )
            if connection.child_name in selected_set:
                child = scene.section(connection.child_name)
                scale = 1.0 if x_axis == "normalized" else child.L
                ax.scatter(
                    [connection.child_end * scale],
                    [connection.child_point[3]],
                    s=42,
                    color="#333333",
                    marker="x",
                    linewidths=1.0,
                    zorder=8,
                )

    if x_axis == "normalized":
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalized Section position x")
    else:
        ax.set_xlim(left=0.0)
        ax.set_xlabel("Distance from authored x=0 (µm)")
    ax.set_ylabel("Diameter (µm)")
    ax.grid(True, linewidth=0.45, alpha=0.25)
    ax.set_title("Authored diameter profiles" if title is None else title)
    if legend:
        handles = [
            Line2D(
                [0],
                [0],
                color=context.section_color(section),
                alpha=_section_alpha(section, highlight_names),
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=section.name,
            )
            for section in selected
        ]
        ax.legend(handles=handles, frameon=False, title="Sections")
    return fig, ax


def inspect_morphology(
    morphology: Morphology,
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
):
    """Build coordinated geometry, topology, and diameter-profile panels."""
    tolerance = _finite(
        connection_tolerance_um,
        name="connection_tolerance_um",
        nonnegative=True,
    )
    scene = build_morphology_scene(morphology)
    topology_scene = build_compartment_topology_scene(
        morphology,
        morphology_scene=scene,
    )
    # Validate once before any artists are added, keeping dashboard creation
    # transactional with respect to user input errors.
    _highlight_names(scene, highlight)
    fig = plt.figure(figsize=_validate_figsize(figsize), dpi=_validate_dpi(dpi))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.08))
    axes = {
        "x": fig.add_subplot(grid[0, 0]),
        "y": fig.add_subplot(grid[0, 1]),
        "z": fig.add_subplot(grid[0, 2]),
        "topology": fig.add_subplot(grid[1, :2]),
        "diameter": fig.add_subplot(grid[1, 2]),
    }
    for view in ("x", "y", "z"):
        plot_morphology(
            morphology,
            view=view,
            color_by="section",
            highlight=highlight,
            show_points=show_points,
            show_connections=True,
            show_compartments=show_compartments,
            show_orientation=show_orientation,
            annotate_sections=annotate_sections,
            # Show optional spatial details once rather than repeating them in
            # all projections; the topology panel itself stays text-free.
            annotate_connections=annotate_connections and view == "z",
            connection_tolerance_um=tolerance,
            legend=False,
            ax=axes[view],
            title=f"View along {view}",
            _scene=scene,
        )
    plot_morphology_topology(
        morphology,
        highlight=highlight,
        interactive=False,
        legend=False,
        ax=axes["topology"],
        title="Compartment topology",
        _scene=scene,
        _topology_scene=topology_scene,
    )
    plot_morphology_diameter_profile(
        morphology,
        highlight=highlight,
        x_axis="normalized",
        show_points=show_points,
        show_compartments=show_compartments,
        show_connections=True,
        legend=True,
        ax=axes["diameter"],
        title="Diameter vs Section x",
        _scene=scene,
    )
    if title is None:
        gaps = sum(connection.gap_um > tolerance for connection in scene.connections)
        title = (
            f"Morphology inspection · {len(scene.sections)} Sections · "
            f"{sum(section.nseg for section in scene.sections)} declared compartments · "
            f"{gaps} spatial gap{'s' if gaps != 1 else ''}"
        )
    fig.suptitle(title, fontsize=14, weight="semibold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig, axes


__all__ = [
    "inspect_morphology",
    "plot_morphology",
    "plot_morphology_3d",
    "plot_morphology_diameter_profile",
    "plot_morphology_shape",
    "plot_morphology_shape_3d",
    "plot_morphology_section_topology",
    "plot_morphology_topology",
]
