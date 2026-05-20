from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

from dendra.models.core import (
    Axon,
    Myelinated,
    Population,
    SingleCompartment,
    Unmyelinated,
)
from dendra.models.extcell import ExtCellAxon, ExtCellTree
from dendra.models.tree import Tree


def _parent_map(classes: Sequence[type]) -> dict[type, type | None]:
    class_set = set(classes)
    parent_map: dict[type, type | None] = {}

    for cls in classes:
        internal_parents = [base for base in cls.__bases__ if base in class_set]
        if len(internal_parents) > 1:
            raise ValueError(
                "draw_class_hierarchy currently assumes single inheritance "
                "within the selected class set."
            )
        parent_map[cls] = internal_parents[0] if internal_parents else None

    return parent_map


def _children_map(classes: Sequence[type]) -> dict[type, list[type]]:
    parent_map = _parent_map(classes)
    children: dict[type, list[type]] = {cls: [] for cls in classes}

    for cls in classes:
        parent = parent_map[cls]
        if parent is not None:
            children[parent].append(cls)

    return children


def _depth_map(classes: Sequence[type]) -> dict[type, int]:
    parent_map = _parent_map(classes)
    depth_map: dict[type, int] = {}

    def depth(cls: type) -> int:
        if cls in depth_map:
            return depth_map[cls]
        parent = parent_map[cls]
        depth_map[cls] = 0 if parent is None else depth(parent) + 1
        return depth_map[cls]

    for cls in classes:
        depth(cls)

    return depth_map


def _tree_positions(
    classes: Sequence[type],
    root_gap: float = 1.0,
) -> dict[type, tuple[float, float]]:
    parent_map = _parent_map(classes)
    children = _children_map(classes)
    depth_map = _depth_map(classes)

    roots = [cls for cls in classes if parent_map[cls] is None]
    positions: dict[type, tuple[float, float]] = {}
    cursor = 0.0

    def assign(node: type) -> float:
        nonlocal cursor
        kids = children[node]

        if not kids:
            x = cursor
            cursor += 1.0
        else:
            child_x = [assign(child) for child in kids]
            x = (child_x[0] + child_x[-1]) / 2.0

        positions[node] = (x, -float(depth_map[node]))
        return x

    for i, root in enumerate(roots):
        if i > 0:
            cursor += root_gap
        assign(root)

    return positions


def _node_label_parts(
    cls: type,
    descriptions: dict[type, str],
    wrap_width: int = 28,
) -> tuple[str, str]:
    class_name = cls.__name__
    desc = descriptions.get(cls, "").strip()
    if desc:
        desc = fill(desc, width=wrap_width)
    return class_name, desc


def _rounded_elbow_path(
    start: tuple[float, float],
    end: tuple[float, float],
    corner_radius: float,
    elbow_fraction: float,
) -> MplPath:
    """Build a downward elbow connector with two rounded corners."""
    sx, sy = start
    ex, ey = end

    if abs(ex - sx) < 1e-9 or abs(ey - sy) < 1e-9:
        vertices = [start, end]
        codes = [MplPath.MOVETO, MplPath.LINETO]
        return MplPath(vertices, codes)

    mid_y = sy + elbow_fraction * (ey - sy)

    sign_x = 1.0 if ex > sx else -1.0
    dx = abs(ex - sx)
    dy_top = abs(sy - mid_y)
    dy_bottom = abs(mid_y - ey)

    radius = min(corner_radius, dx / 2.0, dy_top / 2.0, dy_bottom / 2.0)

    if radius < 1e-9:
        vertices = [start, (sx, mid_y), (ex, mid_y), end]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
        return MplPath(vertices, codes)

    pre_corner_1 = (sx, mid_y + radius)
    post_corner_1 = (sx + sign_x * radius, mid_y)
    pre_corner_2 = (ex - sign_x * radius, mid_y)
    post_corner_2 = (ex, mid_y - radius)

    vertices = [
        start,
        pre_corner_1,
        (sx, mid_y),
        post_corner_1,
        pre_corner_2,
        (ex, mid_y),
        post_corner_2,
        end,
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
    ]
    return MplPath(vertices, codes)


def _zorder_map(
    classes: Sequence[type],
    depth_map: dict[type, int],
    layer_stride: float = 10.0,
    base_zorder: float = 100.0,
) -> dict[type, dict[str, float]]:
    """
    Create depth-aware z-orders.

    For each depth level:
    - the node patch sits above connectors originating from that node
    - node text sits above the patch
    - shallower layers sit above deeper layers

    This yields the intended visual effect for inheritance diagrams:
    each branch tucks under its parent node, while still passing above deeper
    descendants and their connectors.
    """
    if not classes:
        return {}

    max_depth = max(depth_map.values())
    zorders: dict[type, dict[str, float]] = {}

    for cls in classes:
        depth = depth_map[cls]
        layer_base = base_zorder + (max_depth - depth) * layer_stride
        zorders[cls] = {
            "connector": layer_base + 1.0,
            "patch": layer_base + 2.0,
            "text": layer_base + 3.0,
        }

    return zorders


def mermaid_from_classes(
    classes: Sequence[type],
    descriptions: dict[type, str] | None = None,
    direction: str = "TD",
) -> str:
    descriptions = descriptions or {}
    parent_map = _parent_map(classes)

    lines = [f"flowchart {direction}"]

    for cls in classes:
        label = cls.__name__
        desc = descriptions.get(cls, "").strip()
        if desc:
            label += "<br/>" + desc
        lines.append(f'    {cls.__name__}["{label}"]')

    for cls in classes:
        parent = parent_map[cls]
        if parent is not None:
            lines.append(f"    {parent.__name__} --> {cls.__name__}")

    return "\n".join(lines)


def draw_class_hierarchy(
    classes: Sequence[type],
    descriptions: dict[type, str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    x_spacing: float = 3.8,
    y_spacing: float = 2.4,
    box_width: float = 3.5,
    box_height: float = 1.0,
    font_size: float = 10,
    class_font_size: float | None = None,
    description_font_size: float | None = None,
    class_font_weight: str = "bold",
    class_color: str = "black",
    description_color: str = "#6E6E6E",
    description_wrap_width: int = 28,
    connector_linewidth: float = 2.8,
    connector_corner_radius: float = 0.55,
    connector_elbow_fraction: float = 0.50,
    arrow_mutation_scale: float = 16,
    zorder_layer_stride: float = 10.0,
    zorder_base: float = 100.0,
    save_path: str | Path | None = None,
    dpi: int = 200,
    highlight: Sequence[type] | None = None,
):
    descriptions = descriptions or {}

    if class_font_size is None:
        class_font_size = font_size + 2.4
    if description_font_size is None:
        description_font_size = max(font_size - 1.4, 7.4)

    raw_positions = _tree_positions(classes)
    parent_map = _parent_map(classes)
    depth_map = _depth_map(classes)
    zorders = _zorder_map(
        classes,
        depth_map,
        layer_stride=zorder_layer_stride,
        base_zorder=zorder_base,
    )

    positions = {
        cls: (x * x_spacing, y * y_spacing) for cls, (x, y) in raw_positions.items()
    }

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    ncols = int(
        max(x for x, _ in raw_positions.values())
        - min(x for x, _ in raw_positions.values())
        + 1
    )
    nlevels = max(depth_map.values()) + 1

    if figsize is None:
        figsize = (max(10.0, 2.8 * ncols), max(4.5, 2.6 * nlevels))

    fig, ax = plt.subplots(figsize=figsize)

    for cls in classes:
        parent = parent_map[cls]
        if parent is None:
            continue

        x0, y0 = positions[parent]
        x1, y1 = positions[cls]

        start = (x0, y0 - box_height / 2.0)
        end = (x1, y1 + box_height / 2.0)
        path = _rounded_elbow_path(
            start,
            end,
            corner_radius=connector_corner_radius,
            elbow_fraction=connector_elbow_fraction,
        )

        connector = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=arrow_mutation_scale,
            linewidth=connector_linewidth,
            edgecolor="black",
            facecolor="black",
            joinstyle="round",
            capstyle="round",
            shrinkA=0,
            shrinkB=0,
            zorder=zorders[parent]["connector"],
        )
        ax.add_patch(connector)

    for cls in classes:
        x, y = positions[cls]
        if highlight is not None and cls in highlight:
            edgecolor = "blue"
        else:
            edgecolor = "black"
        patch = FancyBboxPatch(
            (x - box_width / 2.0, y - box_height / 2.0),
            box_width,
            box_height,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor="white",
            zorder=zorders[cls]["patch"],
        )
        ax.add_patch(patch)

        class_name, desc = _node_label_parts(
            cls,
            descriptions,
            wrap_width=description_wrap_width,
        )

        if desc:
            class_y = y + box_height * 0.20
            desc_y = y - box_height * 0.14
        else:
            class_y = y
            desc_y = y

        ax.text(
            x,
            class_y,
            class_name,
            ha="center",
            va="center",
            fontsize=class_font_size,
            fontweight=class_font_weight,
            color=class_color,
            multialignment="center",
            clip_path=patch,
            zorder=zorders[cls]["text"],
        )

        if desc:
            ax.text(
                x,
                desc_y,
                desc,
                ha="center",
                va="center",
                fontsize=description_font_size,
                color=description_color,
                linespacing=1.15,
                multialignment="center",
                clip_path=patch,
                zorder=zorders[cls]["text"],
            )

    ax.set_xlim(min(xs) - box_width, max(xs) + box_width)
    ax.set_ylim(min(ys) - box_height, max(ys) + box_height)
    ax.axis("off")

    if title:
        ax.set_title(title, pad=18)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


classes = [
    Population,
    SingleCompartment,
    Axon,
    Unmyelinated,
    Myelinated,
    ExtCellAxon,
    Tree,
    ExtCellTree,
]

highlight = [
    SingleCompartment,
    Unmyelinated,
    Myelinated,
    ExtCellAxon,
    Tree,
    ExtCellTree,
]

descriptions = {
    Population: "generic multicompartment population",
    SingleCompartment: "point-neuron / one-compartment population",
    Axon: "1D cable-like axonal fiber population",
    Unmyelinated: "uniform spatial discretization",
    Myelinated: "nodes of Ranvier + diameter-scaled internodes",
    ExtCellAxon: "axon with two-layer extracellular coupling (accurate myelination)",
    Tree: "branched neuron morphology",
    ExtCellTree: "tree with two-layer extracellular coupling (accurate myelination)",
}


def render():
    _ = draw_class_hierarchy(
        classes,
        highlight=highlight,
        descriptions=descriptions,
        description_font_size=10.0,
        class_font_size=11.0,
    )
