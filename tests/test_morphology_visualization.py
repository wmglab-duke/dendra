"""Contracts for native-Morphology inspection and visualization."""

from __future__ import annotations

import math
import sys
import warnings

import matplotlib

# Select a renderer before importing pyplot or Dendra's plotting implementation.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

import dendra as dn
import dendra.models.morphology_visualization as morphology_viz
import dendra.models.visualization as runtime_viz
from dendra.models._morphology_scene import (
    build_compartment_topology_scene,
    build_morphology_scene,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _plots_never_show(monkeypatch):
    """A library plot must return handles instead of taking over the display."""

    def fail_show(*args, **kwargs):
        del args, kwargs
        pytest.fail("Morphology visualization must never call show().")

    monkeypatch.setattr(plt, "show", fail_show)
    monkeypatch.setattr(Figure, "show", fail_show)
    yield
    plt.close("all")


@pytest.fixture
def diagnostic_morphology() -> dn.Morphology:
    """A tapered, bent tree with interior, reversed, and gapped branches."""
    morphology = dn.Morphology()
    trunk = morphology.section(
        "trunk",
        points=(
            (0.0, 0.0, 0.0, 8.0),
            (10.0, 0.0, 0.0, 6.0),
            (10.0, 30.0, 0.0, 4.0),
        ),
        nseg=4,
        rhoa=80.0,
        cm=0.9,
        labels=("cell_body", "membrane"),
    )
    reverse = morphology.section(
        "reverse",
        # Authored distal-to-proximal: endpoint 1 is the attachment.
        points=((30.0, 10.0, 5.0, 1.0), (10.0, 10.0, 0.0, 3.0)),
        nseg=2,
        rhoa=120.0,
        cm=1.1,
        labels=("dendrite", "membrane", "shared"),
    )
    forward = morphology.section(
        "forward",
        points=((10.0, 0.0, 0.0, 4.0), (15.0, -10.0, 0.0, 2.0)),
        nseg=3,
        rhoa=100.0,
        cm=1.2,
        labels=("dendrite", "membrane", "shared"),
    )
    gapped = morphology.section(
        "gapped",
        points=((11.0, 20.0, 0.0, 2.0), (25.0, 20.0, 0.0, 1.0)),
        nseg=1,
        rhoa=150.0,
        cm=1.0,
        labels=("axon", "membrane"),
    )

    reverse.connect(trunk.at(0.5), child_end=1)
    forward.connect(trunk.at(0.25), child_end=0)
    gapped.connect(trunk.at(0.75), child_end=0)
    return morphology


def _artists_by_gid(ax: Axes) -> dict[str, matplotlib.artist.Artist]:
    artists = {}
    for artist in ax.get_children():
        gid = artist.get_gid()
        if gid is not None:
            assert gid not in artists, f"duplicate artist gid {gid!r}"
            artists[gid] = artist
    return artists


def _artist(ax: Axes, gid: str):
    try:
        return _artists_by_gid(ax)[gid]
    except KeyError as error:
        raise AssertionError(
            f"missing artist gid {gid!r}; found {sorted(_artists_by_gid(ax))!r}"
        ) from error


def _alpha(artist) -> float:
    value = artist.get_alpha()
    return 1.0 if value is None else float(value)


def _draw(fig: Figure) -> None:
    """Exercise the headless renderer without comparing backend-specific pixels."""
    fig.canvas.draw()
    assert fig.canvas.get_width_height()[0] > 0
    assert fig.canvas.get_width_height()[1] > 0


def test_scene_preserves_bent_arclength_geometry_and_discretization(
    diagnostic_morphology,
):
    scene = build_morphology_scene(diagnostic_morphology)
    trunk = scene.section("trunk")

    assert trunk.name == "trunk"
    assert trunk.labels == frozenset({"trunk", "cell_body", "membrane"})
    assert trunk.is_pt3d is True
    assert trunk.L == pytest.approx(40.0)
    assert trunk.nseg == 4
    assert trunk.rhoa == pytest.approx(80.0)
    assert trunk.cm == pytest.approx(0.9)
    assert trunk.points == (
        (0.0, 0.0, 0.0, 8.0),
        (10.0, 0.0, 0.0, 6.0),
        (10.0, 30.0, 0.0, 4.0),
    )
    assert trunk.sample_x == pytest.approx((0.0, 0.25, 1.0))
    assert trunk.boundary_x == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))
    assert trunk.center_x == pytest.approx((0.125, 0.375, 0.625, 0.875))
    np.testing.assert_allclose(
        trunk.boundary_points,
        (
            (0.0, 0.0, 0.0, 8.0),
            (10.0, 0.0, 0.0, 6.0),
            (10.0, 10.0, 0.0, 16.0 / 3.0),
            (10.0, 20.0, 0.0, 14.0 / 3.0),
            (10.0, 30.0, 0.0, 4.0),
        ),
    )
    np.testing.assert_allclose(
        trunk.center_points,
        (
            (5.0, 0.0, 0.0, 7.0),
            (10.0, 5.0, 0.0, 17.0 / 3.0),
            (10.0, 15.0, 0.0, 5.0),
            (10.0, 25.0, 0.0, 13.0 / 3.0),
        ),
    )


def test_scene_resolves_interior_attachments_by_arclength_and_child_orientation(
    diagnostic_morphology,
):
    scene = build_morphology_scene(diagnostic_morphology)
    reverse = scene.section("reverse")
    connection = next(
        item for item in scene.connections if item.child_name == "reverse"
    )

    assert reverse.parent_name == "trunk"
    assert reverse.parent_x == pytest.approx(0.5)
    assert reverse.child_end == 1
    assert reverse.away_direction == -1
    assert connection.parent_name == "trunk"
    assert connection.parent_x == pytest.approx(0.5)
    assert connection.parent_point == pytest.approx((10.0, 10.0, 0.0, 16.0 / 3.0))
    assert connection.child_name == "reverse"
    assert connection.child_end == 1
    assert connection.child_point == pytest.approx((10.0, 10.0, 0.0, 3.0))
    assert connection.gap_um == pytest.approx(0.0)
    # An exact internal compartment boundary belongs to its x-increasing side.
    assert connection.parent_host_segment == 2


def test_scene_records_spatial_connection_gap_without_rejecting_topology(
    diagnostic_morphology,
):
    scene = build_morphology_scene(diagnostic_morphology)
    connection = next(item for item in scene.connections if item.child_name == "gapped")

    assert connection.parent_point[:3] == pytest.approx((10.0, 20.0, 0.0))
    assert connection.child_point[:3] == pytest.approx((11.0, 20.0, 0.0))
    assert connection.gap_um == pytest.approx(1.0)


def test_scene_keeps_stylized_and_two_point_pt3d_declarations_distinct():
    morphology = dn.Morphology()
    morphology.section("stylized", L=12.0, diam=3.0, nseg=3)
    morphology.section(
        "pt3d",
        points=((2.0, 3.0, 4.0, 3.0), (2.0, 3.0, 16.0, 1.0)),
        nseg=3,
    )

    scene = build_morphology_scene(morphology)
    stylized = scene.section("stylized")
    pt3d = scene.section("pt3d")

    assert stylized.is_pt3d is False
    assert stylized.points == ((0.0, 0.0, 0.0, 3.0), (0.0, 0.0, 12.0, 3.0))
    assert pt3d.is_pt3d is True
    assert pt3d.points == ((2.0, 3.0, 4.0, 3.0), (2.0, 3.0, 16.0, 1.0))


def test_scene_supports_empty_and_incomplete_forests_without_compilation():
    empty = build_morphology_scene(dn.Morphology())
    assert empty.sections == ()
    assert empty.connections == ()
    assert empty.roots == ()
    assert empty.children == ()

    morphology = dn.Morphology()
    first = morphology.section("first", L=10.0, diam=2.0)
    child = morphology.section("child", L=5.0, diam=1.0)
    morphology.section("orphan", L=7.0, diam=1.5)
    child.connect(first.at(1.0), child_end=0)

    scene = build_morphology_scene(morphology)
    assert tuple(item.name for item in scene.sections) == ("first", "child", "orphan")
    assert scene.roots == ("first", "orphan")
    assert scene.children_of("first") == ("child",)
    assert scene.children_of("child") == ()
    assert scene.children_of("orphan") == ()


def test_scene_is_a_snapshot_and_fresh_build_reflects_local_update(
    diagnostic_morphology,
):
    before = build_morphology_scene(diagnostic_morphology)
    before_points = before.section("trunk").points

    trunk = diagnostic_morphology.sections[0]
    returned = trunk.at(0.5).update(diam=9.0)
    after = build_morphology_scene(diagnostic_morphology)

    assert returned is trunk
    assert before.section("trunk").points == before_points
    assert len(before_points) == 3
    assert after.section("trunk").points == (
        (0.0, 0.0, 0.0, 8.0),
        (10.0, 0.0, 0.0, 6.0),
        (10.0, 10.0, 0.0, 9.0),
        (10.0, 30.0, 0.0, 4.0),
    )


@pytest.mark.parametrize(
    ("view", "labels"),
    [
        ("x", ("y (µm)", "z (µm)")),
        ("y", ("x (µm)", "z (µm)")),
        ("z", ("x (µm)", "y (µm)")),
    ],
)
def test_plot_projects_authored_geometry_and_returns_handles(
    diagnostic_morphology, view, labels
):
    fig, ax = diagnostic_morphology.plot(view=view, color_by="section")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.figure is fig
    assert (ax.get_xlabel(), ax.get_ylabel()) == labels
    assert ax.get_aspect() == 1.0
    assert _artist(ax, "section:trunk") is not None
    assert _artist(ax, "section:reverse") is not None
    _draw(fig)


@pytest.mark.parametrize(
    "color_by", ["section", "diameter", "length", "nseg", "rhoa", "cm"]
)
def test_plot_supports_all_documented_color_encodings(diagnostic_morphology, color_by):
    fig, ax = diagnostic_morphology.plot(view="y", color_by=color_by, legend=True)

    assert _artist(ax, "section:trunk") is not None
    assert _artist(ax, "section:gapped") is not None
    if color_by == "section":
        assert ax.get_legend() is not None
    else:
        # Scalar encodings carry an explicit color scale rather than a
        # categorical Section legend.
        assert len(fig.axes) == 2
        expected_labels = {
            "diameter": "Diameter (µm)",
            "length": "Section length (µm)",
            "nseg": "Section nseg",
            "rhoa": "Axial resistivity (Ω·cm)",
            "cm": "Specific capacitance (µF/cm²)",
        }
        assert fig.axes[-1].get_ylabel() == expected_labels[color_by]
    _draw(fig)


def test_plot_highlight_matches_exact_name_or_shared_label(diagnostic_morphology):
    fig, ax = diagnostic_morphology.plot(highlight="shared")
    label_artists = _artists_by_gid(ax)

    assert _alpha(label_artists["section:reverse"]) > _alpha(
        label_artists["section:trunk"]
    )
    assert _alpha(label_artists["section:forward"]) > _alpha(
        label_artists["section:gapped"]
    )
    _draw(fig)

    fig, ax = diagnostic_morphology.plot(highlight="reverse")
    name_artists = _artists_by_gid(ax)
    assert _alpha(name_artists["section:reverse"]) > _alpha(
        name_artists["section:forward"]
    )
    _draw(fig)

    with pytest.raises(ValueError, match=r"(?i)highlight|unknown|label"):
        diagnostic_morphology.plot(highlight="rev")


def test_plot_diagnostic_overlays_have_stable_artist_ids(diagnostic_morphology):
    fig, ax = diagnostic_morphology.plot(
        view="y",
        show_points=True,
        show_connections=True,
        show_compartments=True,
        show_orientation=True,
        annotate_sections=True,
        annotate_connections=True,
    )
    gids = _artists_by_gid(ax)

    for section in ("trunk", "reverse", "forward", "gapped"):
        assert f"section:{section}" in gids
        assert f"points:{section}" in gids
        assert f"compartments:{section}" in gids
        assert f"orientation:{section}" in gids
        assert f"annotation-section:{section}" in gids
    for child in ("reverse", "forward", "gapped"):
        assert f"connection:{child}" in gids
        assert f"annotation-connection:{child}" in gids
    _draw(fig)


def test_plot_linewidths_follow_taper_and_respect_floor(diagnostic_morphology):
    fig, ax = diagnostic_morphology.plot(
        view="y",
        diameter_scale=0.5,
        min_linewidth=0.25,
    )
    trunk = _artist(ax, "section:trunk")

    widths = np.asarray(trunk.get_linewidths())
    assert len(widths) > 2
    assert np.all(np.diff(widths) <= 0.0)
    assert widths[0] > widths[-1]
    assert widths[0] == pytest.approx(3.875)
    assert widths[-1] == pytest.approx(2.125)
    assert np.all(widths >= 0.25)
    _draw(fig)


@pytest.mark.parametrize("dimensionality", ["2d", "3d"])
def test_sparse_taper_is_interpolated_in_width_color_and_point_glyphs(
    dimensionality,
):
    morphology = dn.Morphology()
    morphology.section(
        "taper",
        points=((0.0, 0.0, 0.0, 10.0), (100.0, 0.0, 0.0, 1.0)),
        nseg=3,
    )

    if dimensionality == "2d":
        fig, ax = morphology.plot(view="z", color_by="diameter")
    else:
        fig, ax = morphology.plot_3d(color_by="diameter")
    _draw(fig)
    centerline = _artist(ax, "section:taper")
    points = _artist(ax, "points:taper")
    widths = np.asarray(centerline.get_linewidths())
    colors = np.asarray(centerline.get_colors())
    point_colors = np.asarray(points.get_facecolors())

    assert 4 <= len(widths) <= 16
    assert np.all(np.diff(widths) <= 0.0)
    assert widths[0] > widths[-1]
    assert not np.allclose(colors[0], colors[-1])
    assert point_colors.shape[0] == 2
    assert not np.allclose(point_colors[0], point_colors[-1])


def test_orientation_uses_local_span_and_marks_parent_facing_child_end():
    morphology = dn.Morphology()
    parent = morphology.section(
        "parent", points=((0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0))
    )
    child = morphology.section(
        "child",
        points=(
            (20.0, 10.0, 0.0, 1.0),
            (10.0, 10.0, 0.0, 1.0),
            (10.0, 0.0, 0.0, 1.0),
        ),
    )
    child.connect(parent.at(1.0), child_end=1)

    fig, ax = morphology.plot(
        view="z", show_points=False, show_connections=False, show_orientation=True
    )
    orientation = _artist(ax, "orientation:child")
    parent_end = _artist(ax, "parent-end:child")

    # The +x arrow is on the final authored chord next to child_end=1; it does
    # not cut diagonally across the bend as a pair of global x samples would.
    path = orientation.get_path()
    vertices = np.asarray(path.vertices)[np.asarray(path.codes) != path.CLOSEPOLY]
    assert np.ptp(vertices[:3, 0]) < 1e-9
    assert np.ptp(vertices[:3, 1]) > 0.0
    np.testing.assert_allclose(parent_end.get_offsets(), ((10.0, 0.0),))
    assert "parent-end:parent" not in _artists_by_gid(ax)
    _draw(fig)


def test_plot_gap_tolerance_changes_connection_diagnostic(diagnostic_morphology):
    low_fig, low_ax = diagnostic_morphology.plot(
        show_connections=True,
        connection_tolerance_um=0.5,
    )
    high_fig, high_ax = diagnostic_morphology.plot(
        show_connections=True,
        connection_tolerance_um=1.0,
    )

    low_gap = _artist(low_ax, "connection:gapped")
    high_gap = _artist(high_ax, "connection:gapped")
    assert low_gap.get_linestyle() != high_gap.get_linestyle()
    _draw(low_fig)
    _draw(high_fig)


def test_plot_reuses_supplied_axes_and_honors_title(diagnostic_morphology):
    supplied_fig, supplied_ax = plt.subplots()
    fig, ax = diagnostic_morphology.plot(ax=supplied_ax, title="Authored cell")

    assert fig is supplied_fig
    assert ax is supplied_ax
    assert ax.get_title() == "Authored cell"
    _draw(fig)


def test_plot_rejects_invalid_projection_and_encoding(diagnostic_morphology):
    with pytest.raises(ValueError, match=r"(?i)view"):
        diagnostic_morphology.plot(view="diagonal")
    with pytest.raises(ValueError, match=r"(?i)color_by"):
        diagnostic_morphology.plot(color_by="volume")
    with pytest.raises(ValueError, match=r"(?i)tolerance"):
        diagnostic_morphology.plot(connection_tolerance_um=-1.0)


def test_plot_3d_returns_matplotlib_axes_and_diagnostic_artists(
    diagnostic_morphology,
):
    fig, ax = diagnostic_morphology.plot_3d(
        color_by="section",
        highlight="dendrite",
        show_points=True,
        show_connections=True,
        show_compartments=True,
        show_orientation=True,
        annotate_sections=True,
        annotate_connections=True,
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.name == "3d"
    gids = _artists_by_gid(ax)
    assert "section:trunk" in gids
    assert "points:trunk" in gids
    assert "compartments:trunk" in gids
    assert "orientation:reverse" in gids
    assert "connection:gapped" in gids
    _draw(fig)


def test_plot_3d_reuses_supplied_3d_axes(diagnostic_morphology):
    supplied_fig = plt.figure()
    supplied_ax = supplied_fig.add_subplot(111, projection="3d")

    fig, ax = diagnostic_morphology.plot_3d(ax=supplied_ax)

    assert fig is supplied_fig
    assert ax is supplied_ax
    _draw(fig)


def _shape_collection(ax):
    collections = [
        collection
        for collection in ax.collections
        if collection.get_gid() == "morphology-shape"
    ]
    assert len(collections) == 1
    return collections[0]


def test_plot_shape_uses_physical_diameter_and_surface_bounds():
    morphology = dn.Morphology()
    morphology.section(
        "cable",
        points=((0.0, 0.0, 0.0, 4.0), (10.0, 0.0, 0.0, 4.0)),
    )

    fig, ax = morphology.plot_shape(
        view="z",
        radial_segments=8,
        legend=False,
        show_axes=True,
    )
    collection = _shape_collection(ax)
    vertices = collection._dendra_surface_vertices

    np.testing.assert_allclose(vertices.min(axis=0), (0.0, -2.0, -2.0))
    np.testing.assert_allclose(vertices.max(axis=0), (10.0, 2.0, 2.0))
    assert ax.get_xlim()[0] < 0.0 < 10.0 < ax.get_xlim()[1]
    assert ax.get_ylim()[0] < -2.0 < 2.0 < ax.get_ylim()[1]
    assert ax.get_aspect() == 1.0
    assert (ax.get_xlabel(), ax.get_ylabel()) == (
        "x (\N{MICRO SIGN}m)",
        "y (\N{MICRO SIGN}m)",
    )
    _draw(fig)

    scaled_fig, scaled_ax = morphology.plot_shape_3d(
        diameter_scale=2.0,
        radial_segments=8,
        legend=False,
        show_axes=True,
    )
    scaled_vertices = _shape_collection(scaled_ax)._dendra_surface_vertices
    np.testing.assert_allclose(scaled_vertices[:, 0].min(), 0.0)
    np.testing.assert_allclose(scaled_vertices[:, 0].max(), 10.0)
    np.testing.assert_allclose(scaled_vertices[:, 1:].min(axis=0), (-4.0, -4.0))
    np.testing.assert_allclose(scaled_vertices[:, 1:].max(axis=0), (4.0, 4.0))
    assert scaled_ax.get_xlim()[0] < 0.0 < 10.0 < scaled_ax.get_xlim()[1]
    assert scaled_ax.get_ylim()[0] < -4.0 < 4.0 < scaled_ax.get_ylim()[1]
    assert scaled_ax.get_zlim()[0] < -4.0 < 4.0 < scaled_ax.get_zlim()[1]
    _draw(scaled_fig)


def test_shape_tube_rings_preserve_taper_and_bend_frames():
    points = np.asarray(
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 12.0, 4.0)),
        dtype=float,
    )
    diameters = np.asarray((4.0, 2.0, 6.0), dtype=float)

    rings, tangents = morphology_viz._tube_rings(
        points,
        diameters,
        diameter_scale=1.0,
        radial_segments=12,
    )

    assert rings.shape == (3, 12, 3)
    assert tangents.shape == (3, 3)
    assert np.isfinite(rings).all()
    assert np.isfinite(tangents).all()
    np.testing.assert_allclose(rings.mean(axis=1), points, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    for index, diameter in enumerate(diameters):
        offsets = rings[index] - points[index]
        np.testing.assert_allclose(
            np.linalg.norm(offsets, axis=1),
            0.5 * diameter,
        )
        np.testing.assert_allclose(offsets @ tangents[index], 0.0, atol=1e-14)


def test_shape_tube_frames_remain_finite_at_an_exact_reversal():
    points = np.asarray(
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        dtype=float,
    )
    rings, tangents = morphology_viz._tube_rings(
        points,
        np.asarray((2.0, 1.0, 2.0)),
        diameter_scale=1.0,
        radial_segments=8,
    )

    assert np.isfinite(rings).all()
    assert np.isfinite(tangents).all()
    np.testing.assert_allclose(rings.mean(axis=1), points, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    for point, ring, tangent in zip(points, rings, tangents):
        np.testing.assert_allclose((ring - point) @ tangent, 0.0, atol=1e-14)


@pytest.mark.parametrize("dimensionality", ("2d", "3d"))
@pytest.mark.parametrize(
    ("points", "step_x", "expected_face_count"),
    [
        (
            (
                (0.0, 0.0, 0.0, 2.0),
                (0.0, 0.0, 0.0, 4.0),
                (10.0, 0.0, 0.0, 4.0),
            ),
            0.0,
            26,
        ),
        (
            (
                (0.0, 0.0, 0.0, 2.0),
                (5.0, 0.0, 0.0, 2.0),
                (5.0, 0.0, 0.0, 4.0),
                (10.0, 0.0, 0.0, 4.0),
            ),
            5.0,
            38,
        ),
        (
            (
                (0.0, 0.0, 0.0, 2.0),
                (10.0, 0.0, 0.0, 2.0),
                (10.0, 0.0, 0.0, 4.0),
            ),
            10.0,
            26,
        ),
    ],
    ids=("start-step", "interior-step", "end-step"),
)
def test_shape_views_render_finite_annular_diameter_steps(
    dimensionality, points, step_x, expected_face_count
):
    morphology = dn.Morphology()
    morphology.section("step", points=points)

    if dimensionality == "2d":
        # Look along the cable so the zero-length degenerate frustum is seen
        # face-on as an annular shoulder rather than an edge-on radial line.
        fig, ax = morphology.plot_shape(
            view="x", radial_segments=12, legend=False, show_axes=True
        )
    else:
        fig, ax = morphology.plot_shape_3d(
            radial_segments=12, legend=False, show_axes=True
        )
    collection = _shape_collection(ax)
    vertices = collection._dendra_surface_vertices

    assert np.isfinite(vertices).all()
    # Every authored span contributes twelve quad faces, including the
    # coincident-coordinate diameter step; the two end caps complete the tube.
    assert len(collection._dendra_face_sections) == expected_face_count
    step_vertices = vertices[vertices[:, 0] == step_x]
    radii = np.linalg.norm(step_vertices[:, 1:], axis=1)
    np.testing.assert_allclose(np.unique(np.round(radii, decimals=12)), (1.0, 2.0))
    _draw(fig)


def test_diagnostic_views_preserve_an_authored_vertical_diameter_step():
    morphology = dn.Morphology()
    morphology.section(
        "step",
        points=(
            (0.0, 0.0, 0.0, 2.0),
            (5.0, 0.0, 0.0, 2.0),
            (5.0, 0.0, 0.0, 4.0),
            (10.0, 0.0, 0.0, 4.0),
        ),
        nseg=3,
    )

    spatial_fig, spatial_ax = morphology.plot(
        view="z", show_points=True, show_orientation=True, legend=False
    )
    centerline = _artist(spatial_ax, "section:step")
    segments = np.asarray(centerline.get_segments())
    assert np.isfinite(segments).all()
    assert np.any(np.all(segments[:, 0] == segments[:, 1], axis=1))
    _draw(spatial_fig)

    profile_fig, profile_ax = morphology.plot_diameter_profile(
        sections="step",
        x_axis="normalized",
        show_points=True,
        show_compartments=False,
        show_connections=False,
        legend=False,
    )
    profile = _artist(profile_ax, "diameter-profile:step")
    np.testing.assert_allclose(profile.get_xdata(), (0.0, 0.5, 0.5, 1.0))
    np.testing.assert_allclose(profile.get_ydata(), (2.0, 2.0, 4.0, 4.0))
    _draw(profile_fig)


def test_plot_shape_projection_keeps_a_cable_viewed_end_on_visible():
    morphology = dn.Morphology()
    morphology.section(
        "end_on",
        points=((0.0, 0.0, 0.0, 4.0), (10.0, 0.0, 0.0, 4.0)),
    )

    fig, ax = morphology.plot_shape(
        view="x", radial_segments=16, legend=False, show_axes=True
    )
    collection = _shape_collection(ax)
    projected = collection._dendra_surface_vertices[:, (1, 2)]

    np.testing.assert_allclose(projected.min(axis=0), (-2.0, -2.0))
    np.testing.assert_allclose(projected.max(axis=0), (2.0, 2.0))
    assert any(
        np.ptp(path.vertices[:, 0]) > 0.0 and np.ptp(path.vertices[:, 1]) > 0.0
        for path in collection.get_paths()
    )
    _draw(fig)


@pytest.mark.parametrize("dimensionality", ("2d", "3d"))
def test_shape_renderer_uses_one_collection_without_diagnostic_artists(
    dimensionality,
):
    morphology = dn.Morphology()
    for index in range(12):
        morphology.section(
            f"branch_{index}",
            points=(
                (0.0, float(index), 0.0, 1.0),
                (10.0, float(index), float(index % 3), 0.5),
            ),
            nseg=17,
        )

    if dimensionality == "2d":
        fig, ax = morphology.plot_shape(view="z", legend=False)
    else:
        fig, ax = morphology.plot_shape_3d(legend=False)
    collection = _shape_collection(ax)

    assert len(ax.collections) == 1
    assert set(collection._dendra_face_sections) == {
        f"branch_{index}" for index in range(12)
    }
    assert not ax.lines
    assert not ax.texts
    assert ax.get_title() == ""
    assert ax.get_legend() is None
    _draw(fig)


@pytest.mark.parametrize("dimensionality", ("2d", "3d"))
def test_shape_colors_use_canonical_region_families_and_compact_legend(
    dimensionality,
):
    morphology = dn.Morphology()
    morphology.section(
        "soma[0]",
        points=((0.0, 0.0, 0.0, 8.0), (10.0, 0.0, 0.0, 8.0)),
        labels="membrane",
    )
    morphology.section(
        "body",
        points=((0.0, 2.0, 0.0, 8.0), (10.0, 2.0, 0.0, 8.0)),
        labels="cell_body",
    )
    morphology.section(
        "axon[0]",
        points=((0.0, 5.0, 0.0, 1.0), (10.0, 5.0, 0.0, 1.0)),
        labels="axon",
    )
    morphology.section(
        "initial_segment",
        points=((0.0, 7.0, 0.0, 1.0), (10.0, 7.0, 0.0, 1.0)),
        labels="axon",
    )
    morphology.section(
        "dend[0]",
        points=((0.0, 10.0, 0.0, 2.0), (10.0, 10.0, 0.0, 2.0)),
        labels="dendrite",
    )
    morphology.section(
        "dend[1]",
        points=((0.0, 15.0, 0.0, 2.0), (10.0, 15.0, 0.0, 2.0)),
        labels="dendrite",
    )
    morphology.section(
        "apic[0]",
        points=((0.0, 20.0, 0.0, 2.0), (10.0, 20.0, 0.0, 2.0)),
        # A specific name-derived identity must outrank this broader label.
        labels="dendrite",
    )
    morphology.section(
        "left",
        points=((0.0, 25.0, 0.0, 2.0), (10.0, 25.0, 0.0, 2.0)),
        labels=("dendrite", "membrane"),
    )
    morphology.section(
        "tuft",
        points=((0.0, 30.0, 0.0, 2.0), (10.0, 30.0, 0.0, 2.0)),
        labels="apical_dendrite",
    )
    morphology.section(
        "basal_dendrite_0",
        points=((0.0, 32.0, 0.0, 2.0), (10.0, 32.0, 0.0, 2.0)),
    )
    morphology.section(
        "apical_dendrite_0",
        points=((0.0, 34.0, 0.0, 2.0), (10.0, 34.0, 0.0, 2.0)),
    )
    morphology.section(
        "branch[0]",
        points=((0.0, 35.0, 0.0, 2.0), (10.0, 35.0, 0.0, 2.0)),
    )
    morphology.section(
        "branch[1]",
        points=((0.0, 40.0, 0.0, 2.0), (10.0, 40.0, 0.0, 2.0)),
    )
    morphology.section(
        "custom_branch",
        points=((0.0, 45.0, 0.0, 2.0), (10.0, 45.0, 0.0, 2.0)),
    )
    morphology.section(
        "ambiguous",
        points=((0.0, 50.0, 0.0, 2.0), (10.0, 50.0, 0.0, 2.0)),
        labels=("axon", "dendrite"),
    )

    if dimensionality == "2d":
        fig, ax = morphology.plot_shape(view="z", legend=True)
    else:
        fig, ax = morphology.plot_shape_3d(legend=True)
    collection = _shape_collection(ax)
    colors = dict(collection._dendra_section_colors)
    assert runtime_viz.palette_hsv(5)[1:] == [
        "#c8e650",
        "#e65050",
        "#c850e6",
        "#508ce6",
    ]
    assert collection._dendra_section_color_groups == {
        "soma[0]": "soma",
        "body": "soma",
        "axon[0]": "axon",
        "initial_segment": "axon",
        "dend[0]": "dend",
        "dend[1]": "dend",
        "apic[0]": "apic",
        "left": "dend",
        "tuft": "apic",
        "basal_dendrite_0": "dend",
        "apical_dendrite_0": "apic",
        "branch[0]": "branch",
        "branch[1]": "branch",
        "custom_branch": "custom_branch",
        "ambiguous": "ambiguous",
    }

    assert colors["soma[0]"] == pytest.approx(matplotlib.colors.to_rgba("#c8e650"))
    assert colors["body"] == pytest.approx(colors["soma[0]"])
    assert colors["axon[0]"] == pytest.approx(matplotlib.colors.to_rgba("#e65050"))
    assert colors["initial_segment"] == pytest.approx(colors["axon[0]"])
    assert colors["dend[0]"] == pytest.approx(matplotlib.colors.to_rgba("#c850e6"))
    assert colors["apic[0]"] == pytest.approx(matplotlib.colors.to_rgba("#508ce6"))
    assert colors["dend[0]"] == pytest.approx(colors["dend[1]"])
    assert colors["dend[0]"] == pytest.approx(colors["left"])
    assert colors["dend[0]"] == pytest.approx(colors["basal_dendrite_0"])
    assert colors["apic[0]"] == pytest.approx(colors["tuft"])
    assert colors["apic[0]"] == pytest.approx(colors["apical_dendrite_0"])
    assert colors["branch[0]"] == pytest.approx(colors["branch[1]"])
    assert colors["branch[0]"] != pytest.approx(colors["custom_branch"])
    assert colors["ambiguous"] != pytest.approx(colors["axon[0]"])
    assert colors["ambiguous"] != pytest.approx(colors["dend[0]"])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "soma",
        "axon",
        "dend",
        "apic",
        "branch",
        "custom_branch",
        "ambiguous",
    ]
    assert ax.get_legend().get_title().get_text() == "Regions"
    _draw(fig)


def test_shape_generic_fallback_color_is_deterministic_and_append_stable():
    morphology = dn.Morphology()
    morphology.section("custom_branch", L=10.0, diam=2.0)

    before_fig, before_ax = morphology.plot_shape(view="z", legend=False)
    before = _shape_collection(before_ax)._dendra_section_colors["custom_branch"]

    morphology.section("soma", L=12.0, diam=8.0)
    morphology.section("another_region", L=8.0, diam=1.0)
    after_fig, after_ax = morphology.plot_shape(view="z", legend=False)
    after = _shape_collection(after_ax)._dendra_section_colors["custom_branch"]

    reordered = dn.Morphology()
    reordered.section("another_region", L=8.0, diam=1.0)
    reordered.section("custom_branch", L=10.0, diam=2.0)
    reordered_fig, reordered_ax = reordered.plot_shape_3d(legend=False)
    reordered_color = _shape_collection(reordered_ax)._dendra_section_colors[
        "custom_branch"
    ]

    assert after == pytest.approx(before)
    assert reordered_color == pytest.approx(before)
    _draw(before_fig)
    _draw(after_fig)
    _draw(reordered_fig)


def test_shape_gap_warning_does_not_add_a_physical_bridge():
    morphology = dn.Morphology()
    parent = morphology.section(
        "parent", points=((0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0))
    )
    child = morphology.section(
        "child", points=((20.0, 0.0, 0.0, 1.0), (30.0, 0.0, 0.0, 1.0))
    )
    child.connect(parent.at(1.0), child_end=0)

    with pytest.warns(RuntimeWarning, match=r"spatially incoherent|does not draw"):
        fig, ax = morphology.plot_shape(view="z", legend=False)
    collection = _shape_collection(ax)

    assert set(collection._dendra_face_sections) == {"parent", "child"}
    for path in collection.get_paths():
        x = path.vertices[:, 0]
        assert not (float(x.min()) <= 10.0 and float(x.max()) >= 20.0)
    assert {artist.get_gid() for artist in ax.collections} == {"morphology-shape"}
    _draw(fig)

    with warnings.catch_warnings(record=True) as caught:
        accepted_fig, _ = morphology.plot_shape(
            view="z", connection_tolerance_um=10.0, legend=False
        )
    assert not caught
    _draw(accepted_fig)


def test_plot_shape_auto_legend_and_axes_contract():
    small = dn.Morphology()
    for index in range(16):
        small.section(
            f"region_{chr(ord('a') + index)}",
            points=((0.0, float(index), 0.0, 1.0), (1.0, float(index), 0.0, 1.0)),
        )
    small_fig, small_ax = small.plot_shape(view="z", show_axes=False)
    assert small_ax.get_legend() is not None
    assert small_ax.axison is False
    _draw(small_fig)

    large = dn.Morphology()
    for index in range(17):
        large.section(
            f"region_{chr(ord('a') + index)}",
            points=((0.0, float(index), 0.0, 1.0), (1.0, float(index), 0.0, 1.0)),
        )
    large_fig, large_ax = large.plot_shape(view="z", show_axes=True)
    assert large_ax.get_legend() is None
    assert large_ax.axison is True
    assert (large_ax.get_xlabel(), large_ax.get_ylabel()) == (
        "x (\N{MICRO SIGN}m)",
        "y (\N{MICRO SIGN}m)",
    )
    _draw(large_fig)

    forced_fig, forced_ax = large.plot_shape_3d(legend=True, show_axes=True)
    assert len(forced_ax.get_legend().get_texts()) == 17
    assert forced_ax._axis3don is True
    assert (forced_ax.get_xlabel(), forced_ax.get_ylabel(), forced_ax.get_zlabel()) == (
        "x (\N{MICRO SIGN}m)",
        "y (\N{MICRO SIGN}m)",
        "z (\N{MICRO SIGN}m)",
    )
    _draw(forced_fig)

    # The automatic threshold applies to semantic legend groups, not raw
    # Section count. Large imported morphologies should retain a useful key.
    indexed = dn.Morphology()
    for index in range(20):
        indexed.section(
            f"dend[{index}]",
            points=((0.0, float(index), 0.0, 1.0), (1.0, float(index), 0.0, 1.0)),
        )
    indexed_fig, indexed_ax = indexed.plot_shape(view="z")
    assert [text.get_text() for text in indexed_ax.get_legend().get_texts()] == ["dend"]
    _draw(indexed_fig)


def test_shape_views_render_an_empty_morphology_and_an_incomplete_forest():
    empty = dn.Morphology()
    empty_fig, empty_ax = empty.plot_shape()
    empty_3d_fig, empty_3d_ax = empty.plot_shape_3d()
    assert [text.get_text() for text in empty_ax.texts] == ["Empty Morphology"]
    assert [text.get_text() for text in empty_3d_ax.texts] == ["Empty Morphology"]
    assert empty_ax.get_title() == ""
    assert empty_3d_ax.get_title() == ""
    _draw(empty_fig)
    _draw(empty_3d_fig)

    forest = dn.Morphology()
    forest.section("first", L=10.0, diam=2.0, nseg=3)
    forest.section(
        "second",
        points=((20.0, 0.0, 0.0, 1.0), (30.0, 5.0, 0.0, 0.5)),
        nseg=9,
    )
    with pytest.raises(ValueError, match=r"exactly one root"):
        forest.compile()

    forest_fig, forest_ax = forest.plot_shape(view="z", legend=False)
    forest_3d_fig, forest_3d_ax = forest.plot_shape_3d(legend=False)
    assert set(_shape_collection(forest_ax)._dendra_face_sections) == {
        "first",
        "second",
    }
    assert set(_shape_collection(forest_3d_ax)._dendra_face_sections) == {
        "first",
        "second",
    }
    _draw(forest_fig)
    _draw(forest_3d_fig)


@pytest.mark.parametrize("dimensionality", ("2d", "3d"))
@pytest.mark.parametrize(
    ("points", "diameter_scale", "expected_radius"),
    [
        (
            ((0.0, 0.0, 0.0, 1e308), (10.0, 0.0, 0.0, 1e308)),
            4.0,
            2.0,
        ),
        (
            ((0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)),
            float(np.finfo(np.float64).max),
            1.0,
        ),
        (
            ((0.0, 0.0, 0.0, 1e308), (1e308, 0.0, 0.0, 1e308)),
            float(np.finfo(np.float64).max),
            1.0,
        ),
        (
            ((0.0, 0.0, 0.0, 1e308), (10.0, 0.0, 0.0, 1e308)),
            math.nextafter(0.0, 1.0),
            0.5 * (1e308 * math.nextafter(0.0, 1.0)),
        ),
    ],
    ids=(
        "maximum-diameter",
        "maximum-scale",
        "overflowing-scale-product",
        "minimum-subnormal-scale",
    ),
)
def test_shape_extreme_finite_diameter_and_scale_remain_renderable(
    dimensionality, points, diameter_scale, expected_radius
):
    morphology = dn.Morphology()
    morphology.section("extreme", points=points)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if dimensionality == "2d":
            fig, ax = morphology.plot_shape(
                view="z",
                diameter_scale=diameter_scale,
                radial_segments=8,
                legend=False,
                show_axes=True,
            )
        else:
            fig, ax = morphology.plot_shape_3d(
                diameter_scale=diameter_scale,
                radial_segments=8,
                legend=False,
                show_axes=True,
            )
        vertices = _shape_collection(ax)._dendra_surface_vertices
        assert np.isfinite(vertices).all()
        limits = [ax.get_xlim(), ax.get_ylim()]
        if dimensionality == "3d":
            limits.append(ax.get_zlim())
        assert np.isfinite(np.asarray(limits)).all()
        np.testing.assert_allclose(
            vertices[:, 1].min(), -expected_radius, rtol=1e-14, atol=0.0
        )
        np.testing.assert_allclose(
            vertices[:, 1].max(), expected_radius, rtol=1e-14, atol=0.0
        )
        np.testing.assert_allclose(
            vertices[:, 2].min(), -expected_radius, rtol=1e-14, atol=0.0
        )
        np.testing.assert_allclose(
            vertices[:, 2].max(), expected_radius, rtol=1e-14, atol=0.0
        )
        _draw(fig)


@pytest.mark.parametrize("dimensionality", ("2d", "3d"))
def test_shape_extreme_coordinate_and_diameter_use_one_uniform_transform(
    dimensionality,
):
    morphology = dn.Morphology()
    morphology.section(
        "remote",
        points=((1e308, 0.0, 0.0, 1e308), (1e308, 10.0, 0.0, 1e308)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if dimensionality == "2d":
            fig, ax = morphology.plot_shape(
                view="z", diameter_scale=4.0, legend=False, show_axes=True
            )
        else:
            fig, ax = morphology.plot_shape_3d(
                diameter_scale=4.0, legend=False, show_axes=True
            )
        vertices = _shape_collection(ax)._dendra_surface_vertices
        assert np.isfinite(vertices).all()
        # The common x origin is removed and every axis and radius is divided
        # by the same 1e308 scale. No anisotropic normalization is permitted.
        np.testing.assert_allclose(vertices[:, 0].min(), -2.0)
        np.testing.assert_allclose(vertices[:, 0].max(), 2.0)
        np.testing.assert_allclose(vertices[:, 2].min(), -2.0)
        np.testing.assert_allclose(vertices[:, 2].max(), 2.0)
        assert 0.0 < vertices[:, 1].max() <= 2e-307
        _draw(fig)


@pytest.mark.parametrize("dimensionality", ("2d", "3d"))
def test_shape_show_axes_reenables_axes_when_reusing_handles(dimensionality):
    morphology = dn.Morphology()
    morphology.section("cable", L=10.0, diam=2.0)

    if dimensionality == "2d":
        fig, ax = morphology.plot_shape(legend=False, show_axes=False)
        assert ax.axison is False
        returned_fig, returned_ax = morphology.plot_shape(
            legend=False, show_axes=True, ax=ax
        )
        assert returned_ax.axison is True
    else:
        fig, ax = morphology.plot_shape_3d(legend=False, show_axes=False)
        assert ax._axis3don is False
        returned_fig, returned_ax = morphology.plot_shape_3d(
            legend=False, show_axes=True, ax=ax
        )
        assert returned_ax._axis3don is True
    assert returned_fig is fig
    assert returned_ax is ax
    _draw(fig)


def test_compartment_topology_scene_matches_canonical_compile(diagnostic_morphology):
    graph = diagnostic_morphology.compile()
    scene = build_compartment_topology_scene(diagnostic_morphology)

    assert scene.roots == (graph.topology.root,)
    assert scene.edges == graph.topology.edges
    assert len(scene.nodes) == graph.n_compartments
    expected_children = [[] for _ in range(graph.n_compartments)]
    expected_degree = [0 for _ in range(graph.n_compartments)]
    for parent, child in graph.topology.edges:
        expected_children[parent].append(child)
        expected_degree[parent] += 1
        expected_degree[child] += 1
    assert scene.children == tuple(
        (node_id, tuple(children)) for node_id, children in enumerate(expected_children)
    )
    for node in scene.nodes:
        node_id = node.node_id
        parent = graph.topology.parent_index[node_id]
        assert node.component_index == 0
        assert node.local_node_id == node_id
        assert node.parent_id == (None if parent == -1 else parent)
        assert node.degree == expected_degree[node_id]
        assert node.name == graph.metadata.name[node_id]
        assert node.kind == graph.metadata.kind[node_id]
        assert node.section_name == graph.metadata.section_name[node_id]
        assert node.segment_index == graph.metadata.segment_index[node_id]
        assert node.section_x == graph.metadata.section_x[node_id]
        assert node.labels == graph.metadata.labels[node_id]
        assert node.length_um == graph.geometry.length_um[node_id]
        assert node.diameter_um == graph.geometry.diameter_um[node_id]
        assert node.area_um2 == graph.geometry.area_um2[node_id]
        assert node.volume_um3 == graph.geometry.volume_um3[node_id]
        assert node.x_um == graph.geometry.x_um[node_id]
        assert node.y_um == graph.geometry.y_um[node_id]
        assert node.z_um == graph.geometry.z_um[node_id]
        assert node.rhoa_ohm_cm == graph.geometry.rhoa_ohm_cm[node_id]
        assert node.cm_uF_cm2 == graph.geometry.cm_uF_cm2[node_id]
        assert node.edge_length_um == graph.geometry.edge_length_um[node_id]
        assert node.edge_resistance_ohm == graph.geometry.edge_resistance_ohm[node_id]
        assert node.edge_diff_geom_um == graph.geometry.edge_diff_geom_um[node_id]


def test_plot_topology_is_text_free_compartment_graph_with_section_legend(
    diagnostic_morphology,
):
    graph = diagnostic_morphology.compile()
    fig, ax = diagnostic_morphology.plot_topology()
    gids = _artists_by_gid(ax)
    scene = ax._dendra_topology_scene

    assert scene.edges == graph.topology.edges
    assert len(scene.nodes) == graph.n_compartments
    assert "topology:edges" in gids
    assert "topology:compartments" in gids
    assert not ax.texts
    assert ax.get_legend() is not None
    assert {text.get_text() for text in ax.get_legend().get_texts()}.issuperset(
        {"trunk", "reverse", "forward", "gapped"}
    )
    assert len(_artist(ax, "topology:edges").get_segments()) == len(scene.edges)
    _draw(fig)


def test_plot_topology_distinguishes_material_forks_and_algebraic_junctions():
    interior = dn.Morphology()
    trunk = interior.section("trunk", L=30.0, diam=3.0, nseg=3)
    branch = interior.section("branch", L=10.0, diam=1.0)
    branch.connect(trunk.at(0.5), child_end=0)

    fig, ax = interior.plot_topology()
    scene = ax._dendra_topology_scene
    material_forks = [
        node
        for node in scene.nodes
        if node.kind == "compartment" and node.is_branchpoint
    ]
    assert len(material_forks) == 1
    assert material_forks[0].section_name == "trunk"
    assert "topology:branchpoints" in _artists_by_gid(ax)
    assert "topology:junctions" not in _artists_by_gid(ax)
    _draw(fig)

    endpoint = dn.Morphology()
    soma = endpoint.section("soma", L=10.0, diam=8.0)
    left = endpoint.section("left", L=8.0, diam=1.0)
    right = endpoint.section("right", L=9.0, diam=1.0)
    left.connect(soma.at(1.0), child_end=0)
    right.connect(soma.at(1.0), child_end=0)

    fig, ax = endpoint.plot_topology()
    scene = ax._dendra_topology_scene
    junctions = [node for node in scene.nodes if node.kind == "junction"]
    assert len(junctions) == 1
    assert junctions[0].is_branchpoint
    assert "topology:junctions" in _artists_by_gid(ax)
    _draw(fig)


def test_plot_topology_scales_and_clips_material_node_and_fork_areas():
    morphology = dn.Morphology()
    for index, diameter in enumerate((0.5, 10.0, 100.0)):
        root = morphology.section(
            f"root_{index}",
            L=12.0,
            diam=diameter,
            nseg=3,
        )
        branch = morphology.section(
            f"branch_{index}",
            L=5.0,
            diam=1.0,
        )
        branch.connect(root.at(0.5), child_end=0)

    fig, ax = morphology.plot_topology(legend=False)
    scene = ax._dendra_topology_scene
    ordinary = _artist(ax, "topology:compartments")
    forks = _artist(ax, "topology:branchpoints")

    ordinary_sizes = {
        node_id: float(size)
        for node_id, size in zip(
            ordinary._dendra_topology_node_ids,
            ordinary.get_sizes(),
        )
    }
    fork_sizes = {
        node_id: float(size)
        for node_id, size in zip(
            forks._dendra_topology_node_ids,
            forks.get_sizes(),
        )
    }

    assert ordinary_sizes == pytest.approx(
        {
            node_id: float(np.clip(8.0 * scene.node(node_id).diameter_um, 12.0, 240.0))
            for node_id in ordinary._dendra_topology_node_ids
        }
    )
    assert fork_sizes == pytest.approx(
        {
            node_id: float(
                np.clip(
                    1.35
                    * np.clip(
                        8.0 * scene.node(node_id).diameter_um,
                        12.0,
                        240.0,
                    ),
                    24.0,
                    300.0,
                )
            )
            for node_id in forks._dendra_topology_node_ids
        }
    )
    assert sorted(fork_sizes.values()) == pytest.approx((24.0, 108.0, 300.0))
    _draw(fig)


def test_plot_topology_scales_edges_from_material_endpoints_only():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=100.0)
    thin = morphology.section("thin", L=5.0, diam=1.0)
    medium = morphology.section("medium", L=5.0, diam=10.0)
    thin.connect(root.at(1.0), child_end=0)
    medium.connect(root.at(1.0), child_end=0)

    fig, ax = morphology.plot_topology(legend=False)
    scene = ax._dendra_topology_scene
    junction = next(node for node in scene.nodes if node.kind == "junction")
    assert junction.diameter_um == pytest.approx(100.0)

    edges = _artist(ax, "topology:edges")
    widths = {
        edge: float(width)
        for edge, width in zip(
            edges._dendra_topology_edges,
            edges.get_linewidths(),
        )
    }

    expected = {}
    for edge in scene.edges:
        material_diameters = [
            scene.node(node_id).diameter_um
            for node_id in edge
            if scene.node(node_id).kind == "compartment"
        ]
        assert material_diameters
        expected[edge] = float(np.clip(0.12 * np.mean(material_diameters), 0.35, 3.0))

    assert widths == pytest.approx(expected)
    thin_edge = next(
        edge
        for edge in scene.edges
        if {scene.node(node_id).section_name for node_id in edge} == {None, "thin"}
    )
    medium_edge = next(
        edge
        for edge in scene.edges
        if {scene.node(node_id).section_name for node_id in edge} == {None, "medium"}
    )
    assert widths[thin_edge] == pytest.approx(0.35)
    assert widths[medium_edge] == pytest.approx(1.2)
    _draw(fig)


def test_plot_topology_junction_size_is_fixed_and_independent_of_diameter():
    morphology = dn.Morphology()
    for index, diameter in enumerate((0.5, 100.0)):
        root = morphology.section(f"root_{index}", L=10.0, diam=diameter)
        first = morphology.section(f"first_{index}", L=5.0, diam=1.0)
        second = morphology.section(f"second_{index}", L=5.0, diam=2.0)
        first.connect(root.at(1.0), child_end=0)
        second.connect(root.at(1.0), child_end=0)

    fig, ax = morphology.plot_topology(legend=False, junction_size=47.0)
    junctions = _artist(ax, "topology:junctions")

    assert len(junctions._dendra_topology_node_ids) == 2
    # Matplotlib stores one scalar size and broadcasts it across the collection.
    assert junctions.get_sizes() == pytest.approx((47.0,))
    _draw(fig)


def test_plot_topology_legacy_fixed_size_overrides_preserve_historical_scope():
    morphology = dn.Morphology()
    interior_root = morphology.section("interior_root", L=12.0, diam=20.0, nseg=3)
    interior_branch = morphology.section("interior_branch", L=5.0, diam=1.0)
    interior_branch.connect(interior_root.at(0.5), child_end=0)

    endpoint_root = morphology.section("endpoint_root", L=10.0, diam=50.0)
    left = morphology.section("left", L=5.0, diam=1.0)
    right = morphology.section("right", L=5.0, diam=2.0)
    left.connect(endpoint_root.at(1.0), child_end=0)
    right.connect(endpoint_root.at(1.0), child_end=0)

    fig, ax = morphology.plot_topology(
        node_size=19.0,
        branchpoint_size=31.0,
        junction_size=47.0,
        edge_width=2.25,
        legend=False,
    )

    assert np.all(_artist(ax, "topology:compartments").get_sizes() == 19.0)
    assert np.all(_artist(ax, "topology:branchpoints").get_sizes() == 31.0)
    # The legacy branchpoint override historically covered both branchpoint
    # glyph classes and therefore takes precedence over junction_size.
    assert np.all(_artist(ax, "topology:junctions").get_sizes() == 31.0)
    assert np.all(np.asarray(_artist(ax, "topology:edges").get_linewidths()) == 2.25)
    _draw(fig)


def test_plot_topology_highlight_changes_opacity_not_geometry(
    diagnostic_morphology,
):
    plain_fig, plain_ax = diagnostic_morphology.plot_topology(legend=False)
    selected_fig, selected_ax = diagnostic_morphology.plot_topology(
        highlight="shared",
        legend=False,
    )

    assert selected_ax._dendra_topology_scene == plain_ax._dendra_topology_scene
    np.testing.assert_allclose(
        selected_ax._dendra_topology_positions,
        plain_ax._dendra_topology_positions,
    )
    for gid in ("topology:compartments", "topology:branchpoints"):
        plain = _artist(plain_ax, gid)
        selected = _artist(selected_ax, gid)
        assert selected._dendra_topology_node_ids == plain._dendra_topology_node_ids
        np.testing.assert_allclose(selected.get_sizes(), plain.get_sizes())

    plain_edges = _artist(plain_ax, "topology:edges")
    selected_edges = _artist(selected_ax, "topology:edges")
    assert selected_edges._dendra_topology_edges == plain_edges._dendra_topology_edges
    np.testing.assert_allclose(
        selected_edges.get_linewidths(),
        plain_edges.get_linewidths(),
    )
    np.testing.assert_allclose(plain_edges.get_colors()[:, :3], 0.0)
    np.testing.assert_allclose(plain_edges.get_colors()[:, 3], 0.85)
    assert sorted(set(selected_edges.get_colors()[:, 3])) == pytest.approx([0.24, 0.85])
    assert not np.array_equal(
        selected_edges.get_colors()[:, 3],
        plain_edges.get_colors()[:, 3],
    )
    _draw(plain_fig)
    _draw(selected_fig)


def test_plot_topology_highlight_uses_exact_section_names_or_shared_labels(
    diagnostic_morphology,
):
    fig, ax = diagnostic_morphology.plot_topology(highlight="shared")
    collection = _artist(ax, "topology:compartments")
    scene = ax._dendra_topology_scene
    colors = {
        node_id: color
        for node_id, color in zip(
            collection._dendra_topology_node_ids,
            collection.get_facecolors(),
        )
    }
    selected = [
        colors[node.node_id][3]
        for node in scene.nodes
        if node.section_name in {"reverse", "forward"} and node.node_id in colors
    ]
    muted = [
        colors[node.node_id][3]
        for node in scene.nodes
        if node.section_name in {"trunk", "gapped"} and node.node_id in colors
    ]

    assert min(selected) == pytest.approx(1.0)
    assert min(muted) >= 0.28
    assert max(muted) <= 0.35
    assert min(selected) > max(muted)
    edge_alphas = _artist(ax, "topology:edges").get_colors()[:, 3]
    assert edge_alphas.min() >= 0.20
    _draw(fig)
    with pytest.raises(ValueError, match=r"(?i)unknown|highlight|label"):
        diagnostic_morphology.plot_topology(highlight="share")


def test_section_palette_is_saturated_and_stable_when_sections_are_appended():
    morphology = dn.Morphology()
    soma = morphology.section("soma", L=10.0, diam=8.0)
    axon = morphology.section("axon", L=20.0, diam=1.0)
    axon.connect(soma.at(1.0), child_end=0)

    def colors_by_section(ax):
        collection = _artist(ax, "topology:compartments")
        scene = ax._dendra_topology_scene
        return {
            scene.node(node_id).section_name: tuple(color)
            for node_id, color in zip(
                collection._dendra_topology_node_ids,
                collection.get_facecolors(),
            )
        }

    fig, ax = morphology.plot_topology(legend=False)
    initial = colors_by_section(ax)
    assert initial["soma"] == pytest.approx(
        (*matplotlib.colors.hsv_to_rgb((0.0, 0.65, 0.90)), 1.0)
    )
    assert initial["axon"] == pytest.approx(
        (*matplotlib.colors.hsv_to_rgb((2.0 / 3.0, 0.65, 0.90)), 1.0)
    )
    _draw(fig)

    morphology.section("dendrite", L=15.0, diam=2.0)
    fig, ax = morphology.plot_topology(legend=False)
    extended = colors_by_section(ax)
    assert extended["soma"] == pytest.approx(initial["soma"])
    assert extended["axon"] == pytest.approx(initial["axon"])
    assert extended["dendrite"] == pytest.approx(
        (*matplotlib.colors.hsv_to_rgb((1.0 / 3.0, 0.65, 0.90)), 1.0)
    )
    _draw(fig)


def test_plot_topology_warns_when_hover_uses_static_inline_backend(
    diagnostic_morphology,
    monkeypatch,
):
    monkeypatch.setattr(
        matplotlib,
        "get_backend",
        lambda: "module://matplotlib_inline.backend_inline",
    )

    with pytest.warns(
        RuntimeWarning,
        match=r"dendra\[jupyter\]|ipympl",
    ) as caught:
        fig, ax = diagnostic_morphology.plot_topology(interactive=True)

    message = str(caught[0].message)
    assert "separate environments" in message
    assert "entire Jupyter server" in message
    assert "not only the kernel" in message
    assert ax._dendra_topology_hover is not None
    _draw(fig)


def test_plot_topology_interactive_hover_reports_material_metadata(
    diagnostic_morphology,
):
    fig, ax = diagnostic_morphology.plot_topology(interactive=True)
    _draw(fig)
    collection = _artist(ax, "topology:compartments")
    node_id = collection._dendra_topology_node_ids[0]
    position = ax._dendra_topology_positions[node_id]
    display_x, display_y = ax.transData.transform(position)
    event = MouseEvent("motion_notify_event", fig.canvas, display_x, display_y)
    fig.canvas.callbacks.process("motion_notify_event", event)

    controller = ax._dendra_topology_hover
    text = controller.annotation.get_text()
    assert controller.annotation.get_visible()
    assert f"Node {node_id}" in text
    assert "Section:" in text
    assert "L=" in text and "mean diam=" in text
    assert "cm=" in text and "rhoa=" in text
    assert "area=" in text and "volume=" in text
    assert "center xyz=" in text and "labels:" in text

    outside = MouseEvent("motion_notify_event", fig.canvas, -100.0, -100.0)
    fig.canvas.callbacks.process("motion_notify_event", outside)
    assert not controller.annotation.get_visible()


def test_plot_topology_junction_hover_avoids_nonmaterial_storage_fields():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=5.0)
    first = morphology.section("first", L=5.0, diam=1.0)
    second = morphology.section("second", L=5.0, diam=1.0)
    first.connect(root.at(1.0), child_end=0)
    second.connect(root.at(1.0), child_end=0)

    fig, ax = morphology.plot_topology(interactive=True)
    scene = ax._dendra_topology_scene
    junction = next(node for node in scene.nodes if node.kind == "junction")
    controller = ax._dendra_topology_hover
    controller.show_node(junction.node_id)
    text = controller.annotation.get_text()

    assert "zero-area algebraic junction" in text
    assert "L=" not in text
    assert "cm=" not in text
    assert "rhoa=" not in text
    assert "center xyz=" not in text
    _draw(fig)


def test_plot_topology_replaces_owned_artists_and_hover_when_reusing_axes(
    diagnostic_morphology,
):
    fig, ax = diagnostic_morphology.plot_topology(interactive=True)
    first = ax._dendra_topology_hover
    assert first.annotation in ax.texts

    replacement = dn.Morphology()
    replacement.section("replacement", L=12.0, diam=2.0, nseg=2)
    returned_fig, returned_ax = replacement.plot_topology(
        interactive=True,
        legend=False,
        ax=ax,
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert ax._dendra_topology_hover is not first
    assert first.annotation not in ax.texts
    assert ax.get_legend() is None
    assert len(ax._dendra_topology_scene.nodes) == 2
    assert {node.section_name for node in ax._dendra_topology_scene.nodes} == {
        "replacement"
    }
    gids = [
        artist.get_gid() for artist in ax.get_children() if artist.get_gid() is not None
    ]
    assert gids.count("topology:compartments") == 1
    assert gids.count("topology:edges") == 1

    empty = dn.Morphology()
    empty.plot_topology(interactive=False, ax=ax)
    assert not hasattr(ax, "_dendra_topology_hover")
    assert not ax._dendra_topology_scene.nodes
    assert [
        artist.get_gid()
        for artist in ax.get_children()
        if isinstance(artist.get_gid(), str)
        and artist.get_gid().startswith("topology:")
    ] == ["topology:empty"]
    assert ax.get_legend() is None
    _draw(fig)


def test_plot_topology_broad_tree_preserves_visible_depth_spacing():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=4.0)
    for index in range(64):
        child = morphology.section(f"branch_{index}", L=5.0, diam=1.0)
        child.connect(root.at(1.0), child_end=0)

    fig, ax = morphology.plot_topology(legend=False)
    _draw(fig)
    positions = ax._dendra_topology_positions
    parent, child = ax._dendra_topology_scene.edges[0]
    parent_display = ax.transData.transform(positions[parent])
    child_display = ax.transData.transform(positions[child])

    assert abs(child_display[0] - parent_display[0]) > 40.0
    assert ax.get_aspect() == "auto"


def test_plot_topology_mutes_junctions_outside_highlighted_component():
    morphology = dn.Morphology()
    for component_index in range(2):
        labels = ("selected",) if component_index == 0 else ()
        root = morphology.section(
            f"root_{component_index}",
            L=10.0,
            diam=4.0,
            labels=labels,
        )
        for child_index in range(2):
            child = morphology.section(
                f"branch_{component_index}_{child_index}",
                L=5.0,
                diam=1.0,
                labels=labels,
            )
            child.connect(root.at(1.0), child_end=0)

    fig, ax = morphology.plot_topology(highlight="selected", legend=False)
    collection = _artist(ax, "topology:junctions")
    scene = ax._dendra_topology_scene
    alpha_by_component = {
        scene.node(node_id).component_index: color[3]
        for node_id, color in zip(
            collection._dendra_topology_node_ids,
            collection.get_facecolors(),
        )
    }

    assert alpha_by_component[0] > alpha_by_component[1]
    _draw(fig)


def test_plot_section_topology_is_coordinate_independent_and_exposes_parameters(
    diagnostic_morphology,
):
    fig, ax = diagnostic_morphology.plot_section_topology(
        color_by="rhoa",
        highlight="membrane",
        show_parameters=True,
        show_labels=True,
        show_compartments=True,
        annotate_connections=True,
    )
    gids = _artists_by_gid(ax)

    for section in ("trunk", "reverse", "forward", "gapped"):
        assert f"section:{section}" in gids
        assert f"compartments:{section}" in gids
        assert f"annotation-section:{section}" in gids
    for child in ("reverse", "forward", "gapped"):
        assert f"connection:{child}" in gids
        assert f"annotation-connection:{child}" in gids
    assert any("nseg" in text.get_text() for text in ax.texts)
    assert any("membrane" in text.get_text() for text in ax.texts)
    _draw(fig)


def test_plot_topology_supports_an_incomplete_forest():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=2.0)
    child = morphology.section("child", L=8.0, diam=1.0)
    morphology.section("second_root", L=12.0, diam=3.0)
    child.connect(root.at(1.0), child_end=0)

    fig, ax = morphology.plot_topology()

    gids = _artists_by_gid(ax)
    scene = ax._dendra_topology_scene
    assert len(scene.nodes) == root.nseg + child.nseg + 1
    assert len(scene.roots) == 2
    assert "topology:compartments" in gids
    assert "topology:edges" in gids
    _draw(fig)


def test_diameter_profile_uses_normalized_arclength(diagnostic_morphology):
    fig, ax = diagnostic_morphology.plot_diameter_profile(
        sections=("trunk",), x_axis="normalized"
    )
    profile = _artist(ax, "diameter-profile:trunk")

    assert profile.get_xdata() == pytest.approx((0.0, 0.25, 1.0))
    assert profile.get_ydata() == pytest.approx((8.0, 6.0, 4.0))
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    assert "normalized" in ax.get_xlabel().lower()
    _draw(fig)


def test_diameter_profile_uses_physical_arclength_and_exact_highlight(
    diagnostic_morphology,
):
    fig, ax = diagnostic_morphology.plot_diameter_profile(
        sections=("trunk", "reverse", "forward"),
        highlight="reverse",
        x_axis="distance",
    )
    trunk = _artist(ax, "diameter-profile:trunk")
    reverse = _artist(ax, "diameter-profile:reverse")

    assert trunk.get_xdata() == pytest.approx((0.0, 10.0, 40.0))
    assert trunk.get_ydata() == pytest.approx((8.0, 6.0, 4.0))
    assert _alpha(reverse) > _alpha(trunk)
    assert "µm" in ax.get_xlabel()
    _draw(fig)


def test_diameter_profile_validates_section_names_and_axis(diagnostic_morphology):
    with pytest.raises((KeyError, ValueError), match=r"(?i)missing|unknown|section"):
        diagnostic_morphology.plot_diameter_profile(sections=("missing",))
    with pytest.raises(ValueError, match=r"(?i)x_axis|normalized|distance"):
        diagnostic_morphology.plot_diameter_profile(x_axis="index")


def test_inspect_returns_complete_dashboard_without_display(diagnostic_morphology):
    fig, axes = diagnostic_morphology.inspect()

    assert isinstance(fig, Figure)
    assert set(axes) == {"x", "y", "z", "topology", "diameter"}
    assert all(isinstance(ax, Axes) and ax.figure is fig for ax in axes.values())
    assert (axes["x"].get_xlabel(), axes["x"].get_ylabel()) == (
        "y (µm)",
        "z (µm)",
    )
    assert (axes["y"].get_xlabel(), axes["y"].get_ylabel()) == (
        "x (µm)",
        "z (µm)",
    )
    assert (axes["z"].get_xlabel(), axes["z"].get_ylabel()) == (
        "x (µm)",
        "y (µm)",
    )
    assert _artist(axes["topology"], "topology:compartments") is not None
    assert _artist(axes["diameter"], "diameter-profile:trunk") is not None
    _draw(fig)


def test_visualization_does_not_mutate_morphology(diagnostic_morphology):
    before_sections = tuple(
        (
            section.name,
            section.points,
            section.nseg,
            section.rhoa,
            section.cm,
            section.labels,
        )
        for section in diagnostic_morphology.sections
    )
    before_scene = build_morphology_scene(diagnostic_morphology)

    figures = [
        diagnostic_morphology.plot()[0],
        diagnostic_morphology.plot_3d()[0],
        diagnostic_morphology.plot_shape(connection_tolerance_um=2.0)[0],
        diagnostic_morphology.plot_shape_3d(connection_tolerance_um=2.0)[0],
        diagnostic_morphology.plot_topology()[0],
        diagnostic_morphology.plot_section_topology()[0],
        diagnostic_morphology.plot_diameter_profile()[0],
        diagnostic_morphology.inspect()[0],
    ]

    after_sections = tuple(
        (
            section.name,
            section.points,
            section.nseg,
            section.rhoa,
            section.cm,
            section.labels,
        )
        for section in diagnostic_morphology.sections
    )
    assert after_sections == before_sections
    assert build_morphology_scene(diagnostic_morphology) == before_scene
    for fig in figures:
        _draw(fig)


def test_scene_connection_gap_uses_euclidean_distance():
    morphology = dn.Morphology()
    parent = morphology.section(
        "parent", points=((0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0))
    )
    child = morphology.section(
        "child", points=((13.0, 4.0, 0.0, 1.0), (20.0, 4.0, 0.0, 1.0))
    )
    child.connect(parent.at(1.0), child_end=0)

    scene = build_morphology_scene(morphology)

    assert len(scene.connections) == 1
    assert scene.connections[0].gap_um == pytest.approx(math.hypot(3.0, 4.0))


def test_empty_morphology_renders_in_every_public_view():
    morphology = dn.Morphology()
    figures = []

    for view in ("x", "y", "z"):
        fig, ax = morphology.plot(view=view)
        figures.append(fig)
        assert ax.get_title() == "Empty Morphology"
        assert [text.get_text() for text in ax.texts] == ["Empty Morphology"]

    fig, ax = morphology.plot_3d()
    figures.append(fig)
    assert ax.name == "3d"
    assert ax.get_title() == "Empty Morphology"

    for plotter in (morphology.plot_topology, morphology.plot_diameter_profile):
        fig, ax = plotter()
        figures.append(fig)
        assert len(ax.texts) == 1

    fig, axes = morphology.inspect()
    figures.append(fig)
    assert set(axes) == {"x", "y", "z", "topology", "diameter"}
    assert "0 Sections" in fig._suptitle.get_text()

    for figure in figures:
        _draw(figure)


@pytest.mark.parametrize(
    ("method_name", "kwargs", "error"),
    [
        ("plot", {"connection_tolerance_um": True}, TypeError),
        ("plot", {"diameter_scale": 0.0}, ValueError),
        ("plot", {"min_linewidth": float("nan")}, ValueError),
        ("plot", {"dpi": 0}, ValueError),
        ("plot", {"figsize": (8.0,)}, TypeError),
        ("plot", {"cmap": "not-a-colormap", "color_by": "cm"}, ValueError),
        ("plot_shape", {"view": "diagonal"}, ValueError),
        ("plot_shape", {"diameter_scale": True}, TypeError),
        ("plot_shape", {"diameter_scale": 0.0}, ValueError),
        ("plot_shape", {"radial_segments": True}, TypeError),
        ("plot_shape", {"radial_segments": 2}, ValueError),
        ("plot_shape", {"radial_segments": 3.5}, TypeError),
        ("plot_shape", {"connection_tolerance_um": -1.0}, ValueError),
        ("plot_shape", {"legend": "always"}, ValueError),
        ("plot_shape", {"show_axes": 1}, TypeError),
        ("plot_shape", {"dpi": 0}, ValueError),
        ("plot_shape_3d", {"diameter_scale": float("inf")}, ValueError),
        ("plot_shape_3d", {"radial_segments": 1}, ValueError),
        ("plot_shape_3d", {"legend": None}, ValueError),
        ("plot_shape_3d", {"show_axes": "yes"}, TypeError),
        ("plot_shape_3d", {"figsize": (9.0,)}, TypeError),
        ("plot_topology", {"node_size": float("inf")}, ValueError),
        ("plot_topology", {"branchpoint_size": 0.0}, ValueError),
        ("plot_topology", {"edge_width": float("nan")}, ValueError),
        ("plot_topology", {"node_scale": True}, TypeError),
        ("plot_topology", {"node_scale": 0.0}, ValueError),
        ("plot_topology", {"min_node_size": float("nan")}, ValueError),
        (
            "plot_topology",
            {"min_node_size": 20.0, "max_node_size": 19.0},
            ValueError,
        ),
        ("plot_topology", {"branchpoint_scale": float("inf")}, ValueError),
        ("plot_topology", {"min_branchpoint_size": 0.0}, ValueError),
        (
            "plot_topology",
            {"min_branchpoint_size": 30.0, "max_branchpoint_size": 29.0},
            ValueError,
        ),
        ("plot_topology", {"junction_size": False}, TypeError),
        ("plot_topology", {"junction_size": 0.0}, ValueError),
        ("plot_topology", {"edge_scale": 0.0}, ValueError),
        ("plot_topology", {"min_edge_width": float("nan")}, ValueError),
        (
            "plot_topology",
            {"min_edge_width": 2.0, "max_edge_width": 1.0},
            ValueError,
        ),
        ("plot_topology", {"node_size": True}, TypeError),
        ("plot_topology", {"branchpoint_size": float("nan")}, ValueError),
        ("plot_topology", {"edge_width": False}, TypeError),
        ("plot_topology", {"interactive": 1}, TypeError),
        (
            "plot_section_topology",
            {"connection_tolerance_um": float("inf")},
            ValueError,
        ),
        ("inspect", {"dpi": True}, TypeError),
    ],
)
def test_visualization_rejects_invalid_numeric_and_figure_options(
    diagnostic_morphology, method_name, kwargs, error
):
    with pytest.raises(error):
        getattr(diagnostic_morphology, method_name)(**kwargs)


def test_visualization_rejects_axes_of_the_wrong_dimensionality(
    diagnostic_morphology,
):
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    for plotter in (
        diagnostic_morphology.plot,
        diagnostic_morphology.plot_shape,
        diagnostic_morphology.plot_topology,
        diagnostic_morphology.plot_section_topology,
        diagnostic_morphology.plot_diameter_profile,
    ):
        with pytest.raises(ValueError, match=r"(?i)two-dimensional"):
            plotter(ax=ax_3d)

    _, ax_2d = plt.subplots()
    for plotter in (
        diagnostic_morphology.plot_3d,
        diagnostic_morphology.plot_shape_3d,
    ):
        with pytest.raises(ValueError, match=r"(?i)three-dimensional|3d"):
            plotter(ax=ax_2d)


def test_visualization_toggles_remove_their_artists(diagnostic_morphology):
    for plotter in (diagnostic_morphology.plot, diagnostic_morphology.plot_3d):
        fig, ax = plotter(
            show_points=False,
            show_connections=False,
            show_compartments=False,
            show_orientation=False,
            annotate_sections=False,
            annotate_connections=False,
        )
        gids = set(_artists_by_gid(ax))
        assert gids == {
            "section:trunk",
            "section:reverse",
            "section:forward",
            "section:gapped",
        }
        _draw(fig)

    fig, ax = diagnostic_morphology.plot_section_topology(
        show_parameters=False,
        show_labels=False,
        show_compartments=False,
        annotate_connections=False,
    )
    gids = set(_artists_by_gid(ax))
    assert not any(gid.startswith("compartments:") for gid in gids)
    assert not any(gid.startswith("annotation-connection:") for gid in gids)
    assert {text.get_text() for text in ax.texts} == {
        "trunk",
        "reverse",
        "forward",
        "gapped",
    }
    _draw(fig)

    fig, ax = diagnostic_morphology.plot_diameter_profile(
        show_points=False,
        show_compartments=False,
        show_connections=False,
    )
    assert len(ax.collections) == 0
    assert set(_artists_by_gid(ax)) == {
        "diameter-profile:trunk",
        "diameter-profile:reverse",
        "diameter-profile:forward",
        "diameter-profile:gapped",
    }
    _draw(fig)


@pytest.mark.parametrize(
    ("color_by", "value"),
    [
        ("diameter", 2.0),
        ("length", 10.0),
        ("nseg", 1.0),
        ("rhoa", 100.0),
        ("cm", 1.0),
    ],
)
def test_constant_numeric_color_scales_have_finite_nonzero_normalization(
    color_by, value
):
    morphology = dn.Morphology()
    morphology.section("constant", L=10.0, diam=2.0)

    fig, _ = morphology.plot(color_by=color_by)
    colorbar = fig.axes[-1]._colorbar

    assert math.isfinite(colorbar.norm.vmin)
    assert math.isfinite(colorbar.norm.vmax)
    assert colorbar.norm.vmin < value < colorbar.norm.vmax
    _draw(fig)


def test_completed_plot_is_snapshot_independent_of_later_updates(
    diagnostic_morphology,
):
    old_fig, old_ax = diagnostic_morphology.plot(
        show_points=False,
        show_connections=False,
        show_orientation=False,
    )
    old_artist = _artist(old_ax, "section:trunk")
    old_segments = np.asarray(old_artist.get_segments()).copy()
    old_widths = np.asarray(old_artist.get_linewidths()).copy()

    diagnostic_morphology.sections[0].at(0.5).update(diam=9.0)
    new_fig, new_ax = diagnostic_morphology.plot(
        show_points=False,
        show_connections=False,
        show_orientation=False,
    )
    new_artist = _artist(new_ax, "section:trunk")

    np.testing.assert_array_equal(np.asarray(old_artist.get_segments()), old_segments)
    np.testing.assert_array_equal(np.asarray(old_artist.get_linewidths()), old_widths)
    assert len(new_artist.get_segments()) > len(old_segments)
    assert tuple(new_artist.get_linewidths()) != pytest.approx(tuple(old_widths))
    _draw(old_fig)
    _draw(new_fig)


def test_projected_away_spatial_gap_remains_explicitly_diagnosed():
    morphology = dn.Morphology()
    parent = morphology.section(
        "parent", points=((0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0))
    )
    child = morphology.section(
        "child", points=((10.0, 5.0, 0.0, 1.0), (20.0, 5.0, 0.0, 1.0))
    )
    child.connect(parent.at(1.0), child_end=0)

    # Looking along y collapses the 5 µm gap to one displayed coordinate, but
    # the diagnostic must continue to use its true three-dimensional length.
    fig, ax = morphology.plot(
        view="y",
        show_connections=True,
        annotate_connections=True,
        connection_tolerance_um=1.0,
    )
    connector = _artist(ax, "connection:child")
    annotation = _artist(ax, "annotation-connection:child")

    assert connector.get_xdata()[0] == pytest.approx(connector.get_xdata()[1])
    assert connector.get_ydata()[0] == pytest.approx(connector.get_ydata()[1])
    assert connector.get_linestyle() == "--"
    assert "Δ=5 µm" in annotation.get_text()
    assert "spatial gap" in annotation.get_text()
    _draw(fig)


def test_diameter_profile_accepts_only_canonical_section_handles(
    diagnostic_morphology,
):
    canonical = diagnostic_morphology.sections[2]
    fig, ax = diagnostic_morphology.plot_diameter_profile(sections=canonical)

    assert set(_artists_by_gid(ax)) == {"diameter-profile:forward"}
    _draw(fig)

    foreign = dn.Morphology().section("foreign", L=5.0, diam=1.0)
    with pytest.raises(ValueError, match=r"(?i)different Morphology"):
        diagnostic_morphology.plot_diameter_profile(sections=foreign)
    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        diagnostic_morphology.plot_diameter_profile(
            sections=(canonical, canonical.name)
        )
    with pytest.raises(TypeError, match=r"(?i)sections"):
        diagnostic_morphology.plot_diameter_profile(sections=(canonical, 3))


def test_dashboard_propagates_highlight_and_validates_tolerance(
    diagnostic_morphology,
):
    fig, axes = diagnostic_morphology.inspect(highlight="shared")

    for key in ("x", "y", "z"):
        assert _alpha(_artist(axes[key], "section:reverse")) > _alpha(
            _artist(axes[key], "section:trunk")
        )
    topology = _artist(axes["topology"], "topology:compartments")
    topology_scene = axes["topology"]._dendra_topology_scene
    node_ids = topology._dendra_topology_node_ids
    color_by_node = {
        node_id: color for node_id, color in zip(node_ids, topology.get_facecolors())
    }
    reverse_node = next(
        node.node_id
        for node in topology_scene.nodes
        if node.section_name == "reverse" and node.node_id in color_by_node
    )
    trunk_node = next(
        node.node_id
        for node in topology_scene.nodes
        if node.section_name == "trunk" and node.node_id in color_by_node
    )
    assert color_by_node[reverse_node][3] > color_by_node[trunk_node][3]
    assert _alpha(_artist(axes["diameter"], "diameter-profile:forward")) > _alpha(
        _artist(axes["diameter"], "diameter-profile:trunk")
    )
    _draw(fig)

    with pytest.raises(ValueError, match=r"(?i)tolerance"):
        diagnostic_morphology.inspect(connection_tolerance_um=-1.0)
    with pytest.raises(ValueError, match=r"(?i)highlight|unknown"):
        diagnostic_morphology.inspect(highlight="not-a-label")


def _axes_disclosure_text(ax: Axes) -> str:
    """Collect user-visible coordinate context without assuming its placement."""
    values = [
        ax.get_xlabel(),
        ax.get_ylabel(),
        ax.get_title(),
        ax.xaxis.get_offset_text().get_text(),
        ax.yaxis.get_offset_text().get_text(),
    ]
    if hasattr(ax, "get_zlabel"):
        values.extend(
            (
                ax.get_zlabel(),
                ax.zaxis.get_offset_text().get_text(),
            )
        )
    values.extend(text.get_text() for text in ax.texts)
    values.extend(text.get_text() for text in ax.figure.texts)
    return " ".join(value for value in values if value)


def _assert_large_origin_is_disclosed(ax: Axes) -> None:
    disclosure = _axes_disclosure_text(ax).lower().replace(" ", "")
    magnitude_is_visible = any(
        marker in disclosure for marker in ("1e+100", "1e100", "10^{100}", "10^100")
    )
    offset_is_explicit = any(
        marker in disclosure for marker in ("offset", "origin", "relative", "+1e")
    )
    assert magnitude_is_visible, disclosure
    assert offset_is_explicit, disclosure


def test_huge_absolute_coordinate_preserves_local_span_and_discloses_origin():
    morphology = dn.Morphology()
    morphology.section(
        "remote",
        points=((0.0, 1e100, 0.0, 2.0), (10.0, 1e100, 0.0, 1.0)),
    )

    fig_2d, ax_2d = morphology.plot(
        view="z",
        show_points=False,
        show_connections=False,
        show_orientation=False,
    )
    section_2d = _artist(ax_2d, "section:remote")
    segment_2d = np.asarray(section_2d.get_segments(), dtype=float)

    assert np.isfinite(segment_2d).all()
    assert segment_2d.ndim == 3 and segment_2d.shape[1:] == (2, 2)
    assert segment_2d[-1, 1, 0] - segment_2d[0, 0, 0] == pytest.approx(10.0)
    assert segment_2d[-1, 1, 1] - segment_2d[0, 0, 1] == pytest.approx(0.0)
    _assert_large_origin_is_disclosed(ax_2d)
    _draw(fig_2d)

    fig_3d, ax_3d = morphology.plot_3d(
        show_points=False,
        show_connections=False,
        show_orientation=False,
    )
    section_3d = _artist(ax_3d, "section:remote")
    segment_3d = np.asarray(section_3d._segments3d, dtype=float)

    assert np.isfinite(segment_3d).all()
    assert segment_3d.ndim == 3 and segment_3d.shape[1:] == (2, 3)
    assert segment_3d[-1, 1, 0] - segment_3d[0, 0, 0] == pytest.approx(10.0)
    np.testing.assert_allclose(segment_3d[-1, 1, 1:] - segment_3d[0, 0, 1:], 0.0)
    _assert_large_origin_is_disclosed(ax_3d)
    _draw(fig_3d)


@pytest.mark.parametrize("color_by", ["rhoa", "cm"])
@pytest.mark.parametrize(
    ("value", "display_value", "scale_marker"),
    [(sys.float_info.max, 1.0, "e+308"), (math.ulp(0.0), math.ulp(0.0), None)],
    ids=("maximum-finite", "minimum-subnormal"),
)
def test_extreme_constant_continuous_fields_have_finite_color_normalization(
    color_by, value, display_value, scale_marker
):
    morphology = dn.Morphology(rhoa=value, cm=value)
    morphology.section("constant", L=10.0, diam=2.0)

    fig, ax = morphology.plot(
        color_by=color_by,
        show_points=False,
        show_connections=False,
        show_orientation=False,
    )
    section = _artist(ax, "section:constant")
    colorbar = fig.axes[-1]._colorbar

    assert np.isfinite(section.get_colors()).all()
    assert math.isfinite(colorbar.norm.vmin)
    assert math.isfinite(colorbar.norm.vmax)
    assert colorbar.norm.vmin < colorbar.norm.vmax
    # Maximum-exponent values are divided by an explicitly disclosed display
    # scale before reaching Matplotlib. Tiny values remain safe on their raw
    # scale; both avoid overflow/underflow in colorbar boundary arithmetic.
    assert colorbar.norm.vmin <= display_value <= colorbar.norm.vmax
    np.testing.assert_allclose(
        section.get_colors()[0], colorbar.mappable.to_rgba(display_value)
    )
    colorbar_label = fig.axes[-1].get_ylabel().lower().replace(" ", "")
    if scale_marker is None:
        assert "values÷" not in colorbar_label
    else:
        assert "values÷" in colorbar_label
        assert scale_marker in colorbar_label
    _draw(fig)


def _overflowing_connection_gap_morphology() -> dn.Morphology:
    morphology = dn.Morphology()
    parent = morphology.section(
        "parent",
        points=((-1e308, 0.0, 0.0, 2.0), (-1e308, 1.0, 0.0, 2.0)),
    )
    child = morphology.section(
        "child",
        points=((1e308, 1.0, 0.0, 1.0), (1e308, 2.0, 0.0, 1.0)),
    )
    child.connect(parent.at(1.0), child_end=0)
    return morphology


def test_scene_preserves_connection_gap_beyond_binary64_range():
    morphology = _overflowing_connection_gap_morphology()

    # Extreme absolute placement does not invalidate the electrical tree.
    assert morphology.compile().n_compartments == 2
    scene = build_morphology_scene(morphology)
    connection = scene.connections[0]

    assert connection.parent_name == "parent"
    assert connection.child_name == "child"
    assert math.isinf(connection.gap_um)
    assert connection.gap_um > 0.0


def test_section_topology_reports_connection_gap_beyond_binary64_range():
    morphology = _overflowing_connection_gap_morphology()

    fig, ax = morphology.plot_section_topology(annotate_connections=True)
    connector = _artist(ax, "connection:child")
    annotation = _artist(ax, "annotation-connection:child")

    assert connector.get_linestyle() == "--"
    assert "beyond binary64 range" in annotation.get_text()
    _draw(fig)


def _assert_finite_spatial_axes(ax: Axes) -> None:
    limits = [ax.get_xlim(), ax.get_ylim()]
    if getattr(ax, "name", "rectilinear") == "3d":
        limits.append(ax.get_zlim())
    assert np.isfinite(np.asarray(limits, dtype=float)).all()


def _assert_extreme_spatial_scale_is_disclosed(ax: Axes) -> None:
    disclosure = _axes_disclosure_text(ax).lower().replace(" ", "")
    assert "e+308" in disclosure, disclosure
    assert "scale" in disclosure or "÷" in disclosure, disclosure


def test_spatial_views_render_connection_gap_beyond_binary64_range_without_warnings():
    morphology = _overflowing_connection_gap_morphology()

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        fig_2d, ax_2d = morphology.plot(
            view="z",
            show_connections=True,
            annotate_connections=True,
        )
        for section_name in ("parent", "child"):
            segments = np.asarray(
                _artist(ax_2d, f"section:{section_name}").get_segments(),
                dtype=float,
            )
            assert np.isfinite(segments).all()
        connection_2d = _artist(ax_2d, "connection:child")
        connection_xy = np.column_stack(connection_2d.get_data())
        assert np.isfinite(connection_xy).all()
        assert (
            "beyond binary64 range"
            in _artist(ax_2d, "annotation-connection:child").get_text()
        )
        _assert_finite_spatial_axes(ax_2d)
        _assert_extreme_spatial_scale_is_disclosed(ax_2d)
        _draw(fig_2d)

        fig_3d, ax_3d = morphology.plot_3d(
            show_connections=True,
            annotate_connections=True,
        )
        for section_name in ("parent", "child"):
            segments = np.asarray(
                _artist(ax_3d, f"section:{section_name}")._segments3d,
                dtype=float,
            )
            assert np.isfinite(segments).all()
        connection_3d = _artist(ax_3d, "connection:child")
        connection_xyz = np.column_stack(connection_3d.get_data_3d())
        assert np.isfinite(connection_xyz).all()
        assert (
            "beyond binary64 range"
            in _artist(ax_3d, "annotation-connection:child").get_text()
        )
        _assert_finite_spatial_axes(ax_3d)
        _assert_extreme_spatial_scale_is_disclosed(ax_3d)
        _draw(fig_3d)
