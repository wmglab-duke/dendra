"""Contracts for transactional edits to native Morphology declarations."""

from __future__ import annotations

import copy
import math
import pickle
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

import dendra as dn
from dendra.models.morphology import SectionLocation

pytestmark = pytest.mark.cpu


def _section_state(section):
    return (
        section.name,
        section.nseg,
        section.L,
        section.diam,
        section.points,
        section.rhoa,
        section.cm,
        section.labels,
    )


def _section_nodes(graph, section_name):
    return [
        node
        for node, source in enumerate(graph.metadata.section_name)
        if source == section_name
    ]


def test_update_section_by_name_and_handle_preserves_identity_and_order():
    morphology = dn.Morphology()
    soma = morphology.section("soma", L=20.0, diam=10.0, labels="cell_body")
    dend = morphology.section("dend", L=30.0, diam=2.0)
    location = soma.at(0.5)

    result = morphology.update_section(
        "soma",
        rhoa=125.0,
        cm=1.4,
        nseg=3,
        labels={"membrane", "cell_body"},
    )

    assert result is soma
    assert morphology.sections == (soma, dend)
    assert location.section is soma
    assert (soma.rhoa, soma.cm, soma.nseg) == (125.0, 1.4, 3)
    assert soma.labels == frozenset({"soma", "membrane", "cell_body"})
    assert morphology.update_section(soma, rhoa=140.0) is soma
    assert soma.update(cm=1.6) is soma
    assert soma.update() is soma
    assert (soma.rhoa, soma.cm) == (140.0, 1.6)


def test_section_fields_remain_read_only_outside_update_api():
    morphology = dn.Morphology()
    section = morphology.section("soma", L=10.0, diam=4.0)
    location = section.at(0.5)

    with pytest.raises(FrozenInstanceError):
        section.rhoa = 120.0
    with pytest.raises(FrozenInstanceError):
        location.x = 0.25


def test_whole_section_update_recompiles_electrical_geometry_and_labels():
    morphology = dn.Morphology()
    section = morphology.section("cable", L=10.0, diam=2.0, nseg=2, rhoa=100.0, cm=1.0)

    section.update(rhoa=150.0, cm=1.5, nseg=4, labels="active")
    graph = morphology.compile()

    assert graph.n_compartments == 4
    assert graph.geometry.length_um == (2.5,) * 4
    assert graph.geometry.rhoa_ohm_cm == (150.0,) * 4
    assert graph.geometry.cm_uF_cm2 == (1.5,) * 4
    assert graph.nodes_with_label("cable") == (0, 1, 2, 3)
    assert graph.nodes_with_label("active") == (0, 1, 2, 3)
    assert graph.geometry.edge_resistance_ohm[1:] == pytest.approx(
        (150.0 * 1e4 * 2.5 / math.pi,) * 3
    )

    section.update(labels=())
    assert section.labels == frozenset({"cable"})
    assert morphology.compile().nodes_with_label("active") == ()


def test_stylized_length_and_diameter_updates_regenerate_canonical_geometry():
    morphology = dn.Morphology(rhoa=80.0)
    section = morphology.section("cable", L=10.0, diam=2.0, nseg=2)

    assert section.update(L=20.0) is section
    assert section.update(diam=4.0) is section

    assert section.L == 20.0
    assert section.diam == 4.0
    assert section.points == (
        (0.0, 0.0, 0.0, 4.0),
        (0.0, 0.0, 20.0, 4.0),
    )
    assert not section.is_pt3d

    graph = morphology.compile()
    assert graph.geometry.length_um == (10.0, 10.0)
    assert graph.geometry.diameter_um == (4.0, 4.0)
    assert graph.geometry.area_um2 == pytest.approx((40.0 * math.pi,) * 2)
    assert graph.geometry.volume_um3 == pytest.approx((40.0 * math.pi,) * 2)
    assert graph.geometry.edge_resistance_ohm[1] == pytest.approx(
        80.0 * 1e4 * 10.0 / (4.0 * math.pi)
    )


def test_points_replace_pt3d_geometry_and_promote_a_stylized_section():
    morphology = dn.Morphology()
    section = morphology.section("cable", L=10.0, diam=2.0, nseg=2)
    location = section.at(0.5)
    promoted_points = (
        (0.0, 0.0, 0.0, 2.0),
        (3.0, 4.0, 0.0, 4.0),
        (3.0, 4.0, 12.0, 6.0),
    )

    assert morphology.update_section(section, points=promoted_points) is section
    assert section.is_pt3d
    assert section.diam is None
    assert section.L == pytest.approx(17.0)
    assert section.points == promoted_points
    assert location.section is section

    replacement_points = (
        (1.0, 2.0, 3.0, 5.0),
        (4.0, 6.0, 3.0, 1.0),
    )
    assert section.update(points=replacement_points) is section
    assert section.points == replacement_points
    assert section.L == pytest.approx(5.0)
    assert section.diam is None


def test_pt3d_derived_length_must_be_finite_at_declaration():
    morphology = dn.Morphology()

    with pytest.raises(ValueError, match="finite"):
        morphology.section(
            "overflow",
            points=((-1e308, 0.0, 0.0, 2.0), (1e308, 0.0, 0.0, 2.0)),
        )

    assert morphology.sections == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"L": 20.0},
        {"diam": 3.0},
        {"points": ((0, 0, 0, 2), (0, 0, 20, 2)), "L": 20.0},
        {"points": ((0, 0, 0, 2), (0, 0, 20, 2)), "diam": 2.0},
    ],
)
def test_pt3d_rejects_derived_or_mixed_geometry_updates_atomically(kwargs):
    morphology = dn.Morphology()
    section = morphology.section(
        "cable",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0)),
        nseg=2,
    )
    before_state = _section_state(section)
    before_graph = morphology.compile()

    with pytest.raises(ValueError, match=r"points|pt3d|derive"):
        section.update(**kwargs)

    assert _section_state(section) == before_state
    assert morphology.compile() == before_graph


def test_location_update_replaces_an_exact_pt3d_sample_without_moving_it():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 4.0, 4.0),
            (0.0, 0.0, 10.0, 6.0),
        ),
        nseg=4,
    )
    before_xyz = tuple(point[:3] for point in section.points)
    before_length = section.L

    result = section.at(0.4).update(diam=8.0)

    assert result is section
    assert len(section.points) == 3
    assert tuple(point[:3] for point in section.points) == before_xyz
    assert tuple(point[3] for point in section.points) == (2.0, 8.0, 6.0)
    assert section.L == before_length


def test_location_update_inserts_once_and_compiles_like_explicit_pt3d_geometry():
    morphology = dn.Morphology(rhoa=120.0, cm=1.2)
    section = morphology.section(
        "dend",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 6.0)),
        nseg=4,
    )
    old_graph = morphology.compile()

    assert section.at(0.25).update(diam=8.0) is section
    assert section.points == (
        (0.0, 0.0, 0.0, 2.0),
        (0.0, 0.0, 2.5, 8.0),
        (0.0, 0.0, 10.0, 6.0),
    )
    assert section.at(0.25).update(diam=7.0) is section
    assert len(section.points) == 3
    expected_points = (
        (0.0, 0.0, 0.0, 2.0),
        (0.0, 0.0, 2.5, 7.0),
        (0.0, 0.0, 10.0, 6.0),
    )
    assert section.points == expected_points

    updated_graph = morphology.compile()
    fresh = dn.Morphology(rhoa=120.0, cm=1.2)
    fresh.section("dend", points=expected_points, nseg=4)
    assert updated_graph == fresh.compile()
    assert updated_graph != old_graph
    assert old_graph.geometry.diameter_um == pytest.approx((2.5, 3.5, 4.5, 5.5))


def test_saved_location_selects_the_updated_centerline_geometry():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend", points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0))
    )
    midpoint = section.at(0.5)

    section.update(points=((0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 20.0, 4.0)))
    midpoint.update(diam=7.0)

    assert section.points == (
        (0.0, 0.0, 0.0, 2.0),
        (5.0, 0.0, 10.0, 7.0),
        (10.0, 0.0, 20.0, 4.0),
    )


def test_location_update_treats_nextafter_x_as_distinct_when_representable():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 4.0, 4.0),
            (0.0, 0.0, 10.0, 6.0),
        ),
    )
    near_sample = math.nextafter(0.4, 1.0)

    section.at(near_sample).update(diam=9.0)

    assert len(section.points) == 4
    assert section.points[1] == (0.0, 0.0, 4.0, 4.0)
    assert section.points[2][2] > section.points[1][2]
    assert section.points[2][3] == 9.0


def test_small_normalized_x_inserts_when_physical_location_is_distinct():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 1e16, 4.0)),
    )

    section.at(1e-16).update(diam=9.0)

    assert section.points == (
        (0.0, 0.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 9.0),
        (0.0, 0.0, 1e16, 4.0),
    )


def test_repeated_location_update_handles_coordinate_ulp_roundoff():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (67838342.98999065, 28221587.372390587, 69235009.6098231, 2.0),
            (70550516.26624496, 26543285.66916373, 70229877.71574682, 4.0),
        ),
    )
    location = section.at(0.589790239235281)

    location.update(diam=3.0)
    location.update(diam=3.5)

    assert len(section.points) == 3
    assert section.points[1][3] == 3.5


def test_repeated_location_update_recovers_by_coordinate_when_x_drifts():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (
                4.510429032892685e77,
                -4.6306092435346224e77,
                4.598815947209131e77,
                2.0,
            ),
            (
                4.510662576957519e77,
                -4.630208981833672e77,
                4.598281291870139e77,
                4.0,
            ),
        ),
    )
    location = section.at(0.4126443692773669)

    location.update(diam=3.0)
    inserted_xyz = section.points[1][:3]
    location.update(diam=3.5)

    assert len(section.points) == 3
    assert section.points[1] == (*inserted_xyz, 3.5)


def test_repeated_location_update_is_idempotent_for_ordinary_roundoff_case():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (
                727301.0419774503,
                -92124258.58306153,
                -79815751.76220667,
                2.0,
            ),
            (
                824948.0717219504,
                -92184387.42496812,
                -79844040.70194435,
                4.0,
            ),
        ),
    )
    location = section.at(0.7315983062253606)

    location.update(diam=3.0)
    inserted_xyz = section.points[1][:3]
    location.update(diam=3.5)

    assert len(section.points) == 3
    assert section.points[1] == (*inserted_xyz, 3.5)


def test_repeated_location_update_is_idempotent_under_huge_cancellation():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (
                4.288694889803538e73,
                -8.526725871392942e73,
                -9.542399043325504e73,
                2.0,
            ),
            (
                9.884333056223993e73,
                1.414802940144107e73,
                -1.5718357096638304e73,
                4.0,
            ),
        ),
    )
    location = section.at(0.8578537189405654)

    location.update(diam=3.0)
    inserted_xyz = section.points[1][:3]
    location.update(diam=3.5)

    assert len(section.points) == 3
    assert section.points[1] == (*inserted_xyz, 3.5)


def test_one_ulp_control_point_on_short_large_offset_span_is_accepted():
    base = 1e100
    ulp = math.ulp(base)
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (base, 0.0, 0.0, 2.0),
            (base + 100 * ulp, 0.0, 0.0, 4.0),
        ),
    )

    section.at(0.01).update(diam=9.0)

    assert section.points == (
        (base, 0.0, 0.0, 2.0),
        (base + ulp, 0.0, 0.0, 9.0),
        (base + 100 * ulp, 0.0, 0.0, 4.0),
    )


def test_tiny_exact_x_with_distinct_coordinate_does_not_edit_endpoint():
    base = 1e100
    ulp = math.ulp(base)
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (base, 0.0, 0.0, 2.0),
            (base + 10**13 * ulp, 0.0, 0.0, 4.0),
        ),
    )

    section.at(1e-13).update(diam=9.0)

    assert section.points[0] == (base, 0.0, 0.0, 2.0)
    assert section.points[1] == (base + ulp, 0.0, 0.0, 9.0)
    assert len(section.points) == 3


def test_saved_and_new_exact_x_locations_target_one_control_point():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0)),
    )
    saved_location = section.at(0.25)

    saved_location.update(diam=3.0)
    inserted_xyz = section.points[1][:3]
    new_location = section.at(0.25)
    new_location.update(diam=5.0)
    saved_location.update(diam=7.0)

    assert new_location is not saved_location
    assert len(section.points) == 3
    assert section.points[1] == (*inserted_xyz, 7.0)


def test_lower_x_insertion_shifts_higher_control_and_both_remain_updateable():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0)),
    )
    lower = section.at(0.25)
    higher = section.at(0.75)

    higher.update(diam=7.0)
    higher_xyz = section.points[1][:3]
    lower.update(diam=5.0)
    lower_xyz = section.points[1][:3]

    # Inserting the lower-x point shifts the previously registered higher-x
    # control from index 1 to index 2. Both saved locations must follow their
    # own control point rather than whichever point now occupies the old index.
    higher.update(diam=8.0)
    lower.update(diam=6.0)

    assert section.points == (
        (0.0, 0.0, 0.0, 2.0),
        (*lower_xyz, 6.0),
        (*higher_xyz, 8.0),
        (0.0, 0.0, 10.0, 4.0),
    )


def test_whole_points_replacement_clears_local_control_provenance():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0)),
    )
    saved_location = section.at(0.25)
    saved_location.update(diam=7.0)

    replacement = (
        (10.0, 0.0, 0.0, 3.0),
        (20.0, 0.0, 0.0, 5.0),
    )
    section.update(points=replacement)
    saved_location.update(diam=9.0)

    # The saved selector keeps normalized x=0.25, but its old point-index
    # provenance must not survive an explicit replacement centerline.
    assert section.points == (
        replacement[0],
        (12.5, 0.0, 0.0, 9.0),
        replacement[1],
    )


def test_non_geometric_section_update_preserves_local_control_provenance():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (
                727301.0419774503,
                -92124258.58306153,
                -79815751.76220667,
                2.0,
            ),
            (
                824948.0717219504,
                -92184387.42496812,
                -79844040.70194435,
                4.0,
            ),
        ),
    )
    location = section.at(0.7315983062253606)
    location.update(diam=3.0)
    inserted_xyz = section.points[1][:3]

    section.update(rhoa=135.0, cm=1.3, nseg=3, labels={"active"})
    location.update(diam=8.0)

    assert (section.rhoa, section.cm, section.nseg) == (135.0, 1.3, 3)
    assert section.labels == frozenset({"dend", "active"})
    assert len(section.points) == 3
    assert section.points[1] == (*inserted_xyz, 8.0)


def test_endpoint_controls_shift_correctly_after_interior_insertion():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0)),
    )
    start = section.at(0.0)
    middle = section.at(0.5)
    end = section.at(1.0)

    start.update(diam=3.0)
    end.update(diam=5.0)
    middle.update(diam=7.0)

    # The interior insertion shifts the registered x=1 endpoint index while
    # x=0 remains at index 0. Both saved endpoint selectors must still target
    # their endpoints rather than the newly inserted control.
    start.update(diam=6.0)
    end.update(diam=8.0)

    assert section.points == (
        (0.0, 0.0, 0.0, 6.0),
        (0.0, 0.0, 5.0, 7.0),
        (0.0, 0.0, 10.0, 8.0),
    )


@pytest.mark.parametrize("round_trip", ("deepcopy", "pickle"))
def test_shifted_controls_survive_copy_round_trip_with_canonical_owner(round_trip):
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0)),
    )
    section.at(0.75).update(diam=7.0)
    section.at(0.25).update(diam=5.0)
    original_points = section.points

    if round_trip == "deepcopy":
        restored = copy.deepcopy(morphology)
    else:
        restored = pickle.loads(pickle.dumps(morphology))
    restored_section = restored.sections[0]

    assert restored is not morphology
    assert restored_section is not section
    assert restored_section._owner is restored

    restored_section.at(0.75).update(diam=8.0)
    restored_section.at(0.25).update(diam=6.0)

    assert len(restored_section.points) == 4
    assert restored_section.points[1] == (0.0, 0.0, 2.5, 6.0)
    assert restored_section.points[2] == (0.0, 0.0, 7.5, 8.0)
    assert section.points == original_points


def test_invalid_points_replacement_preserves_local_control_provenance():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (
                727301.0419774503,
                -92124258.58306153,
                -79815751.76220667,
                2.0,
            ),
            (
                824948.0717219504,
                -92184387.42496812,
                -79844040.70194435,
                4.0,
            ),
        ),
    )
    location = section.at(0.7315983062253606)
    location.update(diam=3.0)
    inserted_xyz = section.points[1][:3]
    before = _section_state(section)

    with pytest.raises(ValueError, match="distinct coordinates"):
        section.update(points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 2.0)))

    assert _section_state(section) == before
    location.update(diam=8.0)
    assert len(section.points) == 3
    assert section.points[1] == (*inserted_xyz, 8.0)


def test_local_diameter_update_rejects_an_ambiguous_diameter_discontinuity():
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 5.0, 2.0),
            (0.0, 0.0, 5.0, 4.0),
            (0.0, 0.0, 10.0, 4.0),
        ),
    )
    before = _section_state(section)

    with pytest.raises(
        ValueError,
        match=r"(?i)diameter discontinuity|ambiguous.*replace.*points",
    ):
        section.at(0.5).update(diam=3.0)

    assert _section_state(section) == before


def test_unrepresentable_local_control_point_is_rejected_atomically():
    base = 1e100
    delta = 2.0 * math.ulp(base)
    morphology = dn.Morphology()
    section = morphology.section(
        "dend",
        points=(
            (base, base, base, 2.0),
            (base + delta, base + 2.0 * delta, base + 3.0 * delta, 4.0),
        ),
    )
    before = _section_state(section)

    with pytest.raises(ValueError, match="coordinate precision"):
        section.at(0.55).update(diam=3.0)

    assert _section_state(section) == before


@pytest.mark.parametrize(
    ("diameter", "error"),
    [
        (True, TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_invalid_location_diameter_is_atomic(diameter, error):
    morphology = dn.Morphology()
    section = morphology.section(
        "dend", points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0))
    )
    location = section.at(0.5)
    before_state = _section_state(section)
    before_graph = morphology.compile()

    with pytest.raises(error, match=r"diameter|positive|finite|number"):
        location.update(diam=diameter)

    assert _section_state(section) == before_state
    assert morphology.compile() == before_graph


def test_location_update_is_pt3d_only_and_electrical_fields_are_not_local():
    morphology = dn.Morphology()
    stylized = morphology.section("stylized", L=10.0, diam=2.0)
    before = _section_state(stylized)

    with pytest.raises(ValueError, match="pt3d"):
        stylized.at(0.5).update(diam=3.0)
    assert _section_state(stylized) == before

    pt3d = morphology.section(
        "pt3d", points=((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 10.0, 4.0))
    )
    location = pt3d.at(0.5)
    with pytest.raises(TypeError, match="unexpected keyword"):
        location.update(rhoa=150.0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        location.update(cm=1.5)


def test_connections_orientation_and_existing_locations_survive_updates():
    morphology = dn.Morphology()
    root = morphology.section(
        "root", points=((0.0, 0.0, 0.0, 8.0), (0.0, 0.0, 10.0, 6.0))
    )
    child = morphology.section(
        "child",
        points=((0.0, 0.0, 30.0, 2.0), (0.0, 0.0, 10.0, 4.0)),
        nseg=2,
    )
    parent_location = root.at(1.0)
    child_location = child.at(0.5)
    child.connect(parent_location, child_end=1)
    before = morphology.compile()

    assert root.update(rhoa=130.0, cm=1.3) is root
    assert child_location.update(diam=7.0) is child
    after = morphology.compile()

    assert parent_location.section is root
    assert child_location.section is child
    assert after.topology.parent_index == before.topology.parent_index
    child_nodes = _section_nodes(after, "child")
    assert [after.metadata.segment_index[node] for node in child_nodes] == [1, 0]
    assert [after.metadata.section_x[node] for node in child_nodes] == pytest.approx(
        [0.75, 0.25]
    )


def test_location_created_before_update_can_define_a_later_connection():
    morphology = dn.Morphology()
    root = morphology.section(
        "root", points=((0.0, 0.0, 0.0, 4.0), (0.0, 0.0, 20.0, 2.0)), nseg=2
    )
    child = morphology.section("child", L=8.0, diam=1.0)
    location = root.at(0.5)

    root.update(rhoa=160.0, cm=1.6)
    child.connect(location, child_end=0)
    graph = morphology.compile()

    child_node = _section_nodes(graph, "child")[0]
    host = graph.topology.parent_index[child_node]
    assert graph.metadata.section_name[host] == "root"
    assert graph.metadata.segment_index[host] == 1


def test_update_failures_are_atomic_and_the_section_remains_usable():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=4.0, labels="membrane")
    child = morphology.section("child", L=10.0, diam=2.0)
    child.connect(root.at(1.0), child_end=0)

    invalid_updates = (
        (ValueError, {"rhoa": 0.0}),
        (ValueError, {"cm": float("inf")}),
        (TypeError, {"nseg": True}),
        (ValueError, {"labels": {"child"}}),
        (
            ValueError,
            {
                "points": ((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 20.0, 2.0)),
                "L": 20.0,
            },
        ),
        (
            ValueError,
            {"points": ((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 2.0))},
        ),
        (
            ValueError,
            {"points": ((-1e308, 0.0, 0.0, 2.0), (1e308, 0.0, 0.0, 2.0))},
        ),
    )

    for error, kwargs in invalid_updates:
        before_sections = morphology.sections
        before_state = _section_state(root)
        before_graph = morphology.compile()
        with pytest.raises(error):
            root.update(**kwargs)
        assert morphology.sections == before_sections
        assert morphology.sections[0] is root
        assert _section_state(root) == before_state
        assert morphology.compile() == before_graph

    assert root.update(rhoa=175.0) is root
    assert root.rhoa == 175.0


def test_noncanonical_same_owner_and_foreign_section_handles_are_rejected():
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=4.0)
    child = morphology.section("child", L=10.0, diam=2.0)
    forged = replace(root, rhoa=999.0)
    forged_location = SectionLocation(forged, 1.0)

    with pytest.raises(ValueError, match="canonical"):
        morphology.update_section(forged, rhoa=120.0)
    with pytest.raises(ValueError, match="canonical"):
        forged.update(rhoa=120.0)
    with pytest.raises(ValueError, match="canonical"):
        forged.at(0.5)
    with pytest.raises(ValueError, match="canonical"):
        child.connect(forged_location, child_end=0)

    other = dn.Morphology()
    foreign = other.section("foreign", L=5.0, diam=1.0)
    with pytest.raises(ValueError, match="different Morphology"):
        morphology.update_section(foreign, rhoa=120.0)
    with pytest.raises(KeyError, match="Unknown Section"):
        morphology.update_section("missing", rhoa=120.0)

    assert morphology.sections == (root, child)


def test_validated_morphology_defaults_affect_only_future_sections_atomically():
    morphology = dn.Morphology(rhoa=90.0, cm=0.8)
    existing = morphology.section("existing", L=10.0, diam=2.0)

    morphology.rhoa = 140.0
    morphology.cm = 1.4
    future = morphology.section("future", L=10.0, diam=2.0)

    assert (existing.rhoa, existing.cm) == (90.0, 0.8)
    assert (future.rhoa, future.cm) == (140.0, 1.4)

    before_defaults = (morphology.rhoa, morphology.cm)
    with pytest.raises(ValueError, match="positive"):
        morphology.rhoa = 0.0
    with pytest.raises(ValueError, match="finite"):
        morphology.cm = float("nan")
    assert (morphology.rhoa, morphology.cm) == before_defaults
    assert (existing.rhoa, existing.cm) == (90.0, 0.8)
    assert (future.rhoa, future.cm) == (140.0, 1.4)


@pytest.mark.parametrize("model_type", (dn.Tree, dn.Cable), ids=("tree", "cable"))
def test_existing_graph_and_model_are_independent_snapshots_of_updates(model_type):
    morphology = dn.Morphology()
    section = morphology.section("cable", L=10.0, diam=2.0, nseg=2, rhoa=100.0, cm=1.0)
    old_graph = morphology.compile()
    old_model = model_type.from_morphology(morphology, dtype=torch.float64)
    old_model_graph = old_model.compartment_graph
    old_dx = old_model.dx.clone()
    old_diam = old_model.diam.clone()
    old_rhoa = old_model.rhoa.clone()
    old_cm = old_model.cm.clone()

    section.update(L=18.0, diam=3.0, nseg=3, rhoa=180.0, cm=1.8)
    new_graph = morphology.compile()
    new_model = model_type.from_morphology(morphology, dtype=torch.float64)

    assert old_model.compartment_graph is old_model_graph
    assert old_model.compartment_graph == old_graph
    assert old_model.shape == (1, 2)
    assert old_graph.geometry.length_um == (5.0, 5.0)
    torch.testing.assert_close(old_model.dx, old_dx)
    torch.testing.assert_close(old_model.diam, old_diam)
    torch.testing.assert_close(old_model.rhoa, old_rhoa)
    torch.testing.assert_close(old_model.cm, old_cm)

    assert new_graph != old_graph
    assert new_model.compartment_graph == new_graph
    assert new_model.shape == (1, 3)
    assert new_graph.geometry.length_um == (6.0, 6.0, 6.0)
    assert new_model.dx.tolist() == [[6.0, 6.0, 6.0]]
    assert new_model.diam.tolist() == [[3.0, 3.0, 3.0]]
    assert new_model.rhoa.tolist() == [[180.0, 180.0, 180.0]]
    assert new_model.cm.tolist() == [[1.8, 1.8, 1.8]]
