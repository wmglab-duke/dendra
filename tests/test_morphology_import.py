"""Contracts for importing classic SWC into native ``Morphology`` objects."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest

import dendra as dn

pytestmark = pytest.mark.cpu


@dataclass(frozen=True)
class _SwcNode:
    node_id: int
    type_id: int
    x: float
    y: float
    z: float
    radius: float
    parent_id: int

    @property
    def record(self) -> tuple[int, float, float, float, float]:
        return (self.type_id, self.x, self.y, self.z, self.radius)


def _write_swc(tmp_path, text: str, *, name: str = "cell.swc"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _swc_nodes(text: str) -> dict[int, _SwcNode]:
    nodes = {}
    for line in text.splitlines():
        payload = line.split("#", 1)[0].strip()
        if not payload:
            continue
        fields = payload.split()
        assert len(fields) == 7
        node = _SwcNode(
            node_id=int(fields[0]),
            type_id=int(fields[1]),
            x=float(fields[2]),
            y=float(fields[3]),
            z=float(fields[4]),
            radius=float(fields[5]),
            parent_id=int(fields[6]),
        )
        nodes[node.node_id] = node
    return nodes


def _canonical_swc_tree(text: str):
    """Return an ID- and row-order-independent geometric SWC tree."""
    nodes = _swc_nodes(text)
    roots = Counter(node.record for node in nodes.values() if node.parent_id == -1)
    edges = Counter(
        (nodes[node.parent_id].record, node.record)
        for node in nodes.values()
        if node.parent_id != -1
    )
    return roots, edges


def _branched_swc() -> str:
    return (
        "1 3 0 0 0 2 -1\n"
        "2 3 0 10 0 1.5 1\n"
        "3 3 0 20 0 1 2\n"
        "4 3 -10 30 0 0.75 3\n"
        "5 4 10 30 0 0.5 3\n"
        "6 3 -20 40 0 0.5 4\n"
        "7 4 20 40 0 0.25 5\n"
    )


def test_from_swc_accepts_pathlike_comments_inline_comments_and_reordered_rows(
    tmp_path, monkeypatch
):
    path = _write_swc(
        tmp_path,
        "# classic seven-column SWC\n"
        "30 3 0 20 0 0.5 20  # distal sample appears first\n"
        "\n"
        "10 3 0 0 0 2 -1\n"
        "20 3 0 10 0 1 10\n",
    )
    # Native SWC reconstruction must not rely on NEURON's Import3d reader.
    monkeypatch.setitem(sys.modules, "neuron", None)

    morphology = dn.Morphology.from_swc(path)

    assert [section.name for section in morphology.sections] == ["basal_dendrite_0"]
    section = morphology.sections[0]
    assert section.points == (
        (0.0, 0.0, 0.0, 4.0),
        (0.0, 10.0, 0.0, 2.0),
        (0.0, 20.0, 0.0, 1.0),
    )
    assert section.nseg == 1
    assert section.rhoa == 100.0
    assert section.cm == 1.0
    assert {"basal_dendrite", "dendrite", "swc_type_3"} <= section.labels
    assert morphology.swc_section_types == {"basal_dendrite_0": 3}


def test_from_swc_accepts_an_optional_utf8_byte_order_mark(tmp_path):
    path = tmp_path / "bom.swc"
    path.write_bytes(b"\xef\xbb\xbf1 3 0 0 0 2 -1\n2 3 0 10 0 1 1\n")

    morphology = dn.Morphology.from_swc(path)

    assert morphology.sections[0].points == (
        (0.0, 0.0, 0.0, 4.0),
        (0.0, 10.0, 0.0, 2.0),
    )


def test_from_swc_parses_structural_integer_ids_without_binary64_rounding(tmp_path):
    root_id = 9_007_199_254_740_992
    child_id = root_id + 1
    path = _write_swc(
        tmp_path,
        f"{root_id} 3 0 0 0 2 -1\n{child_id} 3 0 10 0 1 {root_id}\n",
    )

    morphology = dn.Morphology.from_swc(path)

    assert morphology.sections[0].points == (
        (0.0, 0.0, 0.0, 4.0),
        (0.0, 10.0, 0.0, 2.0),
    )


def test_from_swc_does_not_round_an_unknown_large_parent_to_an_existing_id(
    tmp_path,
):
    root_id = 9_007_199_254_740_992
    missing_parent = root_id + 1
    child_id = root_id + 2
    path = _write_swc(
        tmp_path,
        f"{root_id} 3 0 0 0 2 -1\n{child_id} 3 0 10 0 1 {missing_parent}\n",
    )

    with pytest.raises(ValueError, match=rf"(?i)unknown parent {missing_parent}"):
        dn.Morphology.from_swc(path)


def test_from_swc_sectionizes_maximal_same_type_paths_at_branches_and_type_changes(
    tmp_path,
):
    morphology = dn.Morphology.from_swc(_write_swc(tmp_path, _branched_swc()))

    assert [section.name for section in morphology.sections] == [
        "basal_dendrite_0",
        "basal_dendrite_1",
        "apical_dendrite_0",
    ]
    root, basal, apical = morphology.sections
    assert root.points == (
        (0.0, 0.0, 0.0, 4.0),
        (0.0, 10.0, 0.0, 3.0),
        (0.0, 20.0, 0.0, 2.0),
    )
    assert basal.points == (
        (0.0, 20.0, 0.0, 2.0),
        (-10.0, 30.0, 0.0, 1.5),
        (-20.0, 40.0, 0.0, 1.0),
    )
    assert apical.points == (
        (0.0, 20.0, 0.0, 2.0),
        (10.0, 30.0, 0.0, 1.0),
        (20.0, 40.0, 0.0, 0.5),
    )
    assert {"dendrite", "basal_dendrite", "swc_type_3"} <= basal.labels
    assert {"dendrite", "apical_dendrite", "swc_type_4"} <= apical.labels
    assert morphology.swc_section_types == {
        "basal_dendrite_0": 3,
        "basal_dendrite_1": 3,
        "apical_dendrite_0": 4,
    }

    # Re-export is also a public check that both children attach to the shared
    # branch sample rather than to an arbitrary neighboring location.
    reexported = morphology.to_swc(section_types=morphology.swc_section_types)
    assert _canonical_swc_tree(reexported) == _canonical_swc_tree(_branched_swc())


def test_custom_type_labels_add_semantics_without_losing_raw_type_provenance(
    tmp_path,
):
    path = _write_swc(
        tmp_path,
        "1 42 0 0 0 2 -1\n2 42 0 10 0 1 1\n",
    )

    morphology = dn.Morphology.from_swc(
        path,
        type_labels={42: ("custom_neurite", "excitable")},
    )

    section = morphology.sections[0]
    assert section.name == "type_42_0"
    assert {"custom_neurite", "excitable", "swc_type_42"} <= section.labels
    assert morphology.swc_section_types == {section.name: 42}


def test_type_labels_accept_numpy_integer_ids_consistently_with_swc_export(tmp_path):
    path = _write_swc(tmp_path, "1 42 0 0 0 2 -1\n2 42 0 1 0 1 1\n")

    section = dn.Morphology.from_swc(
        path, type_labels={np.int64(42): "custom_neurite"}
    ).sections[0]

    assert "custom_neurite" in section.labels


def test_string_type_label_is_one_label_rather_than_an_iterable_of_characters(
    tmp_path,
):
    path = _write_swc(tmp_path, "1 42 0 0 0 2 -1\n2 42 0 1 0 1 1\n")

    section = dn.Morphology.from_swc(path, type_labels={42: "custom_neurite"}).sections[
        0
    ]

    assert "custom_neurite" in section.labels
    assert "c" not in section.labels


def test_swc_section_types_is_read_only_and_can_drive_lossless_type_reexport(
    tmp_path,
):
    morphology = dn.Morphology.from_swc(_write_swc(tmp_path, _branched_swc()))

    section_types = morphology.swc_section_types

    assert isinstance(section_types, Mapping)
    with pytest.raises(TypeError):
        section_types["basal_dendrite_0"] = 99
    assert morphology.swc_section_types["basal_dendrite_0"] == 3
    assert _canonical_swc_tree(
        morphology.to_swc(section_types=morphology.swc_section_types)
    ) == _canonical_swc_tree(_branched_swc())


def test_single_point_soma_expands_to_explicit_area_equivalent_cylinder(
    tmp_path,
):
    source = (
        "1 1 2 3 4 5 -1\n"
        "2 3 2 13 4 1 1\n"
        "3 3 2 23 4 0.5 2\n"
        "4 2 12 3 4 0.5 1\n"
        "5 2 22 3 4 0.25 4\n"
    )
    morphology = dn.Morphology.from_swc(_write_swc(tmp_path, source))

    assert [section.name for section in morphology.sections] == [
        "soma_0",
        "basal_dendrite_0",
        "axon_0",
    ]
    soma, dendrite, axon = morphology.sections
    assert soma.points == (
        (-3.0, 3.0, 4.0, 10.0),
        (2.0, 3.0, 4.0, 10.0),
        (7.0, 3.0, 4.0, 10.0),
    )
    assert soma.L == 10.0
    assert soma.nseg == 1
    assert dendrite.points[0] == (2.0, 3.0, 4.0, 10.0)
    assert axon.points[0] == (2.0, 3.0, 4.0, 10.0)
    assert morphology.swc_section_types == {
        "soma_0": 1,
        "basal_dendrite_0": 3,
        "axon_0": 2,
    }

    exported = _swc_nodes(morphology.to_swc(section_types=morphology.swc_section_types))
    center = next(
        node
        for node in exported.values()
        if (node.x, node.y, node.z, node.radius) == (2.0, 3.0, 4.0, 5.0)
    )
    first_dendrite = next(node for node in exported.values() if node.type_id == 3)
    first_axon = next(node for node in exported.values() if node.type_id == 2)
    assert first_dendrite.parent_id == center.node_id
    assert first_axon.parent_id == center.node_id


def test_isolated_single_point_soma_is_still_a_compilable_morphology(tmp_path):
    morphology = dn.Morphology.from_swc(_write_swc(tmp_path, "1 1 4 5 6 3 -1\n"))

    assert morphology.sections[0].points == (
        (1.0, 5.0, 6.0, 6.0),
        (4.0, 5.0, 6.0, 6.0),
        (7.0, 5.0, 6.0, 6.0),
    )
    assert morphology.compile().n_compartments == 1


def test_single_point_soma_error_policy_rejects_implicit_geometry(tmp_path):
    path = _write_swc(
        tmp_path,
        "1 1 0 0 0 5 -1\n2 3 0 10 0 1 1\n",
    )

    with pytest.raises(ValueError, match=r"(?i)single.point|soma|sphere"):
        dn.Morphology.from_swc(path, single_point_soma="error")


def test_non_soma_root_type_discontinuity_is_rejected_instead_of_retyped(tmp_path):
    path = _write_swc(
        tmp_path,
        "1 0 0 0 0 2 -1\n2 3 0 10 0 1 1\n3 3 0 20 0 0.5 2\n",
    )

    with pytest.raises(
        ValueError,
        match=r"(?i)root.*type|positive.length.*preserve|same.type continuation",
    ):
        dn.Morphology.from_swc(path)


def test_to_swc_emits_authored_controls_without_normalized_x_roundoff():
    morphology = dn.Morphology()
    points = (
        (2.0, 4.0, 8.0, 1.4),
        (5.0, 25.0, 22.0, 2.0),
        (11.0, 20.0, 95.0, 1.8),
    )
    morphology.section("cable", points=points)

    nodes = list(_swc_nodes(morphology.to_swc()).values())

    assert [(node.x, node.y, node.z, 2.0 * node.radius) for node in nodes] == list(
        points
    )


def test_to_swc_retains_controls_with_colliding_normalized_positions():
    morphology = dn.Morphology()
    points = (
        (0.0, 0.0, 0.0, 2.0),
        (1e16, 0.0, 0.0, 1.5),
        (1e16, 1.0, 0.0, 1.0),
    )
    morphology.section("ill_scaled", points=points)

    nodes = list(_swc_nodes(morphology.to_swc()).values())

    assert [(node.x, node.y, node.z, 2.0 * node.radius) for node in nodes] == list(
        points
    )
    assert [node.parent_id for node in nodes] == [-1, 1, 2]


def test_to_swc_reversed_child_retains_nonshared_control_at_colliding_endpoint_x():
    morphology = dn.Morphology()
    parent = morphology.section(
        "parent",
        points=((1e16, 1.0, -10.0, 2.0), (1e16, 1.0, 0.0, 2.0)),
    )
    child = morphology.section(
        "child",
        points=(
            (0.0, 0.0, 0.0, 1.0),
            (1e16, 0.0, 0.0, 1.0),
            (1e16, 1.0, 0.0, 1.0),
        ),
    )
    child.connect(parent.at(1.0), child_end=1)

    nodes = list(
        _swc_nodes(morphology.to_swc(section_types={"parent": 3, "child": 4})).values()
    )
    child_nodes = [node for node in nodes if node.type_id == 4]

    assert [(node.x, node.y, node.z) for node in child_nodes] == [
        (1e16, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert child_nodes[0].parent_id == 2
    assert child_nodes[1].parent_id == child_nodes[0].node_id


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", r"(?i)empty|sample|node"),
        ("# comments only\n", r"(?i)empty|sample|node"),
        ("1 3 0 0 0 1\n", r"(?i)seven|column|line"),
        ("1 3 0 0 0 1 -1 trailing\n", r"(?i)seven|column|line"),
        ("one 3 0 0 0 1 -1\n", r"(?i)node|integer|line"),
        ("1 3 0 nan 0 1 -1\n", r"(?i)finite|coordinate|line"),
        ("1 3 0 0 0 inf -1\n", r"(?i)finite|radius|line"),
        ("1.5 3 0 0 0 1 -1\n", r"(?i)node|integer|line"),
        ("1 3.5 0 0 0 1 -1\n", r"(?i)type|integer|line"),
        ("1 3 0 0 0 1 -1.5\n", r"(?i)parent|integer|line"),
        ("1 -3 0 0 0 1 -1\n", r"(?i)type|non.negative|line"),
        ("1 3 0 0 0 0 -1\n", r"(?i)radius|positive|line"),
        ("1 3 0 0 0 -1 -1\n", r"(?i)radius|positive|line"),
        (
            "1 3 0 0 0 1 -1\n1 3 0 1 0 1 1\n",
            r"(?i)duplicate|node|id",
        ),
        (
            "1 3 0 0 0 1 -1\n2 3 0 1 0 1 99\n",
            r"(?i)parent|unknown|missing",
        ),
        (
            "1 3 0 0 0 1 -1\n2 3 0 1 0 1 2\n",
            r"(?i)self|parent|cycle",
        ),
        ("1 3 0 0 0 1 -2\n", r"(?i)parent|-1|root"),
        (
            "1 3 0 0 0 1 -1\n2 3 0 1 0 1 -1\n",
            r"(?i)root|tree|forest",
        ),
        (
            "1 3 0 0 0 1 -1\n2 3 0 1 0 1 3\n3 3 0 2 0 1 2\n",
            r"(?i)cycle|connected|tree",
        ),
        (
            "1 3 0 0 0 1 -1\n2 3 0 0 0 0.5 1\n",
            r"(?i)distinct|zero|length|coordinate",
        ),
    ],
    ids=(
        "empty",
        "comments-only",
        "six-columns",
        "extra-column",
        "nonnumeric-id",
        "nonfinite-coordinate",
        "nonfinite-radius",
        "fractional-id",
        "fractional-type",
        "fractional-parent",
        "negative-type",
        "zero-radius",
        "negative-radius",
        "duplicate-id",
        "missing-parent",
        "self-parent",
        "invalid-root-sentinel",
        "multiple-roots",
        "disconnected-cycle",
        "zero-length-edge",
    ),
)
def test_from_swc_rejects_malformed_or_unrepresentable_inputs(tmp_path, text, match):
    path = _write_swc(tmp_path, text)

    with pytest.raises((TypeError, ValueError), match=match):
        dn.Morphology.from_swc(path)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"nseg": True}, TypeError, r"(?i)nseg|integer"),
        ({"nseg": 0}, ValueError, r"(?i)nseg|positive"),
        ({"rhoa": 0.0}, ValueError, r"(?i)rhoa|positive"),
        ({"cm": float("inf")}, ValueError, r"(?i)cm|finite"),
        (
            {"single_point_soma": "guess"},
            ValueError,
            r"(?i)single.point|sphere|error",
        ),
        ({"type_labels": []}, TypeError, r"(?i)type.labels|mapping"),
    ],
)
def test_from_swc_validates_import_options_before_constructing(
    tmp_path, kwargs, error, match
):
    path = _write_swc(tmp_path, "1 3 0 0 0 1 -1\n2 3 0 1 0 1 1\n")

    with pytest.raises(error, match=match):
        dn.Morphology.from_swc(path, **kwargs)


def test_imported_morphology_compiles_and_constructs_a_tree_population(tmp_path):
    morphology = dn.Morphology.from_swc(
        _write_swc(tmp_path, _branched_swc()),
        rhoa=123.0,
        cm=1.75,
        nseg=2,
    )

    assert all(section.rhoa == 123.0 for section in morphology.sections)
    assert all(section.cm == 1.75 for section in morphology.sections)
    assert all(section.nseg == 2 for section in morphology.sections)
    graph = morphology.compile()
    model = dn.Tree.from_morphology(morphology, N=2)

    assert graph.n_compartments == 7  # six material compartments + one junction
    assert model.compartment_graph == graph
    assert model.shape == (2, graph.n_compartments)
    assert model.dendrite.shape == (2, 6)
    assert model.apical_dendrite.shape == (2, 2)


def test_dendra_export_import_export_preserves_the_canonical_swc_tree(tmp_path):
    source = dn.Morphology()
    trunk = source.section(
        "trunk",
        points=(
            (0.0, 0.0, 0.0, 4.0),
            (0.0, 0.0, 10.0, 3.0),
            (0.0, 0.0, 20.0, 2.0),
        ),
    )
    reversed_child = source.section(
        "reversed_child",
        points=(
            (10.0, 0.0, 10.0, 1.0),
            (0.0, 0.0, 10.0, 2.5),
        ),
    )
    sibling = source.section(
        "sibling",
        points=(
            (0.0, 0.0, 10.0, 2.5),
            (-10.0, 0.0, 10.0, 1.5),
        ),
    )
    reversed_child.connect(trunk.at(0.5), child_end=1)
    sibling.connect(trunk.at(0.5), child_end=0)
    first = source.to_swc(section_types={"trunk": 3, "reversed_child": 4, "sibling": 3})
    path = _write_swc(tmp_path, first, name="dendra-export.swc")

    restored = dn.Morphology.from_swc(path)
    second = restored.to_swc(section_types=restored.swc_section_types)

    assert _canonical_swc_tree(second) == _canonical_swc_tree(first)
    assert restored.compile().n_compartments >= len(restored.sections)
