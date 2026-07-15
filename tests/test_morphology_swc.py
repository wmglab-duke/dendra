"""Contract tests for deterministic native-Morphology SWC export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import dendra as dn

pytestmark = pytest.mark.cpu


@dataclass(frozen=True)
class _SwcRow:
    node_id: int
    type_id: int
    x: float
    y: float
    z: float
    radius: float
    parent_id: int


def _swc_rows(text: str) -> list[_SwcRow]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        assert len(fields) == 7
        rows.append(
            _SwcRow(
                node_id=int(fields[0]),
                type_id=int(fields[1]),
                x=float(fields[2]),
                y=float(fields[3]),
                z=float(fields[4]),
                radius=float(fields[5]),
                parent_id=int(fields[6]),
            )
        )
    return rows


def _two_section_morphology() -> dn.Morphology:
    morphology = dn.Morphology()
    root = morphology.section(
        "root",
        points=((0.0, 0.0, 0.0, 8.0), (0.0, 0.0, 10.0, 6.0)),
    )
    child = morphology.section(
        "child",
        points=((0.0, 0.0, 10.0, 2.0), (5.0, 0.0, 10.0, 1.0)),
    )
    child.connect(root.at(1.0), child_end=0)
    return morphology


def test_to_swc_serializes_authored_samples_radii_and_not_nseg():
    morphology = dn.Morphology()
    morphology.section(
        "axon",
        points=(
            (-1.0, 2.0, 3.0, 10.0),
            (1.0, 2.0, 3.0, 6.0),
            (1.0, 2.0, 7.0, 2.0),
        ),
        nseg=17,
    )

    text = morphology.to_swc(section_types={"axon": 2})

    assert text.endswith("\n")
    assert any(line.startswith("#") for line in text.splitlines())
    assert _swc_rows(text) == [
        _SwcRow(1, 2, -1.0, 2.0, 3.0, 5.0, -1),
        _SwcRow(2, 2, 1.0, 2.0, 3.0, 3.0, 1),
        _SwcRow(3, 2, 1.0, 2.0, 7.0, 1.0, 2),
    ]


def test_to_swc_uses_positional_numbers_and_canonical_zero():
    morphology = dn.Morphology()
    morphology.section(
        "tiny",
        points=((-0.0, 1e-12, 0.0, 2.0), (1.0, 1e-12, 0.0, 2.0)),
    )

    data_lines = [
        line for line in morphology.to_swc().splitlines() if not line.startswith("#")
    ]

    assert data_lines[0] == "1 0 0 0.000000000001 0 1 -1"
    assert all(
        "e" not in field.lower() for line in data_lines for field in line.split()
    )


def test_to_swc_defaults_to_undefined_type_and_accepts_custom_type_ids():
    morphology = _two_section_morphology()

    rows = _swc_rows(morphology.to_swc(section_types={"child": np.int64(42)}))

    # The shared endpoint is represented by the parent's SWC node. The child
    # type therefore begins at its first non-shared authored sample.
    assert [row.type_id for row in rows] == [0, 0, 42]
    assert rows[-1].parent_id == 2


def test_to_swc_is_parent_first_deterministic_and_preserves_orientation():
    morphology = dn.Morphology()
    # Declaration order deliberately disagrees with both root-first traversal
    # and the order in which the connections are installed.
    forward = morphology.section(
        "forward",
        points=((0.0, 0.0, 5.0, 2.0), (5.0, 0.0, 5.0, 1.0)),
    )
    reverse = morphology.section(
        "reverse",
        points=((0.0, 5.0, 15.0, 1.0), (0.0, 0.0, 15.0, 3.0)),
    )
    shared = morphology.section(
        "shared",
        points=((0.0, 0.0, 5.0, 4.0), (-5.0, 0.0, 5.0, 4.0)),
    )
    root = morphology.section(
        "root",
        points=(
            (0.0, 0.0, 0.0, 8.0),
            (0.0, 0.0, 10.0, 6.0),
            (0.0, 0.0, 20.0, 4.0),
        ),
    )
    reverse.connect(root.at(0.75), child_end=1)
    shared.connect(root.at(0.25), child_end=0)
    forward.connect(root.at(0.25), child_end=0)

    first = morphology.to_swc(
        section_types={"root": 1, "reverse": 2, "forward": 3, "shared": 4}
    )
    second = morphology.to_swc(
        section_types={"root": 1, "reverse": 2, "forward": 3, "shared": 4}
    )

    assert first == second
    assert _swc_rows(first) == [
        # Both interior connection positions are inserted into the root chain.
        _SwcRow(1, 1, 0.0, 0.0, 0.0, 4.0, -1),
        _SwcRow(2, 1, 0.0, 0.0, 5.0, 3.5, 1),
        _SwcRow(3, 1, 0.0, 0.0, 10.0, 3.0, 2),
        _SwcRow(4, 1, 0.0, 0.0, 15.0, 2.5, 3),
        _SwcRow(5, 1, 0.0, 0.0, 20.0, 2.0, 4),
        # Siblings use Section declaration order, not connection-call order.
        # Their connected endpoints reuse the existing root node and are not
        # emitted as zero-length duplicate edges.
        _SwcRow(6, 3, 5.0, 0.0, 5.0, 0.5, 2),
        # child_end=1 reverses traversal away from the shared endpoint.
        _SwcRow(7, 2, 0.0, 5.0, 15.0, 0.5, 4),
        _SwcRow(8, 4, -5.0, 0.0, 5.0, 2.0, 2),
    ]


def test_to_swc_emits_one_root_and_parent_before_child():
    morphology = _two_section_morphology()

    rows = _swc_rows(morphology.to_swc())

    assert [row.node_id for row in rows] == list(range(1, len(rows) + 1))
    assert sum(row.parent_id == -1 for row in rows) == 1
    assert all(row.parent_id == -1 or row.parent_id < row.node_id for row in rows)


def test_write_swc_accepts_pathlike_and_matches_in_memory_serialization(tmp_path):
    morphology = _two_section_morphology()
    path = tmp_path / "nested" / "cell.swc"
    path.parent.mkdir()

    result = morphology.write_swc(path, section_types={"root": 1, "child": 3})

    assert result is None
    assert path.read_text(encoding="utf-8") == morphology.to_swc(
        section_types={"root": 1, "child": 3}
    )


def test_write_swc_validates_spatial_connections_before_replacing_file(tmp_path):
    morphology = dn.Morphology()
    root = morphology.section(
        "root",
        points=((0.0, 0.0, 0.0, 4.0), (0.0, 0.0, 10.0, 4.0)),
    )
    child = morphology.section(
        "child",
        points=((0.0, 0.0, 11.0, 2.0), (5.0, 0.0, 11.0, 2.0)),
    )
    child.connect(root.at(1.0), child_end=0)
    path = tmp_path / "existing.swc"
    path.write_text("do not truncate\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"(?i)spatial|coordinate|coincid"):
        morphology.write_swc(path)

    assert path.read_text(encoding="utf-8") == "do not truncate\n"


def test_swc_export_rejects_native_diameter_steps_without_replacing_file(tmp_path):
    morphology = dn.Morphology()
    morphology.section(
        "dend",
        points=(
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 5.0, 2.0),
            (0.0, 0.0, 5.0, 4.0),
            (0.0, 0.0, 10.0, 4.0),
        ),
    )
    path = tmp_path / "existing.swc"
    path.write_text("do not truncate\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"(?i)SWC.*zero-length diameter discontinu|annular membrane",
    ):
        morphology.write_swc(path)

    assert path.read_text(encoding="utf-8") == "do not truncate\n"


def test_connection_tolerance_only_accommodates_small_coordinate_roundoff():
    morphology = dn.Morphology()
    root = morphology.section(
        "root",
        points=((0.0, 0.0, 0.0, 4.0), (0.0, 0.0, 10.0, 4.0)),
    )
    child = morphology.section(
        "child",
        points=((5e-8, 0.0, 10.0, 2.0), (5.0, 0.0, 10.0, 2.0)),
    )
    child.connect(root.at(1.0), child_end=0)

    with pytest.raises(ValueError, match=r"(?i)spatial|coordinate|coincid"):
        morphology.to_swc()

    rows = _swc_rows(morphology.to_swc(connection_tolerance_um=1e-7))
    assert rows[-1].parent_id == 2


@pytest.mark.parametrize(
    ("section_types", "error", "match"),
    [
        ([("root", 1)], TypeError, "section_types"),
        ({1: 1}, TypeError, r"(?i)section.*name|string"),
        ({"missing": 1}, ValueError, r"(?i)unknown|missing"),
        ({"root": True}, TypeError, r"(?i)type"),
        ({"root": 1.5}, TypeError, r"(?i)type"),
        ({"root": -1}, ValueError, r"(?i)non-negative|negative"),
    ],
)
def test_section_type_mapping_is_exact_and_validated(section_types, error, match):
    morphology = _two_section_morphology()

    with pytest.raises(error, match=match):
        morphology.to_swc(section_types=section_types)


@pytest.mark.parametrize(
    ("default_type", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (-1, ValueError),
    ],
)
def test_default_type_is_a_nonnegative_integer(default_type, error):
    morphology = _two_section_morphology()

    with pytest.raises(error, match=r"(?i)default.*type|type.*default"):
        morphology.to_swc(default_type=default_type)


@pytest.mark.parametrize(
    ("tolerance", "error"),
    [(True, TypeError), (-1.0, ValueError), (float("inf"), ValueError)],
)
def test_connection_tolerance_must_be_finite_and_nonnegative(tolerance, error):
    morphology = _two_section_morphology()

    with pytest.raises(error, match=r"(?i)tolerance|finite|non-negative"):
        morphology.to_swc(connection_tolerance_um=tolerance)
