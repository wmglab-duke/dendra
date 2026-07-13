"""Topology contracts for the public NEURON section-path helpers."""

from __future__ import annotations

import pytest
from neuron import h

from dendra.utils.neuron import path_sections, path_via

pytestmark = pytest.mark.neuron

_SECTION_COUNTER = 0


@pytest.fixture(autouse=True)
def _delete_sections_created_by_test():
    """Keep NEURON's process-global section namespace isolated."""
    sections_before_test = set(h.allsec())
    yield
    for section in list(h.allsec()):
        if section not in sections_before_test:
            h.delete_section(sec=section)


def _section(prefix: str):
    global _SECTION_COUNTER
    _SECTION_COUNTER += 1
    return h.Section(name=f"{prefix}_{_SECTION_COUNTER}")


@pytest.fixture
def branched_tree():
    root = _section("path_root")
    trunk = _section("path_trunk")
    left = _section("path_left")
    left_tip = _section("path_left_tip")
    right = _section("path_right")
    right_tip = _section("path_right_tip")

    trunk.connect(root(1.0), 0.0)
    left.connect(trunk(1.0), 0.0)
    left_tip.connect(left(1.0), 0.0)
    right.connect(trunk(1.0), 0.0)
    right_tip.connect(right(1.0), 0.0)

    return {
        "root": root,
        "trunk": trunk,
        "left": left,
        "left_tip": left_tip,
        "right": right,
        "right_tip": right_tip,
    }


def test_path_sections_orders_connected_ancestor_and_descendant(branched_tree):
    root = branched_tree["root"]
    left = branched_tree["left"]
    left_tip = branched_tree["left_tip"]

    assert path_sections(root, left_tip) == [
        root,
        branched_tree["trunk"],
        left,
        left_tip,
    ]
    assert path_sections(left_tip, root) == [
        left_tip,
        left,
        branched_tree["trunk"],
        root,
    ]
    assert path_sections(left, left) == [left]


def test_path_sections_crosses_lowest_common_ancestor_once(branched_tree):
    left = branched_tree["left"]
    left_tip = branched_tree["left_tip"]
    trunk = branched_tree["trunk"]
    right = branched_tree["right"]
    right_tip = branched_tree["right_tip"]

    assert path_sections(left_tip, right_tip) == [
        left_tip,
        left,
        trunk,
        right,
        right_tip,
    ]
    assert path_sections(right_tip, left_tip) == [
        right_tip,
        right,
        trunk,
        left,
        left_tip,
    ]


def test_path_via_accepts_only_sections_on_the_unique_simple_path(branched_tree):
    left_tip = branched_tree["left_tip"]
    trunk = branched_tree["trunk"]
    right = branched_tree["right"]
    right_tip = branched_tree["right_tip"]

    expected = [left_tip, branched_tree["left"], trunk, right, right_tip]
    assert path_via(left_tip, right_tip, trunk) == expected
    assert path_via(left_tip, right_tip, left_tip) == expected
    assert path_via(left_tip, right_tip, branched_tree["root"]) is None

    # NEURON permits duplicate section names; membership is by identity.
    disconnected_trunk = h.Section(name=trunk.name())
    assert path_via(left_tip, right_tip, disconnected_trunk) is None


def test_disconnected_trees_have_no_path_even_with_a_matching_via_section(
    branched_tree,
):
    # Matching names across trees must not be mistaken for a common ancestor.
    isolated_root = h.Section(name=branched_tree["root"].name())
    isolated_child = _section("isolated_child")
    isolated_child.connect(isolated_root(1.0), 0.0)

    left_tip = branched_tree["left_tip"]
    assert path_sections(left_tip, isolated_child) is None
    assert path_via(left_tip, isolated_child, left_tip) is None
    assert path_via(left_tip, isolated_child, isolated_root) is None


def test_path_via_can_return_an_ordered_neuron_section_list(branched_tree):
    left_tip = branched_tree["left_tip"]
    trunk = branched_tree["trunk"]
    right_tip = branched_tree["right_tip"]

    section_list = path_via(
        left_tip,
        right_tip,
        trunk,
        return_sectionlist=True,
    )

    assert section_list is not None
    assert not isinstance(section_list, list)
    assert list(section_list) == [
        left_tip,
        branched_tree["left"],
        trunk,
        branched_tree["right"],
        right_tip,
    ]
    assert (
        path_via(
            left_tip,
            right_tip,
            branched_tree["root"],
            return_sectionlist=True,
        )
        is None
    )
