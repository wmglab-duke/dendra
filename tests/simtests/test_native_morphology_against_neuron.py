"""NEURON oracles for Dendra's native Section compiler."""

from __future__ import annotations

import itertools
import math

import networkx as nx
import pytest
from neuron import h

import dendra as dn
from dendra.models.io import neuron_to_dendra_graph

pytestmark = pytest.mark.neuron

_SECTION_IDS = itertools.count()


def _neuron_section(name, *, points, nseg, rhoa, cm):
    section = h.Section(name=f"native_{name}_{next(_SECTION_IDS)}")
    section.pt3dclear()
    for point in points:
        section.pt3dadd(*point)
    section.nseg = int(nseg)
    section.Ra = float(rhoa)
    section.cm = float(cm)
    return section


def _close(left, right, *, rtol=3e-7, atol=3e-9):
    # NEURON stores pt3d diameters with float32 precision even when the solver
    # uses doubles. The native compiler deliberately retains source binary64.
    return math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)


def _graphs_match_physics(
    native: nx.DiGraph, neuron: nx.DiGraph, *, section_names
) -> bool:
    node_fields = ("L", "diam", "Ra", "cm", "area", "volume", "x", "y", "z")
    edge_fields = ("L", "R_ohm", "diff_geom_um")

    def region(attributes):
        if attributes["L"] == attributes["area"] == attributes["volume"] == 0.0:
            return "__junction__"
        name = str(attributes.get("section_name", attributes.get("name", "")))
        return next(
            (section_name for section_name in section_names if section_name in name),
            None,
        )

    def node_match(left, right):
        if region(left) != region(right):
            return False
        if region(left) == "__junction__":
            # Diameter, coordinate, Ra, and cm are non-physical placeholders
            # on a zero-area algebraic junction.
            return True
        return all(_close(left[name], right[name]) for name in node_fields)

    def edge_match(left, right):
        return all(_close(left[name], right[name]) for name in edge_fields)

    return nx.is_isomorphic(
        native,
        neuron,
        node_match=node_match,
        edge_match=edge_match,
    )


def test_native_tapered_branch_and_reversed_child_match_neuron_compartment_graph():
    rhoa = 117.0
    cm = 0.83
    specifications = {
        "soma": ([(0, 0, 0, 12), (0, 0, 30, 10)], 3),
        "trunk": ([(0, 0, 30, 5), (0, 60, 30, 3)], 3),
        "tuft_a": ([(0, 60, 30, 3), (40, 105, 35, 1.2)], 3),
        # Authored from the distal end toward the physical branch junction;
        # child_end=1 exercises reversed Section orientation.
        "tuft_b": ([(-35, 100, 20, 1.0), (0, 60, 30, 2.5)], 2),
    }

    morphology = dn.Morphology(rhoa=rhoa, cm=cm)
    native_sections = {
        name: morphology.section(name, points=points, nseg=nseg)
        for name, (points, nseg) in specifications.items()
    }
    native_sections["trunk"].connect(native_sections["soma"].at(1.0), child_end=0)
    native_sections["tuft_a"].connect(native_sections["trunk"].at(1.0), child_end=0)
    native_sections["tuft_b"].connect(native_sections["trunk"].at(1.0), child_end=1)

    neuron_sections = {
        name: _neuron_section(
            name,
            # Native Section points always define authored x=0 → 1.  NEURON
            # maps logical x opposite its stored pt3d order when child_end=1,
            # so reverse this one point sequence to describe the same physical
            # cable and connection in the two authoring APIs.
            points=(list(reversed(points)) if name == "tuft_b" else points),
            nseg=nseg,
            rhoa=rhoa,
            cm=cm,
        )
        for name, (points, nseg) in specifications.items()
    }
    neuron_sections["trunk"].connect(neuron_sections["soma"](1.0), 0.0)
    neuron_sections["tuft_a"].connect(neuron_sections["trunk"](1.0), 0.0)
    neuron_sections["tuft_b"].connect(neuron_sections["trunk"](1.0), 1.0)

    native_graph = morphology.compile().to_networkx()
    neuron_graph, _ = neuron_to_dendra_graph(neuron_sections["soma"])

    assert _graphs_match_physics(
        native_graph, neuron_graph, section_names=specifications
    )


@pytest.mark.parametrize("parent_x", [0.5, 2.0 / 3.0])
def test_native_interior_attachment_matches_neuron_boundary_selection(parent_x):
    rhoa = 91.0
    cm = 1.2
    trunk_points = [(0, 0, 0, 4), (0, 0, 36, 2)]
    branch_points = [(0, 0, 18, 1.5), (18, 0, 18, 1.0)]

    morphology = dn.Morphology(rhoa=rhoa, cm=cm)
    trunk = morphology.section("trunk", points=trunk_points, nseg=3)
    branch = morphology.section("branch", points=branch_points, nseg=3)
    branch.connect(trunk.at(parent_x), child_end=0)

    nrn_trunk = _neuron_section(
        "interior_trunk", points=trunk_points, nseg=3, rhoa=rhoa, cm=cm
    )
    nrn_branch = _neuron_section(
        "interior_branch", points=branch_points, nseg=3, rhoa=rhoa, cm=cm
    )
    nrn_branch.connect(nrn_trunk(parent_x), 0.0)

    native_graph = morphology.compile().to_networkx()
    neuron_graph, _ = neuron_to_dendra_graph(nrn_trunk)

    assert _graphs_match_physics(
        native_graph,
        neuron_graph,
        section_names=("trunk", "branch"),
    )
