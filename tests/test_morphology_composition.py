"""Contracts for purely connecting two native Morphology declarations."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import dendra as dn

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
        section.is_pt3d,
    )


def _connection_state(morphology):
    return tuple(
        (
            child_name,
            connection.parent_name,
            connection.parent_x,
            connection.child_name,
            connection.child_end,
        )
        for child_name, connection in morphology._connections.items()
    )


def _morphology_snapshot(morphology):
    return (
        morphology.rhoa,
        morphology.cm,
        morphology.sections,
        tuple(_section_state(section) for section in morphology.sections),
        _connection_state(morphology),
        dict(morphology.swc_section_types),
        morphology.compile(),
    )


def _section_names(morphology):
    return tuple(section.name for section in morphology.sections)


def _branched_pair():
    host = dn.Morphology(rhoa=91.0, cm=0.8)
    host_root = host.section(
        "host_root",
        L=12.0,
        diam=8.0,
        nseg=2,
        rhoa=83.0,
        cm=0.9,
        labels=("cell_body", "shared"),
    )
    stump = host.section(
        "stump",
        points=((0.0, 0.0, 12.0, 3.0), (10.0, 0.0, 12.0, 2.0)),
        nseg=2,
        rhoa=97.0,
        cm=1.1,
        labels=("dendrite", "shared"),
    )
    stump.connect(host_root.at(1.0), child_end=0)

    donor = dn.Morphology(rhoa=130.0, cm=1.4)
    # Declaration order deliberately disagrees with root-first traversal.
    reverse_tip = donor.section(
        "reverse_tip",
        L=7.0,
        diam=0.7,
        nseg=2,
        rhoa=151.0,
        cm=1.6,
        labels=("terminal", "shared"),
    )
    arbor_root = donor.section(
        "arbor_root",
        points=((10.0, 0.0, 12.0, 2.0), (20.0, 0.0, 12.0, 1.2)),
        nseg=3,
        rhoa=121.0,
        cm=1.3,
        labels=("dendrite", "shared"),
    )
    side = donor.section(
        "side",
        points=((15.0, 0.0, 12.0, 1.6), (15.0, 6.0, 12.0, 0.8)),
        nseg=2,
        rhoa=141.0,
        cm=1.5,
        labels=("dendrite", "shared"),
    )
    reverse_tip.connect(arbor_root.at(1.0), child_end=1)
    side.connect(arbor_root.at(0.5), child_end=0)
    return host, stump, donor, arbor_root


def test_connect_morphologies_returns_an_independent_exact_combined_declaration():
    host, stump, donor, arbor_root = _branched_pair()
    host_before = _morphology_snapshot(host)
    donor_before = _morphology_snapshot(donor)

    combined = dn.connect_morphologies(stump.at(1.0), arbor_root.at(0.0))

    assert combined is not host
    assert combined is not donor
    assert (combined.rhoa, combined.cm) == (host.rhoa, host.cm)
    assert _section_names(combined) == (
        "host_root",
        "stump",
        "reverse_tip",
        "arbor_root",
        "side",
    )
    for source in (*host.sections, *donor.sections):
        copied = combined[source.name]
        assert copied is not source
        assert copied._owner is combined
        assert _section_state(copied) == _section_state(source)

    # Internal topology is copied exactly, then one new root attachment is added.
    assert _connection_state(combined) == (
        ("stump", "host_root", 1.0, "stump", 0),
        ("reverse_tip", "arbor_root", 1.0, "reverse_tip", 1),
        ("side", "arbor_root", 0.5, "side", 0),
        ("arbor_root", "stump", 1.0, "arbor_root", 0),
    )
    assert combined.compile() == combined.compile()
    assert (
        combined.compile().metadata.section_name[combined.compile().topology.root]
        == "host_root"
    )

    assert _morphology_snapshot(host) == host_before
    assert _morphology_snapshot(donor) == donor_before

    combined["host_root"].update(rhoa=199.0)
    assert host["host_root"].rhoa == 83.0
    donor["arbor_root"].update(cm=1.9)
    assert combined["arbor_root"].cm == 1.3


def test_child_prefix_renames_only_donor_names_and_their_automatic_labels():
    host, stump, donor, arbor_root = _branched_pair()
    host_before = _morphology_snapshot(host)
    donor_before = _morphology_snapshot(donor)

    combined = dn.connect_morphologies(
        stump.at(1.0),
        arbor_root.at(0.0),
        child_prefix="graft_",
    )

    assert _section_names(combined) == (
        "host_root",
        "stump",
        "graft_reverse_tip",
        "graft_arbor_root",
        "graft_side",
    )
    assert combined["host_root"].labels == host["host_root"].labels
    assert combined["stump"].labels == stump.labels
    for source in donor.sections:
        copied = combined[f"graft_{source.name}"]
        assert copied.labels == (source.labels - {source.name}) | {copied.name}
        assert copied.nseg == source.nseg
        assert copied.L == source.L
        assert copied.diam == source.diam
        assert copied.points == source.points
        assert copied.rhoa == source.rhoa
        assert copied.cm == source.cm
        assert copied.is_pt3d is source.is_pt3d

    assert _connection_state(combined) == (
        ("stump", "host_root", 1.0, "stump", 0),
        (
            "graft_reverse_tip",
            "graft_arbor_root",
            1.0,
            "graft_reverse_tip",
            1,
        ),
        ("graft_side", "graft_arbor_root", 0.5, "graft_side", 0),
        ("graft_arbor_root", "stump", 1.0, "graft_arbor_root", 0),
    )
    assert _morphology_snapshot(host) == host_before
    assert _morphology_snapshot(donor) == donor_before


def test_connect_morphologies_preserves_and_prefixes_swc_type_provenance(tmp_path):
    host_path = tmp_path / "host.swc"
    host_path.write_text(
        "1 2 0 0 0 1 -1\n2 2 5 0 0 0.75 1\n",
        encoding="utf-8",
    )
    donor_path = tmp_path / "donor.swc"
    donor_path.write_text(
        "1 3 5 0 0 0.75 -1\n2 3 10 0 0 0.5 1\n3 4 15 5 0 0.25 2\n",
        encoding="utf-8",
    )
    host = dn.Morphology.from_swc(host_path, rhoa=87.0, cm=0.7, nseg=2)
    donor = dn.Morphology.from_swc(donor_path, rhoa=137.0, cm=1.7, nseg=3)
    host_root = host["axon_0"]
    donor_root = donor["basal_dendrite_0"]
    host_types = dict(host.swc_section_types)
    donor_types = dict(donor.swc_section_types)

    combined = dn.connect_morphologies(
        host_root.at(1.0),
        donor_root.at(0.0),
        child_prefix="donor_",
    )

    assert (combined.rhoa, combined.cm) == (host.rhoa, host.cm)
    assert dict(combined.swc_section_types) == {
        "axon_0": 2,
        "donor_basal_dendrite_0": 3,
        "donor_apical_dendrite_0": 4,
    }
    assert dict(host.swc_section_types) == host_types
    assert dict(donor.swc_section_types) == donor_types

    # Provenance registries are independent along with the Section declarations.
    combined["donor_apical_dendrite_0"].delete()
    assert dict(combined.swc_section_types) == {
        "axon_0": 2,
        "donor_basal_dendrite_0": 3,
    }
    assert dict(donor.swc_section_types) == donor_types


def test_endpoint_bridge_compiles_with_exact_series_resistance_and_orientation():
    host = dn.Morphology(rhoa=80.0)
    parent = host.section("parent", L=10.0, diam=4.0, rhoa=80.0)
    donor = dn.Morphology(rhoa=120.0)
    child = donor.section("child", L=20.0, diam=2.0, nseg=2, rhoa=120.0)

    combined = dn.connect_morphologies(parent.at(1.0), child.at(1.0))
    graph = combined.compile()

    assert graph.topology.parent_index == (-1, 0, 1)
    child_nodes = [
        node
        for node, section_name in enumerate(graph.metadata.section_name)
        if section_name == "child"
    ]
    assert [graph.metadata.segment_index[node] for node in child_nodes] == [1, 0]
    assert [graph.metadata.section_x[node] for node in child_nodes] == pytest.approx(
        [0.75, 0.25]
    )
    expected_inv_area = 5.0 / (4.0 * math.pi) + 5.0 / math.pi
    assert graph.geometry.edge_resistance_ohm[child_nodes[0]] == pytest.approx(
        1e4 * (80.0 * 5.0 / (4.0 * math.pi) + 120.0 * 5.0 / math.pi)
    )
    assert graph.geometry.edge_diff_geom_um[child_nodes[0]] == pytest.approx(
        1.0 / expected_inv_area
    )


def test_interior_parent_location_attaches_to_the_containing_host_compartment():
    host = dn.Morphology(rhoa=100.0)
    trunk = host.section("trunk", L=30.0, diam=2.0, nseg=3)
    donor = dn.Morphology(rhoa=100.0)
    branch = donor.section("branch", L=10.0, diam=2.0)

    graph = dn.connect_morphologies(trunk.at(0.5), branch.at(0.0)).compile()

    branch_node = graph.metadata.section_name.index("branch")
    host_node = graph.topology.parent_index[branch_node]
    assert graph.metadata.section_name[host_node] == "trunk"
    assert graph.metadata.segment_index[host_node] == 1
    assert graph.geometry.edge_length_um[branch_node] == pytest.approx(5.0)
    assert graph.geometry.edge_resistance_ohm[branch_node] == pytest.approx(
        100.0 * 1e4 * 5.0 / math.pi
    )


@pytest.mark.parametrize(
    "case",
    ("duplicate_name", "donor_label_is_host_name", "host_label_is_donor_name"),
)
def test_cross_morphology_name_and_label_collisions_are_transactional(case):
    host = dn.Morphology()
    host_labels = ("donor_root",) if case == "host_label_is_donor_name" else ()
    parent = host.section("host_root", L=10.0, diam=4.0, labels=host_labels)

    donor = dn.Morphology()
    donor_name = "host_root" if case == "duplicate_name" else "donor_root"
    donor_labels = ("host_root",) if case == "donor_label_is_host_name" else ()
    child = donor.section(donor_name, L=8.0, diam=2.0, labels=donor_labels)
    host_before = _morphology_snapshot(host)
    donor_before = _morphology_snapshot(donor)

    with pytest.raises(ValueError, match=r"(?i)name|label|conflict|collision"):
        dn.connect_morphologies(parent.at(1.0), child.at(0.0))

    assert _morphology_snapshot(host) == host_before
    assert _morphology_snapshot(donor) == donor_before


def test_prefix_collisions_are_rejected_without_mutating_either_source():
    host = dn.Morphology()
    parent = host.section("graft_root", L=10.0, diam=4.0)
    donor = dn.Morphology()
    child = donor.section("root", L=8.0, diam=2.0)
    host_before = _morphology_snapshot(host)
    donor_before = _morphology_snapshot(donor)

    with pytest.raises(ValueError, match=r"(?i)name|label|conflict|collision"):
        dn.connect_morphologies(
            parent.at(1.0),
            child.at(0.0),
            child_prefix="graft_",
        )

    assert _morphology_snapshot(host) == host_before
    assert _morphology_snapshot(donor) == donor_before


@pytest.mark.parametrize("child_x", (0.25, 0.5, 0.75))
def test_donor_attachment_must_be_an_endpoint_of_its_root_section(child_x):
    host = dn.Morphology()
    parent = host.section("host", L=10.0, diam=4.0)
    donor = dn.Morphology()
    child = donor.section("donor", L=8.0, diam=2.0)
    host_before = _morphology_snapshot(host)
    donor_before = _morphology_snapshot(donor)

    with pytest.raises(ValueError, match=r"(?i)root|endpoint|0|1"):
        dn.connect_morphologies(parent.at(1.0), child.at(child_x))

    assert _morphology_snapshot(host) == host_before
    assert _morphology_snapshot(donor) == donor_before


def test_donor_attachment_rejects_a_nonroot_section_transactionally():
    host = dn.Morphology()
    parent = host.section("host", L=10.0, diam=4.0)
    donor = dn.Morphology()
    donor_root = donor.section("donor_root", L=8.0, diam=2.0)
    donor_child = donor.section("donor_child", L=6.0, diam=1.0)
    donor_child.connect(donor_root.at(1.0), child_end=0)
    host_before = _morphology_snapshot(host)
    donor_before = _morphology_snapshot(donor)

    with pytest.raises(ValueError, match=r"(?i)root"):
        dn.connect_morphologies(parent.at(1.0), donor_child.at(0.0))

    assert _morphology_snapshot(host) == host_before
    assert _morphology_snapshot(donor) == donor_before


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("parent", "not a location"),
        ("child", "not a location"),
        ("child_prefix", 3),
    ),
)
def test_connect_morphologies_validates_public_argument_types(argument, value):
    host = dn.Morphology()
    parent = host.section("host", L=10.0, diam=4.0)
    donor = dn.Morphology()
    child = donor.section("donor", L=8.0, diam=2.0)
    kwargs = {
        "parent": parent.at(1.0),
        "child": child.at(0.0),
        "child_prefix": "",
    }
    kwargs[argument] = value

    with pytest.raises(TypeError, match=argument):
        dn.connect_morphologies(**kwargs)


def test_connect_morphologies_rejects_noncanonical_location_handles():
    host = dn.Morphology()
    parent = host.section("host", L=10.0, diam=4.0)
    donor = dn.Morphology()
    child = donor.section("donor", L=8.0, diam=2.0)
    forged_parent = dn.SectionLocation(replace(parent), 1.0)
    host_before = _morphology_snapshot(host)
    donor_before = _morphology_snapshot(donor)

    with pytest.raises(ValueError, match=r"(?i)canonical|registered"):
        dn.connect_morphologies(forged_parent, child.at(0.0))

    assert _morphology_snapshot(host) == host_before
    assert _morphology_snapshot(donor) == donor_before


def test_connect_morphologies_is_exported_from_both_public_namespaces():
    import dendra.models as models
    from dendra.models import morphology

    assert dn.connect_morphologies is models.connect_morphologies
    assert "connect_morphologies" in morphology.__all__
