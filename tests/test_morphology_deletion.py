"""Contracts for deleting native Morphology Sections and subtrees."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import dendra as dn

pytestmark = pytest.mark.cpu


def _branched_morphology():
    """Return a two-branch tree whose declarations and topology interleave."""
    morphology = dn.Morphology()
    soma = morphology.section("soma", L=12.0, diam=8.0, labels="cell_body")
    dend = morphology.section("dend", L=20.0, diam=2.0, nseg=2, labels="dendrite")
    axon = morphology.section("axon", L=30.0, diam=1.0, nseg=3, labels="axon")
    tuft = morphology.section("tuft", L=15.0, diam=1.5, nseg=2, labels="dendrite")
    node = morphology.section("node", L=10.0, diam=0.8, labels="axonal")

    dend.connect(soma.at(0.75), child_end=0)
    axon.connect(soma.at(0.25), child_end=0)
    tuft.connect(dend.at(1.0), child_end=0)
    node.connect(axon.at(1.0), child_end=0)
    return morphology, {
        "soma": soma,
        "dend": dend,
        "axon": axon,
        "tuft": tuft,
        "node": node,
    }


def _section_names(morphology):
    return tuple(section.name for section in morphology.sections)


def test_delete_leaf_by_handle_and_name_preserves_the_remaining_tree():
    morphology, sections = _branched_morphology()

    assert sections["tuft"].delete() == ("tuft",)
    assert _section_names(morphology) == ("soma", "dend", "axon", "node")

    graph = morphology.compile()
    assert graph.nodes_with_label("tuft") == ()
    assert graph.nodes_with_label("dend")
    assert graph.nodes_with_label("axon")

    assert morphology.delete_section("node") == ("node",)
    assert _section_names(morphology) == ("soma", "dend", "axon")
    assert morphology.compile().nodes_with_label("node") == ()


def test_nonrecursive_branch_deletion_is_rejected_atomically():
    morphology, sections = _branched_morphology()
    before_sections = morphology.sections
    before_graph = morphology.compile()

    with pytest.raises(ValueError, match=r"(?i)child|descendant|recursive"):
        morphology.delete_section(sections["axon"])

    assert morphology.sections == before_sections
    assert all(
        current is original
        for current, original in zip(morphology.sections, before_sections)
    )
    assert morphology.compile() == before_graph


def test_recursive_deletion_removes_one_subtree_in_declaration_order():
    morphology, sections = _branched_morphology()

    assert sections["axon"].delete(recursive=True) == ("axon", "node")
    assert _section_names(morphology) == ("soma", "dend", "tuft")
    assert morphology.sections == (
        sections["soma"],
        sections["dend"],
        sections["tuft"],
    )

    graph = morphology.compile()
    assert graph.nodes_with_label("axon") == ()
    assert graph.nodes_with_label("node") == ()
    assert graph.nodes_with_label("dendrite")


def test_recursive_return_order_is_authored_order_not_topological_order():
    morphology = dn.Morphology()
    child = morphology.section("child", L=4.0, diam=1.0)
    parent = morphology.section("parent", L=8.0, diam=2.0)
    child.connect(parent.at(1.0), child_end=0)

    assert parent.delete(recursive=True) == ("child", "parent")
    assert morphology.sections == ()


def test_deleting_the_root_requires_recursion_and_can_empty_the_morphology():
    morphology, sections = _branched_morphology()
    before_graph = morphology.compile()

    with pytest.raises(ValueError, match=r"(?i)child|descendant|recursive"):
        sections["soma"].delete()
    assert morphology.compile() == before_graph

    assert morphology.delete_section("soma", recursive=True) == (
        "soma",
        "dend",
        "axon",
        "tuft",
        "node",
    )
    assert morphology.sections == ()
    with pytest.raises(ValueError, match=r"(?i)empty"):
        morphology.compile()

    singleton = dn.Morphology()
    root = singleton.section("root", L=5.0, diam=2.0)
    assert root.delete() == ("root",)
    assert singleton.sections == ()


def test_deleted_name_can_be_reused_without_revalidating_the_stale_handle():
    morphology = dn.Morphology()
    old = morphology.section("axon", L=20.0, diam=1.0, labels="axon")
    old_location = old.at(0.5)
    old_endpoint = old.at(0.0)

    assert old.delete() == ("axon",)
    replacement = morphology.section("axon", L=40.0, diam=2.0, nseg=4, labels="axon")

    assert morphology.sections == (replacement,)
    assert replacement is not old
    assert morphology.compile().n_compartments == 4

    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        old.update(L=25.0)
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        old.at(0.25)
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        old.delete()
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        morphology.delete_section(old)
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        old_location.update(diam=3.0)
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        morphology.connect(old_location, replacement, child_end=0)
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        morphology.connect(replacement.at(1.0), old_endpoint)

    assert morphology.sections == (replacement,)
    assert replacement.L == 40.0


@pytest.mark.parametrize("recursive", [1, 0, "yes", None])
def test_invalid_recursive_values_are_rejected_transactionally(recursive):
    morphology, _ = _branched_morphology()
    before_sections = morphology.sections
    before_graph = morphology.compile()

    with pytest.raises(TypeError, match=r"(?i)recursive|boolean"):
        morphology.delete_section("axon", recursive=recursive)

    assert morphology.sections == before_sections
    assert morphology.compile() == before_graph


def test_unknown_foreign_and_invalid_section_references_do_not_mutate():
    morphology, sections = _branched_morphology()
    other = dn.Morphology()
    foreign = other.section("foreign", L=5.0, diam=1.0)
    forged = replace(sections["axon"])
    before_sections = morphology.sections
    before_graph = morphology.compile()

    with pytest.raises(KeyError, match=r"(?i)unknown|missing"):
        morphology.delete_section("missing")
    with pytest.raises(ValueError, match=r"(?i)different|morphology"):
        morphology.delete_section(foreign)
    with pytest.raises(ValueError, match=r"(?i)canonical|registered"):
        morphology.delete_section(forged)
    with pytest.raises(TypeError, match=r"(?i)section|name"):
        morphology.delete_section(3)

    assert morphology.sections == before_sections
    assert morphology.compile() == before_graph
    assert other.sections == (foreign,)


def test_deletion_is_available_while_authoring_an_incomplete_forest():
    morphology = dn.Morphology()
    first_root = morphology.section("first_root", L=10.0, diam=3.0)
    surviving_root = morphology.section("surviving_root", L=8.0, diam=2.0)
    child = morphology.section("child", L=4.0, diam=1.0)
    child.connect(first_root.at(1.0), child_end=0)

    with pytest.raises(ValueError, match=r"(?i)exactly one root"):
        morphology.compile()

    assert first_root.delete(recursive=True) == ("first_root", "child")
    assert morphology.sections == (surviving_root,)
    assert morphology.compile().metadata.section_name == ("surviving_root",)


@pytest.mark.parametrize(
    "corruption", ("wrong_type", "key_mismatch", "unknown_parent", "cycle")
)
def test_private_connection_corruption_is_rejected_without_partial_deletion(
    corruption,
):
    morphology, _ = _branched_morphology()
    if corruption == "wrong_type":
        morphology._connections["axon"] = object()
    elif corruption == "key_mismatch":
        morphology._connections["not_axon"] = morphology._connections.pop("axon")
    elif corruption == "unknown_parent":
        morphology._connections["axon"] = replace(
            morphology._connections["axon"], parent_name="missing"
        )
    else:
        morphology._connections["soma"] = replace(
            morphology._connections["axon"],
            parent_name="node",
            child_name="soma",
        )

    before_sections = morphology.sections
    before_connections = dict(morphology._connections)

    with pytest.raises(RuntimeError, match=r"(?i)registry|cycle|deletion"):
        morphology.delete_section("tuft")

    assert morphology.sections == before_sections
    assert morphology._connections == before_connections


def test_compiled_graph_and_constructed_tree_are_independent_snapshots():
    morphology, _ = _branched_morphology()
    graph_before = morphology.compile()
    tree = dn.Tree.from_morphology(morphology)
    area_before = tree.area.clone()
    length_before = tree.dx.clone()
    model_sections = tree.compartment_graph.metadata.section_name

    assert morphology.delete_section("axon", recursive=True) == ("axon", "node")
    graph_after = morphology.compile()

    assert graph_before != graph_after
    assert tree.compartment_graph == graph_before
    assert tree.compartment_graph.metadata.section_name == model_sections
    assert "axon" in model_sections
    assert "node" in model_sections
    torch.testing.assert_close(tree.area, area_before)
    torch.testing.assert_close(tree.dx, length_before)
    assert graph_before.nodes_with_label("axon")
    assert graph_after.nodes_with_label("axon") == ()


def test_delete_and_replace_axon_workflow_builds_the_replacement_geometry():
    morphology = dn.Morphology()
    soma = morphology.section("soma", L=20.0, diam=10.0, labels="cell_body")
    dend = morphology.section("dend", L=50.0, diam=2.0, nseg=2, labels="dendrite")
    old_axon = morphology.section("axon", L=100.0, diam=1.0, nseg=2, labels="axon")
    dend.connect(soma.at(1.0), child_end=0)
    old_axon.connect(soma.at(0.0), child_end=0)
    old_graph = morphology.compile()

    assert morphology.delete_section(old_axon) == ("axon",)
    new_axon = morphology.section(
        "axon",
        points=((0.0, 0.0, 0.0, 2.0), (90.0, 0.0, 0.0, 0.5)),
        nseg=5,
        rhoa=80.0,
        labels=("axon", "replacement"),
    )
    new_axon.connect(soma.at(0.0), child_end=0)

    graph = morphology.compile()
    axon_nodes = graph.nodes_with_label("axon")
    assert _section_names(morphology) == ("soma", "dend", "axon")
    assert len(axon_nodes) == 5
    assert graph.nodes_with_label("replacement") == axon_nodes
    assert (
        tuple(graph.metadata.section_name[node] for node in axon_nodes) == ("axon",) * 5
    )
    assert tuple(graph.geometry.rhoa_ohm_cm[node] for node in axon_nodes) == (80.0,) * 5
    assert graph != old_graph

    tree = dn.Tree.from_morphology(morphology)
    assert tree.compartment_graph == graph
    assert tree.compartment_graph.nodes_with_label("replacement") == axon_nodes
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        old_axon.update(diam=3.0)


def test_deletion_prunes_swc_type_provenance_without_restoring_it_on_reuse(
    tmp_path,
):
    swc = tmp_path / "soma_axon.swc"
    swc.write_text(
        "1 1 0 0 0 2 -1\n2 1 4 0 0 2 1\n3 2 8 0 0 0.5 2\n4 2 12 0 0 0.5 3\n",
        encoding="utf-8",
    )
    morphology = dn.Morphology.from_swc(swc)
    axon = next(section for section in morphology.sections if section.name == "axon_0")
    live_provenance = morphology.swc_section_types

    assert dict(live_provenance) == {"soma_0": 1, "axon_0": 2}
    assert axon.delete() == ("axon_0",)
    assert dict(live_provenance) == {"soma_0": 1}

    replacement = morphology.section("axon_0", L=10.0, diam=1.0, labels="axon")
    assert replacement.name not in live_provenance
