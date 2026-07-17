"""Required, deterministic contracts for NEURON-backed morphology import."""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pytest
from neuron import h

from dendra.models.io import (
    apply_d_lambda,
    edge_diff_geom_um,
    edge_inv_area_integral_um_inv,
    lambda_f,
    neuron_to_dendra_graph,
    r_ohm,
    read_neurolucida,
    read_swc,
    segment_volume_um3,
    xyz,
)
from dendra.models.morphology import Morphology

pytestmark = pytest.mark.neuron

_COUNTER = 0


@pytest.fixture(autouse=True)
def _delete_sections_created_by_test():
    """Keep HOC's process-global section namespace isolated across tests."""
    before = set(h.allsec())
    yield
    for section in list(h.allsec()):
        if section not in before:
            h.delete_section(sec=section)


def _section(prefix, *, L=100.0, diam=2.0, Ra=100.0, cm=1.0, nseg=1):
    global _COUNTER
    _COUNTER += 1
    section = h.Section(name=f"{prefix}_{_COUNTER}")
    section.L = L
    section.diam = diam
    section.Ra = Ra
    section.cm = cm
    section.nseg = nseg
    return section


def _assert_physical_tree(graph):
    assert nx.is_arborescence(graph)
    assert graph.number_of_edges() == graph.number_of_nodes() - 1
    for _, _, attrs in graph.edges(data=True):
        assert math.isfinite(float(attrs["R_ohm"]))
        assert float(attrs["R_ohm"]) > 0.0
        assert math.isfinite(float(attrs["diff_geom_um"]))
        assert float(attrs["diff_geom_um"]) > 0.0


def test_lambda_rule_validates_inputs_and_selects_an_odd_compartment_grid():
    section = _section("lambda", L=100.0, diam=2.0, Ra=100.0, cm=1.0)
    expected_lambda = 1.0e5 * math.sqrt(2.0 / (4.0 * math.pi * 100 * 100 * 1))

    actual_lambda = lambda_f(section, 100.0)
    assert actual_lambda == pytest.approx(expected_lambda, rel=1e-12)

    apply_d_lambda([section], d_lambda=0.1, freq=100.0)
    expected_nseg = int((section.L / (0.1 * actual_lambda) + 0.9) / 2) * 2 + 1
    assert int(section.nseg) == max(1, expected_nseg)
    assert int(section.nseg) % 2 == 1

    for invalid in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="freq_hz must be positive and finite"):
            lambda_f(section, invalid)
        with pytest.raises(ValueError, match="d_lambda must be positive and finite"):
            apply_d_lambda([section], d_lambda=invalid, freq=100.0)
        with pytest.raises(ValueError, match="freq must be positive and finite"):
            apply_d_lambda([section], d_lambda=0.1, freq=invalid)

    with pytest.raises(TypeError, match="freq_hz.*boolean"):
        lambda_f(section, True)
    with pytest.raises(TypeError, match="d_lambda.*boolean"):
        apply_d_lambda([section], d_lambda=True)


def test_pt3d_coordinates_extracellular_layers_volume_and_diffusion_geometry():
    section = _section("taper", nseg=2)
    h.pt3dclear(sec=section)
    h.pt3dadd(0.0, 0.0, 0.0, 4.0, sec=section)
    h.pt3dadd(10.0, 0.0, 0.0, 3.0, sec=section)
    h.pt3dadd(10.0, 30.0, 0.0, 2.0, sec=section)

    assert xyz(section(0.25)) == pytest.approx({"x": 10.0, "y": 0.0, "z": 0.0})
    assert xyz(section(0.625)) == pytest.approx({"x": 10.0, "y": 15.0, "z": 0.0})

    section.insert("extracellular")
    segment = section(0.25)
    for layer in range(2):
        segment.xraxial[layer] = 1.0 + layer
        segment.xc[layer] = 2.0 + layer
        segment.xg[layer] = 3.0 + layer
    coordinates = xyz(segment, extcell=2)
    assert coordinates["xraxial"] == pytest.approx([1.0, 2.0])
    assert coordinates["xc"] == pytest.approx([2.0, 3.0])
    assert coordinates["xg"] == pytest.approx([3.0, 4.0])

    def frustum_volume(length, d0, d1):
        return math.pi * length * (d0 * d0 + d0 * d1 + d1 * d1) / 12.0

    expected_volume = frustum_volume(10.0, 4.0, 3.0) + frustum_volume(30.0, 3.0, 2.0)
    actual_volume = sum(segment_volume_um3(seg) for seg in section)
    assert actual_volume == pytest.approx(expected_volume, rel=1e-12)

    left, right = section(0.25), section(0.75)
    diameter_right = 3.0 + (2.0 - 3.0) * (20.0 / 30.0)
    inverse_area = 4.0 * 20.0 / (math.pi * 3.0 * diameter_right)
    assert edge_diff_geom_um(left, right) == pytest.approx(
        1.0 / inverse_area, rel=1e-12
    )

    graph, _ = neuron_to_dendra_graph(section)
    _assert_physical_tree(graph)
    assert sum(float(attrs["volume"]) for _, attrs in graph.nodes(data=True)) == (
        pytest.approx(expected_volume, rel=1e-12)
    )


def test_pt3d_diameter_step_preserves_incoming_and_outgoing_frusta():
    section = _section("diameter_step", nseg=2)
    h.pt3dclear(sec=section)
    controls = (
        (0.0, 0.0, 0.0, 1.0),
        (5.0, 0.0, 0.0, 2.0),
        (5.0, 0.0, 0.0, 4.0),
        (10.0, 0.0, 0.0, 3.0),
    )
    for control in controls:
        h.pt3dadd(*control, sec=section)

    def frustum_volume(length, d0, d1):
        return math.pi * length * (d0 * d0 + d0 * d1 + d1 * d1) / 12.0

    def frustum_area(length, d0, d1):
        return math.pi * (d0 + d1) / 2.0 * math.hypot(length, (d1 - d0) / 2.0)

    expected_left_volume = frustum_volume(5.0, 1.0, 2.0)
    expected_right_volume = frustum_volume(5.0, 4.0, 3.0)
    expected_annular_area = math.pi * (4.0**2 - 2.0**2) / 4.0
    expected_area = (
        frustum_area(5.0, 1.0, 2.0)
        + expected_annular_area
        + frustum_area(5.0, 4.0, 3.0)
    )
    segments = list(section)
    assert segment_volume_um3(segments[0]) == pytest.approx(
        expected_left_volume, rel=1e-12
    )
    assert segment_volume_um3(segments[1]) == pytest.approx(
        expected_right_volume, rel=1e-12
    )
    assert sum(segment_volume_um3(seg) for seg in segments) == pytest.approx(
        sum(float(seg.volume()) for seg in segments), rel=1e-12
    )

    # The centre-to-centre path uses the incoming taper from d=1.5 to d=2,
    # skips the zero-length jump, then uses the outgoing taper from d=4 to
    # d=3.5. The discontinuity contributes neither volume nor axial path.
    expected_inv_area = 4.0 * 2.5 / (math.pi * 1.5 * 2.0) + 4.0 * 2.5 / (
        math.pi * 4.0 * 3.5
    )
    actual_inv_area = edge_inv_area_integral_um_inv(segments[0], segments[1])
    assert actual_inv_area == pytest.approx(expected_inv_area, rel=1e-12)
    assert edge_diff_geom_um(segments[0], segments[1]) == pytest.approx(
        1.0 / expected_inv_area, rel=1e-12
    )
    expected_resistance_ohm = float(section.Ra) * 1e4 * expected_inv_area
    assert r_ohm(segments[0], segments[1]) == pytest.approx(
        expected_resistance_ohm, rel=1e-12
    )
    assert sum(float(seg.area()) for seg in segments) == pytest.approx(
        expected_area, rel=1e-12
    )

    graph, _ = neuron_to_dendra_graph(section)
    assert sum(float(attrs["volume"]) for _, attrs in graph.nodes(data=True)) == (
        pytest.approx(expected_left_volume + expected_right_volume, rel=1e-12)
    )
    assert sum(float(attrs["area"]) for _, attrs in graph.nodes(data=True)) == (
        pytest.approx(sum(float(seg.area()) for seg in segments), rel=1e-12)
    )
    _, _, edge_attrs = next(iter(graph.edges(data=True)))
    assert edge_attrs["R_ohm"] == pytest.approx(expected_resistance_ohm, rel=1e-12)
    assert edge_attrs["diff_geom_um"] == pytest.approx(
        1.0 / expected_inv_area, rel=1e-12
    )


def test_native_decimal_boundary_diameter_step_matches_neuron_geometry():
    controls = (
        (0.0, 0.0, 0.0, 2.0),
        (0.1, 0.0, 0.0, 2.0),
        (0.1, 0.0, 0.0, 4.0),
        (0.3, 0.0, 0.0, 4.0),
    )
    morphology = Morphology(rhoa=100.0)
    morphology.section("cable", points=controls, nseg=3)
    native = morphology.compile()

    section = _section("decimal_boundary_step", Ra=100.0, nseg=3)
    h.pt3dclear(sec=section)
    for control in controls:
        h.pt3dadd(*control, sec=section)
    segments = list(section)

    # NEURON stores pt3d controls at its own precision, so comparison needs a
    # small tolerance. Both implementations nevertheless assign the complete
    # annular shoulder to the upstream segment despite 0.1 and 0.3 not being
    # exactly representable in binary floating point.
    annular_area = 3.0 * math.pi
    assert float(segments[0].area()) > annular_area
    assert float(segments[1].area()) < annular_area
    assert native.geometry.area_um2[0] > annular_area
    assert native.geometry.area_um2[1] < annular_area
    assert native.geometry.area_um2 == pytest.approx(
        tuple(float(segment.area()) for segment in segments), rel=5e-7
    )
    assert native.geometry.volume_um3 == pytest.approx(
        tuple(float(segment.volume()) for segment in segments), rel=5e-7
    )
    assert native.geometry.diameter_um == pytest.approx(
        tuple(float(segment.diam) for segment in segments), rel=5e-7
    )
    assert native.geometry.edge_resistance_ohm[1:] == pytest.approx(
        tuple(float(segment.ri()) * 1e6 for segment in segments[1:]), rel=5e-7
    )


def test_bundled_neurolucida_diameter_step_matches_neuron_geometry():
    asc_path = Path(__file__).parents[1] / "docs" / "basics" / "example.asc"
    h.load_file("import3d.hoc")
    reader = h.Import3d_Neurolucida3()
    reader.quiet = 1
    reader.input(str(asc_path))
    importer = h.Import3d_GUI(reader, 0)

    class ImportedCell:
        def __init__(self):
            importer.instantiate(self)

    cell = ImportedCell()
    section = next(sec for sec in cell.all if str(sec.name()).endswith(".dend[13]"))
    repeated = [
        index
        for index in range(1, int(section.n3d()))
        if (
            section.x3d(index),
            section.y3d(index),
            section.z3d(index),
        )
        == (
            section.x3d(index - 1),
            section.y3d(index - 1),
            section.z3d(index - 1),
        )
        and section.diam3d(index) != section.diam3d(index - 1)
    ]
    assert repeated == [6]
    assert section.diam3d(5) == pytest.approx(0.33)
    assert section.diam3d(6) == pytest.approx(0.67)

    expected_volume = sum(float(seg.volume()) for seg in section)
    actual_volume = sum(segment_volume_um3(seg) for seg in section)
    assert actual_volume == pytest.approx(expected_volume, rel=1e-12)

    expected_inv_area = 0.0
    for index in range(1, int(section.n3d())):
        length = float(section.arc3d(index) - section.arc3d(index - 1))
        if length > 0.0:
            expected_inv_area += (
                4.0
                * length
                / (
                    math.pi
                    * float(section.diam3d(index - 1))
                    * float(section.diam3d(index))
                )
            )
    assert edge_inv_area_integral_um_inv(section(0.0), section(1.0)) == (
        pytest.approx(expected_inv_area, rel=1e-12)
    )


def test_swc_pathlike_import_preserves_parameters_and_branch_topology(tmp_path):
    swc_path = tmp_path / "small-branched.swc"
    swc_path.write_text(
        "1 1 0 0 0 5 -1\n"
        "2 3 0 10 0 1 1\n"
        "3 3 0 20 0 1 2\n"
        "4 3 10 30 0 1 3\n"
        "5 3 -10 30 0 1 3\n"
    )

    graph, id2seg = read_swc(
        swc_path,
        d_lambda=0.5,
        freq=100.0,
        rhoa=123.0,
        cm=2.0,
        data_func=lambda seg: {"source_location": str(seg)},
    )

    _assert_physical_tree(graph)
    assert set(id2seg) == set(graph)
    assert graph.number_of_nodes() == 5
    assert all(
        attrs["Ra"] == pytest.approx(123.0) for _, attrs in graph.nodes(data=True)
    )
    assert all(attrs["cm"] == pytest.approx(2.0) for _, attrs in graph.nodes(data=True))
    assert all("source_location" in attrs for _, attrs in graph.nodes(data=True))
    branchpoints = [
        node
        for node, attrs in graph.nodes(data=True)
        if str(attrs["name"]).startswith("branchpoint.")
    ]
    assert len(branchpoints) == 1
    assert graph.out_degree(branchpoints[0]) == 2


def test_native_morphology_swc_export_is_neuron_importable(tmp_path):
    morphology = Morphology()
    soma = morphology.section(
        "soma",
        points=((0.0, 0.0, 0.0, 8.0), (0.0, 0.0, 20.0, 8.0)),
    )
    dendrite_a = morphology.section(
        "dendrite_a",
        points=((0.0, 0.0, 10.0, 2.0), (10.0, 0.0, 15.0, 1.0)),
    )
    dendrite_b = morphology.section(
        "dendrite_b",
        points=((0.0, 0.0, 10.0, 2.0), (-10.0, 0.0, 15.0, 1.0)),
    )
    dendrite_a.connect(soma.at(0.5), child_end=0)
    dendrite_b.connect(soma.at(0.5), child_end=0)

    swc_path = tmp_path / "native-export.swc"
    morphology.write_swc(
        swc_path,
        section_types={"soma": 1, "dendrite_a": 3, "dendrite_b": 3},
    )
    graph, id2seg = read_swc(swc_path, d_lambda=0.5, freq=100.0)

    _assert_physical_tree(graph)
    assert set(id2seg) == set(graph)
    assert sum(graph.out_degree(node) == 2 for node in graph) == 1
    assert sum(graph.out_degree(node) == 0 for node in graph) == 2


def test_neurolucida_pathlike_import_applies_electrical_overrides():
    asc_path = Path(__file__).parent / "simtests" / "111200A.asc"
    graph, id2seg = read_neurolucida(
        asc_path,
        d_lambda=1.0,
        freq=100.0,
        rhoa=117.0,
        cm=1.75,
    )

    _assert_physical_tree(graph)
    assert set(id2seg) == set(graph)
    material_attrs = [
        attrs
        for _, attrs in graph.nodes(data=True)
        if not str(attrs["name"]).startswith("branchpoint.")
    ]
    assert material_attrs
    assert all(attrs["Ra"] == pytest.approx(117.0) for attrs in material_attrs)
    assert all(attrs["cm"] == pytest.approx(1.75) for attrs in material_attrs)


def test_subtree_import_prunes_exclusions_and_supports_metadata_free_graphs():
    root = _section("subtree_root", L=90.0, nseg=3)
    child = _section("subtree_child", L=60.0, nseg=2)
    grandchild = _section("subtree_grandchild", L=30.0, nseg=1)
    child.connect(root(1.0), 0.0)
    grandchild.connect(child(1.0), 0.0)

    graph, id2seg = neuron_to_dendra_graph(root, attach_objects=False)
    _assert_physical_tree(graph)
    assert graph.number_of_nodes() == 6
    assert all(not attrs for _, attrs in graph.nodes(data=True))
    assert {seg.sec for seg in id2seg.values()} == {root, child, grandchild}

    subtree, subtree_map = neuron_to_dendra_graph(child, attach_objects=False)
    _assert_physical_tree(subtree)
    assert subtree.number_of_nodes() == 3
    assert {seg.sec for seg in subtree_map.values()} == {child, grandchild}

    pruned, pruned_map = neuron_to_dendra_graph(
        root, exclude={child}, attach_objects=False
    )
    _assert_physical_tree(pruned)
    assert pruned.number_of_nodes() == 3
    assert {seg.sec for seg in pruned_map.values()} == {root}

    with pytest.raises(ValueError, match="empty after exclusions"):
        neuron_to_dendra_graph(root, exclude={root})


def test_custom_metadata_cannot_override_physical_branchpoint_contract():
    root = _section("custom_root", L=20.0)
    child_a = _section("custom_a", L=40.0)
    child_b = _section("custom_b", L=50.0)
    unrelated = _section("custom_unrelated", L=10.0)
    child_a.connect(root(1.0), 0.0)
    child_b.connect(root(1.0), 0.0)

    def custom_data(seg):
        return {
            "name": "spoofed",
            "area": 999.0,
            "volume": 999.0,
            "volume_um3": 999.0,
            "volume_i": 999.0,
            "custom_x": float(seg.x),
        }

    graph, id2seg = neuron_to_dendra_graph(root, data_func=custom_data)
    _assert_physical_tree(graph)
    branchpoints = [
        (node, attrs)
        for node, attrs in graph.nodes(data=True)
        if str(attrs["name"]).startswith("branchpoint.")
    ]
    assert len(branchpoints) == 1
    branch_node, branch_attrs = branchpoints[0]
    assert graph.degree(branch_node) == 3
    for name in ("area", "volume", "volume_um3", "volume_i"):
        assert branch_attrs[name] == 0.0
    assert "custom_x" in branch_attrs
    assert set(id2seg) == set(graph)

    assert r_ohm(root(0.5), root(0.5)) == 0.0
    with pytest.raises(ValueError, match="not a direct NEURON"):
        r_ohm(root(0.5), unrelated(0.5))
