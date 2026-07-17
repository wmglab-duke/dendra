"""Contracts for NEURON-backed Neurolucida ASC import into Morphology."""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pytest

import dendra as dn

pytest.importorskip("neuron")
pytestmark = pytest.mark.neuron

_MORPHOLOGY_DATA = Path(__file__).parent / "data" / "morphologies"
_MIXED_ASC = _MORPHOLOGY_DATA / "neurolucida_mixed.asc"
_REPAIRED_ASC = _MORPHOLOGY_DATA / "neurolucida_repaired.asc"
_TWO_ROOTS_ASC = _MORPHOLOGY_DATA / "neurolucida_two_roots.asc"
_REAL_WORLD_ASC = Path(__file__).parent / "simtests" / "111200A.asc"
_DOCS_EXAMPLE_ASC = Path(__file__).parents[1] / "docs" / "basics" / "example.asc"


def _section_map(morphology: dn.Morphology):
    return {section.name: section for section in morphology.sections}


def _neuron_registry_snapshot():
    import neuron
    from neuron import h

    section_names = tuple(sorted(str(section.name()) for section in h.allsec()))
    section_db = getattr(neuron, "_sec_db", None)
    db_keys = None if not isinstance(section_db, dict) else frozenset(section_db)
    return section_names, db_keys


def test_from_asc_preserves_normalized_geometry_topology_names_and_labels():
    morphology = dn.Morphology.from_asc(_MIXED_ASC)
    sections = _section_map(morphology)

    assert set(sections) == {
        "soma[0]",
        "axon[0]",
        "dend[0]",
        "dend[1]",
        "dend[2]",
        "apic[0]",
    }
    for name, section in sections.items():
        region = name.split("[", 1)[0]
        assert section.labels == frozenset({name, region})
        assert section.nseg == 1
        assert section.rhoa == pytest.approx(100.0)
        assert section.cm == pytest.approx(1.0)

    np.testing.assert_allclose(
        sections["axon[0]"].points,
        ((0.0, 0.0, 0.0, 2.0), (0.0, -10.0, 0.0, 1.0)),
    )
    np.testing.assert_allclose(
        sections["dend[0]"].points,
        ((0.0, 0.0, 0.0, 3.0), (10.0, 0.0, 0.0, 2.0)),
    )
    np.testing.assert_allclose(
        sections["dend[1]"].points,
        ((10.0, 0.0, 0.0, 2.0), (20.0, 5.0, 0.0, 1.0)),
    )
    np.testing.assert_allclose(
        sections["dend[2]"].points,
        ((10.0, 0.0, 0.0, 2.0), (20.0, -5.0, 0.0, 1.5)),
    )
    np.testing.assert_allclose(
        sections["apic[0]"].points,
        ((0.0, 0.0, 0.0, 4.0), (0.0, 0.0, 15.0, 2.0)),
    )

    # Morphology does not yet expose public connection introspection. These
    # assertions pin the bridge's exact electrical attachment semantics rather
    # than merely checking that the resulting graph happens to be connected.
    connections = morphology._connections
    for child_name in ("axon[0]", "dend[0]", "apic[0]"):
        connection = connections[child_name]
        assert connection.parent_name == "soma[0]"
        assert connection.parent_x == pytest.approx(0.5)
        assert connection.child_end == 0
    for child_name in ("dend[1]", "dend[2]"):
        connection = connections[child_name]
        assert connection.parent_name == "dend[0]"
        assert connection.parent_x == pytest.approx(1.0)
        assert connection.child_end == 0

    graph = morphology.compile()
    assert graph.n_compartments == 7
    assert graph.metadata.kind.count("junction") == 1
    assert graph.metadata.section_name[graph.topology.root] == "soma[0]"
    for section_name in sections:
        material_nodes = [
            node
            for node, source in enumerate(graph.metadata.section_name)
            if source == section_name
        ]
        assert len(material_nodes) == 1
        node = material_nodes[0]
        assert sections[section_name].labels == graph.metadata.labels[node]


def test_from_asc_applies_explicit_electrical_values_and_uniform_nseg():
    morphology = dn.Morphology.from_asc(
        _MIXED_ASC,
        rhoa=117.0,
        cm=1.75,
        nseg=3,
    )

    assert morphology.rhoa == pytest.approx(117.0)
    assert morphology.cm == pytest.approx(1.75)
    assert all(section.rhoa == pytest.approx(117.0) for section in morphology.sections)
    assert all(section.cm == pytest.approx(1.75) for section in morphology.sections)
    assert all(section.nseg == 3 for section in morphology.sections)

    graph = morphology.compile()
    for section in morphology.sections:
        assert graph.metadata.section_name.count(section.name) == 3


def test_from_asc_rejects_ambiguous_roots_and_accepts_an_exact_root_name():
    with pytest.raises(ValueError, match=r"root|disconnected") as error:
        dn.Morphology.from_asc(_TWO_ROOTS_ASC)

    message = str(error.value)
    assert "axon[0]" in message
    assert "dend[0]" in message

    axon = dn.Morphology.from_asc(_TWO_ROOTS_ASC, root="axon[0]")
    assert tuple(section.name for section in axon.sections) == ("axon[0]",)
    np.testing.assert_allclose(
        axon.sections[0].points,
        ((0.0, 0.0, 0.0, 1.0), (10.0, 0.0, 0.0, 1.0)),
    )
    assert axon.compile().n_compartments == 1

    dendrite = dn.Morphology.from_asc(_TWO_ROOTS_ASC, root="dend[0]")
    assert tuple(section.name for section in dendrite.sections) == ("dend[0]",)
    assert dendrite.compile().n_compartments == 1


def test_from_asc_rejects_an_unknown_or_non_string_root():
    with pytest.raises(ValueError, match=r"root|unknown"):
        dn.Morphology.from_asc(_TWO_ROOTS_ASC, root="soma[0]")
    with pytest.raises(TypeError, match="root"):
        dn.Morphology.from_asc(_TWO_ROOTS_ASC, root=0)


def test_from_asc_warns_once_and_preserves_neurons_nearest_soma_repair():
    with pytest.warns(
        RuntimeWarning, match=r"outside the soma|logical connection"
    ) as seen:
        morphology = dn.Morphology.from_asc(_REPAIRED_ASC)

    assert len(seen) == 1
    sections = _section_map(morphology)
    assert set(sections) == {"soma[0]", "dend[0]"}
    connection = morphology._connections["dend[0]"]
    assert connection.parent_name == "soma[0]"
    assert connection.parent_x == pytest.approx(0.5)
    assert connection.child_end == 0
    assert morphology.compile().n_compartments == 2


def test_from_asc_releases_temporary_neuron_sections_on_success_and_warning_error():
    # Clear temporary cells from earlier tests before pinning this call's delta.
    gc.collect()
    before = _neuron_registry_snapshot()

    morphology = dn.Morphology.from_asc(_MIXED_ASC)
    assert morphology.sections
    gc.collect()
    assert _neuron_registry_snapshot() == before

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            RuntimeWarning, match=r"outside the soma|logical connection"
        ):
            dn.Morphology.from_asc(_REPAIRED_ASC)
    gc.collect()
    assert _neuron_registry_snapshot() == before


def test_from_asc_validates_paths_and_parse_failures(tmp_path):
    missing = tmp_path / "missing.asc"
    with pytest.raises(FileNotFoundError):
        dn.Morphology.from_asc(missing)

    with pytest.raises(IsADirectoryError):
        dn.Morphology.from_asc(tmp_path)

    empty = tmp_path / "empty.asc"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=r"ASC|section|parse|empty"):
        dn.Morphology.from_asc(empty)

    malformed = tmp_path / "malformed.asc"
    malformed.write_text("( (Axon)\n  (0 0 0 1)\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"ASC|parse|invalid"):
        dn.Morphology.from_asc(malformed)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"nseg": 0}, ValueError, "nseg"),
        ({"nseg": 1.5}, TypeError, "nseg"),
        ({"rhoa": 0.0}, ValueError, "rhoa"),
        ({"cm": float("inf")}, ValueError, "cm"),
    ],
)
def test_from_asc_reuses_native_electrical_and_nseg_validation(
    kwargs, error_type, match
):
    with pytest.raises(error_type, match=match):
        dn.Morphology.from_asc(_MIXED_ASC, **kwargs)


def test_from_asc_coalesces_only_redundant_consecutive_pt3d_samples(tmp_path):
    redundant = tmp_path / "redundant.asc"
    redundant.write_text(
        """( (Axon)
  (0 0 0 2)
  (5 0 0 2)
  (5 0 0 2)
  (10 0 0 1)
)
""",
        encoding="utf-8",
    )

    morphology = dn.Morphology.from_asc(redundant)
    assert len(morphology.sections) == 1
    np.testing.assert_allclose(
        morphology.sections[0].points,
        (
            (0.0, 0.0, 0.0, 2.0),
            (5.0, 0.0, 0.0, 2.0),
            (10.0, 0.0, 0.0, 1.0),
        ),
    )


def test_from_asc_preserves_zero_length_diameter_discontinuities(tmp_path):
    discontinuous = tmp_path / "discontinuous.asc"
    discontinuous.write_text(
        """( (Axon)
  (0 0 0 2)
  (5 0 0 2)
  (5 0 0 3)
  (10 0 0 1)
)
""",
        encoding="utf-8",
    )

    morphology = dn.Morphology.from_asc(discontinuous)

    assert morphology.sections[0].points == (
        (0.0, 0.0, 0.0, 2.0),
        (5.0, 0.0, 0.0, 2.0),
        (5.0, 0.0, 0.0, 3.0),
        (10.0, 0.0, 0.0, 1.0),
    )


def test_from_asc_rejects_an_all_coincident_diameter_profile(tmp_path):
    collapsed = tmp_path / "collapsed-diameter-profile.asc"
    collapsed.write_text(
        """( (Axon)
  (1 2 3 1)
  (1 2 3 2)
  (1 2 3 3)
)
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"axon\[0\]|distinct|cable"):
        dn.Morphology.from_asc(collapsed)


def test_from_asc_rejects_a_centerline_collapsed_by_duplicate_coalescing(tmp_path):
    collapsed = tmp_path / "collapsed.asc"
    collapsed.write_text(
        """( (Axon)
  (1 2 3 2)
  (1 2 3 2)
)
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"axon\[0\]|distinct|cable"):
        dn.Morphology.from_asc(collapsed)


def test_from_asc_compiles_and_constructs_a_tree_snapshot():
    morphology = dn.Morphology.from_asc(_MIXED_ASC)
    graph = morphology.compile()
    tree = dn.Tree.from_morphology(morphology)

    assert tree.shape == (1, graph.n_compartments)
    assert tree._compartment_graph.topology == graph.topology
    assert tree._compartment_graph.geometry == graph.geometry
    assert tree._compartment_graph.metadata == graph.metadata


@pytest.mark.slow
def test_from_asc_handles_the_real_world_duplicate_and_explicit_component():
    morphology = dn.Morphology.from_asc(_REAL_WORLD_ASC, root="soma[0]")
    sections = _section_map(morphology)

    assert len(sections) == 248
    # This source trace contains one exactly repeated consecutive sample. The
    # NEURON snapshot has 139 samples; native Morphology retains 138 physical
    # controls after removing only that redundant point.
    assert len(sections["axon[121]"].points) == 138
    assert all(
        left[:3] != right[:3]
        for left, right in zip(
            sections["axon[121]"].points,
            sections["axon[121]"].points[1:],
        )
    )


@pytest.mark.slow
def test_from_asc_loads_the_documented_example_diameter_step():
    morphology = dn.Morphology.from_asc(_DOCS_EXAMPLE_ASC)
    sections = _section_map(morphology)

    assert len(sections) == 211
    assert sections["dend[13]"].points[5:7] == (
        (
            62.13990020751953,
            -16.510700225830078,
            -6.200329780578613,
            0.33000001311302185,
        ),
        (
            62.13990020751953,
            -16.510700225830078,
            -6.200329780578613,
            0.6700000166893005,
        ),
    )

    graph = morphology.compile()
    node = graph.metadata.section_name.index("dend[13]")
    assert graph.geometry.area_um2[node] == pytest.approx(145.31899142855707)
    assert graph.geometry.volume_um3[node] == pytest.approx(21.213373907703662)
    assert graph.geometry.diameter_um[node] == pytest.approx(0.5515648780271981)
