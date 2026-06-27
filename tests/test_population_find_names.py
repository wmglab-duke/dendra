"""Regression tests for token-aware Population.find morphology matching."""

from __future__ import annotations

import warnings

# Import torch first in this constrained test environment so libgomp initializes
# before Dendra applies its preferred OpenMP affinity settings.
import torch  # noqa: F401

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Dendra: torch was imported before Dendra configured TorchInductor.*",
        category=RuntimeWarning,
    )
    import dendra as dn


def _population_with_names(names):
    population = dn.Population(N=1, C=len(names))
    population.names = list(names)
    return population


def test_find_matches_neuron_names_after_underscore():
    population = _population_with_names(
        [
            "MelnickSG_soma(0.05)",
            "MelnickSG_soma(0.15)",
            "MelnickSG_hillock(0.50)",
            "MelnickSG_dend(0.50)",
        ]
    )

    assert population.find("soma", as_list=True) == [0, 1]
    assert population.find("MelnickSG_soma", as_list=True) == [0, 1]
    assert population.find("hillock", as_list=True) == [2]
    assert population.find("dend", as_list=True) == [3]


def test_find_base_section_name_matches_indexed_sections():
    population = _population_with_names(
        [
            "Cell[0].soma[0](0.25)",
            "Cell[0].soma[0](0.75)",
            "Cell[0].soma[1](0.50)",
            "Cell[0].dend[0](0.50)",
        ]
    )

    assert population.find("soma", as_list=True) == [0, 1, 2]
    assert population.find("soma[0]", as_list=True) == [0, 1]
    assert population.find("soma[1]", as_list=True) == [2]
    assert population.find("soma[*]", as_list=True) == [0, 1, 2]


def test_indexed_selector_is_literal_and_token_bounded():
    population = _population_with_names(
        [
            "Cell_soma[0](0.5)",
            "Cell_soma[1](0.5)",
            "Cell_presoma[0](0.5)",
            "Cell_soma[00](0.5)",
            "Cell_soma[0]extra(0.5)",
        ]
    )

    assert population.find("soma[0]", as_list=True) == [0]


def test_plain_selector_does_not_match_alphanumeric_substrings():
    population = _population_with_names(
        [
            "Cell_soma(0.5)",
            "Cell_presoma(0.5)",
            "Cell_somatic(0.5)",
            "Cell_soma2(0.5)",
            "Cell_soma_aux(0.5)",
        ]
    )

    # Underscores are delimiters, but adjacent letters/digits are not.
    assert population.find("soma", as_list=True) == [0, 4]


def test_myelin_and_unmyelin_are_distinct_identifier_tokens():
    """L23-style myelin labels must remain disjoint from unmyelin labels."""
    population = _population_with_names(
        [
            "L23_PC_cADpyr[0].myelin[0](0.50)",
            "L23_PC_cADpyr[0].unmyelin[0](0.50)",
            "L23_PC_cADpyr[0].myelin[1](0.50)",
            "L23_PC_cADpyr[0].demyelin[0](0.50)",
            "L23_PC_cADpyr[0].myelinated[0](0.50)",
            "L23_PC_cADpyr[0].node[0](0.50)",
        ]
    )

    myelin = population.find("myelin", as_list=True)
    unmyelin = population.find("unmyelin", as_list=True)

    assert myelin == [0, 2]
    assert unmyelin == [1]
    assert population.find("myelin[0]", as_list=True) == [0]
    assert population.find("unmyelin[0]", as_list=True) == [1]
    assert set(myelin).isdisjoint(unmyelin)


def test_asterisk_explicitly_broadens_myelin_matching():
    """A leading ``*`` requests the suffix match rather than token matching."""
    population = _population_with_names(
        [
            "L23_PC_cADpyr[0].myelin[0](0.50)",
            "L23_PC_cADpyr[0].unmyelin[0](0.50)",
            "L23_PC_cADpyr[0].myelin[1](0.50)",
            "L23_PC_cADpyr[0].myelinated[0](0.50)",
            "L23_PC_cADpyr[0].soma[0](0.50)",
        ]
    )

    assert population.find("*myelin", as_list=True) == [0, 1, 2]
    assert population.find("*myelin[0]", as_list=True) == [0, 1]
    assert population.find("myelin*", as_list=True) == [0, 2, 3]
    assert population.find("*myelin*", as_list=True) == [0, 1, 2, 3]


def test_asterisk_is_literal_when_fuzzy_matching_is_disabled():
    population = _population_with_names(["myelin", "*myelin"])

    assert population.find("*myelin", fuzzy=False, as_list=True) == [1]


def test_find_supports_dot_underscore_and_case_control():
    population = _population_with_names(
        [
            "Cell.SOMA(0.5)",
            "Cell_soma(0.5)",
            "Cell.Somatic(0.5)",
        ]
    )

    assert population.find("soma", as_list=True) == [0, 1]
    assert population.find("soma", match_case=True, as_list=True) == [1]
    assert population.find("SOMA", match_case=True, as_list=True) == [0]


def test_loc_refinement_still_operates_after_token_matching():
    population = _population_with_names(
        [
            "MelnickSG_soma(0.05)",
            "MelnickSG_soma(0.15)",
            "MelnickSG_soma(0.95)",
            "MelnickSG_dend(0.15)",
        ]
    )

    assert population.find("soma", loc=0.12, as_list=True) == [1]


def test_default_branchpoint_exclusion_uses_same_boundaries():
    population = _population_with_names(
        [
            "Cell_soma(0.5)",
            "branchpoint.0",
            "Cell_branchpoint_aux",
            "Cell_notbranchpoint(0.5)",
        ]
    )

    assert population.find(as_list=True) == [0, 3]


def test_exact_matching_behavior_is_unchanged_when_fuzzy_is_false():
    population = _population_with_names(["soma", "Cell_soma(0.5)", "soma[0]"])

    assert population.find("soma", fuzzy=False, as_list=True) == [0]
    assert population.find("soma[0]", fuzzy=False, as_list=True) == [2]


def test_l23_tree_slice_labels_keep_myelin_and_unmyelin_disjoint():
    """Exercise the same ``Tree.from_graph``/``slice`` path as the L23 builder."""
    import networkx as nx

    names = [
        "L23_PC_cADpyr[0].soma[0](0.50)",
        "L23_PC_cADpyr[0].axon[0](0.50)",
        "L23_PC_cADpyr[0].myelin[0](0.50)",
        "L23_PC_cADpyr[0].node[0](0.50)",
        "L23_PC_cADpyr[0].unmyelin[0](0.50)",
        "L23_PC_cADpyr[0].myelin[1](0.50)",
    ]
    graph = nx.DiGraph()
    for node, name in enumerate(names):
        graph.add_node(
            node,
            name=name,
            L=10.0,
            diam=1.0,
            Ra=100.0,
            cm=1.0,
            area=31.4,
            x=float(node),
            y=0.0,
            z=0.0,
        )
    for node in range(len(names) - 1):
        graph.add_edge(
            node,
            node + 1,
            R_ohm=1.0e5,
            L=10.0,
            diff_geom_um=1.0,
        )

    cell = dn.Tree.from_graph(graph, N=1)
    cell.slice("myelin").label("myelin")
    cell.slice("unmyelin").label("unmyelin")

    assert cell.find("myelin", as_list=True) == [2, 5]
    assert cell.find("unmyelin", as_list=True) == [4]
    assert tuple(cell.myelin.shape) == (1, 2)
    assert tuple(cell.unmyelin.shape) == (1, 1)


def _indices_as_list(indices, *, total_size: int):
    if isinstance(indices, slice):
        start, stop, step = indices.indices(total_size)
        return list(range(start, stop, step))
    return indices.detach().cpu().tolist()


def test_compound_underscore_selector_and_full_report_are_consistent():
    names = [
        "Cell[0].apic_trunk[0](0.25)",
        "Cell[0].apic_tuft[0](0.50)",
        "Cell[0].apic_trunk[0](0.75)",
        "Cell[0].axon[0](0.50)",
    ]
    population = _population_with_names(names)

    assert population.find("apic_trunk", as_list=True) == [0, 2]
    assert population.find("trunk", as_list=True) == [0, 2]

    report = population.find(["apic_trunk", "axon"], full_report=True)
    assert _indices_as_list(report.indices, total_size=len(names)) == [0, 2, 3]
    assert _indices_as_list(
        report.local_indices["apic_trunk"], total_size=report.total_size
    ) == [0, 1]
    assert _indices_as_list(
        report.local_indices["axon"], total_size=report.total_size
    ) == [2]
    assert report.local_sizes == {"apic_trunk": 2, "axon": 1}
