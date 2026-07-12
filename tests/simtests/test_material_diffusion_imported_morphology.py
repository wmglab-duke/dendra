"""Finite-volume material oracle on an imported tapered NEURON tree.

NEURON is used only to define and discretize the morphology.  The diffusion
reference is assembled independently from the imported compartment volumes and
edge geometry, so this test covers the complete

    NEURON morphology -> graph -> Tree -> MaterialProcess

path without sharing Dendra's spatial-operator implementation.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mechanisms._material_process import DiffusionProcess

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DTYPE = torch.float64
DIFFUSIVITY = 12.5  # um^2 / ms
ORACLE_DT = 0.1
ORACLE_STEPS = 8
REFINEMENT_TSTOP = 1.6

_SECTION_COUNTER = 0


class _ImportedTreeDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(D=DIFFUSIVITY)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("tracer", field="c", D="D")


@pytest.fixture(autouse=True)
def _delete_sections_created_by_test():
    """Keep NEURON's process-global section namespace isolated."""
    before = set(h.allsec())
    yield
    for section in list(h.allsec()):
        if section not in before:
            h.delete_section(sec=section)


def _section(prefix, *, points, ra, cm, nseg):
    global _SECTION_COUNTER
    _SECTION_COUNTER += 1
    section = h.Section(name=f"{prefix}_{_SECTION_COUNTER}")
    section.Ra = float(ra)
    section.cm = float(cm)
    section.nseg = int(nseg)
    h.pt3dclear(sec=section)
    for x, y, z, diameter in points:
        h.pt3dadd(x, y, z, diameter, sec=section)
    return section


def _build_imported_morphology():
    """Build a tapered endpoint bifurcation with an explicit branchpoint."""
    soma = _section(
        "soma_material",
        points=[
            (0.0, 0.0, 0.0, 9.0),
            (20.0, 1.0, 0.0, 6.5),
            (42.0, 10.0, 2.0, 4.0),
        ],
        ra=83.0,
        cm=0.9,
        nseg=3,
    )
    apic = _section(
        "apic_material",
        points=[
            (42.0, 10.0, 2.0, 3.8),
            (52.0, 52.0, 7.0, 2.3),
            (64.0, 101.0, 13.0, 0.75),
        ],
        ra=137.0,
        cm=1.2,
        nseg=5,
    )
    dend = _section(
        "dend_material",
        points=[
            (42.0, 10.0, 2.0, 4.6),
            (12.0, 38.0, -4.0, 2.7),
            (-31.0, 57.0, -11.0, 1.05),
        ],
        ra=109.0,
        cm=1.5,
        nseg=5,
    )

    # Three half-segments meet at soma(1): the soma, apic, and dend cables.
    # The importer must retain this as a zero-volume algebraic branchpoint.
    apic.connect(soma(1.0), 0.0)
    dend.connect(soma(1.0), 0.0)

    graph, node_to_segment = neuron_to_dendra_graph(soma)
    return graph, node_to_segment, (soma, apic, dend)


def _branchpoints(graph):
    return [
        node
        for node, attrs in graph.nodes(data=True)
        if str(attrs.get("name", "")).startswith("branchpoint.")
    ]


def _geometry_system(graph):
    """Assemble ``V dc/dt = -L c`` directly from public graph metadata."""
    size = graph.number_of_nodes()
    volume = torch.tensor(
        [float(graph.nodes[node]["volume_i"]) for node in range(size)],
        dtype=DTYPE,
    )
    laplacian = torch.zeros((size, size), dtype=DTYPE)
    for parent, child, attrs in graph.edges(data=True):
        coupling = DIFFUSIVITY * float(attrs["diff_geom_um"])
        laplacian[parent, parent] += coupling
        laplacian[child, child] += coupling
        laplacian[parent, child] -= coupling
        laplacian[child, parent] -= coupling
    return volume, laplacian


def _consistent_initial_state(graph, node_to_segment, sections, volume, laplacian):
    """Create a positive spatial profile and satisfy zero-volume constraints."""
    soma, apic, dend = sections
    concentration = torch.empty(graph.number_of_nodes(), dtype=DTYPE)
    for node in range(graph.number_of_nodes()):
        if volume[node] == 0.0:
            concentration[node] = 1.0  # replaced by algebraic projection below
            continue
        segment = node_to_segment[node]
        x = float(segment.x)
        if segment.sec is soma:
            concentration[node] = 0.25 + 0.95 * x
        elif segment.sec is apic:
            concentration[node] = 2.1 + 0.35 * math.cos(math.pi * x)
        elif segment.sec is dend:
            concentration[node] = 0.55 + 1.25 * x * x
        else:  # pragma: no cover - construction invariant
            raise AssertionError(f"Unexpected imported section for node {node}")

    positive = torch.nonzero(volume > 0.0, as_tuple=False).flatten()
    algebraic = torch.nonzero(volume == 0.0, as_tuple=False).flatten()
    if algebraic.numel():
        lzz = laplacian.index_select(0, algebraic).index_select(1, algebraic)
        lzp = laplacian.index_select(0, algebraic).index_select(1, positive)
        concentration[algebraic] = torch.linalg.solve(
            lzz, -(lzp @ concentration[positive]).unsqueeze(-1)
        ).squeeze(-1)
    assert torch.all(concentration > 0.0)
    return concentration


def _backward_euler_step(concentration, volume, laplacian, dt):
    """Independent sealed finite-volume backward-Euler update."""
    mass_matrix = torch.diag(volume)
    lhs = mass_matrix + float(dt) * laplacian
    rhs = mass_matrix @ concentration
    return torch.linalg.solve(lhs, rhs)


def _semidiscrete_exact(concentration, volume, laplacian, tstop):
    """Solve the continuous finite-volume DAE after eliminating branchpoints."""
    positive = torch.nonzero(volume > 0.0, as_tuple=False).flatten()
    algebraic = torch.nonzero(volume == 0.0, as_tuple=False).flatten()
    lpp = laplacian.index_select(0, positive).index_select(1, positive)

    if algebraic.numel():
        lpz = laplacian.index_select(0, positive).index_select(1, algebraic)
        lzp = laplacian.index_select(0, algebraic).index_select(1, positive)
        lzz = laplacian.index_select(0, algebraic).index_select(1, algebraic)
        effective_laplacian = lpp - lpz @ torch.linalg.solve(lzz, lzp)
    else:  # pragma: no cover - this fixture intentionally contains a branchpoint
        effective_laplacian = lpp

    generator = -effective_laplacian / volume[positive].unsqueeze(-1)
    material_state = (
        torch.matrix_exp(float(tstop) * generator) @ concentration[positive]
    )
    result = torch.empty_like(concentration)
    result[positive] = material_state
    if algebraic.numel():
        result[algebraic] = torch.linalg.solve(
            lzz, -(lzp @ material_state).unsqueeze(-1)
        ).squeeze(-1)
    return result


def _new_model(graph, initial):
    model = dn.Tree.from_graph(
        graph,
        N=1,
        v_init=-65.0,
        dtype=DTYPE,
    )
    model.material(
        "tracer",
        fields={"c": initial.reshape(1, -1)},
        min_values={"c": 0.0},
        domain="intracellular",
    )
    model.insert(_ImportedTreeDiffusion, D=torch.tensor(DIFFUSIVITY, dtype=DTYPE))
    model.eval()
    model.initialize()
    return model


def _material_state(model):
    return model.mech.materials["tracer"].c[0]


def _branchpoint_flux(concentration, graph, branchpoint):
    flux = concentration.new_zeros(())
    for neighbor in graph.to_undirected().neighbors(branchpoint):
        attrs = (
            graph.edges[branchpoint, neighbor]
            if graph.has_edge(branchpoint, neighbor)
            else graph.edges[neighbor, branchpoint]
        )
        coupling = DIFFUSIVITY * float(attrs["diff_geom_um"])
        flux = flux + coupling * (concentration[neighbor] - concentration[branchpoint])
    return flux


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


def _fixture_state():
    graph, node_to_segment, sections = _build_imported_morphology()
    volume, laplacian = _geometry_system(graph)
    initial = _consistent_initial_state(
        graph, node_to_segment, sections, volume, laplacian
    )
    return graph, node_to_segment, volume, laplacian, initial


def test_imported_tapered_tree_diffusion_matches_every_compartment_oracle():
    graph, node_to_segment, volume, laplacian, initial = _fixture_state()
    branchpoints = _branchpoints(graph)

    assert nx.is_arborescence(graph)
    assert len(branchpoints) == 1
    branchpoint = branchpoints[0]
    assert graph.to_undirected().degree(branchpoint) == 3
    assert volume[branchpoint] == 0.0
    assert len(node_to_segment) == graph.number_of_nodes()
    # The importer's semantic grouping deliberately makes tensor order differ
    # from topological solver order; this exercises both mapping permutations.
    assert list(nx.topological_sort(graph)) != list(range(len(graph)))

    model = _new_model(graph, initial)
    torch.testing.assert_close(model.volume_i[0], volume, rtol=0.0, atol=0.0)
    for child in range(len(graph)):
        parents = list(graph.predecessors(child))
        expected_parent = -1 if not parents else parents[0]
        assert int(model.diff_parent_index[child]) == expected_parent
        expected_geometry = (
            0.0
            if not parents
            else float(graph.edges[expected_parent, child]["diff_geom_um"])
        )
        assert float(model.diff_geom_um[0, child]) == pytest.approx(
            expected_geometry, rel=2e-15, abs=0.0
        )

    oracle = initial.clone()
    initial_mass = torch.dot(volume, initial)
    for _ in range(ORACLE_STEPS):
        oracle = _backward_euler_step(oracle, volume, laplacian, ORACLE_DT)
        model.step(dt=ORACLE_DT)
        actual = _material_state(model)

        # Comparing the complete vector includes the algebraic branchpoint and
        # catches graph/tensor reordering errors that strategic probes miss.
        torch.testing.assert_close(actual, oracle, rtol=3e-12, atol=3e-12)
        torch.testing.assert_close(
            torch.dot(volume, actual), initial_mass, rtol=3e-13, atol=3e-12
        )
        assert torch.all(actual >= 0.0)
        assert abs(float(_branchpoint_flux(actual, graph, branchpoint))) < 2e-12

    assert float(torch.max(torch.abs(oracle - initial))) > 0.02
    neighbor_values = oracle[
        torch.tensor(list(graph.to_undirected().neighbors(branchpoint)))
    ]
    assert neighbor_values.min() <= oracle[branchpoint] <= neighbor_values.max()


def test_imported_tapered_tree_diffusion_converges_under_dt_refinement():
    graph, _, volume, laplacian, initial = _fixture_state()
    exact = _semidiscrete_exact(initial, volume, laplacian, REFINEMENT_TSTOP)
    errors = []

    for dt in (0.2, 0.1, 0.05):
        model = _new_model(graph, initial)
        for _ in range(round(REFINEMENT_TSTOP / dt)):
            model.step(dt=dt)
        actual = _material_state(model)
        errors.append(float(torch.max(torch.abs(actual - exact))))
        assert torch.all(actual >= 0.0)
        torch.testing.assert_close(
            torch.dot(volume, actual),
            torch.dot(volume, initial),
            rtol=4e-13,
            atol=4e-12,
        )

    assert errors[0] > 1e-5  # the refinement signal must be nontrivial
    assert errors[1] < 0.65 * errors[0]
    assert errors[2] < 0.65 * errors[1]


def test_imported_tapered_tree_diffusion_checkpoint_continuation_is_exact():
    graph, _, volume, laplacian, initial = _fixture_state()
    source = _new_model(graph, initial)
    oracle = initial.clone()

    for _ in range(3):
        source.step(dt=ORACLE_DT)
        oracle = _backward_euler_step(oracle, volume, laplacian, ORACLE_DT)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    for _ in range(5):
        source.step(dt=ORACLE_DT)
        oracle = _backward_euler_step(oracle, volume, laplacian, ORACLE_DT)

    resumed = _new_model(graph, initial)
    resumed.restore_dict_from_checkpoint(checkpoint)
    for _ in range(5):
        resumed.step(dt=ORACLE_DT)

    torch.testing.assert_close(
        _material_state(resumed), _material_state(source), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(_material_state(resumed), oracle, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(resumed.t, source.t, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        torch.dot(volume, _material_state(resumed)),
        torch.dot(volume, initial),
        rtol=3e-13,
        atol=3e-12,
    )
    assert torch.all(_material_state(resumed) >= 0.0)
