"""Public contracts for named material geometry on branched Tree populations."""

from __future__ import annotations

import copy

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import DiffusionProcess

DTYPE = torch.float64
DT = 0.075
C = 5
E = C - 1


class _NamedTreeNodeDiffusion(DiffusionProcess):
    DiffusionProcess.RANGE(D=1.0)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE(
        "solute",
        field="c",
        D="D",
        domain="extracellular",
        geometry="chemical_tree",
        D_location="node",
    )


class _NamedTreeEdgeDiffusion(DiffusionProcess):
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE(
        "solute",
        field="c",
        D="edge_diffusivity",
        domain="extracellular",
        geometry="chemical_tree",
        D_location="edge",
    )


class _NamedTreeExplicitEdgeDiffusion(DiffusionProcess):
    DiffusionProcess.METHOD("explicit", solver="dense")
    DiffusionProcess.DIFFUSE(
        "solute",
        field="c",
        D="edge_diffusivity",
        domain="extracellular",
        geometry="chemical_tree",
        D_location="edge",
    )


class _InvalidRangeTreeEdgeDiffusion(DiffusionProcess):
    DiffusionProcess.RANGE(D=1.0)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE(
        "solute",
        field="c",
        D="D",
        domain="extracellular",
        geometry="chemical_tree",
        D_location="edge",
    )


class _NativeTreeDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(D=0.75)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("solute", field="c", D="D", domain="intracellular")


def _branched_graph():
    """Return a canonical graph whose root is storage slot 3, not slot 0."""
    graph = nx.DiGraph()
    # Insertion/storage order is deliberately different from topological order.
    for node in range(C):
        graph.add_node(
            node,
            name=f"compartment.{node}",
            kind="compartment",
            L=1.0,
            diam=1.0,
            Ra=100.0,
            cm=1.0,
            area=1.0,
            volume=20.0 + node,
            volume_i=30.0 + node,
            # Native extracellular storage is deliberately unusable. Named
            # extracellular geometry must be wholly independent of this.
            volume_o=0.0,
            x=float(node),
            y=0.0,
            z=0.0,
        )

    # Compact material-edge order is ascending child storage slot: 0, 1, 2, 4.
    # Native factors are deliberately unrelated to named chemical geometry.
    for parent, child, native_factor in (
        (3, 0, 101.0),
        (0, 1, 103.0),
        (0, 2, 107.0),
        (3, 4, 109.0),
    ):
        graph.add_edge(parent, child, diff_geom_um=native_factor, R_ohm=1.0e8, L=1.0)
    return graph


def _alternate_branched_graph():
    """Return the same compartments with a deliberately different topology."""
    graph = _branched_graph()
    graph.remove_edges_from(list(graph.edges))
    for parent, child, factor in (
        (3, 0, 211.0),
        (0, 1, 223.0),
        (1, 2, 227.0),
        (2, 4, 229.0),
    ):
        graph.add_edge(parent, child, diff_geom_um=factor, R_ohm=1.0e8, L=1.0)
    return graph


def _as_rows(value, rows, width, *, dtype=DTYPE):
    value = torch.as_tensor(value, dtype=dtype)
    target = (rows, width)
    if value.ndim == 0:
        return value.expand(target)
    while value.ndim < 2:
        value = value.unsqueeze(0)
    return torch.broadcast_to(value, target)


def _make_tree(
    initial,
    volume,
    *,
    edge_factor=None,
    edge_area=None,
    edge_distance=None,
    edge_diffusivity=1.0,
    process=_NamedTreeEdgeDiffusion,
):
    initial = torch.as_tensor(initial, dtype=DTYPE)
    if initial.ndim == 1:
        initial = initial.unsqueeze(0)
    rows = int(initial.shape[0])
    model = dn.Tree.from_graph(_branched_graph(), N=rows, v_init=-65.0, dtype=DTYPE)
    model.material(
        "solute",
        fields={"c": initial.clone()},
        min_values={"c": 0.0},
        domain="extracellular",
    )
    geometry = {"volume": torch.as_tensor(volume, dtype=DTYPE)}
    if edge_factor is not None:
        geometry["edge_factor"] = torch.as_tensor(edge_factor, dtype=DTYPE)
    else:
        geometry["edge_area"] = torch.as_tensor(edge_area, dtype=DTYPE)
        geometry["edge_distance"] = torch.as_tensor(edge_distance, dtype=DTYPE)
    model.register_material_geometry(
        "chemical_tree", domain="extracellular", **geometry
    )
    model.register_buffer(
        "edge_diffusivity", torch.as_tensor(edge_diffusivity, dtype=DTYPE)
    )
    return model


def _dense_backward_euler(
    concentration,
    volume,
    edge_factor,
    diffusivity,
    edge_index,
    *,
    D_location,
    dt=DT,
    node_mask=None,
):
    """Independent finite-volume solve in Tree storage order."""
    concentration = torch.as_tensor(concentration, dtype=DTYPE)
    if concentration.ndim == 1:
        concentration = concentration.unsqueeze(0)
    rows, compartments = concentration.shape
    parent, child = edge_index
    parent = parent.detach().cpu().tolist()
    child = child.detach().cpu().tolist()
    edges = len(parent)
    assert compartments == C and edges == E

    volume = _as_rows(volume, rows, compartments)
    edge_factor = _as_rows(edge_factor, rows, edges)
    if D_location == "node":
        node_D = _as_rows(diffusivity, rows, compartments)
    else:
        edge_D = _as_rows(diffusivity, rows, edges)
    if node_mask is None:
        node_mask = torch.ones_like(concentration, dtype=torch.bool)
    else:
        node_mask = _as_rows(node_mask, rows, compartments, dtype=torch.bool)

    answers = []
    for row in range(rows):
        # Identity rows make excluded nodes exact no-ops. Active zero-volume
        # junctions retain a zero storage row and are fixed by incident fluxes.
        matrix = torch.eye(compartments, dtype=DTYPE)
        rhs = concentration[row].clone()
        active = node_mask[row]
        for node in torch.nonzero(active, as_tuple=False).flatten().tolist():
            matrix[node].zero_()
            matrix[node, node] = volume[row, node]
            rhs[node] = volume[row, node] * concentration[row, node]

        for edge, (p, c) in enumerate(zip(parent, child)):
            if not bool(active[p] and active[c]):
                continue
            if D_location == "node":
                D_e = 0.5 * (node_D[row, p] + node_D[row, c])
            else:
                D_e = edge_D[row, edge]
            coupling = float(dt) * D_e * edge_factor[row, edge]
            matrix[p, p] += coupling
            matrix[c, c] += coupling
            matrix[p, c] -= coupling
            matrix[c, p] -= coupling
        answers.append(torch.linalg.solve(matrix, rhs))
    return torch.stack(answers)


def _initialize_and_step(model, *, dt=DT, training=False):
    model.train(training)
    model.initialize()
    model.step(dt=dt)
    return model.mech.materials["solute"].c


def test_tree_material_edges_are_compact_stable_and_root_agnostic():
    model = dn.Tree.from_graph(_branched_graph(), N=2, dtype=DTYPE)

    expected = torch.tensor([[3, 0, 0, 3], [0, 1, 2, 4]], dtype=torch.long)
    assert model.material_edge_index.shape == (2, E)
    assert model.material_edge_index.dtype == torch.long
    assert model.material_edge_index.device == model.diam.device
    assert torch.equal(model.material_edge_index, expected)

    parent, child = model.material_edges
    assert torch.equal(parent, expected[0])
    assert torch.equal(child, expected[1])
    assert model.diff_parent_index.tolist() == [3, 0, 0, -1, 3]

    # Topology has no population axis and remains integer-valued across dtype moves.
    model.to(dtype=torch.float32)
    assert model.material_edge_index.shape == (2, E)
    assert model.material_edge_index.dtype == torch.long
    assert torch.equal(model.material_edge_index.cpu(), expected)


def test_low_level_tree_constructor_compiles_stable_material_topology():
    graph = _branched_graph()
    model = dn.Tree(1, C, graph=graph, dtype=DTYPE)
    expected = torch.tensor([[3, 0, 0, 3], [0, 1, 2, 4]], dtype=torch.long)

    assert torch.equal(model.material_edge_index, expected)
    assert torch.equal(model.diff_parent_index, torch.tensor([3, 0, 0, -1, 3]))

    # Mutating the NetworkX interoperability view cannot recompile either
    # voltage or material topology on a constructed simulation object.
    replacement = _alternate_branched_graph()
    model.graph.remove_edges_from(list(model.graph.edges))
    model.graph.add_edges_from(replacement.edges(data=True))
    model.initialize()
    model.step(dt=DT)
    solver_order = model.integrator.solver_order.detach().cpu()
    parent_solver = model.integrator.parent_idx.detach().cpu()
    electrical_edges = []
    for child_solver, parent in enumerate(parent_solver.tolist()):
        if parent >= 0:
            electrical_edges.append(
                (int(solver_order[parent]), int(solver_order[child_solver]))
            )
    electrical_edges.sort(key=lambda edge: edge[1])
    assert electrical_edges == list(map(tuple, expected.T.tolist()))


def test_named_tree_node_diffusivity_matches_dense_oracle_with_nonzero_root():
    initial = torch.tensor([[8.0, 1.0, 4.0, 11.0, 2.0]], dtype=DTYPE)
    volume = torch.tensor([[0.7, 1.8, 0.4, 2.2, 1.1]], dtype=DTYPE)
    edge_factor = torch.tensor([[0.3, 1.7, 0.8, 2.2]], dtype=DTYPE)
    # Binary-exact values keep this geometry/topology oracle independent of
    # the established RANGE-parameter initialization precision contract.
    node_D = torch.tensor([[0.25, 1.0, 0.5, 2.0, 0.75]], dtype=DTYPE)
    with dn.ctx(DTYPE=DTYPE):
        model = _make_tree(
            initial,
            volume,
            edge_factor=edge_factor,
            edge_diffusivity=torch.ones(E),
            process=_NamedTreeNodeDiffusion,
        )
        model.insert(_NamedTreeNodeDiffusion, D=node_D)

    actual = _initialize_and_step(model)
    expected = _dense_backward_euler(
        initial,
        volume,
        edge_factor,
        node_D,
        model.material_edge_index,
        D_location="node",
    )
    torch.testing.assert_close(actual, expected, rtol=4e-13, atol=4e-13)


def test_named_tree_edge_diffusivity_batches_and_ignores_native_geometry():
    initial = torch.tensor(
        [[8.0, 1.0, 4.0, 11.0, 2.0], [0.3, 7.0, 2.2, 1.5, 9.0]],
        dtype=DTYPE,
    )
    volume = torch.tensor(
        [[0.7, 1.8, 0.4, 2.2, 1.1], [1.3, 0.5, 2.0, 0.9, 1.7]],
        dtype=DTYPE,
    )
    edge_area = torch.tensor([[0.15, 0.8, 0.2, 1.1], [0.4, 0.3, 1.2, 0.5]], dtype=DTYPE)
    edge_distance = torch.tensor(
        [[0.5, 0.4, 1.0, 2.0], [0.2, 1.5, 0.6, 0.25]], dtype=DTYPE
    )
    edge_D = torch.tensor([[0.2, 1.1, 0.6, 1.7], [1.3, 0.4, 2.0, 0.1]], dtype=DTYPE)
    model = _make_tree(
        initial,
        volume,
        edge_area=edge_area,
        edge_distance=edge_distance,
        edge_diffusivity=edge_D,
    )
    model.insert(_NamedTreeEdgeDiffusion)

    assert torch.count_nonzero(model.volume_o) == 0
    actual = _initialize_and_step(model)
    edge_factor = edge_area / edge_distance
    expected = _dense_backward_euler(
        initial,
        volume,
        edge_factor,
        edge_D,
        model.material_edge_index,
        D_location="edge",
    )
    torch.testing.assert_close(actual, expected, rtol=4e-13, atol=4e-13)

    # Each row uses its own named volume and edge coefficients, while topology
    # remains shared. Sealed diffusion conserves named-domain mass per row.
    torch.testing.assert_close(
        (volume * actual).sum(-1),
        (volume * initial).sum(-1),
        rtol=4e-13,
        atol=4e-13,
    )


def test_regional_tree_masks_omit_or_include_zero_volume_junction_explicitly():
    initial = torch.tensor([[7.0, 9.0, 1.0, 13.0, 17.0]], dtype=DTYPE)
    # Slot 0 is a chemical zero-volume junction, although it is an ordinary
    # positive-volume compartment in the native morphology.
    volume = torch.tensor([[0.0, 1.3, 0.8, 2.0, 1.1]], dtype=DTYPE)
    edge_factor = torch.tensor([[float("nan"), 1.2, 0.7, float("nan")]], dtype=DTYPE)
    edge_D = torch.tensor([[float("nan"), 0.5, 1.4, -123.0]], dtype=DTYPE)

    omitted = _make_tree(
        initial, volume, edge_factor=edge_factor, edge_diffusivity=edge_D
    )
    omitted[:, [1, 2]].insert(_NamedTreeEdgeDiffusion)
    omitted_actual = _initialize_and_step(omitted)
    # Siblings do not acquire a synthetic edge when their junction is omitted.
    torch.testing.assert_close(omitted_actual, initial, rtol=0.0, atol=0.0)

    included = _make_tree(
        initial, volume, edge_factor=edge_factor, edge_diffusivity=edge_D
    )
    included[:, [0, 1, 2]].insert(_NamedTreeEdgeDiffusion)
    included_actual = _initialize_and_step(included)
    mask = torch.tensor([[True, True, True, False, False]])
    expected = _dense_backward_euler(
        initial,
        volume,
        edge_factor,
        edge_D,
        included.material_edge_index,
        D_location="edge",
        node_mask=mask,
    )
    torch.testing.assert_close(included_actual, expected, rtol=4e-13, atol=4e-13)
    assert included_actual[0, 1] < initial[0, 1]
    assert included_actual[0, 2] > initial[0, 2]
    assert torch.equal(included_actual[:, [3, 4]], initial[:, [3, 4]])


def test_implicit_tree_accepts_anchored_zero_volume_junction_but_rejects_zero_mass_component():
    initial = torch.tensor([[7.0, 9.0, 1.0, 13.0, 17.0]], dtype=DTYPE)
    edge_factor = torch.ones((1, E), dtype=DTYPE)
    edge_D = torch.ones((1, E), dtype=DTYPE)

    anchored = _make_tree(
        initial,
        volume=torch.tensor([[0.0, 1.0, 2.0, 1.0, 1.0]], dtype=DTYPE),
        edge_factor=edge_factor,
        edge_diffusivity=edge_D,
    )
    anchored[:, [0, 1, 2]].insert(_NamedTreeEdgeDiffusion)
    actual = _initialize_and_step(anchored)
    assert torch.isfinite(actual).all()

    unanchored = _make_tree(
        initial,
        volume=torch.tensor([[0.0, 0.0, 2.0, 1.0, 1.0]], dtype=DTYPE),
        edge_factor=edge_factor,
        edge_diffusivity=edge_D,
    )
    unanchored[:, [0, 1]].insert(_NamedTreeEdgeDiffusion)
    with pytest.raises(
        ValueError, match="positive material volume|zero-volume component"
    ):
        _initialize_and_step(unanchored)


def test_explicit_named_tree_diffusion_rejects_active_zero_volume_nodes():
    model = _make_tree(
        initial=[[7.0, 9.0, 1.0, 13.0, 17.0]],
        volume=[[0.0, 1.0, 2.0, 1.0, 1.0]],
        edge_factor=torch.ones(E, dtype=DTYPE),
        edge_diffusivity=torch.ones(E, dtype=DTYPE),
        process=_NamedTreeExplicitEdgeDiffusion,
    )
    model[:, [0, 1]].insert(_NamedTreeExplicitEdgeDiffusion)
    with pytest.raises(RuntimeError, match="[Zz]ero-volume|positive volume"):
        _initialize_and_step(model)


def test_named_tree_geometry_and_edge_diffusivity_preserve_gradients_and_refresh():
    initial = torch.tensor([[8.0, 1.0, 4.0, 11.0, 2.0]], dtype=DTYPE)
    model = dn.Tree.from_graph(_branched_graph(), N=1, v_init=-65.0, dtype=DTYPE)
    model.material(
        "solute",
        fields={"c": initial.clone()},
        min_values={"c": 0.0},
        domain="extracellular",
    )
    model.register_parameter(
        "chemical_volume",
        torch.nn.Parameter(torch.tensor([[0.7, 1.8, 0.4, 2.2, 1.1]], dtype=DTYPE)),
    )
    model.register_parameter(
        "chemical_edge_factor",
        torch.nn.Parameter(torch.tensor([[0.3, 1.7, 0.8, 2.2]], dtype=DTYPE)),
    )
    model.register_parameter(
        "edge_diffusivity",
        torch.nn.Parameter(torch.tensor([[0.2, 1.1, 0.6, 1.7]], dtype=DTYPE)),
    )
    model.register_material_geometry(
        "chemical_tree",
        domain="extracellular",
        volume="chemical_volume",
        edge_factor="chemical_edge_factor",
    )
    model.insert(_NamedTreeEdgeDiffusion)

    actual = _initialize_and_step(model, training=True)
    expected = _dense_backward_euler(
        initial,
        model.chemical_volume,
        model.chemical_edge_factor,
        model.edge_diffusivity,
        model.material_edge_index,
        D_location="edge",
    )
    torch.testing.assert_close(actual, expected, rtol=4e-13, atol=4e-13)

    weights = torch.tensor([[0.3, -0.8, 1.1, 0.4, -0.2]], dtype=DTYPE)
    parameters = (
        model.chemical_volume,
        model.chemical_edge_factor,
        model.edge_diffusivity,
    )
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        assert torch.count_nonzero(actual_grad) > 0
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-12, atol=2e-12)

    first = actual.detach().clone()
    with torch.no_grad():
        model.chemical_edge_factor.mul_(1.7)
        model.edge_diffusivity.mul_(0.6)
    # Initialization reconstructs state and forces the next timestep to sample
    # the current registered geometry/coefficient values.
    model.initialize()
    refreshed = model.step(dt=DT).mech.materials["solute"].c
    refreshed_expected = _dense_backward_euler(
        initial,
        model.chemical_volume,
        model.chemical_edge_factor,
        model.edge_diffusivity,
        model.material_edge_index,
        D_location="edge",
    )
    torch.testing.assert_close(refreshed, refreshed_expected, rtol=4e-13, atol=4e-13)
    assert not torch.allclose(refreshed, first)


def test_named_tree_requires_compact_edge_shape():
    model = _make_tree(
        initial=[[8.0, 1.0, 4.0, 11.0, 2.0]],
        volume=torch.ones(C, dtype=DTYPE),
        # A child-indexed C-vector is not the public compact E-vector contract.
        edge_factor=torch.ones(C, dtype=DTYPE),
        edge_diffusivity=torch.ones(E, dtype=DTYPE),
    )
    model.insert(_NamedTreeEdgeDiffusion)
    with pytest.raises(ValueError, match="edge.*shape|compact|C - 1"):
        _initialize_and_step(model)


def test_edge_located_diffusivity_rejects_node_aligned_range_parameter():
    model = _make_tree(
        initial=[[8.0, 1.0, 4.0, 11.0, 2.0]],
        volume=torch.ones(C, dtype=DTYPE),
        edge_factor=torch.ones(E, dtype=DTYPE),
        edge_diffusivity=torch.ones(E, dtype=DTYPE),
        process=_InvalidRangeTreeEdgeDiffusion,
    )
    model.insert(_InvalidRangeTreeEdgeDiffusion, D=torch.ones(C, dtype=DTYPE))

    with pytest.raises(ValueError, match="Edge-located diffusivity.*RANGE"):
        _initialize_and_step(model)


def test_node_diffusivity_rejects_negative_active_endpoint_before_averaging():
    model = _make_tree(
        initial=[[8.0, 1.0, 4.0, 11.0, 2.0]],
        volume=torch.ones(C, dtype=DTYPE),
        edge_factor=torch.ones(E, dtype=DTYPE),
        edge_diffusivity=torch.ones(E, dtype=DTYPE),
        process=_NamedTreeNodeDiffusion,
    )
    # Edge 3 -> 0 would have a positive arithmetic mean despite the invalid
    # negative endpoint; diffusivity itself is required to be non-negative.
    node_D = torch.tensor([-1.0, 2.0, 2.0, 3.0, 2.0], dtype=DTYPE)
    model.insert(_NamedTreeNodeDiffusion, D=node_D)

    with pytest.raises(ValueError, match="incident.*non-negative diffusivity"):
        _initialize_and_step(model)


def test_named_tree_transport_uses_compiled_topology_after_graph_mutation():
    initial = torch.tensor([[8.0, 1.0, 4.0, 11.0, 2.0]], dtype=DTYPE)
    volume = torch.tensor([[0.7, 1.8, 0.4, 2.2, 1.1]], dtype=DTYPE)
    edge_factor = torch.tensor([[0.3, 1.7, 0.8, 2.2]], dtype=DTYPE)
    edge_D = torch.tensor([[0.2, 1.1, 0.6, 1.7]], dtype=DTYPE)
    model = _make_tree(
        initial,
        volume,
        edge_factor=edge_factor,
        edge_diffusivity=edge_D,
    )
    compiled_edges = model.material_edge_index.clone()
    model.insert(_NamedTreeEdgeDiffusion)

    # `graph` is retained for NetworkX interoperability and is mutable. Named
    # transport must remain tied to the immutable compiled morphology buffers.
    replacement = _alternate_branched_graph()
    model.graph.remove_edges_from(list(model.graph.edges))
    model.graph.add_edges_from(replacement.edges(data=True))

    actual = _initialize_and_step(model)
    expected = _dense_backward_euler(
        initial,
        volume,
        edge_factor,
        edge_D,
        compiled_edges,
        D_location="edge",
    )
    torch.testing.assert_close(actual, expected, rtol=4e-13, atol=4e-13)
    assert torch.equal(model.material_edge_index, compiled_edges)

    solver_order = model.integrator.solver_order.detach().cpu()
    parent_solver = model.integrator.parent_idx.detach().cpu()
    electrical_edges = []
    for child_solver, parent in enumerate(parent_solver.tolist()):
        if parent >= 0:
            electrical_edges.append(
                (int(solver_order[parent]), int(solver_order[child_solver]))
            )
    electrical_edges.sort(key=lambda edge: edge[1])
    assert electrical_edges == list(map(tuple, compiled_edges.T.tolist()))


def test_native_tree_transport_uses_compiled_geometry_after_graph_mutation():
    initial = torch.tensor([[8.0, 1.0, 4.0, 11.0, 2.0]], dtype=DTYPE)
    model = dn.Tree.from_graph(_branched_graph(), N=1, v_init=-65.0, dtype=DTYPE)
    model.material(
        "solute",
        fields={"c": initial.clone()},
        min_values={"c": 0.0},
        domain="intracellular",
    )
    model.insert(_NativeTreeDiffusion)
    compiled_edges = model.material_edge_index.clone()
    compiled_factor = model.diff_geom_um.index_select(-1, compiled_edges[1])

    replacement = _alternate_branched_graph()
    model.graph.remove_edges_from(list(model.graph.edges))
    model.graph.add_edges_from(replacement.edges(data=True))

    actual = _initialize_and_step(model)
    expected = _dense_backward_euler(
        initial,
        model.volume_i,
        compiled_factor,
        torch.tensor(0.75, dtype=DTYPE),
        compiled_edges,
        D_location="node",
    )
    torch.testing.assert_close(actual, expected, rtol=4e-13, atol=4e-13)


def test_tree_state_dict_rejects_different_compiled_topology():
    source = dn.Tree.from_graph(_branched_graph(), N=1, dtype=DTYPE)
    target = dn.Tree.from_graph(_alternate_branched_graph(), N=1, dtype=DTYPE)

    with pytest.raises(RuntimeError, match="different.*compiled material topology"):
        target.load_state_dict(source.state_dict())

    # Matching compiled topology remains a normal state-dict round trip.
    matching = dn.Tree.from_graph(_branched_graph(), N=1, dtype=DTYPE)
    result = matching.load_state_dict(source.state_dict())
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_extcell_tree_retains_and_protects_compiled_topology():
    source = dn.ExtCellTree.from_graph(_branched_graph(), N=1, dtype=DTYPE)
    target = dn.ExtCellTree.from_graph(_alternate_branched_graph(), N=1, dtype=DTYPE)

    assert source.compartment_graph is not None
    assert target.compartment_graph is not None
    with pytest.raises(RuntimeError, match="different.*compiled material topology"):
        target.load_state_dict(source.state_dict())


def test_direct_parameter_geometry_is_optimizer_discoverable():
    model = dn.Tree.from_graph(_branched_graph(), N=1, dtype=DTYPE)
    volume = torch.nn.Parameter(torch.ones((1, C), dtype=DTYPE))
    factor = torch.nn.Parameter(torch.ones((1, E), dtype=DTYPE))
    model.register_material_geometry(
        "direct_parameter_geometry",
        domain="extracellular",
        volume=volume,
        edge_factor=factor,
    )

    config = model.material_geometry("direct_parameter_geometry")
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assert parameters[config["volume"]] is volume
    assert parameters[config["edge_factor"]] is factor
    assert config["volume"] not in buffers
    assert config["edge_factor"] not in buffers


@pytest.mark.parametrize("clone_kind", ("deepcopy", "pickleable"))
def test_clone_sanitizes_differentiable_geometry_caches_and_rebuilds(clone_kind):
    initial = torch.tensor([[8.0, 1.0, 4.0, 11.0, 2.0]], dtype=DTYPE)
    model = dn.Tree.from_graph(_branched_graph(), N=1, v_init=-65.0, dtype=DTYPE)
    model.material(
        "solute",
        fields={"c": initial.clone()},
        min_values={"c": 0.0},
        domain="extracellular",
    )
    model.register_parameter(
        "chemical_volume", torch.nn.Parameter(torch.ones((1, C), dtype=DTYPE))
    )
    model.register_parameter(
        "chemical_edge_factor",
        torch.nn.Parameter(torch.ones((1, E), dtype=DTYPE)),
    )
    model.register_parameter(
        "edge_diffusivity",
        torch.nn.Parameter(torch.full((1, E), 0.5, dtype=DTYPE)),
    )
    model.register_material_geometry(
        "chemical_tree",
        domain="extracellular",
        volume="chemical_volume",
        edge_factor="chemical_edge_factor",
    )
    model.insert(_NamedTreeEdgeDiffusion)
    _initialize_and_step(model, training=True)

    cloned = (
        copy.deepcopy(model)
        if clone_kind == "deepcopy"
        else model.pickleable(clone=True)
    )
    process = next(iter(cloned.mech.material_processes.values()))
    assert process._mp_population is cloned
    assert process._spatial_configured is False
    assert cloned.mech.materials["solute"].c.grad_fn is None

    result = cloned.step(dt=DT).mech.materials["solute"].c
    gradients = torch.autograd.grad(
        result.square().sum(),
        (
            cloned.chemical_volume,
            cloned.chemical_edge_factor,
            cloned.edge_diffusivity,
        ),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
