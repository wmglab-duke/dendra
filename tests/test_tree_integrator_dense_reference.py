import networkx as nx
import pytest
import torch

from dendra.models.integrators.tree import (
    DENDRA_SOLVERS_AVAILABLE,
    _dhs,
    build_dhs_layers,
    build_morphology,
    graph_to_parent_and_axial,
)
from dendra.models.integrators.tree_bt import (
    _dhs_bt,
    _topo_parent_depth,
    assemble_rhs,
)

DTYPE = torch.float64
pytestmark = pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="CPU dense-reference tests require dendra_solvers",
)


def _graph(edges, *, nodes=None, resistances=None):
    if nodes is None:
        node_set = {node for edge in edges for node in edge}
        nodes = sorted(node_set) if node_set else [0]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    resistances = resistances or [2.0e8 + i * 0.75e8 for i in range(len(edges))]
    for edge, resistance in zip(edges, resistances):
        graph.add_edge(*edge, R_ohm=float(resistance))
    return graph


TOPOLOGIES = [
    pytest.param(_graph([], nodes=[0]), id="one-node"),
    pytest.param(_graph([(0, 1), (1, 2), (2, 3)]), id="chain"),
    pytest.param(
        _graph([(2, 0), (2, 1), (0, 3)], nodes=[3, 1, 0, 2]),
        id="branched-reordered",
    ),
]


class TreeModel(torch.nn.Module):
    def __init__(self, graph, *, shape=None, graphs=None):
        super().__init__()
        self.graph = graph if graphs is None else graphs
        compartments = graph.number_of_nodes()
        self.shape = tuple(shape or (1, compartments))
        self.celsius = 34.0
        self.area_scale = 1.0
        self.cm_scale = 1.0
        self.rhoa_scale = 1.0
        self.v_init = -65.0

        self.jit = False
        self.jit_network_solves = False
        self.jit_network_ops = False
        self.backend = "inductor"
        self.fullgraph = False
        self.dynamic = False
        self.compile_mode = None
        self.compile_options = None

        base = torch.arange(compartments, dtype=DTYPE)
        area = (0.7e-5 + base * 0.13e-5).expand(self.shape).clone()
        cm = (0.8 + base * 0.17).expand(self.shape).clone()
        v = (-68.0 + base * 2.25).expand(self.shape).clone()
        self.register_buffer("area", area)
        self.register_buffer("cm", cm)
        self.register_buffer("v", v)

    def device(self):
        return self.v.device

    def dtype(self):
        return self.v.dtype

    def expanded_v_init(self):
        return torch.as_tensor(self.v_init, dtype=self.v.dtype).expand_as(self.v)


class BlockTreeModel(TreeModel):
    def __init__(self, graph, *, shape=None):
        super().__init__(graph, shape=shape)
        compartments = graph.number_of_nodes()
        base = torch.arange(compartments, dtype=DTYPE)
        dx = (8.0 + 2.5 * base).expand(self.shape).clone()
        shell_shape = self.shape + (2,)
        xraxial = torch.empty(shell_shape, dtype=DTYPE)
        xraxial[..., 0] = (2.0 + 0.2 * base).expand(self.shape)
        xraxial[..., 1] = (3.0 + 0.3 * base).expand(self.shape)
        xc = torch.empty_like(xraxial)
        xc[..., 0] = (0.25 + 0.03 * base).expand(self.shape)
        xc[..., 1] = (0.15 + 0.02 * base).expand(self.shape)
        xg = torch.empty_like(xraxial)
        xg[..., 0] = (1.5e-4 + 0.1e-4 * base).expand(self.shape)
        xg[..., 1] = (2.0e-4 + 0.2e-4 * base).expand(self.shape)
        self.register_buffer("dx", dx)
        self.register_buffer("xraxial", xraxial)
        self.register_buffer("xc", xc)
        self.register_buffer("xg", xg)


class LinearMechanism(torch.nn.Module):
    def __init__(self, conductance=0.0, reversal=-52.0, *, currents=True):
        super().__init__()
        self.conductance = float(conductance)
        self.reversal = float(reversal)
        self.currents = currents
        self.advance_calls = 0
        self.dt = None

    def update_v(self, voltage):
        return voltage

    def advance(self, voltage, dt, temp):
        self.advance_calls += 1

    def i(self, voltage):
        conductance = torch.full_like(voltage, self.conductance)
        return conductance * (voltage - self.reversal), conductance

    def set_dt(self, dt):
        self.dt = float(dt)


def _dense_hines(main, axial, rhs, parent):
    batch, compartments = main.shape
    matrix = torch.diag_embed(main)
    for child, parent_node in enumerate(parent.tolist()):
        if parent_node < 0:
            continue
        edge = axial[:, child]
        matrix[:, child, child] += edge
        matrix[:, parent_node, parent_node] += edge
        matrix[:, child, parent_node] -= edge
        matrix[:, parent_node, child] -= edge
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


def _dense_block_hines(main, axial, rhs, parent):
    batch, compartments, block, _ = main.shape
    matrix = torch.zeros(
        batch,
        compartments * block,
        compartments * block,
        dtype=main.dtype,
        device=main.device,
    )
    for node in range(compartments):
        node_slice = slice(node * block, (node + 1) * block)
        matrix[:, node_slice, node_slice] = main[:, node]
    for child, parent_node in enumerate(parent.tolist()):
        if parent_node < 0:
            continue
        child_slice = slice(child * block, (child + 1) * block)
        parent_slice = slice(parent_node * block, (parent_node + 1) * block)
        edge = torch.diag_embed(axial[:, child])
        matrix[:, child_slice, child_slice] += edge
        matrix[:, parent_slice, parent_slice] += edge
        matrix[:, child_slice, parent_slice] -= edge
        matrix[:, parent_slice, child_slice] -= edge
    flat_rhs = rhs.reshape(batch, compartments * block, 1)
    return torch.linalg.solve(matrix, flat_rhs).reshape_as(rhs)


def _scalar_dense_reference(integrator, voltage, mechanism, intra=None, ve=None):
    batch_voltage = voltage.reshape(-1, integrator.K)
    current, conductance = mechanism.i(voltage)
    conductance = conductance.reshape_as(batch_voltage)
    current = current.reshape_as(batch_voltage)
    scale = integrator.scale.reshape_as(batch_voltage)
    rhs = (conductance * batch_voltage - current) * scale
    rhs = rhs + integrator.cmdt * batch_voltage

    if intra is not None:
        rhs = rhs + torch.broadcast_to(intra, voltage.shape).reshape_as(rhs)
    if ve is not None:
        ve_flat = torch.broadcast_to(ve, voltage.shape).reshape_as(rhs)
        edge = integrator.edge_gax_orig * (
            ve_flat.index_select(1, integrator.edge_child_orig)
            - ve_flat.index_select(1, integrator.edge_parent_orig)
        )
        rhs = rhs.clone()
        rhs.scatter_add_(1, integrator.edge_child_orig.expand_as(edge), -edge)
        rhs.scatter_add_(1, integrator.edge_parent_orig.expand_as(edge), edge)

    main = integrator.cmdt + conductance * scale
    order = integrator.solver_order
    solved = _dense_hines(
        main.index_select(1, order),
        integrator.a_geom,
        rhs.index_select(1, order),
        integrator.parent_idx,
    )
    return solved.index_select(1, integrator.inv_solver_order).reshape_as(voltage)


def _block_rhs(integrator, vc, d_lin, e_ext=None):
    c_rad = torch.cat([integrator.cm_dt.unsqueeze(-1), integrator.xc_dt], dim=-1)
    rhs = torch.zeros_like(vc)
    capacitive = c_rad[..., :-1] * (vc[..., :-1] - vc[..., 1:])
    rhs[..., :-1] += capacitive
    rhs[..., 1:] -= capacitive
    rhs[..., 0] += d_lin
    rhs[..., 1] -= d_lin
    rhs[..., -1] += c_rad[..., -1] * vc[..., -1]
    if e_ext is not None:
        rhs[..., -1] += integrator.xg[..., -1] * e_ext
    return rhs


def _block_dense_reference(integrator, vc, voltage, mechanism, intra=None, ve=None):
    flat_voltage = voltage.reshape(-1, integrator.K)
    current, conductance = mechanism.i(voltage)
    current = current.reshape_as(flat_voltage)
    conductance = conductance.reshape_as(flat_voltage)
    d_lin = (conductance * flat_voltage - current) * integrator.area
    if intra is not None:
        d_lin = d_lin + torch.broadcast_to(intra, voltage.shape).reshape_as(d_lin)
    ve_flat = None
    if ve is not None:
        ve_flat = torch.broadcast_to(ve, voltage.shape).reshape_as(flat_voltage)
    rhs = _block_rhs(integrator, vc, d_lin, ve_flat)
    order = integrator.solver_order
    rhs = rhs.index_select(1, order)
    conductance = (conductance * integrator.area).index_select(1, order)
    main = integrator.main_blocks.clone()
    main[..., 0, 0] += conductance
    main[..., 1, 1] += conductance
    main[..., 0, 1] -= conductance
    main[..., 1, 0] -= conductance
    solved = _dense_block_hines(
        main, integrator.g_to_parent, rhs, integrator.parent_idx
    )
    return solved.index_select(1, integrator.inv_solver_order).reshape(
        integrator.base_shape
    )


def test_tree_topology_helpers_return_known_parent_depth_and_layers():
    parent, children, depth = build_morphology([-1, 0, 0, 1, 1])
    assert parent.tolist() == [-1, 0, 0, 1, 1]
    assert children == [[1, 2], [3, 4], [], [], []]
    assert depth.tolist() == [0, 1, 1, 2, 2]

    order, layer_ptr = build_dhs_layers(depth, 2)
    assert order.tolist() == [3, 4, 1, 2, 0]
    assert layer_ptr.tolist() == [0, 2, 4, 5]


@pytest.mark.parametrize(
    "parents, match",
    [
        ([], "at least one"),
        ([-1, -1], "exactly one root"),
        ([1, 0], "exactly one root"),
        ([-1, 7], "parent index"),
        ([-1, 1], "own parent"),
        ([-1, 2, 1], "cycle|unreachable"),
    ],
)
def test_build_morphology_rejects_invalid_parent_arrays(parents, match):
    with pytest.raises(ValueError, match=match):
        build_morphology(parents)


@pytest.mark.parametrize("threads", [0, -16, 3, 33, 1.5, True])
def test_tree_integrators_reject_invalid_thread_counts(threads):
    graph = _graph([(0, 1)])
    model = TreeModel(graph)
    mechanism = LinearMechanism()
    with pytest.raises((TypeError, ValueError), match="threads"):
        _dhs(model, mechanism, threads=threads)
    with pytest.raises((TypeError, ValueError), match="threads"):
        _dhs_bt(BlockTreeModel(graph), LinearMechanism(), threads=threads)


def test_graph_conversion_accepts_single_graph_and_preserves_float64_geometry():
    resistance = 123_456_789.12345679
    graph = _graph([(0, 1)], resistances=[resistance])
    parent, axial, nodes = graph_to_parent_and_axial(graph, dtype_axial=DTYPE)
    assert parent.tolist() == [-1, 0]
    assert nodes == [0, 1]
    assert axial.dtype == DTYPE
    assert axial.shape == (1, 2)
    assert axial[0, 1].item() == pytest.approx(1.0 / resistance, rel=1e-14)


def test_graph_conversion_supports_heterogeneous_geometry_with_shared_topology():
    first = _graph([(2, 0), (2, 1), (0, 3)], nodes=[3, 1, 0, 2])
    second = _graph(
        [(2, 0), (2, 1), (0, 3)],
        nodes=[0, 1, 2, 3],
        resistances=[5.0e8, 6.0e8, 7.0e8],
    )
    parent, axial, nodes = graph_to_parent_and_axial([first, second], DTYPE)
    assert nodes == [2, 0, 1, 3]
    assert parent.tolist() == [-1, 0, 0, 1]
    assert axial.shape == (2, 4)
    assert not torch.equal(axial[0], axial[1])


def test_graph_conversion_rejects_nonidentical_or_non_tree_topologies():
    chain = _graph([(0, 1), (1, 2)])
    relabeled_chain = _graph([(2, 1), (1, 0)])
    with pytest.raises(ValueError, match="labeled topology"):
        graph_to_parent_and_axial([chain, relabeled_chain])

    with pytest.raises(ValueError, match="exactly one root"):
        graph_to_parent_and_axial(_graph([(0, 1)], nodes=[0, 1, 2]))

    multi_parent = _graph([(0, 2), (1, 2)], nodes=[0, 1, 2])
    with pytest.raises(ValueError, match="parents"):
        graph_to_parent_and_axial(multi_parent)

    cyclic = _graph([(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="acyclic"):
        graph_to_parent_and_axial(cyclic)


@pytest.mark.parametrize("graph", TOPOLOGIES)
def test_scalar_tree_step_matches_dense_reference(graph):
    model = TreeModel(graph)
    mechanism = LinearMechanism(conductance=2.7e-4, reversal=-49.0)
    integrator = _dhs(model, mechanism, threads=2)
    integrator._initialize(model, 0.037)
    intra = torch.linspace(0.1, 0.1 * model.shape[-1], model.shape[-1], dtype=DTYPE)

    actual, imem = integrator._step(model.v, 0.037, model.celsius, intra=intra)
    expected = _scalar_dense_reference(integrator, model.v, mechanism, intra=intra)
    assert imem is None
    assert torch.allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_scalar_tree_supports_heterogeneous_batched_geometry():
    edges = [(0, 1), (0, 2), (2, 3)]
    first = _graph(edges, resistances=[2.0e8, 3.0e8, 4.0e8])
    second = _graph(edges, resistances=[7.0e8, 5.0e8, 9.0e8])
    model = TreeModel(first, shape=(2, 4), graphs=[first, second])
    model.v[1] += torch.tensor([4.0, -1.0, 2.0, -3.0], dtype=DTYPE)
    mechanism = LinearMechanism(conductance=1.1e-4)
    integrator = _dhs(model, mechanism, threads=4)
    integrator._initialize(model, 0.025)

    actual, _ = integrator._step(model.v, 0.025, model.celsius)
    expected = _scalar_dense_reference(integrator, model.v, mechanism)
    assert integrator.a_geom.shape == (2, 4)
    assert not torch.equal(integrator.a_geom[0], integrator.a_geom[1])
    assert torch.allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_scalar_tree_flattens_leading_batches_and_matches_dense_gradients():
    graph = _graph([(0, 1), (0, 2), (2, 3)])
    model = TreeModel(graph, shape=(2, 3, 4))
    mechanism = LinearMechanism(conductance=3.0e-4)
    integrator = _dhs(model, mechanism, threads=2)
    integrator._initialize(model, 0.05)
    voltage = model.v.detach().clone().requires_grad_()
    intra = torch.linspace(0.02, 0.08, 4, dtype=DTYPE).requires_grad_()

    actual, _ = integrator._step(voltage, 0.05, model.celsius, intra=intra)
    expected = _scalar_dense_reference(integrator, voltage, mechanism, intra=intra)
    assert actual.shape == model.shape
    assert torch.allclose(actual, expected, rtol=2e-12, atol=2e-12)

    actual_grad = torch.autograd.grad(actual.square().sum(), (voltage, intra))
    expected_grad = torch.autograd.grad(expected.square().sum(), (voltage, intra))
    for got, want in zip(actual_grad, expected_grad):
        assert torch.allclose(got, want, rtol=2e-11, atol=2e-11)


def test_scalar_tree_uniform_extracellular_field_is_gauge_invariant():
    graph = _graph([(0, 1), (0, 2), (2, 3)])
    model = TreeModel(graph, shape=(2, 4))
    mechanism = LinearMechanism(conductance=2.0e-4)
    integrator = _dhs(model, mechanism, threads=2)
    integrator._initialize(model, 0.05)

    baseline, _ = integrator._step(model.v, 0.05, model.celsius)
    shifted, _ = integrator._step(
        model.v, 0.05, model.celsius, ve=torch.full((4,), 123.0, dtype=DTYPE)
    )
    nonuniform = torch.tensor([0.0, 2.0, -1.0, 4.0], dtype=DTYPE)
    driven, _ = integrator._step(model.v, 0.05, model.celsius, ve=nonuniform)
    expected = _scalar_dense_reference(integrator, model.v, mechanism, ve=nonuniform)
    assert torch.allclose(shifted, baseline, rtol=0.0, atol=2e-13)
    assert not torch.allclose(driven, baseline)
    assert torch.allclose(driven, expected, rtol=2e-12, atol=2e-12)


def test_scalar_tree_extracellular_field_has_neuron_polarity():
    graph = _graph([(0, 1)])
    model = TreeModel(graph)
    model.v.fill_(-65.0)
    mechanism = LinearMechanism(conductance=0.0)
    integrator = _dhs(model, mechanism, threads=2)
    integrator._initialize(model, 0.05)

    baseline, _ = integrator._step(model.v, 0.05, model.celsius)
    driven, _ = integrator._step(
        model.v,
        0.05,
        model.celsius,
        ve=torch.tensor([0.0, 5.0], dtype=DTYPE),
    )

    # NEURON's extracellular mechanism defines v as transmembrane voltage and
    # the intracellular potential as v + vext.  Raising vext at the child
    # therefore sends axial current toward the parent: parent depolarizes and
    # child hyperpolarizes.
    assert torch.equal(baseline, model.v)
    assert driven[0, 0] > baseline[0, 0]
    assert driven[0, 1] < baseline[0, 1]


def test_scalar_tree_imem_matches_discrete_membrane_balance():
    graph = _graph([(0, 1), (0, 2)])
    model = TreeModel(graph)
    mechanism = LinearMechanism(conductance=1.7e-4)
    integrator = _dhs(model, mechanism, imem=True, threads=2)
    integrator._initialize(model, 0.04)
    actual, imem = integrator._step(model.v, 0.04, model.celsius)
    current, conductance = mechanism.i(model.v)
    scale = integrator.scale.reshape_as(model.v)
    main = integrator.cmdt.reshape_as(model.v) + conductance * scale
    expected_imem = main * (actual - model.v) + current * scale
    assert torch.allclose(imem, expected_imem, rtol=2e-12, atol=2e-12)


def test_block_rhs_matches_manual_radial_balance_and_outer_drive():
    previous = torch.tensor([[[3.0, 1.0, -2.0], [4.0, -1.0, 0.5]]], dtype=DTYPE)
    capacitance = torch.tensor([[[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]], dtype=DTYPE)
    drive = torch.tensor([[17.0, 19.0]], dtype=DTYPE)
    xg = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=DTYPE)
    e_ext = torch.tensor([[23.0, 29.0]], dtype=DTYPE)
    expected = _block_rhs(
        type(
            "Buffers",
            (),
            {"cm_dt": capacitance[..., 0], "xc_dt": capacitance[..., 1:], "xg": xg},
        )(),
        previous,
        drive,
        e_ext,
    )
    assert torch.equal(assemble_rhs(previous, capacitance, drive, xg, e_ext), expected)


@pytest.mark.parametrize("graph", TOPOLOGIES)
def test_block_tree_step_matches_dense_reference(graph):
    model = BlockTreeModel(graph)
    mechanism = LinearMechanism(conductance=2.2e-4, reversal=-51.0)
    integrator = _dhs_bt(model, mechanism, threads=2)
    integrator._initialize(model, 0.03)
    compartments = model.shape[-1]
    vc = torch.empty(model.shape + (3,), dtype=DTYPE)
    vc[..., 0] = model.v
    vc[..., 1] = torch.linspace(-2.0, 1.0, compartments, dtype=DTYPE)
    vc[..., 2] = torch.linspace(0.5, -1.5, compartments, dtype=DTYPE)
    voltage = vc[..., 0] - vc[..., 1]
    intra = torch.linspace(0.01, 0.04, compartments, dtype=DTYPE)
    external = torch.linspace(-3.0, 5.0, compartments, dtype=DTYPE)

    actual_vc, actual_v = integrator._step(
        vc, voltage, 0.03, model.celsius, ve=external, intra=intra
    )
    expected_vc = _block_dense_reference(
        integrator, vc, voltage, mechanism, intra=intra, ve=external
    )
    assert torch.allclose(actual_vc, expected_vc, rtol=2e-10, atol=5e-10)
    assert torch.allclose(actual_v, expected_vc[..., 0] - expected_vc[..., 1])


def test_block_tree_supports_batched_inputs_and_dense_gradients():
    graph = _graph([(2, 0), (2, 1), (0, 3)], nodes=[3, 1, 0, 2])
    model = BlockTreeModel(graph, shape=(2, 3, 4))
    mechanism = LinearMechanism(conductance=1.9e-4)
    integrator = _dhs_bt(model, mechanism, threads=2)
    integrator._initialize(model, 0.045)
    vc = torch.zeros(model.shape + (3,), dtype=DTYPE)
    vc[..., 0] = model.v
    vc[..., 1] = torch.tensor([-1.0, 0.5, 2.0, -0.5], dtype=DTYPE)
    vc[..., 2] = torch.tensor([0.5, -0.75, 1.0, 0.25], dtype=DTYPE)
    vc.requires_grad_()
    voltage = (vc[..., 0] - vc[..., 1]).requires_grad_()
    intra = torch.tensor([0.02, -0.01, 0.04, 0.03], dtype=DTYPE).requires_grad_()
    external = torch.tensor([-3.0, 1.0, 4.0, -2.0], dtype=DTYPE).requires_grad_()

    actual, _ = integrator._step(
        vc.reshape(-1, integrator.K, 3),
        voltage,
        0.045,
        model.celsius,
        ve=external,
        intra=intra,
    )
    expected = _block_dense_reference(
        integrator,
        vc.reshape(-1, integrator.K, 3),
        voltage,
        mechanism,
        intra=intra,
        ve=external,
    )
    assert actual.shape == model.shape + (3,)
    assert torch.allclose(actual, expected, rtol=2e-10, atol=5e-10)
    inputs = (vc, voltage, intra, external)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for got, want in zip(actual_grad, expected_grad):
        assert torch.allclose(got, want, rtol=2e-9, atol=2e-8)


def test_block_tree_zero_optional_drives_match_none_and_init_v_resets_state():
    graph = _graph([(0, 1), (0, 2)])
    model = BlockTreeModel(graph, shape=(2, 3))
    mechanism = LinearMechanism(conductance=1.0e-4)
    integrator = _dhs_bt(model, mechanism, imem=True, threads=2)
    integrator._initialize(model, 0.05)
    baseline = integrator._step(model.vc, model.v, 0.05, model.celsius)[0]
    zeros = torch.zeros(3, dtype=DTYPE)
    with_zeros = integrator._step(
        model.vc, model.v, 0.05, model.celsius, ve=zeros, intra=zeros
    )[0]
    assert torch.allclose(with_zeros, baseline, rtol=0.0, atol=0.0)

    model.vc.fill_(42.0)
    model.v.fill_(7.0)
    integrator.init_v(model)
    assert torch.equal(model.v, torch.full_like(model.v, model.v_init))
    assert torch.equal(model.vc[..., 0], model.v)
    assert torch.count_nonzero(model.vc[..., 1:]) == 0
    assert torch.count_nonzero(model.i_membrane) == 0


def test_topology_helper_rejects_forest_multi_parent_and_cycle():
    with pytest.raises(ValueError, match="exactly one root"):
        _topo_parent_depth(_graph([(0, 1)], nodes=[0, 1, 2]))
    with pytest.raises(ValueError, match="parents"):
        _topo_parent_depth(_graph([(0, 2), (1, 2)], nodes=[0, 1, 2]))
    with pytest.raises(ValueError, match="acyclic"):
        _topo_parent_depth(_graph([(0, 1), (1, 0)]))
