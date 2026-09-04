import math

import networkx as nx
import pytest
import torch

from dendra.models.integrators.tree import (
    DENDRA_SOLVERS_AVAILABLE,
    _dhs,
    build_dhs_layers,
    graph_to_parent_and_axial,
)
from dendra.models.integrators.tree_bt import _dhs_bt

DTYPE = torch.float64
pytestmark = pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="CPU dense-reference tests require dendra_solvers",
)


def _graph(edges, resistances, *, node_order):
    graph = nx.DiGraph()
    graph.add_nodes_from(node_order)
    for (parent, child), resistance in zip(edges, resistances):
        graph.add_edge(parent, child, R_ohm=float(resistance))
    return graph


EDGES = [(2, 0), (2, 1), (0, 3), (0, 4)]


class _TreeModel(torch.nn.Module):
    def __init__(self, graph, *, shape=(2, 5)):
        super().__init__()
        self.graph = graph
        self.shape = tuple(shape)
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

        compartments = self.shape[-1]
        base = torch.arange(compartments, dtype=DTYPE)
        lead = math.prod(self.shape[:-1])
        row = torch.arange(lead, dtype=DTYPE).reshape(*self.shape[:-1], 1)
        self.register_buffer(
            "area", (0.9e-5 + 0.11e-5 * base + 0.03e-5 * row).expand(self.shape).clone()
        )
        self.register_buffer(
            "cm", (0.75 + 0.09 * base + 0.04 * row).expand(self.shape).clone()
        )
        self.register_buffer(
            "v", (-72.0 + 2.1 * base + 0.7 * row).expand(self.shape).clone()
        )

    def device(self):
        return self.v.device

    def dtype(self):
        return self.v.dtype

    def expanded_v_init(self):
        return torch.as_tensor(self.v_init, dtype=self.v.dtype).expand_as(self.v)


class _BlockTreeModel(_TreeModel):
    def __init__(self, graph, *, shape=(2, 5)):
        super().__init__(graph, shape=shape)
        compartments = self.shape[-1]
        base = torch.arange(compartments, dtype=DTYPE)
        lead = math.prod(self.shape[:-1])
        row = torch.arange(lead, dtype=DTYPE).reshape(*self.shape[:-1], 1)
        self.register_buffer(
            "dx", (7.0 + 1.7 * base + 0.3 * row).expand(self.shape).clone()
        )
        shell_shape = self.shape + (2,)
        xraxial = torch.empty(shell_shape, dtype=DTYPE)
        xraxial[..., 0] = (1.8 + 0.17 * base + 0.05 * row).expand(self.shape)
        xraxial[..., 1] = (2.7 + 0.23 * base + 0.07 * row).expand(self.shape)
        xc = torch.empty_like(xraxial)
        xc[..., 0] = (0.22 + 0.025 * base + 0.01 * row).expand(self.shape)
        xc[..., 1] = (0.14 + 0.019 * base + 0.008 * row).expand(self.shape)
        xg = torch.empty_like(xraxial)
        xg[..., 0] = (1.4e-4 + 0.12e-4 * base + 0.03e-4 * row).expand(self.shape)
        xg[..., 1] = (2.1e-4 + 0.16e-4 * base + 0.04e-4 * row).expand(self.shape)
        self.register_buffer("xraxial", xraxial)
        self.register_buffer("xc", xc)
        self.register_buffer("xg", xg)


class _LinearMechanism(torch.nn.Module):
    def __init__(self, conductance=0.0, reversal=-51.0, *, currents=True):
        super().__init__()
        self.conductance = float(conductance)
        self.reversal = float(reversal)
        self.currents = currents

    def update_v(self, voltage):
        return voltage

    def advance(self, voltage, dt, temp):
        pass

    def i(self, voltage):
        conductance = torch.full_like(voltage, self.conductance)
        return conductance * (voltage - self.reversal), conductance

    def set_dt(self, dt):
        pass


def _graphs_for_batched_scalar():
    first_resistances = [
        9_000_000.1234567,
        13_000_000.7654321,
        17_000_000.2468135,
        21_000_000.9753186,
    ]
    second_resistances = [
        7_000_000.3141592,
        11_000_000.2718281,
        19_000_000.1618034,
        23_000_000.1414213,
    ]
    return [
        _graph(EDGES, first_resistances, node_order=[4, 1, 0, 3, 2]),
        _graph(EDGES, second_resistances, node_order=[3, 2, 4, 0, 1]),
    ]


def _add_edge_laplacian(matrix, graph, *, channel=0, shell_g=None, rhoa_scale=1.0):
    for parent, child, data in graph.edges(data=True):
        if shell_g is None:
            child_scale = (
                rhoa_scale[child] if torch.is_tensor(rhoa_scale) else rhoa_scale
            )
            conductance = 1.0 / (data["R_ohm"] * child_scale)
        else:
            conductance = shell_g[child]
        p = (
            3 * parent + channel
            if matrix.shape[-1] != graph.number_of_nodes()
            else parent
        )
        c = (
            3 * child + channel
            if matrix.shape[-1] != graph.number_of_nodes()
            else child
        )
        matrix[p, p] += conductance
        matrix[c, c] += conductance
        matrix[p, c] -= conductance
        matrix[c, p] -= conductance


def _dense_scalar_reference(model, mechanism, voltage, dt, intra, extracellular):
    graphs = model.graph if isinstance(model.graph, list) else [model.graph]
    flat_voltage = voltage.reshape(-1, model.shape[-1])
    flat_area = model.area.reshape_as(flat_voltage) * model.area_scale
    flat_cm = model.cm.reshape_as(flat_voltage)
    flat_intra = torch.broadcast_to(intra, voltage.shape).reshape_as(flat_voltage)
    flat_ve = torch.broadcast_to(extracellular, voltage.shape).reshape_as(flat_voltage)
    flat_rhoa_scale = torch.broadcast_to(
        torch.as_tensor(model.rhoa_scale, dtype=voltage.dtype), voltage.shape
    ).reshape_as(flat_voltage)
    output = []
    for row, graph in enumerate(graphs):
        area = flat_area[row]
        cmdt = 1e-6 * flat_cm[row] * area * model.cm_scale / (dt * 1e-3)
        membrane_g = mechanism.conductance * area
        matrix = torch.diag(cmdt + membrane_g)
        _add_edge_laplacian(matrix, graph, rhoa_scale=flat_rhoa_scale[row])
        rhs = (
            cmdt * flat_voltage[row] + membrane_g * mechanism.reversal + flat_intra[row]
        )
        for parent, child, data in graph.edges(data=True):
            conductance = 1.0 / (data["R_ohm"] * flat_rhoa_scale[row, child])
            drive = conductance * (flat_ve[row, child] - flat_ve[row, parent])
            rhs = rhs.clone()
            rhs[parent] += drive
            rhs[child] -= drive
        output.append(torch.linalg.solve(matrix, rhs))
    return torch.stack(output).reshape_as(voltage)


def _dense_block_reference(model, mechanism, vc, voltage, dt, intra, extracellular):
    graph = model.graph
    batch = math.prod(model.shape[:-1])
    compartments = model.shape[-1]
    area_scale = torch.broadcast_to(
        torch.as_tensor(
            model.area_scale,
            dtype=model.area.dtype,
            device=model.area.device,
        ),
        model.shape,
    ).reshape(batch, compartments)
    cm_scale = torch.broadcast_to(
        torch.as_tensor(
            model.cm_scale,
            dtype=model.cm.dtype,
            device=model.cm.device,
        ),
        model.shape,
    ).reshape(batch, compartments)
    rhoa_scale = torch.broadcast_to(
        torch.as_tensor(
            model.rhoa_scale,
            dtype=model.area.dtype,
            device=model.area.device,
        ),
        model.shape,
    ).reshape(batch, compartments)
    area = model.area.reshape(batch, compartments) * area_scale
    cm = model.cm.reshape(batch, compartments) * cm_scale
    dx_cm = 1e-4 * model.dx.reshape(batch, compartments)
    xraxial = model.xraxial.reshape(batch, compartments, 2)
    xc = model.xc.reshape(batch, compartments, 2)
    xg_param = model.xg.reshape(batch, compartments, 2)
    vc = vc.reshape(batch, compartments, 3)
    voltage = voltage.reshape(batch, compartments)
    intra = torch.broadcast_to(intra, model.shape).reshape(batch, compartments)
    extracellular = torch.broadcast_to(extracellular, model.shape).reshape(
        batch, compartments
    )
    output = []
    for row in range(batch):
        cm_dt = 1e-6 * cm[row] * area[row] / (dt * 1e-3)
        xc_dt = 1e-6 * xc[row] * area[row, :, None] / (dt * 1e-3)
        xg = xg_param[row] * area[row, :, None]
        size = 3 * compartments
        matrix = torch.zeros(size, size, dtype=vc.dtype)
        rhs = torch.zeros(size, dtype=vc.dtype)
        current, conductance = mechanism.i(voltage[row])
        membrane_g = conductance * area[row]
        membrane_drive = (conductance * voltage[row] - current) * area[row] + intra[row]

        for node in range(compartments):
            vi, ve0, ve1 = 3 * node, 3 * node + 1, 3 * node + 2
            matrix[vi, vi] += cm_dt[node] + membrane_g[node]
            matrix[ve0, ve0] += (
                cm_dt[node] + membrane_g[node] + xc_dt[node, 0] + xg[node, 0]
            )
            matrix[ve1, ve1] += (
                xc_dt[node, 0] + xg[node, 0] + xc_dt[node, 1] + xg[node, 1]
            )
            matrix[vi, ve0] -= cm_dt[node] + membrane_g[node]
            matrix[ve0, vi] -= cm_dt[node] + membrane_g[node]
            matrix[ve0, ve1] -= xc_dt[node, 0] + xg[node, 0]
            matrix[ve1, ve0] -= xc_dt[node, 0] + xg[node, 0]

            old_vi, old_ve0, old_ve1 = vc[row, node]
            rhs[vi] += cm_dt[node] * (old_vi - old_ve0) + membrane_drive[node]
            rhs[ve0] += (
                -cm_dt[node] * (old_vi - old_ve0)
                + xc_dt[node, 0] * (old_ve0 - old_ve1)
                - membrane_drive[node]
            )
            rhs[ve1] += (
                -xc_dt[node, 0] * (old_ve0 - old_ve1)
                + xc_dt[node, 1] * old_ve1
                + xg[node, 1] * extracellular[row, node]
            )

        _add_edge_laplacian(
            matrix,
            graph,
            channel=0,
            rhoa_scale=rhoa_scale[row],
        )
        for shell in range(2):
            shell_g = torch.zeros(compartments, dtype=vc.dtype)
            for parent, child in graph.edges():
                resistance = (
                    0.5
                    * (
                        xraxial[row, child, shell] * dx_cm[row, child]
                        + xraxial[row, parent, shell] * dx_cm[row, parent]
                    )
                    * 1e6
                )
                shell_g[child] = 1.0 / resistance
            _add_edge_laplacian(matrix, graph, channel=shell + 1, shell_g=shell_g)

        output.append(torch.linalg.solve(matrix, rhs).reshape(compartments, 3))
    return torch.stack(output).reshape(model.shape + (3,))


def test_scalar_tree_matches_independent_dense_system_and_input_gradients():
    model = _TreeModel(_graphs_for_batched_scalar())
    model.rhoa_scale = torch.tensor(
        [
            [0.7, 1.1, 1.6, 0.85, 1.35],
            [1.4, 0.75, 1.2, 1.8, 0.95],
        ],
        dtype=DTYPE,
    )
    mechanism = _LinearMechanism(conductance=3.2e-4, reversal=-47.5)
    integrator = _dhs(model, mechanism, threads=2)
    dt = 0.19
    integrator._initialize(model, dt)
    voltage = model.v.detach().clone().requires_grad_()
    intra = torch.tensor(
        [[0.013, -0.021, 0.034, 0.008, -0.017]], dtype=DTYPE, requires_grad=True
    )
    extracellular = torch.tensor(
        [[-4.0, 1.5, 6.0, -2.5, 3.0], [3.0, -5.0, 2.0, 7.0, -1.0]],
        dtype=DTYPE,
        requires_grad=True,
    )

    actual, _ = integrator._step(
        voltage, dt, model.celsius, intra=intra, ve=extracellular
    )
    expected = _dense_scalar_reference(
        model, mechanism, voltage, dt, intra, extracellular
    )
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)

    inputs = (voltage, intra, extracellular)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=2e-11, atol=2e-11)


def test_scalar_tree_conserves_capacitive_charge_and_uniform_voltage():
    graph = _graph(
        EDGES,
        [8.0e6, 11.0e6, 15.0e6, 20.0e6],
        node_order=[4, 0, 2, 1, 3],
    )
    model = _TreeModel(graph, shape=(1, 5))
    mechanism = _LinearMechanism(currents=False)
    integrator = _dhs(model, mechanism, threads=2)
    dt = 0.08
    integrator._initialize(model, dt)

    redistributed, _ = integrator._step(model.v, dt, model.celsius)
    torch.testing.assert_close(
        (integrator.cmdt * redistributed).sum(),
        (integrator.cmdt * model.v).sum(),
        rtol=2e-13,
        atol=2e-13,
    )
    uniform = torch.full_like(model.v, -61.25)
    unchanged, _ = integrator._step(uniform, dt, model.celsius)
    torch.testing.assert_close(unchanged, uniform, rtol=0.0, atol=2e-13)


def test_block_tree_matches_independent_dense_system_and_input_gradients():
    graph = _graph(
        EDGES,
        [8_000_000.123, 12_000_000.456, 16_000_000.789, 22_000_000.321],
        node_order=[4, 1, 3, 0, 2],
    )
    model = _BlockTreeModel(graph)
    mechanism = _LinearMechanism(conductance=2.4e-4, reversal=-49.0)
    integrator = _dhs_bt(model, mechanism, threads=2)
    dt = 0.07
    integrator._initialize(model, dt)
    vc = torch.empty(model.shape + (3,), dtype=DTYPE)
    vc[..., 0] = model.v
    vc[..., 1] = torch.tensor([-1.0, 0.5, 2.0, -0.75, 1.25], dtype=DTYPE)
    vc[..., 2] = torch.tensor([0.4, -0.6, 0.9, 1.1, -0.3], dtype=DTYPE)
    vc.requires_grad_()
    voltage = (vc[..., 0] - vc[..., 1]).requires_grad_()
    intra = torch.tensor(
        [0.011, -0.007, 0.019, 0.003, -0.013], dtype=DTYPE
    ).requires_grad_()
    extracellular = torch.tensor(
        [-3.0, 1.0, 5.0, -2.0, 4.0], dtype=DTYPE, requires_grad=True
    )

    actual, _, _ = integrator._step(
        vc.reshape(-1, 5, 3),
        voltage,
        dt,
        model.celsius,
        intra=intra,
        ve=extracellular,
    )
    expected = _dense_block_reference(
        model,
        mechanism,
        vc,
        voltage,
        dt,
        intra,
        extracellular,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=8e-10)

    inputs = (vc, voltage, intra, extracellular)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=3e-9, atol=3e-8)


def test_block_tree_applies_population_scales_and_preserves_their_gradients():
    graph = _graph(
        EDGES,
        [8_000_000.123, 12_000_000.456, 16_000_000.789, 22_000_000.321],
        node_order=[4, 1, 3, 0, 2],
    )
    model = _BlockTreeModel(graph)
    model.area_scale = torch.tensor(
        [[0.85, 1.10, 1.35, 0.95, 1.20], [1.25, 0.90, 1.05, 1.40, 0.80]],
        dtype=DTYPE,
        requires_grad=True,
    )
    model.cm_scale = torch.tensor(
        [[1.15, 0.90, 1.25, 0.80, 1.05], [0.75, 1.30, 0.95, 1.10, 1.20]],
        dtype=DTYPE,
        requires_grad=True,
    )
    model.rhoa_scale = torch.tensor(
        [[0.70, 1.10, 1.60, 0.85, 1.35], [1.40, 0.75, 1.20, 1.80, 0.95]],
        dtype=DTYPE,
        requires_grad=True,
    )
    mechanism = _LinearMechanism(conductance=2.4e-4, reversal=-49.0)
    integrator = _dhs_bt(model, mechanism, threads=2)
    dt = 0.07
    integrator._initialize(model, dt)
    vc = torch.empty(model.shape + (3,), dtype=DTYPE)
    vc[..., 0] = model.v
    vc[..., 1] = torch.tensor([-1.0, 0.5, 2.0, -0.75, 1.25], dtype=DTYPE)
    vc[..., 2] = torch.tensor([0.4, -0.6, 0.9, 1.1, -0.3], dtype=DTYPE)
    voltage = vc[..., 0] - vc[..., 1]
    intra = torch.tensor([0.011, -0.007, 0.019, 0.003, -0.013], dtype=DTYPE)
    extracellular = torch.tensor([-3.0, 1.0, 5.0, -2.0, 4.0], dtype=DTYPE)

    actual, _, _ = integrator._step(
        vc.reshape(-1, 5, 3),
        voltage,
        dt,
        model.celsius,
        intra=intra,
        ve=extracellular,
    )
    expected = _dense_block_reference(
        model,
        mechanism,
        vc,
        voltage,
        dt,
        intra,
        extracellular,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=8e-10)

    scales = (model.area_scale, model.cm_scale, model.rhoa_scale)
    actual_grad = torch.autograd.grad(actual.square().sum(), scales, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), scales)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=3e-9, atol=3e-8)


def test_block_tree_common_mode_equal_to_bath_is_invariant():
    graph = _graph(
        EDGES,
        [8.0e6, 12.0e6, 16.0e6, 22.0e6],
        node_order=[3, 1, 4, 0, 2],
    )
    model = _BlockTreeModel(graph, shape=(1, 5))
    integrator = _dhs_bt(model, _LinearMechanism(), threads=2)
    dt = 0.05
    integrator._initialize(model, dt)
    common_mode = 17.25
    vc = torch.full(model.shape + (3,), common_mode, dtype=DTYPE)
    voltage = torch.zeros(model.shape, dtype=DTYPE)
    bath = torch.full(model.shape, common_mode, dtype=DTYPE)

    actual, membrane, i_membrane = integrator._step(
        vc.reshape(-1, 5, 3), voltage, dt, model.celsius, ve=bath
    )
    torch.testing.assert_close(actual, vc, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(membrane, voltage, rtol=0.0, atol=1e-10)
    assert i_membrane is None


def test_block_tree_imem_reports_discrete_membrane_balance():
    graph = _graph(
        EDGES,
        [9.0e6, 13.0e6, 17.0e6, 21.0e6],
        node_order=[0, 4, 2, 3, 1],
    )
    model = _BlockTreeModel(graph, shape=(1, 5))
    mechanism = _LinearMechanism(conductance=1.8e-4, reversal=-48.0)
    integrator = _dhs_bt(model, mechanism, imem=True, threads=2)
    dt = 0.04
    integrator._initialize(model, dt)
    model.vc[..., 1] = torch.tensor([-1.0, 0.0, 1.0, -0.5, 0.5], dtype=DTYPE)
    model.vc[..., 2] = torch.tensor([0.5, -0.25, 0.75, 0.0, -0.5], dtype=DTYPE)
    model.v = model.vc[..., 0] - model.vc[..., 1]
    old_voltage = model.v.clone()
    old_current, old_conductance = mechanism.i(old_voltage)

    integrator.step(
        model,
        dt,
        ve=torch.tensor([1.0, -2.0, 3.0, 0.5, -1.5], dtype=DTYPE),
        intra=torch.tensor([0.01, 0.0, -0.02, 0.03, -0.01], dtype=DTYPE),
    )
    expected = (integrator.cm_dt + old_conductance.reshape(1, 5) * integrator.area) * (
        model.v.reshape(1, 5) - old_voltage.reshape(1, 5)
    )
    expected = expected + old_current.reshape(1, 5) * integrator.area
    torch.testing.assert_close(model.i_membrane, expected.reshape(model.shape))


def test_block_tree_dt_reinitialization_preserves_live_state():
    graph = _graph(
        EDGES,
        [9.0e6, 13.0e6, 17.0e6, 21.0e6],
        node_order=[4, 3, 1, 0, 2],
    )
    model = _BlockTreeModel(graph, shape=(1, 5))
    integrator = _dhs_bt(model, _LinearMechanism(), threads=2)
    integrator._initialize(model, 0.04)
    model.vc.copy_(torch.linspace(-70.0, 4.0, 15, dtype=DTYPE).reshape(1, 5, 3))
    model.v.copy_(model.vc[..., 0] - model.vc[..., 1])
    expected_vc = model.vc.clone()
    expected_v = model.v.clone()

    integrator._initialize(model, 0.09)
    torch.testing.assert_close(model.vc, expected_vc, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.v, expected_v, rtol=0.0, atol=0.0)


def test_failed_tree_reinitialization_is_retried_after_topology_repair():
    graph = _graph(
        EDGES,
        [8.0e6, 12.0e6, 16.0e6, 20.0e6],
        node_order=[0, 1, 2, 3, 4],
    )
    model = _TreeModel(graph, shape=(1, 5))
    mechanism = _LinearMechanism(conductance=2.0e-4)
    integrator = _dhs(model, mechanism, threads=2)
    dt = 0.06
    integrator._initialize(model, dt)

    invalid = nx.DiGraph()
    invalid.add_nodes_from(range(5))
    invalid.add_edges_from([(0, 1), (1, 0), (0, 2), (2, 3), (2, 4)])
    model.graph = invalid
    with pytest.raises(ValueError, match="acyclic"):
        integrator._initialize(model, dt, force=True)
    assert not integrator.initialized

    repaired = _graph(
        EDGES,
        [7.0e6, 11.0e6, 15.0e6, 19.0e6],
        node_order=[4, 2, 0, 3, 1],
    )
    model.graph = repaired
    integrator._initialize(model, dt)
    expected_axial_mechanism_order = torch.tensor(
        [1 / 7.0e6, 1 / 11.0e6, 0.0, 1 / 15.0e6, 1 / 19.0e6],
        dtype=DTYPE,
    )
    assert integrator.initialized
    torch.testing.assert_close(
        integrator.a_geom.squeeze(0).index_select(0, integrator.inv_solver_order),
        expected_axial_mechanism_order,
    )


@pytest.mark.parametrize(
    "attributes, match",
    [
        ({"L": 10.0, "diam": 0.0, "Ra": 100.0}, "diam"),
        ({"L": math.inf, "diam": 1.0, "Ra": 100.0}, "finite"),
        ({"L": 10.0, "diam": 1.0, "Ra": -1.0}, "positive"),
    ],
)
def test_graph_conversion_rejects_nonphysical_fallback_geometry(attributes, match):
    graph = nx.DiGraph()
    graph.add_node(0, **attributes)
    graph.add_node(1, **attributes)
    graph.add_edge(0, 1)
    with pytest.raises(ValueError, match=match):
        graph_to_parent_and_axial(graph, dtype_axial=DTYPE)


@pytest.mark.parametrize(
    "depth, error",
    [
        (torch.tensor([0, -1], dtype=torch.int32), ValueError),
        (torch.tensor([0.0, 1.5]), TypeError),
    ],
)
def test_layer_builder_rejects_invalid_depth_vectors(depth, error):
    with pytest.raises(error, match="depth"):
        build_dhs_layers(depth, 2)
