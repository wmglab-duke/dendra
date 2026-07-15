"""Independent CPU contracts for universal scalar multi-model packing."""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.core import Axon, Population
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE, _dhs_multi
from dendra.models.tree import Tree

DTYPE = torch.float64
pytestmark = pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="Universal scalar multi tests require the CPU dendra_solvers package",
)


class LinearMechanism(torch.nn.Module):
    """Small linear membrane current model in composite mechanism order."""

    currents = True

    def __init__(self, conductance, reversal):
        super().__init__()
        self.register_buffer("conductance", torch.as_tensor(conductance, dtype=DTYPE))
        self.register_buffer("reversal", torch.as_tensor(reversal, dtype=DTYPE))
        self.dt = None

    def update_v(self, voltage):
        return voltage

    def advance(self, voltage, dt, temp):
        return None

    def i(self, voltage):
        conductance = self.conductance.to(voltage).expand_as(voltage)
        reversal = self.reversal.to(voltage).expand_as(voltage)
        return conductance * (voltage - reversal), conductance

    def set_dt(self, dt):
        self.dt = float(dt)


def _non_topological_tree_graph():
    graph = nx.DiGraph()
    areas = [0.85e8, 1.15e8, 0.95e8, 1.35e8]
    cms = [0.75, 1.05, 1.25, 0.9]
    # Insertion order deliberately differs from both numeric and topological
    # order. Mechanism state nevertheless remains in numeric node-label order.
    for node in (3, 1, 0, 2):
        graph.add_node(
            node,
            name=f"section[{node}](0.5)",
            L=5.0 + node,
            diam=1.0 + 0.1 * node,
            Ra=80.0 + 3.0 * node,
            cm=cms[node],
            area=areas[node],
        )
    for edge, resistance in zip(((2, 0), (2, 1), (0, 3)), (55.0, 80.0, 105.0)):
        graph.add_edge(*edge, R_ohm=resistance)
    return graph


def _mixed_components():
    point = dn.Population(
        N=2,
        C=3,
        v_init=[-61.0, -63.0, -65.0],
        cm=torch.tensor([[0.7, 1.1, 1.5], [1.9, 2.3, 2.7]], dtype=DTYPE),
        dtype=DTYPE,
    )
    point.diam.copy_(torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=DTYPE))
    point.dx.copy_(torch.tensor([[11.0, 13.0, 17.0], [19.0, 23.0, 29.0]], dtype=DTYPE))
    point.area_scale.fill_(1.2)
    point.cm_scale.fill_(0.85)

    single = dn.SingleCompartment(N=2, C=1, dtype=DTYPE)
    single.cm.copy_(torch.tensor([[1.4], [2.1]], dtype=DTYPE))
    single.diam.copy_(torch.tensor([[4.5], [7.5]], dtype=DTYPE))
    single.dx.copy_(torch.tensor([[12.0], [18.0]], dtype=DTYPE))
    single.area_scale.fill_(0.95)
    single.cm_scale.fill_(1.05)

    tree = dn.Tree.from_graph(_non_topological_tree_graph(), N=2, dtype=DTYPE)
    tree.area_scale.fill_(1.35)
    tree.cm_scale.fill_(0.8)
    tree.rhoa_scale.fill_(1.25)

    unmyelinated = dn.Unmyelinated(diameters=[1.5, 3.0], L=30.0, dx=10.0, dtype=DTYPE)
    unmyelinated.cm.copy_(torch.tensor([[0.8, 1.0, 1.2], [1.1, 1.3, 1.5]], dtype=DTYPE))
    unmyelinated.rhoa.copy_(
        torch.tensor([[31.0, 37.0, 43.0], [47.0, 53.0, 59.0]], dtype=DTYPE)
    )
    unmyelinated.area_scale.fill_(0.9)
    unmyelinated.cm_scale.fill_(1.15)
    unmyelinated.rhoa_scale.fill_(0.75)

    myelinated = dn.Myelinated(diameters=[8.0], n_node=4, node_length=2.0, dtype=DTYPE)
    # Myelinated axial geometry is an in-graph parametrization. Populate it
    # before this low-level integrator test so the independent reference and
    # packed solver both consume the effective internodal resistivity.
    myelinated.populate_parameter_buffers()
    myelinated.cm.copy_(torch.tensor([[0.65, 0.8, 0.95, 1.1]], dtype=DTYPE))
    myelinated.area_scale.fill_(1.1)
    myelinated.cm_scale.fill_(0.7)
    myelinated.rhoa_scale.fill_(1.4)

    return {
        "point": point,
        "single": single,
        "tree": tree,
        "unmyelinated": unmyelinated,
        "myelinated": myelinated,
    }


def _make_mixed(*, write_back=True):
    components = _mixed_components()
    model = dn.concat_models(components, write_back=write_back)
    width = model.shape[-1]
    mechanism = LinearMechanism(
        torch.linspace(0.0012, 0.0038, width, dtype=DTYPE),
        torch.linspace(-72.0, -48.0, width, dtype=DTYPE),
    )
    integrator = _dhs_multi(model, mechanism, threads=2, write_back=write_back)
    return model, mechanism, integrator


def _group_planes(value, population, P, B, K):
    value = torch.as_tensor(value, device=population.device(), dtype=population.dtype())
    return torch.broadcast_to(value, tuple(population.shape)).reshape(P, B, K)


def _component_layout(population, P):
    """Return independent reference layout and physical axial edges."""
    if isinstance(population, Tree):
        B, K = population.np, population.nc
        graphs = population.graph
        graphs = graphs if isinstance(graphs, list) else [graphs]
        if len(graphs) not in (1, B):
            raise AssertionError("Tree reference requires one graph or one per neuron")

        rhoa_scale = _group_planes(population.rhoa_scale, population, P, B, K)
        edges = []
        for parent, child in graphs[0].edges:
            resistance = torch.tensor(
                [
                    graphs[0 if len(graphs) == 1 else row].edges[parent, child]["R_ohm"]
                    for row in range(B)
                ],
                dtype=population.dtype(),
                device=population.device(),
            )
            conductance = resistance.reciprocal().reshape(1, B).expand(P, B)
            conductance = conductance / rhoa_scale[..., child]
            edges.append((parent, child, conductance))
        return B, K, edges

    if isinstance(population, Axon):
        B, K = population.np, population.nc
        diam = _group_planes(population.diam, population, P, B, K)
        dx = _group_planes(population.dx, population, P, B, K)
        rhoa = _group_planes(population.rhoa, population, P, B, K)
        rhoa_scale = _group_planes(population.rhoa_scale, population, P, B, K)
        radius_cm = diam * 1e-4 / 2
        dx_cm = dx * 1e-4
        segment_resistance = rhoa * rhoa_scale * dx_cm / (torch.pi * radius_cm.square())
        edge_conductance = 2 / (
            segment_resistance[..., :-1] + segment_resistance[..., 1:]
        )
        edges = [
            (child - 1, child, edge_conductance[..., child - 1])
            for child in range(1, K)
        ]
        return B, K, edges

    if isinstance(population, Population):
        # Every ordinary Population element is an independent one-node graph.
        return population.np * population.nc, 1, []

    raise TypeError(f"No independent scalar reference for {type(population)!r}")


def _dense_reference(model, mechanism, voltage, dt, *, ve=None, intra=None):
    """Solve every physical scalar group directly in mechanism order."""
    P = int(math.prod(voltage.shape[:-2])) if voltage.ndim > 2 else 1
    width = voltage.shape[-1]
    voltage_flat = voltage.reshape(P, width)
    current, conductance = mechanism.i(voltage)
    current_flat = current.reshape(P, width)
    conductance_flat = conductance.reshape(P, width)
    intra_flat = (
        torch.zeros_like(voltage_flat)
        if intra is None
        else torch.broadcast_to(intra, voltage.shape).reshape(P, width)
    )
    ve_flat = (
        None if ve is None else torch.broadcast_to(ve, voltage.shape).reshape(P, width)
    )

    result = torch.empty_like(voltage_flat)
    offset = 0
    dt_seconds = dt * 1e-3
    for population in model.populations.values():
        B, K, edges = _component_layout(population, P)
        component_width = B * K
        group_shape = (P, B, K)
        group_slice = slice(offset, offset + component_width)

        v_group = voltage_flat[:, group_slice].reshape(group_shape)
        i_group = current_flat[:, group_slice].reshape(group_shape)
        g_group = conductance_flat[:, group_slice].reshape(group_shape)
        intra_group = intra_flat[:, group_slice].reshape(group_shape)
        ve_group = (
            None if ve_flat is None else ve_flat[:, group_slice].reshape(group_shape)
        )

        area = _group_planes(population.area, population, P, B, K)
        area = area * _group_planes(population.area_scale, population, P, B, K)
        capacitance = (
            1e-6
            * _group_planes(population.cm, population, P, B, K)
            * area
            * _group_planes(population.cm_scale, population, P, B, K)
        )
        cdt = capacitance / dt_seconds
        main = cdt + g_group * area
        rhs = (g_group * v_group - i_group) * area + cdt * v_group
        rhs = rhs + intra_group
        matrix = torch.diag_embed(main)

        for parent, child, edge_g in edges:
            orientation = torch.zeros(K, dtype=voltage.dtype, device=voltage.device)
            orientation[parent] = 1
            orientation[child] = -1
            laplacian = torch.outer(orientation, orientation)
            matrix = matrix + edge_g[..., None, None] * laplacian
            if ve_group is not None:
                delta_ve = ve_group[..., child] - ve_group[..., parent]
                rhs = rhs + (edge_g * delta_ve)[..., None] * orientation

        solved = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
        result[:, group_slice] = solved.reshape(P, component_width)
        offset += component_width

    assert offset == width
    return result.reshape_as(voltage)


def _sample_inputs(model):
    width = model.shape[-1]
    P = int(math.prod(model.shape[:-2])) if len(model.shape) > 2 else 1
    voltage = torch.linspace(-76.0, -44.0, width, dtype=DTYPE).expand(P, -1).clone()
    ve = torch.linspace(-5.0, 8.0, width, dtype=DTYPE).expand(P, -1).clone()
    intra = torch.linspace(-3e-9, 5e-9, width, dtype=DTYPE).expand(P, -1).clone()
    if P > 1:
        plane = torch.arange(P, dtype=DTYPE).reshape(P, 1)
        voltage = voltage + plane * torch.linspace(0.1, 1.0, width, dtype=DTYPE)
        ve = ve - plane * torch.linspace(0.2, 0.7, width, dtype=DTYPE)
        intra = intra + plane * 1e-10
    return (
        voltage.reshape(model.shape),
        intra.reshape(model.shape),
        ve.reshape(model.shape),
    )


def test_public_concat_selects_universal_scalar_solver_for_mixed_models():
    model = dn.concat_models(_mixed_components())

    assert issubclass(model._integrator_class, _dhs_multi)


def test_public_concat_rejects_a_nested_multi_population():
    nested = dn.concat_models(
        {
            "left": dn.Population(N=1, C=1, dtype=DTYPE),
            "right": dn.SingleCompartment(N=1, C=1, dtype=DTYPE),
        }
    )
    tree = dn.Tree.from_graph(_non_topological_tree_graph(), dtype=DTYPE)

    with pytest.raises(TypeError, match="do not expose a supported"):
        dn.concat_models({"nested": nested, "tree": tree})


def test_multi_population_populates_and_refreshes_myelinated_geometry():
    point = dn.Population(N=1, C=2, dtype=DTYPE)
    myelinated = dn.Myelinated(
        diameters=[8.0, 12.0], n_node=3, node_length=2.0, dtype=DTYPE
    )
    oracle = dn.Myelinated(
        diameters=[8.0, 12.0], n_node=3, node_length=2.0, dtype=DTYPE
    )
    baseline_rhoa = myelinated.rhoa.clone()
    model = dn.concat_models({"point": point, "myelinated": myelinated})

    model.populate_parameter_buffers()
    oracle.populate_parameter_buffers()

    assert not torch.equal(myelinated.rhoa, baseline_rhoa)
    torch.testing.assert_close(myelinated.rhoa, oracle.rhoa)
    point_width = point.numelc()
    torch.testing.assert_close(
        model.rhoa[..., point_width:], myelinated.rhoa.reshape(1, -1)
    )
    torch.testing.assert_close(
        model.area[..., point_width:], myelinated.area.reshape(1, -1)
    )


def test_mixed_scalar_models_match_dense_reference_and_pack_points_as_k1():
    model, mechanism, integrator = _make_mixed()
    dt = 0.041
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)

    actual, _ = integrator._step(voltage, dt, model.celsius, ve=ve, intra=intra)
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)

    assert integrator.group_B == [6, 2, 2, 2, 1]
    assert integrator.group_K == [1, 1, 4, 3, 4]
    assert integrator.B_total == 13
    assert integrator.K_stride == 4
    torch.testing.assert_close(actual, expected, rtol=4e-11, atol=4e-11)


def test_batched_axon_live_geometry_values_and_gradients_match_dense_reference():
    axon = dn.Unmyelinated(diameters=[1.7, 3.4], L=30.0, dx=10.0, dtype=DTYPE)
    model = dn.concat_models({"axon": axon})
    model.batch(2)

    plane_scale = torch.tensor([0.75, 1.35], dtype=DTYPE).reshape(2, 1, 1)
    diam = torch.broadcast_to(axon.diam, axon.shape).clone().detach()
    diam = (diam * plane_scale).requires_grad_()
    dx = torch.broadcast_to(axon.dx, axon.shape).clone().detach()
    dx = (dx * torch.flip(plane_scale, (0,))).requires_grad_()
    rhoa = torch.broadcast_to(axon.rhoa, axon.shape).clone().detach()
    rhoa = (
        rhoa
        * torch.tensor([0.8, 1.25], dtype=DTYPE).reshape(2, 1, 1)
        * torch.linspace(0.9, 1.2, axon.nc, dtype=DTYPE)
    ).requires_grad_()
    rhoa_scale = torch.tensor([0.7, 1.6], dtype=DTYPE).reshape(2, 1, 1)
    rhoa_scale.requires_grad_()
    axon.diam = diam
    axon.dx = dx
    axon.rhoa = rhoa
    axon.rhoa_scale = rhoa_scale

    width = model.shape[-1]
    mechanism = LinearMechanism(
        torch.linspace(0.0015, 0.003, width, dtype=DTYPE),
        torch.linspace(-70.0, -50.0, width, dtype=DTYPE),
    )
    integrator = _dhs_multi(model, mechanism, threads=2)
    dt = 0.037
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)
    voltage.requires_grad_()
    intra.requires_grad_()
    ve.requires_grad_()

    actual, _ = integrator._step(voltage, dt, model.celsius, ve=ve, intra=intra)
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)
    torch.testing.assert_close(actual, expected, rtol=8e-11, atol=8e-11)

    inputs = (voltage, intra, ve, diam, dx, rhoa, rhoa_scale)
    weights = torch.linspace(0.4, 1.3, actual.numel(), dtype=DTYPE).reshape_as(actual)
    actual_grad = torch.autograd.grad(
        (actual.square() * weights).sum(), inputs, retain_graph=True
    )
    expected_grad = torch.autograd.grad((expected.square() * weights).sum(), inputs)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=8e-9, atol=8e-9)
        assert torch.isfinite(got).all()
        assert got.abs().sum() > 0


@pytest.mark.parametrize("write_back", [True, False])
def test_mixed_scalar_writeback_contract(write_back):
    model, mechanism, integrator = _make_mixed(write_back=write_back)
    dt = 0.033
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)
    voltage.requires_grad_()
    model.v = voltage
    component_before = {
        name: population.v.clone() for name, population in model.populations.items()
    }
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)

    integrator.step(model, dt, ve=ve, intra=intra)
    torch.testing.assert_close(model.v, expected, rtol=5e-11, atol=5e-11)

    offset = 0
    if write_back:
        weights = torch.linspace(0.3, 1.4, model.v.numel(), dtype=DTYPE).reshape_as(
            model.v
        )
        component_loss = torch.zeros((), dtype=DTYPE)
        for population in model.populations.values():
            width = population.numelc()
            expected_component = model.v[..., offset : offset + width].reshape_as(
                population.v
            )
            torch.testing.assert_close(population.v, expected_component)
            component_weights = weights[..., offset : offset + width].reshape_as(
                population.v
            )
            component_loss = component_loss + (population.v * component_weights).sum()
            offset += width
        assert offset == model.shape[-1]

        expected_loss = (expected * weights).sum()
        actual_grad = torch.autograd.grad(component_loss, voltage, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected_loss, voltage)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-10, atol=5e-10)
    else:
        for name, population in model.populations.items():
            assert torch.equal(population.v, component_before[name])


def test_mixed_scalar_reinitializes_after_repeated_batching():
    model, mechanism, integrator = _make_mixed(write_back=True)
    integrator._initialize(model, 0.02)
    assert integrator.P == 1

    model.batch(2)
    model.batch(3)
    dt = 0.071
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)
    model.v = voltage
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)

    integrator.step(model, dt, ve=ve, intra=intra)

    assert integrator.P == 6
    assert integrator.group_B == [6, 2, 2, 2, 1]
    assert integrator.group_K == [1, 1, 4, 3, 4]
    torch.testing.assert_close(model.v, expected, rtol=8e-11, atol=8e-11)
    offset = 0
    for population in model.populations.values():
        width = population.numelc()
        expected_component = model.v[..., offset : offset + width].reshape_as(
            population.v
        )
        torch.testing.assert_close(population.v, expected_component)
        offset += width
