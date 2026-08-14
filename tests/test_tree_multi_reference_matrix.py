"""Independent physical references for the packed multi-tree integrator."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE, _dhs_multi
from dendra.models.integrators.triton import dhs_solve_multi_cuda

DTYPE = torch.float64
pytestmark = pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="CPU multi-tree reference tests require dendra_solvers",
)


class LinearMechanism(torch.nn.Module):
    """Small linear current model whose parameters stay in mechanism order."""

    currents = True

    def __init__(self, conductance, reversal):
        super().__init__()
        self.register_buffer("conductance", torch.as_tensor(conductance))
        self.register_buffer("reversal", torch.as_tensor(reversal))
        self.advance_calls = 0
        self.dt = None

    def update_v(self, voltage):
        return voltage

    def advance(self, voltage, dt, temp):
        self.advance_calls += 1

    def i(self, voltage):
        conductance = self.conductance.to(voltage).expand_as(voltage)
        reversal = self.reversal.to(voltage).expand_as(voltage)
        return conductance * (voltage - reversal), conductance

    def set_dt(self, dt):
        self.dt = float(dt)


def _graph(edges, *, nodes, areas, cms, resistances):
    graph = nx.DiGraph()
    # Deliberately preserve the supplied insertion order: graph insertion order
    # must not be confused with either mechanism or topological solver order.
    for node in nodes:
        graph.add_node(
            node,
            name=f"section[{node}](0.5)",
            L=5.0 + node,
            diam=1.0 + 0.1 * node,
            Ra=80.0 + 3.0 * node,
            cm=float(cms[node]),
            # Dendra converts graph areas from um^2 to cm^2.
            area=float(areas[node]),
        )
    for edge, resistance in zip(edges, resistances):
        graph.add_edge(*edge, R_ohm=float(resistance))
    return graph


def _make_multi(*, batch=None, imem=False, write_back=True, threads=2):
    branched = _graph(
        [(2, 0), (2, 1), (0, 3)],
        nodes=[3, 1, 0, 2],
        areas=[0.85e8, 1.15e8, 0.95e8, 1.35e8],
        cms=[0.75, 1.05, 1.25, 0.9],
        resistances=[55.0, 80.0, 105.0],
    )
    chain = _graph(
        [(0, 1), (1, 2)],
        nodes=[2, 0, 1],
        areas=[1.4e8, 0.7e8, 1.1e8],
        cms=[1.3, 0.8, 1.1],
        resistances=[65.0, 125.0],
    )
    left = dn.Tree.from_graph(branched, N=2, dtype=DTYPE)
    right = dn.Tree.from_graph(chain, N=1, dtype=DTYPE)

    left.area_scale.fill_(1.7)
    left.cm_scale.fill_(0.65)
    left.rhoa_scale.fill_(1.4)
    right.area_scale.fill_(0.8)
    right.cm_scale.fill_(1.55)
    right.rhoa_scale.fill_(0.75)

    multi = dn.concat_models({"branched": left, "chain": right}, write_back=write_back)
    if batch is not None:
        multi.batch(batch)

    n_total = multi.shape[-1]
    conductance = torch.linspace(0.0015, 0.0045, n_total, dtype=DTYPE)
    reversal = torch.linspace(-58.0, -43.0, n_total, dtype=DTYPE)
    mechanism = LinearMechanism(conductance, reversal)
    integrator = _dhs_multi(
        multi,
        mechanism,
        imem=imem,
        threads=threads,
        write_back=write_back,
    )
    return multi, mechanism, integrator


def _broadcast_group_parameter(value, shape, *, dtype, device):
    value = torch.as_tensor(value, dtype=dtype, device=device)
    while value.ndim < len(shape):
        value = value.unsqueeze(0)
    return torch.broadcast_to(value, shape)


def _dense_reference(multi, mechanism, voltage, dt, *, ve=None, intra=None):
    """Assemble each physical Hines matrix directly in original node order."""

    flat_shape = (int(voltage.numel() // voltage.shape[-1]), voltage.shape[-1])
    voltage_flat = voltage.reshape(flat_shape)
    current, conductance = mechanism.i(voltage)
    current_flat = current.reshape(flat_shape)
    conductance_flat = conductance.reshape(flat_shape)
    intra_flat = (
        torch.zeros_like(voltage_flat)
        if intra is None
        else torch.broadcast_to(intra, voltage.shape).reshape(flat_shape)
    )
    ve_flat = (
        None
        if ve is None
        else torch.broadcast_to(ve, voltage.shape).reshape(flat_shape)
    )

    result = torch.empty_like(voltage_flat)
    dt_seconds = dt * 1e-3
    offset = 0
    for population in multi.populations.values():
        neurons = population.shape[-2]
        compartments = population.shape[-1]
        width = neurons * compartments
        group_shape = (flat_shape[0], neurons, compartments)

        v_group = voltage_flat[:, offset : offset + width].reshape(group_shape)
        i_group = current_flat[:, offset : offset + width].reshape(group_shape)
        g_group = conductance_flat[:, offset : offset + width].reshape(group_shape)
        intra_group = intra_flat[:, offset : offset + width].reshape(group_shape)

        area = _broadcast_group_parameter(
            population.area,
            group_shape,
            dtype=voltage.dtype,
            device=voltage.device,
        ) * _broadcast_group_parameter(
            population.area_scale,
            group_shape,
            dtype=voltage.dtype,
            device=voltage.device,
        )
        capacitance = (
            1e-6
            * _broadcast_group_parameter(
                population.cm,
                group_shape,
                dtype=voltage.dtype,
                device=voltage.device,
            )
            * area
            * _broadcast_group_parameter(
                population.cm_scale,
                group_shape,
                dtype=voltage.dtype,
                device=voltage.device,
            )
        )
        cdt = capacitance / dt_seconds
        main = cdt + g_group * area
        rhs = (g_group * v_group - i_group) * area + cdt * v_group + intra_group

        graphs = (
            population.graph
            if isinstance(population.graph, list)
            else [population.graph]
        )
        graph = graphs[0]
        axial = []
        for parent, child, edge_data in graph.edges(data=True):
            resistance = torch.as_tensor(
                [candidate.edges[parent, child]["R_ohm"] for candidate in graphs],
                dtype=voltage.dtype,
                device=voltage.device,
            )
            resistance = resistance.expand(neurons).reshape(1, neurons)
            rhoa_scale = _broadcast_group_parameter(
                population.rhoa_scale,
                group_shape,
                dtype=voltage.dtype,
                device=voltage.device,
            )[..., child]
            axial.append((parent, child, 1.0 / resistance / rhoa_scale))

        if ve_flat is not None:
            ve_group = ve_flat[:, offset : offset + width].reshape(group_shape)
            rhs = rhs.clone()
            for parent, child, edge_g in axial:
                edge_current = (ve_group[..., child] - ve_group[..., parent]) * edge_g
                rhs[..., child] -= edge_current
                rhs[..., parent] += edge_current

        matrix = torch.diag_embed(main)
        for parent, child, edge_g in axial:
            matrix[..., child, child] += edge_g
            matrix[..., parent, parent] += edge_g
            matrix[..., child, parent] -= edge_g
            matrix[..., parent, child] -= edge_g
        solved = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
        result[:, offset : offset + width] = solved.reshape(flat_shape[0], width)
        offset += width

    assert offset == voltage.shape[-1]
    return result.reshape_as(voltage)


def _sample_inputs(model):
    n_total = model.shape[-1]
    leading = model.shape[:-1]
    voltage = torch.linspace(-73.0, -47.0, n_total, dtype=DTYPE)
    voltage = voltage.expand(*leading, n_total).clone()
    if voltage.ndim > 2:
        voltage[1] += torch.linspace(3.0, -2.0, n_total, dtype=DTYPE)
    intra = torch.linspace(-0.012, 0.018, n_total, dtype=DTYPE)
    ve = torch.linspace(-4.0, 7.0, n_total, dtype=DTYPE)
    return voltage, intra, ve


def _make_plane_varying_geometry(model):
    """Replace broadcast geometry buffers with distinct differentiable planes."""

    geometry_inputs = []
    for group_index, population in enumerate(model.populations.values()):
        planes = model.shape[0]
        cm = torch.broadcast_to(population.cm, population.shape).clone().detach()
        cm[0] *= 0.7 + 0.1 * group_index
        cm[1] *= torch.linspace(
            1.15 + 0.1 * group_index,
            1.75 + 0.1 * group_index,
            population.shape[-1],
            dtype=DTYPE,
        )
        cm.requires_grad_()
        population.cm = cm

        area_scale = torch.tensor(
            [1.0 + 0.2 * group_index, 1.45 - 0.1 * group_index], dtype=DTYPE
        ).reshape(planes, 1, 1)
        area_scale.requires_grad_()
        population.area_scale = area_scale

        rhoa_scale = torch.tensor(
            [0.8 + 0.15 * group_index, 1.6 - 0.2 * group_index], dtype=DTYPE
        ).reshape(planes, 1, 1)
        rhoa_scale.requires_grad_()
        population.rhoa_scale = rhoa_scale

        geometry_inputs.extend((cm, area_scale, rhoa_scale))
    return geometry_inputs


def test_multi_tree_matches_dense_physical_reference_with_scales_and_forcing():
    model, mechanism, integrator = _make_multi(imem=True)
    dt = 0.037
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)

    actual, actual_imem = integrator._step(
        voltage, dt, model.celsius, ve=ve, intra=intra
    )
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)
    assert actual.dtype == DTYPE
    assert torch.allclose(actual, expected, rtol=3e-12, atol=3e-12)

    current, conductance = mechanism.i(voltage)
    scale = torch.cat(
        [
            (population.area * population.area_scale).reshape(-1)
            for population in model.populations.values()
        ]
    ).reshape_as(voltage)
    cdt = torch.cat(
        [
            (
                1e-6
                * population.cm
                * population.area
                * population.area_scale
                * population.cm_scale
                / (dt * 1e-3)
            ).reshape(-1)
            for population in model.populations.values()
        ]
    ).reshape_as(voltage)
    expected_imem = (cdt + conductance * scale) * (actual - voltage) + current * scale
    assert torch.allclose(actual_imem, expected_imem, rtol=3e-12, atol=3e-12)

    offset = 0
    for population in model.populations.values():
        width = population.shape[-2] * population.shape[-1]
        membrane_balance = actual_imem[..., offset : offset + width].reshape(
            population.shape[-2], population.shape[-1]
        )
        applied = intra[offset : offset + width].reshape(
            population.shape[-2], population.shape[-1]
        )
        # Axial and extracellular edge currents cancel pairwise within each
        # neuron, leaving only the externally applied intracellular current.
        assert torch.allclose(
            membrane_balance.sum(-1), applied.sum(-1), rtol=2e-11, atol=2e-11
        )
        offset += width


def test_multi_tree_batched_planes_match_dense_values_and_gradients():
    model, mechanism, integrator = _make_multi(batch=2)
    dt = 0.043
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)
    voltage.requires_grad_()
    intra.requires_grad_()
    ve.requires_grad_()
    mechanism.conductance.requires_grad_()
    mechanism.reversal.requires_grad_()

    actual, _ = integrator._step(voltage, dt, model.celsius, ve=ve, intra=intra)
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)
    assert integrator.P == 2
    assert actual.shape == model.shape
    assert torch.allclose(actual, expected, rtol=3e-12, atol=3e-12)

    weights = torch.linspace(0.7, 1.4, actual.numel(), dtype=DTYPE).reshape_as(actual)
    differentiable_inputs = (
        voltage,
        intra,
        ve,
        mechanism.conductance,
        mechanism.reversal,
    )
    actual_grad = torch.autograd.grad(
        (actual.square() * weights).sum(),
        differentiable_inputs,
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        (expected.square() * weights).sum(), differentiable_inputs
    )
    for got, want in zip(actual_grad, expected_grad):
        assert torch.allclose(got, want, rtol=2e-11, atol=2e-11)


def test_multi_tree_preserves_plane_varying_geometry_and_its_gradients():
    model, mechanism, integrator = _make_multi(batch=2)
    geometry_inputs = _make_plane_varying_geometry(model)
    dt = 0.046
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)
    voltage.requires_grad_()
    intra.requires_grad_()
    ve.requires_grad_()

    actual, _ = integrator._step(voltage, dt, model.celsius, ve=ve, intra=intra)
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)

    assert integrator.CMDT_MECH.shape == (2, model.shape[-1])
    assert integrator.SCALE_MECH.shape == (2, model.shape[-1])
    assert integrator.EDGE_GAX_FLAT.shape[0] == 2
    assert not torch.equal(integrator.CMDT_MECH[0], integrator.CMDT_MECH[1])
    assert not torch.equal(integrator.SCALE_MECH[0], integrator.SCALE_MECH[1])
    assert not torch.equal(integrator.EDGE_GAX_FLAT[0], integrator.EDGE_GAX_FLAT[1])
    assert torch.allclose(actual, expected, rtol=4e-12, atol=4e-12)

    differentiable_inputs = (voltage, intra, ve, *geometry_inputs)
    weights = torch.linspace(0.3, 1.1, actual.numel(), dtype=DTYPE).reshape_as(actual)
    actual_grad = torch.autograd.grad(
        (actual.square() * weights).sum(),
        differentiable_inputs,
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        (expected.square() * weights).sum(), differentiable_inputs
    )
    for got, want in zip(actual_grad, expected_grad):
        assert torch.allclose(got, want, rtol=4e-10, atol=4e-10)


def test_multi_tree_multistep_gradients_match_dense_reference():
    """Packed scratch storage must not invalidate tensors saved for BPTT."""

    model, mechanism, integrator = _make_multi()
    dt = 0.041
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)
    voltage.requires_grad_()
    intra.requires_grad_()
    ve.requires_grad_()
    mechanism.conductance.requires_grad_()
    mechanism.reversal.requires_grad_()

    actual_1, _ = integrator._step(voltage, dt, model.celsius, ve=ve, intra=intra)
    actual_2, _ = integrator._step(actual_1, dt, model.celsius, ve=ve, intra=intra)

    expected_1 = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)
    expected_2 = _dense_reference(model, mechanism, expected_1, dt, ve=ve, intra=intra)
    assert torch.allclose(actual_2, expected_2, rtol=5e-12, atol=5e-12)

    weights = torch.linspace(0.4, 1.3, actual_2.numel(), dtype=DTYPE).reshape_as(
        actual_2
    )
    differentiable_inputs = (
        voltage,
        intra,
        ve,
        mechanism.conductance,
        mechanism.reversal,
    )
    actual_grad = torch.autograd.grad(
        (actual_2.square() * weights).sum(),
        differentiable_inputs,
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        (expected_2.square() * weights).sum(), differentiable_inputs
    )
    for got, want in zip(actual_grad, expected_grad):
        assert torch.allclose(got, want, rtol=8e-10, atol=8e-10)


def test_multi_tree_common_mode_and_uniform_extracellular_gauge_are_invariant():
    model, mechanism, integrator = _make_multi()
    mechanism.conductance.zero_()
    dt = 0.05
    integrator._initialize(model, dt)
    voltage = torch.empty_like(model.v)
    voltage[..., :8] = -61.25
    voltage[..., 8:] = -48.75

    baseline, _ = integrator._step(voltage, dt, model.celsius)
    uniform_ve = torch.empty_like(voltage)
    uniform_ve[..., :8] = 123.0
    uniform_ve[..., 8:] = -91.0
    shifted, _ = integrator._step(voltage, dt, model.celsius, ve=uniform_ve)
    assert torch.allclose(baseline, voltage, rtol=0.0, atol=2e-13)
    assert torch.allclose(shifted, baseline, rtol=0.0, atol=2e-13)


def test_multi_tree_reinitializes_dt_and_new_batch_shape_without_stale_plan():
    model, mechanism, integrator = _make_multi()
    first_dt = 0.02
    integrator._initialize(model, first_dt)
    first_cmdt = integrator.CMDT_MECH.clone()
    assert len(integrator.base_shapes) == len(model)

    model.batch(3)
    second_dt = 0.08
    integrator._initialize(model, second_dt)
    voltage, intra, ve = _sample_inputs(model)
    actual, _ = integrator._step(voltage, second_dt, model.celsius, ve=ve, intra=intra)
    expected = _dense_reference(
        model, mechanism, voltage, second_dt, ve=ve, intra=intra
    )
    assert integrator.P == 3
    assert integrator._d_plane.shape == (3 * integrator.B_total, integrator.K_stride)
    assert len(integrator.base_shapes) == len(model)
    assert torch.allclose(integrator.CMDT_MECH, first_cmdt * 0.25)
    assert torch.allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_multi_tree_step_writes_each_group_back_in_original_layout():
    model, mechanism, integrator = _make_multi(write_back=True)
    dt = 0.031
    integrator._initialize(model, dt)
    voltage, intra, ve = _sample_inputs(model)
    model.v = voltage.clone()
    expected = _dense_reference(model, mechanism, voltage, dt, ve=ve, intra=intra)

    integrator.step(model, dt, ve=ve, intra=intra)
    assert torch.allclose(model.v, expected, rtol=3e-12, atol=3e-12)
    assert torch.equal(model.populations["branched"].v, model.v[..., :8].reshape(2, 4))
    assert torch.equal(model.populations["chain"].v, model.v[..., 8:].reshape(1, 3))


def test_multi_tree_mutable_graph_view_does_not_recompile_group_topology():
    model, mechanism, integrator = _make_multi()
    invalid = model.populations["chain"].graph.copy()
    invalid.add_edge(2, 0)
    model.populations["chain"]._graph = invalid

    # Tree factories retain an immutable CompartmentGraph snapshot. The
    # NetworkX graph is an interoperability view; mutating/replacing it does
    # not change the compiled electrical (or material) topology.
    integrator._initialize(model, 0.05)
    assert integrator.initialized is True
    assert integrator.dt == pytest.approx(0.05)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 32])
def test_multi_tree_cpu_cuda_value_and_gradient_parity(threads):
    dt = 0.043
    cpu_model, cpu_mechanism, cpu_integrator = _make_multi(batch=2, threads=threads)
    gpu_model, gpu_mechanism, gpu_integrator = _make_multi(batch=2, threads=threads)
    gpu_model.cuda()
    gpu_integrator.cuda()

    cpu_integrator._initialize(cpu_model, dt)
    gpu_integrator._initialize(gpu_model, dt)
    assert gpu_integrator.solve.func is dhs_solve_multi_cuda

    cpu_voltage, cpu_intra, cpu_ve = _sample_inputs(cpu_model)
    cpu_voltage.requires_grad_()
    cpu_intra.requires_grad_()
    cpu_ve.requires_grad_()
    cpu_mechanism.conductance.requires_grad_()
    cpu_mechanism.reversal.requires_grad_()

    gpu_voltage = cpu_voltage.detach().cuda().requires_grad_()
    gpu_intra = cpu_intra.detach().cuda().requires_grad_()
    gpu_ve = cpu_ve.detach().cuda().requires_grad_()
    gpu_mechanism.conductance.requires_grad_()
    gpu_mechanism.reversal.requires_grad_()

    cpu_actual, _ = cpu_integrator._step(
        cpu_voltage, dt, cpu_model.celsius, ve=cpu_ve, intra=cpu_intra
    )
    gpu_actual, _ = gpu_integrator._step(
        gpu_voltage, dt, gpu_model.celsius, ve=gpu_ve, intra=gpu_intra
    )
    cpu_expected = _dense_reference(
        cpu_model,
        cpu_mechanism,
        cpu_voltage,
        dt,
        ve=cpu_ve,
        intra=cpu_intra,
    )

    torch.testing.assert_close(cpu_actual, cpu_expected, rtol=3.0e-12, atol=3.0e-12)
    torch.testing.assert_close(gpu_actual.cpu(), cpu_actual, rtol=3.0e-10, atol=3.0e-10)

    cpu_weights = torch.linspace(0.7, 1.4, cpu_actual.numel(), dtype=DTYPE).reshape_as(
        cpu_actual
    )
    gpu_weights = cpu_weights.cuda()
    cpu_inputs = (
        cpu_voltage,
        cpu_intra,
        cpu_ve,
        cpu_mechanism.conductance,
        cpu_mechanism.reversal,
    )
    gpu_inputs = (
        gpu_voltage,
        gpu_intra,
        gpu_ve,
        gpu_mechanism.conductance,
        gpu_mechanism.reversal,
    )
    cpu_grad = torch.autograd.grad(
        (cpu_actual.square() * cpu_weights).sum(), cpu_inputs
    )
    gpu_grad = torch.autograd.grad(
        (gpu_actual.square() * gpu_weights).sum(), gpu_inputs
    )
    for got, want in zip(gpu_grad, cpu_grad):
        torch.testing.assert_close(got.cpu(), want, rtol=3.0e-8, atol=3.0e-9)
