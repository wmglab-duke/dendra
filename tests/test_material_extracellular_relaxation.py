"""Known-answer tests for extracellular geometry and fused reservoir coupling."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import DiffusionProcess
from dendra.models.mechanisms._spatial import SpatialOperator1D, SpatialOperatorTree

DTYPE = torch.float64


class _ExtracellularEdgeRelaxation(DiffusionProcess):
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE(
        "solute",
        field="c",
        D="edge_diffusivity",
        domain="extracellular",
        geometry="periaxonal",
        D_location="edge",
    )
    DiffusionProcess.RELAX(
        "solute",
        field="c",
        rate="bath_rate",
        target="bath_target",
        where="bath_mask",
    )


class _ExplicitRelaxation(DiffusionProcess):
    DiffusionProcess.METHOD("explicit", solver="dense")
    DiffusionProcess.DIFFUSE("solute", field="c", D=0.0)
    DiffusionProcess.RELAX("solute", field="c", rate=1.0, target=0.0)


def _dense_chain_reaction_diffusion(c, volume, conductance, dt, rate, target):
    """Independent backward-Euler finite-volume oracle."""
    c = c.reshape(-1)
    volume = volume.reshape(-1)
    conductance = conductance.reshape(-1)
    rate = rate.reshape(-1)
    target = target.reshape(-1)
    count = int(c.numel())
    matrix = torch.diag(volume + dt * volume * rate)
    for edge, coupling in enumerate(conductance):
        left, right = edge, edge + 1
        matrix[left, left] += dt * coupling
        matrix[right, right] += dt * coupling
        matrix[left, right] -= dt * coupling
        matrix[right, left] -= dt * coupling
    rhs = volume * c + dt * volume * rate * target
    return torch.linalg.solve(matrix, rhs).reshape(1, count)


def _extracellular_chain(
    initial, volume, edge_area, edge_distance, edge_D, rate, target, mask
):
    initial = torch.as_tensor(initial, dtype=DTYPE).reshape(1, -1)
    count = int(initial.shape[-1])
    with dn.ctx(DTYPE=DTYPE):
        model = dn.Population(
            N=1,
            C=count,
            integrator=dn.bwd_euler_sc(),
            v_init=-65.0,
            dtype=DTYPE,
        )
        model.material(
            "solute",
            fields={"c": initial},
            min_values={"c": 0.0},
            domain="extracellular",
        )
        model.register_material_geometry(
            "periaxonal",
            domain="extracellular",
            volume=torch.as_tensor(volume, dtype=DTYPE).reshape(1, -1),
            edge_area=torch.as_tensor(edge_area, dtype=DTYPE).reshape(1, -1),
            edge_distance=torch.as_tensor(edge_distance, dtype=DTYPE).reshape(1, -1),
        )
        model.register_buffer(
            "edge_diffusivity", torch.as_tensor(edge_D, dtype=DTYPE).reshape(1, -1)
        )
        model.register_buffer(
            "bath_rate", torch.as_tensor(rate, dtype=DTYPE).reshape(1, -1)
        )
        model.register_buffer(
            "bath_target", torch.as_tensor(target, dtype=DTYPE).reshape(1, -1)
        )
        model.register_buffer(
            "bath_mask", torch.as_tensor(mask, dtype=torch.bool).reshape(1, -1)
        )
        model.insert(_ExtracellularEdgeRelaxation)
        model.eval()
        model.initialize()
    return model


def test_named_extracellular_edge_diffusion_and_masked_relaxation_match_dense_oracle():
    initial = torch.tensor([14.0, 2.0, 7.0, 20.0], dtype=DTYPE)
    volume = torch.tensor([0.2, 1.7, 0.4, 2.2], dtype=DTYPE)
    edge_area = torch.tensor([0.03, 0.9, 0.12], dtype=DTYPE)
    edge_distance = torch.tensor([0.5, 1.5, 0.25], dtype=DTYPE)
    edge_D = torch.tensor([0.4, 3.0, 0.15], dtype=DTYPE)
    # The large unmasked value proves that reservoir support is independent of
    # the transport domain and that masked coefficients are semantically inert.
    rate = torch.tensor([35.0, 9.0e8, 0.0, 0.8], dtype=DTYPE)
    target = torch.tensor([5.0, float("nan"), 11.0, 3.0], dtype=DTYPE)
    mask = torch.tensor([True, False, False, True])
    dt = 0.07

    model = _extracellular_chain(
        initial, volume, edge_area, edge_distance, edge_D, rate, target, mask
    )
    before = model.mech.materials["solute"].c.clone()
    model.step(dt=dt)
    actual = model.mech.materials["solute"].c

    conductance = edge_D * edge_area / edge_distance
    effective_rate = torch.where(mask, rate, torch.zeros_like(rate))
    effective_target = torch.where(mask, target, torch.zeros_like(target))
    expected = _dense_chain_reaction_diffusion(
        initial,
        volume,
        conductance,
        dt,
        effective_rate,
        effective_target,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)

    process = next(iter(model.mech.material_processes.values()))
    operator = next(iter(process._spatial_operators.values()))
    torch.testing.assert_close(
        operator.g_edge.reshape(-1), conductance, rtol=2e-15, atol=2e-15
    )

    # Sealed edge fluxes cancel pairwise. The only modeled-domain mass change
    # is the backward-Euler exchange with the fixed reservoir.
    mass_change = (volume * (actual.reshape(-1) - before.reshape(-1))).sum()
    bath_flux = (
        -dt * (volume * effective_rate * (actual.reshape(-1) - effective_target)).sum()
    )
    torch.testing.assert_close(mass_change, bath_flux, rtol=3e-13, atol=3e-13)


def test_zero_edge_diffusion_singleton_stiff_relaxation_uses_implicit_update():
    initial = torch.tensor([17.0], dtype=DTYPE)
    volume = torch.tensor([0.004], dtype=DTYPE)
    rate = torch.tensor([2.0e7], dtype=DTYPE)
    target = torch.tensor([5.0], dtype=DTYPE)
    dt = 0.2
    model = _extracellular_chain(
        initial,
        volume,
        torch.empty(0, dtype=DTYPE),
        torch.empty(0, dtype=DTYPE),
        torch.empty(0, dtype=DTYPE),
        rate,
        target,
        torch.tensor([True]),
    )

    model.step(dt=dt)
    actual = model.mech.materials["solute"].c
    expected = (initial + dt * rate * target) / (1.0 + dt * rate)
    torch.testing.assert_close(actual.reshape(-1), expected, rtol=2e-15, atol=2e-15)
    assert torch.isfinite(actual).all()
    assert abs(actual.item() - target.item()) < 4e-6


def test_relaxation_rejects_explicit_process_and_negative_active_rate():
    model = dn.Population(N=1, C=2, integrator=dn.bwd_euler_sc(), dtype=DTYPE)
    model.material(
        "solute",
        fields={"c": torch.ones((1, 2), dtype=DTYPE)},
        domain="intracellular",
    )
    model.insert(_ExplicitRelaxation)
    with pytest.raises(NotImplementedError, match="RELAX.*implicit"):
        model.initialize()

    operator = _configured_tree_operator()
    with pytest.raises(ValueError, match="finite and non-negative"):
        operator.configure_relaxation(
            torch.tensor([[1.0, -0.1, 1.0]], dtype=DTYPE),
            target=0.0,
        )

    # Invalid coefficients outside the independent reaction mask are ignored.
    operator = _configured_tree_operator()
    operator.configure_relaxation(
        torch.tensor([[2.0, -123.0, float("nan")]], dtype=DTYPE),
        target=torch.tensor([[4.0, float("nan"), float("nan")]], dtype=DTYPE),
        where=torch.tensor([[True, False, False]]),
    )
    updated = operator.diffuse_implicit_configured(
        torch.tensor([[9.0, 2.0, 1.0]], dtype=DTYPE)
    )
    assert torch.isfinite(updated).all()


def _configured_tree_operator():
    graph = nx.DiGraph()
    for node, volume in enumerate((0.7, 1.1, 0.4)):
        graph.add_node(node, volume_i=volume)
    graph.add_edge(0, 1, diff_geom_um=0.3)
    graph.add_edge(0, 2, diff_geom_um=1.4)

    class _TreeGeometry:
        def __init__(self):
            self.graph = graph
            self.volume_i = torch.tensor([[0.7, 1.1, 0.4]], dtype=DTYPE)

        def material_volume(self, domain):
            assert domain == "intracellular"
            return self.volume_i

    operator = SpatialOperatorTree(solver="dense")
    operator.configure_diffusion(
        torch.zeros((1, 3), dtype=DTYPE),
        0.1,
        torch.tensor([[0.6, 0.2, 1.0]], dtype=DTYPE),
        _TreeGeometry(),
        domain="intracellular",
        solver="dense",
    )
    return operator


def test_tree_masked_relaxation_matches_independent_dense_system_and_has_gradients():
    operator = _configured_tree_operator()
    initial = torch.tensor([[8.0, 2.0, 11.0]], dtype=DTYPE)
    rate = torch.tensor([[4.0, 1000.0, 0.3]], dtype=DTYPE, requires_grad=True)
    target = torch.tensor([[1.5, 99.0, 6.0]], dtype=DTYPE, requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    operator.configure_relaxation(rate, target, where=mask)
    actual = operator.diffuse_implicit_configured(initial)

    volume = torch.tensor([0.7, 1.1, 0.4], dtype=DTYPE)
    node_D = torch.tensor([0.6, 0.2, 1.0], dtype=DTYPE)
    conductance = torch.tensor(
        [0.5 * (node_D[0] + node_D[1]) * 0.3, 0.5 * (node_D[0] + node_D[2]) * 1.4],
        dtype=DTYPE,
    )
    effective_rate = torch.where(mask.reshape(-1), rate.reshape(-1), 0.0)
    effective_target = torch.where(mask.reshape(-1), target.reshape(-1), 0.0)
    matrix = torch.diag(volume / 0.1 + volume * effective_rate)
    for parent, child, coupling in ((0, 1, conductance[0]), (0, 2, conductance[1])):
        matrix[parent, parent] += coupling
        matrix[child, child] += coupling
        matrix[parent, child] -= coupling
        matrix[child, parent] -= coupling
    rhs = (
        volume / 0.1 * initial.reshape(-1) + volume * effective_rate * effective_target
    )
    expected = torch.linalg.solve(matrix, rhs).reshape_as(actual)
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)

    loss = actual.square().sum()
    rate_grad, target_grad = torch.autograd.grad(loss, (rate, target))
    assert torch.isfinite(rate_grad).all() and torch.isfinite(target_grad).all()
    assert rate_grad[0, 0] != 0 and rate_grad[0, 2] != 0
    assert target_grad[0, 0] != 0 and target_grad[0, 2] != 0
    assert rate_grad[0, 1] == 0 and target_grad[0, 1] == 0


def test_named_geometry_refreshes_live_population_edge_buffers():
    model = _extracellular_chain(
        initial=[9.0, 1.0],
        volume=[1.0, 1.0],
        edge_area=[1.0],
        edge_distance=[1.0],
        edge_D=[0.5],
        rate=[0.0, 0.0],
        target=[0.0, 0.0],
        mask=[False, False],
    )
    process = next(iter(model.mech.material_processes.values()))
    material = model.mech.materials["solute"]
    initial = material.c.clone()
    dt = 0.1

    process.set_dt(dt)
    process.advance_materials(dt)
    first = material.c.clone()

    material.c.copy_(initial)
    model.edge_diffusivity.fill_(1.0)
    process.set_dt(dt)
    process.advance_materials(dt)
    second = material.c.clone()

    operator = next(iter(process._spatial_operators.values()))
    torch.testing.assert_close(operator.g_edge, torch.ones_like(operator.g_edge))
    assert second[0, 0] < first[0, 0]
    assert second[0, 1] > first[0, 1]


def test_nodewise_dt_row_scaling_preserves_constant_fields():
    concentration = torch.full((1, 3), 4.5, dtype=DTYPE)
    operator = SpatialOperator1D(solver="dense")
    operator.configure_diffusion(
        concentration,
        torch.tensor([[0.1, 0.2, 0.3]], dtype=DTYPE),
        D_um2_per_ms=torch.tensor([[0.4, 1.2, 0.7]], dtype=DTYPE),
        diam_um=torch.tensor([[1.0, 2.0, 0.5]], dtype=DTYPE),
        dx_um=torch.tensor([[0.5, 1.0, 2.0]], dtype=DTYPE),
        solver="dense",
    )
    actual = operator.diffuse_implicit_configured(concentration)
    torch.testing.assert_close(actual, concentration, rtol=2e-15, atol=2e-15)
