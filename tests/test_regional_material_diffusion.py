"""Public-API contracts for compartment-restricted material diffusion."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mechanisms._material_process import DiffusionProcess
from dendra.models.mechanisms._spatial import SpatialOperator1D

DTYPE = torch.float64


class _RegionalExplicitDiffusion(DiffusionProcess):
    DiffusionProcess.RANGE(D=1.0)
    DiffusionProcess.METHOD("explicit", solver="dense")
    DiffusionProcess.DIFFUSE("tracer", field="c", D="D")


class _RegionalImplicitDiffusion(DiffusionProcess):
    DiffusionProcess.RANGE(D=1.0)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("tracer", field="c", D="D")


class _SecondRegionalImplicitDiffusion(DiffusionProcess):
    DiffusionProcess.RANGE(D=1.0)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("tracer", field="c", D="D")


class _RegionalAutoImplicitDiffusion(DiffusionProcess):
    DiffusionProcess.RANGE(D=1.0)
    DiffusionProcess.METHOD("implicit", solver="auto")
    DiffusionProcess.DIFFUSE("tracer", field="c", D="D")


class _TreeSeriesDiffusion(DiffusionProcess):
    DiffusionProcess.RANGE(D=1.0)
    DiffusionProcess.METHOD("implicit", interface_scheme="series")
    DiffusionProcess.DIFFUSE("tracer", field="c", D="D")


class _ConstantDiffusivity(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, buffer):
        return buffer.new_tensor(self.value)


def _chain(initial, *, diam=None, dx=None):
    initial = torch.as_tensor(initial, dtype=DTYPE)
    compartments = int(initial.numel())
    model = dn.Population(
        N=1,
        C=compartments,
        integrator=dn.bwd_euler_sc(),
        v_init=-65.0,
        dtype=DTYPE,
    )
    diam = (
        torch.full((compartments,), 2.0, dtype=DTYPE)
        if diam is None
        else torch.as_tensor(diam, dtype=DTYPE)
    )
    dx = (
        torch.ones(compartments, dtype=DTYPE)
        if dx is None
        else torch.as_tensor(dx, dtype=DTYPE)
    )
    model.diam.copy_(diam.reshape_as(model.diam))
    model.dx.copy_(dx.reshape_as(model.dx))
    model.material(
        "tracer",
        fields={"c": initial.reshape(1, compartments)},
        min_values={"c": 0.0},
        domain="intracellular",
    )
    return model


def _tree(initial, *, root_volume=1.0):
    graph = nx.DiGraph()
    for node in range(5):
        volume = root_volume if node == 0 else 1.0
        is_junction = node == 0 and root_volume == 0.0
        graph.add_node(
            node,
            name=f"compartment.{node}",
            kind="junction" if is_junction else "compartment",
            L=0.0 if is_junction else 1.0,
            diam=1.0,
            Ra=100.0,
            cm=1.0,
            area=0.0 if is_junction else 1.0,
            volume=volume,
            volume_i=volume,
            volume_o=0.0,
            x=float(node),
            y=0.0,
            z=0.0,
        )
    for parent, child in ((0, 1), (0, 2), (2, 3), (2, 4)):
        graph.add_edge(
            parent,
            child,
            diff_geom_um=1.0,
            R_ohm=1.0e8,
            L=1.0,
        )
    model = dn.Tree.from_graph(graph, N=1, v_init=-65.0, dtype=DTYPE)
    model.material(
        "tracer",
        fields={"c": torch.as_tensor(initial, dtype=DTYPE).reshape(1, -1)},
        min_values={"c": 0.0},
        domain="intracellular",
    )
    return model


def _initialize(model):
    model.eval()
    model.initialize()
    return model


def _concentration(model):
    return model.mech.materials["tracer"].c


def test_series_interface_scheme_matches_half_compartment_resistances():
    operator = SpatialOperator1D(interface_scheme="series")
    diam = torch.tensor([[2.0, 4.0]], dtype=DTYPE)
    dx = torch.tensor([[1.0, 3.0]], dtype=DTYPE)
    diffusivity = torch.tensor([[1.0, 0.5]], dtype=DTYPE)

    _, conductance = operator.edge_conductance(
        diffusivity, diam, dx, area_fraction=0.25
    )
    area = torch.pi * (0.5 * diam) ** 2
    expected = 0.25 / (
        0.5 * dx[:, :1] / (diffusivity[:, :1] * area[:, :1])
        + 0.5 * dx[:, 1:] / (diffusivity[:, 1:] * area[:, 1:])
    )
    torch.testing.assert_close(conductance, expected, rtol=2e-15, atol=2e-15)

    _, blocked = operator.edge_conductance(
        torch.tensor([[1.0, 0.0]], dtype=DTYPE), diam, dx
    )
    assert torch.equal(blocked, torch.zeros_like(blocked))

    regional = SpatialOperator1D(interface_scheme="series", solver="dense")
    regional.configure_diffusion(
        torch.zeros((1, 3), dtype=DTYPE),
        0.1,
        torch.tensor([[1.0, 1.0, -123.0]], dtype=DTYPE),
        torch.full((1, 3), 2.0, dtype=DTYPE),
        torch.ones((1, 3), dtype=DTYPE),
        node_mask=torch.tensor([[True, True, False]]),
    )
    assert regional.g_edge[0, 0] > 0
    assert regional.g_edge[0, 1] == 0


def test_regional_chain_uses_induced_edges_for_disconnected_islands():
    initial = torch.tensor([1.0, 0.0, 23.0, 29.0, 10.0, 0.0, 31.0], dtype=DTYPE)
    model = _chain(initial)
    model[:, [0, 1, 4, 5]].insert(_RegionalExplicitDiffusion, D=1.0)
    _initialize(model)

    before = _concentration(model).clone()
    model.step(dt=0.1)
    actual = _concentration(model)

    # Sparse support order is [0, 1, 4, 5], but physical compartments 1 and 4
    # must never acquire a synthetic edge.
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.9, 0.1, 23.0, 29.0, 9.0, 1.0, 31.0]], dtype=DTYPE),
        rtol=2e-15,
        atol=2e-15,
    )
    torch.testing.assert_close(actual[:, [0, 1]].sum(), before[:, [0, 1]].sum())
    torch.testing.assert_close(actual[:, [4, 5]].sum(), before[:, [4, 5]].sum())
    assert torch.equal(actual[:, [2, 3, 6]], before[:, [2, 3, 6]])


def test_regional_range_diffusivity_is_scattered_before_edge_masking():
    model = _chain([1.0, 0.0, 99.0])
    model[:, [0, 1]].insert(
        _RegionalExplicitDiffusion,
        D=torch.tensor([2.0, 4.0], dtype=DTYPE),
    )
    _initialize(model)

    before = _concentration(model).clone()
    model.step(dt=0.1)
    actual = _concentration(model)

    # D_edge=(2+4)/2=3 on 0--1.  The crossing edge 1--2 is sealed after
    # endpoint averaging, even though its selected endpoint has nonzero D.
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.7, 0.3, 99.0]], dtype=DTYPE),
        rtol=2e-15,
        atol=2e-15,
    )
    assert torch.equal(actual[:, 2], before[:, 2])


def test_full_slice_regional_diffusion_matches_global_insertion():
    initial = torch.tensor([1.3, 0.2, 0.9, 0.1, 2.0], dtype=DTYPE)
    diam = torch.tensor([1.4, 2.0, 1.2, 2.4, 1.8], dtype=DTYPE)
    dx = torch.tensor([0.7, 1.2, 0.9, 1.4, 0.8], dtype=DTYPE)
    diffusivity = torch.tensor([0.2, 0.5, 0.9, 0.4, 0.7], dtype=DTYPE)

    global_model = _chain(initial, diam=diam, dx=dx)
    global_model.insert(_RegionalImplicitDiffusion, D=diffusivity)
    _initialize(global_model)
    regional_model = _chain(initial, diam=diam, dx=dx)
    regional_model[:, :].insert(_RegionalImplicitDiffusion, D=diffusivity)
    _initialize(regional_model)

    for _ in range(5):
        global_model.step(dt=0.07)
        regional_model.step(dt=0.07)
    torch.testing.assert_close(
        _concentration(regional_model),
        _concentration(global_model),
        rtol=2e-13,
        atol=2e-13,
    )


def test_regional_tree_support_is_an_induced_forest():
    model = _tree([100.0, 5.0, 1.0, 0.0, 200.0])
    model[:, [1, 2, 3]].insert(_RegionalExplicitDiffusion, D=1.0)
    _initialize(model)

    before = _concentration(model).clone()
    before_mass = torch.dot(model.volume_i[0, [1, 2, 3]], before[0, [1, 2, 3]])
    model.step(dt=0.1)
    actual = _concentration(model)

    torch.testing.assert_close(
        actual,
        torch.tensor([[100.0, 5.0, 0.9, 0.1, 200.0]], dtype=DTYPE),
        rtol=2e-15,
        atol=2e-15,
    )
    assert torch.equal(actual[:, [0, 4]], before[:, [0, 4]])
    after_mass = torch.dot(model.volume_i[0, [1, 2, 3]], actual[0, [1, 2, 3]])
    torch.testing.assert_close(after_mass, before_mass, rtol=2e-15, atol=2e-15)


def test_excluded_zero_volume_tree_junction_is_an_identity_row():
    model = _tree([7.0, 1.0, 2.0, 3.0, 4.0], root_volume=0.0)
    model[:, [1, 2, 3, 4]].insert(_RegionalImplicitDiffusion, D=1.0)
    _initialize(model)

    before = _concentration(model).clone()
    model.step(dt=0.1)
    assert torch.equal(_concentration(model)[:, 0], before[:, 0])
    assert torch.isfinite(_concentration(model)).all()


def test_active_zero_volume_tree_component_is_rejected_before_solve():
    model = _tree([7.0, 1.0, 2.0, 3.0, 4.0], root_volume=0.0)
    model[:, [0]].insert(_RegionalAutoImplicitDiffusion, D=1.0)
    model.initialize()
    with pytest.raises(ValueError, match="positive material volume"):
        model.step(dt=0.1)


def test_tree_rejects_1d_only_series_interface_scheme():
    model = _tree([1.0, 0.0, 0.0, 0.0, 0.0])
    model.insert(_TreeSeriesDiffusion, D=1.0)
    with pytest.raises(NotImplementedError, match="analytic 1-D"):
        model.initialize()


def test_repeated_regions_union_and_duplicate_slots_are_rejected():
    model = _chain([0.0, 0.0, 1.0, 0.0, 0.0])
    model[:, [0, 1, 2]].insert(_RegionalExplicitDiffusion, D=1.0)
    model[:, [2, 3, 4]].insert(_RegionalExplicitDiffusion, D=1.0)
    _initialize(model)
    assert len(model.mech.material_processes) == 1
    model.step(dt=0.1)
    torch.testing.assert_close(
        _concentration(model),
        torch.tensor([[0.0, 0.1, 0.8, 0.1, 0.0]], dtype=DTYPE),
        rtol=2e-15,
        atol=2e-15,
    )

    duplicate = _chain([1.0, 0.0, 0.0])
    duplicate[:, [0, 1]].insert(_RegionalExplicitDiffusion, D=1.0, copies=2)
    with pytest.raises(ValueError, match="set-valued"):
        duplicate.initialize()


def test_overlapping_regions_require_unambiguous_diffusivity_and_writer():
    conflicting = _chain([1.0, 0.0, 0.0, 0.0, 0.0])
    conflicting[:, [0, 1, 2]].insert(_RegionalImplicitDiffusion, alias="left", D=1.0)
    conflicting[:, [2, 3, 4]].insert(_RegionalImplicitDiffusion, alias="right", D=2.0)
    with pytest.raises(ValueError, match="conflicting.*diffusivity"):
        conflicting.initialize()

    separate = _chain([1.0, 0.0, 0.0, 0.0])
    separate[:, [0, 1]].insert(_RegionalImplicitDiffusion, D=1.0)
    separate[:, [1, 2]].insert(_SecondRegionalImplicitDiffusion, D=1.0)
    with pytest.raises(ValueError, match="Overlapping DiffusionProcess"):
        separate.initialize()

    disjoint = _chain([1.0, 0.0, 0.0, 0.0])
    disjoint[:, [0, 1]].insert(_RegionalImplicitDiffusion, D=1.0)
    disjoint[:, [2, 3]].insert(_SecondRegionalImplicitDiffusion, D=1.0)
    _initialize(disjoint)
    assert len(disjoint.mech.material_processes) == 2


def test_overlapping_dynamic_diffusivity_overrides_are_rejected():
    model = _chain([1.0, 0.0, 0.0, 0.0])
    model[:, [0, 1, 2]].insert(
        _RegionalImplicitDiffusion,
        alias="left",
        D=_ConstantDiffusivity(1.0),
    )
    model[:, [2, 3]].insert(
        _RegionalImplicitDiffusion,
        alias="right",
        D=_ConstantDiffusivity(1.0),
    )
    with pytest.raises(ValueError, match="dynamic module-valued diffusivity"):
        model.initialize()


@pytest.mark.parametrize("regional", [False, True])
def test_1d_diffusion_preserves_live_diameter_gradients(regional):
    model = _chain([1.0, 0.0, 0.0], diam=[1.4, 2.0, 2.6])
    model.diam.requires_grad_(True)
    if regional:
        model[:, [0, 1]].insert(_RegionalExplicitDiffusion, D=1.0)
    else:
        model.insert(_RegionalExplicitDiffusion, D=1.0)
    model.train()
    model.initialize()

    model.step(dt=0.05)
    _concentration(model)[0, 1].backward()

    assert model.diam.grad is not None
    assert torch.isfinite(model.diam.grad).all()
    assert torch.any(model.diam.grad != 0)
