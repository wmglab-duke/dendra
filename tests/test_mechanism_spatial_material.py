"""Known-answer tests for spatial operators and material processes."""

from __future__ import annotations

import math
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

from dendra.models.mechanisms import _spatial as spatial
from dendra.models.mechanisms._material_process import (
    ClampProcess,
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
    MaterialProcess,
    _canonical_domain,
    _material_field_ref,
    _safe_key,
)
from dendra.models.mechanisms._materials import Material, MaterialFieldSpec

SHAPE = (1, 3)


def _process(cls, *, shape=SHAPE, diam=2.0, key=None):
    return cls(
        name=cls.__name__.lower(),
        celsius=torch.tensor(37.0),
        diameters=torch.full(shape, diam),
        shape=shape,
        shape_f=shape,
        key=key,
    )


def _material(name, values, *, domain="i"):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    spec = MaterialFieldSpec("c", initial=values, domain=domain)
    return Material(name, tuple(values.shape), specs={"c": spec})


def _bind(process, materials, population=None):
    return process._bind_materials(materials.__getitem__, population=population)


def _snapshot_material_state(materials):
    return {
        name: {
            field: (material._buffers[field], material._buffers[field].detach().clone())
            for field in material.fields
        }
        for name, material in materials.items()
    }


def _assert_material_state_unchanged(materials, snapshot):
    assert materials.keys() == snapshot.keys()
    for name, fields in snapshot.items():
        material = materials[name]
        assert set(material.fields) == set(fields)
        for field, (original, expected) in fields.items():
            assert material._buffers[field] is original
            assert torch.equal(material._buffers[field], expected)


def _snapshot_module_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_module_state_unchanged(module, snapshot):
    actual = module.state_dict()
    assert actual.keys() == snapshot.keys()
    for name, expected in snapshot.items():
        assert torch.equal(actual[name], expected), name


def _snapshot_diffusion_process(process):
    return {
        "state": _snapshot_module_state(process),
        "spatial_configured": process._spatial_configured,
        "operators": tuple(
            (
                id(op),
                op.configured,
                op.solver_name,
                op.B,
                op.K,
                op.base_shape,
            )
            for op in process._spatial_operators.values()
        ),
    }


def _assert_diffusion_process_unchanged(process, snapshot):
    _assert_module_state_unchanged(process, snapshot["state"])
    assert process._spatial_configured is snapshot["spatial_configured"]
    assert (
        tuple(
            (
                id(op),
                op.configured,
                op.solver_name,
                op.B,
                op.K,
                op.base_shape,
            )
            for op in process._spatial_operators.values()
        )
        == snapshot["operators"]
    )


class ExactClearance(ClearanceProcess):
    ClearanceProcess.CLEAR("x", field="c", rate=0.5, target=1.0)


class ImplicitClearance(ClearanceProcess):
    ClearanceProcess.METHOD("implicit")
    ClearanceProcess.CLEAR("x", field="c", rate=0.5, target=1.0)


class ExplicitClearance(ClearanceProcess):
    ClearanceProcess.METHOD("explicit")
    ClearanceProcess.CLEAR("x", field="c", rate=0.5, target=1.0)


class ClampedNegativeClearance(ClearanceProcess):
    ClearanceProcess.CLEAR("x", field="c", rate=-1.0)


class TerminalSet(ClampProcess):
    ClampProcess.SET("x", field="c", value=5.0, where="terminal")


class RangeClamp(ClampProcess):
    ClampProcess.BOUNDS("x", field="c", lower=0.0, upper=1.0)


class IndexedMaximum(ClampProcess):
    ClampProcess.MAX("x", field="c", value=1.0, where=1)


class ExactExchange(ExchangeProcess):
    ExchangeProcess.EXCHANGE("a.c", "b.c", rate=1.0, volume_a=1.0, volume_b=1.0)


class ImplicitExchange(ExchangeProcess):
    ExchangeProcess.METHOD("implicit")
    ExchangeProcess.EXCHANGE("a.c", "b.c", rate=1.0, volume_a=1.0, volume_b=1.0)


class ExplicitExchange(ExchangeProcess):
    ExchangeProcess.METHOD("explicit")
    ExchangeProcess.EXCHANGE("a.c", "b.c", rate=1.0, volume_a=1.0, volume_b=1.0)


class WeightedExchange(ExchangeProcess):
    ExchangeProcess.EXCHANGE("a.c", "b.c", rate=1.0, volume_a=1.0, volume_b=2.0)


class ExplicitDiffusion(DiffusionProcess):
    DiffusionProcess.METHOD("explicit", solver="dense")
    DiffusionProcess.DIFFUSE("x", field="c", D=1.0)


class ImplicitDiffusion(DiffusionProcess):
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("x", field="c", D=1.0)


def test_spatial_shape_and_geometry_helpers():
    target = torch.zeros((2, 3), dtype=torch.float64)
    assert spatial._as_tensor_like(2, target).dtype == torch.float64
    assert torch.equal(
        spatial._broadcast_to(torch.tensor([1, 2, 3]), target),
        torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float64),
    )
    flat, shape = spatial._flatten_last(target)
    assert flat.shape == (2, 3) and shape == (2, 3)
    with pytest.raises(ValueError, match="spatial dimension"):
        spatial._flatten_last(torch.tensor(1.0))

    op = spatial.SpatialOperator1D(solver="dense")
    diam = torch.full((1, 3), 2.0)
    volume, edge_area = op.finite_volume_geometry(diam, torch.ones_like(diam))
    assert torch.allclose(volume, torch.full_like(volume, torch.pi))
    assert torch.allclose(edge_area, torch.full_like(edge_area, torch.pi))
    volume, conductance = op.edge_conductance(2.0, diam, torch.ones_like(diam))
    assert torch.allclose(volume, torch.full_like(volume, torch.pi))
    assert torch.allclose(conductance, torch.full_like(conductance, 2 * torch.pi))


def test_dense_tridiagonal_known_answer_and_singleton():
    a = torch.tensor([[1.0, 1.0]])
    b = torch.tensor([[4.0, 4.0, 3.0]])
    c = torch.tensor([[1.0, 1.0]])
    rhs = torch.tensor([[6.0, 12.0, 11.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(
        spatial.solve_tridiagonal_1d(a, b, c, rhs, solver="dense"), expected
    )
    assert torch.allclose(spatial._dense_tridiagonal_solve(a, b, c, rhs), expected)
    assert torch.equal(
        spatial.solve_tridiagonal_1d(
            torch.empty(1, 0),
            torch.tensor([[2.0]]),
            torch.empty(1, 0),
            torch.tensor([[6.0]]),
        ),
        torch.tensor([[3.0]]),
    )


def test_solver_selection_normalization_and_cpu_fallback(monkeypatch):
    assert spatial._normalize_solver_name("inv") == "thomas"
    assert spatial._normalize_solver_name("torch") == "dense"
    assert spatial.select_tridiagonal_solver("dense", "cpu") == (None, "dense")
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    monkeypatch.setattr(spatial, "pcr_solve_t", None)
    assert spatial.select_tridiagonal_solver("auto", "cpu") == (None, "dense")
    with pytest.warns(UserWarning, match="Unknown material"):
        assert spatial.select_tridiagonal_solver("mystery", "cpu") == (None, "dense")
    with pytest.raises(ImportError, match="requires"):
        spatial.select_tridiagonal_solver("pcr", "cpu")


def test_chain_diffusion_known_answers_and_mass_conservation():
    c = torch.tensor([[1.0, 0.0, 0.0]])
    geom = torch.full_like(c, 2.0)
    dx = torch.ones_like(c)
    explicit = spatial.SpatialOperator1D(solver="dense").diffuse_explicit(
        c, 0.1, 1.0, geom, dx
    )
    assert torch.allclose(explicit, torch.tensor([[0.9, 0.1, 0.0]]))

    implicit = spatial.SpatialOperator1D(solver="dense").diffuse_implicit(
        c, 0.1, 1.0, geom, dx
    )
    matrix = torch.tensor([[1.1, -0.1, 0.0], [-0.1, 1.2, -0.1], [0.0, -0.1, 1.1]])
    expected = torch.linalg.solve(matrix, c[0]).unsqueeze(0)
    assert torch.allclose(implicit, expected, atol=1e-6)
    assert explicit.sum() == pytest.approx(c.sum().item())
    assert implicit.sum() == pytest.approx(c.sum().item())


def test_chain_configuration_guards_and_single_compartment():
    with pytest.raises(NotImplementedError, match="sealed"):
        spatial.SpatialOperator1D(boundary="open")
    op = spatial.SpatialOperator1D(solver="dense")
    with pytest.raises(RuntimeError, match="not been configured"):
        op.diffuse_explicit_configured(torch.ones(1, 2))
    with pytest.raises(ValueError, match="final compartment"):
        op.configure_diffusion(
            torch.tensor(1.0), 0.1, 1.0, torch.tensor(2.0), torch.tensor(1.0)
        )
    single = torch.tensor([[4.0]])
    op.configure_diffusion(
        single, 0.1, 1.0, torch.ones_like(single), torch.ones_like(single)
    )
    assert op.solver_name == "trivial"
    assert op.diffuse_implicit_configured(single) is single
    assert op.diffuse_explicit_configured(single) is single


def test_tree_morphology_layers_and_invalid_parent_vectors():
    parent, children, depth = spatial._build_tree_morphology([-1, 0, 0, 2])
    assert parent.tolist() == [-1, 0, 0, 2]
    assert children == [[1, 2], [], [3], []]
    assert depth.tolist() == [0, 1, 1, 2]
    order, layer_ptr = spatial._build_dhs_layers(depth, threads=2)
    assert order.tolist() == [3, 1, 2, 0]
    assert layer_ptr.tolist() == [0, 1, 3, 4]

    for bad, message in [
        ([-1, -1], "more than one root"),
        ([0], "one root"),
        ([-1, 3], "invalid parent"),
        ([-1, 2, 1], "connected"),
    ]:
        with pytest.raises(ValueError, match=message):
            spatial._build_tree_morphology(bad)


@pytest.mark.parametrize("threads", [0, -1, 3, 64])
def test_tree_operator_rejects_invalid_thread_counts(threads):
    with pytest.raises(ValueError, match="positive"):
        spatial.SpatialOperatorTree(threads=threads)


def test_graph_geometry_conversion_and_validation():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, diff_geom_um=2.0)
    graph.add_edge(0, 2, diff_geom_um=3.0)
    parent, geom, nodes = spatial.graph_to_parent_and_diffusion(graph)
    assert parent.tolist() == [-1, 0, 0]
    assert geom.tolist() == [[0.0, 2.0, 3.0]]
    assert nodes == [0, 1, 2]

    with pytest.raises(ValueError, match="at least one graph"):
        spatial.graph_to_parent_and_diffusion([])
    multiparent = nx.DiGraph([(0, 2), (1, 2)])
    with pytest.raises(ValueError, match="2 parents"):
        spatial.graph_to_parent_and_diffusion(multiparent)


def test_tree_diffusion_known_answers_and_zero_volume_guard():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, diff_geom_um=1.0)
    model = SimpleNamespace(graph=graph, volume_i=torch.ones(1, 2))
    c = torch.tensor([[1.0, 0.0]])
    op = spatial.SpatialOperatorTree(solver="dense")
    explicit = op.diffuse_explicit(c, 0.1, 1.0, model)
    assert torch.allclose(explicit, torch.tensor([[0.9, 0.1]]))
    implicit = op.diffuse_implicit(c, 0.1, 1.0, model, solver="dense")
    assert torch.allclose(implicit, torch.tensor([[11 / 12, 1 / 12]]), atol=1e-6)
    assert implicit.sum() == pytest.approx(1.0)

    model.volume_i = torch.tensor([[0.0, 1.0]])
    op.configure_diffusion(c, 0.1, 1.0, model, solver="dense")
    with pytest.raises(RuntimeError, match="zero-volume"):
        op.diffuse_explicit_configured(c)


def test_material_lifecycle_minimum_and_detach():
    spec = MaterialFieldSpec("c", initial=torch.tensor([[-1.0, 2.0]]), min_value=0.0)
    material = Material("x", (1, 2), specs={"c": spec})
    assert material.fields == ("c",)
    assert material.has_field("c") and not material.has_field("missing")
    assert material.field_spec("c") is spec
    material.initialize()
    assert torch.equal(material.c, torch.tensor([[0.0, 2.0]]))
    material.c = material.c.clone().requires_grad_()
    assert material.detach() is material
    assert material.c.grad_fn is None
    with pytest.raises(ValueError, match="has no fields"):
        Material("empty", (1,))


def test_material_process_reference_helpers_and_unbound_error():
    assert _canonical_domain("cytosol") == "intracellular"
    assert _canonical_domain("outside") == "extracellular"
    assert _canonical_domain("custom") == "custom"
    assert _material_field_ref("ca.cai") == ("ca", "cai")
    assert _material_field_ref(("k", "ko")) == ("k", "ko")
    assert _material_field_ref({"name": "ip3"}) == ("ip3", "ip3i")
    assert _safe_key(2, "ca++", "c.a") == "2_ca___c_a"
    with pytest.raises(ValueError, match="Material field reference"):
        _material_field_ref(3)
    base = _process(MaterialProcess)
    with pytest.raises(RuntimeError, match="not been bound"):
        base._get_material("x")
    with pytest.raises(NotImplementedError, match="advance_materials"):
        base.advance_materials(0.1)


@pytest.mark.parametrize(
    "cls, expected",
    [
        (ExactClearance, 1.0 + 2.0 * math.exp(-1.0)),
        (ImplicitClearance, 2.0),
        (ExplicitClearance, 1.0),
    ],
)
def test_clearance_methods_known_answers(cls, expected):
    material = _material("x", [3.0, 3.0, 3.0])
    process = _bind(_process(cls), {"x": material})
    process.advance_materials(2.0)
    assert torch.allclose(material.c, torch.full_like(material.c, expected))


def test_clearance_clamps_negative_rates_and_validates_fields():
    material = _material("x", [2.0, 3.0, 4.0])
    process = _bind(_process(ClampedNegativeClearance), {"x": material})
    before = material.c.clone()
    process.advance_materials(1.0)
    assert torch.equal(material.c, before)

    class MissingClearance(ClearanceProcess):
        ClearanceProcess.CLEAR("x", field="missing", rate=1.0)

    with pytest.raises(ValueError, match="has no field"):
        _bind(_process(MissingClearance), {"x": material})


def test_clearance_multispec_late_resolution_failure_is_transactional():
    class LateFailureClearance(ClearanceProcess):
        ClearanceProcess.CLEAR("x", field="c", rate=0.5, target=0.0)
        ClearanceProcess.CLEAR(
            "y", field="c", rate="missing_clearance_rate", target=0.0
        )

    materials = {
        "x": _material("x", [3.0, 2.0, 1.0]),
        "y": _material("y", [4.0, 5.0, 6.0]),
    }
    process = _bind(_process(LateFailureClearance), materials)
    material_snapshot = _snapshot_material_state(materials)
    process_snapshot = _snapshot_module_state(process)

    with pytest.raises(AttributeError, match="missing_clearance_rate"):
        process.advance_materials(0.25)

    _assert_material_state_unchanged(materials, material_snapshot)
    _assert_module_state_unchanged(process, process_snapshot)


def test_clamp_modes_and_regions():
    terminal = _material("x", [1.0, 2.0, 3.0])
    process = _bind(
        _process(TerminalSet),
        {"x": terminal},
        SimpleNamespace(graph=None, shape=SHAPE),
    )
    process.advance_materials(99.0)
    assert torch.equal(terminal.c, torch.tensor([[5.0, 2.0, 5.0]]))

    bounded = _material("x", [-1.0, 0.5, 3.0])
    _bind(_process(RangeClamp), {"x": bounded}).advance_materials(0.1)
    assert torch.equal(bounded.c, torch.tensor([[0.0, 0.5, 1.0]]))

    indexed = _material("x", [-1.0, 2.0, 3.0])
    _bind(_process(IndexedMaximum), {"x": indexed}).advance_materials(0.1)
    assert torch.equal(indexed.c, torch.tensor([[-1.0, 1.0, 3.0]]))


def test_clamp_rejects_unknown_region_and_bad_range():
    class UnknownRegion(ClampProcess):
        ClampProcess.SET("x", field="c", where="nowhere")

    with pytest.raises(ValueError, match="Unsupported clamp region"):
        _bind(_process(UnknownRegion), {"x": _material("x", [1, 2, 3])})

    class BadRange(ClampProcess):
        ClampProcess.CLAMP("x", field="c", mode="range", value=1.0)

    proc = _bind(_process(BadRange), {"x": _material("x", [1, 2, 3])})
    with pytest.raises(ValueError, match="requires value"):
        proc.advance_materials(0.1)


def test_clamp_multispec_late_resolution_failure_is_transactional():
    class LateFailureClamp(ClampProcess):
        ClampProcess.SET("x", field="c", value=9.0)
        ClampProcess.SET("y", field="c", value="missing_clamp_value")

    materials = {
        "x": _material("x", [1.0, 2.0, 3.0]),
        "y": _material("y", [4.0, 5.0, 6.0]),
    }
    process = _bind(_process(LateFailureClamp), materials)
    material_snapshot = _snapshot_material_state(materials)
    process_snapshot = _snapshot_module_state(process)

    with pytest.raises(AttributeError, match="missing_clamp_value"):
        process.advance_materials(0.1)

    _assert_material_state_unchanged(materials, material_snapshot)
    _assert_module_state_unchanged(process, process_snapshot)


def test_transactional_multispec_success_preserves_declaration_order():
    class SequentialClamp(ClampProcess):
        ClampProcess.SET("x", field="c", value=4.0)
        ClampProcess.MAX("x", field="c", value=2.0)

    material = _material("x", [0.0, 1.0, 3.0])
    _bind(_process(SequentialClamp), {"x": material}).advance_materials(0.1)
    assert torch.equal(material.c, torch.full_like(material.c, 2.0))


@pytest.mark.parametrize(
    "cls, expected",
    [
        (ExactExchange, (1 + math.exp(-0.5), 1 - math.exp(-0.5))),
        (ImplicitExchange, (5 / 3, 1 / 3)),
        (ExplicitExchange, (1.5, 0.5)),
    ],
)
def test_exchange_methods_known_answers_and_conservation(cls, expected):
    a, b = _material("a", [2, 2, 2]), _material("b", [0, 0, 0])
    process = _bind(_process(cls), {"a": a, "b": b})
    process.advance_materials(0.25)
    assert torch.allclose(a.c, torch.full_like(a.c, expected[0]), atol=1e-6)
    assert torch.allclose(b.c, torch.full_like(b.c, expected[1]), atol=1e-6)
    assert torch.allclose(a.c + b.c, torch.full_like(a.c, 2.0))


def test_weighted_exchange_conserves_volume_weighted_mass():
    a, b = _material("a", [2, 2, 2]), _material("b", [0, 0, 0])
    process = _bind(_process(WeightedExchange), {"a": a, "b": b})
    process.advance_materials(1.0)
    assert torch.allclose(a.c + 2 * b.c, torch.full_like(a.c, 2.0))
    assert torch.all(a.c > b.c)


def test_exchange_declaration_and_field_validation():
    with pytest.raises(ValueError, match="either rate or conductance"):
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=1.0, conductance=1.0)

    class SelfExchange(ExchangeProcess):
        ExchangeProcess.EXCHANGE("a.c", "a.c", rate=1.0)

    a = _material("a", [1, 1, 1])
    with pytest.raises(ValueError, match="cannot exchange a field with itself"):
        _bind(_process(SelfExchange), {"a": a})


def test_exchange_multispec_late_resolution_failure_is_transactional():
    class LateFailureExchange(ExchangeProcess):
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=0.5, volume_a=1.0, volume_b=1.0)
        ExchangeProcess.EXCHANGE(
            "c.c",
            "d.c",
            rate="missing_exchange_rate",
            volume_a=1.0,
            volume_b=1.0,
        )

    materials = {
        "a": _material("a", [2.0, 2.0, 2.0]),
        "b": _material("b", [0.0, 0.0, 0.0]),
        "c": _material("c", [4.0, 4.0, 4.0]),
        "d": _material("d", [1.0, 1.0, 1.0]),
    }
    process = _bind(_process(LateFailureExchange), materials)
    material_snapshot = _snapshot_material_state(materials)
    process_snapshot = _snapshot_module_state(process)

    with pytest.raises(AttributeError, match="missing_exchange_rate"):
        process.advance_materials(0.25)

    _assert_material_state_unchanged(materials, material_snapshot)
    _assert_module_state_unchanged(process, process_snapshot)


def test_diffusion_process_chain_known_answers_and_dt_reconfiguration():
    population = SimpleNamespace(dx=torch.ones(SHAPE), graph=None, shape=SHAPE)
    material = _material("x", [1.0, 0.0, 0.0])
    process = _bind(_process(ExplicitDiffusion), {"x": material}, population)
    process.set_dt(0.2)
    process.advance_materials(0.2)
    assert torch.allclose(material.c, torch.tensor([[0.8, 0.2, 0.0]]))

    material = _material("x", [1.0, 0.0, 0.0])
    process = _bind(_process(ImplicitDiffusion), {"x": material}, population)
    process.advance_materials(0.1)
    matrix = torch.tensor([[1.1, -0.1, 0.0], [-0.1, 1.2, -0.1], [0.0, -0.1, 1.1]])
    expected = torch.linalg.solve(matrix, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(material.c[0], expected, atol=1e-6)


def test_diffusion_reconfiguration_preserves_eval_mode_on_staged_operators():
    population = SimpleNamespace(dx=torch.ones(SHAPE), graph=None, shape=SHAPE)
    material = _material("x", [1.0, 0.0, 0.0])
    process = _bind(_process(ExplicitDiffusion), {"x": material}, population)
    process.set_dt(0.1)
    process.eval()

    process.set_dt(0.2)

    assert process.training is False
    assert all(
        not operator.training for operator in process._spatial_operators.values()
    )


def test_diffusion_late_reconfiguration_failure_is_transactional():
    class DynamicMultiDiffusion(DiffusionProcess):
        DiffusionProcess.METHOD("explicit", solver="dense")
        DiffusionProcess.DIFFUSE("x", field="c", D="D_x")
        DiffusionProcess.DIFFUSE("y", field="c", D="D_y")

    population = SimpleNamespace(dx=torch.ones(SHAPE), graph=None, shape=SHAPE)
    materials = {
        "x": _material("x", [1.0, 0.0, 0.0]),
        "y": _material("y", [0.0, 1.0, 0.0]),
    }
    process = _process(DynamicMultiDiffusion)
    process.register_buffer("D_x", torch.tensor(1.0))
    process.register_buffer("D_y", torch.tensor(0.5))
    _bind(process, materials, population)
    process.set_dt(0.1)

    # Simulate a dynamic parameter disappearing before a timestep change. The
    # first operator can be reconfigured successfully; resolution fails only on
    # the second specification.
    del process._buffers["D_y"]
    material_snapshot = _snapshot_material_state(materials)
    process_snapshot = _snapshot_diffusion_process(process)

    with pytest.raises(AttributeError, match="D_y"):
        process.set_dt(0.2)

    _assert_material_state_unchanged(materials, material_snapshot)
    _assert_diffusion_process_unchanged(process, process_snapshot)


def test_diffusion_multispec_late_execution_failure_is_transactional():
    class MultiDiffusion(DiffusionProcess):
        DiffusionProcess.METHOD("explicit", solver="dense")
        DiffusionProcess.DIFFUSE("x", field="c", D=1.0)
        DiffusionProcess.DIFFUSE("y", field="c", D=1.0)

    population = SimpleNamespace(dx=torch.ones(SHAPE), graph=None, shape=SHAPE)
    materials = {
        "x": _material("x", [1.0, 0.0, 0.0]),
        "y": _material("y", [0.0, 1.0, 0.0]),
    }
    process = _bind(_process(MultiDiffusion), materials, population)
    process.set_dt(0.1)

    # Make the second field incompatible only after both operators have been
    # configured. The first solve succeeds, then the second reshape fails.
    materials["y"]._buffers["c"] = torch.tensor([[0.0, 1.0]])
    material_snapshot = _snapshot_material_state(materials)
    process_snapshot = _snapshot_diffusion_process(process)

    with pytest.raises(RuntimeError, match="shape"):
        process.advance_materials(0.1)

    _assert_material_state_unchanged(materials, material_snapshot)
    _assert_diffusion_process_unchanged(process, process_snapshot)


def test_diffusion_process_tree_and_configuration_validation():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, diff_geom_um=1.0)
    population = SimpleNamespace(graph=graph, volume_i=torch.ones(1, 2), shape=(1, 2))
    material = _material("x", [1.0, 0.0])
    process = _bind(
        _process(ImplicitDiffusion, shape=(1, 2)), {"x": material}, population
    )
    process.advance_materials(0.1)
    assert torch.allclose(material.c, torch.tensor([[11 / 12, 1 / 12]]), atol=1e-6)

    with pytest.raises(RuntimeError, match="requires population geometry"):
        _bind(_process(ImplicitDiffusion), {"x": _material("x", [1, 0, 0])})

    bad_population = SimpleNamespace(graph=graph, shape=(1, 2))
    with pytest.raises(RuntimeError, match="volume buffers"):
        _bind(
            _process(ImplicitDiffusion, shape=(1, 2)),
            {"x": _material("x", [1, 0])},
            bad_population,
        )


def test_graph_diffusion_geometry_fallbacks_and_batched_topology_contracts():
    direct = nx.DiGraph()
    direct.add_edge(0, 1, diff_geom_um=2.5, R_ohm=1.0)
    assert spatial._edge_diff_geom_from_graph(direct, 0, 1) == pytest.approx(2.5)

    resistance = nx.DiGraph()
    resistance.add_node(0, Ra=80.0)
    resistance.add_node(1, Ra=120.0)
    resistance.add_edge(0, 1, R_ohm=2.0e6)
    assert spatial._edge_diff_geom_from_graph(resistance, 0, 1) == pytest.approx(0.5)
    del resistance.nodes[0]["Ra"]
    assert spatial._edge_diff_geom_from_graph(resistance, 0, 1) == pytest.approx(0.6)
    resistance.nodes[0]["Ra"] = 80.0
    del resistance.nodes[1]["Ra"]
    assert spatial._edge_diff_geom_from_graph(resistance, 0, 1) == pytest.approx(0.4)
    del resistance.nodes[0]["Ra"]
    with pytest.raises(KeyError, match="neither endpoint has Ra"):
        spatial._edge_diff_geom_from_graph(resistance, 0, 1)

    stylized = nx.DiGraph()
    stylized.add_node(0, diam=2.0)
    stylized.add_node(1, diam=4.0)
    stylized.add_edge(0, 1, L=3.0)
    a0, a1 = torch.pi * 1.0**2, torch.pi * 2.0**2
    expected = 1.0 / (0.5 * 3.0 / a0 + 0.5 * 3.0 / a1)
    assert spatial._edge_diff_geom_from_graph(stylized, 0, 1) == pytest.approx(
        float(expected)
    )
    stylized.edges[0, 1]["L"] = 0.0
    assert spatial._edge_diff_geom_from_graph(stylized, 0, 1) == 0.0
    stylized.edges[0, 1]["L"] = 3.0
    stylized.nodes[1]["diam"] = 0.0
    assert spatial._edge_diff_geom_from_graph(stylized, 0, 1) == 0.0

    graph_a = nx.DiGraph()
    graph_a.add_edge(0, 1, diff_geom_um=1.0)
    graph_a.add_edge(0, 2, diff_geom_um=2.0)
    graph_b = nx.DiGraph()
    graph_b.add_edge(0, 1, diff_geom_um=3.0)
    graph_b.add_edge(0, 2, diff_geom_um=4.0)
    parent, geometry, nodes = spatial.graph_to_parent_and_diffusion(
        [graph_a, graph_b], dtype=torch.float64
    )
    assert parent.tolist() == [-1, 0, 0]
    assert nodes == [0, 1, 2]
    torch.testing.assert_close(
        geometry,
        torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=torch.float64),
    )

    different_order = nx.DiGraph()
    different_order.add_edge(0, 2, diff_geom_um=1.0)
    different_order.add_edge(2, 1, diff_geom_um=1.0)
    with pytest.raises(AssertionError, match="topological order"):
        spatial.graph_to_parent_and_diffusion([graph_a, different_order])

    different_parent = nx.DiGraph()
    different_parent.add_edge(0, 1, diff_geom_um=1.0)
    different_parent.add_edge(1, 2, diff_geom_um=1.0)
    with pytest.raises(AssertionError, match="parent/child topology"):
        spatial.graph_to_parent_and_diffusion([graph_a, different_parent])


def test_tree_diffusion_batched_geometry_and_tensor_dt_match_dense_oracle():
    graph_a = nx.DiGraph()
    graph_a.add_edge(2, 0, diff_geom_um=0.5)
    graph_a.add_edge(2, 1, diff_geom_um=1.5)
    graph_b = nx.DiGraph()
    graph_b.add_edge(2, 0, diff_geom_um=2.0)
    graph_b.add_edge(2, 1, diff_geom_um=0.75)
    volume = torch.tensor([[1.0, 1.5, 0.75], [2.0, 0.5, 1.25]], dtype=torch.float64)
    concentration = torch.tensor(
        [[0.25, 1.5, 0.75], [1.25, 0.1, 2.0]], dtype=torch.float64
    )
    diffusivity = torch.tensor(
        [[0.5, 1.0, 1.5], [1.25, 0.75, 0.25]], dtype=torch.float64
    )
    dt = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=torch.float64)
    model = SimpleNamespace(graph=[graph_a, graph_b], volume_i=volume)
    operator = spatial.SpatialOperatorTree(solver="dense", threads=4)
    operator.configure_diffusion(concentration, dt, diffusivity, model, solver="dense")
    actual = operator.diffuse_implicit_configured(concentration)

    expected_rows = []
    for row, graph in enumerate((graph_a, graph_b)):
        dmem = volume[row] / dt[row]
        matrix = torch.diag(dmem).clone()
        for parent, child, attrs in graph.edges(data=True):
            coupling = (
                0.5
                * (diffusivity[row, parent] + diffusivity[row, child])
                * attrs["diff_geom_um"]
            )
            matrix[parent, parent] += coupling
            matrix[child, child] += coupling
            matrix[parent, child] -= coupling
            matrix[child, parent] -= coupling
        expected_rows.append(torch.linalg.solve(matrix, dmem * concentration[row]))
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected, atol=2.0e-15, rtol=2.0e-15)
    torch.testing.assert_close(
        (volume * actual).sum(-1),
        (volume * concentration).sum(-1),
        atol=2.0e-15,
        rtol=2.0e-15,
    )

    too_many_graphs = SimpleNamespace(
        graph=[graph_a, graph_b, graph_a], volume_i=volume
    )
    with pytest.raises(ValueError, match="graph batch has 3 rows"):
        spatial.SpatialOperatorTree(solver="dense").configure_diffusion(
            concentration, 0.1, diffusivity, too_many_graphs, solver="dense"
        )


def test_clamp_region_masks_cover_graph_alias_tensor_boolean_and_indices():
    process = _process(ClampProcess, shape=(2, 4))
    like = torch.zeros((2, 4))
    graph = nx.DiGraph([(0, 1), (0, 2), (2, 3)])
    population = SimpleNamespace(graph=[graph])

    for alias in ("", "all", "everywhere", "global", "none"):
        assert process._build_where_mask(alias, like, population=population) is None

    terminal = process._build_where_mask("leaves", like, population=population)
    root = process._build_where_mask("soma", like, population=population)
    assert torch.equal(
        terminal,
        torch.tensor([[False, True, False, True], [False, True, False, True]]),
    )
    assert torch.equal(
        root,
        torch.tensor([[True, False, False, False], [True, False, False, False]]),
    )

    process.register_buffer("named_mask", torch.tensor([False, True, False, True]))
    assert torch.equal(
        process._build_where_mask("named_mask", like), process.named_mask
    )
    assert process._build_where_mask(True, like) is None
    assert not process._build_where_mask(False, like).any()
    assert torch.equal(
        process._build_where_mask(torch.tensor([1, 0, 1, 0]), like),
        torch.tensor([True, False, True, False]),
    )
    assert process._build_where_mask(slice(1, 3), like).tolist() == [
        [False, True, True, False],
        [False, True, True, False],
    ]
    assert process._build_where_mask((slice(None), 2), like).tolist() == [
        [False, False, True, False],
        [False, False, True, False],
    ]
    assert process._build_where_mask([0, 3], like).tolist() == [
        [True, False, False, True],
        [True, False, False, True],
    ]
    with pytest.raises(ValueError, match="Could not interpret"):
        process._build_where_mask(object(), like)
    with pytest.raises(ValueError, match="Unsupported clamp region"):
        process._build_where_mask("missing_mask", like)

    no_graph = SimpleNamespace(graph=None)
    empty = torch.zeros((1, 0))
    assert not process._terminal_mask(empty, no_graph).any()
    assert not process._root_mask(empty, no_graph).any()
    assert process._terminal_mask(like, no_graph)[0].tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert process._root_mask(like, no_graph)[0].tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_exchange_geometry_domains_aliases_and_negative_rate_policy():
    class GeometryExchange(ExchangeProcess):
        ExchangeProcess.METHOD("exact", require_volumes=True)
        ExchangeProcess.EXCHANGE(
            "a.c",
            "b.c",
            rate=0.5,
            volume_a="intracellular",
            volume_b="extracellular",
        )

    volume_i = torch.tensor([[1.0, 2.0, 3.0]])
    volume_o = torch.tensor([[4.0, 5.0, 6.0]])
    area_cm2 = torch.tensor([[2.0e-8, 3.0e-8, 4.0e-8]])
    population = SimpleNamespace(
        volume_i=volume_i,
        volume_o=volume_o,
        dx=torch.ones_like(volume_i),
        area=area_cm2,
        shape=SHAPE,
    )
    materials = {
        "a": _material("a", [2.0, 3.0, 4.0], domain="intracellular"),
        "b": _material("b", [0.5, 0.25, 1.0], domain="extracellular"),
    }
    process = _bind(_process(GeometryExchange), materials, population)
    before_mass = volume_i * materials["a"].c + volume_o * materials["b"].c
    process.advance_materials(0.2)
    after_mass = volume_i * materials["a"].c + volume_o * materials["b"].c
    torch.testing.assert_close(after_mass, before_mass, atol=2.0e-6, rtol=2.0e-6)

    like = materials["a"].c
    for alias in ("auto", "field", "domain", "i", "volume_i"):
        torch.testing.assert_close(
            process._resolve_exchange_volume(
                alias, like, domain="intracellular", what="test"
            ),
            volume_i,
        )
    for alias in ("o", "volume_o", "extracellular_volume"):
        torch.testing.assert_close(
            process._resolve_exchange_volume(
                alias, like, domain="extracellular", what="test"
            ),
            volume_o,
        )
    for alias in ("area", "surface_area", "membrane_area"):
        torch.testing.assert_close(
            process._resolve_exchange_volume(alias, like, domain=None, what="test"),
            area_cm2 * 1.0e8,
        )
    torch.testing.assert_close(
        process._resolve_exchange_volume(
            torch.tensor(7.0), like, domain=None, what="test"
        ),
        torch.tensor(7.0),
    )
    assert process._resolve_exchange_volume(3.0, like, domain=None, what="test") == 3

    class NegativeExchange(ExchangeProcess):
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=-1.0, volume_a=1.0, volume_b=1.0)

    a = _material("a", [2.0, 3.0, 4.0])
    b = _material("b", [0.0, 1.0, 2.0])
    _bind(_process(NegativeExchange), {"a": a, "b": b}).advance_materials(0.5)
    torch.testing.assert_close(a.c, torch.tensor([[2.0, 3.0, 4.0]]))
    torch.testing.assert_close(b.c, torch.tensor([[0.0, 1.0, 2.0]]))

    class MissingRequiredVolume(ExchangeProcess):
        ExchangeProcess.METHOD("exact", require_volumes=True)
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=1.0)

    missing = {
        "a": _material("a", [1.0, 2.0, 3.0], domain="extracellular"),
        "b": _material("b", [3.0, 2.0, 1.0], domain="extracellular"),
    }
    missing_process = _bind(_process(MissingRequiredVolume), missing)
    missing_snapshot = _snapshot_material_state(missing)
    with pytest.raises(NotImplementedError, match="Could not infer"):
        missing_process.advance_materials(0.1)
    _assert_material_state_unchanged(missing, missing_snapshot)


def test_diffusion_geometry_kind_and_material_volume_failure_contracts():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, diff_geom_um=1.0)
    population = SimpleNamespace(
        graph=graph,
        volume_i=torch.tensor([[1.0, 2.0]]),
        volume_o=torch.tensor([[3.0, 4.0]]),
        shape=(1, 2),
    )
    process = _bind(
        _process(ImplicitDiffusion, shape=(1, 2)),
        {"x": _material("x", [1.0, 0.0])},
        population,
    )
    torch.testing.assert_close(process.material_volume("cytosol"), population.volume_i)
    torch.testing.assert_close(process.material_volume("outside"), population.volume_o)
    with pytest.raises(NotImplementedError, match="Membrane"):
        process.material_volume("surface")
    with pytest.raises(NotImplementedError, match="Unsupported"):
        process.material_volume("nucleus")

    missing_edges = nx.DiGraph()
    missing_edges.add_edge(0, 1)
    with pytest.raises(RuntimeError, match="diff_geom_um or R_ohm"):
        _bind(
            _process(ImplicitDiffusion, shape=(1, 2)),
            {"x": _material("x", [1.0, 0.0])},
            SimpleNamespace(
                graph=missing_edges,
                volume_i=torch.ones((1, 2)),
                shape=(1, 2),
            ),
        )

    class ExtracellularDiffusion(DiffusionProcess):
        DiffusionProcess.DIFFUSE("x", field="c", D=1.0, domain="extracellular")

    material = _material("x", [1.0, 0.0], domain="extracellular")
    snapshot = _snapshot_material_state({"x": material})
    with pytest.raises(NotImplementedError, match="only supports intracellular"):
        _bind(
            _process(ExtracellularDiffusion, shape=(1, 2)),
            {"x": material},
            population,
        )
    _assert_material_state_unchanged({"x": material}, snapshot)
