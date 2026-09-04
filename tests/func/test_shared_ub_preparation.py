"""Parity contracts for the shared unbranched backward-Euler workspace."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.integrators.cable import unbranched_edge_conductance
from dendra.models.integrators.core import _as_solve_matrix
from dendra.models.integrators.implicit import _bwd_euler_ub
from dendra.models.mod import pas

DT = 0.01
WORKSPACE_NAMES = (
    "diag_base",
    "lower",
    "upper",
    "g_edge_Cinv",
    "g_edge_Cinv_right",
    "cm_inv",
    "scale",
)
WORKSPACE_SCHEMA = (
    ("diag_base", "node"),
    ("lower", "edge"),
    ("upper", "edge"),
    ("g_edge_Cinv", "edge"),
    ("g_edge_Cinv_right", "edge"),
    ("cm_inv", "node"),
    ("scale", "node"),
)

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _LifecycleCounter(dn.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.pre_loop = 0
        self.pre_step = 0
        self.post_step = 0
        self.post_loop = 0

    def pre_loop_hook(self, model):
        self.pre_loop += 1

    def pre_step_hook(self, model):
        self.pre_step += 1

    def post_step_hook(self, model):
        self.post_step += 1

    def post_loop_hook(self, model):
        self.post_loop += 1


def _population(*, dtype, batch_calls=(), require_grad=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=int(require_grad)):
        population = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            dtype=dtype,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-70.0)
        for size in batch_calls:
            population.batch(size)
        population.initialize()
    return population


def _workspace_snapshot(integrator):
    return {
        name: (
            id(getattr(integrator, name)),
            getattr(integrator, name).untyped_storage().data_ptr(),
            getattr(integrator, name)._version,
            getattr(integrator, name).detach().clone(),
        )
        for name in WORKSPACE_NAMES
    }


def _assert_workspace_snapshot(integrator, snapshot):
    for name, (identity, storage, version, expected) in snapshot.items():
        actual = getattr(integrator, name)
        assert id(actual) == identity
        assert actual.untyped_storage().data_ptr() == storage
        assert actual._version == version
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _workspace_inputs(population):
    cm = _as_solve_matrix(population.cm, population) * _as_solve_matrix(
        population.cm_scale,
        population,
    )
    area = _as_solve_matrix(population.area, population) * _as_solve_matrix(
        population.area_scale,
        population,
    )
    edge_conductance = unbranched_edge_conductance(population)
    return cm, area, edge_conductance


def _flat_workspace(dt, cm, area, edge_conductance):
    workspace = _bwd_euler_ub._prepare_workspace(
        dt,
        cm=cm,
        area=area,
        edge_conductance=edge_conductance,
    )
    return torch.cat(tuple(workspace[name].reshape(-1) for name in WORKSPACE_NAMES))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_calls", [(), (3, 2)])
def test_functional_preparation_exactly_matches_initialized_imperative_workspace(
    dtype,
    batch_calls,
):
    population = _population(dtype=dtype, batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(population, dt=DT)

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    dt = torch.as_tensor(DT, device=population.device(), dtype=dtype)
    population.integrator._initialize(
        population,
        dt,
        force=True,
        compile_scope="population",
    )
    cm, area, edge_conductance = _workspace_inputs(population)
    pure_workspace = _bwd_euler_ub._prepare_workspace(
        dt,
        cm=cm,
        area=area,
        edge_conductance=edge_conductance,
    )

    prepared_workspace = prepared.values["integrator"]
    assert _bwd_euler_ub._PREPARED_WORKSPACE_SCHEMA == WORKSPACE_SCHEMA
    assert tuple(pure_workspace) == WORKSPACE_NAMES
    for name in WORKSPACE_NAMES:
        actual = prepared_workspace[name]
        expected = getattr(population.integrator, name)
        assert actual.dtype == expected.dtype == dtype
        assert actual.device == expected.device == population.device()
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            pure_workspace[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("node_shape", [(2, 1), (3, 5)])
def test_pure_workspace_schema_shapes_dtype_and_input_immutability(
    dtype,
    node_shape,
):
    batch, nodes = node_shape
    cm = torch.linspace(1.1, 1.9, batch * nodes, dtype=dtype).reshape(node_shape)
    area = torch.linspace(0.7, 1.3, batch * nodes, dtype=dtype).reshape(node_shape)
    edge_conductance = torch.linspace(
        0.2,
        0.8,
        batch * (nodes - 1),
        dtype=dtype,
    ).reshape(batch, nodes - 1)
    dt = torch.tensor(0.025, dtype=dtype)
    inputs = (dt, cm, area, edge_conductance)
    snapshots = tuple(
        (id(value), value._version, value.detach().clone()) for value in inputs
    )

    workspace = _bwd_euler_ub._prepare_workspace(
        dt,
        cm=cm,
        area=area,
        edge_conductance=edge_conductance,
    )

    assert _bwd_euler_ub._PREPARED_WORKSPACE_SCHEMA == WORKSPACE_SCHEMA
    assert tuple(workspace) == tuple(name for name, _role in WORKSPACE_SCHEMA)
    input_storage = {
        value.untyped_storage().data_ptr() for value in inputs if value.numel()
    }
    for name, role in WORKSPACE_SCHEMA:
        value = workspace[name]
        expected_shape = node_shape if role == "node" else (batch, nodes - 1)
        assert value.shape == expected_shape
        assert value.dtype == dtype
        assert value.device.type == "cpu"
        assert torch.isfinite(value).all()
        if value.numel():
            assert value.untyped_storage().data_ptr() not in input_storage

    for value, (identity, version, expected) in zip(inputs, snapshots, strict=True):
        assert id(value) == identity
        assert value._version == version
        torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def test_pure_workspace_passes_float64_autograd_gradcheck():
    dt = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    cm = torch.tensor(
        [[1.1e6, 1.5e6, 1.9e6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    area = torch.tensor(
        [[0.7, 1.0, 1.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_conductance = torch.tensor(
        [[0.2, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        _flat_workspace,
        (dt, cm, area, edge_conductance),
        atol=1.0e-6,
        rtol=1.0e-4,
    )


def test_pure_workspace_composes_vmap_with_reverse_mode_gradients():
    lanes, batch, nodes = 3, 2, 4
    dt = torch.linspace(0.01, 0.03, lanes, dtype=torch.float64)
    cm = torch.linspace(
        1.1e6,
        2.3e6,
        lanes * batch * nodes,
        dtype=torch.float64,
    ).reshape(lanes, batch, nodes)
    area = torch.linspace(
        0.6,
        1.4,
        lanes * batch * nodes,
        dtype=torch.float64,
    ).reshape(lanes, batch, nodes)
    edge_conductance = torch.linspace(
        0.1,
        0.7,
        lanes * batch * (nodes - 1),
        dtype=torch.float64,
    ).reshape(lanes, batch, nodes - 1)

    def objective(local_dt, local_cm, local_area, local_edge):
        values = _flat_workspace(local_dt, local_cm, local_area, local_edge)
        weights = torch.linspace(
            0.5,
            1.5,
            values.numel(),
            device=values.device,
            dtype=values.dtype,
        )
        return (values.square() * weights).mean()

    gradient = torch.func.grad(objective, argnums=(0, 1, 2, 3))
    actual = torch.vmap(gradient)(dt, cm, area, edge_conductance)
    expected = tuple(
        torch.stack(
            tuple(
                gradient(
                    dt[index],
                    cm[index],
                    area[index],
                    edge_conductance[index],
                )[argument]
                for index in range(lanes)
            )
        )
        for argument in range(4)
    )

    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient) == actual_gradient.numel()


def test_pure_workspace_vmap_accepts_an_empty_outer_batch():
    lanes, batch, nodes = 0, 2, 4
    dt = torch.empty(lanes, dtype=torch.float64)
    cm = torch.empty(lanes, batch, nodes, dtype=torch.float64)
    area = torch.empty_like(cm)
    edge_conductance = torch.empty(
        lanes,
        batch,
        nodes - 1,
        dtype=torch.float64,
    )

    workspace = torch.vmap(
        lambda local_dt, local_cm, local_area, local_edge: (
            _bwd_euler_ub._prepare_workspace(
                local_dt,
                cm=local_cm,
                area=local_area,
                edge_conductance=local_edge,
            )
        )
    )(dt, cm, area, edge_conductance)

    for name, role in WORKSPACE_SCHEMA:
        trailing_shape = (batch, nodes) if role == "node" else (batch, nodes - 1)
        assert workspace[name].shape == (lanes, *trailing_shape)


@pytest.mark.parametrize("owner", ["population", "network"])
def test_imperative_runners_reuse_workspace_while_preserving_callbacks(owner):
    population = _population(dtype=torch.float64, require_grad=False)
    callback = _LifecycleCounter()

    if owner == "population":
        # A zero-step call materializes the dt-dependent workspace without
        # advancing state, so the following run can exercise same-dt reuse.
        population.run(tstop=0.0, dt=DT)
        model = population
        integrator = population.integrator
    else:
        model = dn.Network({"population": population})
        model.initialize(DT)
        integrator = model.population.integrator

    snapshot = _workspace_snapshot(integrator)
    if owner == "population":
        model.run(tstop=3 * DT, dt=DT, callbacks=[callback])
    else:
        model.run(3 * DT, callbacks=[callback])

    assert callback.pre_loop == 1
    assert callback.pre_step == 3
    assert callback.post_step == 3
    assert callback.post_loop == 1
    assert callback.dt == DT
    _assert_workspace_snapshot(integrator, snapshot)
