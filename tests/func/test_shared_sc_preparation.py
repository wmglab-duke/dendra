"""Parity contracts for shared single-compartment backward-Euler preparation."""

from __future__ import annotations

import copy
import math

import pytest
import torch

import dendra as dn
from dendra.models.integrators.implicit import _bwd_euler_sc
from dendra.models.mod import pas

DT = 0.01
ALT_DT = 0.025
WORKSPACE_NAMES = ("cmdt", "area")
WORKSPACE_SCHEMA = (
    ("cmdt", "parameter"),
    ("area", "geometry"),
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
        population = dn.SingleCompartment(
            N=2,
            C=3,
            dtype=dtype,
            integrator=dn.bwd_euler_sc(imem=False),
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
    cm = population.cm * population.cm_scale
    area = population.area * population.area_scale
    return cm, area


def _flat_workspace(dt, cm, area):
    workspace = _bwd_euler_sc._prepare_workspace(dt, cm=cm, area=area)
    return torch.cat(tuple(workspace[name].reshape(-1) for name in WORKSPACE_NAMES))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_calls", [(), (3, 2)])
def test_pure_workspace_exactly_matches_initialized_imperative_workspace(
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
    cm, area = _workspace_inputs(population)
    pure_workspace = _bwd_euler_sc._prepare_workspace(dt, cm=cm, area=area)

    assert _bwd_euler_sc._PREPARED_WORKSPACE_SCHEMA == WORKSPACE_SCHEMA
    assert tuple(pure_workspace) == WORKSPACE_NAMES
    for name in WORKSPACE_NAMES:
        actual = pure_workspace[name]
        expected = getattr(population.integrator, name)
        functional_value = prepared.values["integrator"][name]
        assert actual.dtype == expected.dtype == dtype
        assert actual.device == expected.device == population.device()
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            functional_value,
            expected,
            rtol=0.0,
            atol=0.0,
        )

    expected_parameter_shape = population._calc_shape_p()
    expected_geometry_shape = population.core_shape()
    assert population.integrator.cmdt.shape == expected_parameter_shape
    assert population.integrator.area.shape == expected_geometry_shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("parameter_shape", "geometry_shape"),
    [
        ((2, 3), (2, 3)),
        ((1, 1, 2, 3), (2, 3)),
    ],
)
def test_pure_workspace_schema_shapes_dtype_non_aliasing_and_input_immutability(
    dtype,
    parameter_shape,
    geometry_shape,
):
    cm = torch.linspace(
        1.1,
        1.9,
        math.prod(parameter_shape),
        dtype=dtype,
    ).reshape(parameter_shape)
    area = torch.linspace(
        0.7,
        1.3,
        math.prod(geometry_shape),
        dtype=dtype,
    ).reshape(geometry_shape)
    dt = torch.tensor(0.025, dtype=dtype)
    inputs = (dt, cm, area)
    snapshots = tuple(
        (id(value), value._version, value.detach().clone()) for value in inputs
    )

    workspace = _bwd_euler_sc._prepare_workspace(dt, cm=cm, area=area)

    assert _bwd_euler_sc._PREPARED_WORKSPACE_SCHEMA == WORKSPACE_SCHEMA
    assert tuple(workspace) == WORKSPACE_NAMES
    assert workspace["cmdt"].shape == parameter_shape
    assert workspace["area"].shape == geometry_shape
    input_storage = {
        value.untyped_storage().data_ptr() for value in inputs if value.numel()
    }
    for value in workspace.values():
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
        [[[[1.1, 1.5, 1.9], [1.3, 1.7, 2.1]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    area = torch.tensor(
        [[0.7, 1.0, 1.3], [0.8, 1.1, 1.4]],
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        _flat_workspace,
        (dt, cm, area),
        atol=1.0e-6,
        rtol=1.0e-4,
    )


def test_pure_workspace_composes_vmap_with_reverse_mode_gradients():
    lanes, rows, compartments = 3, 2, 3
    dt = torch.linspace(0.01, 0.03, lanes, dtype=torch.float64)
    cm = torch.linspace(
        1.1,
        2.3,
        lanes * rows * compartments,
        dtype=torch.float64,
    ).reshape(lanes, 1, rows, compartments)
    area = torch.linspace(
        0.6,
        1.4,
        lanes * rows * compartments,
        dtype=torch.float64,
    ).reshape(lanes, rows, compartments)

    def objective(local_dt, local_cm, local_area):
        values = _flat_workspace(local_dt, local_cm, local_area)
        weights = torch.linspace(
            0.5,
            1.5,
            values.numel(),
            device=values.device,
            dtype=values.dtype,
        )
        return (values.square() * weights).mean()

    gradient = torch.func.grad(objective, argnums=(0, 1, 2))
    actual = torch.vmap(gradient)(dt, cm, area)
    expected = tuple(
        torch.stack(
            tuple(
                gradient(dt[index], cm[index], area[index])[argument]
                for index in range(lanes)
            )
        )
        for argument in range(3)
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
    lanes, rows, compartments = 0, 2, 3
    dt = torch.empty(lanes, dtype=torch.float64)
    cm = torch.empty(lanes, 1, rows, compartments, dtype=torch.float64)
    area = torch.empty(lanes, rows, compartments, dtype=torch.float64)

    workspace = torch.vmap(
        lambda local_dt, local_cm, local_area: _bwd_euler_sc._prepare_workspace(
            local_dt,
            cm=local_cm,
            area=local_area,
        )
    )(dt, cm, area)

    assert workspace["cmdt"].shape == (lanes, 1, rows, compartments)
    assert workspace["area"].shape == (lanes, rows, compartments)


def test_initialize_stages_the_complete_workspace_before_installation(monkeypatch):
    population = _population(dtype=torch.float64)
    integrator = population.integrator
    integrator._initialize(population, DT, force=True)
    snapshot = _workspace_snapshot(integrator)

    def incomplete_workspace(dt, *, cm, area):
        return {"cmdt": torch.full_like(cm, 123.0)}

    monkeypatch.setattr(
        _bwd_euler_sc,
        "_prepare_workspace",
        staticmethod(incomplete_workspace),
    )
    with pytest.raises(KeyError, match="area"):
        integrator.initialize(population, ALT_DT)

    _assert_workspace_snapshot(integrator, snapshot)


def test_functional_step_contract_matches_hot_path_and_ignores_uniform_ve():
    population = _population(dtype=torch.float64, require_grad=False)
    population.integrator._initialize(population, DT, force=True)
    imperative = copy.deepcopy(population.integrator)
    functional = copy.deepcopy(population.integrator)
    dt = torch.as_tensor(DT, dtype=population.dtype(), device=population.device())
    intra = torch.linspace(
        -1.0e-8,
        1.0e-8,
        population.v.numel(),
        dtype=population.dtype(),
        device=population.device(),
    ).reshape(population.shape)
    ve = torch.full_like(population.v, 42.0)

    expected = imperative._step(
        population.v.clone(),
        dt,
        population.celsius,
        intra=intra,
    )
    actual = functional._step(
        population.v.clone(),
        dt,
        population.celsius,
        ve,
        intra,
        solver=object(),
        call_local_currents=True,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    assert actual[1] is expected[1] is None


@pytest.mark.parametrize("owner", ["population", "network"])
def test_imperative_runners_reuse_workspace_while_preserving_callbacks(owner):
    population = _population(dtype=torch.float64, require_grad=False)
    callback = _LifecycleCounter()

    if owner == "population":
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
