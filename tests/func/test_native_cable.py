"""Functional contracts for exact native ``Cable`` geometry and UB preparation."""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.cable import (
    _cylindrical_edge_conductance,
    _cylindrical_membrane_area,
    unbranched_edge_conductance,
)
from dendra.models.integrators.core import _as_solve_matrix
from dendra.models.integrators.implicit import _bwd_euler_ub
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh, pas

DT = 0.01
STEPS = 3
WORKSPACE_NAMES = (
    "diag_base",
    "lower",
    "upper",
    "g_edge_Cinv",
    "g_edge_Cinv_right",
    "cm_inv",
    "scale",
)
AREA = "canonical_area_cm2"
RESISTANCE = "canonical_edge_resistance_ohm"
RHOA_RAW = "rhoa_param.rho"
CM_RAW = "cm_param.rho"
RHOA_SCALE_RAW = "rhoa_scale_param.rho"
AREA_SCALE_RAW = "area_scale_param.rho"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _morphology(*, one_compartment=False, variant=0):
    morphology = dn.Morphology(rhoa=91.0, cm=1.0)
    if one_compartment:
        morphology.section(
            "single",
            points=[(0.0, 0.0, 0.0, 5.0), (17.0, 3.0, 1.0, 1.5)],
            nseg=1,
        )
        return morphology

    root = morphology.section(
        "root",
        points=[
            (0.0, 0.0, 0.0, 7.0),
            (14.0 + variant, 9.0, 2.0, 3.2),
        ],
        nseg=2,
        rhoa=83.0,
        cm=0.8,
    )
    tail = morphology.section(
        "tail",
        points=[
            (14.0 + variant, 9.0, 2.0, 3.2),
            (43.0, 31.0, -4.0, 0.8 + 0.1 * variant),
        ],
        nseg=2,
        rhoa=139.0,
        cm=1.3,
    )
    tail.connect(root.at(1.0))
    return morphology


def _model(
    *,
    dtype=torch.float64,
    batch_calls=(),
    one_compartment=False,
    variant=0,
    require_grad=True,
    mechanism="pas",
    method="pcr",
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=int(require_grad)):
        model = dn.Cable.from_morphology(
            _morphology(one_compartment=one_compartment, variant=variant),
            N=2,
            dtype=dtype,
            v_init=-64.0,
            integrator=dn.bwd_euler_ub(method=method, imem=False),
        )
        if mechanism == "pas":
            model.insert(pas, g=4.0e-4, e=-71.0)
        elif mechanism == "hh":
            model.insert(hh)
        else:  # pragma: no cover - private test helper guard
            raise ValueError(f"unsupported test mechanism {mechanism!r}")
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _drives(model, steps=STEPS):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -2.0,
        3.0,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -2.0e-9,
        3.0e-9,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _imperative_step(model, ve=None, intra=None):
    dt = torch.as_tensor(DT, dtype=model.dtype(), device=model.device())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _clone_tree(value):
    return torch.utils._pytree.tree_map(lambda tensor: tensor.detach().clone(), value)


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_calls", [(), (3, 2)])
def test_native_cable_prepared_workspace_exactly_matches_imperative_exact_geometry(
    dtype,
    batch_calls,
):
    model = _model(dtype=dtype, batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    dt = torch.as_tensor(DT, dtype=dtype, device=model.device())
    model.integrator._initialize(model, dt, force=True)
    expected_node_shape = tuple(int(size) for size in model.batched_shape())
    expected_edge_shape = (*expected_node_shape[:-1], expected_node_shape[-1] - 1)

    assert set(tensors.constants) == {"diam", "dx", AREA, RESISTANCE, "dt"}
    assert tensors.constants[AREA].shape == model.core_shape()
    assert tensors.constants[RESISTANCE].shape == model.core_shape()
    assert torch.equal(tensors.constants[AREA], model._canonical_area_cm2)
    assert torch.equal(
        tensors.constants[RESISTANCE],
        model._canonical_edge_resistance_ohm,
    )
    assert (
        tensors.constants[AREA].untyped_storage().data_ptr()
        != model._canonical_area_cm2.untyped_storage().data_ptr()
    )
    assert (
        tensors.constants[RESISTANCE].untyped_storage().data_ptr()
        != model._canonical_edge_resistance_ohm.untyped_storage().data_ptr()
    )

    for name in WORKSPACE_NAMES:
        actual = prepared.values["integrator"][name]
        expected = getattr(model.integrator, name)
        expected_shape = (
            expected_edge_shape
            if name in {"lower", "upper", "g_edge_Cinv", "g_edge_Cinv_right"}
            else expected_node_shape
        )
        assert actual.shape == expected.shape == expected_shape
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    cylindrical_area = _cylindrical_membrane_area(model.diam, model.dx)
    assert not torch.allclose(
        tensors.constants[AREA],
        cylindrical_area,
        rtol=1.0e-5,
        atol=0.0,
    )
    cylindrical_edges = _cylindrical_edge_conductance(
        _as_solve_matrix(model.diam, model),
        _as_solve_matrix(model.dx, model),
        _as_solve_matrix(model.rhoa * model.rhoa_scale, model),
    )
    assert not torch.allclose(
        unbranched_edge_conductance(model),
        cylindrical_edges,
        rtol=1.0e-5,
        atol=0.0,
    )


def test_native_cable_k1_and_explicit_batch_preserve_empty_edge_contract():
    source = _model(one_compartment=True, batch_calls=(3,))
    imperative = _model(one_compartment=True, batch_calls=(3,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    batch = math.prod(source.shape[:-1])

    assert source.nc == 1
    for name in ("lower", "upper", "g_edge_Cinv", "g_edge_Cinv_right"):
        assert prepared.values["integrator"][name].shape == (batch, 0)
    for name in ("diag_base", "cm_inv", "scale"):
        assert prepared.values["integrator"][name].shape == (batch, 1)

    _ve, intra = _drives(source, steps=1)
    actual, _ = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(intra=intra[0]),
    )
    _imperative_step(imperative, intra=intra[0])
    _assert_tree_close(
        actual,
        functional.extract(imperative).state,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("drive_kind", ["none", "ve", "intra", "both"])
def test_native_cable_chained_and_fused_transition_match_imperative_every_leaf(
    drive_kind,
):
    source = _model()
    imperative = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)
    if drive_kind not in {"ve", "both"}:
        ve = None
    if drive_kind not in {"intra", "both"}:
        intra = None
    state = tensors.state

    for index in range(STEPS):
        ve_step = None if ve is None else ve[index]
        intra_step = None if intra is None else intra[index]
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve_step, intra=intra_step),
        )
        _imperative_step(imperative, ve_step, intra_step)
        _assert_tree_close(
            state,
            functional.extract(imperative).state,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        torch.testing.assert_close(auxiliary["v"], state["integrator"]["v"])

    rolled, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
        steps=STEPS,
    )
    _assert_tree_close(rolled, state, rtol=2.0e-12, atol=2.0e-12)


def test_native_cable_stateful_hh_matches_every_imperative_leaf():
    source = _model(mechanism="hh")
    imperative = _model(mechanism="hh")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=2)
    state = tensors.state

    for index in range(2):
        state, _ = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(imperative, ve[index], intra[index])
        _assert_tree_close(
            state,
            functional.extract(imperative).state,
            rtol=2.0e-12,
            atol=2.0e-12,
        )


def test_native_cable_canonical_area_and_resistance_jacrev_jacfwd_semantics():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    _ve, intra = _drives(model, steps=1)

    def voltage(canonical_area, canonical_resistance):
        constants = dict(tensors.constants)
        constants[AREA] = canonical_area
        constants[RESISTANCE] = canonical_resistance
        return functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
            dn.func.StepInput(intra=intra[0]),
        )[0]["integrator"]["v"]

    arguments = (tensors.constants[AREA], tensors.constants[RESISTANCE])
    reverse = torch.func.jacrev(voltage, argnums=(0, 1))(*arguments)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(voltage, argnums=(0, 1))(*arguments)

    for actual, expected in zip(reverse, forward, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)
    assert torch.count_nonzero(reverse[0]) > 0
    assert torch.count_nonzero(reverse[1][..., 0]) == 0
    assert torch.count_nonzero(reverse[1][..., 1:]) > 0

    def voltage_from_raw_rhoa(raw_rhoa):
        parameters = dict(tensors.parameters)
        parameters[RHOA_RAW] = raw_rhoa
        return functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(intra=intra[0]),
        )[0]["integrator"]["v"]

    rhoa_jacobian = torch.func.jacrev(voltage_from_raw_rhoa)(
        tensors.parameters[RHOA_RAW]
    )
    assert torch.count_nonzero(rhoa_jacobian) == 0

    def voltage_from_other_dynamics(
        cm_raw,
        rhoa_scale_raw,
        area_scale_raw,
        dt,
        diam,
        dx,
    ):
        parameters = dict(tensors.parameters)
        parameters[CM_RAW] = cm_raw
        parameters[RHOA_SCALE_RAW] = rhoa_scale_raw
        parameters[AREA_SCALE_RAW] = area_scale_raw
        constants = dict(tensors.constants)
        constants["dt"] = dt
        constants["diam"] = diam
        constants["dx"] = dx
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
            dn.func.StepInput(intra=intra[0]),
        )[0]["integrator"]["v"]

    dynamics_jacobians = torch.func.jacrev(
        voltage_from_other_dynamics,
        argnums=(0, 1, 2, 3, 4, 5),
    )(
        tensors.parameters[CM_RAW],
        tensors.parameters[RHOA_SCALE_RAW],
        tensors.parameters[AREA_SCALE_RAW],
        tensors.constants["dt"],
        tensors.constants["diam"],
        tensors.constants["dx"],
    )
    for jacobian in dynamics_jacobians:
        assert torch.isfinite(jacobian).all()
    for jacobian in dynamics_jacobians[:4]:
        assert torch.count_nonzero(jacobian) > 0
    # Canonical area and resistance, not a cylinder reconstructed from these
    # representative geometry views, own the passive electrical sensitivity.
    for jacobian in dynamics_jacobians[4:]:
        assert torch.count_nonzero(jacobian) == 0


def test_native_cable_vmap_zero_lanes_and_compile_jacrev():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    area = tensors.constants[AREA]
    resistance = tensors.constants[RESISTANCE]

    def voltage(local_area, local_resistance):
        constants = dict(tensors.constants)
        constants[AREA] = local_area
        constants[RESISTANCE] = local_resistance
        return functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
        )[0]["integrator"]["v"]

    lane_areas = torch.stack((area * 0.95, area * 1.05))
    lane_resistances = torch.stack((resistance * 0.9, resistance * 1.1))
    actual = torch.vmap(voltage)(lane_areas, lane_resistances)
    expected = torch.stack(
        tuple(voltage(lane_areas[index], lane_resistances[index]) for index in range(2))
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)

    empty = torch.vmap(voltage)(
        area.new_empty((0, *area.shape)),
        resistance.new_empty((0, *resistance.shape)),
    )
    assert empty.shape == (0, *model.shape)

    eager_jacobian = torch.func.jacrev(voltage, argnums=0)(area, resistance)
    with torch_compiler_warning_context():
        compiled = torch.compile(
            torch.func.jacrev(voltage, argnums=0),
            backend="aot_eager",
            fullgraph=True,
        )
        compiled_jacobian = compiled(area, resistance)
    torch.testing.assert_close(
        compiled_jacobian,
        eager_jacobian,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="native Thomas transform facade requires dendra-solvers",
)
def test_native_cable_thomas_facade_preserves_parity_and_area_jacobian():
    source = _model(method="thomas")
    imperative = _model(method="thomas")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    _ve, intra = _drives(source, steps=1)

    def voltage(area):
        constants = dict(tensors.constants)
        constants[AREA] = area
        return functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
            dn.func.StepInput(intra=intra[0]),
        )[0]["integrator"]["v"]

    actual = voltage(tensors.constants[AREA])
    _imperative_step(imperative, intra=intra[0])
    torch.testing.assert_close(actual, imperative.v, rtol=2.0e-12, atol=2.0e-12)
    jacobian = torch.func.jacrev(voltage)(tensors.constants[AREA])
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) > 0


def _runner_case(functional, tensors, intra, *, checkpointed):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    area = constants[AREA].requires_grad_()
    resistance = constants[RESISTANCE].requires_grad_()
    state = _clone_tree(tensors.state)
    initial_voltage = state["integrator"]["v"].requires_grad_()
    local_intra = intra.detach().clone().requires_grad_()
    prepared = functional.prepare(parameters, constants)
    step = partial(functional.step, parameters, prepared)
    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.longrun
    final, auxiliary = runner(
        functional,
        step,
        state,
        STEPS * DT,
        2,
        dn.func.RolloutInput(intra=local_intra),
    )
    loss = final["integrator"]["v"].square().mean() + 0.01 * auxiliary["v"].mean()
    gradients = torch.autograd.grad(
        loss,
        (area, resistance, initial_voltage, local_intra),
    )
    return final, auxiliary, loss, gradients


def test_native_cable_checkpointed_runner_matches_ordinary_bptt():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    _ve, intra = _drives(model)
    ordinary = _runner_case(functional, tensors, intra, checkpointed=False)
    checkpointed = _runner_case(functional, tensors, intra, checkpointed=True)

    _assert_tree_close(checkpointed[0], ordinary[0], rtol=2.0e-12, atol=2.0e-12)
    _assert_tree_close(checkpointed[1], ordinary[1], rtol=2.0e-12, atol=2.0e-12)
    torch.testing.assert_close(checkpointed[2], ordinary[2], rtol=2.0e-12, atol=2.0e-12)
    for actual, expected in zip(checkpointed[3], ordinary[3], strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


@pytest.mark.parametrize("constant_name", [AREA, RESISTANCE])
def test_native_cable_prepared_freshness_tracks_canonical_constants(constant_name):
    model = _model()
    source_area = model._canonical_area_cm2.detach().clone()
    source_resistance = model._canonical_edge_resistance_ohm.detach().clone()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    constants = dict(tensors.constants)
    prepared = functional.prepare(tensors.parameters, constants)

    constants[constant_name].reshape(-1)[-1].mul_(1.01)
    with pytest.raises(dn.func.FunctionalizationError, match="constants changed"):
        functional.step(tensors.parameters, prepared, tensors.state)

    refreshed = functional.prepare(tensors.parameters, constants)
    state, _ = functional.step(tensors.parameters, refreshed, tensors.state)
    assert torch.isfinite(state["integrator"]["v"]).all()
    torch.testing.assert_close(
        model._canonical_area_cm2, source_area, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        model._canonical_edge_resistance_ohm,
        source_resistance,
        rtol=0.0,
        atol=0.0,
    )


def test_native_cable_provenance_state_dict_and_commit_fail_closed():
    source = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    checkpoint, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=2,
    )

    same = _model()
    same.integrator._initialize(same, DT, force=True)
    snapshot = source.state_dict()
    same.load_state_dict(snapshot)
    assert not same.integrator.initialized
    extracted = functional.extract(same)
    assert torch.equal(extracted.constants[AREA], same._canonical_area_cm2)
    functional.commit_state_(same, checkpoint)
    assert torch.equal(same._canonical_area_cm2, source._canonical_area_cm2)
    assert torch.equal(
        same._canonical_edge_resistance_ohm,
        source._canonical_edge_resistance_ohm,
    )

    different = _model(variant=1)
    with pytest.raises(
        dn.func.FunctionalizationError, match="structure does not match"
    ):
        functional.extract(different)
    with pytest.raises(
        dn.func.FunctionalizationError, match="structure does not match"
    ):
        functional.commit_state_(different, checkpoint)

    corrupt = _model()
    corrupt._canonical_edge_resistance_ohm[..., -1].mul_(1.01)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="canonical geometry is invalid.*modified",
    ):
        dn.func.make_functional(corrupt, dt=DT)


@pytest.mark.parametrize(
    "name",
    ["celsius", "cm", "cm_scale", "area_scale", "rhoa_scale"],
)
def test_native_cable_rejects_live_effective_parameter_overrides(name):
    model = _model()
    value = getattr(model, name)
    setattr(model, name, value * 1.01)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="live effective values.*raw parameter sources",
    ):
        dn.func.make_functional(model, dt=DT)


def test_native_cable_rejects_noncanonical_constructor_and_subclass():
    low_level = dn.Cable(
        2,
        4,
        dtype=torch.float64,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    low_level.insert(pas)
    low_level.initialize()

    class CableSubclass(dn.Cable):
        pass

    subclass = CableSubclass.from_morphology(
        _morphology(),
        dtype=torch.float64,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    subclass.insert(pas)
    subclass.initialize()

    for unsupported in (low_level, subclass):
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="current supported topologies",
        ):
            dn.func.make_functional(unsupported, dt=DT)


def test_native_cable_rejects_integrator_subclasses_outside_audited_contract():
    class AlteredUB(_bwd_euler_ub):
        def __init__(self, model, mech, imem=None):
            super().__init__(model, mech, method="pcr", imem=imem)

    model = dn.Cable.from_morphology(
        _morphology(),
        dtype=torch.float64,
        integrator=AlteredUB,
    )
    model.insert(pas)
    model.initialize()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="native Cable functionalization requires bwd_euler_ub",
    ):
        dn.func.make_functional(model, dt=DT)

    def replacement(*args, **kwargs):
        return None

    rebound = _model()
    rebound.integrator._step = replacement
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="native Cable functionalization requires bwd_euler_ub",
    ):
        dn.func.make_functional(rebound, dt=DT)


def test_native_cable_uses_widened_runtime_geometry_after_dtype_roundtrip():
    model = _model(dtype=torch.float32)
    graph = model.compartment_graph
    model.double()
    model.initialize(force_rebuild=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    graph_area = (
        torch.as_tensor(
            graph.geometry.area_um2,
            dtype=torch.float64,
        )
        .reshape(1, -1)
        .expand(model.core_shape())
        * 1.0e-8
    )
    graph_resistance = (
        torch.as_tensor(
            graph.geometry.edge_resistance_ohm,
            dtype=torch.float64,
        )
        .reshape(1, -1)
        .expand(model.core_shape())
    )

    assert torch.equal(tensors.constants[AREA], model._canonical_area_cm2)
    assert torch.equal(
        tensors.constants[RESISTANCE],
        model._canonical_edge_resistance_ohm,
    )
    assert not torch.equal(tensors.constants[AREA], graph_area)
    assert not torch.equal(tensors.constants[RESISTANCE], graph_resistance)
    assert functional.shape == model.shape


def test_native_cable_source_mutation_after_lowering_fails_without_cache_updates():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    cache_signature = model._validated_runtime_contract_signature
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)
    assert model._validated_runtime_contract_signature is cache_signature

    model._canonical_area_cm2[..., -1].mul_(1.01)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="canonical geometry changed after make_functional",
    ):
        functional.prepare(tensors.parameters, tensors.constants)
