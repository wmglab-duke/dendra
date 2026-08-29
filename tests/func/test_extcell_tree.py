"""Functional acceptance contracts for branched ``ExtCellTree`` models."""

from __future__ import annotations

import math
from functools import partial

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.tree_bt import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas

DT = 0.01
DTYPE = torch.float64
STEPS = 4
PAS_G = "integrator.mech.mechanisms.pas.g_param"
CM_RAW = "cm_param.rho"
RHOA_RAW = "rhoa_param.rho"
RHOA_SCALE_RAW = "rhoa_scale_param.rho"
CM_SCALE_RAW = "cm_scale_param.rho"
AREA_SCALE_RAW = "area_scale_param.rho"
AREA = "canonical_area_cm2"
RESISTANCE = "canonical_edge_resistance_ohm"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="ExtCellTree CPU execution requires dendra-solvers",
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _branched_nonzero_root_graph():
    """Return exact heterogeneous geometry whose root is storage slot three."""
    graph = nx.DiGraph()
    areas = (130.0, 85.0, 105.0, 240.0)
    cms = (0.75, 1.20, 0.95, 1.10)
    for node in range(4):
        graph.add_node(
            node,
            name=f"compartment.{node}",
            kind="compartment",
            L=8.0 + node,
            diam=1.2 + 0.2 * node,
            Ra=80.0 + 7.0 * node,
            cm=cms[node],
            area=areas[node],
            xraxial=[1.4 + 0.2 * node, 2.2 + 0.25 * node],
            xc=[0.06 + 0.01 * node, 0.11 + 0.015 * node],
            xg=[1.2e-4 + 0.2e-4 * node, 2.1e-4 + 0.3e-4 * node],
        )
    for parent, child, resistance in (
        (3, 0, 1.7e8),
        (3, 2, 2.9e8),
        (2, 1, 4.3e8),
    ):
        graph.add_edge(parent, child, R_ohm=resistance, L=1.0)
    return graph


def _alternate_graph():
    """Return the same payload with one leaf attached to another parent."""
    graph = _branched_nonzero_root_graph()
    edge = dict(graph.edges[2, 1])
    graph.remove_edge(2, 1)
    graph.add_edge(0, 1, **edge)
    return graph


def _one_compartment_graph():
    graph = nx.DiGraph()
    graph.add_node(
        0,
        name="soma",
        kind="compartment",
        L=10.0,
        diam=2.0,
        Ra=113.0,
        cm=0.85,
        area=79.0,
        xraxial=[1.7, 2.4],
        xc=[0.08, 0.13],
        xg=[1.5e-4, 2.5e-4],
    )
    return graph


class _AreaOverrideExtCellTree(dn.ExtCellTree):
    """Subclass whose exact area semantics are no longer framework-owned."""

    @property
    def area(self):
        return super().area


def _seed_circuit_state(model):
    """Install genuine shell carry instead of a membrane-only initial state."""
    inner_shell = torch.linspace(
        -2.0,
        1.0,
        model.v.numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(model.shape)
    outer_shell = torch.linspace(
        0.5,
        -1.5,
        model.v.numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(model.shape)
    with torch.no_grad():
        model.vc[..., 1].copy_(inner_shell)
        model.vc[..., 2].copy_(outer_shell)
        model.vc[..., 0].copy_(model.v + inner_shell)


def _build_model(
    graph,
    *,
    batch_calls=(),
    model_type=dn.ExtCellTree,
    imem=False,
    integrator=None,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=imem):
        if integrator is None:
            integrator = dhs_bt(threads=2, imem=imem)
        model = model_type.from_graph(
            graph,
            N=2,
            dtype=DTYPE,
            v_init=-64.0,
            integrator=integrator,
        )
        model.rhoa_scale_param.set(1.20)
        model.cm_scale_param.set(0.85)
        model.area_scale_param.set(1.10)
        model.insert(pas, g=4.0e-4, e=-71.0)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
        if hasattr(model, "vc"):
            _seed_circuit_state(model)
    return model


def _model(*, batch_calls=(), alternate_topology=False, imem=False):
    graph = _alternate_graph() if alternate_topology else _branched_nonzero_root_graph()
    return _build_model(graph, batch_calls=batch_calls, imem=imem)


def _one_compartment_model():
    return _build_model(_one_compartment_graph())


def _drives(model, *, steps=STEPS):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -1.5,
        2.0,
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


def _initialize_imperative(model):
    dt = torch.as_tensor(DT, dtype=model.dtype(), device=model.device())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    return dt


def _imperative_step(model, dt, ve, intra):
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _clone_tree(tree):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), tree)


def _assert_tree_close(actual, expected, *, rtol=2.0e-10, atol=2.0e-11):
    actual_with_paths, actual_spec = torch.utils._pytree.tree_flatten_with_path(actual)
    expected_with_paths, expected_spec = torch.utils._pytree.tree_flatten_with_path(
        expected
    )
    assert actual_spec == expected_spec
    for (actual_path, actual_leaf), (expected_path, expected_leaf) in zip(
        actual_with_paths,
        expected_with_paths,
        strict=True,
    ):
        assert actual_path == expected_path
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"tensor leaf {actual_path}: {message}",
        )


def _assert_voltage_relation(state):
    voltage = state["integrator"]["v"]
    circuit_voltage = state["integrator"]["vc"]
    torch.testing.assert_close(
        voltage,
        circuit_voltage[..., 0] - circuit_voltage[..., 1],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def _while_loop_nodes(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.higher_order.while_loop
    ]


def test_extcell_tree_state_topology_geometry_and_workspace_schema():
    from dendra_solvers import dhs_bt_solve

    model = _model()
    topology = model.compartment_graph.topology
    assert topology.root == 3
    assert topology.parent_index == (3, 2, 3, -1)

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert functional._transition.solver is dhs_bt_solve
    assert set(tensors.state) == {"integrator", "clock", "control"}
    assert set(tensors.state["integrator"]) == {"v", "vc"}
    assert tensors.state["integrator"]["v"].shape == model.shape
    assert tensors.state["integrator"]["vc"].shape == (*model.shape, 3)
    assert torch.count_nonzero(tensors.state["integrator"]["vc"][..., 1:])
    _assert_voltage_relation(tensors.state)

    expected_constants = {
        "dx": model.core_shape(),
        AREA: model.core_shape(),
        RESISTANCE: model.core_shape(),
        "xraxial": (*model.core_shape(), 2),
        "xc": (*model.core_shape(), 2),
        "xg": (*model.core_shape(), 2),
    }
    for name, shape in expected_constants.items():
        assert tensors.constants[name].shape == tuple(shape)
        assert tensors.constants[name].dtype == DTYPE
    torch.testing.assert_close(tensors.constants[AREA], model.area)
    torch.testing.assert_close(
        tensors.constants[RESISTANCE],
        model.edge_resistance_ohm,
    )
    for name in ("rhoa_scale", "cm_scale", "area_scale"):
        assert not torch.equal(
            getattr(model, name),
            torch.ones_like(getattr(model, name)),
        )

    batch = math.prod(model.shape[:-1])
    compartments = model.shape[-1]
    workspace_shapes = {
        "main_blocks": (batch, compartments, 3, 3),
        "g_to_parent": (batch, compartments, 3),
        "area": (batch, compartments),
        "cm_dt": (batch, compartments),
        "xc_dt": (batch, compartments, 2),
        "c_rad": (batch, compartments, 3),
        "xg": (batch, compartments, 2),
    }
    workspace = prepared.values["integrator"]
    assert workspace["dt"].shape == ()
    for name, shape in workspace_shapes.items():
        assert workspace[name].shape == shape
        assert workspace[name].dtype == DTYPE
        assert workspace[name].device.type == "cpu"


@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_extcell_tree_step_and_rollout_match_every_imperative_leaf(batch_calls):
    source = _model(batch_calls=batch_calls)
    reference = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)
    dt = _initialize_imperative(reference)
    initial_circuit_voltage = tensors.state["integrator"]["vc"].clone()
    state = tensors.state

    for index in range(STEPS):
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(reference, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)
        _assert_voltage_relation(state)
        torch.testing.assert_close(
            auxiliary["v"],
            state["integrator"]["v"],
            rtol=0.0,
            atol=0.0,
        )

    assert not torch.equal(state["integrator"]["vc"], initial_circuit_voltage)
    rolled = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_tree_close(rolled[0], state)
    _assert_voltage_relation(rolled[0])
    torch.testing.assert_close(
        rolled[1]["v"],
        rolled[0]["integrator"]["v"],
        rtol=0.0,
        atol=0.0,
    )

    if batch_calls:
        assert source.shape == (2, *source.core_shape())
        assert rolled[0]["integrator"]["vc"].shape == (*source.shape, 3)


def test_extcell_tree_one_compartment_zero_edge_parity_and_transforms():
    source = _one_compartment_model()
    reference = _one_compartment_model()
    topology = source.compartment_graph.topology
    assert topology.parent_index == (-1,)
    assert source.material_edge_index.shape == (2, 0)
    assert torch.count_nonzero(source.edge_resistance_ohm) == 0

    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    workspace = prepared.values["integrator"]
    assert workspace["main_blocks"].shape == (source.np, 1, 3, 3)
    assert workspace["g_to_parent"].shape == (source.np, 1, 3)
    assert torch.count_nonzero(workspace["g_to_parent"]) == 0

    ve, intra = _drives(source, steps=1)
    actual, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(ve=ve[0], intra=intra[0]),
    )
    dt = _initialize_imperative(reference)
    _imperative_step(reference, dt, ve[0], intra[0])
    _assert_tree_close(actual, functional.extract(reference).state)
    _assert_voltage_relation(actual)

    def circuit_voltage(canonical_area, canonical_resistance, xg):
        constants = {
            **tensors.constants,
            AREA: canonical_area,
            RESISTANCE: canonical_resistance,
            "xg": xg,
        }
        state, _auxiliary = functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )
        return state["integrator"]["vc"]

    arguments = (
        tensors.constants[AREA],
        tensors.constants[RESISTANCE],
        tensors.constants["xg"],
    )
    reverse = torch.func.jacrev(circuit_voltage, argnums=(0, 1, 2))(*arguments)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(circuit_voltage, argnums=(0, 1, 2))(*arguments)
    for index, (actual_jacobian, expected_jacobian) in enumerate(
        zip(reverse, forward, strict=True)
    ):
        assert torch.isfinite(actual_jacobian).all()
        torch.testing.assert_close(
            actual_jacobian,
            expected_jacobian,
            rtol=2.0e-10,
            # Area is expressed in cm², so its Jacobian is O(1e4-1e5).
            # Reverse/forward cross terms near zero accumulate correspondingly
            # larger absolute roundoff while retaining ~1e-12 relative error.
            atol=2.0e-7 if index == 0 else 2.0e-11,
        )
    assert torch.count_nonzero(reverse[0])
    assert torch.count_nonzero(reverse[1]) == 0
    assert torch.count_nonzero(reverse[2])

    def membrane_voltage_from_xg(xg):
        constants = {**tensors.constants, "xg": xg}
        state, _auxiliary = functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )
        return state["integrator"]["v"]

    # With no axial edge, shell conductance changes only absolute/common-mode
    # circuit potentials. The membrane difference vi-ve0 is exactly invariant.
    membrane_xg = torch.func.jacrev(membrane_voltage_from_xg)(tensors.constants["xg"])
    assert torch.count_nonzero(membrane_xg) == 0


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_extcell_tree_host_runners_match_direct_rollout(runner_name):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)
    inputs = dn.func.RolloutInput(ve=ve, intra=intra)
    expected = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
    )
    step = partial(functional.step, tensors.parameters, prepared)

    if runner_name == "run":
        actual = dn.func.run(functional, step, tensors.state, inputs)
    else:
        actual = getattr(dn.func, runner_name)(
            functional,
            step,
            tensors.state,
            STEPS * DT,
            2,
            inputs,
        )

    _assert_tree_close(actual, expected)
    _assert_voltage_relation(actual[0])


def _runner_gradient_case(functional, tensors, ve, intra, *, checkpointed):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    state = _clone_tree(tensors.state)

    parameters[PAS_G].requires_grad_()
    parameters[CM_RAW].requires_grad_()
    parameters[RHOA_SCALE_RAW].requires_grad_()
    parameters[CM_SCALE_RAW].requires_grad_()
    parameters[AREA_SCALE_RAW].requires_grad_()
    constants[AREA].requires_grad_()
    constants[RESISTANCE].requires_grad_()
    constants["xraxial"].requires_grad_()
    constants["xc"].requires_grad_()
    constants["xg"].requires_grad_()
    state["integrator"]["v"].requires_grad_()
    state["integrator"]["vc"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()
    targets = (
        parameters[PAS_G],
        parameters[CM_RAW],
        parameters[RHOA_SCALE_RAW],
        parameters[CM_SCALE_RAW],
        parameters[AREA_SCALE_RAW],
        constants[AREA],
        constants[RESISTANCE],
        constants["xraxial"],
        constants["xc"],
        constants["xg"],
        state["integrator"]["v"],
        state["integrator"]["vc"],
        ve,
        intra,
    )

    prepared = functional.prepare(parameters, constants)
    step = partial(functional.step, parameters, prepared)
    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.longrun
    final, auxiliary = runner(
        functional,
        step,
        state,
        STEPS * DT,
        2,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    loss = (
        final["integrator"]["v"].square().mean()
        + 0.01 * final["integrator"]["vc"].square().mean()
        + 0.001 * auxiliary["v"].sin().mean()
    )
    gradients = torch.autograd.grad(loss, targets)
    return final, auxiliary, loss, gradients


def test_extcell_tree_checkpointed_runner_matches_ordinary_full_block_bptt():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    ordinary = _runner_gradient_case(
        functional,
        tensors,
        ve,
        intra,
        checkpointed=False,
    )
    checkpointed = _runner_gradient_case(
        functional,
        tensors,
        ve,
        intra,
        checkpointed=True,
    )

    _assert_tree_close(checkpointed[0], ordinary[0])
    _assert_tree_close(checkpointed[1], ordinary[1])
    torch.testing.assert_close(checkpointed[2], ordinary[2])
    for index, (actual, expected) in enumerate(
        zip(checkpointed[3], ordinary[3], strict=True)
    ):
        assert torch.isfinite(actual).all()
        # For a purely passive linearized current, old membrane ``v`` cancels
        # algebraically. ``vc`` is the genuine block carry and remains active.
        if index == 10:
            assert torch.count_nonzero(actual) == 0
        else:
            assert torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


def test_extcell_tree_exact_geometry_jacrev_and_jacfwd_agree():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=1)

    names = (AREA, RESISTANCE, "dx", "xraxial", "xc", "xg")

    def voltage(*geometry):
        constants = {**tensors.constants, **dict(zip(names, geometry, strict=True))}
        state, _auxiliary = functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )
        return state["integrator"]["v"]

    arguments = tuple(tensors.constants[name] for name in names)
    argnums = tuple(range(len(arguments)))
    reverse = torch.func.jacrev(voltage, argnums=argnums)(*arguments)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(voltage, argnums=argnums)(*arguments)

    for actual, expected in zip(reverse, forward, strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)
    # Canonical resistance owns the authored intracellular resistivity. The
    # child-indexed root slot is structural and cannot influence the solve.
    assert torch.count_nonzero(reverse[1][..., 3]) == 0
    assert torch.count_nonzero(reverse[1][..., :3])


def test_extcell_tree_raw_scale_jacrev_and_jacfwd_semantics():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=1)
    names = (RHOA_RAW, RHOA_SCALE_RAW, CM_SCALE_RAW, AREA_SCALE_RAW)

    def voltage(*raw_scales):
        parameters = {
            **tensors.parameters,
            **dict(zip(names, raw_scales, strict=True)),
        }
        state, _auxiliary = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )
        return state["integrator"]["v"]

    arguments = tuple(tensors.parameters[name] for name in names)
    argnums = tuple(range(len(arguments)))
    reverse = torch.func.jacrev(voltage, argnums=argnums)(*arguments)
    forward = torch.func.jacfwd(voltage, argnums=argnums)(*arguments)
    for actual, expected in zip(reverse, forward, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    # The exact compiled resistance already includes authored ``rhoa``. Only
    # the explicit runtime scale participates after morphology compilation.
    assert torch.count_nonzero(reverse[0]) == 0
    for jacobian in reverse[1:]:
        assert torch.count_nonzero(jacobian)


def _transform_case(*, batch_calls=()):
    model = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=1)

    def response(raw_rhoa_scale, intra_value):
        parameters = {
            **tensors.parameters,
            RHOA_SCALE_RAW: raw_rhoa_scale,
        }
        state, _auxiliary = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra_value),
        )
        return state["integrator"]["v"]

    return model, response, tensors.parameters[RHOA_SCALE_RAW], intra[0]


def test_extcell_tree_jacrev_jacfwd_and_vmap_including_empty_lanes():
    model, response, raw_rhoa_scale, intra = _transform_case()

    reverse = torch.func.jacrev(response, argnums=0)(raw_rhoa_scale, intra)
    forward = torch.func.jacfwd(response, argnums=0)(raw_rhoa_scale, intra)
    assert torch.isfinite(reverse).all()
    assert torch.count_nonzero(reverse)
    torch.testing.assert_close(reverse, forward, rtol=2.0e-10, atol=2.0e-11)

    scale_lanes = raw_rhoa_scale + raw_rhoa_scale.new_tensor((-0.1, 0.0, 0.15))
    intra_lanes = torch.stack((intra - 1.0e-10, intra, intra + 2.0e-10))
    actual = torch.vmap(response)(scale_lanes, intra_lanes)
    expected = torch.stack(
        tuple(
            response(scale, drive)
            for scale, drive in zip(scale_lanes, intra_lanes, strict=True)
        )
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    empty = torch.vmap(response)(
        raw_rhoa_scale.new_empty((0,)),
        intra.new_empty((0, *model.shape)),
    )
    assert empty.shape == (0, *model.shape)


def test_extcell_tree_nested_explicit_batch_and_nested_empty_vmap_compose():
    model, response, raw_rhoa_scale, intra = _transform_case(batch_calls=(2, 3))
    assert model.shape == (3, 2, 2, 4)

    offsets = intra.new_tensor(
        (
            (-2.0e-10, -1.0e-10, 0.0),
            (1.0e-10, 2.0e-10, 3.0e-10),
        )
    )
    scale_lanes = raw_rhoa_scale + raw_rhoa_scale.new_tensor(
        (
            (-0.15, 0.0, 0.10),
            (0.20, 0.30, 0.45),
        )
    )
    intra_lanes = intra.reshape((1, 1, *model.shape)) + offsets.reshape(
        (2, 3, *((1,) * len(model.shape)))
    )

    mapped = torch.vmap(torch.vmap(response))
    actual = mapped(scale_lanes, intra_lanes)
    expected = torch.stack(
        tuple(
            torch.stack(
                tuple(
                    response(scale_lanes[outer, inner], intra_lanes[outer, inner])
                    for inner in range(3)
                )
            )
            for outer in range(2)
        )
    )
    assert actual.shape == (2, 3, *model.shape)
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    nested_empty = mapped(
        raw_rhoa_scale.new_empty((0, 3)),
        intra.new_empty((0, 3, *model.shape)),
    )
    assert nested_empty.shape == (0, 3, *model.shape)


def test_extcell_tree_compile_composes_with_jacrev():
    _model_value, response, raw_rhoa_scale, intra = _transform_case()
    transformed = torch.func.jacrev(response, argnums=0)
    expected = transformed(raw_rhoa_scale, intra)

    with torch_compiler_warning_context():
        compiled = torch.compile(transformed, backend="eager", fullgraph=True)
        actual = compiled(raw_rhoa_scale, intra)

    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_extcell_tree_compiled_chunk_uses_one_bounded_native_dhs_bt_loop():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)
    inputs = dn.func.RolloutInput(ve=ve, intra=intra)
    expected = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
    )
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    chunk = functional.compile_rollout_chunk(STEPS, backend=backend)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = chunk(
            tensors.parameters,
            prepared,
            tensors.state,
            inputs,
        )

    _assert_tree_close(actual, expected)
    assert len(graphs) == 1
    assert len(_while_loop_nodes(graphs[0])) == 1
    graph_modules = (
        module
        for module in graphs[0].modules()
        if isinstance(module, torch.fx.GraphModule)
    )
    assert any(
        node.op == "call_function"
        and node.target is torch.ops.dendra_solvers.dhs_bt_solve.default
        for graph_module in graph_modules
        for node in graph_module.graph.nodes
    )


def test_extcell_tree_commit_and_imperative_resume_are_topology_safe():
    source = _model(batch_calls=(2,))
    target = _model(batch_calls=(2,))
    reference = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=5)

    checkpoint, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:3], intra=intra[:3]),
    )
    functional.commit_state_(target, checkpoint)
    _assert_tree_close(functional.extract(target).state, checkpoint)

    target_dt = _initialize_imperative(target)
    for index in range(3, 5):
        _imperative_step(target, target_dt, ve[index], intra[index])
    reference_dt = _initialize_imperative(reference)
    for index in range(5):
        _imperative_step(reference, reference_dt, ve[index], intra[index])
    _assert_tree_close(
        functional.extract(target).state,
        functional.extract(reference).state,
    )

    for incompatible in (
        _model(batch_calls=(3,)),
        _model(batch_calls=(2,), alternate_topology=True),
    ):
        voltage_before = incompatible.v.detach().clone()
        circuit_before = incompatible.vc.detach().clone()
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="structure does not match",
        ):
            functional.commit_state_(incompatible, checkpoint)
        torch.testing.assert_close(incompatible.v, voltage_before)
        torch.testing.assert_close(incompatible.vc, circuit_before)


def test_extcell_tree_rejects_rebound_target_vc_layout_transactionally():
    source = _model()
    target = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    target.vc = target.vc[..., :2].clone()
    voltage_before = target.v.detach().clone()
    circuit_before = target.vc.detach().clone()
    time_before = target.t.detach().clone()

    for operation in (
        lambda: functional.extract(target),
        lambda: functional.commit_state_(target, tensors.state),
    ):
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="runtime-state layout does not match",
        ):
            operation()
        torch.testing.assert_close(target.v, voltage_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(target.vc, circuit_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(target.t, time_before, rtol=0.0, atol=0.0)


def test_extcell_tree_requires_exact_v_and_vc_integrator_state_contract():
    model = _model()
    model.integrator.v_vars = ["v"]

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"must declare exactly the \('v', 'vc'\) integrator state contract",
    ):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.parametrize(
    ("model_factory", "message"),
    [
        (lambda: _model(imem=True), "i_membrane recording is not supported yet"),
        (
            lambda: _build_model(
                _branched_nonzero_root_graph(),
                integrator=dn.dhs(threads=2, imem=False),
            ),
            "requires exactly dhs_bt",
        ),
        (
            lambda: _build_model(
                _branched_nonzero_root_graph(),
                model_type=_AreaOverrideExtCellTree,
            ),
            "standard exact area",
        ),
    ],
)
def test_extcell_tree_unsupported_numerical_contracts_fail_closed(
    model_factory,
    message,
):
    model = model_factory()
    with pytest.raises(dn.func.FunctionalizationError, match=message):
        dn.func.make_functional(model, dt=DT)


def test_extcell_tree_low_level_noncanonical_construction_fails_closed():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.ExtCellTree(
            2,
            4,
            graph=_branched_nonzero_root_graph(),
            dtype=DTYPE,
            v_init=-64.0,
            integrator=dhs_bt(threads=2, imem=False),
        )
        model.insert(pas, g=4.0e-4, e=-71.0)
        model.initialize()
        model.train()
        _seed_circuit_state(model)

    assert model.compartment_graph is None
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="current supported topologies",
    ):
        dn.func.make_functional(model, dt=DT)


def test_real_l4_topology_extcell_tree_matches_one_imperative_step_when_available():
    cortical = pytest.importorskip("dendra_models.models.cells.cortical")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE):
        authored = cortical.L4_NBC_dNAC(
            8989,
            N=1,
            integrator=dn.dhs(threads=2, imem=False),
        )

    def build():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE):
            model = dn.ExtCellTree.from_graph(
                authored.graph.copy(),
                N=1,
                integrator=dhs_bt(threads=2, imem=False),
            )
            model.xraxial.copy_(
                torch.linspace(
                    1.5,
                    3.0,
                    model.xraxial.numel(),
                    dtype=DTYPE,
                ).reshape_as(model.xraxial)
            )
            model.xc.copy_(
                torch.linspace(
                    0.06,
                    0.18,
                    model.xc.numel(),
                    dtype=DTYPE,
                ).reshape_as(model.xc)
            )
            model.xg.copy_(
                torch.linspace(
                    1.0e-4,
                    3.0e-4,
                    model.xg.numel(),
                    dtype=DTYPE,
                ).reshape_as(model.xg)
            )
            model.insert(pas, g=4.0e-4, e=-71.0)
            model.initialize()
            model.train()
            _seed_circuit_state(model)
        return model

    source = build()
    reference = build()
    assert source.shape == (1, 874)
    assert any(source.graph.out_degree(node) > 1 for node in source.graph)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=1)

    actual, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(ve=ve[0], intra=intra[0]),
    )
    dt = _initialize_imperative(reference)
    _imperative_step(reference, dt, ve[0], intra[0])
    _assert_tree_close(actual, functional.extract(reference).state)
    _assert_voltage_relation(actual)
