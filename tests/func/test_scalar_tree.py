"""Functional acceptance contracts for branched scalar ``Tree`` models."""

from __future__ import annotations

from functools import partial

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas

DT = 0.01
DTYPE = torch.float64
STEPS = 4
PAS_G = "integrator.mech.mechanisms.pas.g_param"
CM_RAW = "cm_param.rho"
RHOA_SCALE_RAW = "rhoa_scale_param.rho"
AREA = "canonical_area_cm2"
RESISTANCE = "canonical_edge_resistance_ohm"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="scalar Tree CPU execution requires dendra-solvers",
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _morphology():
    """Return a small branch with a retained zero-volume junction."""
    morphology = dn.Morphology(rhoa=91.0, cm=1.0)
    soma = morphology.section(
        "soma",
        L=18.0,
        diam=16.0,
        nseg=1,
        rhoa=83.0,
        cm=0.8,
    )
    trunk = morphology.section(
        "trunk",
        points=[
            (0.0, 0.0, 0.0, 4.0),
            (0.0, 60.0, 0.0, 2.0),
        ],
        nseg=3,
        rhoa=107.0,
        cm=1.1,
    )
    left = morphology.section(
        "left",
        points=[
            (0.0, 60.0, 0.0, 2.0),
            (-35.0, 95.0, 0.0, 1.1),
        ],
        nseg=2,
        rhoa=129.0,
        cm=1.25,
    )
    right = morphology.section(
        "right",
        points=[
            (0.0, 60.0, 0.0, 1.8),
            (42.0, 98.0, 4.0, 0.9),
        ],
        nseg=2,
        rhoa=143.0,
        cm=0.95,
    )
    trunk.connect(soma.at(1.0), child_end=0)
    left.connect(trunk.at(1.0), child_end=0)
    right.connect(trunk.at(1.0), child_end=0)
    return morphology


def _alternate_graph():
    """Return the same graph payload with one leaf reparented."""
    graph = _morphology().compile().to_networkx()
    leaves = sorted(node for node in graph if graph.out_degree(node) == 0)
    new_parent, moved = leaves[:2]
    old_parent = next(graph.predecessors(moved))
    edge = dict(graph.edges[old_parent, moved])
    graph.remove_edge(old_parent, moved)
    graph.add_edge(new_parent, moved, **edge)
    return graph


def _reordered_root_graph():
    """Return exact geometry whose root is storage slot 3, not slot 0."""
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
        )
    for parent, child, resistance in (
        (3, 0, 1.7e8),
        (3, 2, 2.9e8),
        (2, 1, 4.3e8),
    ):
        graph.add_edge(parent, child, R_ohm=resistance, L=1.0)
    return graph


def _one_compartment_graph():
    """Return an exact zero-edge Tree with deliberately noncylindrical area."""
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
    )
    return graph


class _AreaOverrideTree(dn.Tree):
    """Tree subclass whose area semantics are no longer framework-owned."""

    @property
    def area(self):
        return super().area


def _model(*, batch_calls=(), alternate_topology=False, imem=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=imem):
        integrator = dn.dhs(threads=2)
        if alternate_topology:
            model = dn.Tree.from_graph(
                _alternate_graph(),
                N=2,
                dtype=DTYPE,
                v_init=-64.0,
                integrator=integrator,
            )
        else:
            model = dn.Tree.from_morphology(
                _morphology(),
                N=2,
                dtype=DTYPE,
                v_init=-64.0,
                integrator=integrator,
            )
        model.insert(pas, g=4.0e-4, e=-71.0)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _one_compartment_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Tree.from_graph(
            _one_compartment_graph(),
            N=2,
            dtype=DTYPE,
            v_init=-64.0,
            integrator=dn.dhs(threads=2, imem=False),
        )
        model.insert(pas, g=4.0e-4, e=-71.0)
        model.initialize()
        model.train()
    return model


def _reordered_root_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Tree.from_graph(
            _reordered_root_graph(),
            N=2,
            dtype=DTYPE,
            v_init=-64.0,
            integrator=dn.dhs(threads=2, imem=False),
        )
        model.insert(pas, g=4.0e-4, e=-71.0)
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


def _clone_tree(value):
    return torch.utils._pytree.tree_map(
        lambda tensor: tensor.detach().clone(),
        value,
    )


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


def _while_loop_nodes(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.higher_order.while_loop
    ]


def test_scalar_tree_step_and_rollout_match_imperative_every_leaf():
    source = _model()
    reference = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)
    dt = _initialize_imperative(reference)
    state = tensors.state
    step_auxiliary = []

    for index in range(STEPS):
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        step_auxiliary.append(auxiliary)
        _imperative_step(reference, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)

    rolled_state, rolled_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_tree_close(rolled_state, state)
    _assert_tree_close(rolled_auxiliary, step_auxiliary[-1])


def test_scalar_tree_reordered_root_matches_imperative_with_both_drives():
    source = _reordered_root_model()
    reference = _reordered_root_model()
    topology = source.compartment_graph.topology
    assert topology.root == 3
    assert topology.parent_index == (3, 2, 3, -1)

    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert torch.equal(tensors.constants[AREA], source.area)
    assert torch.equal(tensors.constants[RESISTANCE], source.edge_resistance_ohm)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=3)
    dt = _initialize_imperative(reference)
    state = tensors.state

    for index in range(3):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(reference, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)


def test_scalar_tree_one_compartment_zero_edge_parity_and_transforms():
    source = _one_compartment_model()
    reference = _one_compartment_model()
    topology = source.compartment_graph.topology
    assert topology.parent_index == (-1,)
    assert source.material_edge_index.shape == (2, 0)
    assert torch.count_nonzero(source.edge_resistance_ohm) == 0
    authored_area = (
        torch.as_tensor(79.0, dtype=source.dtype(), device=source.device()) * 1.0e-8
    )
    torch.testing.assert_close(
        source.area,
        authored_area.expand_as(source.area),
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(source.area, torch.pi * source.diam * source.dx * 1.0e-8)

    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    workspace = prepared.values["integrator"]
    assert workspace["axial_conductance"].shape == source.shape
    assert workspace["edge_conductance"].shape == (source.np, 0)
    assert torch.count_nonzero(workspace["axial_conductance"]) == 0

    ve, intra = _drives(source, steps=3)
    dt = _initialize_imperative(reference)
    state = tensors.state
    for index in range(3):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(reference, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)

    def voltage(raw_cm, canonical_area, canonical_resistance):
        parameters = {**tensors.parameters, CM_RAW: raw_cm}
        constants = {
            **tensors.constants,
            AREA: canonical_area,
            RESISTANCE: canonical_resistance,
        }
        next_state, _auxiliary = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )
        return next_state["integrator"]["v"]

    arguments = (
        tensors.parameters[CM_RAW],
        tensors.constants[AREA],
        tensors.constants[RESISTANCE],
    )
    reverse = torch.func.jacrev(voltage, argnums=(0, 1, 2))(*arguments)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(voltage, argnums=(0, 1, 2))(*arguments)
    for actual, expected in zip(reverse, forward, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    assert torch.count_nonzero(reverse[0]) > 0
    assert torch.count_nonzero(reverse[1]) > 0
    assert torch.count_nonzero(reverse[2]) == 0


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_scalar_tree_host_runners_match_direct_rollout(runner_name):
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
    state["integrator"]["v"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()
    targets = (
        parameters[PAS_G],
        parameters[CM_RAW],
        parameters[RHOA_SCALE_RAW],
        state["integrator"]["v"],
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
    loss = final["integrator"]["v"].square().mean()
    loss = loss + 0.01 * auxiliary["v"].sin().mean()
    gradients = torch.autograd.grad(loss, targets)
    return final, auxiliary, loss, gradients


def test_scalar_tree_checkpointed_runner_matches_ordinary_bptt():
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
    for actual, expected in zip(checkpointed[3], ordinary[3], strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


def test_scalar_tree_exact_area_and_resistance_jacrev_jacfwd_semantics():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=1)
    assert torch.equal(tensors.constants[AREA], model.area)
    assert torch.equal(tensors.constants[RESISTANCE], model.edge_resistance_ohm)

    def voltage(canonical_area, canonical_resistance):
        constants = {
            **tensors.constants,
            AREA: canonical_area,
            RESISTANCE: canonical_resistance,
        }
        state, _auxiliary = functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )
        return state["integrator"]["v"]

    arguments = (tensors.constants[AREA], tensors.constants[RESISTANCE])
    reverse = torch.func.jacrev(voltage, argnums=(0, 1))(*arguments)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(voltage, argnums=(0, 1))(*arguments)

    for actual, expected in zip(reverse, forward, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)
    assert torch.count_nonzero(reverse[0][..., 1:]) > 0
    # The root has no parent edge, while every non-root resistance participates
    # in the child-indexed Hines conductance plane.
    assert torch.count_nonzero(reverse[1][..., 0]) == 0
    assert torch.count_nonzero(reverse[1][..., 1:]) > 0


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


def test_scalar_tree_jacrev_jacfwd_and_vmap_compose():
    model, response, raw_rhoa_scale, intra = _transform_case()

    reverse = torch.func.jacrev(response, argnums=0)(raw_rhoa_scale, intra)
    forward = torch.func.jacfwd(response, argnums=0)(raw_rhoa_scale, intra)
    assert torch.isfinite(reverse).all()
    assert torch.count_nonzero(reverse) > 0
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


def test_scalar_tree_nested_explicit_batch_and_nested_empty_vmap_compose():
    model, response, raw_rhoa_scale, intra = _transform_case(batch_calls=(2, 3))
    assert model.shape == (3, 2, 2, 9)

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

    empty = torch.vmap(response)(
        raw_rhoa_scale.new_empty((0,)),
        intra.new_empty((0, *model.shape)),
    )
    assert empty.shape == (0, *model.shape)


def test_scalar_tree_compile_composes_with_jacrev():
    _model_value, response, raw_rhoa_scale, intra = _transform_case()
    transformed = torch.func.jacrev(response, argnums=0)
    expected = transformed(raw_rhoa_scale, intra)

    with torch_compiler_warning_context():
        compiled = torch.compile(transformed, backend="eager", fullgraph=True)
        actual = compiled(raw_rhoa_scale, intra)

    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_scalar_tree_compiled_chunk_uses_one_bounded_native_dhs_loop():
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
        and node.target is torch.ops.dendra_solvers.dhs_solve.default
        for graph_module in graph_modules
        for node in graph_module.graph.nodes
    )


def test_scalar_tree_explicit_batch_commit_and_imperative_resume_are_safe():
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
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="structure does not match",
        ):
            functional.commit_state_(incompatible, checkpoint)
        torch.testing.assert_close(incompatible.v, voltage_before)


def test_scalar_tree_low_level_noncanonical_construction_fails_closed():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Tree(
            2,
            4,
            graph=_reordered_root_graph(),
            dtype=DTYPE,
            v_init=-64.0,
            integrator=dn.dhs(threads=2, imem=False),
        )
        model.insert(pas, g=4.0e-4, e=-71.0)
        model.initialize()
        model.train()

    assert model.compartment_graph is None
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="current supported topologies",
    ):
        dn.func.make_functional(model, dt=DT)


def test_scalar_tree_imem_fails_closed():
    model = _model(imem=True)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="i_membrane recording is not supported yet",
    ):
        dn.func.make_functional(model, dt=DT)


def test_scalar_tree_custom_exact_geometry_override_fails_closed():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = _AreaOverrideTree.from_morphology(
            _morphology(),
            N=2,
            dtype=DTYPE,
            v_init=-64.0,
            integrator=dn.dhs(threads=2, imem=False),
        )
        model.insert(pas, g=4.0e-4, e=-71.0)
        model.initialize()
        model.train()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="standard exact area",
    ):
        dn.func.make_functional(model, dt=DT)


def test_real_l4_tree_matches_one_imperative_step_when_models_are_installed():
    cortical = pytest.importorskip("dendra_models.models.cells.cortical")

    def build():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE):
            model = cortical.L4_NBC_dNAC(
                8989,
                N=1,
                integrator=dn.dhs(threads=2, imem=False),
            )
            model.initialize()
            model.train()
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
