from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _population as functional_population_module
from dendra.models.mod import hh

DT = 0.01

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model(*, dtype=torch.float64, method="pcr", jit=0):
    with dn.ctx(JIT=jit, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -60.0, -56.0, -59.0, -63.0]),
            dtype=dtype,
            integrator=dn.bwd_euler_ub(method=method, imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    return model


def _drives(model, steps):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -2.0,
        2.0,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.0e-9,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _imperative_step(model, ve=None, intra=None):
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _assert_state_matches_model(state, model, *, rtol=0.0, atol=0.0):
    torch.testing.assert_close(state["integrator"]["v"], model.v, rtol=rtol, atol=atol)
    for name in ("m", "h", "n"):
        torch.testing.assert_close(
            state["mechanisms"]["hh"][name],
            getattr(model.mech.hh, name),
            rtol=rtol,
            atol=atol,
        )
    torch.testing.assert_close(state["clock"]["t"], model.t, rtol=0.0, atol=0.0)


def _assert_every_state_leaf_close(actual, expected, *, rtol=0.0, atol=0.0):
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
            msg=lambda message: f"state leaf {actual_path}: {message}",
        )


def _while_loop_nodes(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.higher_order.while_loop
    ]


def _tensor_snapshot(tensor):
    return (
        id(tensor),
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tensor.detach().clone(),
    )


def _source_snapshot(model):
    handler = model.mech
    integrator = model.integrator
    return {
        "parameters": {
            name: _tensor_snapshot(value)
            for name, value in model.named_parameters(remove_duplicate=False)
        },
        "buffers": {
            name: _tensor_snapshot(value)
            for name, value in model.named_buffers(remove_duplicate=False)
        },
        "buf_i_list": id(handler._buf_i),
        "buf_i": tuple(_tensor_snapshot(value) for value in handler._buf_i),
        "buf_g_list": id(handler._buf_g),
        "buf_g": tuple(_tensor_snapshot(value) for value in handler._buf_g),
        "workspace": {
            name: _tensor_snapshot(getattr(integrator, name))
            for name in (
                "diag_base",
                "lower",
                "upper",
                "g_edge_Cinv",
                "g_edge_Cinv_right",
                "cm_inv",
                "scale",
            )
        },
        "integrator_initialized": integrator.initialized,
        "integrator_dt": integrator.dt,
        "integrator_shape": integrator.shape,
        "compiled_kernels": dict(integrator._compiled_kernels),
        "population_caches": dict(model._caches),
        "training": tuple(
            (name, module.training) for name, module in model.named_modules()
        ),
    }


def _assert_tensor_snapshot(value, snapshot):
    identity, storage, version, expected = snapshot
    assert id(value) == identity
    assert value.untyped_storage().data_ptr() == storage
    assert value._version == version
    assert torch.equal(value, expected)


def _assert_source_unchanged(model, snapshot):
    for name, value in model.named_parameters(remove_duplicate=False):
        _assert_tensor_snapshot(value, snapshot["parameters"][name])
    for name, value in model.named_buffers(remove_duplicate=False):
        _assert_tensor_snapshot(value, snapshot["buffers"][name])

    assert id(model.mech._buf_i) == snapshot["buf_i_list"]
    assert id(model.mech._buf_g) == snapshot["buf_g_list"]
    for value, expected in zip(model.mech._buf_i, snapshot["buf_i"], strict=True):
        _assert_tensor_snapshot(value, expected)
    for value, expected in zip(model.mech._buf_g, snapshot["buf_g"], strict=True):
        _assert_tensor_snapshot(value, expected)
    for name, expected in snapshot["workspace"].items():
        _assert_tensor_snapshot(getattr(model.integrator, name), expected)
    assert model.integrator.initialized == snapshot["integrator_initialized"]
    assert model.integrator.dt == snapshot["integrator_dt"]
    assert model.integrator.shape == snapshot["integrator_shape"]
    assert model.integrator._compiled_kernels == snapshot["compiled_kernels"]
    assert model._caches == snapshot["population_caches"]
    assert (
        tuple((name, module.training) for name, module in model.named_modules())
        == snapshot["training"]
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 2.0e-5, 2.0e-6),
        (torch.float64, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("drive_kind", ["none", "ve", "intra", "both"])
def test_functional_step_matches_complete_imperative_state(
    dtype, rtol, atol, drive_kind
):
    functional_model = _model(dtype=dtype)
    imperative_model = _model(dtype=dtype)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = (value[0] for value in _drives(functional_model, 1))
    if drive_kind not in {"ve", "both"}:
        ve = None
    if drive_kind not in {"intra", "both"}:
        intra = None

    actual, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(ve=ve, intra=intra),
    )
    _imperative_step(imperative_model, ve=ve, intra=intra)

    _assert_state_matches_model(actual, imperative_model, rtol=rtol, atol=atol)


def test_functional_rollout_matches_imperative_at_every_boundary():
    functional_model = _model()
    imperative_model = _model()
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(functional_model, 6)
    state = tensors.state

    for index in range(6):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(imperative_model, ve[index], intra[index])
        _assert_state_matches_model(state, imperative_model)

    rolled, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_state_matches_model(rolled, imperative_model)


def test_functional_calls_and_transforms_do_not_mutate_source_or_input_state():
    model = _model(method="pcr")
    source_snapshot = _source_snapshot(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    _assert_source_unchanged(model, source_snapshot)
    execution_snapshot = _source_snapshot(functional._transition.population)
    ve, intra = _drives(model, 2)
    ve.requires_grad_()
    intra.requires_grad_()
    explicit_tensors = (
        *tensors.parameters.values(),
        *tensors.constants.values(),
        *torch.utils._pytree.tree_leaves(prepared.values),
        *torch.utils._pytree.tree_leaves(tensors.state),
        ve,
        intra,
    )
    explicit_snapshots = tuple(_tensor_snapshot(value) for value in explicit_tensors)

    step_input = dn.func.StepInput(ve=ve[0], intra=intra[0])
    first, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        step_input,
    )
    second, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        step_input,
    )
    for left, right in zip(
        torch.utils._pytree.tree_leaves(first),
        torch.utils._pytree.tree_leaves(second),
        strict=True,
    ):
        assert torch.equal(left, right)

    torch.func.jacrev(
        lambda state: functional.step(tensors.parameters, prepared, state)[0][
            "integrator"
        ]["v"]
    )(tensors.state)
    rolled, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"
    torch.autograd.grad(
        rolled["integrator"]["v"].square().mean(),
        (tensors.parameters[parameter_name], ve, intra),
    )

    _assert_source_unchanged(model, source_snapshot)
    _assert_source_unchanged(functional._transition.population, execution_snapshot)
    for value, expected in zip(explicit_tensors, explicit_snapshots, strict=True):
        _assert_tensor_snapshot(value, expected)


def test_vmap_over_independent_states_and_inputs_matches_explicit_lanes():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, _intra = _drives(model, 3)
    lane_offsets = torch.tensor([-0.3, 0.0, 0.25], dtype=model.dtype())
    states = torch.utils._pytree.tree_map(
        lambda value: torch.stack(
            [
                value + offset if value.dtype.is_floating_point else value
                for offset in lane_offsets
            ]
        ),
        tensors.state,
    )

    def run_lane(state, lane_ve):
        return functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=lane_ve),
        )[0]

    actual = torch.vmap(run_lane)(states, ve)
    expected = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(
            run_lane(
                torch.utils._pytree.tree_map(lambda value: value[index], states),
                ve[index],
            )
            for index in range(3)
        ),
    )
    for got, want in zip(
        torch.utils._pytree.tree_leaves(actual),
        torch.utils._pytree.tree_leaves(expected),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=2.0e-10, atol=2.0e-11)


def test_vmap_accepts_an_empty_population_lane_batch():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    states = torch.utils._pytree.tree_map(
        lambda value: value.new_empty((0, *value.shape)),
        tensors.state,
    )
    ve = model.v.new_empty((0, *model.shape))

    result = torch.vmap(
        lambda state, lane_ve: functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=lane_ve),
        )[0]
    )(states, ve)

    for result_leaf, state_leaf in zip(
        torch.utils._pytree.tree_leaves(result),
        torch.utils._pytree.tree_leaves(tensors.state),
        strict=True,
    ):
        assert result_leaf.shape == (0, *state_leaf.shape)


def test_vmap_over_parameters_with_shared_state_matches_explicit_lanes():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"
    base = tensors.parameters[parameter_name]
    lanes = torch.stack((0.8 * base, base, 1.2 * base))

    def run_lane(gnabar):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = gnabar
        prepared = functional.prepare(parameters, tensors.constants)
        return functional.step(parameters, prepared, tensors.state)[0]["integrator"][
            "v"
        ]

    actual = torch.vmap(run_lane)(lanes)
    expected = torch.stack(tuple(run_lane(value) for value in lanes))
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    empty = torch.vmap(run_lane)(base.new_empty((0, *base.shape)))
    assert empty.shape == (0, *model.v.shape)


def test_raw_parameter_gradients_and_hessian_are_finite_and_nonzero():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"

    def loss(gnabar):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = gnabar
        prepared = functional.prepare(parameters, tensors.constants)
        state, _aux = functional.rollout(
            parameters,
            prepared,
            tensors.state,
            steps=3,
        )
        return state["integrator"]["v"].square().mean()

    gnabar = tensors.parameters[parameter_name]
    gradient = torch.func.grad(loss)(gnabar)
    hessian = torch.func.jacrev(torch.func.grad(loss))(gnabar)

    assert torch.isfinite(gradient)
    assert torch.isfinite(hessian)
    assert gradient.abs() > 0
    assert hessian.abs() > 0


def test_functional_step_is_independent_of_dendra_jit_and_compiles_fullgraph():
    model = _model(method="pcr", jit=1)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def voltage(state):
        return functional.step(tensors.parameters, prepared, state)[0]["integrator"][
            "v"
        ]

    def atomic_voltage(state):
        return functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )[0]["integrator"]["v"]

    compiled = torch.compile(atomic_voltage, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(tensors.state)
    expected = voltage(tensors.state)

    torch.testing.assert_close(actual, expected)


def test_native_transformable_solver_runs_real_hh_jacrev_when_available():
    try:
        from dendra_solvers import thomas_solve_t
    except ImportError:
        pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    model = _model(method="thomas")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert functional._transition.solver is thomas_solve_t
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    jacobian = torch.func.jacrev(
        lambda state: functional.step(tensors.parameters, prepared, state)[0][
            "integrator"
        ]["v"]
    )(tensors.state)

    assert jacobian["integrator"]["v"].shape == (*model.shape, *model.shape)
    assert torch.isfinite(jacobian["integrator"]["v"]).all()


def test_native_thomas_matches_imperative_and_higher_order_pcr_oracle():
    try:
        from dendra_solvers import thomas_solve_t
    except ImportError:
        pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    native_model = _model(method="thomas")
    imperative_model = _model(method="thomas")
    native, native_tensors = dn.func.make_functional(native_model, dt=DT)
    assert native._transition.solver is thomas_solve_t
    native_prepared = native.prepare(
        native_tensors.parameters,
        native_tensors.constants,
    )
    ve, intra = _drives(native_model, 4)
    native_state, _aux = native.rollout(
        native_tensors.parameters,
        native_prepared,
        native_tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    for index in range(4):
        _imperative_step(imperative_model, ve[index], intra[index])
    _assert_state_matches_model(native_state, imperative_model)

    pcr_model = _model(method="pcr")
    pcr, pcr_tensors = dn.func.make_functional(pcr_model, dt=DT)
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"

    def make_loss(functional, tensors):
        def loss(gnabar):
            parameters = dict(tensors.parameters)
            parameters[parameter_name] = gnabar
            prepared = functional.prepare(parameters, tensors.constants)
            state, _aux = functional.rollout(
                parameters,
                prepared,
                tensors.state,
                steps=3,
            )
            return state["integrator"]["v"].square().mean()

        return loss

    native_loss = make_loss(native, native_tensors)
    pcr_loss = make_loss(pcr, pcr_tensors)
    gnabar = native_tensors.parameters[parameter_name]
    native_hessian = torch.func.hessian(native_loss)(gnabar)
    reverse_hessian = torch.func.jacrev(torch.func.grad(native_loss))(gnabar)
    pcr_hessian = torch.func.hessian(pcr_loss)(pcr_tensors.parameters[parameter_name])
    torch.testing.assert_close(
        native_hessian, reverse_hessian, rtol=2.0e-9, atol=2.0e-10
    )
    torch.testing.assert_close(native_hessian, pcr_hessian, rtol=2.0e-9, atol=2.0e-10)

    eps = 1.0e-4
    finite_difference = (
        torch.func.grad(native_loss)(gnabar + eps)
        - torch.func.grad(native_loss)(gnabar - eps)
    ) / (2.0 * eps)
    torch.testing.assert_close(
        native_hessian,
        finite_difference,
        rtol=2.0e-6,
        atol=2.0e-8,
    )


def test_functional_gradients_match_imperative_parameter_state_and_drive_oracle():
    functional_model = _model(method="pcr")
    imperative_model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)

    functional_state = copy.deepcopy(tensors.state)
    functional_state["integrator"]["v"].requires_grad_()
    for name in ("m", "h", "n"):
        functional_state["mechanisms"]["hh"][name].requires_grad_()

    imperative_v0 = functional_state["integrator"]["v"].detach().clone()
    imperative_v0.requires_grad_()
    imperative_model.v = imperative_v0
    imperative_gate0 = {}
    for name in ("m", "h", "n"):
        value = functional_state["mechanisms"]["hh"][name].detach().clone()
        value.requires_grad_()
        imperative_gate0[name] = value
        imperative_model.mech.hh._buffers[name] = value

    ve, intra = _drives(functional_model, 3)
    functional_ve = ve.detach().clone().requires_grad_()
    functional_intra = intra.detach().clone().requires_grad_()
    imperative_ve = ve.detach().clone().requires_grad_()
    imperative_intra = intra.detach().clone().requires_grad_()

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional_result, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        functional_state,
        dn.func.RolloutInput(ve=functional_ve, intra=functional_intra),
    )
    functional_loss = functional_result["integrator"]["v"].square().mean()

    dt = torch.as_tensor(DT, dtype=imperative_model.dtype())
    imperative_model.integrator._initialize(
        imperative_model,
        dt,
        force=True,
        compile_scope="population",
    )
    for index in range(3):
        imperative_model.integrator.step(
            imperative_model,
            dt,
            imperative_ve[index],
            imperative_intra[index],
        )
        imperative_model.t = imperative_model.t + dt
    imperative_loss = imperative_model.v.square().mean()

    functional_parameters = tuple(tensors.parameters.values())
    imperative_parameters = tuple(dict(imperative_model.named_parameters()).values())
    assert tuple(tensors.parameters) == tuple(dict(imperative_model.named_parameters()))
    functional_state_targets = (
        functional_state["integrator"]["v"],
        *(functional_state["mechanisms"]["hh"][name] for name in ("m", "h", "n")),
    )
    imperative_initial_state = (
        imperative_v0,
        *(imperative_gate0[name] for name in ("m", "h", "n")),
    )
    functional_targets = (
        *functional_parameters,
        *functional_state_targets,
        functional_ve,
        functional_intra,
    )
    imperative_targets = (
        *imperative_parameters,
        *imperative_initial_state,
        imperative_ve,
        imperative_intra,
    )
    functional_gradients = torch.autograd.grad(
        functional_loss,
        functional_targets,
        allow_unused=True,
    )
    imperative_gradients = torch.autograd.grad(
        imperative_loss,
        imperative_targets,
        allow_unused=True,
    )

    torch.testing.assert_close(functional_loss, imperative_loss, rtol=0.0, atol=0.0)
    for actual, expected in zip(
        functional_gradients,
        imperative_gradients,
        strict=True,
    ):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)


@pytest.mark.parametrize("method", ["pcr", "thomas"])
def test_multistep_rollout_aot_compiled_backward_matches_eager(method):
    if method == "thomas":
        try:
            from dendra_solvers import thomas_solve_t
        except ImportError:
            pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    model = _model(method=method)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    if method == "thomas":
        assert functional._transition.solver is thomas_solve_t
    ve, intra = _drives(model, 3)
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"

    def loss(gnabar):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = gnabar
        state, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        return state["integrator"]["v"].square().mean()

    gnabar = tensors.parameters[parameter_name]
    expected = torch.autograd.grad(loss(gnabar), gnabar)[0]
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    with torch_compiler_warning_context():
        compiled_loss = compiled(gnabar)
    actual = torch.autograd.grad(compiled_loss, gnabar)[0]
    torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)


def test_multistep_fullgraph_is_reused_for_stable_shapes():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 3)
    compile_count = 0

    def backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def voltage(state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]["integrator"]["v"]

    compiled = torch.compile(voltage, backend=backend, fullgraph=True)
    with torch_compiler_warning_context():
        first = compiled(tensors.state, ve, intra)
        second = compiled(tensors.state, ve + 0.1, intra)
    torch.testing.assert_close(first, voltage(tensors.state, ve, intra))
    torch.testing.assert_close(second, voltage(tensors.state, ve + 0.1, intra))
    assert compile_count == 1


def test_structured_step_capture_is_lazy_and_signature_specific(monkeypatch):
    make_fx_calls = []
    original_make_fx = functional_population_module.make_fx

    def counting_make_fx(*args, **kwargs):
        make_fx_calls.append(None)
        return original_make_fx(*args, **kwargs)

    monkeypatch.setattr(functional_population_module, "make_fx", counting_make_fx)
    functional, _tensors = dn.func.make_functional(_model(method="pcr"), dt=DT)

    assert make_fx_calls == []
    assert functional._structured_step_graphs == {}
    assert functional.prewarm_structured_rollout(ve=True, intra=False)
    assert len(make_fx_calls) == 1
    assert set(functional._structured_step_graphs) == {(True, False)}

    # Reusing one signature is free; requesting another captures only that one.
    assert functional.prewarm_structured_rollout(ve=True, intra=False)
    assert len(make_fx_calls) == 1
    assert functional.prewarm_structured_rollout(ve=False, intra=True)
    assert len(make_fx_calls) == 2
    assert set(functional._structured_step_graphs) == {
        (True, False),
        (False, True),
    }


def test_structured_capture_failure_is_cached_per_signature(monkeypatch):
    original_make_fx = functional_population_module.make_fx
    make_fx_calls = 0

    def fail_first_make_fx(*args, **kwargs):
        nonlocal make_fx_calls
        make_fx_calls += 1
        if make_fx_calls == 1:
            raise RuntimeError("authored step is not traceable")
        return original_make_fx(*args, **kwargs)

    monkeypatch.setattr(functional_population_module, "make_fx", fail_first_make_fx)
    functional, _tensors = dn.func.make_functional(_model(method="pcr"), dt=DT)

    assert not functional.prewarm_structured_rollout(ve=False, intra=False)
    assert not functional.prewarm_structured_rollout(ve=False, intra=False)
    assert make_fx_calls == 1
    assert (False, False) in functional._structured_capture_errors

    assert functional.prewarm_structured_rollout(ve=True, intra=True)
    assert make_fx_calls == 2
    assert set(functional._structured_step_graphs) == {(True, True)}


def test_no_grad_fullgraph_without_prewarm_uses_correct_unrolled_fallback():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 5)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def atomic(state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]

    expected = atomic(tensors.state, ve, intra)
    compiled = torch.compile(atomic, backend=backend, fullgraph=True)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = compiled(tensors.state, ve, intra)

    _assert_every_state_leaf_close(actual, expected)
    assert functional._structured_step_graphs == {}
    assert len(graphs) == 1
    assert not _while_loop_nodes(graphs[0])


@pytest.mark.parametrize("drive_kind", ["none", "ve", "intra", "both"])
def test_no_grad_compiled_rollout_uses_structured_loop_for_optional_drives(
    drive_kind,
):
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 5)
    selected_ve = ve if drive_kind in {"ve", "both"} else None
    selected_intra = intra if drive_kind in {"intra", "both"} else None
    signature = (selected_ve is not None, selected_intra is not None)
    assert not functional._structured_step_graphs
    assert functional.prewarm_structured_rollout(
        ve=signature[0],
        intra=signature[1],
    )
    assert set(functional._structured_step_graphs) == {signature}
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def atomic(state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
            steps=5,
        )[0]

    expected = atomic(tensors.state, selected_ve, selected_intra)
    compiled = torch.compile(atomic, backend=backend, fullgraph=True)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = compiled(tensors.state, selected_ve, selected_intra)

    _assert_every_state_leaf_close(actual, expected)
    assert len(graphs) == 1
    assert len(_while_loop_nodes(graphs[0])) == 1


@pytest.mark.parametrize("state_kind", ["noncontiguous", "reversed"])
def test_no_grad_structured_rollout_canonicalizes_state_carry(state_kind):
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert functional.prewarm_structured_rollout(ve=True, intra=True)
    ve, intra = _drives(model, 5)
    if state_kind == "noncontiguous":
        state = torch.utils._pytree.tree_map(
            lambda value: (
                value.transpose(0, 1).contiguous().transpose(0, 1)
                if value.ndim == 2
                else value.clone()
            ),
            tensors.state,
        )
        assert any(
            not value.is_contiguous()
            for value in torch.utils._pytree.tree_leaves(state)
        )
    else:

        def reverse_mapping_order(value):
            if isinstance(value, dict):
                return {
                    key: reverse_mapping_order(item)
                    for key, item in reversed(value.items())
                }
            return value.clone()

        state = reverse_mapping_order(tensors.state)
        assert tuple(state) == tuple(reversed(tensors.state))
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def atomic(current, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            current,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]

    expected = atomic(state, ve, intra)
    compiled = torch.compile(atomic, backend=backend, fullgraph=True)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = compiled(state, ve, intra)

    _assert_every_state_leaf_close(actual, expected)
    assert len(graphs) == 1
    assert len(_while_loop_nodes(graphs[0])) == 1


def test_structured_rollout_graph_size_is_bounded_by_block_not_step_count():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert functional.prewarm_structured_rollout(ve=True, intra=True)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def atomic(state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]

    compiled = torch.compile(
        atomic,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    cases = tuple((steps, *_drives(model, steps)) for steps in (8, 68))
    with torch.no_grad(), torch_compiler_warning_context():
        for _repeat in range(2):
            for _steps, ve, intra in cases:
                actual = compiled(tensors.state, ve, intra)
                expected = atomic(tensors.state, ve, intra)
                _assert_every_state_leaf_close(actual, expected)

    # Each time-axis shape needs one specialized graph, but repeated calls do
    # not recompile and neither graph contains a timestep-sized unrolled body.
    assert len(graphs) == len(cases)
    assert all(len(_while_loop_nodes(graph)) == 1 for graph in graphs)
    recursive_sizes = [
        tuple(
            (name, len(tuple(module.graph.nodes)))
            for name, module in graph.named_modules()
            if isinstance(module, torch.fx.GraphModule)
        )
        for graph in graphs
    ]
    assert recursive_sizes[0] == recursive_sizes[1]


def test_no_grad_compiled_zero_step_rollout_preserves_every_state_leaf():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    empty_ve, empty_intra = _drives(model, 0)

    def without_drives(state):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            steps=0,
        )[0]

    def with_empty_drives(state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]

    compiled_without_drives = torch.compile(
        without_drives,
        backend="eager",
        fullgraph=True,
    )
    compiled_with_empty_drives = torch.compile(
        with_empty_drives,
        backend="eager",
        fullgraph=True,
    )
    with torch.no_grad(), torch_compiler_warning_context():
        without = compiled_without_drives(tensors.state)
        with_empty = compiled_with_empty_drives(
            tensors.state,
            empty_ve,
            empty_intra,
        )

    _assert_every_state_leaf_close(without, tensors.state)
    _assert_every_state_leaf_close(with_empty, tensors.state)


def test_no_grad_compiled_vmap_retains_unrolled_transform_fallback():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, _intra = _drives(model, 3)
    offsets = model.v.new_tensor((-0.25, 0.125))
    states = torch.utils._pytree.tree_map(
        lambda value: torch.stack((value, value)),
        tensors.state,
    )
    states["integrator"]["v"] = states["integrator"]["v"] + offsets.reshape(
        2, *([1] * model.v.ndim)
    )
    lane_ve = torch.stack((ve, ve + 0.2))
    graphs = []

    def run_lane(state, ve_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values),
        )[0]

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    vmapped = torch.vmap(run_lane)
    compiled = torch.compile(vmapped, backend=backend, fullgraph=True)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = compiled(states, lane_ve)
        expected = torch.utils._pytree.tree_map(
            lambda *values: torch.stack(values),
            *(
                run_lane(
                    torch.utils._pytree.tree_map(
                        lambda value: value[index],
                        states,
                    ),
                    lane_ve[index],
                )
                for index in range(2)
            ),
        )

    _assert_every_state_leaf_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    assert graphs
    assert all(not _while_loop_nodes(graph) for graph in graphs)

    empty_states = torch.utils._pytree.tree_map(
        lambda value: value.new_empty((0, *value.shape)),
        tensors.state,
    )
    empty_ve = ve.new_empty((0, *ve.shape))
    with torch.no_grad():
        empty_result = vmapped(empty_states, empty_ve)
    for result_leaf, state_leaf in zip(
        torch.utils._pytree.tree_leaves(empty_result),
        torch.utils._pytree.tree_leaves(tensors.state),
        strict=True,
    ):
        assert result_leaf.shape == (0, *state_leaf.shape)


def test_grad_enabled_compile_retains_differentiable_unrolled_rollout():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"
    graphs = []

    def loss(gnabar):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = gnabar
        state, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=5,
        )
        return state["integrator"]["v"].square().mean()

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    gnabar = tensors.parameters[parameter_name]
    expected = torch.autograd.grad(loss(gnabar), gnabar)[0]
    compiled = torch.compile(loss, backend=backend, fullgraph=True)
    with torch_compiler_warning_context():
        compiled_loss = compiled(gnabar)
    actual = torch.autograd.grad(compiled_loss, gnabar)[0]

    torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)
    assert graphs
    assert all(not _while_loop_nodes(graph) for graph in graphs)


def test_no_grad_direct_forward_ad_retains_differentiable_unrolled_rollout():
    model = _model(method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"
    graphs = []

    def final_voltage(gnabar):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = gnabar
        state, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=5,
        )
        return state["integrator"]["v"]

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    primal = tensors.parameters[parameter_name].detach()
    tangent = torch.full_like(primal, 0.125)
    compiled = torch.compile(final_voltage, backend=backend, fullgraph=True)
    with (
        torch.no_grad(),
        torch.autograd.forward_ad.dual_level(),
        torch_compiler_warning_context(),
    ):
        gnabar = torch.autograd.forward_ad.make_dual(primal, tangent)
        expected = torch.autograd.forward_ad.unpack_dual(final_voltage(gnabar))
        actual = torch.autograd.forward_ad.unpack_dual(compiled(gnabar))

    torch.testing.assert_close(actual.primal, expected.primal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.tangent,
        expected.tangent,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    assert graphs
    assert all(not _while_loop_nodes(graph) for graph in graphs)


def test_no_grad_structured_rollout_supports_native_thomas_solver():
    try:
        from dendra_solvers import thomas_solve_t
    except ImportError:
        pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    model = _model(method="thomas")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert functional._transition.solver is thomas_solve_t
    assert functional.prewarm_structured_rollout(ve=True, intra=True)
    ve, intra = _drives(model, 5)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def atomic(state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]

    expected = atomic(tensors.state, ve, intra)
    compiled = torch.compile(atomic, backend=backend, fullgraph=True)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = compiled(tensors.state, ve, intra)

    _assert_every_state_leaf_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    assert len(graphs) == 1
    assert len(_while_loop_nodes(graphs[0])) == 1


@pytest.mark.parametrize("training", [False, True])
def test_commit_then_imperative_resume_matches_uninterrupted_execution(training):
    source = _model(method="pcr")
    target = _model(method="pcr")
    reference = _model(method="pcr")
    source.train(training)
    target.train(training)
    reference.train(training)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, 5)

    checkpoint, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:3], intra=intra[:3]),
    )
    functional.commit_state_(target, checkpoint)
    for index in range(3, 5):
        _imperative_step(target, ve[index], intra[index])
    for index in range(5):
        _imperative_step(reference, ve[index], intra[index])

    _assert_state_matches_model(
        {
            "integrator": {"v": target.v},
            "mechanisms": {
                "hh": {name: getattr(target.mech.hh, name) for name in ("m", "h", "n")}
            },
            "clock": {"t": target.t},
        },
        reference,
    )
