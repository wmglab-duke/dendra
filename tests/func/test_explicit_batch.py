"""Functional Population contracts for explicit ``Population.batch()`` axes."""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import hh
from dendra.units import nA

DT = 0.01
STEPS = 3
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _SingletonBufferMechanism(Mechanism):
    """Exercise local mutable carry over the complete runtime batch shape."""

    Mechanism.CARRY("gain")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        del v
        return {"gain": torch.ones_like(values["gain"])}

    def advance(self, v, dt, values):
        del v, dt
        return {"gain": values["gain"] + 1.0}

    def i(self, v):
        return self.gain * v

    def i_with_conductance(self, v):
        return self.i(v), self.gain.expand_as(v)


class _ShapeDriftingBufferMechanism(Mechanism):
    """Deliberately violate the initialized CARRY storage contract."""

    Mechanism.CARRY("gain")

    def initial_values(self, v, values):
        del v
        return {"gain": torch.ones_like(values["gain"])}

    def advance(self, v, dt, values):
        del dt, values
        return {"gain": torch.ones_like(v[0])}


def _model(*, batch_calls=(2,), method="pcr"):
    """Construct one deterministic HH cable with explicit batch axes.

    Each call to ``Population.batch`` prepends an axis, so ``batch_calls=(3, 2)``
    produces state shape ``(2, 3, N, C)``.
    """
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -60.0, -56.0, -59.0, -63.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method=method, imem=False),
        )
        model.insert(hh)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _buffer_model(*, batch_calls=(3, 2), mechanism=_SingletonBufferMechanism):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0, 1.5],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(mechanism)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _drives(model, steps=STEPS):
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


def _initialize_imperative_integrator(model):
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    return dt


def _imperative_step(model, dt, ve=None, intra=None):
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _clone_state(state):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), state)


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
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


@pytest.mark.parametrize(
    ("batch_calls", "expected_batch_shape"),
    [
        ((1,), (1,)),
        ((3,), (3,)),
        ((3, 2), (2, 3)),
    ],
)
def test_explicit_batch_keeps_geometry_shared_and_solver_workspaces_flattened(
    batch_calls,
    expected_batch_shape,
):
    model = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    expected_state_shape = (*expected_batch_shape, *model.core_shape())
    expected_parameter_shape = (
        *((1,) * len(expected_batch_shape)),
        *model.core_shape(),
    )
    expected_solve_shape = (
        math.prod(expected_state_shape[:-1]),
        expected_state_shape[-1],
    )

    assert functional.shape == expected_state_shape
    assert tensors.state["integrator"]["v"].shape == expected_state_shape
    for value in tensors.state["mechanisms"]["hh"].values():
        assert value.shape == expected_state_shape
    assert tensors.state["clock"]["t"].shape == ()
    assert tensors.state["control"]["duration_remainder"].shape == ()

    # Morphology and raw parameters are shared by every explicit replica.
    assert tensors.constants["diam"].shape == model.core_shape()
    assert tensors.constants["dx"].shape == model.core_shape()
    assert prepared.values["population"]["cm"].shape == expected_parameter_shape
    assert prepared.values["population"]["rhoa"].shape == expected_parameter_shape

    solver = prepared.values["integrator"]
    assert solver["diag_base"].shape == expected_solve_shape
    assert solver["cm_inv"].shape == expected_solve_shape
    assert solver["scale"].shape == expected_solve_shape
    expected_edge_shape = (expected_solve_shape[0], expected_solve_shape[1] - 1)
    for name in ("lower", "upper", "g_edge_Cinv", "g_edge_Cinv_right"):
        assert solver[name].shape == expected_edge_shape


@pytest.mark.parametrize("constant_name", ["diam", "dx"])
def test_lane_specific_full_shape_geometry_is_rejected(constant_name):
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    constants = dict(tensors.constants)
    core_value = constants[constant_name]
    full_value = core_value.unsqueeze(0).expand(model.shape).clone()
    full_value[1] = full_value[1] + 0.125
    constants[constant_name] = full_value

    with pytest.raises(ValueError, match="shared geometry shape"):
        functional.prepare(tensors.parameters, constants)


def test_batching_the_source_after_lowering_invalidates_the_plan():
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    original_shape = functional.shape

    model.batch(3)
    model.initialize(force_rebuild=True)

    assert functional.shape == original_shape
    assert model.shape != original_shape
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)


def test_singleton_prefixed_mutable_buffer_remains_complete_functional_carry():
    source = _buffer_model()
    imperative = _buffer_model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    gain = tensors.state["mechanism_buffers"]["_SingletonBufferMechanism"]["gain"]
    expected_shape = source.shape
    original = gain.detach().clone()
    original_version = gain._version

    assert source.shape == (2, 3, *source.core_shape())
    assert gain.shape == expected_shape
    next_state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    next_gain = next_state["mechanism_buffers"]["_SingletonBufferMechanism"]["gain"]

    assert gain._version == original_version
    torch.testing.assert_close(gain, original)
    assert next_gain.shape == expected_shape
    assert not torch.equal(next_gain, gain)

    imperative.step(dt=DT)
    _assert_tree_close(
        next_state,
        functional.extract(imperative).state,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    def next_voltage(local_gain):
        state = {
            **tensors.state,
            "mechanism_buffers": {
                **tensors.state["mechanism_buffers"],
                "_SingletonBufferMechanism": {"gain": local_gain},
            },
        }
        return functional.step(tensors.parameters, prepared, state)[0]["integrator"][
            "v"
        ]

    jacobian = torch.func.jacrev(next_voltage)(gain)
    assert jacobian.shape == (*source.shape, *expected_shape)
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) > 0


def test_batched_mutable_buffer_cannot_change_its_initialized_schema():
    model = _buffer_model(
        batch_calls=(2,),
        mechanism=_ShapeDriftingBufferMechanism,
    )
    gain = model.mech._ShapeDriftingBufferMechanism.gain
    original_gain = gain.detach().clone()
    original_version = gain._version
    original_v = model.v.detach().clone()
    original_t = model.t.detach().clone()

    assert gain.shape == model.shape
    with pytest.raises(
        ValueError,
        match=r"advance\(\)\['gain'\] must match",
    ):
        dn.func.make_functional(model, dt=DT)

    # Admission validates canonical returned-output schemas on a disposable
    # audit clone; failure must not advance or otherwise mutate the source.
    assert model.mech._ShapeDriftingBufferMechanism.gain is gain
    assert gain._version == original_version
    assert gain.shape == model.shape
    torch.testing.assert_close(gain, original_gain)
    torch.testing.assert_close(model.v, original_v)
    torch.testing.assert_close(model.t, original_t)


def test_repeated_explicit_batch_matches_imperative_at_every_boundary_and_fused():
    functional_model = _model(batch_calls=(3, 2))
    imperative_model = _model(batch_calls=(3, 2))
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(functional_model, steps=4)
    dt = _initialize_imperative_integrator(imperative_model)
    state = tensors.state

    for index in range(4):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(imperative_model, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(imperative_model).state)

    rolled, auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_tree_close(rolled, state)
    torch.testing.assert_close(auxiliary["v"], rolled["integrator"]["v"])


def test_explicit_batch_lanes_match_independent_unbatched_models():
    batched_model = _model(batch_calls=(3,))
    functional, tensors = dn.func.make_functional(batched_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(batched_model)

    actual, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )

    references = [_model(batch_calls=()) for _index in range(3)]
    for batch_index, reference in enumerate(references):
        dt = _initialize_imperative_integrator(reference)
        for step_index in range(STEPS):
            _imperative_step(
                reference,
                dt,
                ve[step_index, batch_index],
                intra[step_index, batch_index],
            )

    torch.testing.assert_close(
        actual["integrator"]["v"],
        torch.stack(tuple(reference.v for reference in references)),
    )
    for name in ("m", "h", "n"):
        torch.testing.assert_close(
            actual["mechanisms"]["hh"][name],
            torch.stack(
                tuple(getattr(reference.mech.hh, name) for reference in references)
            ),
        )


def _run_direct_input(functional, tensors, *, entrypoint, input_name, value):
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    input_type = dn.func.StepInput if entrypoint == "step" else dn.func.RolloutInput
    inputs = input_type(**{input_name: value})
    method = functional.step if entrypoint == "step" else functional.rollout
    return method(tensors.parameters, prepared, tensors.state, inputs)[0]


@pytest.mark.parametrize("input_name", ["ve", "intra"])
@pytest.mark.parametrize("entrypoint", ["step", "rollout"])
def test_direct_inputs_prefer_trailing_spatial_axes_when_batch_equals_compartment(
    input_name,
    entrypoint,
):
    # B == C is deliberately ambiguous. A bare [C] sample retains its trailing
    # compartment meaning; [B, 1, 1] is the explicit batch-varying form.
    model = _model(batch_calls=(5,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    scale = 1.0 if input_name == "ve" else 1.0e-9
    raw = scale * model.v.new_tensor((-1.0, -0.25, 0.2, 0.7, 1.3))
    trailing = raw.reshape(1, 1, 5).expand(model.shape)
    batch_only = raw.reshape(5, 1, 1).expand(model.shape)
    if entrypoint == "rollout":
        raw = raw.unsqueeze(0)
        trailing = trailing.unsqueeze(0)
        batch_only = batch_only.unsqueeze(0)

    actual = _run_direct_input(
        functional,
        tensors,
        entrypoint=entrypoint,
        input_name=input_name,
        value=raw,
    )
    expected_trailing = _run_direct_input(
        functional,
        tensors,
        entrypoint=entrypoint,
        input_name=input_name,
        value=trailing,
    )
    explicit_batch = _run_direct_input(
        functional,
        tensors,
        entrypoint=entrypoint,
        input_name=input_name,
        value=batch_only,
    )

    _assert_tree_close(actual, expected_trailing)
    assert not torch.equal(
        actual["integrator"]["v"],
        explicit_batch["integrator"]["v"],
    )


@pytest.mark.parametrize("input_name", ["ve", "intra"])
@pytest.mark.parametrize("entrypoint", ["step", "rollout"])
def test_direct_inputs_right_align_batch_fallback_over_repeated_batch_axes(
    input_name,
    entrypoint,
):
    model = _model(batch_calls=(3, 2))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    scale = 1.0 if input_name == "ve" else 1.0e-9
    raw = scale * model.v.new_tensor((-0.75, 0.1, 1.2))
    expected_full = raw.reshape(1, 3, 1, 1).expand(model.shape)
    if entrypoint == "rollout":
        raw = raw.unsqueeze(0)
        expected_full = expected_full.unsqueeze(0)

    actual = _run_direct_input(
        functional,
        tensors,
        entrypoint=entrypoint,
        input_name=input_name,
        value=raw,
    )
    expected = _run_direct_input(
        functional,
        tensors,
        entrypoint=entrypoint,
        input_name=input_name,
        value=expected_full,
    )
    _assert_tree_close(actual, expected)


def test_explicit_batch_gradients_match_batched_imperative_oracle():
    functional_model = _model(batch_calls=(2,))
    imperative_model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)

    parameters = dict(tensors.parameters)
    raw_conductance = parameters[GNABAR].detach().clone().requires_grad_()
    parameters[GNABAR] = raw_conductance
    constants = {
        name: value.detach().clone().requires_grad_()
        for name, value in tensors.constants.items()
    }
    state = _clone_state(tensors.state)
    functional_initial_state = (
        state["integrator"]["v"],
        *(state["mechanisms"]["hh"][name] for name in ("m", "h", "n")),
    )
    for value in functional_initial_state:
        value.requires_grad_()
    ve, intra = _drives(functional_model)
    functional_ve = ve.detach().clone().requires_grad_()
    functional_intra = intra.detach().clone().requires_grad_()

    prepared = functional.prepare(parameters, constants)
    result, _auxiliary = functional.rollout(
        parameters,
        prepared,
        state,
        dn.func.RolloutInput(ve=functional_ve, intra=functional_intra),
    )
    weights = torch.linspace(
        0.5,
        1.5,
        functional_model.v.numel(),
        dtype=functional_model.dtype(),
    ).reshape(functional_model.shape)
    functional_loss = (result["integrator"]["v"] * weights).sum() + 0.03 * result[
        "mechanisms"
    ]["hh"]["m"].square().sum()
    functional_targets = (
        raw_conductance,
        constants["diam"],
        constants["dx"],
        *functional_initial_state,
        functional_ve,
        functional_intra,
    )
    functional_gradients = torch.autograd.grad(
        functional_loss,
        functional_targets,
        allow_unused=True,
    )

    imperative_parameter = dict(imperative_model.named_parameters())[GNABAR]
    imperative_model.diam.requires_grad_()
    imperative_model.dx.requires_grad_()
    imperative_initial_state = (
        imperative_model.v,
        *(getattr(imperative_model.mech.hh, name) for name in ("m", "h", "n")),
    )
    for value in imperative_initial_state:
        value.requires_grad_()
    imperative_ve = ve.detach().clone().requires_grad_()
    imperative_intra = intra.detach().clone().requires_grad_()
    dt = _initialize_imperative_integrator(imperative_model)
    for index in range(STEPS):
        _imperative_step(
            imperative_model,
            dt,
            imperative_ve[index],
            imperative_intra[index],
        )
    imperative_loss = (
        imperative_model.v * weights
    ).sum() + 0.03 * imperative_model.mech.hh.m.square().sum()
    imperative_targets = (
        imperative_parameter,
        imperative_model.diam,
        imperative_model.dx,
        *imperative_initial_state,
        imperative_ve,
        imperative_intra,
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
            assert torch.isfinite(actual).all()
            assert torch.count_nonzero(actual) > 0
            torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)


def test_outer_vmap_over_explicit_batch_and_core_geometry_matches_explicit_lanes():
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, _intra = _drives(model, steps=2)
    lane_states = torch.utils._pytree.tree_map(
        lambda value: torch.stack((value, value)),
        tensors.state,
    )
    lane_states["integrator"]["v"] = lane_states["integrator"]["v"] + (
        model.v.new_tensor((-0.2, 0.15)).reshape(2, 1, 1, 1)
    )
    lane_ve = torch.stack((ve - 0.1, ve + 0.2))
    lane_diam = torch.stack(
        (0.9 * tensors.constants["diam"], 1.1 * tensors.constants["diam"])
    )

    def run_lane(diam, state, ve_values):
        constants = dict(tensors.constants)
        constants["diam"] = diam
        return functional.prepare_and_rollout(
            tensors.parameters,
            constants,
            state,
            dn.func.RolloutInput(ve=ve_values),
        )[0]

    actual = torch.vmap(run_lane)(lane_diam, lane_states, lane_ve)
    expected = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(
            run_lane(
                lane_diam[index],
                torch.utils._pytree.tree_map(
                    lambda value: value[index],
                    lane_states,
                ),
                lane_ve[index],
            )
            for index in range(2)
        ),
    )
    _assert_tree_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_outer_vmap_accepts_zero_lanes_around_an_explicit_batch():
    model = _model(batch_calls=(2,))
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


def test_compile_of_outer_vmap_around_explicit_batch_matches_eager_lanes():
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, _intra = _drives(model, steps=2)
    states = torch.utils._pytree.tree_map(
        lambda value: torch.stack((value, value)),
        tensors.state,
    )
    lane_ve = torch.stack((ve - 0.15, ve + 0.25))

    def run_lane(state, ve_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values),
        )[0]

    vmapped = torch.vmap(run_lane)
    expected = vmapped(states, lane_ve)
    compiled = torch.compile(vmapped, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(states, lane_ve)
    _assert_tree_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_native_thomas_outer_vmap_preserves_explicit_solver_batches():
    try:
        from dendra_solvers import thomas_solve_t
    except ImportError:
        pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    model = _model(batch_calls=(2,), method="thomas")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert functional._transition.solver is thomas_solve_t
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, _intra = _drives(model, steps=1)
    states = torch.utils._pytree.tree_map(
        lambda value: torch.stack((value, value)),
        tensors.state,
    )
    lane_ve = torch.stack((ve[0] - 0.1, ve[0] + 0.2))

    def run_lane(state, ve_value):
        return functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve_value),
        )[0]

    actual = torch.vmap(run_lane)(states, lane_ve)
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
    _assert_tree_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_explicit_batch_aot_compiled_backward_matches_eager():
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=2)

    def loss(raw_conductance, diam, ve_values, intra_values):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = raw_conductance
        constants = dict(tensors.constants)
        constants["diam"] = diam
        state, _auxiliary = functional.prepare_and_rollout(
            parameters,
            constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )
        return state["integrator"]["v"].square().mean()

    eager_inputs = (
        tensors.parameters[GNABAR].detach().clone().requires_grad_(),
        tensors.constants["diam"].detach().clone().requires_grad_(),
        ve.detach().clone().requires_grad_(),
        intra.detach().clone().requires_grad_(),
    )
    eager_loss = loss(*eager_inputs)
    eager_gradients = torch.autograd.grad(eager_loss, eager_inputs)

    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    with torch_compiler_warning_context():
        compiled_loss = compiled(*eager_inputs)
    compiled_gradients = torch.autograd.grad(compiled_loss, eager_inputs)

    torch.testing.assert_close(compiled_loss, eager_loss)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)


def test_explicit_batch_no_grad_compilation_uses_structured_rollout():
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert functional.prewarm_structured_rollout(ve=True, intra=True)
    ve, intra = _drives(model, steps=5)
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

    _assert_tree_close(actual, expected)
    assert len(graphs) == 1
    assert len(_while_loop_nodes(graphs[0])) == 1


def _differentiable_runner_case(tensors, ve, intra):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    parameters[GNABAR].requires_grad_()
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    constants["diam"].requires_grad_()
    state = _clone_state(tensors.state)
    state["integrator"]["v"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()
    targets = (
        parameters[GNABAR],
        constants["diam"],
        state["integrator"]["v"],
        ve,
        intra,
    )
    return parameters, constants, state, ve, intra, targets


def _run_with_runner(functional, tensors, ve, intra, runner):
    parameters, constants, state, ve, intra, targets = _differentiable_runner_case(
        tensors,
        ve,
        intra,
    )
    prepared = functional.prepare(parameters, constants)

    def step(current, inputs):
        return functional.step(parameters, prepared, current, inputs)

    if runner == "longrun":
        final, auxiliary = dn.func.longrun(
            functional,
            step,
            state,
            STEPS * DT,
            2,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
    else:
        final, auxiliary = dn.func.longrun_checkpointed(
            functional,
            step,
            state,
            STEPS * DT,
            2,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
    loss = final["integrator"]["v"].square().mean() + 0.01 * auxiliary["v"].sin().mean()
    gradients = torch.autograd.grad(loss, targets)
    return final, loss, gradients


def test_checkpointed_host_runner_preserves_explicit_batch_state_and_gradients():
    model = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)

    ordinary = _run_with_runner(functional, tensors, ve, intra, "longrun")
    checkpointed = _run_with_runner(
        functional,
        tensors,
        ve,
        intra,
        "checkpointed",
    )
    ordinary_state, ordinary_loss, ordinary_gradients = ordinary
    checkpointed_state, checkpointed_loss, checkpointed_gradients = checkpointed

    _assert_tree_close(
        checkpointed_state,
        ordinary_state,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        checkpointed_loss,
        ordinary_loss,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    for actual, expected in zip(
        checkpointed_gradients,
        ordinary_gradients,
        strict=True,
    ):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


def test_explicit_batch_commit_then_imperative_resume_is_transactional():
    source = _model(batch_calls=(2,))
    target = _model(batch_calls=(2,))
    reference = _model(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=5)

    checkpoint_state, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:3], intra=intra[:3]),
    )
    functional.commit_state_(target, checkpoint_state)

    target_dt = _initialize_imperative_integrator(target)
    for index in range(3, 5):
        _imperative_step(target, target_dt, ve[index], intra[index])
    reference_dt = _initialize_imperative_integrator(reference)
    for index in range(5):
        _imperative_step(reference, reference_dt, ve[index], intra[index])
    _assert_tree_close(
        functional.extract(target).state,
        functional.extract(reference).state,
    )

    wrong_batch = _model(batch_calls=(3,))
    voltage_before = wrong_batch.v.detach().clone()
    with pytest.raises(
        dn.func.FunctionalizationError, match="structure does not match"
    ):
        functional.commit_state_(wrong_batch, checkpoint_state)
    torch.testing.assert_close(wrong_batch.v, voltage_before)


def _injected_model(order):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(hh)
        waveform = dn.constant(
            value=model.v.new_tensor((0.12, 0.2, 0.31)) * nA,
        )
        if order == "before":
            model[:, 1:2].inject(waveform)
        model.batch(3)
        if order == "after":
            model[:, :, 1:2].inject(waveform)
        model.initialize()
        model.train()
    return model


@pytest.mark.parametrize("order", ["before", "after"])
def test_registered_batch_sweep_matches_imperative_before_or_after_batch(order):
    functional_model = _injected_model(order)
    imperative_model = _injected_model(order)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state

    for _index in range(STEPS):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
        )
        imperative_model.step(dt=DT)
        _assert_tree_close(
            state,
            functional.extract(imperative_model).state,
            rtol=2.0e-12,
            atol=2.0e-12,
        )


def test_registered_batch_sweep_jacobian_is_lane_local():
    model = _injected_model("before")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    value_name = next(
        name
        for name in tensors.parameters
        if name.startswith("stimulation.intra.") and name.endswith(".value")
    )
    base_value = tensors.parameters[value_name]

    def response(value):
        parameters = dict(tensors.parameters)
        parameters[value_name] = value
        state, _auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=2,
        )
        return state["integrator"]["v"][..., 1].mean(dim=-1)

    jacobian = torch.func.jacrev(response)(base_value)
    assert jacobian.shape == (3, 3)
    diagonal = torch.diagonal(jacobian)
    assert torch.isfinite(diagonal).all()
    assert torch.count_nonzero(diagonal) == diagonal.numel()
    torch.testing.assert_close(
        jacobian - torch.diag_embed(diagonal),
        torch.zeros_like(jacobian),
        rtol=0.0,
        atol=0.0,
    )


def _extra(model):
    shared_field = torch.linspace(
        -0.6,
        0.8,
        math.prod(model.core_shape()),
        dtype=model.dtype(),
    ).reshape(model.core_shape())
    batch_field = torch.stack(
        tuple((index + 1.0) * shared_field for index in range(model.shape[0]))
    )
    batch_waveform = dn.constant(
        value=model.v.new_tensor((0.3, 0.5, 0.9)),
    )
    shared_waveform = dn.constant(value=0.25)
    return [
        (shared_field, batch_waveform),
        (batch_field, shared_waveform),
    ]


def test_bound_multicontact_extra_preserves_shared_and_batch_broadcasting():
    functional_model = _model(batch_calls=(3,))
    imperative_model = _model(batch_calls=(3,))
    functional_extra = _extra(functional_model)
    imperative_extra = _extra(imperative_model)
    functional, tensors = dn.func.make_functional(
        functional_model,
        dt=DT,
        extra=functional_extra,
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state

    for _index in range(STEPS):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
        )
        imperative_model.step(dt=DT, extra=imperative_extra)
        _assert_tree_close(
            state,
            functional.extract(imperative_model).state,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
