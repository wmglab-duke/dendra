from __future__ import annotations

import gc
import weakref
from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh

DT = 0.01
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -60.0, -56.0, -59.0, -63.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
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
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.0e-9,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _clone_mapping(values: Mapping[str, torch.Tensor], *, requires_grad=False):
    return {
        name: value.detach().clone().requires_grad_(requires_grad)
        for name, value in values.items()
    }


def _clone_state(state):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), state)


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
        )


def _differentiable_case(tensors, ve, intra):
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()
    constants = _clone_mapping(tensors.constants, requires_grad=True)
    state = _clone_state(tensors.state)
    state["integrator"]["v"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()
    targets = (
        parameters[GNABAR],
        state["integrator"]["v"],
        constants["diam"],
        constants["dx"],
        ve,
        intra,
    )
    return parameters, constants, state, ve, intra, targets


def _loss(state, aux):
    return (
        state["integrator"]["v"].square().mean()
        + 0.1 * state["mechanisms"]["hh"]["m"].square().mean()
        + 0.01 * aux["v"].sin().mean()
    )


@pytest.mark.parametrize("steps", [True, 0, -1, 1.5, "4"])
def test_compile_rollout_chunk_requires_a_positive_integer(steps):
    functional, _tensors = dn.func.make_functional(_model(), dt=DT)

    with pytest.raises(ValueError, match="positive integer"):
        functional.compile_rollout_chunk(steps, backend="eager")


def test_compiled_chunk_reports_its_fixed_length_and_rejects_outer_compile():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(2, backend="eager")

    assert isinstance(chunk, dn.func.CompiledPopulationChunk)
    assert chunk.steps == 2

    compiled_outer = torch.compile(
        lambda state: chunk(tensors.parameters, prepared, state)[0],
        backend="eager",
        fullgraph=True,
    )
    with pytest.raises(Exception, match="already an outer Python wrapper"):
        compiled_outer(tensors.state)


def test_no_grad_compiled_chunk_lazily_captures_only_its_drive_signature():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, _intra = _drives(model, 5)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    chunk = functional.compile_rollout_chunk(5, backend=backend)
    assert not functional._structured_step_graphs

    with torch.no_grad(), torch_compiler_warning_context():
        chunk(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=ve),
        )

    assert set(functional._structured_step_graphs) == {(True, False)}
    assert len(graphs) == 1
    assert any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for node in graphs[0].graph.nodes
    )


def test_repeated_compiled_chunks_match_eager_bptt_for_all_explicit_sources():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 8)

    eager_case = _differentiable_case(tensors, ve, intra)
    (
        eager_parameters,
        eager_constants,
        eager_state,
        eager_ve,
        eager_intra,
        eager_targets,
    ) = eager_case
    eager_prepared = functional.prepare(eager_parameters, eager_constants)
    eager_final, eager_aux = functional.rollout(
        eager_parameters,
        eager_prepared,
        eager_state,
        dn.func.RolloutInput(ve=eager_ve, intra=eager_intra),
    )
    eager_loss = _loss(eager_final, eager_aux)
    eager_gradients = torch.autograd.grad(eager_loss, eager_targets)

    compiled_case = _differentiable_case(tensors, ve, intra)
    (
        compiled_parameters,
        compiled_constants,
        compiled_state,
        compiled_ve,
        compiled_intra,
        compiled_targets,
    ) = compiled_case
    compiled_prepared = functional.prepare(compiled_parameters, compiled_constants)
    chunk = functional.compile_rollout_chunk(4, backend="aot_eager")
    with torch_compiler_warning_context():
        for start in range(0, 8, chunk.steps):
            compiled_state, compiled_aux = chunk(
                compiled_parameters,
                compiled_prepared,
                compiled_state,
                dn.func.RolloutInput(
                    ve=compiled_ve[start : start + chunk.steps],
                    intra=compiled_intra[start : start + chunk.steps],
                ),
            )
    compiled_loss = _loss(compiled_state, compiled_aux)
    compiled_gradients = torch.autograd.grad(compiled_loss, compiled_targets)

    _assert_tree_close(compiled_state, eager_final, rtol=2.0e-10, atol=2.0e-11)
    _assert_tree_close(compiled_aux, eager_aux, rtol=2.0e-10, atol=2.0e-11)
    torch.testing.assert_close(
        compiled_loss,
        eager_loss,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    for actual, expected in zip(
        compiled_gradients,
        eager_gradients,
        strict=True,
    ):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_chunk_preserves_direct_forward_ad(backend):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    chunk = functional.compile_rollout_chunk(2, backend=backend)
    primal = tensors.parameters[GNABAR].detach()
    tangent = torch.full_like(primal, 0.125)

    with (
        torch.no_grad(),
        torch.autograd.forward_ad.dual_level(),
        torch_compiler_warning_context(),
    ):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = torch.autograd.forward_ad.make_dual(primal, tangent)
        prepared = functional.prepare(parameters, tensors.constants)
        expected_state, _aux = functional.rollout(
            parameters,
            prepared,
            tensors.state,
            steps=2,
        )
        actual_state, _aux = chunk(
            parameters,
            prepared,
            tensors.state,
        )
        expected = torch.autograd.forward_ad.unpack_dual(
            expected_state["integrator"]["v"]
        )
        actual = torch.autograd.forward_ad.unpack_dual(actual_state["integrator"]["v"])

    torch.testing.assert_close(actual.primal, expected.primal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.tangent,
        expected.tangent,
        rtol=2.0e-9,
        atol=2.0e-10,
    )


@pytest.mark.parametrize("executor_name", ["eager", "compiled"])
@pytest.mark.parametrize(
    ("mutation_target", "error_match"),
    [("source", "parameters changed"), ("workspace", "workspaces changed")],
)
def test_prepared_rollout_rejects_mutated_forward_ad_tangents(
    executor_name,
    mutation_target,
    error_match,
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    chunk = functional.compile_rollout_chunk(2, backend="eager")
    parameters = dict(tensors.parameters)
    constants = dict(tensors.constants)

    with (
        torch.no_grad(),
        torch.autograd.forward_ad.dual_level(),
        torch_compiler_warning_context(),
    ):
        if mutation_target == "source":
            tangent = torch.full_like(parameters[GNABAR], 0.125)
            parameters[GNABAR] = torch.autograd.forward_ad.make_dual(
                parameters[GNABAR],
                tangent,
            )
        else:
            diam_tangent = torch.full_like(constants["diam"], 0.125)
            constants["diam"] = torch.autograd.forward_ad.make_dual(
                constants["diam"],
                diam_tangent,
            )

        prepared = functional.prepare(parameters, constants)

        def execute():
            if executor_name == "compiled":
                return chunk(parameters, prepared, tensors.state)
            return functional.rollout(
                parameters,
                prepared,
                tensors.state,
                steps=2,
            )

        # Exercise a valid reuse first. For CompiledPopulationChunk this also
        # installs the weak prepared-plan cache whose fast freshness path must
        # notice the later tangent mutation.
        execute()
        if mutation_target == "workspace":
            tangent = torch.autograd.forward_ad.unpack_dual(
                prepared.values["integrator"]["diag_base"]
            ).tangent
            assert tangent is not None
        tangent.add_(0.5)

        with pytest.raises(dn.func.FunctionalizationError, match=error_match):
            execute()


@pytest.mark.parametrize(
    ("inference_component", "error_match"),
    [
        ("primal", "inference-tensor parameters or constants"),
        ("tangent", "forward-AD tangents.*clone them outside inference mode"),
    ],
)
def test_prepare_rejects_inference_forward_ad_components_under_inference_mode(
    inference_component,
    error_match,
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    with torch.inference_mode():
        inference_value = torch.ones_like(tensors.parameters[GNABAR])

    with torch.autograd.forward_ad.dual_level():
        parameters = dict(tensors.parameters)
        primal = (
            inference_value if inference_component == "primal" else parameters[GNABAR]
        )
        tangent = (
            inference_value
            if inference_component == "tangent"
            else torch.ones_like(parameters[GNABAR])
        )
        parameters[GNABAR] = torch.autograd.forward_ad.make_dual(
            primal,
            tangent,
        )
        # Inference mode hides dual tangents from unpack_dual() by default;
        # prepare() must still expose and reject either unversioned component.
        with (
            torch.inference_mode(),
            pytest.raises(
                dn.func.FunctionalizationError,
                match=error_match,
            ),
        ):
            functional.prepare(parameters, tensors.constants)


def test_compiled_chunk_hoists_preparation_and_reuses_one_shape_graph(monkeypatch):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 2)
    # Lazy structured inference capture performs one example
    # preparation. Prewarm this drive signature so the counter below measures
    # only the caller-visible prepare() operations whose hoisting is under test.
    assert functional.prewarm_structured_rollout(ve=True, intra=True)
    preparation_calls = 0
    compile_calls = 0
    original_prepare_values = functional._prepare_values

    def counted_prepare_values(parameters, constants):
        nonlocal preparation_calls
        preparation_calls += 1
        return original_prepare_values(parameters, constants)

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_calls
        compile_calls += 1
        return graph_module.forward

    monkeypatch.setattr(functional, "_prepare_values", counted_prepare_values)
    parameters_a = _clone_mapping(tensors.parameters, requires_grad=True)
    constants_a = _clone_mapping(tensors.constants)
    parameters_b = _clone_mapping(tensors.parameters, requires_grad=True)
    constants_b = _clone_mapping(tensors.constants)
    prepared_a = functional.prepare(parameters_a, constants_a)
    prepared_b = functional.prepare(parameters_b, constants_b)
    chunk = functional.compile_rollout_chunk(2, backend=counting_backend)

    with torch.no_grad(), torch_compiler_warning_context():
        state, _aux = chunk(
            parameters_a,
            prepared_a,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        chunk(
            parameters_a,
            prepared_a,
            state,
            dn.func.RolloutInput(ve=ve + 0.1, intra=intra),
        )
        chunk(
            parameters_b,
            prepared_b,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )

    assert preparation_calls == 2
    assert compile_calls == 1


@pytest.mark.parametrize("drive_kind", ["none", "ve", "intra", "both"])
def test_compiled_chunk_supports_every_optional_drive_signature(drive_kind):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, 3)
    selected_ve = ve if drive_kind in {"ve", "both"} else None
    selected_intra = intra if drive_kind in {"intra", "both"} else None
    inputs = dn.func.RolloutInput(ve=selected_ve, intra=selected_intra)
    expected = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
        steps=3,
    )
    chunk = functional.compile_rollout_chunk(3, backend="eager")

    with torch_compiler_warning_context():
        actual = chunk(tensors.parameters, prepared, tensors.state, inputs)

    _assert_tree_close(actual, expected, rtol=0.0, atol=0.0)


def test_separate_tail_chunk_matches_one_eager_rollout():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, 7)
    expected = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    main = functional.compile_rollout_chunk(4, backend="eager")
    tail = functional.compile_rollout_chunk(3, backend="eager")

    with torch_compiler_warning_context():
        state, _aux = main(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=ve[:4], intra=intra[:4]),
        )
        actual = tail(
            tensors.parameters,
            prepared,
            state,
            dn.func.RolloutInput(ve=ve[4:], intra=intra[4:]),
        )

    _assert_tree_close(actual, expected, rtol=0.0, atol=0.0)


def test_compiled_chunk_rejects_stale_wrong_plan_and_malformed_boundaries():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    chunk = functional.compile_rollout_chunk(2, backend="eager")
    ve, intra = _drives(model, 2)
    inputs = dn.func.RolloutInput(ve=ve, intra=intra)

    parameters = _clone_mapping(tensors.parameters, requires_grad=True)
    constants = _clone_mapping(tensors.constants)
    prepared = functional.prepare(parameters, constants)
    with torch.no_grad():
        parameters[GNABAR].add_(0.1)
    with pytest.raises(dn.func.FunctionalizationError, match="parameters changed"):
        chunk(parameters, prepared, tensors.state, inputs)

    parameters = _clone_mapping(tensors.parameters, requires_grad=True)
    constants = _clone_mapping(tensors.constants)
    prepared = functional.prepare(parameters, constants)
    constants["diam"].add_(0.1)
    with pytest.raises(dn.func.FunctionalizationError, match="constants changed"):
        chunk(parameters, prepared, tensors.state, inputs)

    parameters = _clone_mapping(tensors.parameters, requires_grad=True)
    constants = _clone_mapping(tensors.constants)
    prepared = functional.prepare(parameters, constants)
    constants["diam"] = constants["diam"].clone()
    with pytest.raises(dn.func.FunctionalizationError, match="constants changed"):
        chunk(parameters, prepared, tensors.state, inputs)

    parameters = _clone_mapping(tensors.parameters, requires_grad=True)
    constants = _clone_mapping(tensors.constants)
    prepared = functional.prepare(parameters, constants)
    with torch.no_grad():
        prepared.values["integrator"]["diag_base"].add_(0.1)
    with pytest.raises(dn.func.FunctionalizationError, match="workspaces changed"):
        chunk(parameters, prepared, tensors.state, inputs)

    parameters = _clone_mapping(tensors.parameters)
    constants = _clone_mapping(tensors.constants)
    prepared = functional.prepare(parameters, constants)
    parameters[GNABAR].requires_grad_()
    with pytest.raises(dn.func.FunctionalizationError, match="parameters changed"):
        chunk(parameters, prepared, tensors.state, inputs)

    parameters = _clone_mapping(tensors.parameters, requires_grad=True)
    constants = _clone_mapping(tensors.constants)
    prepared = functional.prepare(parameters, constants)
    integrator_values = prepared.values["integrator"]
    prepared.values["integrator"] = {
        ("renamed_diag_base" if name == "diag_base" else name): value
        for name, value in integrator_values.items()
    }
    with pytest.raises(dn.func.FunctionalizationError, match="workspaces changed"):
        chunk(parameters, prepared, tensors.state, inputs)

    other_functional, other_tensors = dn.func.make_functional(_model(), dt=2 * DT)
    other_prepared = other_functional.prepare(
        other_tensors.parameters,
        other_tensors.constants,
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different.*plan"):
        chunk(
            other_tensors.parameters,
            other_prepared,
            tensors.state,
            inputs,
        )

    with pytest.raises(TypeError, match="opaque value returned by prepare"):
        chunk(tensors.parameters, None, tensors.state, inputs)
    malformed_state = {**tensors.state, "integrator": tensors.state["integrator"]["v"]}
    with pytest.raises(TypeError, match=r"state\['integrator'\].*mapping"):
        chunk(
            tensors.parameters,
            functional.prepare(tensors.parameters, tensors.constants),
            malformed_state,
            inputs,
        )
    with pytest.raises(ValueError, match="expects 2"):
        chunk(
            tensors.parameters,
            functional.prepare(tensors.parameters, tensors.constants),
            tensors.state,
            dn.func.RolloutInput(ve=ve[:1], intra=intra[:1]),
        )


@pytest.mark.parametrize("consumer", ["eager", "compiled"])
def test_grad_enabled_consumption_rejects_grad_sources_prepared_under_no_grad(
    consumer,
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    with torch.no_grad():
        prepared = functional.prepare(tensors.parameters, tensors.constants)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="gradient recording disabled",
    ):
        if consumer == "eager":
            functional.rollout(
                tensors.parameters,
                prepared,
                tensors.state,
                steps=2,
            )
        else:
            chunk = functional.compile_rollout_chunk(2, backend="eager")
            chunk(tensors.parameters, prepared, tensors.state)


def test_no_grad_preparation_is_safe_for_no_grad_or_detached_sources():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    chunk = functional.compile_rollout_chunk(2, backend="eager")
    with torch.no_grad():
        grad_source_prepared = functional.prepare(
            tensors.parameters,
            tensors.constants,
        )
        no_grad_state, _aux = chunk(
            tensors.parameters,
            grad_source_prepared,
            tensors.state,
        )
    assert not no_grad_state["integrator"]["v"].requires_grad

    parameters = _clone_mapping(tensors.parameters)
    constants = _clone_mapping(tensors.constants)
    state = _clone_state(tensors.state)
    state["integrator"]["v"].requires_grad_()
    ve, _intra = _drives(_model(), 2)
    ve.requires_grad_()
    with torch.no_grad():
        detached_prepared = functional.prepare(parameters, constants)

    with torch_compiler_warning_context():
        final, _aux = chunk(
            parameters,
            detached_prepared,
            state,
            dn.func.RolloutInput(ve=ve),
        )
    state_gradient, drive_gradient = torch.autograd.grad(
        final["integrator"]["v"].square().mean(),
        (state["integrator"]["v"], ve),
    )
    assert torch.isfinite(state_gradient).all()
    assert torch.isfinite(drive_gradient).all()


def test_inference_mode_preparation_materializes_versioned_workspaces():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    chunk = functional.compile_rollout_chunk(2, backend="eager")
    regular_prepared = functional.prepare(tensors.parameters, tensors.constants)
    expected = functional.rollout(
        tensors.parameters,
        regular_prepared,
        tensors.state,
        steps=2,
    )

    with torch.inference_mode(), torch_compiler_warning_context():
        inference_prepared = functional.prepare(
            tensors.parameters,
            tensors.constants,
        )
        in_inference_mode = chunk(
            tensors.parameters,
            inference_prepared,
            tensors.state,
        )
    assert all(
        not torch.is_inference(value)
        for value in torch.utils._pytree.tree_leaves(inference_prepared.values)
    )
    with torch.no_grad(), torch_compiler_warning_context():
        in_no_grad = chunk(
            tensors.parameters,
            inference_prepared,
            tensors.state,
        )

    _assert_tree_close(in_inference_mode, expected, rtol=0.0, atol=0.0)
    _assert_tree_close(in_no_grad, expected, rtol=0.0, atol=0.0)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="gradient recording disabled",
    ):
        chunk(tensors.parameters, inference_prepared, tensors.state)

    with torch.inference_mode():
        mutated_prepared = functional.prepare(
            tensors.parameters,
            tensors.constants,
        )
        mutated_prepared.values["integrator"]["diag_base"].add_(1.0)
    with (
        torch.no_grad(),
        pytest.raises(
            dn.func.FunctionalizationError,
            match="workspaces changed",
        ),
    ):
        chunk(tensors.parameters, mutated_prepared, tensors.state)

    with torch.inference_mode(), torch.enable_grad():
        nested_enable_grad_prepared = functional.prepare(
            tensors.parameters,
            tensors.constants,
        )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="gradient recording disabled",
    ):
        chunk(tensors.parameters, nested_enable_grad_prepared, tensors.state)


def test_inference_mode_lowering_and_structured_capture_own_versioned_tensors():
    model = _model()
    with torch.inference_mode():
        functional, tensors = dn.func.make_functional(model, dt=DT)
        assert functional.prewarm_structured_rollout(ve=True, intra=False)

    owned_trees = (
        tensors.constants,
        tensors.state,
        functional._base_mapping,
        functional._preparation_base_mapping,
    )
    assert all(
        not torch.is_inference(value)
        for tree in owned_trees
        for value in torch.utils._pytree.tree_leaves(tree)
    )
    assert set(functional._structured_step_graphs) == {(True, False)}


def test_prepare_rejects_unversioned_inference_tensor_sources():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    with torch.inference_mode():
        parameters = {name: value.clone() for name, value in tensors.parameters.items()}
        constants = {name: value.clone() for name, value in tensors.constants.items()}

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="inference-tensor parameters",
    ):
        functional.prepare(parameters, constants)


def test_compiled_chunk_cache_does_not_retain_preparation_autograd_graph():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    chunk = functional.compile_rollout_chunk(2, backend="eager")

    def run_iteration():
        parameters = _clone_mapping(tensors.parameters, requires_grad=True)
        constants = _clone_mapping(tensors.constants, requires_grad=True)
        prepared = functional.prepare(parameters, constants)
        references = {
            "prepared": weakref.ref(prepared),
            "parameter": weakref.ref(parameters[GNABAR]),
            "constant": weakref.ref(constants["diam"]),
            "workspace": weakref.ref(prepared.values["integrator"]["diag_base"]),
        }
        with torch_compiler_warning_context():
            final, _aux = chunk(parameters, prepared, tensors.state)
        final["integrator"]["v"].square().mean().backward()
        return references

    references = run_iteration()
    gc.collect()

    assert {name: reference() for name, reference in references.items()} == {
        "prepared": None,
        "parameter": None,
        "constant": None,
        "workspace": None,
    }


def test_compiled_chunk_weak_cache_keeps_fast_validation_path(monkeypatch):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(2, backend="eager")
    full_validation_calls = 0
    freshness_calls = 0
    original_validate_prepared = functional._validate_prepared
    original_validate_freshness = functional._validate_prepared_freshness

    def counted_validate_prepared(candidate, parameters):
        nonlocal full_validation_calls
        full_validation_calls += 1
        return original_validate_prepared(candidate, parameters)

    def counted_validate_freshness(candidate, parameters):
        nonlocal freshness_calls
        freshness_calls += 1
        return original_validate_freshness(candidate, parameters)

    monkeypatch.setattr(functional, "_validate_prepared", counted_validate_prepared)
    monkeypatch.setattr(
        functional,
        "_validate_prepared_freshness",
        counted_validate_freshness,
    )

    with torch.no_grad(), torch_compiler_warning_context():
        first, _aux = chunk(tensors.parameters, prepared, tensors.state)
        chunk(tensors.parameters, prepared, first)

    assert full_validation_calls == 1
    # Full schema validation includes a freshness check; the second call then
    # takes the cheap freshness-only path through the live weak reference.
    assert freshness_calls == 2


def test_compiled_chunk_uses_transform_safe_eager_path_for_torch_func():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    compile_calls = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_calls
        compile_calls += 1
        return graph_module.forward

    chunk = functional.compile_rollout_chunk(2, backend=counting_backend)
    base = tensors.parameters[GNABAR]
    lanes = torch.stack((0.9 * base, base, 1.1 * base))

    def run(gnabar, executor):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        prepared = functional.prepare(parameters, tensors.constants)
        final, _aux = executor(parameters, prepared)
        return final["integrator"]["v"]

    def eager(parameters, prepared):
        return functional.rollout(
            parameters,
            prepared,
            tensors.state,
            steps=2,
        )

    def through_chunk(parameters, prepared):
        return chunk(parameters, prepared, tensors.state)

    def eager_run(gnabar):
        return run(gnabar, eager)

    def chunk_run(gnabar):
        return run(gnabar, through_chunk)

    actual_vmap = torch.func.vmap(chunk_run)(lanes)
    expected_vmap = torch.func.vmap(eager_run)(lanes)
    actual_jacrev = torch.func.jacrev(chunk_run)(base)
    expected_jacrev = torch.func.jacrev(eager_run)(base)
    actual_jacfwd = torch.func.jacfwd(chunk_run)(base)
    expected_jacfwd = torch.func.jacfwd(eager_run)(base)

    torch.testing.assert_close(actual_vmap, expected_vmap, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_jacrev,
        expected_jacrev,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        actual_jacfwd,
        expected_jacfwd,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert compile_calls == 0
