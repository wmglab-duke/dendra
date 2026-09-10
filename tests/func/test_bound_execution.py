"""Public bindings preserve the checked transition and host-runner contracts."""

import gc
import weakref
from functools import partial

import pytest
import torch
from test_runner_compiled_callbacks import _callbacks
from test_runner_compiled_chunks import _run as _legacy_run
from test_runners import (
    DT,
    GNABAR,
    _assert_tree_close,
    _clone_mapping,
    _clone_state,
    _differentiable_case,
    _drives,
    _loss,
    _model,
    _replace_voltage,
)

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _bind(functional, parameters, prepared, kernel=None):
    return (functional if kernel is None else kernel).bind(parameters, prepared)


def _evaluate(
    functional,
    tensors,
    ve,
    intra,
    *,
    kernel=None,
    runner="canonical",
    options=None,
    callbacks=None,
    factor=1.0,
):
    parameters, constants, state, ve, intra, targets = _differentiable_case(
        tensors, ve, intra
    )
    with torch.no_grad():
        parameters[GNABAR].mul_(factor)
        constants["diam"].mul_(0.9 + 0.1 * factor)
    prepared = functional.prepare(parameters, constants)
    inputs = dn.func.RolloutInput(ve, intra)
    if runner == "partial":
        step = partial(functional.step, parameters, prepared)
    else:
        step = _bind(functional, parameters, prepared, kernel)
    if runner == "canonical":
        final, auxiliary = dn.func.run(
            step, state, inputs=inputs, callbacks=callbacks, **(options or {})
        )
    else:
        final, auxiliary = _legacy_run(
            "run" if runner == "partial" else runner,
            functional,
            step,
            state,
            inputs,
            steps=ve.shape[0],
            chunklength=4,
            callbacks=callbacks,
        )
    if callbacks is None:
        loss = _loss(final, auxiliary)
    else:
        result = auxiliary["callbacks"]
        loss = result["mse"] + 0.1 * result["trace"]["v"].square().mean()
    gradients = dict(
        zip(targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True)
    )
    return final, auxiliary, loss, gradients


def _assert_evaluation(actual, expected):
    _assert_tree_close(actual[:3], expected[:3], rtol=2e-10, atol=2e-11)
    assert actual[3].keys() == expected[3].keys()
    for name, reference in expected[3].items():
        observed = actual[3][name]
        assert torch.isfinite(observed).all(), name
        assert torch.isfinite(reference).all(), name
        scale = reference.abs().amax()
        assert scale > 0, name
        assert (observed - reference).abs().amax() / scale <= 2e-8, name


@pytest.mark.parametrize("width", [1, 3])
def test_binding_metadata_and_parameter_snapshot_are_read_only(width):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    prepared = functional.prepare(parameters, tensors.constants)
    kernel = (
        None if width == 1 else functional.compile_rollout_chunk(width, backend="eager")
    )
    bound = _bind(functional, parameters, prepared, kernel)

    assert isinstance(bound, dn.func.BoundPopulation)
    assert bound.functional is functional
    assert bound.prepared is prepared
    assert bound.compiled_chunk is kernel
    assert bound.steps == width
    assert bound.parameters is not parameters
    original = parameters[GNABAR]
    assert bound.parameters[GNABAR] is original
    for name in ("functional", "parameters", "prepared", "compiled_chunk", "steps"):
        with pytest.raises(AttributeError):
            setattr(bound, name, getattr(bound, name))
    with pytest.raises(TypeError):
        bound.parameters[GNABAR] = original.clone()

    # A structural edit of the caller's mapping must not redirect the binding.
    parameters[GNABAR] = original + 0.05
    parameters["unrelated"] = original
    assert bound.parameters[GNABAR] is original
    assert "unrelated" not in bound.parameters
    with torch.no_grad(), torch_compiler_warning_context():
        actual = bound(tensors.state)
        expected = functional.rollout(
            dict(bound.parameters), prepared, tensors.state, steps=width
        )
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("width", [1, 3])
def test_direct_bound_call_consumes_the_owned_transition_input_shape(width):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, width)
    kernel = (
        None if width == 1 else functional.compile_rollout_chunk(width, backend="eager")
    )
    bound = _bind(functional, tensors.parameters, prepared, kernel)
    inputs = (
        dn.func.StepInput(ve[0], intra[0])
        if width == 1
        else dn.func.RolloutInput(ve, intra)
    )
    with torch.no_grad(), torch_compiler_warning_context():
        actual = bound(tensors.state, inputs)
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, dn.func.RolloutInput(ve, intra)
        )
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    if width > 1:
        with pytest.raises(ValueError, match="expects|steps|length"):
            bound(tensors.state, dn.func.RolloutInput(ve[:1], intra[:1]))


@pytest.mark.parametrize(
    ("compiled", "options"),
    [
        (False, {}),
        (True, {}),
        (True, {"host_span_steps": 4}),
        (True, {"checkpoint_every": 4}),
        (False, {"checkpoint_every": 4}),
    ],
)
def test_canonical_bound_run_preserves_callbacks_and_all_source_gradients(
    compiled, options
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    callbacks = _callbacks(functional, model, 7)
    expected = _evaluate(
        functional, tensors, ve, intra, runner="partial", callbacks=callbacks
    )
    kernel = (
        functional.compile_rollout_chunk(3, backend="aot_eager") if compiled else None
    )
    # T=7, C=3 and H=4 require both kernel and host/checkpoint tails.
    with torch_compiler_warning_context():
        actual = _evaluate(
            functional,
            tensors,
            ve,
            intra,
            kernel=kernel,
            callbacks=callbacks,
            options=options,
        )
    _assert_evaluation(actual, expected)
    assert actual[1]["callbacks"]["trace"]["v"].shape[0] == 8
    assert actual[1]["callbacks"].state["mse"]["frames"].item() == 8


@pytest.mark.parametrize("runner", ["run", "longrun", "longrun_checkpointed"])
@pytest.mark.parametrize("compiled", [False, True])
def test_legacy_runners_accept_explicit_bindings(runner, compiled):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 5)
    expected = _evaluate(functional, tensors, ve, intra, runner="partial")
    kernel = functional.compile_rollout_chunk(3, backend="eager") if compiled else None
    with torch_compiler_warning_context():
        actual = _evaluate(functional, tensors, ve, intra, kernel=kernel, runner=runner)
    _assert_evaluation(actual, expected)


def test_rebinding_a_shared_kernel_uses_fresh_parameters_and_preparation():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 5)
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")
    losses = []
    for factor in (0.8, 1.3):
        expected = _evaluate(
            functional, tensors, ve, intra, runner="partial", factor=factor
        )
        with torch_compiler_warning_context():
            actual = _evaluate(
                functional, tensors, ve, intra, kernel=kernel, factor=factor
            )
        _assert_evaluation(actual, expected)
        losses.append(actual[2].detach())
    assert not torch.isclose(*losses, rtol=1e-5, atol=1e-7)


def test_bound_callback_resume_includes_zero_segments_without_duplicate_samples():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 5)
    callbacks = _callbacks(functional, model, 5)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    expected, expected_aux = dn.func.run(
        functional,
        partial(functional.step, tensors.parameters, prepared),
        tensors.state,
        dn.func.RolloutInput(ve, intra),
        callbacks=callbacks,
    )
    kernel = functional.compile_rollout_chunk(3, backend="eager")
    bound = kernel.bind(tensors.parameters, prepared)
    state, callback_state = tensors.state, None
    frames, start = [], 0
    with torch.no_grad(), torch_compiler_warning_context():
        for count in (0, 2, 0, 3, 0):
            state, auxiliary = dn.func.run(
                bound,
                state,
                inputs=dn.func.RolloutInput(
                    ve[start : start + count], intra[start : start + count]
                ),
                steps=count,
                checkpoint_every=4,
                callbacks=callbacks,
                callback_state=callback_state,
            )
            result = auxiliary["callbacks"]
            frames.append(result["trace"]["v"])
            callback_state = result.state
            start += count
            assert callback_state["mse"]["frames"].item() == start + 1
    _assert_tree_close(state, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        torch.cat(frames),
        expected_aux["callbacks"]["trace"]["v"],
        rtol=2e-10,
        atol=2e-11,
    )
    torch.testing.assert_close(result["mse"], expected_aux["callbacks"]["mse"])


def test_count_horizons_preserve_remainder_and_duration_consumes_it():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)
    state, _ = dn.func.run(bound, tensors.state, duration=0.6 * DT)
    assert state["clock"]["t"].item() == 0
    state, _ = dn.func.run(bound, state, duration=0.6 * DT)
    assert state["clock"]["t"].item() == pytest.approx(DT)
    remainder = state["control"]["duration_remainder"].clone()
    assert remainder.item() == pytest.approx(0.2 * DT)
    expected, _ = functional.rollout(tensors.parameters, prepared, state, steps=2)
    actual, _ = dn.func.run(bound, state, steps=2)
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        actual["control"]["duration_remainder"], remainder, rtol=0, atol=0
    )
    empty = torch.empty((0, *functional.shape), dtype=functional.dtype)
    resumed, _ = dn.func.run(bound, actual, inputs=dn.func.RolloutInput(ve=empty))
    _assert_tree_close(resumed, actual)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_count_shorter_than_bound_kernel_is_scheduled_without_changing_width(count):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(4, backend="eager")
    bound = kernel.bind(tensors.parameters, prepared)
    with torch.no_grad(), torch_compiler_warning_context():
        actual, _ = dn.func.run(bound, tensors.state, steps=count, host_span_steps=3)
        expected, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=count
        )
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    assert bound.steps == kernel.steps == 4


@pytest.mark.parametrize("options", [{"steps": 3}, {"duration": 3 * DT}])
def test_explicit_horizon_must_match_time_first_drives(options):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)
    ve, intra = _drives(model, 2)
    with pytest.raises(ValueError, match="steps|length|expects|horizon|duration"):
        dn.func.run(
            bound, tensors.state, inputs=dn.func.RolloutInput(ve, intra), **options
        )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"steps": 2, "duration": 2 * DT},
        {"steps": 2, "host_span_steps": 2, "checkpoint_every": 2},
        {"steps": True},
        {"steps": -1},
        {"steps": 1.5},
        {"duration": float("nan")},
        {"steps": 2, "checkpoint_every": 0},
    ],
)
def test_canonical_runner_rejects_ambiguous_or_invalid_scheduling(options):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)
    with pytest.raises(ValueError):
        dn.func.run(bound, tensors.state, **options)


def test_canonical_runner_infers_prepared_dt_and_rejects_a_conflicting_override():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    constants = _clone_mapping(tensors.constants)
    constants["dt"] = constants["dt"].new_tensor(2 * DT)
    prepared = functional.prepare(tensors.parameters, constants)
    bound = functional.bind(tensors.parameters, prepared)
    ve, intra = _drives(model, 3)
    inputs = dn.func.RolloutInput(ve, intra)
    actual = dn.func.run(bound, tensors.state, inputs=inputs, duration=6 * DT)
    expected = dn.func.run(
        functional,
        partial(functional.step, tensors.parameters, prepared),
        tensors.state,
        inputs,
        dt=2 * DT,
    )
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    assert actual[0]["clock"]["t"].item() == pytest.approx(6 * DT)
    with pytest.raises(dn.func.FunctionalizationError, match="dt|timestep"):
        dn.func.run(bound, tensors.state, steps=0, dt=DT)


def test_nominal_float32_dt_preserves_duration_and_threshold_callback_windows():
    model = _model(dtype=torch.float32)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks(
        {
            "spikes": dn.func.APCount(
                threshold=-60.0,
                node_check=(0, 4),
                t_start_check=DT,
                t_end_check=3 * DT,
            ),
            "trace": dn.func.Recorder(["t"]),
        }
    )
    bound = functional.bind(tensors.parameters, prepared)
    actual = dn.func.run(bound, tensors.state, duration=3 * DT, callbacks=callbacks)
    expected = dn.func.run(
        functional,
        partial(functional.step, tensors.parameters, prepared),
        tensors.state,
        tstop=3 * DT,
        callbacks=callbacks,
    )
    _assert_tree_close(actual, expected)
    assert actual[1]["callbacks"]["trace"]["t"].shape[0] == 4
    assert actual[0]["control"]["duration_remainder"].item() == 0


@pytest.mark.parametrize("source", ["parameters", "constants", "workspaces"])
def test_binding_rechecks_tensor_freshness_at_zero_and_nonzero_boundaries(source):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    constants = _clone_mapping(tensors.constants)
    prepared = functional.prepare(parameters, constants)
    bound = functional.compile_rollout_chunk(3, backend="eager").bind(
        parameters, prepared
    )
    changed = {
        "parameters": parameters[GNABAR],
        "constants": constants["diam"],
        "workspaces": prepared.values["integrator"]["diag_base"],
    }[source]
    with torch.no_grad():
        changed.add_(0.1)
    for count in (0, 2):
        with pytest.raises(dn.func.FunctionalizationError, match=f"{source} changed"):
            dn.func.run(bound, tensors.state, steps=count)


def test_bound_checkpoint_rechecks_parameter_freshness_during_backward():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    bound = functional.bind(parameters, prepared)
    final, auxiliary = dn.func.run(bound, tensors.state, steps=3, checkpoint_every=2)
    loss = _loss(final, auxiliary)
    with torch.no_grad():
        parameters[GNABAR].add_(0.1)
    with pytest.raises(dn.func.FunctionalizationError, match="parameters changed"):
        torch.autograd.grad(loss, parameters[GNABAR])


def test_bindings_reject_wrong_preparation_and_legacy_runner_plan():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    other, other_tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    other_prepared = other.prepare(other_tensors.parameters, other_tensors.constants)
    with pytest.raises(dn.func.FunctionalizationError, match="different.*plan"):
        functional.bind(other_tensors.parameters, other_prepared)
    bound = functional.bind(tensors.parameters, prepared)
    with pytest.raises(dn.func.FunctionalizationError, match="different.*plan"):
        dn.func.run(other, bound, other_tensors.state, tstop=0.0)


@pytest.mark.parametrize("compiled", [False, True])
def test_binding_rejects_a_functional_subclass_whose_step_could_be_bypassed(compiled):
    class OverriddenPopulation(dn.func.FunctionalPopulation):
        def step(self, *_args, **_kwargs):
            raise AssertionError("An owned runner must never bypass this override")

    functional = OverriddenPopulation(_model(), dt=DT)
    tensors = functional.extract()
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(2, backend="eager") if compiled else None
    with pytest.raises(TypeError, match="exact FunctionalPopulation"):
        _bind(functional, tensors.parameters, prepared, kernel)


def test_eager_binding_rejects_an_existing_instance_step_override(monkeypatch):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def override(*_args, **_kwargs):
        raise AssertionError("Binding must reject an existing overridden transition")

    monkeypatch.setattr(functional, "step", override)
    with pytest.raises(
        dn.func.FunctionalizationError, match=r"owned FunctionalPopulation\.step"
    ):
        functional.bind(tensors.parameters, prepared)


def test_existing_eager_binding_retains_its_transition_after_instance_override(
    monkeypatch,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)
    ve, intra = _drives(model, 2)
    inputs = dn.func.RolloutInput(ve, intra)
    expected_direct = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve[:1], intra[:1]),
    )
    expected_run = functional.rollout(
        tensors.parameters, prepared, tensors.state, inputs
    )

    def override(*_args, **_kwargs):
        raise AssertionError(
            "An existing binding retains its captured owned transition"
        )

    monkeypatch.setattr(functional, "step", override)
    actual_direct = bound(tensors.state, dn.func.StepInput(ve[0], intra[0]))
    actual_run = dn.func.run(bound, tensors.state, inputs=inputs)
    _assert_tree_close(actual_direct, expected_direct, rtol=2e-10, atol=2e-11)
    _assert_tree_close(actual_run, expected_run, rtol=2e-10, atol=2e-11)


def test_bound_zero_step_run_rejects_a_changed_source_population_plan():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)
    model.to(dtype=torch.float32)
    with pytest.raises(dn.func.FunctionalizationError, match="structure|lower.*again"):
        dn.func.run(bound, tensors.state, steps=0)


def test_dead_binding_does_not_leave_differentiable_preparation_in_kernel_cache():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    kernel = functional.compile_rollout_chunk(2, backend="eager")

    def iteration():
        parameters = _clone_mapping(tensors.parameters)
        parameters[GNABAR].requires_grad_()
        constants = _clone_mapping(tensors.constants)
        constants["diam"].requires_grad_()
        prepared = functional.prepare(parameters, constants)
        bound = kernel.bind(parameters, prepared)
        references = {
            "prepared": weakref.ref(prepared),
            "parameter": weakref.ref(parameters[GNABAR]),
            "constant": weakref.ref(constants["diam"]),
            "workspace": weakref.ref(prepared.values["integrator"]["diag_base"]),
        }
        with torch_compiler_warning_context():
            final, _ = bound(tensors.state)
        final["integrator"]["v"].square().mean().backward()
        return references

    references = iteration()
    gc.collect()
    assert all(reference() is None for reference in references.values())


@pytest.mark.parametrize("compiled", [False, True])
def test_binding_created_inside_torch_func_preserves_grad_and_vmap(compiled):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    compile_calls = []

    def backend(graph, _inputs):
        compile_calls.append(graph)
        return graph.forward

    kernel = functional.compile_rollout_chunk(2, backend=backend) if compiled else None
    width = 2 if compiled else 1
    base = tensors.parameters[GNABAR]

    def evaluate(gnabar, *, bound):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        prepared = functional.prepare(parameters, tensors.constants)
        if bound:
            final, _ = _bind(functional, parameters, prepared, kernel)(tensors.state)
        else:
            final, _ = functional.rollout(
                parameters, prepared, tensors.state, steps=width
            )
        return final["integrator"]["v"].square().mean()

    actual = partial(evaluate, bound=True)
    expected = partial(evaluate, bound=False)
    observed_gradient = torch.func.grad(actual)(base)
    reference_gradient = torch.func.grad(expected)(base)
    assert reference_gradient.abs().amax() > 0
    torch.testing.assert_close(
        observed_gradient, reference_gradient, rtol=2e-10, atol=2e-11
    )
    lanes = torch.stack((0.9 * base, 1.1 * base))
    torch.testing.assert_close(
        torch.func.vmap(actual)(lanes), torch.func.vmap(expected)(lanes)
    )
    assert not compile_calls


def test_canonical_host_run_rejects_opaque_callables_and_torch_func_scheduling():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)

    def opaque(state, inputs=None):
        return bound(state, inputs)

    with pytest.raises(
        ValueError,
        match="require run\\(bound",
    ):
        dn.func.run(opaque, tensors.state, steps=1)

    def scheduled(voltage):
        state = _replace_voltage(tensors.state, voltage)
        return dn.func.run(bound, state, steps=1)[0]["integrator"]["v"]

    lanes = torch.stack((tensors.state["integrator"]["v"],) * 2)
    with pytest.raises(
        dn.func.FunctionalizationError, match="not a torch.func transform"
    ):
        torch.func.vmap(scheduled)(lanes)


@pytest.mark.parametrize("construct_inside", [False, True])
def test_binding_rejects_opaque_outer_compile_crossing(construct_inside):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)

    def outer(state):
        local = (
            functional.bind(tensors.parameters, prepared) if construct_inside else bound
        )
        return local(state)[0]["integrator"]["v"]

    compiled = torch.compile(outer, backend="eager", fullgraph=True)
    with pytest.raises(Exception, match="compile|wrapper|BoundPopulation|binding"):
        compiled(_clone_state(tensors.state))
