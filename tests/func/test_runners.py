from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._lowered import _LoweredPopulationChunk
from dendra.models.mod import hh

DT = 0.01
STEPS = 5
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model(*, dtype=torch.float64):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor(
                [-64.0, -60.0, -56.0, -59.0, -63.0],
                dtype=dtype,
            ),
            dtype=dtype,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    return model


def _drives(model, steps=STEPS):
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


def _clone_mapping(values: Mapping[str, torch.Tensor]):
    return {name: value.detach().clone() for name, value in values.items()}


def _clone_state(state):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), state)


def _replace_voltage(state, voltage):
    replaced = dict(state)
    replaced["integrator"] = dict(state["integrator"])
    replaced["integrator"]["v"] = voltage
    return replaced


def _step_to_rollout(inputs):
    return dn.func.RolloutInput(
        ve=None if inputs.ve is None else inputs.ve.unsqueeze(0),
        intra=None if inputs.intra is None else inputs.intra.unsqueeze(0),
    )


def _first_step_input(inputs):
    return dn.func.StepInput(
        ve=None if inputs.ve is None else inputs.ve[0],
        intra=None if inputs.intra is None else inputs.intra[0],
    )


def _make_bound_step(name, functional, parameters, prepared, state, inputs):
    """Bind today's three execution forms to ``(state, StepInput)``."""
    if name == "bound_functional_step":
        return partial(functional.step, parameters, prepared)

    if name == "eager_step":

        def eager_step(local_state, step_inputs):
            return functional.step(
                parameters,
                prepared,
                local_state,
                step_inputs,
            )

        return eager_step

    if name == "compiled_tensor_step":
        lowered = _LoweredPopulationChunk(functional, steps=1)
        first = _step_to_rollout(_first_step_input(inputs))
        bound = lowered.bind(parameters, prepared, state, first)

        def tensor_step(local_state, step_inputs):
            rollout_inputs = _step_to_rollout(step_inputs)
            return lowered(
                parameters,
                bound.prepared,
                local_state,
                rollout_inputs.ve,
                rollout_inputs.intra,
            )

        return torch.compile(
            tensor_step,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )

    if name == "compiled_chunk_one":
        chunk = functional.compile_rollout_chunk(1, backend="aot_eager")

        def compiled_chunk_step(local_state, step_inputs):
            # A one-step chunk accepts StepInput directly; no time-axis adapter
            # belongs in the host runner or its bound callable.
            return chunk(parameters, prepared, local_state, step_inputs)

        return compiled_chunk_step

    if name == "bound_compiled_chunk_one":
        chunk = functional.compile_rollout_chunk(1, backend="aot_eager")
        return partial(chunk, parameters, prepared)

    raise AssertionError(f"unknown executor {name!r}")


def _invoke_runner(
    name,
    functional,
    step,
    state,
    inputs,
    *,
    steps,
    chunklength=2,
):
    duration = steps * functional.dt
    if name == "run":
        return dn.func.run(
            functional,
            step,
            state,
            inputs,
            tstop=duration,
        )
    if name == "longrun":
        return dn.func.longrun(
            functional,
            step,
            state,
            duration,
            chunklength,
            inputs,
        )
    if name == "longrun_checkpointed":
        return dn.func.longrun_checkpointed(
            functional,
            step,
            state,
            duration,
            chunklength,
            inputs,
        )
    raise AssertionError(f"unknown runner {name!r}")


def _differentiable_case(tensors, ve, intra, *, geometry_only=False):
    parameters = _clone_mapping(tensors.parameters)
    constants = _clone_mapping(tensors.constants)
    state = _clone_state(tensors.state)
    ve = ve.detach().clone()
    intra = intra.detach().clone()

    constants["diam"].requires_grad_()
    targets = {"diam": constants["diam"]}
    if not geometry_only:
        parameters[GNABAR].requires_grad_()
        constants["dx"].requires_grad_()
        state["integrator"]["v"].requires_grad_()
        ve.requires_grad_()
        intra.requires_grad_()
        targets = {
            "parameter": parameters[GNABAR],
            "diam": constants["diam"],
            "dx": constants["dx"],
            "initial_voltage": state["integrator"]["v"],
            "ve": ve,
            "intra": intra,
        }
    return parameters, constants, state, ve, intra, targets


def _loss(state, auxiliary):
    return (
        state["integrator"]["v"].square().mean()
        + 0.1 * state["mechanisms"]["hh"]["m"].square().mean()
        + 0.01 * auxiliary["v"].sin().mean()
    )


def _execute(
    functional,
    tensors,
    ve,
    intra,
    executor_name,
    runner_name,
    *,
    geometry_only=False,
):
    parameters, constants, state, ve, intra, targets = _differentiable_case(
        tensors,
        ve,
        intra,
        geometry_only=geometry_only,
    )
    prepared = functional.prepare(parameters, constants)
    inputs = dn.func.RolloutInput(ve, intra)
    step = _make_bound_step(
        executor_name,
        functional,
        parameters,
        prepared,
        state,
        inputs,
    )
    final, auxiliary = _invoke_runner(
        runner_name,
        functional,
        step,
        state,
        inputs,
        steps=ve.shape[0],
    )
    loss = _loss(final, auxiliary)
    gradients = dict(
        zip(
            targets,
            torch.autograd.grad(loss, tuple(targets.values())),
            strict=True,
        )
    )
    return final, loss, gradients


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


def _invoke_duration_runner_with_dt(
    runner_name,
    functional,
    step,
    state,
    duration,
    dt,
    *,
    inputs=None,
    extra=None,
):
    if runner_name == "run":
        return dn.func.run(
            functional,
            step,
            state,
            inputs,
            tstop=duration,
            dt=dt,
            extra=extra,
        )
    return getattr(dn.func, runner_name)(
        functional,
        step,
        state,
        duration,
        2,
        inputs,
        dt=dt,
        extra=extra,
    )


@pytest.mark.parametrize(
    "executor_name",
    [
        "eager_step",
        "bound_functional_step",
        "compiled_tensor_step",
        "compiled_chunk_one",
        "bound_compiled_chunk_one",
    ],
)
@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_runners_preserve_state_loss_and_all_raw_gradients(
    executor_name,
    runner_name,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    expected = _execute(
        functional,
        tensors,
        ve,
        intra,
        "eager_step",
        "run",
    )

    with torch_compiler_warning_context():
        actual = _execute(
            functional,
            tensors,
            ve,
            intra,
            executor_name,
            runner_name,
        )

    actual_state, actual_loss, actual_gradients = actual
    expected_state, expected_loss, expected_gradients = expected
    _assert_tree_close(actual_state, expected_state, rtol=2.0e-10, atol=2.0e-11)
    torch.testing.assert_close(
        actual_loss,
        expected_loss,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in actual_gradients:
        assert torch.isfinite(actual_gradients[name]).all(), name
        assert torch.count_nonzero(actual_gradients[name]) > 0, name
        torch.testing.assert_close(
            actual_gradients[name],
            expected_gradients[name],
            rtol=2.0e-8,
            atol=2.0e-10,
            msg=lambda message: f"gradient {name}: {message}",
        )


@pytest.mark.parametrize(
    "executor_name",
    [
        "eager_step",
        "bound_functional_step",
        "compiled_tensor_step",
        "compiled_chunk_one",
        "bound_compiled_chunk_one",
    ],
)
def test_checkpointed_runner_retains_geometry_only_preparation_dependency(
    executor_name,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 3)
    expected = _execute(
        functional,
        tensors,
        ve,
        intra,
        "eager_step",
        "run",
        geometry_only=True,
    )

    with torch_compiler_warning_context():
        actual = _execute(
            functional,
            tensors,
            ve,
            intra,
            executor_name,
            "longrun_checkpointed",
            geometry_only=True,
        )

    actual_state, actual_loss, actual_gradients = actual
    expected_state, expected_loss, expected_gradients = expected
    _assert_tree_close(actual_state, expected_state, rtol=2.0e-10, atol=2.0e-11)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2.0e-10, atol=2.0e-11)
    actual_gradient = actual_gradients["diam"]
    assert torch.isfinite(actual_gradient).all()
    assert torch.count_nonzero(actual_gradient) > 0
    torch.testing.assert_close(
        actual_gradient,
        expected_gradients["diam"],
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_checkpointed_runner_reuses_one_preparation_during_backward(monkeypatch):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 3)
    parameters, constants, state, ve, intra, targets = _differentiable_case(
        tensors,
        ve,
        intra,
        geometry_only=True,
    )
    prepared = functional.prepare(parameters, constants)
    chunk = functional.compile_rollout_chunk(1, backend="aot_eager")

    def bound_chunk(local_state, step_inputs):
        return chunk(parameters, prepared, local_state, step_inputs)

    preparation_calls = 0
    original_prepare_values = functional._prepare_values

    def counted_prepare_values(local_parameters, local_constants):
        nonlocal preparation_calls
        preparation_calls += 1
        return original_prepare_values(local_parameters, local_constants)

    monkeypatch.setattr(functional, "_prepare_values", counted_prepare_values)
    with torch_compiler_warning_context():
        final, auxiliary = dn.func.longrun_checkpointed(
            functional,
            bound_chunk,
            state,
            3 * DT,
            2,
            dn.func.RolloutInput(ve, intra),
        )
        torch.autograd.grad(_loss(final, auxiliary), targets["diam"])

    assert preparation_calls == 0


@pytest.mark.parametrize(
    ("runner_name", "expected_state_validations"),
    [("run", 2), ("longrun", 4)],
)
@pytest.mark.parametrize("binding_kind", ["functional", "compiled"])
def test_owned_bound_step_amortizes_boundary_validation(
    monkeypatch,
    runner_name,
    expected_state_validations,
    binding_kind,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)
    if binding_kind == "functional":
        step = partial(functional.step, tensors.parameters, prepared)
    else:
        chunk = functional.compile_rollout_chunk(1, backend="aot_eager")
        step = partial(chunk, tensors.parameters, prepared)
    calls = {
        "_validate_source": 0,
        "_validate_parameters": 0,
        "_validate_prepared": 0,
        "_validate_state": 0,
    }

    for name in calls:
        original = getattr(functional, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(functional, name, counted)

    with torch_compiler_warning_context():
        final, auxiliary = _invoke_runner(
            runner_name,
            functional,
            step,
            tensors.state,
            dn.func.RolloutInput(ve, intra),
            steps=STEPS,
            chunklength=2,
        )

    assert auxiliary is not None
    assert final["clock"]["t"] > tensors.state["clock"]["t"]
    assert calls == {
        "_validate_source": 1,
        "_validate_parameters": 1,
        "_validate_prepared": 1,
        "_validate_state": expected_state_validations,
    }


def test_opaque_step_keeps_per_timestep_public_validation(monkeypatch):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)
    validation_calls = 0
    original = functional._validate_prepared

    def counted(*args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(functional, "_validate_prepared", counted)

    # A user wrapper may perform arbitrary work. It must remain on the public,
    # fully checked path; only an exact ordinary partial is optimized.
    def opaque_step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    dn.func.run(
        functional,
        opaque_step,
        tensors.state,
        dn.func.RolloutInput(ve, intra),
    )

    assert validation_calls == STEPS


def test_bound_functional_step_checks_freshness_even_for_zero_steps():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    prepared = functional.prepare(parameters, tensors.constants)
    step = partial(functional.step, parameters, prepared)
    with torch.no_grad():
        parameters[GNABAR].add_(1.0)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"parameters changed after prepare\(\)",
    ):
        dn.func.run(functional, step, tensors.state, tstop=0.0)


def test_checkpointed_bound_step_rechecks_freshness_during_backward():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    step = partial(functional.step, parameters, prepared)

    final, auxiliary = dn.func.longrun_checkpointed(
        functional,
        step,
        tensors.state,
        3 * DT,
        2,
    )
    loss = _loss(final, auxiliary)
    with torch.no_grad():
        parameters[GNABAR].add_(1.0)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"parameters changed after prepare\(\)",
    ):
        torch.autograd.grad(loss, parameters[GNABAR])


def test_duration_remainder_is_shared_and_direct_run_preserves_it():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    initial_time = tensors.state["clock"]["t"].clone()
    state, auxiliary = dn.func.run(
        functional,
        step,
        tensors.state,
        tstop=0.004,
    )
    assert auxiliary is None
    torch.testing.assert_close(state["clock"]["t"], initial_time)
    assert state["control"]["duration_remainder"].item() == pytest.approx(0.004)
    assert state["control"]["duration_remainder"].device.type == "cpu"
    assert state["control"]["duration_remainder"].dtype == torch.float64

    one_ve, one_intra = _drives(model, 1)
    state, _aux = dn.func.run(
        functional,
        step,
        state,
        dn.func.RolloutInput(one_ve, one_intra),
        tstop=1000.0,
    )
    torch.testing.assert_close(
        state["clock"]["t"],
        initial_time + initial_time.new_tensor(DT),
    )
    assert state["control"]["duration_remainder"].item() == pytest.approx(0.004)

    state, _aux = dn.func.longrun(
        functional,
        step,
        state,
        0.003,
        2,
    )
    assert state["control"]["duration_remainder"].item() == pytest.approx(0.007)
    state, _aux = dn.func.longrun_checkpointed(
        functional,
        step,
        state,
        0.003,
        2,
    )
    torch.testing.assert_close(
        state["clock"]["t"],
        initial_time + initial_time.new_tensor(2 * DT),
    )
    assert state["control"]["duration_remainder"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_explicit_runner_dt_controls_duration_budget(runner_name):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    runner_dt = 2 * DT
    constants = _clone_mapping(tensors.constants)
    constants["dt"] = constants["dt"].new_tensor(runner_dt)
    prepared = functional.prepare(tensors.parameters, constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    final, auxiliary = _invoke_duration_runner_with_dt(
        runner_name,
        functional,
        step,
        tensors.state,
        2.5 * runner_dt,
        runner_dt,
    )

    assert auxiliary is not None
    torch.testing.assert_close(
        final["clock"]["t"],
        tensors.state["clock"]["t"] + 2 * runner_dt,
    )
    assert final["control"]["duration_remainder"].item() == pytest.approx(
        0.5 * runner_dt
    )


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_host_runners_reject_bound_step_with_mismatched_runner_dt(runner_name):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    step = partial(
        functional.step,
        tensors.parameters,
        prepared,
    )
    runner_dt = 2 * DT

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="clock advance does not match runner dt",
    ):
        _invoke_duration_runner_with_dt(
            runner_name,
            functional,
            step,
            tensors.state,
            2 * runner_dt,
            runner_dt,
        )


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_host_runners_reject_bound_step_with_mismatched_default_dt(runner_name):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    constants = _clone_mapping(tensors.constants)
    constants["dt"] = constants["dt"].new_tensor(2 * DT)
    prepared = functional.prepare(tensors.parameters, constants)
    step = partial(
        functional.step,
        tensors.parameters,
        prepared,
    )

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="clock advance does not match runner dt",
    ):
        _invoke_runner(
            runner_name,
            functional,
            step,
            tensors.state,
            None,
            steps=2,
        )


@pytest.mark.parametrize("start_time", [1000.0, 10000.0])
def test_float32_clock_guard_accepts_matching_dt_at_nonzero_clock(start_time):
    functional, tensors = dn.func.make_functional(
        _model(dtype=torch.float32),
        dt=DT,
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = _clone_state(tensors.state)
    state["clock"]["t"] = state["clock"]["t"].new_tensor(start_time)
    step = partial(functional.step, tensors.parameters, prepared)

    final, _auxiliary = dn.func.run(
        functional,
        step,
        state,
        tstop=10 * DT,
    )

    assert final["clock"]["t"] > state["clock"]["t"]


@pytest.mark.parametrize(
    ("start_time", "step_dt"),
    [(1000.0, 1.1 * DT), (10000.0, 2.0 * DT)],
)
def test_float32_clock_guard_rejects_opaque_mismatch_at_nonzero_clock(
    start_time,
    step_dt,
):
    functional, tensors = dn.func.make_functional(
        _model(dtype=torch.float32),
        dt=DT,
    )
    constants = _clone_mapping(tensors.constants)
    constants["dt"] = constants["dt"].new_tensor(step_dt)
    prepared = functional.prepare(tensors.parameters, constants)
    state = _clone_state(tensors.state)
    state["clock"]["t"] = state["clock"]["t"].new_tensor(start_time)

    # Deliberately opaque: the fallback must diagnose these material
    # mismatches from total clock motion rather than prepared-plan inspection.
    def step(local_state, inputs):
        return functional.step(tensors.parameters, prepared, local_state, inputs)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="clock advance does not match runner dt",
    ):
        dn.func.run(
            functional,
            step,
            state,
            tstop=10 * DT,
        )


def test_float32_visible_binding_catches_dt_hidden_by_clock_quantization():
    functional, tensors = dn.func.make_functional(
        _model(dtype=torch.float32),
        dt=DT,
    )
    constants = _clone_mapping(tensors.constants)
    constants["dt"] = constants["dt"].new_tensor(1.1 * DT)
    prepared = functional.prepare(tensors.parameters, constants)
    state = _clone_state(tensors.state)
    state["clock"]["t"] = state["clock"]["t"].new_tensor(10000.0)
    step = partial(functional.step, tensors.parameters, prepared)

    # At this clock magnitude both timesteps round to the same float32 clock
    # increment. Public partial bindings still expose the exact prepared dt.
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="bound prepared dt",
    ):
        dn.func.run(
            functional,
            step,
            state,
            tstop=10 * DT,
        )


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_explicit_runner_dt_controls_runtime_extra_waveform_times(runner_name):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    runner_dt = 2 * DT
    steps = 3
    constants = _clone_mapping(tensors.constants)
    constants["dt"] = constants["dt"].new_tensor(runner_dt)
    prepared = functional.prepare(tensors.parameters, constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    field = torch.linspace(
        -1.0,
        1.0,
        model.v.numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(model.shape)
    waveform = dn.sin(amp=0.2, freq=7.0, phase=0.3)
    extra = functional.make_extra((field, waveform))
    stimulation = extra.extract()
    times = (
        tensors.state["clock"]["t"].to(torch.float64)
        + torch.arange(steps, dtype=torch.float64) * runner_dt
    ).to(model.dtype())
    explicit_ve = extra.assemble_tensors(stimulation, times)
    nominal_times = (
        tensors.state["clock"]["t"].to(torch.float64)
        + torch.arange(steps, dtype=torch.float64) * functional.dt
    ).to(model.dtype())
    nominal_ve = extra.assemble_tensors(stimulation, nominal_times)
    assert not torch.equal(explicit_ve, nominal_ve)

    runtime, _ = _invoke_duration_runner_with_dt(
        runner_name,
        functional,
        step,
        tensors.state,
        steps * runner_dt,
        runner_dt,
        extra=extra,
    )
    precomputed, _ = _invoke_duration_runner_with_dt(
        runner_name,
        functional,
        step,
        tensors.state,
        steps * runner_dt,
        runner_dt,
        inputs=dn.func.RolloutInput(ve=explicit_ve),
    )

    _assert_tree_close(runtime, precomputed, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
@pytest.mark.parametrize("invalid_dt", [0.0, -DT, float("nan"), float("inf"), True])
def test_host_runners_reject_invalid_explicit_dt(runner_name, invalid_dt):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    with pytest.raises((TypeError, ValueError), match="dt"):
        _invoke_duration_runner_with_dt(
            runner_name,
            functional,
            step,
            tensors.state,
            DT,
            invalid_dt,
        )


def test_runner_slices_time_first_inputs_into_exact_step_frames():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, 3)
    observed = []

    def recording_step(state, inputs):
        observed.append((inputs.ve.detach().clone(), inputs.intra.detach().clone()))
        return functional.step(tensors.parameters, prepared, state, inputs)

    dn.func.longrun(
        functional,
        recording_step,
        tensors.state,
        3 * DT,
        2,
        dn.func.RolloutInput(ve, intra),
    )

    assert len(observed) == 3
    for index, (observed_ve, observed_intra) in enumerate(observed):
        assert observed_ve.shape == model.shape
        assert observed_intra.shape == model.shape
        torch.testing.assert_close(observed_ve, ve[index], rtol=0.0, atol=0.0)
        torch.testing.assert_close(observed_intra, intra[index], rtol=0.0, atol=0.0)


def test_one_step_compiled_chunk_is_directly_bindable_to_runner():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, 2)
    chunk = functional.compile_rollout_chunk(1, backend="aot_eager")
    seen_input_types = []

    def bound_chunk(state, inputs):
        seen_input_types.append(type(inputs))
        return chunk(tensors.parameters, prepared, state, inputs)

    with torch_compiler_warning_context():
        actual, _aux = dn.func.run(
            functional,
            bound_chunk,
            tensors.state,
            dn.func.RolloutInput(ve, intra),
        )
    expected, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve, intra),
    )

    assert seen_input_types == [dn.func.StepInput, dn.func.StepInput]
    _assert_tree_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_host_runners_fail_closed_inside_outer_compile(runner_name):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    def scheduled(state):
        final, _aux = _invoke_runner(
            runner_name,
            functional,
            step,
            state,
            None,
            steps=1,
        )
        return final["integrator"]["v"]

    compiled = torch.compile(
        scheduled,
        backend="eager",
        fullgraph=True,
    )
    with pytest.raises(Exception, match="host schedulers cannot be passed"):
        compiled(tensors.state)


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_host_runners_fail_closed_inside_vmap(runner_name):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    initial_voltage = tensors.state["integrator"]["v"]
    lanes = torch.stack((initial_voltage - 0.25, initial_voltage + 0.25))

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    def scheduled(voltage):
        final, _aux = _invoke_runner(
            runner_name,
            functional,
            step,
            _replace_voltage(tensors.state, voltage),
            None,
            steps=1,
        )
        return final["integrator"]["v"]

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="not a torch.func transform",
    ):
        torch.func.vmap(scheduled)(lanes)


def test_longrun_rejects_time_first_drives_that_do_not_cover_duration():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, _intra = _drives(model, 2)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    with pytest.raises(ValueError, match="rollout expects 3"):
        dn.func.longrun(
            functional,
            step,
            tensors.state,
            3 * DT,
            2,
            dn.func.RolloutInput(ve=ve),
        )


@pytest.mark.parametrize("executor_name", ["eager", "compiled"])
def test_runner_rejects_visible_step_plan_mismatch(executor_name):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    other, other_tensors = dn.func.make_functional(_model(), dt=2 * DT)
    other_prepared = other.prepare(
        other_tensors.parameters,
        other_tensors.constants,
    )
    if executor_name == "eager":
        step = partial(
            other.step,
            other_tensors.parameters,
            other_prepared,
        )
    else:
        compiled_one = other.compile_rollout_chunk(1, backend="eager")
        step = partial(
            compiled_one,
            other_tensors.parameters,
            other_prepared,
        )

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="different FunctionalPopulation plan",
    ):
        dn.func.run(functional, step, tensors.state, tstop=2 * DT)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("dt", 2 * DT),
        ("shape", (1,)),
        ("dtype", torch.float32),
        ("device", torch.device("meta")),
    ],
)
def test_functional_execution_plan_metadata_is_read_only(name, value):
    functional, _tensors = dn.func.make_functional(_model(), dt=DT)

    with pytest.raises(AttributeError):
        setattr(functional, name, value)


@pytest.mark.parametrize(
    ("replacement", "error_type", "error_match"),
    [
        (
            torch.tensor(float("nan"), dtype=torch.float64),
            ValueError,
            "finite and non-negative",
        ),
        (
            torch.tensor(-0.1, dtype=torch.float64),
            ValueError,
            "finite and non-negative",
        ),
        (
            torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
            dn.func.FunctionalizationError,
            "must not require gradients",
        ),
        (
            torch.tensor(0.25, dtype=torch.float64),
            dn.func.FunctionalizationError,
            "changed host-owned",
        ),
    ],
)
def test_runner_rejects_invalid_or_step_modified_duration_carry(
    replacement,
    error_type,
    error_match,
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def corrupting_step(state, inputs):
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            inputs,
        )
        state = dict(state)
        state["control"] = dict(state["control"])
        state["control"]["duration_remainder"] = replacement
        return state, auxiliary

    with pytest.raises(error_type, match=error_match):
        dn.func.longrun(
            functional,
            corrupting_step,
            tensors.state,
            DT,
            1,
        )


def test_checkpointed_runner_materializes_inference_state_and_drives():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    with torch.inference_mode():
        state = torch.utils._pytree.tree_map(
            lambda value: value.clone(),
            tensors.state,
        )
        ve, intra = _drives(model, 3)

    def step(local_state, inputs):
        return functional.step(parameters, prepared, local_state, inputs)

    final, auxiliary = dn.func.longrun_checkpointed(
        functional,
        step,
        state,
        3 * DT,
        2,
        dn.func.RolloutInput(ve, intra),
    )
    gradient = torch.autograd.grad(_loss(final, auxiliary), parameters[GNABAR])[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_checkpointed_runner_bypasses_checkpoint_when_gradients_are_disabled(
    monkeypatch,
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    def unexpected_checkpoint(*_args, **_kwargs):
        raise AssertionError("checkpoint should not run under no_grad")

    monkeypatch.setattr("dendra.func._runners.checkpoint", unexpected_checkpoint)
    with torch.no_grad():
        final, auxiliary = dn.func.longrun_checkpointed(
            functional,
            step,
            tensors.state,
            2 * DT,
            1,
        )

    assert auxiliary is not None
    torch.testing.assert_close(
        final["clock"]["t"],
        tensors.state["clock"]["t"] + 2 * DT,
    )


def test_runner_validates_host_arguments_and_step_result_contract():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def step(state, inputs):
        return functional.step(tensors.parameters, prepared, state, inputs)

    with pytest.raises(TypeError, match="functional must be"):
        dn.func.run(object(), step, tensors.state, tstop=DT)
    with pytest.raises(TypeError, match="step must be callable"):
        dn.func.run(functional, None, tensors.state, tstop=DT)
    with pytest.raises(ValueError, match="tstop must be provided"):
        dn.func.run(functional, step, tensors.state)
    with pytest.raises(TypeError, match="RolloutInput"):
        dn.func.run(
            functional,
            step,
            tensors.state,
            dn.func.StepInput(),
            tstop=DT,
        )
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            dn.func.longrun(
                functional,
                step,
                tensors.state,
                DT,
                invalid,
            )

    def invalid_step(_state, _inputs):
        return tensors.state

    with pytest.raises(TypeError, match=r"exactly \(next_state, auxiliary\)"):
        dn.func.run(functional, invalid_step, tensors.state, tstop=DT)


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_zero_duration_runner_returns_an_independent_canonical_state(runner_name):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)

    def step(state, inputs):
        raise AssertionError("a zero-duration runner must not call step")

    if runner_name == "run":
        final, auxiliary = dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=0.0,
        )
    else:
        final, auxiliary = getattr(dn.func, runner_name)(
            functional,
            step,
            tensors.state,
            0.0,
            2,
        )

    assert auxiliary is None
    _assert_tree_close(final, tensors.state)
    for actual, original in zip(
        torch.utils._pytree.tree_leaves(final),
        torch.utils._pytree.tree_leaves(tensors.state),
        strict=True,
    ):
        assert actual is not original
        assert actual.data_ptr() != original.data_ptr()
