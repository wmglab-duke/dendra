from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import APCount as ImperativeAPCount
from dendra.models.callbacks import Raster as ImperativeRaster
from dendra.models.callbacks import Recorder
from dendra.models.mod import hh
from dendra.units import nA

DT = 0.005
FunctionalRecorder = dn.func.Recorder
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"
GKBAR = "integrator.mech.mechanisms.hh.gkbar_param"


def _model(*, gnabar=0.05, gkbar=0.05, trainable=False):
    model = dn.Unmyelinated(
        [2.0],
        L=40.0,
        dx=10.0,
        celsius=6.3,
        v_init=-65.0,
        rhoa=100.0,
        integrator=dn.bwd_euler_ub(method="thomas", imem=False),
    ).double()
    model.insert(hh, gnabar=gnabar, gkbar=gkbar)
    model[0, 2].inject(
        dn.mono_rect(amp=2.0 * nA, delay=0.01, pw=0.03),
    )
    model.train()
    if trainable:
        model.build()
        model.unfreeze_("hh.gnabar", "hh.gkbar")
    model.initialize()
    return model


def _imperative_recording(model, recorder, steps):
    model.run(tstop=steps * DT, dt=DT, callbacks=[recorder])
    return {name: recorder.stack(name) for name in recorder.states}


def _functional_case(*, trainable=False):
    model = _model(trainable=trainable)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    return model, functional, tensors


def _replace_voltage(state, voltage):
    next_state = dict(state)
    next_state["integrator"] = dict(state["integrator"])
    next_state["integrator"]["v"] = voltage
    return next_state


def _threshold_sequence_run(
    runner,
    functional,
    tensors,
    callbacks,
    voltage,
    *,
    state=None,
    callback_state=None,
    chunklength=2,
):
    if state is None:
        state = tensors.state
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def prescribed_voltage_step(local_state, inputs):
        next_state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            local_state,
            dn.func.StepInput(),
        )
        next_state = _replace_voltage(next_state, inputs.ve)
        return next_state, {**auxiliary, "v": inputs.ve}

    inputs = dn.func.RolloutInput(ve=voltage)
    if runner == "run":
        return dn.func.run(
            functional,
            prescribed_voltage_step,
            state,
            inputs,
            callbacks=callbacks,
            callback_state=callback_state,
        )
    return getattr(dn.func, runner)(
        functional,
        prescribed_voltage_step,
        state,
        voltage.shape[0] * DT,
        chunklength,
        inputs,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def _run_with_callbacks(
    runner,
    functional,
    tensors,
    callbacks,
    steps,
    *,
    parameters=None,
    callback_state=None,
    compiled=False,
    chunklength=3,
):
    if parameters is None:
        parameters = tensors.parameters
    prepared = functional.prepare(parameters, tensors.constants)
    if compiled:
        one_step = functional.compile_rollout_chunk(1, backend="aot_eager")
        step = partial(one_step, parameters, prepared)
    else:
        step = partial(functional.step, parameters, prepared)
    if runner == "run":
        return dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=steps * DT,
            callbacks=callbacks,
            callback_state=callback_state,
        )
    return getattr(dn.func, runner)(
        functional,
        step,
        tensors.state,
        steps * DT,
        chunklength,
        callbacks=callbacks,
        callback_state=callback_state,
    )


@pytest.mark.parametrize("runner", ["run", "longrun", "longrun_checkpointed"])
@pytest.mark.parametrize("steps", [1, 7])
def test_functional_recorder_matches_initial_and_post_step_values(runner, steps):
    expected_model = _model()
    expected_recorder = Recorder(["v", "hh.m", "t"])
    expected = _imperative_recording(expected_model, expected_recorder, steps)

    source, functional, tensors = _functional_case()
    recorder = FunctionalRecorder(["v", "hh.m", "t"])
    callbacks = functional.make_callbacks({"trace": recorder})
    source_state = torch.utils._pytree.tree_map(torch.clone, tensors.state)

    _final, auxiliary = _run_with_callbacks(
        runner,
        functional,
        tensors,
        callbacks,
        steps,
    )
    result = auxiliary["callbacks"]

    assert result.state["trace"] == ()
    for name in expected:
        torch.testing.assert_close(result["trace"][name], expected[name])
    assert source.t.item() == 0.0
    for actual, expected_leaf in zip(
        torch.utils._pytree.tree_leaves(tensors.state),
        torch.utils._pytree.tree_leaves(source_state),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected_leaf)


@pytest.mark.parametrize("compiled", [False, True])
def test_functional_recorder_composes_with_eager_and_compiled_steps(compiled):
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    _final, auxiliary = _run_with_callbacks(
        "longrun",
        functional,
        tensors,
        callbacks,
        5,
        compiled=compiled,
        chunklength=2,
    )
    assert auxiliary["callbacks"]["trace"]["v"].shape == (6, 1, 5)


def test_fixed_rollout_recorder_matches_host_runner_exactly():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v", "hh.m"])})
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    fixed_state, fixed_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=7,
        callbacks=callbacks,
    )
    host_state, host_auxiliary = _run_with_callbacks(
        "longrun",
        functional,
        tensors,
        callbacks,
        7,
        chunklength=3,
    )

    for actual, expected in zip(
        torch.utils._pytree.tree_leaves(fixed_state),
        torch.utils._pytree.tree_leaves(host_state),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for state_name in ("v", "hh.m"):
        torch.testing.assert_close(
            fixed_auxiliary["callbacks"]["trace"][state_name],
            host_auxiliary["callbacks"]["trace"][state_name],
            rtol=0.0,
            atol=0.0,
        )
    assert fixed_auxiliary["callbacks"].state["trace"] == ()


def test_fixed_rollout_recorder_resume_and_zero_step_lifecycle():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    zero_state, zero_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    assert zero_auxiliary["callbacks"]["trace"]["v"].shape == (1, 1, 5)
    assert zero_auxiliary["callbacks"].state["trace"] == ()

    first_state, first_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        zero_state,
        steps=3,
        callbacks=callbacks,
        callback_state=zero_auxiliary["callbacks"].state,
    )
    final_state, second_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        first_state,
        steps=4,
        callbacks=callbacks,
        callback_state=first_auxiliary["callbacks"].state,
    )
    resumed = torch.cat(
        (
            zero_auxiliary["callbacks"]["trace"]["v"],
            first_auxiliary["callbacks"]["trace"]["v"],
            second_auxiliary["callbacks"]["trace"]["v"],
        )
    )
    expected_state, expected_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=7,
        callbacks=callbacks,
    )

    torch.testing.assert_close(
        resumed,
        expected_auxiliary["callbacks"]["trace"]["v"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        final_state["integrator"]["v"],
        expected_state["integrator"]["v"],
        rtol=0.0,
        atol=0.0,
    )


def test_fixed_rollout_trace_loss_supports_torch_func_grad_and_hessian():
    steps = 8
    _reference_source, reference, reference_tensors = _functional_case()
    reference_callbacks = reference.make_callbacks({"trace": FunctionalRecorder(["v"])})
    reference_parameters = dict(reference_tensors.parameters)
    reference_parameters[GNABAR] = reference_parameters[GNABAR].new_tensor(0.12)
    with torch.no_grad():
        _state, reference_auxiliary = reference.prepare_and_rollout(
            reference_parameters,
            reference_tensors.constants,
            reference_tensors.state,
            steps=steps,
            callbacks=reference_callbacks,
        )
        target = reference_auxiliary["callbacks"]["trace"]["v"]

    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})

    def loss(gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        _state, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=steps,
            callbacks=callbacks,
        )
        return torch.mean((auxiliary["callbacks"]["trace"]["v"] - target).square())

    gnabar = tensors.parameters[GNABAR]
    gradient = torch.func.grad(loss)(gnabar)
    nested_gradient = torch.func.grad(torch.func.grad(loss))(gnabar)
    hessian = torch.func.jacrev(torch.func.grad(loss))(gnabar)

    assert torch.isfinite(gradient)
    assert torch.isfinite(hessian)
    assert gradient.abs() > 0
    assert hessian.abs() > 0
    torch.testing.assert_close(nested_gradient, hessian)


def test_recorded_loss_supports_selected_parameter_pytree_hessian():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    independent = tensors.independent_parameters(trainable=(GNABAR, GKBAR))
    selected = {name: independent[name] for name in (GNABAR, GKBAR)}
    fixed = {name: value for name, value in independent.items() if name not in selected}

    def loss(selected_parameters):
        parameters = {**fixed, **selected_parameters}
        _state, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=3,
            callbacks=callbacks,
        )
        return auxiliary["callbacks"]["trace"]["v"].square().mean()

    with torch_compiler_warning_context():
        hessian = torch.func.hessian(loss)(selected)

    assert set(hessian) == set(selected)
    assert all(set(row) == set(selected) for row in hessian.values())
    assert all(
        torch.isfinite(block) for row in hessian.values() for block in row.values()
    )
    torch.testing.assert_close(hessian[GNABAR][GKBAR], hessian[GKBAR][GNABAR])


def test_compile_of_nested_grad_over_fixed_recording_matches_eager_transform():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})

    def loss(gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        first_state, first_auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=1,
            callbacks=callbacks,
        )
        _state, second_auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            first_state,
            steps=1,
            callbacks=callbacks,
            callback_state=first_auxiliary["callbacks"].state,
        )
        trace = torch.cat(
            (
                first_auxiliary["callbacks"]["trace"]["v"],
                second_auxiliary["callbacks"]["trace"]["v"],
            )
        )
        return trace.square().mean()

    transformed = torch.func.grad(torch.func.grad(loss))
    expected = transformed(tensors.parameters[GNABAR])
    compiled = torch.compile(
        transformed,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        actual = compiled(tensors.parameters[GNABAR])

    torch.testing.assert_close(actual, expected)


def test_compiled_resume_reuses_one_graph_across_callback_carry_instances():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})

    first_state, first_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=1,
        callbacks=callbacks,
    )
    second_state, second_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        first_state,
        steps=1,
        callbacks=callbacks,
        callback_state=first_auxiliary["callbacks"].state,
    )
    compile_count = 0

    def backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def resume(state, callback_state):
        final_state, auxiliary = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            steps=1,
            callbacks=callbacks,
            callback_state=callback_state,
        )
        results = auxiliary["callbacks"]
        return final_state, results.state, results["trace"]["v"]

    expected_first = resume(first_state, first_auxiliary["callbacks"].state)
    expected_second = resume(second_state, second_auxiliary["callbacks"].state)
    compiled = torch.compile(resume, backend=backend, fullgraph=True, dynamic=False)
    with torch_compiler_warning_context():
        actual_first = compiled(first_state, first_auxiliary["callbacks"].state)
        actual_second = compiled(second_state, second_auxiliary["callbacks"].state)

    for actual, expected in zip(
        torch.utils._pytree.tree_leaves((actual_first, actual_second)),
        torch.utils._pytree.tree_leaves((expected_first, expected_second)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    assert compile_count == 1


def test_vmap_can_resume_model_and_callback_carry_without_boundary_duplication():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    base = tensors.parameters[GNABAR]
    lanes = torch.stack((0.8 * base, 1.2 * base))

    def parameters_for(gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        return parameters

    def first(gnabar):
        parameters = parameters_for(gnabar)
        state, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=2,
            callbacks=callbacks,
        )
        results = auxiliary["callbacks"]
        return state, results.state, results["trace"]["v"]

    first_states, callback_states, first_traces = torch.vmap(first)(lanes)

    def resume(gnabar, state, callback_state):
        parameters = parameters_for(gnabar)
        final, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            state,
            steps=2,
            callbacks=callbacks,
            callback_state=callback_state,
        )
        return final["integrator"]["v"], auxiliary["callbacks"]["trace"]["v"]

    final_voltages, second_traces = torch.vmap(resume)(
        lanes,
        first_states,
        callback_states,
    )
    resumed_traces = torch.cat((first_traces, second_traces), dim=1)

    def complete(gnabar):
        parameters = parameters_for(gnabar)
        final, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=4,
            callbacks=callbacks,
        )
        return final["integrator"]["v"], auxiliary["callbacks"]["trace"]["v"]

    expected_voltages, expected_traces = torch.vmap(complete)(lanes)
    torch.testing.assert_close(final_voltages, expected_voltages)
    torch.testing.assert_close(resumed_traces, expected_traces)


def test_functional_recorder_resume_carry_avoids_duplicate_boundary_frame():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    step = partial(functional.step, tensors.parameters, prepared)

    first_state, first_auxiliary = dn.func.longrun(
        functional,
        step,
        tensors.state,
        3 * DT,
        2,
        callbacks=callbacks,
    )
    first_results = first_auxiliary["callbacks"]
    final_state, second_auxiliary = dn.func.longrun(
        functional,
        step,
        first_state,
        4 * DT,
        3,
        callbacks=callbacks,
        callback_state=first_results.state,
    )
    resumed = torch.cat(
        (
            first_results["trace"]["v"],
            second_auxiliary["callbacks"]["trace"]["v"],
        )
    )

    expected_state, expected_auxiliary = dn.func.longrun(
        functional,
        step,
        tensors.state,
        7 * DT,
        4,
        callbacks=callbacks,
    )
    torch.testing.assert_close(
        resumed,
        expected_auxiliary["callbacks"]["trace"]["v"],
    )
    torch.testing.assert_close(
        final_state["integrator"]["v"],
        expected_state["integrator"]["v"],
    )
    assert second_auxiliary["callbacks"].state["trace"] == ()


def test_functional_recorder_selection_matches_imperative_recorder():
    expected_indexed = _imperative_recording(
        _model(),
        Recorder(["v"], node_indices=[0, 3]),
        4,
    )["v"]
    expected_partitioned = _imperative_recording(
        _model(),
        Recorder(["v"], max_only=True, partition=[2, 3]),
        4,
    )["v"]
    expected_full = _imperative_recording(
        _model(),
        Recorder(["v"], partition=[999]),
        4,
    )["v"]
    expected_max = _imperative_recording(
        _model(),
        Recorder(["v"], max_only=True, node_indices=[0]),
        4,
    )["v"]

    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks(
        {
            "indexed": FunctionalRecorder(["v"], node_indices=[0, 3]),
            "partitioned": FunctionalRecorder(["v"], max_only=True, partition=[2, 3]),
            "partition_ignored": FunctionalRecorder(["v"], partition=[999]),
            "indices_ignored": FunctionalRecorder(
                ["v"], max_only=True, node_indices=[0]
            ),
        }
    )
    _final, auxiliary = _run_with_callbacks(
        "longrun", functional, tensors, callbacks, 4, chunklength=3
    )
    results = auxiliary["callbacks"]
    torch.testing.assert_close(results["indexed"]["v"], expected_indexed)
    torch.testing.assert_close(results["partitioned"]["v"], expected_partitioned)
    torch.testing.assert_close(results["partition_ignored"]["v"], expected_full)
    torch.testing.assert_close(results["indices_ignored"]["v"], expected_max)


def test_functional_recorder_zero_step_lifecycle_matches_runner_families():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    step = partial(functional.step, tensors.parameters, prepared)

    _state, run_auxiliary = dn.func.run(
        functional,
        step,
        tensors.state,
        tstop=0.0,
        callbacks=callbacks,
    )
    assert run_auxiliary["callbacks"]["trace"]["v"].shape[0] == 1

    for runner in (dn.func.longrun, dn.func.longrun_checkpointed):
        zero_state, auxiliary = runner(
            functional,
            step,
            tensors.state,
            0.0,
            2,
            callbacks=callbacks,
        )
        assert auxiliary["callbacks"]["trace"]["v"].shape == (1, 1, 5)
        assert auxiliary["callbacks"].state["trace"] == ()

        _state, resumed_auxiliary = runner(
            functional,
            step,
            zero_state,
            DT,
            2,
            callbacks=callbacks,
            callback_state=auxiliary["callbacks"].state,
        )
        assert resumed_auxiliary["callbacks"]["trace"]["v"].shape == (1, 1, 5)


def test_functional_recorder_trace_loss_and_conductance_gradients_match_imperative():
    steps = 20
    target = _imperative_recording(
        _model(gnabar=0.12, gkbar=0.036),
        Recorder(["v"]),
        steps,
    )["v"].detach()

    imperative = _model(trainable=True)
    imperative_recorder = Recorder(["v"])
    actual_imperative = _imperative_recording(
        imperative,
        imperative_recorder,
        steps,
    )["v"]
    imperative_loss = torch.mean((actual_imperative - target).square())
    imperative_targets = (
        imperative.integrator.mech.mechanisms.hh.gnabar_param,
        imperative.integrator.mech.mechanisms.hh.gkbar_param,
    )
    imperative_gradients = torch.autograd.grad(imperative_loss, imperative_targets)

    _source, functional, tensors = _functional_case(trainable=True)
    parameters = dict(tensors.parameters)
    parameters[GNABAR] = parameters[GNABAR].detach().clone().requires_grad_()
    parameters[GKBAR] = parameters[GKBAR].detach().clone().requires_grad_()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    _state, auxiliary = _run_with_callbacks(
        "longrun_checkpointed",
        functional,
        tensors,
        callbacks,
        steps,
        parameters=parameters,
        chunklength=6,
    )
    actual_functional = auxiliary["callbacks"]["trace"]["v"]
    functional_loss = torch.mean((actual_functional - target).square())
    functional_gradients = torch.autograd.grad(
        functional_loss,
        (parameters[GNABAR], parameters[GKBAR]),
    )

    torch.testing.assert_close(actual_functional, actual_imperative)
    torch.testing.assert_close(functional_loss, imperative_loss)
    for actual, expected in zip(
        functional_gradients,
        imperative_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_functional_recorder_one_adam_update_matches_imperative_tutorial():
    steps = 20
    target = _imperative_recording(
        _model(gnabar=0.12, gkbar=0.036),
        Recorder(["v"]),
        steps,
    )["v"].detach()

    imperative = _model(trainable=True)
    imperative_optimizer = torch.optim.Adam(imperative.parameters(), lr=5e-3)
    imperative_optimizer.zero_grad()
    imperative_trace = _imperative_recording(
        imperative,
        Recorder(["v"]),
        steps,
    )["v"]
    torch.mean((imperative_trace - target).square()).backward()
    imperative_optimizer.step()

    _source, functional, tensors = _functional_case(trainable=True)
    parameters = tensors.independent_parameters(
        trainable=("gnabar", "gkbar"),
        within="model",
    )
    trainable = {name: parameters[name] for name in (GNABAR, GKBAR)}
    fixed = {name: value for name, value in parameters.items() if name not in trainable}
    optimizer = torch.optim.Adam(trainable.values(), lr=5e-3)
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})

    def loss(selected):
        local_parameters = {**fixed, **selected}
        _state, auxiliary = functional.prepare_and_rollout(
            local_parameters,
            tensors.constants,
            tensors.state,
            steps=steps,
            callbacks=callbacks,
        )
        return torch.mean((auxiliary["callbacks"]["trace"]["v"] - target).square())

    optimizer.zero_grad()
    gradients = torch.func.grad(loss)(trainable)
    for name, parameter in trainable.items():
        parameter.grad = gradients[name]
    optimizer.step()

    torch.testing.assert_close(
        trainable[GNABAR],
        imperative.integrator.mech.mechanisms.hh.gnabar_param,
    )
    torch.testing.assert_close(
        trainable[GKBAR],
        imperative.integrator.mech.mechanisms.hh.gkbar_param,
    )


def test_functional_recorder_checkpoint_preserves_waveform_parameter_gradient():
    def execute(runner):
        _source, functional, tensors = _functional_case()
        parameters = dict(tensors.parameters)
        amplitude_name = next(
            name
            for name in parameters
            if name.startswith("stimulation.intra.") and name.endswith(".amp")
        )
        parameters[amplitude_name] = (
            parameters[amplitude_name].detach().clone().requires_grad_()
        )
        callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
        _state, auxiliary = _run_with_callbacks(
            runner,
            functional,
            tensors,
            callbacks,
            9,
            parameters=parameters,
            chunklength=4,
        )
        trace = auxiliary["callbacks"]["trace"]["v"]
        weights = torch.linspace(0.5, 1.5, trace.shape[0], dtype=trace.dtype)
        loss = (trace * weights.reshape(-1, 1, 1)).square().mean()
        gradient = torch.autograd.grad(loss, parameters[amplitude_name])[0]
        return trace, loss, gradient

    ordinary = execute("longrun")
    checkpointed = execute("longrun_checkpointed")
    for actual, expected in zip(checkpointed, ordinary, strict=True):
        torch.testing.assert_close(actual, expected)
    assert torch.isfinite(checkpointed[-1])
    assert checkpointed[-1].abs() > 0


def test_inference_authored_recorder_indices_are_ordinary_for_backward():
    _source, functional, tensors = _functional_case()
    with torch.inference_mode():
        recorder = FunctionalRecorder(["v"], node_indices=[1, 3])
        callbacks = functional.make_callbacks({"trace": recorder})

    parameters = dict(tensors.parameters)
    parameters[GNABAR] = parameters[GNABAR].detach().clone().requires_grad_()
    _state, auxiliary = _run_with_callbacks(
        "longrun_checkpointed",
        functional,
        tensors,
        callbacks,
        5,
        parameters=parameters,
        chunklength=2,
    )
    loss = auxiliary["callbacks"]["trace"]["v"].square().mean()
    gradient = torch.autograd.grad(loss, parameters[GNABAR])[0]
    assert torch.isfinite(gradient)


def test_functional_recorder_preserves_explicit_population_batch_axes():
    model = _model()
    model.batch(2)
    model.initialize()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(
        {"trace": FunctionalRecorder(["v"], node_indices=[1, 4])}
    )
    _state, auxiliary = _run_with_callbacks(
        "longrun",
        functional,
        tensors,
        callbacks,
        3,
        chunklength=2,
    )
    assert auxiliary["callbacks"]["trace"]["v"].shape == (4, 2, 1, 2)


def _threshold_voltage_sequence(shape):
    voltage = torch.full((6, *shape), -1.0, dtype=torch.float64)
    selected = torch.tensor(
        [
            [-1.0, -1.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ],
        dtype=voltage.dtype,
    )
    voltage[..., 0] = selected[:, 0].reshape(6, *([1] * (voltage.ndim - 2)))
    voltage[..., -1] = selected[:, 1].reshape(6, *([1] * (voltage.ndim - 2)))
    return voltage


@pytest.mark.parametrize("runner", ["run", "longrun", "longrun_checkpointed"])
def test_functional_raster_and_apcount_match_imperative_crossings(runner):
    source, functional, tensors = _functional_case()
    voltage = _threshold_voltage_sequence(source.shape)
    callbacks = functional.make_callbacks(
        {
            "raster": dn.func.Raster(threshold=0.0, node_check=[0, -1]),
            "count": dn.func.APCount(threshold=0.0, node_check=[0, -1]),
        }
    )
    _final, auxiliary = _threshold_sequence_run(
        runner,
        functional,
        tensors,
        callbacks,
        voltage,
    )
    results = auxiliary["callbacks"]

    expected_model = _model()
    expected_raster = ImperativeRaster(
        threshold=0.0,
        node_check=[0, -1],
        dt=DT,
    )
    expected_count = ImperativeAPCount(
        threshold=0.0,
        node_check=[0, -1],
        dt=DT,
    )
    expected_raster.pre_loop_hook(expected_model)
    expected_count.pre_loop_hook(expected_model)
    for frame in voltage:
        expected_model.v.copy_(frame)
        expected_raster.post_step_hook(expected_model)
        expected_count.post_step_hook(expected_model)

    assert results["raster"].dtype == torch.bool
    assert results["count"].dtype == torch.int64
    torch.testing.assert_close(results["raster"], expected_raster.stack())
    torch.testing.assert_close(results["count"], expected_count.record.to(torch.int64))
    torch.testing.assert_close(
        results["count"],
        results["raster"].sum(dim=0),
    )
    below, count, step = results.state["count"]
    assert below.shape == count.shape == (1, 2)
    assert step.shape == ()


def test_functional_threshold_window_is_aligned_and_false_masked():
    source, functional, tensors = _functional_case()
    voltage = _threshold_voltage_sequence(source.shape)
    start_step, end_step = 2, 5
    callbacks = functional.make_callbacks(
        {
            "raster": dn.func.Raster(
                threshold=0.0,
                t_start_check=start_step * DT,
                t_end_check=end_step * DT,
                node_check=[0, -1],
                dt=DT,
            ),
            "count": dn.func.APCount(
                threshold=0.0,
                t_start_check=start_step * DT,
                t_end_check=end_step * DT,
                node_check=[0, -1],
                dt=DT,
            ),
        }
    )
    _final, auxiliary = _threshold_sequence_run(
        "run",
        functional,
        tensors,
        callbacks,
        voltage,
    )
    results = auxiliary["callbacks"]

    expected_model = _model()
    expected_raster = ImperativeRaster(
        threshold=0.0,
        t_start_check=start_step * DT,
        t_end_check=end_step * DT,
        node_check=[0, -1],
        dt=DT,
    )
    expected_count = ImperativeAPCount(
        threshold=0.0,
        t_start_check=start_step * DT,
        t_end_check=end_step * DT,
        node_check=[0, -1],
        dt=DT,
    )
    expected_raster.pre_loop_hook(expected_model)
    expected_count.pre_loop_hook(expected_model)
    for frame in voltage:
        expected_model.v.copy_(frame)
        expected_raster.post_step_hook(expected_model)
        expected_count.post_step_hook(expected_model)

    assert results["raster"].shape == (6, 1, 2)
    assert not torch.any(results["raster"][:start_step])
    assert not torch.any(results["raster"][end_step:])
    torch.testing.assert_close(
        results["raster"][start_step:end_step],
        expected_raster.stack(),
    )
    torch.testing.assert_close(results["count"], expected_count.record.to(torch.int64))


def test_functional_threshold_callbacks_resume_without_rearming_or_rephasing():
    source, functional, tensors = _functional_case()
    voltage = _threshold_voltage_sequence(source.shape)
    callbacks = functional.make_callbacks(
        {
            "raster": dn.func.Raster(threshold=0.0, node_check=[0, -1]),
            "count": dn.func.APCount(threshold=0.0, node_check=[0, -1]),
        }
    )

    zero_state, zero_auxiliary = dn.func.run(
        functional,
        partial(
            functional.step,
            tensors.parameters,
            functional.prepare(tensors.parameters, tensors.constants),
        ),
        tensors.state,
        tstop=0.0,
        callbacks=callbacks,
    )
    zero_results = zero_auxiliary["callbacks"]
    assert zero_results["raster"].shape == (0, 1, 2)
    assert torch.count_nonzero(zero_results["count"]) == 0

    first_state, first_auxiliary = _threshold_sequence_run(
        "longrun",
        functional,
        tensors,
        callbacks,
        voltage[:3],
        state=zero_state,
        callback_state=zero_results.state,
    )
    final_state, second_auxiliary = _threshold_sequence_run(
        "longrun_checkpointed",
        functional,
        tensors,
        callbacks,
        voltage[3:],
        state=first_state,
        callback_state=first_auxiliary["callbacks"].state,
    )
    _complete_state, complete_auxiliary = _threshold_sequence_run(
        "run",
        functional,
        tensors,
        callbacks,
        voltage,
    )

    resumed_raster = torch.cat(
        (
            first_auxiliary["callbacks"]["raster"],
            second_auxiliary["callbacks"]["raster"],
        )
    )
    torch.testing.assert_close(
        resumed_raster,
        complete_auxiliary["callbacks"]["raster"],
    )
    torch.testing.assert_close(
        second_auxiliary["callbacks"]["count"],
        complete_auxiliary["callbacks"]["count"],
    )
    torch.testing.assert_close(
        final_state["integrator"]["v"],
        voltage[-1],
    )

    _state, resumed_zero_auxiliary = dn.func.run(
        functional,
        partial(
            functional.step,
            tensors.parameters,
            functional.prepare(tensors.parameters, tensors.constants),
        ),
        final_state,
        tstop=0.0,
        callbacks=callbacks,
        callback_state=second_auxiliary["callbacks"].state,
    )
    assert resumed_zero_auxiliary["callbacks"]["raster"].shape == (0, 1, 2)
    torch.testing.assert_close(
        resumed_zero_auxiliary["callbacks"]["count"],
        second_auxiliary["callbacks"]["count"],
    )


def test_functional_threshold_callbacks_preserve_explicit_batch_axes():
    model = _model()
    model.batch(2)
    model.initialize()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _threshold_voltage_sequence(model.shape)
    voltage[:, 0, 0, 0] = -1.0
    callbacks = functional.make_callbacks(
        {
            "raster": dn.func.Raster(node_check=[0, -1]),
            "count": dn.func.APCount(node_check=[0, -1]),
        }
    )
    _state, auxiliary = _threshold_sequence_run(
        "longrun",
        functional,
        tensors,
        callbacks,
        voltage,
    )
    results = auxiliary["callbacks"]
    assert results["raster"].shape == (6, 2, 1, 2)
    assert results["count"].shape == (2, 1, 2)
    torch.testing.assert_close(results["count"], results["raster"].sum(dim=0))


def test_functional_threshold_node_selection_supports_scalar_duplicates_and_empty():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks(
        {
            "scalar": dn.func.Raster(threshold=-100.0, node_check=0),
            "duplicates": dn.func.Raster(
                threshold=-100.0,
                node_check=[0, 0],
            ),
            "empty": dn.func.APCount(threshold=-100.0, node_check=[]),
        }
    )
    _state, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=1,
        callbacks=callbacks,
    )
    results = auxiliary["callbacks"]
    assert results["scalar"].shape == (1, 1, 1)
    assert results["duplicates"].shape == (1, 1, 2)
    assert results["empty"].shape == (1, 0)
    torch.testing.assert_close(
        results["duplicates"],
        results["scalar"].expand(-1, -1, 2),
    )


def test_functional_threshold_callbacks_compose_with_vmap_and_fullgraph_compile():
    _source, functional, tensors = _functional_case()
    callbacks = functional.make_callbacks(
        {
            "raster": dn.func.Raster(threshold=-100.0, node_check=[0, -1]),
            "count": dn.func.APCount(threshold=-100.0, node_check=[0, -1]),
        }
    )
    base = tensors.parameters[GNABAR]

    def execute(gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        final, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=2,
            callbacks=callbacks,
        )
        results = auxiliary["callbacks"]
        return final["integrator"]["v"], results["raster"], results["count"]

    lanes = torch.stack((0.9 * base, 1.1 * base))
    actual_batched = torch.vmap(execute)(lanes)
    expected_batched = tuple(
        torch.stack(values)
        for values in zip(*(execute(lane) for lane in lanes), strict=True)
    )
    for actual, expected in zip(actual_batched, expected_batched, strict=True):
        torch.testing.assert_close(actual, expected)

    compiled = torch.compile(
        execute,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        actual_compiled = compiled(base)
    expected = execute(base)
    for actual, expected_value in zip(actual_compiled, expected, strict=True):
        torch.testing.assert_close(actual, expected_value)


@pytest.mark.parametrize("callback_type", [dn.func.Raster, dn.func.APCount])
def test_functional_threshold_callbacks_fail_closed_on_invalid_configuration(
    callback_type,
):
    _source, functional, tensors = _functional_case()
    with pytest.raises(IndexError, match="valid compartment indices"):
        functional.make_callbacks({"invalid": callback_type(node_check=[5])})
    with pytest.raises(dn.func.FunctionalizationError, match="dt must be positive"):
        functional.make_callbacks({"invalid": callback_type(node_check=[0], dt=0)})
    with pytest.raises(dn.func.FunctionalizationError, match="real scalars"):
        functional.make_callbacks(
            {"invalid": callback_type(node_check=[0], threshold="invalid")}
        )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="must match the FunctionalPopulation timestep",
    ):
        functional.make_callbacks({"invalid": callback_type(node_check=[0], dt=2 * DT)})

    callbacks = functional.make_callbacks({"threshold": callback_type(node_check=[0])})
    constants = dict(tensors.constants)
    constants["dt"] = constants["dt"].new_tensor(2 * DT)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="prepared timestep",
    ):
        functional.prepare_and_rollout(
            tensors.parameters,
            constants,
            tensors.state,
            steps=0,
            callbacks=callbacks,
        )

    prepared = functional.prepare(tensors.parameters, constants)
    step = partial(functional.step, tensors.parameters, prepared)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="runtime timestep",
    ):
        dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=2 * DT,
            dt=2 * DT,
            callbacks=callbacks,
        )


def test_functional_recorder_fails_closed_on_unsupported_or_stale_configuration():
    _source, functional, tensors = _functional_case()

    class CustomRecorder(Recorder):
        pass

    with pytest.raises(TypeError, match="FunctionalCallback"):
        functional.make_callbacks({"trace": CustomRecorder(["v"])})
    with pytest.raises(dn.func.FunctionalizationError, match="explicit transition"):
        functional.make_callbacks({"trace": FunctionalRecorder(["diam"])})
    with pytest.raises(dn.func.FunctionalizationError, match="does not yet support"):
        functional.make_callbacks({"trace": FunctionalRecorder(["v"], dt=0.01)})
    with pytest.raises(dn.func.FunctionalizationError, match="sliding_window"):
        functional.make_callbacks(
            {"trace": FunctionalRecorder(["v"], sliding_window=3)}
        )
    with pytest.raises(dn.func.FunctionalizationError, match="non-scalar state"):
        functional.make_callbacks({"trace": FunctionalRecorder(["t"], max_only=True)})

    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    step = partial(functional.step, tensors.parameters, prepared)
    with pytest.raises(TypeError, match="FunctionalCallbackState"):
        dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=DT,
            callbacks=callbacks,
            callback_state={"wrong": torch.zeros((), dtype=torch.int64)},
        )

    _zero_state, zero_auxiliary = dn.func.run(
        functional,
        step,
        tensors.state,
        tstop=0.0,
        callbacks=callbacks,
    )
    initial_callback_state = zero_auxiliary["callbacks"].state
    wrong_keys = dn.func.FunctionalCallbackState(
        carries={"wrong": torch.zeros((), dtype=torch.int64)},
        _token=initial_callback_state.plan_token,
        _emission_schemas=initial_callback_state.emission_schemas,
    )
    with pytest.raises(ValueError, match="callback_state keys"):
        dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=DT,
            callbacks=callbacks,
            callback_state=wrong_keys,
        )
    with pytest.raises(ValueError, match="callback_state requires callbacks"):
        dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=DT,
            callback_state={"trace": torch.zeros((), dtype=torch.int64)},
        )

    _other_source, other_functional, _other_tensors = _functional_case()
    other_prepared = other_functional.prepare(
        _other_tensors.parameters,
        _other_tensors.constants,
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different"):
        dn.func.run(
            other_functional,
            partial(
                other_functional.step,
                _other_tensors.parameters,
                other_prepared,
            ),
            _other_tensors.state,
            tstop=DT,
            callbacks=callbacks,
        )


def test_functional_recorder_carry_is_bound_to_callback_plan_and_independent():
    _source, functional, tensors = _functional_case()
    first_callbacks = functional.make_callbacks(
        {"trace": FunctionalRecorder(["v"], node_indices=[0])}
    )
    second_callbacks = functional.make_callbacks(
        {"trace": FunctionalRecorder(["v"], node_indices=[4])}
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    step = partial(functional.step, tensors.parameters, prepared)
    _state, auxiliary = dn.func.run(
        functional,
        step,
        tensors.state,
        tstop=DT,
        callbacks=first_callbacks,
    )

    with pytest.raises(dn.func.FunctionalizationError, match="callback plan"):
        dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=DT,
            callbacks=second_callbacks,
            callback_state=auxiliary["callbacks"].state,
        )

    pair = functional.make_callbacks(
        {"first": FunctionalRecorder(["v"]), "second": FunctionalRecorder(["v"])}
    )
    _state, pair_auxiliary = dn.func.run(
        functional,
        step,
        tensors.state,
        tstop=0.0,
        callbacks=pair,
    )
    pair_state = pair_auxiliary["callbacks"].state
    _state, resumed = dn.func.run(
        functional,
        step,
        tensors.state,
        tstop=DT,
        callbacks=pair,
        callback_state=pair_state,
    )
    assert resumed["callbacks"].state["first"] == ()
    assert resumed["callbacks"].state["second"] == ()


def test_functional_callback_results_and_state_are_tensor_pytrees():
    _source, functional, tensors = _functional_case(trainable=True)
    parameters = dict(tensors.parameters)
    parameters[GNABAR] = parameters[GNABAR].detach().clone().requires_grad_()
    callbacks = functional.make_callbacks({"trace": FunctionalRecorder(["v"])})
    _state, auxiliary = _run_with_callbacks(
        "longrun",
        functional,
        tensors,
        callbacks,
        2,
        parameters=parameters,
    )
    results = auxiliary["callbacks"]

    leaves, _spec = torch.utils._pytree.tree_flatten(results)
    assert leaves
    assert all(torch.is_tensor(leaf) for leaf in leaves)
    cloned = torch.utils._pytree.tree_map(torch.clone, results)
    assert isinstance(cloned, dn.func.FunctionalCallbackResults)
    assert isinstance(cloned.state, dn.func.FunctionalCallbackState)
    assert cloned.state.plan_token is results.state.plan_token
    assert cloned.state["trace"] == results.state["trace"] == ()
    torch.testing.assert_close(cloned["trace"]["v"], results["trace"]["v"])

    gradient = torch.autograd.grad(
        cloned["trace"]["v"].square().mean(),
        parameters[GNABAR],
    )[0]
    assert torch.isfinite(gradient)
