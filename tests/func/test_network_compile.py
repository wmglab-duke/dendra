"""The Network compile facade preserves explicit rollout and snapshot contracts."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Mapping

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _loss,
    _mixed_network,
    _stim_network,
)
from tests.func.test_network_rollout import _driven_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _mutable(tree):
    """Copy readonly containers without cloning or untying tensor owners."""
    if isinstance(tree, Mapping):
        return {name: _mutable(value) for name, value in tree.items()}
    if isinstance(tree, tuple):
        values = [_mutable(value) for value in tree]
        return type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)
    return tree


def _event_names(model):
    return tuple(f"synapses.{name}.weights" for name in model.synapses)


def _targets(runner, names):
    return tuple(runner.get_parameter(name) for name in names)


def _recording():
    return {"trace": dn.func.Recorder(("cell.v", "cell.mech.relay.level", "t"))}


def _compile(model, *, steps=4, **options):
    return dn.compile(model, dt=DT, steps=steps, backend="aot_eager", **options)


def _voltage_loss(state):
    return sum(
        population["integrator"]["v"].square().sum()
        for population in state["populations"].values()
    )


@pytest.mark.parametrize("mode", ("train", "eval"))
def test_compile_network_snapshot_restart_matches_native_and_preserves_source(mode):
    model, native = _mixed_network(), _mixed_network()
    for network in (model, native):
        if mode == "eval":
            network.eval()
        network._reset_runtime_clock(2.75)
        network._set_duration_remainder(0.3 * DT)
    runner = _compile(model, callbacks=_recording())
    assert isinstance(runner, dn.func.CompiledNetworkRunner)
    assert runner.execution_report() is None
    initial = _clone(runner.initial_state, detached=True)
    source = runner.functional.extract().state
    observations = {
        "cell.v": [native.cell.v.clone()],
        "cell.mech.relay.level": [native.cell.mech.relay.level.clone()],
        "t": [native.t.clone()],
    }
    with torch.no_grad():
        for _ in range(4):
            native.step()
            observations["cell.v"].append(native.cell.v.clone())
            observations["cell.mech.relay.level"].append(
                native.cell.mech.relay.level.clone()
            )
            observations["t"].append(native.t.clone())
    expected_trace = {
        name: torch.stack(values) for name, values in observations.items()
    }
    first, auxiliary = runner()
    second, repeated = runner()
    _close(first, runner.functional.extract(native).state)
    _close(second, first)
    _close(auxiliary["callbacks"]["trace"], expected_trace)
    _close(repeated["callbacks"]["trace"], expected_trace)
    _close(runner.initial_state, initial)
    _close(runner.functional.extract().state, source)
    assert auxiliary["callbacks"]["trace"]["cell.v"].shape[0] == 5
    assert not first["populations"]["cell"]["integrator"]["v"].requires_grad
    assert runner.execution_report().strategy == "compiled_scan"


def test_explicit_network_continuation_preserves_recordings_and_prefix_gradients():
    model = _mixed_network()
    names = _event_names(model)
    runner = _compile(
        model, steps=3, inference=False, trainable=names, callbacks=_recording()
    )
    relay = runner.initial_state["populations"]["cell"]["mechanism_buffers"]["relay"]
    initial_level = relay["level"].detach().clone().requires_grad_()
    relay["level"] = initial_level
    reference_callbacks = runner.functional.make_callbacks(_recording())
    expected, reference = runner.functional.prepare_and_rollout(
        runner.parameters,
        runner.constants,
        runner.initial_state,
        steps=6,
        callbacks=reference_callbacks,
    )
    prefix, prefix_aux = runner()
    final, auxiliary = runner(prefix, callback_state=prefix_aux["callbacks"].state)
    trace = {
        name: torch.cat((prefix_aux["callbacks"]["trace"][name], values))
        for name, values in auxiliary["callbacks"]["trace"].items()
    }
    _close(final, expected)
    _close(trace, reference["callbacks"]["trace"])
    targets = (initial_level, *_targets(runner, names))
    actual_gradients = torch.autograd.grad(
        _loss(final) + 0.01 * trace["cell.v"].square().sum(), targets
    )
    reference_gradients = torch.autograd.grad(
        _loss(expected)
        + 0.01 * reference["callbacks"]["trace"]["cell.v"].square().sum(),
        targets,
    )
    _close(actual_gradients, reference_gradients)
    assert all(value.abs().sum() > 0 for value in actual_gradients)
    assert trace["cell.v"].shape[0] == 7


def test_network_runner_keeps_parameter_tree_readonly_and_live_tensor_leaves():
    model = _mixed_network()
    name = _event_names(model)[0]
    runner = _compile(model, inference=False, trainable=(name,))
    connection = next(iter(model.synapses))
    selected = runner.get_parameter(name)
    assert isinstance(selected, torch.nn.Parameter)
    assert runner.named_parameters[name] is selected
    assert runner.parameters["synapses"][connection]["weights"] is selected
    assert runner.parameter_name(name) == name
    assert selected.data_ptr() != model.synapses[connection].weight.w.data_ptr()
    assert all(
        not value.requires_grad
        for key, value in runner.named_parameters.items()
        if key != name
    )
    with pytest.raises(TypeError):
        runner.parameters["synapses"][connection]["weights"] = selected.clone()
    with pytest.raises(TypeError):
        runner.named_parameters[name] = selected.clone()
    with pytest.raises(TypeError):
        runner.constants["populations"]["cell"]["unknown"] = torch.zeros(())
    native = model.synapses[connection].weight.w.detach().clone()
    before, _ = runner()
    with torch.no_grad():
        selected.add_(0.02)
    changed, _ = runner()
    assert not torch.equal(
        before["populations"]["cell"]["mechanism_buffers"]["relay"]["level"],
        changed["populations"]["cell"]["mechanism_buffers"]["relay"]["level"],
    )
    torch.testing.assert_close(model.synapses[connection].weight.w, native)


def test_network_trainable_selection_and_checkpoint_preserve_shared_weight_identity():
    model = _mixed_network()
    first, second = tuple(model.synapses.values())
    second.weight.w = first.weight.w
    names = _event_names(model)
    runner = _compile(
        model,
        steps=7,
        inference=False,
        checkpoint_every=3,
        trainable=(names[1],),
    )
    target = runner.get_parameter(names[0])
    assert target is runner.get_parameter(names[1])
    assert isinstance(target, torch.nn.Parameter)
    assert target.data_ptr() != first.weight.w.data_ptr()
    expected, _ = runner.functional.prepare_and_rollout(
        runner.parameters, runner.constants, runner.initial_state, steps=7
    )
    actual, _ = runner()
    _close(actual, expected)
    actual_gradient = torch.autograd.grad(_loss(actual), target)[0]
    expected_gradient = torch.autograd.grad(_loss(expected), target)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0
    broken = _mutable(runner.parameters)
    second_name = tuple(model.synapses)[1]
    broken["synapses"][second_name]["weights"] = target.clone()
    with pytest.raises(ValueError, match="shared|alias"):
        runner(parameters=broken)


def test_network_compile_preserves_cross_population_raw_parameter_sharing():
    model, native = _driven_network(), _driven_network()
    for network in (model, native):
        network.right.mech.pas.g_param = network.left.mech.pas.g_param
        network.initialize(DT)
    left = "populations.left.integrator.mech.mechanisms.pas.g_param"
    right = "populations.right.integrator.mech.mechanisms.pas.g_param"
    runner = _compile(model, inference=False, checkpoint_every=3, trainable=(right,))
    target = runner.get_parameter(left)
    assert target is runner.get_parameter(right)
    assert isinstance(target, torch.nn.Parameter)
    assert target is not model.left.mech.pas.g_param
    actual, _ = runner()
    for _ in range(4):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    observed = torch.autograd.grad(_voltage_loss(actual), target)[0]
    expected = torch.autograd.grad(
        sum(cell.v.square().sum() for cell in native.populations.values()),
        native.left.mech.pas.g_param,
    )[0]
    torch.testing.assert_close(observed, expected)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0


def test_noisy_network_snapshot_restart_and_continuation_preserve_rng_position():
    model = _stim_network()
    runner = _compile(
        model,
        steps=5,
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    native_rng = model.netstim._rng.get_state().clone()
    initial = _clone(runner.initial_state, detached=True)
    first, first_aux = runner()
    restarted, restarted_aux = runner()
    _close(first, restarted)
    _close(first_aux["callbacks"]["trace"], restarted_aux["callbacks"]["trace"])
    resumed, resumed_aux = runner(first, callback_state=first_aux["callbacks"].state)
    callbacks = runner.functional.make_callbacks(
        {"trace": dn.func.Recorder(("cell.v", "t"))}
    )
    with torch.no_grad():
        expected, reference = runner.functional.run(
            runner.parameters,
            runner.constants,
            initial,
            steps=10,
            callbacks=callbacks,
        )
    _close(resumed, expected)
    joined = {
        name: torch.cat((first_aux["callbacks"]["trace"][name], values))
        for name, values in resumed_aux["callbacks"]["trace"].items()
    }
    _close(joined, reference["callbacks"]["trace"])
    _close(runner.initial_state, initial)
    torch.testing.assert_close(model.netstim._rng.get_state(), native_rng)
    assert not torch.equal(
        first["netstim"]["rng_state"], initial["netstim"]["rng_state"]
    )


@pytest.mark.parametrize("explicit_draws", (False, True))
def test_compiled_network_checkpoint_matches_explicit_draw_and_drive_gradients(
    explicit_draws,
):
    model = _stim_network()
    weight_name = _event_names(model)[0]
    runner = _compile(
        model,
        steps=7,
        inference=False,
        checkpoint_every=3,
        trainable=(weight_name, "netstim.parameters.noise"),
        callbacks={"trace": dn.func.Recorder("cell.v")},
    )
    voltage = runner.initial_state["populations"]["cell"]["integrator"]["v"]
    current = voltage.new_full((7, *voltage.shape), 0.02).requires_grad_()
    tape = runner.functional.sample_draw_tape(runner.initial_state, 7)
    if explicit_draws:
        tape = dn.func.NetStimDraw(tape.exp_rand.requires_grad_(), tape.rng_state)
    supplied = dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=current)},
        netstim=tape if explicit_draws else None,
    )
    oracle_inputs = dn.func.NetworkRolloutInput(
        populations=supplied.populations, netstim=tape
    )
    callbacks = runner.functional.make_callbacks({"trace": dn.func.Recorder("cell.v")})
    expected, expected_aux = runner.functional.prepare_and_rollout(
        runner.parameters,
        runner.constants,
        runner.initial_state,
        oracle_inputs,
        callbacks=callbacks,
    )
    actual, auxiliary = runner(inputs=supplied)
    _close(actual, expected)
    _close(auxiliary["callbacks"]["trace"], expected_aux["callbacks"]["trace"])
    targets = (*_targets(runner, (weight_name, "netstim.parameters.noise")), current)
    if explicit_draws:
        targets += (tape.exp_rand,)
    actual_gradient = torch.autograd.grad(
        auxiliary["callbacks"]["trace"]["cell.v"].square().sum(), targets
    )
    expected_gradient = torch.autograd.grad(
        expected_aux["callbacks"]["trace"]["cell.v"].square().sum(), targets
    )
    _close(actual_gradient, expected_gradient)
    assert all(value.abs().sum() > 0 for value in actual_gradient)


def test_per_population_network_compile_tapes_keep_independent_input_gradients():
    model = _driven_network()
    runner = _compile(model, steps=3, inference=False, checkpoint_every=2)
    inputs = dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(
                ve=torch.linspace(-0.3, 0.5, 3 * cell.v.numel(), dtype=DTYPE)
                .reshape(3, *cell.shape)
                .requires_grad_(),
                intra=cell.v.new_full((3, *cell.shape), 1e-8).requires_grad_(),
            )
            for name, cell in model.populations.items()
        }
    )
    expected, _ = runner.functional.prepare_and_rollout(
        runner.parameters, runner.constants, runner.initial_state, inputs
    )
    actual, _ = runner(inputs=inputs)
    _close(actual, expected)
    targets = tuple(value for pair in inputs.populations.values() for value in pair)
    observed = torch.autograd.grad(_voltage_loss(actual), targets)
    oracle = torch.autograd.grad(_voltage_loss(expected), targets)
    _close(observed, oracle)
    assert all(value.abs().sum() > 0 for value in observed)


def test_external_network_parameters_keep_user_history_without_replacing_defaults():
    model = _mixed_network()
    runner = _compile(model, inference=False)
    name = _event_names(model)[0]
    connection = next(iter(model.synapses))
    defaults = _clone(_mutable(runner.parameters), detached=True)
    gain = torch.tensor(1.2, dtype=DTYPE, requires_grad=True)
    parameters = _mutable(runner.parameters)
    parameters["synapses"][connection]["weights"] = runner.get_parameter(name) * gain
    expected, _ = runner.functional.prepare_and_rollout(
        parameters, runner.constants, runner.initial_state, steps=4
    )
    actual, _ = runner(parameters=parameters)
    _close(actual, expected)
    observed = torch.autograd.grad(_loss(actual), gain, retain_graph=True)[0]
    oracle = torch.autograd.grad(_loss(expected), gain)[0]
    torch.testing.assert_close(observed, oracle)
    assert observed.abs() > 0
    _close(_mutable(runner.parameters), defaults)


def test_repeated_training_network_calls_release_outputs_and_preparation_graphs():
    model = _mixed_network()
    name = _event_names(model)[0]
    with torch.inference_mode():
        runner = _compile(model, inference=False, trainable=(name,))
    leaves = (
        *runner.named_parameters.values(),
        *pytree.tree_leaves(runner.initial_state),
    )
    assert all(
        not torch.is_inference(value) for value in leaves if torch.is_tensor(value)
    )
    gradients = []
    for _ in range(3):
        state, auxiliary = runner()
        gradients.append(
            torch.autograd.grad(_loss(state), runner.get_parameter(name))[0]
        )
        output_ref = weakref.ref(state["populations"]["cell"]["integrator"]["v"])
        del state, auxiliary
        gc.collect()
        assert output_ref() is None
    _close(gradients[0], gradients[1])
    _close(gradients[1], gradients[2])


def test_inference_network_live_population_parameters_refresh_numerical_preparation():
    runner = _compile(_mixed_network())
    before, _ = runner()
    cm = runner.get_parameter("populations.cell.cm_param")
    with torch.no_grad():
        cm.mul_(1.5)
    changed, _ = runner()
    with torch.no_grad():
        expected, _ = runner.functional.prepare_and_rollout(
            runner.parameters, runner.constants, runner.initial_state, steps=4
        )
    _close(changed, expected)
    assert not torch.equal(
        changed["populations"]["cell"]["integrator"]["v"],
        before["populations"]["cell"]["integrator"]["v"],
    )


@pytest.mark.parametrize("mutation", ("mode", "wiring"))
def test_network_compile_rechecks_source_after_warm_calls(mutation):
    model = _mixed_network()
    runner = _compile(model)
    runner()
    report = runner.execution_report()
    if mutation == "mode":
        model.eval()
    else:
        model.synapses.pop(next(iter(model.synapses)))
    with pytest.raises(
        dn.func.FunctionalizationError, match="changed|wiring|configuration"
    ):
        runner()
    assert runner.execution_report() is report


@pytest.mark.parametrize(
    ("options", "match"),
    (
        ({"dt": 2 * DT}, "dt|timestep"),
        ({"initialization": "fresh"}, "snapshot|fresh|initialization"),
        ({"execution": "while"}, "scan|while|execution"),
        ({"extra": object()}, "extra|NetworkRolloutInput|population"),
    ),
)
def test_network_compile_rejects_unsupported_constructor_policies(options, match):
    keywords = {"dt": DT, "steps": 4, "backend": "aot_eager", **options}
    with pytest.raises(
        (ValueError, TypeError, dn.func.FunctionalizationError), match=match
    ):
        dn.compile(_mixed_network(), **keywords)


def test_network_compile_rejects_training_on_a_frozen_eval_network():
    model = _mixed_network()
    model.eval()
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="train|eval|mode"
    ):
        _compile(model, inference=False)


def test_network_compile_rejects_callback_continuation_without_explicit_state():
    runner = _compile(_mixed_network(), callbacks=_recording())
    _, auxiliary = runner()
    with pytest.raises(ValueError, match="state|continu"):
        runner(callback_state=auxiliary["callbacks"].state)
    with pytest.raises(ValueError, match="snapshot|initialization"):
        runner(initialization=object())


@pytest.mark.parametrize("transform", ("grad", "forward_ad"))
def test_network_compile_is_a_host_runner_with_lower_level_transform_guidance(
    transform,
):
    runner = _compile(_mixed_network())
    with pytest.raises(dn.func.FunctionalizationError, match="host-side.*lower-level"):
        if transform == "grad":
            torch.func.grad(lambda value: _voltage_loss(runner()[0]) + value * 0)(
                torch.tensor(0.0, dtype=DTYPE)
            )
        else:
            with torch.autograd.forward_ad.dual_level():
                runner()


def test_network_compile_rejects_an_outer_torch_compile_boundary():
    runner = _compile(_mixed_network())
    wrapped = torch.compile(runner, backend="eager", fullgraph=True)
    with pytest.raises(
        (dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported),
        match="outside torch.compile|[Oo]bserved exception",
    ):
        wrapped()
    assert runner.execution_report() is None
