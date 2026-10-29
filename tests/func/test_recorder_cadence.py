"""Recorder cadence follows native sampling phase across pure loop engines."""

from __future__ import annotations

import io
import os
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Recorder as NativeRecorder
from tests.func.test_lowering_api import DT, _close, _network, _population, _target

pytestmark = pytest.mark.cpu
STEPS = 9


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _indices(steps, stride):
    return [0, *range(1, steps + 1, stride)]


def _select(trace, indices):
    return {
        name: value.index_select(0, value.new_tensor(indices, dtype=torch.long))
        for name, value in trace.items()
    }


@pytest.mark.parametrize("save_dt", (None, 0.1, 0.25, 0.3))
@pytest.mark.parametrize("streaming", (False, True), ids=("memory", "stream-phase"))
def test_native_recorder_keeps_initial_then_preincrement_poststep_sampling(
    save_dt, streaming
):
    solver_dt = 0.1
    stride = None if save_dt is None else int(save_dt / solver_dt)
    model = SimpleNamespace(v=torch.zeros((1, 2), dtype=torch.float64))
    recorder = NativeRecorder(["v"], dt=save_dt)
    recorder.dt = solver_dt
    # Exercise native stream-phase accounting without starting an I/O worker.
    recorder.cache_with_hdf5 = streaming
    recorder.cache_every = 1000
    try:
        recorder.pre_loop_hook(model)
        for step in range(1, 8):
            model.v = model.v.new_full(model.v.shape, step)
            recorder.post_step_hook(model)
        if stride is None:
            expected = list(range(8))
        elif streaming:
            expected = [0, *range(stride, 8, stride)]
        else:
            expected = _indices(7, stride)
        torch.testing.assert_close(
            recorder.stack("v")[:, 0, 0], torch.tensor(expected, dtype=torch.float64)
        )
        assert recorder.i == 7 + int(streaming) and recorder.save_every == stride
        if save_dt == 0.3:
            assert stride == 2  # Preserve int(0.3 / 0.1), including float truncation.
        recorder.reset()
        assert recorder.i == 0 and recorder.rec == {"v": []}
    finally:
        recorder.cache_with_hdf5 = False


class _EveryStep(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        voltage = state["integrator"]["v"]
        return carry + voltage.sum(), voltage.mean()

    def finalize(self, carry, emissions):
        return {
            "total": carry,
            "samples": carry.new_empty((0,)) if emissions is None else emissions,
        }


def _callbacks(functional, network, *, custom=True):
    names = ("left.v", "t") if network else ("v", "t")
    slow = dn.func.Recorder(("v", "t"), dt=2.5 * DT)
    specifications = {
        "dense": dn.func.Recorder(names),
        "slow": dn.func.PopulationCallback("left", slow) if network else slow,
        "sparse": dn.func.Recorder(names, dt=6.5 * DT),
    }
    if custom:
        callback = _EveryStep()
        specifications["every"] = (
            dn.func.PopulationCallback("left", callback) if network else callback
        )
    return functional.make_callbacks(specifications)


def _inputs(network, tape):
    value = dn.func.Inputs(intra=tape)
    return {"left": value} if network else value


def _loss(outputs, network):
    key = "left.v" if network else "v"
    return outputs["slow"]["v"].square().sum() + outputs["sparse"][key].square().sum()


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_recorder_cadences_preserve_selected_gradients_and_segment_phase(
    network, execution
):
    model = _network() if network else _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    shape = model.left.shape if network else model.shape
    tape = torch.linspace(1e-7, 4e-7, STEPS * torch.Size(shape).numel()).reshape(
        STEPS, *shape
    )
    tape.requires_grad_()
    callbacks = _callbacks(functional, network)
    key = "left.v" if network else "v"
    dense_callbacks = functional.make_callbacks({"dense": dn.func.Recorder((key, "t"))})
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    bound = kernel.bind(tensors)
    with torch.no_grad() if execution == "while" else nullcontext():
        expected_state, reference = dn.func.run(
            functional.bind(tensors),
            tensors.state,
            steps=STEPS,
            inputs=_inputs(network, tape),
            callbacks=dense_callbacks,
        )
        eager_state, eager = dn.func.run(
            functional.bind(tensors),
            tensors.state,
            steps=STEPS,
            inputs=_inputs(network, tape),
            callbacks=callbacks,
        )
        actual_state, actual = dn.func.run(
            bound,
            tensors.state,
            steps=STEPS,
            inputs=_inputs(network, tape),
            callbacks=callbacks,
        )
    dense = reference["callbacks"]["dense"]
    slow = _select(dense, _indices(STEPS, 2))
    slow = {"v": slow[key], "t": slow["t"]}
    sparse = _select(dense, _indices(STEPS, 6))
    for result in (eager, actual):
        outputs = result["callbacks"]
        _close(outputs["dense"], dense)
        _close(outputs["slow"], slow)
        _close(outputs["sparse"], sparse)
        _close(
            outputs["every"]["samples"], dense[key][1:].flatten(start_dim=1).mean(dim=1)
        )
        _close(outputs["every"]["total"], dense[key][1:].sum())
    _close(eager_state, expected_state)
    _close(actual_state, expected_state)
    assert kernel.execution_report().steps == 1
    if execution == "scan":
        targets = (_target(tensors.parameters, network), tape)
        expected_gradients = torch.autograd.grad(
            _loss({"slow": slow, "sparse": sparse}, network), targets
        )
        observed_gradients = torch.autograd.grad(
            _loss(actual["callbacks"], network), targets
        )
        _close(observed_gradients, expected_gradients)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0
            for value in observed_gradients
        )
    with torch.no_grad() if execution == "while" else nullcontext():
        state, prefix = dn.func.run(
            bound,
            tensors.state,
            steps=1,
            inputs=_inputs(network, tape[:1]),
            callbacks=callbacks,
        )
        report = kernel.execution_report()
        unchanged, empty = dn.func.run(
            bound, prefix["continuation"], steps=0, callbacks=callbacks
        )
        assert kernel.execution_report() is report
        _close(unchanged, state)
        for name in ("dense", "slow", "sparse"):
            assert all(
                value.shape[0] == 0 for value in empty["callbacks"][name].values()
            )
        state, middle = dn.func.run(
            bound,
            prefix["continuation"],
            steps=4,
            inputs=_inputs(network, tape[1:5]),
            callbacks=callbacks,
        )
        assert middle["callbacks"]["sparse"][key].shape[0] == 0
        assert kernel.execution_report().strategy == (
            "compiled_scan" if execution == "scan" else "compiled_while_loop"
        )
        state, suffix = dn.func.run(
            bound,
            middle["continuation"],
            steps=4,
            inputs=_inputs(network, tape[5:]),
            callbacks=callbacks,
        )
    _close(state, expected_state)
    joined = {}
    for name in ("dense", "slow", "sparse"):
        joined[name] = {
            state_name: torch.cat(
                [
                    part["callbacks"][name][state_name]
                    for part in (prefix, middle, suffix)
                ]
            )
            for state_name in actual["callbacks"][name]
        }
        _close(joined[name], actual["callbacks"][name])
    _close(suffix["callbacks"]["every"]["total"], actual["callbacks"]["every"]["total"])
    if execution == "scan":
        gradients = torch.autograd.grad(_loss(joined, network), targets)
        _close(gradients, expected_gradients)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_recorder_cadence_portable_continuation_restores_phase_to_a_new_chunk_width(
    network,
):
    functional, tensors = dn.func.make_functional(
        _network() if network else _population(), dt=DT
    )
    callbacks = _callbacks(functional, network, custom=False)
    source = functional.compile_rollout_chunk(4, execution="while", backend="eager")
    with torch.no_grad():
        expected, whole = dn.func.run(
            source.bind(tensors), tensors.state, steps=STEPS, callbacks=callbacks
        )
        _, first = dn.func.run(
            source.bind(tensors), tensors.state, steps=5, callbacks=callbacks
        )
    checkpoint = dn.func.export_continuation(
        functional, first["continuation"], callbacks=callbacks
    )
    disk = io.BytesIO()
    torch.save(checkpoint, disk)
    disk.seek(0)
    receiving, _ = dn.func.make_functional(
        _network() if network else _population(), dt=DT
    )
    receiving_callbacks = _callbacks(receiving, network, custom=False)
    restored = dn.func.restore_continuation(
        receiving, torch.load(disk, weights_only=True), callbacks=receiving_callbacks
    )
    receiver = receiving.compile_rollout_chunk(5, execution="scan", backend="eager")
    with torch.no_grad():
        actual, last = dn.func.run(
            receiver.bind(restored), restored, steps=4, callbacks=receiving_callbacks
        )
    _close(actual, expected)
    for name in whole["callbacks"]:
        observed = {
            state_name: torch.cat(
                (
                    first["callbacks"][name][state_name],
                    last["callbacks"][name][state_name],
                )
            )
            for state_name in whole["callbacks"][name]
        }
        _close(observed, whole["callbacks"][name])
    names = ("left.v", "t") if network else ("v", "t")
    different_slow = dn.func.Recorder(("v", "t"), dt=3 * DT)
    different = receiving.make_callbacks(
        {
            "dense": dn.func.Recorder(names),
            "slow": dn.func.PopulationCallback("left", different_slow)
            if network
            else different_slow,
            "sparse": dn.func.Recorder(names, dt=6.5 * DT),
        }
    )
    with pytest.raises(dn.func.FunctionalizationError, match="configuration"):
        different.restore_state(callbacks.export_state(first["callbacks"].state))


def test_regional_recorder_cadence_selects_the_same_physical_coordinates():
    from tests.func.test_callback_checkpoints import _model

    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(
        {
            "all": dn.func.Recorder(("v", "t")),
            "region": dn.func.Recorder(("v", "t"), region=model.even, dt=2.5 * DT),
        }
    )
    tape = (
        torch.arange(1, 1 + 5 * model.v.numel(), dtype=torch.float64).reshape(
            5, *model.shape
        )
        * 1e-7
    )
    with torch.no_grad():
        _, auxiliary = dn.func.run(
            functional.bind(tensors),
            tensors.state,
            steps=5,
            inputs=dn.func.Inputs(intra=tape),
            callbacks=callbacks,
        )
    expected = _select(auxiliary["callbacks"]["all"], _indices(5, 2))
    expected["v"] = expected["v"].index_select(-1, torch.tensor([0, 2]))
    _close(auxiliary["callbacks"]["region"], expected)


def test_native_checkpoint_callback_bridge_preserves_cadence_phase_and_selected_vjp():
    from tests.func.test_imperative_checkpoint_callbacks import DT as bridge_dt
    from tests.func.test_imperative_checkpoint_callbacks import (
        _callbacks as bridge_callbacks,
    )
    from tests.func.test_imperative_checkpoint_callbacks import (
        _checkpoint,
        _gnabar,
    )
    from tests.func.test_imperative_checkpoint_callbacks import _model as bridge_model

    reference_model = bridge_model(require_grad=True)
    functional, tensors = dn.func.make_functional(reference_model, dt=bridge_dt)
    dense = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    _, whole = dn.func.run(
        functional.bind(tensors), tensors.state, steps=7, callbacks=dense
    )
    expected = _select(whole["callbacks"]["trace"], _indices(7, 3))
    parameter_name = tensors.parameter_name("gnabar")
    expected_gradient = torch.autograd.grad(
        expected["v"].square().sum(), tensors.parameters[parameter_name]
    )[0]

    native = bridge_model(require_grad=True)
    callbacks = bridge_callbacks(
        native, {"trace": dn.func.Recorder(("v", "t"), dt=3.5 * bridge_dt)}
    )
    first_loss, first = _checkpoint(native, callbacks, steps=2)
    middle_loss, middle = _checkpoint(
        native, callbacks, steps=2, callback_state=first.state
    )
    last_loss, last = _checkpoint(
        native, callbacks, steps=3, callback_state=middle.state
    )
    assert first_loss is middle_loss is last_loss is None
    observed = {
        name: torch.cat([part["trace"][name] for part in (first, middle, last)])
        for name in expected
    }
    _close(observed, expected)
    final_voltage, final_time = native.v.detach().clone(), native.t.detach().clone()
    actual_gradient = torch.autograd.grad(
        observed["v"].square().sum(), _gnabar(native)
    )[0]
    _close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0
    _close(native.v, final_voltage)
    _close(native.t, final_time)


def test_default_recorder_keeps_legacy_portable_configuration_and_empty_carry():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    _, result = dn.func.run(
        functional.bind(tensors), tensors.state, steps=1, callbacks=callbacks
    )
    checkpoint = callbacks.export_state(result["callbacks"].state)
    descriptor = checkpoint["configuration"]["callbacks"][0]["callback"]
    assert descriptor["kind"] == "Recorder"
    assert tuple(descriptor["fields"]) == (
        "states",
        "paths",
        "max_only",
        "node_indices",
        "partition",
        "region",
    )
    assert checkpoint["carries"]["trace"] == ()


@pytest.mark.parametrize("save_dt", (0, -1, float("nan"), float("inf"), DT / 2))
def test_functional_recorder_rejects_invalid_or_substep_cadence_at_binding(save_dt):
    functional, _ = dn.func.make_functional(_population(), dt=DT)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError),
        match="dt|finite|positive|record|timestep",
    ):
        functional.make_callbacks({"trace": dn.func.Recorder("v", dt=save_dt)})


def test_cadenced_recorder_rejects_a_changed_prepared_or_runtime_timestep():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder("v", dt=2.5 * DT)})
    constants = {**tensors.constants, "dt": tensors.constants["dt"].new_tensor(2 * DT)}
    with pytest.raises(dn.func.FunctionalizationError, match="prepared timestep"):
        functional.prepare_and_rollout(
            tensors.parameters, constants, tensors.state, steps=0, callbacks=callbacks
        )
    prepared = functional.prepare(tensors.parameters, constants)
    with pytest.raises(dn.func.FunctionalizationError, match="runtime timestep"):
        dn.func.run(
            functional,
            partial(functional.step, tensors.parameters, prepared),
            tensors.state,
            tstop=2 * DT,
            dt=2 * DT,
            callbacks=callbacks,
        )


@pytest.mark.parametrize("transform", ("grad", "jacfwd", "vmap"))
def test_cadenced_recorder_supports_fresh_functional_transforms(transform):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder("v", dt=2.5 * DT)})
    dense_callbacks = functional.make_callbacks({"trace": dn.func.Recorder("v")})

    def evaluate(voltage, decimate):
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": voltage},
        }
        _, auxiliary = functional.rollout(
            tensors.parameters,
            prepared,
            state,
            steps=5,
            callbacks=callbacks if decimate else dense_callbacks,
        )
        observed = auxiliary["callbacks"]["trace"]["v"]
        if not decimate:
            observed = observed.index_select(
                0, observed.new_tensor(_indices(5, 2), dtype=torch.long)
            )
        return observed.square().mean()

    voltage = tensors.state["integrator"]["v"].detach()
    if transform == "vmap":
        voltage = torch.stack((voltage, voltage + 1))
    operation = getattr(torch.func, transform)
    expected = operation(lambda value: evaluate(value, False))(voltage)
    actual = operation(lambda value: evaluate(value, True))(voltage)
    _close(actual, expected)


def test_atomic_recorder_cadence_compiles_with_dynamic_output_capture():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder("v", dt=2.5 * DT)})
    dense_callbacks = functional.make_callbacks({"trace": dn.func.Recorder("v")})

    def observed(voltage, selected):
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": voltage},
        }
        _, auxiliary = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            steps=5,
            callbacks=callbacks if selected else dense_callbacks,
        )
        trace = auxiliary["callbacks"]["trace"]["v"]
        if not selected:
            trace = trace.index_select(
                0, trace.new_tensor(_indices(5, 2), dtype=torch.long)
            )
        return trace

    voltage = tensors.state["integrator"]["v"].detach().clone().requires_grad_()
    expected = observed(voltage, False)
    expected_gradient = torch.autograd.grad(expected.square().sum(), voltage)[0]
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(
            lambda value: observed(value, True), backend="aot_eager", fullgraph=True
        )
        actual = compiled(voltage)
        actual_gradient = torch.autograd.grad(actual.square().sum(), voltage)[0]
    _close(actual, expected)
    _close(actual_gradient, expected_gradient)


@pytest.mark.mps
def test_mps_while_recorder_cadence_matches_selected_dense_samples():
    from dendra.func._callbacks import _BoundCadencedRecorder, _BoundRecorder

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    # A Network recording can combine accelerator values with CPU RNG bytes.
    # Exercise both devices as the phase owner without another compiler case.
    for states in (("gpu", "cpu"), ("cpu", "gpu")):
        recorder = _BoundCadencedRecorder(
            recorder=_BoundRecorder(
                states=states,
                paths=tuple((name,) for name in states),
                max_only=False,
                node_indices=None,
                partition=None,
            ),
            save_every=2,
            callback_dt=DT,
        )

        def frame(step):
            return {
                "gpu": torch.tensor(
                    [step + 0.25, step + 0.75], dtype=torch.float32, device="mps"
                ),
                "cpu": torch.tensor([step, step + 10], dtype=torch.uint8, device="cpu"),
            }

        initial = frame(0)
        carry, first = recorder.initialize(initial, None)
        assert carry.device == initial[states[0]].device
        parts = [first]
        for step in range(1, 6):
            carry, emitted = recorder.update(carry, frame(step), None)
            parts.append(emitted)
        emitted = {name: torch.stack([part[name] for part in parts]) for name in states}
        selected = recorder.finalize(carry, emitted)
        for name, value in emitted.items():
            expected = value.index_select(
                0, torch.tensor([0, 1, 3, 5], dtype=torch.long, device=value.device)
            )
            assert selected[name].device == value.device
            torch.testing.assert_close(selected[name], expected)
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        model = _population()
        program = dn.compile(
            model,
            dt=DT,
            steps=STEPS,
            chunk_steps=4,
            execution="while",
            backend="aot_eager",
            callbacks={"trace": dn.func.Recorder(("v", "t"), dt=2.5 * DT)},
        )
        actual, auxiliary = program()
        dense = program.functional.make_callbacks(
            {"trace": dn.func.Recorder(("v", "t"))}
        )
        with torch.no_grad():
            expected, reference = dn.func.run(
                program.functional.bind(
                    program.parameters, constants=program.constants
                ),
                program.initial_state,
                steps=STEPS,
                callbacks=dense,
            )
        _close(actual, expected)
        _close(
            auxiliary["callbacks"]["trace"],
            _select(reference["callbacks"]["trace"], _indices(STEPS, 2)),
        )
        assert auxiliary["callbacks"]["trace"]["v"].device.type == "mps"
        assert program.execution_report().steps == 1
