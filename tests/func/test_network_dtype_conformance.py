"""Mixed solver Networks retain each Population's numerical precision."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import graded_syn, pas
from dendra.units import nA
from tests.func.test_functional_initialization import (
    _assert_tensor_snapshot,
    _tensor_snapshot,
)
from tests.func.test_network import _RAW_WEIGHT, _clone
from tests.func.test_network_rollout import _native_driven_step

DT = 0.0001
STEPS = 4
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def _network(*, training=False):
    point = dn.Population(
        N=2,
        C=1,
        dtype=torch.float32,
        v_init=-62.0,
        integrator=dn.rk4(imem=True),
    )
    path = dn.Unmyelinated(
        [2.0, 2.5],
        L=30.0,
        dx=10.0,
        dtype=torch.float64,
        v_init=-65.0,
        integrator=dn.bwd_euler_ub(method="thomas", imem=True),
    )
    for population in (point, path):
        population.insert(pas, g=0.0004, e=-70.0)
        population.insert(graded_syn.rename("analog"), e=0.0)
    path[..., [2, 0, 2]].label("recorded")
    network = dn.Network({"point": point, "path": path})
    for source, target, source_slice, target_slice in (
        (point, path, point[:], path[:, 0]),
        (path, point, path[:, -1], point[:]),
    ):
        network.connect_continuous_one_to_one(
            source_slice,
            target_slice,
            target.mech.analog,
            input="g_pre",
            transform=_release,
            weight=0.0002,
            delay=DT,
        )
    network.train(training)
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    return network


def _close(actual, expected):
    values, schema = pytree.tree_flatten(actual)
    references, other = pytree.tree_flatten(expected)
    assert schema == other
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            assert value.dtype == reference.dtype
            torch.testing.assert_close(
                value,
                reference,
                rtol=4e-5 if value.dtype == torch.float32 else 2e-8,
                atol=3e-6 if value.dtype == torch.float32 else 2e-9,
            )
        else:
            assert value == reference


def _snapshot(network):
    return {
        (kind, name): _tensor_snapshot(value)
        for kind, values in (
            ("parameters", network.named_parameters(remove_duplicate=False)),
            ("buffers", network.named_buffers(remove_duplicate=False)),
        )
        for name, value in values
    }


def _assert_snapshot(network, snapshot):
    values = {
        (kind, name): value
        for kind, entries in (
            ("parameters", network.named_parameters(remove_duplicate=False)),
            ("buffers", network.named_buffers(remove_duplicate=False)),
        )
        for name, value in entries
    }
    assert values.keys() == snapshot.keys()
    for key, value in values.items():
        _assert_tensor_snapshot(value, snapshot[key])


def _inputs(network, *, trainable=False):
    return dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(
                intra=(
                    torch.linspace(
                        -0.2 * nA,
                        0.4 * nA,
                        2 * STEPS * population.v.numel(),
                        dtype=population.v.dtype,
                    )
                    .reshape(2 * STEPS, *population.shape)
                    .requires_grad_(trainable)
                )
            )
            for name, population in network.populations.items()
        }
    )


def _slice(inputs, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


def _native_step(network, inputs, index):
    _native_driven_step(
        network,
        {
            name: dn.func.StepInput(intra=packet.intra[index])
            for name, packet in inputs.populations.items()
        },
    )


def _callbacks(network):
    return {
        "trace": dn.func.Recorder(
            ("point.v", "point.i_membrane", "path.v", "path.i_membrane", "t")
        ),
        "region": dn.func.Recorder(("path.v", "path.t"), region=network.path.recorded),
    }


def _assert_dtypes(state, callbacks):
    for name, dtype in (("point", torch.float32), ("path", torch.float64)):
        for field in ("v", "i_membrane"):
            assert state["populations"][name]["integrator"][field].dtype == dtype
            assert callbacks["trace"][f"{name}.{field}"].dtype == dtype
    assert callbacks["region"]["path.v"].dtype == torch.float64
    assert callbacks["region"]["path.t"].dtype == torch.float64


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_mixed_dtype_while_recording_restart_and_checkpoint_continuation(
    initialization,
):
    source, native = _network(), _network()
    snapshot = _snapshot(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization=initialization,
        inference=True,
        execution="while",
        callbacks=_callbacks(source),
        backend="aot_eager",
        fullgraph=True,
    )
    frozen = _clone(runner.functional.extract(), detached=True)
    inputs = _inputs(source)
    frozen_inputs = _clone(inputs, detached=True)
    if initialization == "fresh":
        native.initialize(DT)
    frames = {
        f"{name}.{field}": [getattr(population, field).clone()]
        for name, population in native.populations.items()
        for field in ("v", "i_membrane")
    }
    frames["t"] = [native.t.clone()]
    with torch.no_grad():
        for index in range(2 * STEPS):
            _native_step(native, inputs, index)
            for name, population in native.populations.items():
                for field in ("v", "i_membrane"):
                    frames[f"{name}.{field}"].append(getattr(population, field).clone())
            frames["t"].append(native.t.clone())
            if index == STEPS - 1:
                middle = _clone(runner.functional.extract(native).state, detached=True)
    prefix, first = runner(inputs=_slice(inputs, 0, STEPS))
    replay, replay_aux = runner(inputs=_slice(inputs, 0, STEPS))
    _close(prefix, middle)
    _close(replay, prefix)
    _close(replay_aux["callbacks"].outputs, first["callbacks"].outputs)
    parameters = first.get("parameters", runner.parameters)
    constants = first.get("constants", runner.constants)
    final, second = runner(
        prefix,
        parameters=parameters,
        constants=constants,
        inputs=_slice(inputs, STEPS, 2 * STEPS),
        callback_state=first["callbacks"].state,
    )
    _close(final, runner.functional.extract(native).state)
    for name, values in frames.items():
        trace = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        _close(trace, torch.stack(values))
    for state, auxiliary in ((prefix, first), (final, second)):
        _assert_dtypes(state, auxiliary["callbacks"])
        _close(
            auxiliary["callbacks"]["region"]["path.v"],
            auxiliary["callbacks"]["trace"]["path.v"][..., [2, 0, 2]],
        )
    assert runner.execution_report().strategy == "compiled_while_loop"
    resumed, destination = _network(), _network()
    runner.functional.commit_state_(destination, prefix)
    resumed.restore_dict_from_checkpoint(destination.state_dict_for_checkpoint())
    with torch.no_grad():
        for index in range(STEPS, 2 * STEPS):
            _native_step(resumed, inputs, index)
    _close(final, runner.functional.extract(resumed).state)
    _close(runner.functional.extract(), frozen)
    _close(inputs, frozen_inputs)
    _assert_snapshot(source, snapshot)


def test_mixed_dtype_fresh_scan_checkpoint_preserves_cross_dtype_weight_and_drive_vjps():
    source, native = _network(training=True), _network(training=True)
    snapshot = _snapshot(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=2 * STEPS,
        initialization="fresh",
        inference=False,
        trainable=(f"continuous_synapses.*.{_RAW_WEIGHT}",),
        checkpoint_every=2,
        callbacks=_callbacks(source),
        backend="aot_eager",
        fullgraph=True,
    )
    frozen = _clone(runner.functional.extract(), detached=True)
    inputs, native_inputs = (
        _inputs(source, trainable=True),
        _inputs(native, trainable=True),
    )
    frozen_inputs = _clone(inputs, detached=True)
    final, auxiliary = runner(inputs=inputs)
    native.initialize(DT)
    traces = {
        name: [population.v.clone()] for name, population in native.populations.items()
    }
    for index in range(2 * STEPS):
        _native_step(native, native_inputs, index)
        for name, population in native.populations.items():
            traces[name].append(population.v.clone())
    _close(final, runner.functional.extract(native).state)
    connection_names = tuple(source.continuous_synapses)
    targets = tuple(
        runner.get_parameter(f"continuous_synapses.{name}.{_RAW_WEIGHT}")
        for name in connection_names
    ) + tuple(packet.intra for packet in inputs.populations.values())
    native_targets = tuple(
        dict(native.continuous_synapses[name].named_parameters())[_RAW_WEIGHT]
        for name in connection_names
    ) + tuple(packet.intra for packet in native_inputs.populations.values())
    loss = sum(
        auxiliary["callbacks"]["trace"][f"{name}.v"].square().mean()
        for name in source.populations
    )
    native_loss = sum(torch.stack(values).square().mean() for values in traces.values())
    gradients = torch.autograd.grad(loss, targets)
    expected = torch.autograd.grad(native_loss, native_targets)
    _close(gradients, expected)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in gradients)
    _assert_dtypes(final, auxiliary["callbacks"])
    assert runner.execution_report().strategy == "compiled_scan"
    _close(runner.functional.extract(), frozen)
    _close(inputs, frozen_inputs)
    _assert_snapshot(source, snapshot)
