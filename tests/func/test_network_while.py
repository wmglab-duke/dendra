"""Network while inference retains native phases, explicit carry, and recording."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import graded_syn, pas
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _loss,
    _mixed_network,
    _stim_network,
)
from tests.func.test_network_bitpacked_inference import (
    _recurrent,
    _stimulus,
    _sync_stimulus,
)
from tests.func.test_network_pending_calendar import _network as _calendar_network
from tests.func.test_network_rollout import (
    _driven_network,
    _native_driven_step,
    _VoltageEnergy,
)
from tests.func.test_network_source_history import _stim_history

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _eval(network):
    network.eval()
    network.initialize(DT, reinit_weights=False, reinit_delays=False)
    return network


def _recorders(functional):
    return functional.make_callbacks({"trace": dn.func.Recorder(("cell.v", "t"))})


def _transport(backend):
    if backend != "dense":
        return _calendar_network(backend, noise=0.6, interval=0.5)
    network = _eval(_stim_network())
    connection = next(iter(network.synapses.values()))
    connection.schedule(
        con_indices=[0, 0, 0],
        times_ms=[0.0, DT, 3 * DT],
        weight=torch.tensor([0.25, -0.5, 0.7], dtype=DTYPE),
    )
    return network


class _LayoutRelease(torch.nn.Module):
    """A pure transform whose public Tensor metadata affects its arithmetic."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4], dtype=DTYPE))

    def forward(self, value):
        layout_gain = 1.0 if self.scale.is_contiguous() else 2.0
        layout_gain *= self.scale.storage_offset() + 1
        return torch.sigmoid(value * 0.01) * self.scale * layout_gain


def _layout_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        pre = dn.SingleCompartment(
            N=1, C=3, v_init=-55.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        post = dn.SingleCompartment(
            N=1, C=3, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        pre.insert(pas, g=0.001, e=-55.0)
        post.insert(pas, g=0.001, e=-65.0)
        post.insert(graded_syn.rename("analog"), e=0.0)
        network = dn.Network({"pre": pre, "post": post})
        network.connect_continuous_one_to_one(
            pre[:],
            post[:],
            post.mech.analog,
            pre_var="v",
            input="g_pre",
            weight=0.002,
            delay=0.0,
            transform=_LayoutRelease(),
        )
        network.eval()
        network.initialize(DT)
    return network


@pytest.mark.parametrize("record", (False, True))
def test_while_mixed_recurrent_event_and_continuous_phases_match_eager_scan_and_native(
    record,
):
    model, native = _eval(_mixed_network()), _eval(_mixed_network())
    functional, tensors = dn.func.make_functional_network(model)
    frozen = _clone(tensors, detached=True)
    steps = 6
    drive = torch.linspace(
        -2e-6, 4e-6, steps * model.cell.v.numel(), dtype=DTYPE
    ).reshape(steps, *model.cell.shape)
    inputs = dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=drive)}
    )
    callbacks = _recorders(functional) if record else None
    while_chunk = functional.compile_rollout_chunk(
        steps, execution="while", backend="aot_eager"
    )
    scan_chunk = functional.compile_rollout_chunk(
        steps, execution="scan", backend="aot_eager"
    )
    with torch.no_grad():
        expected, expected_aux = functional.prepare_and_rollout(
            *tensors, inputs, steps=steps, callbacks=callbacks
        )
        actual, auxiliary = functional.run(
            *tensors,
            inputs,
            steps=steps,
            compiled_chunk=while_chunk,
            callbacks=callbacks,
        )
        scanned, scan_aux = functional.run(
            *tensors,
            inputs,
            steps=steps,
            compiled_chunk=scan_chunk,
            callbacks=callbacks,
        )
        for index in range(steps):
            _native_driven_step(native, {"cell": dn.func.StepInput(intra=drive[index])})
    _close(actual, expected)
    _close(actual, scanned)
    _close(actual, functional.extract(native).state)
    if record:
        _close(auxiliary["callbacks"]["trace"], expected_aux["callbacks"]["trace"])
        _close(auxiliary["callbacks"]["trace"], scan_aux["callbacks"]["trace"])
    assert while_chunk.execution_report().strategy == "compiled_while_loop"
    assert while_chunk.execution_report().callbacks is record
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)


@pytest.mark.parametrize("steps", (1, 3, 4, 5))
def test_while_report_matches_the_captured_loop_or_bounded_tail_without_callbacks(
    steps,
):
    functional, tensors = dn.func.make_functional_network(_recurrent())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    graphs = []

    def capture(graph, _examples):
        graphs.append(graph)
        return graph.forward

    chunk = functional.compile_rollout_chunk(steps, execution="while", backend=capture)
    assert chunk.capabilities.execution == "while"
    assert chunk.execution_report() is None
    with torch.no_grad():
        actual, _ = chunk(tensors.parameters, prepared, tensors.state)
        expected, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=steps
        )
    _close(actual, expected)
    has_loop = any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for graph in graphs
        for node in graph.graph.nodes
    )
    assert graphs and has_loop is (steps >= 4)
    report = chunk.execution_report()
    assert report.requested_execution == "while"
    assert report.strategy == (
        "compiled_while_loop" if steps >= 4 else "compiled_unrolled"
    )
    assert report.steps == steps and not report.grad_enabled and not report.callbacks
    assert report.reason is None if steps >= 4 else report.reason is not None


@pytest.mark.parametrize("backend", ("dense", "bitpacked_history", "sparse_calendar"))
@pytest.mark.parametrize("draws", ("automatic", "explicit"))
def test_while_transport_schedules_noisy_rng_and_input_tapes_match_native(
    backend, draws
):
    model, native = _transport(backend), _transport(backend)
    _sync_stimulus(model, native)
    functional, tensors = dn.func.make_functional_network(model, pending_capacity=64)
    frozen = _clone(tensors, detached=True)
    steps = 7
    drive = torch.linspace(
        -1e-6, 3e-6, steps * model.cell.v.numel(), dtype=DTYPE
    ).reshape(steps, *model.cell.shape)
    tape = functional.sample_draw_tape(tensors.state, steps)
    explicit = dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=drive)}, netstim=tape
    )
    selected = explicit if draws == "explicit" else explicit._replace(netstim=None)
    callbacks = _recorders(functional)
    chunk = functional.compile_rollout_chunk(
        steps, execution="while", backend="aot_eager"
    )
    with torch.no_grad():
        expected, oracle = functional.prepare_and_rollout(
            *tensors, explicit, steps=steps, callbacks=callbacks
        )
        actual, auxiliary = functional.run(
            *tensors, selected, steps=steps, callbacks=callbacks, compiled_chunk=chunk
        )
        for index in range(steps):
            _native_driven_step(native, {"cell": dn.func.StepInput(intra=drive[index])})
    _close(actual, expected)
    _close(actual, functional.extract(native).state)
    _close(auxiliary["callbacks"]["trace"], oracle["callbacks"]["trace"])
    assert not torch.equal(
        actual["netstim"]["rng_state"], tensors.state["netstim"]["rng_state"]
    )
    assert chunk.execution_report().strategy == "compiled_while_loop"
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)


def test_while_distinct_population_extracellular_and_intracellular_tapes_match_native():
    model, native = _eval(_driven_network()), _eval(_driven_network())
    functional, tensors = dn.func.make_functional_network(model)
    steps = 5
    tapes = {}
    for index, (name, cell) in enumerate(model.populations.items()):
        ve = torch.linspace(-0.02, 0.03, steps * cell.v.numel(), dtype=DTYPE).reshape(
            steps, *cell.shape
        ) * (index + 1)
        intra = torch.linspace(
            -2e-6, 5e-6, steps * cell.v.numel(), dtype=DTYPE
        ).reshape(steps, *cell.shape) * (index + 1)
        tapes[name] = dn.func.RolloutInput(ve=ve, intra=intra)
    inputs = dn.func.NetworkRolloutInput(populations=tapes)
    frozen = _clone((tensors, inputs), detached=True)
    chunk = functional.compile_rollout_chunk(
        steps, execution="while", backend="aot_eager"
    )
    with torch.no_grad():
        actual, _ = functional.run(*tensors, inputs, steps=steps, compiled_chunk=chunk)
        for index in range(steps):
            _native_driven_step(
                native,
                {
                    name: dn.func.StepInput(
                        ve=value.ve[index], intra=value.intra[index]
                    )
                    for name, value in tapes.items()
                },
            )
    _close(actual, functional.extract(native).state)
    _close((tensors, inputs), frozen)


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_while_runner_restart_continuation_and_recording_keep_snapshots(initialization):
    model, native = _transport("bitpacked_history"), _transport("bitpacked_history")
    _sync_stimulus(model, native)
    for _ in range(2):
        model.step()
        native.step()
    runner = dn.compile(
        model,
        dt=DT,
        steps=5,
        execution="while",
        inference=True,
        initialization=initialization,
        pending_capacity=64,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    initial = _clone(runner.initial_state, detached=True)
    source = runner.functional.extract().state
    defaults = _clone((runner._parameters, runner._constants), detached=True)
    if initialization == "fresh":
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    trace = [native.cell.v.clone()]
    prefix, first = runner()
    for _ in range(5):
        native.step()
        trace.append(native.cell.v.clone())
    _close(prefix, runner.functional.extract(native).state)
    repeated, repeated_aux = runner()
    _close(repeated, prefix)
    _close(repeated_aux["callbacks"]["trace"], first["callbacks"]["trace"])
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    for _ in range(5):
        native.step()
        trace.append(native.cell.v.clone())
    _close(final, runner.functional.extract(native).state)
    torch.testing.assert_close(
        torch.cat(
            (
                first["callbacks"]["trace"]["cell.v"],
                second["callbacks"]["trace"]["cell.v"],
            )
        ),
        torch.stack(trace),
    )
    assert runner.execution_report().strategy == "compiled_while_loop"
    assert runner.execution_report().requested_execution == "while"
    assert not final["populations"]["cell"]["integrator"]["v"].requires_grad
    _close(runner.initial_state, initial)
    _close(runner.functional.extract().state, source)
    _close((runner._parameters, runner._constants), defaults)


@pytest.mark.parametrize("backend", ("bitpacked_history", "sparse_calendar"))
def test_while_overflow_retains_prior_report_callback_rng_and_restart_state(backend):
    model = _calendar_network(
        backend,
        noise=0.6,
        start=100.0,
        schedules=([0, 1], [6 * DT, 6 * DT], [1.0, 1.0]),
    )
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        execution="while",
        inference=True,
        pending_capacity=1,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    prefix, auxiliary = runner()
    prior = runner.execution_report()
    assert prior.strategy == "compiled_while_loop"
    callback_state = auxiliary["callbacks"].state
    untouched = _clone(
        (prefix, callback_state.carries, runner.initial_state), detached=True
    )
    source = runner.functional.extract().state
    with pytest.raises(RuntimeError, match="overflow"):
        runner(prefix, callback_state=callback_state)
    assert runner.execution_report() is prior
    _close((prefix, callback_state.carries, runner.initial_state), untouched)
    _close(runner.functional.extract().state, source)


def test_while_runner_preserves_shared_parameter_aliases_and_explicit_override_leaves():
    runner = dn.compile(
        _stimulus(twins=True),
        dt=DT,
        steps=4,
        execution="while",
        inference=True,
        trainable="synapses.*.weight.weights.0.param.rho",
        backend="aot_eager",
    )
    names = [
        name
        for name in runner.named_parameters
        if name.startswith("synapses.") and name.endswith(".weight.weights.0.param.rho")
    ]
    assert len(names) == 2
    first, second = (runner.get_parameter(name) for name in names)
    assert first is second and first.requires_grad
    assert (
        runner.trainable_parameters[names[0]] is runner.trainable_parameters[names[1]]
    )
    original = first.detach().clone()
    upstream = first.detach().clone().requires_grad_()
    shared = upstream * 1.7
    parameters = {
        **runner._parameters,
        "synapses": {
            name: {**values, "weight.weights.0.param.rho": shared}
            for name, values in runner._parameters["synapses"].items()
        },
    }
    with torch.no_grad():
        expected, _ = runner.functional.run(
            parameters, runner._constants, runner.initial_state, steps=4
        )
    actual, _ = runner(parameters=parameters)
    _close(actual, expected)
    assert all(
        values["weight.weights.0.param.rho"] is shared
        for values in parameters["synapses"].values()
    )
    assert shared.requires_grad and shared.grad_fn is not None
    assert not actual["populations"]["cell"]["integrator"]["v"].requires_grad
    torch.testing.assert_close(first, original)
    prior = runner.execution_report()
    broken = {
        **parameters,
        "synapses": {
            name: {**values} for name, values in parameters["synapses"].items()
        },
    }
    next(iter(broken["synapses"].values()))["weight.weights.0.param.rho"] = (
        shared.clone()
    )
    with pytest.raises(ValueError, match="shared|aliases"):
        runner(parameters=broken)
    assert runner.execution_report() is prior


def test_while_continuous_transform_preserves_explicit_parameter_view_metadata():
    runner = dn.compile(
        _layout_network(),
        dt=DT,
        steps=4,
        execution="while",
        inference=True,
        trainable="continuous_synapses.*.transform_parameters.scale",
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("post.v", "t"))},
    )
    functional = runner.functional
    name = next(iter(functional.continuous_synapses))
    original = runner._parameters["continuous_synapses"][name]
    scan = functional.compile_rollout_chunk(4, execution="scan", backend="aot_eager")
    frozen = _clone((runner._parameters, runner.initial_state), detached=True)
    values = original["transform_parameters"]["scale"].detach().clone()
    outputs = []
    for offset, stride in ((0, 1), (1, 2), (2, 2)):
        storage = values.new_zeros(offset + stride * values.numel()).requires_grad_()
        with torch.no_grad():
            storage[offset::stride].copy_(values)
        parameter = storage[offset::stride]
        assert parameter.requires_grad and parameter.grad_fn is not None
        assert parameter.storage_offset() == offset and parameter.stride() == (stride,)
        parameters = {
            **runner._parameters,
            "continuous_synapses": {
                name: {
                    **original,
                    "transform_parameters": {
                        **original["transform_parameters"],
                        "scale": parameter,
                    },
                },
            },
        }
        native = _layout_network()
        native.continuous_synapses[name].transform.scale = torch.nn.Parameter(
            parameter.detach()
        )
        with torch.no_grad():
            expected, oracle = functional.prepare_and_rollout(
                parameters,
                runner._constants,
                runner.initial_state,
                steps=4,
                callbacks=runner._callbacks,
            )
            scanned, scan_aux = functional.run(
                parameters,
                runner._constants,
                runner.initial_state,
                steps=4,
                callbacks=runner._callbacks,
                compiled_chunk=scan,
            )
            for _ in range(4):
                native.step()
        actual, auxiliary = runner(parameters=parameters)
        _close(actual, expected)
        _close(actual, scanned)
        _close(actual, functional.extract(native).state)
        _close(auxiliary["callbacks"]["trace"], oracle["callbacks"]["trace"])
        _close(auxiliary["callbacks"]["trace"], scan_aux["callbacks"]["trace"])
        assert (
            parameters["continuous_synapses"][name]["transform_parameters"]["scale"]
            is parameter
        )
        assert parameter.storage_offset() == offset and parameter.stride() == (stride,)
        assert parameter.requires_grad and parameter.grad_fn is not None
        outputs.append(actual["populations"]["post"]["integrator"]["v"])
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])
    assert runner.execution_report().strategy == "compiled_while_loop"
    _close((runner._parameters, runner.initial_state), frozen)


@pytest.mark.parametrize(("surrogate", "taps"), ((True, 2), (False, 3)))
def test_while_inference_keeps_training_source_history_and_differentiable_delay_policy(
    surrogate, taps
):
    model, native = (
        _stim_history(surrogate=surrogate, taps=taps, noise=0.6) for _ in range(2)
    )
    _sync_stimulus(model, native)
    steps = 8
    runner = dn.compile(
        model,
        dt=DT,
        steps=steps,
        execution="while",
        inference=True,
        trainable=(
            "synapses.*.weight.weights.0.param.rho",
            "synapses.*.delay_ms.weights.0.param.rho",
        ),
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    functional = runner.functional
    assert model.training and functional.training
    assert all(
        plan.runtime == "source_history" for plan in functional.synapses.values()
    )
    assert all(
        values["delay_ms.weights.0.param.rho"] is not None
        for values in runner._parameters["synapses"].values()
    )
    inputs = dn.func.NetworkRolloutInput(
        netstim=functional.sample_draw_tape(runner.initial_state, steps)
    )
    frozen = _clone(
        (runner._parameters, runner._constants, runner.initial_state, inputs),
        detached=True,
    )
    scan = functional.compile_rollout_chunk(
        steps, execution="scan", backend="aot_eager"
    )
    with torch.no_grad():
        expected, oracle = functional.prepare_and_rollout(
            runner._parameters,
            runner._constants,
            runner.initial_state,
            inputs,
            steps=steps,
            callbacks=runner._callbacks,
        )
        scanned, scan_aux = functional.run(
            runner._parameters,
            runner._constants,
            runner.initial_state,
            inputs,
            steps=steps,
            compiled_chunk=scan,
            callbacks=runner._callbacks,
        )
        for _ in range(steps):
            native.step()
    actual, auxiliary = runner(inputs=inputs)
    _close(actual, expected)
    _close(actual, scanned)
    _close(actual, functional.extract(native).state)
    _close(auxiliary["callbacks"]["trace"], oracle["callbacks"]["trace"])
    _close(auxiliary["callbacks"]["trace"], scan_aux["callbacks"]["trace"])
    history = "source_gate_history" if surrogate else "spike_history_packed"
    for name, plan in functional.synapses.items():
        assert (
            plan.runtime
            == native.synapses[name]._checkpoint_runtime_kind()
            == "source_history"
        )
        assert plan._diff_delays and plan._taps == taps
        carry = actual["synapses"][name]
        assert carry["delivery_buffer"].shape[0] == 1
        assert carry[history].any()
        assert carry[history].dtype == (DTYPE if surrogate else torch.int64)
        assert not carry[history].requires_grad
    assert runner.execution_report().strategy == "compiled_while_loop"
    assert not actual["populations"]["cell"]["integrator"]["v"].requires_grad
    _close(
        (runner._parameters, runner._constants, runner.initial_state, inputs), frozen
    )
    _close(functional.extract().state, frozen[2])


def test_while_rejects_ordinary_training_gradients_and_scan_retains_them():
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="aot_eager")
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="[Ss]can|inference"
    ):
        chunk(tensors.parameters, prepared, tensors.state)
    assert chunk.execution_report() is None
    with torch.no_grad():
        actual, _ = chunk(tensors.parameters, prepared, tensors.state)
        expected, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=4
        )
    _close(actual, expected)
    prior = chunk.execution_report()
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="[Ss]can|inference"
    ):
        chunk(tensors.parameters, prepared, tensors.state)
    assert chunk.execution_report() is prior
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="[Ss]can|inference"
    ):
        dn.compile(model, dt=DT, steps=4, execution="while", inference=False)
    scan = functional.compile_rollout_chunk(4, execution="scan", backend="aot_eager")
    final, _ = scan(tensors.parameters, prepared, tensors.state)
    weights = next(iter(tensors.parameters["synapses"].values()))[
        "weight.weights.0.param.rho"
    ]
    gradient = torch.autograd.grad(_loss(final), weights)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("steps", (1, 4))
def test_while_custom_callback_reducers_work_for_tail_and_loop_dispatch(steps):
    model = _eval(_mixed_network())
    functional, tensors = dn.func.make_functional_network(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(
        steps, execution="while", backend="aot_eager"
    )
    callbacks = functional.make_callbacks({"energy": _VoltageEnergy()})
    with torch.no_grad():
        actual, auxiliary = chunk(
            tensors.parameters, prepared, tensors.state, callbacks=callbacks
        )
        expected, reference = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            steps=steps,
            callbacks=callbacks,
        )
    _close(actual, expected)
    _close(auxiliary["callbacks"]["energy"], reference["callbacks"]["energy"])
    assert chunk.execution_report().strategy == (
        "compiled_while_loop" if steps >= 4 else "compiled_unrolled"
    )
    runner = dn.compile(
        model,
        dt=DT,
        steps=steps,
        execution="while",
        inference=True,
        callbacks={"energy": _VoltageEnergy()},
        backend="aot_eager",
    )
    observed, result = runner()
    _close(observed, expected)
    _close(result["callbacks"]["energy"], reference["callbacks"]["energy"])


def test_while_disabled_lower_level_dispatch_is_reported_and_convenience_rejects_it():
    model = _recurrent()
    functional, tensors = dn.func.make_functional_network(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(4, execution="while", disable=True)
    with torch.no_grad():
        actual, _ = chunk(tensors.parameters, prepared, tensors.state)
        expected, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=4
        )
    _close(actual, expected)
    assert chunk.execution_report().strategy == "eager_disabled"
    assert chunk.execution_report().requested_execution == "while"
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="disable|compil"
    ):
        dn.compile(
            model, dt=DT, steps=4, execution="while", inference=True, disable=True
        )


@pytest.mark.parametrize("transform", ("grad", "jacfwd", "vmap"))
def test_while_explicit_transforms_keep_the_pure_eager_fallback(transform):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="aot_eager")

    def evaluate(voltage, compiled):
        population = tensors.state["populations"]["cell"]
        state = {
            **tensors.state,
            "populations": {
                "cell": {
                    **population,
                    "integrator": {**population["integrator"], "v": voltage},
                }
            },
        }
        if compiled:
            final, _ = chunk(tensors.parameters, prepared, state)
        else:
            final, _ = functional.rollout(tensors.parameters, prepared, state, steps=4)
        return _loss(final)

    voltage = tensors.state["populations"]["cell"]["integrator"]["v"].detach()
    if transform == "vmap":
        voltage = torch.stack((voltage, voltage + 0.2))
    operation = getattr(torch.func, transform)
    expected = operation(lambda value: evaluate(value, False))(voltage)
    actual = operation(lambda value: evaluate(value, True))(voltage)
    torch.testing.assert_close(actual, expected)
    assert chunk.execution_report().strategy == "eager_transform_fallback"


def test_while_source_reinitialization_rejects_cached_graph_and_preserves_report():
    model = _recurrent()
    functional, tensors = dn.func.make_functional_network(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="aot_eager")
    with torch.no_grad():
        first, _ = chunk(tensors.parameters, prepared, tensors.state)
    frozen = _clone(first, detached=True)
    prior = chunk.execution_report()
    model.initialize(DT, reinit_weights=False, reinit_delays=False)
    with (
        torch.no_grad(),
        pytest.raises(dn.func.FunctionalizationError, match="changed|lower|initializ"),
    ):
        chunk(tensors.parameters, prepared, tensors.state)
    assert chunk.execution_report() is prior
    _close(first, frozen)
