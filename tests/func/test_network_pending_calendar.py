"""Connected compact calendars preserve delivery, continuation, and failures."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, pas
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_bitpacked_inference import _sync_stimulus
from tests.func.test_network_rollout import _native_driven_step, _slice, _VoltageEnergy

pytestmark = pytest.mark.cpu
BACKENDS = ("bitpacked_history", "sparse_calendar")


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _network(backend, *, noise=0.0, start=0.0, interval=100.0, schedules=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(
            N=2, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.1, e=0.0)
        stimulus = dn.NetStim(
            N=2, interval=interval, start=start, noise=noise, max_spikes=30, seed=812
        )
        stimulus.set_diff_config(diff_scheduled_times=False)
        network = dn.Network(
            {"cell": cell},
            netstim=stimulus,
            netcon_delay_backend=backend,
            track_netcon_events=backend == "sparse_calendar",
        )
        network.connect_one_to_one(
            stimulus[:],
            cell[:],
            cell.mech.syn,
            weight=torch.tensor([0.002, 0.003], dtype=DTYPE),
            delay=torch.tensor([2 * DT, 3 * DT], dtype=DTYPE),
        )
        network.eval()
        network.initialize(DT, max_delay_ms=6 * DT)
        connection = next(iter(network.synapses.values()))
        if schedules is None:
            schedules = (
                [0, 1, 0, 1],
                [0.0, 0.0, 4 * DT, 7 * DT],
                [0.25, -2, 1.5, 0.75],
            )
        indices, times, amplitudes = schedules
        connection.schedule(
            con_indices=indices,
            times_ms=times,
            weight=torch.tensor(amplitudes, dtype=DTYPE),
        )
        assert connection._checkpoint_runtime_kind() == backend
        assert connection.delivery_buffer.shape[0] == 1
        return network


def _pending_count(network):
    return sum(
        values.numel()
        for connection in network.synapses.values()
        for chunks in connection.state_dict_for_checkpoint()["backend_state"][
            "sparse_calendar"
        ].values()
        for _, values in chunks
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("compiled", (False, True))
def test_connected_schedules_preserve_enqueue_weights_and_native_checkpoint(
    backend, compiled
):
    model, native, resumed = (_network(backend) for _ in range(3))
    _sync_stimulus(model, native, resumed)
    functional, tensors = dn.func.make_functional_network(model, pending_capacity=64)
    frozen = _clone(tensors, detached=True)
    name = next(iter(model.synapses))
    advance = functional.prepare_and_step
    if compiled:
        advance = torch.compile(advance, backend="aot_eager", fullgraph=True)
    state, _ = advance(
        *tensors, netstim_draw=functional.sample_step_draw(tensors.state)
    )
    native.step()
    _close(state, functional.extract(native).state)
    assert _pending_count(native) == 2
    checkpoint = _clone(native.state_dict_for_checkpoint(), detached=True)
    resumed.restore_dict_from_checkpoint(checkpoint)
    original = tensors.parameters["synapses"][name]
    assert original["delays"] is None
    weights = original["weights"] * 4
    parameters = {
        **tensors.parameters,
        "synapses": {name: {**original, "weights": weights}},
    }
    for target in (native, resumed):
        target.synapses[name].weight.w = weights.detach().clone()
    deliveries = []
    for _ in range(12):
        incoming = state
        untouched = _clone(incoming, detached=True)
        state, _ = advance(
            parameters,
            tensors.constants,
            incoming,
            netstim_draw=functional.sample_step_draw(incoming),
        )
        for target in (native, resumed):
            target.step()
            _close(state, functional.extract(target).state)
        _close(incoming, untouched)
        deliveries.append(state["synapses"][name]["delivery_buffer"].clone())
    # Packed intrinsic history uses replacement weights at delivery. Both
    # runtimes retain the weights already multiplied into scheduled traffic.
    factors = [10.25, 5.0] if backend == "bitpacked_history" else [7.25, 2.0]
    torch.testing.assert_close(
        torch.stack(deliveries).sum(dim=0).flatten(),
        original["weights"] * torch.tensor(factors, dtype=DTYPE),
    )
    assert _pending_count(native) == 0
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)


def test_connected_sparse_event_counts_survive_cancellation_and_zero_weights():
    schedules = ([0, 1], [0.0, 0.0], [-1.0, 2.0])
    model, native = (_network("sparse_calendar", schedules=schedules) for _ in range(2))
    _sync_stimulus(model, native)
    name = next(iter(model.synapses))
    for network in (model, native):
        network.synapses[name].weight.w = torch.tensor([0.002, 0.0], dtype=DTYPE)
    functional, tensors = dn.func.make_functional_network(model, pending_capacity=8)
    state = tensors.state
    deliveries, counts = [], []
    for _ in range(7):
        state, _ = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
            netstim_draw=functional.sample_step_draw(state),
        )
        native.step()
        _close(state, functional.extract(native).state)
        deliveries.append(state["synapses"][name]["delivery_buffer"])
        counts.append(state["synapses"][name]["events"])
    assert torch.stack(deliveries).count_nonzero() == 0
    torch.testing.assert_close(
        torch.stack(counts).sum(dim=0, dtype=torch.int32).flatten(),
        torch.tensor([2, 2], dtype=torch.int32),
    )
    queue = getattr(next(iter(native.synapses.values())), "event_queue", None)
    assert queue is None or queue.numel() == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("execution", ("scan", "unrolled"))
def test_connected_pending_rollout_chunks_preserve_rng_inputs_and_native_commit(
    backend, execution
):
    model, native, resumed = (
        _network(backend, noise=0.6, interval=0.5) for _ in range(3)
    )
    _sync_stimulus(model, native, resumed)
    functional, tensors = dn.func.make_functional_network(model)
    assert all(plan.pending_capacity == 16 for plan in functional.synapses.values())
    frozen = _clone(tensors, detached=True)
    steps = 12
    drive = torch.linspace(
        -2e-6, 4e-6, steps * model.cell.v.numel(), dtype=DTYPE
    ).reshape(steps, *model.cell.shape)
    inputs = dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=drive)},
        netstim=functional.sample_draw_tape(tensors.state, steps),
    )
    expected, _ = functional.prepare_and_rollout(*tensors, inputs, steps=steps)
    chunk = functional.compile_rollout_chunk(
        3, execution=execution, backend="aot_eager", fullgraph=True
    )
    prefix, _ = functional.run(
        *tensors, _slice(inputs, 0, 2), steps=2, compiled_chunk=chunk, chunklength=3
    )
    for index in range(2):
        _native_driven_step(native, {"cell": dn.func.StepInput(intra=drive[index])})
    _close(prefix, functional.extract(native).state)
    assert _pending_count(native) > 0
    functional.commit_state_(resumed, prefix)
    actual, _ = functional.run(
        tensors.parameters,
        tensors.constants,
        prefix,
        _slice(inputs, 2, steps),
        steps=steps - 2,
        compiled_chunk=chunk,
        chunklength=3,
        checkpoint_every=4,
    )
    for index in range(2, steps):
        for target in (native, resumed):
            _native_driven_step(target, {"cell": dn.func.StepInput(intra=drive[index])})
    _close(actual, expected)
    _close(actual, functional.extract(native).state)
    _close(actual, functional.extract(resumed).state)
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_compile_pending_network_restarts_and_continues_recordings(
    backend, initialization
):
    model, native = (_network(backend) for _ in range(2))
    _sync_stimulus(model, native)
    for network in (model, native):
        network.step()
    name = next(iter(model.synapses))
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        inference=True,
        initialization=initialization,
        pending_capacity={name: 32},
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    assert runner.functional.synapses[name].pending_capacity == 32
    initial = _clone(runner.initial_state, detached=True)
    source = runner.functional.extract().state
    if initialization == "fresh":
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    trace = [native.cell.v.clone()]
    prefix, first = runner()
    for _ in range(4):
        native.step()
        trace.append(native.cell.v.clone())
    _close(prefix, runner.functional.extract(native).state)
    repeated, repeated_aux = runner()
    _close(repeated, prefix)
    _close(repeated_aux["callbacks"]["trace"], first["callbacks"]["trace"])
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    for _ in range(4):
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
    _close(runner.initial_state, initial)
    _close(runner.functional.extract().state, source)
    assert not final["populations"]["cell"]["integrator"]["v"].requires_grad


@pytest.mark.parametrize("backend", BACKENDS)
def test_pending_overflow_rejects_host_results_and_commit_without_mutation(backend):
    model = _network(backend, start=100.0, schedules=([0, 1], [DT, DT], [1.0, 1.0]))
    functional, tensors = dn.func.make_functional_network(model, pending_capacity=1)
    frozen = _clone(tensors, detached=True)
    state, _ = functional.prepare_and_step(
        *tensors, netstim_draw=functional.sample_step_draw(tensors.state)
    )
    untouched = _clone(state, detached=True)
    draw = functional.sample_step_draw(state)
    with pytest.raises(RuntimeError, match="pending.*overflow|overflow.*pending"):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, state, netstim_draw=draw
        )
    compiled = torch.compile(
        functional.prepare_and_step, backend="aot_eager", fullgraph=True
    )
    poisoned, _ = compiled(
        tensors.parameters, tensors.constants, state, netstim_draw=draw
    )
    with pytest.raises(RuntimeError, match="overflow"):
        functional.check_overflow(poisoned)
    with pytest.raises(RuntimeError, match="overflow"):
        functional.run(tensors.parameters, tensors.constants, state, steps=2)
    with pytest.raises(RuntimeError, match="overflow"):
        functional.commit_state_(model, poisoned)
    _close(state, untouched)
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)
    # Native inference can grow its calendar rather than publishing partial traffic.
    model.step()
    model.step()
    assert _pending_count(model) == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_pending_overflow_keeps_prior_execution_report_and_snapshot(backend):
    model = _network(backend, start=100.0, schedules=([0, 1], [DT, DT], [1.0, 1.0]))
    runner = dn.compile(
        model, dt=DT, steps=1, inference=True, pending_capacity=1, backend="aot_eager"
    )
    source = runner.functional.extract().state
    initial = _clone(runner.initial_state, detached=True)
    prefix, _ = runner()
    report = runner.execution_report()
    untouched = _clone(prefix, detached=True)
    with pytest.raises(RuntimeError, match="overflow"):
        runner(prefix)
    assert runner.execution_report() == report
    _close(prefix, untouched)
    _close(runner.initial_state, initial)
    _close(runner.functional.extract().state, source)


@pytest.mark.parametrize("backend", BACKENDS)
def test_late_overflow_preserves_rng_callback_continuation_and_prior_report(backend):
    model = _network(
        backend,
        noise=0.6,
        start=100.0,
        schedules=([0, 1], [4 * DT, 4 * DT], [1.0, 1.0]),
    )
    functional, tensors = dn.func.make_functional_network(model, pending_capacity=1)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("cell.v", "t")),
            "energy": _VoltageEnergy(),
        }
    )
    chunk = functional.compile_rollout_chunk(2, disable=True)
    prefix, auxiliary = functional.run(
        *tensors, steps=1, compiled_chunk=chunk, callbacks=callbacks
    )
    callback_state = auxiliary["callbacks"].state
    prior_report = chunk.execution_report()
    assert prior_report is not None and prior_report.steps == 1
    frozen = _clone(tensors, detached=True)
    incoming = _clone(prefix, detached=True)
    carries = _clone(callback_state.carries, detached=True)
    emission_schemas = callback_state.emission_schemas
    global_rng = torch.random.get_rng_state().clone()
    # The first two-step span succeeds and publishes a different shared report;
    # scheduled events overflow in the following span after RNG/callback updates.
    with pytest.raises(RuntimeError, match="overflow"):
        functional.run(
            tensors.parameters,
            tensors.constants,
            prefix,
            steps=5,
            chunklength=2,
            compiled_chunk=chunk,
            callbacks=callbacks,
            callback_state=callback_state,
        )
    assert chunk.execution_report() is prior_report
    _close(prefix, incoming)
    _close(callback_state.carries, carries)
    assert callback_state.emission_schemas == emission_schemas
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng)
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)


@pytest.mark.parametrize(
    ("backend", "traffic"),
    (
        ("bitpacked_history", "payload"),
        ("sparse_calendar", "payload"),
        ("sparse_calendar", "events"),
    ),
)
def test_positive_capacity_rejects_larger_imported_payload_or_event_calendar(
    backend, traffic
):
    model = _network(
        backend,
        start=100.0,
        schedules=([0, 1], [0.0, 0.0], [1.0, 1.0]),
    )
    connection = next(iter(model.synapses.values()))
    if traffic == "events":
        connection.weight.w = torch.zeros_like(connection.weight.w)
    model.step()
    checkpoint = _clone(model.state_dict_for_checkpoint(), detached=True)
    assert _pending_count(model) == (0 if traffic == "events" else 2)
    if traffic == "events":
        assert (
            sum(
                counts.numel()
                for chunks in connection._sparse_event_calendar.values()
                for _, counts in chunks
            )
            == 2
        )
    with pytest.raises(ValueError, match="pending_capacity|capacity"):
        dn.func.make_functional_network(model, pending_capacity=1)
    _close(model.state_dict_for_checkpoint(), checkpoint)
