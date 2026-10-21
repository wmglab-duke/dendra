"""Compiled packed inference preserves native receives, compact carry, and RNG."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, pas
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_rollout import _native_driven_step, _slice
from tests.networks.test_functional_event_delivery import ThresholdRelay

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _initialize(network, *, batch=None):
    if batch is not None:
        network.batch(batch)
    network.eval()
    network.initialize(DT, max_delay_ms=6 * DT)
    for connection in network.synapses.values():
        assert connection._checkpoint_runtime_kind() == "bitpacked_history"
        assert connection.advance.__name__ == "advance_non_diff_bitpacked_history"
        assert connection.delivery_buffer.shape[0] == 1
    return network


def _recurrent(*, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(ThresholdRelay.rename("relay"))
        network = dn.Network({"cell": cell}, netcon_delay_backend="bitpacked_history")
        for field, threshold, weight in (("level", 0.0, 0.2), ("mirror", 1.5, 0.3)):
            network.connect_one_to_one(
                cell[:],
                cell[:],
                cell.mech.relay,
                pre_var=f"mech.relay.{field}",
                threshold=threshold,
                weight=weight,
                delay=DT,
                allow_autapses=True,
            )
        return _initialize(network, batch=batch)


def _stimulus(*, noise=0.6, schedule=None, long_interval=False, twins=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(
            N=2, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(pas, g=0.001, e=-65.0)
        targets = ("syn", "other") if twins else ("syn",)
        for name in targets:
            cell.insert(expsyn.rename(name), tau=1.1, e=0.0)
        stimulus = dn.NetStim(
            N=2,
            interval=100.0 if long_interval or schedule else 0.5,
            start=100.0 if schedule else 0.0,
            noise=noise,
            max_spikes=30,
            seed=812,
        )
        stimulus.set_diff_config(diff_scheduled_times=False)
        network = dn.Network(
            {"cell": cell}, netstim=stimulus, netcon_delay_backend="bitpacked_history"
        )
        for name in targets:
            network.connect_one_to_one(
                stimulus[:],
                cell[:],
                cell.mech.mechanisms[name],
                weight=torch.tensor([0.002, 0.003], dtype=DTYPE),
                delay=torch.tensor([2 * DT, 3 * DT], dtype=DTYPE),
            )
        _initialize(network)
        if twins:
            first, second = network.synapses.values()
            second.weight.w = first.weight.w
        if schedule in ("tensor", "mixed"):
            stimulus.schedule(
                [0, 1, 1],
                torch.tensor([0.0, DT, 3 * DT], dtype=DTYPE),
                weight=torch.tensor([0.25, -2.0, 0.0], dtype=DTYPE),
                allow_past=True,
                differentiable=True,
            )
        if schedule in ("heap", "mixed"):
            stimulus.schedule(
                [0, 1, 0],
                [2 * DT, 4 * DT, 4 * DT],
                allow_past=True,
                differentiable=False,
            )
        return network


def _sync_stimulus(source, *targets):
    # A fixed renewal seed does not fix the independent entropy-seeder packet.
    for target in targets:
        target.netstim.restore_dict_from_checkpoint(
            _clone(source.netstim.state_dict_for_checkpoint(), detached=True)
        )


def _assert_scratch(functional, state, native):
    assert all(
        plan.runtime == "bitpacked_history" for plan in functional.synapses.values()
    )
    for name, carry in state["synapses"].items():
        torch.testing.assert_close(
            carry["delivery_buffer"], native.synapses[name].delivery_buffer
        )
        assert carry["delivery_buffer"].shape[0] == 1


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("batch", (None, 2))
def test_packed_recurrent_inference_preserves_receive_observe_order_and_inputs(
    compiled, batch
):
    model, native = _recurrent(batch=batch), _recurrent(batch=batch)
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors, detached=True)
    state = tensors.state
    advance = functional.prepare_and_step
    if compiled:
        advance = torch.compile(advance, backend="aot_eager", fullgraph=True)
    delivered = False
    for index in range(11):
        incoming = state
        untouched = _clone(incoming, detached=True)
        state, _ = advance(tensors.parameters, tensors.constants, incoming)
        native.step()
        _close(state, functional.extract(native).state)
        _close(incoming, untouched)
        _assert_scratch(functional, state, native)
        delivered |= any(
            carry["delivery_buffer"].count_nonzero()
            for carry in state["synapses"].values()
        )
        if index == 0:
            relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
            torch.testing.assert_close(
                relay["level"], torch.full_like(relay["level"], 1.5)
            )
            # The second projection sees the threshold reached by its own
            # receive, including a zero-payload receive on the first step.
            assert all(
                carry["bitpack_source_has_spiked"].all()
                for carry in state["synapses"].values()
            )
    assert delivered
    _close(tensors, before)
    _close(functional.extract().state, before.state)


def test_packed_pending_source_uses_delivery_time_weights_and_fixed_offsets():
    model, native = (_stimulus(noise=0.0, long_interval=True) for _ in range(2))
    _sync_stimulus(model, native)
    functional, tensors = dn.func.make_functional_network(model)
    name = next(iter(model.synapses))
    state, _ = functional.prepare_and_step(
        *tensors, netstim_draw=functional.sample_step_draw(tensors.state)
    )
    native.step()
    _close(state, functional.extract(native).state)
    assert state["netstim"]["spike_counts"].eq(1).all()
    original = tensors.parameters["synapses"][name]
    assert original["delays"] is None
    weights = original["weights"] * 4.0
    parameters = {
        **tensors.parameters,
        "synapses": {name: {**original, "weights": weights}},
    }
    native.synapses[name].weight.w = weights.detach().clone()
    scratch = []
    for _ in range(8):
        state, _ = functional.prepare_and_step(
            parameters,
            tensors.constants,
            state,
            netstim_draw=functional.sample_step_draw(state),
        )
        native.step()
        _close(state, functional.extract(native).state)
        scratch.append(state["synapses"][name]["delivery_buffer"].clone())
    # The source emitted before replacement; its two distinct initialized
    # integer delays still deliver the replacement weights exactly once.
    torch.testing.assert_close(torch.stack(scratch).sum(dim=0).reshape(-1), weights)
    assert state["synapses"][name]["delivery_buffer"].count_nonzero() == 0


def test_packed_noisy_rng_and_population_input_tapes_match_native_and_commit():
    model, native, resumed = (_stimulus() for _ in range(3))
    _sync_stimulus(model, native, resumed)
    functional, tensors = dn.func.make_functional_network(model)
    frozen = _clone(tensors, detached=True)
    steps = 12
    drive = (
        torch.linspace(-0.2, 0.4, steps * model.cell.v.numel(), dtype=DTYPE).reshape(
            steps, *model.cell.shape
        )
        * 1e-5
    )
    inputs = dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=drive)},
        netstim=functional.sample_draw_tape(tensors.state, steps),
    )
    prefix, _ = functional.run(*tensors, _slice(inputs, 0, 5), steps=5)
    for index in range(5):
        _native_driven_step(native, {"cell": dn.func.StepInput(intra=drive[index])})
    _close(prefix, functional.extract(native).state)
    functional.commit_state_(resumed, prefix)
    actual, _ = functional.run(
        tensors.parameters, tensors.constants, prefix, _slice(inputs, 5, 12), steps=7
    )
    for index in range(5, steps):
        for target in (native, resumed):
            _native_driven_step(target, {"cell": dn.func.StepInput(intra=drive[index])})
    _close(actual, functional.extract(native).state)
    _close(actual, functional.extract(resumed).state)
    _assert_scratch(functional, actual, native)
    _close(tensors, frozen)
    _close(functional.extract().state, frozen.state)


@pytest.mark.parametrize("kind", ("tensor", "heap", "mixed"))
def test_packed_netstim_schedules_use_hard_source_spikes_and_continue_from_checkpoint(
    kind,
):
    model, native, resumed = (_stimulus(noise=0.0, schedule=kind) for _ in range(3))
    _sync_stimulus(model, native, resumed)
    functional, tensors = dn.func.make_functional_network(model)
    state = tensors.state
    name = next(iter(model.synapses))
    checkpoint = None
    for index in range(10):
        incoming = state
        state, auxiliary = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
            netstim_draw=functional.sample_step_draw(state),
        )
        native.step()
        _close(state, functional.extract(native).state)
        slot = incoming["synapses"][name]["current_time_step"]
        packed = state["synapses"][name]["spike_history_packed"][slot].reshape(-1)
        expected_word = (
            native.netstim.spikes.reshape(-1).to(torch.int64).dot(torch.tensor([1, 2]))
        )
        torch.testing.assert_close(packed, expected_word.reshape(1))
        torch.testing.assert_close(auxiliary["netstim"].spikes, native.netstim.spikes)
        if kind in ("tensor", "mixed") and index in (0, 1, 3):
            assert native.netstim.spikes.any()
            assert not torch.equal(
                native.netstim.spike_gate, native.netstim.spikes.to(DTYPE)
            )
        if index == 3:
            checkpoint = _clone(state)
    assert checkpoint is not None
    if kind in ("heap", "mixed"):
        assert state["netstim"]["heap_cursor"].sum() > 0
    functional.commit_state_(resumed, checkpoint)
    replay = checkpoint
    for _ in range(4, 10):
        replay, _ = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            replay,
            netstim_draw=functional.sample_step_draw(replay),
        )
        resumed.step()
        _close(replay, functional.extract(resumed).state)
    _close(replay, state)


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_compiled_packed_inference_restarts_and_continues_nonempty_carry(
    initialization,
):
    model, native, resumed = (_stimulus(twins=True) for _ in range(3))
    _sync_stimulus(model, native, resumed)
    for _ in range(3):
        model.step()
        native.step()
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        inference=True,
        initialization=initialization,
        backend="aot_eager",
        trainable="synapses.*.weights",
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    initial = _clone(runner.initial_state, detached=True)
    source = runner.functional.extract().state
    assert any(
        value["spike_history_packed"].count_nonzero()
        for value in initial["synapses"].values()
    )
    aliases = tuple(runner.trainable_parameters.values())
    assert len(aliases) == 2 and aliases[0] is aliases[1]
    if initialization == "fresh":
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    expected_trace = [native.cell.v.clone()]
    prefix, first = runner()
    for _ in range(4):
        native.step()
        expected_trace.append(native.cell.v.clone())
    _close(prefix, runner.functional.extract(native).state)
    repeated, repeated_aux = runner()
    _close(repeated, prefix)
    _close(repeated_aux["callbacks"]["trace"], first["callbacks"]["trace"])
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    for _ in range(4):
        native.step()
        expected_trace.append(native.cell.v.clone())
    _close(final, runner.functional.extract(native).state)
    trace = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(trace, torch.stack(expected_trace))
    assert trace.shape[0] == 9 and not trace.requires_grad
    assert not final["populations"]["cell"]["integrator"]["v"].requires_grad
    runner.functional.commit_state_(resumed, final)
    resumed.step()
    native.step()
    _close(
        runner.functional.extract(resumed).state,
        runner.functional.extract(native).state,
    )
    _close(runner.initial_state, initial)
    _close(runner.functional.extract().state, source)
    assert runner.execution_report().strategy == "compiled_scan"


def test_packed_eval_network_rejects_gradient_runner_and_accepts_netcon_side_traffic():
    network = _stimulus()
    with pytest.raises(dn.func.FunctionalizationError, match="training mode"):
        dn.compile(network, dt=DT, steps=2, inference=False, backend="eager")
    connection = next(iter(network.synapses.values()))
    connection.schedule(con_indices=[0], times_ms=[DT], weight=0.7)
    functional, _ = dn.func.make_functional_network(network)
    assert all(
        plan.runtime == "bitpacked_history" for plan in functional.synapses.values()
    )
    other = _stimulus()
    connection = next(iter(other.synapses.values()))
    connection._sparse_calendar[0] = [
        (connection.post_idx[:1], connection.weight.w[:1])
    ]
    _, tensors = dn.func.make_functional_network(other)
    assert next(iter(tensors.state["synapses"].values()))["pending_size"].item() == 1
