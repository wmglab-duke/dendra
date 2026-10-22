"""Multi-step Network execution retains native state, RNG, recordings, and AD."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _loss,
    _mixed_network,
    _native_loss,
    _stim_network,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _known_compiler_warnings():
    with torch_compiler_warning_context():
        yield


@dataclass(frozen=True)
class _VoltageEnergy(dn.func.FunctionalCallback):
    """A differentiable reducer exercising callback carry inside scan."""

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["populations"]["cell"]["integrator"]["v"]
        return {
            "energy": voltage.square().mean(),
            "frames": voltage.new_ones((), dtype=torch.long),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["populations"]["cell"]["integrator"]["v"]
        return {
            "energy": carry["energy"] + voltage.square().mean(),
            "frames": carry["frames"] + 1,
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry["energy"] / carry["frames"]


def _callbacks(functional):
    return functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("cell.v", "cell.mech.relay.level", "t")),
            "energy": _VoltageEnergy(),
        }
    )


def _mixed_targets(parameters, initial_level):
    return (
        initial_level,
        *(value["weights"] for value in parameters["synapses"].values()),
        *(value["weights"] for value in parameters["continuous_synapses"].values()),
        *(
            leaf
            for value in parameters["continuous_synapses"].values()
            for leaf in (value["transform_parameters"] or {}).values()
            if leaf.requires_grad
        ),
    )


def _native_mixed_targets(network, initial_level):
    return (
        initial_level,
        *(connection.weight.w for connection in network.synapses.values()),
        *(connection.weight.w for connection in network.continuous_synapses.values()),
        *(
            leaf
            for connection in network.continuous_synapses.values()
            if isinstance(connection.transform, torch.nn.Module)
            for leaf in connection.transform.parameters()
        ),
    )


def _native_observation(network):
    return {
        "cell.v": network.cell.v.clone(),
        "cell.mech.relay.level": network.cell.mech.relay.level.clone(),
        "t": network.t.clone(),
    }


def _slice(inputs, start, stop):
    if inputs is None:
        return None
    populations = None
    if inputs.populations is not None:
        populations = {
            name: dn.func.RolloutInput(
                None if value.ve is None else value.ve[start:stop],
                None if value.intra is None else value.intra[start:stop],
            )
            for name, value in inputs.populations.items()
        }
    netstim = None
    if inputs.netstim is not None:
        netstim = dn.func.NetStimDraw(
            inputs.netstim.exp_rand[start:stop], inputs.netstim.rng_state[start:stop]
        )
    return dn.func.NetworkRolloutInput(populations, netstim)


def _execute(functional, tensors, state, steps, callbacks, execution, inputs=None):
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    if execution == "eager":
        return functional.rollout(
            tensors.parameters,
            prepared,
            state,
            inputs,
            steps=steps,
            callbacks=callbacks,
        )
    chunk = functional.compile_rollout_chunk(
        steps, execution="scan", backend="aot_eager"
    )
    return chunk(tensors.parameters, prepared, state, inputs, callbacks=callbacks)


@pytest.mark.parametrize("execution", ("eager", "scan"))
def test_network_rollout_matches_native_recordings_carry_and_all_parameter_gradients(
    execution,
):
    steps = 6
    model, native = _mixed_network(), _mixed_network()
    for network in (model, native):
        network._reset_runtime_clock(2.75)
        network._set_duration_remainder(0.3 * DT)
    functional, tensors = dn.func.make_functional_network(model)
    frozen = _clone(tensors.state, detached=True)
    state = _clone(tensors.state)
    relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
    initial_level = relay["level"].detach().clone().requires_grad_()
    relay["level"] = initial_level
    native_level = initial_level.detach().clone().requires_grad_()
    native.cell.mech.relay.level = native_level
    observations = [_native_observation(native)]
    native_energy = native.cell.v.square().mean()
    for _ in range(steps):
        native.step()
        observations.append(_native_observation(native))
        native_energy = native_energy + native.cell.v.square().mean()
    expected_recording = {
        name: torch.stack([value[name] for value in observations])
        for name in observations[0]
    }
    callbacks = _callbacks(functional)
    final, auxiliary = _execute(functional, tensors, state, steps, callbacks, execution)
    result = auxiliary["callbacks"]
    _close(final, functional.extract(native).state)
    _close(result["trace"], expected_recording)
    torch.testing.assert_close(result["energy"], native_energy / (steps + 1))
    assert result.state["energy"]["frames"].dtype == torch.long
    assert result.state["energy"]["frames"].item() == steps + 1
    assert not final["clock"]["duration_remainder"].requires_grad
    loss = _loss(final) + result["energy"]
    expected_loss = _native_loss(native) + native_energy / (steps + 1)
    gradients = torch.autograd.grad(
        loss, _mixed_targets(tensors.parameters, initial_level)
    )
    expected = torch.autograd.grad(
        expected_loss, _native_mixed_targets(native, native_level)
    )
    _close(gradients, expected)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    _close(tensors.state, frozen)
    _close(functional.extract().state, frozen)


def _driven_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        populations = {
            name: dn.Unmyelinated(
                [2.0],
                L=4.0,
                dx=1.0,
                dtype=DTYPE,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
            for name in ("left", "right")
        }
        for name, cell in populations.items():
            cell.insert(pas, g=0.001, e=-60.0 if name == "left" else -70.0)
        network = dn.Network(populations)
        network.train()
        network.initialize(DT)
    return network


def _native_driven_step(network, values):
    """Use native phase execution with already assembled per-step drives."""
    network._sync_runtime_clock()
    network._step(
        network._population_step_items,
        network._synapse_step_items,
        network._continuous_synapse_step_items,
        network._continuous_target_step_items,
        network.netstim,
        network.t,
        network.dt,
        extra={
            name: value.ve for name, value in values.items() if value.ve is not None
        },
        intra={
            name: value.intra
            for name, value in values.items()
            if value.intra is not None
        },
        compile_network_ops=False,
    )
    network._advance_runtime_clock()


@pytest.mark.parametrize("execution", ("eager", "scan"))
def test_per_population_input_tapes_match_native_values_and_input_gradients(execution):
    steps = 4
    model, native = _driven_network(), _driven_network()
    functional, tensors = dn.func.make_functional_network(model)
    populations, native_inputs = {}, {}
    for index, (name, cell) in enumerate(model.populations.items()):
        shape = (steps, *cell.shape)
        ve = (
            torch.linspace(
                -0.3 + index, 0.5 + index, steps * cell.v.numel(), dtype=DTYPE
            )
            .reshape(shape)
            .requires_grad_()
        )
        intra = (
            torch.linspace(-0.7, 1.1 + index, steps * cell.v.numel(), dtype=DTYPE)
            .reshape(shape)
            .mul(1e-8)
            .requires_grad_()
        )
        populations[name] = dn.func.RolloutInput(ve, intra)
        native_inputs[name] = dn.func.RolloutInput(
            ve.detach().clone().requires_grad_(),
            intra.detach().clone().requires_grad_(),
        )
    inputs = dn.func.NetworkRolloutInput(populations=populations)
    before = _clone(inputs, detached=True)
    final, _ = _execute(
        functional, tensors, tensors.state, steps, None, execution, inputs
    )
    for step in range(steps):
        _native_driven_step(
            native,
            {
                name: dn.func.StepInput(value.ve[step], value.intra[step])
                for name, value in native_inputs.items()
            },
        )
    _close(final, functional.extract(native).state)
    actual_loss = sum(
        value["integrator"]["v"].square().sum()
        for value in final["populations"].values()
    )
    expected_loss = sum(cell.v.square().sum() for cell in native.populations.values())
    targets = tuple(
        leaf for pair in populations.values() for leaf in (pair.ve, pair.intra)
    )
    native_targets = tuple(
        leaf for pair in native_inputs.values() for leaf in (pair.ve, pair.intra)
    )
    actual = torch.autograd.grad(
        actual_loss, targets, allow_unused=True, materialize_grads=True
    )
    expected = torch.autograd.grad(
        expected_loss, native_targets, allow_unused=True, materialize_grads=True
    )
    _close(actual, expected)
    assert all(value.abs().sum() > 0 for value in actual)
    _close(inputs, before)


def test_netstim_tape_sampling_matches_sequential_packets_without_changing_rng():
    model = _stim_network()
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors.state, detached=True)
    native_rng = model.netstim._rng.get_state().clone()
    global_rng = torch.random.get_rng_state().clone()
    tape = functional.sample_draw_tape(tensors.state, 7)
    current = _clone(tensors.state)
    draws = []
    for _ in range(7):
        draw = functional.sample_step_draw(current)
        draws.append(draw)
        current = {
            **current,
            "netstim": {**current["netstim"], "rng_state": draw.rng_state},
        }
    torch.testing.assert_close(
        tape.exp_rand, torch.stack([value.exp_rand for value in draws])
    )
    torch.testing.assert_close(
        tape.rng_state, torch.stack([value.rng_state for value in draws])
    )
    assert tape.exp_rand.shape == (
        7,
        *tensors.state["netstim"]["next_stoch_time"].shape,
    )
    assert tape.rng_state.dtype == torch.uint8
    _close(tensors.state, before)
    torch.testing.assert_close(model.netstim._rng.get_state(), native_rng)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng)
    empty = functional.sample_draw_tape(tensors.state, 0)
    assert empty.exp_rand.shape[0] == empty.rng_state.shape[0] == 0
    resumed = {
        **tensors.state,
        "netstim": {**tensors.state["netstim"], "rng_state": tape.rng_state[-1]},
    }
    assert resumed["netstim"]["rng_state"].storage_offset() > 0
    _close(
        functional.sample_step_draw(resumed),
        functional.sample_step_draw(_clone(resumed, detached=True)),
    )


@pytest.mark.parametrize("execution", ("eager", "scan"))
def test_explicit_netstim_draw_tape_preserves_native_rng_clock_and_gradients(execution):
    steps = 9
    model, native = _stim_network(), _stim_network()
    for network in (model, native):
        network.netstim.noise.requires_grad_()
        network.netstim.tau = torch.tensor(0.17, dtype=DTYPE, requires_grad=True)
    native.netstim.restore_dict_from_checkpoint(
        _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
    )
    functional, tensors = dn.func.make_functional_network(model)
    tape = functional.sample_draw_tape(tensors.state, steps)
    inputs = dn.func.NetworkRolloutInput(netstim=tape)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("cell.v", "t"))})
    final, auxiliary = _execute(
        functional, tensors, tensors.state, steps, callbacks, execution, inputs
    )
    native_voltage = [native.cell.v.clone()]
    for _ in range(steps):
        native.step()
        native_voltage.append(native.cell.v.clone())
    _close(final, functional.extract(native).state)
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["cell.v"], torch.stack(native_voltage)
    )
    values = tensors.parameters["netstim"]["parameters"]
    weights = next(iter(tensors.parameters["synapses"].values()))["weights"]
    actual_loss = auxiliary["callbacks"]["trace"]["cell.v"].square().sum()
    expected_loss = torch.stack(native_voltage).square().sum()
    actual = torch.autograd.grad(
        actual_loss, (values["interval"], values["noise"], values["tau"], weights)
    )
    expected = torch.autograd.grad(
        expected_loss,
        (
            native.netstim.interval.rho,
            native.netstim.noise,
            native.netstim.tau,
            next(iter(native.synapses.values())).weight.w,
        ),
    )
    _close(
        (
            actual[0] * torch.sigmoid(native.netstim.interval.rho),
            actual[1],
            actual[2].sum(),
            actual[3],
        ),
        expected,
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in actual
    )


@pytest.mark.parametrize("compiled", (False, True))
def test_chunk_tails_checkpoint_backward_and_recording_preserve_prefix_gradients(
    compiled,
):
    steps = 8
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    callbacks = _callbacks(functional)

    def evaluate(checkpointed):
        state = _clone(tensors.state)
        relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
        initial = relay["level"].detach().clone().requires_grad_()
        relay["level"] = initial
        kernel = None
        if compiled:
            kernel = functional.compile_rollout_chunk(
                3, execution="scan", backend="aot_eager"
            )
        final, auxiliary = functional.run(
            tensors.parameters,
            tensors.constants,
            state,
            steps=steps,
            chunklength=5,
            checkpoint_every=4 if checkpointed else None,
            callbacks=callbacks,
            compiled_chunk=kernel,
        )
        result = auxiliary["callbacks"]
        loss = (
            _loss(final)
            + result["energy"]
            + 0.01 * result["trace"]["cell.v"].square().mean()
        )
        gradients = torch.autograd.grad(
            loss, _mixed_targets(tensors.parameters, initial)
        )
        return final, result, gradients

    expected = evaluate(False)
    actual = evaluate(True)
    _close(actual[:2], expected[:2])
    _close(actual[2], expected[2])
    assert actual[1]["trace"]["cell.v"].shape[0] == steps + 1
    assert actual[1].state["energy"]["frames"].item() == steps + 1


@pytest.mark.parametrize("compiled", (False, True))
def test_segmented_zero_steps_and_short_tails_do_not_duplicate_recordings(compiled):
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    callbacks = _callbacks(functional)
    expected, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=7,
        callbacks=callbacks,
    )
    reference = auxiliary["callbacks"]
    kernel = None
    if compiled:
        kernel = functional.compile_rollout_chunk(
            3, execution="scan", backend="aot_eager"
        )
    state, callback_state = tensors.state, None
    observations = []
    for steps in (0, 2, 0, 5, 0):
        state, result = functional.run(
            tensors.parameters,
            tensors.constants,
            state,
            steps=steps,
            chunklength=4,
            callbacks=callbacks,
            callback_state=callback_state,
            compiled_chunk=kernel,
        )
        outputs = result["callbacks"]
        callback_state = outputs.state
        observations.append(outputs["trace"])
    _close(state, expected)
    joined = {
        name: torch.cat([part[name] for part in observations])
        for name in reference["trace"]
    }
    _close(joined, reference["trace"])
    assert callback_state["energy"]["frames"].item() == 8


def test_noisy_netstim_host_checkpoint_chunks_replay_explicit_rng_tape():
    model = _stim_network()
    model.netstim.noise.requires_grad_()
    functional, tensors = dn.func.make_functional_network(model)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("cell.v", "t"))})
    draws = functional.sample_draw_tape(tensors.state, 9)
    tape = dn.func.NetworkRolloutInput(
        netstim=dn.func.NetStimDraw(draws.exp_rand.requires_grad_(), draws.rng_state)
    )
    expected, expected_aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        tape,
        steps=9,
        callbacks=callbacks,
    )
    chunk = functional.compile_rollout_chunk(4, execution="scan", backend="aot_eager")
    actual, actual_aux = functional.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        tape,
        steps=9,
        chunklength=5,
        checkpoint_every=3,
        callbacks=callbacks,
        compiled_chunk=chunk,
    )
    _close(actual, expected)
    _close(actual_aux["callbacks"]["trace"], expected_aux["callbacks"]["trace"])
    values = tensors.parameters["netstim"]["parameters"]
    weights = next(iter(tensors.parameters["synapses"].values()))["weights"]
    targets = (values["noise"], values["interval"], weights, tape.netstim.exp_rand)
    actual_gradients = torch.autograd.grad(
        actual_aux["callbacks"]["trace"]["cell.v"].square().sum(), targets
    )
    expected_gradients = torch.autograd.grad(
        expected_aux["callbacks"]["trace"]["cell.v"].square().sum(), targets
    )
    _close(actual_gradients, expected_gradients)
    assert actual_gradients[-1].abs().sum() > 0
    prefix, prefix_aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        _slice(tape, 0, 4),
        steps=4,
        callbacks=callbacks,
    )
    suffix, suffix_aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        prefix,
        _slice(tape, 4, 9),
        steps=5,
        callbacks=callbacks,
        callback_state=prefix_aux["callbacks"].state,
    )
    _close(suffix, expected)
    joined = torch.cat(
        (
            prefix_aux["callbacks"]["trace"]["cell.v"],
            suffix_aux["callbacks"]["trace"]["cell.v"],
        )
    )
    torch.testing.assert_close(joined, expected_aux["callbacks"]["trace"]["cell.v"])


def test_checkpoint_backward_preserves_shared_parameter_identity_and_gradient():
    model = _mixed_network()
    connections = tuple(model.synapses.values())
    connections[1].weight.w = connections[0].weight.w
    functional, tensors = dn.func.make_functional_network(model)
    values = tuple(tensors.parameters["synapses"].values())
    assert values[0]["weights"] is values[1]["weights"]
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=7
    )
    actual, _ = functional.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=7,
        chunklength=4,
        checkpoint_every=3,
    )
    _close(actual, expected)
    target = values[0]["weights"]
    actual_gradient = torch.autograd.grad(_loss(actual), target)[0]
    expected_gradient = torch.autograd.grad(_loss(expected), target)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0


def test_host_sampled_netstim_chunks_match_supplied_tape_rng_and_gradients():
    model = _stim_network()
    model.netstim.noise.requires_grad_()
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors.state, detached=True)
    native_rng = model.netstim._rng.get_state().clone()
    inputs = dn.func.NetworkRolloutInput(
        netstim=functional.sample_draw_tape(tensors.state, 11)
    )
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, inputs, steps=11
    )
    actual, _ = functional.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=11,
        chunklength=3,
        checkpoint_every=4,
    )
    _close(actual, expected)
    noise = tensors.parameters["netstim"]["parameters"]["noise"]
    actual_gradient = torch.autograd.grad(
        actual["populations"]["cell"]["integrator"]["v"].square().sum(), noise
    )[0]
    expected_gradient = torch.autograd.grad(
        expected["populations"]["cell"]["integrator"]["v"].square().sum(), noise
    )[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    _close(tensors.state, before)
    torch.testing.assert_close(model.netstim._rng.get_state(), native_rng)


@pytest.mark.parametrize("mutation", ("parameter", "draw_tape", "source"))
def test_checkpoint_replay_rejects_changed_parameters_tapes_and_source(mutation):
    model = _stim_network()
    model.netstim.noise.requires_grad_()
    functional, tensors = dn.func.make_functional_network(model)
    tape = functional.sample_draw_tape(tensors.state, 8)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder("cell.v")})
    _, auxiliary = functional.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.NetworkRolloutInput(netstim=tape),
        steps=8,
        chunklength=5,
        checkpoint_every=3,
        callbacks=callbacks,
    )
    target = next(iter(tensors.parameters["synapses"].values()))["weights"]
    if mutation == "source":
        model.eval()
    else:
        with torch.no_grad():
            if mutation == "parameter":
                target.add_(0.001)
            else:
                tape.exp_rand.add_(0.1)
    with pytest.raises((dn.func.FunctionalizationError, RuntimeError)):
        torch.autograd.grad(
            auxiliary["callbacks"]["trace"]["cell.v"].square().sum(), target
        )


@pytest.mark.parametrize(
    "invalid", ("length", "population", "rng_dtype", "steps", "missing_draws")
)
def test_invalid_network_tapes_reject_without_mutating_source_or_carry(invalid):
    model = _stim_network()
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors.state, detached=True)
    tape = functional.sample_draw_tape(tensors.state, 3)
    inputs = dn.func.NetworkRolloutInput(netstim=tape)
    steps = 3
    if invalid == "length":
        inputs = _slice(inputs, 0, 2)
    elif invalid == "population":
        inputs = dn.func.NetworkRolloutInput({"unknown": dn.func.RolloutInput()}, tape)
    elif invalid == "rng_dtype":
        inputs = dn.func.NetworkRolloutInput(
            netstim=dn.func.NetStimDraw(tape.exp_rand, tape.rng_state.float())
        )
    elif invalid == "steps":
        steps = -1
    else:
        inputs = None
    with pytest.raises((ValueError, TypeError, dn.func.FunctionalizationError)):
        functional.prepare_and_rollout(
            tensors.parameters, tensors.constants, tensors.state, inputs, steps=steps
        )
    _close(tensors.state, before)
    _close(functional.extract().state, before)


def test_unrolled_compiled_network_chunk_preserves_values_and_gradients():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(
        3, execution="unrolled", backend="aot_eager"
    )
    expected, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, steps=3
    )
    actual, _ = chunk(tensors.parameters, prepared, tensors.state)
    _close(actual, expected)
    target = next(iter(tensors.parameters["continuous_synapses"].values()))["weights"]
    actual_gradient = torch.autograd.grad(_loss(actual), target)[0]
    expected_gradient = torch.autograd.grad(_loss(expected), target)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient.abs().sum() > 0


def test_compiled_network_chunk_uses_eager_fallback_during_transforms():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    compile_calls = []

    def backend(graph_module, example_inputs):
        del example_inputs
        compile_calls.append(graph_module)
        return graph_module.forward

    chunk = functional.compile_rollout_chunk(2, execution="scan", backend=backend)

    def execute(offset, compiled):
        state = _clone(tensors.state)
        relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
        relay["level"] = relay["level"] + offset
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        if compiled:
            final, _ = chunk(tensors.parameters, prepared, state)
        else:
            final, _ = functional.rollout(tensors.parameters, prepared, state, steps=2)
        return final["populations"]["cell"]["integrator"]["v"]

    def actual(offset):
        return execute(offset, True)

    def expected(offset):
        return execute(offset, False)

    lanes = torch.tensor([-0.1, 0.0, 0.1], dtype=DTYPE)
    torch.testing.assert_close(torch.vmap(actual)(lanes), torch.vmap(expected)(lanes))
    torch.testing.assert_close(
        torch.func.jacrev(actual)(lanes[0]), torch.func.jacrev(expected)(lanes[0])
    )
    torch.testing.assert_close(
        torch.func.jvp(actual, (lanes[0],), (torch.ones_like(lanes[0]),)),
        torch.func.jvp(expected, (lanes[0],), (torch.ones_like(lanes[0]),)),
    )
    assert compile_calls == []


def test_network_scan_rejects_higher_order_backward():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(2, execution="scan", backend="aot_eager")
    final, _ = chunk(tensors.parameters, prepared, tensors.state)
    target = next(iter(tensors.parameters["continuous_synapses"].values()))["weights"]
    with pytest.raises(dn.func.FunctionalizationError, match="first-order"):
        torch.autograd.grad(_loss(final), target, create_graph=True)


def test_cached_network_chunk_consumes_fresh_preparations_after_backward():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    chunk = functional.compile_rollout_chunk(3, execution="scan", backend="aot_eager")
    voltages = []
    for scale in (0.7, 1.3):
        parameters = _clone(tensors.parameters, detached=True)
        selected = next(iter(parameters["continuous_synapses"].values()))
        selected["weights"] = (scale * selected["weights"]).requires_grad_()
        prepared = functional.prepare(parameters, tensors.constants)
        expected, _ = functional.rollout(parameters, prepared, tensors.state, steps=3)
        actual, _ = chunk(parameters, prepared, tensors.state)
        _close(actual, expected)
        actual_gradient = torch.autograd.grad(_loss(actual), selected["weights"])[0]
        expected_gradient = torch.autograd.grad(_loss(expected), selected["weights"])[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert actual_gradient.abs().sum() > 0
        voltages.append(actual["populations"]["cell"]["integrator"]["v"])
    assert not torch.equal(*voltages)


def test_eval_network_scan_accepts_inference_owned_carry_and_recordings():
    model = _mixed_network()
    model.eval()
    functional, tensors = dn.func.make_functional_network(model)
    callbacks = _callbacks(functional)
    expected, reference = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=3,
        callbacks=callbacks,
    )
    with torch.inference_mode():
        state = _clone(tensors.state)
    assert state["populations"]["cell"]["integrator"]["v"].is_inference()
    chunk = functional.compile_rollout_chunk(2, execution="scan", backend="aot_eager")
    actual, auxiliary = functional.run(
        tensors.parameters,
        tensors.constants,
        state,
        steps=3,
        chunklength=2,
        checkpoint_every=2,
        callbacks=callbacks,
        compiled_chunk=chunk,
    )
    _close(actual, expected)
    _close(auxiliary["callbacks"], reference["callbacks"])
    assert not actual["populations"]["cell"]["integrator"]["v"].requires_grad
