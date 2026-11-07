"""Compact source-history Networks preserve native training and continuation."""

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


def _configure(network, *, surrogate, taps):
    network.train()
    network.initialize(DT, max_delay_ms=8 * DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=taps is not None,
        diff_spiking=surrogate,
        taps=2 if taps is None else taps,
        sigma=0.43,
        tau=0.23,
        diff_scheduled_times=False,
    )
    assert network.netcon_train_backend == "auto"
    for connection in network.synapses.values():
        assert connection.train_delay_backend == "auto"
        assert connection._checkpoint_runtime_kind() == "source_history"
        assert connection.advance.__name__ == "advance_diff_source_history"
        assert connection.delivery_buffer.shape[0] == 1
    return network


def _recurrent_history(*, surrogate=True, taps=2):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(ThresholdRelay.rename("relay"))
        network = dn.Network({"cell": cell})
        for field, threshold, weight, delay in (
            ("level", 0.0, 0.2, 2.4 * DT if taps else DT),
            ("mirror", 1.5, 0.3, 3.2 * DT if taps else DT),
        ):
            network.connect_one_to_one(
                cell[:],
                cell[:],
                cell.mech.relay,
                pre_var=f"mech.relay.{field}",
                threshold=threshold,
                weight=weight,
                delay=delay,
                allow_autapses=True,
            )
        return _configure(network, surrogate=surrogate, taps=taps)


def _stim_history(
    *, surrogate=True, taps=2, noise=0.6, twins=False, long_interval=False
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(
            N=2, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(pas, g=0.001, e=-65.0)
        targets = ("syn", "other") if twins else ("syn",)
        for name in targets:
            cell.insert(expsyn.rename(name), e=0.0, tau=0.9)
        stimulus = dn.NetStim(
            N=2,
            interval=100.0 if long_interval else 0.5,
            start=0.0,
            noise=noise,
            max_spikes=30,
            seed=812,
        )
        network = dn.Network({"cell": cell}, netstim=stimulus)
        for name in targets:
            network.connect_one_to_one(
                stimulus[:],
                cell[:],
                cell.mech.mechanisms[name],
                weight=torch.tensor([0.002, 0.003], dtype=DTYPE),
                delay=torch.tensor([4.3 * DT, 5.2 * DT], dtype=DTYPE)
                if taps
                else torch.tensor([2 * DT, 3 * DT], dtype=DTYPE),
            )
        _configure(network, surrogate=surrogate, taps=taps)
        if twins:
            first, second = network.synapses.values()
            second.weight.weights[0].param = first.weight.weights[0].param
            second.weight.init()
        return network


def _sync_stim_state(source, *targets):
    # A fixed renewal seed does not fix the independent entropy seeder packet.
    for target in targets:
        target.netstim.restore_dict_from_checkpoint(
            _clone(source.netstim.state_dict_for_checkpoint(), detached=True)
        )


def _relay_loss(state):
    relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
    return relay["level"].square().sum() + 0.01 * relay["ledger"].square().sum()


def _native_relay_loss(model):
    return (
        model.cell.mech.relay.level.square().sum()
        + 0.01 * model.cell.mech.relay.ledger.square().sum()
    )


@pytest.mark.parametrize(
    "surrogate,taps,compiled",
    (
        (False, None, False),
        (True, None, False),
        (False, 2, False),
        (True, 2, True),
        (False, 3, True),
        (True, 3, False),
    ),
)
def test_auto_source_history_native_order_values_and_gradients(
    surrogate, taps, compiled
):
    model, native = (
        _recurrent_history(surrogate=surrogate, taps=taps) for _ in range(2)
    )
    functional, tensors = dn.func.make_functional_network(model)
    before = _clone(tensors, detached=True)
    state = _clone(tensors.state)
    initial = (
        state["populations"]["cell"]["mechanism_buffers"]["relay"]["level"]
        .detach()
        .clone()
        .requires_grad_()
    )
    state["populations"]["cell"]["mechanism_buffers"]["relay"]["level"] = initial
    native_initial = initial.detach().clone().requires_grad_()
    native.cell.mech.relay.level = native_initial
    advance = functional.prepare_and_step
    if compiled:
        advance = torch.compile(advance, backend="aot_eager", fullgraph=True)
    for index in range(11):
        incoming = state
        untouched = _clone(incoming, detached=True)
        state, _ = advance(tensors.parameters, tensors.constants, incoming)
        native.step()
        _close(state, functional.extract(native).state)
        _close(incoming, untouched)
        if index == 0:
            relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
            torch.testing.assert_close(relay["level"], initial.new_tensor([[1.5]]))
            # Each projection samples its source after its own zero-payload
            # receive, including the second threshold reached exactly at 1.5.
            assert all(
                carry["bitpack_source_has_spiked"].all()
                for carry in state["synapses"].values()
            )
    targets = (
        initial,
        *(
            value["weight.weights.0.param.rho"]
            for value in tensors.parameters["synapses"].values()
        ),
    )
    native_targets = (
        native_initial,
        *(value.weight.weights[0].param.rho for value in native.synapses.values()),
    )
    if taps:
        targets += tuple(
            value["delay_ms.weights.0.param.rho"]
            for value in tensors.parameters["synapses"].values()
        )
        native_targets += tuple(
            value.delay_ms.weights[0].param.rho for value in native.synapses.values()
        )
    observed = torch.autograd.grad(_relay_loss(state), targets)
    expected = torch.autograd.grad(_native_relay_loss(native), native_targets)
    _close(observed, expected)
    assert all(torch.isfinite(value).all() for value in observed)
    assert all(value.abs().sum() > 0 for value in observed[:3])
    _close(tensors, before)
    _close(functional.extract().state, before.state)


@pytest.mark.parametrize("taps", (2, 3))
def test_delivery_uses_weight_and_delay_values_supplied_after_source_emission(taps):
    model, native = (
        _stim_history(taps=taps, noise=0.0, long_interval=True) for _ in range(2)
    )
    _sync_stim_state(model, native)
    functional, tensors = dn.func.make_functional_network(model)
    name = next(iter(model.synapses))
    native_connection = native.synapses[name]
    first_draw = functional.sample_step_draw(tensors.state)
    prefix, _ = functional.prepare_and_step(*tensors, netstim_draw=first_draw)
    native.step()
    _close(prefix, functional.extract(native).state)
    assert prefix["netstim"]["spike_counts"].eq(1).all()
    original = tensors.parameters["synapses"][name]
    weights = (
        native_connection.weight.weights[0]
        .param._inverse_transform(native_connection.weight.w.detach() * 2.5)
        .requires_grad_()
    )
    delays = (
        native_connection.delay_ms.weights[0]
        .param._inverse_transform(
            original["delay_ms.weights.0.param.rho"].new_tensor([1.6 * DT, 2.3 * DT])
        )
        .requires_grad_()
    )
    parameters = {
        **tensors.parameters,
        "synapses": {
            name: {
                **original,
                "weight.weights.0.param.rho": weights,
                "delay_ms.weights.0.param.rho": delays,
            }
        },
    }
    native_weights = torch.nn.Parameter(weights.detach().clone())
    native_delays = torch.nn.Parameter(delays.detach().clone())
    native_connection.weight.weights[0].param.rho = native_weights
    native_connection.weight.init()
    native_connection.delay_ms.weights[0].param.rho = native_delays
    native_connection.delay_ms.init()
    state = prefix
    for _ in range(8):
        draw = functional.sample_step_draw(state)
        state, _ = functional.prepare_and_step(
            parameters, tensors.constants, state, netstim_draw=draw
        )
        native.step()
        _close(state, functional.extract(native).state)
    loss = state["populations"]["cell"]["mechanisms"]["syn"]["g"].sum()
    expected_loss = native.cell.mech.syn.g.sum()
    actual = torch.autograd.grad(
        loss,
        (weights, delays, original["weight.weights.0.param.rho"]),
        allow_unused=True,
    )
    expected = torch.autograd.grad(expected_loss, (native_weights, native_delays))
    _close(actual[:2], expected)
    assert all(value.abs().sum() > 0 for value in actual[:2])
    assert actual[2] is None or actual[2].eq(0).all()


@pytest.mark.parametrize("surrogate", (False, True))
def test_noisy_history_network_matches_native_rng_drive_and_parameter_gradients(
    surrogate,
):
    model, native = (_stim_history(surrogate=surrogate, taps=2) for _ in range(2))
    _sync_stim_state(model, native)
    model.netstim.noise.requires_grad_()
    native.netstim.noise.requires_grad_()
    functional, tensors = dn.func.make_functional_network(model)
    steps = 12
    drive = (
        torch.linspace(-0.1, 0.4, steps * model.cell.v.numel(), dtype=DTYPE)
        .reshape(steps, *model.cell.shape)
        .mul(1e-5)
        .requires_grad_()
    )
    native_drive = drive.detach().clone().requires_grad_()
    inputs = dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=drive)},
        netstim=functional.sample_draw_tape(tensors.state, steps),
    )
    actual, _ = functional.run(*tensors, inputs, steps=steps)
    for index in range(steps):
        _native_driven_step(
            native, {"cell": dn.func.StepInput(intra=native_drive[index])}
        )
    _close(actual, functional.extract(native).state)
    connection = next(iter(tensors.parameters["synapses"].values()))
    native_connection = next(iter(native.synapses.values()))
    loss = (
        actual["populations"]["cell"]["integrator"]["v"].square().sum()
        + actual["netstim"]["next_stoch_time"].square().sum()
    )
    native_loss = (
        native.cell.v.square().sum() + native.netstim.next_stoch_time.square().sum()
    )
    gradients = torch.autograd.grad(
        loss,
        (
            connection["weight.weights.0.param.rho"],
            connection["delay_ms.weights.0.param.rho"],
            tensors.parameters["netstim"]["parameters"]["noise"],
            drive,
        ),
    )
    expected = torch.autograd.grad(
        native_loss,
        (
            native_connection.weight.weights[0].param.rho,
            native_connection.delay_ms.weights[0].param.rho,
            native.netstim.noise,
            native_drive,
        ),
    )
    _close(gradients, expected)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )


@pytest.mark.parametrize("surrogate", (False, True))
@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_compiled_history_checkpoint_continuation_keeps_shared_parameters_inputs_and_rng(
    surrogate, initialization
):
    model = _stim_history(surrogate=surrogate, twins=True)
    if initialization == "snapshot":
        for _ in range(3):
            model.step()
    names = tuple(model.synapses)
    runner = dn.compile(
        model,
        dt=DT,
        steps=5,
        inference=False,
        initialization=initialization,
        checkpoint_every=3,
        backend="aot_eager",
        trainable=(
            f"synapses.{names[1]}.weight.weights.0.param.rho",
            "netstim.parameters.noise",
        ),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
    )
    shared = runner.get_parameter(f"synapses.{names[0]}.weight.weights.0.param.rho")
    assert shared is runner.get_parameter(
        f"synapses.{names[1]}.weight.weights.0.param.rho"
    )
    assert isinstance(shared, torch.nn.Parameter)
    assert all(
        plan.runtime == "source_history" for plan in runner.functional.synapses.values()
    )
    before = _clone(runner.functional.extract().state, detached=True)
    voltage = (
        runner.initialization_inputs.populations["cell"]
        .v_init.detach()
        .clone()
        .requires_grad_()
    )
    initialization_inputs = runner.initialization_inputs._replace(
        populations={
            "cell": runner.initialization_inputs.populations["cell"]._replace(
                v_init=voltage
            )
        }
    )
    if initialization == "fresh":
        initial = runner.functional.initialize(
            runner.parameters,
            runner.constants,
            runner.initial_state,
            initialization_inputs,
            netstim_draw=runner.functional.sample_initialization_draw(
                runner.parameters, runner.initial_state
            ),
        )
    else:
        runner.initial_state["populations"]["cell"]["integrator"]["v"] = voltage
        initial = dn.func.NetworkTensors(
            runner.parameters, runner.constants, runner.initial_state
        )
    drive = (
        torch.linspace(-0.2, 0.5, 10 * model.cell.v.numel(), dtype=DTYPE)
        .reshape(10, *model.cell.shape)
        .mul(1e-5)
        .requires_grad_()
    )
    inputs = dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=drive)},
        netstim=runner.functional.sample_draw_tape(initial.state, 10),
    )
    expected, reference = runner.functional.run(
        *initial,
        inputs,
        steps=10,
        callbacks=runner.functional.make_callbacks(
            {"trace": dn.func.Recorder(("cell.v", "t"))}
        ),
    )
    prefix, prefix_aux = runner(
        inputs=_slice(inputs, 0, 5),
        initialization=initialization_inputs if initialization == "fresh" else None,
    )
    actual, auxiliary = runner(
        prefix,
        parameters=prefix_aux.get("parameters", runner.parameters),
        inputs=_slice(inputs, 5, 10),
        callback_state=prefix_aux["callbacks"].state,
    )
    trace = {
        name: torch.cat((prefix_aux["callbacks"]["trace"][name], values))
        for name, values in auxiliary["callbacks"]["trace"].items()
    }
    _close(actual, expected)
    _close(trace, reference["callbacks"]["trace"])
    assert trace["cell.v"].shape[0] == 11
    noise = runner.get_parameter("netstim.parameters.noise")
    targets = (voltage, shared, noise, drive)

    def loss(state, recordings):
        return (
            recordings["cell.v"].square().sum()
            + state["netstim"]["next_stoch_time"].square().sum()
        )

    gradients = torch.autograd.grad(loss(actual, trace), targets)
    oracle = torch.autograd.grad(
        loss(expected, reference["callbacks"]["trace"]), targets
    )
    _close(gradients, oracle)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    _close(runner.functional.extract().state, before)
    assert runner.execution_report().strategy == "compiled_scan"


@pytest.mark.parametrize("surrogate", (False, True))
def test_history_checkpoint_commits_to_native_and_continues_with_exact_rng(surrogate):
    model, native, resumed = (_stim_history(surrogate=surrogate) for _ in range(3))
    _sync_stim_state(model, native, resumed)
    functional, tensors = dn.func.make_functional_network(model)
    prefix, _ = functional.run(*tensors, steps=5, checkpoint_every=3)
    for _ in range(5):
        native.step()
    _close(prefix, functional.extract(native).state)
    functional.commit_state_(resumed, prefix)
    _close(functional.extract(resumed).state, prefix)
    final, _ = functional.run(tensors.parameters, tensors.constants, prefix, steps=7)
    for _ in range(7):
        native.step()
        resumed.step()
    _close(final, functional.extract(native).state)
    _close(final, functional.extract(resumed).state)
    assert final["clock"]["step"].item() == 12


def test_scheduled_connection_keeps_auto_dense_fallback_and_native_execution():
    model, native = (_stim_history() for _ in range(2))
    _sync_stim_state(model, native)
    for network in (model, native):
        connection = next(iter(network.synapses.values()))
        connection.schedule(con_indices=[0, 1], times_ms=[0.0, 2 * DT], weight=0.7)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=True,
            taps=2,
            sigma=0.43,
            tau=0.23,
            diff_scheduled_times=False,
        )
        assert connection._checkpoint_runtime_kind() == "dense"
        assert connection.advance.__name__ == "advance_diff"
    functional, tensors = dn.func.make_functional_network(model)
    actual, _ = functional.run(*tensors, steps=8)
    for _ in range(8):
        native.step()
    _close(actual, functional.extract(native).state)
