"""Connected population gradients and saved-state replay with physical delays."""

import runpy
from pathlib import Path

import pytest
import torch

import dendra as dn
from dendra._bootstrap import reset_torch_compiler
from dendra.models.networks import netcon

DEMO = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "examples"
        / "networks"
        / "functional_delay_interpolation.py"
    )
)


@pytest.fixture(autouse=True)
def _compiler_cache():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("synapse,batch", (("expsyn", None), ("exp2syn", 2)))
def test_interpolated_connections_match_native_voltages_gradients_and_replay(
    taps, synapse, batch
):
    DEMO["run_demo"](taps=taps, synapse=synapse, batch=batch)


@pytest.mark.parametrize("taps", (2, 3))
def test_fullgraph_interpolated_network_composition_keeps_checkpoint_gradients(taps):
    DEMO["run_demo"](taps=taps, synapse="exp2syn", batch=2, compiled=True)


@pytest.mark.parametrize("taps", (2, 3))
def test_native_rollout_delay_gradients_match_individual_steps(taps, monkeypatch):
    dt, steps = DEMO["DT"], DEMO["STEPS"]
    rollout = DEMO["make_network"](taps=taps)
    single = DEMO["make_network"](taps=taps)
    for network in (rollout, single):
        network.pre.mech.source.gate = network.pre.v.new_tensor([[0.25], [0.5]])

    interpolate = netcon.delay_interpolation
    calls = []

    def count(*args, **kwargs):
        calls.append(None)
        return interpolate(*args, **kwargs)

    monkeypatch.setattr(netcon, "delay_interpolation", count)
    rollout.run(steps=steps)
    assert len(calls) == 1
    for _ in range(steps):
        single.step()

    def loss_and_grads(network):
        (connection,) = network.synapses.values()
        loss = network.post.v.square().sum() + connection.delivery_buffer.square().sum()
        gradients = torch.autograd.grad(
            loss, (connection.delay_ms.w, connection.weight.w)
        )
        return loss, gradients

    actual, actual_grads = loss_and_grads(rollout)
    expected, expected_grads = loss_and_grads(single)
    torch.testing.assert_close(rollout.post.v, single.post.v, rtol=0, atol=0)
    torch.testing.assert_close(rollout.t, rollout.t.new_tensor(steps * dt))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.count_nonzero(actual_grads[0])


def test_native_rollout_recovers_taps_after_in_place_delay_edit():
    rollout = DEMO["make_network"]()
    single = DEMO["make_network"]()
    for network in (rollout, single):
        network.pre.mech.source.gate = network.pre.v.new_tensor([[0.25], [0.5]])

    def edit_delays(network):
        (connection,) = network.synapses.values()
        connection.delay_ms.w.add_(DEMO["DT"])

    class EditDelay(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.steps = 0

        def post_step_hook(self, network):
            self.steps += 1
            if self.steps == 3:
                edit_delays(network)

    with torch.no_grad():
        rollout.run(steps=8, callbacks=[EditDelay()])
        for step in range(8):
            single.step()
            if step == 2:
                edit_delays(single)

    torch.testing.assert_close(rollout.post.v, single.post.v, rtol=0, atol=0)
    (actual,) = rollout.synapses.values()
    (expected,) = single.synapses.values()
    torch.testing.assert_close(
        actual.delivery_buffer, expected.delivery_buffer, rtol=0, atol=0
    )


def test_functional_rollout_prepares_event_taps_once_across_chunks(monkeypatch):
    network = DEMO["make_network"]()
    network.pre.mech.source.gate = network.pre.v.new_tensor([[0.25], [0.5]])
    plan, tensors = dn.func.make_functional_network(network)
    (projection_type,) = {type(projection) for projection in plan.synapses.values()}
    prepare = projection_type._delay_taps
    calls = []

    def count(projection, delays):
        calls.append(None)
        return prepare(projection, delays)

    monkeypatch.setattr(projection_type, "_delay_taps", count)
    state, _ = plan.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=DEMO["STEPS"],
        chunklength=4,
    )
    assert len(calls) == 1
    (projection,) = tensors.parameters["synapses"].values()
    raw_delays = projection["delay_ms.weights.0.param.rho"]
    loss = state["populations"]["post"]["integrator"]["v"].square().sum()
    (gradient,) = torch.autograd.grad(loss, (raw_delays,))
    assert torch.count_nonzero(gradient)
