#!/usr/bin/env python3
"""Sample and train connection sources through fresh Network initialization."""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.distributions import Normal, TruncatedNormal
from dendra.models.mod import expsyn, graded_syn, pas

DT = 0.125
STEPS = 4
DTYPE = torch.float64


def release(voltage):
    return torch.sigmoid(voltage + 65.0)


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        # region sources
        mean = torch.nn.Parameter(torch.tensor(0.003, dtype=DTYPE))
        weight = Normal(mean=mean, std=torch.tensor(0.0004, dtype=DTYPE))
        analog_weight = TruncatedNormal(
            mean=mean, std=torch.tensor(0.0005, dtype=DTYPE), low=0.0, high=0.02
        )
        delay = TruncatedNormal(
            mean=torch.tensor(1.5 * DT, dtype=DTYPE),
            std=torch.tensor(0.1 * DT, dtype=DTYPE),
            low=DT,
            high=2 * DT,
        )
        # endregion sources
        cell = dn.SingleCompartment(
            N=2, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("event"), tau=1.2, e=0.0)
        cell.insert(graded_syn.rename("analog"), e=-10.0)
        stimulus = dn.NetStim(
            N=2, interval=0.25, start=0.0, noise=0.0, max_spikes=100, seed=17
        )
        network = dn.Network(
            {"cell": cell},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            stimulus[:], cell[:], cell.mech.event, weight=weight, delay=delay
        )
        network.connect_continuous_one_to_one(
            cell[:],
            cell[:],
            cell.mech.analog,
            input="g_pre",
            weight=analog_weight,
            delay=DT,
            transform=release,
            allow_autapses=True,
        )
        network.train()
        network.initialize(DT, max_delay_ms=4 * DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=False,
            train_delay_backend="dense",
        )
    return network, mean, weight, delay


def run_demo():
    network, _, _, _ = make_network()
    native, native_mean, native_weight, native_delay = make_network()
    # region compile
    # Lowering captures the current Torch RNG for the explicit sampling carry.
    torch.manual_seed(83)
    global_rng = torch.get_rng_state().clone()
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("synapses.*.weight.weights.*", "synapses.*.delay_ms.weights.*"),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        checkpoint_every=2,
        backend="aot_eager",
    )
    event, analog = (
        next(iter(network.synapses)),
        next(iter(network.continuous_synapses)),
    )
    mean = runner.get_parameter(f"synapses.{event}.weight.weights.0.param.mean")
    std = runner.get_parameter(f"synapses.{event}.weight.weights.0.param.std")
    delay_scale = runner.get_parameter(
        f"synapses.{event}.delay_ms.weights.0.param._log_std"
    )
    assert mean is runner.get_parameter(
        f"continuous_synapses.{analog}.weight.weights.0.param.mean"
    )
    # endregion compile
    # region continuation
    prefix, first = runner()
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    voltage = torch.cat(
        (
            first["callbacks"]["trace"]["cell.v"],
            second["callbacks"]["trace"]["cell.v"],
        )
    )
    assert torch.equal(torch.get_rng_state(), global_rng)
    # endregion continuation
    native.initialize(DT, max_delay_ms=4 * DT)
    expected = [native.cell.v.clone()]
    for _ in range(2 * STEPS):
        native.step()
        expected.append(native.cell.v.clone())
    expected = torch.stack(expected)
    torch.testing.assert_close(voltage, expected)
    assert torch.equal(
        final["connection_sampling"]["rng"]["cpu"], torch.get_rng_state()
    )
    # region gradients
    gradients = torch.autograd.grad(voltage.square().mean(), (mean, std, delay_scale))
    reference = torch.autograd.grad(
        expected.square().mean(),
        (native_mean, native_weight.std, native_delay._log_std),
    )
    for actual, oracle in zip(gradients, reference, strict=True):
        torch.testing.assert_close(actual, oracle)
        assert actual.isfinite().all() and actual.abs().sum() > 0
    # endregion gradients
    return voltage.detach(), tuple(value.detach() for value in gradients)


if __name__ == "__main__":
    voltage, gradients = run_demo()
    print("Native voltage, raw-source gradients, and connection RNG match.")
    print("Recorded samples:", voltage.shape[0])
    print("Gradient norms:", [value.norm().item() for value in gradients])
