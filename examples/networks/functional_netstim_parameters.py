#!/usr/bin/env python3
"""Train raw NetStim sources through a continued functional Network.

Run from the repository root:
    python examples/networks/functional_netstim_parameters.py
"""

from __future__ import annotations

import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.125
STEPS = 4
DTYPE = torch.float64


# region transform
class ScaleRaw(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.02, dtype=DTYPE))

    def forward(self, original):
        return original * self.factor + self.offset


# endregion transform


def make_network():
    # region network
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        stimulus = dn.NetStim(
            N=1, interval=0.25, start=0.0, noise=0.6, max_spikes=100, seed=17
        )
        register_parametrization(stimulus.interval, "rho", ScaleRaw())
        cell = dn.SingleCompartment(
            N=1,
            C=1,
            cm=stimulus.interval,
            integrator=dn.bwd_euler_sc(imem=False),
            v_init=-65.0,
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        network = dn.Network(
            {"cell": cell},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            stimulus[:], cell[:], cell.mech.syn, weight=stimulus.interval, delay=DT
        )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            train_delay_backend="dense",
        )
    # endregion network
    return network


def run_demo():
    network, native = make_network(), make_network()
    # region compile
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("netstim.parameters.interval.*", "netstim.parameters.noise"),
        callbacks={"trace": dn.func.Recorder(("cell.v", "t"))},
        checkpoint_every=2,
        backend="aot_eager",
    )
    raw_path = "interval.parametrizations.rho.original"
    factor_path = "interval.parametrizations.rho.0.factor"
    original = runner.get_parameter(f"netstim.parameters.{raw_path}")
    factor = runner.get_parameter(f"netstim.parameters.{factor_path}")
    noise = runner.get_parameter("netstim.parameters.noise")
    assert original is runner.get_parameter(
        "populations.cell.cm_param.parametrizations.rho.original"
    )
    connection = next(iter(network.synapses))
    assert original is runner.get_parameter(
        f"synapses.{connection}.weight.weights.0.param.parametrizations.rho.original"
    )
    print("Selected parameters:", tuple(runner.trainable_parameters))
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
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    # endregion continuation
    native.initialize(DT)
    native.netstim.noise.requires_grad_()
    expected = [native.cell.v.clone()]
    for _ in range(2 * STEPS):
        native.step()
        expected.append(native.cell.v.clone())
    expected = torch.stack(expected)
    torch.testing.assert_close(voltage, expected)
    torch.testing.assert_close(
        final["netstim"]["next_stoch_time"], native.netstim.next_stoch_time
    )
    torch.testing.assert_close(
        final["netstim"]["rng_state"], native.netstim._rng.get_state()
    )
    # region gradients
    loss = voltage.square().mean() + final["netstim"]["next_stoch_time"].square().mean()
    gradients = torch.autograd.grad(loss, (original, factor, noise))
    native_loss = (
        expected.square().mean() + native.netstim.next_stoch_time.square().mean()
    )
    oracle = torch.autograd.grad(
        native_loss,
        (
            native.netstim.interval.parametrizations.rho.original,
            native.netstim.interval.parametrizations.rho[0].factor,
            native.netstim.noise,
        ),
    )
    for value, reference in zip(gradients, oracle, strict=True):
        torch.testing.assert_close(value, reference)
        assert torch.isfinite(value).all() and value.abs().sum() > 0
    # endregion gradients
    assert final["clock"]["step"].item() == 2 * STEPS
    assert network.t.item() == 0.0
    return voltage.detach(), tuple(value.detach() for value in gradients)


def main():
    voltage, gradients = run_demo()
    print("Native voltage, raw-source gradients, and clock/RNG continuation match.")
    print("Recorded samples:", voltage.shape[0])
    print("Gradient norms:", [value.norm().item() for value in gradients])


if __name__ == "__main__":
    main()
