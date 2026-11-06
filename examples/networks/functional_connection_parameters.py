#!/usr/bin/env python3
"""Train native connection sources through a continued functional Network.

Run from the repository root:
    python examples/networks/functional_connection_parameters.py
"""

from __future__ import annotations

import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.models.mod import expsyn, graded_syn, pas
from dendra.models.parametric import PositiveParam

DT = 0.125
STEPS = 4
DTYPE = torch.float64


# region transform
class ScaleRaw(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.0, dtype=DTYPE))

    def forward(self, original):
        return original * self.factor


def release(voltage):
    return torch.sigmoid(voltage + 65.0)


# endregion transform


def make_network():
    # region network
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        weight = PositiveParam(torch.tensor(0.003, dtype=DTYPE))
        register_parametrization(weight, "rho", ScaleRaw())
        delay = PositiveParam(torch.tensor(1.5 * DT, dtype=DTYPE))
        cell = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
        )
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("event"), tau=1.2, e=0.0)
        cell.insert(graded_syn.rename("analog"), e=-10.0)
        stimulus = dn.NetStim(
            N=1, interval=0.25, start=0.0, noise=0.0, max_spikes=100, seed=17
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
            weight=weight,
            delay=DT,
            transform=release,
            allow_autapses=True,
        )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=False,
            train_delay_backend="dense",
        )
    # endregion network
    return network, weight, delay


def run_demo():
    network, _, _ = make_network()
    native, native_weight, native_delay = make_network()
    # region compile
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
    event = next(iter(network.synapses))
    analog = next(iter(network.continuous_synapses))
    raw = "weight.weights.0.param.parametrizations.rho.original"
    factor = "weight.weights.0.param.parametrizations.rho.0.factor"
    delay = "delay_ms.weights.0.param.rho"
    original = runner.get_parameter(f"synapses.{event}.{raw}")
    scale = runner.get_parameter(f"synapses.{event}.{factor}")
    delay_raw = runner.get_parameter(f"synapses.{event}.{delay}")
    assert original is runner.get_parameter(f"continuous_synapses.{analog}.{raw}")
    assert scale is runner.get_parameter(f"continuous_synapses.{analog}.{factor}")
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
    expected = [native.cell.v.clone()]
    for _ in range(2 * STEPS):
        native.step()
        expected.append(native.cell.v.clone())
    expected = torch.stack(expected)
    torch.testing.assert_close(voltage, expected)
    # region gradients
    gradients = torch.autograd.grad(
        voltage.square().mean(), (original, scale, delay_raw)
    )
    oracle = torch.autograd.grad(
        expected.square().mean(),
        (
            native_weight.parametrizations.rho.original,
            native_weight.parametrizations.rho[0].factor,
            native_delay.rho,
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
    print("Native voltage and raw-source gradients match.")
    print("Recorded samples:", voltage.shape[0])
    print("Gradient norms:", [value.norm().item() for value in gradients])


if __name__ == "__main__":
    main()
