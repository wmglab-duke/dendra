#!/usr/bin/env python3
"""Author a shared pure connection distribution and compare native execution."""

from __future__ import annotations

import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.models.distributions import Distribution
from dendra.models.mod import expsyn, graded_syn, pas

DT = 0.125
STEPS = 4
DTYPE = torch.float64


# region distribution
class UniformConnection(Distribution):
    """Uniform samples on [mean - half_width, mean + half_width]."""

    def __init__(self, mean, half_width):
        super().__init__()
        self.mean = mean
        self.half_width = torch.nn.Parameter(
            torch.tensor(half_width, dtype=mean.dtype, device=mean.device)
        )

    def sample_draw_schema(self, sample_shape=torch.Size()):
        return (
            torch.Size(sample_shape)
            + torch.broadcast_shapes(self.mean.shape, self.half_width.shape),
            self.mean.dtype,
            self.mean.device,
        )

    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def sample_draw_placeholder(self, sample_shape=torch.Size()):
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.full(shape, 0.5, dtype=dtype, device=device)

    def sample_from_draw(self, draw):
        return self.mean + (2.0 * draw - 1.0) * self.half_width

    def log_prob(self, value):
        inside = (value - self.mean).abs() <= self.half_width
        return torch.where(inside, -(2.0 * self.half_width).log(), -torch.inf)


class Positive(torch.nn.Module):
    def forward(self, value):
        return value.exp()

    def right_inverse(self, value):
        return value.log()


# endregion distribution


def release(voltage):
    return torch.sigmoid(voltage + 65.0)


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        # region sources
        mean = torch.nn.Parameter(torch.tensor(0.003, dtype=DTYPE))
        weight = UniformConnection(mean, half_width=0.0004)
        analog_weight = UniformConnection(mean, half_width=0.0005)
        delay = UniformConnection(
            torch.nn.Parameter(torch.tensor(1.5 * DT, dtype=DTYPE)),
            half_width=0.1 * DT,
        )
        for distribution in (weight, analog_weight, delay):
            register_parametrization(distribution, "half_width", Positive())
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
    network, _, weight, _ = make_network()
    native, native_mean, native_weight, native_delay = make_network()
    # region explicit-draw
    generator = torch.Generator(device="cpu").manual_seed(7)
    global_before = torch.get_rng_state().clone()
    draw = weight.sample_draw((3,), generator=generator)
    sample = weight.sample_from_draw(draw)
    torch.testing.assert_close(sample, weight.mean + (2 * draw - 1) * weight.half_width)
    assert torch.equal(torch.get_rng_state(), global_before)
    assert not draw.requires_grad and sample.requires_grad
    # endregion explicit-draw
    # region compile
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
    event = next(iter(network.synapses))
    analog = next(iter(network.continuous_synapses))
    mean = runner.get_parameter(f"synapses.{event}.weight.weights.0.param.mean")
    weight_width = runner.get_parameter(
        f"synapses.{event}.weight.weights.0.param.parametrizations.half_width.original"
    )
    delay_width = runner.get_parameter(
        f"synapses.{event}.delay_ms.weights.0.param.parametrizations.half_width.original"
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
    gradients = torch.autograd.grad(
        voltage.square().mean(), (mean, weight_width, delay_width)
    )
    reference = torch.autograd.grad(
        expected.square().mean(),
        (
            native_mean,
            native_weight.parametrizations.half_width.original,
            native_delay.parametrizations.half_width.original,
        ),
    )
    for actual, oracle in zip(gradients, reference, strict=True):
        torch.testing.assert_close(actual, oracle)
        assert actual.isfinite().all() and actual.abs().sum() > 0
    # endregion gradients
    return voltage.detach(), tuple(value.detach() for value in gradients)


if __name__ == "__main__":
    voltage, gradients = run_demo()
    print("Custom distribution voltage, raw-source gradients, and RNG match native.")
    print("Recorded samples:", voltage.shape[0])
    print("Gradient norms:", [value.norm().item() for value in gradients])
