"""Differentiate mechanism-owned waveforms and continue a compiled Network.

Run from the repository root: python examples/models/functional_mechanism_injections.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, VoltageProcess
from dendra.models.mod import graded_syn

DT = 0.125
STEPS = 4


# region current-owner
class DrivenLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.ASSIGNED("i_inj")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def assigned_values(self, v, values):
        del values
        return {"i_inj": self.injection_values(v, t=self.t)}

    def i_affine_terms(self, v):
        del v
        # i = g*v - (g*e + gain*i_inj): positive drive is inward current.
        return self.g, self.g * self.e + self.gain * self.i_inj


# endregion current-owner


# region voltage-owner
class DrivenVoltage(VoltageProcess):
    VoltageProcess.GLOBAL(gain=0.7)
    VoltageProcess.CARRY("owned")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def initial_values(self, v, values):
        del values
        return {"owned": v}

    def voltage_values(self, v, dt, values):
        drive = self.injection_values(v, t=self.t)
        voltage = values["owned"] + dt * self.gain * drive
        return voltage, {"owned": voltage}


# endregion voltage-owner


def make_population(mechanism=DrivenLeak, *, waveform=None):
    population = dn.SingleCompartment(
        N=2,
        C=2,
        v_init=-47.0,
        integrator=(
            dn.scnv(imem=False)
            if mechanism is DrivenVoltage
            else dn.bwd_euler_sc(imem=False)
        ),
    )
    population.insert(mechanism.rename("drive"))
    if waveform is None:
        waveform = dn.sin(amp=0.7, freq=0.23, phase=0.4, delay=-0.5)
    # region attach
    population[:, 0].inject(waveform)
    population.train()
    population.initialize()
    assert population.mechanism_injection_accepted == [True]
    assert population.build_intra() is None
    # endregion attach
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def run_population_demo(mechanism=DrivenLeak):
    source, native = make_population(mechanism), make_population(mechanism)
    # region differentiate
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*gain_param", "*injected_waveforms.*.amp"),
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="aot_eager",
    )
    final, auxiliary = runner()
    amplitude = runner.get_parameter("*injected_waveforms.*.amp")
    gradient = torch.autograd.grad(final["integrator"]["v"].square().mean(), amplitude)[
        0
    ]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # endregion differentiate
    with torch.no_grad():
        for _ in range(STEPS):
            native.step(DT)
    torch.testing.assert_close(final["integrator"]["v"], native.v)
    assert source.t.item() == 0.0
    return auxiliary["callbacks"]["trace"]["v"].detach()


def run_network_demo():
    # region shared-network
    waveform = dn.sin(amp=0.7, freq=0.23, phase=0.4, delay=-0.5)
    left = make_population(waveform=waveform)
    right = make_population(waveform=waveform)
    right.insert(graded_syn.rename("graded"), e=-10.0)
    network = dn.Network({"left": left, "right": right})
    network.connect_continuous_one_to_one(
        left[:],
        right[:],
        right.mech.graded,
        pre_var="v",
        input="g_pre",
        weight=0.001,
        delay=0.0,
        transform=torch.sigmoid,
    )
    network.eval()
    network.initialize(DT)
    # endregion shared-network
    # region continue-network
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=True,
        execution="while",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="aot_eager",
    )
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    assert final["clock"]["t"].item() == 2 * STEPS * DT
    # endregion continue-network
    return torch.cat(
        (first["callbacks"]["trace"]["left.v"], second["callbacks"]["trace"]["left.v"])
    )


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        current_trace = run_population_demo()
        voltage_trace = run_population_demo(DrivenVoltage)
        network_trace = run_network_demo()
    print("Current-owning Population trace:", tuple(current_trace.shape))
    print("Voltage-owning Population trace:", tuple(voltage_trace.shape))
    print("Continued Network trace:", tuple(network_trace.shape))


if __name__ == "__main__":
    main()
