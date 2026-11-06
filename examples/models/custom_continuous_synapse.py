#!/usr/bin/env python3
"""Author a graded density synapse and deliver a dynamic analog release gate.

Run from the repository root::

    python examples/models/custom_continuous_synapse.py

``ContinuousSynapse.INPUT`` declares a persistent analog delivery buffer that
Network resets and fills every timestep. ``State.INPUT`` instead declares an
ephemeral, read-only entry in a nested State's ``values`` frame; use
``Population.bind_input`` to supply such an entry from a declared field in the
same Population. Declaring a State input does not create a network connection.

This example checks ordinary Network execution. The same continuous delivery
and receive contracts support functional lowering through ``dn.compile(network)``.
``functional_continuous_connectivity.py`` demonstrates explicit projection and
receiver composition with gradients and checkpoint continuation.
"""

from __future__ import annotations

# region imports
import torch

import dendra as dn
from dendra.models.callbacks import RecorderLambda
from dendra.models.mechanisms import ContinuousSynapse, Mechanism, State
from dendra.models.mod import pas

# endregion imports


# region declarations
class ReleaseGate(State):
    """A dimensionless presynaptic gate relaxing toward a voltage-dependent level."""

    State.STATE("release")
    State.ASSIGNED("release_inf")
    State.RANGE(tau=2.0, midpoint=-45.0, slope=5.0, initial_release=0.25)
    State.DERIVATIVE("release' = (release_inf - release) / tau")

    def state_defaults(self, v, values):
        return {"release": torch.zeros_like(v) + self.initial_release}

    def assigned_values(self, v, values):
        return {"release_inf": torch.sigmoid((v - self.midpoint) / self.slope)}


class AnalogSource(Mechanism):
    Mechanism.STATE_BUNDLE(ReleaseGate)


class GradedSynapse(ContinuousSynapse):
    # g_pre is conductance density in S/cm²; its previous value is g_pre_old.
    ContinuousSynapse.INPUT("g_pre")
    ContinuousSynapse.RANGE(e=-10.0)
    ContinuousSynapse.NONSPECIFIC_CURRENT("i")
    ContinuousSynapse.AFFINE("i")

    def i(self, v):
        # S/cm² * mV = mA/cm², with outward-positive current.
        return self.g_pre * (v - self.e)


# endregion declarations


# region network
def make_network():
    """Connect two presynaptic cells to two postsynaptic cells one-to-one."""
    pre = dn.SingleCompartment(N=2, C=1, v_init=-35.0, dtype=torch.float64)
    pre.insert(pas, g=0.001, e=-35.0)  # Holds the release target level constant.
    pre.insert(AnalogSource.rename("source"))
    post = dn.SingleCompartment(N=2, C=1, v_init=-65.0, dtype=torch.float64)
    post.insert(pas, g=0.001, e=-65.0)
    post.insert(GradedSynapse.rename("graded"))

    network = dn.Network({"pre": pre, "post": post})
    g_max = torch.tensor([0.002, 0.004], dtype=torch.float64)  # S/cm² per edge.
    network.connect_continuous_one_to_one(
        pre[:],
        post[:],
        post.mech.graded,
        pre_var="mech.source.release",  # Live declared State, selected each step.
        input="g_pre",
        weight=g_max,
        delay=0.0,
        reduce="sum",
    )
    return network, g_max


# endregion network


# region simulation
def run_demo():
    """Run four timesteps and check delivered conductances and density currents."""
    dt, steps = 0.1, 4  # Time is in ms.
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        network, g_max = make_network()
        network.eval()
        network.initialize(dt)
        recorder = RecorderLambda(
            {
                "t": lambda net: net.t.clone(),
                "release": lambda net: net.pre.mech.source.release.clone(),
                "g_pre": lambda net: net.post.mech.graded.g_pre.clone(),
                "g_pre_old": lambda net: net.post.mech.graded.g_pre_old.clone(),
                "v_post": lambda net: net.post.v.clone(),
                "i_density": lambda net: net.post.mech.graded.i(net.post.v).clone(),
            }
        )
        network.run(tstop=steps * dt, callbacks=[recorder])
        traces = {name: recorder.stack(name) for name in recorder.funcs}

        # Recorder includes t=0. Zero-delay delivery samples release BEFORE
        # either population advances, so sample k drives the next recorded frame.
        expected_input = traces["release"][:-1] * g_max.reshape(1, 2, 1)
        torch.testing.assert_close(traces["g_pre"][1:], expected_input)
        torch.testing.assert_close(traces["g_pre_old"][1:], traces["g_pre"][:-1])
        torch.testing.assert_close(
            traces["i_density"],
            traces["g_pre"] * (traces["v_post"] - network.post.mech.graded.e),
        )
        assert torch.all(traces["i_density"][1:] < 0)  # Inward at v < e.

        # Constant presynaptic voltage makes the gate's analytic trajectory known.
        steady = torch.sigmoid(traces["t"].new_tensor(2.0))
        expected_release = steady + (0.25 - steady) * torch.exp(-traces["t"] / 2.0)
        torch.testing.assert_close(
            traces["release"],
            expected_release[:, None, None].expand_as(traces["release"]),
        )
        assert traces["g_pre"].shape[0] == steps + 1
        return traces


# endregion simulation


def main():
    traces = run_demo()
    print("Final conductance density [S/cm²]:", traces["g_pre"][-1].flatten().tolist())
    print(
        "Final outward current density [mA/cm²]:",
        traces["i_density"][-1].flatten().tolist(),
    )
    print("Verified zero-delay analog sampling, input history, and release dynamics.")


if __name__ == "__main__":
    main()
