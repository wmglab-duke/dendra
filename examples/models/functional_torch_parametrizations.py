"""Train a shared PyTorch parametrization through a compiled Network.

Run from the repository root: python examples/models/functional_torch_parametrizations.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.125
STEPS = 4


# region transform
class CapacitanceScale(torch.nn.Module):
    def __init__(self, log_scale):
        super().__init__()
        self.log_scale = log_scale
        self.register_buffer("offset", log_scale.new_tensor(0.05))

    def forward(self, original):
        return original * self.log_scale.exp() + self.offset


# endregion transform


# region network
def make_network():
    shared = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    populations = {}
    for name, initial_voltage in (("left", -55.0), ("right", -50.0)):
        population = dn.SingleCompartment(
            N=1,
            C=2,
            v_init=initial_voltage,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        population.insert(pas, g=0.001, e=-65.0)
        population.register_parametrization("cm", CapacitanceScale(shared))
        populations[name] = population
    network = dn.Network(populations)
    network.train()
    network.initialize(DT)
    return network


# endregion network


def objective(state, trace):
    return sum(
        population["integrator"]["v"].square().mean()
        for population in state["populations"].values()
    ) + 0.01 * sum(trace[f"{name}.v"].square().mean() for name in ("left", "right"))


def run_demo():
    source, native = make_network(), make_network()
    # region compile
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*parametrizations.cm.0.log_scale",),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        checkpoint_every=2,
        backend="aot_eager",
    )
    left = runner.get_parameter("*left*log_scale")
    right = runner.get_parameter("*right*log_scale")
    assert left is right
    assert len({id(value) for value in runner.trainable_parameters.values()}) == 1
    # endregion compile
    # region continue
    prefix, first = runner()
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    trace = {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in second["callbacks"]["trace"].items()
    }
    # endregion continue
    # region native
    native_trace = {
        name: [population.v.clone()] for name, population in native.populations.items()
    }
    times = [native.t.clone()]
    for _ in range(2 * STEPS):
        native.step()
        times.append(native.t.clone())
        for name, population in native.populations.items():
            native_trace[name].append(population.v.clone())
    expected_trace = {
        f"{name}.v": torch.stack(values) for name, values in native_trace.items()
    }
    expected_trace["t"] = torch.stack(times)
    for name, expected in expected_trace.items():
        torch.testing.assert_close(trace[name], expected)
    expected_state = {
        "populations": {
            name: {"integrator": {"v": population.v}}
            for name, population in native.populations.items()
        }
    }
    native_scale = native.populations["left"].parametrizations.cm[0].log_scale
    gradient = torch.autograd.grad(objective(final, trace), left)[0]
    expected_gradient = torch.autograd.grad(
        objective(expected_state, expected_trace), native_scale
    )[0]
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.isfinite(gradient) and gradient.abs() > 0
    assert source.t.item() == 0.0
    # endregion native
    return trace, gradient, tuple(runner.trainable_parameters)


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        trace, gradient, names = run_demo()
    print("Recorded samples:", trace["t"].shape[0])
    print("Shared parameter aliases:", names)
    print("Shared log-scale gradient:", gradient.item())
    print("Native trajectories and checkpointed compiled gradients agree.")


if __name__ == "__main__":
    main()
