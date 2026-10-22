"""Train a custom MaterialProcess and continue a compiled Network.

Run from the repository root: python examples/functional_material_processes.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import MaterialProcess
from dendra.models.mod import pas

DT = 0.125
STEPS = 4


# region process
class Transfer(MaterialProcess):
    MaterialProcess.GLOBAL(rate=0.4)
    MaterialProcess.PHASE("post_local")

    @property
    def material_fields(self):
        return (("pool", "amount"), ("reservoir", "amount"))

    def prepare_materials(self, fields, dt):
        pool = fields[("pool", "amount")]
        rate = self._resolve_quantity("rate", pool)
        fraction = -torch.expm1(-dt * rate)
        return (fraction,)

    def material_updates(self, fields, prepared):
        (fraction,) = prepared
        pool = fields[("pool", "amount")]
        reservoir = fields[("reservoir", "amount")]
        transferred = fraction * pool
        updates = {
            ("pool", "amount"): pool - transferred,
            ("reservoir", "amount"): reservoir + transferred,
        }
        return self._mask_material_updates(fields, updates)


# endregion process


# region population
def make_population(rate):
    population = dn.SingleCompartment(N=1, C=3, v_init=-60.0)
    population.material("pool", fields={"amount": 2.0}, min_values={"amount": 0.0})
    population.material("reservoir", fields={"amount": 0.5}, min_values={"amount": 0.0})
    population.insert(pas)
    population.insert(Transfer, rate=rate)
    population.train()
    population.initialize()
    return population


def make_network():
    network = dn.Network({"left": make_population(0.4), "right": make_population(0.7)})
    network.train()
    network.initialize(DT)
    return network


# endregion population


def run_demo():
    source, native = make_network(), make_network()
    # region compile
    names = (
        "left.pool_material.amount",
        "left.reservoir_material.amount",
        "right.pool_material.amount",
        "right.reservoir_material.amount",
        "t",
    )
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*rate_param",),
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(names)},
        backend="aot_eager",
    )
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
        name: torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        for name in names
    }
    # endregion continue
    # region gradients
    left_rate = runner.get_parameter("*left*rate_param")
    right_rate = runner.get_parameter("*right*rate_param")
    loss = sum(
        final["populations"][name]["materials"]["reservoir"]["amount"].mean()
        for name in ("left", "right")
    )
    gradients = torch.autograd.grad(loss, (left_rate, right_rate))
    elapsed = 2 * STEPS * DT
    for gradient, rate in zip(gradients, (left_rate, right_rate), strict=True):
        expected = 2.0 * elapsed * torch.exp(-elapsed * rate)
        torch.testing.assert_close(gradient, expected)
    # endregion gradients

    def read(name):
        if name == "t":
            return native.t.clone()
        population, material, field = name.split(".")
        material = material.removesuffix("_material")
        return getattr(
            native.populations[population].mech.materials[material], field
        ).clone()

    native.initialize(DT)
    expected = {name: [read(name)] for name in names}
    for _ in range(2 * STEPS):
        native.step()
        for name in names:
            expected[name].append(read(name))
    for name in names:
        torch.testing.assert_close(trace[name], torch.stack(expected[name]))
    for name in ("left", "right"):
        total = (
            trace[f"{name}.pool_material.amount"]
            + trace[f"{name}.reservoir_material.amount"]
        )
        torch.testing.assert_close(total, torch.full_like(total, 2.5))
        torch.testing.assert_close(
            trace[f"{name}.pool_material.amount"][0],
            torch.full_like(trace[f"{name}.pool_material.amount"][0], 2.0),
        )
    assert source.t.item() == 0.0
    return trace, gradients


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        trace, gradients = run_demo()
    print("Recorded samples:", trace["t"].shape[0])
    print("Transfer rate gradients:", [value.item() for value in gradients])
    print("Native trajectories and total stored amount agree.")


if __name__ == "__main__":
    main()
