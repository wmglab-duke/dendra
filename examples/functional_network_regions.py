"""Record named physical regions from ordinary and concatenated Networks.

Run from the repository root: python examples/functional_network_regions.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.1
STEPS = 4


def make_network(*, concatenated=False):
    # region declarations
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64, DEVICE="cpu"):
        left = dn.SingleCompartment(N=2, C=3, integrator=dn.bwd_euler_sc(imem=True))
        right = dn.SingleCompartment(N=2, C=2, integrator=dn.bwd_euler_sc(imem=True))
        left[:, 0].label("soma")
        left[1, :].label("cell_1")
        right[:, 0].label("soma")
        left.insert(pas, g=0.001, e=-55.0)
        right.insert(pas, g=0.001, e=-65.0)
        right.insert(expsyn.rename("event"), tau=2.0, e=0.0)
        names = ("exc", "inh") if concatenated else ("left", "right")
        network = dn.Network(dict(zip(names, (left, right), strict=True)))
        network.connect_one_to_one(
            left.soma,
            right.soma,
            right.mech.event,
            threshold=-64.0,
            weight=0.002,
            delay=DT,
        )
        if concatenated:
            network = network.concat()
        network.batch(2)
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            train_delay_backend="dense",
        )
    # endregion declarations
    return network


def run_demo(*, execution="scan", concatenated=False, backend="aot_eager"):
    network = make_network(concatenated=concatenated)
    # region regions
    if concatenated:
        population = network.all_populations
        owner = "all_populations"
        component = population.exc
        soma = population.exc.soma
    else:
        population = network.left
        owner = "left"
        component = population[:]
        soma = population.soma
    # All regions come from this exact Network, after concatenation/batching.
    repeated = population[[1, 0, 1], :, [5, 1, 5] if concatenated else [2, 0, 2]]
    empty = population[..., []]
    # endregion regions
    voltage = f"{owner}.v"
    current = f"{owner}.i_membrane"
    # region compile
    callbacks = {
        "full": dn.func.Recorder((voltage, current, "t")),
        "component": dn.func.Recorder((voltage, "t"), region=component),
        "soma": dn.func.Recorder((voltage, current, "t"), region=soma),
        "repeated": dn.func.Recorder((voltage,), region=repeated),
        "empty": dn.func.Recorder((voltage, "t"), region=empty),
    }
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=execution == "while",
        callbacks=callbacks,
        backend=backend,
    )
    # endregion compile
    initial = runner.initial_state
    initial_voltage = initial["populations"][owner]["integrator"]["v"].detach().clone()
    initial_voltage.requires_grad_(execution == "scan")
    initial = {
        **initial,
        "populations": {
            **initial["populations"],
            owner: {
                **initial["populations"][owner],
                "integrator": {
                    **initial["populations"][owner]["integrator"],
                    "v": initial_voltage,
                },
            },
        },
    }
    # region continuation
    prefix, first = runner(initial)
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    traces = {
        name: {
            field: torch.cat((first["callbacks"][name][field], values))
            for field, values in second["callbacks"][name].items()
        }
        for name in callbacks
    }
    assert traces["soma"][voltage].shape[0] == 2 * STEPS + 1
    # endregion continuation
    for name, region in (
        ("component", component),
        ("soma", soma),
        ("repeated", repeated),
        ("empty", empty),
    ):
        for field, values in traces[name].items():
            expected = (
                traces["full"]["t"]
                if field == "t"
                else torch.stack(
                    [frame[region.index] for frame in traces["full"][field]]
                )
            )
            torch.testing.assert_close(values, expected)
    assert traces["empty"][voltage].numel() == 0
    # region gradients
    if execution == "scan":
        selected = traces["repeated"][voltage]
        expected = torch.stack(
            [frame[repeated.index] for frame in traces["full"][voltage]]
        )
        gradient = torch.autograd.grad(
            selected.square().sum(), initial_voltage, retain_graph=True
        )[0]
        reference = torch.autograd.grad(expected.square().sum(), initial_voltage)[0]
        torch.testing.assert_close(gradient, reference)
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    else:
        assert not traces["soma"][voltage].requires_grad
        assert runner.execution_report().strategy == "compiled_while_loop"
    # endregion gradients
    assert final["clock"]["step"].item() == 2 * STEPS
    assert network.t.item() == 0.0
    assert runner.initial_state["clock"]["step"].item() == 0
    return traces


def main():
    for concatenated in (False, True):
        for execution in ("scan", "while"):
            traces = run_demo(execution=execution, concatenated=concatenated)
            label = "concatenated" if concatenated else "ordinary"
            field = "all_populations.v" if concatenated else "left.v"
            print(label, execution, "soma shape", tuple(traces["soma"][field].shape))


if __name__ == "__main__":
    main()
