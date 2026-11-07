"""Compile packed inference with source schedules and native continuation.

Run from the repository root with ``python examples/networks/functional_bitpacked_inference.py``.
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.125
STEPS = 5
DTYPE = torch.float64


def make_network():
    # region declarations
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE=DTYPE, DEVICE="cpu", IMEM=False):
        cell = dn.SingleCompartment(N=2, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        stimulus = dn.NetStim(N=2, interval=0.5, start=0.0, noise=0.3, seed=812)
        # Both schedules produce source spikes; neither schedules a NetCon edge.
        stimulus.schedule([0, 0, 1], [0.125, 0.375, 0.625], differentiable=False)
        stimulus.schedule(
            [0, 1], [0.25, 0.875], weight=[0.25, -0.5], differentiable=True
        )
        stimulus.set_diff_config(diff_scheduled_times=False)
        network = dn.Network(
            {"cell": cell}, netstim=stimulus, netcon_delay_backend="bitpacked_history"
        )
        network.connect_one_to_one(
            stimulus[:],
            cell[:],
            cell.mech.syn,
            weight=torch.tensor([0.002, 0.003], dtype=DTYPE),
            delay=torch.tensor([2 * DT, 3 * DT], dtype=DTYPE),
        )
        network.eval()
        network.initialize(DT, max_delay_ms=1.0)
    # endregion declarations
    return network


def clone_tree(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def assert_tree(actual, expected):
    values, spec = pytree.tree_flatten(actual)
    references, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def run_demo(*, initialization="snapshot", backend="aot_eager"):
    source, native = make_network(), make_network()
    native.netstim.restore_dict_from_checkpoint(
        clone_tree(source.netstim.state_dict_for_checkpoint())
    )
    # Capture a nonempty history for snapshot restarts. Fresh initialization
    # clears the connection history while retaining consumed source cursors.
    for _ in range(2):
        source.step()
        native.step()
    # region compile
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=True,
        initialization=initialization,
        backend=backend,
        callbacks={"trace": dn.func.Recorder(("cell.v",))},
    )
    name = next(iter(source.synapses))
    assert runner.functional.synapses[name].runtime == "bitpacked_history"
    # endregion compile
    before = runner.functional.extract().state
    if initialization == "fresh":
        native.initialize(DT, max_delay_ms=1.0)
    observations = [native.cell.v.clone()]
    prefix, first = runner()
    for _ in range(STEPS):
        native.step()
        observations.append(native.cell.v.clone())
    assert_tree(prefix, runner.functional.extract(native).state)
    # region continuation
    parameters = dict(runner.parameters)
    parameters["synapses"] = dict(parameters["synapses"])
    parameters["synapses"][name] = {
        **parameters["synapses"][name],
        "weight.weights.0.param.rho": parameters["synapses"][name][
            "weight.weights.0.param.rho"
        ]
        + 0.2,
    }
    final, second = runner(
        prefix,
        parameters=parameters,
        callback_state=first["callbacks"].state,
    )
    # endregion continuation
    effective = runner.functional._materialize_connections(
        parameters, first.get("constants", runner.constants)
    )["synapses"][name]["weights"]
    native.synapses[name].weight.w = effective.clone()
    for _ in range(STEPS):
        native.step()
        observations.append(native.cell.v.clone())
    assert_tree(final, runner.functional.extract(native).state)
    trace = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(trace, torch.stack(observations))
    assert not trace.requires_grad
    # region commit
    resumed = make_network()
    resumed.synapses[name].weight.w = effective.detach().clone()
    runner.functional.commit_state_(resumed, final)
    resumed.step()
    native.step()
    assert_tree(
        runner.functional.extract(resumed).state,
        runner.functional.extract(native).state,
    )
    # endregion commit
    assert_tree(runner.functional.extract().state, before)
    return final, trace


def main():
    for initialization in ("snapshot", "fresh"):
        state, trace = run_demo(initialization=initialization)
        print(
            initialization,
            "step",
            state["clock"]["step"].item(),
            "samples",
            trace.shape[0],
        )


if __name__ == "__main__":
    main()
