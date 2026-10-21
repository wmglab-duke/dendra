"""Compile scheduled packed/sparse traffic and continue through native checkpoints.

Run with ``python examples/functional_pending_calendar.py`` from the repository root.
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.125
STEPS = 3
DTYPE = torch.float64


def make_network(runtime="bitpacked_history"):
    # region declarations
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE=DTYPE, DEVICE="cpu", IMEM=False):
        cell = dn.SingleCompartment(N=2, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        stimulus = dn.NetStim(N=2, interval=100.0, start=0.0, noise=0.0, seed=812)
        network = dn.Network(
            {"cell": cell}, netstim=stimulus, netcon_delay_backend=runtime
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
        connection = next(iter(network.synapses.values()))
        connection.set_diff_config(diff_scheduled_times=False)
        connection.schedule(
            con_indices=[0, 1, 0, 1],
            times_ms=[0.0, 0.0, 4 * DT, 4 * DT],
            weight=torch.tensor([0.5, -0.25, 0.75, 0.5], dtype=DTYPE),
        )
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


def run_demo(runtime="bitpacked_history", *, backend="aot_eager"):
    source, native, resumed = (make_network(runtime) for _ in range(3))
    for target in (native, resumed):
        target.netstim.restore_dict_from_checkpoint(
            clone_tree(source.netstim.state_dict_for_checkpoint())
        )
    # region compile
    name = next(iter(source.synapses))
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        pending_capacity={name: 16},
        backend=backend,
        callbacks={"trace": dn.func.Recorder(("cell.v",))},
    )
    assert runner.functional.synapses[name].runtime == runtime
    prefix, first = runner()
    # endregion compile
    trace = [native.cell.v.clone()]
    for _ in range(STEPS):
        native.step()
        trace.append(native.cell.v.clone())
    assert_tree(prefix, runner.functional.extract(native).state)

    # region continuation
    parameters = dict(runner.parameters)
    parameters["synapses"] = {
        name: {
            **parameters["synapses"][name],
            "weights": parameters["synapses"][name]["weights"] * 2,
        }
    }
    final, second = runner(
        prefix,
        parameters=parameters,
        callback_state=first["callbacks"].state,
    )
    # endregion continuation
    native.synapses[name].weight.w = parameters["synapses"][name]["weights"].clone()
    for _ in range(STEPS):
        native.step()
        trace.append(native.cell.v.clone())
    assert_tree(final, runner.functional.extract(native).state)
    recording = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(recording, torch.stack(trace))

    # region checkpoint
    runner.functional.commit_state_(resumed, prefix)
    checkpoint = clone_tree(resumed.state_dict_for_checkpoint())
    restored = make_network(runtime)
    restored.restore_dict_from_checkpoint(checkpoint)
    for target in (resumed, restored):
        target.synapses[name].weight.w = parameters["synapses"][name]["weights"].clone()
        for _ in range(STEPS):
            target.step()
        assert_tree(final, runner.functional.extract(target).state)
    # endregion checkpoint
    return final, recording


def run_overflow_demo(*, backend="aot_eager"):
    source = make_network("sparse_calendar")
    # region overflow
    small = dn.compile(source, dt=DT, steps=1, pending_capacity=1, backend=backend)
    snapshot = clone_tree(small.initial_state)
    try:
        small()
    except RuntimeError as error:
        assert "pending_capacity" in str(error)
    else:
        raise AssertionError("The insufficient calendar capacity must raise")
    assert_tree(small.initial_state, snapshot)
    # Recompile from the unchanged native snapshot with enough pending storage.
    larger = dn.compile(source, dt=DT, steps=1, pending_capacity=16, backend=backend)
    state, _ = larger()
    larger.functional.check_overflow(state)
    # endregion overflow
    return state


def main():
    for runtime in ("bitpacked_history", "sparse_calendar"):
        state, trace = run_demo(runtime)
        print(runtime, "step", state["clock"]["step"].item(), "samples", len(trace))
    run_overflow_demo()
    print("Overflow rejected; retry with larger capacity succeeded.")


if __name__ == "__main__":
    main()
