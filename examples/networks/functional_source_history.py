"""Compile auto-policy source-history training and continue after parameter edits.

Run from the repository root with ``python examples/networks/functional_source_history.py``.
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.125
STEPS = 5
DTYPE = torch.float64


def make_network(surrogate=True):
    # region declarations
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu", IMEM=False):
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        stimulus = dn.NetStim(N=1, interval=0.5, start=0.0, noise=0.0, seed=812)
        network = dn.Network({"cell": cell}, netstim=stimulus)
        network.connect_one_to_one(
            stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=2.25 * DT
        )
        network.train()
        network.initialize(DT, max_delay_ms=0.75)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=surrogate,
            diff_scheduled_times=False,
            taps=2,
            train_delay_backend="auto",
        )
    # endregion declarations
    return network


def assert_tree(actual, expected):
    values, spec = pytree.tree_flatten(actual)
    references, expected_spec = pytree.tree_flatten(expected)
    assert spec == expected_spec
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def run_demo(*, surrogate=True, backend="aot_eager"):
    source, native = make_network(surrogate), make_network(surrogate)
    # Match the complete RNG checkpoint, including the independent entropy seeder.
    native.netstim.restore_dict_from_checkpoint(
        pytree.tree_map(
            lambda value: value.detach().clone() if torch.is_tensor(value) else value,
            source.netstim.state_dict_for_checkpoint(),
        )
    )
    name = next(iter(source.synapses))
    # region compile
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=(
            f"synapses.{name}.weight.weights.0.param.rho",
            f"synapses.{name}.delay_ms.weights.0.param.rho",
        ),
        checkpoint_every=3,
        backend=backend,
        callbacks={"trace": dn.func.Recorder(("cell.v",))},
    )
    assert runner.functional.synapses[name].runtime == "source_history"
    # endregion compile
    before = runner.functional.extract().state
    weights = runner.get_parameter(f"synapses.{name}.weight.weights.0.param.rho")
    delays = runner.get_parameter(f"synapses.{name}.delay_ms.weights.0.param.rho")
    connection = native.synapses[name]
    native_weights = connection.weight.weights[0].param.rho
    native_delays = connection.delay_ms.weights[0].param.rho
    native_trace = [native.cell.v.clone()]
    prefix, first = runner()
    for _ in range(STEPS):
        native.step()
        native_trace.append(native.cell.v.clone())
    assert_tree(prefix, runner.functional.extract(native).state)
    # region changed-parameters
    parameters = dict(first["parameters"])
    parameters["synapses"] = dict(parameters["synapses"])
    parameters["synapses"][name] = {
        **parameters["synapses"][name],
        "weight.weights.0.param.rho": weights + 0.2,
        "delay_ms.weights.0.param.rho": delays - 0.15,
    }
    final, second = runner(
        prefix, parameters=parameters, callback_state=first["callbacks"].state
    )
    # endregion changed-parameters
    # Native transport reads initialized effective tensors. Materialize the same
    # raw-coordinate edits using the shared connection source plan.
    native_parameters = runner.functional.extract(native).parameters
    native_parameters["synapses"][name].update(
        {
            "weight.weights.0.param.rho": native_weights + 0.2,
            "delay_ms.weights.0.param.rho": native_delays - 0.15,
        }
    )
    effective = runner.functional._materialize_connections(
        native_parameters, first["constants"]
    )["synapses"][name]
    connection.weight.w = effective["weights"]
    connection.delay_ms.w = effective["delays"]
    for _ in range(STEPS):
        native.step()
        native_trace.append(native.cell.v.clone())
    assert_tree(final, runner.functional.extract(native).state)
    trace = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    expected = torch.stack(native_trace)
    torch.testing.assert_close(trace, expected)
    # region gradients
    gradients = torch.autograd.grad(trace.square().mean(), (weights, delays))
    references = torch.autograd.grad(
        expected.square().mean(), (native_weights, native_delays)
    )
    for gradient, reference in zip(gradients, references, strict=True):
        torch.testing.assert_close(gradient, reference)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # endregion gradients
    # region commit
    resumed = make_network(surrogate)
    resumed.synapses[name].weight.w = connection.weight.w.detach().clone()
    resumed.synapses[name].delay_ms.w = connection.delay_ms.w.detach().clone()
    runner.functional.commit_state_(resumed, final)
    resumed.step()
    native.step()
    assert_tree(
        runner.functional.extract(resumed).state,
        runner.functional.extract(native).state,
    )
    # endregion commit
    assert_tree(runner.functional.extract().state, before)
    return final, trace, gradients


def main():
    for surrogate in (True, False):
        state, trace, gradients = run_demo(surrogate=surrogate)
        print(
            "Source history:",
            "floating surrogate gates" if surrogate else "packed hard spikes",
        )
        print("Steps:", state["clock"]["step"].item(), "samples:", trace.shape[0])
        print(
            "Weight/delay gradient norms:", [value.norm().item() for value in gradients]
        )


if __name__ == "__main__":
    main()
