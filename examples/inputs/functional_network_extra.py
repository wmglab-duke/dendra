"""Bind and optimize extracellular waveforms in a functional Network.

Run from the repository root: python examples/inputs/functional_network_extra.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.05
STEPS = 4


def make_network():
    # region network
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        populations = {}
        for name in ("left", "right"):
            cell = dn.Unmyelinated(
                [2.0],
                L=40.0,
                dx=10.0,
                rhoa=100.0,
                v_init=-65.0,
                integrator=dn.bwd_euler_ub(method="thomas", imem=False),
            )
            cell.insert(pas, g=0.001, e=-65.0)
            populations[name] = cell
        populations["right"].insert(expsyn.rename("event"), tau=2.0, e=0.0)
        network = dn.Network(populations)
        network.connect_one_to_one(
            network.left[:, 0],
            network.right[:, 0],
            network.right.mech.event,
            threshold=-64.5,
            weight=0.001,
            delay=DT,
        )
        network.batch(2)
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            train_delay_backend="dense",
        )
    # endregion network
    return network


def make_extra(network):
    # region extra
    # A (cells=1, compartments=5) field broadcasts over both batch lanes.
    field = torch.linspace(
        -1.0, 1.25, network.left.shape[-1], dtype=torch.float64
    ).reshape(1, -1)
    other_field = field.flip(-1).clone()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        shared_waveform = dn.sin(amp=0.2, freq=0.7, phase=0.3)
        other_waveform = dn.cos(amp=0.1, freq=0.4, phase=-0.2)
    extra = {
        "left": [(field, shared_waveform), (other_field, other_waveform)],
        "right": (field, shared_waveform),
    }
    assert field.shape == network.left.shape[-2:]
    # endregion extra
    return extra


def run_demo(*, execution="scan", backend="aot_eager"):
    network = make_network()
    extra = make_extra(network)
    # region compile
    options = {
        "dt": DT,
        "extra": extra,
        "execution": execution,
        "inference": execution == "while",
        "trainable": (
            "populations.*.stimulation.extra.contacts.*.field",
            "populations.*.stimulation.extra.waveforms.*.amp",
            "populations.*.stimulation.extra.waveforms.*.freq",
            "populations.*.stimulation.extra.waveforms.*.phase",
        ),
        "callbacks": {"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        "backend": backend,
    }
    if execution == "scan":
        options["checkpoint_every"] = 2
    runner = dn.compile(network, steps=STEPS, **options)
    uninterrupted = dn.compile(network, steps=2 * STEPS, **options)
    # endregion compile
    # region parameters
    left = runner.parameters["populations"]["left"]
    right = runner.parameters["populations"]["right"]
    field_name = "stimulation.extra.contacts.0.field"
    amplitude_name = "stimulation.extra.waveforms.0.amp"
    assert left[field_name] is right[field_name]
    assert left[amplitude_name] is right[amplitude_name]
    selected = runner.trainable_parameters
    assert f"populations.left.{field_name}" in selected
    # Deduplicate shared aliases before passing leaves to an optimizer.
    optimizer_leaves = {id(value): value for value in selected.values()}
    assert len(optimizer_leaves) < len(selected)
    # endregion parameters
    # region continuation
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    reference, whole = uninterrupted()
    trace = {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in second["callbacks"]["trace"].items()
    }
    for name, values in trace.items():
        torch.testing.assert_close(values, whole["callbacks"]["trace"][name])
    for name in network.populations:
        torch.testing.assert_close(
            final["populations"][name]["integrator"]["v"],
            reference["populations"][name]["integrator"]["v"],
        )
    assert trace["left.v"].shape == (2 * STEPS + 1, *network.left.shape)
    assert final["clock"]["step"].item() == 2 * STEPS
    # endregion continuation
    # region gradients
    if execution == "scan":
        targets = (
            left[field_name],
            left[amplitude_name],
            left["stimulation.extra.waveforms.0.freq"],
            left["stimulation.extra.waveforms.0.phase"],
        )
        voltage = trace["left.v"] + 0.7 * trace["right.v"]
        weights = torch.linspace(0.7, 1.3, voltage.shape[-1], dtype=voltage.dtype)
        loss = (weights * voltage).square().mean()
        gradients = torch.autograd.grad(loss, targets)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
    else:
        assert not trace["left.v"].requires_grad
        assert runner.execution_report().strategy == "compiled_while_loop"
    # endregion gradients
    # region mixed-inputs
    # Bind only left. Supply an already assembled voltage tape for right.
    plan, tensors = dn.func.make_functional_network(
        network, extra={"left": extra["left"]}
    )
    field, waveform = extra["right"]
    times = torch.arange(STEPS, dtype=torch.float64) * DT
    ve = waveform(times).reshape(STEPS, 1, 1, 1) * field
    ve = ve.expand(STEPS, *network.right.shape)
    inputs = dn.func.NetworkRolloutInput(
        populations={"right": dn.func.RolloutInput(ve=ve)}
    )
    callbacks = plan.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    )
    _, explicit = plan.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        callbacks=callbacks,
    )
    for name, values in explicit["callbacks"]["trace"].items():
        torch.testing.assert_close(values, first["callbacks"]["trace"][name])
    # endregion mixed-inputs
    assert network.t.item() == 0.0
    return {name: value.detach() for name, value in trace.items()}


def main():
    scan = run_demo(execution="scan")
    while_trace = run_demo(execution="while")
    for name, values in scan.items():
        torch.testing.assert_close(values, while_trace[name])
    print("scan/while voltage shape:", tuple(scan["left.v"].shape))
    print("final time:", scan["t"][-1].item())


if __name__ == "__main__":
    main()
