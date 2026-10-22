"""Observe individual Populations in a compiled functional Network.

Run from the repository root: python examples/functional_network_callbacks.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.1
STEPS = 4


def make_network():
    # region network
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64, DEVICE="cpu"):
        left = dn.SingleCompartment(N=2, C=3, integrator=dn.bwd_euler_sc(imem=True))
        right = dn.SingleCompartment(N=2, C=2, integrator=dn.bwd_euler_sc(imem=True))
        left[:, 0].label("soma")
        left.insert(pas, g=0.001, e=-55.0)
        right.insert(pas, g=0.001, e=-65.0)
        right.insert(expsyn.rename("event"), tau=2.0, e=0.0)
        network = dn.Network({"left": left, "right": right})
        network.connect_one_to_one(
            left.soma,
            right[:, 0],
            right.mech.event,
            threshold=-64.0,
            weight=0.002,
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


def run_demo(*, execution="scan", backend="aot_eager"):
    network = make_network()
    # region callbacks
    lead_field = torch.tensor(
        [[1.0, 0.5, 0.25], [0.75, 0.5, 0.125]],
        dtype=torch.float64,
        requires_grad=execution == "scan",
    )
    fields = [lead_field, -0.5 * lead_field]
    # Voltage/current: (batch=2, cells=2, compartments=3).
    # Each (cells=2, compartments=3) field broadcasts over the batch axis.
    selected = {
        "threshold": -100.0,
        "node_check": [0, -1],
        "t_start_check": 2 * DT,
        "t_end_check": 6 * DT,
    }
    scoped = dn.func.PopulationCallback
    callbacks = {
        "lfp": scoped("left", dn.func.LFP(fields)),
        "raster": scoped("left", dn.func.Raster(**selected)),
        "counts": scoped("left", dn.func.APCount(**selected)),
        "active": scoped("left", dn.func.Active(**selected)),
        "site_quorum": scoped("left", dn.func.ActiveAL(**selected, at_least=2)),
        "event_quorum": scoped("left", dn.func.ActiveALCount(**selected, at_least=2)),
        "anomaly": scoped("right", dn.func.AnomalyDetector()),
        "soma": scoped("left", dn.func.Recorder(("v", "t"), region=network.left.soma)),
        "trace": dn.func.Recorder(("left.v", "left.i_membrane", "t")),
    }
    # endregion callbacks
    # region compile
    options = {
        "dt": DT,
        "execution": execution,
        "inference": execution == "while",
        "callbacks": callbacks,
        "backend": backend,
    }
    runner = dn.compile(network, steps=STEPS, **options)
    uninterrupted = dn.compile(network, steps=2 * STEPS, **options)
    # endregion compile
    # region continuation
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    reference, whole = uninterrupted()
    first, second, whole = (
        first["callbacks"],
        second["callbacks"],
        whole["callbacks"],
    )
    lfp = torch.cat((first["lfp"], second["lfp"]))
    raster = torch.cat((first["raster"], second["raster"]))
    trace = {
        field: torch.cat((first["trace"][field], values))
        for field, values in second["trace"].items()
    }
    soma = torch.cat((first["soma"]["v"], second["soma"]["v"]))
    # endregion continuation
    torch.testing.assert_close(lfp, whole["lfp"])
    torch.testing.assert_close(raster, whole["raster"])
    for name in ("counts", "active", "site_quorum", "event_quorum", "anomaly"):
        torch.testing.assert_close(second[name], whole[name])
    for field, values in trace.items():
        torch.testing.assert_close(values, whole["trace"][field])
    for name in ("left", "right"):
        torch.testing.assert_close(
            final["populations"][name]["integrator"]["v"],
            reference["populations"][name]["integrator"]["v"],
        )
    # region results
    assert lfp.shape == (2 * STEPS + 1, 2, 2)  # Frames, batch, contacts.
    assert raster.shape == (2 * STEPS, 2, 2, 2)  # Steps, batch, cells, sites.
    assert soma.shape == (2 * STEPS + 1, 2, 2)
    assert second["counts"].shape == (2, 2, 2)
    assert second["anomaly"].shape == (2, 2)
    torch.testing.assert_close(second["counts"], raster.sum(dim=0))
    torch.testing.assert_close(second["active"], (second["counts"] > 0).any(dim=-1))
    torch.testing.assert_close(
        second["site_quorum"], (second["counts"] > 0).sum(dim=-1) >= 2
    )
    torch.testing.assert_close(
        second["event_quorum"], second["counts"].sum(dim=-1) >= 2
    )
    # This low threshold illustrates callback phase, not physiological spikes.
    assert not raster[:2].any() and raster[2].all() and not raster[3:].any()
    assert not second["anomaly"].any()
    # endregion results
    # region gradients
    current = trace["left.i_membrane"]  # (frames, batch, cells, compartments).
    projected = torch.einsum("tbnc,knc->tbk", current, torch.stack(fields))
    torch.testing.assert_close(lfp, projected)
    if execution == "scan":
        gradient = torch.autograd.grad(
            lfp.square().sum(), lead_field, retain_graph=True
        )[0]
        expected = torch.autograd.grad(projected.square().sum(), lead_field)[0]
        torch.testing.assert_close(gradient, expected)
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    else:
        assert not lfp.requires_grad
        assert runner.execution_report().strategy == "compiled_while_loop"
    # endregion gradients
    assert network.t.item() == 0.0
    assert final["clock"]["step"].item() == 2 * STEPS
    return lfp.detach(), raster, second["counts"]


def main():
    scan = run_demo(execution="scan")
    while_results = run_demo(execution="while")
    for actual, expected in zip(while_results, scan, strict=True):
        torch.testing.assert_close(actual, expected)
    print("scan/while LFP shape:", tuple(scan[0].shape))
    print("continued counts:", scan[2].tolist())


if __name__ == "__main__":
    main()
