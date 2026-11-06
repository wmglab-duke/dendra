#!/usr/bin/env python3
"""Save a custom pure callback and resume through explicit receiving callbacks.

Run from the repository root::

    python examples/callbacks/functional_callback_checkpoint_protocol.py
    python examples/callbacks/functional_callback_checkpoint_protocol.py --backend aot_eager

The short CPU examples cover a Population and a connected Network. They check
recording phase, shared parameters and live/detached continuation gradients.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT, PREFIX, SUFFIX = 0.025, 4, 4
G = "integrator.mech.mechanisms.leak.g_param"


# region custom_callback
@dataclass(frozen=True)
class VoltageEnergy(dn.func.FunctionalCallback):
    reference: float
    scale: torch.Tensor

    def checkpoint_configuration(self):
        return {
            "kind": "org.example.voltage_energy",
            "version": 1,
            "fields": {"reference": self.reference, "scale": self.scale},
        }

    def initialize(self, state, auxiliary):
        voltage = state["integrator"]["v"]
        return {
            "total": voltage.new_zeros(()),
            "count": voltage.new_zeros((), dtype=torch.long),
        }, None

    def update(self, carry, state, auxiliary):
        voltage = state["integrator"]["v"]
        energy = (self.scale * (voltage - self.reference)).square().mean()
        count = carry["count"] + 1
        return {"total": carry["total"] + energy, "count": count}, {
            "energy": energy,
            "sample": count,
        }

    def finalize(self, carry, emissions):
        if emissions is None:
            emissions = {
                "energy": carry["total"].new_empty((0,)),
                "sample": carry["count"].new_empty((0,)),
            }
        return {**carry, **emissions}


# endregion custom_callback


def cell(initial, reversal):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=initial, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=reversal)
    return model


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def make_model(network):
    left = cell(-60.0, -55.0)
    if network:
        right = cell(-65.0, -70.0)
        right.insert(graded_syn.rename("continuous"), e=0.0)
        model = dn.Network({"left": left, "right": right})
        right.mech.leak.g_param = left.mech.leak.g_param
        model.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.continuous,
            pre_var="v",
            input="g_pre",
            weight=0.0005,
            delay=0.0,
            transform=release,
        )
    else:
        model = left
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def make_callbacks(functional, network):
    # region callback_configuration
    energy = VoltageEnergy(
        reference=-70.0, scale=torch.tensor(0.1, dtype=torch.float64)
    )
    if network:
        energy = dn.func.PopulationCallback("left", energy)
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    callbacks = functional.make_callbacks(
        {
            "energy": energy,
            "trace": dn.func.Recorder(paths, dt=3 * DT),
        }
    )
    # endregion callback_configuration
    return callbacks


def conductance(packet, network):
    local = packet.parameters["populations"]["left"] if network else packet.parameters
    return local[G]


def demo(network, backend, directory):
    functional, tensors = dn.func.make_functional(make_model(network), dt=DT)
    parameters = tensors.independent_parameters(trainable="*.g_param")
    initial = dn.func.initialize(functional, tensors._replace(parameters=parameters))
    callbacks = make_callbacks(functional, network)
    kernel = functional.compile_rollout_chunk(3, backend=backend)
    _, first = dn.func.run(kernel, initial, steps=PREFIX, callbacks=callbacks)
    final, last = dn.func.run(
        kernel, first["continuation"], steps=SUFFIX, callbacks=callbacks
    )
    expected, whole = dn.func.run(
        kernel, initial, steps=PREFIX + SUFFIX, callbacks=callbacks
    )
    torch.testing.assert_close(final, expected)
    source = conductance(initial, network)
    live_gradient = torch.autograd.grad(last["callbacks"]["energy"]["total"], source)[0]
    assert torch.isfinite(live_gradient).all() and live_gradient.abs().sum() > 0

    # region custom_checkpoint
    saved = dn.func.export_continuation(
        functional, first["continuation"], callbacks=callbacks
    )
    path = directory / ("network.pt" if network else "population.pt")
    torch.save(saved, path)

    # The receiving model and callback implementation are constructed explicitly.
    receiving, _ = dn.func.make_functional(make_model(network), dt=DT)
    receiving_callbacks = make_callbacks(receiving, network)
    restored = dn.func.restore_continuation(
        receiving, torch.load(path, weights_only=True), callbacks=receiving_callbacks
    )
    receiving_kernel = receiving.compile_rollout_chunk(3, backend=backend)
    restored_final, restored_aux = dn.func.run(
        receiving_kernel, restored, steps=SUFFIX, callbacks=receiving_callbacks
    )
    # endregion custom_checkpoint
    torch.testing.assert_close(restored_final, final)
    for name in ("energy", "trace"):
        torch.testing.assert_close(
            dict(restored_aux["callbacks"][name]), dict(last["callbacks"][name])
        )
    for name, suffix in restored_aux["callbacks"]["trace"].items():
        joined = torch.cat((first["callbacks"]["trace"][name], suffix))
        torch.testing.assert_close(joined, whole["callbacks"]["trace"][name])
        assert joined.shape[0] == 3  # Frames 0, 3 and 6 retain the Recorder phase.
        assert suffix.shape[0] == 1
    joined_samples = torch.cat(
        (
            first["callbacks"]["energy"]["sample"],
            restored_aux["callbacks"]["energy"]["sample"],
        )
    )
    torch.testing.assert_close(joined_samples, torch.arange(1, PREFIX + SUFFIX + 1))
    torch.testing.assert_close(
        restored_aux["callbacks"]["energy"]["total"],
        whole["callbacks"]["energy"]["total"],
    )

    # region custom_gradients
    report = dn.func.inspect_sources(restored)
    leaves = report.trainable_leaves
    loss = restored_aux["callbacks"]["energy"]["total"]
    gradients = torch.autograd.grad(loss, (*leaves, source), allow_unused=True)
    assert all(value is None or torch.isfinite(value).all() for value in gradients[:-1])
    assert any(value is not None and value.abs().sum() > 0 for value in gradients[:-1])
    assert gradients[-1] is None  # The portable boundary cuts the prefix graph.
    # endregion custom_gradients
    saved_g = conductance(restored, network)
    assert saved_g is not source and saved_g.is_leaf and saved_g.requires_grad
    if network:
        assert saved_g is restored.parameters["populations"]["right"][G]
    print(
        f"{'Scoped Network' if network else 'Population'}: custom callback metadata, "
        "disk continuation, recording phase and gradients passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-custom-callback-") as temporary,
    ):
        for network in (False, True):
            demo(network, args.backend, Path(temporary))


if __name__ == "__main__":
    main()
