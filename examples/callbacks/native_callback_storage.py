#!/usr/bin/env python3
"""Stream pure native callback observations to HDF5 with explicit carry.

Run from the repository root::

    python examples/callbacks/native_callback_storage.py

The bounded CPU example uses temporary files for a Population and a Network.
It continues mixed disk/in-memory recording and a pure reducer through manual
steps, run, checkpointed longrun and longrun. Functional values and gradients
provide an independent reference; file readback checks cadence and replay.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.data import H5Reader
from dendra.models.mod import pas

DT = 0.025
MANUAL, RUN, CHECKPOINTED, LONG = 3, 2, 3, 1
STEPS = MANUAL + RUN + CHECKPOINTED + LONG
G = "integrator.mech.mechanisms.leak.g_param"


class VoltageEnergy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return {
            "energy": carry["energy"] + state["integrator"]["v"].square().mean(),
            "steps": carry["steps"] + 1,
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


def make_model(network):
    populations = {}
    specifications = (
        (("left", 2, -55.0), ("right", 3, -75.0)) if network else (("cell", 2, -55.0),)
    )
    for name, compartments, reversal in specifications:
        population = dn.SingleCompartment(
            N=1,
            C=compartments,
            v_init=-65.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        population.insert(pas.rename("leak"), g=0.001, e=reversal)
        populations[name] = population
    model = dn.Network(populations) if network else populations["cell"]
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def make_callbacks(functional, network, disk=None):
    scope = dn.callbacks.PopulationCallback
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    configured = {
        "memory": dn.Recorder(fields, dt=2 * DT),
        "summary": scope("right", VoltageEnergy()) if network else VoltageEnergy(),
    }
    if disk is not None:
        configured["disk"] = scope("right", disk) if network else disk
    return functional.make_callbacks(configured)


def run_demo(directory, network):
    model = make_model(network)
    path = directory / ("network.h5" if network else "population.h5")
    # region configuration
    disk = dn.Recorder(("v", "t"), dt=2 * DT)
    disk.set_hdf5(str(path), cache_every=3)
    # endregion configuration
    clock = {} if network else {"dt": DT}
    try:
        functional, _ = dn.func.make_functional(model, dt=DT)
        observations = make_callbacks(functional, network, disk)
        # region continuation
        parts = []
        carry = None
        for _ in range(MANUAL):
            result = model.step(
                **clock,
                callbacks=observations,
                callback_state=carry,
            )
            parts.append(result)
            carry = result.state
        result = model.run(
            tstop=RUN * DT,
            **clock,
            callbacks=observations,
            callback_state=carry,
        )
        parts.append(result)
        hook_loss, result = model.longrun_checkpointed(
            tstop=CHECKPOINTED * DT,
            chunklength=2,
            **clock,
            callbacks=observations,
            callback_state=result.state,
            safe_checkpoint=True,
        )
        assert hook_loss is None
        parts.append(result)
        result = model.longrun(
            tstop=LONG * DT,
            chunklength=2,
            **clock,
            callbacks=observations,
            callback_state=result.state,
        )
        parts.append(result)
        # endregion continuation
        fields = ("left.v", "right.v", "t") if network else ("v", "t")
        samples = {
            field: torch.cat(tuple(part["memory"][field] for part in parts))
            for field in fields
        }
        assert all("disk" not in part and "disk" in part.state for part in parts)
        assert result["summary"]["steps"].item() == STEPS
        assert parts[2]["memory"]["t"].numel() == 0
        assert parts[-1]["memory"]["t"].numel() == 0
        torch.testing.assert_close(
            samples["t"], torch.arange(0, STEPS + 1, 2, dtype=torch.float64) * DT
        )
        segments = sum(part["memory"]["t"].numel() > 0 for part in parts)
        assert disk.run_number == segments == 4

        reference_model = make_model(network)
        reference, tensors = dn.func.make_functional(reference_model, dt=DT)
        _, auxiliary = reference.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=STEPS,
            callbacks=make_callbacks(reference, network),
        )
        expected = auxiliary["callbacks"]
        torch.testing.assert_close(samples, expected["memory"])
        torch.testing.assert_close(result["summary"], expected["summary"])
        targets = (
            tuple(
                model.populations[name].mech.leak.g_param for name in ("left", "right")
            )
            if network
            else (model.mech.leak.g_param,)
        )
        reference_targets = (
            tuple(
                tensors.parameters["populations"][name][G] for name in ("left", "right")
            )
            if network
            else (tensors.parameters[G],)
        )
        objective = result["summary"]["energy"] + sum(
            samples[field].square().mean() for field in fields[:-1]
        )
        reference_objective = expected["summary"]["energy"] + sum(
            expected["memory"][field].square().mean() for field in fields[:-1]
        )
        # region gradients
        runs_before_backward = disk.run_number
        observed = torch.autograd.grad(objective, targets)
        wanted = torch.autograd.grad(reference_objective, reference_targets)
        for gradient, reference_gradient in zip(observed, wanted, strict=True):
            torch.testing.assert_close(gradient, reference_gradient)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert disk.run_number == runs_before_backward
        # endregion gradients
    finally:
        # Finish the writer before opening the file or removing its directory.
        disk.close()

    # region readback
    voltage = "right.v" if network else "v"
    with H5Reader(path) as recording:
        assert recording.vars() == ["t", "v"]
        for disk_field, memory_field in (("v", voltage), ("t", "t")):
            assert recording[disk_field].n() == segments
            persisted = torch.cat(
                tuple(
                    torch.from_numpy(recording[disk_field][run].data.compute())
                    for run in range(segments)
                )
            )
            torch.testing.assert_close(persisted, samples[memory_field].detach().cpu())
            assert not persisted.requires_grad
    # endregion readback
    print(
        f"{'Network' if network else 'Population'}: continued native HDF5 recording, "
        "cadence, gradients and replay passed"
    )


def main():
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-native-callback-storage-") as temporary,
    ):
        for network in (False, True):
            run_demo(Path(temporary), network)


if __name__ == "__main__":
    main()
