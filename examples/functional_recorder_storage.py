#!/usr/bin/env python3
"""Record HH state in memory and HDF5 with the same Recorder configuration.

Run from the repository root::

    python examples/functional_recorder_storage.py
    python examples/functional_recorder_storage.py --backend aot_eager
    python examples/functional_recorder_storage.py --execution while

The short CPU example checks mechanism-state selection, sampling cadence,
continuation and file readback. Scan also checks gradients through the
in-memory trajectory while the independent disk recorder stores detached data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.data import H5Reader
from dendra.models.mod import hh

DT = 0.025
STEPS = 9
PREFIX = 5
FIELDS = ("v", "hh.m", "t")


def make_model():
    model = dn.SingleCompartment(
        N=1,
        C=2,
        v_init=-63.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(hh)
    model.train()
    model.initialize()
    return model


def run_demo(directory, *, backend, execution):
    model = make_model()
    training = execution == "scan"
    path = directory / f"{execution}.h5"
    # region configuration
    memory = dn.Recorder(FIELDS, dt=2 * DT)
    disk = dn.Recorder(FIELDS, dt=2 * DT)
    disk.set_hdf5(str(path), cache_every=5)
    # endregion configuration
    options = {
        "dt": DT,
        "steps": STEPS,
        "chunk_steps": 4,
        "backend": backend,
        "execution": execution,
        "inference": not training,
    }
    if training:
        options["checkpoint_every"] = 4
    drive = (
        torch.linspace(1e-6, 3e-6, STEPS * model.v.numel(), dtype=model.v.dtype)
        .reshape(STEPS, *model.v.shape)
        .requires_grad_(training)
    )
    try:
        # region continuation
        program = dn.compile(
            model, callbacks={"memory": memory, "disk": disk}, **options
        )
        _, first = program(steps=PREFIX, inputs=dn.func.Inputs(intra=drive[:PREFIX]))
        final, last = program(
            first["continuation"],
            steps=STEPS - PREFIX,
            inputs=dn.func.Inputs(intra=drive[PREFIX:]),
        )
        samples = {
            name: torch.cat((first["callbacks"]["memory"][name], value))
            for name, value in last["callbacks"]["memory"].items()
        }
        assert "disk" not in first["callbacks"]
        assert "disk" in last["continuation"].callback_state
        # endregion continuation
        reference = dn.compile(
            model,
            callbacks={"memory": dn.Recorder(FIELDS, dt=2 * DT)},
            **options,
        )
        expected, whole = reference(inputs=dn.func.Inputs(intra=drive))
        torch.testing.assert_close(final, expected)
        torch.testing.assert_close(samples, dict(whole["callbacks"]["memory"]))
        torch.testing.assert_close(
            samples["t"],
            torch.arange(0, STEPS + 1, 2, dtype=torch.float64) * DT,
        )
        if training:
            # region gradients
            gradient = torch.autograd.grad(samples["hh.m"].square().sum(), drive)[0]
            expected_gradient = torch.autograd.grad(
                whole["callbacks"]["memory"]["hh.m"].square().sum(), drive
            )[0]
            torch.testing.assert_close(gradient, expected_gradient)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            # endregion gradients
    finally:
        # Complete queued writes before opening the file for reading.
        disk.close()

    # region readback
    with H5Reader(path) as recording:
        assert recording.vars() == ["hh.m", "t", "v"]
        for name in FIELDS:
            assert recording[name].n() == 2
            segments = [
                torch.from_numpy(recording[name][run].data.compute())
                for run in range(2)
            ]
            torch.testing.assert_close(
                torch.cat(segments), samples[name].detach().cpu()
            )
    # endregion readback
    assert model.t.item() == 0.0
    print(
        f"{execution}: HH state, aligned cadence, continued HDF5 segments "
        + ("and scan gradients passed" if training else "and inference passed")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    parser.add_argument("--execution", choices=("all", "scan", "while"), default="all")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-recorder-storage-") as temporary,
    ):
        for execution in ("scan", "while"):
            if arguments.execution in ("all", execution):
                run_demo(
                    Path(temporary), backend=arguments.backend, execution=execution
                )


if __name__ == "__main__":
    main()
