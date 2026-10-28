#!/usr/bin/env python3
"""Save callback continuation and resume an equivalent reconstructed runner.

Run from the repository root::

    python examples/functional_callback_checkpoints.py
    python examples/functional_callback_checkpoints.py --backend aot_eager
    python examples/functional_callback_checkpoints.py --execution while

The small CPU examples check disk round-trips and continuation, not performance.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.models.mod import expsyn, graded_syn, pas

DT, STEPS = 0.025, 4


def snapshot(tree):
    """Own detached tensor storage while retaining shared parameter aliases."""
    tensors = {}

    def copy(value):
        if isinstance(value, torch.Tensor):
            identity = id(value)
            if identity not in tensors:
                with torch.inference_mode(False):
                    tensors[identity] = value.detach().clone()
            return tensors[identity]
        if isinstance(value, Mapping):
            return {name: copy(child) for name, child in value.items()}
        if isinstance(value, tuple):
            children = tuple(copy(child) for child in value)
            return type(value)(*children) if hasattr(value, "_fields") else children
        if isinstance(value, list):
            return [copy(child) for child in value]
        return value

    return copy(tree)


# region models
def make_population():
    model = dn.SingleCompartment(
        N=2, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.diam.fill_(20.0)
    model.dx.fill_(20.0)
    model.insert(pas.rename("leak"), g=0.001, e=-45.0)
    model.eval()
    model.initialize(DT)
    return model


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def make_network():
    left = dn.SingleCompartment(
        N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    right = dn.SingleCompartment(
        N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    left.insert(pas.rename("leak"), g=0.001, e=-55.0)
    right.insert(pas.rename("leak"), g=0.001, e=-70.0)
    right.insert(expsyn.rename("event"), tau=2.0, e=0.0)
    right.insert(graded_syn.rename("continuous"), e=0.0)
    stimulus = dn.NetStim(N=1, interval=0.04, start=0.0, noise=0.4, seed=71)
    network = dn.Network(
        {"left": left, "right": right},
        netstim=stimulus,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    network.connect_one_to_one(
        stimulus[:], right[:], right.mech.event, weight=0.002, delay=DT
    )
    network.connect_continuous_one_to_one(
        left[:],
        right[:],
        right.mech.continuous,
        pre_var="v",
        input="g_pre",
        weight=0.0005,
        delay=0.0,
        transform=release,
    )
    # The checkpoint keeps these two raw parameter paths aliased.
    right.mech.leak.g_param = left.mech.leak.g_param
    network.eval()
    network.initialize(DT)
    return network


# endregion models


# region compile
def compile_runner(model, *, network, backend, execution, steps=STEPS, threshold=None):
    if threshold is None:
        threshold = -59.8 if network else -59.5
    options = dict(
        threshold=threshold,
        node_check=[0],
        dt=DT,
        t_start_check=0.0,
        t_end_check=2 * STEPS * DT,
    )
    if network:
        callbacks = {
            "trace": dn.func.Recorder(("left.v", "right.v", "t")),
            "raster": dn.func.PopulationCallback("left", dn.func.Raster(**options)),
            "counts": dn.func.PopulationCallback("left", dn.func.APCount(**options)),
            "anomalies": dn.func.PopulationCallback("right", dn.func.AnomalyDetector()),
        }
    else:
        callbacks = {
            "trace": dn.func.Recorder(("v", "t")),
            "raster": dn.func.Raster(**options),
            "counts": dn.func.APCount(**options),
            "anomalies": dn.func.AnomalyDetector(),
        }
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        execution=execution,
        callbacks=callbacks,
        backend=backend,
    )


# endregion compile


# region checkpoint
def save_checkpoint(path, runner, state, auxiliary):
    continuation = auxiliary["continuation"]
    # Callback export owns detached carry and metadata, excluding old traces.
    callback_packet = runner.callbacks.export_state(continuation.callback_state)
    payload = snapshot(
        {
            "state": state,
            "parameters": continuation.parameters,
            "constants": continuation.constants,
            "callbacks": callback_packet,
        }
    )
    torch.save(payload, path)
    return payload


def resume_checkpoint(path, receiver):
    saved = torch.load(path, weights_only=True)
    carry = receiver.callbacks.restore_state(saved["callbacks"])
    # A returned state continues without running initialization again.
    return receiver(
        saved["state"],
        parameters=saved["parameters"],
        constants=saved["constants"],
        callback_state=carry,
    )


# endregion checkpoint


def joined_trace(first, second):
    return {
        name: torch.cat((first["callbacks"]["trace"][name], value))
        for name, value in second["callbacks"]["trace"].items()
    }


def verify_continuation(first, resumed, expected):
    resumed_state, resumed_auxiliary = resumed
    expected_state, expected_auxiliary = expected
    torch.testing.assert_close(resumed_state, expected_state)
    observed = joined_trace(first, resumed_auxiliary)
    torch.testing.assert_close(observed, expected_auxiliary["callbacks"]["trace"])
    for value in observed.values():
        assert value.shape[0] == 2 * STEPS + 1
    for value in resumed_auxiliary["callbacks"]["trace"].values():
        assert value.shape[0] == STEPS  # No repeated boundary sample.
    raster = torch.cat(
        (first["callbacks"]["raster"], resumed_auxiliary["callbacks"]["raster"])
    )
    torch.testing.assert_close(raster, expected_auxiliary["callbacks"]["raster"])
    for name in ("counts", "anomalies"):
        torch.testing.assert_close(
            resumed_auxiliary["callbacks"][name], expected_auxiliary["callbacks"][name]
        )
    assert resumed_auxiliary["callbacks"]["counts"].sum().item() > 0


# region demo
def run_demo(*, network, backend, execution, directory):
    make_model = make_network if network else make_population
    model = make_model()
    runner = compile_runner(
        model, network=network, backend=backend, execution=execution
    )
    native_start = snapshot(runner.functional.extract().state)
    prefix, first = runner()
    path = directory / ("network.pt" if network else "population.pt")
    payload = save_checkpoint(path, runner, prefix, first)

    # Rebuild ordinary model, lowering plan, callbacks and compiled runner.
    receiver = compile_runner(
        make_model(), network=network, backend=backend, execution=execution
    )
    assert receiver.callbacks is not runner.callbacks
    resumed = resume_checkpoint(path, receiver)

    whole = compile_runner(
        model,
        network=network,
        backend=backend,
        execution=execution,
        steps=2 * STEPS,
    )
    verify_continuation(first, resumed, whole())
    # Comparing complete Network state also checks queues and NetStim RNG carry.
    torch.testing.assert_close(runner.functional.extract().state, native_start)

    incompatible = compile_runner(
        make_model(),
        network=network,
        backend=backend,
        execution=execution,
        threshold=-58.0,
    )
    try:
        incompatible.callbacks.restore_state(payload["callbacks"])
    except dn.func.FunctionalizationError:
        pass
    else:
        raise AssertionError("a different threshold must reject callback restoration")
    print(
        f"{'Network' if network else 'Population'}: disk checkpoint, reconstructed runner, "
        "recording/reducer continuation and configuration validation passed"
    )


# endregion demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", choices=("all", "population", "network"), default="all"
    )
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    parser.add_argument("--execution", choices=("scan", "while"), default="scan")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-callback-checkpoint-") as temporary,
    ):
        for network in (False, True):
            name = "network" if network else "population"
            if arguments.section in ("all", name):
                run_demo(
                    network=network,
                    backend=arguments.backend,
                    execution=arguments.execution,
                    directory=Path(temporary),
                )


if __name__ == "__main__":
    main()
