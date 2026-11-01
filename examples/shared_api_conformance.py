#!/usr/bin/env python3
"""Continue a training simulation across native, compiled and saved execution.

Run from the repository root::

    python examples/shared_api_conformance.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT = 0.025
G = "integrator.mech.mechanisms.leak.g_param"


def population():
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-55.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-70.0)
    model.insert(pas.rename("paired"), g=0.001, e=-65.0)
    model.train()
    model.initialize()
    model.mech.paired.g_param = model.mech.leak.g_param
    model.initialize()
    return model


def recordings():
    return {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}


def compile_program(model, *, callbacks=None):
    return dn.compile(
        model,
        dt=DT,
        steps=3,
        chunk_steps=3,
        inference=False,
        trainable="*g_param",
        callbacks=recordings() if callbacks is None else callbacks,
        backend="eager",
    )


def inputs(model):
    shape = (10, *model.shape)
    count = 10 * model.v.numel()
    voltage = torch.linspace(
        -0.2, 0.3, count, dtype=model.dtype(), device=model.device()
    ).reshape(shape)
    current = torch.linspace(
        0.8e-10, 1.2e-10, count, dtype=model.dtype(), device=model.device()
    ).reshape(shape)
    return tuple(
        dn.func.Inputs(
            voltage[start:stop].clone().requires_grad_(),
            current[start:stop].clone().requires_grad_(),
        )
        for start, stop in ((0, 2), (2, 5), (5, 6), (6, 8), (8, 10))
    )


def demonstrate(directory):
    # region native_prefix
    model = population()
    drives = inputs(model)
    native_callbacks = dn.func.make_callbacks(model, recordings(), dt=DT)
    prefix = model.run(
        duration=2.4 * DT, dt=DT, inputs=drives[0], callbacks=native_callbacks
    )
    # Two steps execute; 0.4*DT remains available to a later duration call.
    torch.testing.assert_close(model.t, model.t.new_tensor(2 * DT))
    # endregion native_prefix

    # region compiled
    program = compile_program(model)  # Snapshot the current native state.
    callback_state = program.callbacks.restore_state(
        native_callbacks.export_state(prefix.state)
    )
    _, first = dn.func.run(
        program, steps=3, inputs=drives[1], callback_state=callback_state
    )
    _, second = dn.func.run(
        program, first["continuation"], duration=DT, inputs=drives[2]
    )
    # endregion compiled

    # region checkpoint
    path = directory / "continuation.pt"
    torch.save(program.export_continuation(second["continuation"]), path)
    receiver = compile_program(population())
    restored = receiver.restore_continuation(torch.load(path, weights_only=True))
    _, third = dn.func.run(receiver, restored, steps=2, inputs=drives[3])
    continuation = third["continuation"]
    # The restored training sources are independent leaves.
    restored_parameter = restored.parameters[G]
    assert restored_parameter.is_leaf and restored_parameter.requires_grad
    assert restored_parameter is not program.parameters[G]
    # endregion checkpoint

    # region native_handoff
    target = population()
    receiving_parameter = target.mech.leak.g_param
    target_callbacks = dn.func.make_callbacks(target, recordings(), dt=DT)
    callback_state = target_callbacks.restore_state(
        receiver.callbacks.export_state(continuation.callback_state)
    )
    dn.func.commit_continuation_(receiver, target, continuation)
    assert target.mech.leak.g_param is receiving_parameter
    assert target.mech.paired.g_param is receiving_parameter
    _, suffix = target.longrun_checkpointed(
        duration=1.6 * DT,
        chunklength=2,
        dt=DT,
        inputs=drives[4],
        callbacks=target_callbacks,
        callback_state=callback_state,
        safe_checkpoint=True,
    )
    # endregion native_handoff

    with torch.no_grad():
        expected, auxiliary = dn.func.run(
            receiver, continuation, duration=1.6 * DT, inputs=drives[4]
        )
    torch.testing.assert_close(target.v, expected["integrator"]["v"])
    torch.testing.assert_close(
        suffix["trace"]["v"], auxiliary["callbacks"]["trace"]["v"]
    )
    results = (
        prefix,
        first["callbacks"],
        second["callbacks"],
        third["callbacks"],
        suffix,
    )
    times = torch.cat(tuple(result["trace"]["t"] for result in results))
    torch.testing.assert_close(times, torch.arange(0, 11, 2, dtype=target.dtype()) * DT)

    # region gradients
    saved_prefix_voltage = continuation.state["integrator"]["v"]
    gradients = torch.autograd.grad(
        target.v.square().mean() + suffix["trace"]["v"].square().mean(),
        (receiving_parameter, *drives[4], restored_parameter, saved_prefix_voltage),
        allow_unused=True,
    )
    assert all(
        value.isfinite().all() and value.abs().sum() > 0 for value in gradients[:3]
    )
    assert gradients[3:] == (None, None)
    # The independent target leaves the saved compiled graph available.
    saved_gradient = torch.autograd.grad(
        saved_prefix_voltage.square().mean(), restored_parameter
    )[0]
    assert saved_gradient.isfinite().all() and saved_gradient.abs().sum() > 0
    # endregion gradients
    print("Native → compiled → portable restore → native: state and cadence passed")
    print("Source aliases, suffix input gradients and detached handoff passed")


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


# region connection_factory
def network():
    left, right = population(), population()
    right.insert(graded_syn.rename("graded"), e=0.0)
    right.initialize()
    right.mech.leak.g_param = left.mech.leak.g_param
    right.mech.paired.g_param = left.mech.leak.g_param
    model = dn.Network({"left": left, "right": right})
    model.connect_continuous_one_to_one(
        left[:],
        right[:],
        right.mech.graded,
        pre_var="v",
        input="g_pre",
        weight=0.0005,
        delay=DT,
        transform=release,
    )
    model.train()
    model.initialize(DT)
    return model


# endregion connection_factory


def demonstrate_network(directory):
    # region network
    model = network()
    drives = inputs(model.left)
    configuration = {"trace": dn.Recorder(("left.v", "right.v", "t"), dt=2 * DT)}

    def packet(index):
        return {"left": drives[index], "right": drives[index]}

    native_callbacks = dn.func.make_callbacks(model, configuration, dt=DT)
    prefix = model.run(duration=2.4 * DT, inputs=packet(0), callbacks=native_callbacks)
    program = compile_program(model, callbacks=configuration)
    carry = program.callbacks.restore_state(native_callbacks.export_state(prefix.state))
    _, first = dn.func.run(
        program, program.initial_state, steps=3, inputs=packet(1), callback_state=carry
    )
    _, second = dn.func.run(
        program, first["continuation"], duration=DT, inputs=packet(2)
    )
    path = directory / "network-continuation.pt"
    torch.save(program.export_continuation(second["continuation"]), path)
    receiver = compile_program(network(), callbacks=configuration)
    restored = receiver.restore_continuation(torch.load(path, weights_only=True))
    _, third = dn.func.run(receiver, restored, steps=2, inputs=packet(3))
    continuation = third["continuation"]

    target = network()
    receiving_parameter = target.left.mech.leak.g_param
    target_callbacks = dn.func.make_callbacks(target, configuration, dt=DT)
    carry = target_callbacks.restore_state(
        receiver.callbacks.export_state(continuation.callback_state)
    )
    dn.func.commit_continuation_(receiver, target, continuation)
    assert target.left.mech.leak.g_param is receiving_parameter
    assert target.right.mech.leak.g_param is receiving_parameter
    suffix = target.longrun(
        duration=1.6 * DT,
        chunklength=2,
        inputs=packet(4),
        callbacks=target_callbacks,
        callback_state=carry,
    )
    # endregion network

    with torch.no_grad():
        expected, auxiliary = dn.func.run(
            receiver, continuation, duration=1.6 * DT, inputs=packet(4)
        )
    for name, owner in target.populations.items():
        torch.testing.assert_close(
            owner.v, expected["populations"][name]["integrator"]["v"]
        )
        assert owner.mech.paired.g_param is receiving_parameter
        torch.testing.assert_close(
            suffix["trace"][f"{name}.v"], auxiliary["callbacks"]["trace"][f"{name}.v"]
        )
    results = (
        prefix,
        first["callbacks"],
        second["callbacks"],
        third["callbacks"],
        suffix,
    )
    times = torch.cat(tuple(result["trace"]["t"] for result in results))
    torch.testing.assert_close(
        times, torch.arange(0, 11, 2, dtype=target.dtype(), device=target.device()) * DT
    )
    restored_parameter = restored.parameters["populations"]["left"][G]
    assert restored_parameter is restored.parameters["populations"]["right"][G]
    saved_prefix_voltage = continuation.state["populations"]["left"]["integrator"]["v"]
    gradients = torch.autograd.grad(
        sum(owner.v.square().mean() for owner in target.populations.values()),
        (receiving_parameter, *drives[4], restored_parameter, saved_prefix_voltage),
        allow_unused=True,
    )
    assert all(
        value.isfinite().all() and value.abs().sum() > 0 for value in gradients[:3]
    )
    assert gradients[3:] == (None, None)
    print(
        "Connected Network: portable continuation, native handoff and gradients passed"
    )


def main():
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False),
        TemporaryDirectory() as directory,
    ):
        demonstrate(Path(directory))
        demonstrate_network(Path(directory))


if __name__ == "__main__":
    main()
