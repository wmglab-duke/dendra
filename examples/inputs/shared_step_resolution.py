#!/usr/bin/env python3
"""Size deterministic input tapes from an explicit execution horizon.

Run from the repository root::

    python examples/inputs/shared_step_resolution.py

The bounded CPU example covers Population and Network input allocation,
fractional time, gradients and native/portable checkpoint continuation.
"""

from __future__ import annotations

import io

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
PREFIX = 0.015
DURATION = 0.015
SUFFIX = 0.02


def make_model(network):
    cells = {}
    for index, name in enumerate(("left", "right") if network else ("cell",)):
        cell = dn.SingleCompartment(
            N=index + 1,
            C=1,
            v_init=-55.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        cell.insert(pas.rename("leak"), g=0.001, e=-70.0)
        cells[name] = cell
    model = dn.Network(cells) if network else cells["cell"]
    model.train()
    model.initialize(DT) if network else model.initialize()
    return model


def clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf.detach().clone(), value)


def assert_equal(actual, expected):
    values, structure = pytree.tree_flatten(actual)
    references, expected_structure = pytree.tree_flatten(expected)
    assert structure == expected_structure
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def make_inputs(state, network, steps):
    frames = state["populations"] if network else {"cell": state}
    inputs, currents = {}, []
    for index, (name, frame) in enumerate(frames.items()):
        voltage = frame["integrator"]["v"]
        current = voltage.new_full((steps, *voltage.shape), (index + 1) * 0.2e-9)
        current.requires_grad_()
        inputs[name] = dn.func.Inputs(intra=current)
        currents.append(current)
    return (inputs if network else inputs["cell"]), tuple(currents)


def voltage_loss(state, network):
    frames = state["populations"].values() if network else (state,)
    return sum(frame["integrator"]["v"].square().mean() for frame in frames)


def demonstrate(network):
    model = make_model(network)
    native_options = {} if network else {"dt": DT}
    assert dn.func.resolve_steps(model, duration=PREFIX, **native_options) == 0
    model.run(duration=PREFIX, **native_options)
    checkpoint = clone(model.state_dict_for_checkpoint())

    program = dn.compile(
        model,
        dt=DT,
        steps=2,
        chunk_steps=2,
        inference=False,
        trainable="*g_param",
        backend="eager",
        disable=True,
    )
    trial = dn.func.capture_continuation(program)
    saved_state = clone(trial.state)
    # region resolution
    request = {"duration": DURATION}
    steps = dn.func.resolve_steps(program, trial, **request)
    # The pending 0.6dt plus this 0.6dt request accepts one complete timestep.
    assert steps == 1
    inputs, currents = make_inputs(trial.state, network, steps)
    state, auxiliary = dn.func.run(program, trial, inputs=inputs, **request)
    # endregion resolution
    assert dn.func.resolve_steps(model, **request, **native_options) == steps
    torch.testing.assert_close(trial.state, saved_state)
    assert_equal(model.state_dict_for_checkpoint(), checkpoint)
    wanted = torch.autograd.grad(voltage_loss(state, network), currents)

    # Keep duration= on execution so the fractional remainder is published.
    model.longrun_checkpointed(
        inputs=inputs,
        chunklength=1,
        safe_checkpoint=True,
        **request,
        **native_options,
    )
    cells = model.populations.values() if network else (model,)
    observed = torch.autograd.grad(
        sum(cell.v.square().mean() for cell in cells), currents
    )
    torch.testing.assert_close(observed, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    torch.testing.assert_close(program.functional.extract(model).state, state)
    torch.testing.assert_close(model.t, model.t.new_tensor(DT))
    pending = model.state_dict_for_checkpoint()["duration_remainder"]
    torch.testing.assert_close(pending, pending.new_tensor(0.2 * DT))

    # Native and portable checkpoints preserve the same incoming time budget.
    buffer = io.BytesIO()
    torch.save(dn.func.export_continuation(program, auxiliary["continuation"]), buffer)
    buffer.seek(0)
    restored = dn.func.restore_continuation(
        program, torch.load(buffer, weights_only=True)
    )
    receiver = make_model(network)
    receiver.restore_dict_from_checkpoint(clone(model.state_dict_for_checkpoint()))
    steps = dn.func.resolve_steps(program, restored, duration=SUFFIX)
    assert steps == 1
    assert dn.func.resolve_steps(receiver, duration=SUFFIX, **native_options) == steps
    suffix_inputs, _ = make_inputs(restored.state, network, steps)
    with torch.no_grad():
        final, _ = dn.func.run(program, restored, duration=SUFFIX, inputs=suffix_inputs)
        receiver.longrun(
            duration=SUFFIX, inputs=suffix_inputs, chunklength=1, **native_options
        )
    torch.testing.assert_close(program.functional.extract(receiver).state, final)
    torch.testing.assert_close(receiver.t, receiver.t.new_tensor(2 * DT))
    assert receiver.state_dict_for_checkpoint()["duration_remainder"].item() == 0.0
    print(
        f"{'Network' if network else 'Population'}: exact input lengths, fractional "
        "carry, gradients and checkpoint continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
