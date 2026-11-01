#!/usr/bin/env python3
"""Continue current native Populations and Networks through functional execution.

Run from the repository root::

    python examples/functional_continuation_capture.py

The bounded CPU example covers lowered plans and compiled programs, shared
sources, independent suffix gradients, callback boundaries and portable replay.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT, PREFIX_STEPS, SUFFIX_STEPS = 0.025, 2, 3
G = "integrator.mech.mechanisms.leak.g_param"


def make_model(network):
    def cell(reversal):
        population = dn.SingleCompartment(
            N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas.rename("leak"), g=0.001, e=reversal)
        return population

    model = cell(-45.0)
    if network:
        model = dn.Network({"left": model, "right": cell(-55.0)})
        model.right.mech.leak.g_param = model.left.mech.leak.g_param
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def advance(model, network, steps):
    if network:
        model.run(steps=steps)
    else:
        model.run(steps=steps, dt=DT)


def conductance(parameters, network):
    return parameters["populations"]["left"][G] if network else parameters[G]


def voltage(state, network):
    if network:
        return torch.cat(
            [
                part["integrator"]["v"].flatten()
                for part in state["populations"].values()
            ]
        )
    return state["integrator"]["v"]


def demonstrate(network, compiled):
    native = make_model(network)
    # A native run configures the Population integrator at the capture timestep.
    advance(native, network, PREFIX_STEPS)
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    configurations = {"trace": dn.Recorder(paths)}
    if compiled:
        target = dn.compile(
            native,
            dt=DT,
            steps=SUFFIX_STEPS,
            inference=False,
            callbacks=configurations,
            backend="eager",
            disable=True,
        )
        callbacks = None
        # Capture reads native sources, including their trainability. Edited
        # program defaults and its trainable selection do not replace them.
        with torch.no_grad():
            conductance(target.parameters, network).mul_(1.5)
    else:
        target, _ = dn.func.make_functional(native, dt=DT)
        callbacks = target.make_callbacks(configurations)
    advance(native, network, PREFIX_STEPS)
    boundary_time = native.t.detach().clone()
    original = native.left.mech.leak.g_param if network else native.mech.leak.g_param

    # region capture
    continuation = dn.func.capture_continuation(target)
    # An explicit compatible native model is also accepted.
    explicit = dn.func.capture_continuation(target, native_model=native)
    torch.testing.assert_close(continuation.state, explicit.state)
    captured = conductance(continuation.parameters, network)
    torch.testing.assert_close(captured, original)
    assert captured is not original and captured.data_ptr() != original.data_ptr()
    assert captured.is_leaf and captured.requires_grad == original.requires_grad
    assert continuation.callback_state is None
    if network:
        assert captured is continuation.parameters["populations"]["right"][G]
    # endregion capture

    # region checkpoint
    saved = dn.func.export_continuation(target, continuation, callbacks=callbacks)
    restored = dn.func.restore_continuation(target, saved, callbacks=callbacks)
    state, auxiliary = dn.func.run(
        target, restored, steps=SUFFIX_STEPS, callbacks=callbacks
    )
    # Fresh callbacks include the capture boundary as their initial sample.
    trace = auxiliary["callbacks"]["trace"]
    torch.testing.assert_close(trace["t"][0], boundary_time.reshape(()))
    assert trace["t"].shape[0] == SUFFIX_STEPS + 1
    # endregion checkpoint

    advance(native, network, SUFFIX_STEPS)
    expected = (
        torch.cat([part.v.flatten() for part in native.populations.values()])
        if network
        else native.v
    )
    torch.testing.assert_close(voltage(state, network), expected)
    restored_source = conductance(restored.parameters, network)
    gradient, prefix_gradient = torch.autograd.grad(
        voltage(state, network).square().sum(),
        (restored_source, original),
        allow_unused=True,
    )
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    assert prefix_gradient is None
    print(
        f"{'Network' if network else 'Population'} "
        f"{'program' if compiled else 'plan'}: current native capture, "
        "shared sources, callback boundary, checkpoint replay and suffix gradients passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            for compiled in (False, True):
                demonstrate(network, compiled)


if __name__ == "__main__":
    main()
