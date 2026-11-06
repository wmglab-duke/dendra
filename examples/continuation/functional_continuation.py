#!/usr/bin/env python3
"""Continue Population/Network runners and explicit raw bindings.

Run from the repository root::

    python examples/continuation/functional_continuation.py
    python examples/continuation/functional_continuation.py --backend aot_eager

The short CPU examples check ownership, recording and gradients, not speed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT, STEPS = 0.025, 2


def cell(*, initial=-60.0, reversal=-70.0):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=initial, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=reversal)
    return model


def make_population():
    model = cell()
    model.train()
    model.initialize()
    return model


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def make_network():
    left, right = cell(reversal=-55.0), cell(initial=-65.0)
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
    model.train()
    model.initialize(DT)
    return model


def containers(tree, leaf=lambda value: value):
    """Copy containers and retain tensor leaves, including shared sources."""
    if isinstance(tree, Mapping):
        return {name: containers(value, leaf) for name, value in tree.items()}
    if isinstance(tree, tuple):
        values = tuple(containers(value, leaf) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    if isinstance(tree, list):
        return [containers(value, leaf) for value in tree]
    return leaf(tree)


def gain_parameters(program, gain, source):
    replacement = source * gain
    return containers(
        program.parameters,
        lambda value: replacement if value is source else value,
    )


def compile_program(model, *, network, backend, initialization="snapshot"):
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    return dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        inference=False,
        initialization=initialization,
        checkpoint_every=1,
        trainable="*g_param",
        callbacks={"trace": dn.func.Recorder(paths)},
        backend=backend,
    )


def trace_loss(trace):
    return sum(value.square().mean() for name, value in trace.items() if name != "t")


def run_model(model, *, network, backend):
    program = compile_program(model, network=network, backend=backend)
    source = next(iter(program.trainable_parameters.values()))
    gain = source.new_tensor(1.2, requires_grad=True)
    parameters = gain_parameters(program, gain, source)

    # region continuation
    prefix, first = program(parameters=parameters)
    continuation = first["continuation"]
    final, last = program(continuation)
    # endregion continuation
    assert isinstance(continuation, dn.func.Continuation)

    # The packet captures the supplied override, not the runner's defaults.
    selected = (
        continuation.parameters["populations"]["left"]
        if network
        else continuation.parameters
    )
    name = next(name for name in selected if name.endswith("g_param"))
    overridden = parameters["populations"]["left"] if network else parameters
    assert selected[name] is overridden[name] and selected[name] is not source
    if network:
        assert selected[name] is continuation.parameters["populations"]["right"][name]

    # region binding_bridge
    bound = program.kernel.bind(
        continuation.parameters, constants=continuation.constants
    )
    explicit, bound_aux = dn.func.run(
        bound,
        continuation,
        steps=STEPS,
        callbacks=program.callbacks,
        checkpoint_every=1,
    )
    # endregion binding_bridge
    torch.testing.assert_close(containers(final), containers(explicit))
    torch.testing.assert_close(
        dict(last["callbacks"]["trace"]), dict(bound_aux["callbacks"]["trace"])
    )

    # region binding_resume
    same_bound, continued_aux = dn.func.run(
        bound,
        bound_aux["continuation"],
        steps=STEPS,
        callbacks=program.callbacks,
        checkpoint_every=1,
    )
    # endregion binding_resume
    next_state, next_aux = program(bound_aux["continuation"])
    torch.testing.assert_close(containers(same_bound), containers(next_state))
    torch.testing.assert_close(
        dict(continued_aux["callbacks"]["trace"]),
        dict(next_aux["callbacks"]["trace"]),
    )
    assert continued_aux["continuation"].callback_state.plan_token is (
        continuation.callback_state.plan_token
    )

    reference_gain = gain.detach().clone().requires_grad_()
    reference_parameters = gain_parameters(program, reference_gain, source)
    expected, whole = program.functional.prepare_and_rollout(
        reference_parameters,
        program.constants,
        program.initial_state,
        steps=2 * STEPS,
        callbacks=program.callbacks,
    )
    torch.testing.assert_close(containers(final), containers(expected))
    trace = {
        name: torch.cat((first["callbacks"]["trace"][name], suffix))
        for name, suffix in bound_aux["callbacks"]["trace"].items()
    }
    torch.testing.assert_close(trace, dict(whole["callbacks"]["trace"]))
    for name, values in trace.items():
        assert values.shape[0] == 2 * STEPS + 1
        assert bound_aux["callbacks"]["trace"][name].shape[0] == STEPS
    gradient = torch.autograd.grad(trace_loss(trace), gain)[0]
    reference_gradient = torch.autograd.grad(
        trace_loss(whole["callbacks"]["trace"]), reference_gain
    )[0]
    torch.testing.assert_close(gradient, reference_gradient)
    assert torch.isfinite(gradient) and gradient.abs() > 0
    torch.testing.assert_close(
        program.functional.extract().state, program.initial_state
    )

    # region restored_binding
    receiver = compile_program(
        make_network() if network else make_population(),
        network=network,
        backend=backend,
    )
    restored = receiver.restore_continuation(program.export_continuation(continuation))
    restored_bound = receiver.kernel.bind(
        restored.parameters, constants=restored.constants
    )
    restored_state, restored_aux = dn.func.run(
        restored_bound,
        restored,
        steps=STEPS,
        callbacks=receiver.callbacks,
        checkpoint_every=1,
    )
    # endregion restored_binding
    torch.testing.assert_close(containers(restored_state), containers(final))
    torch.testing.assert_close(
        dict(restored_aux["callbacks"]["trace"]),
        dict(bound_aux["callbacks"]["trace"]),
    )

    # region fresh
    fresh = compile_program(
        model, network=network, backend=backend, initialization="fresh"
    )
    fresh_state, fresh_aux = fresh()
    carry = fresh_aux["continuation"]
    resumed, _ = fresh(carry)
    manual, _ = fresh(
        fresh_state,
        parameters=fresh_aux["parameters"],
        constants=fresh_aux["constants"],
        callback_state=fresh_aux["callbacks"].state,
    )
    # endregion fresh
    torch.testing.assert_close(containers(resumed), containers(manual))
    torch.testing.assert_close(containers(carry.state), containers(fresh_state))
    for frame, saved in (
        (fresh_aux["parameters"], carry.parameters),
        (fresh_aux["constants"], carry.constants),
    ):

        def check_identity(value):
            return id(value) if isinstance(value, torch.Tensor) else value

        assert containers(frame, check_identity) == containers(saved, check_identity)
    print(
        f"{'Network' if network else 'Population'}: effective overrides, aliases, "
        "bound continuation, checkpointed gradients and explicit restored bindings passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="eager")
    args = parser.parse_args()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        run_model(make_population(), network=False, backend=args.backend)
        run_model(make_network(), network=True, backend=args.backend)


if __name__ == "__main__":
    main()
