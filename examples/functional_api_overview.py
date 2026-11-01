#!/usr/bin/env python3
"""Exercise the main functional workflows using ordinary Dendra models.

Run from the repository root::

    python examples/functional_api_overview.py
    python examples/functional_api_overview.py --section explicit --backend aot_eager

Sections are inference, training, explicit, draws, extra and transforms. The short CPU
examples check API contracts; they are not performance or convergence studies.
The eager compiler backend exercises capture without native code generation.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import expsyn, graded_syn, pas
from dendra.units import nA

DT, STEPS, WIDTH = 0.025, 4, 2


def copy_state(state):
    """Make an independent checkpoint, intentionally breaking its old graph."""
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), state
    )


def window(tape, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], tape)


def joined_trace(first, second):
    return {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in second["callbacks"]["trace"].items()
    }


def assert_gradients(gradients):
    assert gradients
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


# region cable
def make_cable(*, injected=False):
    model = dn.Unmyelinated(
        [2.0],
        L=3.0,
        dx=1.0,
        v_init=-60.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-70.0)
    if injected:
        model[:, 1].inject(dn.sin(amp=0.2 * nA, freq=0.7, phase=0.2))
    model.train()
    model.initialize()
    return model


# endregion cable


# region inference
def inference_demo(*, backend):
    model = make_cable(injected=True)
    original_voltage = model.v.detach().clone()
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=backend,
    )
    final, first = runner()
    repeated, _ = runner()  # No state means restart from the captured snapshot.
    torch.testing.assert_close(final, repeated)
    assert first["callbacks"]["trace"]["v"].shape[0] == STEPS + 1
    assert not first["callbacks"]["trace"]["v"].requires_grad

    continued, second = runner(first["continuation"])
    whole = dn.compile(
        model,
        dt=DT,
        steps=2 * STEPS,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=backend,
    )
    expected, expected_aux = whole()
    torch.testing.assert_close(continued, expected)
    torch.testing.assert_close(
        joined_trace(first, second), expected_aux["callbacks"]["trace"]
    )
    torch.testing.assert_close(model.v, original_voltage)
    print("inference: registered injection, restart, recording and continuation passed")
    print("execution strategy:", runner.execution_report().strategy)


# endregion inference


def release(voltage):
    """An ordinary deterministic transform used by continuous connectivity."""
    return torch.sigmoid(0.1 * (voltage + 60.0))


# region network
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
    # Two named aliases deliberately share one trainable raw source.
    right.mech.leak.g_param = left.mech.leak.g_param
    network.train()
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    return network


# endregion network


# region training
def training_demo(*, backend):
    model = make_network()
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=(
            "*.g_param",
            "synapses.*.weight.weights.*.param.rho",
            "continuous_synapses.*.weight.weights.*.param.rho",
        ),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        checkpoint_every=2,
        backend=backend,
    )
    shared = runner.find_parameters("*.g_param")
    assert len(shared) == 2 and len({id(value) for value in shared.values()}) == 1
    for name, value in runner.trainable_parameters.items():
        print(name, tuple(value.shape), value.dtype, value.requires_grad)
    unique = {id(value): value for value in runner.trainable_parameters.values()}
    optimizer = torch.optim.SGD(list(unique.values()), lr=1e-8)
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, auxiliary = runner()
        trace = auxiliary["callbacks"]["trace"]
        loss = trace["right.v"].square().mean() + trace["left.v"].square().mean()
        loss.backward()
        assert_gradients([value.grad for value in unique.values()])
        optimizer.step()
        print(f"training iteration {iteration}: loss={loss.item():.8g}")
    print(
        "training: connected event/continuous Network, shared sources and gradients passed"
    )


# endregion training


# region explicit
def explicit_demo(*, backend):
    model = make_cable()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*.g_param")
    initial = copy_state(tensors.state)
    voltage = initial["integrator"]["v"]
    current = voltage.new_full((STEPS, *voltage.shape), 0.02 * nA).requires_grad_()
    inputs = dn.func.Inputs(intra=current)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    kernel = functional.compile_rollout_chunk(WIDTH, backend=backend)
    bound = kernel.bind(tensors._replace(parameters=parameters))

    # Direct calls consume exactly WIDTH time-leading input frames.
    prefix, first = bound(initial, inputs=window(inputs, 0, WIDTH), callbacks=callbacks)
    final, second = bound(
        prefix,
        inputs=window(inputs, WIDTH, STEPS),
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    # The host scheduler handles arbitrary horizons, tails and checkpoint spans.
    expected, whole = dn.func.run(
        bound,
        initial,
        inputs=inputs,
        steps=STEPS,
        checkpoint_every=3,
        callbacks=callbacks,
    )
    torch.testing.assert_close(final, expected)
    trace = joined_trace(first, second)
    torch.testing.assert_close(trace, whole["callbacks"]["trace"])
    assert trace["v"].shape[0] == STEPS + 1
    targets = tuple(
        {
            id(value): value
            for value in pytree.tree_leaves(parameters)
            if value.requires_grad
        }.values()
    ) + (current,)
    actual_gradients = torch.autograd.grad(trace["v"].square().mean(), targets)
    expected_gradients = torch.autograd.grad(
        whole["callbacks"]["trace"]["v"].square().mean(), targets
    )
    assert_gradients(actual_gradients)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    print(
        "explicit: raw-bound direct callbacks, input gradients and checkpointed run agree"
    )


# endregion explicit


# region draws
def draws_demo(*, backend):
    model = make_network()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initial = copy_state(tensors.state)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    )
    kernel = functional.compile_rollout_chunk(WIDTH, backend=backend)
    bound = kernel.bind(tensors)
    # Sampling consumes explicit carry, leaving the native Network/RNG unchanged.
    draws = functional.sample_runtime_draws(initial, steps=STEPS)
    assert isinstance(draws, dn.func.RuntimeDraws)
    with torch.no_grad():
        prefix, first = bound(
            initial, draws=window(draws, 0, WIDTH), callbacks=callbacks
        )
        final, second = bound(
            prefix,
            draws=window(draws, WIDTH, STEPS),
            callbacks=callbacks,
            callback_state=first["callbacks"].state,
        )
        expected, whole = dn.func.run(
            bound, initial, draws=draws, steps=STEPS, callbacks=callbacks
        )
        automatic, automatic_aux = dn.func.run(
            bound, initial, steps=STEPS, host_span_steps=3, callbacks=callbacks
        )
        torch.testing.assert_close(final, expected)
        torch.testing.assert_close(automatic, expected)
        torch.testing.assert_close(
            joined_trace(first, second), whole["callbacks"]["trace"]
        )
        torch.testing.assert_close(
            automatic_aux["callbacks"]["trace"], whole["callbacks"]["trace"]
        )
        torch.testing.assert_close(
            functional.sample_runtime_draws(prefix, steps=WIDTH),
            window(draws, WIDTH, STEPS),
        )
        torch.testing.assert_close(functional.extract().state, initial)
    print(
        "draws: connected Network explicit replay, automatic sampling and RNG continuation agree"
    )


# endregion draws


# region extra
def extra_demo(*, backend):
    model = make_cable()
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        extra=dn.VarInput(),
        chunk_steps=WIDTH,
        inference=False,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=backend,
    )
    field = torch.linspace(-1.0, 1.0, model.v.numel(), dtype=model.dtype())
    field = field.reshape(model.shape).requires_grad_()  # Toy field [mV/mA].
    sine = dn.sin(amp=0.1, freq=0.7, phase=0.2)  # Current amplitude [mA].
    pulse = dn.mono_rect(amp=0.2, pw=2 * STEPS * DT)
    one_contact = (field, sine)
    two_contacts = [(field, pulse), (field.flip(-1), sine)]
    first_state, first = runner(extra=one_contact)
    changed, _ = runner(extra=two_contacts)
    zero, _ = runner(extra=None)
    assert not torch.equal(changed["integrator"]["v"], first_state["integrator"]["v"])
    assert not torch.equal(zero["integrator"]["v"], first_state["integrator"]["v"])
    final, second = runner(first["continuation"], extra=one_contact)
    whole_runner = dn.compile(
        model,
        dt=DT,
        steps=2 * STEPS,
        extra=dn.VarInput(),
        chunk_steps=WIDTH,
        inference=False,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=backend,
    )
    expected, whole = whole_runner(extra=one_contact)
    torch.testing.assert_close(final, expected)
    trace = joined_trace(first, second)
    torch.testing.assert_close(trace, whole["callbacks"]["trace"])
    targets = (field, sine.amp, sine.freq, sine.phase)
    actual_gradients = torch.autograd.grad(trace["v"].square().mean(), targets)
    expected_gradients = torch.autograd.grad(
        whole["callbacks"]["trace"]["v"].square().mean(), targets
    )
    assert_gradients(actual_gradients)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    print(
        "extra: different deterministic types/contact counts, gradients and continuation agree"
    )


# endregion extra


# region transforms
def transforms_demo(*, backend):
    del backend  # This section deliberately uses pure eager execution.
    model = dn.SingleCompartment(
        N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas, g=0.001, e=-70.0)
    model.train()
    model.initialize()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initial = copy_state(tensors.state)
    tape = initial["integrator"]["v"].new_full((STEPS, *model.shape), 0.02 * nA)

    def objective(amplitude):
        final, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            initial,
            dn.func.Inputs(intra=amplitude * tape),
            steps=STEPS,
        )
        return final["integrator"]["v"].square().mean()

    amplitude = tape.new_tensor(1.0)
    derivative = torch.func.grad(objective)(amplitude)
    ordinary = amplitude.detach().clone().requires_grad_()
    reference = torch.autograd.grad(objective(ordinary), ordinary)[0]
    torch.testing.assert_close(derivative, reference)
    assert_gradients((derivative,))
    amplitudes = tape.new_tensor([0.5, 1.0, 1.5])
    batched = torch.func.vmap(objective)(amplitudes)
    separate = torch.stack([objective(value) for value in amplitudes])
    torch.testing.assert_close(batched, separate)
    assert torch.isfinite(batched).all()
    print("transforms: pure eager grad/autograd and vmap/individual rollouts agree")


# endregion transforms


def main():
    sections = {
        "inference": inference_demo,
        "training": training_demo,
        "explicit": explicit_demo,
        "draws": draws_demo,
        "extra": extra_demo,
        "transforms": transforms_demo,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("all", *sections), default="all")
    parser.add_argument(
        "--backend", choices=("eager", "aot_eager", "inductor"), default="eager"
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    selected = (
        sections.values()
        if arguments.section == "all"
        else (sections[arguments.section],)
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for section in selected:
            section(backend=arguments.backend)


if __name__ == "__main__":
    main()
