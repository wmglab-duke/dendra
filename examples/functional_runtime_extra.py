#!/usr/bin/env python3
"""Choose extracellular contacts per compiled call and differentiate their inputs.

Run from the repository root::

    python examples/functional_runtime_extra.py
    python examples/functional_runtime_extra.py --backend aot_eager
    python examples/functional_runtime_extra.py --backend inductor

The CPU example uses a short passive cable. It compares continued
checkpointed training with one uninterrupted run, including caller-owned field
and Waveform gradients. The default eager backend exercises compiler capture
without native kernel generation.
"""

from __future__ import annotations

import argparse

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.01
STEPS = 8
CHUNK_STEPS = 4


def make_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        model = dn.Unmyelinated(
            [2.0],
            L=3.0,
            dx=1.0,
            v_init=-65.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-4, e=-65.0)
        model.train()
        model.initialize()
        model.t.fill_(0.175)
    return model


def make_runner(model, *, steps=STEPS, backend="eager"):
    # A runtime slot keeps the contact recipe outside the compiled model.
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        extra=dn.VarInput(),
        chunk_steps=CHUNK_STEPS,
        inference=False,
        checkpoint_every=CHUNK_STEPS,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=backend,
    )


def loss(voltage):
    weights = torch.linspace(0.7, 1.3, voltage.shape[-1], dtype=voltage.dtype)
    return (voltage * weights).square().mean() + 0.1 * voltage.sin().mean()


def run_demo(*, backend="eager"):
    model = make_model()
    source_voltage = model.v.detach().clone()
    source_time = model.t.detach().clone()
    # Toy unit-current field [mV/mA] and waveform amplitudes [mA]. Runtime
    # inputs retain their caller-owned gradients; no trainable selector is needed.
    field = torch.linspace(-1.25, 1.75, model.v.numel(), dtype=model.dtype())
    field = field.reshape(model.shape).requires_grad_()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        sine = dn.sin(amp=0.18, freq=0.9, phase=0.2)
        pulse = dn.mono_rect(amp=0.3, delay=0.19, pw=0.025)
        cosine = dn.cos(amp=-0.11, freq=0.55, phase=-0.15)
    saved_field = field.detach().clone()
    saved_waveform = {
        name: value.detach().clone() for name, value in sine.named_parameters()
    }
    one_contact = (field, sine)
    two_contacts = [(field, pulse), (field.flip(-1), cosine)]

    runner = make_runner(model, backend=backend)
    # Each call without state restarts from the same snapshot. The same runner
    # accepts another Waveform type, another contact count, and explicit zero.
    prefix, first = runner(extra=one_contact)
    changed, _ = runner(extra=two_contacts)
    zero, _ = runner(extra=None)
    assert not torch.equal(changed["integrator"]["v"], prefix["integrator"]["v"])
    assert not torch.equal(zero["integrator"]["v"], prefix["integrator"]["v"])
    assert not any(name.startswith("stimulation.extra.") for name in runner.parameters)

    # Carry simulation and callback state explicitly. Waveform time follows
    # the returned simulation clock, and the recording omits a second boundary.
    final, second = runner(
        prefix,
        extra=one_contact,
        callback_state=first["callbacks"].state,
    )
    trace = {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in second["callbacks"]["trace"].items()
    }
    assert trace["v"].shape == (2 * STEPS + 1, *model.shape)
    torch.testing.assert_close(final["clock"]["t"], source_time + 2 * STEPS * DT)

    uninterrupted = make_runner(model, steps=2 * STEPS, backend=backend)
    expected, whole = uninterrupted(extra=one_contact)
    for name, values in trace.items():
        torch.testing.assert_close(values, whole["callbacks"]["trace"][name])
    torch.testing.assert_close(final["integrator"]["v"], expected["integrator"]["v"])

    # Checkpoint replay retains the complete gradient through the prefix and
    # continuation. Both runs differentiate the original field and Waveform.
    targets = (field, sine.amp, sine.freq, sine.phase)
    gradients = torch.autograd.grad(loss(trace["v"]), targets)
    expected_gradients = torch.autograd.grad(
        loss(whole["callbacks"]["trace"]["v"]), targets
    )
    for actual, reference in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual).all() and actual.abs().sum().item() > 0
        torch.testing.assert_close(actual, reference, rtol=2e-8, atol=2e-10)

    torch.testing.assert_close(model.v, source_voltage, rtol=0, atol=0)
    torch.testing.assert_close(model.t, source_time, rtol=0, atol=0)
    torch.testing.assert_close(field, saved_field, rtol=0, atol=0)
    for name, value in sine.named_parameters():
        torch.testing.assert_close(value, saved_waveform[name], rtol=0, atol=0)
    assert field.grad is None
    assert all(value.grad is None for value in sine.parameters())

    print("same runner accepted contact counts: 1, 2, 0")
    print(f"continued recording: {trace['v'].shape[0]} frames, no duplicate boundary")
    print("field/amplitude/frequency/phase gradient norms:")
    print(" ".join(f"{value.norm().item():.6g}" for value in gradients))
    print(f"execution strategy: {runner.execution_report().strategy}")
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("eager", "aot_eager", "inductor"), default="eager"
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    run_demo(backend=arguments.backend)


if __name__ == "__main__":
    main()
