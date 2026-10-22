"""Whole-model injection transforms and compiled shared-waveform execution."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_mechanism_injection_execution import (
    DT,
    _close,
    _InjectedLeak,
    _NamedInjectedLeak,
    _network,
    _population,
)
from tests.func.test_waveform_purity import PURE_WAVEFORM_FACTORIES


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


@pytest.mark.cpu
@pytest.mark.parametrize("owner", (_InjectedLeak, _NamedInjectedLeak))
def test_population_injection_jacobians_jvp_and_vmap(owner):
    functional, tensors = dn.func.make_functional(_population(owner), dt=DT)
    amplitude = tensors.parameter_name("*injected_waveforms.0.amp")
    scale = next(
        name for name in tensors.constants if name.endswith("_injection_scale_0")
    )

    def evaluate(amp, strength, voltage, time, dt):
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": voltage},
            "clock": {**tensors.state["clock"], "t": time},
        }
        final, _ = functional.prepare_and_rollout(
            {**tensors.parameters, amplitude: amp},
            {**tensors.constants, scale: strength, "dt": dt},
            state,
            steps=4,
        )
        return final["integrator"]["v"].flatten()

    inputs = (
        tensors.parameters[amplitude].detach(),
        tensors.constants[scale],
        tensors.state["integrator"]["v"],
        tensors.state["clock"]["t"] + 0.3,
        tensors.constants["dt"],
    )
    argnums = tuple(range(len(inputs)))
    reverse = torch.func.jacrev(evaluate, argnums=argnums)(*inputs)
    forward = torch.func.jacfwd(evaluate, argnums=argnums)(*inputs)
    _close(reverse, forward)
    assert all(value.abs().sum() > 0 for value in reverse)
    _, tangent = torch.func.jvp(
        evaluate, inputs, tuple(torch.ones_like(value) for value in inputs)
    )
    _close(tangent, sum(value.reshape(value.shape[0], -1).sum(1) for value in reverse))
    batched = tuple(torch.stack((value, value + 0.01)) for value in inputs)
    expected = torch.stack(
        [evaluate(*(value[index] for value in batched)) for index in range(2)]
    )
    _close(torch.vmap(evaluate)(*batched), expected)
    # Empty state lanes work with a shared waveform. Mapping scalar waveform
    # operands against a time vector retains PyTorch's scalar-promotion vmap
    # limitation, also present when evaluating the native Waveform directly.
    empty = torch.vmap(
        lambda voltage: evaluate(inputs[0], inputs[1], voltage, inputs[3], inputs[4])
    )(batched[2][:0])
    assert empty.shape == (0, expected.shape[1])


@pytest.mark.cpu
@pytest.mark.parametrize("waveform_factory", PURE_WAVEFORM_FACTORIES)
def test_owned_builtin_waveforms_preserve_native_rollout(waveform_factory):
    source = _population(waveform=waveform_factory())
    native = _population(waveform=waveform_factory())
    functional, tensors = dn.func.make_functional(source, dt=DT)
    final, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=4
    )
    for _ in range(4):
        native.step(DT)
    _close(final, functional.extract(native).state)


@pytest.mark.cpu
@pytest.mark.parametrize("owner", (_InjectedLeak, _NamedInjectedLeak))
def test_inductor_shared_waveform_network_checkpointed_gradients(owner):
    source, native = _network(owner, shared=True), _network(owner, shared=True)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=False,
        backend="inductor",
        checkpoint_every=2,
        trainable=("*injected_waveforms.*.amp", "*gain_param"),
    )
    amplitude = runner.get_parameter("*left*injected_waveforms.0.amp")
    assert amplitude is runner.get_parameter("*right*injected_waveforms.0.amp")
    for population in native.populations.values():
        population.mech.owner.populate()
    final, _ = runner()
    for _ in range(4):
        native.step()
    _close(final, runner.functional.extract(native).state)
    observed = torch.autograd.grad(
        sum(
            state["integrator"]["v"].square().sum()
            for state in final["populations"].values()
        ),
        (
            amplitude,
            runner.get_parameter("*left*gain_param"),
            runner.get_parameter("*right*gain_param"),
        ),
    )
    expected = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(),
        (
            native.left.mech.owner.injected_waveforms[0].amp,
            native.left.mech.owner.gain_param,
            native.right.mech.owner.gain_param,
        ),
    )
    assert all(value.abs().sum() > 0 for value in expected)
    _close(observed, expected)
    assert runner.execution_report().strategy == "compiled_scan"


@pytest.mark.cpu
@pytest.mark.parametrize("owner", (_InjectedLeak, _NamedInjectedLeak))
def test_inductor_owned_injections_while_recording_and_continuation(owner):
    source, native = _network(owner, shared=True), _network(owner, shared=True)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=True,
        execution="while",
        backend="inductor",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = [native.right.v.clone()]
    with torch.no_grad():
        for _ in range(8):
            native.step()
            expected.append(native.right.v.clone())
    _close(final, runner.functional.extract(native).state)
    _close(
        torch.cat(
            (
                first["callbacks"]["trace"]["right.v"],
                second["callbacks"]["trace"]["right.v"],
            )
        ),
        torch.stack(expected),
    )
    assert runner.execution_report().strategy == "compiled_while_loop"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("owner", (_InjectedLeak, _NamedInjectedLeak))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_owned_injection_continuation(owner, dtype):
    with dn.ctx(DEVICE="cuda", DTYPE=dtype):
        source, native = _population(owner), _population(owner)
    runner = dn.compile(source, dt=DT, steps=4, backend="aot_eager")
    prefix, _ = runner()
    final, _ = runner(prefix)
    with torch.no_grad():
        for _ in range(8):
            native.step(DT)
    tolerance = 5e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(
        final, runner.functional.extract(native).state, rtol=tolerance, atol=tolerance
    )
