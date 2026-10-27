"""Deterministic runtime extra preserves a stochastic model's shared draw order."""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_poisson_stimulation import DT, DTYPE, _clone, _close
from tests.func.test_stimulation_rng_sharing import (
    _draw_input,
    _one_stream,
    _population,
    _slice,
)

STEPS = 8
CHUNK_STEPS = 4
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=1,
            IMEM=False,
            DTYPE=DTYPE,
            DEVICE="cpu",
            CHECK_PURITY=False,
        ),
    ):
        yield


def _voltage_tape(state, field, waveform, dt, steps):
    # Independent deterministic sine formula. Recurrent additions follow the
    # accepted clock; the model's registered stochastic inputs use other tapes.
    time = state["clock"]["t"]
    frames = []
    for _ in range(steps):
        value = waveform.amp * torch.sin(
            2 * math.pi * waveform.freq * time.to(field) + waveform.phase
        )
        frames.append(field * value)
        time = time + dt.to(time)
    return torch.stack(frames)


def _reference(runner, inputs, steps):
    functional = runner.functional
    prepared = functional.prepare(runner.parameters, runner.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    return dn.func.run(
        functional.bind(runner.parameters, prepared),
        runner.initial_state,
        steps=steps,
        inputs=inputs,
        callbacks=callbacks,
    )


def _trace(result):
    return dict(result[1]["callbacks"]["trace"])


def _exercise(execution):
    # The existing fixture shares one stream across registered NOISE, State
    # SDE draws, and repeated Poisson evaluations. Runtime extra stays pure.
    model, _bound_extra, pulse = _population()
    inference = execution == "while"
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        extra=dn.VarInput(),
        chunk_steps=CHUNK_STEPS,
        execution=execution,
        inference=inference,
        checkpoint_every=None if inference else CHUNK_STEPS,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="aot_eager",
    )
    functional = runner.functional
    assert functional._stimulation_rng_plan.enabled
    field = torch.linspace(-1.25, 1.75, model.v.numel(), dtype=DTYPE)
    field = field.reshape(model.shape).requires_grad_()
    waveform = dn.sin(amp=0.18, freq=0.9, phase=0.2)
    extra = field, waveform
    original = _clone(functional.extract())
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()

    # Both explicit tapes originate from the same initial carry, retaining the
    # interleaved stream order rather than sampling the two families in series.
    draws = _draw_input(functional, runner.initial_state, 2 * STEPS)
    assert draws.noise is not None and draws.stimulation is not None
    frozen_draws = _clone(draws)
    inputs = draws._replace(
        ve=_voltage_tape(
            runner.initial_state, field, waveform, runner.constants["dt"], 2 * STEPS
        )
    )
    context = torch.no_grad() if inference else torch.enable_grad()
    with context:
        expected_prefix = _reference(runner, _slice(inputs, 0, STEPS), STEPS)
        expected_full = _reference(runner, inputs, 2 * STEPS)

    prefix, first = runner(extra=extra)
    _close(prefix, expected_prefix[0])
    _close(dict(first["callbacks"]["trace"]), _trace(expected_prefix))
    if inference:
        # Repeating while preflight and its warm execution must not consume
        # either the authored shared generator or the global generator.
        repeated = runner(extra=extra)
        _close(repeated[0], prefix)
        _close(_trace(repeated), _trace(expected_prefix))
        _close(pulse.generator.get_state(), source_rng)
        _close(torch.get_rng_state(), global_rng)

    suffix_draws = _draw_input(functional, prefix, STEPS)
    _close(suffix_draws, _slice(draws, STEPS, 2 * STEPS))
    final, second = runner(
        prefix,
        extra=extra,
        callback_state=first["callbacks"].state,
    )
    joined = {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in second["callbacks"]["trace"].items()
    }
    assert joined["v"].shape[0] == 2 * STEPS + 1
    _close(final, expected_full[0])
    _close(joined, _trace(expected_full))
    _close(_one_stream(final), _one_stream(expected_full[0]))
    assert not torch.equal(_one_stream(final), _one_stream(runner.initial_state))
    assert runner.execution_report().strategy == (
        "compiled_while_loop" if inference else "compiled_scan"
    )

    if inference:
        assert not final["integrator"]["v"].requires_grad
    else:
        # Backward replays the checkpoint spans without resampling the model.
        targets = field, waveform.amp
        gradients = torch.autograd.grad(joined["v"].square().mean(), targets)
        reference_gradients = torch.autograd.grad(
            _trace(expected_full)["v"].square().mean(), targets
        )
        assert all(
            torch.isfinite(value).all() and torch.count_nonzero(value) > 0
            for value in gradients
        )
        _close(gradients, reference_gradients)

    _close(functional.extract(), original)
    _close(draws, frozen_draws)
    _close(pulse.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


def test_runtime_extra_checkpointed_scan_preserves_shared_model_draw_order():
    _exercise("scan")


def test_runtime_extra_while_preflight_and_continuation_preserve_shared_model_draw_order():
    _exercise("while")
