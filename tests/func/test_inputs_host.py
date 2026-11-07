"""Shared drive packets retain host scheduling and legacy callable semantics."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_raw_bindings import DT, _case, _close
from tests.func.test_raw_bindings import _environment as _environment

pytestmark = pytest.mark.cpu


def _drives(shape, steps):
    count = steps * torch.Size(shape).numel()
    ve = torch.linspace(-0.2, 0.3, count).reshape(steps, *shape).requires_grad_()
    intra = (
        torch.linspace(0.8e-10, 1.2e-10, count).reshape(steps, *shape).requires_grad_()
    )
    return dn.func.Inputs(ve=ve, intra=intra)


def _loss(state):
    voltage = state["integrator"]["v"]
    weights = torch.linspace(0.7, 1.3, voltage.shape[-1])
    return (voltage * weights).square().mean()


def _snapshot(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


@pytest.mark.parametrize("compiled", (False, True))
def test_shared_inputs_preserve_checkpointed_callback_continuation_and_all_gradients(
    compiled,
):
    _, functional, parameters, constants, state, name = _case()
    inputs = _drives(functional.shape, 5)
    originals = _snapshot(inputs)
    targets = (parameters[name], inputs.ve, inputs.intra)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    expected, reference_aux = functional.prepare_and_rollout(
        parameters,
        constants,
        state,
        inputs=dn.func.RolloutInput(*inputs),
        steps=5,
        callbacks=callbacks,
    )
    reference = torch.autograd.grad(_loss(expected), targets)
    owner = (
        functional.compile_rollout_chunk(3, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(parameters, constants=constants)
    prefix, first = dn.func.run(
        bound,
        state,
        inputs=dn.func.Inputs(inputs.ve[:3], inputs.intra[:3]),
        steps=3,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    empty, middle = dn.func.run(
        bound,
        prefix,
        inputs=dn.func.Inputs(inputs.ve[:0], inputs.intra[:0]),
        steps=0,
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    final, auxiliary = dn.func.run(
        bound,
        empty,
        inputs=dn.func.Inputs(inputs.ve[3:], inputs.intra[3:]),
        steps=2,
        checkpoint_every=2,
        callbacks=callbacks,
        callback_state=middle["callbacks"].state,
    )
    _close(final, expected)
    actual = torch.autograd.grad(_loss(final), targets)
    _close(actual, reference)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in actual
    )
    for field in ("v", "t"):
        assert middle["callbacks"]["trace"][field].shape[0] == 0
        resumed = torch.cat(
            (first["callbacks"]["trace"][field], auxiliary["callbacks"]["trace"][field])
        )
        assert resumed.shape[0] == 6
        _close(resumed, reference_aux["callbacks"]["trace"][field])
    _close(inputs, originals)


def test_compiled_runner_shared_intra_and_runtime_extra_match_accepted_step_reference():
    model, _, _, _, _, _ = _case()
    runner = dn.compile(
        model,
        dt=DT,
        steps=3,
        inference=False,
        trainable="*g_param",
        extra=dn.VarInput(),
        chunk_steps=2,
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="aot_eager",
    )
    source = _snapshot(model.state_dict())
    intra = _drives(runner.functional.shape, 3).intra
    field = (
        torch.linspace(-0.2, 0.3, model.v.numel()).reshape(model.shape).requires_grad_()
    )
    waveform = dn.sin(amp=0.2, freq=0.8, phase=0.3)
    amplitude = dict(waveform.named_parameters())["amp"]
    conductance = next(
        value for name, value in runner.parameters.items() if name.endswith("g_param")
    )
    targets = (conductance, intra, field, amplitude)
    expected = runner.initial_state
    for index in range(3):
        ve = field * waveform(expected["clock"]["t"])
        expected, _ = runner.functional.prepare_and_step(
            runner.parameters,
            runner.constants,
            expected,
            dn.func.StepInput(ve=ve, intra=intra[index]),
        )
    reference = torch.autograd.grad(_loss(expected), targets)
    actual, auxiliary = runner(
        inputs=dn.func.Inputs(intra=intra), extra=[(field, waveform)]
    )
    _close(actual, expected)
    gradients = torch.autograd.grad(_loss(actual), targets)
    _close(gradients, reference)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    assert auxiliary["callbacks"]["trace"]["v"].shape[0] == 4
    _close(model.state_dict(), source)


def test_shared_inputs_adapt_to_exact_step_input_for_authored_legacy_callables():
    _, functional, parameters, constants, state, name = _case()
    inputs = _drives(functional.shape, 3)
    expected, _ = functional.prepare_and_rollout(
        parameters, constants, state, dn.func.RolloutInput(*inputs), steps=3
    )
    reference = torch.autograd.grad(
        _loss(expected), (parameters[name], inputs.ve, inputs.intra)
    )
    prepared = functional.prepare(parameters, constants)
    observed = []

    def authored(current, frame):
        assert type(frame) is dn.func.StepInput
        index = len(observed)
        assert frame.ve.data_ptr() == inputs.ve[index].data_ptr()
        assert frame.intra.data_ptr() == inputs.intra[index].data_ptr()
        assert frame.noise is None and frame.stimulation is None
        observed.append(current["clock"]["t"].detach().clone())
        return functional.step(parameters, prepared, current, frame)

    actual, _ = dn.func.run(functional, authored, state, inputs=inputs)
    _close(actual, expected)
    gradients = torch.autograd.grad(
        _loss(actual), (parameters[name], inputs.ve, inputs.intra)
    )
    _close(gradients, reference)
    assert len(observed) == 3
    _close(torch.stack(observed), torch.arange(3) * DT + state["clock"]["t"])


@pytest.mark.parametrize("field", ("ve", "intra"))
def test_shared_input_mutation_invalidates_checkpoint_replay(field):
    model, functional, parameters, constants, state, name = _case()
    inputs = _drives(functional.shape, 4)
    extra = None
    if field == "intra":
        inputs = dn.func.Inputs(intra=inputs.intra)
        extra = [
            (
                torch.linspace(-0.2, 0.3, model.v.numel()).reshape(model.shape),
                dn.sin(amp=0.2, freq=0.8, phase=0.3),
            )
        ]
    bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
        parameters, constants=constants
    )
    final, _ = dn.func.run(
        bound, state, inputs=inputs, steps=4, checkpoint_every=2, extra=extra
    )
    with torch.no_grad():
        getattr(inputs, field).add_(0.01 if field == "ve" else 1e-12)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(_loss(final), parameters[name])


@pytest.mark.parametrize("mutation", ("value", "rebind"))
def test_shared_inputs_keep_explicit_draw_checkpoint_guards_and_source_rng(mutation):
    from tests.func.test_runtime_draws_checkpoint_freshness import _rng_slot
    from tests.func.test_stimulation_rng_sharing import _lower, _population

    model, extra, _ = _population()
    functional, tensors = _lower(model, extra)
    original = _snapshot(functional.extract())
    global_rng = torch.get_rng_state().clone()
    inputs = dn.func.Inputs(intra=_drives(functional.shape, 4).intra)
    packet = functional.sample_runtime_draws(tensors.state, steps=4)
    bound = functional.bind(tensors.parameters, constants=tensors.constants)
    final, _ = dn.func.run(
        bound, tensors.state, inputs=inputs, draws=packet, steps=4, checkpoint_every=2
    )
    amplitude = next(
        value
        for name, value in tensors.parameters.items()
        if name.endswith("waveform.amp")
    )
    slot = _rng_slot(packet.rng)
    assert slot is not None
    owner, name = slot
    if mutation == "value":
        with torch.no_grad():
            owner[name].reshape(-1)[0] ^= 1
        error = RuntimeError
        message = "runtime draws changed after checkpoint forward|modified by an inplace operation"
    else:
        owner[name] = owner[name].clone()
        error = dn.func.FunctionalizationError
        message = "runtime draws changed after checkpoint forward"
    with pytest.raises(error, match=message):
        torch.autograd.grad(_loss(final), amplitude)
    _close(functional.extract(), original)
    _close(torch.get_rng_state(), global_rng)


def test_shared_input_adapters_preserve_source_and_initialization_error_precedence():
    model, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    runner = dn.compile(model, dt=DT, steps=1, backend="eager")
    with pytest.raises(ValueError, match="initialization overrides require"):
        runner(inputs=object(), initialization=object())
    model._flag_rebuild = True
    with pytest.raises(dn.func.FunctionalizationError, match="pending structural"):
        dn.func.run(bound, state, inputs=object(), steps=1)
    with pytest.raises(dn.func.FunctionalizationError, match="pending structural"):
        runner(inputs=object())


@pytest.mark.parametrize("checkpointed", (False, True))
def test_legacy_noise_sampler_canonical_returns_use_separate_runtime_operand(
    monkeypatch, checkpointed
):
    from dendra.func import _runners
    from tests.func.test_runtime_noise_execution import DT as NOISE_DT
    from tests.func.test_runtime_noise_execution import _case as noise_case

    _, functional, tensors, _ = noise_case()
    assert functional._poisson_plan is None
    parameters = tensors.independent_parameters(trainable="*g_param")
    conductance = next(
        value for name, value in parameters.items() if name.endswith("g_param")
    )
    intra = _drives(functional.shape, 4).intra
    inputs = dn.func.Inputs(intra=intra)
    packet = functional.sample_runtime_draws(tensors.state, steps=4)
    expected, _ = functional.prepare_and_rollout(
        parameters, tensors.constants, tensors.state, inputs=inputs, draws=packet
    )
    targets = (conductance, intra)
    reference = torch.autograd.grad(_loss(expected), targets)
    source = _snapshot(functional.extract())
    global_rng = torch.get_rng_state().clone()
    calls, frames = [], []

    def sample(current, steps):
        calls.append((current["clock"]["t"].clone(), steps))
        return functional.sample_runtime_draws(current, steps=steps)

    execute = _runners._execute_steps

    def observed(step, current, selected, steps):
        assert selected.noise is None and selected.stimulation is None
        assert isinstance(selected.draws, dn.func.RuntimeDraws)
        frames.append(selected)
        return execute(step, current, selected, steps)

    monkeypatch.setattr(functional, "sample_noise_draw_tape", sample)
    monkeypatch.setattr(_runners, "_execute_steps", observed)
    bound = functional.bind(parameters, constants=tensors.constants)
    options = {"checkpoint_every" if checkpointed else "host_span_steps": 2}
    actual, _ = dn.func.run(bound, tensors.state, inputs=inputs, steps=4, **options)
    _close(actual, expected)
    gradients = torch.autograd.grad(_loss(actual), targets)
    _close(gradients, reference)
    assert all(
        value.abs().sum() > 0 and torch.isfinite(value).all() for value in gradients
    )
    assert [steps for _, steps in calls] == [2, 2]
    _close(
        torch.stack([time for time, _ in calls]),
        tensors.state["clock"]["t"] + torch.arange(0, 4, 2) * NOISE_DT,
    )
    assert len(frames) >= 2
    _close(functional.extract(), source)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("runtime_extra", (False, True))
def test_legacy_noise_sampler_canonical_returns_support_strict_while_runner(
    monkeypatch, runtime_extra
):
    from tests.func.test_runtime_noise_execution import DT as NOISE_DT
    from tests.func.test_runtime_noise_execution import _case as noise_case

    model, _, _, _ = noise_case()
    options = {"extra": dn.VarInput(), "chunk_steps": 4} if runtime_extra else {}
    runner = dn.compile(
        model,
        dt=NOISE_DT,
        steps=4,
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="eager",
        **options,
    )
    functional = runner.functional
    assert functional._poisson_plan is None
    source = _snapshot(functional.extract())
    global_rng = torch.get_rng_state().clone()
    inputs = dn.func.Inputs(intra=_drives(functional.shape, 4).intra)
    draws = functional.sample_runtime_draws(runner.initial_state, steps=4)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    with torch.no_grad():
        expected, reference = functional.prepare_and_rollout(
            runner.parameters,
            runner.constants,
            runner.initial_state,
            inputs,
            draws=draws,
            callbacks=callbacks,
        )
    calls = []

    def sample(current, steps):
        calls.append(steps)
        return functional.sample_runtime_draws(current, steps=steps)

    monkeypatch.setattr(functional, "sample_noise_draw_tape", sample)
    actual, auxiliary = runner(
        inputs=inputs, **({"extra": None} if runtime_extra else {})
    )
    _close(actual, expected)
    _close(dict(auxiliary["callbacks"]), dict(reference["callbacks"]))
    assert calls == [4] * (2 if runtime_extra else 1)
    assert runner.execution_report().strategy == "compiled_while_loop"
    _close(functional.extract(), source)
    _close(torch.get_rng_state(), global_rng)


def test_authored_legacy_canonical_sampler_source_changes_fail_before_execution(
    monkeypatch,
):
    from dendra.func import _runners
    from tests.func.test_runtime_noise_execution import _case as noise_case

    model, functional, tensors, _ = noise_case()
    bound = functional.bind(tensors.parameters, constants=tensors.constants)

    def sample(current, steps):
        packet = functional.sample_runtime_draws(current, steps=steps)
        model._flag_rebuild = True
        return packet

    def unexpected_execute(*args, **kwargs):
        pytest.fail(
            "Authored sampler source edits must fail before unchecked execution"
        )

    monkeypatch.setattr(functional, "sample_noise_draw_tape", sample)
    monkeypatch.setattr(_runners, "_execute_steps", unexpected_execute)
    with pytest.raises(dn.func.FunctionalizationError, match="pending structural"):
        dn.func.run(bound, tensors.state, inputs=dn.func.Inputs(), steps=4)
