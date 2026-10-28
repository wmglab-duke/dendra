"""Shared deterministic inputs retain Population execution and draw ownership."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_raw_bindings import DT, _case, _close
from tests.func.test_raw_bindings import _environment as _environment
from tests.func.test_raw_bindings import _loss

pytestmark = pytest.mark.cpu


def _drives(model, steps):
    ve = (
        torch.linspace(-0.2, 0.3, steps * model.v.numel(), dtype=model.dtype())
        .reshape(steps, *model.shape)
        .requires_grad_()
    )
    intra = (
        torch.linspace(0.01, 0.04, steps * model.v.numel(), dtype=model.dtype())
        .reshape(steps, *model.shape)
        .requires_grad_()
    )
    return ve, intra


@pytest.mark.parametrize(
    "entry",
    (
        "step",
        "atomic_step",
        "rollout",
        "atomic_rollout",
        "chunk",
        "raw_step",
        "host",
        "convenience",
    ),
)
def test_inputs_share_values_and_nonzero_gradients_across_population_boundaries(entry):
    model, functional, parameters, constants, state, name = _case()
    steps = 1 if entry in ("step", "atomic_step", "raw_step") else 3
    ve, intra = _drives(model, steps)
    constants["diam"].requires_grad_()
    targets = (parameters[name], constants["diam"], ve, intra)
    expected, _ = functional.prepare_and_rollout(
        parameters, constants, state, dn.func.RolloutInput(ve, intra), steps=steps
    )
    reference = torch.autograd.grad(_loss(expected), targets)
    source_ve, source_intra = (ve[0], intra[0]) if steps == 1 else (ve, intra)
    inputs = dn.func.Inputs(source_ve, source_intra)
    if entry == "atomic_step":
        actual, _ = functional.prepare_and_step(parameters, constants, state, inputs)
    elif entry == "atomic_rollout":
        actual, _ = functional.prepare_and_rollout(
            parameters, constants, state, inputs, steps=steps
        )
    elif entry == "step":
        actual, _ = functional.step(
            parameters, functional.prepare(parameters, constants), state, inputs
        )
    elif entry == "rollout":
        actual, _ = functional.rollout(
            parameters,
            functional.prepare(parameters, constants),
            state,
            inputs,
            steps=steps,
        )
    elif entry == "chunk":
        kernel = functional.compile_rollout_chunk(steps, backend="aot_eager")
        actual, _ = kernel(
            parameters, functional.prepare(parameters, constants), state, inputs
        )
    elif entry == "raw_step":
        actual, _ = functional.bind(parameters, constants=constants)(state, inputs)
    elif entry == "host":
        bound = functional.compile_rollout_chunk(2, backend="aot_eager").bind(
            parameters, constants=constants
        )
        actual, _ = dn.func.run(
            bound, state, inputs=inputs, steps=steps, checkpoint_every=2
        )
    else:
        runner = dn.compile(
            model, dt=DT, steps=steps, inference=False, backend="aot_eager"
        )
        actual, _ = runner(
            state, parameters=parameters, constants=constants, inputs=inputs
        )
    _close(actual, expected)
    observed = torch.autograd.grad(_loss(actual), targets)
    _close(observed, reference)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert inputs.ve is source_ve
    assert inputs.intra is source_intra


def test_inputs_are_transformable_without_detaching_drive_or_parameter_gradients():
    model, functional, parameters, constants, state, name = _case()
    ve, intra = _drives(model, 2)

    def evaluate(conductance, *, legacy=False):
        selected = {**parameters, name: conductance}
        inputs = (
            dn.func.RolloutInput(ve, intra) if legacy else dn.func.Inputs(ve, intra)
        )
        final, _ = functional.prepare_and_rollout(
            selected, constants, state, inputs, steps=2
        )
        return _loss(final)

    base = parameters[name]
    reference = torch.func.grad(lambda value: evaluate(value, legacy=True))(base)
    assert reference.abs().sum() > 0
    _close(torch.func.grad(evaluate)(base), reference)
    lanes = torch.stack((base * 0.9, base * 1.1))
    _close(
        torch.func.vmap(evaluate)(lanes),
        torch.func.vmap(lambda value: evaluate(value, legacy=True))(lanes),
    )


@pytest.mark.parametrize("atomic", (False, True))
def test_population_source_errors_precede_legacy_input_container_errors(atomic):
    model, functional, parameters, constants, state, _ = _case()
    prepared = functional.prepare(parameters, constants)
    model._flag_rebuild = True
    with pytest.raises(
        dn.func.FunctionalizationError, match="pending structural changes"
    ):
        if atomic:
            functional.prepare_and_rollout(
                parameters, constants, state, object(), steps=1
            )
        else:
            functional.rollout(parameters, prepared, state, object(), steps=1)


def test_inputs_zero_steps_and_continuation_retain_clock_and_recorder_boundaries():
    model, functional, parameters, constants, state, _ = _case()
    ve, intra = _drives(model, 4)
    bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
        parameters, constants=constants
    )
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    whole, full = dn.func.run(
        bound, state, inputs=dn.func.Inputs(ve, intra), steps=4, callbacks=callbacks
    )
    first, head = dn.func.run(
        bound,
        state,
        inputs=dn.func.Inputs(ve[:2], intra[:2]),
        steps=2,
        callbacks=callbacks,
    )
    empty, middle = dn.func.run(
        bound,
        first,
        inputs=dn.func.Inputs(ve[:0], intra[:0]),
        steps=0,
        callbacks=callbacks,
        callback_state=head["callbacks"].state,
    )
    actual, tail = dn.func.run(
        bound,
        empty,
        inputs=dn.func.Inputs(ve[2:], intra[2:]),
        duration=2 * DT,
        callbacks=callbacks,
        callback_state=middle["callbacks"].state,
    )
    _close(actual, whole)
    _close(empty, first)
    for key, values in full["callbacks"]["trace"].items():
        assert middle["callbacks"]["trace"][key].shape[0] == 0
        _close(
            torch.cat(
                (head["callbacks"]["trace"][key], tail["callbacks"]["trace"][key])
            ),
            values,
        )
    targets = (ve, intra)
    _close(
        torch.autograd.grad(_loss(actual), targets),
        torch.autograd.grad(_loss(whole), targets),
    )


def test_inputs_checkpoint_replay_rejects_mutation_of_live_drive_tapes():
    model, functional, parameters, constants, state, _ = _case()
    ve, intra = _drives(model, 4)
    bound = functional.bind(parameters, constants=constants)
    final, _ = dn.func.run(
        bound, state, inputs=dn.func.Inputs(ve, intra), steps=4, checkpoint_every=2
    )
    with torch.no_grad():
        ve.add_(0.1)
    with pytest.raises(
        RuntimeError,
        match="modified by an inplace operation|changed after checkpoint forward",
    ):
        torch.autograd.grad(_loss(final), intra)


def test_inputs_keep_canonical_noise_sde_and_poisson_authority_separate(monkeypatch):
    from tests.func.test_stimulation_rng_sharing import _lower, _population

    model, extra, pulse = _population("mechanism")
    functional, tensors = _lower(model, extra)
    ve, intra = _drives(model, 4)
    inputs = dn.func.Inputs(ve, intra)
    tape = functional.sample_runtime_draws(tensors.state, steps=4)
    frozen = pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tape)
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    expected, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve, intra),
        steps=4,
        draws=tape,
    )
    original = functional._transition_rollout_values
    observed = []

    def transition(
        parameters,
        prepared,
        state,
        ve,
        intra,
        steps,
        noise=None,
        stimulation=None,
        draws=None,
    ):
        assert noise is None and stimulation is None
        assert isinstance(draws, dn.func.RuntimeDraws)
        observed.append(draws)
        return original(
            parameters, prepared, state, ve, intra, steps, noise, stimulation, draws
        )

    monkeypatch.setattr(functional, "_transition_rollout_values", transition)
    actual, _ = functional.rollout(
        tensors.parameters,
        functional.prepare(tensors.parameters, tensors.constants),
        tensors.state,
        inputs,
        steps=4,
        draws=tape,
    )
    _close(actual, expected)
    assert len(observed) == 4
    _close(tape, frozen)
    _close(pulse.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)
    _close(
        torch.autograd.grad(_loss(actual), (ve, intra)),
        torch.autograd.grad(_loss(expected), (ve, intra)),
    )


def test_inputs_scan_explicit_and_automatic_draws_continue_with_native_order():
    from tests.func.test_stimulation_rng_sharing import _lower, _population

    model, extra, _ = _population("mechanism")
    functional, tensors = _lower(model, extra)
    ve, intra = _drives(model, 4)
    bound = functional.compile_rollout_chunk(
        3, execution="scan", backend="aot_eager"
    ).bind(tensors.parameters, constants=tensors.constants)
    tape = functional.sample_runtime_draws(tensors.state, steps=4)
    explicit, _ = dn.func.run(
        bound,
        tensors.state,
        inputs=dn.func.Inputs(ve, intra),
        steps=4,
        draws=tape,
        checkpoint_every=2,
    )
    automatic, _ = dn.func.run(
        bound,
        tensors.state,
        inputs=dn.func.Inputs(ve, intra),
        steps=4,
        checkpoint_every=2,
    )
    _close(explicit, automatic)
    _close(
        torch.autograd.grad(_loss(explicit), (ve, intra)),
        torch.autograd.grad(_loss(automatic), (ve, intra)),
    )
