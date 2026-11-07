"""Owned Inductor compilation preserves training and portable continuation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_runner_chunking import (
    DT,
    _callbacks,
    _clone,
    _close,
    _inputs,
    _kernel_cache,
    _loss,
    _model,
    _reference,
    _trace,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _runner(model, network):
    return dn.compile(
        model,
        dt=DT,
        steps=5,
        chunk_steps=3,
        checkpoint_every=2,
        execution="scan",
        inference=False,
        backend="inductor",
        callbacks=_callbacks(network),
        trainable="*g_param",
    )


def _source_values(subject):
    return tuple(entry.tensor for entry in dn.func.inspect_sources(subject).entries)


def _drive(population, network, start, stop):
    current = (
        torch.linspace(
            start,
            stop,
            5 * population.v.numel(),
            dtype=population.dtype(),
            device=population.device(),
        ).reshape(5, *population.shape)
        * dn.units.nA
    ).requires_grad_()
    return current, _inputs(network, dn.func.Inputs(intra=current))


def _gradients(result, targets):
    gradients = torch.autograd.grad(_loss(result), targets)
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value)
        for value in gradients
    )
    return gradients


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_inductor_training_repeated_backward_and_portable_checkpoint(network, tmp_path):
    model, receiver_model = _model(network), _model(network)
    population = model.cell if network else model
    runner, receiver = _runner(model, network), _runner(receiver_model, network)
    leaves = dn.func.inspect_sources(runner).trainable_leaves
    assert leaves
    current, inputs = _drive(population, network, 0.1, 0.3)
    targets = (*leaves, current)
    sources = _source_values(runner), _source_values(receiver)
    frozen = _clone(
        (
            sources,
            runner.initial_state,
            receiver.initial_state,
            inputs,
            runner.functional.extract(),
            receiver.functional.extract(),
        )
    )
    rng = torch.get_rng_state().clone()
    expected = _reference(runner, inputs=inputs)
    gradients = _gradients(expected, targets)
    cached = None
    for _ in range(2):
        actual = runner(inputs=inputs)
        _close(actual[0], expected[0])
        _close(_trace(actual), _trace(expected))
        assert all(value.shape[0] == 6 for value in _trace(actual).values())
        _close(_gradients(actual, targets), gradients)
        kernels = _kernel_cache(runner, network)
        assert kernels
        if cached is not None:
            assert kernels == cached
        cached = kernels

    path = tmp_path / "continuation.pt"
    torch.save(runner.export_continuation(actual[1]["continuation"]), path)
    restored = receiver.restore_continuation(torch.load(path, weights_only=True))
    restored_leaves = dn.func.inspect_sources(restored).trainable_leaves
    assert len(restored_leaves) == len(leaves)
    assert all(value.is_leaf and value.requires_grad for value in restored_leaves)
    assert not {id(value) for value in restored_leaves} & {
        id(value)
        for value in (*leaves, *dn.func.inspect_sources(receiver).trainable_leaves)
    }
    _close(restored_leaves, leaves)
    assert restored.callback_state.plan_token is receiver.callbacks._state_token
    assert restored.callback_state.plan_token is not runner.callbacks._state_token
    suffix, suffix_inputs = _drive(population, network, 0.2, 0.4)
    resumed_targets = (*restored_leaves, suffix)
    restored_sources = _source_values(restored)
    suffix_frozen = _clone(
        (
            restored.state,
            restored_sources,
            restored.callback_state.carries,
            suffix_inputs,
        )
    )
    reference = receiver.functional.prepare_and_rollout(
        restored.parameters,
        restored.constants,
        restored.state,
        inputs=suffix_inputs,
        steps=5,
        callbacks=receiver.callbacks,
        callback_state=restored.callback_state,
    )
    resumed_gradients = _gradients(reference, resumed_targets)
    for _ in range(2):
        resumed = receiver(restored, inputs=suffix_inputs)
        _close(resumed[0], reference[0])
        _close(_trace(resumed), _trace(reference))
        assert all(value.shape[0] == 5 for value in _trace(resumed).values())
        observed = torch.autograd.grad(
            _loss(resumed), (*resumed_targets, *targets), allow_unused=True
        )
        _close(observed[: len(resumed_targets)], resumed_gradients)
        assert all(value is None for value in observed[len(resumed_targets) :])

    full_inputs = _inputs(network, dn.func.Inputs(intra=torch.cat((current, suffix))))
    full = _reference(runner, inputs=full_inputs, steps=10)
    _close(resumed[0], full[0])
    for name, prefix in _trace(actual).items():
        _close(torch.cat((prefix, _trace(resumed)[name])), _trace(full)[name])
    _close(
        (
            restored.state,
            restored_sources,
            restored.callback_state.carries,
            suffix_inputs,
        ),
        suffix_frozen,
    )
    _close(
        (
            sources,
            runner.initial_state,
            receiver.initial_state,
            inputs,
            runner.functional.extract(),
            receiver.functional.extract(),
        ),
        frozen,
    )
    _close(torch.get_rng_state(), rng)
