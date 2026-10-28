"""Raw bindings share the Population and Network host execution contract."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas

DT = 0.025
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _case(*, shared=False):
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-55.0,
        dtype=torch.float64,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-70.0)
    if shared:
        model.insert(pas.rename("other"), g=0.001, e=-70.0)
    model.train()
    model.initialize()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*g_param")
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    state = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )
    # Nonuniform voltage makes both membrane and axial coefficients observable.
    voltage = state["integrator"]["v"]
    state["integrator"]["v"] = torch.linspace(
        -58.0, -52.0, voltage.numel(), dtype=voltage.dtype
    ).reshape_as(voltage)
    name = next(name for name in parameters if name.endswith("leak.g_param"))
    if shared:
        alias = next(key for key in parameters if key.endswith("other.g_param"))
        parameters[alias] = parameters[name]
    return model, functional, parameters, constants, state, name


def _close(actual, expected):
    left, spec = pytree.tree_flatten(actual)
    right, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=2e-9, atol=2e-10)
        else:
            assert value == reference


def _loss(state):
    return state["integrator"]["v"].square().mean()


def _bind(functional, parameters, constants, *, compiled):
    owner = (
        functional.compile_rollout_chunk(3, backend="aot_eager")
        if compiled
        else functional
    )
    return owner.bind(parameters, constants=constants)


@pytest.mark.parametrize("compiled", (False, True))
def test_raw_population_binding_reuses_live_leaves_across_backward_and_optimizer_updates(
    compiled,
):
    _, functional, parameters, constants, state, name = _case()
    constants["diam"].requires_grad_()
    targets = (parameters[name], constants["diam"])
    optimizer = torch.optim.SGD(targets, lr=1.0e-7)
    bound = _bind(functional, parameters, constants, compiled=compiled)
    outputs = []

    for iteration in range(3):
        expected, _ = functional.prepare_and_rollout(
            parameters, constants, state, steps=4
        )
        reference = torch.autograd.grad(_loss(expected), targets)
        actual, _ = dn.func.run(bound, state, steps=4, host_span_steps=3)
        optimizer.zero_grad(set_to_none=True)
        _loss(actual).backward()
        _close(actual, expected)
        _close(tuple(value.grad for value in targets), reference)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in reference
        )
        outputs.append(actual["integrator"]["v"].detach().clone())
        if iteration == 1:
            optimizer.step()
    _close(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])
    assert bound.prepared is None


def test_raw_population_binding_snapshots_mappings_and_preserves_shared_tensor_aliases():
    _, functional, parameters, constants, state, name = _case(shared=True)
    alias = next(key for key in parameters if key.endswith("other.g_param"))
    assert parameters[name] is parameters[alias]
    bound = functional.bind(parameters, constants=constants)
    original, diameter = parameters[name], constants["diam"]
    assert bound.parameters[name] is original
    assert bound.parameters[alias] is original
    assert bound.constants["diam"] is diameter
    for mapping, key in ((bound.parameters, name), (bound.constants, "diam")):
        with pytest.raises(TypeError):
            mapping[key] = mapping[key].clone()
    parameters[name] = original + 0.01
    constants["diam"] = diameter + 1.0
    assert bound.parameters[name] is original
    assert bound.constants["diam"] is diameter
    with torch.no_grad():
        original.mul_(1.1)
        actual, _ = bound(state)
        expected, _ = functional.prepare_and_rollout(
            bound.parameters, bound.constants, state, steps=1
        )
    _close(actual, expected)


def test_explicit_prepared_binding_keeps_its_existing_lifetime_and_errors():
    _, functional, parameters, constants, state, name = _case()
    prepared = functional.prepare(parameters, constants)
    bound = functional.bind(parameters, prepared)
    assert bound.prepared is prepared
    assert bound.constants is None
    with pytest.raises(TypeError, match="mutually exclusive"):
        functional.bind(parameters, prepared, constants=constants)
    with torch.no_grad():
        parameters[name].add_(0.01)
    with pytest.raises(
        dn.func.FunctionalizationError, match="parameters changed after prepare"
    ):
        bound(state)


@pytest.mark.parametrize("compiled", (False, True))
def test_raw_binding_direct_inputs_and_duration_preserve_chunk_tails_and_remainders(
    compiled,
):
    model, functional, parameters, constants, state, _ = _case()
    bound = _bind(functional, parameters, constants, compiled=compiled)
    ve = torch.linspace(
        -0.1, 0.2, bound.steps * model.v.numel(), dtype=model.dtype()
    ).reshape(bound.steps, *model.shape)
    inputs = dn.func.RolloutInput(ve=ve)
    direct = dn.func.StepInput(ve=ve[0]) if not compiled else inputs
    actual, _ = bound(state, direct)
    expected, _ = functional.prepare_and_rollout(
        parameters, constants, state, inputs=inputs, steps=bound.steps
    )
    _close(actual, expected)
    first, _ = dn.func.run(bound, state, duration=0.6 * DT)
    second, _ = dn.func.run(bound, first, duration=0.6 * DT)
    third, _ = dn.func.run(bound, second, steps=4, host_span_steps=2)
    reference, _ = functional.prepare_and_rollout(parameters, constants, state, steps=5)
    _close(third["integrator"]["v"], reference["integrator"]["v"])
    assert third["control"]["duration_remainder"].item() == pytest.approx(0.2 * DT)


def test_existing_raw_eager_binding_retains_its_captured_transition(monkeypatch):
    model, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    ve = torch.linspace(-0.1, 0.2, 2 * model.v.numel(), dtype=model.dtype()).reshape(
        2, *model.shape
    )
    inputs = dn.func.RolloutInput(ve=ve)
    expected_direct = functional.prepare_and_rollout(
        parameters, constants, state, inputs=dn.func.RolloutInput(ve=ve[:1]), steps=1
    )
    expected_run = functional.prepare_and_rollout(
        parameters, constants, state, inputs=inputs, steps=2
    )

    def override(*arguments, **keywords):
        raise AssertionError("An existing raw binding retains its captured transition")

    monkeypatch.setattr(functional, "step", override)
    _close(bound(state, dn.func.StepInput(ve=ve[0])), expected_direct)
    actual, auxiliary = dn.func.run(bound, state, inputs=inputs)
    continuation = auxiliary["continuation"]
    _close(continuation.state, actual)
    assert continuation.callback_state is None
    assert all(
        continuation.parameters[key] is value for key, value in parameters.items()
    )
    assert all(continuation.constants[key] is value for key, value in constants.items())
    _close(
        (
            actual,
            {key: value for key, value in auxiliary.items() if key != "continuation"},
        ),
        expected_run,
    )


@pytest.mark.parametrize("compiled", (False, True))
def test_raw_population_binding_does_not_retain_training_preparation_graphs(
    monkeypatch, compiled
):
    _, functional, parameters, constants, state, _ = _case()
    constants["diam"].requires_grad_()
    bound = _bind(functional, parameters, constants, compiled=compiled)
    references = []
    prepare = functional._prepare_source_validated

    def observe(*arguments):
        prepared = prepare(*arguments)
        references.append(
            (
                weakref.ref(prepared),
                weakref.ref(prepared.values["integrator"]["diag_base"]),
            )
        )
        return prepared

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    for _ in range(2):
        final, auxiliary = dn.func.run(bound, state, steps=4)
        _loss(final).backward()
        del final, auxiliary
        gc.collect()
        assert all(reference() is None for frame in references for reference in frame)
    assert len(references) == 2
    assert bound._inference_prepared is None


@pytest.mark.parametrize("compiled", (False, True))
def test_network_raw_keyword_binding_preserves_positional_compatibility_and_live_optimizer_edits(
    compiled,
):
    from tests.func.test_network import (
        _RAW_WEIGHT,
    )
    from tests.func.test_network import _loss as network_loss
    from tests.func.test_network import (
        _mixed_network,
    )

    functional, tensors = dn.func.make_functional_network(_mixed_network())
    names = tuple(
        name
        for name in tensors.named_parameters
        if name.startswith("synapses.") and name.endswith(f".{_RAW_WEIGHT}")
    )
    parameters = tensors.independent_parameters(trainable=names)
    targets = tuple(
        value
        for name, value in tensors._replace(
            parameters=parameters
        ).named_parameters.items()
        if name in names
    )
    owner = (
        functional.compile_rollout_chunk(3, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(parameters, constants=tensors.constants)
    legacy = owner.bind(parameters, tensors.constants)
    optimizer = torch.optim.SGD(targets, lr=0.01)
    for _ in range(2):
        expected, _ = functional.prepare_and_rollout(
            parameters, tensors.constants, tensors.state, steps=9
        )
        reference = torch.autograd.grad(network_loss(expected), targets)
        actual, _ = dn.func.run(bound, tensors.state, steps=9)
        optimizer.zero_grad(set_to_none=True)
        network_loss(actual).backward()
        _close(actual, expected)
        _close(tuple(value.grad for value in targets), reference)
        compatible, _ = dn.func.run(legacy, tensors.state, steps=9)
        _close(compatible, actual)
        optimizer.step()
    assert not hasattr(bound, "_prepared")


@pytest.mark.parametrize("source", ("parameter", "constant"))
def test_raw_binding_checkpoint_replay_rejects_late_input_mutation(source):
    _, functional, parameters, constants, state, name = _case()
    bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
        parameters, constants=constants
    )
    final, _ = dn.func.run(bound, state, steps=4, checkpoint_every=2)
    changed = parameters[name] if source == "parameter" else constants["diam"]
    with torch.no_grad():
        changed.add_(0.001)
    with pytest.raises(
        dn.func.FunctionalizationError, match=f"{source}s changed after prepare"
    ):
        _loss(final).backward()


def test_raw_binding_checkpoint_continuation_preserves_recording_and_gradients():
    _, functional, parameters, constants, state, name = _case()
    bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
        parameters, constants=constants
    )
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    expected, whole = dn.func.run(bound, state, steps=6, callbacks=callbacks)
    reference = torch.autograd.grad(_loss(expected), parameters[name])[0]
    prefix, first = dn.func.run(
        bound, state, steps=3, checkpoint_every=2, callbacks=callbacks
    )
    empty, middle = dn.func.run(
        bound,
        prefix,
        steps=0,
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    final, auxiliary = dn.func.run(
        bound,
        empty,
        steps=3,
        checkpoint_every=2,
        callbacks=callbacks,
        callback_state=middle["callbacks"].state,
    )
    _close(final, expected)
    _close(torch.autograd.grad(_loss(final), parameters[name])[0], reference)
    for field in ("v", "t"):
        prefix_trace = first["callbacks"]["trace"][field]
        empty_trace = middle["callbacks"]["trace"][field]
        suffix_trace = auxiliary["callbacks"]["trace"][field]
        assert prefix_trace.shape[0] == 4
        assert empty_trace.shape[0] == 0
        assert suffix_trace.shape[0] == 3
        _close(
            torch.cat((prefix_trace, suffix_trace)), whole["callbacks"]["trace"][field]
        )


def test_raw_binding_accepts_canonical_runtime_draws_and_call_local_extra():
    from tests.func.test_stimulation_rng_sharing import _lower, _population

    model, extra, _ = _population("mechanism")
    functional, tensors = _lower(model, extra)
    bound = functional.bind(tensors.parameters, constants=tensors.constants)
    draws = functional.sample_runtime_draws(tensors.state, steps=3)
    field = torch.linspace(-1.0, 1.0, model.v.numel(), dtype=model.dtype()).reshape(
        model.shape
    )
    waveform = dn.sin(amp=0.2, freq=0.8, phase=0.3)
    options = dict(steps=3, draws=draws, extra=[(field, waveform)])
    actual, _ = dn.func.run(bound, tensors.state, **options)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    expected, _ = dn.func.run(
        functional.bind(tensors.parameters, prepared), tensors.state, **options
    )
    _close(actual, expected)
