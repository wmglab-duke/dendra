"""Direct bound and fixed-chunk calls share callback carry and kernel ownership."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_inputs_population import _drives
from tests.func.test_raw_bindings import DT, _case, _close
from tests.func.test_raw_bindings import _environment as _environment
from tests.func.test_runner_stimulation_clock import STEPS as PULSE_STEPS
from tests.func.test_runner_stimulation_clock import _case as _pulse_case

pytestmark = pytest.mark.cpu


class _VoltageEnergy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].square().mean(), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return carry + state["integrator"]["v"].square().mean(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


def _callbacks(functional):
    return functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "t"]), "energy": _VoltageEnergy()}
    )


def _loss(auxiliary):
    results = auxiliary["callbacks"]
    return results["energy"] + results["trace"]["v"].square().mean()


def _call(
    functional, parameters, constants, state, inputs, callbacks, entry, execution
):
    steps = inputs.intra.shape[0]
    kernel = functional.compile_rollout_chunk(
        steps, backend="aot_eager", execution=execution
    )
    if entry == "raw":
        return kernel.bind(parameters, constants=constants)(
            state, inputs, callbacks=callbacks
        )
    prepared = functional.prepare(parameters, constants)
    if entry == "prepared":
        return kernel.bind(parameters, prepared)(state, inputs, callbacks=callbacks)
    return kernel(parameters, prepared, state, inputs, callbacks=callbacks)


@pytest.mark.parametrize("entry", ("chunk", "prepared", "raw"))
@pytest.mark.parametrize("execution", ("default", "scan"))
def test_direct_population_callbacks_preserve_values_and_nonzero_gradients(
    entry, execution
):
    model, functional, parameters, constants, state, name = _case()
    ve, intra = _drives(model, 3)
    state["integrator"]["v"].requires_grad_()
    callbacks = _callbacks(functional)
    inputs = dn.func.Inputs(ve, intra)
    targets = (parameters[name], state["integrator"]["v"], ve, intra)
    expected, expected_auxiliary = functional.prepare_and_rollout(
        parameters, constants, state, inputs, steps=3, callbacks=callbacks
    )
    reference = torch.autograd.grad(_loss(expected_auxiliary), targets)
    actual, auxiliary = _call(
        functional, parameters, constants, state, inputs, callbacks, entry, execution
    )
    gradients = torch.autograd.grad(_loss(auxiliary), targets)
    _close(actual, expected)
    _close(dict(auxiliary["callbacks"]), dict(expected_auxiliary["callbacks"]))
    _close(gradients, reference)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    assert auxiliary["callbacks"]["trace"]["v"].shape[0] == 4


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("raw", (False, True))
def test_bound_population_callback_continuation_has_no_duplicate_sample(compiled, raw):
    model, functional, parameters, constants, state, name = _case()
    target = parameters[name]
    width = 2 if compiled else 1
    ve, intra = _drives(model, 2 * width)
    callbacks = _callbacks(functional)
    expected, expected_auxiliary = functional.prepare_and_rollout(
        parameters,
        constants,
        state,
        dn.func.Inputs(ve, intra),
        steps=2 * width,
        callbacks=callbacks,
    )
    reference = torch.autograd.grad(expected_auxiliary["callbacks"]["energy"], target)[
        0
    ]
    owner = (
        functional.compile_rollout_chunk(width, backend="aot_eager")
        if compiled
        else functional
    )
    bound = (
        owner.bind(parameters, constants=constants)
        if raw
        else owner.bind(parameters, functional.prepare(parameters, constants))
    )
    inputs = dn.func.Inputs(ve[:width], intra[:width])
    if not compiled:
        inputs = pytree.tree_map_only(torch.Tensor, lambda value: value[0], inputs)
    prefix, prefix_auxiliary = bound(state, inputs, callbacks=callbacks)
    inputs = dn.func.Inputs(ve[width:], intra[width:])
    if not compiled:
        inputs = pytree.tree_map_only(torch.Tensor, lambda value: value[0], inputs)
    final, auxiliary = bound(
        prefix,
        inputs,
        callbacks=callbacks,
        callback_state=prefix_auxiliary["callbacks"].state,
    )
    _close(final, expected)
    results = auxiliary["callbacks"]
    for name in ("v", "t"):
        combined = torch.cat(
            (prefix_auxiliary["callbacks"]["trace"][name], results["trace"][name])
        )
        _close(combined, expected_auxiliary["callbacks"]["trace"][name])
    _close(results["energy"], expected_auxiliary["callbacks"]["energy"])
    # The cumulative reducer retains both segments' parameter graph.
    observed = torch.autograd.grad(results["energy"], target)[0]
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    _close(observed, reference)
    assert results["trace"]["v"].shape[0] == width


@pytest.mark.parametrize("raw", (False, True))
def test_eager_population_callbacks_use_the_captured_step(raw):
    _, functional, parameters, constants, state, _ = _case()
    callbacks = _callbacks(functional)
    bound = (
        functional.bind(parameters, constants=constants)
        if raw
        else functional.bind(parameters, functional.prepare(parameters, constants))
    )

    def replaced(*args, **kwargs):
        raise AssertionError("binding must retain its original step")

    functional.step = replaced
    final, auxiliary = bound(state, callbacks=callbacks)
    assert auxiliary["callbacks"]["trace"]["v"].shape[0] == 2
    assert final["clock"]["t"].item() == pytest.approx(DT)


@pytest.mark.parametrize("stimulus_kind", ("intra", "extra"))
@pytest.mark.parametrize("execution", ("default", "scan"))
def test_direct_callbacks_preserve_atomic_pulse_clock_and_separate_host_cache(
    stimulus_kind, execution
):
    functional, tensors, parameters, prepared, _ = _pulse_case(stimulus_kind)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    kernel = functional.compile_rollout_chunk(
        PULSE_STEPS, backend="aot_eager", execution=execution
    )
    with torch.no_grad():
        expected, _ = functional.rollout(
            parameters, prepared, tensors.state, steps=PULSE_STEPS
        )
        ordinary, _ = kernel(parameters, prepared, tensors.state)
        actual, auxiliary = kernel(
            parameters, prepared, tensors.state, callbacks=callbacks
        )
        bound = kernel.bind(parameters, prepared)
        host, _ = dn.func.run(bound, tensors.state, steps=PULSE_STEPS)
        recorded_host, _ = dn.func.run(
            bound, tensors.state, steps=PULSE_STEPS, callbacks=callbacks
        )
    _close(actual, expected)
    _close(ordinary, expected)
    _close(recorded_host, host)
    assert not torch.equal(host["integrator"]["v"], ordinary["integrator"]["v"])
    _close(auxiliary["callbacks"]["trace"]["v"][-1], ordinary["integrator"]["v"])
    kernels = kernel._callback_chunks[callbacks]
    assert set(kernels) == {(PULSE_STEPS, False), (PULSE_STEPS, True)}
    assert kernels[(PULSE_STEPS, False)] is not kernels[(PULSE_STEPS, True)]


def test_direct_callback_cache_is_weak_and_does_not_own_prepared_graphs():
    _, functional, parameters, constants, state, _ = _case()
    kernel = functional.compile_rollout_chunk(2, disable=True)
    callbacks = _callbacks(functional)
    prepared = functional.prepare(parameters, constants)
    callback_ref, prepared_ref = weakref.ref(callbacks), weakref.ref(prepared)
    final, auxiliary = kernel(parameters, prepared, state, callbacks=callbacks)
    assert len(kernel._callback_chunks) == 1
    del final, auxiliary, callbacks, prepared
    gc.collect()
    assert callback_ref() is None
    assert prepared_ref() is None
    assert not kernel._callback_chunks


def test_direct_callback_resume_preserves_discovered_emission_schema_and_gradients():
    from tests.func.test_runner_compiled_callbacks import _PostStepVoltage

    _, functional, parameters, constants, state, name = _case()
    callbacks = functional.make_callbacks({"post": _PostStepVoltage()})
    expected, expected_auxiliary = functional.prepare_and_rollout(
        parameters, constants, state, steps=4, callbacks=callbacks
    )
    reference = torch.autograd.grad(
        expected_auxiliary["callbacks"]["post"].square().mean(), parameters[name]
    )[0]
    bound = functional.compile_rollout_chunk(2, backend="aot_eager").bind(
        parameters, constants=constants
    )
    prefix, prefix_auxiliary = bound(state, callbacks=callbacks)
    final, auxiliary = bound(
        prefix, callbacks=callbacks, callback_state=prefix_auxiliary["callbacks"].state
    )
    result = torch.cat(
        (prefix_auxiliary["callbacks"]["post"], auxiliary["callbacks"]["post"])
    )
    _close(final, expected)
    _close(result, expected_auxiliary["callbacks"]["post"])
    observed = torch.autograd.grad(result.square().mean(), parameters[name])[0]
    _close(observed, reference)
    assert observed.abs().sum() > 0
    assert callbacks._plans[0].emission_schema is None
    assert auxiliary["callbacks"].state.emission_schemas[0] is not None


def test_direct_callback_resume_rejects_changed_discovered_emission_shape():
    class AuxiliaryVoltage(dn.func.FunctionalCallback):
        def initialize(self, state, auxiliary):
            del auxiliary
            return state["integrator"]["v"].new_zeros(()), None

        def update(self, carry, state, auxiliary):
            del state
            return carry, auxiliary["v"]

        def finalize(self, carry, emissions):
            return carry if emissions is None else emissions.square().sum()

    _, functional, parameters, constants, state, _ = _case()
    prepared = functional.prepare(parameters, constants)
    callbacks = functional.make_callbacks({"emitter": AuxiliaryVoltage()})

    def narrow(current, inputs):
        current, auxiliary = functional.step(parameters, prepared, current, inputs)
        return current, {**auxiliary, "v": auxiliary["v"][..., :1]}

    prefix, auxiliary = dn.func.run(
        functional, narrow, state, tstop=DT, callbacks=callbacks
    )
    kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
    with pytest.raises(
        dn.func.FunctionalizationError, match="emission changed a leaf's shape"
    ):
        kernel(
            parameters,
            prepared,
            prefix,
            callbacks=callbacks,
            callback_state=auxiliary["callbacks"].state,
        )
    assert kernel.execution_report() is None


def test_direct_callbacks_keep_atomic_pulse_clock_with_explicit_noise_draws():
    from tests.func.test_network_runtime_noise import _NoisyLeak
    from tests.func.test_runner_stimulation_clock import DT as pulse_dt

    old_functional, _, _, _, _ = _pulse_case("intra")
    model = old_functional._population
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float32):
        model.insert(_NoisyLeak)
        model.initialize()
        model.t.fill_(1000.0)
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=pulse_dt)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    packet = functional.sample_runtime_draws(tensors.state, steps=PULSE_STEPS)
    original = pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), packet)
    kernel = functional.compile_rollout_chunk(PULSE_STEPS, backend="aot_eager")
    unrolled = functional.compile_rollout_chunk(
        PULSE_STEPS, backend="aot_eager", execution="unrolled"
    )
    with torch.no_grad():
        expected, _ = unrolled(
            tensors.parameters, prepared, tensors.state, draws=packet
        )
        ordinary, _ = kernel(tensors.parameters, prepared, tensors.state, draws=packet)
        assert kernel.execution_report().strategy == "compiled_while_loop"
        actual, auxiliary = kernel(
            tensors.parameters,
            prepared,
            tensors.state,
            callbacks=callbacks,
            draws=packet,
        )
        replay, replay_auxiliary = kernel(
            tensors.parameters,
            prepared,
            tensors.state,
            callbacks=callbacks,
            draws=packet,
        )
    assert kernel.execution_report().strategy == "compiled_while_loop"
    torch.testing.assert_close(ordinary, expected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(replay, actual)
    _close(dict(auxiliary["callbacks"]), dict(replay_auxiliary["callbacks"]))
    _close(packet, original)


def test_direct_callback_loss_supports_grad_vmap_and_forward_ad():
    model, functional, parameters, constants, state, name = _case()
    ve, intra = _drives(model, 2)
    callbacks = _callbacks(functional)
    kernel = functional.compile_rollout_chunk(2, backend="aot_eager")

    def loss(conductance, *, compiled):
        selected = {**parameters, name: conductance}
        prepared = functional.prepare(selected, constants)
        if compiled:
            _, auxiliary = kernel(
                selected,
                prepared,
                state,
                dn.func.Inputs(ve, intra),
                callbacks=callbacks,
            )
        else:
            _, auxiliary = functional.rollout(
                selected,
                prepared,
                state,
                dn.func.Inputs(ve, intra),
                steps=2,
                callbacks=callbacks,
            )
        return _loss(auxiliary)

    base = parameters[name]
    _close(
        torch.func.grad(lambda value: loss(value, compiled=True))(base),
        torch.func.grad(lambda value: loss(value, compiled=False))(base),
    )
    lanes = torch.stack((base * 0.9, base * 1.1))
    _close(
        torch.vmap(lambda value: loss(value, compiled=True))(lanes),
        torch.vmap(lambda value: loss(value, compiled=False))(lanes),
    )
    observed = torch.func.jvp(
        lambda value: loss(value, compiled=True), (base,), (base,)
    )
    expected = torch.func.jvp(
        lambda value: loss(value, compiled=False), (base,), (base,)
    )
    _close(observed, expected)
    assert observed[1].abs().sum() > 0


def test_direct_callback_errors_preserve_validation_order_and_successful_report():
    _, functional, parameters, constants, state, name = _case()
    kernel = functional.compile_rollout_chunk(2, disable=True)
    prepared = functional.prepare(parameters, constants)
    callbacks = _callbacks(functional)
    kernel(parameters, prepared, state, callbacks=callbacks)
    previous = kernel.execution_report()
    with pytest.raises(ValueError, match="callback_state requires callbacks"):
        kernel(parameters, prepared, state, callback_state={})
    with pytest.raises(TypeError, match="callbacks must be a FunctionalCallbacks"):
        kernel(parameters, prepared, state, callbacks={})
    other = _case()[1]
    with pytest.raises(dn.func.FunctionalizationError, match="different functional"):
        kernel(parameters, prepared, state, callbacks=_callbacks(other))
    with pytest.raises(TypeError, match="callback_state must be"):
        kernel(parameters, prepared, state, callbacks=callbacks, callback_state={})
    with pytest.raises((TypeError, ValueError), match="length|steps|expects"):
        kernel(
            parameters,
            prepared,
            state,
            dn.func.Inputs(intra=state["integrator"]["v"].unsqueeze(0)),
            callbacks={},
        )
    with torch.no_grad():
        parameters[name].add_(1.0e-5)
    with pytest.raises(dn.func.FunctionalizationError, match="stale|changed|modified"):
        kernel(parameters, prepared, state, callbacks={})
    assert kernel.execution_report() is previous


def test_raw_population_callback_call_keeps_source_structure_checks():
    model, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    callbacks = _callbacks(functional)
    model._flag_rebuild = True
    with pytest.raises(dn.func.FunctionalizationError, match="pending structural"):
        bound(state, callbacks=callbacks)
