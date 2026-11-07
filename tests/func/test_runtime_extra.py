"""Runtime extracellular slots reuse a compiled Population across contact recipes."""

from __future__ import annotations

import inspect
import io
import math

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population import FunctionalPopulation
from dendra.models.mod import pas
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_stimulation import (
    _assert_module_unchanged,
)
from tests.func.test_stimulation import _module_snapshot as _waveform_snapshot

DT = 0.01
STEPS = 8
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


def _model(*, dtype=torch.float64, batch=2):
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-65.0,
        dtype=dtype,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=1.0e-4, e=-65.0)
    if batch:
        model.batch(batch)
    model.initialize()
    model.t.fill_(0.175)
    model.train()
    return model


def _field(model, *, reverse=False):
    value = torch.linspace(
        -1.25, 1.75, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)
    return value.flip(-1) if reverse else value


def _runner(model, *, steps=STEPS, chunk_steps=3, **keywords):
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        extra=dn.VarInput(),
        chunk_steps=chunk_steps,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=keywords.pop("backend", "eager"),
        **keywords,
    )


def _trace(result):
    return dict(result[1]["callbacks"]["trace"])


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


def _reference(runner, *, extra=None, ve=None, steps=None):
    functional = runner.functional
    prepared = functional.prepare(runner.parameters, runner.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    return dn.func.run(
        functional.bind(runner.parameters, prepared),
        runner.initial_state,
        steps=runner._steps if steps is None else steps,
        extra=extra,
        inputs=None if ve is None else dn.func.RolloutInput(ve=ve),
        callbacks=callbacks,
    )


def _analytic_tape(runner, extra, steps):
    # Independent sine/cosine formula, sampled by the recurrent clock rather
    # than start + arange * dt. All samples are away from timing gates.
    time = runner.initial_state["clock"]["t"]
    dt = runner.constants["dt"].to(time)
    frames = []
    for _ in range(steps):
        contributions = []
        for (field, waveform), function in zip(
            extra, (torch.sin, torch.cos), strict=True
        ):
            sampled_time = time.to(field)
            value = waveform.amp * function(
                2 * math.pi * waveform.freq * sampled_time + waveform.phase
            )
            contributions.append(field * value)
        frames.append(sum(contributions))
        time = time + dt
    return torch.stack(frames)


def _differentiable_extra(model, scale_value):
    scale = model.v.new_tensor(scale_value, requires_grad=True)
    field = _field(model) * scale
    first = dn.sin(amp=0.18, freq=0.9, phase=0.2)
    second = dn.cos(amp=-0.11, freq=0.55, phase=-0.15)
    extra = [(field, first), (_field(model, reverse=True), second)]
    targets = (scale, first.amp, first.freq, first.phase, second.amp)
    assert not field.is_leaf
    return extra, targets


def _loss(result):
    voltage = _trace(result)["v"]
    weights = torch.linspace(0.7, 1.3, voltage.shape[-1], dtype=voltage.dtype)
    return (voltage * weights).square().mean() + 0.1 * voltage.sin().mean()


def _gradients(result, targets):
    gradients = torch.autograd.grad(_loss(result), targets)
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all(torch.count_nonzero(value) > 0 for value in gradients)
    return gradients


def test_runtime_extra_marker_is_exported_and_chunk_steps_is_explicit():
    assert dn.VarInput is dn.func.VarInput
    assert "VarInput" in dn.__all__
    assert "VarInput" in dn.func.__all__
    for factory in (dn.compile, dn.func.CompiledRunner):
        parameter = inspect.signature(factory).parameters["chunk_steps"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


@pytest.mark.parametrize("chunk_steps", (1, 2, 3))
def test_runtime_population_while_requires_a_full_loop_chunk(chunk_steps):
    with pytest.raises(ValueError, match="chunk_steps.*four"):
        _runner(_model(), execution="while", chunk_steps=chunk_steps)


def test_runtime_while_rechecks_source_after_replaced_sampler(monkeypatch):
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward

    runner = _runner(_model(), execution="while", chunk_steps=4, backend=backend)

    def mutate_source(state, steps):
        assert steps == 4
        runner.functional._population._flag_rebuild = True
        return None

    monkeypatch.setattr(runner.functional, "sample_noise_draw_tape", mutate_source)
    with pytest.raises(dn.func.FunctionalizationError, match="structur|rebuild"):
        runner(extra=None)
    assert not compilations


def test_runtime_extra_container_cannot_change_source_before_capture():
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward

    runner = _runner(_model(), backend=backend)

    class Contacts:
        def __iter__(self):
            runner.functional._population._flag_rebuild = True
            yield _field(runner.functional._population), dn.sin(amp=0.2)

    with pytest.raises(dn.func.FunctionalizationError, match="structur|rebuild"):
        runner(extra=Contacts())
    assert not compilations


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_runtime_extra_without_callbacks_preserves_initialization_policy(
    initialization,
):
    model = _model(batch=())
    extra = (_field(model), dn.sin(amp=0.2, freq=0.8, phase=0.3))
    options = dict(
        dt=DT,
        steps=5,
        initialization=initialization,
        backend="eager",
    )
    runtime = dn.compile(model, extra=dn.VarInput(), chunk_steps=2, **options)
    bound = dn.compile(model, extra=extra, **options)
    actual, auxiliary = runtime(extra=extra)
    expected, _ = bound()
    _close(actual, expected)
    assert "callbacks" not in auxiliary


@pytest.mark.parametrize("chunk_steps", [True, False, 0, -1, 2.5, "4"])
def test_invalid_chunk_steps_fails_before_functionalization(monkeypatch, chunk_steps):
    import dendra.func._convenience as convenience

    def must_not_lower(*args, **keywords):
        pytest.fail("invalid chunk_steps must fail before functionalization")

    monkeypatch.setattr(convenience, "make_functional", must_not_lower)
    with pytest.raises(ValueError, match="chunk_steps.*positive integer"):
        dn.compile(
            object(), dt=DT, steps=STEPS, extra=dn.VarInput(), chunk_steps=chunk_steps
        )


def test_chunk_steps_bounds_ordinary_and_runtime_compiled_widths():
    model = _model()
    ordinary = dn.compile(model, dt=DT, steps=STEPS, chunk_steps=4, backend="eager")
    assert ordinary.kernel.steps == 4
    assert _runner(model, steps=5, chunk_steps=20).kernel.steps == 5
    assert _runner(model, steps=73, chunk_steps=None).kernel.steps == 64


def test_runtime_extra_is_required_but_explicit_none_is_zero_stimulation():
    runner = _runner(_model())
    with pytest.raises(TypeError, match="extra"):
        runner()
    actual = runner(extra=None)
    zeros = runner.initial_state["integrator"]["v"].new_zeros(
        (STEPS, *runner.initial_state["integrator"]["v"].shape)
    )
    expected = _reference(runner, ve=zeros)
    _close(actual[0], expected[0])
    _close(_trace(actual), _trace(expected))
    assert not runner.functional.extra.enabled
    assert not any(name.startswith("stimulation.extra.") for name in runner.parameters)


@pytest.mark.parametrize("bound", [False, True])
def test_nonruntime_runner_rejects_call_local_extra_even_none(bound):
    model = _model()
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        extra=(_field(model), dn.constant(value=0.2)) if bound else None,
        backend="eager",
    )
    with pytest.raises(TypeError, match="extra|VarInput"):
        runner(extra=None)


def test_runtime_extra_rejects_an_explicit_ve_source():
    model = _model()
    runner = _runner(model)
    inputs = dn.func.RolloutInput(ve=model.v.new_zeros((STEPS, *model.shape)))
    with pytest.raises(ValueError, match="ve"):
        runner(extra=(_field(model), dn.constant(value=0.2)), inputs=inputs)


def test_changing_contacts_and_none_reuses_the_model_graph_and_preparation(monkeypatch):
    graphs, preparations = [], []
    original = FunctionalPopulation.prepare

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    def prepare(functional, parameters, constants):
        preparations.append(functional)
        return original(functional, parameters, constants)

    monkeypatch.setattr(FunctionalPopulation, "prepare", prepare)
    model = _model()
    runner = _runner(model, chunk_steps=4, backend=backend)
    first = (_field(model), dn.sin(amp=0.18, freq=0.9, phase=0.2))
    first_result = runner(extra=first)
    warmed_graphs, warmed_preparations = len(graphs), len(preparations)
    assert warmed_graphs > 0
    assert warmed_preparations > 0
    cases = (
        (_field(model), dn.mono_rect(amp=0.3, delay=0.19, pw=0.025)),
        [
            (_field(model), dn.cos(amp=0.24, freq=0.5)),
            (_field(model, reverse=True), dn.constant(value=-0.12)),
        ],
        None,
        (_field(model) * 1.2, dn.sin(amp=0.27, freq=0.7, phase=-0.1)),
    )
    outputs = [runner(extra=extra) for extra in cases]
    repeated = runner(extra=first)
    assert len(graphs) == warmed_graphs
    assert len(preparations) == warmed_preparations
    _close(_trace(repeated), _trace(first_result))
    assert any(
        not torch.equal(_trace(result)["v"], _trace(first_result)["v"])
        for result in outputs
    )
    # Check numerical results after counting so the independent preparation
    # in each reference does not contribute to the reuse assertion.
    for extra, actual in zip(cases, outputs, strict=True):
        expected = _reference(runner, extra=extra)
        _close(actual[0], expected[0])
        _close(_trace(actual), _trace(expected))


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_runtime_multicontact_matches_native_steps_including_a_short_tail(execution):
    model, native = _model(), _model()
    runner = _runner(model, steps=7, chunk_steps=4, execution=execution)
    extra = [
        (_field(model), dn.sin(amp=0.18, freq=0.9, phase=0.2)),
        (_field(model, reverse=True), dn.cos(amp=-0.11, freq=0.55, phase=-0.15)),
    ]
    source = _module_snapshot(model)
    fields = [field.clone() for field, _ in extra]
    waveforms = [_waveform_snapshot(waveform) for _, waveform in extra]
    actual = runner(extra=extra)
    frames = [{"v": native.v.clone(), "t": native.t.clone()}]
    with torch.no_grad():
        for _ in range(7):
            native.step(dt=DT, extra=extra)
            frames.append({"v": native.v.clone(), "t": native.t.clone()})
    expected = {
        name: torch.stack([frame[name] for frame in frames]) for name in ("v", "t")
    }
    _close(_trace(actual), expected)
    _close(actual[0]["integrator"]["v"], native.v)
    assert not actual[0]["integrator"]["v"].requires_grad
    _assert_module_snapshot(model, source)
    for (field, waveform), saved_field, saved_waveform in zip(
        extra, fields, waveforms, strict=True
    ):
        torch.testing.assert_close(field, saved_field, rtol=0, atol=0)
        _assert_module_unchanged(waveform, saved_waveform)


@pytest.mark.parametrize("checkpoint_every", [None, 5])
def test_runtime_field_and_waveform_gradients_match_an_explicit_tape(checkpoint_every):
    model = _model()
    runner = _runner(model, inference=False, checkpoint_every=checkpoint_every)
    source = _module_snapshot(model)
    for scale_value in (0.8, 1.2):
        extra, targets = _differentiable_extra(model, scale_value)
        saved_fields = [field.detach().clone() for field, _ in extra]
        saved_waveforms = [_waveform_snapshot(waveform) for _, waveform in extra]
        actual = runner(extra=extra)
        actual_gradients = _gradients(actual, targets)
        reference_extra, reference_targets = _differentiable_extra(model, scale_value)
        expected = _reference(runner, ve=_analytic_tape(runner, reference_extra, STEPS))
        _close(actual[0], expected[0])
        _close(_trace(actual), _trace(expected))
        _close(actual_gradients, _gradients(expected, reference_targets))
        for (field, waveform), saved_field, saved_waveform in zip(
            extra, saved_fields, saved_waveforms, strict=True
        ):
            torch.testing.assert_close(field.detach(), saved_field, rtol=0, atol=0)
            _assert_module_unchanged(waveform, saved_waveform)
            assert all(parameter.grad is None for parameter in waveform.parameters())
    _assert_module_snapshot(model, source)


def test_tensor_temporal_contacts_slice_the_global_tape_and_preserve_gradients():
    model = _model()
    runner = _runner(model, steps=7, inference=False, checkpoint_every=5)
    gain = model.v.new_tensor(0.7, requires_grad=True)
    field = _field(model) * gain
    first = torch.linspace(0.1, 0.4, 7, dtype=model.dtype(), requires_grad=True)
    second = torch.linspace(-0.2, 0.3, 14, dtype=model.dtype()).reshape(2, 1, 7)
    extra = [(field, first), (_field(model, reverse=True), second)]
    actual = runner(extra=extra)
    actual_gradients = _gradients(actual, (gain, first))
    reference_gain = gain.detach().clone().requires_grad_()
    reference_first = first.detach().clone().requires_grad_()
    reference_field = _field(model) * reference_gain
    ve = reference_field.unsqueeze(0) * reference_first.reshape(7, 1, 1, 1) + _field(
        model, reverse=True
    ).unsqueeze(0) * second.movedim(-1, 0).unsqueeze(-1)
    expected = _reference(runner, ve=ve)
    _close(actual[0], expected[0])
    _close(_trace(actual), _trace(expected))
    _close(actual_gradients, _gradients(expected, (reference_gain, reference_first)))
    with pytest.raises(ValueError, match="samples|timepoints|steps"):
        runner(extra=(field, first[:-1]))


def test_checkpointed_continuation_keeps_waveform_clock_callbacks_and_prefix_gradients():
    model = _model()
    runner = _runner(model, inference=False, checkpoint_every=5)
    extra, targets = _differentiable_extra(model, 0.8)
    expected = _reference(
        runner, ve=_analytic_tape(runner, extra, 2 * STEPS), steps=2 * STEPS
    )
    expected_gradients = _gradients(expected, targets)
    extra, targets = _differentiable_extra(model, 0.8)
    middle, first = runner(extra=extra)
    final, second = runner(middle, extra=extra, callback_state=first["callbacks"].state)
    joined = {
        name: torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        for name in ("v", "t")
    }
    actual = final, {"callbacks": {"trace": joined}}
    assert joined["v"].shape[0] == 2 * STEPS + 1
    _close(final, expected[0])
    _close(joined, _trace(expected))
    _close(_gradients(actual, targets), expected_gradients)


def test_serialized_continuation_restores_state_and_callback_tensor_carry():
    model = _model()
    runner = _runner(model)
    extra = (_field(model), dn.sin(amp=0.18, freq=0.9, phase=0.2))
    middle, first = runner(extra=extra)
    # Callback metadata belongs to this plan. Serialize its tensor leaves
    # alongside simulation carry, retaining that plan's PyTree specification.
    leaves, spec = pytree.tree_flatten((middle, first["callbacks"].state))
    checkpoint = io.BytesIO()
    torch.save(leaves, checkpoint)
    checkpoint.seek(0)
    restored_state, restored_callbacks = pytree.tree_unflatten(
        torch.load(checkpoint, weights_only=True), spec
    )
    final, second = runner(
        restored_state, extra=extra, callback_state=restored_callbacks
    )
    expected = _reference(runner, extra=extra, steps=2 * STEPS)
    _close(final, expected[0])
    for name in ("v", "t"):
        joined = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        assert joined.shape[0] == 2 * STEPS + 1
        _close(joined, _trace(expected)[name])


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_runtime_waveform_uses_the_accepted_float32_clock_at_a_pulse_edge(execution):
    model = _model(dtype=torch.float32, batch=None)
    model.t.fill_(1000.0)
    with dn.ctx(DTYPE=torch.float32):
        waveform = dn.mono_rect(amp=1.0, delay=1000.070068359375, pw=0.001)
    extra = (_field(model), waveform)
    runner = _runner(
        model,
        execution=execution,
        inference=execution == "while",
        checkpoint_every=2 if execution == "scan" else None,
        chunk_steps=4,
    )
    expected = _reference(runner, extra=extra)
    actual = runner(extra=extra)
    native = _model(dtype=torch.float32, batch=None)
    native.t.fill_(1000.0)
    recorder = dn.Recorder(("v", "t"))
    native.run(steps=STEPS, dt=DT, extra=extra, callbacks=[recorder])
    native_trace = {
        name: torch.from_numpy(recorder.numpy(name)).clone() for name in ("v", "t")
    }
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(_trace(actual), _trace(expected))
    torch.testing.assert_close(_trace(actual), native_trace)
    torch.testing.assert_close(actual[0]["integrator"]["v"], native.v)
    torch.testing.assert_close(actual[0]["clock"]["t"], native.t)
    if execution == "scan":
        expected_gradient = torch.autograd.grad(_loss(expected), waveform.amp)[0]
        actual_gradient = torch.autograd.grad(_loss(actual), waveform.amp)[0]
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient) == 0
        torch.testing.assert_close(actual_gradient, expected_gradient)
    unstimulated = runner(extra=None)
    torch.testing.assert_close(
        actual[0]["integrator"]["v"], unstimulated[0]["integrator"]["v"], rtol=0, atol=0
    )
    # Native.step intentionally samples at its accepted clock instead of the
    # whole-run invocation grid. This same pulse remains active there.
    stepped = _model(dtype=torch.float32, batch=None)
    stepped.t.fill_(1000.0)
    for _ in range(STEPS):
        stepped.step(dt=DT, extra=extra)
    assert not torch.equal(stepped.v, unstimulated[0]["integrator"]["v"])
    if execution == "scan":
        stepped_gradient = torch.autograd.grad(stepped.v.square().mean(), waveform.amp)[
            0
        ]
        assert torch.isfinite(stepped_gradient).all()
        assert torch.count_nonzero(stepped_gradient) > 0


def test_randomized_runtime_poisson_is_rejected_without_consuming_rng():
    model = _model()
    runner = _runner(model)
    generator = torch.Generator().manual_seed(17)
    waveform = dn.mono_rect(amp=0.2, pw=0.045).poisson(
        interval=0.055,
        n=5,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    saved_generator = generator.get_state().clone()
    saved_global_rng = torch.get_rng_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="Randomized Poisson"):
        runner(extra=(_field(model), waveform))
    torch.testing.assert_close(generator.get_state(), saved_generator, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), saved_global_rng, rtol=0, atol=0)
