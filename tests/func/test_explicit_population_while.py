"""Raw Population while requests share strict inference and bounded host scheduling."""

from __future__ import annotations

import io
import os
from dataclasses import asdict

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population_while import _PopulationWhileExecutor
from tests.func.test_runtime_extra import DT, _close, _field
from tests.func.test_runtime_extra import _model as _passive
from tests.func.test_runtime_noise import DT as NOISE_DT
from tests.func.test_runtime_noise import _population as _noisy
from tests.func.test_stimulation_rng_sharing import _clone, _slice

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


def _case():
    model = _passive(batch=())
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    return model, functional, tensors, prepared


def _capture_backend(graphs):
    def capture(graph, _examples):
        graphs.append(graph)
        return graph.forward

    return capture


def _has_while(graphs):
    return any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for graph in graphs
        for node in graph.graph.nodes
    )


@pytest.mark.parametrize("steps", (3, 4, 5))
def test_direct_while_reports_actual_loop_and_short_tail_without_recording(steps):
    model, functional, tensors, prepared = _case()
    graphs = []
    kernel = functional.compile_rollout_chunk(
        steps, execution="while", backend=_capture_backend(graphs)
    )
    inputs = dn.func.Inputs(
        ve=torch.stack([_field(model) * (0.1 + index / 20) for index in range(steps)])
    )
    original = _clone((tensors, inputs))
    assert kernel.execution_report() is None
    assert kernel.capabilities.execution == "while"
    assert kernel.capabilities.compiled_reverse_mode == "unsupported"
    assert kernel.capabilities.compiled_higher_order_reverse_mode == "rejected"
    assert not kernel.capabilities.structured_training
    with torch.inference_mode() if steps == 4 else torch.no_grad():
        actual, auxiliary = kernel(
            tensors.parameters, prepared, tensors.state, inputs=inputs
        )
        expected, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, inputs=inputs, steps=steps
        )
    _close(actual, expected)
    _close((tensors, inputs), original)
    assert "callbacks" not in auxiliary
    assert graphs and _has_while(graphs) is (steps >= 4)
    report = kernel.execution_report()
    assert report.requested_execution == "while"
    assert report.strategy == (
        "compiled_while_loop" if steps >= 4 else "compiled_unrolled"
    )
    assert report.steps == steps and not report.grad_enabled and not report.callbacks
    assert (report.reason is None) is (steps >= 4)
    assert all(
        value is None or type(value) in (str, bool, int)
        for value in asdict(report).values()
    )


def test_direct_aot_while_callbacks_resume_without_duplicate_samples():
    _model, functional, tensors, prepared = _case()
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    kernel = functional.compile_rollout_chunk(4, execution="while", backend="aot_eager")
    current = tensors.state["integrator"]["v"].new_full((9, *_model.shape), 3e-7)
    with torch.no_grad():
        first, prefix = kernel(
            tensors.parameters,
            prepared,
            tensors.state,
            inputs=dn.func.Inputs(intra=current[:4]),
            callbacks=callbacks,
        )
        actual, suffix = kernel(
            tensors.parameters,
            prepared,
            first,
            inputs=dn.func.Inputs(intra=current[4:8]),
            callbacks=callbacks,
            callback_state=prefix["callbacks"].state,
        )
        expected, reference = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            inputs=dn.func.Inputs(intra=current[:8]),
            steps=8,
            callbacks=callbacks,
        )
    _close(actual, expected)
    for name, value in reference["callbacks"]["trace"].items():
        observed = torch.cat(
            (prefix["callbacks"]["trace"][name], suffix["callbacks"]["trace"][name])
        )
        _close(observed, value)
        assert suffix["callbacks"]["trace"][name].shape[0] == 4
    assert kernel.execution_report().requested_execution == "while"
    assert kernel.execution_report().strategy == "compiled_while_loop"
    assert kernel.execution_report().callbacks
    with torch.no_grad():
        bounded, bounded_aux = dn.func.run(
            kernel.bind(tensors),
            tensors.state,
            inputs=dn.func.Inputs(intra=current),
            steps=9,
            callbacks=callbacks,
        )
        expected, reference = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            inputs=dn.func.Inputs(intra=current),
            steps=9,
            callbacks=callbacks,
        )
    _close(bounded, expected)
    _close(bounded_aux["callbacks"]["trace"], reference["callbacks"]["trace"])
    assert bounded_aux["callbacks"]["trace"]["v"].shape[0] == 10
    assert bounded_aux["continuation"].callback_state is not None
    assert kernel.execution_report().requested_execution == "while"
    assert kernel.execution_report().steps == 1 and kernel.execution_report().callbacks
    assert kernel.execution_report().strategy == "compiled_unrolled"


def test_bound_while_noise_tails_and_portable_continuation_preserve_rng():
    model = _noisy()
    functional, tensors = dn.func.make_functional(model, dt=NOISE_DT)
    kernel = functional.compile_rollout_chunk(4, execution="while", backend="eager")
    bound = kernel.bind(tensors)
    original = _clone(tensors)
    global_rng = torch.get_rng_state().clone()
    tape = functional.sample_runtime_draws(tensors.state, steps=14)
    frozen = _clone(tape)
    with torch.no_grad():
        automatic, auxiliary = dn.func.run(bound, tensors.state, steps=9)
        explicit, prefix = dn.func.run(
            bound, tensors.state, steps=9, draws=_slice(tape, 0, 9)
        )
        expected, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=14,
            draws=tape,
        )
    _close(automatic, explicit)
    assert auxiliary["continuation"].callback_state is None
    assert "callbacks" not in auxiliary
    report = kernel.execution_report()
    assert report.requested_execution == "while"
    assert report.strategy == "compiled_unrolled" and report.steps == 1
    assert kernel._runner_chunks[1].execution_report() is report
    saved = io.BytesIO()
    torch.save(dn.func.export_continuation(functional, prefix["continuation"]), saved)
    saved.seek(0)
    receiving, _ = dn.func.make_functional(_noisy(), dt=NOISE_DT)
    restored = dn.func.restore_continuation(
        receiving, torch.load(saved, weights_only=True)
    )
    receiver = receiving.compile_rollout_chunk(5, execution="while", backend="eager")
    with torch.no_grad():
        actual, resumed = dn.func.run(receiver.bind(restored), restored, steps=5)
    _close(actual, expected)
    assert resumed["continuation"].callback_state is None
    assert receiver.execution_report().strategy == "compiled_while_loop"
    _close(tensors, original)
    _close(tape, frozen)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("stimulus_kind", ("intra", "extra"))
def test_bound_while_waveforms_use_native_clock_and_reuse_the_effective_child(
    monkeypatch, stimulus_kind
):
    from tests.func.test_runner_stimulation_clock import DT as CLOCK_DT
    from tests.func.test_runner_stimulation_clock import STEPS
    from tests.func.test_runner_stimulation_clock import _case as clock_case

    functional, tensors, parameters, prepared, _amplitude = clock_case(stimulus_kind)
    kernel = functional.compile_rollout_chunk(STEPS, execution="while", backend="eager")
    executions = []
    kernel_for = _PopulationWhileExecutor.kernel_for

    def observe(executor, *args, **kwargs):
        if executor._functional is functional:
            executions.append(executor)
        return kernel_for(executor, *args, **kwargs)

    monkeypatch.setattr(_PopulationWhileExecutor, "kernel_for", observe)
    with torch.no_grad():
        expected = tensors.state
        for _ in range(STEPS + 1):
            expected, _ = functional.step(parameters, prepared, expected)
        actual, auxiliary = dn.func.run(
            kernel.bind(parameters, prepared), tensors.state, steps=STEPS + 1
        )
    torch.testing.assert_close(actual["integrator"]["v"], expected["integrator"]["v"])
    torch.testing.assert_close(actual["clock"]["t"], expected["clock"]["t"])
    assert actual["clock"]["t"].item() > tensors.state["clock"]["t"].item() + CLOCK_DT
    assert "callbacks" not in auxiliary
    assert "continuation" not in auxiliary
    full = [executor for executor in executions if executor._steps == STEPS]
    assert full and all(executor is full[0] for executor in full)
    assert full[0]._stepwise_stimulation
    assert kernel.execution_report().steps == 1


@pytest.mark.parametrize("record", (False, True))
def test_explicit_while_capture_failures_preserve_last_report_and_backend(
    monkeypatch, record
):
    _model, functional, tensors, prepared = _case()
    graphs = []
    kernel = functional.compile_rollout_chunk(
        4, execution="while", backend=_capture_backend(graphs)
    )
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        if record
        else None
    )

    def unavailable(*args, **kwargs):
        return None, "sentinel explicit Population while capture"

    with monkeypatch.context() as scoped:
        scoped.setattr(_PopulationWhileExecutor, "kernel_for", unavailable)
        with (
            torch.no_grad(),
            pytest.raises(dn.func.FunctionalizationError, match="sentinel explicit"),
        ):
            kernel(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    assert graphs == [] and kernel.execution_report() is None
    with torch.no_grad():
        kernel(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    report, count = kernel.execution_report(), len(graphs)
    assert count and _has_while(graphs)
    with monkeypatch.context() as scoped:
        scoped.setattr(_PopulationWhileExecutor, "kernel_for", unavailable)
        with (
            torch.no_grad(),
            pytest.raises(dn.func.FunctionalizationError, match="sentinel explicit"),
        ):
            kernel(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    assert len(graphs) == count
    assert kernel.execution_report() is report


def test_explicit_while_rejects_training_but_admitted_runner_kernel_retains_gradients():
    model, functional, tensors, prepared = _case()
    strict = functional.compile_rollout_chunk(4, execution="while", backend="eager")
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
    for collection in (None, callbacks):
        with pytest.raises(
            (ValueError, dn.func.FunctionalizationError), match="inference|scan"
        ):
            strict(tensors.parameters, prepared, tensors.state, callbacks=collection)
    assert strict.execution_report() is None
    with torch.no_grad():
        strict(tensors.parameters, prepared, tensors.state)
    prior = strict.execution_report()
    for collection in (None, callbacks):
        with pytest.raises(
            (ValueError, dn.func.FunctionalizationError), match="inference|scan"
        ):
            strict(tensors.parameters, prepared, tensors.state, callbacks=collection)
    assert strict.execution_report() is prior
    program = dn.compile(model, dt=DT, steps=4, execution="while", backend="aot_eager")
    program()
    assert program.kernel.capabilities.execution == "default"
    assert program.kernel.capabilities.compiled_reverse_mode == "first_order"
    voltage = program.initial_state["integrator"]["v"].detach().clone().requires_grad_()
    state = {
        **program.initial_state,
        "integrator": {**program.initial_state["integrator"], "v": voltage},
    }
    prepared = program.functional.prepare(program.parameters, program.constants)
    observed, _ = program.kernel(program.parameters, prepared, state)
    expected, _ = program.functional.rollout(
        program.parameters, prepared, state, steps=4
    )
    actual_grad = torch.autograd.grad(observed["integrator"]["v"].sum(), voltage)[0]
    expected_grad = torch.autograd.grad(expected["integrator"]["v"].sum(), voltage)[0]
    _close(observed, expected)
    _close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
    assert program.kernel.execution_report().strategy == "compiled_unrolled"
    assert program.kernel.execution_report().requested_execution == "default"


@pytest.mark.parametrize("transform", ("grad", "jacfwd", "forward_ad"))
def test_explicit_while_keeps_transform_fallback_without_compiler_dispatch(transform):
    _model, functional, tensors, prepared = _case()

    def unexpected(*args, **kwargs):
        pytest.fail("explicit transform reached the compiler")

    kernel = functional.compile_rollout_chunk(4, execution="while", backend=unexpected)

    def evaluate(voltage, executable):
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": voltage},
        }
        if executable:
            final, _ = kernel(tensors.parameters, prepared, state)
        else:
            final, _ = functional.rollout(tensors.parameters, prepared, state, steps=4)
        return final["integrator"]["v"].square().mean()

    voltage = tensors.state["integrator"]["v"].detach()
    if transform == "forward_ad":
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(
                voltage, torch.ones_like(voltage)
            )
            expected = torch.autograd.forward_ad.unpack_dual(
                evaluate(dual, False)
            ).tangent
            actual = torch.autograd.forward_ad.unpack_dual(evaluate(dual, True)).tangent
            _close(actual, expected)
        strategy = "eager_forward_ad_fallback"
    else:
        operation = getattr(torch.func, transform)
        expected = operation(lambda value: evaluate(value, False))(voltage)
        actual = operation(lambda value: evaluate(value, True))(voltage)
        _close(actual, expected)
        strategy = "eager_transform_fallback"
    assert kernel.execution_report().requested_execution == "while"
    assert kernel.execution_report().strategy == strategy


def test_disabled_explicit_while_is_an_eager_diagnostic_dispatch():
    _model, functional, tensors, prepared = _case()
    kernel = functional.compile_rollout_chunk(4, execution="while", disable=True)
    with torch.no_grad():
        actual, _ = kernel(tensors.parameters, prepared, tensors.state)
        expected, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=4
        )
    _close(actual, expected)
    assert kernel.execution_report().requested_execution == "while"
    assert kernel.execution_report().strategy == "eager_disabled"


def test_explicit_while_accepts_custom_reducers_for_a_short_chunk():
    class Custom(dn.func.FunctionalCallback):
        def initialize(self, state, auxiliary):
            return state["integrator"]["v"].new_zeros(()), None

        def update(self, carry, state, auxiliary):
            return carry + state["integrator"]["v"].sum(), None

        def finalize(self, carry, emissions):
            return carry

    _model, functional, tensors, prepared = _case()
    callbacks = functional.make_callbacks({"custom": Custom()})
    kernel = functional.compile_rollout_chunk(3, execution="while", backend="eager")
    with torch.no_grad():
        actual, auxiliary = kernel(
            tensors.parameters, prepared, tensors.state, callbacks=callbacks
        )
        expected, reference = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=3, callbacks=callbacks
        )
    _close(actual, expected)
    _close(auxiliary["callbacks"]["custom"], reference["callbacks"]["custom"])
    assert kernel.execution_report().strategy == "compiled_unrolled"
    assert kernel.execution_report().callbacks


@pytest.mark.mps
def test_mps_explicit_population_while_matches_eager_with_no_callbacks():
    from tests.func.test_runtime_extra_backends import _environment as environment
    from tests.func.test_runtime_extra_backends import _field as field
    from tests.func.test_runtime_extra_backends import _model as model_factory

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with environment("mps"):
        model = model_factory("mps")
        functional, tensors = dn.func.make_functional(model, dt=DT)
        kernel = functional.compile_rollout_chunk(
            4, execution="while", backend="aot_eager"
        )
        inputs = dn.func.Inputs(
            ve=torch.stack([field(model) * (0.1 + index / 20) for index in range(4)])
        )
        with torch.no_grad():
            prepared = functional.prepare(tensors.parameters, tensors.constants)
            actual, auxiliary = kernel(
                tensors.parameters, prepared, tensors.state, inputs=inputs
            )
            expected, _ = functional.rollout(
                tensors.parameters, prepared, tensors.state, inputs=inputs, steps=4
            )
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
        assert "callbacks" not in auxiliary
        assert kernel.execution_report().requested_execution == "while"
        assert kernel.execution_report().strategy == "compiled_while_loop"
