"""Execution reports describe the dispatched kernel without retaining inputs."""

import gc
import warnings
import weakref
from dataclasses import FrozenInstanceError, asdict
from functools import partial

import pytest
import torch
from test_compiled_chunk import DT, _assert_tree_close, _model
from test_runner_stimulation_clock import _case as _stimulus_case

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._execution import ExecutionCapabilities, ExecutionReport

requires_scan_torch = pytest.mark.skipif(
    torch.__version__.split("+")[0] != "2.14.0",
    reason="scan compatibility is validated against PyTorch 2.14.0",
)


@pytest.fixture(autouse=True)
def _known_compiler_warnings():
    with torch_compiler_warning_context(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated.*",
            category=FutureWarning,
            module=r"torch\.jit\._script",
        )
        yield


def _case():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    return functional, tensors, prepared


def _graph_backend(observed):
    def backend(graph, _examples):
        observed.append({str(node.target) for node in graph.graph.nodes})
        return graph.forward

    return backend


def test_capabilities_and_reports_are_immutable_and_descriptive():
    functional, tensors, prepared = _case()
    kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
    capabilities = kernel.capabilities
    assert isinstance(capabilities, ExecutionCapabilities)
    assert capabilities.execution == "default"
    assert capabilities.compiled_reverse_mode == "first_order"
    assert capabilities.compiled_higher_order_reverse_mode == "backend_limited"
    assert capabilities.functional_transforms == "eager_fallback"
    assert capabilities.forward_mode == "eager_fallback"
    assert capabilities.structured_inference == "conditional_while_loop"
    assert not capabilities.structured_training
    assert kernel.execution_report() is None
    with pytest.raises(FrozenInstanceError):
        capabilities.execution = "scan"

    final, _ = kernel(tensors.parameters, prepared, tensors.state)
    expected, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, steps=2
    )
    _assert_tree_close(final, expected)
    report = kernel.execution_report()
    assert isinstance(report, ExecutionReport)
    assert report.strategy == "compiled_unrolled"
    assert report.backend == "aot_eager"
    assert report.grad_enabled
    with pytest.raises(FrozenInstanceError):
        report.steps = 100
    assert all(
        value is None or type(value) in (str, bool, int)
        for value in asdict(report).values()
    )


@pytest.mark.parametrize("steps", [1, 3, 4, 5])
def test_inference_report_matches_actual_graph_including_short_chunks(steps):
    functional, tensors, prepared = _case()
    observed = []
    kernel = functional.compile_rollout_chunk(steps, backend=_graph_backend(observed))
    with torch.no_grad():
        actual = kernel(tensors.parameters, prepared, tensors.state)
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=steps
        )
    _assert_tree_close(actual, expected)
    report = kernel.execution_report()
    has_loop = any("while_loop" in targets for targets in observed)
    assert has_loop == (steps >= 4)
    assert report.strategy == (
        "compiled_while_loop" if has_loop else "compiled_unrolled"
    )
    assert report.steps == steps and not report.grad_enabled
    assert report.reason is None if has_loop else report.reason is not None


def test_failed_capture_reports_unrolled_inference(monkeypatch):
    functional, tensors, prepared = _case()
    observed = []
    kernel = functional.compile_rollout_chunk(4, backend=_graph_backend(observed))

    def reject_capture(*_args):
        raise RuntimeError("unsupported capture for this test")

    monkeypatch.setattr(functional, "_capture_structured_step_graph", reject_capture)
    with torch.no_grad():
        kernel(tensors.parameters, prepared, tensors.state)
    report = kernel.execution_report()
    assert report.strategy == "compiled_unrolled"
    assert report.reason == "No captured inference step graph is available."
    assert observed and not any("while_loop" in targets for targets in observed)


def test_explicitly_disabled_compilation_is_not_reported_as_compiled():
    functional, tensors, prepared = _case()
    kernel = functional.compile_rollout_chunk(4, disable=True)
    with torch.no_grad():
        kernel(tensors.parameters, prepared, tensors.state)
    assert kernel.execution_report().strategy == "eager_disabled"


@pytest.mark.parametrize("transform", ["grad", "jacfwd", "vmap"])
def test_transform_fallback_report_and_failed_compile_preserve_last_success(transform):
    functional, tensors, prepared = _case()

    def reject_backend(_graph, _examples):
        raise RuntimeError("report test compiler failure")

    kernel = functional.compile_rollout_chunk(2, backend=reject_backend)

    def loss(voltage):
        state = dict(tensors.state)
        state["integrator"] = dict(state["integrator"], v=voltage)
        final, _ = kernel(tensors.parameters, prepared, state)
        return final["integrator"]["v"].square().mean()

    voltage = tensors.state["integrator"]["v"].detach()
    if transform == "vmap":
        voltage = torch.stack((voltage, voltage + 1))
    assert kernel.execution_report() is None
    actual = getattr(torch.func, transform)(loss)(voltage)
    assert torch.isfinite(actual).all()
    report = kernel.execution_report()
    assert report.strategy == "eager_transform_fallback"
    with pytest.raises(Exception, match="report test compiler failure"):
        kernel(tensors.parameters, prepared, tensors.state)
    assert kernel.execution_report() is report


def test_direct_forward_ad_report():
    functional, tensors, prepared = _case()

    def reject_backend(_graph, _examples):
        raise AssertionError("forward AD should not enter the compiler")

    kernel = functional.compile_rollout_chunk(2, backend=reject_backend)
    voltage = tensors.state["integrator"]["v"].detach()
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(voltage, torch.ones_like(voltage))
        state = dict(tensors.state)
        state["integrator"] = dict(state["integrator"], v=dual)
        final, _ = kernel(tensors.parameters, prepared, state)
        tangent = torch.autograd.forward_ad.unpack_dual(
            final["integrator"]["v"]
        ).tangent
        assert tangent is not None and torch.isfinite(tangent).all()
    assert kernel.execution_report().strategy == "eager_forward_ad_fallback"


@pytest.mark.parametrize("with_callbacks", [False, True])
def test_runner_tail_and_callback_dispatches_update_parent(with_callbacks):
    functional, tensors, prepared = _case()
    observed = []
    kernel = functional.compile_rollout_chunk(4, backend=_graph_backend(observed))
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        if with_callbacks
        else None
    )
    with torch.no_grad():
        dn.func.run(
            functional,
            partial(kernel, tensors.parameters, prepared),
            tensors.state,
            tstop=5 * DT,
            callbacks=callbacks,
        )
    report = kernel.execution_report()
    assert report.steps == 1
    assert report.callbacks is with_callbacks
    assert report.strategy == "compiled_unrolled"
    assert any("while_loop" in targets for targets in observed) is not with_callbacks
    if with_callbacks:
        callback_ref = weakref.ref(callbacks)
        del callbacks
        gc.collect()
        assert callback_ref() is None
        assert not kernel._callback_chunks
    else:
        assert kernel._runner_chunks[1].execution_report() is report


def test_inference_callback_body_is_unrolled_even_with_prewarmed_graph():
    functional, tensors, prepared = _case()
    assert functional.prewarm_structured_rollout()
    observed = []
    kernel = functional.compile_rollout_chunk(4, backend=_graph_backend(observed))
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
    with torch.no_grad():
        dn.func.run(
            functional,
            partial(kernel, tensors.parameters, prepared),
            tensors.state,
            tstop=4 * DT,
            callbacks=callbacks,
        )
    assert kernel.execution_report().strategy == "compiled_unrolled"
    assert kernel.execution_report().callbacks
    assert observed and not any("while_loop" in targets for targets in observed)


def test_checkpoint_replay_report_is_the_latest_completed_kernel_dispatch():
    functional, tensors, prepared = _case()
    kernel = functional.compile_rollout_chunk(4, backend="eager")
    state = dict(tensors.state)
    voltage = tensors.state["integrator"]["v"].detach().clone().requires_grad_()
    state["integrator"] = dict(state["integrator"], v=voltage)
    # Disable early stopping so each replayed kernel returns normally and is
    # eligible to replace the most-recent-success report.
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        final, _ = dn.func.longrun_checkpointed(
            functional,
            partial(kernel, tensors.parameters, prepared),
            state,
            6 * DT,
            chunklength=4,
        )
    assert kernel.execution_report().steps == 2
    gradient = torch.autograd.grad(final["integrator"]["v"].square().mean(), voltage)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    assert kernel.execution_report().steps == 4


@pytest.mark.parametrize("stimulus_kind", ["intra", "extra"])
def test_stepwise_bound_stimulation_reports_unrolled_inference(stimulus_kind):
    functional, tensors, parameters, prepared, _amplitude = _stimulus_case(
        stimulus_kind
    )
    observed = []
    kernel = functional.compile_rollout_chunk(4, backend=_graph_backend(observed))
    with torch.no_grad():
        dn.func.run(
            functional,
            partial(kernel, parameters, prepared),
            tensors.state,
            tstop=4 * DT,
        )
    report = kernel.execution_report()
    assert report.strategy == "compiled_unrolled"
    assert report.reason == "Bound waveforms are sampled at each accepted timestep."
    assert observed and not any("while_loop" in targets for targets in observed)


@requires_scan_torch
@pytest.mark.parametrize("with_callbacks", [False, True])
def test_scan_reports_dispatched_loop_and_distinct_capabilities(with_callbacks):
    functional, tensors, prepared = _case()
    observed = []
    kernel = functional.compile_rollout_chunk(
        3, execution="scan", backend=_graph_backend(observed)
    )
    assert kernel.capabilities.execution == "scan"
    assert kernel.capabilities.compiled_higher_order_reverse_mode == "rejected"
    assert kernel.capabilities.structured_inference == "scan"
    assert kernel.capabilities.structured_training
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        if with_callbacks
        else None
    )
    with torch.no_grad():
        dn.func.run(
            functional,
            partial(kernel, tensors.parameters, prepared),
            tensors.state,
            tstop=5 * DT,
            callbacks=callbacks,
        )
    report = kernel.execution_report()
    assert report.strategy == "compiled_scan"
    assert report.requested_execution == "scan"
    assert report.steps == 2 and report.callbacks is with_callbacks
    assert observed and all("scan" in targets for targets in observed)
