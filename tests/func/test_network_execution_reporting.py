"""Network dispatch reports share the existing tensor-free execution contract."""

from dataclasses import FrozenInstanceError, asdict
from types import MappingProxyType

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import _close, _loss, _mixed_network, _stim_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _case():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    return functional, tensors, prepared


@pytest.mark.parametrize("execution", ("scan", "unrolled"))
def test_network_capabilities_and_reports_are_immutable_and_tensor_free(execution):
    functional, tensors, prepared = _case()
    kernel = functional.compile_rollout_chunk(
        2, execution=execution, backend="aot_eager"
    )
    capabilities = kernel.capabilities
    assert isinstance(capabilities, dn.func.ExecutionCapabilities)
    assert capabilities.execution == execution
    assert capabilities.inference
    assert capabilities.compiled_reverse_mode == "first_order"
    assert capabilities.compiled_higher_order_reverse_mode == (
        "rejected" if execution == "scan" else "backend_limited"
    )
    assert capabilities.structured_inference == (
        "scan" if execution == "scan" else "none"
    )
    assert capabilities.structured_training is (execution == "scan")
    assert (
        capabilities.functional_transforms
        == capabilities.forward_mode
        == "eager_fallback"
    )
    assert kernel.execution_report() is None
    with pytest.raises(FrozenInstanceError):
        capabilities.execution = "default"

    actual, _ = kernel(tensors.parameters, prepared, tensors.state)
    expected, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, steps=2
    )
    _close(actual, expected)
    report = kernel.execution_report()
    assert isinstance(report, dn.func.ExecutionReport)
    assert report.requested_execution == execution
    assert report.strategy == f"compiled_{execution}"
    assert report.steps == 2 and report.grad_enabled and not report.callbacks
    assert report.backend == "aot_eager"
    assert all(
        value is None or type(value) in (str, bool, int)
        for value in asdict(report).values()
    )
    with pytest.raises(FrozenInstanceError):
        report.steps = 5


@pytest.mark.parametrize("callbacks_enabled", (False, True))
def test_network_tail_and_callback_dispatches_share_the_parent_report(
    callbacks_enabled,
):
    functional, tensors, _ = _case()
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder("cell.v")})
        if callbacks_enabled
        else None
    )
    with torch.no_grad():
        actual, _ = functional.run(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=5,
            chunklength=4,
            compiled_chunk=kernel,
            callbacks=callbacks,
        )
        expected, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=5,
        )
    _close(actual, expected)
    report = kernel.execution_report()
    assert report.steps == 1 and report.strategy == "compiled_scan"
    assert report.callbacks is callbacks_enabled
    assert not report.grad_enabled
    assert kernel._runner_chunk(1).execution_report() is report
    functional.run(
        tensors.parameters, tensors.constants, actual, steps=0, compiled_chunk=kernel
    )
    assert kernel.execution_report() is report


def test_network_checkpoint_replay_can_replace_the_shared_report():
    functional, tensors, _ = _case()
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder("cell.v")})
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        final, auxiliary = functional.run(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=5,
            chunklength=4,
            checkpoint_every=4,
            compiled_chunk=kernel,
            callbacks=callbacks,
        )
    original = kernel.execution_report()
    weight = next(iter(tensors.parameters["synapses"].values()))["weights"]
    gradient = torch.autograd.grad(
        _loss(final) + auxiliary["callbacks"]["trace"]["cell.v"].square().sum(), weight
    )[0]
    assert torch.isfinite(gradient).all()
    report = kernel.execution_report()
    assert report is not original
    assert (
        report.strategy == "compiled_scan" and report.callbacks and report.grad_enabled
    )
    assert report.steps in (1, 3)
    assert kernel._runner_chunk(report.steps).execution_report() is report


@pytest.mark.parametrize("transform", ("grad", "jacfwd", "vmap"))
def test_network_transform_fallback_reports_survive_a_later_compile_failure(transform):
    functional, tensors, prepared = _case()

    def reject_backend(_graph, _examples):
        raise RuntimeError("Network report compiler rejection")

    kernel = functional.compile_rollout_chunk(2, backend=reject_backend)

    def loss(voltage):
        state = dict(tensors.state)
        state["populations"] = dict(state["populations"])
        cell = dict(state["populations"]["cell"])
        cell["integrator"] = dict(cell["integrator"], v=voltage)
        state["populations"]["cell"] = cell
        final, _ = kernel(tensors.parameters, prepared, state)
        return final["populations"]["cell"]["integrator"]["v"].square().mean()

    voltage = tensors.state["populations"]["cell"]["integrator"]["v"].detach()
    if transform == "vmap":
        voltage = torch.stack((voltage, voltage + 1))
    actual = getattr(torch.func, transform)(loss)(voltage)
    assert torch.isfinite(actual).all()
    report = kernel.execution_report()
    assert report.strategy == "eager_transform_fallback"
    with pytest.raises(Exception, match="Network report compiler rejection"):
        kernel(tensors.parameters, prepared, tensors.state)
    assert kernel.execution_report() is report


def test_network_direct_forward_ad_report():
    functional, tensors, prepared = _case()

    def reject_backend(_graph, _examples):
        raise AssertionError("forward AD must use the authored eager recurrence")

    kernel = functional.compile_rollout_chunk(2, backend=reject_backend)
    voltage = tensors.state["populations"]["cell"]["integrator"]["v"].detach()
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(voltage, torch.ones_like(voltage))
        state = dict(tensors.state)
        state["populations"] = dict(state["populations"])
        cell = dict(state["populations"]["cell"])
        cell["integrator"] = dict(cell["integrator"], v=dual)
        state["populations"]["cell"] = cell
        final, _ = kernel(tensors.parameters, prepared, state)
        tangent = torch.autograd.forward_ad.unpack_dual(
            final["populations"]["cell"]["integrator"]["v"]
        ).tangent
        assert tangent is not None and torch.isfinite(tangent).all()
    report = kernel.execution_report()
    assert report.strategy == "eager_forward_ad_fallback"
    assert not report.grad_enabled


def test_network_disabled_compilation_reports_eager_execution_and_callable_backend():
    functional, tensors, prepared = _case()

    def backend(_graph, _examples):
        raise AssertionError("disabled kernels must not invoke backend")

    kernel = functional.compile_rollout_chunk(2, backend=backend, disable=True)
    kernel(tensors.parameters, prepared, tensors.state)
    report = kernel.execution_report()
    assert report.strategy == "eager_disabled" and report.reason
    assert report.backend == backend.__qualname__


class _RejectFinalize(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["clock"]["step"].clone(), None

    def update(self, carry, state, auxiliary):
        del state, auxiliary
        return carry + 1, None

    def finalize(self, carry, emissions):
        del emissions
        if bool(carry > 0):
            raise ValueError("Reject callback finalization")
        return carry


def test_network_failed_public_validation_and_callback_finalization_preserve_report():
    functional, tensors, prepared = _case()
    kernel = functional.compile_rollout_chunk(2, disable=True)
    kernel(tensors.parameters, prepared, tensors.state)
    original = kernel.execution_report()
    with pytest.raises(ValueError, match="component schema"):
        kernel(tensors.parameters, prepared, {})
    assert kernel.execution_report() is original
    callbacks = functional.make_callbacks({"reject": _RejectFinalize()})
    with pytest.raises(ValueError, match="Reject callback finalization"):
        kernel(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    assert kernel.execution_report() is original


def test_network_run_failed_callback_finalization_preserves_prior_report():
    functional, tensors, prepared = _case()
    kernel = functional.compile_rollout_chunk(2, disable=True)
    kernel(tensors.parameters, prepared, tensors.state)
    original = kernel.execution_report()
    assert original is not None and not original.callbacks
    callbacks = functional.make_callbacks({"reject": _RejectFinalize()})
    # A successful full chunk and tail both change the shared report before
    # finalization rejects the complete host invocation.
    with pytest.raises(ValueError, match="Reject callback finalization"):
        functional.run(
            *tensors,
            steps=3,
            chunklength=2,
            compiled_chunk=kernel,
            callbacks=callbacks,
        )
    assert kernel.execution_report() is original


def _read_only_tree(tree):
    if isinstance(tree, dict):
        return MappingProxyType(
            {name: _read_only_tree(value) for name, value in tree.items()}
        )
    if isinstance(tree, tuple):
        values = tuple(_read_only_tree(value) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    return tree


@pytest.mark.parametrize("stimulus", (False, True))
def test_compiled_network_read_only_packets_reuse_fresh_preparation_and_gradients(
    stimulus,
):
    model = _stim_network() if stimulus else _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    parameters = _read_only_tree(tensors.parameters)
    constants = _read_only_tree(tensors.constants)
    state = _read_only_tree(tensors.state)
    steps = 9 if stimulus else 4
    inputs = None
    if stimulus:
        inputs = dn.func.NetworkRolloutInput(
            netstim=functional.sample_draw_tape(state, steps)
        )
    kernel = functional.compile_rollout_chunk(steps, backend="aot_eager")
    weight = next(iter(parameters["synapses"].values()))["weights"]
    for _ in range(2):
        prepared = functional.prepare(parameters, constants)
        final, _ = kernel(parameters, prepared, state, inputs)
        expected, _ = functional.prepare_and_rollout(
            parameters, constants, state, inputs, steps=steps
        )
        _close(final, expected)
        observed = torch.autograd.grad(
            final["populations"]["cell"]["integrator"]["v"].square().sum(), weight
        )[0]
        oracle = torch.autograd.grad(
            expected["populations"]["cell"]["integrator"]["v"].square().sum(), weight
        )[0]
        torch.testing.assert_close(observed, oracle)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
        with torch.no_grad():
            weight.add_(0.0001)
    assert kernel.execution_report().strategy == "compiled_scan"
