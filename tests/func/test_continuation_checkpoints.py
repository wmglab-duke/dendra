"""Portable runner checkpoints own detached data and preserve effective sources."""

from __future__ import annotations

import os
from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _drive,
    _joined,
    _plain,
    _runner,
    _snapshot,
)
from tests.func.test_lowering_api import DT, G, _population, _target, _voltage

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _tensors(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, Mapping):
        for value in tree.values():
            yield from _tensors(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _tensors(value)


def _frames(continuation):
    return (
        continuation.state,
        continuation.parameters,
        continuation.constants,
        None
        if continuation.callback_state is None
        else continuation.callback_state.carries,
    )


def _plain_checkpoint(tree):
    if isinstance(tree, torch.Tensor):
        assert type(tree) is torch.Tensor
        assert tree.grad_fn is None and tree.is_leaf
        assert not tree.is_inference()
    elif type(tree) is dict:
        assert all(type(key) in (str, int) for key in tree)
        for value in tree.values():
            _plain_checkpoint(value)
    elif type(tree) in (tuple, list):
        for value in tree:
            _plain_checkpoint(value)
    else:
        assert type(tree) in (str, int, float, bool, type(None)), type(tree)


def _copy_checkpoint(tree, memo=None):
    memo = {} if memo is None else memo
    if isinstance(tree, torch.Tensor):
        if id(tree) not in memo:
            with torch.inference_mode(False), torch.no_grad():
                value = tree.detach().clone()
                value.requires_grad_(tree.requires_grad)
            memo[id(tree)] = value
        return memo[id(tree)]
    if isinstance(tree, Mapping):
        return {key: _copy_checkpoint(value, memo) for key, value in tree.items()}
    if type(tree) in (list, tuple):
        values = [_copy_checkpoint(value, memo) for value in tree]
        return tuple(values) if type(tree) is tuple else values
    return tree


def _disk(packet, path):
    torch.save(packet, path)
    return torch.load(path, weights_only=True)


def _leaf_flags(tree):
    return [value.requires_grad for value in _tensors(tree)]


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("callbacks", (False, True))
@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_weights_only_disk_checkpoint_rebuilds_runner_with_effective_overrides_and_suffix_gradients(
    network, callbacks, backend, tmp_path
):
    options = dict(callbacks=callbacks, backend=backend, disable=backend == "eager")
    runner, receiver = _runner(network, **options), _runner(network, **options)
    parameters = _plain(runner.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = 1.3 * upstream
    _assign(parameters, network, replacement)
    prefix, first = runner(parameters=parameters)
    source = first["continuation"]
    defaults = _snapshot(
        (receiver.parameters, receiver.constants, receiver.initial_state)
    )
    packet = runner.export_continuation(source)
    _plain_checkpoint(packet)
    loaded = _disk(packet, tmp_path / "continuation.pt")
    restored = receiver.restore_continuation(loaded)
    assert isinstance(restored, dn.func.Continuation)
    selected = _target(restored.parameters, network)
    assert selected is not replacement
    assert selected.grad_fn is None and selected.is_leaf and selected.requires_grad
    torch.testing.assert_close(selected, replacement)
    assert selected.data_ptr() != replacement.data_ptr()
    if network:
        assert selected is restored.parameters["populations"]["right"][G]
    if callbacks:
        assert (
            restored.callback_state.plan_token is not source.callback_state.plan_token
        )
        assert restored.callback_state.plan_token is receiver.callbacks._state_token
    else:
        assert restored.callback_state is None
    current, inputs = _drive(receiver, network, 3, requires_grad=True)
    actual, last = receiver(restored, inputs=inputs)
    expected, reference = receiver.functional.prepare_and_rollout(
        restored.parameters,
        restored.constants,
        restored.state,
        inputs,
        steps=3,
        callbacks=receiver.callbacks,
        callback_state=restored.callback_state,
    )
    _close(actual, expected)
    actual_loss = _voltage(actual, network).square().sum()
    reference_loss = _voltage(expected, network).square().sum()
    if callbacks:
        _close(last["callbacks"].outputs, reference["callbacks"].outputs)
        assert last["callbacks"]["trace"]["left.v" if network else "v"].shape[0] == 3
        actual_loss = (
            actual_loss
            + last["callbacks"]["trace"]["left.v" if network else "v"].square().mean()
        )
        reference_loss = (
            reference_loss
            + reference["callbacks"]["trace"]["left.v" if network else "v"]
            .square()
            .mean()
        )
    observed = torch.autograd.grad(
        actual_loss, (selected, current, upstream), allow_unused=True
    )
    gradients = torch.autograd.grad(reference_loss, (selected, current))
    assert observed[2] is None  # Export cuts the prefix graph.
    _close(observed[:2], gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed[:2]
    )
    assert upstream.grad is None
    _close((receiver.parameters, receiver.constants, receiver.initial_state), defaults)
    _close(runner.functional.extract().state, runner.initial_state)
    _close(restored.state, prefix)


@pytest.mark.parametrize("network", (False, True))
def test_checkpoint_roundtrip_retains_joined_recording_and_cumulative_callback_state(
    network, tmp_path
):
    runner, receiver = (
        _runner(network, inference=True),
        _runner(network, inference=True),
    )
    initial = _snapshot(runner.initial_state)
    expected, reference = runner.functional.prepare_and_rollout(
        runner.parameters,
        runner.constants,
        initial,
        steps=6,
        callbacks=runner.callbacks,
    )
    _, first = runner()
    restored = receiver.restore_continuation(
        _disk(
            runner.export_continuation(first["continuation"]), tmp_path / "recording.pt"
        )
    )
    actual, last = receiver(restored)
    _close(actual, expected)
    _close(_joined(first, last), reference["callbacks"]["trace"])
    _close(last["callbacks"]["count"], reference["callbacks"]["count"])
    assert _joined(first, last)["left.v" if network else "v"].shape[0] == 7


@pytest.mark.parametrize("network", (False, True))
def test_export_restore_inside_inference_mode_owns_ordinary_storage_and_trainability(
    network,
):
    runner, receiver = _runner(network), _runner(network)
    _, auxiliary = runner()
    original = auxiliary["continuation"]
    frozen = _snapshot(_frames(original))
    with torch.inference_mode():
        packet = runner.export_continuation(original)
        restored = receiver.restore_continuation(packet)
    _plain_checkpoint(packet)
    _close(_frames(restored), frozen)
    assert _leaf_flags(restored.parameters) == _leaf_flags(original.parameters)
    assert _leaf_flags(restored.constants) == _leaf_flags(original.constants)
    for value in _tensors(_frames(restored)):
        assert type(value) is torch.Tensor and not value.is_inference()
        assert value.grad_fn is None
    packet_storage = {value.data_ptr() for value in _tensors(packet) if value.numel()}
    source_storage = {
        value.data_ptr() for value in _tensors(_frames(original)) if value.numel()
    }
    assert all(
        value.data_ptr() not in source_storage
        for value in _tensors(packet)
        if value.numel()
    )
    assert all(
        value.data_ptr() not in packet_storage
        for value in _tensors(_frames(restored))
        if value.numel()
    )
    with torch.no_grad():
        _target(restored.parameters, network).add_(0.004)
    _close(_frames(original), frozen)


@pytest.mark.parametrize("operation", ("export", "restore"))
@pytest.mark.parametrize("context", ("grad", "forward_ad"))
def test_continuation_checkpoint_boundary_rejects_function_transforms(
    operation, context
):
    runner = _runner(False)
    _, auxiliary = runner()
    continuation = auxiliary["continuation"]
    packet = runner.export_continuation(continuation)

    def work(value):
        runner.export_continuation(
            continuation
        ) if operation == "export" else runner.restore_continuation(packet)
        return value.square()

    value = torch.tensor(0.5)
    with pytest.raises(dn.func.FunctionalizationError, match="outside.*torch.compile"):
        if context == "grad":
            torch.func.grad(work)(value)
        else:
            with torch.autograd.forward_ad.dual_level():
                work(torch.autograd.forward_ad.make_dual(value, torch.ones_like(value)))
    runner.restore_continuation(packet)


def test_cold_export_restore_does_not_initialize_prepare_sample_execute_or_change_report(
    monkeypatch,
):
    runner, receiver = _runner(False), _runner(False)
    _, auxiliary = runner()
    receiver()
    source_report, receiver_report = (
        runner.execution_report(),
        receiver.execution_report(),
    )
    before = _snapshot(
        (
            runner.initial_state,
            receiver.initial_state,
            runner.parameters,
            receiver.parameters,
        )
    )
    rng = torch.random.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError(
            "cold checkpoint operations must not execute model machinery"
        )

    for program in (runner, receiver):
        for name in (
            "initialize_host",
            "prepare",
            "sample_runtime_draws",
            "prepare_and_rollout",
        ):
            monkeypatch.setattr(program.functional, name, reject)
    restored = receiver.restore_continuation(
        runner.export_continuation(auxiliary["continuation"])
    )
    _close(restored.state, auxiliary["continuation"].state)
    assert runner.execution_report() is source_report
    assert receiver.execution_report() is receiver_report
    _close(
        (
            runner.initial_state,
            receiver.initial_state,
            runner.parameters,
            receiver.parameters,
        ),
        before,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize(
    "kind",
    (
        "version",
        "boolean_version",
        "format",
        "missing",
        "extra",
        "tensor_type",
        "tensor_shape",
        "tensor_dtype",
        "grad_count",
        "grad_type",
    ),
)
def test_malformed_checkpoint_rejection_leaves_packet_and_receiver_unchanged(kind):
    runner, receiver = _runner(False), _runner(False)
    _, auxiliary = runner()
    packet = _copy_checkpoint(runner.export_continuation(auxiliary["continuation"]))
    if kind == "version":
        packet["version"] += 1
    elif kind == "boolean_version":
        packet["version"] = True
    elif kind == "format":
        packet["format"] = "other-format"
    elif kind == "missing":
        packet.pop("owner")
    elif kind == "extra":
        packet["unknown"] = None
    elif kind == "tensor_type":
        packet["frames"]["tensors"][0] = 0
    elif kind == "tensor_shape":
        packet["frames"]["tensors"][0] = packet["frames"]["tensors"][0].unsqueeze(0)
    elif kind == "tensor_dtype":
        packet["frames"]["tensors"][0] = packet["frames"]["tensors"][0].to(
            torch.float32
        )
    elif kind == "grad_count":
        packet["frames"]["requires_grad"] = []
    else:
        packet["frames"]["requires_grad"][0] = 1
    frozen = _copy_checkpoint(packet)
    defaults = _snapshot(
        (receiver.parameters, receiver.constants, receiver.initial_state)
    )
    report = receiver.execution_report()
    with pytest.raises(
        (TypeError, ValueError, KeyError, dn.func.FunctionalizationError)
    ):
        receiver.restore_continuation(packet)
    _assert_packet_equal(packet, frozen)
    _close((receiver.parameters, receiver.constants, receiver.initial_state), defaults)
    assert receiver.execution_report() is report


@pytest.mark.parametrize(
    "changed", ("state", "parameters", "constants", "aliases", "callback")
)
def test_export_checks_receiving_contracts_before_publishing_bad_continuations(changed):
    runner = _runner(True)
    _, auxiliary = runner()
    source = auxiliary["continuation"]
    state, parameters, constants = (
        _plain(source.state),
        _plain(source.parameters),
        _plain(source.constants),
    )
    callback_state = source.callback_state
    if changed == "state":
        state["populations"]["left"]["integrator"]["v"] = state["populations"]["left"][
            "integrator"
        ]["v"].reshape(-1)
    elif changed == "parameters":
        parameters["populations"]["left"][G] = parameters["populations"]["left"][
            G
        ].unsqueeze(0)
    elif changed == "constants":
        constants["populations"]["left"].pop("diam")
    elif changed == "aliases":
        parameters["populations"]["right"][G] = parameters["populations"]["left"][
            G
        ].clone()
    else:
        _, other = _runner(True)()
        callback_state = other["continuation"].callback_state
    invalid = dn.func.Continuation(state, parameters, constants, callback_state)
    with pytest.raises(
        (TypeError, ValueError, KeyError, dn.func.FunctionalizationError)
    ):
        runner.export_continuation(invalid)


@pytest.mark.parametrize(
    "changed", ("kind", "dt", "mode", "shape", "callbacks", "threshold")
)
def test_portable_checkpoint_rejects_incompatible_receivers(changed):
    with_callbacks = changed != "dt"
    runner = _runner(False, callbacks=with_callbacks)
    _, auxiliary = runner()
    packet = runner.export_continuation(auxiliary["continuation"])
    if changed == "kind":
        receiver = _runner(True)
    elif changed == "callbacks":
        receiver = _runner(False, callbacks=False)
    elif changed == "threshold":
        specification = _callbacks(False)
        specification["count"] = dn.func.APCount(node_check=[0], threshold=-58.5, dt=DT)
        receiver = dn.compile(
            _population(),
            dt=DT,
            steps=3,
            callbacks=specification,
            inference=False,
            backend="eager",
            disable=True,
        )
    else:
        model = _population()
        dt = 2 * DT if changed == "dt" else DT
        if changed == "mode":
            model.eval()
        if changed == "shape":
            model = dn.SingleCompartment(
                N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
            )
            from dendra.models.mod import pas

            model.insert(pas.rename("leak"), g=0.001, e=-45.0)
        model.initialize(dt)
        receiver = dn.compile(
            model,
            dt=dt,
            steps=3,
            callbacks=_callbacks(False) if with_callbacks else None,
            inference=changed == "mode",
            backend="eager",
            disable=True,
        )
    frozen = _copy_checkpoint(packet)
    with pytest.raises(
        (TypeError, ValueError, KeyError, dn.func.FunctionalizationError)
    ):
        receiver.restore_continuation(packet)
    _assert_packet_equal(packet, frozen)


def test_callback_free_checkpoint_does_not_silently_start_receiving_analysis():
    source = _runner(False, callbacks=False)
    _, auxiliary = source()
    packet = source.export_continuation(auxiliary["continuation"])
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="callback"):
        _runner(False).restore_continuation(packet)


class _CustomSum(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        return state["integrator"]["v"].sum(), None

    def update(self, carry, state, auxiliary):
        return carry + state["integrator"]["v"].sum(), None

    def finalize(self, carry, emissions):
        return carry


def test_custom_callback_execution_remains_available_but_portable_checkpointing_rejects_explicitly():
    runner = dn.compile(
        _population(),
        dt=DT,
        steps=2,
        callbacks={"custom": _CustomSum()},
        inference=False,
        backend="eager",
        disable=True,
    )
    _, auxiliary = runner()
    with pytest.raises(
        dn.func.FunctionalizationError, match="custom|built-in|unsupported"
    ):
        runner.export_continuation(auxiliary["continuation"])


@pytest.mark.mps
@pytest.mark.parametrize("network", (False, True))
def test_mps_checkpoint_roundtrip_keeps_devices_and_suffix_training_gradients(
    network, tmp_path
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    from tests.func.test_runtime_extra_backends import (
        _environment as backend_environment,
    )

    with backend_environment("mps"):
        source = _runner(network, backend="aot_eager", disable=False, steps=2)
        receiver = _runner(network, backend="aot_eager", disable=False, steps=2)
        _, first = source()
        restored = receiver.restore_continuation(
            _disk(
                source.export_continuation(first["continuation"]), tmp_path / "mps.pt"
            )
        )
        target = _target(restored.parameters, network)
        assert target.device.type == "mps" and target.dtype is torch.float32
        assert target.requires_grad and target.is_leaf and target.grad_fn is None
        state, _ = receiver(restored)
        gradient = torch.autograd.grad(
            _voltage(state, network).square().mean(), target
        )[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for value in _tensors(restored.state):
            assert not value.is_inference()
