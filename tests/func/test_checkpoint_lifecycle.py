"""Portable checkpoints distinguish fresh initialization from completed calls."""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _callbacks, _close, _runner, _snapshot
from tests.func.test_continuation_checkpoints import (
    _copy_checkpoint,
    _disk,
    _plain_checkpoint,
)
from tests.func.test_lowering_api import G, _target, _voltage

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _export(program, packet, route):
    if route == "method":
        return program.export_continuation(packet)
    if route == "runner":
        return dn.func.export_continuation(program, packet)
    return dn.func.export_continuation(
        program.functional, packet, callbacks=program.callbacks
    )


def _restore(program, packet, route):
    if route == "method":
        return program.restore_continuation(packet)
    if route == "runner":
        return dn.func.restore_continuation(program, packet)
    return dn.func.restore_continuation(
        program.functional, packet, callbacks=program.callbacks
    )


def _trace(auxiliary, network):
    return auxiliary["callbacks"]["trace"]["left.v" if network else "v"]


def _assert_owned_sources(actual, expected, network):
    _close(actual.state, expected.state)
    _close(actual.parameters, expected.parameters)
    _close(actual.constants, expected.constants)
    selected, original = (
        _target(actual.parameters, network),
        _target(expected.parameters, network),
    )
    assert selected is not original and selected.data_ptr() != original.data_ptr()
    assert selected.is_leaf and selected.grad_fn is None and selected.requires_grad
    assert not selected.is_inference()
    if network:
        assert selected is actual.parameters["populations"]["right"][G]


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_private_lifecycle_keeps_public_continuation_fields_and_constructor(network):
    assert tuple(field.name for field in fields(dn.func.Continuation)) == (
        "state",
        "parameters",
        "constants",
        "callback_state",
    )
    assert tuple(inspect.signature(dn.func.Continuation).parameters) == (
        "state",
        "parameters",
        "constants",
        "callback_state",
    )
    program = _runner(network)
    initial = dn.func.initialize(program)
    assert initial._lifecycle == "unstarted"
    manual = dn.func.Continuation(initial.state, initial.parameters, initial.constants)
    assert manual._lifecycle == "unspecified"
    assert manual.callback_state is None
    with pytest.raises(TypeError):
        dn.func.Continuation(
            initial.state, initial.parameters, initial.constants, lifecycle="unstarted"
        )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("route", ("runner", "method", "explicit"))
def test_initialized_checkpoint_defers_callbacks_and_starts_recording_once(
    network, route, tmp_path, monkeypatch
):
    source, receiver = _runner(network), _runner(network)
    calls = []
    initialize = receiver.callbacks._initialize

    def counted(*args, **kwargs):
        calls.append(None)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(receiver.callbacks, "_initialize", counted)
    initial = dn.func.initialize(source)
    assert initial.callback_state is None and initial._lifecycle == "unstarted"
    saved = _export(source, initial, route)
    assert saved["version"] == 2 and saved["lifecycle"] == "unstarted"
    assert saved["callbacks"] is None
    _plain_checkpoint(saved)
    restored = _restore(receiver, _disk(saved, tmp_path / "initial.pt"), route)
    assert restored._lifecycle == "unstarted"
    assert restored.callback_state is None and calls == []
    _assert_owned_sources(restored, initial, network)
    # Restore remains a cold operation and does not change receiving defaults.
    defaults = _snapshot(
        (receiver.parameters, receiver.constants, receiver.initial_state)
    )
    _, first = receiver(restored, steps=2)
    assert calls == [None]
    assert first["continuation"]._lifecycle == "active"
    assert _trace(first, network).shape[0] == 3
    active = _restore(
        receiver,
        _disk(_export(receiver, first["continuation"], route), tmp_path / "active.pt"),
        route,
    )
    assert active._lifecycle == "active" and active.callback_state is not None
    assert active.callback_state.plan_token is receiver.callbacks._state_token
    final, last = receiver(active, steps=1)
    assert calls == [None] and _trace(last, network).shape[0] == 1
    expected, reference = receiver.functional.prepare_and_rollout(
        active.parameters,
        active.constants,
        active.state,
        steps=1,
        callbacks=receiver.callbacks,
        callback_state=active.callback_state,
    )
    _close(final, expected)
    _close(last["callbacks"], reference["callbacks"])
    _close((receiver.parameters, receiver.constants, receiver.initial_state), defaults)
    targets = dn.func.inspect_sources(active).trainable_leaves
    gradients = torch.autograd.grad(
        _voltage(final, network).square().sum(), targets, allow_unused=True
    )
    assert any(value is not None and value.abs().sum() > 0 for value in gradients)
    assert all(value is None or torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("direction", ("runner_to_plan", "plan_to_runner"))
def test_explicit_initialization_and_runner_codecs_share_unstarted_provenance(
    network, direction, tmp_path
):
    source, receiver = _runner(network), _runner(network)
    tensors = source.functional.extract()._replace(
        parameters=source.parameters,
        constants=source.constants,
        state=source.initial_state,
    )
    initial = dn.func.initialize(source.functional, tensors)
    assert initial._lifecycle == "unstarted"
    export_route, restore_route = (
        ("runner", "explicit")
        if direction == "runner_to_plan"
        else ("explicit", "runner")
    )
    restored = _restore(
        receiver,
        _disk(_export(source, initial, export_route), tmp_path / "cross.pt"),
        restore_route,
    )
    assert restored._lifecycle == "unstarted" and restored.callback_state is None
    bound = receiver.functional.bind(restored)
    _, first = dn.func.run(bound, restored, steps=1, callbacks=receiver.callbacks)
    assert first["continuation"]._lifecycle == "active"
    assert _trace(first, network).shape[0] == 2


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_zero_step_completion_is_active_and_preserves_callback_phase(
    network, tmp_path, monkeypatch
):
    program = _runner(network)
    calls = []
    initialize = program.callbacks._initialize

    def counted(*args, **kwargs):
        calls.append(None)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(program.callbacks, "_initialize", counted)
    initial = dn.func.initialize(program)
    state, first = program(initial, steps=0)
    assert calls == [None] and _trace(first, network).shape[0] == 1
    _close(state, initial.state)
    packet = first["continuation"]
    assert packet._lifecycle == "active" and packet.callback_state is not None
    restored = dn.func.restore_continuation(
        program,
        _disk(dn.func.export_continuation(program, packet), tmp_path / "zero.pt"),
    )
    _, again = program(restored, steps=0)
    assert calls == [None] and _trace(again, network).shape[0] == 0
    _, last = program(again["continuation"], steps=1)
    assert calls == [None] and _trace(last, network).shape[0] == 1


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("steps", (0, 1))
def test_active_callback_free_checkpoint_is_strict_even_at_zero_steps(network, steps):
    source, receiver = _runner(network, callbacks=False), _runner(network)
    _, auxiliary = source(steps=steps)
    active = auxiliary["continuation"]
    assert active._lifecycle == "active" and active.callback_state is None
    saved = dn.func.export_continuation(source, active)
    assert saved["lifecycle"] == "active" and saved["callbacks"] is None
    for route in ("runner", "method", "explicit"):
        with pytest.raises(dn.func.FunctionalizationError, match="callback"):
            _restore(receiver, saved, route)
        with pytest.raises(dn.func.FunctionalizationError, match="callback"):
            _export(receiver, active, route)
    # Strictness belongs to the portable boundary. Attaching analysis directly
    # to an in-memory continuation retains its established public behavior.
    _, first = receiver(active, steps=0)
    assert _trace(first, network).shape[0] == 1


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_manual_continuation_remains_unspecified_and_strict(network):
    program = _runner(network)
    initial = dn.func.initialize(program)
    manual = dn.func.Continuation(initial.state, initial.parameters, initial.constants)
    assert manual._lifecycle == "unspecified"
    with pytest.raises(dn.func.FunctionalizationError, match="callback"):
        dn.func.export_continuation(program, manual)
    saved = dn.func.export_continuation(program.functional, manual)
    assert saved["lifecycle"] == "unspecified"
    with pytest.raises(dn.func.FunctionalizationError, match="callback"):
        dn.func.restore_continuation(program, saved)
    restored = dn.func.restore_continuation(program.functional, saved)
    assert restored._lifecycle == "unspecified"


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("callbacks", (False, True))
def test_version_one_remains_readable_without_unstarted_relaxation(
    network, callbacks, tmp_path
):
    source, receiver = (
        _runner(network, callbacks=callbacks),
        _runner(network, callbacks=callbacks),
    )
    _, auxiliary = source(steps=0)
    saved = dn.func.export_continuation(source, auxiliary["continuation"])
    saved.pop("lifecycle")
    saved["version"] = 1
    restored = dn.func.restore_continuation(receiver, _disk(saved, tmp_path / "v1.pt"))
    assert restored._lifecycle == "unspecified"
    _assert_owned_sources(restored, auxiliary["continuation"], network)
    assert (restored.callback_state is None) is (not callbacks)
    opposite = _runner(network, callbacks=not callbacks)
    with pytest.raises(dn.func.FunctionalizationError, match="callback"):
        dn.func.restore_continuation(opposite, saved)
    # A v1 packet at t=0 may not be inferred to mean fresh initialization.
    if not callbacks:
        with pytest.raises(dn.func.FunctionalizationError, match="callback"):
            dn.func.export_continuation(opposite, restored)


@pytest.mark.parametrize("operation", ("export", "restore"))
def test_runner_targets_reject_competing_explicit_callback_owners(operation):
    program = _runner(False)
    initial = dn.func.initialize(program)
    saved = dn.func.export_continuation(program, initial)
    function = (
        (
            lambda callbacks: dn.func.export_continuation(
                program, initial, callbacks=callbacks
            )
        )
        if operation == "export"
        else (
            lambda callbacks: dn.func.restore_continuation(
                program, saved, callbacks=callbacks
            )
        )
    )
    for callbacks in (program.callbacks, _callbacks(False), object()):
        with pytest.raises((TypeError, ValueError), match="callback"):
            function(callbacks)
    # Explicit None has the same meaning as the canonical API's default.
    function(None)


@pytest.mark.parametrize("lifecycle", (None, True, 1, [], {}, "future"))
def test_invalid_lifecycle_rejects_before_frames_or_authored_hooks(
    lifecycle, monkeypatch
):
    program = _runner(False)
    saved = dn.func.export_continuation(program, dn.func.initialize(program))
    saved["lifecycle"] = lifecycle
    before = _copy_checkpoint(saved)
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    report = program.execution_report()

    def reject(*args, **kwargs):
        raise AssertionError(
            "invalid lifecycle must reject before owner/frame validation"
        )

    monkeypatch.setattr(program.functional, "_validate_source", reject)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="lifecycle"):
        dn.func.restore_continuation(program, saved)
    _assert_packet_equal(saved, before)
    _close((program.parameters, program.constants, program.initial_state), defaults)
    assert program.execution_report() is report


@pytest.mark.parametrize("changed", ("missing", "unknown_key", "v1_extra", "version"))
def test_versioned_lifecycle_keys_are_checked_before_receiving_target(changed):
    program = _runner(False, callbacks=False)
    saved = dn.func.export_continuation(program, dn.func.initialize(program))
    if changed == "missing":
        saved.pop("lifecycle")
    elif changed == "unknown_key":
        saved["unknown"] = None
    elif changed == "v1_extra":
        saved["version"] = 1
    else:
        saved["version"] = True
    with pytest.raises(ValueError, match="keys|version"):
        dn.func.restore_continuation(object(), saved, callbacks=object())


def test_unstarted_checkpoint_cannot_contain_active_callback_payload():
    program = _runner(False)
    _, auxiliary = program(steps=0)
    saved = dn.func.export_continuation(program, auxiliary["continuation"])
    saved["lifecycle"] = "unstarted"
    with pytest.raises(
        dn.func.FunctionalizationError, match="callback|unstarted|lifecycle"
    ):
        dn.func.restore_continuation(program, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_compiled_runner_checkpoints_preserve_deferred_start_and_resume(
    network, execution, tmp_path
):
    program = _runner(
        network,
        execution=execution,
        inference=execution == "while",
        steps=4,
        chunk_steps=4,
        backend="aot_eager",
        disable=False,
    )
    initial = dn.func.initialize(program)
    restored = dn.func.restore_continuation(
        program,
        _disk(dn.func.export_continuation(program, initial), tmp_path / "initial.pt"),
    )
    _, first = program(restored)
    assert program.execution_report() is not None
    assert first["continuation"]._lifecycle == "active"
    assert _trace(first, network).shape[0] == 5
    active = dn.func.restore_continuation(
        program,
        _disk(
            dn.func.export_continuation(program, first["continuation"]),
            tmp_path / "active.pt",
        ),
    )
    final, resumed = program(active)
    assert resumed["continuation"]._lifecycle == "active"
    assert _trace(resumed, network).shape[0] == 4
    if execution == "scan":
        gradients = torch.autograd.grad(
            _voltage(final, network).square().sum(),
            dn.func.inspect_sources(active).trainable_leaves,
            allow_unused=True,
        )
        assert any(value is not None and value.abs().sum() > 0 for value in gradients)
        assert all(value is None or torch.isfinite(value).all() for value in gradients)
