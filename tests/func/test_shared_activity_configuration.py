"""Native activity configurations preserve queries with explicit pure carry."""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_bound_continuation import _case
from tests.func.test_callback_checkpoints import _assert_packet_equal, _clone
from tests.func.test_continuation import _close, _joined, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_lowering_api import DT, _network, _population, _target, _voltage
from tests.func.test_shared_threshold_callbacks import (
    _configuration,
    _execute,
    _manual,
    _threshold_case,
    _voltage_sequence,
)

pytestmark = pytest.mark.cpu
_FAMILIES = (
    (dn.callbacks.Active, dn.func.Active, {}, dn.func.ActiveAL, "ActiveAL"),
    (
        dn.callbacks.ActiveAL,
        dn.func.ActiveAL,
        {"at_least": 2},
        dn.func.ActiveAL,
        "ActiveAL",
    ),
    (
        dn.callbacks.ActiveALCount,
        dn.func.ActiveALCount,
        {"at_least": 2},
        dn.func.ActiveALCount,
        "ActiveALCount",
    ),
)


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


def _initial(tensors, parameters=None, constants=None):
    return dn.func.Continuation(
        tensors.state,
        tensors.parameters if parameters is None else parameters,
        tensors.constants if constants is None else constants,
    )


def _native(callback, voltage):
    model = SimpleNamespace(
        v=voltage[0],
        shape=tuple(voltage.shape[1:]),
        nc=voltage.shape[-1],
        device=lambda: voltage.device,
    )
    callback.pre_loop_hook(model)
    for frame in voltage:
        model.v = frame
        callback.post_step_hook(model)
    return callback


def _measure(count, *, total):
    return count.sum(-1) if total else torch.count_nonzero(count, dim=-1)


@pytest.mark.parametrize("native_type,pure_type,options,canonical,kind", _FAMILIES)
@pytest.mark.parametrize("inv", (False, True))
def test_native_activity_configuration_matches_independent_counts_and_native_query(
    native_type, pure_type, options, canonical, kind, inv
):
    functional, _ = _threshold_case()
    configured = native_type(**_configuration(), **options, inv=inv)
    collection = functional.make_callbacks({"active": configured})
    callback = collection._plans[0].callback
    assert type(callback) is canonical
    assert not collection._plans[0].validate_updates and callback.partition is None
    voltage = _voltage_sequence((2, 3, *functional.shape)).requires_grad_()
    carry, observed = _execute(callback, voltage)
    below, count, _ = _manual(voltage, torch.tensor([0, 0, 3, 3]))
    expected = _measure(count, total=kind == "ActiveALCount") >= options.get(
        "at_least", 1
    )
    if inv:
        expected = ~expected
    _close(observed, expected)
    _close(carry[0], below)
    _close(carry[1], count)
    assert carry[1].dtype == torch.int32 and carry[2].item() == len(voltage)
    assert observed.dtype == torch.bool and not observed.requires_grad
    reference = _native(native_type(**_configuration(), **options, inv=inv), voltage)
    _close(observed, reference.is_active())
    assert configured.record is configured.state_cache is None and configured.i == 0
    assert configured.node_check == [0, 0, -1, -1]


@pytest.mark.parametrize("target", (1, 2, np.int64(3)))
def test_positive_integral_targets_share_queries_and_normalized_checkpoint_metadata(
    target, tmp_path
):
    functional, tensors = _threshold_case()
    voltage = _voltage_sequence(functional.shape)
    for family, pure_family, total in (
        (dn.callbacks.ActiveAL, dn.func.ActiveAL, False),
        (dn.callbacks.ActiveALCount, dn.func.ActiveALCount, True),
    ):
        configured = family(**_configuration(), at_least=target)
        assert type(configured.at_least) is int and configured.at_least == target
        callbacks = functional.make_callbacks({"active": configured})
        pure_configured = pure_family(**_configuration(), at_least=target)
        assert (
            type(pure_configured.at_least) is int and pure_configured.at_least == target
        )
        pure = functional.make_callbacks({"active": pure_configured})
        callback = callbacks._plans[0].callback
        carry, observed = _execute(callback, voltage)
        pure_carry, pure_observed = _execute(pure._plans[0].callback, voltage)
        expected = _measure(carry[1], total=total) >= target
        _close(observed, expected)
        _close(pure_observed, expected)
        _close(pure_carry, carry)
        _close(
            observed,
            _native(family(**_configuration(), at_least=target), voltage).is_active(),
        )
        _, initial = dn.func.run(
            functional, _initial(tensors), steps=0, callbacks=callbacks
        )
        saved = callbacks.export_state(initial["callbacks"].state)
        _, pure_initial = dn.func.run(
            functional, _initial(tensors), steps=0, callbacks=pure
        )
        _assert_packet_equal(saved, pure.export_state(pure_initial["callbacks"].state))
        loaded = _disk(saved, tmp_path / f"{family.__name__}.pt")
        captured = loaded["configuration"]["callbacks"][0]["callback"]["fields"][
            "at_least"
        ]
        assert type(captured) is int and captured == target and loaded["version"] == 1
        _assert_packet_equal(loaded, saved)
        receiver = functional.make_callbacks(
            {"active": family(**_configuration(), at_least=target)}
        )
        restored = receiver.restore_state(loaded)
        _close(restored.carries, initial["callbacks"].state.carries)
        _close(pure.restore_state(loaded).carries, restored.carries)
        configured.at_least = 100
        assert callback.at_least == target
        assert not _execute(
            functional.make_callbacks({"active": configured})._plans[0].callback,
            voltage,
        )[1].any()


@pytest.mark.parametrize(
    "target,error",
    (
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (False, TypeError),
        (1.5, TypeError),
        (2.0, TypeError),
        (-1.0, TypeError),
        (float("nan"), TypeError),
        (float("inf"), TypeError),
        ("2", TypeError),
        (torch.tensor(2), TypeError),
    ),
)
def test_native_and_functional_count_targets_share_constructor_and_edit_validation(
    target, error
):
    functional, _ = _threshold_case()
    explicit = (
        dn.callbacks.ActiveAL,
        dn.callbacks.ActiveALCount,
        dn.func.ActiveAL,
        dn.func.ActiveALCount,
    )
    for family in explicit:
        with pytest.raises(error, match="positive integer"):
            family(node_check=[0], at_least=target)
    for family in (*explicit, dn.callbacks.Active, dn.func.Active):
        configured = family(node_check=[0])
        configured.at_least = target
        with pytest.raises(error, match="positive integer"):
            functional.make_callbacks({"active": configured})
        if isinstance(configured, dn.callbacks.APCount):
            configured.record = torch.zeros((1, 1), dtype=torch.int32)
            with pytest.raises(error, match="positive integer"):
                configured.is_active()


@pytest.mark.parametrize(
    "inv", (0, 1, "", "invert", torch.tensor(False), torch.tensor(True))
)
def test_native_and_functional_inversion_use_captured_scalar_truth(inv):
    functional, _ = _threshold_case()
    voltage = _voltage_sequence(functional.shape)
    expected = _native(
        dn.callbacks.ActiveAL(**_configuration(), at_least=1, inv=inv), voltage
    ).is_active()
    for family in (dn.callbacks.ActiveAL, dn.func.ActiveAL):
        configured = family(**_configuration(), at_least=1, inv=inv)
        callback = functional.make_callbacks({"active": configured})._plans[0].callback
        assert type(callback.inv) is bool and callback.inv == bool(inv)
        _close(_execute(callback, voltage)[1], expected)
        with pytest.raises((RuntimeError, ValueError), match="ambiguous|Boolean|bool"):
            functional.make_callbacks(
                {"active": family(**_configuration(), inv=torch.tensor([True, False]))}
            )


def test_edited_active_target_and_reused_native_sources_are_captured_without_history():
    functional, _ = _threshold_case()
    nodes = torch.tensor([0, -1])
    configured = dn.callbacks.Active(node_check=nodes, threshold=0.0)
    configured.at_least = 2
    previous = _native(configured, _voltage_sequence((2, 3)))
    assert previous is configured
    history = (
        configured.record,
        configured.state_cache,
        configured.i,
        configured.node_check,
    )
    frozen = _snapshot(history)
    callback = functional.make_callbacks({"active": configured})._plans[0].callback
    assert type(callback) is dn.func.ActiveAL and callback.at_least == 2
    assert callback.callback_dt == DT and callback.end_step == int(1e9 / DT)
    _close(callback.node_indices, torch.tensor([0, 3]))
    configured.at_least = 100
    configured.inv = True
    nodes[0] = 1
    rebound = functional.make_callbacks({"active": configured})._plans[0].callback
    assert callback.at_least == 2 and callback.inv is False
    assert rebound.at_least == 100 and rebound.inv is True
    _close(rebound.node_indices, torch.tensor([1, 3]))
    assert configured.record is history[0] and configured.state_cache is history[1]
    assert configured.i == history[2] and configured.node_check is history[3]
    _close(history, frozen)


@pytest.mark.parametrize("native_type,pure_type,options,canonical,kind", _FAMILIES)
def test_empty_selection_zero_step_queries_are_inactive_unless_inverted(
    native_type, pure_type, options, canonical, kind
):
    functional, tensors = _threshold_case()
    for inv in (False, True):
        configured = native_type(node_check=[], inv=inv, **options)
        callbacks = functional.make_callbacks({"active": configured})
        _, auxiliary = dn.func.run(
            functional, _initial(tensors), steps=0, callbacks=callbacks
        )
        result = auxiliary["callbacks"]
        _close(
            result["active"],
            torch.full(functional.shape[:-1], inv, dtype=torch.bool),
        )
        assert result.state["active"][1].shape == (*functional.shape[:-1], 0)
        assert result.state["active"][1].dtype == torch.int32
        assert result.state["active"][2].item() == 0
        assert configured.record is configured.state_cache is None and configured.i == 0


def test_native_partition_queries_remain_posthoc_and_pure_partition_configuration_remains_explicit():
    functional, _ = _threshold_case()
    voltage = _voltage_sequence(functional.shape)
    voltage[..., -1] = -1
    for native_type, pure_type in (
        (dn.callbacks.ActiveAL, dn.func.ActiveAL),
        (dn.callbacks.ActiveALCount, dn.func.ActiveALCount),
    ):
        native = _native(native_type(**_configuration(), at_least=2), voltage)
        before = _snapshot(native.record)
        identity = native.record
        whole = native.is_active()
        partitioned = native.is_active([2, 2])
        assert partitioned.shape == (*whole.shape, 2)
        assert partitioned[..., 0].all() and not partitioned[..., 1].any()
        _close(torch.from_numpy(native.numpy([2, 2])), partitioned)
        configured = functional.make_callbacks({"active": native})._plans[0].callback
        assert configured.partition is None
        _close(_execute(configured, voltage)[1], whole)
        pure = pure_type(**_configuration(), at_least=2, partition=[2, 2])._bind(
            functional
        )
        _close(_execute(pure, voltage)[1], partitioned)
        assert native.record is identity
        _close(native.record, before)


@pytest.mark.parametrize("native_type,pure_type,options,canonical,kind", _FAMILIES)
def test_native_subclasses_and_overridden_queries_or_hooks_are_not_silently_lowered(
    native_type, pure_type, options, canonical, kind
):
    functional, _ = _threshold_case()

    class Derived(native_type):
        pass

    for name in (None, "is_active", "numpy", "post_step_hook"):
        configured = (
            Derived(node_check=[0]) if name is None else native_type(node_check=[0])
        )
        if name is not None:
            setattr(configured, name, lambda *args, **kwargs: None)
        with pytest.raises(
            (TypeError, dn.func.FunctionalizationError),
            match="FunctionalCallback|custom|query|hook|configuration",
        ):
            functional.make_callbacks({"active": configured})


def test_explicit_functional_callback_authority_precedes_native_activity_query_checks():
    functional, tensors = _threshold_case()
    calls = []

    class Pure(dn.callbacks.ActiveAL, dn.func.FunctionalCallback):
        def _bind(self, plan):
            calls.append(plan)
            return self

        def is_active(self, partition=None):
            raise AssertionError("explicit functional hooks own execution")

        def initialize(self, state, auxiliary):
            return (), state["integrator"]["v"].sum()

        def update(self, carry, state, auxiliary):
            return carry, state["integrator"]["v"].sum()

        def finalize(self, carry, emissions):
            return emissions

    configured = Pure(node_check=[0])
    callbacks = functional.make_callbacks({"active": configured})
    assert calls == [functional] and callbacks._plans[0].callback is configured
    assert callbacks._plans[0].validate_updates
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=2, callbacks=callbacks
    )
    assert auxiliary["callbacks"]["active"].shape == (3,)
    assert configured.record is configured.state_cache is None and configured.i == 0


@pytest.mark.parametrize("native_type,pure_type,options,canonical,kind", _FAMILIES)
@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_activity_keeps_pure_v1_checkpoint_configuration_and_legacy_counts(
    native_type, pure_type, options, canonical, kind, network, tmp_path
):
    functional, tensors, parameters, constants, _, _ = _case(network)
    settings = dict(
        node_check=[0, 0, -1],
        threshold=-59.99,
        t_start_check=DT,
        t_end_check=7 * DT,
        dt=DT,
        inv=True,
        **options,
    )

    def scoped(callback):
        return dn.func.PopulationCallback("left", callback) if network else callback

    native = functional.make_callbacks({"active": scoped(native_type(**settings))})
    pure = functional.make_callbacks({"active": scoped(pure_type(**settings))})
    _, first = dn.func.run(
        functional, _initial(tensors, parameters, constants), steps=3, callbacks=native
    )
    _, reference = dn.func.run(
        functional, _initial(tensors, parameters, constants), steps=3, callbacks=pure
    )
    packet = native.export_state(first["callbacks"].state)
    _assert_packet_equal(packet, pure.export_state(reference["callbacks"].state))
    assert packet["version"] == 1
    declaration = packet["configuration"]["callbacks"][0]["callback"]
    if network:
        declaration = declaration["fields"]["callback"]
    assert declaration["kind"] == kind
    assert set(declaration["fields"]) == {
        "node_indices",
        "threshold",
        "start_step",
        "end_step",
        "callback_dt",
        "at_least",
        "inv",
        "partition",
    }
    legacy = _clone(packet)
    below, count, step = legacy["carries"]["active"]
    legacy["carries"]["active"] = (below, count.to(torch.int64), step)
    restored = pure.restore_state(_disk(legacy, tmp_path / "callbacks.pt"))
    assert restored["active"][1].dtype == torch.int32
    _close(restored.carries, first["callbacks"].state.carries)


def _specifications(network):
    settings = dict(
        node_check=[0, 0, -1],
        threshold=-59.99,
        t_start_check=DT,
        t_end_check=8 * DT,
        dt=DT,
    )
    sources = (
        dn.callbacks.Active(**settings, inv=True),
        dn.callbacks.ActiveAL(**settings, at_least=2),
        dn.callbacks.ActiveALCount(**settings, at_least=4),
    )

    def scoped(callback):
        return dn.func.PopulationCallback("left", callback) if network else callback

    return {
        "any": scoped(sources[0]),
        "sites": scoped(sources[1]),
        "events": scoped(sources[2]),
        "count": scoped(dn.func.APCount(**settings)),
        "trace": dn.Recorder(("left.v", "right.v", "t") if network else ("v", "t")),
    }, sources


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_compile_accepts_native_activity_configurations_without_native_history(network):
    specifications, sources = _specifications(network)
    program = dn.compile(
        _network() if network else _population(),
        dt=DT,
        steps=3,
        callbacks=specifications,
        backend="eager",
        disable=True,
    )
    _, auxiliary = program()
    assert all(not plan.validate_updates for plan in program.callbacks._plans)
    assert set(auxiliary["callbacks"]) == set(specifications)
    for name in ("any", "sites", "events"):
        assert auxiliary["callbacks"][name].dtype == torch.bool
        assert auxiliary["callbacks"].state[name][2].item() == 3
    assert all(
        source.record is source.state_cache is None and source.i == 0
        for source in sources
    )


def _compiled_case(network, execution, tmp_path):
    functional, tensors, parameters, constants, _, _ = _case(network)
    specifications, sources = _specifications(network)
    for source in sources:
        source.record = torch.full((2, 3), 37, dtype=torch.int32)
        source.state_cache = torch.zeros((2, 3), dtype=torch.bool)
        source.i = 37
    history = [(source.record, source.state_cache, source.i) for source in sources]
    frozen = _snapshot(history)
    callbacks = functional.make_callbacks(specifications)
    initial = _initial(tensors, parameters, constants)
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    checkpoint = {"checkpoint_every": 3} if execution == "scan" else {}
    with nullcontext() if execution == "scan" else torch.no_grad():
        expected, whole = dn.func.run(functional, initial, steps=9, callbacks=callbacks)
        _, first = dn.func.run(
            kernel, initial, steps=5, callbacks=callbacks, **checkpoint
        )
        _, empty = dn.func.run(
            kernel, first["continuation"], steps=0, callbacks=callbacks, **checkpoint
        )
        actual, last = dn.func.run(
            kernel, empty["continuation"], steps=4, callbacks=callbacks, **checkpoint
        )
    _close(actual, expected)
    trace = _joined(first, last)
    _close(trace, whole["callbacks"]["trace"])
    count = whole["callbacks"]["count"]
    independent = {
        "any": ~(torch.count_nonzero(count, dim=-1) >= 1),
        "sites": torch.count_nonzero(count, dim=-1) >= 2,
        "events": count.sum(-1) >= 4,
    }
    for name, value in independent.items():
        _close(last["callbacks"][name], value)
        _close(last["callbacks"][name], whole["callbacks"][name])
        _close(empty["callbacks"][name], first["callbacks"][name])
        assert value.dtype == torch.bool and not value.requires_grad
    if execution == "scan":
        target = _target(parameters, network)
        observed_loss = _voltage(actual, network).square().sum() + sum(
            value.square().mean() for name, value in trace.items() if name != "t"
        )
        expected_loss = _voltage(expected, network).square().sum() + sum(
            value.square().mean()
            for name, value in whole["callbacks"]["trace"].items()
            if name != "t"
        )
        observed = torch.autograd.grad(observed_loss, target)[0]
        reference = torch.autograd.grad(expected_loss, target)[0]
        _close(observed, reference)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    assert kernel.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    receiving, _ = _specifications(network)
    receiving = functional.make_callbacks(receiving)
    saved = _disk(
        dn.func.export_continuation(
            functional, first["continuation"], callbacks=callbacks
        ),
        tmp_path / "continuation.pt",
    )
    resumed = dn.func.restore_continuation(functional, saved, callbacks=receiving)
    with torch.no_grad():
        final, tail = dn.func.run(kernel, resumed, steps=4, callbacks=receiving)
    _close(final, expected)
    _close(tail["callbacks"].outputs, last["callbacks"].outputs)
    assert tail["callbacks"]["trace"]["t"].shape[0] == 4
    for source, (record, latch, phase) in zip(sources, history, strict=True):
        assert (
            source.record is record
            and source.state_cache is latch
            and source.i == phase
        )
    _close(history, frozen)
    return actual


@pytest.mark.parametrize("network,execution", ((False, "scan"), (True, "while")))
def test_native_activity_compiled_execution_preserves_gradients_and_portable_continuation(
    network, execution, tmp_path
):
    _compiled_case(network, execution, tmp_path)


@pytest.mark.mps
def test_mps_native_activity_while_and_checkpoint_continuation(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        actual = _compiled_case(False, "while", tmp_path)
        assert actual["integrator"]["v"].device.type == "mps"
