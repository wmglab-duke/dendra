"""Native callback carry crosses the same owned boundary as model frames."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models._recorder_storage import _RecorderHDF5Session
from tests.func.test_continuation import _close, _drive, _snapshot
from tests.func.test_continuation_capture import (  # noqa: F401 - shared autouse fixture
    _advance,
    _assert_owned,
    _environment,
    _native_parameter,
    _program,
)
from tests.func.test_continuation_checkpoints import _disk, _frames, _tensors
from tests.func.test_continuation_detach import GNABAR, _History
from tests.func.test_lfp import DT as LFP_DT
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_native_handoff import _capture as _native_snapshot
from tests.func.test_native_handoff import _unchanged as _native_unchanged
from tests.func.test_shared_lfp_configuration import _case as _lfp_case
from tests.func.test_shared_lfp_configuration import _scoped
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu


def _specifications(network):
    def scope(callback):
        return dn.func.PopulationCallback("left", callback) if network else callback

    return {
        "trace": dn.Recorder(
            ("left.v", "right.v", "t") if network else ("v", "t"), dt=2 * DT
        ),
        "count": scope(dn.func.APCount(node_check=[0], threshold=-59.0, dt=DT)),
        "history": scope(_History()),
    }


def _case(network):
    model = _network() if network else _population()
    return model, _program(model, callbacks=_specifications(network))


def _options(program, runner):
    return {} if runner else {"callbacks": program.callbacks}


def _capture(program, runner, *, state=None, native=None):
    return dn.func.capture_continuation(
        program if runner else program.functional,
        native,
        callback_state=state,
        **_options(program, runner),
    )


def _run(program, runner, packet, *, steps, **options):
    return dn.func.run(
        program if runner else program.functional,
        packet,
        steps=steps,
        **_options(program, runner),
        **options,
    )


def _native(model, program, network, *, steps, inputs=None):
    return model.run(
        steps=steps,
        inputs=inputs,
        callbacks=program.callbacks,
        **({} if network else {"dt": DT}),
    )


def _assert_carry_owned(actual, original):
    originals = tuple(_tensors(original.carries))
    storages = {
        (value.device, value.untyped_storage().data_ptr())
        for value in originals
        if value.numel()
    }
    assert actual is not original
    assert actual.plan_token is original.plan_token
    assert actual.emission_schemas is original.emission_schemas
    _close(actual.carries, original.carries)
    for value in _tensors(actual.carries):
        assert type(value) is torch.Tensor
        assert value.is_leaf and value.grad_fn is None and not value.is_inference()
        assert isinstance(value._version, int)
        assert all(value is not old for old in originals)
        if value.numel():
            assert (value.device, value.untyped_storage().data_ptr()) not in storages


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
@pytest.mark.parametrize("prefix_steps", (0, 3), ids=("zero-prefix", "offphase-prefix"))
def test_native_callback_capture_resumes_phase_reductions_and_zero_step_segments(
    network, runner, prefix_steps
):
    model, program = _case(network)
    first = _native(model, program, network, steps=prefix_steps)
    before = _native_snapshot(model)
    packet = _capture(program, runner, state=first.state)
    assert packet._lifecycle == "active"
    _assert_owned(packet, model)
    _assert_carry_owned(packet.callback_state, first.state)
    frozen = _snapshot(_frames(packet))
    unchanged, empty = _run(program, runner, packet, steps=0)
    _close(unchanged, packet.state)
    assert empty["callbacks"]["trace"]["t"].shape == (0,)
    _close(empty["callbacks"].state.carries, packet.callback_state.carries)
    _, middle = _run(program, runner, empty["continuation"], steps=1)
    final, last = _run(program, runner, middle["continuation"], steps=3)

    reference, oracle = _case(network)
    whole = _native(reference, oracle, network, steps=prefix_steps + 4)
    for field, expected in whole["trace"].items():
        joined = torch.cat(
            [
                part["trace"][field]
                for part in (
                    first,
                    empty["callbacks"],
                    middle["callbacks"],
                    last["callbacks"],
                )
            ]
        )
        _close(joined, expected)
        assert joined.shape[0] == (prefix_steps + 4) // 2 + 1
    _close(last["callbacks"]["count"], whole["count"])
    _close(last["callbacks"]["history"], whole["history"])
    _close(final, oracle.functional.extract(reference).state)
    assert last["callbacks"].state["trace"].item() == prefix_steps + 4
    _close(_frames(packet), frozen)
    _native_unchanged(model, before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_omitting_native_callback_carry_still_starts_at_the_captured_boundary(
    network, runner
):
    model, program = _case(network)
    _native(model, program, network, steps=3)
    packet = _capture(program, runner)
    assert packet._lifecycle == "unstarted" and packet.callback_state is None
    _, auxiliary = _run(program, runner, packet, steps=0)
    assert auxiliary["callbacks"]["trace"]["t"].shape == (1,)
    _close(auxiliary["callbacks"]["trace"]["t"][0], packet.state["clock"]["t"])
    assert auxiliary["callbacks"].state["trace"].item() == 0
    assert auxiliary["callbacks"]["history"]["energy"].item() == 0


class _Pair(NamedTuple):
    value: torch.Tensor
    transpose: torch.Tensor


class _DeferredPair(dn.func.FunctionalCallback):
    """A custom carry with no portable checkpoint declaration."""

    def initialize(self, state, auxiliary):
        del auxiliary
        value = state["integrator"]["v"]
        return _Pair(value, value.t()), None

    def update(self, carry, state, auxiliary):
        del carry, auxiliary
        value = state["integrator"]["v"]
        return _Pair(value, value.t()), value

    def finalize(self, carry, emissions):
        return {
            "carry": carry,
            "samples": carry.value.new_empty((0, *carry.value.shape))
            if emissions is None
            else emissions,
        }


@dataclass(frozen=True)
class _RegisteredCarry:
    latest: torch.Tensor
    repeated: torch.Tensor
    label: str = "native-voltage"


pytree.register_pytree_node(
    _RegisteredCarry,
    lambda carry: ([carry.latest, carry.repeated], carry.label),
    lambda values, label: _RegisteredCarry(*values, label),
)


class _RegisteredHistory(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return _RegisteredCarry(voltage, voltage), None

    def update(self, carry, state, auxiliary):
        del carry
        return self.initialize(state, auxiliary)

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_retains_registered_carry_nodes_and_native_frame_tensor_identity(
    network, runner, monkeypatch
):
    model = _network() if network else _population()
    callback = _RegisteredHistory()
    if network:
        callback = dn.func.PopulationCallback("left", callback)
    program = _program(model, callbacks={"history": callback})
    first = _native(model, program, network, steps=2)
    population = model.left if network else model
    # Native observations own their storage. Deliberately supply the live frame
    # object to exercise identity ties across the native extraction and carry.
    carry = dn.func.FunctionalCallbackState(
        {"history": _RegisteredCarry(population.v, population.v)},
        first.state.plan_token,
        first.state.emission_schemas,
    )
    native = _native_snapshot(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("registered custom carry must be copied without hooks")

    with monkeypatch.context() as guard:
        for name in ("initialize", "update", "finalize", "checkpoint_configuration"):
            guard.setattr(_RegisteredHistory, name, forbidden)
        packet = _capture(program, runner, state=carry)
    captured = packet.callback_state["history"]
    local = packet.state["populations"]["left"] if network else packet.state
    assert type(captured) is _RegisteredCarry
    assert captured.label == "native-voltage"
    assert captured.latest is captured.repeated is local["integrator"]["v"]
    assert captured.latest is not population.v
    assert not torch._C._is_alias_of(captured.latest, population.v)
    assert captured.latest.is_leaf and captured.latest.grad_fn is None
    assert not captured.latest.is_inference()
    _close(captured.latest, population.v)
    assert packet.callback_state.plan_token is first.state.plan_token
    assert packet.callback_state.emission_schemas is first.state.emission_schemas
    _native_unchanged(model, native)
    with torch.no_grad():
        captured.latest.add_(1)
    _close(local["integrator"]["v"], population.v + 1)
    _native_unchanged(model, native)
    _, last = _run(program, runner, packet, steps=1)
    assert type(last["callbacks"]["history"]) is _RegisteredCarry
    assert last["callbacks"]["history"].label == "native-voltage"


@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_same_collection_custom_capture_preserves_pytree_aliases_and_deferred_metadata(
    runner, monkeypatch
):
    model = _population()
    program = _program(
        model, callbacks={"first": _DeferredPair(), "second": _DeferredPair()}
    )
    first = _native(model, program, False, steps=2)
    original = first.state
    assert all(schema is not None for schema in original.emission_schemas)
    value = original["first"].value
    carry = dn.func.FunctionalCallbackState(
        {
            "first": _Pair(value, value.t()),
            "second": _Pair(value, value.t()),
        },
        original.plan_token,
        original.emission_schemas,
    )
    native = _native_snapshot(model)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("capture must only read native frames and supplied carry")

    with monkeypatch.context() as guard:
        for name in ("initialize", "update", "finalize", "checkpoint_configuration"):
            guard.setattr(_DeferredPair, name, forbidden)
        for name in ("_initialize", "_update", "export_state", "restore_state"):
            guard.setattr(program.callbacks, name, forbidden)
        for name in ("initialize", "initialize_host", "sample_runtime_draws"):
            guard.setattr(program.functional, name, forbidden)
        for name in ("rand", "randn", "rand_like", "randn_like"):
            guard.setattr(torch, name, forbidden)
        with torch.inference_mode():
            packet = _capture(program, runner, state=carry)
    captured = packet.callback_state
    _assert_carry_owned(captured, carry)
    _assert_owned(packet, model)
    assert type(captured["first"]) is type(captured["second"]) is _Pair
    assert captured["first"].value is captured["second"].value
    assert captured["first"].transpose is not captured["second"].transpose
    assert not torch._C._is_alias_of(
        captured["first"].value, captured["first"].transpose
    )
    _native_unchanged(model, native)
    _close(torch.get_rng_state(), rng)
    saved = _snapshot(captured.carries)
    with torch.no_grad():
        value.add_(4)
    _close(captured.carries, saved)
    with torch.no_grad():
        captured["first"].value.add_(2)
    assert not torch.equal(captured["first"].value, value)
    _, empty = _run(program, runner, packet, steps=0)
    assert empty["callbacks"]["first"]["samples"].shape == (0, *model.shape)
    _, last = _run(program, runner, packet, steps=1)
    assert last["callbacks"]["first"]["samples"].shape == (1, *model.shape)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_cuts_native_model_and_callback_prefix_gradients_in_inference_mode(
    network, runner
):
    model, program = _case(network)
    prefix_drive, inputs = _drive(program, network, 2, requires_grad=True)
    first = _native(model, program, network, steps=2, inputs=inputs)
    assert first.state["history"]["energy"].grad_fn is not None
    assert (model.left.v if network else model.v).grad_fn is not None
    with torch.inference_mode():
        packet = _capture(program, runner, state=first.state)
    _assert_owned(packet, model)
    _assert_carry_owned(packet.callback_state, first.state)
    selected = _target(packet.parameters, network)
    if network:
        assert selected is packet.parameters["populations"]["right"][G]
    suffix_drive, inputs = _drive(program, network, 3, requires_grad=True)
    final, last = _run(program, runner, packet, steps=3, inputs=inputs)
    loss = (
        _voltage(final, network).square().sum() + last["callbacks"]["history"]["energy"]
    )
    parameter, drive, native, prefix = torch.autograd.grad(
        loss,
        (selected, suffix_drive, _native_parameter(model, network), prefix_drive),
        allow_unused=True,
    )
    assert all(
        value is not None and value.isfinite().all() and value.abs().sum() > 0
        for value in (parameter, drive)
    )
    assert native is None and prefix is None
    with torch.no_grad():
        _advance(model, network, steps=3, inputs=inputs)
    _close(final, program.functional.extract(model).state)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
@pytest.mark.parametrize("mode", ("grad", "no_grad", "inference"))
def test_capture_refreshes_owned_lfp_fields_with_caller_grad_mode_after_prefix_backward(
    network, runner, mode
):
    model, population, _, _ = _lfp_case(network)
    field = torch.nn.Parameter(torch.ones_like(population.v))
    program = _program(
        model,
        dt=LFP_DT,
        callbacks={
            "lfp": _scoped(dn.callbacks.LFP([field, field]), network),
            "trace": _scoped(dn.Recorder(("v", "i_membrane", "t")), network),
            "history": _scoped(_History(), network),
        },
    )

    def inputs(current):
        packet = dn.func.Inputs(intra=current)
        return {"observed": packet} if network else packet

    def drive(steps):
        return (
            torch.linspace(
                -2e-7,
                3e-7,
                steps * population.v.numel(),
                dtype=population.v.dtype,
                device=population.v.device,
            )
            .reshape(steps, *population.shape)
            .requires_grad_()
        )

    prefix_drive = drive(2)
    first = model.run(
        steps=2,
        inputs=inputs(prefix_drive),
        callbacks=program.callbacks,
        **({} if network else {"dt": LFP_DT}),
    )
    ((first["lfp"] * 1e6).square().mean() + first["history"]["energy"]).backward()
    assert field.grad is not None and field.grad.abs().sum() > 0
    assert prefix_drive.grad is not None and prefix_drive.grad.abs().sum() > 0
    supplied = _snapshot(first.state.carries)
    with torch.no_grad():
        field.add_(0.5)
    field.grad = None
    prefix_drive.grad = None
    native_parameter = population.mech.hh.gnabar_param
    native_parameter.grad = None
    native = _native_snapshot(model)
    context = (
        torch.no_grad()
        if mode == "no_grad"
        else torch.inference_mode()
        if mode == "inference"
        else nullcontext()
    )
    with context:
        packet = _capture(program, runner, state=first.state)
        assert torch.is_grad_enabled() == (mode == "grad")
        assert torch.is_inference_mode_enabled() == (mode == "inference")
    captured_field = packet.callback_state["lfp"]
    _close(captured_field, torch.stack((field, field)))
    assert captured_field.requires_grad == (mode == "grad")
    assert (captured_field.grad_fn is not None) == (mode == "grad")
    assert not captured_field.is_inference()
    assert isinstance(captured_field._version, int)
    assert not torch._C._is_alias_of(captured_field, field)
    assert not torch._C._is_alias_of(captured_field, first.state["lfp"])
    assert not torch.equal(captured_field, first.state["lfp"])
    for value in _tensors(packet.callback_state["history"]):
        assert value.is_leaf and value.grad_fn is None and not value.is_inference()
    _close(first.state.carries, supplied)
    _native_unchanged(model, native)
    suffix_drive = drive(3)
    final, last = _run(program, runner, packet, steps=3, inputs=inputs(suffix_drive))
    loss = (last["callbacks"]["lfp"] * 1e6).square().mean() + 1e-3 * last["callbacks"][
        "history"
    ]["latest"].square().mean()
    parameters = (
        packet.parameters["populations"]["observed"] if network else packet.parameters
    )
    selected, suffix, actual_field, native_gradient, prefix = torch.autograd.grad(
        loss,
        (parameters[GNABAR], suffix_drive, field, native_parameter, prefix_drive),
        allow_unused=True,
    )
    assert all(
        value is not None and value.isfinite().all() and value.abs().sum() > 0
        for value in (selected, suffix)
    )
    assert native_gradient is None and prefix is None
    if mode == "grad":
        currents = last["callbacks"]["trace"]["i_membrane"].detach()
        proxy = field.detach().clone().requires_grad_()
        projection = (currents[:, None] * torch.stack((proxy, proxy))[None]).sum(
            (-1, -2)
        )
        wanted = torch.autograd.grad((projection * 1e6).square().mean(), proxy)[0]
        _close(actual_field, wanted)
        assert actual_field.isfinite().all() and actual_field.abs().sum() > 0
    else:
        assert actual_field is None
    assert final["clock"]["t"].item() == pytest.approx(5 * LFP_DT)
    _native_unchanged(model, native)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_captured_callback_carry_round_trips_as_an_active_continuation(
    network, runner, tmp_path
):
    model, program = _case(network)
    first = _native(model, program, network, steps=3)
    packet = _capture(program, runner, state=first.state)
    subject = program if runner else program.functional
    saved = _disk(
        dn.func.export_continuation(subject, packet, **_options(program, runner)),
        tmp_path / "captured-callbacks.pt",
    )
    _, receiver = _case(network)
    restored = dn.func.restore_continuation(
        receiver if runner else receiver.functional,
        saved,
        **_options(receiver, runner),
    )
    assert restored._lifecycle == "active"
    assert restored.callback_state.plan_token is receiver.callbacks._state_token
    assert restored.callback_state.plan_token is not packet.callback_state.plan_token
    _close(_frames(restored), _frames(packet))
    final, last = _run(program, runner, packet, steps=4)
    replay, auxiliary = _run(receiver, runner, restored, steps=4)
    _close(replay, final)
    _close(auxiliary["callbacks"].outputs, last["callbacks"].outputs)
    times = auxiliary["callbacks"]["trace"]["t"]
    _close(times, times.new_tensor([4 * DT, 6 * DT]))


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_callback_carry_ownership_does_not_narrow_compatible_native_capture(
    network, runner
):
    source, program = _case(network)
    first = _native(source, program, network, steps=2)
    alternate = _network() if network else _population()
    with torch.no_grad():
        _native_parameter(alternate, network).add_(0.0002)
        _advance(alternate, network, steps=2)
    expected = program.functional.extract(alternate)
    source_before, alternate_before = (
        _native_snapshot(source),
        _native_snapshot(alternate),
    )
    packet = _capture(program, runner, state=first.state, native=alternate)
    assert packet._lifecycle == "active"
    _close(
        (packet.state, packet.parameters, packet.constants),
        (expected.state, expected.parameters, expected.constants),
    )
    _assert_carry_owned(packet.callback_state, first.state)
    _assert_owned(packet, alternate)
    _native_unchanged(source, source_before)
    _native_unchanged(alternate, alternate_before)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_capture_rejects_mismatched_callback_collections_and_malformed_carry_atomically(
    network, monkeypatch
):
    model, program = _case(network)
    first = _native(model, program, network, steps=2)
    callbacks = program.callbacks
    other = program.functional.make_callbacks(_specifications(network))
    _, foreign = _case(network)
    before = _native_snapshot(model)
    frozen = _snapshot(first.state.carries)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "invalid callback operands must fail before initialization"
        )

    monkeypatch.setattr(callbacks, "_initialize", forbidden)
    for malformed in (object(), first):
        with pytest.raises(TypeError, match="FunctionalCallbackState|callback_state"):
            _capture(program, True, state=malformed)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="callback"):
        dn.func.capture_continuation(program.functional, callback_state=first.state)
    with pytest.raises(TypeError, match="FunctionalCallbacks"):
        dn.func.capture_continuation(program.functional, callbacks={})
    with pytest.raises(dn.func.FunctionalizationError, match="different.*plan"):
        dn.func.capture_continuation(
            program.functional, callbacks=other, callback_state=first.state
        )
    with pytest.raises(dn.func.FunctionalizationError, match="different.*plan"):
        dn.func.capture_continuation(program.functional, callbacks=foreign.callbacks)
    for explicit in ({}, callbacks):
        with pytest.raises(TypeError, match="callbacks.*omitted|callbacks"):
            dn.func.capture_continuation(
                program, callbacks=explicit, callback_state=first.state
            )
    malformed = dn.func.FunctionalCallbackState(
        {}, first.state.plan_token, first.state.emission_schemas
    )
    with pytest.raises(ValueError, match="keys.*match"):
        _capture(program, True, state=malformed)
    malformed = dn.func.FunctionalCallbackState(
        {
            **first.state.carries,
            "history": {**first.state["history"], "latest": torch.zeros(3)},
        },
        first.state.plan_token,
        first.state.emission_schemas,
    )
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="shape"):
        _capture(program, True, state=malformed)
    malformed = dn.func.FunctionalCallbackState(
        first.state.carries, first.state.plan_token, ()
    )
    with pytest.raises(ValueError, match="emission schema count"):
        _capture(program, True, state=malformed)
    _native_unchanged(model, before)
    _close(first.state.carries, frozen)


def test_capture_of_streamed_native_carry_does_not_open_or_write_recording_storage(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
):
    model = _population()
    recorder = dn.Recorder(("v", "t"), dt=2 * DT).set_hdf5(
        str(tmp_path / "native-prefix.h5"), cache_every=2
    )
    program = _program(model, callbacks={"trace": recorder})
    try:
        first = _native(model, program, False, steps=3)
        queued, run_number = len(local_hdf5_writer.queue.items), recorder.run_number
        native = _native_snapshot(model)

        def forbidden(*args, **kwargs):
            raise AssertionError("capture must not create recording storage")

        monkeypatch.setattr(_RecorderHDF5Session, "__init__", forbidden)
        packet = _capture(program, True, state=first.state)
        assert packet._lifecycle == "active"
        _assert_carry_owned(packet.callback_state, first.state)
        assert recorder._hdf5_session is None and recorder.run_number == run_number
        assert len(local_hdf5_writer.queue.items) == queued
        _native_unchanged(model, native)
    finally:
        recorder.close()


@pytest.mark.parametrize(
    ("network", "execution"),
    ((False, "scan"), (True, "while")),
    ids=("population-training-scan", "network-inference-while"),
)
def test_native_callback_capture_runs_through_real_fullgraph_compiled_suffixes(
    network, execution
):
    model = _network() if network else _population()
    inference = execution == "while"
    specifications = _specifications(network)
    specifications.pop("history")
    program = dn.compile(
        model,
        dt=DT,
        steps=8,
        chunk_steps=4,
        inference=inference,
        execution=execution,
        checkpoint_every=None if inference else 4,
        trainable=() if inference else "*g_param",
        callbacks=specifications,
        backend="aot_eager",
        fullgraph=True,
    )
    with torch.no_grad():
        first = _native(model, program, network, steps=3)
    packet = _capture(program, True, state=first.state)
    native = _native_snapshot(model)
    with torch.no_grad() if inference else nullcontext():
        final, last = program(packet)
        expected, reference = program.functional.prepare_and_rollout(
            packet.parameters,
            packet.constants,
            packet.state,
            steps=8,
            callbacks=program.callbacks,
            callback_state=packet.callback_state,
        )
    _close(final, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    times = last["callbacks"]["trace"]["t"]
    _close(times, times.new_tensor([4 * DT, 6 * DT, 8 * DT, 10 * DT]))
    assert program.execution_report().strategy == (
        "compiled_while_loop" if inference else "compiled_scan"
    )
    if not inference:
        selected = _target(packet.parameters, network)
        observed = torch.autograd.grad(
            _voltage(final, network).square().sum(), selected
        )[0]
        wanted = torch.autograd.grad(
            _voltage(expected, network).square().sum(), selected
        )[0]
        _close(observed, wanted)
        assert observed.isfinite().all() and observed.abs().sum() > 0
    _native_unchanged(model, native)


def test_lfp_refresh_preserves_aliased_raw_source_values_and_owned_leaf():
    class FieldCapacitance(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source

        def forward(self, value):
            return value + 0.01 * self.source.mean()

    model, population, _, _ = _lfp_case()
    source = torch.nn.Parameter(population.v.new_full((2, *population.shape), 3.25))
    population.register_parametrization_in_graph("cm", FieldCapacitance(source))
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    source_name = next(
        name for name, parameter in population.named_parameters() if parameter is source
    )
    field = torch.nn.Parameter(torch.ones_like(population.v))
    program = _program(model, dt=LFP_DT, callbacks={"lfp": dn.func.LFP([field, field])})
    first = model.run(steps=0, dt=LFP_DT, callbacks=program.callbacks)
    assert first.state["lfp"].shape == source.shape
    supplied = dn.func.FunctionalCallbackState(
        {"lfp": source}, first.state.plan_token, first.state.emission_schemas
    )

    packet = _capture(program, True, state=supplied)
    captured_source = packet.parameters[source_name]
    refreshed = packet.callback_state["lfp"]
    _close(captured_source, source)
    assert type(captured_source) is torch.Tensor
    assert captured_source.is_leaf and captured_source.grad_fn is None
    assert captured_source.requires_grad and not captured_source.is_inference()
    assert not torch._C._is_alias_of(captured_source, source)
    assert captured_source is not refreshed
    assert not torch._C._is_alias_of(captured_source, refreshed)
    _close(refreshed, torch.stack((field, field)))
    assert refreshed.requires_grad and refreshed.grad_fn is not None

    field_gradient, native_gradient = torch.autograd.grad(
        refreshed.sum(), (field, source), allow_unused=True
    )
    _close(field_gradient, torch.full_like(field, 2))
    assert native_gradient is None
    captured_gradient, native_gradient = torch.autograd.grad(
        captured_source.square().sum(), (captured_source, source), allow_unused=True
    )
    _close(captured_gradient, 2 * source)
    assert native_gradient is None
