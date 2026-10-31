"""Truncated training cuts recurrent graphs while retaining live raw sources."""

from __future__ import annotations

import os
from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models._recorder_storage import _RecorderHDF5Session
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _drive,
    _plain,
    _snapshot,
)
from tests.func.test_continuation_checkpoints import _disk, _tensors
from tests.func.test_lfp import DT as LFP_DT
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_network_source_history import _stim_history
from tests.func.test_shared_lfp_configuration import _case as _lfp_case
from tests.func.test_shared_lfp_configuration import _scoped
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _History(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "initial": voltage,
            "latest": voltage,
            "energy": voltage.new_zeros(()),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "initial": carry["initial"],
            "latest": voltage,
            "energy": carry["energy"] + voltage.square().mean(),
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry

    def checkpoint_configuration(self):
        return {"kind": "tests.continuation-detach.history", "version": 1, "fields": {}}


def _program(network=False):
    model = _network() if network else _population()
    callbacks = _callbacks(network)
    callbacks["history"] = (
        dn.callbacks.PopulationCallback("left", _History()) if network else _History()
    )
    callbacks["trace"] = dn.Recorder(
        ("left.v", "right.v", "t") if network else ("v", "t"), dt=2 * DT
    )
    return dn.compile(
        model,
        dt=DT,
        steps=3,
        inference=False,
        trainable="*g_param",
        callbacks=callbacks,
        backend="eager",
        disable=True,
    )


def _initial(program):
    return dn.func.Continuation(
        program.initial_state, program.parameters, program.constants
    )


def _run(program, explicit, packet=None, *, steps=3, **options):
    if explicit:
        return dn.func.run(
            program.functional,
            _initial(program) if packet is None else packet,
            steps=steps,
            callbacks=program.callbacks,
            **options,
        )
    return (
        program(packet, steps=steps, **options)
        if packet is not None
        else program(steps=steps, **options)
    )


def _detach(program, explicit, packet):
    return (
        dn.func.detach_continuation(
            program.functional, packet, callbacks=program.callbacks
        )
        if explicit
        else dn.func.detach_continuation(program, packet)
    )


def _loss(state, auxiliary, network):
    return (
        _voltage(state, network).square().sum()
        + auxiliary["callbacks"]["history"]["energy"]
        + auxiliary["callbacks"]["history"]["latest"].square().mean()
    )


def _same_sources(actual, expected):
    if torch.is_tensor(expected):
        assert actual is expected
        return
    if isinstance(expected, Mapping):
        assert tuple(actual) == tuple(expected)
        for name, value in expected.items():
            _same_sources(actual[name], value)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _same_sources(value, reference)
    else:
        assert actual is expected


def _zero_grad(parameters):
    for value in _tensors(parameters):
        value.grad = None


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("explicit", (False, True), ids=("runner", "plan"))
def test_detached_continuation_cuts_prefix_inputs_and_keeps_strict_suffix_gradients(
    network, explicit
):
    program = _program(network)
    prefix_drive, inputs = _drive(program, network, 3, requires_grad=True)
    state, first = _run(program, explicit, inputs=inputs)
    _loss(state, first, network).backward()
    assert prefix_drive.grad is not None and prefix_drive.grad.abs().sum() > 0
    original = first["continuation"]
    frozen = _snapshot((original.state, original.callback_state.carries))
    detached = _detach(program, explicit, original)
    assert (
        detached is not original
        and detached._lifecycle == original._lifecycle == "active"
    )
    _same_sources(detached.parameters, original.parameters)
    _same_sources(detached.constants, original.constants)
    if network:
        assert (
            _target(detached.parameters, True)
            is detached.parameters["populations"]["right"][G]
        )
    assert detached.callback_state.plan_token is original.callback_state.plan_token
    assert (
        detached.callback_state.emission_schemas
        is original.callback_state.emission_schemas
    )
    for value in _tensors((detached.state, detached.callback_state.carries)):
        assert value.grad_fn is None and value.is_leaf and not value.requires_grad
    _close((detached.state, detached.callback_state.carries), frozen)
    _close((original.state, original.callback_state.carries), frozen)
    _zero_grad(program.parameters)
    prefix_drive.grad = None
    suffix_drive, suffix_inputs = _drive(program, network, 3, requires_grad=True)
    final, last = _run(program, explicit, detached, inputs=suffix_inputs)
    _loss(final, last, network).backward()
    selected = _target(detached.parameters, network)
    observed = selected.grad.clone(), suffix_drive.grad.clone()
    assert prefix_drive.grad is None
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    expected, reference = dn.func.run(
        program.functional,
        detached,
        steps=3,
        inputs=suffix_inputs,
        callbacks=program.callbacks,
    )
    wanted = torch.autograd.grad(
        _loss(expected, reference, network), (selected, suffix_drive)
    )
    _close(observed, wanted)
    _close(final, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    assert last["callbacks"]["trace"]["t"].shape == (2,)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("explicit", (False, True), ids=("runner", "plan"))
def test_lfp_invariants_rematerialize_after_backward_and_optimizer_updates(
    network, explicit
):
    model, population, _, _ = _lfp_case(network)
    stacked = explicit and not network
    field = torch.nn.Parameter(
        torch.ones(
            (2, *population.shape), device=population.device(), dtype=population.dtype()
        )
        if stacked
        else torch.ones_like(population.v)
    )
    lfp = (
        dn.callbacks.LFP([torch.ones_like(population.v)])
        if stacked
        else dn.callbacks.LFP([field, field])
    )
    if stacked:
        lfp.v_unit = field
    prefix = "observed." if network else ""
    program = dn.compile(
        model,
        dt=LFP_DT,
        steps=2,
        inference=False,
        trainable="*gnabar_param",
        callbacks={
            "lfp": _scoped(lfp, network),
            "trace": dn.Recorder((prefix + "i_membrane", "t")),
            "history": _scoped(_History(), network),
        },
        backend="eager",
        disable=True,
    )
    parameters = (
        program.parameters["populations"]["observed"] if network else program.parameters
    )
    conductance = parameters[GNABAR]
    optimizer = torch.optim.SGD(
        [{"params": [conductance], "lr": 1e-8}, {"params": [field], "lr": 1e-4}]
    )
    packet = None
    for iteration in range(3):
        drive = torch.linspace(
            -2e-7,
            3e-7,
            2 * population.v.numel(),
            device=population.device(),
            dtype=population.dtype(),
        ).reshape(2, *population.shape)
        inputs = dn.func.Inputs(intra=drive)
        state, auxiliary = _run(
            program,
            explicit,
            packet,
            steps=2,
            inputs={"observed": inputs} if network else inputs,
        )
        currents = auxiliary["callbacks"]["trace"][prefix + "i_membrane"]
        proxy = field.detach().clone().requires_grad_()
        fields = proxy if stacked else torch.stack((proxy, proxy))
        expected = (
            (currents.detach().unsqueeze(1) * fields.unsqueeze(0))
            .sum((-1, -2))
            .movedim(1, -1)
        )
        _close(auxiliary["callbacks"]["lfp"], expected)
        wanted = torch.autograd.grad((expected * 1e6).square().mean(), proxy)[0]
        loss = (auxiliary["callbacks"]["lfp"] * 1e6).square().mean() + 1e-3 * auxiliary[
            "callbacks"
        ]["history"]["latest"].square().mean()
        loss.backward()
        _close(field.grad, wanted)
        assert (
            torch.isfinite(conductance.grad).all() and conductance.grad.abs().sum() > 0
        )
        assert field.grad.abs().sum() > 0
        old = auxiliary["continuation"]
        old_fields = old.callback_state["lfp"]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        packet = _detach(program, explicit, old)
        assert packet.callback_state["lfp"] is not old_fields
        assert packet.callback_state["lfp"].grad_fn is not None
        assert packet.callback_state["lfp"].requires_grad
        _close(
            packet.callback_state["lfp"],
            field if stacked else torch.stack((field, field)),
        )
        assert all(
            not value.requires_grad and value.grad_fn is None
            for value in _tensors((packet.state, packet.callback_state["history"]))
        )
        assert packet.callback_state.plan_token is old.callback_state.plan_token
        selected = (
            packet.parameters["populations"]["observed"]
            if network
            else packet.parameters
        )
        assert (
            packet.parameters is not old.parameters and selected[GNABAR] is conductance
        )
        assert auxiliary["callbacks"]["trace"]["t"].shape[0] == (
            3 if iteration == 0 else 2
        )


@pytest.mark.parametrize(
    "conflict", (False, True), ids=("same_sources", "conflicting_sources")
)
def test_lfp_aliases_refresh_once_preserve_custom_ties_and_follow_gradient_mode(
    conflict,
):
    _, population, functional, tensors = _lfp_case()
    field = torch.nn.Parameter(torch.ones_like(population.v))
    other = torch.nn.Parameter(torch.ones_like(field)) if conflict else field

    class Echo(dn.func.FunctionalCallback):
        def initialize(self, state, auxiliary):
            return torch.stack((field, field)), None

        def update(self, carry, state, auxiliary):
            return carry, None

        def finalize(self, carry, emissions):
            return carry

    callbacks = functional.make_callbacks(
        {
            "first": dn.func.LFP([field, field]),
            "second": dn.func.LFP([other, other]),
            "echo": Echo(),
        }
    )
    _, auxiliary = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=1,
        callbacks=callbacks,
    )
    original = auxiliary["continuation"]
    shared = original.callback_state["first"]
    state = dn.func.FunctionalCallbackState(
        {"first": shared, "second": shared, "echo": shared},
        original.callback_state.plan_token,
        original.callback_state.emission_schemas,
    )
    packet = dn.func.Continuation(
        original.state, original.parameters, original.constants, state
    )
    if conflict:
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="aliased LFP carry.*different field sources",
        ):
            dn.func.detach_continuation(functional, packet, callbacks=callbacks)
        assert packet.callback_state["second"] is shared
        return
    detached = dn.func.detach_continuation(functional, packet, callbacks=callbacks)
    refreshed = detached.callback_state["first"]
    assert (
        refreshed
        is detached.callback_state["second"]
        is detached.callback_state["echo"]
    )
    assert refreshed is not shared and refreshed.grad_fn is not None
    _close(torch.autograd.grad(refreshed.sum(), field)[0], torch.full_like(field, 2))
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            frozen = dn.func.detach_continuation(
                functional, packet, callbacks=callbacks
            )
        carry = frozen.callback_state["first"]
        assert carry is frozen.callback_state["second"] is frozen.callback_state["echo"]
        assert (
            not carry.requires_grad
            and carry.grad_fn is None
            and not torch.is_inference(carry)
        )
        assert frozen.parameters is not packet.parameters
        _same_sources(frozen.parameters, packet.parameters)


def test_detach_preserves_recurrent_aliases_views_and_splits_raw_source_identity():
    program = _program()
    _, first = program()
    original = first["continuation"]
    state = _plain(original.state)
    voltage = state["integrator"]["v"]
    distinct_view = voltage.view_as(voltage)
    state["clock"]["t"] = original.parameters[G]
    carries = {
        **original.callback_state.carries,
        "history": {
            **original.callback_state["history"],
            "initial": distinct_view,
            "latest": voltage,
        },
    }
    callback_state = dn.func.FunctionalCallbackState(
        carries,
        original.callback_state.plan_token,
        original.callback_state.emission_schemas,
    )
    packet = dn.func.Continuation(
        state, original.parameters, original.constants, callback_state
    )
    detached = dn.func.detach_continuation(program, packet)
    assert detached._lifecycle == packet._lifecycle == "unspecified"
    value = detached.state["integrator"]["v"]
    assert value is detached.callback_state["history"]["latest"]
    assert value is not voltage and value.data_ptr() == voltage.data_ptr()
    view = detached.callback_state["history"]["initial"]
    assert view is not value and view.data_ptr() == distinct_view.data_ptr()
    assert (
        view.stride() == distinct_view.stride()
        and view.storage_offset() == distinct_view.storage_offset()
    )
    assert detached.parameters[G] is original.parameters[G]
    assert detached.state["clock"]["t"] is not detached.parameters[G]
    assert detached.state["clock"]["t"].data_ptr() == detached.parameters[G].data_ptr()
    assert (
        detached.parameters[G].requires_grad
        and not detached.state["clock"]["t"].requires_grad
    )


def test_detach_retains_nonleaf_raw_parameters_and_all_source_flags():
    program = _program(True)
    parameters = _plain(program.parameters)
    upstream = _target(parameters, True).detach().clone().requires_grad_()
    replacement = 1.3 * upstream
    _assign(parameters, True, replacement)
    _, first = program(parameters=parameters)
    original = first["continuation"]
    detached = dn.func.detach_continuation(program, original)
    assert _target(detached.parameters, True) is replacement
    assert replacement is detached.parameters["populations"]["right"][G]
    assert replacement.grad_fn is not None and replacement.requires_grad
    _same_sources(detached.constants, original.constants)
    _same_sources(detached.parameters, original.parameters)
    _close(
        torch.autograd.grad(_target(detached.parameters, True).sum(), upstream)[0],
        torch.full_like(upstream, 1.3),
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_detachment_preserves_active_and_unstarted_portable_lifecycle(
    network, tmp_path
):
    program = _program(network)
    initial = dn.func.initialize(program)
    _, first = program(steps=2)
    for name, original in (("unstarted", initial), ("active", first["continuation"])):
        detached = dn.func.detach_continuation(program, original)
        assert detached._lifecycle == original._lifecycle == name
        if name == "unstarted":
            assert detached.callback_state is None
        portable = _disk(
            dn.func.export_continuation(program, detached), tmp_path / f"{name}.pt"
        )
        restored = dn.func.restore_continuation(program, portable)
        with torch.no_grad():
            actual, last = program(detached, steps=1)
            expected, reference = program(restored, steps=1)
        _close(actual, expected)
        _close(last["callbacks"].outputs, reference["callbacks"].outputs)
        assert last["callbacks"]["trace"]["t"].shape[0] == (
            1 if name == "unstarted" else 0
        )


def test_source_history_rng_and_pending_delivery_continue_exactly_after_detachment():
    model = _stim_history(noise=0.6, twins=True)
    functional, tensors = dn.func.make_functional(model, dt=model.dt)
    callbacks = functional.make_callbacks({"trace": dn.Recorder(("cell.v", "t"))})
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    _, first = dn.func.run(functional, initial, steps=3, callbacks=callbacks)
    original = first["continuation"]
    native = _snapshot(model.state_dict_for_checkpoint())
    rng = torch.get_rng_state().clone()
    detached = dn.func.detach_continuation(functional, original, callbacks=callbacks)
    _close(detached.state, original.state)
    _same_sources(detached.parameters, original.parameters)
    assert all(
        value.grad_fn is None and not value.requires_grad
        for value in _tensors(detached.state)
    )
    with torch.no_grad():
        actual, last = dn.func.run(functional, detached, steps=8, callbacks=callbacks)
        expected, reference = dn.func.run(
            functional, original, steps=8, callbacks=callbacks
        )
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    _close(torch.get_rng_state(), rng)
    _assert_packet_equal(model.state_dict_for_checkpoint(), native)


def test_detach_is_cold_and_validates_targets_packets_callback_ownership(monkeypatch):
    program = _program()
    _, first = program()
    packet = first["continuation"]
    functional, callbacks = program.functional, program.callbacks
    with pytest.raises(TypeError, match="requires a Continuation"):
        dn.func.detach_continuation(program, object())
    for target in (object(), functional.bind(packet), program.kernel):
        with pytest.raises(
            TypeError, match="FunctionalPopulation|FunctionalNetwork|runner"
        ):
            dn.func.detach_continuation(target, packet)
    with pytest.raises(TypeError, match="callbacks must be omitted"):
        dn.func.detach_continuation(program, packet, callbacks=callbacks)
    with pytest.raises(dn.func.FunctionalizationError, match="callback"):
        dn.func.detach_continuation(functional, packet)
    with pytest.raises(TypeError, match="FunctionalCallbacks"):
        dn.func.detach_continuation(functional, packet, callbacks={})
    other = _program()
    with pytest.raises(
        dn.func.FunctionalizationError, match="different functional execution plan"
    ):
        dn.func.detach_continuation(functional, packet, callbacks=other.callbacks)
    equivalent = functional.make_callbacks(
        {"trace": dn.Recorder(("v", "t")), "history": _History()}
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        dn.func.detach_continuation(functional, packet, callbacks=equivalent)
    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(dn.func.FunctionalizationError, match="outside"):
            dn.func.detach_continuation(object(), object())
    with torch.autograd.forward_ad.dual_level():
        with pytest.raises(dn.func.FunctionalizationError, match="outside"):
            dn.func.detach_continuation(object(), object())
    with pytest.raises(dn.func.FunctionalizationError, match="outside"):
        torch.func.grad(lambda value: dn.func.detach_continuation(object(), object()))(
            torch.ones(())
        )


def test_detach_never_initializes_samples_prepares_or_opens_recording_storage(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
    monkeypatch,
):
    disk = dn.Recorder(("v", "t")).set_hdf5(str(tmp_path / "history.h5"), cache_every=2)
    program = dn.compile(
        _population(),
        dt=DT,
        steps=2,
        callbacks={"disk": disk, "history": _History()},
        inference=False,
        trainable="*g_param",
        backend="eager",
        disable=True,
    )
    try:
        _, first = program()
        queued, run = len(local_hdf5_writer.queue.items), disk.run_number
        report = program.execution_report()
        source = _snapshot(
            (
                program.initial_state,
                program.parameters,
                program.constants,
                program.functional.extract().state,
            )
        )
        native = _snapshot(program.functional._population.state_dict_for_checkpoint())
        rng = torch.get_rng_state().clone()

        def forbidden(*args, **kwargs):
            raise AssertionError("detach_continuation must not execute or write")

        for name in (
            "initialize_host",
            "prepare",
            "sample_runtime_draws",
            "prepare_and_rollout",
            "extract",
        ):
            monkeypatch.setattr(program.functional, name, forbidden)
        monkeypatch.setattr(_RecorderHDF5Session, "__init__", forbidden)
        monkeypatch.setattr(_History, "initialize", forbidden)
        packet = dn.func.detach_continuation(program, first["continuation"])
        assert packet.callback_state.plan_token is program.callbacks._state_token
        assert program.execution_report() is report
        assert disk._hdf5_session is None and disk.run_number == run
        assert len(local_hdf5_writer.queue.items) == queued
        _close(
            (program.initial_state, program.parameters, program.constants), source[:3]
        )
        _assert_packet_equal(
            program.functional._population.state_dict_for_checkpoint(), native
        )
        _close(torch.get_rng_state(), rng)
    finally:
        disk.close()


@pytest.mark.mps
def test_mps_detachment_allows_a_fresh_suffix_backward_without_copying_sources():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        program = _program()
        state, first = program()
        _loss(state, first, False).backward()
        packet = dn.func.detach_continuation(program, first["continuation"])
        _zero_grad(program.parameters)
        state, last = program(packet)
        _loss(state, last, False).backward()
        gradient = program.parameters[G].grad
        assert gradient.device.type == "mps" and gradient.abs().sum() > 0
        assert packet.parameters[G] is program.parameters[G]
        assert packet.state["integrator"]["v"].device.type == "mps"
