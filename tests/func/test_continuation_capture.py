"""Current native boundaries become independent functional continuations."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _callbacks, _close, _drive, _snapshot
from tests.func.test_continuation_checkpoints import _disk, _frames, _tensors
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_native_handoff import _capture as _native_snapshot
from tests.func.test_native_handoff import _unchanged as _native_unchanged
from tests.func.test_native_handoff_stochastic import (
    _assert_borrowed_streams,
)
from tests.func.test_native_handoff_stochastic import _case as _stochastic_case
from tests.func.test_native_handoff_stochastic import (
    _populations,
)
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_connection_sampling import (
    _check_native as _check_sampled_connections,
)
from tests.func.test_network_connection_sampling import (
    _native_targets as _native_sampled_targets,
)
from tests.func.test_network_connection_sampling import _network as _sampled_network
from tests.func.test_network_connection_sampling import _targets as _sampled_targets
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_network_initialization_ordering import (
    RAW_CM,
)
from tests.func.test_network_initialization_ordering import (
    _native_voltage as _ordered_native_voltage,
)
from tests.func.test_network_initialization_ordering import _network as _ordered_network
from tests.func.test_network_initialization_ordering import _voltage as _ordered_voltage
from tests.func.test_network_stochastic_conformance import DT as CONNECTED_DT
from tests.func.test_network_stochastic_conformance import RAW as CONNECTED_RAW
from tests.func.test_network_stochastic_conformance import (
    _network as _connected_network,
)
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _one_stream, _step_native

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                DTYPE=torch.float64,
                DEVICE="cpu",
                IMEM=False,
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _program(model, *, dt=DT, callbacks=None, initialization="fresh", **options):
    return dn.compile(
        model,
        dt=dt,
        steps=2,
        initialization=initialization,
        inference=False,
        trainable=(),
        callbacks=callbacks,
        backend="eager",
        disable=True,
        **options,
    )


def _native_parameter(model, network):
    return model.left.mech.leak.g_param if network else model.mech.leak.g_param


def _advance(model, network, *, steps=None, duration=None, inputs=None):
    options = {"steps": steps} if steps is not None else {"duration": duration}
    if not network:
        options["dt"] = DT
    return model.run(inputs=inputs, **options)


def _assert_owned(packet, model):
    native = tuple(model.parameters()) + tuple(model.buffers())
    storages = {
        (value.device, value.untyped_storage().data_ptr())
        for value in native
        if value.numel()
    }
    for value in _tensors(_frames(packet)):
        assert type(value) is torch.Tensor
        assert value.is_leaf and value.grad_fn is None
        assert not value.is_inference()
        assert isinstance(value._version, int)
        if value.numel():
            assert (value.device, value.untyped_storage().data_ptr()) not in storages


def _assert_independent_network_clocks(packet, native):
    native_clocks = [native.t, *(value.t for value in native.populations.values())]
    captured_clocks = [
        packet.state["clock"]["t"],
        *(
            packet.state["populations"][name]["clock"]["t"]
            for name in native.populations
        ),
    ]
    assert len({id(value) for value in native_clocks}) == len(native_clocks)
    assert len({id(value) for value in captured_clocks}) == len(captured_clocks)
    for captured, original in zip(captured_clocks, native_clocks, strict=True):
        assert captured is not original
        torch.testing.assert_close(captured, original)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
@pytest.mark.parametrize("explicit", (False, True), ids=("source", "compatible_native"))
def test_capture_reads_current_native_boundary_without_using_runner_defaults(
    network, runner, explicit, monkeypatch
):
    source = _network() if network else _population()
    program = _program(source, callbacks=_callbacks(network))
    functional = program.functional
    subject = program if runner else functional
    native = (_network() if network else _population()) if explicit else source
    initial = _snapshot(functional.extract(native).state)
    with torch.no_grad():
        _native_parameter(native, network).add_(0.0003)
    _advance(native, network, duration=2.4 * DT)
    expected = functional.extract(native)
    assert not torch.allclose(
        _voltage(expected.state, network), _voltage(initial, network)
    )

    # These are independent constructor defaults. Capture selects native frames
    # even when a fresh-policy runner has been used and its defaults edited.
    program(steps=1)
    with torch.no_grad():
        _target(program.parameters, network).add_(0.002)
        if network:
            program.initial_state["populations"]["left"]["integrator"]["v"].add_(7)
            program.constants["populations"]["left"]["dt"].mul_(2)
        else:
            program.initial_state["integrator"]["v"].add_(7)
            program.constants["dt"].mul_(2)
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    saved = _native_snapshot(native)
    report = program.execution_report()
    prepared = None if network else program._prepared

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "capture must not initialize, step, sample or start callbacks"
        )

    with monkeypatch.context() as guard:
        guard.setattr(functional, "initialize", forbidden)
        guard.setattr(
            functional, "_initialize_host" if network else "initialize_host", forbidden
        )
        guard.setattr(functional, "sample_runtime_draws", forbidden)
        guard.setattr(program.callbacks, "_initialize", forbidden)
        packet = (
            dn.func.capture_continuation(subject, native)
            if explicit
            else (dn.func.capture_continuation(subject))
        )

    assert type(packet) is dn.func.Continuation
    assert packet.callback_state is None and packet._lifecycle == "unstarted"
    _close(
        (packet.state, packet.parameters, packet.constants),
        (expected.state, expected.parameters, expected.constants),
    )
    assert _target(packet.parameters, network).requires_grad == (
        _native_parameter(native, network).requires_grad
    )
    assert not _target(program.parameters, network).requires_grad
    if network:
        _assert_independent_network_clocks(packet, native)
        assert (
            _target(packet.parameters, True)
            is packet.parameters["populations"]["right"][G]
        )
        assert packet.state["clock"]["duration_remainder"].item() == pytest.approx(
            0.4 * DT
        )
    else:
        assert packet.state["control"]["duration_remainder"].item() == pytest.approx(
            0.4 * DT
        )
    _assert_owned(packet, native)
    _native_unchanged(native, saved)
    _close((program.parameters, program.constants, program.initial_state), defaults)
    assert program.execution_report() is report
    if not network:
        assert program._prepared is prepared

    frozen = _snapshot(_frames(packet))
    with torch.no_grad():
        _native_parameter(native, network).add_(0.0001)
        population = native.left if network else native
        population.v.add_(1)
    _close(_frames(packet), frozen)
    after_native_edit = _native_snapshot(native)
    with torch.no_grad():
        _target(packet.parameters, network).add_(0.0002)
        local = packet.state["populations"]["left"] if network else packet.state
        local["integrator"]["v"].add_(2)
    _native_unchanged(native, after_native_edit)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("mode", ("grad", "no_grad", "inference"))
@pytest.mark.parametrize("trainable", (False, True))
def test_capture_preserves_native_source_flags_in_every_caller_grad_mode(
    network, mode, trainable
):
    native = _network() if network else _population()
    program = _program(native)
    native_parameter = _native_parameter(native, network)
    native_parameter.requires_grad_(trainable)
    context = (
        torch.no_grad()
        if mode == "no_grad"
        else torch.inference_mode()
        if mode == "inference"
        else nullcontext()
    )
    with context:
        packet = dn.func.capture_continuation(program, native_model=None)
        assert torch.is_grad_enabled() == (mode == "grad")
        assert torch.is_inference_mode_enabled() == (mode == "inference")
    selected = _target(packet.parameters, network)
    assert selected.requires_grad is trainable
    assert selected is not native_parameter
    _assert_owned(packet, native)


@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
@pytest.mark.parametrize("reader_first", (False, True))
def test_capture_preserves_native_parameter_to_constant_aliases_and_trainability(
    runner, reader_first
):
    native, key = _cross_population_binding(reader_first=reader_first)
    program = _program(native, dt=NETWORK_DT)
    parameter = dict(native.left.named_parameters())[key]
    parameter.requires_grad_(True)
    assert native.right.factor is parameter
    subject = program if runner else program.functional
    packet = dn.func.capture_continuation(subject)
    selected = packet.parameters["populations"]["left"][key]
    assert (
        selected is packet.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert selected.requires_grad and selected.is_leaf and selected.grad_fn is None
    assert selected is not parameter and selected.data_ptr() != parameter.data_ptr()
    torch.testing.assert_close(selected, parameter)
    _assert_owned(packet, native)
    with torch.no_grad():
        selected.add_(0.25)
    torch.testing.assert_close(parameter, parameter.new_tensor([1.0]))


@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
@pytest.mark.parametrize("explicit", (False, True), ids=("source", "compatible_native"))
def test_capture_rebinds_parameter_constant_aliases_after_native_detach(
    runner, explicit
):
    source, key = _cross_population_binding()
    program = _program(source, dt=NETWORK_DT)
    native = _cross_population_binding()[0] if explicit else source
    parameter = dict(native.left.named_parameters())[key]
    parameter.requires_grad_(True)
    native.run(steps=2)
    native.detach()
    assert native.right.factor is not parameter
    assert torch._C._is_alias_of(native.right.factor, parameter)
    assert parameter.requires_grad and not native.right.factor.requires_grad
    saved = _native_snapshot(native)
    subject = program if runner else program.functional
    packet = dn.func.capture_continuation(subject, native if explicit else None)
    selected = packet.parameters["populations"]["left"][key]
    assert (
        selected is packet.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert selected.requires_grad and selected.is_leaf and selected.grad_fn is None
    assert selected is not parameter and selected.data_ptr() != parameter.data_ptr()
    _assert_owned(packet, native)
    _native_unchanged(native, saved)

    restored = dn.func.restore_continuation(
        subject, dn.func.export_continuation(subject, packet)
    )
    actual, _ = dn.func.run(subject, packet, steps=3)
    replay, _ = dn.func.run(subject, restored, steps=3)
    _close(actual, replay)
    gradient, upstream = torch.autograd.grad(
        actual["populations"]["right"]["integrator"]["v"].square().sum(),
        (selected, parameter),
        allow_unused=True,
    )
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    assert upstream is None
    _native_unchanged(native, saved)
    with torch.no_grad():
        native.run(steps=3)
    _close(actual, program.functional.extract(native).state)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_capture_retains_standalone_parametrization_buffer_trainability(network):
    owner, _ = _cross_population_binding(shared=False)
    native = owner if network else owner.right
    program = _program(native, dt=NETWORK_DT)
    buffer = owner.right.factor
    assert type(buffer) is torch.Tensor and not buffer.requires_grad
    buffer.requires_grad_(True)
    saved = _native_snapshot(native)
    with torch.inference_mode():
        packet = dn.func.capture_continuation(program)
    constants = (
        packet.constants["populations"]["right"] if network else packet.constants
    )
    selected = constants["parametrizations.factor"]
    assert selected.requires_grad and selected.is_leaf and selected.grad_fn is None
    assert selected is not buffer and selected.data_ptr() != buffer.data_ptr()
    torch.testing.assert_close(selected, buffer)
    _assert_owned(packet, native)
    _native_unchanged(native, saved)


def test_capture_owns_distinct_native_source_views_as_independent_leaves():
    native, key = _cross_population_binding(shared=False)
    parameter = dict(native.left.named_parameters())[key]
    native.right.factor = parameter.detach().view_as(parameter).requires_grad_()
    assert native.right.factor is not parameter
    assert torch._C._is_alias_of(native.right.factor, parameter)
    program = _program(native, dt=NETWORK_DT)
    saved = _native_snapshot(native)
    packet = dn.func.capture_continuation(program)
    raw = packet.parameters["populations"]["left"][key]
    config = packet.constants["populations"]["right"]["parametrizations.factor"]
    assert raw is not config and not torch._C._is_alias_of(raw, config)
    assert not raw.requires_grad and config.requires_grad
    torch.testing.assert_close(raw, config)
    _assert_owned(packet, native)
    with torch.no_grad():
        raw.add_(0.25)
    torch.testing.assert_close(config, config.new_tensor([1.0]))
    _native_unchanged(native, saved)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_cuts_native_prefix_graphs_and_trains_independent_suffix_sources(
    network, runner
):
    native = _network() if network else _population()
    program = _program(native, checkpoint_every=2)
    subject = program if runner else program.functional
    prefix_drive, prefix_inputs = _drive(program, network, 2, requires_grad=True)
    _advance(native, network, steps=2, inputs=prefix_inputs)
    assert (native.left.v if network else native.v).grad_fn is not None
    packet = dn.func.capture_continuation(subject)
    selected = _target(packet.parameters, network)
    suffix_drive, suffix_inputs = _drive(program, network, 3, requires_grad=True)
    final, _ = dn.func.run(
        subject,
        packet,
        steps=3,
        inputs=suffix_inputs,
        **({} if runner else {"checkpoint_every": 2}),
    )
    observed, drive, native_gradient, prefix_gradient = torch.autograd.grad(
        _voltage(final, network).square().sum(),
        (selected, suffix_drive, _native_parameter(native, network), prefix_drive),
        allow_unused=True,
    )
    assert all(
        value.isfinite().all() and value.abs().sum() > 0 for value in (observed, drive)
    )
    assert native_gradient is None and prefix_gradient is None
    with torch.no_grad():
        _advance(native, network, steps=3, inputs=suffix_inputs)
    _close(final, program.functional.extract(native).state)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_resumes_shared_stochastic_streams_without_consuming_randomness(
    network, runner, monkeypatch
):
    native, pulses = _stochastic_case(network)
    program = _program(native, dt=STOCHASTIC_DT)
    functional = program.functional
    subject = program if runner else functional
    for _ in range(2):
        _step_native(native)
    expected = functional.extract(native)
    saved = _native_snapshot(native)
    streams = tuple(cell.mech.probe.stream for cell in _populations(native))
    generators = tuple(stream.rng for stream in streams)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "capturing an advanced native boundary sampled or initialized"
        )

    with monkeypatch.context() as guard:
        for name in ("rand", "randn", "rand_like", "randn_like"):
            guard.setattr(torch, name, forbidden)
        guard.setattr(functional, "initialize", forbidden)
        guard.setattr(
            functional, "_initialize_host" if network else "initialize_host", forbidden
        )
        guard.setattr(functional, "sample_runtime_draws", forbidden)
        packet = dn.func.capture_continuation(subject)
    _native_unchanged(native, saved)
    _assert_borrowed_streams(native, pulses, streams, generators)
    _close(packet.state, expected.state)
    _close(_one_stream(packet.state), generators[0].get_state())
    _assert_owned(packet, native)
    frozen = _snapshot(_frames(packet))
    continuation = packet
    for _ in range(3):
        actual, auxiliary = dn.func.run(subject, continuation, steps=1)
        continuation = auxiliary["continuation"]
        _step_native(native)
        _close(actual, functional.extract(native).state)
        _close(_one_stream(actual), generators[0].get_state())
        _assert_borrowed_streams(native, pulses, streams, generators)
    _close(_frames(packet), frozen)


def test_connected_capture_retains_pending_traffic_clocks_and_portable_rng(
    monkeypatch, tmp_path
):
    native = _connected_network(training=True)
    program = _program(native, dt=CONNECTED_DT)
    functional = program.functional
    # Capture while delivery is pending, after both connection kinds have
    # affected the mixed cable/point Network.
    for _ in range(12):
        native.step()
        if (
            native.cell.mech.events.g.abs().sum() > 0
            and any(
                connection.delivery_buffer.abs().sum() > 0
                for connection in native.synapses.values()
            )
            and any(
                connection.delivery_buffer.abs().sum() > 0
                for connection in native.continuous_synapses.values()
            )
        ):
            break
    else:
        pytest.fail("the connected fixture did not reach an active delivery boundary")
    expected = functional.extract(native)
    saved = _native_snapshot(native)
    streams = (native.cable.mech.probe.primary, native.cable.mech.probe.secondary)
    generators = tuple(stream.rng for stream in streams)
    netstim_generator = native.netstim._rng

    def forbidden(*args, **kwargs):
        raise AssertionError("connected capture must not initialize or sample")

    with monkeypatch.context() as guard:
        for name in ("rand", "randn", "rand_like", "randn_like"):
            guard.setattr(torch, name, forbidden)
        guard.setattr(functional, "_initialize_host", forbidden)
        guard.setattr(functional, "sample_runtime_draws", forbidden)
        packet = dn.func.capture_continuation(program)
    _native_unchanged(native, saved)
    _assert_owned(packet, native)
    _close(
        (packet.state, packet.parameters, packet.constants),
        (expected.state, expected.parameters, expected.constants),
    )
    assert (
        packet.parameters["populations"]["cable"][CONNECTED_RAW]
        is packet.parameters["populations"]["cell"][CONNECTED_RAW]
    )
    _assert_independent_network_clocks(packet, native)
    restored = program.restore_continuation(
        _disk(program.export_continuation(packet), tmp_path / "connected.pt")
    )
    _assert_independent_network_clocks(restored, native)
    frozen = _snapshot(_frames(packet))
    actual, _ = program(packet, steps=3)
    replay, _ = program(restored, steps=3)
    _close(actual, replay)
    _native_unchanged(native, saved)
    native.run(steps=3)
    _close(actual, functional.extract(native).state)
    assert native.netstim._rng is netstim_generator
    assert all(
        stream.rng is generator
        for stream, generator in zip(streams, generators, strict=True)
    )
    _close(_frames(packet), frozen)


def test_capture_retains_sampled_connection_sources_draw_history_and_rng(monkeypatch):
    native, weights, delays, _ = _sampled_network(parametrized=True, interpolated=True)
    functional, _ = dn.func.make_functional(native, dt=NETWORK_DT)
    native.run(steps=4)
    expected = functional.extract(native)
    assert "connection_sampling" in expected.state
    saved = _native_snapshot(native)

    def forbidden(*args, **kwargs):
        raise AssertionError("capturing retained connection draws must not sample")

    with monkeypatch.context() as guard:
        for name in ("rand", "randn", "rand_like", "randn_like"):
            guard.setattr(torch, name, forbidden)
        guard.setattr(functional, "_initialize_host", forbidden)
        packet = dn.func.capture_continuation(functional)
    _native_unchanged(native, saved)
    _assert_owned(packet, native)
    _close(
        (packet.state, packet.parameters, packet.constants),
        (expected.state, expected.parameters, expected.constants),
    )
    _check_sampled_connections(functional, packet, native)
    for selected, source in zip(
        _sampled_targets(packet, True),
        _native_sampled_targets(weights, delays, True),
        strict=True,
    ):
        assert selected.requires_grad == source.requires_grad
        assert selected is not source and selected.data_ptr() != source.data_ptr()
        torch.testing.assert_close(selected, source)
    frozen = _snapshot(_frames(packet))
    actual, _ = dn.func.run(functional, packet, steps=3)
    _native_unchanged(native, saved)
    native.run(steps=3)
    _close(actual, functional.extract(native).state)
    _close(_frames(packet), frozen)


def test_capture_rebases_stale_native_workspaces_without_changing_runner_defaults(
    monkeypatch,
):
    native = _ordered_network()
    native.initialize(NETWORK_DT)
    program = _program(native, dt=NETWORK_DT, initialization="snapshot")
    original, _ = program(steps=2)
    defaults = _snapshot((program.parameters, program.constants, program.initial_state))
    with torch.no_grad():
        native.left.cm_scale_param.rho.add_(0.15)
    expected = program.functional.extract(native)
    saved = _native_snapshot(native)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "stale native preparation must be captured without initialization"
        )

    with monkeypatch.context() as guard:
        guard.setattr(program.functional, "_initialize_host", forbidden)
        packet = dn.func.capture_continuation(program)
        actual, _ = program(packet, steps=2)
    _native_unchanged(native, saved)
    _close(
        (packet.state, packet.parameters, packet.constants),
        (expected.state, expected.parameters, expected.constants),
    )
    raw = packet.parameters["populations"]["left"][RAW_CM]
    assert raw is packet.parameters["populations"]["right"][RAW_CM]
    torch.testing.assert_close(raw, native.left.cm_scale_param.rho)
    assert not torch.equal(raw, program.parameters["populations"]["left"][RAW_CM])
    native.run(steps=2)
    torch.testing.assert_close(
        _ordered_voltage(actual), _ordered_native_voltage(native)
    )
    _close(actual, program.functional.extract(native).state)
    repeated, _ = program(steps=2)
    _close(repeated, original)
    _close((program.parameters, program.constants, program.initial_state), defaults)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_captured_native_boundary_runs_through_fullgraph_compiled_backends(
    network, execution
):
    native = _network() if network else _population()
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    specifications = {"trace": dn.func.Recorder(fields)}
    program = dn.compile(
        native,
        dt=DT,
        steps=5,
        chunk_steps=4,
        inference=True,
        execution=execution,
        callbacks=specifications,
        backend="aot_eager",
        fullgraph=True,
    )
    with torch.no_grad():
        _advance(native, network, steps=2)
    packet = dn.func.capture_continuation(program)
    saved = _native_snapshot(native)
    actual, auxiliary = program(packet)
    _native_unchanged(native, saved)
    assert not _voltage(actual, network).requires_grad
    callbacks = dn.func.make_callbacks(native, specifications, dt=DT)
    with torch.no_grad():
        expected = native.run(
            steps=5,
            callbacks=callbacks,
            **({} if network else {"dt": DT}),
        )
    _close(actual, program.functional.extract(native).state)
    _close(auxiliary["callbacks"].outputs, expected.outputs)
    assert all(
        value.shape[0] == 6 for value in auxiliary["callbacks"]["trace"].values()
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_captured_checkpoint_starts_callbacks_once_at_the_advanced_boundary(
    network, monkeypatch, tmp_path
):
    native = _network() if network else _population()
    program = _program(native, callbacks=_callbacks(network), checkpoint_every=1)
    _advance(native, network, steps=2)
    calls = []
    initialize = program.callbacks._initialize

    def counted(*args, **kwargs):
        calls.append(None)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(program.callbacks, "_initialize", counted)
    packet = dn.func.capture_continuation(program)
    portable = _disk(program.export_continuation(packet), tmp_path / "captured.pt")
    restored = program.restore_continuation(portable)
    assert calls == [] and restored.callback_state is None
    assert restored._lifecycle == packet._lifecycle == "unstarted"
    _close(_frames(restored), _frames(packet))
    assert _target(restored.parameters, network).requires_grad

    def forbidden(*args, **kwargs):
        raise AssertionError("captured state must bypass fresh runner initialization")

    monkeypatch.setattr(
        program.functional,
        "_initialize_host" if network else "initialize_host",
        forbidden,
    )
    _, first = program(restored, steps=2)
    assert calls == [None]
    key = "left.v" if network else "v"
    boundary = restored.state["populations"]["left"] if network else restored.state
    torch.testing.assert_close(
        first["callbacks"]["trace"][key][0], boundary["integrator"]["v"]
    )
    assert first["callbacks"]["trace"][key].shape[0] == 3
    resumed = program.restore_continuation(
        _disk(
            program.export_continuation(first["continuation"]), tmp_path / "active.pt"
        )
    )
    _, last = program(resumed, steps=1)
    assert calls == [None]
    assert last["callbacks"]["trace"][key].shape[0] == 1


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_rejects_incompatible_native_models_without_changing_them(
    network, runner
):
    source = _network() if network else _population()
    program = _program(source)
    subject = program if runner else program.functional
    wrong_kind = _population() if network else _network()
    saved = _native_snapshot(wrong_kind)
    with pytest.raises((dn.func.FunctionalizationError, TypeError, ValueError)):
        dn.func.capture_continuation(subject, wrong_kind)
    _native_unchanged(wrong_kind, saved)
    if network:
        broken = _network()
        previous = broken.right.mech.leak.g_param
        broken.right.mech.leak.g_param = torch.nn.Parameter(previous.detach().clone())
        saved = _native_snapshot(broken)
        with pytest.raises(ValueError, match="shared tensor aliases") as error:
            dn.func.capture_continuation(subject, broken)
        assert type(error.value) is ValueError
        _native_unchanged(broken, saved)


@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_rejects_a_configured_population_timestep_mismatch_atomically(runner):
    source, native = _population(), _population()
    program = _program(source)
    subject = program if runner else program.functional
    native.run(steps=1, dt=2 * DT)
    assert native.initialized and native.integrator.initialized
    assert native.integrator.dt != program.functional.dt
    source_saved, native_saved = _native_snapshot(source), _native_snapshot(native)
    with pytest.raises(dn.func.FunctionalizationError, match="timestep.*match"):
        dn.func.capture_continuation(subject, native)
    _native_unchanged(source, source_saved)
    _native_unchanged(native, native_saved)


@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_rejects_a_built_uninitialized_population_atomically(runner):
    source, native = _population(), _population()
    program = _program(source)
    subject = program if runner else program.functional
    native.initialized = False
    assert native.is_built and not native.initialized
    saved = _native_snapshot(native)
    with pytest.raises(dn.func.FunctionalizationError, match="Initialize.*Population"):
        dn.func.capture_continuation(subject, native)
    assert not native.initialized and native.is_built
    _native_unchanged(native, saved)


@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_capture_rejects_unsupported_target_types(runner):
    program = _program(_population())
    subject = program if runner else program.functional

    class Derived(type(subject)):
        pass

    for invalid in (object(), _population(), object.__new__(Derived)):
        with pytest.raises(TypeError, match="plan|runner|FunctionalPopulation"):
            dn.func.capture_continuation(invalid)


@pytest.mark.parametrize("context", ("compile", "grad", "forward_ad"))
def test_capture_rejects_transforms_before_native_extraction(monkeypatch, context):
    program = _program(_population())

    def forbidden(*args, **kwargs):
        raise AssertionError("transform rejection must precede native extraction")

    monkeypatch.setattr(program.functional, "_extract", forbidden)

    def work(value):
        dn.func.capture_continuation(program)
        return value.square()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="capture_continuation must run outside torch.compile",
    ):
        if context == "compile":
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            work(torch.tensor(0.5))
        elif context == "grad":
            torch.func.grad(work)(torch.tensor(0.5))
        else:
            with torch.autograd.forward_ad.dual_level():
                work(torch.tensor(0.5))


@pytest.mark.parametrize("kind", ("cable", "multi"))
def test_capture_cable_and_mixed_population_histories_continue_native_execution(kind):
    if kind == "cable":
        from tests.func.test_native_cable import DT as dt
        from tests.func.test_native_cable import _model as factory
    else:
        from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
        from tests.func.test_multi_population import DT as dt
        from tests.func.test_multi_population import _mixed_multi as factory

        if not DENDRA_SOLVERS_AVAILABLE:
            pytest.skip("mixed MultiPopulation requires dendra-solvers")
    native = factory()
    functional, _ = dn.func.make_functional(native, dt=dt)
    native.run(steps=2, dt=dt)
    saved = _native_snapshot(native)
    packet = dn.func.capture_continuation(functional)
    _native_unchanged(native, saved)
    _assert_owned(packet, native)
    actual, _ = dn.func.run(functional, packet, steps=3)
    native.run(steps=3, dt=dt)
    _close(actual, functional.extract(native).state)
