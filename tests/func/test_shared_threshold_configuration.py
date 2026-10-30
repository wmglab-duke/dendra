"""Native threshold configurations prepare independent pure callback execution."""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

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
_THRESHOLDS = (
    (dn.callbacks.APCount, dn.func.APCount),
    (dn.callbacks.Raster, dn.func.Raster),
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


def _native_model(voltage):
    return SimpleNamespace(
        v=voltage,
        shape=tuple(voltage.shape),
        n_ax=voltage.shape[-2],
        nc=voltage.shape[-1],
        device=lambda: voltage.device,
    )


def _initial(tensors, parameters=None, constants=None):
    return dn.func.Continuation(
        tensors.state,
        tensors.parameters if parameters is None else parameters,
        tensors.constants if constants is None else constants,
    )


@pytest.mark.parametrize("native_type,pure_type", _THRESHOLDS)
def test_native_configuration_matches_independent_threshold_and_native_window_semantics(
    native_type,
    pure_type,
):
    functional, _ = _threshold_case()
    voltage = _voltage_sequence((2, 3, *functional.shape))
    configured = native_type(**_configuration())
    callbacks = functional.make_callbacks({"analysis": configured})
    prepared = callbacks._plans[0].callback
    assert type(prepared) is pure_type
    assert not callbacks._plans[0].validate_updates
    carry, observed = _execute(prepared, voltage)
    below, counts, events = _manual(voltage, torch.tensor([0, 0, 3, 3]))
    _close(carry[0], below)
    assert carry[-1].item() == voltage.shape[0]
    expected = counts if pure_type is dn.func.APCount else events[1:6]
    _close(observed, expected)
    assert configured.record is configured.state_cache is None and configured.i == 0
    assert configured.node_check == [0, 0, -1, -1]
    reference = native_type(**_configuration())
    model = _native_model(voltage[0])
    reference.pre_loop_hook(model)
    for frame in voltage:
        model.v = frame
        reference.post_step_hook(model)
    _close(reference.state_cache, below)
    if pure_type is dn.func.APCount:
        assert observed.dtype == reference.record.dtype == torch.int32
        _close(observed, reference.record)
    else:
        assert observed.dtype == torch.bool and observed.shape[0] == 5
        assert len(reference.record) == 5
        _close(reference.stack(), observed)


@pytest.mark.parametrize("native", (False, True), ids=("functional", "native"))
@pytest.mark.parametrize("start,end", ((1, 6), (6, 1), (8, 9), (-2, 3), (0, None)))
def test_raster_returns_only_window_rows_across_empty_and_off_window_segments(
    native, start, end
):
    functional, _ = _threshold_case()
    voltage = _voltage_sequence(functional.shape)
    family = dn.callbacks.Raster if native else dn.func.Raster
    callback = (
        functional.make_callbacks(
            {
                "raster": family(
                    threshold=0.0,
                    node_check=[0, 0, -1],
                    t_start_check=start * DT,
                    t_end_check=None if end is None else end * DT,
                    dt=DT,
                ),
            }
        )
        ._plans[0]
        .callback
    )
    beginning = int(start * DT / DT)
    ending = voltage.shape[0] if end is None else int(end * DT / DT)
    below, _, full = _manual(
        voltage, torch.tensor([0, 0, 3]), start=beginning, end=ending
    )
    carry, initial = callback.initialize({"integrator": {"v": voltage[0]}}, None)
    assert initial is None
    pieces = []
    for begin, stop in ((0, 1), (1, 1), (1, 3), (3, 7), (7, 7)):
        emissions = []
        for frame in voltage[begin:stop]:
            carry, emitted = callback.update(carry, {"integrator": {"v": frame}}, None)
            emissions.append(emitted)
        observed = callback.finalize(
            carry, torch.stack(emissions) if emissions else None
        )
        lower, upper = max(begin, beginning), min(stop, ending)
        expected = full[lower:upper] if lower < upper else full[:0]
        _close(observed, expected)
        assert observed.shape[0] == max(upper - lower, 0)
        assert observed.dtype == torch.bool and not observed.requires_grad
        pieces.append(observed)
    lower, upper = max(0, beginning), min(voltage.shape[0], ending)
    expected = full[lower:upper] if lower < upper else full[:0]
    _close(torch.cat(pieces), expected)
    _close(carry[0], below)
    assert carry[-1].item() == voltage.shape[0]


def test_finite_window_raster_pure_hooks_support_vmap_and_atomic_dynamic_capture():
    functional, _ = _threshold_case()
    observer = (
        functional.make_callbacks(
            {
                "raster": dn.func.Raster(**_configuration()),
            }
        )
        ._plans[0]
        .callback
    )
    voltage = _voltage_sequence(functional.shape)
    lanes = torch.stack((voltage, voltage.flip(0)))

    def execute(frames):
        return _execute(observer, frames)[1]

    expected = torch.stack(
        [_manual(frames, torch.tensor([0, 0, 3, 3]))[2][1:6] for frames in lanes]
    )
    _close(torch.vmap(execute)(lanes), expected)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(execute, backend="aot_eager", fullgraph=True)
        observed = compiled(voltage.requires_grad_())
    _close(observed, expected[0])
    assert not observed.requires_grad


@pytest.mark.parametrize("native_type,pure_type", _THRESHOLDS)
def test_reused_native_configuration_keeps_authored_negative_indices_and_omitted_dt(
    native_type,
    pure_type,
    monkeypatch,
):
    from dendra.models.backend import Backend

    functional, tensors = _threshold_case()
    nodes = torch.tensor([0, -1])
    monkeypatch.setattr(Backend, "dt", 7 * DT)
    configured = native_type(node_check=nodes, threshold=0.0)
    previous = _native_model(torch.zeros((2, 3), dtype=torch.float64))
    configured.pre_loop_hook(previous)
    previous.v.fill_(1.0)
    configured.post_step_hook(previous)
    published = configured.node_check
    _close(published, torch.tensor([0, 2]))
    record, latch, counter, effective_dt = (
        configured.record,
        configured.state_cache,
        configured.i,
        configured.dt,
    )
    history_values = _snapshot((record, latch, published))
    with torch.inference_mode():
        callbacks = functional.make_callbacks({"analysis": configured})
    prepared = callbacks._plans[0].callback
    assert type(prepared) is pure_type
    _close(prepared.node_indices, torch.tensor([0, 3]))
    assert not prepared.node_indices.is_inference()
    assert prepared.callback_dt == DT
    assert prepared.end_step == int(1e9 / DT)
    nodes[0] = 1
    configured.threshold = 99.0
    rebound = functional.make_callbacks({"analysis": configured})
    _close(rebound._plans[0].callback.node_indices, torch.tensor([1, 3]))
    _close(prepared.node_indices, torch.tensor([0, 3]))
    assert prepared.threshold == 0.0
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=2, callbacks=callbacks
    )
    assert auxiliary["callbacks"].state["analysis"][-1].item() == 2
    assert configured.record is record and configured.state_cache is latch
    assert configured.node_check is published and configured.i == counter
    assert configured.dt == effective_dt
    _close((record, latch, published), history_values)


@pytest.mark.parametrize("native_type,pure_type", _THRESHOLDS)
def test_public_native_selector_edits_and_explicit_dt_remain_authoritative(
    native_type, pure_type
):
    functional, _ = _threshold_case()
    configured = native_type(node_check=[0, -1], dt=DT)
    configured.pre_loop_hook(_native_model(torch.zeros((2, 3), dtype=torch.float64)))
    configured.node_check.fill_(1)
    first = functional.make_callbacks({"analysis": configured})._plans[0].callback
    assert type(first) is pure_type
    _close(first.node_indices, torch.tensor([1, 1]))
    replacement = [0, -1, 0]
    configured.node_check = replacement
    replacement[0] = -2
    second = functional.make_callbacks({"analysis": configured})._plans[0].callback
    _close(second.node_indices, torch.tensor([2, 3, 0]))
    configured.dt = 2 * DT
    with pytest.raises(dn.func.FunctionalizationError, match="dt.*match|timestep"):
        functional.make_callbacks({"analysis": configured})
    configured.dt = DT
    configured.node_check = [functional.shape[-1]]
    with pytest.raises(IndexError, match="valid compartment indices"):
        functional.make_callbacks({"analysis": configured})


@pytest.mark.parametrize("shape", ((2, 4), (1, 1), (2, 3, 1, 4)))
def test_native_anomaly_configuration_starts_fresh_and_preserves_all_batch_axes(shape):
    functional, tensors = _threshold_case()
    configured = dn.callbacks.AnomalyDetector()
    marker = torch.tensor([True, False])
    configured.rec = marker
    callbacks = functional.make_callbacks({"anomaly": configured})
    assert type(callbacks._plans[0].callback) is dn.func.AnomalyDetector
    assert not callbacks._plans[0].validate_updates
    initial, emitted = callbacks._initialize(tensors.state)
    assert emitted is None and not initial["anomaly"].any()
    _, zero = dn.func.run(functional, _initial(tensors), steps=0, callbacks=callbacks)
    assert zero["callbacks"]["anomaly"].shape == functional.shape[:-1]
    assert not zero["callbacks"]["anomaly"].any()
    observer = callbacks._plans[0].callback
    voltage = torch.zeros(shape, dtype=torch.float64)
    carry, _ = observer.initialize({"integrator": {"v": voltage}}, None)
    native = dn.callbacks.AnomalyDetector()
    model = _native_model(voltage)
    native.pre_loop_hook(model)
    assert native.rec.shape == carry.shape == shape[:-1]
    expected = torch.zeros(shape[:-1], dtype=torch.bool)
    for value, row in ((float("nan"), 0), (float("inf"), -1), (0.0, 0)):
        voltage = torch.zeros_like(voltage)
        voltage.reshape(-1, shape[-1])[row, 0] = value
        if value != 0.0:
            expected.reshape(-1)[row] = True
        model.v = voltage
        native.post_step_hook(model)
        carry, event = observer.update(carry, {"integrator": {"v": voltage}}, None)
        assert event is None
        assert native.rec.shape == carry.shape == shape[:-1]
        _close(carry, expected)
        _close(observer.finalize(carry, None), native.rec)
    assert carry.dtype == torch.bool and carry.any()
    assert configured.rec is marker
    _close(marker, torch.tensor([True, False]))


@pytest.mark.parametrize(
    "native_type",
    (dn.callbacks.APCount, dn.callbacks.Raster, dn.callbacks.AnomalyDetector),
)
def test_native_population_callbacks_require_explicit_network_scope(native_type):
    functional, tensors, parameters, constants, _, _ = _case(True)
    options = {} if native_type is dn.callbacks.AnomalyDetector else {"node_check": [0]}
    configured = native_type(**options)
    with pytest.raises(
        dn.func.FunctionalizationError, match="Population-specific|PopulationCallback"
    ):
        functional.make_callbacks({"analysis": configured})
    callbacks = functional.make_callbacks(
        {
            "analysis": dn.func.PopulationCallback("left", configured),
        }
    )
    _, auxiliary = dn.func.run(
        functional,
        _initial(tensors, parameters, constants),
        steps=3,
        callbacks=callbacks,
    )
    assert tuple(auxiliary["callbacks"]) == ("analysis",)
    if native_type is dn.callbacks.AnomalyDetector:
        assert configured.rec is None
    else:
        assert configured.record is None


@pytest.mark.parametrize(
    "native_type",
    (dn.callbacks.APCount, dn.callbacks.Raster, dn.callbacks.AnomalyDetector),
)
def test_native_subclasses_and_custom_imperative_hooks_are_not_silently_lowered(
    native_type,
):
    functional, _ = _threshold_case()
    options = {} if native_type is dn.callbacks.AnomalyDetector else {"node_check": [0]}

    class Derived(native_type):
        pass

    for configured in (Derived(**options), native_type(**options)):
        if type(configured) is native_type:
            configured.post_step_hook = lambda model: None
        with pytest.raises(
            (TypeError, dn.func.FunctionalizationError),
            match="FunctionalCallback|pure|custom|hook|configuration",
        ):
            functional.make_callbacks({"analysis": configured})
    for excluded in (
        dn.callbacks.Active,
        dn.callbacks.ActiveAL,
        dn.callbacks.ActiveALCount,
    ):
        with pytest.raises(
            (TypeError, dn.func.FunctionalizationError),
            match="FunctionalCallback|pure|custom|configuration",
        ):
            functional.make_callbacks({"analysis": excluded(node_check=[0])})


def test_explicit_functional_binding_retains_authority_for_a_native_subclass():
    functional, tensors = _threshold_case()
    calls = []

    class PureCounter(dn.callbacks.APCount, dn.func.FunctionalCallback):
        def _bind(self, plan):
            calls.append(plan)
            return self

        def initialize(self, state, auxiliary):
            return (), state["integrator"]["v"].sum()

        def update(self, carry, state, auxiliary):
            return carry, state["integrator"]["v"].sum()

        def finalize(self, carry, emissions):
            return emissions

    configured = PureCounter(node_check=[0])
    callbacks = functional.make_callbacks({"custom": configured})
    assert calls == [functional] and callbacks._plans[0].callback is configured
    assert callbacks._plans[0].validate_updates
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=2, callbacks=callbacks
    )
    assert auxiliary["callbacks"]["custom"].shape == (3,)
    assert configured.record is configured.state_cache is None and configured.i == 0


def _specifications(network, *, native):
    namespace = dn.callbacks if native else dn.func
    options = dict(
        threshold=-59.99,
        node_check=[0, 0, -1],
        t_start_check=2 * DT,
        t_end_check=7 * DT,
    )
    sources = (
        namespace.APCount(**options),
        namespace.Raster(**options),
        namespace.AnomalyDetector(),
    )

    def scoped(callback):
        return dn.func.PopulationCallback("left", callback) if network else callback

    return {
        "count": scoped(sources[0]),
        "raster": scoped(sources[1]),
        "anomaly": scoped(sources[2]),
        "trace": dn.Recorder(("left.v", "right.v", "t") if network else ("v", "t")),
    }, sources


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_compile_accepts_native_analysis_configurations_without_mutating_history(
    network,
):
    configured, sources = _specifications(network, native=True)
    program = dn.compile(
        _network() if network else _population(),
        dt=DT,
        steps=3,
        callbacks=configured,
        backend="eager",
        disable=True,
    )
    assert all(not plan.validate_updates for plan in program.callbacks._plans)
    _, auxiliary = program()
    assert set(auxiliary["callbacks"]) == set(configured)
    assert auxiliary["callbacks"].state["count"][-1].item() == 3
    assert sources[0].record is sources[0].state_cache is None and sources[0].i == 0
    assert sources[1].record is sources[1].state_cache is None and sources[1].i == 0
    assert sources[2].rec is None


def test_native_configurations_keep_existing_callback_checkpoint_metadata(tmp_path):
    functional, tensors, parameters, constants, _, _ = _case(False)
    native, _ = _specifications(False, native=True)
    pure, _ = _specifications(False, native=False)
    callbacks, equivalent = (
        functional.make_callbacks(native),
        functional.make_callbacks(pure),
    )
    initial = _initial(tensors, parameters, constants)
    _, first = dn.func.run(functional, initial, steps=3, callbacks=callbacks)
    _, reference = dn.func.run(functional, initial, steps=3, callbacks=equivalent)
    saved = callbacks.export_state(first["callbacks"].state)
    expected = equivalent.export_state(reference["callbacks"].state)
    assert saved["format"] == "dendra.func.callback_state" and saved["version"] == 1
    _assert_packet_equal(saved, expected)
    received = equivalent.restore_state(_disk(saved, tmp_path / "callbacks.pt"))
    assert received.plan_token is equivalent._state_token
    _close(received.carries, first["callbacks"].state.carries)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("legacy_dtype", (torch.int64, torch.float32))
def test_legacy_count_leaves_migrate_in_portable_continuations_without_changing_phase(
    network, legacy_dtype, tmp_path
):
    functional, tensors, parameters, constants, _, _ = _case(network)

    def configured(native):
        namespace = dn.callbacks if native else dn.func
        options = dict(node_check=[0, 0, -1], t_end_check=7 * DT)
        entries = {
            "count": namespace.APCount(**options),
            "active": dn.func.ActiveAL(**options, at_least=2),
            "events": dn.func.ActiveALCount(**options, at_least=2),
        }
        return {
            name: dn.func.PopulationCallback("left", callback) if network else callback
            for name, callback in entries.items()
        }

    callbacks = functional.make_callbacks(configured(native=True))
    receiver = functional.make_callbacks(configured(native=False))
    _, prefix = dn.func.run(
        functional,
        _initial(tensors, parameters, constants),
        steps=3,
        callbacks=callbacks,
    )
    packet = dn.func.export_continuation(
        functional, prefix["continuation"], callbacks=callbacks
    )
    canonical = _clone(packet["callbacks"])
    for name, (below, count, step) in canonical["carries"].items():
        values = torch.arange(count.numel(), dtype=torch.int32).reshape(count.shape) + 3
        canonical["carries"][name] = (below, values, step)
        packet["callbacks"]["carries"][name] = (below, values.to(legacy_dtype), step)
    loaded = _disk(packet, tmp_path / "legacy-continuation.pt")
    frozen = _clone(loaded)
    resumed = dn.func.restore_continuation(functional, loaded, callbacks=receiver)
    _assert_packet_equal(loaded, frozen)
    assert loaded["callbacks"]["version"] == 1
    for name, carry in resumed.callback_state.carries.items():
        assert carry[0].dtype == torch.bool and carry[1].dtype == torch.int32
        assert carry[2].dtype == torch.int64 and carry[2].item() == 3
        _close(carry, canonical["carries"][name])
    reference = dn.func.Continuation(
        prefix["continuation"].state,
        parameters,
        constants,
        receiver.restore_state(canonical),
    )
    with torch.no_grad():
        actual, tail = dn.func.run(functional, resumed, steps=2, callbacks=receiver)
        expected, control = dn.func.run(
            functional, reference, steps=2, callbacks=receiver
        )
    _close(actual, expected)
    _close(tail["callbacks"].outputs, control["callbacks"].outputs)
    for carry in tail["callbacks"].state.carries.values():
        assert carry[1].dtype == torch.int32 and carry[2].item() == 5


def _compiled_case(network, execution, tmp_path):
    functional, tensors, parameters, constants, _, _ = _case(network)
    configurations, sources = _specifications(network, native=True)
    sources[0].record = torch.full((2, 3), 37.0)
    sources[0].state_cache = torch.zeros((2, 3), dtype=torch.bool)
    sources[0].i = 37
    sources[1].record = [torch.ones((2, 3), dtype=torch.bool)]
    sources[1].state_cache = torch.zeros((2, 3), dtype=torch.bool)
    sources[1].i = 37
    sources[2].rec = torch.tensor([True, False])
    history = [
        (callback.record, callback.state_cache, callback.i) for callback in sources[:2]
    ]
    flags = sources[2].rec
    history_values = _snapshot((history, flags))
    callbacks = functional.make_callbacks(configurations)
    pure, _ = _specifications(network, native=False)
    equivalent = functional.make_callbacks(pure)
    initial = _initial(tensors, parameters, constants)
    source_model = _snapshot(functional.extract().state)
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    checkpoint = {"checkpoint_every": 3} if execution == "scan" else {}
    with nullcontext() if execution == "scan" else torch.no_grad():
        expected, whole = dn.func.run(
            functional, initial, steps=9, callbacks=equivalent
        )
        _, first = dn.func.run(
            kernel, initial, steps=5, callbacks=callbacks, **checkpoint
        )
        actual, last = dn.func.run(
            kernel, first["continuation"], steps=4, callbacks=callbacks, **checkpoint
        )
    _close(actual, expected)
    joined = _joined(first, last)
    _close(joined, whole["callbacks"]["trace"])
    _close(
        torch.cat((first["callbacks"]["raster"], last["callbacks"]["raster"])),
        whole["callbacks"]["raster"],
    )
    for name in ("count", "anomaly"):
        _close(last["callbacks"][name], whole["callbacks"][name])
    assert last["callbacks"]["count"].dtype == torch.int32
    assert (
        last["callbacks"]["raster"].dtype
        == last["callbacks"]["anomaly"].dtype
        == torch.bool
    )
    if execution == "scan":
        target = _target(parameters, network)
        observed_loss = _voltage(actual, network).square().sum() + sum(
            value.square().mean() for name, value in joined.items() if name != "t"
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
    saved = _disk(
        dn.func.export_continuation(
            functional, first["continuation"], callbacks=callbacks
        ),
        tmp_path / "continuation.pt",
    )
    receiving, _ = _specifications(network, native=True)
    receiving = functional.make_callbacks(receiving)
    restored = dn.func.restore_continuation(functional, saved, callbacks=receiving)
    with torch.no_grad():
        resumed, tail = dn.func.run(kernel, restored, steps=4, callbacks=receiving)
    _close(resumed, expected)
    _close(tail["callbacks"].outputs, last["callbacks"].outputs)
    assert tail["callbacks"]["trace"]["t"].shape[0] == 4
    for callback, (record, latch, counter) in zip(sources[:2], history, strict=True):
        assert (
            callback.record is record
            and callback.state_cache is latch
            and callback.i == counter
        )
    assert sources[2].rec is flags
    _close((history, flags), history_values)
    _close(functional.extract().state, source_model)
    return actual


@pytest.mark.parametrize("network,execution", ((False, "scan"), (True, "while")))
def test_compiled_native_configurations_preserve_gradients_and_portable_continuation(
    network, execution, tmp_path
):
    _compiled_case(network, execution, tmp_path)


@pytest.mark.mps
def test_mps_native_threshold_configurations_support_compiled_while_and_checkpoint_continuation(
    tmp_path,
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        actual = _compiled_case(False, "while", tmp_path)
        assert _voltage(actual, False).device.type == "mps"
