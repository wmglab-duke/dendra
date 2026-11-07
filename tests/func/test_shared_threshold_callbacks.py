"""Native and prepared public threshold callbacks share pure event semantics."""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Active as NativeActive
from dendra.models.callbacks import ActiveAL as NativeActiveAL
from dendra.models.callbacks import ActiveALCount as NativeActiveALCount
from dendra.models.callbacks import APCount as NativeAPCount
from dendra.models.callbacks import Raster as NativeRaster
from tests.func.test_bound_continuation import _case
from tests.func.test_callback_checkpoints import _assert_packet_equal, _model
from tests.func.test_continuation import _close, _joined, _snapshot
from tests.func.test_lowering_api import DT, _target, _voltage

pytestmark = pytest.mark.cpu
_FAMILIES = (
    (dn.func.Raster, NativeRaster, {}, dn.func.Raster, "Raster"),
    (dn.func.APCount, NativeAPCount, {}, dn.func.APCount, "APCount"),
    (dn.func.Active, NativeActive, {"inv": True}, dn.func.ActiveAL, "ActiveAL"),
    (dn.func.ActiveAL, NativeActiveAL, {"at_least": 2}, dn.func.ActiveAL, "ActiveAL"),
    (
        dn.func.ActiveALCount,
        NativeActiveALCount,
        {"at_least": 2, "inv": True},
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


def _threshold_case():
    with dn.ctx(IMEM=True):
        return dn.func.make_functional(_model(), dt=DT)


def _configuration(**changes):
    return {
        "threshold": 0.0,
        "node_check": [0, 0, -1, -1],
        "t_start_check": DT,
        "t_end_check": 6 * DT,
        "dt": DT,
        **changes,
    }


def _voltage_sequence(shape):
    voltage = torch.full((7, *shape), -1.0, dtype=torch.float64)
    view = (7, *([1] * (voltage.ndim - 2)))
    voltage[..., 0] = voltage.new_tensor([1, 1, -1, 0, float("nan"), 1, -1]).reshape(
        view
    )
    voltage[..., -1] = voltage.new_tensor([-1, 1, 1, -1, 0, 0, 1]).reshape(view)
    return voltage


def _manual(voltage, indices, start=1, end=6):
    selected = voltage.index_select(-1, indices)
    below = torch.ones_like(selected[0], dtype=torch.bool)
    counts = torch.zeros_like(selected[0], dtype=torch.int32)
    events = []
    for step, frame in enumerate(selected):
        event = torch.zeros_like(below)
        if start <= step < end:
            above = frame >= 0.0
            event = above & below
            below = ~above
            counts = counts + event.to(counts.dtype)
        events.append(event)
    return below, counts, torch.stack(events)


def _execute(callback, voltage):
    carry, initial = callback.initialize({"integrator": {"v": voltage[0]}}, None)
    assert initial is None
    events = []
    for frame in voltage:
        carry, event = callback.update(carry, {"integrator": {"v": frame}}, None)
        if event is not None:
            events.append(event)
    emissions = torch.stack(events) if events else None
    return carry, callback.finalize(carry, emissions)


@pytest.mark.parametrize("family,native_type,options,canonical,kind", _FAMILIES)
def test_prepared_public_families_match_native_nan_equality_windows_and_duplicate_slots(
    family, native_type, options, canonical, kind
):
    functional, _ = _threshold_case()
    shape = (2, 3, *functional.shape)
    voltage = _voltage_sequence(shape)
    activity = issubclass(family, dn.func.ActiveAL)
    partition = [2, 2] if activity else None
    authored = family(
        **_configuration(), **options, **({"partition": partition} if activity else {})
    )
    prepared = authored._bind(functional)
    assert type(prepared) is canonical
    carry, result = _execute(prepared, voltage)
    below, counts, raster = _manual(voltage, torch.tensor([0, 0, 3, 3]))
    torch.testing.assert_close(carry[0], below)
    assert carry[-1].item() == 7
    if kind == "Raster":
        assert len(carry) == 2 and result.dtype == torch.bool
        torch.testing.assert_close(result, raster[1:6])
        assert result.shape[0] == 5
    else:
        assert len(carry) == 3 and carry[1].dtype == torch.int32
        torch.testing.assert_close(carry[1], counts)
        assert not result.requires_grad
    native = native_type(**_configuration(), **options)
    model = SimpleNamespace(
        v=voltage[0], shape=shape, nc=shape[-1], device=lambda: voltage.device
    )
    native.pre_loop_hook(model)
    for frame in voltage:
        model.v = frame
        native.post_step_hook(model)
    torch.testing.assert_close(native.state_cache, below)
    if kind == "Raster":
        assert len(native.record) == 5
        torch.testing.assert_close(native.stack(), result)
    else:
        assert native.record.dtype == torch.int32
        torch.testing.assert_close(native.record, counts)
        expected = native.is_active(partition) if activity else native.record
        torch.testing.assert_close(result, expected)


def test_shared_activity_reduction_distinguishes_repeated_events_from_distinct_slots():
    functional, _ = _threshold_case()
    voltage = _voltage_sequence(functional.shape)
    voltage[..., -1] = -1.0
    configuration = _configuration(node_check=[0, -1])
    sites = dn.func.ActiveAL(**configuration, at_least=2)._bind(functional)
    events = dn.func.ActiveALCount(**configuration, at_least=3)._bind(functional)
    sites_carry, sites_result = _execute(sites, voltage)
    events_carry, events_result = _execute(events, voltage)
    torch.testing.assert_close(sites_carry, events_carry)
    assert not sites_result.any() and events_result.all()


@pytest.mark.parametrize("family,native_type,options,canonical,kind", _FAMILIES)
def test_binding_snapshots_immutable_public_execution_configuration_without_mutating_authored_objects(
    family, native_type, options, canonical, kind
):
    functional, tensors = _threshold_case()
    nodes = torch.tensor([0, 0, -1, -1])
    activity = issubclass(family, dn.func.ActiveAL)
    partition = [2, 2]
    authored = family(
        **_configuration(node_check=nodes),
        **options,
        **({"partition": partition} if activity else {}),
    )
    with torch.inference_mode():
        prepared = authored._bind(functional)
    assert type(prepared) is canonical and prepared is not authored
    assert authored.node_check is nodes
    assert prepared.node_indices is not nodes
    assert not prepared.node_indices.is_inference()
    assert prepared.node_indices.dtype == torch.int64
    assert (
        prepared.callback_dt == DT
        and prepared.start_step == 1
        and prepared.end_step == 6
    )
    assert prepared._bind(functional) is prepared
    # Ordinary indices must also work when voltage participates in autograd.
    _execute(prepared, _voltage_sequence(functional.shape).requires_grad_())
    if activity:
        assert authored.partition is partition and prepared.partition == (2, 2)
    before = _snapshot(prepared.node_indices)
    nodes.fill_(1)
    partition[:] = [1, 3]
    authored.threshold = 100.0
    authored.dt = 2 * DT
    authored.t_start_check = 0.0
    torch.testing.assert_close(prepared.node_indices, before)
    assert prepared.threshold == 0.0 and prepared.callback_dt == DT
    if activity:
        assert prepared.partition == (2, 2)
    for name, value in (
        ("threshold", 1.0),
        ("node_indices", nodes),
        ("new_attribute", None),
    ):
        with pytest.raises((AttributeError, TypeError)):
            setattr(prepared, name, value)
    with pytest.raises((AttributeError, TypeError)):
        del prepared.threshold
    collection = functional.make_callbacks({"analysis": prepared})
    assert collection._plans[0].callback is prepared
    assert collection._plans[0].validate_updates is False
    with pytest.raises(dn.func.FunctionalizationError, match="threshold callback dt"):
        collection._validate_runtime_dt(2 * DT)
    carry, emission = prepared.initialize(tensors.state, None)
    assert emission is None and carry[-1].item() == 0
    repeated, _ = prepared.initialize(tensors.state, None)
    for first, second in zip(carry, repeated, strict=True):
        assert first is not second and first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("family,native_type,options,canonical,kind", _FAMILIES)
def test_authored_execution_remains_guarded_and_default_subclasses_collapse_to_canonical_family(
    family, native_type, options, canonical, kind
):
    functional, tensors = _threshold_case()
    authored = family(**_configuration(), **options)
    for execute in (
        lambda: authored.initialize(tensors.state, None),
        lambda: authored.update(None, tensors.state, None),
        lambda: authored.finalize(None, None),
    ):
        with pytest.raises(dn.func.FunctionalizationError, match="make_callbacks"):
            execute()
    with pytest.raises(
        dn.func.FunctionalizationError, match="unsupported|custom callbacks"
    ):
        authored.checkpoint_configuration()

    class Inherited(family):
        def initialize(self, state, auxiliary):
            raise AssertionError(
                "default binding must retain canonical family behavior"
            )

        def update(self, carry, state, auxiliary):
            raise AssertionError(
                "default binding must retain canonical family behavior"
            )

        def finalize(self, carry, emissions):
            raise AssertionError(
                "default binding must retain canonical family behavior"
            )

    inherited = Inherited(**_configuration(), **options)
    collection = functional.make_callbacks({"analysis": inherited})
    assert type(collection._plans[0].callback) is canonical
    assert collection._plans[0].validate_updates is False


class _Replacement(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        return carry + 1, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


@pytest.mark.parametrize("family,native_type,options,canonical,kind", _FAMILIES)
def test_custom_bound_type_factories_keep_the_normalized_positional_argument_contract(
    family, native_type, options, canonical, kind
):
    functional, _ = _threshold_case()
    calls = []
    replacement = _Replacement()

    def factory(*arguments):
        calls.append(arguments)
        return replacement

    class CustomFactory(family):
        _bound_type = staticmethod(factory)

    activity = issubclass(family, dn.func.ActiveAL)
    authored = CustomFactory(
        **_configuration(), **options, **({"partition": [2, 2]} if activity else {})
    )
    collection = functional.make_callbacks({"analysis": authored})
    assert collection._plans[0].callback is replacement
    assert collection._plans[0].validate_updates is True
    assert len(calls) == 1 and len(calls[0]) == (8 if activity else 5)
    arguments = calls[0]
    torch.testing.assert_close(arguments[0], torch.tensor([0, 0, 3, 3]))
    assert arguments[1:5] == (0.0, 1, 6, DT)
    if activity:
        assert arguments[5:] == (
            options.get("at_least", 1),
            options.get("inv", False),
            (2, 2),
        )


def test_authored_bind_override_keeps_custom_execution_and_validation():
    functional, _ = _threshold_case()
    calls = []
    replacement = _Replacement()

    class Authored(dn.func.APCount):
        def _bind(self, plan):
            calls.append(plan)
            return replacement

    collection = functional.make_callbacks({"analysis": Authored(node_check=[999])})
    assert calls == [functional]
    assert collection._plans[0].callback is replacement
    assert collection._plans[0].validate_updates is True


@pytest.mark.parametrize("family,native_type,options,canonical,kind", _FAMILIES)
def test_prepared_public_families_keep_legacy_checkpoint_kinds_fields_and_carry(
    family, native_type, options, canonical, kind
):
    functional, tensors = _threshold_case()
    activity = issubclass(family, dn.func.ActiveAL)
    authored = family(
        **_configuration(), **options, **({"partition": [2, 2]} if activity else {})
    )
    callbacks = functional.make_callbacks({"analysis": authored})
    packet = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    _, auxiliary = dn.func.run(functional, packet, steps=1, callbacks=callbacks)
    saved = callbacks.export_state(auxiliary["callbacks"].state)
    expected_fields = {
        "node_indices": {
            "tensor": torch.tensor([0, 0, 3, 3]),
            "dtype": "torch.int64",
            "device": "cpu",
        },
        "threshold": 0.0,
        "start_step": 1,
        "end_step": 6,
        "callback_dt": DT,
    }
    if activity:
        expected_fields.update(
            at_least=options.get("at_least", 1),
            inv=options.get("inv", False),
            partition=[2, 2],
        )
    assert saved["version"] == 1
    _assert_packet_equal(
        saved["configuration"]["callbacks"][0]["callback"],
        {"kind": kind, "fields": expected_fields},
    )
    assert len(saved["carries"]["analysis"]) == (2 if kind == "Raster" else 3)
    receiving = functional.make_callbacks(
        {
            "analysis": family(
                **_configuration(),
                **options,
                **({"partition": [2, 2]} if activity else {}),
            )
        }
    )
    assert type(receiving._plans[0].callback) is canonical
    restored = receiving.restore_state(saved)
    _close(restored.carries, auxiliary["callbacks"].state.carries)
    assert restored.plan_token is receiving._state_token
    assert restored.plan_token is not callbacks._state_token


def _specifications(network):
    def scoped(callback):
        return dn.func.PopulationCallback("left", callback) if network else callback

    options = dict(
        threshold=-59.99, node_check=[0], t_start_check=2 * DT, t_end_check=7 * DT
    )
    return {
        "raster": scoped(dn.func.Raster(**options)),
        "count": scoped(dn.func.APCount(**options)),
        "active": scoped(dn.func.Active(**options)),
        "sites": scoped(dn.func.ActiveAL(**options, at_least=1)),
        "events": scoped(dn.func.ActiveALCount(**options, at_least=2)),
        "trace": dn.func.Recorder(
            ("left.v", "right.v", "t") if network else ("v", "t")
        ),
    }


def _compiled_case(network, execution):
    functional, tensors, parameters, constants, _, _ = _case(network)
    initial = dn.func.Continuation(tensors.state, parameters, constants)
    callbacks = functional.make_callbacks(_specifications(network))
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    checkpoint = {"checkpoint_every": 3} if execution == "scan" else {}
    with torch.no_grad() if execution == "while" else nullcontext():
        expected, whole = dn.func.run(functional, initial, steps=9, callbacks=callbacks)
        _, first = dn.func.run(
            kernel, initial, steps=5, callbacks=callbacks, **checkpoint
        )
        actual, last = dn.func.run(
            kernel, first["continuation"], steps=4, callbacks=callbacks, **checkpoint
        )
    _close(actual, expected)
    _close(_joined(first, last), whole["callbacks"]["trace"])
    for name in ("active", "count", "sites", "events"):
        _close(last["callbacks"][name], whole["callbacks"][name])
    _close(
        torch.cat((first["callbacks"]["raster"], last["callbacks"]["raster"])),
        whole["callbacks"]["raster"],
    )
    if execution == "scan":
        target = _target(parameters, network)
        actual_loss = _voltage(actual, network).square().sum() + sum(
            value.square().mean()
            for name, value in _joined(first, last).items()
            if name != "t"
        )
        expected_loss = _voltage(expected, network).square().sum() + sum(
            value.square().mean()
            for name, value in whole["callbacks"]["trace"].items()
            if name != "t"
        )
        observed = torch.autograd.grad(actual_loss, target)[0]
        reference = torch.autograd.grad(expected_loss, target)[0]
        _close(observed, reference)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    assert kernel.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    return actual


@pytest.mark.parametrize("network,execution", ((False, "scan"), (True, "while")))
def test_public_prepared_threshold_callbacks_preserve_compiled_continuation_and_recording_gradients(
    network, execution
):
    _compiled_case(network, execution)


@pytest.mark.mps
def test_mps_public_prepared_threshold_callbacks_share_while_execution():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        actual = _compiled_case(False, "while")
        assert _voltage(actual, False).device.type == "mps"
