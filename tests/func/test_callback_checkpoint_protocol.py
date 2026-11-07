"""Authored pure callbacks declare portable configuration without reconstruction."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import fields
from numbers import Integral

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import (
    _assert_packet_equal,
)
from tests.func.test_callback_checkpoints import _model as _builtin_model
from tests.func.test_callback_checkpoints import (
    _plain,
)
from tests.func.test_callback_checkpoints import (
    _specifications as _builtin_specifications,
)
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_generic_callbacks import _callback_loss, _Statistics, _VoltageSum
from tests.func.test_lowering_api import DT, _network, _population, _target, _voltage

pytestmark = pytest.mark.cpu


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


class _PortableStatistics(_Statistics):
    def checkpoint_configuration(self):
        return {
            "kind": "tests.Statistics",
            "version": 1,
            "fields": {"scale": self.scale},
        }


class _InitialStatistics(_PortableStatistics):
    def initialize(self, state, auxiliary):
        carry, _ = super().initialize(state, auxiliary)
        value = state["integrator"]["v"].mean() * carry["scale"]
        return carry, {"value": value, "alias": value}

    def checkpoint_configuration(self):
        return {**super().checkpoint_configuration(), "kind": "tests.InitialStatistics"}


class _PortableSum(_VoltageSum):
    def checkpoint_configuration(self):
        return {"kind": "tests.Sum", "version": 1, "fields": {}}


class _DeclaredSum(_PortableSum):
    def __init__(self, declaration):
        self.declaration = declaration

    def checkpoint_configuration(self):
        return self.declaration


class _PortableAnomaly(dn.func.AnomalyDetector):
    def checkpoint_configuration(self):
        # A custom declaration cannot impersonate the legacy built-in payload.
        return {"kind": "AnomalyDetector", "version": 1, "fields": {}}


def _case(network=False):
    functional, tensors = dn.func.make_functional(
        _network() if network else _population(), dt=DT
    )
    return functional, tensors._replace(
        parameters=tensors.independent_parameters(trainable="*g_param")
    )


def _specifications(network, scale, *, initial=False):
    def scoped(callback):
        return dn.func.PopulationCallback("left", callback) if network else callback

    return {
        "statistics": scoped(
            (_InitialStatistics if initial else _PortableStatistics)(scale)
        ),
        "sum": scoped(_PortableSum()),
    }


def _initial(tensors):
    return dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)


def _joined_statistics(first, last):
    return {
        name: torch.cat((first["statistics"]["samples"][name], value))
        for name, value in last["statistics"]["samples"].items()
    }


def _declaration(**changes):
    return {
        "kind": "uninstalled.application.Callback",
        "version": 1,
        "fields": {},
        **changes,
    }


def _declared_case(declaration):
    functional, tensors = _case()
    callbacks = functional.make_callbacks({"custom": _DeclaredSum(declaration)})
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=0, callbacks=callbacks
    )
    return functional, tensors, callbacks, auxiliary["callbacks"].state


@pytest.mark.parametrize("network", (False, True), ids=("population", "scoped_network"))
@pytest.mark.parametrize(
    "initial", (False, True), ids=("post_step_emitter", "initial_emitter")
)
def test_custom_reducer_and_emitter_callback_state_roundtrips_without_restore_probes(
    network, initial, tmp_path, monkeypatch
):
    source, tensors = _case(network)
    receiver, received = _case(network)
    scale = torch.tensor(0.25)
    callbacks = source.make_callbacks(_specifications(network, scale, initial=initial))
    receiving_callbacks = receiver.make_callbacks(
        _specifications(network, scale.clone(), initial=initial)
    )
    native = _snapshot(receiver.extract())
    with torch.no_grad():
        expected, whole = dn.func.run(
            source, _initial(tensors), steps=6, callbacks=callbacks
        )
        prefix, first = dn.func.run(
            source, _initial(tensors), steps=2, callbacks=callbacks
        )
    saved = _disk(
        callbacks.export_state(first["callbacks"].state), tmp_path / "callbacks.pt"
    )
    assert saved["format"] == "dendra.func.callback_state" and saved["version"] == 1
    _plain(saved)

    def reject(*args, **kwargs):
        raise AssertionError("restoration must not probe or execute authored callbacks")

    with monkeypatch.context() as guarded:
        for implementation in (_PortableStatistics, _PortableSum):
            for method in ("initialize", "update", "finalize"):
                guarded.setattr(implementation, method, reject)
        guarded.setattr(receiving_callbacks, "_probe", reject)
        restored = receiving_callbacks.restore_state(saved)
    assert restored.plan_token is receiving_callbacks._state_token
    assert restored.plan_token is not first["callbacks"].state.plan_token
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        dn.func.run(
            receiver.bind(received),
            _snapshot(prefix),
            steps=1,
            callbacks=receiving_callbacks,
            callback_state=first["callbacks"].state,
        )
    with torch.no_grad():
        actual, last = dn.func.run(
            receiver.bind(received),
            _snapshot(prefix),
            steps=4,
            callbacks=receiving_callbacks,
            callback_state=restored,
        )
    _close(actual, expected)
    _close(
        _joined_statistics(first["callbacks"], last["callbacks"]),
        whole["callbacks"]["statistics"]["samples"],
    )
    _close(
        last["callbacks"]["statistics"]["carry"],
        whole["callbacks"]["statistics"]["carry"],
    )
    _close(last["callbacks"]["sum"], whole["callbacks"]["sum"])
    assert last["callbacks"]["statistics"]["samples"]["value"].shape[0] == 4
    _close(receiver.extract(), native)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("runner", (False, True), ids=("explicit", "runner"))
def test_custom_callback_continuation_checkpoints_preserve_suffix_gradients_and_graph_cut(
    network, runner, tmp_path
):
    source_scale = torch.tensor(0.25, requires_grad=True)
    receiving_scale = torch.tensor(0.25, requires_grad=True)
    if runner:

        def program(scale):
            return dn.compile(
                _network() if network else _population(),
                dt=DT,
                steps=2,
                inference=False,
                trainable="*g_param",
                callbacks=_specifications(network, scale),
                backend="eager",
                disable=True,
            )

        source, receiver = program(source_scale), program(receiving_scale)
        source_callbacks, receiving_callbacks = source.callbacks, receiver.callbacks
        functional, receiving = source.functional, receiver.functional
        source_parameter = _target(source.parameters, network)
        _, first = dn.func.run(source)
        saved = dn.func.export_continuation(source, first["continuation"])
        restored = dn.func.restore_continuation(
            receiver, _disk(saved, tmp_path / "runner.pt")
        )
        actual = dn.func.run(receiver, restored, steps=3)
    else:
        functional, tensors = _case(network)
        receiving, _ = _case(network)
        source_callbacks = functional.make_callbacks(
            _specifications(network, source_scale)
        )
        receiving_callbacks = receiving.make_callbacks(
            _specifications(network, receiving_scale)
        )
        source_parameter = _target(tensors.parameters, network)
        _, first = dn.func.run(
            functional, _initial(tensors), steps=2, callbacks=source_callbacks
        )
        saved = dn.func.export_continuation(
            functional, first["continuation"], callbacks=source_callbacks
        )
        restored = dn.func.restore_continuation(
            receiving,
            _disk(saved, tmp_path / "explicit.pt"),
            callbacks=receiving_callbacks,
        )
        actual = dn.func.run(
            receiving, restored, steps=3, callbacks=receiving_callbacks
        )
    assert restored.callback_state.plan_token is receiving_callbacks._state_token
    assert restored.callback_state.plan_token is not source_callbacks._state_token
    target = _target(restored.parameters, network)
    assert target.is_leaf and target.requires_grad and target is not source_parameter
    carry = restored.callback_state["statistics"]
    assert not carry["scale"].requires_grad and carry["scale"].grad_fn is None
    expected = dn.func.run(
        receiving.bind(restored), restored, steps=3, callbacks=receiving_callbacks
    )
    _close(actual[0], expected[0])
    _close(actual[1]["callbacks"], expected[1]["callbacks"])
    observed = torch.autograd.grad(
        _callback_loss(actual[1]),
        (target, source_parameter, source_scale, receiving_scale),
        allow_unused=True,
    )
    reference = torch.autograd.grad(_callback_loss(expected[1]), target)[0]
    _close(observed[0], reference)
    assert torch.isfinite(observed[0]).all() and observed[0].abs().sum() > 0
    assert observed[1:] == (None, None, None)
    assert actual[1]["callbacks"]["statistics"]["samples"]["value"].shape[0] == 3


@pytest.mark.parametrize("network,execution", ((False, "scan"), (True, "while")))
def test_custom_callback_checkpoint_protocol_reuses_compiled_scan_and_while(
    network, execution, tmp_path
):
    source, tensors = _case(network)
    receiving, _ = _case(network)
    callbacks = source.make_callbacks(_specifications(network, torch.tensor(0.25)))
    receiving_callbacks = receiving.make_callbacks(
        _specifications(network, torch.tensor(0.25))
    )
    kernel = source.compile_rollout_chunk(4, execution=execution, backend="aot_eager")
    receiver_kernel = receiving.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    with torch.no_grad() if execution == "while" else nullcontext():
        expected, whole = dn.func.run(
            source, _initial(tensors), steps=8, callbacks=callbacks
        )
        _, first = dn.func.run(kernel, _initial(tensors), steps=4, callbacks=callbacks)
        restored = dn.func.restore_continuation(
            receiving,
            _disk(
                dn.func.export_continuation(
                    source, first["continuation"], callbacks=callbacks
                ),
                tmp_path / "compiled.pt",
            ),
            callbacks=receiving_callbacks,
        )
        actual, last = dn.func.run(
            receiver_kernel, restored, steps=4, callbacks=receiving_callbacks
        )
    _close(actual, expected)
    _close(
        _joined_statistics(first["callbacks"], last["callbacks"]),
        whole["callbacks"]["statistics"]["samples"],
    )
    _close(last["callbacks"]["sum"], whole["callbacks"]["sum"])
    report = receiver_kernel.execution_report()
    assert report.requested_execution == execution and report.steps == 4
    assert report.strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


def test_metadata_hook_is_cold_and_called_once_per_configuration_traversal(monkeypatch):
    functional, tensors = _case()
    calls = []
    original = _PortableSum.checkpoint_configuration

    def counted(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(_PortableSum, "checkpoint_configuration", counted)
    source = _PortableSum()
    receiving = _PortableSum()
    callbacks = functional.make_callbacks({"custom": source})
    receiver_callbacks = functional.make_callbacks({"custom": receiving})
    _, first = dn.func.run(functional, _initial(tensors), steps=2, callbacks=callbacks)
    assert calls == []
    saved = callbacks.export_state(first["callbacks"].state)
    assert calls == [source]
    restored = receiver_callbacks.restore_state(saved)
    assert calls == [source, receiving]
    dn.func.run(
        functional.bind(tensors),
        first["continuation"].state,
        steps=1,
        callbacks=receiver_callbacks,
        callback_state=restored,
    )
    assert calls == [source, receiving]


@pytest.mark.parametrize(
    "mismatch",
    ("kind", "version", "field", "tensor_dtype", "tensor_shape", "tuple_list"),
)
def test_receiving_custom_configuration_must_match_before_carry_is_read(mismatch):
    source = _declaration(
        fields={"gain": torch.tensor(0.5, dtype=torch.float64), "name": "voltage"}
    )
    receiving = _declaration(
        fields={"gain": torch.tensor(0.5, dtype=torch.float64), "name": "voltage"}
    )
    if mismatch == "kind":
        receiving["kind"] = "another.Callback"
    elif mismatch == "version":
        receiving["version"] = 2
    elif mismatch == "field":
        receiving["fields"]["name"] = "current"
    elif mismatch == "tensor_dtype":
        receiving["fields"]["gain"] = torch.tensor(0.5, dtype=torch.float32)
    elif mismatch == "tensor_shape":
        receiving["fields"]["gain"] = torch.tensor([0.5])
    else:
        source["fields"]["sequence"] = (1, 2)
        receiving["fields"]["sequence"] = [1, 2]
    functional, _, callbacks, carry = _declared_case(source)
    saved = callbacks.export_state(carry)
    receiver = functional.make_callbacks({"custom": _DeclaredSum(receiving)})
    saved["carries"] = object()
    with pytest.raises(dn.func.FunctionalizationError, match="configuration"):
        receiver.restore_state(saved)


def _invalid_declaration(case):
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    class TupleSubclass(tuple):
        pass

    class TensorSubclass(torch.Tensor):
        pass

    declaration = _declaration()
    if case == "outer_type":
        return []
    if case == "outer_subclass":
        return DictSubclass(declaration)
    if case == "missing_key":
        declaration.pop("kind")
    elif case == "unknown_key":
        declaration["extra"] = None
    elif case == "empty_kind":
        declaration["kind"] = ""
    elif case == "nonstring_kind":
        declaration["kind"] = 3
    elif case in ("zero_version", "negative_version", "bool_version", "float_version"):
        declaration["version"] = {
            "zero_version": 0,
            "negative_version": -1,
            "bool_version": True,
            "float_version": 1.0,
        }[case]
    elif case == "fields_type":
        declaration["fields"] = []
    elif case == "fields_subclass":
        declaration["fields"] = DictSubclass()
    elif case == "nonstring_key":
        declaration["fields"] = {1: "value"}
    elif case == "dict_cycle":
        declaration["fields"]["cycle"] = declaration["fields"]
    elif case == "list_cycle":
        cycle = []
        cycle.append(cycle)
        declaration["fields"]["cycle"] = cycle
    else:
        declaration["fields"]["value"] = {
            "unknown_object": object(),
            "nested_dict_subclass": DictSubclass(),
            "list_subclass": ListSubclass(),
            "tuple_subclass": TupleSubclass(),
            "tensor_subclass": torch.tensor(0.5).as_subclass(TensorSubclass),
            "dtype_object": torch.float64,
            "device_object": torch.device("cpu"),
            "slice_object": slice(0, 1),
        }[case]
    return declaration


@pytest.mark.parametrize(
    "case",
    (
        "outer_type",
        "outer_subclass",
        "missing_key",
        "unknown_key",
        "empty_kind",
        "nonstring_kind",
        "zero_version",
        "negative_version",
        "bool_version",
        "float_version",
        "fields_type",
        "fields_subclass",
        "nonstring_key",
        "dict_cycle",
        "list_cycle",
        "unknown_object",
        "nested_dict_subclass",
        "list_subclass",
        "tuple_subclass",
        "tensor_subclass",
        "dtype_object",
        "device_object",
        "slice_object",
    ),
)
def test_custom_checkpoint_metadata_rejects_nonplain_or_malformed_declarations(case):
    _, _, callbacks, carry = _declared_case(_invalid_declaration(case))
    with pytest.raises((TypeError, ValueError, dn.func.FunctionalizationError)):
        callbacks.export_state(carry)


def test_custom_metadata_owns_detached_tensor_snapshots_and_accepts_plain_nested_data(
    tmp_path,
):
    upstream = torch.tensor(0.25, requires_grad=True)
    parameter = torch.nn.Parameter(torch.tensor([0.5, 1.0]))
    fields = {
        "nested": [None, True, 3, 0.5, "label", {"values": (upstream * 2, parameter)}],
    }
    declaration = _declaration(fields=fields)
    functional, _, callbacks, carry = _declared_case(declaration)
    receiver = functional.make_callbacks(
        {"custom": _DeclaredSum(_snapshot(declaration))}
    )
    saved = callbacks.export_state(carry)
    frozen = _snapshot(saved)
    _plain(saved)
    originals = tuple(
        value for value in pytree.tree_leaves(fields) if isinstance(value, torch.Tensor)
    )
    copies = tuple(
        value for value in pytree.tree_leaves(saved) if isinstance(value, torch.Tensor)
    )
    assert copies and all(
        all(value is not original for original in originals) for value in copies
    )
    with torch.no_grad():
        fields["nested"][-1]["values"][0].add_(1)
        parameter.add_(1)
    _assert_packet_equal(saved, frozen)
    restored = receiver.restore_state(_disk(saved, tmp_path / "metadata.pt"))
    assert restored.plan_token is receiver._state_token


def test_custom_carry_and_known_emission_schemas_remain_validated():
    functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {"statistics": _InitialStatistics(torch.tensor(0.25))}
    )
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=1, callbacks=callbacks
    )
    saved = callbacks.export_state(auxiliary["callbacks"].state)
    malformed = _snapshot(saved)
    malformed["carries"]["statistics"]["energy"] = torch.zeros(2)
    with pytest.raises(dn.func.FunctionalizationError, match="carry|shape|structure"):
        callbacks.restore_state(malformed)
    malformed = _snapshot(saved)
    malformed["emission_schemas"][0]["children"][0]["shape"] = [2]
    with pytest.raises(dn.func.FunctionalizationError, match="emission|schema"):
        callbacks.restore_state(malformed)


@pytest.mark.parametrize("kind", ("NoneType", "dict", "list", "tuple"))
def test_deferred_emission_schema_rejects_empty_trees_without_probing(kind):
    functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {"statistics": _PortableStatistics(torch.tensor(0.25))}
    )
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=1, callbacks=callbacks
    )
    saved = callbacks.export_state(auxiliary["callbacks"].state)
    saved["emission_schemas"][0] = {"kind": kind, "children": []}
    if kind == "dict":
        saved["emission_schemas"][0]["keys"] = []
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="emission|schema|Tensor"
    ):
        callbacks.restore_state(saved)


def test_restored_deferred_emission_schema_is_enforced_on_the_next_update():
    functional, tensors = _case()
    callbacks = functional.make_callbacks(
        {"statistics": _PortableStatistics(torch.tensor(0.25))}
    )
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=1, callbacks=callbacks
    )
    saved = callbacks.export_state(auxiliary["callbacks"].state)
    saved["emission_schemas"][0]["children"][0]["shape"] = [2]
    restored = callbacks.restore_state(saved)
    with pytest.raises(dn.func.FunctionalizationError, match="emission|shape|rank"):
        dn.func.run(
            functional.bind(tensors),
            auxiliary["continuation"].state,
            steps=1,
            callbacks=callbacks,
            callback_state=restored,
        )


def test_unknown_metadata_does_not_invoke_authored_class_getters_or_metaclass_equality():
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append("equality")
            raise AssertionError(
                "metadata admission must compare actual classes by identity"
            )

    class Unknown(metaclass=Meta):
        @property
        def __class__(self):
            calls.append("class getter")
            raise AssertionError(
                "metadata admission must not consult authored __class__"
            )

    _, _, callbacks, carry = _declared_case(_declaration(fields={"object": Unknown()}))
    with pytest.raises(TypeError, match="plain|Tensor|configuration"):
        callbacks.export_state(carry)
    assert calls == []


def test_custom_callbacks_without_opt_in_keep_legacy_rejection_and_explicit_builtin_override_is_distinct():
    class AuthoredAnomaly(dn.func.AnomalyDetector):
        pass

    functional, tensors = _case()
    for callback in (_VoltageSum(), AuthoredAnomaly()):
        callbacks = functional.make_callbacks({"custom": callback})
        _, auxiliary = dn.func.run(
            functional, _initial(tensors), steps=0, callbacks=callbacks
        )
        with pytest.raises(
            dn.func.FunctionalizationError, match="custom callbacks|unsupported"
        ):
            callbacks.export_state(auxiliary["callbacks"].state)
    custom = functional.make_callbacks({"custom": _PortableAnomaly()})
    builtin = functional.make_callbacks({"custom": dn.func.AnomalyDetector()})
    _, custom_aux = dn.func.run(
        functional, _initial(tensors), steps=0, callbacks=custom
    )
    _, builtin_aux = dn.func.run(
        functional, _initial(tensors), steps=0, callbacks=builtin
    )
    custom_saved = custom.export_state(custom_aux["callbacks"].state)
    builtin_saved = builtin.export_state(builtin_aux["callbacks"].state)
    custom.restore_state(custom_saved)
    for receiver, saved in ((custom, builtin_saved), (builtin, custom_saved)):
        with pytest.raises(dn.func.FunctionalizationError, match="configuration"):
            receiver.restore_state(saved)


def test_legacy_recorder_checkpoint_payload_is_unchanged():
    functional, tensors = _case()
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v",))})
    _, auxiliary = dn.func.run(
        functional, _initial(tensors), steps=0, callbacks=callbacks
    )
    expected = {
        "format": "dendra.func.callback_state",
        "version": 1,
        "configuration": {
            "owner": {
                "kind": "Population",
                "populations": [
                    {
                        "name": None,
                        "shape": list(functional.shape),
                        "dtype": str(functional.dtype),
                        "device": str(functional.device),
                    }
                ],
                "dt": DT,
            },
            "callbacks": [
                {
                    "name": "trace",
                    "callback": {
                        "kind": "Recorder",
                        "fields": {
                            "states": ["v"],
                            "paths": [["integrator", "v"]],
                            "max_only": False,
                            "node_indices": None,
                            "partition": None,
                            "region": None,
                        },
                    },
                }
            ],
        },
        "carries": {"trace": ()},
        "emission_schemas": [
            {
                "kind": "dict",
                "keys": ["v"],
                "children": [
                    {
                        "kind": "tensor",
                        "shape": list(tensors.state["integrator"]["v"].shape),
                        "dtype": str(functional.dtype),
                        "device": str(functional.device),
                    }
                ],
            }
        ],
    }
    _assert_packet_equal(callbacks.export_state(auxiliary["callbacks"].state), expected)
    restored = callbacks.restore_state(expected)
    assert restored.plan_token is callbacks._state_token


def _legacy_configuration_value(value):
    """Independent description of the established built-in version-1 encoding."""
    if isinstance(value, str):
        return str.__str__(value)
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, torch.Tensor):
        return {
            "tensor": value.detach().to("cpu").clone(),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if type(value) in (tuple, list):
        return [_legacy_configuration_value(child) for child in value]
    if type(value) is slice:
        components = (value.start, value.stop, value.step)
        assert all(part is None or isinstance(part, Integral) for part in components)
        return {"slice": [None if part is None else int(part) for part in components]}
    if type(value) is dn.func.AnomalyDetector:
        return {"kind": "AnomalyDetector"}
    kinds = {
        "Recorder": "Recorder",
        "_BoundCadencedRecorder": "CadencedRecorder",
        "LFP": "LFP",
        "Raster": "Raster",
        "APCount": "APCount",
        "ActiveAL": "ActiveAL",
        "ActiveALCount": "ActiveALCount",
        "_BoundPopulationCallback": "PopulationCallback",
        "_RecorderRegion": "RecorderRegion",
    }
    kind = kinds[type(value).__name__]
    if kind in ("Raster", "APCount", "ActiveAL", "ActiveALCount"):
        names = ("node_indices", "threshold", "start_step", "end_step", "callback_dt")
        if kind in ("ActiveAL", "ActiveALCount"):
            names += ("at_least", "inv", "partition")
    elif kind == "Recorder":
        names = ("states", "paths", "max_only", "node_indices", "partition", "region")
    elif kind == "LFP":
        names = ("fields", "rule", "padding", "device", "dtype")
    else:
        names = tuple(field.name for field in fields(value))
    return {
        "kind": kind,
        "fields": {
            name: _legacy_configuration_value(getattr(value, name)) for name in names
        },
    }


@pytest.mark.parametrize("network", (False, True))
def test_all_builtin_configuration_descriptions_keep_the_legacy_wire_encoding(network):
    with dn.ctx(IMEM=True):
        if network:
            model = dn.Network({"left": _builtin_model(), "right": _builtin_model()})
            model.train()
            model.initialize(DT)
            population = model.left
        else:
            model = population = _builtin_model()
        specifications = _builtin_specifications(population)
        specifications["cadence"] = dn.func.Recorder(("v", "t"), dt=2 * DT)
        specifications["region"] = dn.func.Recorder(
            ("v",), region=population[:, [0, 2]]
        )
        if network:
            specifications = {
                name: dn.func.PopulationCallback("left", callback)
                for name, callback in specifications.items()
            }
        functional, tensors = dn.func.make_functional(model, dt=DT)
        callbacks = functional.make_callbacks(specifications)
        _, auxiliary = dn.func.run(
            functional, _initial(tensors), steps=0, callbacks=callbacks
        )
        saved = callbacks.export_state(auxiliary["callbacks"].state)
        populations = (
            functional.populations.items() if network else ((None, functional),)
        )
        expected = {
            "owner": {
                "kind": "Network" if network else "Population",
                "populations": [
                    {
                        "name": name,
                        "shape": list(plan.shape),
                        "dtype": str(plan.dtype),
                        "device": str(plan.device),
                    }
                    for name, plan in populations
                ],
                "dt": functional.dt,
            },
            "callbacks": [
                {
                    "name": plan.name,
                    "callback": _legacy_configuration_value(plan.callback),
                }
                for plan in callbacks._plans
            ],
        }
        assert saved["version"] == 1
        _assert_packet_equal(saved["configuration"], expected)


@pytest.mark.parametrize("operation", ("export", "restore"))
@pytest.mark.parametrize("context", ("compile", "grad", "forward_ad"))
def test_callback_checkpoint_host_guard_precedes_authored_metadata(
    operation, context, monkeypatch
):
    _, _, callbacks, carry = _declared_case(_declaration())
    saved = callbacks.export_state(carry)

    def reject(self):
        raise AssertionError("cold boundary must run before authored metadata")

    monkeypatch.setattr(_DeclaredSum, "checkpoint_configuration", reject)

    def execute(value):
        callbacks.export_state(
            carry
        ) if operation == "export" else callbacks.restore_state(saved)
        return value.square()

    with pytest.raises(dn.func.FunctionalizationError, match="outside torch.compile"):
        if context == "compile":
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            execute(torch.tensor(0.5))
        elif context == "grad":
            torch.func.grad(execute)(torch.tensor(0.5))
        else:
            with torch.autograd.forward_ad.dual_level():
                execute(
                    torch.autograd.forward_ad.make_dual(
                        torch.tensor(0.5), torch.tensor(1.0)
                    )
                )


def test_callback_checkpoint_operand_validation_precedes_authored_metadata(monkeypatch):
    _, _, callbacks, carry = _declared_case(_declaration())
    saved = callbacks.export_state(carry)

    def reject(self):
        raise AssertionError("invalid operands must fail before authored metadata")

    monkeypatch.setattr(_DeclaredSum, "checkpoint_configuration", reject)
    with pytest.raises(TypeError, match="callback_state"):
        callbacks.export_state(None)
    for malformed in ({}, {**saved, "format": "other"}, {**saved, "version": 2}):
        with pytest.raises((TypeError, ValueError)):
            callbacks.restore_state(malformed)


@pytest.mark.mps
def test_mps_custom_callback_checkpoint_metadata_preserves_device_and_continuation(
    tmp_path,
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        functional, tensors = _case()
        receiving, received = _case()
        scale = torch.tensor(0.25, dtype=torch.float32, device="mps")
        callbacks = functional.make_callbacks(_specifications(False, scale))
        receiving_callbacks = receiving.make_callbacks(
            _specifications(False, scale.clone())
        )
        with torch.no_grad():
            expected, whole = dn.func.run(
                functional, _initial(tensors), steps=4, callbacks=callbacks
            )
            prefix, first = dn.func.run(
                functional, _initial(tensors), steps=2, callbacks=callbacks
            )
        saved = _disk(
            callbacks.export_state(first["callbacks"].state), tmp_path / "mps.pt"
        )
        restored = receiving_callbacks.restore_state(saved)
        assert restored["statistics"]["scale"].device.type == "mps"
        wrong_device = receiving.make_callbacks(_specifications(False, scale.cpu()))
        with pytest.raises(dn.func.FunctionalizationError, match="configuration"):
            wrong_device.restore_state(saved)
        with torch.no_grad():
            actual, last = dn.func.run(
                receiving.bind(received),
                _snapshot(prefix),
                steps=2,
                callbacks=receiving_callbacks,
                callback_state=restored,
            )
        _close(actual, expected)
        _close(
            _joined_statistics(first["callbacks"], last["callbacks"]),
            whole["callbacks"]["statistics"]["samples"],
        )
        assert _voltage(actual, False).device.type == "mps"
