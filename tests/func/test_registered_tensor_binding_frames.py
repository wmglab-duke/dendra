"""Registered diagnostic frames reuse only side-effect-free tensor captures."""

from __future__ import annotations

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra.func import _population

pytestmark = pytest.mark.cpu


def _record_captures(monkeypatch, events):
    original = _population._audit_tensor_binding

    def capture(value):
        events.append(id(value))
        return original(value)

    monkeypatch.setattr(_population, "_audit_tensor_binding", capture)


def test_plain_aliases_share_one_capture_across_registered_slot_categories(monkeypatch):
    value = torch.nn.Parameter(torch.arange(6.0, dtype=torch.float64).reshape(2, 3).t())
    child = torch.nn.Module()
    child.register_buffer("value", value)
    root = torch.nn.Module()
    root.register_parameter("weight", value)
    root.register_parameter("weight_alias", value)
    root.register_buffer("cache", value)
    root.register_buffer("absent", None)
    root.child = child
    root.child_alias = child
    expected = _population._audit_tensor_binding(value)
    captures = []
    _record_captures(monkeypatch, captures)

    result = _population._registered_tensor_bindings(root)

    assert list(result) == [
        ("", "_parameters", "weight"),
        ("", "_parameters", "weight_alias"),
        ("", "_buffers", "cache"),
        ("child", "_buffers", "value"),
        ("child_alias", "_buffers", "value"),
    ]
    assert list(result.values()) == [expected] * 5
    assert captures == [id(value)]


def test_new_frame_recaptures_data_writes_that_do_not_change_tensor_version(
    monkeypatch,
):
    value = torch.tensor([1.0, -0.0], dtype=torch.float64)
    module = torch.nn.Module()
    module.register_buffer("first", value)
    module.register_buffer("second", value)
    captures = []
    _record_captures(monkeypatch, captures)

    first = _population._registered_tensor_bindings(module)
    version = value._version
    value.data.add_(3.0)
    assert value._version == version
    second = _population._registered_tensor_bindings(module)

    assert captures == [id(value), id(value)]
    assert first.keys() == second.keys()
    for key in first:
        assert first[key][:-1] == second[key][:-1]
        assert first[key][-1] != second[key][-1]
    assert second[("", "_buffers", "first")] == second[("", "_buffers", "second")]


def test_distinct_tensor_views_keep_distinct_metadata_and_value_captures(monkeypatch):
    value = torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
    view = value.t()
    assert view is not value and torch._C._is_alias_of(value, view)
    module = torch.nn.Module()
    module.register_buffer("value", value)
    module.register_buffer("view", view)
    expected = tuple(_population._audit_tensor_binding(item) for item in (value, view))
    captures = []
    _record_captures(monkeypatch, captures)

    result = _population._registered_tensor_bindings(module)

    assert tuple(result.values()) == expected
    assert captures == [id(value), id(view)]
    assert result[("", "_buffers", "value")][0] != result[("", "_buffers", "view")][0]


def test_empty_and_none_slots_keep_empty_frames_and_empty_tensor_metadata(monkeypatch):
    module = torch.nn.Module()
    module.register_parameter("absent_parameter", None)
    module.register_buffer("absent_buffer", None)
    captures = []
    _record_captures(monkeypatch, captures)
    first = _population._registered_tensor_bindings(module)
    second = _population._registered_tensor_bindings(module)
    assert first == second == {} and first is not second
    assert captures == []

    value = torch.empty(2, 0, dtype=torch.float64).expand(3, 2, 0)
    module.register_buffer("empty", value)
    module.register_buffer("empty_alias", value)
    result = _population._registered_tensor_bindings(module)
    assert list(result) == [("", "_buffers", "empty"), ("", "_buffers", "empty_alias")]
    assert result[("", "_buffers", "empty")][2:5] == (
        (3, 2, 0),
        value.stride(),
        value.storage_offset(),
    )
    assert captures == [id(value)]


def _observed_result(capture, events, failure):
    events.clear()
    try:
        result = tuple(capture())
    except RuntimeError as error:
        assert error is failure
        return None, tuple(events), error
    return result, tuple(events), None


@pytest.mark.parametrize("fail", (False, True))
def test_tensor_subclasses_keep_original_per_alias_dispatch_and_exception(fail):
    events, failure = [], RuntimeError("second alias detach")

    class ObservedTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            events.append(function.__name__)
            if fail and function is torch.Tensor.detach and events.count("detach") == 3:
                raise failure
            return super().__torch_function__(function, types, args, kwargs or {})

    value = torch.Tensor._make_subclass(
        ObservedTensor, torch.arange(3.0, dtype=torch.float64), False
    )
    module = torch.nn.Module()
    module.register_buffer("first", value)
    module.register_buffer("second", value)
    expected = _observed_result(
        lambda: (_population._audit_tensor_binding(value) for _ in range(2)),
        events,
        failure,
    )
    actual = _observed_result(
        lambda: _population._registered_tensor_bindings(module).values(),
        events,
        failure,
    )
    assert actual == expected
    assert actual[1].count("detach") == (3 if fail else 4)


@pytest.mark.parametrize("kind", ("function", "dispatch"))
@pytest.mark.parametrize("fail", (False, True))
def test_active_modes_keep_original_per_alias_operations_and_exception(kind, fail):
    events, failure = [], RuntimeError("second alias detach")
    value = torch.arange(3.0, dtype=torch.float64)
    module = torch.nn.Module()
    module.register_buffer("first", value)
    module.register_buffer("second", value)

    def invoke(function, args, kwargs):
        name = str(function) if kind == "dispatch" else function.__name__
        events.append(name)
        detaches = sum("detach" in event for event in events)
        if fail and "detach" in name and detaches == (5 if kind == "dispatch" else 3):
            raise failure
        return function(*args, **(kwargs or {}))

    if kind == "function":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                return invoke(function, args, kwargs)
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return invoke(function, args, kwargs)

    with Mode():
        expected = _observed_result(
            lambda: (_population._audit_tensor_binding(value) for _ in range(2)),
            events,
            failure,
        )
    with Mode():
        actual = _observed_result(
            lambda: _population._registered_tensor_bindings(module).values(),
            events,
            failure,
        )
    assert actual == expected
    assert bool(actual[2]) is fail
    assert actual[1]


@pytest.mark.parametrize("authored", ("named_modules", "registry", "getattribute"))
def test_authored_traversal_reads_remain_lazy_and_recapture_between_aliases(
    authored, monkeypatch
):
    events = []
    value = torch.tensor([1.0, 2.0], dtype=torch.float64)
    initial_digest = _population._audit_tensor_digest(value)
    changed_digest = _population._audit_tensor_digest(value + 3.0)
    original = _population._audit_tensor_binding

    def capture(tensor):
        events.append("capture")
        return original(tensor)

    monkeypatch.setattr(_population, "_audit_tensor_binding", capture)

    if authored == "named_modules":
        child = torch.nn.Module()
        child.register_buffer("value", value)

        class Root(torch.nn.Module):
            def named_modules(self, *args, **kwargs):
                events.append("walk")
                yield "first", child
                events.append("mutate")
                value.data.add_(3.0)
                yield "second", child

        root = Root()
        keys = [("first", "_buffers", "value"), ("second", "_buffers", "value")]
        expected_events = ["walk", "capture", "mutate", "capture"]
    elif authored == "registry":

        class Registry(dict):
            def items(self):
                events.append("items")
                yield "first", value
                events.append("mutate")
                value.data.add_(3.0)
                yield "second", value

        root = torch.nn.Module()
        object.__setattr__(root, "_buffers", Registry())
        keys = [("", "_buffers", "first"), ("", "_buffers", "second")]
        expected_events = ["items", "capture", "mutate", "capture"]
    else:

        class Child(torch.nn.Module):
            def __getattribute__(self, name):
                if name == "_buffers" and object.__getattribute__(self, "armed"):
                    events.append("buffers")
                    if events.count("buffers") == 2:
                        value.data.add_(3.0)
                return super().__getattribute__(name)

        child = Child()
        object.__setattr__(child, "armed", False)
        child.register_buffer("value", value)
        root = torch.nn.Module()
        root.first = child
        root.second = child
        object.__setattr__(child, "armed", True)
        keys = [("first", "_buffers", "value"), ("second", "_buffers", "value")]
        expected_events = ["buffers", "capture", "buffers", "capture"]
    version = value._version

    result = _population._registered_tensor_bindings(root)

    assert events == expected_events
    assert list(result) == keys
    assert [result[key][-1] for key in keys] == [initial_digest, changed_digest]
    assert value._version == version


def test_authored_module_hash_cannot_reuse_an_earlier_plain_tensor_capture(monkeypatch):
    value = torch.tensor([1.0, 2.0], dtype=torch.float64)
    initial = _population._audit_tensor_digest(value)
    events = []

    class Child(torch.nn.Module):
        def __hash__(self):
            events.append("hash")
            if events.count("hash") == 1:
                value.data.add_(3.0)
            return object.__hash__(self)

    child = Child()
    child.register_buffer("alias", value)
    root = torch.nn.Module()
    root.register_buffer("first", value)
    root.child = child
    original = _population._audit_tensor_binding

    def capture(tensor):
        events.append("capture")
        return original(tensor)

    monkeypatch.setattr(_population, "_audit_tensor_binding", capture)
    result = _population._registered_tensor_bindings(root)

    assert list(result) == [("", "_buffers", "first"), ("child", "_buffers", "alias")]
    assert events == ["capture", "hash", "capture"]
    assert tuple(entry[-1] for entry in result.values()) == (
        initial,
        _population._audit_tensor_digest(value),
    )
    assert (
        result[("", "_buffers", "first")][-1]
        != result[("child", "_buffers", "alias")][-1]
    )


def test_authored_registry_key_hash_keeps_capture_before_publication_order(monkeypatch):
    value = torch.tensor([1.0, 2.0], dtype=torch.float64)
    initial = _population._audit_tensor_digest(value)
    events, armed = [], False

    class Name(str):
        def __hash__(self):
            if armed:
                events.append("hash")
                value.data.add_(3.0)
            return super().__hash__()

    module = torch.nn.Module()
    object.__setattr__(
        module,
        "_buffers",
        {"first": value, Name("second"): value, "third": value},
    )
    original = _population._audit_tensor_binding

    def capture(tensor):
        events.append("capture")
        return original(tensor)

    monkeypatch.setattr(_population, "_audit_tensor_binding", capture)
    armed = True
    result = _population._registered_tensor_bindings(module)
    armed = False

    assert [str(key[2]) for key in result] == ["first", "second", "third"]
    assert events == ["capture", "capture", "hash", "capture"]
    assert [entry[-1] for entry in result.values()] == [
        initial,
        initial,
        _population._audit_tensor_digest(value),
    ]
    assert next(iter(result.values()))[-1] != tuple(result.values())[-1][-1]


def test_tensor_subclass_side_effects_force_plain_alias_recapture_for_whole_frame(
    monkeypatch,
):
    plain = torch.tensor([1.0, 2.0], dtype=torch.float64)
    initial = _population._audit_tensor_digest(plain)
    armed, changed = False, False

    class AuthoredTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            nonlocal changed
            if armed and function is torch.Tensor.detach and not changed:
                plain.data.add_(3.0)
                changed = True
            return super().__torch_function__(function, types, args, kwargs or {})

    authored = torch.Tensor._make_subclass(
        AuthoredTensor, torch.ones(2, dtype=torch.float64), False
    )
    module = torch.nn.Module()
    module.register_buffer("first", plain)
    module.register_buffer("authored", authored)
    module.register_buffer("last", plain)
    captures = []
    _record_captures(monkeypatch, captures)
    armed = True

    result = _population._registered_tensor_bindings(module)

    assert captures == [id(plain), id(authored), id(plain)]
    assert changed
    assert result[("", "_buffers", "first")][-1] == initial
    assert result[("", "_buffers", "last")][-1] == _population._audit_tensor_digest(
        plain
    )
    assert result[("", "_buffers", "first")][-1] != result[("", "_buffers", "last")][-1]


def test_plain_tensor_authored_method_shadow_keeps_original_per_alias_calls():
    value = torch.tensor([1.0, 2.0], dtype=torch.float64)
    initial = _population._audit_tensor_digest(value)
    detach, events = value.detach, []

    def authored_detach():
        events.append("detach")
        if len(events) == 2:
            value.data.add_(3.0)
        return detach()

    value.detach = authored_detach
    assert type(value) is torch.Tensor
    module = torch.nn.Module()
    module.register_buffer("first", value)
    module.register_buffer("second", value)
    version = value._version

    result = _population._registered_tensor_bindings(module)

    assert events == ["detach", "detach"]
    assert value._version == version
    assert result[("", "_buffers", "first")][-1] == initial
    assert result[("", "_buffers", "second")][-1] != initial


def test_canonical_dendra_device_context_reuses_plain_tensor_aliases(monkeypatch):
    with dn.ctx(DEVICE="cpu"):
        value = torch.arange(3.0, dtype=torch.float64)
        module = torch.nn.Module()
        module.register_buffer("first", value)
        module.register_buffer("second", value)
        expected = _population._audit_tensor_binding(value)
        captures = []
        _record_captures(monkeypatch, captures)

        result = _population._registered_tensor_bindings(module)

    assert list(result) == [("", "_buffers", "first"), ("", "_buffers", "second")]
    assert tuple(result.values()) == (expected, expected)
    assert captures == [id(value)]


def test_pre_dispatch_mode_keeps_original_alias_error_and_capture_trace(monkeypatch):
    from torch._subclasses.functional_tensor import FunctionalTensorMode

    value = torch.arange(3.0, dtype=torch.float64)
    module = torch.nn.Module()
    module.register_buffer("first", value)
    module.register_buffer("second", value)
    events, failure = [], RuntimeError("second alias capture")
    original = _population._audit_tensor_binding

    def capture(tensor):
        events.append("capture")
        if events.count("capture") == 2:
            raise failure
        return original(tensor)

    class Mode(FunctionalTensorMode):
        def __init__(self):
            super().__init__(pre_dispatch=True)

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            events.append(str(function))
            return function(*args, **(kwargs or {}))

    monkeypatch.setattr(_population, "_audit_tensor_binding", capture)
    with Mode() as mode:
        assert (
            torch._ops._get_dispatch_mode_pre_dispatch(
                torch._C._TorchDispatchModeKey.FUNCTIONAL
            )
            is mode
        )
        expected = _observed_result(
            lambda: (_population._audit_tensor_binding(value) for _ in range(2)),
            events,
            failure,
        )
    with Mode() as mode:
        assert (
            torch._ops._get_dispatch_mode_pre_dispatch(
                torch._C._TorchDispatchModeKey.FUNCTIONAL
            )
            is mode
        )
        actual = _observed_result(
            lambda: _population._registered_tensor_bindings(module).values(),
            events,
            failure,
        )

    assert actual == expected
    assert actual[2] is failure
    assert actual[1].count("capture") == 2
