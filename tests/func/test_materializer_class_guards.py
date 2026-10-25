"""Materializer constructors retain raw authored class guards and boundaries."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.func import _connection_parameters, _netstim_parameters, _population
from tests.func.test_connection_parameter_sources import _connection, _Source
from tests.func.test_netstim_parameter_sources import _source

pytestmark = pytest.mark.cpu

_KINDS = ("connection", "netstim")


@pytest.fixture(autouse=True)
def _context():
    with dn.ctx(
        JIT=0, REQUIRE_GRAD=True, DEVICE="cpu", DTYPE=torch.float64, CHECK_PURITY=False
    ):
        yield


def _case(kind):
    descriptor_calls = []

    class RawDescriptor:
        def __get__(self, owner, cls=None):
            descriptor_calls.append((owner, cls))
            raise AssertionError("Raw class descriptor was invoked")

    class Excluded:
        def __getattribute__(self, name):
            if name == "__code__":
                raise AssertionError("Framework class value was inspected")
            return super().__getattribute__(name)

    def method(*args):
        return None

    common_values = {
        "base_marker": object(),
        "raw_descriptor": RawDescriptor(),
        "function": method,
        "static_method": staticmethod(method),
        "class_method": classmethod(method),
        "stop_marker": object(),
    }
    common = type(
        "Common",
        (torch.nn.Module,),
        {
            "__module__": __name__,
            "__doc__": "Authored common metadata.",
            "__annotations__": {"base_marker": object},
            **common_values,
        },
    )
    left = type("Left", (common,), {"__module__": __name__, "left_marker": object()})
    right = type("Right", (common,), {"__module__": __name__, "right_marker": object()})
    framework = type(
        "Framework",
        (torch.nn.Module,),
        {
            "__module__": "torch.synthetic_materializer_guards",
            "excluded_marker": Excluded(),
        },
    )
    leaf = type(
        "Leaf",
        (left, right, framework),
        {
            "__module__": __name__,
            "leaf_marker": object(),
            "none_marker": None,
        },
    )
    sibling = type(
        "Sibling", (common,), {"__module__": __name__, "sibling_marker": object()}
    )
    shared = leaf()
    if kind == "connection":
        source = _connection(_Source())
        first, second = source.weight.weights[0].param, source.delay_ms.weights[0].param
        plan_type = _connection_parameters.ConnectionParameterPlan
    else:
        source = _source()
        first = second = source.interval
        plan_type = _netstim_parameters.NetStimParameterPlan
    first.add_module("guard_leaf", shared)
    first.add_module("guard_alias", shared)
    second.add_module("guard_cross_root", shared)
    second.add_module("guard_sibling", sibling())
    plan = object.__new__(plan_type)
    names = (
        (leaf, ("leaf_marker", "none_marker")),
        (left, ("left_marker",)),
        (right, ("right_marker",)),
        (common, tuple(common_values)),
        (sibling, ("sibling_marker",)),
    )
    expected = [
        (cls, name, vars(cls)[name]) for cls, fields in names for name in fields
    ]

    def construct():
        plan_type.__init__(plan, source)
        return plan

    return SimpleNamespace(
        construct=construct,
        plan=plan,
        expected=expected,
        classes={cls for cls, _names in names},
        descriptor_calls=descriptor_calls,
        framework=framework,
        stop=common_values["stop_marker"],
    )


def _authored(case):
    return [record for record in case.plan._class_guards if record[0] in case.classes]


def _assert_prefix(case, expected):
    actual = _authored(case)
    assert [(cls, name) for cls, name, _value, _signature in actual] == [
        (cls, name) for cls, name, _value in expected
    ]
    assert all(
        record[2] is wanted[2] for record, wanted in zip(actual, expected, strict=True)
    )


def _observe_bindings(case, monkeypatch, *, fail_binding=False, fail_guard=False):
    events = []
    binding_error, guard_error = (
        RuntimeError("source binding"),
        RuntimeError("class guard"),
    )
    binding = _connection_parameters._binding
    callable_binding = _connection_parameters._callable_binding

    def bind(*args, **kwargs):
        events.append("source-binding")
        if fail_binding:
            raise binding_error
        return binding(*args, **kwargs)

    def capture(value):
        if value is case.stop:
            events.append("class-guard")
            if fail_guard:
                raise guard_error
        return callable_binding(value)

    for module in (_connection_parameters, _netstim_parameters):
        monkeypatch.setattr(module, "_binding", bind)
        monkeypatch.setattr(module, "_callable_binding", capture)
    return events, binding_error, guard_error


@pytest.mark.parametrize("kind", _KINDS)
def test_real_constructor_captures_diamond_mro_once_without_invoking_descriptors(kind):
    case = _case(kind)
    plan = case.construct()
    assert plan._check_purity is False
    _assert_prefix(case, case.expected)
    assert case.descriptor_calls == []
    assert all(
        cls not in (torch.nn.Module, object)
        and not cls.__module__.startswith("torch.")
        and name not in _population._CLASS_STATE_METADATA_NAMES
        for cls, name, _value, _signature in plan._class_guards
    )
    assert not any(cls is case.framework for cls, *_rest in plan._class_guards)
    for _cls, _name, value, signature in _authored(case):
        target = (
            value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
        )
        assert signature == (id(target), id(getattr(target, "__code__", None)))


@pytest.mark.parametrize("kind", _KINDS)
def test_guard_failure_retains_partial_prefix_at_each_callers_binding_boundary(
    kind, monkeypatch
):
    case = _case(kind)
    events, _binding_error, guard_error = _observe_bindings(
        case, monkeypatch, fail_guard=True
    )
    with pytest.raises(RuntimeError) as caught:
        case.construct()
    assert caught.value is guard_error
    stop = next(
        index
        for index, (_cls, _name, value) in enumerate(case.expected)
        if value is case.stop
    )
    _assert_prefix(case, case.expected[:stop])
    assert events[-1] == "class-guard"
    assert all(event == "source-binding" for event in events[:-1])
    assert bool(events[:-1]) is (kind == "connection")
    assert case.descriptor_calls == []


@pytest.mark.parametrize("kind", _KINDS)
def test_competing_binding_and_guard_failures_preserve_constructor_precedence(
    kind, monkeypatch
):
    case = _case(kind)
    events, binding_error, guard_error = _observe_bindings(
        case, monkeypatch, fail_binding=True, fail_guard=True
    )
    with pytest.raises(RuntimeError) as caught:
        case.construct()
    if kind == "connection":
        assert caught.value is binding_error
        assert events == ["source-binding"]
        assert case.plan._class_guards == []
    else:
        assert caught.value is guard_error
        assert events == ["class-guard"]
        stop = next(
            index
            for index, (_cls, _name, value) in enumerate(case.expected)
            if value is case.stop
        )
        _assert_prefix(case, case.expected[:stop])


@pytest.mark.parametrize("kind", _KINDS)
def test_source_binding_failure_keeps_only_the_guards_captured_before_it(
    kind, monkeypatch
):
    case = _case(kind)
    events, binding_error, _guard_error = _observe_bindings(
        case, monkeypatch, fail_binding=True
    )
    with pytest.raises(RuntimeError) as caught:
        case.construct()
    assert caught.value is binding_error
    if kind == "connection":
        assert events == ["source-binding"]
        assert case.plan._class_guards == []
    else:
        assert events == ["class-guard", "source-binding"]
        _assert_prefix(case, case.expected)
