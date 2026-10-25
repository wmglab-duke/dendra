"""Required class guards retain raw namespaces and per-class audit ordering."""

import builtins

import pytest
import torch

from dendra.func import _continuous_transform as transforms
from tests.func.test_continuous_binding_guard_contracts import _collect, _plan

pytestmark = pytest.mark.cpu

_COLLECTORS = ("binding", "diagnostic")
_REQUIRED = (
    "__call__",
    "_call_impl",
    "_wrapped_call_impl",
    "__getattr__",
    "__getattribute__",
    "named_modules",
    "modules",
    "named_parameters",
    "named_buffers",
    "named_children",
    "parameters",
    "buffers",
    "__deepcopy__",
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
    "forward",
)


def _source():
    getters = []

    class Grand(torch.nn.Module):
        def forward(self, value):
            return value

    class Left(Grand):
        pass

    class Right(torch.nn.Module):
        @property
        def __getstate__(self):
            getters.append("getter")
            raise AssertionError("Class guard capture resolved a descriptor")

    class Root(Left, Right):
        marker = 19

    class Child(Root):
        pass

    root, child = Root(), Child()
    root.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.0)))
    root.register_buffer("optional", None)
    child.register_parameter("offset", torch.nn.Parameter(torch.tensor(2.0)))
    child.register_buffer("static", torch.tensor(3.0))
    root.add_module("child", child)
    root.add_module("alias", child)
    root.add_module("absent", None)
    # Controlled registrations exercise duplicate names without numerical
    # admission or descriptor resolution, neither of which belongs here.
    root._parameters["forward"] = None
    Right.optional = None
    return root, child, (Root, Left, Grand, Right), getters


def _names(child):
    return (
        *_REQUIRED,
        *child._parameters,
        *child._buffers,
        *child._modules,
        *transforms._INTERNAL_FIELDS,
    )


def _expected(kind, children):
    records = []
    metadata = transforms._helpers()._CLASS_STATE_METADATA_NAMES
    for child in children:
        for cls in type(child).__mro__:
            if cls in (torch.nn.Module, object):
                continue
            namespace = vars(cls)
            if kind == "diagnostic":
                records.extend(
                    (cls, name, True, value)
                    for name, value in namespace.items()
                    if name not in metadata
                )
            records.extend(
                (cls, name, name in namespace, namespace.get(name))
                for name in _names(child)
            )
    return records


@pytest.mark.parametrize("kind", _COLLECTORS)
def test_exact_mro_records_keep_raw_descriptors_none_duplicates_and_interleaving(kind):
    root, child, classes, getters = _source()
    plan = _plan(kind)
    _collect(plan, kind, root)
    assert plan._guard_classes == _expected(kind, (root, child))
    Root, _Left, Grand, Right = classes
    assert tuple(type(root).__mro__[:4]) == classes
    assert (Root, "optional", False, None) in plan._guard_classes
    assert (Right, "optional", True, None) in plan._guard_classes
    assert (Root, "forward", False, None) in plan._guard_classes
    assert (Grand, "forward", True, vars(Grand)["forward"]) in plan._guard_classes
    descriptor = vars(Right)["__getstate__"]
    assert any(
        cls is Right and name == "__getstate__" and value is descriptor
        for cls, name, _present, value in plan._guard_classes
    )
    assert plan._guard_classes.count((Root, "forward", False, None)) == 3
    assert getters == []


@pytest.mark.parametrize("kind", _COLLECTORS)
def test_required_names_are_read_again_at_each_class_boundary(kind):
    root, _child, classes, _getters = _source()
    root._modules.clear()
    Root, *_ = classes
    plan = _plan(kind)
    last_name = transforms._INTERNAL_FIELDS[-1]
    assert last_name not in vars(Root)

    class Records(list):
        def append(self, entry):
            super().append(entry)
            if entry[:2] == (Root, last_name):
                root.register_parameter("late", torch.nn.Parameter(torch.tensor(4.0)))

    plan._guard_classes = Records()
    _collect(plan, kind, root)
    assert "late" in root._parameters
    assert (Root, "late", False, None) not in plan._guard_classes
    for cls in classes[1:]:
        assert (cls, "late", False, None) in plan._guard_classes
    assert plan._guard_modules[0][1] == ("weight", "forward")


@pytest.mark.parametrize("kind", _COLLECTORS)
def test_append_failure_retains_only_the_completed_raw_and_required_prefix(kind):
    root, child, classes, getters = _source()
    expected = _expected(kind, (root, child))
    failure = RuntimeError("class guard append failed")
    target = (classes[0], "named_parameters", False, None)
    plan = _plan(kind)

    class Records(list):
        def append(self, entry):
            if entry == target:
                raise failure
            super().append(entry)

    plan._guard_classes = Records()
    with pytest.raises(RuntimeError) as caught:
        _collect(plan, kind, root)
    assert caught.value is failure
    assert plan._guard_classes == expected[: expected.index(target)]
    assert [entry[0] for entry in plan._guard_modules] == [root]
    assert plan._guard_functions == []
    assert getters == []


@pytest.mark.parametrize("kind", _COLLECTORS)
def test_raw_presence_and_value_reads_keep_namespace_lookup_and_error_order(
    kind, monkeypatch
):
    root, _child, classes, getters = _source()
    Root, *_ = classes
    qualified = transforms._safe_modules(root)
    namespace = vars(Root)
    events, failure = [], RuntimeError("raw class value unavailable")

    class Namespace:
        def items(self):
            events.append("raw-scan")
            return namespace.items()

        def __contains__(self, name):
            events.append(("present", name))
            return name in namespace

        def get(self, name):
            events.append(("value", name))
            if name == "named_parameters":
                raise failure
            return namespace.get(name)

    def observed_vars(value):
        if value is Root:
            events.append("namespace")
            return Namespace()
        return builtins.vars(value)

    monkeypatch.setattr(transforms, "_safe_modules", lambda _source: qualified)
    monkeypatch.setattr(transforms, "vars", observed_vars, raising=False)
    plan = _plan(kind)
    with pytest.raises(RuntimeError) as caught:
        _collect(plan, kind, root)
    assert caught.value is failure
    expected = ["namespace", "raw-scan"] if kind == "diagnostic" else []
    for name in _REQUIRED[: _REQUIRED.index("named_parameters") + 1]:
        expected.extend(("namespace", ("present", name), "namespace", ("value", name)))
    assert events == expected
    assert plan._guard_functions == []
    assert getters == []
