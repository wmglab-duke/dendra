"""Nested frame endpoints preserve lazy lookups and in-place publication order."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import pytest
import torch

from dendra.func import _callbacks, _network, _network_bindings, _population, _scan_tree

pytestmark = pytest.mark.cpu
_READERS = (
    pytest.param(_callbacks._get_path, id="callbacks"),
    pytest.param(_population._get_path, id="population"),
    pytest.param(_scan_tree._get_path, id="scan_tree"),
    pytest.param(_network_bindings._tree_value, id="network_binding"),
    pytest.param(_network._at, id="network"),
)
_WRITERS = (
    pytest.param(_callbacks._put_path, id="callbacks"),
    pytest.param(_population._put_path, id="population"),
)


class _Mapping(Mapping):
    def __init__(self, values, events, label, failure=None):
        self.values, self.events, self.label, self.failure = (
            values,
            events,
            label,
            failure,
        )

    def __getitem__(self, key):
        self.events.append((self.label, key))
        if self.failure is not None:
            raise self.failure
        return self.values[key]

    def __iter__(self):
        raise AssertionError("lookup must not enumerate a frame")

    def __len__(self):
        raise AssertionError("lookup must not inspect frame length")


@pytest.mark.parametrize("read", _READERS)
def test_empty_read_returns_the_exact_root_without_inspecting_it(read):
    events = []
    roots = None, [], torch.tensor([2.0]), _Mapping({}, events, "unused")

    class EmptyPath:
        def __iter__(self):
            events.append("path")
            return iter(())

        def __len__(self):
            raise AssertionError("path must not be materialized")

    for root in roots:
        assert read(root, EmptyPath()) is root
    assert events == ["path"] * len(roots)


@pytest.mark.parametrize("read", _READERS)
def test_nested_mapping_sequence_lookup_returns_the_same_tensor_and_ad_graph(read):
    events = []
    source = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    scale = torch.arange(1.0, 5.0, dtype=torch.float64)
    tensor = (source * scale).sin()
    branch = _Mapping({"value": tensor}, events, "branch")
    root = _Mapping({"stage": [branch, None]}, events, "root")
    version = tensor._version

    actual = read(root, ("stage", 0, "value"))

    assert actual is tensor and tensor._version == version
    assert events == [("root", "stage"), ("branch", "value")]
    derivative = torch.autograd.grad(actual.sum(), source)[0]
    torch.testing.assert_close(derivative, (scale * (source * scale).cos()).sum())


@pytest.mark.parametrize("read", _READERS)
def test_authored_path_and_getitem_errors_preserve_lazy_interleaving_and_stop(read):
    marker = object()
    for failure_phase in (None, "path", "getitem"):
        events, failure = [], RuntimeError("authored lookup failure")
        child = _Mapping(
            {"leaf": marker},
            events,
            "child",
            failure if failure_phase == "getitem" else None,
        )
        root = _Mapping({"branch": child}, events, "root")

        def path():
            events.append("yield:branch")
            yield "branch"
            if failure_phase == "path":
                events.append("path:failure")
                raise failure
            events.append("yield:leaf")
            yield "leaf"
            events.append("path:end")

        if failure_phase is None:
            assert read(root, path()) is marker
        else:
            with pytest.raises(RuntimeError) as caught:
                read(root, path())
            assert caught.value is failure
        prefix = ["yield:branch", ("root", "branch")]
        if failure_phase == "path":
            assert events == [*prefix, "path:failure"]
        elif failure_phase == "getitem":
            assert events == [*prefix, "yield:leaf", ("child", "leaf")]
        else:
            assert events == [*prefix, "yield:leaf", ("child", "leaf"), "path:end"]

    with pytest.raises(KeyError) as missing:
        read({"branch": {}}, ("branch", "missing"))
    assert missing.value.args == ("missing",)
    with pytest.raises(TypeError) as none:
        read({"branch": None}, ("branch", "leaf"))
    assert str(none.value) == "'NoneType' object is not subscriptable"
    with pytest.raises(IndexError) as empty:
        read({"branch": []}, ("branch", 0))
    assert str(empty.value) == "list index out of range"


class _ObservedDict(dict):
    def __init__(self, label, events, defaults, values=()):
        super().__init__(values)
        self.label, self.events, self.defaults = label, events, defaults

    def setdefault(self, key, default):
        self.events.append((self.label, "setdefault", key))
        self.defaults.append(default)
        return super().setdefault(key, default)

    def __setitem__(self, key, value):
        self.events.append((self.label, "assign", key))
        return super().__setitem__(key, value)


@pytest.mark.parametrize("put", _WRITERS)
def test_write_keeps_existing_branch_aliases_and_allocates_fresh_plain_defaults(put):
    events, defaults = [], []
    branch = _ObservedDict("branch", events, defaults)
    root = _ObservedDict("root", events, defaults, {"first": branch, "second": branch})
    source = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    value = source.sin()

    assert put(root, ("first", "a", "leaf"), value) is None
    assert put(root, ("second", "b", "leaf"), value) is None
    assert put(root, ("first", "direct"), value) is None

    assert events == [
        ("root", "setdefault", "first"),
        ("branch", "setdefault", "a"),
        ("root", "setdefault", "second"),
        ("branch", "setdefault", "b"),
        ("root", "setdefault", "first"),
        ("branch", "assign", "direct"),
    ]
    assert root["first"] is root["second"] is branch
    assert branch["a"] is defaults[1] and branch["b"] is defaults[3]
    assert len(defaults) == 5 and len({id(item) for item in defaults}) == 5
    assert all(type(item) is dict for item in defaults)
    assert defaults[0] == defaults[2] == defaults[4] == {}
    assert branch["a"]["leaf"] is branch["b"]["leaf"] is branch["direct"] is value
    torch.testing.assert_close(
        torch.autograd.grad(branch["direct"], source)[0], source.cos()
    )

    def unexpected_factory():
        raise AssertionError("setdefault must provide its own ordinary dict")

    native = defaultdict(unexpected_factory)
    put(native, ("new", "leaf"), value)
    assert type(native["new"]) is dict and native["new"]["leaf"] is value


@pytest.mark.parametrize("put", _WRITERS)
def test_write_failures_keep_prior_mutations_and_exact_path_publication_order(put):
    for failure_phase in ("setdefault", "iteration", "last", "assignment"):
        events, failure, marker = [], RuntimeError("authored write failure"), object()

        class Refuse(dict):
            def __setitem__(self, key, value):
                events.append(("assign", key))
                super().__setitem__(key, value)
                raise failure

        class Root(dict):
            def setdefault(self, key, default):
                events.append(("setdefault", key))
                result = super().setdefault(key, default)
                if failure_phase == "setdefault":
                    raise failure
                return result

        branch = Refuse() if failure_phase == "assignment" else None
        root = Root({"created": branch}) if branch is not None else Root()

        class Path:
            def __getitem__(self, key):
                if isinstance(key, slice):
                    assert key == slice(None, -1)
                    events.append("slice")
                    return self.prefix()
                assert key == -1
                events.append("last")
                if failure_phase == "last":
                    raise failure
                return "leaf"

            def prefix(self):
                events.append("yield:created")
                yield "created"
                if failure_phase == "iteration":
                    events.append("iteration:failure")
                    raise failure
                events.append("prefix:end")

        with pytest.raises(RuntimeError) as caught:
            put(root, Path(), marker)
        assert caught.value is failure
        prefix = ["slice", "yield:created", ("setdefault", "created")]
        if failure_phase == "setdefault":
            assert events == prefix
        elif failure_phase == "iteration":
            assert events == [*prefix, "iteration:failure"]
        elif failure_phase == "last":
            assert events == [*prefix, "prefix:end", "last"]
        else:
            assert events == [*prefix, "prefix:end", "last", ("assign", "leaf")]
        if failure_phase == "assignment":
            assert root["created"] is branch and branch["leaf"] is marker
        else:
            assert root == {"created": {}}


@pytest.mark.parametrize("put", _WRITERS)
def test_empty_write_preserves_native_index_error_without_touching_destination(put):
    events, defaults, marker = [], [], object()
    root = _ObservedDict("root", events, defaults, {"existing": marker})

    with pytest.raises(IndexError) as caught:
        put(root, (), object())

    assert str(caught.value) == "tuple index out of range"
    assert root == {"existing": marker}
    assert root["existing"] is marker
    assert events == defaults == []
