"""Registered source walks preserve alias, validation, and suspension boundaries."""

import pytest
import torch

from dendra.func import _continuous_transform as transforms
from dendra.func import _initialization_sources as initializers
from dendra.func import _stimuli as stimuli
from dendra.func._connection_parameters import _registered
from dendra.func._types import FunctionalizationError

pytestmark = pytest.mark.cpu


class _ObservedChildren(dict):
    def __init__(self, children, reads, label):
        super().__init__(children)
        self.reads, self.label = reads, label

    def items(self):
        self.reads.append(self.label)
        return super().items()


def _observe_children(module, reads, label):
    namespace = object.__getattribute__(module, "__dict__")
    namespace["_modules"] = _ObservedChildren(namespace["_modules"], reads, label)


def _diamond():
    root, left, right, leaf = (torch.nn.Module() for _ in range(4))
    root.add_module("left", left)
    root.add_module("absent", None)
    root.add_module("right", right)
    left.add_module("leaf", leaf)
    right.add_module("leaf", leaf)
    leaf.add_module("back", root)
    return root, left, right, leaf


@pytest.mark.parametrize(
    "walk",
    (
        transforms._safe_modules,
        initializers._registered_modules,
        stimuli._waveform_registered_modules,
    ),
)
def test_canonical_walk_keeps_first_paths_and_stops_alias_back_edges(walk):
    root, left, right, leaf = _diamond()
    entries = tuple(walk(root))
    assert tuple(path for path, _ in entries) == ("", "left", "left.leaf", "right")
    assert all(
        actual is expected
        for (_, actual), expected in zip(
            entries, (root, left, leaf, right), strict=True
        )
    )


def test_all_alias_walk_revisits_diamond_paths_but_not_ancestors():
    root, left, right, leaf = _diamond()
    entries = tuple(initializers._registered_modules(root, duplicates=True))
    assert tuple(path for path, _ in entries) == (
        "",
        "left",
        "left.leaf",
        "right",
        "right.leaf",
    )
    assert all(
        actual is expected
        for (_, actual), expected in zip(
            entries, (root, left, leaf, right, leaf), strict=True
        )
    )


@pytest.mark.parametrize(
    "walk", (initializers._registered_modules, stimuli._waveform_registered_modules)
)
def test_lazy_walk_does_not_read_children_until_iteration_resumes(walk):
    root, first, later = (torch.nn.Module() for _ in range(3))
    root.add_module("first", first)
    reads = []
    _observe_children(root, reads, "root")
    _observe_children(first, reads, "first")
    iterator = walk(root)
    assert reads == []
    path, module = next(iterator)
    assert path == "" and module is root
    assert reads == []
    root.add_module("later", later)
    path, module = next(iterator)
    assert path == "first" and module is first
    assert reads == ["root"]
    iterator.close()
    assert reads == ["root"]
    assert tuple(path for path, _ in walk(root)) == ("", "first", "later")


class _CustomCall(torch.nn.Module):
    def __call__(self, value):
        raise AssertionError("authored call must not execute during qualification")


def test_transform_qualification_is_eager_before_child_registration_reads():
    root, invalid = torch.nn.Module(), _CustomCall()
    root.add_module("invalid", invalid)
    reads = []
    _observe_children(root, reads, "root")
    _observe_children(invalid, reads, "invalid")
    with pytest.raises(FunctionalizationError) as caught:
        transforms._safe_modules(root)
    assert str(caught.value) == (
        "Continuous transform needs canonical module call/attribute dispatch."
    )
    assert reads == ["root"]


def test_waveform_qualification_is_lazy_before_child_registration_reads():
    root, invalid = torch.nn.Module(), _CustomCall()
    root.add_module("invalid", invalid)
    reads = []
    _observe_children(root, reads, "root")
    _observe_children(invalid, reads, "invalid")
    iterator = stimuli._waveform_registered_modules(root)
    assert reads == []
    assert next(iterator) == ("", root)
    assert reads == []
    with pytest.raises(FunctionalizationError) as caught:
        next(iterator)
    assert str(caught.value) == (
        "Waveform 'invalid' requires canonical module call/attribute dispatch; "
        "express custom behavior in forward or fn and use registered "
        "parameters, buffers, and child modules."
    )
    assert reads == ["root"]


def test_waveform_suspended_child_lookup_keeps_authored_namespace_exception():
    namespace_descriptor = torch.nn.Module.__dict__["__dict__"]
    active, fail_lookup = False, False
    reads = []
    failure = AttributeError("authored namespace lookup failure")

    class ObservedNamespace(torch.nn.Module):
        @property
        def __dict__(self):
            if active:
                reads.append("namespace")
                if fail_lookup:
                    raise failure
            return namespace_descriptor.__get__(self, type(self))

    root = ObservedNamespace()
    active = True
    iterator = stimuli._waveform_registered_modules(root)
    assert reads == []
    assert next(iterator) == ("", root)
    qualified_reads = len(reads)
    assert qualified_reads > 0
    fail_lookup = True
    with pytest.raises(AttributeError) as caught:
        next(iterator)
    assert caught.value is failure
    assert len(reads) == qualified_reads + 1


def test_transform_storage_descriptor_rejection_never_executes_getter(monkeypatch):
    class PoisonedStorage(torch.nn.Module):
        pass

    root, invalid = torch.nn.Module(), PoisonedStorage()
    root.add_module("invalid", invalid)
    calls = []

    def poison(module):
        calls.append("getter")
        raise AssertionError("authored registration descriptor must not execute")

    monkeypatch.setattr(PoisonedStorage, "_modules", property(poison), raising=False)
    with pytest.raises(FunctionalizationError) as caught:
        transforms._safe_modules(root)
    assert str(caught.value) == (
        "Continuous transform shadows framework storage with a descriptor or class field."
    )
    assert calls == []


def test_tensor_consumers_keep_distinct_order_and_cross_category_alias_selection():
    root, child = torch.nn.Module(), torch.nn.Module()
    root.register_parameter("root_parameter", torch.nn.Parameter(torch.tensor(1.0)))
    shared = torch.nn.Parameter(torch.tensor(2.0))
    root.register_buffer("root_alias", shared)
    child.register_parameter("child_parameter", shared)
    child.register_buffer("child_buffer", torch.tensor(3.0))
    root.add_module("child", child)
    root.add_module("child_alias", child)
    named = transforms._named_tensors(root)
    assert tuple(named) == (
        "root_parameter",
        "child.child_parameter",
        "child.child_buffer",
    )
    assert named["child.child_parameter"] is shared
    slots = tuple(initializers._registered_tensor_slots(root))
    assert tuple(name for name, _ in slots) == (
        "root_parameter",
        "root_alias",
        "child.child_parameter",
        "child.child_buffer",
    )
    assert slots[1][1] is slots[2][1] is shared
    all_slots = tuple(initializers._registered_tensor_slots(root, duplicates=True))
    assert tuple(name for name, _ in all_slots) == (
        "root_parameter",
        "root_alias",
        "child.child_parameter",
        "child.child_buffer",
        "child_alias.child_parameter",
        "child_alias.child_buffer",
    )
    assert all_slots[4][1] is shared
    registered = tuple(_registered(root))
    assert tuple((name, kind) for name, _, kind in registered) == (
        ("root_parameter", "_parameters"),
        ("root_alias", "_buffers"),
        ("child.child_parameter", "_parameters"),
        ("child.child_buffer", "_buffers"),
    )
    assert registered[1][1] is registered[2][1] is shared
