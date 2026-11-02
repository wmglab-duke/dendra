"""Canonical projection shares traversal while authored clones retain their effects."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, MethodType, SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra.func._network import FunctionalNetwork, _TensorSchema
from dendra.func._population import FunctionalPopulation
from dendra.func._state_projection import _clone_state_leaf, _state_projection
from dendra.func._tensor_contracts import _clone_contiguous_tree
from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile

pytestmark = pytest.mark.cpu


class _Leaf:
    def __init__(self, name, events, failure=None):
        self.name, self.events, self.failure = name, events, failure

    def clone(self, **kwargs):
        assert kwargs == {"memory_format": torch.contiguous_format}
        self.events.append(("clone", self.name))
        if self.failure is not None:
            raise self.failure
        return _Leaf(self.name, self.events)


class _Observed(Mapping):
    def __init__(self, name, values, events, failures=None):
        self.name, self.values, self.events = name, values, events
        self.failures = {} if failures is None else failures

    def __getitem__(self, name):
        self.events.append(("get", self.name, name))
        if name in self.failures:
            raise self.failures[name]
        return self.values[name]

    def __iter__(self):
        raise AssertionError("canonical lookup must use the declared paths")

    def __len__(self):
        raise AssertionError("canonical lookup must not inspect mapping length")


def _population_plan():
    plan = object.__new__(FunctionalPopulation)
    plan._state_layout = tuple(
        SimpleNamespace(public_path=path)
        for path in (("integrator", "v"), ("clock", "t"), ("integrator", "alias"))
    )
    # Publication order differs from the traversal's random-before-stimulation.
    plan._state_schema = dict.fromkeys(("clock", "integrator", "stimulation", "random"))
    plan._random_plan = plan._poisson_plan = object()
    return plan


def _population_state(events, *, clone_failure=None, lookup_failure=None):
    shared = _Leaf("voltage", events, clone_failure)
    return _Observed(
        "state",
        {
            "stimulation": {"schedule": _Leaf("schedule", events)},
            "random": {"rng": _Leaf("rng", events)},
            "integrator": _Observed(
                "integrator", {"v": shared, "alias": shared}, events
            ),
            "clock": _Observed("clock", {"t": _Leaf("time", events)}, events),
        },
        events,
        {} if lookup_failure is None else {"clock": lookup_failure},
    )


_LOOKUPS = [
    ("get", "state", "integrator"),
    ("get", "integrator", "v"),
    ("get", "state", "clock"),
    ("get", "clock", "t"),
    ("get", "state", "integrator"),
    ("get", "integrator", "alias"),
    ("get", "state", "random"),
    ("get", "state", "stimulation"),
]


def test_population_projection_and_clone_share_order_without_changing_ownership():
    events = []
    plan, state = _population_plan(), _population_state(events)
    projected = plan._project_state_tree(state)
    assert events == _LOOKUPS
    assert tuple(projected) == tuple(plan._state_schema)
    assert tuple(projected["integrator"]) == ("v", "alias")
    assert projected["integrator"]["v"] is projected["integrator"]["alias"]
    assert projected["random"]["rng"] is state.values["random"]["rng"]
    events.clear()
    cloned = plan._clone_state_tree(state)
    assert events == [
        *_LOOKUPS[:2],
        ("clone", "voltage"),
        *_LOOKUPS[2:4],
        ("clone", "time"),
        *_LOOKUPS[4:6],
        ("clone", "voltage"),
        _LOOKUPS[6],
        ("clone", "rng"),
        _LOOKUPS[7],
        ("clone", "schedule"),
    ]
    assert tuple(cloned) == tuple(projected)
    assert cloned["integrator"]["v"] is not cloned["integrator"]["alias"]
    for group in projected:
        for name, value in projected[group].items():
            assert cloned[group][name] is not value


@pytest.mark.parametrize("first", ("clone", "lookup"))
def test_population_clone_keeps_authored_exception_identity_and_read_stop(first):
    events = []
    clone_error, lookup_error = RuntimeError("clone"), RuntimeError("lookup")
    state = _population_state(
        events,
        clone_failure=clone_error if first == "clone" else None,
        lookup_failure=lookup_error,
    )
    with pytest.raises(RuntimeError) as caught:
        _population_plan()._clone_state_tree(state)
    assert caught.value is (clone_error if first == "clone" else lookup_error)
    expected = [*_LOOKUPS[:2], ("clone", "voltage")]
    assert events == (expected if first == "clone" else [*expected, _LOOKUPS[2]])


def test_population_clone_retains_contiguous_per_occurrence_autograd_paths():
    plan = _population_plan()
    plan._state_schema = dict.fromkeys(("integrator", "clock"))
    plan._random_plan = plan._poisson_plan = None
    source = torch.arange(6, dtype=torch.float64, requires_grad=True)
    value = source.reshape(2, 3).T
    state = {
        "clock": {"t": torch.tensor(0.0)},
        "integrator": {"alias": value, "v": value},
    }
    projected = plan._project_state_tree(state)
    assert projected["integrator"]["v"] is projected["integrator"]["alias"] is value
    cloned = plan._clone_state_tree(state)
    left, right = cloned["integrator"].values()
    assert left.is_contiguous() and right.is_contiguous()
    assert not torch._C._is_alias_of(left, right)
    assert not torch._C._is_alias_of(left, source)
    assert left.requires_grad and right.requires_grad
    (gradient,) = torch.autograd.grad(
        left.square().sum() + right.square().sum(), source
    )
    torch.testing.assert_close(gradient, 4 * source)


def test_network_projection_preserves_schema_order_namedtuple_none_and_aliases():
    class Frame(NamedTuple):
        value: object
        optional: object

    plan = object.__new__(FunctionalNetwork)
    leaf = _TensorSchema((), torch.float64, torch.device("cpu"))
    plan._state_schema = {
        "populations": {"right": {"v": leaf}, "left": {"v": leaf}},
        "transport": Frame(leaf, None),
        "none": None,
    }
    events = []
    shared = _Leaf("shared", events)
    state = {
        "none": None,
        "transport": Frame(shared, None),
        "populations": {"left": {"v": shared}, "right": {"v": shared}},
    }
    projected = plan._project_state_tree(state)
    assert events == []
    assert tuple(projected) == tuple(plan._state_schema)
    assert tuple(projected["populations"]) == ("right", "left")
    assert type(projected["transport"]) is Frame
    assert projected["transport"].optional is projected["none"] is None
    assert projected["transport"].value is shared
    assert all(entry["v"] is shared for entry in projected["populations"].values())
    cloned = plan._clone_state_tree(state)
    assert events == [("clone", "shared")] * 3
    assert type(cloned["transport"]) is Frame
    assert cloned["none"] is cloned["transport"].optional is None
    copies = [entry["v"] for entry in cloned["populations"].values()]
    copies.append(cloned["transport"].value)
    assert len({id(value) for value in copies}) == 3
    assert all(value is not shared for value in copies)


@pytest.mark.parametrize("replacement", ("instance", "class", "code"))
def test_projection_association_rejects_replaced_clone_policies(
    monkeypatch, replacement
):
    class Adapter:
        def clone(self, state):
            return {"value": _clone_state_leaf(state["value"])}

        @_state_projection(clone)
        def project(self, state):
            return state

    def changed(self, state):
        return {"value": _clone_state_leaf(state["value"] + 5)}

    adapter = Adapter()
    if replacement == "instance":
        adapter.clone = MethodType(changed, adapter)
    elif replacement == "class":
        monkeypatch.setattr(Adapter, "clone", changed)
    else:
        monkeypatch.setattr(Adapter.clone, "__code__", changed.__code__)
    engine = _CompiledTensorTreeWhile(
        lambda *args: None,
        steps=1,
        compile_options={},
        device="cpu",
        clone_state=adapter.clone,
        project_state=adapter.project,
    )
    layout, packs, _ = engine._pack({"value": torch.tensor(2.0)}, {})
    state, carries = layout.unpack(packs)
    assert state["value"].item() == 7 and carries == {}


def test_generic_clone_adapter_without_projection_keeps_its_value_transform():
    engine = _CompiledTensorTreeWhile(
        lambda *args: None,
        steps=1,
        compile_options={},
        device="cpu",
        clone_state=lambda state: {"value": _clone_state_leaf(state["value"] + 5)},
    )
    layout, packs, _ = engine._pack({"value": torch.tensor(2.0)}, {})
    state, _ = layout.unpack(packs)
    assert state["value"].item() == 7


def test_authored_mapping_falls_back_before_reads_and_clones_each_leaf_immediately():
    class Adapter:
        def clone(self, state):
            return {name: _clone_state_leaf(state[name]) for name in ("first", "last")}

        @_state_projection(clone)
        def project(self, state):
            raise AssertionError("authored mappings must not be projected")

    events = []
    first, last = torch.tensor(1.0), torch.tensor(2.0)

    class Authored(Mapping):
        def __getitem__(self, name):
            events.append(name)
            if name == "last":
                first.add_(10)
            return first if name == "first" else last

        def __iter__(self):
            raise AssertionError("admission must not enumerate authored mappings")

        def __len__(self):
            raise AssertionError("admission must not inspect authored mapping length")

    adapter = Adapter()
    engine = _CompiledTensorTreeWhile(
        lambda *args: None,
        steps=1,
        compile_options={},
        device="cpu",
        clone_state=adapter.clone,
        project_state=adapter.project,
    )
    layout, packs, _ = engine._pack(Authored(), {})
    state, _ = layout.unpack(packs)
    assert events == ["first", "last"]
    assert first.item() == 11
    assert state["first"].item() == 1 and state["last"].item() == 2


def _stock_packing_engine():
    @_state_projection(_clone_contiguous_tree)
    def project(state):
        return state

    return _CompiledTensorTreeWhile(
        lambda *args: None,
        steps=1,
        compile_options={},
        device="cpu",
        clone_state=_clone_contiguous_tree,
        project_state=project,
    )


@pytest.mark.parametrize("method", ("clone", "reshape"))
def test_tensor_instance_method_overrides_keep_the_original_clone_path(
    monkeypatch, method
):
    engine = _stock_packing_engine()
    value = torch.tensor(2.0)
    version = value._version
    events = []
    original_clone = torch.Tensor.clone

    def clone(self, **kwargs):
        events.append(("clone", kwargs))
        return original_clone(self, **kwargs) + 3

    def reshape(self, *shape):
        raise AssertionError("packing must not reshape an authored source instance")

    monkeypatch.setattr(
        value, method, MethodType(clone if method == "clone" else reshape, value)
    )
    layout, packs, _ = engine._pack({"first": value, "again": value}, {})
    state, carries = layout.unpack(packs)
    assert carries == {} and engine._layouts == {}
    assert events == (
        [("clone", {"memory_format": torch.contiguous_format})] * 2
        if method == "clone"
        else []
    )
    expected = 5 if method == "clone" else 2
    assert state["first"].item() == state["again"].item() == expected
    assert value.item() == 2 and value._version == version
    assert all(not torch._C._is_alias_of(pack, value) for pack in packs)


def test_tensor_class_clone_replacement_after_import_keeps_per_occurrence_effects(
    monkeypatch,
):
    engine = _stock_packing_engine()
    value = torch.tensor(2.0)
    original = torch.Tensor.clone
    events = []

    def changed(self, **kwargs):
        copied = original(self, **kwargs)
        if self is value:
            events.append(kwargs)
            return copied + 3
        return copied

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", changed)
        layout, packs, _ = engine._pack({"first": value, "again": value}, {})
        state, _ = layout.unpack(packs)
    assert events == [{"memory_format": torch.contiguous_format}] * 2
    assert state["first"].item() == state["again"].item() == 5
    assert value.item() == 2 and engine._layouts == {}


def test_stock_default_device_context_keeps_canonical_projection_eligible():
    events = []

    class Adapter:
        def clone(self, state):
            events.append("clone")
            return _clone_contiguous_tree(state)

        @_state_projection(clone)
        def project(self, state):
            events.append("project")
            return state

    adapter = Adapter()
    engine = _CompiledTensorTreeWhile(
        lambda *args: None,
        steps=1,
        compile_options={},
        device="cpu",
        clone_state=adapter.clone,
        project_state=adapter.project,
    )
    value = torch.tensor(2.0)
    with torch.device("cpu"):
        layout, packs, _ = engine._pack({"value": value}, {})
    state, _ = layout.unpack(packs)
    assert events == ["project"] and len(engine._layouts) == 1
    assert state["value"].item() == 2
    assert not torch._C._is_alias_of(state["value"], value)


def test_functional_tensor_wrapper_retains_the_clone_fallback():
    engine = _stock_packing_engine()
    value = torch._to_functional_tensor(torch.tensor(2.0))
    assert type(value) is torch.Tensor and torch._is_functional_tensor(value)
    layout, packs, _ = engine._pack({"value": value}, {})
    state, _ = layout.unpack(packs)
    assert engine._layouts == {}
    torch._sync(state["value"])
    torch.testing.assert_close(
        torch._from_functional_tensor(state["value"]), torch.tensor(2.0)
    )


def test_layout_cache_distinguishes_none_positions_and_tree_structure_at_equal_sizes():
    engine = _stock_packing_engine()
    value = torch.tensor([2.0, 3.0])
    states = (
        {"left": value, "right": None},
        {"left": None, "right": value},
        {"left": {"nested": value}, "right": None},
        {"left": value + 5, "right": None},
    )
    layouts, sizes = [], []
    for original in states:
        layout, packs, _ = engine._pack(original, {})
        actual, carries = layout.unpack(packs)
        assert carries == {}
        assert pytree.tree_structure(actual) == pytree.tree_structure(original)
        for leaf, reference in zip(
            pytree.tree_leaves(actual), pytree.tree_leaves(original), strict=True
        ):
            if torch.is_tensor(reference):
                torch.testing.assert_close(leaf, reference)
            else:
                assert leaf is reference is None
        layouts.append(layout)
        sizes.append(tuple(pack.numel() for pack in packs))
    assert sizes[0] == sizes[1] == sizes[2] == sizes[3]
    assert len(engine._layouts) == 3
    assert layouts[0] is layouts[3]
    assert layouts[0] != layouts[1] and layouts[0] != layouts[2]


@pytest.mark.parametrize("container", ("namedtuple", "readonly"))
def test_typed_or_readonly_state_uses_canonical_clone_fallback(container):
    class Frame(NamedTuple):
        value: torch.Tensor
        optional: object

    source = torch.tensor(2.0)
    leaf = _TensorSchema((), source.dtype, source.device)
    plan = object.__new__(FunctionalNetwork)
    if container == "namedtuple":
        plan._state_schema = {"frame": Frame(leaf, None), "none": None}
        original = {"none": None, "frame": Frame(source, None)}
    else:
        plan._state_schema = {"first": leaf, "last": None}
        original = MappingProxyType({"last": None, "first": source})
    events = []

    class Adapter:
        def clone(self, state):
            events.append("clone")
            return plan._clone_state_tree(state)

        @_state_projection(clone)
        def project(self, state):
            raise AssertionError(
                "observable container must not enter reference projection"
            )

    adapter = Adapter()
    engine = _CompiledTensorTreeWhile(
        lambda *args: None,
        steps=1,
        compile_options={},
        device="cpu",
        clone_state=adapter.clone,
        project_state=adapter.project,
    )
    layout, packs, _ = engine._pack(original, {})
    actual, _ = layout.unpack(packs)
    assert events == ["clone"] and engine._layouts == {}
    assert tuple(actual) == tuple(plan._state_schema)
    if container == "namedtuple":
        assert type(actual["frame"]) is Frame
        assert actual["none"] is actual["frame"].optional is None
        copied = actual["frame"].value
    else:
        assert actual["last"] is None
        copied = actual["first"]
    torch.testing.assert_close(copied, source)
    assert not torch._C._is_alias_of(copied, source)
