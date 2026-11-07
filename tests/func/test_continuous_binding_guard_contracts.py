"""Continuous transform collectors retain binding inventories and audit gates."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.func import _continuous_transform as transforms
from dendra.func._types import FunctionalizationError

pytestmark = pytest.mark.cpu

_COLLECTORS = ("binding", "diagnostic")


class _Node(torch.nn.Module):
    class_note = 17

    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.gain = 1.5

    def forward(self, value):
        return value


def _identity(value):
    return value


def _plan(kind, *, module=True):
    plan = object.__new__(transforms.ContinuousTransformPlan)
    plan.is_module = module
    plan._check_purity = kind == "diagnostic"
    for name in (
        "_guard_attributes",
        "_guard_classes",
        "_guard_functions",
        "_guard_modules",
        "_guard_tensor_slots",
        "_guard_instances",
    ):
        setattr(plan, name, [])
    plan._guard_function_ids, plan._guard_aliases = set(), {}
    return plan


def _collect(plan, kind, source):
    collector = (
        plan._collect_binding_guards if kind == "binding" else plan._collect_guards
    )
    collector(source)


def _diamond():
    root, left, leaf, right = (_Node(tag) for tag in ("root", "left", "leaf", "right"))
    weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    shared = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    static = torch.tensor([1.0, 2.0], dtype=torch.float32)
    root.register_parameter("weight", weight)
    root.register_parameter("missing", None)
    root.register_buffer("cross_alias", shared)
    root.register_buffer("optional", None)
    left.register_parameter("shared", shared)
    leaf.register_buffer("static", static)
    right.register_parameter("again", shared)
    root.add_module("left", left)
    root.add_module("absent", None)
    root.add_module("right", right)
    left.add_module("leaf", leaf)
    right.add_module("leaf", leaf)
    leaf.add_module("back", root)
    leaf.training = False
    return (root, left, leaf, right), (weight, shared, static)


@pytest.mark.parametrize("kind", _COLLECTORS)
def test_collectors_keep_live_namespaces_ordered_slots_and_one_global_alias_memo(kind):
    children, tensors = _diamond()
    root, left, leaf, right = children
    weight, shared, static = tensors
    versions = tuple(tensor._version for tensor in tensors)
    plan = _plan(kind)
    keys = tuple(tuple(vars(child)) for child in children)
    _collect(plan, kind, root)
    assert tuple(entry[0] for entry in plan._guard_modules) == children
    for child, saved_keys, (storage, captured_keys) in zip(
        children, keys, plan._guard_instances, strict=True
    ):
        assert storage is vars(child)
        assert captured_keys == saved_keys
    assert plan._guard_modules == [
        (
            root,
            ("weight", "missing"),
            ("cross_alias", "optional"),
            (("left", left), ("absent", None), ("right", right)),
        ),
        (left, ("shared",), (), (("leaf", leaf),)),
        (leaf, (), ("static",), (("back", root),)),
        (right, ("again",), (), (("leaf", leaf),)),
    ]
    scalar = ((), torch.float64, torch.device("cpu"))
    assert plan._guard_tensor_slots == [
        (root, "_parameters", "weight", scalar, 0),
        (root, "_parameters", "missing", None, None),
        (root, "_buffers", "cross_alias", scalar, 1),
        (root, "_buffers", "optional", None, None),
        (left, "_parameters", "shared", scalar, 1),
        (leaf, "_buffers", "static", ((2,), torch.float32, torch.device("cpu")), 2),
        (right, "_parameters", "again", scalar, 1),
    ]
    assert plan._guard_aliases == {id(weight): 0, id(shared): 1, id(static): 2}
    assert plan._guard_attributes == [
        entry
        for child in children
        for entry in (
            (child, "training", child.training, "config"),
            (child, "tag", child.tag, "config"),
            (child, "gain", child.gain, "config"),
        )
    ]
    assert len(plan._guard_functions) == (4 if kind == "binding" else 1)
    assert any(name == "class_note" for _cls, name, *_ in plan._guard_classes) is (
        kind == "diagnostic"
    )
    root.later = "new namespace key"
    assert plan._guard_instances[0][0]["later"] == "new namespace key"
    assert "later" not in plan._guard_instances[0][1]
    another = _Node("another")
    another.register_buffer("known", shared)
    fresh = torch.tensor(4.0)
    another.register_buffer("fresh", fresh)
    _collect(plan, kind, another)
    assert plan._guard_aliases[id(shared)] == 1
    assert plan._guard_aliases[id(fresh)] == 3
    assert [entry[4] for entry in plan._guard_tensor_slots[-2:]] == [1, 3]
    assert weight.requires_grad and shared.requires_grad
    assert weight.grad is None and shared.grad is None
    assert tuple(tensor._version for tensor in tensors) == versions
    torch.testing.assert_close(weight, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(shared, torch.tensor(3.0, dtype=torch.float64))
    torch.testing.assert_close(static, torch.tensor([1.0, 2.0], dtype=torch.float32))


@pytest.mark.parametrize("kind", _COLLECTORS)
@pytest.mark.parametrize("source", (None, _identity, torch.sin))
def test_nonmodule_collectors_keep_their_distinct_helper_resolution_boundaries(
    kind, source, monkeypatch
):
    plan, events = _plan(kind, module=False), []
    original = transforms._helpers

    def helpers():
        events.append("helpers")
        return original()

    def unexpected(*args):
        pytest.fail("Nonmodule collector attempted registered-module traversal")

    monkeypatch.setattr(transforms, "_helpers", helpers)
    monkeypatch.setattr(transforms, "_safe_modules", unexpected)
    _collect(plan, kind, source)
    assert events == (["helpers"] if kind == "binding" else [])
    assert len(plan._guard_functions) == (1 if source is _identity else 0)
    if source is _identity:
        assert plan._guard_functions[0] == (
            source,
            source.__code__,
            source.__defaults__,
            source.__kwdefaults__,
        )
    assert (
        plan._guard_instances == plan._guard_modules == plan._guard_tensor_slots == []
    )


@pytest.mark.parametrize("kind", _COLLECTORS)
def test_module_helper_resolution_failure_precedes_qualification(kind, monkeypatch):
    plan, events = _plan(kind), []
    failure = RuntimeError("helpers unavailable")

    def helpers():
        events.append("helpers")
        raise failure

    def unexpected(*args):
        pytest.fail("Qualification ran after helper resolution failed")

    monkeypatch.setattr(transforms, "_helpers", helpers)
    monkeypatch.setattr(transforms, "_safe_modules", unexpected)
    with pytest.raises(RuntimeError) as caught:
        _collect(plan, kind, _Node("root"))
    assert caught.value is failure
    assert events == ["helpers"]
    assert plan._guard_instances == plan._guard_modules == plan._guard_attributes == []


@pytest.mark.parametrize("kind", _COLLECTORS)
def test_eager_child_qualification_finishes_before_any_binding_capture(kind):
    class Invalid(torch.nn.Module):
        def __call__(self, value):
            raise AssertionError("Authored call must not execute")

    root, plan = _Node("root"), _plan(kind)
    root.add_module("invalid", Invalid())
    with pytest.raises(FunctionalizationError) as caught:
        _collect(plan, kind, root)
    assert str(caught.value) == (
        "Continuous transform needs canonical module call/attribute dispatch."
    )
    assert plan._guard_instances == plan._guard_modules == plan._guard_attributes == []
    assert (
        plan._guard_tensor_slots == plan._guard_classes == plan._guard_functions == []
    )


@pytest.mark.parametrize("kind", _COLLECTORS)
@pytest.mark.parametrize("failure_phase", ("shape", "dtype", "device"))
def test_tensor_schema_failure_keeps_attribute_read_and_partial_capture_order(
    kind, failure_phase
):
    events, plan = [], _plan(kind)
    failure = RuntimeError(f"{failure_phase} lookup failed")

    class Metadata:
        def __getattribute__(self, name):
            if name in ("shape", "dtype", "device"):
                events.append(name)
                if name == failure_phase:
                    raise failure
                return {
                    "shape": (2,),
                    "dtype": torch.float64,
                    "device": torch.device("cpu"),
                }[name]
            return object.__getattribute__(self, name)

    child = _Node("root")
    # Controlled slot metadata exposes collector read boundaries without
    # invoking numerical code or the constructor's earlier tensor admission.
    vars(child)["_parameters"]["observed"] = Metadata()
    with pytest.raises(RuntimeError) as caught:
        _collect(plan, kind, child)
    assert caught.value is failure
    order = ["shape", "dtype", "device"]
    assert events == order[: order.index(failure_phase) + 1]
    assert plan._guard_instances[0][0] is vars(child)
    assert plan._guard_modules == [(child, ("observed",), (), ())]
    assert plan._guard_tensor_slots == plan._guard_attributes == []
    assert plan._guard_aliases == {}


@pytest.mark.parametrize("kind", _COLLECTORS)
@pytest.mark.parametrize("fail", (False, True))
def test_training_guard_precedes_ordered_instance_predicates_and_configuration(
    kind, fail, monkeypatch
):
    child, plan, events = _Node("root"), _plan(kind), []
    child.register_buffer("offset", torch.tensor(0.25))
    real = transforms._helpers()
    failure = RuntimeError("gain classification failed")

    class Attributes(list):
        def append(self, entry):
            events.append(("attribute", entry[1]))
            super().append(entry)

    def internal(owner, name):
        assert owner is child
        events.append(("predicate", name))
        if fail and name == "gain":
            raise failure
        return real._is_internal_module_state(owner, name)

    def helpers():
        events.append(("helpers",))
        return SimpleNamespace(
            _is_internal_module_state=internal,
            _CLASS_STATE_METADATA_NAMES=real._CLASS_STATE_METADATA_NAMES,
        )

    plan._guard_attributes = Attributes()
    monkeypatch.setattr(transforms, "_helpers", helpers)
    if fail:
        with pytest.raises(RuntimeError) as caught:
            _collect(plan, kind, child)
        assert caught.value is failure
    else:
        _collect(plan, kind, child)
    assert events[:2] == [("helpers",), ("attribute", "training")]
    names = list(vars(child))
    if fail:
        names = names[: names.index("gain") + 1]
    expected_events = [("helpers",), ("attribute", "training")]
    for name in names:
        expected_events.append(("predicate", name))
        if name == "tag" or name == "gain" and not fail:
            expected_events.append(("attribute", name))
    assert events == expected_events
    assert [(entry[1], entry[2]) for entry in plan._guard_attributes] == (
        [("training", True), ("tag", "root")]
        if fail
        else [("training", True), ("tag", "root"), ("gain", 1.5)]
    )
    assert plan._guard_tensor_slots[0][2] == "offset"
    if fail:
        assert plan._guard_classes == plan._guard_functions == []


class _Gain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.gain = 1.5

    def forward(self, values):
        return values * self.weight * self.gain


@pytest.mark.parametrize("diagnostic", (False, True))
def test_real_plan_captures_purity_policy_and_retains_explicit_parameter_gradients(
    diagnostic, monkeypatch
):
    calls, source = [], _Gain()
    original_probe = transforms.ContinuousTransformPlan._probe

    def probe(plan):
        calls.append("probe")
        return original_probe(plan)

    monkeypatch.setattr(transforms.ContinuousTransformPlan, "_probe", probe)
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = transforms.ContinuousTransformPlan(
            source, size=2, device="cpu", dtype=torch.float64
        )
    assert calls == (["probe"] if diagnostic else [])
    with dn.ctx(CHECK_PURITY=not diagnostic):
        parameters = plan.extract(source)
        values = torch.tensor([-1.0, 2.0], dtype=torch.float64, requires_grad=True)
        result = plan.apply(values, parameters)
        plan.validate_source(source)
    assert plan._check_purity is diagnostic
    gradients = torch.autograd.grad(result.sum(), (values, parameters["weight"]))
    torch.testing.assert_close(gradients[0], torch.full_like(values, 3.0))
    torch.testing.assert_close(gradients[1], torch.tensor(1.5, dtype=torch.float64))
    assert source.weight.grad is None
