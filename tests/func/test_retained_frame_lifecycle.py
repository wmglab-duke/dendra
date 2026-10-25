"""Retained frame transitions preserve operation order and tensor ownership."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from tests.func.test_initialization_execution import _assert_snapshot, _snapshot

DT = 0.025
DTYPE = torch.float64
pytestmark = pytest.mark.cpu
_TIED_PATHS = tuple(
    f"integrator.mech.mechanisms._LifecycleLeak.{name}" for name in ("g", "e")
)


class _LifecycleLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-61.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * (v - self.e)


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(
        JIT=0,
        REQUIRE_GRAD=True,
        IMEM=False,
        DEVICE="cpu",
        DTYPE=DTYPE,
        CHECK_PURITY=False,
    ):
        yield


def _plan(*, retained=True):
    population = dn.SingleCompartment(
        N=2,
        C=3,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    population.insert(_LifecycleLeak)
    population.train()
    if not retained:
        population.initialize()
        return dn.func.make_functional(population, dt=DT)
    network = dn.Network({"cell": population})
    network.train()
    network.initialize(DT)
    functional, tensors = dn.func.make_functional_network(network)
    plan = functional.populations["cell"]
    packet = SimpleNamespace(
        parameters=tensors.parameters["populations"]["cell"],
        constants=tensors.constants["populations"]["cell"],
    )
    return plan, packet


def _case(operation, monkeypatch, *, tied=False, failures=(), prepare=False):
    plan, packet = _plan()
    paths = tuple(
        key.removeprefix("materializations.values.")
        for key in packet.constants
        if key.startswith("materializations.values.")
    )
    constants = dict(packet.constants)
    materialized = dict(
        plan.prepare(packet.parameters, packet.constants).values["materializations"]
    )
    leaf = torch.arange(6.0, dtype=DTYPE).reshape(3, 2).t().requires_grad_()
    shared = leaf.sin()
    if tied:
        target = constants if operation == "clear" else materialized
        for path in _TIED_PATHS:
            target[
                f"materializations.values.{path}" if operation == "clear" else path
            ] = shared
    events, enabled = [], False
    errors = {phase: RuntimeError(phase) for phase in failures}

    def observe(phase):
        if enabled:
            events.append(phase)
            if phase in errors:
                raise errors[phase]

    class Observed(Mapping):
        def __init__(self, name, values):
            self.name, self.values = name, values

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

        def keys(self):
            observe(f"{self.name}:keys")
            return self.values.keys()

        def __getitem__(self, key):
            observe(f"{self.name}:get:{key}")
            return self.values[key]

        def get(self, key, default=None):
            observe(f"{self.name}:get:{key}")
            return self.values.get(key, default)

    explicit = Observed("constants", constants)
    values = Observed("values", materialized)
    prepared = Observed("prepared", {"materializations": values})
    constant_labels = {id(value): key for key, value in constants.items()}
    value_labels = {id(value): path for path, value in materialized.items()}
    if tied:
        (constant_labels if operation == "clear" else value_labels)[id(shared)] = (
            "shared"
        )
    detach, clone = torch.Tensor.detach, torch.Tensor.clone
    zeros, ones = torch.zeros_like, torch.ones_like
    tensor_call_depth = 0

    def tensor_call(function, phase, labels, value, args, kwargs):
        nonlocal tensor_call_depth
        # A default-device TorchFunctionMode can dispatch a patched method
        # through this wrapper twice. Observe the authored outer operation.
        if not tensor_call_depth and id(value) in labels:
            observe(f"{phase}:{labels[id(value)]}")
            if phase == "clone" and enabled:
                assert args == () and kwargs == {"memory_format": torch.preserve_format}
        tensor_call_depth += 1
        try:
            return function(value, *args, **kwargs)
        finally:
            tensor_call_depth -= 1

    def detached(value, *args, **kwargs):
        return tensor_call(detach, "detach", constant_labels, value, args, kwargs)

    def cloned(value, *args, **kwargs):
        return tensor_call(clone, "clone", value_labels, value, args, kwargs)

    def zeroed(value, *args, **kwargs):
        return tensor_call(zeros, "zeros", constant_labels, value, args, kwargs)

    def oned(value, *args, **kwargs):
        return tensor_call(ones, "ones", constant_labels, value, args, kwargs)

    monkeypatch.setattr(torch.Tensor, "detach", detached)
    monkeypatch.setattr(torch.Tensor, "clone", cloned)
    monkeypatch.setattr(torch, "zeros_like", zeroed)
    monkeypatch.setattr(torch, "ones_like", oned)

    def prepare_values(parameters, actual_constants, *, use_snapshot_offsets):
        observe("prepare")
        assert parameters is packet.parameters and actual_constants is explicit
        assert use_snapshot_offsets is False
        return prepared

    monkeypatch.setattr(plan, "_prepare_values", prepare_values)
    expected = ["prepare"] if prepare else []
    if operation == "capture":
        expected += ["prepared:get:materializations"]
    expected += ["constants:keys", *(f"constants:get:{key}" for key in constants)]
    if operation == "clear":
        for path in paths:
            key = f"materializations.values.{path}"
            expected += [
                f"constants:get:{key}",
                f"detach:{constant_labels[id(constants[key])]}",
            ]
        for path in paths:
            key = f"materializations.active.{path}"
            expected += [
                f"constants:get:{key}",
                f"zeros:{constant_labels[id(constants[key])]}",
            ]
    else:
        for path in paths:
            expected += [
                f"values:get:{path}",
                f"clone:{value_labels[id(materialized[path])]}",
            ]
        for phase, group in (("ones", "active"), ("zeros", "baseline")):
            for path in paths:
                key = f"materializations.{group}.{path}"
                expected += [
                    f"constants:get:{key}",
                    f"{phase}:{constant_labels[id(constants[key])]}",
                ]

    def invoke():
        nonlocal enabled
        enabled = True
        try:
            if operation == "clear":
                return plan._clear_retained_materializations(explicit)
            return plan._retain_materializations(
                packet.parameters, explicit, prepared=None if prepare else prepared
            )
        finally:
            enabled = False

    return SimpleNamespace(
        invoke=invoke,
        expected=expected,
        events=events,
        errors=errors,
        paths=paths,
        constants=constants,
        values=materialized,
        leaf=leaf,
        shared=shared,
    )


def test_clear_detaches_in_order_without_cloning_shared_storage(monkeypatch):
    case = _case("clear", monkeypatch, tied=True)
    before = _snapshot(case.constants)
    result = case.invoke()
    assert case.events == case.expected
    assert tuple(result) == tuple(case.constants)
    for path in case.paths:
        value, active, baseline = (
            f"materializations.{kind}.{path}"
            for kind in ("values", "active", "baseline")
        )
        assert result[value] is not case.constants[value]
        assert torch._C._is_alias_of(result[value], case.constants[value])
        assert result[value].stride() == case.constants[value].stride()
        assert not result[value].requires_grad and result[value].grad_fn is None
        assert result[active] is not case.constants[active]
        assert not result[active].item()
        assert result[baseline] is case.constants[baseline]
    _assert_snapshot(case.constants, before)
    left, right = (result[f"materializations.values.{path}"] for path in _TIED_PATHS)
    assert left is not right and torch._C._is_alias_of(left, right)
    left.add_(1)
    torch.testing.assert_close(right, case.shared)


def test_capture_clones_each_row_before_activation_and_baseline_publication(
    monkeypatch,
):
    case = _case("capture", monkeypatch, tied=True)
    before = _snapshot((case.constants, case.values))
    result = case.invoke()
    assert case.events == case.expected
    assert tuple(result) == tuple(case.constants)
    for path in case.paths:
        value, active, baseline = (
            f"materializations.{kind}.{path}"
            for kind in ("values", "active", "baseline")
        )
        torch.testing.assert_close(result[value], case.values[path])
        assert not torch._C._is_alias_of(result[value], case.values[path])
        assert result[value].stride() == case.values[path].stride()
        assert result[active].item()
        assert result[active] is not case.constants[active]
        assert result[baseline] is not case.constants[baseline]
        torch.testing.assert_close(
            result[baseline], torch.zeros_like(case.constants[baseline]), rtol=0, atol=0
        )
    left, right = (result[f"materializations.values.{path}"] for path in _TIED_PATHS)
    assert left is not right and not torch._C._is_alias_of(left, right)
    actual = torch.autograd.grad(left.sum() + right.sum(), case.leaf)[0]
    torch.testing.assert_close(actual, 2 * case.leaf.cos())
    _assert_snapshot((case.constants, case.values), before)


@pytest.mark.parametrize("operation", ("clear", "capture"))
def test_empty_retained_plan_returns_constants_without_touching_dependencies(
    operation, monkeypatch
):
    plan, _ = _plan(retained=False)
    constants = object()

    class Untouched:
        def get(self, *args, **kwargs):
            pytest.fail("Empty retained plan inspected prepared values")

    def prepare(*args, **kwargs):
        pytest.fail("Empty retained plan prepared numerical workspaces")

    monkeypatch.setattr(plan, "_prepare_values", prepare)
    if operation == "clear":
        assert plan._clear_retained_materializations(constants) is constants
    else:
        assert (
            plan._retain_materializations(Untouched(), constants, prepared=Untouched())
            is constants
        )
        assert plan._retain_materializations(Untouched(), constants) is constants


@pytest.mark.parametrize(
    ("operation", "failures"),
    (
        ("clear", ("constants:get:dt", "detach:materializations.values.celsius")),
        (
            "clear",
            (
                "detach:materializations.values.rhoa_scale",
                "zeros:materializations.active.celsius",
            ),
        ),
        ("clear", ("zeros:materializations.active.rhoa_scale",)),
        ("capture", ("prepared:get:materializations", "constants:get:dt")),
        ("capture", ("clone:rhoa_scale", "ones:materializations.active.celsius")),
        (
            "capture",
            (
                "ones:materializations.active.rhoa_scale",
                "zeros:materializations.baseline.celsius",
            ),
        ),
        ("capture", ("zeros:materializations.baseline.rhoa_scale",)),
    ),
)
def test_lifecycle_failures_preserve_exact_exception_and_completed_prefix(
    operation, failures, monkeypatch
):
    case = _case(operation, monkeypatch, failures=failures)
    before = _snapshot((case.constants, case.values))
    first = min(failures, key=case.expected.index)
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.errors[first]
    assert case.events == case.expected[: case.expected.index(first) + 1]
    _assert_snapshot((case.constants, case.values), before)


def test_capture_prepares_once_before_reading_prepared_and_constant_packets(
    monkeypatch,
):
    case = _case("capture", monkeypatch, prepare=True)
    case.invoke()
    assert case.events == case.expected
