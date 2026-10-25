"""Raw-source materializers retain canonical-key cloning and lookup boundaries."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func import _connection_parameters, _netstim_parameters
from dendra.func._raw_sources import _RawSourceInventory

pytestmark = pytest.mark.cpu

_KINDS = ("connection", "netstim")


def _expected(kind):
    events = ["helpers", "clone:private-overwritten", "clone:private"]
    events += ["group:parameter"]
    if kind == "netstim":
        events.append("group:constant")
    events += [
        "items:parameter",
        "get:parameter.shared",
        "clone:shared",
        "get:parameter.other",
        "clone:shared",
    ]
    if kind == "connection":
        events.append("group:constant")
    return events + [
        "items:constant",
        "get:constant.config",
        "clone:config",
        "get:constant.meta",
        "clone:meta",
    ]


def _case(kind, monkeypatch, failures=()):
    events, active = [], False
    errors = {
        label: KeyError("shared")
        if label == "get:parameter.shared"
        else RuntimeError(label)
        for label in failures
    }

    def observe(label):
        if active:
            events.append(label)
            if label in errors:
                raise errors[label]

    class Values(dict):
        def __init__(self, label, values):
            super().__init__(values)
            self.label = label

        def __getitem__(self, key):
            observe(f"get:{self.label}.{key}")
            return super().__getitem__(key)

    class Aliases(dict):
        def __init__(self, label, values):
            super().__init__(values)
            self.label = label

        def items(self):
            observe(f"items:{self.label}")
            return super().items()

    raw = torch.arange(6.0, dtype=torch.float64).reshape(2, 3).t().requires_grad_()
    config = torch.arange(6.0, dtype=torch.float32).reshape(2, 3).t().requires_grad_()
    collision = torch.full_like(raw, 99.0, requires_grad=True)
    meta = torch.empty(2, dtype=torch.int32, device="meta")
    private_a, private_b = torch.tensor(17.0), torch.tensor(23.0)
    parameters = Values("parameter", {"shared": raw, "other": raw})
    constants = Values(
        "constant", {"shared": collision, "config": config, "meta": meta}
    )
    parameter_aliases = Aliases(
        "parameter", {"first": "shared", "again": "shared", "other": "other"}
    )
    constant_aliases = Aliases(
        "constant", {"cross": "shared", "config": "config", "meta": "meta"}
    )
    execution = object()
    slots = (("source.first", private_a), ("private", private_b))
    labels = {
        id(raw): "shared",
        id(config): "config",
        id(meta): "meta",
        id(private_a): "private-overwritten",
        id(private_b): "private",
    }
    original_clone = torch.Tensor.clone
    allocations = []

    def clone(value, *args, **kwargs):
        if id(value) in labels:
            observe(f"clone:{labels[id(value)]}")
        result = original_clone(value, *args, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    case = SimpleNamespace(
        raw=raw,
        config=config,
        collision=collision,
        meta=meta,
        events=events,
        errors=errors,
        mapping=None,
        allocations=allocations,
    )

    def helpers():
        nonlocal active
        active = True
        observe("helpers")
        return SimpleNamespace(_module_tensor_slots=lambda actual: slots)

    base = (
        _connection_parameters.ConnectionParameterPlan
        if kind == "connection"
        else _netstim_parameters.NetStimParameterPlan
    )
    inventory = _RawSourceInventory.from_paths(
        {key: key for key in parameters},
        {key: key for key in constants},
        routes={
            "weights" if kind == "connection" else None: (
                parameter_aliases,
                constant_aliases,
            )
        },
    )
    inventory = inventory.with_schema("parameter", parameters)
    inventory = inventory.with_schema("constant", constants)

    class ObservedInventory:
        def __getattr__(self, name):
            return getattr(inventory, name)

        def aliases(self, category, scope=None):
            observe(f"group:{category}")
            return Aliases(category, inventory.aliases(category, scope))

    plan = object.__new__(base)
    plan._raw_sources = ObservedInventory()
    if kind == "connection":
        monkeypatch.setattr(_connection_parameters, "_helpers", helpers)

        def invoke():
            return plan._mapping("weights", parameters, constants, execution)

    else:
        plan.validate_source = lambda: None
        plan._execution = execution
        plan._schema = (tuple(raw.shape), raw.dtype, raw.device)
        monkeypatch.setattr(_netstim_parameters, "_helpers", helpers)

        def materialize(actual, mapping, args, *, tie_weights):
            assert actual is execution and args == () and tie_weights is False
            case.mapping = mapping
            return {
                name: mapping["source.first"]
                for name in ("interval", "start", "noise", "tau")
            }

        monkeypatch.setattr(_netstim_parameters, "functional_call", materialize)

        def invoke():
            plan.materialize(parameters, constants)
            return case.mapping

    case.invoke = invoke
    return case


@pytest.mark.parametrize("kind", _KINDS)
def test_source_bindings_clone_once_per_key_across_groups_and_keep_gradients(
    kind, monkeypatch
):
    case = _case(kind, monkeypatch)
    versions = (case.raw._version, case.config._version, case.collision._version)
    mapping = case.invoke()
    assert case.events == _expected(kind)
    assert tuple(mapping) == (
        "source.first",
        "private",
        "source.again",
        "source.other",
        "source.cross",
        "source.config",
        "source.meta",
    )
    assert mapping["source.first"] is mapping["source.again"] is mapping["source.cross"]
    assert mapping["source.other"] is not mapping["source.first"]
    for key in ("source.first", "source.other"):
        torch.testing.assert_close(mapping[key], case.raw)
        assert mapping[key].stride() == case.raw.stride()
        assert not torch._C._is_alias_of(mapping[key], case.raw)
    assert not torch._C._is_alias_of(mapping["source.first"], mapping["source.other"])
    assert mapping["source.config"].dtype == torch.float32
    assert mapping["source.config"].stride() == case.config.stride()
    assert mapping["source.meta"].dtype == torch.int32
    assert mapping["source.meta"].device == torch.device("meta")
    assert mapping["source.meta"].shape == (2,)
    assert len(case.allocations) == 6
    loss = sum(
        mapping[key].sum()
        for key in (
            "source.first",
            "source.again",
            "source.cross",
            "source.other",
            "source.config",
        )
    )
    gradients = torch.autograd.grad(
        loss, (case.raw, case.config, case.collision), allow_unused=True
    )
    torch.testing.assert_close(gradients[0], torch.full_like(case.raw, 4.0))
    torch.testing.assert_close(gradients[1], torch.ones_like(case.config))
    assert gradients[2] is None
    assert (
        case.raw._version,
        case.config._version,
        case.collision._version,
    ) == versions
    torch.testing.assert_close(case.collision, torch.full_like(case.collision, 99.0))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "failure",
    (
        "clone:private-overwritten",
        "items:parameter",
        "get:parameter.shared",
        "clone:shared",
    ),
)
def test_private_clone_alias_map_key_and_public_clone_failures_stop_binding(
    kind, failure, monkeypatch
):
    case = _case(kind, monkeypatch, failures=(failure,))
    with pytest.raises((RuntimeError, KeyError)) as caught:
        case.invoke()
    assert caught.value is case.errors[failure]
    expected = _expected(kind)
    assert case.events == expected[: expected.index(failure) + 1]


@pytest.mark.parametrize("kind", _KINDS)
def test_constant_group_lookup_keeps_connection_lazy_and_netstim_eager_precedence(
    kind, monkeypatch
):
    case = _case(kind, monkeypatch, failures=("group:constant", "clone:shared"))
    first = "clone:shared" if kind == "connection" else "group:constant"
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.errors[first]
    expected = _expected(kind)
    assert case.events == expected[: expected.index(first) + 1]
