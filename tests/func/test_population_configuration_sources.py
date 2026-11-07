"""Network consumers preserve raw Population configuration-source boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.func import _network as network
from dendra.func import _network_initialization as initialization
from dendra.func._population import FunctionalPopulation
from tests.func.test_initializer_cache_config_aliases import _network

pytestmark = pytest.mark.cpu
_ENDPOINTS = ("extract", "aliases", "cache")
_KEYS = ("first", "nullable", "first", "alias", "same")
_BUFFERS = ("primary", "none", "replacement", "replacement", "replacement")


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(
        JIT=0,
        REQUIRE_GRAD=True,
        IMEM=False,
        DEVICE="cpu",
        DTYPE=torch.float64,
        CHECK_PURITY=False,
    ):
        yield


class _BoundaryReached(Exception):
    pass


def _case(monkeypatch, endpoint, *, empty=False, failure=None):
    events = []
    case = SimpleNamespace(
        events=events,
        endpoint=endpoint,
        failure=failure,
        key_error=RuntimeError("key wins over owner and module lookup"),
        later_error=RuntimeError("later module metadata must not be read"),
        missing_error=None,
        boundary=_BoundaryReached(),
        observed=[],
        bound_reads=0,
    )

    class Buffers(dict):
        def __init__(self, name, entries):
            super().__init__(entries)
            self.name = name

        def __getitem__(self, name):
            events.append(f"read:{self.name}:{name}")
            try:
                return super().__getitem__(name)
            except KeyError as error:
                case.missing_error = error
                raise

    class Owner:
        def __init__(self, name, values):
            self.name, self.values = name, Buffers(name, values)

        @property
        def _buffers(self):
            events.append(f"buffers:{self.name}")
            return self.values

    class Source:
        def __init__(self, name, value):
            self.name = name
            self.owner = Owner(
                name, {"primary": value + 10, "replacement": value, "none": None}
            )

        def __bool__(self):
            events.append(f"bool:{self.name}")
            return False

        @property
        def get_submodule(self):
            events.append(f"get-submodule:{self.name}")

            def resolve(path):
                events.append(f"resolve:{self.name}:{path}")
                return self.owner

            return resolve

    values = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (1.0, 2.0, 3.0)
    ]
    explicit, first, second = (
        Source(name, value)
        for name, value in zip(("explicit", "bound0", "bound1"), values, strict=True)
    )
    sources = (first, second)

    class Entry:
        def __init__(self, index):
            self.index = index

        @property
        def key(self):
            events.append(f"key:{self.index}")
            if failure == "key" and self.index == 1:
                raise case.key_error
            return _KEYS[self.index]

        @property
        def module_path(self):
            events.append(f"module:{self.index}")
            if failure == "key" and self.index == 1:
                raise case.later_error
            return f"module{self.index}"

        @property
        def buffer_name(self):
            events.append(f"buffer:{self.index}")
            return (
                "missing"
                if failure == "missing" and self.index == 1
                else _BUFFERS[self.index]
            )

    class Inventory:
        def __iter__(self):
            events.append("iterate")
            for index in range(0 if empty else len(_KEYS)):
                events.append(f"row:{index}")
                yield Entry(index)

    # Inherit the actual Population implementation: the original consumers use
    # their dicts, and migrated consumers dispatch through the real new method.
    class Plan(FunctionalPopulation):
        def __init__(self):
            pass

        @property
        def _parametrization_constants(self):
            events.append("inventory")
            return Inventory()

        @property
        def _population(self):
            events.append("bound")
            if endpoint != "cache" or empty:
                raise AssertionError(
                    "explicit and empty projections must not resolve the bound owner"
                )
            source = sources[case.bound_reads % 2]
            case.bound_reads += 1
            return source

        def _parameter_sources(self, population=None):
            events.append("parameters")
            assert population is (None if endpoint == "cache" else explicit)
            return {}

        def extract(self, population=None):
            events.append("population-extract")
            assert population is explicit
            return SimpleNamespace(parameters={}, constants={}, state={})

    plan = Plan()
    functional = SimpleNamespace(
        populations={"cell": plan},
        _connection_parameter_plans={},
        synapses={},
        continuous_synapses={},
        netstim=None,
    )
    source_network = SimpleNamespace(populations={"cell": explicit})

    def reached(mapping):
        events.append("boundary")
        case.observed.append(mapping)
        raise case.boundary

    if endpoint == "extract":

        class Snapshots:
            def __init__(self):
                self.calls = 0

            def tree(self, source, values):
                self.calls += 1
                if self.calls == 1:
                    events.append("parameter-snapshot")
                    return values
                return reached(source)

        monkeypatch.setattr(network, "AliasSnapshots", Snapshots)
    else:
        tensor_paths = network._tensor_paths

        def paths(tree, path=()):
            if not path and endpoint == "aliases":
                return reached(tree["populations", "cell"])
            if not path and "netstim" not in tree:
                return reached(tree["populations"]["cell"])
            return tensor_paths(tree, path)

        monkeypatch.setattr(network, "_tensor_paths", paths)

    def invoke():
        try:
            if endpoint == "extract":
                network.FunctionalNetwork._extract(functional, source_network)
            elif endpoint == "aliases":
                network.FunctionalNetwork._native_parameter_constant_aliases(
                    functional, source_network
                )
            else:
                initialization._cache_alias_plan(functional, {}, {})
        except _BoundaryReached as error:
            assert error is case.boundary
            return case.observed[-1]
        pytest.fail("consumer did not reach its preexisting post-projection boundary")

    case.invoke, case.values = invoke, values
    case.prefix = (
        ["population-extract", "parameters", "parameter-snapshot"]
        if endpoint == "extract"
        else ["parameters"]
    )
    case.expected = [*case.prefix, "inventory", "iterate"]
    for index in range(0 if empty else len(_KEYS)):
        source_name = sources[index % 2].name if endpoint == "cache" else explicit.name
        case.expected += [f"row:{index}", f"key:{index}"]
        if endpoint == "cache":
            case.expected.append("bound")
        case.expected += [
            f"get-submodule:{source_name}",
            f"module:{index}",
            f"resolve:{source_name}:module{index}",
            f"buffers:{source_name}",
            f"buffer:{index}",
            f"read:{source_name}:{'missing' if failure == 'missing' and index == 1 else _BUFFERS[index]}",
        ]
    case.expected.append("boundary")
    return case


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_consumers_keep_live_source_identity_aliases_none_and_duplicate_key_order(
    monkeypatch, endpoint
):
    case = _case(monkeypatch, endpoint)
    result = case.invoke()
    assert case.events == case.expected
    assert tuple(result) == ("first", "nullable", "alias", "same")
    assert result["nullable"] is None
    assert result["first"] is result["same"]
    if endpoint == "cache":
        assert case.bound_reads == 5
        assert result["first"] is case.values[1]
        assert result["alias"] is case.values[2]
    else:
        assert case.bound_reads == 0
        assert result["first"] is result["alias"] is case.values[0]
    assert all(value.requires_grad for value in case.values)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_empty_inventory_is_fresh_and_does_not_resolve_any_population_owner(
    monkeypatch, endpoint
):
    case = _case(monkeypatch, endpoint, empty=True)
    first = case.invoke()
    second = case.invoke()
    assert first == second == {} and first is not second
    assert case.bound_reads == 0
    assert case.events == case.expected * 2


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("failure", ("key", "missing"))
def test_source_failures_keep_key_owner_buffer_precedence_and_stop_later_rows(
    monkeypatch, endpoint, failure
):
    case = _case(monkeypatch, endpoint, failure=failure)
    with pytest.raises(RuntimeError if failure == "key" else KeyError) as caught:
        case.invoke()
    event = (
        "key:1"
        if failure == "key"
        else f"read:{'bound1' if endpoint == 'cache' else 'explicit'}:missing"
    )
    assert case.events == case.expected[: case.expected.index(event) + 1]
    assert case.observed == []
    assert caught.value is (case.key_error if failure == "key" else case.missing_error)
    if failure == "missing":
        assert caught.value.args == ("missing",)
    assert case.bound_reads == (
        (1 if failure == "key" else 2) if endpoint == "cache" else 0
    )


def test_real_alternate_network_extract_uses_its_own_parametrization_buffers():
    source, alternate = _network("network"), _network("network")
    functional, original = dn.func.make_functional_network(source)
    with torch.no_grad():
        alternate.left.config.saved_binding.fill_(0.9)
    actual = functional.extract(alternate)
    key = "parametrizations.parametrizations.cm.0.offset"
    value = actual.constants["populations"]["right"][key]
    torch.testing.assert_close(value, torch.full_like(value, 0.9))
    torch.testing.assert_close(
        original.constants["populations"]["right"][key], torch.full_like(value, 0.25)
    )
    native = alternate.right.parametrizations.cm[0].offset
    assert value is not native and not torch._C._is_alias_of(value, native)
