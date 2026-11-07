"""Population and Network guards preserve registered generator ownership."""

import builtins
from types import SimpleNamespace

import pytest
import torch

from dendra.func import _network_rng, _stimulation_rng
from dendra.func._types import FunctionalizationError
from dendra.models.rng import RNGModule

pytestmark = pytest.mark.cpu

_ENDPOINTS = ("population", "network")
_MESSAGE = (
    "Poisson generator aliases an untracked registered RNGModule; "
    "use a declared RAND/NOISE/SDE stream or a separate generator"
)


def _case(endpoint, monkeypatch):
    case = SimpleNamespace(
        events=[],
        failures={},
        owners=[],
        declared={},
        requested=[],
        handles={},
        labels={},
        generator_labels={},
        frames=[],
        identities=[],
        scanning=False,
    )
    global_frame = torch.get_rng_state().clone()

    def phase(name):
        case.events.append(name)
        if name in case.failures:
            raise case.failures[name]

    class Devices(dict):
        def __init__(self, label, generators):
            # Ownership uses handle identity; these device entries use CPU handles.
            super().__init__(zip(("cuda:0", "cuda:1"), generators, strict=True))
            self.label = label

        def items(self):
            phase(f"items:{self.label}")
            return super().items()

        def values(self):
            phase(f"values:{self.label}")

            def expand():
                for index, generator in enumerate(dict.values(self)):
                    phase(f"expand:{self.label}.{index}")
                    yield generator

            return expand()

    class Owner(RNGModule):
        def __getattribute__(self, name):
            label = object.__getattribute__(self, "__dict__").get("_contract_label")
            if label is not None and name in ("_cpu_gen", "_device_gens"):
                field = "cpu" if name == "_cpu_gen" else "devices"
                phase(f"{'scan' if case.scanning else 'source'}:{label}.{field}")
            return super().__getattribute__(name)

    def make_owner(label):
        owner = Owner(811, shape_p=(), shape_f=())
        cpu = object.__getattribute__(owner, "_cpu_gen")
        devices = [torch.Generator().manual_seed(seed) for seed in (821, 823)]
        mapping = Devices(label, devices)
        object.__setattr__(owner, "_device_gens", mapping)
        object.__setattr__(owner, "_contract_label", label)
        case.labels[builtins.id(owner)] = label
        case.identities.append((owner, cpu, mapping, tuple(devices)))
        for field, generator in zip(
            ("cpu", "device0", "device1"), (cpu, *devices), strict=True
        ):
            name = f"{label}.{field}"
            case.handles[name] = generator
            case.generator_labels[builtins.id(generator)] = name
            case.frames.append((generator, generator.get_state().clone()))
        return owner

    def identity(value):
        result = builtins.id(value)
        if case.scanning and result in case.generator_labels:
            phase(f"member:{case.generator_labels[result]}")
        return result

    monkeypatch.setattr(_stimulation_rng, "id", identity, raising=False)
    monkeypatch.setattr(_network_rng, "id", identity, raising=False)

    def modules():
        phase("modules")
        case.scanning = True
        for owner in case.owners:
            phase(f"visit:{case.labels[builtins.id(owner)]}")
            yield owner

    population = object()
    root = SimpleNamespace(modules=modules, populations={"cell": population})

    class Bindings(dict):
        def values(self):
            phase("declarations")
            for label, owner in self.items():
                phase(f"declared:{label}")
                yield owner

    class Requests:
        def __iter__(self):
            phase("requests")
            for generator in case.requested:
                phase(f"request:{case.generator_labels[builtins.id(generator)]}")
                yield generator

    class Streams(dict):
        def items(self):
            phase("declarations")
            return super().items()

    class Sources(dict):
        def __getitem__(self, label):
            phase(f"source:{label}")
            return super().__getitem__(label)

    class Waveform:
        def __init__(self, generator):
            self._handle = generator

        @property
        def generator(self):
            phase(f"request:{case.generator_labels[builtins.id(self._handle)]}")
            return self._handle

    def owners(actual):
        assert actual is population
        phase("owners")
        return Sources(
            {
                label: SimpleNamespace(rng=owner)
                for label, owner in case.declared.items()
            }
        )

    def resolve(actual):
        assert actual is population
        phase("requests")
        return {
            index: Waveform(generator) for index, generator in enumerate(case.requested)
        }

    def invoke():
        case.scanning = False
        if endpoint == "population":
            return _stimulation_rng._validate_tracked_generators(
                root, Bindings(case.declared), Requests()
            )
        plan = object.__new__(_network_rng.NetworkStimulationRNGPlan)
        plan.random = SimpleNamespace(
            streams=Streams(
                {
                    f"model_{index}": ("cell", label, "rng")
                    for index, label in enumerate(case.declared)
                }
            ),
            _owners=owners,
        )
        entries = [
            SimpleNamespace(key=index, stream=index)
            for index in range(len(case.requested))
        ]
        plan.stimulation = SimpleNamespace(
            plans={"cell": SimpleNamespace(_resolved_sources=resolve, entries=entries)},
            local={
                "cell": {index: f"poisson_{index}" for index in range(len(entries))}
            },
        )
        return plan._current_borrowed(root)

    def unchanged():
        for owner, cpu, devices, handles in case.identities:
            assert object.__getattribute__(owner, "_cpu_gen") is cpu
            assert object.__getattribute__(owner, "_device_gens") is devices
            assert all(
                actual is expected
                for actual, expected in zip(dict.values(devices), handles, strict=True)
            )
        for generator, frame in case.frames:
            assert torch.equal(generator.get_state(), frame)
        assert torch.equal(torch.get_rng_state(), global_frame)

    case.make_owner, case.invoke, case.unchanged = make_owner, invoke, unchanged
    return case


def _prefix(endpoint, *, declared=(), requested=()):
    result = ["declarations"]
    for label in declared:
        result += (
            [f"declared:{label}"]
            if endpoint == "population"
            else [
                "owners",
                f"source:{label}",
                f"source:{label}.cpu",
                f"source:{label}.devices",
                f"items:{label}",
            ]
        )
    return result + [
        "requests",
        *(f"request:{label}" for label in requested),
        "modules",
    ]


def _scan(label):
    return [
        f"visit:{label}",
        f"scan:{label}.cpu",
        f"scan:{label}.devices",
        f"values:{label}",
        f"expand:{label}.0",
        f"expand:{label}.1",
        f"member:{label}.cpu",
        f"member:{label}.device0",
        f"member:{label}.device1",
    ]


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_inputs_precede_tree_traversal_and_skipped_owners_leave_aliases_untouched(
    endpoint, monkeypatch
):
    case = _case(endpoint, monkeypatch)
    declared, ordinary = case.make_owner("tracked"), case.make_owner("ordinary")

    class NonRNG:
        def __getattribute__(self, name):
            if name in ("_cpu_gen", "_device_gens"):
                raise AssertionError("non-RNG internals inspected")
            return super().__getattribute__(name)

    non_rng = NonRNG()
    case.labels[builtins.id(non_rng)] = "non-rng"
    case.owners = [non_rng, declared, ordinary, ordinary]
    case.declared = {"tracked": declared}
    case.requested = [case.handles["tracked.cpu"]] * 2
    case.failures["scan:tracked.cpu"] = RuntimeError("declared internals inspected")
    result = case.invoke()
    assert result == (
        None
        if endpoint == "population"
        else {
            "poisson_0": ("model_0", "cpu"),
            "poisson_1": ("model_0", "cpu"),
        }
    )
    assert case.events == _prefix(
        endpoint, declared=("tracked",), requested=("tracked.cpu",) * 2
    ) + [
        "visit:non-rng",
        "visit:tracked",
        *_scan("ordinary"),
        *_scan("ordinary"),
    ]
    case.unchanged()


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("requested", ("cpu", "device0", "device1", "missing", "empty"))
def test_eager_device_reads_cpu_first_membership_and_first_owner_failure(
    endpoint, requested, monkeypatch
):
    case = _case(endpoint, monkeypatch)
    target = case.make_owner("target")
    independent = case.make_owner("independent")
    case.owners = [target]
    if requested != "empty":
        label = "independent.cpu" if requested == "missing" else f"target.{requested}"
        case.requested = [case.handles[label]]
    else:
        label = None
    hit = requested not in ("missing", "empty")
    if hit:
        case.owners.append(independent)
        case.failures["scan:independent.cpu"] = RuntimeError("later owner inspected")
        with pytest.raises(FunctionalizationError) as caught:
            case.invoke()
        assert str(caught.value) == _MESSAGE
        scanned = _scan("target")[: _scan("target").index(f"member:{label}") + 1]
    else:
        assert case.invoke() == (None if endpoint == "population" else {})
        scanned = _scan("target")
    assert (
        case.events
        == _prefix(endpoint, requested=() if label is None else (label,)) + scanned
    )
    case.unchanged()


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize(
    "failure",
    ("scan:target.cpu", "scan:target.devices", "values:target", "expand:target.0"),
)
def test_eager_read_failures_keep_identity_and_precede_even_a_cpu_alias(
    endpoint, failure, monkeypatch
):
    case = _case(endpoint, monkeypatch)
    case.owners = [case.make_owner("target")]
    case.requested = [case.handles["target.cpu"]]
    scan = _scan("target")
    index = scan.index(failure)
    case.failures = {name: RuntimeError(name) for name in scan[index:]}
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.failures[failure]
    assert (
        case.events == _prefix(endpoint, requested=("target.cpu",)) + scan[: index + 1]
    )
    case.unchanged()


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("failure", ("declarations", "requests"))
def test_input_failures_precede_registered_module_traversal(
    endpoint, failure, monkeypatch
):
    case = _case(endpoint, monkeypatch)
    tracked = case.make_owner("tracked")
    case.declared = {"tracked": tracked}
    case.requested = [case.handles["tracked.cpu"]]
    case.owners = [tracked]
    case.failures = {name: RuntimeError(name) for name in (failure, "modules")}
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.failures[failure]
    expected = _prefix(endpoint, declared=("tracked",), requested=("tracked.cpu",))
    assert case.events == expected[: expected.index(failure) + 1]
    case.unchanged()
