"""Cold cache endpoints preserve element spans and distinct byte conventions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils import parametrize
from torch.overrides import TorchFunctionMode

import dendra as dn
from dendra.func import _network_initialization as network
from dendra.func import _population_cache as population
from dendra.models.networks._population_cache import replay_population_cache_storage
from tests.func.test_retained_frame_lifecycle import DTYPE, _plan

pytestmark = pytest.mark.cpu


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


def _entry(name, value):
    return name, value, population._cache_storage_frame(value)


@pytest.mark.parametrize("kind", ("scalar", "adjacent", "strided"))
def test_replay_element_boundaries_keep_scalar_shapes_and_native_last_copy(kind):
    base = torch.arange(8.0, dtype=DTYPE)
    if kind == "adjacent":
        assert (
            population.cache_storage_replays(
                [_entry("left", base[:2])], [_entry("right", base[2:4])]
            )
            == ()
        )
        return
    if kind == "scalar":
        writes = [_entry("scalar", base[2])]
        outputs = [_entry("vector", base[2:3])]
        values = {"scalar": base.new_tensor(9)}
        expected = (base.new_tensor([9]),)
    else:
        writes = [
            _entry("old", base[::2]),
            _entry("tip", base[6]),
            _entry("final", base[::2]),
        ]
        outputs = [_entry("vector", base[::2]), _entry("scalar", base[6])]
        values = {
            "old": base.new_tensor([1, 2, 3, 4]),
            "tip": base.new_tensor(90),
            "final": base.new_tensor([10, 20, 30, 40]),
        }
        native = base.clone()
        native[::2].copy_(values["old"])
        native[6].copy_(values["tip"])
        native[::2].copy_(values["final"])
        expected = native[::2], native[6]
    recipes = population.cache_storage_replays(writes, outputs)
    assert len(recipes) == 1
    recipe = recipes[0]
    assert recipe.write_paths == (("scalar",) if kind == "scalar" else ("tip", "final"))
    assert recipe.size == (1 if kind == "scalar" else 4)
    actual = replay_population_cache_storage(
        tuple(values[name] for name in recipe.write_paths),
        recipe.write_indices,
        recipe.output_indices,
        recipe.output_shapes,
        size=recipe.size,
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)


def test_replay_guards_precede_strict_layout_iteration():
    source = torch.ones(2, dtype=DTYPE)
    frame = population._cache_storage_frame(source)
    malformed = (*frame[:5], (2,), ())
    different_dtype = (*frame[:2], torch.float32, *frame[3:])
    with pytest.raises(
        dn.func.FunctionalizationError, match="sharing storage with different dtypes"
    ):
        population.cache_storage_replays(
            [("write", source, malformed)], [("read", source, different_dtype)]
        )
    assert (
        population.cache_storage_replays(
            [("write", source, malformed)], [("read", source, malformed)]
        )
        == ()
    )
    assert population.cache_storage_replays([("write", source, malformed)], []) == ()
    assert (
        population.cache_storage_replays(
            [("write", source, ("owned_buffer", 1, 2))], [("read", source, malformed)]
        )
        == ()
    )
    assert population.cache_storage_replays([], []) == ()


def _alias_case(writer, reader):
    writer_plan = SimpleNamespace(
        write_keys=("saved",),
        storage_frames={"saved": population._cache_storage_frame(writer)},
        sources={"saved": writer},
        _parameter_sources=lambda: {},
        _parametrization_constants=(),
    )
    reader_plan = SimpleNamespace(
        parameters={"value": "reader"},
        constants={},
        states=(),
        storage_frames={"reader": population._cache_storage_frame(reader)},
        sources={"reader": reader},
        _parameter_sources=lambda: {"value": reader},
        _parametrization_constants=(),
    )
    for plan in (writer_plan, reader_plan):
        plan._parametrization_constant_sources = (
            dn.func.FunctionalPopulation._parametrization_constant_sources.__get__(plan)
        )
    functional = SimpleNamespace(
        populations={"writer": writer_plan, "reader": reader_plan},
        synapses={},
        continuous_synapses={},
        netstim=None,
        _parameter_aliases=(),
        _parameter_constant_aliases=(),
    )
    return functional, {"writer": writer_plan}, {"reader": reader_plan}


@pytest.mark.parametrize("byte", (3, 4), ids=("last-byte", "adjacent-byte"))
def test_network_alias_plan_uses_inclusive_byte_edges(byte):
    storage = torch.arange(8, dtype=torch.uint8)
    case = _alias_case(storage[:4].view(torch.float32), storage[byte])
    if byte == 3:
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="sharing storage with different dtypes",
        ):
            network._cache_alias_plan(*case)
    else:
        assert network._cache_alias_plan(*case) == ((), ())


class _Dtype:
    def __init__(self, events, error=None):
        self.events, self.error = events, error

    @property
    def itemsize(self):
        self.events.append("dtype.itemsize")
        if self.error is not None:
            raise self.error
        return 8


def test_network_alias_dtype_size_is_read_before_strict_zip_errors():
    storage = torch.arange(3.0, dtype=DTYPE)
    functional, writers, readers = _alias_case(storage[:2], storage[1])
    events, error = [], RuntimeError("dtype metadata")
    dtype = _Dtype(events, error)
    frame = readers["reader"].storage_frames["reader"]
    readers["reader"].storage_frames["reader"] = (
        *frame[:2],
        dtype,
        *frame[3:5],
        (2,),
        (),
    )
    with pytest.raises(RuntimeError) as caught:
        network._cache_alias_plan(functional, writers, readers)
    assert caught.value is error and events == ["dtype.itemsize"]
    dtype.error = None
    events.clear()
    with pytest.raises(ValueError, match=r"zip\(\).*shorter"):
        network._cache_alias_plan(functional, writers, readers)
    assert events == ["dtype.itemsize"]


class _Offset(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("offset", value, persistent=False)

    def forward(self, value):
        return value + self.offset


def _nonpersistent_case(writer, reader):
    owner = torch.nn.Module()
    owner.register_buffer("weight", torch.zeros_like(reader))
    parametrize.register_parametrization(owner, "weight", _Offset(reader))
    owner.mech = torch.nn.Module()
    owner.mech.material_processes = torch.nn.ModuleDict()
    reader_plan = SimpleNamespace(
        _population=owner,
        _owned_cache_plan=None,
        _parametrization_constants=(
            SimpleNamespace(
                key="config.offset",
                module_path="parametrizations.weight.0",
                buffer_name="offset",
            ),
        ),
    )
    writer_plan = SimpleNamespace(
        write_keys=("saved",),
        storage_frames={"saved": population._cache_storage_frame(writer)},
    )
    return SimpleNamespace(
        _cache_enabled=True,
        _population_cache_plans={"writer": writer_plan},
        _cache_alias_bindings=(),
        populations={"reader": reader_plan},
    )


@pytest.mark.parametrize("byte", (3, 4), ids=("last-byte", "adjacent-byte"))
def test_nonpersistent_network_reader_uses_half_open_byte_edges(byte):
    storage = torch.arange(8, dtype=torch.uint8)
    functional = _nonpersistent_case(storage[:4].view(torch.float32), storage[byte])
    failures = network.nonpersistent_material_cache_failures(functional)
    assert failures == (
        ()
        if byte == 4
        else (
            "Population 'reader' nonpersistent torch parametrization config 'config.offset' shares storage with another Population's cached tensor. This cache restoration is currently unsupported; use independent config buffers or register the shared config as persistent on every reader.",
        )
    )


def test_nonpersistent_network_strict_zip_precedes_dtype_metadata(monkeypatch):
    storage = torch.arange(3.0, dtype=DTYPE)
    reader = storage[1]
    functional = _nonpersistent_case(storage[:2], reader)
    events = []
    dtype = _Dtype(events, RuntimeError("must not read dtype before shape"))
    frame = population._cache_storage_frame(reader)
    original = population._cache_storage_frame

    def malformed(value, **kwargs):
        return (
            (*frame[:2], dtype, *frame[3:5], (2,), ())
            if value is reader
            else original(value, **kwargs)
        )

    monkeypatch.setattr(population, "_cache_storage_frame", malformed)
    with pytest.raises(ValueError, match=r"zip\(\).*shorter"):
        network.nonpersistent_material_cache_failures(functional)
    assert events == []


def test_nonpersistent_empty_and_owned_frames_skip_storage_metadata(monkeypatch):
    assert (
        network.nonpersistent_material_cache_failures(
            SimpleNamespace(_cache_enabled=True, _population_cache_plans={})
        )
        == ()
    )
    assert (
        network.nonpersistent_material_cache_failures(
            SimpleNamespace(_cache_enabled=False, populations={})
        )
        == ()
    )
    storage = torch.arange(3.0, dtype=DTYPE)
    reader = storage[1]
    functional = _nonpersistent_case(storage[:2], reader)
    original = population._cache_storage_frame
    monkeypatch.setattr(
        population,
        "_cache_storage_frame",
        lambda value, **kwargs: ("owned_buffer", 1, 2)
        if value is reader
        else original(value, **kwargs),
    )
    assert network.nonpersistent_material_cache_failures(functional) == ()


def _population_case(monkeypatch, writer, reader):
    plan, _packet = _plan(retained=False)
    source = plan._population
    source.config = torch.nn.Module()
    source.config.register_buffer("saved", writer)
    source.config.register_buffer("reader", reader, persistent=False)
    monkeypatch.setattr(
        plan,
        "_parametrization_constants",
        (
            SimpleNamespace(
                key="config.reader", module_path="config", buffer_name="reader"
            ),
        ),
    )
    return plan


@pytest.mark.parametrize("byte", (3, 4), ids=("last-byte", "adjacent-byte"))
def test_population_constant_reader_uses_half_open_byte_edges(monkeypatch, byte):
    storage = torch.arange(8, dtype=torch.uint8)
    plan = _population_case(monkeypatch, storage[:4].view(torch.float32), storage[byte])
    if byte == 3:
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="Nonpersistent config views partially overlapping cached tensor storage",
        ):
            population.PopulationCachePlan(plan, bindings_only=True)
    else:
        cached = population.PopulationCachePlan(plan, bindings_only=True)
        assert "config.reader" not in cached.constants
        assert cached.storage_frames["config.reader"][4] == 4


class _ElementSizes(TorchFunctionMode):
    def __init__(self, labels, events, error=None):
        self.labels, self.events, self.error = labels, events, error

    def __torch_function__(self, function, types, args=(), kwargs=None):
        if function is torch.Tensor.element_size and id(args[0]) in self.labels:
            self.events.append(self.labels[id(args[0])])
            if self.error is not None:
                raise self.error
        return function(*args, **({} if kwargs is None else kwargs))


@pytest.mark.parametrize("broken", ("reader", "writer"))
def test_population_constant_span_reads_element_size_only_after_its_layout(
    monkeypatch, broken
):
    storage = torch.arange(3.0, dtype=DTYPE)
    writer, reader = storage[:2], storage[1]
    plan = _population_case(monkeypatch, writer, reader)
    target = reader if broken == "reader" else writer
    frame = population._cache_storage_frame(target)
    original = population._cache_storage_frame
    monkeypatch.setattr(
        population,
        "_cache_storage_frame",
        lambda value, **kwargs: (*frame[:5], (2,), ())
        if value is target
        else original(value, **kwargs),
    )
    events = []
    labels = {id(reader): "reader.element_size", id(writer): "writer.element_size"}
    with (
        _ElementSizes(labels, events),
        pytest.raises(ValueError, match=r"zip\(\).*shorter"),
    ):
        population.PopulationCachePlan(plan, bindings_only=True)
    assert events == (
        [] if broken == "reader" else ["reader.element_size", "reader.element_size"]
    )
    if broken == "writer":
        error = RuntimeError("reader metadata wins over later malformed writer")
        events.clear()
        with (
            _ElementSizes(labels, events, error),
            pytest.raises(RuntimeError) as caught,
        ):
            population.PopulationCachePlan(plan, bindings_only=True)
        assert caught.value is error and events == ["reader.element_size"]
