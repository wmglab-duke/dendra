"""Prepared constants preserve ordered ownership across their slot providers."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func import (
    _custom_material_processes,
    _initialization_sources,
    _mechanism_delays,
    _mechanism_injections,
    _population,
    _tables,
    _torch_parametrizations,
)

pytestmark = pytest.mark.cpu


def _inventory(monkeypatch, *, parameters=(), table_buffers=None):
    events, records, slots = [], [], {}
    owners = (("", object()), ("owner.second", object()))
    paths = {id(owner): path for path, owner in owners}
    table = SimpleNamespace(_buffers=table_buffers or {})
    population = SimpleNamespace(
        parameters=lambda: iter(parameters),
        named_modules=lambda: iter((("table", table),)),
    )
    monkeypatch.setattr(_population, "_parameterized_module_paths", lambda model: ())
    monkeypatch.setattr(
        _population, "_parameterized_owner_paths", lambda model: iter(owners)
    )
    monkeypatch.setattr(
        _tables, "table_owners", lambda model: (("", model, "table", table),)
    )
    monkeypatch.setattr(
        _initialization_sources, "initialization_source_roots", lambda model: ()
    )
    monkeypatch.setattr(_torch_parametrizations, "buffer_slots", lambda model: ())

    def provider(kind):
        def collect(owner):
            path = paths[id(owner)]
            events.append(("call", kind, path))

            def values():
                for value in slots.get((kind, path), ()):
                    if isinstance(value, Exception):
                        events.append(("raise", kind, path))
                        raise value
                    events.append(("yield", kind, path))
                    yield value

            return values()

        return collect

    monkeypatch.setattr(
        _mechanism_delays, "mechanism_delay_constant_slots", provider("delay")
    )
    monkeypatch.setattr(
        _mechanism_injections,
        "mechanism_injection_constant_slots",
        provider("injection"),
    )
    monkeypatch.setattr(
        _custom_material_processes,
        "custom_material_constant_slots",
        provider("material"),
    )
    prepared_buffer = _population._PreparedBuffer

    def record(*args, **kwargs):
        result = prepared_buffer(*args, **kwargs)
        records.append(result)
        return result

    monkeypatch.setattr(_population, "_PreparedBuffer", record)
    return SimpleNamespace(
        events=events,
        records=records,
        slots=slots,
        collect=lambda: _population._prepared_buffer_layout(population),
    )


def test_constants_keep_shared_identity_inventory_and_first_relative_paths(monkeypatch):
    table = torch.ones(2, dtype=torch.float64)
    delay = torch.ones(3, dtype=torch.int64)
    parameter = torch.nn.Parameter(torch.tensor(7.0))
    local = torch.arange(6.0, dtype=torch.float32).reshape(2, 3).t()
    view = local.t()
    assert view is not local and torch._C._is_alias_of(view, local)
    nested = torch.ones(4, dtype=torch.bool)
    custom = torch.empty((2, 0), dtype=torch.int32, device="meta")
    second = torch.tensor(4.0, dtype=torch.float64)
    case = _inventory(
        monkeypatch,
        parameters=(parameter,),
        table_buffers={
            "config": table,
            "alias": table,
            "none": None,
            "parameter": parameter,
        },
    )
    case.slots.update(
        {
            ("delay", ""): (("steps", delay), ("raw_parameter", parameter)),
            ("injection", ""): (
                ("old_table", table),
                ("old_delay", delay),
                ("old_parameter", parameter),
                ("local", local),
                ("nested.flag", nested),
                ("same_local", local),
            ),
            ("material", ""): (("config.deep.value", custom), ("same_local", local)),
            ("delay", "owner.second"): (("old_local", local),),
            ("injection", "owner.second"): (
                ("old_custom", custom),
                ("local.view", view),
            ),
            ("material", "owner.second"): (
                ("local.value", second),
                ("old_table", table),
            ),
        }
    )
    result = case.collect()
    assert result == tuple(case.records)
    assert [
        (
            entry.key,
            entry.module_path,
            entry.buffer_name,
            entry.shape,
            entry.dtype,
            entry.device,
            entry.kind,
        )
        for entry in result
    ] == [
        (
            "table.config",
            "table",
            "config",
            (2,),
            torch.float64,
            table.device,
            "constant",
        ),
        (".steps", "", "steps", (3,), torch.int64, delay.device, "constant"),
        (
            ".raw_parameter",
            "",
            "raw_parameter",
            (),
            parameter.dtype,
            parameter.device,
            "constant",
        ),
        (".local", "", "local", (3, 2), torch.float32, local.device, "constant"),
        (
            ".nested.flag",
            ".nested",
            "flag",
            (4,),
            torch.bool,
            nested.device,
            "constant",
        ),
        (
            ".config.deep.value",
            ".config.deep",
            "value",
            (2, 0),
            torch.int32,
            custom.device,
            "constant",
        ),
        (
            "owner.second.local.view",
            "owner.second.local",
            "view",
            (2, 3),
            torch.float32,
            view.device,
            "constant",
        ),
        (
            "owner.second.local.value",
            "owner.second.local",
            "value",
            (),
            torch.float64,
            second.device,
            "constant",
        ),
    ]
    assert [event for event in case.events if event[0] == "call"] == [
        ("call", provider, path)
        for path in ("", "owner.second")
        for provider in ("delay", "injection", "material")
    ]
    assert local._version == parameter._version == 0


@pytest.mark.parametrize("provider", ("injection", "material"))
def test_provider_failure_after_a_record_precedes_later_provider_and_owner_calls(
    provider, monkeypatch
):
    case = _inventory(monkeypatch)
    error = RuntimeError("provider stopped")
    case.slots[(provider, "")] = (("first", torch.tensor(1.0)), error)
    with pytest.raises(RuntimeError) as caught:
        case.collect()
    assert caught.value is error
    expected = [("call", "delay", ""), ("call", "injection", "")]
    if provider == "material":
        expected.append(("call", "material", ""))
    assert case.events == expected + [("yield", provider, ""), ("raise", provider, "")]
    assert [entry.key for entry in case.records] == [".first"]


def test_parameters_and_seen_values_are_excluded_before_path_or_metadata_reads(
    monkeypatch,
):
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError(f"excluded value accessed {name}")

    parameter = Poison()
    bad_path = Poison()
    shared = torch.ones(2)
    case = _inventory(
        monkeypatch, parameters=(parameter,), table_buffers={"first": shared}
    )
    for path in ("", "owner.second"):
        for provider in ("injection", "material"):
            case.slots[(provider, path)] = ((bad_path, parameter), (bad_path, shared))
    result = case.collect()
    assert [entry.key for entry in result] == ["table.first"]
    assert len(case.records) == 1


@pytest.mark.parametrize("provider", ("injection", "material"))
@pytest.mark.parametrize("failure", ("path", "key", "shape", "dtype", "device"))
def test_record_field_failures_preserve_partial_records_and_access_order(
    provider, failure, monkeypatch
):
    case = _inventory(monkeypatch)
    phases = ("path", "key", "shape", "dtype", "device")
    index = phases.index(failure)
    errors = {name: RuntimeError(name) for name in phases[index:]}

    def phase(name):
        case.events.append(name)
        if name in errors:
            raise errors[name]

    class RelativePath(str):
        def rpartition(self, separator):
            phase("path")
            return super().rpartition(separator)

        def __format__(self, spec):
            phase("key")
            return super().__format__(spec)

    class Buffer:
        @property
        def shape(self):
            phase("shape")
            return (2, 3)

        @property
        def dtype(self):
            phase("dtype")
            return torch.float64

        @property
        def device(self):
            phase("device")
            return torch.device("cpu")

    case.slots[(provider, "")] = (
        ("first", torch.tensor(1.0)),
        (RelativePath("nested.value"), Buffer()),
    )
    with pytest.raises(RuntimeError) as caught:
        case.collect()
    assert caught.value is errors[failure]
    expected = [("call", "delay", ""), ("call", "injection", "")]
    if provider == "material":
        expected.append(("call", "material", ""))
    assert case.events == expected + [("yield", provider, "")] * 2 + list(
        phases[: index + 1]
    )
    assert [entry.key for entry in case.records] == [".first"]
