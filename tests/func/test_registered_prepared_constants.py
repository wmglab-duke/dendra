"""Registered constant collection retains independent discovery and ownership."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func import (
    _initialization_sources,
    _population,
    _tables,
    _torch_parametrizations,
)

pytestmark = pytest.mark.cpu


def _case(monkeypatch, *, parameters=()):
    case = SimpleNamespace(
        events=[],
        records=[],
        initial=[],
        torch=[],
        runtime=[],
        roots=True,
        walks=[[], [], []],
        walk=0,
        failures={},
    )
    source = object()

    def phase(name):
        case.events.append(name)
        if name in case.failures:
            raise case.failures[name]

    def modules():
        case.walk += 1
        number = case.walk
        phase(f"walk:{number}")

        def visit():
            for entry in case.walks[number - 1]:
                if isinstance(entry, Exception):
                    phase(f"fail:walk:{number}")
                    raise entry
                path, module = entry
                phase(f"visit:{number}:{path}")
                yield path, module
            phase(f"end:{number}")

        return visit()

    population = SimpleNamespace(
        parameters=lambda: iter(parameters), named_modules=modules
    )
    table = SimpleNamespace(_buffers={})
    monkeypatch.setattr(_population, "_parameterized_module_paths", lambda model: ())
    monkeypatch.setattr(_population, "_parameterized_owner_paths", lambda model: ())
    monkeypatch.setattr(
        _tables, "table_owners", lambda model: (("", model, "table", table),)
    )

    def entries(kind):
        phase(f"{kind}-slots")
        for entry in getattr(case, kind):
            if isinstance(entry, Exception):
                phase(f"fail:{kind}")
                raise entry
            phase(f"{kind}:{entry[0]}")
            yield entry

    def roots(model):
        phase("roots")
        return (("source", source),) if case.roots else ()

    def runtime(model):
        phase("runtime")
        return {id(value) for value in case.runtime}

    monkeypatch.setattr(_initialization_sources, "initialization_source_roots", roots)
    monkeypatch.setattr(
        _initialization_sources, "initialization_source_runtime_buffer_ids", runtime
    )
    monkeypatch.setattr(
        _initialization_sources,
        "initialization_source_constant_slots",
        lambda owner: entries("initial"),
    )
    monkeypatch.setattr(
        _torch_parametrizations, "buffer_slots", lambda model: entries("torch")
    )
    prepared_buffer = _population._PreparedBuffer

    def record(*args, **kwargs):
        result = prepared_buffer(*args, **kwargs)
        phase(f"record:{result.key}")
        case.records.append(result)
        return result

    monkeypatch.setattr(_population, "_PreparedBuffer", record)
    case.table, case.phase = table, phase
    case.collect = lambda: _population._prepared_buffer_layout(population)
    return case


class _Poison:
    def __getattribute__(self, name):
        raise AssertionError(f"excluded buffer metadata read: {name}")


def _module(**buffers):
    return SimpleNamespace(_buffers=buffers)


def test_discovery_and_fresh_traversals_preserve_cross_phase_identity_and_metadata(
    monkeypatch,
):
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    initial = torch.arange(6.0, dtype=torch.float64).reshape(2, 3).t().requires_grad_()
    view = initial.t()
    assert view is not initial and torch._C._is_alias_of(initial, view)
    prior = torch.ones(2, dtype=torch.float32)
    meta = torch.empty((2, 0), dtype=torch.int32, device="meta")
    late = torch.tensor(3.0, dtype=torch.float32)
    runtime, orphan = _Poison(), _Poison()
    case = _case(monkeypatch, parameters=(parameter,))
    root = _module(
        runtime=runtime, parameter=parameter, first=initial, prior=prior, none=None
    )
    child = _module(alias=initial, view=view, meta=meta)
    case.table._buffers = {"config": prior}
    case.walks = [
        [("table", case.table)],
        [("", root), ("initial.child", child)],
        [("new.child", child), ("", root), ("late", _module(value=late))],
    ]
    case.initial = [
        ("runtime", runtime),
        ("parameter", parameter),
        ("first", initial),
        ("meta", meta),
        ("prior", prior),
        ("orphan", orphan),
    ]
    case.runtime = [runtime]
    case.torch = [("first", initial), ("view", view), ("meta", meta), ("late", late)]
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
            prior.dtype,
            prior.device,
            "constant",
        ),
        (".first", "", "first", (3, 2), initial.dtype, initial.device, "constant"),
        (
            "initial.child.meta",
            "initial.child",
            "meta",
            (2, 0),
            meta.dtype,
            meta.device,
            "constant",
        ),
        (
            "new.child.view",
            "new.child",
            "view",
            (2, 3),
            view.dtype,
            view.device,
            "constant",
        ),
        ("late.value", "late", "value", (), late.dtype, late.device, "constant"),
    ]
    landmarks = [
        event
        for event in case.events
        if event.startswith(("walk:", "end:", "record:"))
        or event in ("roots", "runtime", "initial-slots", "torch-slots")
    ]
    assert landmarks == [
        "walk:1",
        "record:table.config",
        "end:1",
        "roots",
        "runtime",
        "initial-slots",
        "walk:2",
        "record:.first",
        "record:initial.child.meta",
        "end:2",
        "torch-slots",
        "walk:3",
        "record:new.child.view",
        "record:late.value",
        "end:3",
    ]
    assert case.events.index("initial:orphan") < case.events.index("walk:2")
    assert case.events.index("torch:late") < case.events.index("walk:3")
    assert initial._version == parameter._version == 0


def test_filtering_skips_none_noneligible_seen_and_parameters_before_metadata(
    monkeypatch,
):
    parameter, noneligible = _Poison(), _Poison()
    prior, accepted = torch.ones(2), torch.tensor(4.0)
    case = _case(monkeypatch, parameters=(parameter,))
    root = _module(
        none=None,
        excluded=noneligible,
        prior=prior,
        parameter=parameter,
        accepted=accepted,
        alias=accepted,
    )
    case.table._buffers = {"prior": prior}
    case.walks = [[("table", case.table)], [("", root)], [("", root)]]
    case.initial = case.torch = [
        ("none", None),
        ("prior", prior),
        ("parameter", parameter),
        ("accepted", accepted),
    ]
    result = case.collect()
    assert [entry.key for entry in result] == ["table.prior", ".accepted"]


@pytest.mark.parametrize("roots", (False, True))
def test_empty_eligible_sets_keep_both_traversals_and_conditional_runtime_discovery(
    roots, monkeypatch
):
    case = _case(monkeypatch)
    case.roots = roots
    root = _module(excluded=_Poison())
    case.walks = [[], [("", root)], [("", root)]]
    assert case.collect() == ()
    assert case.events == [
        "walk:1",
        "end:1",
        "roots",
        *(("runtime", "initial-slots") if roots else ()),
        "walk:2",
        "visit:2:",
        "end:2",
        "torch-slots",
        "walk:3",
        "visit:3:",
        "end:3",
    ]


@pytest.mark.parametrize("kind,number", (("initial", 2), ("torch", 3)))
def test_discovery_failure_precedes_its_traversal_and_retains_earlier_records(
    kind, number, monkeypatch
):
    case = _case(monkeypatch)
    first, second = torch.tensor(1.0), torch.tensor(2.0)
    root = _module(first=first, second=second)
    case.walks = [[], [("", root)], [("", root)]]
    case.initial, case.torch = [("first", first)], [("second", second)]
    error = RuntimeError("discovery failed")
    getattr(case, kind).append(error)
    case.failures[f"walk:{number}"] = RuntimeError("traversal began too soon")
    with pytest.raises(RuntimeError) as caught:
        case.collect()
    assert caught.value is error
    assert case.events[-1] == f"fail:{kind}"
    assert f"walk:{number}" not in case.events
    assert [entry.key for entry in case.records] == (
        [] if kind == "initial" else [".first"]
    )


@pytest.mark.parametrize("kind,number", (("initial", 2), ("torch", 3)))
def test_traversal_failure_stops_later_phases_after_completed_records(
    kind, number, monkeypatch
):
    case = _case(monkeypatch)
    value = torch.tensor(1.0)
    root = _module(value=value)
    case.walks = [[], [("", root)], [("", root)]]
    setattr(case, kind, [("value", value)])
    error = RuntimeError("registered traversal failed")
    case.walks[number - 1].append(error)
    with pytest.raises(RuntimeError) as caught:
        case.collect()
    assert caught.value is error
    assert case.events[-1] == f"fail:walk:{number}"
    assert [entry.key for entry in case.records] == [".value"]
    if kind == "initial":
        assert "torch-slots" not in case.events


@pytest.mark.parametrize("kind", ("initial", "torch"))
@pytest.mark.parametrize("failure", ("shape", "device"))
def test_metadata_failures_preserve_access_order_and_partial_records(
    kind, failure, monkeypatch
):
    case = _case(monkeypatch)
    phases = ("shape", "dtype", "device")
    errors = {name: RuntimeError(name) for name in phases[phases.index(failure) :]}

    class Buffer:
        def __getattribute__(self, name):
            if name in phases:
                case.phase(name)
                if name in errors:
                    raise errors[name]
                return {
                    "shape": (2,),
                    "dtype": torch.float64,
                    "device": torch.device("cpu"),
                }[name]
            return super().__getattribute__(name)

    first, bad = torch.tensor(1.0), Buffer()
    root = _module(first=first, bad=bad)
    case.walks = [[], [("", root)], [("", root)]]
    setattr(case, kind, [("first", first), ("bad", bad)])
    with pytest.raises(RuntimeError) as caught:
        case.collect()
    assert caught.value is errors[failure]
    index = case.events.index("shape")
    assert case.events[index:] == list(phases[: phases.index(failure) + 1])
    assert [entry.key for entry in case.records] == [".first"]
