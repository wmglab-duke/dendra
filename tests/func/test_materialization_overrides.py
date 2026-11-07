"""Original override boundaries preserve packet and baseline ownership."""

from __future__ import annotations

from collections.abc import MutableMapping
from types import SimpleNamespace

import pytest
import torch

from dendra.func import _population as population
from dendra.models.mechanisms import _noise_workspaces

pytestmark = pytest.mark.cpu
DTYPE = torch.float64


class _Observed(MutableMapping):
    def __init__(self, label, entries, observe):
        self.label, self.entries, self.observe = label, entries, observe

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __bool__(self):
        raise AssertionError("override packets must not be normalized by truthiness")

    def __getitem__(self, key):
        self.observe(f"{self.label}:get:{key}")
        return self.entries[key]

    def __setitem__(self, key, value):
        self.observe(f"{self.label}:set:{key}")
        self.entries[key] = value

    def __delitem__(self, key):
        del self.entries[key]

    def __contains__(self, key):
        self.observe(f"{self.label}:contains:{key}")
        return key in self.entries

    def get(self, key, default=None):
        self.observe(f"{self.label}:get-default:{key}")
        return self.entries.get(key, default)


def _case(*, failure=None):
    events = []
    error = RuntimeError(failure)

    def observe(event):
        events.append(event)
        if event == failure:
            raise error

    slots = tuple(
        SimpleNamespace(
            path=path,
            value_key=f"materializations.values.{path}",
            active_key=f"materializations.active.{path}",
            baseline_key=f"materializations.baseline.{path}",
        )
        for path in (
            "populations.one.cm",
            "populations.two.cm",
            "populations.one.derived",
        )
    )
    specs = (
        ("cached.other-owner", "populations.two.cm", (), DTYPE, torch.device("cpu")),
        ("cached.cm", "populations.one.cm", (), DTYPE, torch.device("cpu")),
        ("cached.other", "populations.one.other", (), DTYPE, torch.device("cpu")),
    )
    entries = {
        name: torch.tensor(index + 1.0, dtype=DTYPE)
        for index, (name, *_) in enumerate(specs)
    }
    for slot in slots:
        entries[slot.value_key] = torch.tensor(4.0, dtype=DTYPE, requires_grad=True)
        entries[slot.active_key] = torch.tensor(True)
        entries[slot.baseline_key] = torch.tensor(5.0, dtype=DTYPE)
    entries["cache.active"] = torch.tensor(False)

    class Owner:
        @property
        def _cache_value_specs(self):
            observe("owner:cache-specs")
            return specs

        @property
        def _retained_materializations(self):
            observe("owner:retained")
            return SimpleNamespace(slots=slots)

    return SimpleNamespace(
        owner=Owner(),
        constants=_Observed("constants", entries, observe),
        entries=entries,
        slots=slots,
        events=events,
        error=error,
    )


def _construction_events(*, offsets=True, snapshot=False):
    events = [
        "owner:cache-specs",
        "owner:cache-specs",
        "constants:get:cached.cm",
        "constants:get:cached.other",
        "owner:retained",
    ]
    for path in ("populations.one.cm", "populations.one.derived"):
        events.append(f"constants:get:materializations.values.{path}")
        if not snapshot:
            events.append(f"constants:get:materializations.active.{path}")
            if offsets:
                events.append(f"constants:get:materializations.baseline.{path}")
    return [*events, "constants:get-default:cache.active"]


@pytest.mark.parametrize("offsets", (False, True))
def test_prefix_construction_preserves_overwrite_order_references_and_reads(offsets):
    case = _case()
    before = dict(case.entries)
    values, active = population.FunctionalPopulation._cache_preparation_inputs(
        case.owner,
        case.constants,
        prefix="populations.one.",
        use_snapshot_offsets=offsets,
    )
    assert case.events == _construction_events(offsets=offsets)
    assert tuple(values) == ("cm", "other", "derived")
    assert active is case.entries["cache.active"]
    assert values["other"] is case.entries["cached.other"]
    for name in ("cm", "derived"):
        slot = f"populations.one.{name}"
        selected = values[name]
        assert isinstance(selected, population._RetainedMaterialization)
        assert selected.value is case.entries[f"materializations.values.{slot}"]
        assert selected.active is case.entries[f"materializations.active.{slot}"]
        expected = (
            case.entries[f"materializations.baseline.{slot}"] if offsets else None
        )
        assert selected.baseline is expected
    assert values["cm"].cached is case.entries["cached.cm"]
    assert values["derived"].cached is None
    assert tuple(case.entries) == tuple(before)
    assert all(case.entries[key] is value for key, value in before.items())


def test_empty_snapshot_capture_uses_native_paths_without_active_or_baseline_reads():
    case = _case()
    baselines = {}
    values, active = population.FunctionalPopulation._cache_preparation_inputs(
        case.owner,
        case.constants,
        prefix="populations.one.",
        snapshot_baseline_capture=baselines,
    )
    assert case.events == _construction_events(snapshot=True)
    assert tuple(values) == ("cm", "other", "derived")
    assert active is case.entries["cache.active"] and baselines == {}
    for name in ("cm", "derived"):
        selected = values[name]
        assert isinstance(selected, population._SnapshotMaterialization)
        assert selected.path == f"populations.one.{name}"
        assert selected.baselines is baselines
        assert (
            selected.value is case.entries[f"materializations.values.{selected.path}"]
        )


@pytest.mark.parametrize(
    "failure",
    (
        "constants:get:cached.cm",
        "constants:get:materializations.values.populations.one.derived",
        "constants:get-default:cache.active",
    ),
)
def test_construction_failure_keeps_exact_prefix_exception_and_source(failure):
    case = _case(failure=failure)
    before = dict(case.entries)
    with pytest.raises(RuntimeError) as caught:
        population.FunctionalPopulation._cache_preparation_inputs(
            case.owner,
            case.constants,
            prefix="populations.one.",
        )
    assert caught.value is case.error
    expected = _construction_events()
    assert case.events == expected[: expected.index(failure) + 1]
    assert tuple(case.entries) == tuple(before)
    assert all(case.entries[key] is value for key, value in before.items())


def test_globally_empty_and_nonmatching_prefix_keep_distinct_packet_boundaries():
    events = []
    constants = _Observed("constants", {"cache.active": object()}, events.append)
    empty = SimpleNamespace(
        _cache_value_specs=(), _retained_materializations=SimpleNamespace(slots=())
    )
    assert population.FunctionalPopulation._cache_preparation_inputs(
        empty, constants
    ) == (None, None)
    assert events == []
    case = _case()
    values, active = population.FunctionalPopulation._cache_preparation_inputs(
        case.owner,
        case.constants,
        prefix="absent.",
    )
    assert values == {} and active is case.entries["cache.active"]
    assert case.events == [
        "owner:cache-specs",
        "owner:cache-specs",
        "owner:retained",
        "constants:get-default:cache.active",
    ]


@pytest.mark.parametrize("failure", (None, "clone", "publish"))
def test_snapshot_selection_detaches_preserves_format_and_publishes_after_clone(
    monkeypatch, failure
):
    events, allocations = [], []
    error = RuntimeError(failure)
    raw = torch.arange(12.0, dtype=DTYPE).reshape(3, 4).t().requires_grad_()
    selected = raw.detach().clone().requires_grad_()
    path, previous = "populations.one.cm", object()

    def observe(event):
        events.append(event)
        if event == "baselines:set:" + path and failure == "publish":
            raise error

    baselines = _Observed("baselines", {path: previous}, observe)
    packet = _Observed(
        "packet",
        {"cm": population._SnapshotMaterialization(selected, path, baselines)},
        observe,
    )
    detached = []
    original_detach, original_clone = torch.Tensor.detach, torch.Tensor.clone

    def detach(value):
        result = original_detach(value)
        if value is raw:
            events.append("detach")
            detached.append(result)
        return result

    def clone(value, *args, **kwargs):
        if detached and value is detached[0]:
            events.append("clone")
            assert args == () and kwargs == {"memory_format": torch.preserve_format}
            if failure == "clone":
                raise error
            result = original_clone(value, *args, **kwargs)
            allocations.append(result)
            return result
        return original_clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "detach", detach)
    monkeypatch.setattr(torch.Tensor, "clone", clone)
    expected = ["packet:contains:cm", "packet:get:cm", "detach", "clone"]
    if failure != "clone":
        expected.append("baselines:set:" + path)
    if failure is not None:
        with pytest.raises(RuntimeError) as caught:
            population._cache_materialization(raw, "cm", packet, None)
        assert caught.value is error and baselines.entries[path] is previous
    else:
        result = population._cache_materialization(raw, "cm", packet, None)
        assert result is selected and baselines.entries[path] is allocations[0]
        baseline = baselines.entries[path]
        assert baseline.stride() == raw.stride()
        assert baseline.grad_fn is None and not baseline.requires_grad
        assert not torch._C._is_alias_of(baseline, raw)
        torch.testing.assert_close(baseline, raw)
        gradients = torch.autograd.grad(
            result.sum(), (raw, selected), allow_unused=True
        )
        assert gradients[0] is None
        torch.testing.assert_close(gradients[1], torch.ones_like(selected))
    assert events == expected


def test_selection_fast_paths_preserve_identity_and_skip_packet_reads():
    value, events = torch.tensor(3.0, dtype=DTYPE, requires_grad=True), []
    assert population._cache_materialization(value, "missing", None, None) is value
    packet = _Observed("packet", {}, events.append)
    assert (
        population._cache_materialization(value, "missing", packet, object()) is value
    )
    assert events == ["packet:contains:missing"]


def _workspace(monkeypatch):
    events = []
    buffers = ["a", "b"]

    class Owner:
        @property
        def _derived_buffers(self):
            events.append("buffers")
            return tuple(buffers)

    owner = Owner()

    def phases(actual):
        assert actual is owner
        events.append("phases")
        return ("derived",)

    monkeypatch.setattr(_noise_workspaces, "noise_workspace_phases", phases)
    return owner, events, buffers


def test_workspace_fast_paths_preserve_phase_and_first_missing_field_order(monkeypatch):
    owner, events, buffers = _workspace(monkeypatch)
    assert (
        population._retained_workspace_builder("m", owner, "derived", None, None)
        is None
    )
    assert events == []
    packet = _Observed("packet", {}, events.append)
    assert (
        population._retained_workspace_builder("m", owner, "unsupported", packet, None)
        is None
    )
    assert events == ["phases"]
    events.clear()
    assert (
        population._retained_workspace_builder("m", owner, "derived", packet, None)
        is None
    )
    assert events == ["phases", "buffers", "packet:get-default:m.a"]
    events.clear()
    buffers.clear()
    assert (
        population._retained_workspace_builder("m", owner, "derived", packet, None)
        is None
    )
    assert events == ["phases", "buffers"]


@pytest.mark.parametrize(
    "cached, global_active, expected",
    ((True, True, True), (False, True, False), (True, None, None)),
)
def test_workspace_mixed_retained_and_cached_masks_keep_existing_precedence(
    monkeypatch, cached, global_active, expected
):
    owner, events, _ = _workspace(monkeypatch)
    value = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)
    flag = None if global_active is None else torch.tensor(global_active)
    packet = _Observed(
        "packet",
        {
            "m.a": population._RetainedMaterialization(
                value if cached else None, value, torch.tensor(False), None
            ),
            "m.b": value,
        },
        events.append,
    )
    actual = population._retained_workspace_builder("m", owner, "derived", packet, flag)
    assert events == [
        "phases",
        "buffers",
        "packet:get-default:m.a",
        "packet:get-default:m.b",
    ]
    if expected is None:
        assert actual is None
    else:
        assert actual.shape == () and actual.dtype == torch.bool
        assert actual.item() is expected
    assert value.grad is None and packet.entries["m.a"].value is value
