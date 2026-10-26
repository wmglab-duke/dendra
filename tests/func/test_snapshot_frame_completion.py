"""Snapshot extraction publishes complete phases with independent storage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode

import dendra as dn
from tests.func.test_retained_frame_lifecycle import _TIED_PATHS, DTYPE, _plan

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


def _key(kind, path):
    return f"materializations.{kind}.{path}"


def _partial_constants(error, plan):
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == "extract" and frame.f_locals.get("self") is plan:
            return frame.f_locals["constants"]
        traceback = traceback.tb_next
    raise AssertionError("failure did not cross the existing extract endpoint")


class _SeedOperations(TorchFunctionMode):
    def __init__(self, case):
        self.case = case
        self.detached = {}
        self.clones = {}
        self.active_index = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        case = self.case
        event = None
        path = case.current
        if case.seeding and path is not None:
            if function is torch.Tensor.detach and args[0] is case.native[path]:
                event = f"detach:{path}"
            elif function is torch.Tensor.clone and id(args[0]) in self.detached:
                path = self.detached[id(args[0])][0]
                event = f"clone:{path}"
                assert kwargs == {"memory_format": torch.preserve_format}
            elif function is torch.zeros:
                path = case.paths[self.active_index]
                self.active_index += 1
                event = f"active:{path}"
                assert args == ((),) and kwargs["dtype"] == torch.bool
            elif function is torch.zeros_like and id(args[0]) in self.clones:
                path = self.clones[id(args[0])][0]
                event = f"baseline:{path}"
        if event is not None:
            case.events.append(event)
            if event == case.failure:
                raise case.error
        result = function(*args, **kwargs)
        if event == f"detach:{path}":
            self.detached[id(result)] = (path, result)
        elif event == f"clone:{path}":
            self.clones[id(result)] = (path, result)
        return result


def _case(
    monkeypatch, *, failure=None, baseline_mode=None, offsets=False, with_dt=False
):
    plan, packet = _plan()
    source = plan._population
    paths = tuple(
        name.removeprefix("materializations.values.")
        for name in packet.constants
        if name.startswith("materializations.values.")
    )
    if with_dt:
        inventory = plan._retained_materializations
        dt = SimpleNamespace(
            path="integrator.dt",
            value_key=_key("values", "integrator.dt"),
            active_key=_key("active", "integrator.dt"),
            baseline_key=_key("baseline", "integrator.dt"),
            shape=(),
            dtype=plan.dtype,
            device=plan.device,
            prepared_path=("integrator", "dt"),
        )
        monkeypatch.setattr(
            plan,
            "_retained_materializations",
            SimpleNamespace(
                slots=(*inventory.slots, dt), by_path={**inventory.by_path, dt.path: dt}
            ),
        )
        paths = (*paths, "integrator.dt")
    raw = torch.arange(7.0, dtype=DTYPE)[1:].reshape(3, 2).t().requires_grad_()
    mechanism = source.mech.mechanisms["_LifecycleLeak"]
    mechanism.g = mechanism.e = raw
    native = {
        path: packet.constants["dt"]
        if path == "integrator.dt"
        else source.get_buffer(path)
        for path in paths
    }
    case = SimpleNamespace(
        plan=plan,
        source=source,
        paths=paths,
        native=native,
        raw=raw,
        current=None,
        seeding=True,
        failure=failure,
        error=RuntimeError(failure),
        events=[],
        constants=None,
        seeded=None,
        captured=None,
        different=False,
        lookup_frames=[],
    )
    getter = source.get_buffer

    def get_buffer(path):
        if path in paths:
            case.current = path
            event = f"read:{path}"
            case.events.append(event)
            if event == case.failure:
                raise case.error
        # Ordinary point plans do not retain dt; expose its real native value
        # through the getter to exercise the existing dt-exclusion contract.
        return native[path] if path == "integrator.dt" else getter(path)

    monkeypatch.setattr(source, "get_buffer", get_buffer)
    by_path = plan._retained_materializations.by_path

    class Paths(dict):
        def __getitem__(self, path):
            event = f"lookup:{path}"
            case.events.append(event)
            case.lookup_frames.append(dict(case.constants))
            if event == case.failure:
                raise case.error
            return super().__getitem__(path)

    monkeypatch.setattr(
        plan,
        "_retained_materializations",
        SimpleNamespace(
            slots=plan._retained_materializations.slots, by_path=Paths(by_path)
        ),
    )

    def prepare(parameters, constants, *, _snapshot_baseline_capture):
        case.seeding = False
        case.events.append("prepare")
        assert not torch.is_grad_enabled()
        case.constants = constants
        case.seeded = dict(constants)
        capture = _snapshot_baseline_capture
        for index, path in enumerate(reversed(paths)):
            if path != "integrator.dt":
                capture[path] = constants[_key("values", path)].clone()
                if case.different and index == 0:
                    capture[path] = capture[path] + 1
        if baseline_mode == "missing_and_unexpected":
            for path in (paths[1], paths[-2]):
                del capture[path]
            capture["z.unexpected"] = capture["a.unexpected"] = torch.tensor(1.0)
        elif baseline_mode == "unexpected_dt":
            capture["integrator.dt"] = constants["dt"]
        case.captured = dict(capture)
        return {}

    monkeypatch.setattr(plan, "_prepare_values", prepare)
    plan._has_snapshot_materialization_offsets = offsets

    def invoke():
        with _SeedOperations(case):
            return plan.extract()

    case.invoke = invoke
    case.seed_events = (
        [
            event
            for path in paths
            for event in (f"read:{path}", f"detach:{path}", f"clone:{path}")
        ]
        + [f"active:{path}" for path in paths]
        + [f"baseline:{path}" for path in paths]
    )
    return case


@pytest.mark.parametrize("phase", ("read", "clone", "active", "baseline"))
def test_seed_failures_publish_only_preceding_complete_dictionary_phases(
    monkeypatch, phase
):
    case = _case(monkeypatch)
    case.failure = f"{phase}:{case.paths[1]}"
    case.error = RuntimeError(case.failure)
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.error
    assert case.events == case.seed_events[: case.seed_events.index(case.failure) + 1]
    constants = _partial_constants(caught.value, case.plan)
    groups = () if phase in ("read", "clone") else ("values",)
    if phase == "baseline":
        groups = ("values", "active")
    assert tuple(
        name for name in constants if name.startswith("materializations.")
    ) == tuple(_key(kind, path) for kind in groups for path in case.paths)
    assert (
        case.constants is None
        and case.plan._has_snapshot_materialization_offsets is False
    )


def test_snapshot_clones_aliased_native_buffers_independently_without_gradients(
    monkeypatch,
):
    case = _case(monkeypatch)
    result = case.invoke()
    order = tuple(reversed(case.paths))
    assert case.events == [
        *case.seed_events,
        "prepare",
        *(f"lookup:{path}" for path in order),
    ]
    assert tuple(
        name for name in result.constants if name.startswith("materializations.")
    ) == tuple(
        _key(kind, path)
        for kind in ("values", "active", "baseline")
        for path in case.paths
    )
    first, second = (result.constants[_key("values", path)] for path in _TIED_PATHS)
    for value in (first, second):
        torch.testing.assert_close(value, case.raw)
        assert value.stride() == case.raw.stride() and value.storage_offset() == 0
        assert not value.requires_grad and value.grad_fn is None
        assert not torch._C._is_alias_of(value, case.raw)
    assert first is not second and not torch._C._is_alias_of(first, second)
    for path in case.paths:
        assert result.constants[_key("baseline", path)] is case.captured[path]
        assert not result.constants[_key("active", path)].item()
        torch.testing.assert_close(
            case.seeded[_key("baseline", path)], torch.zeros_like(case.native[path])
        )
    for frame in case.lookup_frames:
        assert all(
            frame[_key("baseline", path)] is case.seeded[_key("baseline", path)]
            for path in case.paths
        )


@pytest.mark.parametrize("mode", ("missing_and_unexpected", "unexpected_dt"))
def test_baseline_key_errors_report_exact_sets_before_any_publication(
    monkeypatch, mode
):
    case = _case(monkeypatch, baseline_mode=mode)
    with pytest.raises(dn.func.FunctionalizationError) as caught:
        case.invoke()
    missing = (
        sorted((case.paths[1], case.paths[-2]))
        if mode == "missing_and_unexpected"
        else []
    )
    unexpected = (
        ["a.unexpected", "z.unexpected"]
        if mode == "missing_and_unexpected"
        else ["integrator.dt"]
    )
    assert str(caught.value) == (
        "Snapshot preparation did not visit every retained operand: "
        f"missing={missing}, unexpected={unexpected}"
    )
    assert case.events == [*case.seed_events, "prepare"]
    assert all(case.constants[key] is value for key, value in case.seeded.items())
    assert case.plan._has_snapshot_materialization_offsets is False


def test_integrator_dt_needs_seed_storage_but_no_captured_baseline(monkeypatch):
    case = _case(monkeypatch, with_dt=True)
    result = case.invoke()
    assert "integrator.dt" not in case.captured
    assert not result.constants[_key("active", "integrator.dt")].item()
    assert (
        result.constants[_key("baseline", "integrator.dt")]
        is case.seeded[_key("baseline", "integrator.dt")]
    )
    assert case.events == [
        *case.seed_events,
        "prepare",
        *(f"lookup:{path}" for path in reversed(case.paths) if path != "integrator.dt"),
    ]


def test_late_baseline_lookup_failure_keeps_every_seeded_baseline(monkeypatch):
    case = _case(monkeypatch)
    order = tuple(reversed(case.paths))
    case.failure = f"lookup:{order[1]}"
    case.error = RuntimeError(case.failure)
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.error
    assert case.events == [
        *case.seed_events,
        "prepare",
        *(f"lookup:{path}" for path in order[:2]),
    ]
    assert all(case.constants[key] is value for key, value in case.seeded.items())
    assert case.constants[_key("baseline", order[0])] is not case.captured[order[0]]


@pytest.mark.parametrize("offsets", (None, False, True))
def test_offset_detection_follows_completed_capture_and_keeps_existing_policy(
    monkeypatch, offsets
):
    case = _case(monkeypatch, offsets=offsets)
    case.different = True
    result = case.invoke()
    order = tuple(reversed(case.paths))
    expected = [*case.seed_events, "prepare", *(f"lookup:{path}" for path in order)]
    if offsets is None:
        expected.append(f"lookup:{order[0]}")
    assert case.events == expected
    assert case.plan._has_snapshot_materialization_offsets is (offsets is not False)
    assert all(
        result.constants[_key("baseline", path)] is case.captured[path]
        for path in case.paths
    )
    if offsets is None:
        assert all(
            case.lookup_frames[-1][_key("baseline", path)] is case.captured[path]
            for path in case.paths
        )


def test_empty_retained_inventory_skips_snapshot_preparation(monkeypatch):
    plan, _packet = _plan(retained=False)

    def forbidden(*args, **kwargs):
        pytest.fail("empty retained extraction must not prepare snapshot baselines")

    monkeypatch.setattr(plan, "_prepare_values", forbidden)
    result = plan.extract()
    assert not any(name.startswith("materializations.") for name in result.constants)
    assert plan._has_snapshot_materialization_offsets is None
