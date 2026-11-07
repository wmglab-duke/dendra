"""Source reports expose live identities with immutable inspection metadata."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn

pytestmark = pytest.mark.cpu


@pytest.mark.mps
def test_mps_inspection_retains_device_aliases_and_distinct_leaf_views():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    value = torch.tensor([1.0, 2.0], device="mps", requires_grad=True)
    view = value.detach().requires_grad_()
    continuation = _continuation({"shared": value, "view": view}, {"shared": value})
    report = dn.func.inspect_sources(continuation)
    assert tuple(map(id, report.trainable_leaves)) == (id(value), id(view))
    assert report.entries[0].aliases == ("constants.shared",)
    assert all(entry.tensor.device.type == "mps" for entry in report.entries)


def _continuation(parameters=None, constants=None, *, state=None):
    return dn.func.Continuation(
        {} if state is None else state,
        {} if parameters is None else parameters,
        {} if constants is None else constants,
    )


def test_report_keeps_order_named_schedule_fields_and_cross_domain_identity_aliases():
    shared = torch.tensor([1.0, 2.0], requires_grad=True)
    independent = torch.tensor([1.0, 2.0], requires_grad=True)
    derived = shared * 2
    frozen = torch.tensor([3.0])
    schedule = dn.func.TensorScheduleTensors(shared, derived, shared, None)
    continuation = _continuation(
        {
            "weight": shared,
            "nested": {"schedule": schedule, "optional": None},
            "sequence": (independent, None),
        },
        {"shared": shared, "frozen": frozen},
    )
    report = dn.func.inspect_sources(continuation)
    assert isinstance(report, dn.func.SourceReport)
    assert type(report.entries) is tuple
    assert [entry.path for entry in report.entries] == [
        "parameters.weight",
        "parameters.nested.schedule.times",
        "parameters.nested.schedule.amplitudes",
        "parameters.nested.schedule.time_source",
        "parameters.sequence.0",
        "constants.shared",
        "constants.frozen",
    ]
    assert [entry.domain for entry in report.entries] == [
        "parameters",
        "parameters",
        "parameters",
        "parameters",
        "parameters",
        "constants",
        "constants",
    ]
    group = (
        "parameters.weight",
        "parameters.nested.schedule.times",
        "parameters.nested.schedule.time_source",
        "constants.shared",
    )
    expected = (shared, shared, derived, shared, independent, shared, frozen)
    for entry, value in zip(report.entries, expected, strict=True):
        assert isinstance(entry, dn.func.SourceEntry)
        assert entry.tensor is value
        assert entry.requires_grad is value.requires_grad
        assert entry.is_leaf is value.is_leaf
        assert entry.aliases == (
            tuple(path for path in group if path != entry.path)
            if value is shared
            else ()
        )
    assert tuple(map(id, report.trainable_leaves)) == (id(shared), id(independent))


def test_report_metadata_and_paths_are_snapshots_while_tensor_values_remain_live():
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    continuation = _continuation({"original": value})
    report = dn.func.inspect_sources(continuation)
    entry = report.entries[0]
    value.requires_grad_(False)
    with torch.no_grad():
        value.add_(3)
    assert entry.requires_grad and entry.is_leaf
    assert entry.tensor is value
    torch.testing.assert_close(entry.tensor, torch.tensor([4.0, 5.0]))
    assert report.trainable_leaves[0] is value
    refreshed = dn.func.inspect_sources(continuation)
    assert not refreshed.entries[0].requires_grad
    assert refreshed.trainable_leaves == ()

    replacement = torch.tensor([9.0], requires_grad=True)
    continuation.parameters.clear()
    continuation.parameters["replacement"] = replacement
    assert entry.path == "parameters.original" and entry.tensor is value
    fresh = dn.func.inspect_sources(continuation)
    assert fresh.entries[0].path == "parameters.replacement"
    assert fresh.entries[0].tensor is replacement


def test_trainable_nonleaf_sources_remain_visible_without_discovering_upstream_leaves():
    upstream = torch.tensor([2.0, 3.0], requires_grad=True)
    derived = upstream.square()
    report = dn.func.inspect_sources(_continuation({"derived": derived}))
    assert report.entries[0].tensor is derived
    assert report.entries[0].requires_grad and not report.entries[0].is_leaf
    assert report.trainable_leaves == ()
    assert all(entry.tensor is not upstream for entry in report.entries)
    derived.sum().backward()
    torch.testing.assert_close(upstream.grad, 2 * upstream)


def test_state_is_excluded_even_when_it_contains_trainable_leaves():
    state = torch.ones(2, requires_grad=True)
    parameter = torch.zeros(2, requires_grad=True)
    report = dn.func.inspect_sources(
        _continuation({"p": parameter}, state={"v": state})
    )
    assert len(report.entries) == 1
    assert report.entries[0].tensor is parameter
    assert report.trainable_leaves[0] is parameter


def test_distinct_tensor_views_are_not_deduplicated_by_storage_or_value():
    backing = torch.arange(4.0)
    first = backing[:2].requires_grad_()
    second = backing[:2].requires_grad_()
    report = dn.func.inspect_sources(_continuation({"first": first, "second": second}))
    assert first is not second and first.data_ptr() == second.data_ptr()
    assert tuple(map(id, report.trainable_leaves)) == (id(first), id(second))
    assert all(entry.aliases == () for entry in report.entries)


def test_reports_and_entries_are_read_only_and_do_not_compare_tensor_contents():
    first = dn.func.inspect_sources(_continuation({"x": torch.ones(2)}))
    second = dn.func.inspect_sources(_continuation({"x": torch.ones(2)}))
    with pytest.raises(FrozenInstanceError):
        first.entries = ()
    with pytest.raises(FrozenInstanceError):
        first.entries[0].path = "changed"
    assert first != second
    assert first.entries[0] != second.entries[0]


def test_inspection_does_not_clone_detach_or_dispatch_tensor_operators():
    events = []

    class Tracked(torch.Tensor):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            events.append(function.__name__)
            return super().__torch_function__(function, types, args, kwargs or {})

    value = torch.Tensor._make_subclass(Tracked, torch.ones(2), True)
    continuation = _continuation({"first": value, "second": value}, {"third": value})
    events.clear()

    class NoOperators(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            raise AssertionError(f"inspection dispatched {operation}")

    with NoOperators():
        report = dn.func.inspect_sources(continuation)
    assert all(entry.tensor is value for entry in report.entries)
    assert report.trainable_leaves[0] is value
    assert all(name == "__get__" for name in events)


def test_inspection_under_inference_mode_retains_actual_trainability():
    value = torch.ones(2, requires_grad=True)
    continuation = _continuation({"value": value})
    with torch.inference_mode():
        report = dn.func.inspect_sources(continuation)
    assert report.entries[0].requires_grad and report.entries[0].is_leaf
    assert report.trainable_leaves[0] is value
    assert value.requires_grad and not value.is_inference()


@pytest.mark.parametrize("kind", ("object", "mapping", "module"))
def test_arbitrary_objects_and_modules_are_not_accepted_as_source_owners(kind):
    subject = {"object": object(), "mapping": {}, "module": torch.nn.Module()}[kind]
    with pytest.raises(TypeError, match="runner|Continuation|subject"):
        dn.func.inspect_sources(subject)


def test_a_duck_typed_object_does_not_read_authored_source_properties():
    class Authored:
        @property
        def parameters(self):
            raise AssertionError("unsupported subjects must not be inspected")

        constants = {}

    with pytest.raises(TypeError, match="runner|Continuation|subject"):
        dn.func.inspect_sources(Authored())
    with pytest.raises(TypeError, match="runner|Continuation|subject"):
        dn.func.inspect_sources(SimpleNamespace(parameters={}, constants={}))


@pytest.mark.parametrize("invalid", (1.0, [], object()))
def test_unsupported_frame_leaves_keep_existing_tree_errors(invalid):
    with pytest.raises(TypeError, match="mappings, tuples, Tensors, and None"):
        dn.func.inspect_sources(_continuation({"invalid": invalid}))


def test_ambiguous_dotted_paths_reject_instead_of_hiding_a_source():
    value = torch.ones(2)
    continuation = _continuation({"a.b": value, "a": {"b": value}})
    with pytest.raises(ValueError, match="paths collide"):
        dn.func.inspect_sources(continuation)


def test_empty_source_frames_produce_an_empty_report():
    report = dn.func.inspect_sources(_continuation())
    assert report.entries == () and report.trainable_leaves == ()


def test_report_display_summarizes_sources_and_shows_compressed_paths_and_aliases():
    shared = torch.full((2, 3), 137.625, dtype=torch.float32, requires_grad=True)
    frozen = torch.zeros(1, dtype=torch.float32)
    derived = shared * 2
    report = dn.func.inspect_sources(
        _continuation(
            {"layers": {"weight": shared, "frozen": frozen}, "tied": shared},
            {"copy": shared, "derived": derived},
        )
    )
    displayed = repr(report)
    assert str(report) == displayed
    assert displayed.startswith("SourceReport") and len(displayed.splitlines()) > 4
    assert "Sources: 5 named paths; 3 unique tensors; 1 alias group" in displayed
    assert (
        "Gradients: 1 trainable leaf; 1 nonleaf tensor requiring gradients" in displayed
    )
    assert "parameters (3 paths)" in displayed and "constants (2 paths)" in displayed
    for label in (
        "parameters",
        "constants",
        "layers",
        "weight",
        "frozen",
        "tied",
        "derived",
    ):
        assert label in displayed
    assert "parameters.layers.weight" in displayed
    assert displayed.count("-> parameters.layers.weight") == 2
    assert "(2 aliases)" in displayed
    assert "nonleaf, requires_grad=True" in displayed
    assert "requires_grad=False" in displayed
    assert "float32" in displayed and "cpu" in displayed
    compact = displayed.replace(" ", "")
    assert any(shape in compact for shape in ("(2,3)", "[2,3]", "2×3"))
    assert "137.625" not in displayed and "tensor(" not in displayed


def test_empty_report_display_includes_both_domains_and_zero_counts():
    report = dn.func.inspect_sources(_continuation())
    displayed = str(report)
    assert displayed.startswith("SourceReport")
    assert "Sources: 0 named paths; 0 unique tensors; 0 alias groups" in displayed
    assert (
        "Gradients: 0 trainable leaves; 0 nonleaf tensors requiring gradients"
        in displayed
    )
    assert "parameters (0 paths)" in displayed and "constants (0 paths)" in displayed
    assert displayed.count("(none)") == 2
    assert displayed == repr(report)
    assert "tensor(" not in displayed


def test_large_report_display_is_bounded_and_prioritizes_flagged_sources():
    parameters = {f"p{index:02d}": torch.zeros(1) for index in range(24)}
    constants = {f"c{index:02d}": torch.zeros(1) for index in range(24)}
    parameters["p23"].requires_grad_()
    constants["c23"].requires_grad_()
    report = dn.func.inspect_sources(_continuation(parameters, constants))
    displayed = str(report)
    assert len(report.entries) == 48
    assert len(report.trainable_leaves) == 2
    assert "48 named paths; 48 unique tensors; 0 alias groups" in displayed
    assert "2 trainable leaves" in displayed
    assert "p23" in displayed and "c23" in displayed
    assert "p22" not in displayed and "c22" not in displayed
    assert displayed.count("12 paths omitted") == 2
    assert "entries" in displayed
    assert len(displayed.splitlines()) < 40
    assert sum("─ p" in line for line in displayed.splitlines()) == 12
    assert sum("─ c" in line for line in displayed.splitlines()) == 12
    assert displayed.index("p00") < displayed.index("p23")
    assert displayed.index("c00") < displayed.index("c23")
    assert report.entries[22].path == "parameters.p22"
    assert report.entries[46].path == "constants.c22"


def test_report_display_uses_captured_flags_without_reading_changed_tensor_values():
    first = torch.full((2,), 17.125, requires_grad=True)
    second = torch.full((2,), 19.625)
    continuation = _continuation({"first": first, "second": second})
    report = dn.func.inspect_sources(continuation)
    displayed = str(report)
    first.requires_grad_(False)
    second.requires_grad_()
    with torch.no_grad():
        first.add_(100)
        second.add_(100)
    assert str(report) == displayed
    assert str(dn.func.inspect_sources(continuation)) != displayed
    assert "17.125" not in displayed and "19.625" not in displayed


def test_report_display_never_renders_tensor_contents_or_dispatches_operators():
    metadata = []

    class NoValues(torch.Tensor):
        def __repr__(self):
            raise AssertionError("report must not render Tensor values")

        def __str__(self):
            raise AssertionError("report must not render Tensor values")

        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            metadata.append(function.__name__)
            if function.__name__ != "__get__":
                raise AssertionError(f"report called {function.__name__}")
            return super().__torch_function__(function, types, args, kwargs or {})

    class NoOperators(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            raise AssertionError(f"report dispatched {operation}")

    value = torch.Tensor._make_subclass(
        NoValues, torch.ones(2, 3, dtype=torch.float32), True
    )
    continuation = _continuation({"first": value}, {"alias": value})
    with NoOperators():
        report = dn.func.inspect_sources(continuation)
        displayed = str(report)
        assert displayed == repr(report)
    assert metadata and all(name == "__get__" for name in metadata)
    assert "float32" in displayed and "cpu" in displayed
    assert "parameters.first" in displayed
    assert "tensor(" not in displayed
