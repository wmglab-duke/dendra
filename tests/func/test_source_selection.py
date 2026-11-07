"""Source selection preserves captured metadata and complete Tensor identities."""

from __future__ import annotations

import os
import re
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _assign, _plain, _runner, _voltage
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_explicit_source_inspection import _case, _path
from tests.func.test_lowering_api import G, _target
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_source_inspection import _continuation

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _ids(values):
    return tuple(map(id, values))


def _paths(report):
    return tuple(entry.path for entry in report.entries)


def _sample():
    shared = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    separate = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    derived = shared.square()
    frozen = torch.tensor([4.0], dtype=torch.float64)
    subject = _continuation(
        {
            "block": {"first": shared, "second": separate},
            "alias": shared,
            "derived": derived,
        },
        {"shared": shared, "frozen": frozen},
    )
    return (
        subject,
        dn.func.inspect_sources(subject),
        (shared, separate, derived, frozen),
    )


def _assert_view(view, report, identities):
    expected = tuple(
        entry for entry in report.entries if id(entry.tensor) in identities
    )
    assert type(view) is dn.func.SourceReport
    assert _paths(view) == tuple(entry.path for entry in expected)
    assert all(
        actual is original
        for actual, original in zip(view.entries, expected, strict=True)
    )
    assert _ids(view.trainable_leaves) == tuple(
        id(value) for value in report.trainable_leaves if id(value) in identities
    )


@pytest.mark.parametrize(
    "query,indices",
    (
        ("parameters.block.first", (0,)),
        ("second", (1,)),
        ("constants.shared", (0,)),
        ("glob:*first", (0,)),
        ("parameters.block.*", (0, 1)),
        (r"re:^constants\.shared$", (0,)),
        (re.compile(r"^CONSTANTS\.SHARED$", re.IGNORECASE), (0,)),
        (("parameters.block.second", "constants.shared", "parameters.alias"), (0, 1)),
        ("*", (0, 1, 2, 3)),
        ((), ()),
    ),
)
def test_selectors_keep_full_alias_groups_in_original_report_order(query, indices):
    _, report, values = _sample()
    selected = report.select(query)
    _assert_view(selected, report, {id(values[index]) for index in indices})
    for entry in selected.entries:
        assert entry.aliases == next(
            original.aliases for original in report.entries if original is entry
        )
    assert _ids(report.trainable_leaves) == _ids(values[:2])


def test_exact_literal_paths_take_priority_over_glob_syntax():
    bracket = torch.ones(1, requires_grad=True)
    plain = torch.ones(1, requires_grad=True)
    report = dn.func.inspect_sources(
        _continuation({"literal[0]": bracket, "literal0": plain})
    )
    assert report.select("parameters.literal[0]").trainable_leaves[0] is bracket
    assert report.select("glob:parameters.literal[0]").trainable_leaves[0] is plain


def test_iterable_queries_are_consumed_once_and_union_in_source_order():
    _, report, values = _sample()
    calls = []

    def queries():
        for name in ("parameters.block.second", "constants.shared", "parameters.alias"):
            calls.append(name)
            yield name

    selected = report.select(queries())
    assert calls == ["parameters.block.second", "constants.shared", "parameters.alias"]
    _assert_view(selected, report, {id(values[0]), id(values[1])})
    assert _ids(selected.trainable_leaves) == _ids(values[:2])


@pytest.mark.parametrize(
    "query,error",
    (
        ("", ValueError),
        ("glob:", ValueError),
        ("re:", ValueError),
        ("re:[", ValueError),
        (re.compile(b"first"), TypeError),
        (None, TypeError),
        (1, TypeError),
        (("parameters.block.first", ""), ValueError),
        (("parameters.block.first", "missing"), KeyError),
        ("missing", KeyError),
        ("glob:missing.*", KeyError),
        ("parameters.block", KeyError),
    ),
)
def test_invalid_unmatched_or_ambiguous_queries_leave_the_report_unchanged(
    query, error
):
    _, report, _ = _sample()
    entries, leaves = report.entries, report.trainable_leaves
    with pytest.raises(error):
        report.select(query)
    assert report.entries is entries and report.trainable_leaves is leaves


def test_chained_and_empty_selections_are_scoped_to_the_existing_view():
    _, report, values = _sample()
    selected = report.select("constants.shared")
    _assert_view(selected.select("parameters.alias"), report, {id(values[0])})
    _assert_view(selected.select("*"), report, {id(values[0])})
    with pytest.raises(KeyError):
        selected.select("parameters.block.second")
    empty = selected.select([])
    assert empty.entries == () and empty.trainable_leaves == ()
    assert empty.select([]).entries == ()
    with pytest.raises(KeyError):
        empty.select("*")
    with pytest.raises(FrozenInstanceError):
        selected.entries = ()


def test_selection_uses_captured_flags_and_paths_without_rediscovering_upstream_leaves():
    subject, report, (shared, separate, derived, frozen) = _sample()
    shared.requires_grad_(False)
    frozen.requires_grad_(True)
    subject.parameters.clear()
    subject.constants.clear()
    selected = report.select(
        ("parameters.block.first", "parameters.derived", "constants.frozen")
    )
    _assert_view(selected, report, {id(shared), id(derived), id(frozen)})
    assert _ids(selected.trainable_leaves) == (id(shared),)
    assert next(
        entry for entry in selected.entries if entry.tensor is shared
    ).requires_grad
    assert not next(
        entry for entry in selected.entries if entry.tensor is frozen
    ).requires_grad
    assert (
        next(entry for entry in selected.entries if entry.tensor is derived).is_leaf
        is False
    )
    assert all(
        value is not derived and value is not separate
        for value in selected.trainable_leaves
    )
    assert dn.func.inspect_sources(subject).entries == ()


def test_distinct_views_are_selected_by_identity_instead_of_storage_or_values():
    storage = torch.arange(4.0)
    first = storage[:2].requires_grad_()
    second = storage[:2].requires_grad_()
    assert first.data_ptr() == second.data_ptr() and first is not second
    report = dn.func.inspect_sources(_continuation({"first": first, "second": second}))
    assert _ids(report.select("parameters.first").trainable_leaves) == (id(first),)
    assert _ids(report.select("parameters.*").trainable_leaves) == (
        id(first),
        id(second),
    )


def test_selection_does_not_read_tensor_flags_dispatch_operators_or_retraverse_sources(
    monkeypatch,
):
    from dendra.func import _source_inspection

    events = []

    class Tracked(torch.Tensor):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            events.append(function.__name__)
            return super().__torch_function__(function, types, args, kwargs or {})

    value = torch.Tensor._make_subclass(Tracked, torch.ones(2), True)
    report = dn.func.inspect_sources(
        _continuation({"first": value, "second": value}, {"third": value})
    )
    events.clear()

    def reject(*args, **kwargs):
        raise AssertionError("selection must only read the captured inventory")

    monkeypatch.setattr(_source_inspection, "named_parameters", reject)
    monkeypatch.setattr(_source_inspection, "_source_frames", reject)

    class NoOperators(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            raise AssertionError(f"selection dispatched {operation}")

    with NoOperators(), torch.inference_mode():
        selected = report.select("constants.third")
    assert events == []
    assert _ids(selected.trainable_leaves) == (id(value),)
    assert all(entry.tensor is value for entry in selected.entries)


def test_selected_report_display_keeps_cross_domain_aliases_and_captured_flags():
    _, report, _ = _sample()
    selected = report.select("constants.shared")
    displayed = str(selected)
    assert displayed == repr(selected)
    assert "3 named paths; 1 unique tensor; 1 alias group" in displayed
    assert "Gradients: 1 trainable leaf; 0 nonleaf tensors" in displayed
    assert "parameters (2 paths)" in displayed and "constants (1 path)" in displayed
    assert displayed.count("-> parameters.block.first") == 2
    assert "(2 aliases)" in displayed
    assert "tensor(" not in displayed
    assert "Sources: 0 named paths; 0 unique tensors; 0 alias groups" in str(
        report.select([])
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize(
    "kind", ("runner", "bundle", "raw_binding", "compiled_binding", "continuation")
)
def test_all_inspection_subjects_support_selection_without_changing_source_ownership(
    network, kind
):
    if kind == "runner":
        subject = _runner(network, callbacks=False)
        target = _target(subject.parameters, network)
    else:
        _, tensors, parameters, constants, bound = _case(
            network, compiled=kind == "compiled_binding"
        )
        target = _target(parameters, network)
        subject = (
            tensors._replace(parameters=parameters, constants=constants)
            if kind == "bundle"
            else _continuation(parameters, constants, state=tensors.state)
            if kind == "continuation"
            else bound
        )
    report = dn.func.inspect_sources(subject)
    selected = report.select(_path(network))
    _assert_view(selected, report, {id(target)})
    assert _ids(selected.trainable_leaves) == (id(target),)
    if network:
        assert f"parameters.populations.right.{G}" in _paths(selected)


def test_raw_network_selection_keeps_canonical_readers_without_rewriting_authored_frames():
    model, key = _cross_population_binding()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(f"populations.left.{key}")
    constants = _plain(tensors.constants)
    bound = functional.bind(parameters, constants=constants)
    canonical = parameters["populations"]["left"][key]
    authored = constants["populations"]["right"]["parametrizations.factor"]
    path = "constants.populations.right.parametrizations.factor"
    effective = dn.func.inspect_sources(bound)
    selected = effective.select(path)
    _assert_view(selected, effective, {id(canonical)})
    assert f"parameters.populations.left.{key}" in _paths(selected)
    assert _ids(selected.trainable_leaves) == (id(canonical),)
    stored = dn.func.inspect_sources(
        dn.func.NetworkTensors(parameters, constants, tensors.state)
    ).select(path)
    assert all(entry.tensor is authored for entry in stored.entries)
    assert constants["populations"]["right"]["parametrizations.factor"] is authored
    assert (
        bound.constants["populations"]["right"]["parametrizations.factor"] is authored
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_selection_follows_effective_override_and_restored_sources(network, tmp_path):
    program = _runner(network, callbacks=False)
    default = _target(program.parameters, network)
    parameters = _plain(program.parameters)
    upstream = default.detach().clone().requires_grad_()
    override = upstream * 1.3
    _assign(parameters, network, override)
    state, auxiliary = dn.func.run(program, parameters=parameters, steps=1)
    selected = dn.func.inspect_sources(auxiliary["continuation"]).select(
        "parameters.*g_param"
    )
    assert all(entry.tensor is override for entry in selected.entries)
    assert selected.trainable_leaves == ()
    assert all(entry.requires_grad and not entry.is_leaf for entry in selected.entries)
    (gradient,) = torch.autograd.grad(_voltage(state, network).square().sum(), upstream)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert _target(program.parameters, network) is default
    checkpoint = dn.func.export_continuation(program, auxiliary["continuation"])
    restored = dn.func.restore_continuation(
        program, _disk(checkpoint, tmp_path / "sources.pt")
    )
    restored_value = _target(restored.parameters, network)
    restored_report = dn.func.inspect_sources(restored).select("parameters.*g_param")
    assert _ids(restored_report.trainable_leaves) == (id(restored_value),)
    assert restored_value is not override and restored_value.is_leaf
    if network:
        assert restored_value is restored.parameters["populations"]["right"][G]


@pytest.mark.parametrize("mode", ("groups", "separate"))
def test_selections_supply_independent_optimizer_groups_or_separate_optimizers(mode):
    fast = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    slow = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    report = dn.func.inspect_sources(
        _continuation(
            {"fast": fast, "slow": slow, "fast_alias": fast}, {"shared": fast}
        )
    )
    fast_view, slow_view = (
        report.select("parameters.fast"),
        report.select("parameters.slow"),
    )
    assert _ids(fast_view.trainable_leaves) == (id(fast),)
    assert _ids(slow_view.trainable_leaves) == (id(slow),)
    if mode == "groups":
        optimizers = [
            torch.optim.Adam(
                [
                    {"params": fast_view.trainable_leaves, "lr": 0.1},
                    {"params": slow_view.trainable_leaves, "lr": 0.01},
                ]
            )
        ]
    else:
        optimizers = [
            torch.optim.Adam(fast_view.trainable_leaves, lr=0.1),
            torch.optim.SGD(slow_view.trainable_leaves, lr=0.01),
        ]
    (fast.square().sum() + slow.square().sum()).backward()
    for optimizer in optimizers:
        optimizer.step()
    torch.testing.assert_close(fast, torch.tensor([1.9], dtype=torch.float64))
    torch.testing.assert_close(
        slow, torch.tensor([2.99 if mode == "groups" else 2.94], dtype=torch.float64)
    )
    assert all(entry.tensor is fast for entry in fast_view.entries)
    assert len(optimizers[0].param_groups[0]["params"]) == 1


def test_overlapping_views_expose_the_same_leaf_and_keep_pytorch_optimizer_rules():
    value = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    report = dn.func.inspect_sources(_continuation({"first": value, "alias": value}))
    first, alias = report.select("parameters.first"), report.select("parameters.alias")
    assert _ids(first.trainable_leaves) == _ids(alias.trainable_leaves) == (id(value),)
    with pytest.raises(ValueError, match="more than one parameter group"):
        torch.optim.SGD(
            [{"params": first.trainable_leaves}, {"params": alias.trainable_leaves}],
            lr=0.1,
        )
    optimizers = [
        torch.optim.SGD(first.trainable_leaves, lr=0.1),
        torch.optim.SGD(alias.trainable_leaves, lr=0.2),
    ]
    value.square().sum().backward()
    for optimizer in optimizers:
        optimizer.step()
    torch.testing.assert_close(value, torch.tensor([0.8], dtype=torch.float64))


@pytest.mark.mps
def test_mps_selection_and_optimizer_groups_retain_devices_and_aliases():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    fast = torch.tensor([2.0], dtype=torch.float32, device="mps", requires_grad=True)
    slow = torch.tensor([3.0], dtype=torch.float32, device="mps", requires_grad=True)
    report = dn.func.inspect_sources(
        _continuation({"fast": fast, "slow": slow}, {"shared": fast})
    )
    first, second = report.select("parameters.fast"), report.select("parameters.slow")
    assert first.trainable_leaves[0] is fast and second.trainable_leaves[0] is slow
    assert all(
        entry.tensor.device.type == "mps" for entry in first.entries + second.entries
    )
    optimizer = torch.optim.SGD(
        [
            {"params": first.trainable_leaves, "lr": 0.1},
            {"params": second.trainable_leaves, "lr": 0.01},
        ]
    )
    (fast.square().sum() + slow.square().sum()).backward()
    optimizer.step()
    torch.testing.assert_close(fast.cpu(), torch.tensor([1.6], dtype=torch.float32))
    torch.testing.assert_close(slow.cpu(), torch.tensor([2.94], dtype=torch.float32))


def test_manual_report_selection_preserves_leaf_membership_order_and_deduplicates():
    first = torch.ones(1, requires_grad=True)
    second = torch.zeros(1)
    outside = torch.ones(1, requires_grad=True)
    entries = (
        dn.func.SourceEntry("parameters.first", "parameters", first, False, False),
        dn.func.SourceEntry("constants.second", "constants", second, False, True),
        dn.func.SourceEntry("constants.first", "constants", first, True, True),
    )
    report = dn.func.SourceReport(entries, (second, first, second, outside, first))
    selected = report.select("*")
    assert all(
        actual is original
        for actual, original in zip(selected.entries, entries, strict=True)
    )
    assert _ids(selected.trainable_leaves) == (id(second), id(first))
    first_only = report.select("parameters.first")
    assert _paths(first_only) == ("parameters.first", "constants.first")
    assert _ids(first_only.trainable_leaves) == (id(first),)
    assert first_only.entries[0].requires_grad is False
    assert first_only.entries[0].is_leaf is False
    assert _ids(report.trainable_leaves) == (
        id(second),
        id(first),
        id(second),
        id(outside),
        id(first),
    )


def test_selection_rejects_compile_capture_before_iterating_queries(monkeypatch):
    _, report, _ = _sample()

    class Queries:
        def __iter__(self):
            raise AssertionError("the host guard must precede query iteration")

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(dn.func.FunctionalizationError, match="outside torch.compile"):
        report.select(Queries())


def test_selection_adds_no_functorch_or_forward_ad_restriction():
    _, report, values = _sample()

    def work(value):
        selected = report.select("parameters.block.first")
        assert selected.trainable_leaves[0] is values[0]
        return value.square() + selected.trainable_leaves[0].sum()

    torch.testing.assert_close(
        torch.func.grad(work)(torch.tensor(2.0)), torch.tensor(4.0)
    )
    with torch.autograd.forward_ad.dual_level():
        selected = report.select("parameters.block.first")
    _assert_view(selected, report, {id(values[0])})
