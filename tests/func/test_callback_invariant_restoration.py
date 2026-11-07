"""Callback invariant restoration retains lookup order and explicit operands."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from dendra.func._callback_carry import _CallbackCarryPartition

pytestmark = pytest.mark.cpu
_NAMES = ("r_first", "i_first", "r_last", "i_last")
_INVARIANTS = ("i_last", "i_first")
_ORIGINAL_READS = [
    ("original", "r_first"),
    ("original", "r_last"),
    ("original", "i_last"),
    ("original", "i_first"),
]


class _Values(Mapping):
    def __init__(self, label, entries, events, failures=None):
        self.label, self.entries, self.events = label, entries, events
        self.failures = {} if failures is None else failures

    def __getitem__(self, key):
        self.events.append((self.label, key))
        if key in self.failures:
            raise self.failures[key]
        return self.entries[key]

    def __iter__(self):
        raise AssertionError("restoration must use declared names without enumeration")

    def __len__(self):
        raise AssertionError("restoration must not inspect operand length")


def test_restore_keeps_complete_lookup_order_aliases_and_autograd_without_cloning():
    partition = _CallbackCarryPartition(_NAMES, _INVARIANTS)
    events = []
    source = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    old = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
    lead, recurrent = 3 * source, source.square()
    original = _Values(
        "original",
        {"r_first": old, "i_first": lead, "r_last": old, "i_last": lead},
        events,
    )
    updated = _Values(
        "updated",
        {"r_first": recurrent, "r_last": recurrent},
        events,
        {
            "i_first": AssertionError("updated invariant read"),
            "i_last": AssertionError("updated invariant read"),
        },
    )
    version = source._version

    result = partition.restore_invariants(updated, original)

    assert events == [*_ORIGINAL_READS, ("updated", "r_first"), ("updated", "r_last")]
    assert list(result) == list(_NAMES)
    assert result["r_first"] is result["r_last"] is recurrent
    assert result["i_first"] is result["i_last"] is lead
    assert source._version == version
    gradient, unused = torch.autograd.grad(
        sum(result.values()), (source, old), allow_unused=True
    )
    torch.testing.assert_close(gradient, 4 * source + 6)
    assert unused is None


def test_no_invariants_return_updated_identity_without_inspecting_either_operand():
    class Unreadable:
        def __getitem__(self, key):
            raise AssertionError("operand read")

        def __iter__(self):
            raise AssertionError("operand iteration")

        def __len__(self):
            raise AssertionError("operand length")

    partition = _CallbackCarryPartition(("unknown",), ())
    updated, original = Unreadable(), Unreadable()

    assert partition.restore_invariants(updated, original) is updated


def test_each_restoration_uses_fresh_original_invariants_and_updated_recurrence():
    partition = _CallbackCarryPartition(("lead", "value"), ("lead",))
    old_recurrent = object()
    first_lead, second_lead = torch.tensor([1.0]), torch.tensor([2.0])
    first_value, second_value = torch.tensor([3.0]), torch.tensor([4.0])
    original = {"lead": first_lead, "value": old_recurrent}

    first = partition.restore_invariants({"value": first_value}, original)
    original["lead"] = second_lead
    second = partition.restore_invariants({"value": second_value}, original)

    assert first is not second
    assert list(first) == list(second) == ["lead", "value"]
    assert first["lead"] is first_lead and first["value"] is first_value
    assert second["lead"] is second_lead and second["value"] is second_value
    assert original["value"] is old_recurrent


@pytest.mark.parametrize(
    "phase",
    ("missing_original", "authored_original", "missing_updated", "authored_updated"),
)
def test_lookup_failures_preserve_native_errors_authored_identity_and_read_stop(phase):
    partition = _CallbackCarryPartition(_NAMES, _INVARIANTS)
    events, failure = [], RuntimeError("authored carry failure")
    originals = {name: object() for name in _NAMES}
    updates = {"r_first": object(), "r_last": object()}
    original_failures, updated_failures = {}, {}
    if phase == "missing_original":
        del originals["r_last"]
        expected = _ORIGINAL_READS[:2]
        missing = "r_last"
    elif phase == "authored_original":
        original_failures["i_last"] = failure
        expected = _ORIGINAL_READS[:3]
    elif phase == "missing_updated":
        del updates["r_last"]
        expected = [*_ORIGINAL_READS, ("updated", "r_first"), ("updated", "r_last")]
        missing = "r_last"
    else:
        updated_failures["r_first"] = failure
        expected = [*_ORIGINAL_READS, ("updated", "r_first")]
    original = _Values("original", originals, events, original_failures)
    updated = _Values("updated", updates, events, updated_failures)
    expected_type = KeyError if phase.startswith("missing") else RuntimeError

    with pytest.raises(expected_type) as caught:
        partition.restore_invariants(updated, original)

    if expected_type is RuntimeError:
        assert caught.value is failure
    else:
        assert caught.value.args == (missing,)
    assert events == expected


def test_merge_is_bound_before_split_even_when_split_replaces_the_class_method(
    monkeypatch,
):
    partition = _CallbackCarryPartition(("value", "lead"), ("lead",))
    events = []
    updated, original, invariant, result = object(), object(), object(), object()

    def first_merge(self, current, fixed):
        events.append("bound merge")
        assert self is partition and current is updated and fixed is invariant
        return result

    def replacement_merge(self, current, fixed):
        raise AssertionError("merge was looked up after split")

    def split(self, carries):
        events.append("split")
        assert self is partition and carries is original
        monkeypatch.setattr(_CallbackCarryPartition, "merge", replacement_merge)
        return object(), invariant

    monkeypatch.setattr(_CallbackCarryPartition, "merge", first_merge)
    monkeypatch.setattr(_CallbackCarryPartition, "split", split)

    assert partition.restore_invariants(updated, original) is result
    assert events == ["split", "bound merge"]
