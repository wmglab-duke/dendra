"""Native recomputation restores live bindings without cloning Tensor graphs."""

from __future__ import annotations

from collections import namedtuple
from types import MappingProxyType

import pytest
import torch

from dendra.models._checkpoint import _checkpoint_recompute_contexts

pytestmark = pytest.mark.cpu


class _Model:
    def __init__(self, state):
        self.state = state
        self.events = []
        self.read_error = self.restore_error = None

    def state_dict_for_checkpoint(self):
        self.events.append(("read", self.state))
        if self.read_error is not None:
            raise self.read_error
        return self.state

    def restore_dict_from_checkpoint(self, state):
        self.events.append(("restore", state))
        if self.restore_error is not None:
            raise self.restore_error
        self.state = state


def test_context_creation_and_forward_context_do_not_read_or_restore_model_state():
    model = _Model({})
    forward, _recompute = _checkpoint_recompute_contexts(model)
    assert model.events == []
    with forward as value:
        assert value is None
        assert model.events == []
    assert model.events == []


def test_recompute_copies_containers_and_preserves_tensor_identity_aliases_and_graphs():
    raw = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    captured = raw.square()
    marker = object()
    Pair = namedtuple("Pair", ("first", "second"))
    mapping = {"value": captured, "alias": captured}
    values = [captured]
    state = {
        "mapping": MappingProxyType(mapping),
        "values": values,
        "tuple": Pair(captured, marker),
        "marker": marker,
    }
    model = _Model(state)
    _, recompute = _checkpoint_recompute_contexts(model)
    with recompute as value:
        assert value is None
        mapping["value"] = raw * 7
        mapping["new"] = raw
        values.append(raw)
        model.state = {"recomputed": raw}
    restored = model.state
    assert restored is not state and type(restored["mapping"]) is dict
    assert restored["mapping"] == {"value": captured, "alias": captured}
    assert restored["values"] == [captured]
    assert type(restored["tuple"]) is tuple
    assert restored["tuple"] == (captured, marker)
    assert restored["mapping"]["value"] is restored["mapping"]["alias"] is captured
    assert restored["values"][0] is restored["tuple"][0] is captured
    assert restored["marker"] is marker
    gradient = torch.autograd.grad(restored["mapping"]["value"].sum(), raw)[0]
    torch.testing.assert_close(gradient, 2 * raw)


def test_recompute_context_can_be_reused_and_nested_with_entry_time_state():
    original, middle, inner, later = (object() for _ in range(4))
    model = _Model({"value": original})
    _, recompute = _checkpoint_recompute_contexts(model)
    with recompute:
        model.state = {"value": middle}
        with recompute:
            model.state = {"value": inner}
        assert model.state == {"value": middle}
    assert model.state == {"value": original}
    model.state = {"value": later}
    with recompute:
        model.state = {"value": inner}
    assert model.state == {"value": later}
    assert [event for event, _state in model.events] == [
        "read",
        "read",
        "restore",
        "restore",
        "read",
        "restore",
    ]
    assert recompute._previous == []


@pytest.mark.parametrize("exception_type", (RuntimeError, BaseException))
def test_recompute_restores_on_early_stop_and_propagates_original_exception(
    exception_type,
):
    original, changed = object(), object()
    model = _Model({"value": original})
    _, recompute = _checkpoint_recompute_contexts(model)
    failure = exception_type("partial recomputation stopped")
    with pytest.raises(exception_type) as caught:
        with recompute:
            model.state = {"value": changed}
            raise failure
    assert caught.value is failure
    assert model.state == {"value": original}
    assert recompute._previous == []


def test_failed_entry_does_not_push_state_or_attempt_restoration():
    model = _Model({"value": object()})
    _, recompute = _checkpoint_recompute_contexts(model)
    failure = RuntimeError("snapshot failed")
    model.read_error = failure
    with pytest.raises(RuntimeError) as caught:
        with recompute:
            raise AssertionError("failed entry must not enter recomputation")
    assert caught.value is failure
    assert [event for event, _state in model.events] == ["read"]
    assert recompute._previous == []


def test_restoration_failure_retains_existing_exception_precedence_and_clears_stack():
    original, changed = object(), object()
    model = _Model({"value": original})
    _, recompute = _checkpoint_recompute_contexts(model)
    replay_error = RuntimeError("replay failed")
    restore_error = ValueError("restore failed")
    model.restore_error = restore_error
    with pytest.raises(ValueError) as caught:
        with recompute:
            model.state = {"value": changed}
            raise replay_error
    assert caught.value is restore_error
    assert restore_error.__context__ is replay_error
    assert model.state == {"value": changed}
    assert recompute._previous == []
    model.restore_error = None
    with recompute:
        model.state = {"value": original}
    assert model.state == {"value": changed}
