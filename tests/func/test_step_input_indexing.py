"""Step selection preserves tape views, PyTrees, indexing order, and AD."""

from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra.func._types import (
    RuntimeNoiseDraw,
    StepInput,
    StimulationDraw,
    _index_step_input,
)

pytestmark = pytest.mark.cpu


@dataclass(frozen=True)
class _Envelope:
    label: str
    payload: object


def _flatten(envelope):
    return [envelope.payload], envelope.label


def _unflatten(children, label):
    return _Envelope(label, children[0])


pytree.register_pytree_node(_Envelope, _flatten, _unflatten)


class _Pair(NamedTuple):
    first: object
    second: object


@pytest.mark.parametrize("missing", ("ve", "intra", "noise", "stimulation"))
def test_absent_field_does_not_change_other_selected_rows(missing):
    ve = torch.arange(6.0).reshape(3, 2)
    intra = ve + 10
    noise = RuntimeNoiseDraw({"eta": ve + 20}, {})
    stimulation = StimulationDraw({"events": ve + 30}, {})
    tapes = dict(ve=ve, intra=intra, noise=noise, stimulation=stimulation)
    tapes[missing] = None
    selected = _index_step_input(**tapes, index=1)
    assert isinstance(selected, StepInput)
    assert getattr(selected, missing) is None
    if missing != "ve":
        torch.testing.assert_close(selected.ve, torch.tensor([2.0, 3.0]))
    if missing != "intra":
        torch.testing.assert_close(selected.intra, torch.tensor([12.0, 13.0]))
    if missing != "noise":
        torch.testing.assert_close(
            selected.noise.draws["eta"], torch.tensor([22.0, 23.0])
        )
    if missing != "stimulation":
        torch.testing.assert_close(
            selected.stimulation.draws["events"], torch.tensor([32.0, 33.0])
        )


@pytest.mark.parametrize(
    "noise, stimulation",
    ((None, None), ({}, []), (RuntimeNoiseDraw({}, {}), StimulationDraw({}, {}))),
)
def test_none_and_empty_trees_keep_their_distinct_container_contract(
    noise, stimulation
):
    selected = _index_step_input(None, None, noise, stimulation, 3)
    assert selected == StepInput(None, None, noise, stimulation)
    assert type(selected.noise) is type(noise)
    assert type(selected.stimulation) is type(stimulation)


def test_registered_pytree_metadata_namedtuples_and_rng_leaf_dtypes_are_preserved():
    values = torch.arange(6.0, dtype=torch.float64).reshape(3, 2)
    rng = torch.arange(12, dtype=torch.uint8).reshape(3, 4)
    noise = _Envelope(
        "static metadata",
        RuntimeNoiseDraw({"eta": _Pair(values, ())}, {"stream": {"cpu": rng}}),
    )
    stimulation = StimulationDraw({"empty": {}, "events": _Pair(values + 10, ())}, {})
    selected = _index_step_input(None, None, noise, stimulation, 1)
    assert pytree.tree_structure(selected.noise) == pytree.tree_structure(noise)
    assert pytree.tree_structure(selected.stimulation) == pytree.tree_structure(
        stimulation
    )
    assert isinstance(selected.noise, _Envelope)
    assert selected.noise.label == "static metadata"
    assert isinstance(selected.noise.payload, RuntimeNoiseDraw)
    assert isinstance(selected.noise.payload.draws["eta"], _Pair)
    torch.testing.assert_close(selected.noise.payload.draws["eta"].first, values[1])
    assert selected.noise.payload.draws["eta"].second == ()
    assert tuple(selected.noise.payload.rng) == ("stream",)
    torch.testing.assert_close(selected.noise.payload.rng["stream"]["cpu"], rng[1])
    assert selected.noise.payload.rng["stream"]["cpu"].dtype == torch.uint8
    assert tuple(selected.stimulation.draws) == ("empty", "events")
    assert selected.stimulation.draws["empty"] == {}
    assert selected.stimulation.draws["events"].second == ()


@pytest.mark.parametrize("field", ("noise", "stimulation"))
def test_none_leaves_inside_a_stochastic_tree_retain_literal_indexing_failure(field):
    tapes = dict(ve=None, intra=None, noise=None, stimulation=None)
    tapes[field] = {"invalid": None}
    with pytest.raises(TypeError) as caught:
        _index_step_input(**tapes, index=0)
    assert str(caught.value) == "'NoneType' object is not subscriptable"


_ORDER = (
    "ve",
    "intra",
    "noise.first",
    "noise.again",
    "noise.rng",
    "stimulation.events",
    "stimulation.rng",
)


@pytest.mark.parametrize("first_failure", (None, *_ORDER))
def test_indexing_order_and_competing_failures_follow_packet_field_and_leaf_order(
    first_failure,
):
    events, selector = [], slice(1, 2)
    errors = {label: RuntimeError(label) for label in _ORDER}
    failure_index = (
        len(_ORDER) if first_failure is None else _ORDER.index(first_failure)
    )

    class Indexed:
        def __init__(self, label):
            self.label = label

        def __getitem__(self, index):
            assert index is selector
            events.append(self.label)
            # Every later leaf also fails, so the first error is observable.
            if _ORDER.index(self.label) >= failure_index:
                raise errors[self.label]
            return self.label

    noise = RuntimeNoiseDraw(
        {"first": Indexed("noise.first"), "again": Indexed("noise.again")},
        {"stream": {"cpu": Indexed("noise.rng")}},
    )
    stimulation = StimulationDraw(
        {"events": Indexed("stimulation.events")},
        {"stream": Indexed("stimulation.rng")},
    )
    if first_failure is None:
        result = _index_step_input(
            Indexed("ve"), Indexed("intra"), noise, stimulation, selector
        )
        assert result.ve == "ve" and result.intra == "intra"
    else:
        with pytest.raises(RuntimeError) as caught:
            _index_step_input(
                Indexed("ve"), Indexed("intra"), noise, stimulation, selector
            )
        assert caught.value is errors[first_failure]
    assert events == list(_ORDER[: failure_index + 1])


def test_noncontiguous_rows_keep_views_and_repeated_leaf_selection_is_not_memoized():
    backing = torch.arange(120.0, dtype=torch.float64)
    tape = backing.reshape(5, 4, 6).transpose(1, 2)[:, 1::2, ::2].requires_grad_()
    frozen, version = tape.detach().clone(), tape._version
    noise = RuntimeNoiseDraw({"first": tape, "again": tape}, {})
    stimulation = StimulationDraw({"events": tape}, {})
    selected = _index_step_input(tape, tape, noise, stimulation, 2)
    rows = (
        selected.ve,
        selected.intra,
        selected.noise.draws["first"],
        selected.noise.draws["again"],
        selected.stimulation.draws["events"],
    )
    assert len({id(row) for row in rows}) == 5
    for row in rows:
        torch.testing.assert_close(row, frozen[2])
        assert torch._C._is_alias_of(row, tape)
        assert row.stride() == tape.stride()[1:]
        assert row.storage_offset() == tape.storage_offset() + 2 * tape.stride(0)
        assert row.requires_grad
    assert tape._version == version
    torch.testing.assert_close(tape, frozen)
    with torch.no_grad():
        selected.ve[0, 0].add_(100)
    assert all(row[0, 0].item() == frozen[2, 0, 0].item() + 100 for row in rows)
    assert tape[2, 0, 0].item() == frozen[2, 0, 0].item() + 100


def _combine(ve, intra, noise, stimulation):
    row = _index_step_input(
        ve,
        intra,
        RuntimeNoiseDraw({"first": noise, "again": noise}, {}),
        StimulationDraw({"events": stimulation}, {}),
        1,
    )
    return (
        row.ve.square()
        + 3 * row.intra
        + row.noise.draws["first"]
        + 2 * row.noise.draws["again"]
        + 4 * row.stimulation.draws["events"]
    )


def _tapes(noncontiguous=False):
    base = (
        torch.arange(6.0, dtype=torch.float64).reshape(2, 3).t()
        if noncontiguous
        else torch.arange(6.0, dtype=torch.float64).reshape(3, 2)
    )
    inputs = tuple((base + offset).requires_grad_() for offset in (0, 10, 20, 30))
    if noncontiguous:
        assert all(not value.is_contiguous() for value in inputs)
    return inputs


@pytest.mark.parametrize("noncontiguous", (False, True))
def test_reverse_ad_reaches_only_selected_rows_and_accumulates_repeated_leaves(
    noncontiguous,
):
    inputs = _tapes(noncontiguous)
    gradients = torch.autograd.grad(_combine(*inputs).sum(), inputs)
    expected_ve = torch.zeros_like(inputs[0])
    expected_ve[1] = 2 * inputs[0][1].detach()
    expected = [expected_ve]
    for tape, coefficient in zip(inputs[1:], (3, 3, 4), strict=True):
        gradient = torch.zeros_like(tape)
        gradient[1] = coefficient
        expected.append(gradient)
    for actual, reference in zip(gradients, expected, strict=True):
        torch.testing.assert_close(actual, reference)


def test_forward_ad_preserves_selected_row_tangents():
    inputs = _tapes(noncontiguous=True)
    tangents = tuple(
        torch.full_like(tape, scale)
        for tape, scale in zip(inputs, (0.1, 0.2, 0.3, 0.4), strict=True)
    )
    value, tangent = torch.func.jvp(_combine, inputs, tangents)
    torch.testing.assert_close(
        value,
        inputs[0][1].square() + 3 * inputs[1][1] + 3 * inputs[2][1] + 4 * inputs[3][1],
    )
    torch.testing.assert_close(tangent, 0.2 * inputs[0][1] + 0.6 + 0.9 + 1.6)


@pytest.mark.parametrize("lanes", (0, 2))
def test_vmap_selects_time_rows_inside_each_lane_including_empty_batches(lanes):
    base = torch.arange(lanes * 6, dtype=torch.float64).reshape(lanes, 3, 2)
    inputs = tuple(base + offset for offset in (0, 10, 20, 30))
    actual = torch.vmap(_combine)(*inputs)
    expected = (
        inputs[0][:, 1].square()
        + 3 * inputs[1][:, 1]
        + 3 * inputs[2][:, 1]
        + 4 * inputs[3][:, 1]
    )
    torch.testing.assert_close(actual, expected)
