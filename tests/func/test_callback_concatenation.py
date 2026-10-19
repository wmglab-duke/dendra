"""Internal emission borrowing must not change public recording ownership."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func._callbacks import (
    FunctionalCallback,
    FunctionalCallbacks,
    _concatenate_emissions,
    _empty_emissions,
    _tree_schema,
)
from dendra.func._runners import (
    _attach_callback_results,
    _execute_steps_with_callbacks,
    _OwnedStepBinding,
    _StepExecution,
)
from dendra.func._types import FunctionalizationError, RolloutInput


def _tree(value):
    return {"v": value, "derived": (value.square().mean(dim=-1),)}


def _leaves(value):
    return torch.utils._pytree.tree_leaves(value)


def _schema():
    return _tree_schema(
        _tree(torch.ones(1, 2, dtype=torch.float64)),
        label="test recording",
        allow_empty=False,
    )


@pytest.mark.parametrize("discover_schema", [False, True])
def test_internal_single_part_borrows_only_after_validation(
    monkeypatch, discover_schema
):
    part = _tree(torch.arange(6.0).reshape(3, 1, 2).double())
    schema = None if discover_schema else _schema()

    def unexpected_cat(*args, **kwargs):
        raise AssertionError("internal singleton assembly should not concatenate")

    monkeypatch.setattr(torch, "cat", unexpected_cat)
    actual = _concatenate_emissions(
        [None, part, None], schema, label="test recording", borrow_single=True
    )
    assert actual is part


@pytest.mark.parametrize("discover_schema", [False, True])
def test_final_single_part_is_a_fresh_copy(discover_schema):
    part = _tree(torch.arange(6.0).reshape(3, 1, 2).double())
    schema = None if discover_schema else _schema()
    first = _concatenate_emissions([part], schema, label="test recording")
    second = _concatenate_emissions([part], schema, label="test recording")
    for source, left, right in zip(
        _leaves(part), _leaves(first), _leaves(second), strict=True
    ):
        torch.testing.assert_close(left, source)
        torch.testing.assert_close(right, source)
        assert left.data_ptr() != source.data_ptr()
        assert right.data_ptr() not in (source.data_ptr(), left.data_ptr())
        left.add_(10)
        torch.testing.assert_close(right, source)


@pytest.mark.parametrize("borrow_single", [False, True])
@pytest.mark.parametrize("parts", [[], [None], [None, None]])
def test_absent_emissions_keep_none_or_explicit_empty_schema(parts, borrow_single):
    assert (
        _concatenate_emissions(
            parts, None, label="test recording", borrow_single=borrow_single
        )
        is None
    )
    actual = _concatenate_emissions(
        parts, _schema(), label="test recording", borrow_single=borrow_single
    )
    assert actual["v"].shape == (0, 1, 2)
    assert actual["derived"][0].shape == (0, 1)
    assert all(leaf.dtype == torch.float64 for leaf in _leaves(actual))


def test_explicit_zero_length_part_can_be_borrowed():
    part = _empty_emissions(_schema())
    assert (
        _concatenate_emissions(
            [part], _schema(), label="test recording", borrow_single=True
        )
        is part
    )


@pytest.mark.parametrize("borrow_single", [False, True])
@pytest.mark.parametrize("discover_schema", [False, True])
def test_multiple_parts_preserve_nested_tree_and_concatenate(
    monkeypatch, borrow_single, discover_schema
):
    parts = [
        _tree(torch.arange(4.0).reshape(2, 1, 2).double()),
        _tree(torch.arange(4.0, 10.0).reshape(3, 1, 2).double()),
    ]
    schema = None if discover_schema else _schema()
    original = torch.cat
    calls = []

    def record_cat(tensors, *args, **kwargs):
        calls.append(len(tensors))
        return original(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", record_cat)
    actual = _concatenate_emissions(
        parts, schema, label="test recording", borrow_single=borrow_single
    )
    assert calls == [2, 2]
    for index, leaf in enumerate(_leaves(actual)):
        expected = original([_leaves(part)[index] for part in parts])
        torch.testing.assert_close(leaf, expected)
        assert all(leaf.data_ptr() != source.data_ptr() for source in _leaves(parts))


@pytest.mark.parametrize(
    "invalid",
    [
        {"v": torch.ones(3, 1, 2)},
        {"v": torch.ones(3, 1, 3).double(), "derived": (torch.ones(3, 1).double(),)},
        {"v": torch.ones(3, 1, 2), "derived": (torch.ones(3, 1).double(),)},
        {"v": torch.ones(1, 2).double(), "derived": (torch.ones(3, 1).double(),)},
        {"v": "not a tensor", "derived": (torch.ones(3, 1).double(),)},
    ],
)
def test_borrow_rejects_invalid_bound_singleton_before_returning(invalid):
    with pytest.raises(FunctionalizationError):
        _concatenate_emissions(
            [invalid], _schema(), label="test recording", borrow_single=True
        )


@pytest.mark.parametrize("invalid", [{}, {"v": 1}, {"v": torch.tensor(1.0)}])
def test_borrow_rejects_invalid_discovered_singleton(invalid):
    with pytest.raises(FunctionalizationError, match="Tensor"):
        _concatenate_emissions(
            [invalid], None, label="test recording", borrow_single=True
        )


@pytest.mark.parametrize("count", [1, 2])
def test_borrow_and_materialize_preserve_first_and_second_derivatives(count):
    actuals = []
    for borrow_single in (False, True):
        value = torch.arange(1.0, 9.0, dtype=torch.float64).requires_grad_()
        parts = [_tree(piece.reshape(-1, 1, 2)) for piece in value.chunk(count)]
        intermediate = _concatenate_emissions(
            parts, _schema(), label="test recording", borrow_single=borrow_single
        )
        public = _concatenate_emissions(
            [intermediate], _schema(), label="test recording"
        )
        loss = sum(leaf.square().sum() for leaf in _leaves(public))
        first = torch.autograd.grad(loss, value, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), value)[0]
        actuals.append((public, first, second))
    for actual, expected in zip(_leaves(actuals[1]), _leaves(actuals[0]), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _NestedRecording(FunctionalCallback):
    def __init__(self, emit_initial):
        self.emit_initial = emit_initial

    def initialize(self, state, auxiliary):
        del auxiliary
        return state["v"], _tree(state["v"]) if self.emit_initial else None

    def update(self, carry, state, auxiliary):
        del carry, auxiliary
        return state["v"], _tree(state["v"])

    def finalize(self, carry, emissions):
        if emissions is None:
            return _tree(carry.unsqueeze(0)[:0])
        return emissions


class _NoEmission(FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return carry + state["v"].mean(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


def _callbacks(state, *, emit_initial):
    # Test the callback/runner boundary without compilation or a model fixture.
    # Real compiled and checkpointed runners have separate integration coverage.
    callbacks = object.__new__(FunctionalCallbacks)
    callbacks._state_token = object()
    callbacks._plans = tuple(
        FunctionalCallbacks._probe(name, callback, state)
        for name, callback in (
            ("trace", _NestedRecording(emit_initial)),
            ("sum", _NoEmission()),
        )
    )
    return callbacks


class _CallbackChunk:
    def __init__(self, steps, callbacks, captured):
        self.steps = steps
        self.callbacks = callbacks
        self.captured = captured

    def _rollout_values(self, parameters, prepared, state, inputs, carries):
        del parameters, prepared
        emissions = []
        for index in range(self.steps):
            state = {"v": state["v"] + inputs.intra[index]}
            emitted = {}
            next_carries = {}
            for plan in self.callbacks._plans:
                carry, value = plan.callback.update(carries[plan.name], state, {})
                next_carries[plan.name] = carry
                if value is not None:
                    emitted[plan.name] = value
            carries = next_carries
            emissions.append(emitted)
        stacked = self.callbacks._stack(emissions)
        self.captured.append(stacked)
        return state, {}, carries, stacked


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("emit_initial", [False, True])
@pytest.mark.parametrize("steps,width,inner_parts", [(4, 4, 1), (5, 3, 2)])
def test_compiled_callback_assembly_borrows_only_intermediate_single_chunk(
    monkeypatch, resume, emit_initial, steps, width, inner_parts
):
    state = {"v": torch.ones(1, 2, dtype=torch.float64)}
    callbacks = _callbacks(state, emit_initial=emit_initial)
    callback_state, initial = callbacks._initialize(state)
    parts = []
    if not resume and initial is not None:
        parts.append(callbacks._stack([initial]))
    captured = []
    kernels = {
        size: _CallbackChunk(size, callbacks, captured)
        for size in {width, steps % width or width}
    }
    execution = _StepExecution(
        public_step=None,
        binding=_OwnedStepBinding(
            functional=None,
            parameters={},
            prepared=None,
            compiled_chunk=SimpleNamespace(steps=width),
        ),
        prepared_values={},
        compiled_chunks=kernels,
    )
    calls = []
    original = torch.cat

    def record_cat(tensors, *args, **kwargs):
        calls.append(len(tensors))
        return original(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", record_cat)
    final, auxiliary, callback_state, emitted = _execute_steps_with_callbacks(
        execution,
        state,
        RolloutInput(intra=torch.ones(steps, 1, 2, dtype=torch.float64)),
        steps,
        callbacks,
        callback_state,
    )
    assert calls == ([] if inner_parts == 1 else [inner_parts, inner_parts])
    if inner_parts == 1:
        assert emitted["trace"] is captured[0]["trace"]
    parts.append(emitted)
    calls.clear()
    result = _attach_callback_results(auxiliary, callbacks, callback_state, parts)
    assert calls == [len(parts), len(parts)]
    public = result["callbacks"]["trace"]
    assert public["v"].shape[0] == steps + bool(emit_initial and not resume)
    for source, output in zip(_leaves(emitted["trace"]), _leaves(public), strict=True):
        assert source.data_ptr() != output.data_ptr()
    assert public["v"].data_ptr() != final["v"].data_ptr()
    snapshot = [leaf.clone() for leaf in _leaves(public)]
    later = _attach_callback_results(auxiliary, callbacks, callback_state, parts)
    for output in _leaves(later["callbacks"]["trace"]):
        output.add_(1)
    for output, expected in zip(_leaves(public), snapshot, strict=True):
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
