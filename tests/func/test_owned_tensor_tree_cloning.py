"""Compiled-loop clone endpoints own each tensor occurrence without cutting AD."""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra.func import _scan, _scan_tree, _tensor_tree_while

pytestmark = pytest.mark.cpu
_ENDPOINTS = (
    pytest.param(_scan._clone, id="population_scan"),
    pytest.param(_scan_tree._clone_tree, id="tree_scan"),
    pytest.param(_tensor_tree_while._clone_tree, id="tree_while"),
)


class _Frame(NamedTuple):
    value: torch.Tensor
    metadata: object


@pytest.mark.parametrize("clone", _ENDPOINTS)
def test_nested_structure_and_repeated_leaves_have_independent_owned_storage(clone):
    value = torch.arange(6.0, dtype=torch.float64).reshape(2, 3).requires_grad_()
    integer = torch.tensor([2, 4], dtype=torch.int64)
    marker = object()
    tree = {
        "first": [value, None],
        "frame": _Frame(value, ("keep", marker)),
        "last": {"number": 7, "integer": integer},
    }
    versions = value._version, integer._version

    result = clone(tree)

    assert list(result) == ["first", "frame", "last"]
    assert type(result["first"]) is list and type(result["frame"]) is _Frame
    assert list(result["last"]) == ["number", "integer"]
    assert result["first"][1] is None
    assert result["frame"].metadata == ("keep", marker)
    assert result["frame"].metadata[1] is marker
    assert result["last"]["number"] == 7
    assert pytree.tree_structure(result) == pytree.tree_structure(tree)
    owned = result["first"][0], result["frame"].value, result["last"]["integer"]
    assert all(out.is_contiguous() for out in owned)
    assert all(not torch._C._is_alias_of(out, value) for out in owned)
    assert all(not torch._C._is_alias_of(out, integer) for out in owned)
    assert not torch._C._is_alias_of(owned[0], owned[1])
    assert owned[0].requires_grad and owned[1].requires_grad
    assert not owned[2].requires_grad
    assert (value._version, integer._version) == versions

    with torch.no_grad():
        owned[0].add_(10.0)
        value.add_(20.0)
        integer.add_(30)
    torch.testing.assert_close(
        owned[0], torch.arange(6.0, dtype=torch.float64).reshape(2, 3) + 10
    )
    torch.testing.assert_close(
        owned[1], torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
    )
    torch.testing.assert_close(owned[2], torch.tensor([2, 4], dtype=torch.int64))


@pytest.mark.parametrize("clone", _ENDPOINTS)
def test_offset_expanded_empty_conjugate_and_negative_views_become_contiguous(clone):
    base = torch.arange(24.0, dtype=torch.float64).reshape(4, 6)
    complex_value = torch.complex(base[:2, :3], base[:2, :3] + 0.5)
    tree = {
        "offset": base[1:, 1::2],
        "expanded": torch.tensor([2.0], dtype=torch.float64).expand(2, 3),
        "empty": torch.empty(2, 0, dtype=torch.float64).expand(3, 2, 0),
        "conjugate": complex_value.conj(),
        "negative": torch._neg_view(base[:, 1:]),
    }
    assert tree["offset"].storage_offset() > 0
    assert tree["conjugate"].is_conj() and tree["negative"].is_neg()
    versions = {name: value._version for name, value in tree.items()}

    result = clone(tree)

    assert list(result) == list(tree)
    for name, value in tree.items():
        torch.testing.assert_close(result[name], value, rtol=0, atol=0)
        assert result[name].shape == value.shape
        assert result[name].dtype == value.dtype
        assert result[name].device == value.device
        assert result[name].is_contiguous() and result[name].storage_offset() == 0
        assert not torch._C._is_alias_of(result[name], value)
        assert not result[name].is_conj() and not result[name].is_neg()
        assert value._version == versions[name]


@pytest.mark.parametrize("clone", _ENDPOINTS)
def test_clone_preserves_reverse_forward_and_second_derivative_paths(clone):
    value = torch.tensor([-0.4, 0.7], dtype=torch.float64, requires_grad=True)
    result = clone({"first": value, "square": value.square(), "again": value})
    assert not result["first"].is_leaf and result["first"].grad_fn is not None
    loss = (result["first"].pow(3) + 2 * result["square"] + 3 * result["again"]).sum()
    gradient = torch.autograd.grad(loss, value, create_graph=True)[0]
    torch.testing.assert_close(gradient, 3 * value.square() + 4 * value + 3)
    second = torch.autograd.grad(gradient.sum(), value)[0]
    torch.testing.assert_close(second, 6 * value + 4)

    def evaluate(source):
        copied = clone({"first": source, "square": source.square(), "again": source})
        return copied["first"] + 2 * copied["square"] + 3 * copied["again"]

    tangent = torch.tensor([0.3, -0.2], dtype=torch.float64)
    primal, forward = torch.func.jvp(evaluate, (value,), (tangent,))
    torch.testing.assert_close(primal, 4 * value + 2 * value.square())
    torch.testing.assert_close(forward, (4 + 4 * value) * tangent)


@pytest.mark.parametrize("clone", _ENDPOINTS)
def test_clone_obeys_ambient_no_grad_and_inference_modes(clone):
    value = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    version = value._version
    with torch.no_grad():
        ordinary = clone([value, value])
        assert all(not out.requires_grad and not out.is_inference() for out in ordinary)
    with torch.inference_mode():
        inference = clone([value, value])
        assert all(not out.requires_grad and out.is_inference() for out in inference)
    for outputs in (ordinary, inference):
        assert not torch._C._is_alias_of(outputs[0], outputs[1])
        for out in outputs:
            assert not torch._C._is_alias_of(out, value)
            torch.testing.assert_close(out, value)
    assert value.requires_grad and value._version == version


@pytest.mark.parametrize("clone", _ENDPOINTS)
@pytest.mark.parametrize("kind", ("subclass", "function_mode", "dispatch_mode"))
def test_clone_keeps_per_leaf_dispatch_arguments_order_and_exception(clone, kind):
    events = []
    failure = RuntimeError("second authored clone")
    fail = False

    def record(function, args, kwargs):
        is_clone = (
            function is torch.ops.aten.clone.default
            if kind == "dispatch_mode"
            else function is torch.Tensor.clone
        )
        if is_clone:
            assert args[1:] == ()
            assert kwargs == {"memory_format": torch.contiguous_format}
            events.append(id(args[0]))
            if fail and len(events) == 2:
                raise failure

    if kind == "subclass":

        class Value(torch.Tensor):
            @classmethod
            def __torch_function__(cls, function, types, args=(), kwargs=None):
                record(function, args, kwargs or {})
                return super().__torch_function__(function, types, args, kwargs or {})

        first = torch.Tensor._make_subclass(
            Value, torch.arange(4.0, dtype=torch.float64), False
        )
        second = torch.Tensor._make_subclass(
            Value, torch.ones(2, dtype=torch.float64), False
        )
        context = nullcontext
    else:
        first, second = (
            torch.arange(4.0, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
        )
        if kind == "function_mode":

            class Mode(TorchFunctionMode):
                def __torch_function__(self, function, types, args=(), kwargs=None):
                    record(function, args, kwargs or {})
                    return function(*args, **(kwargs or {}))
        else:

            class Mode(TorchDispatchMode):
                def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                    record(function, args, kwargs or {})
                    return function(*args, **(kwargs or {}))

        context = Mode
    leaves = first, second, first
    tree = {"first": first, "optional": None, "nested": (second, [first])}

    def run(call):
        events.clear()
        try:
            with context():
                result = call()
        except RuntimeError as error:
            assert error is failure
            return None, tuple(events), error
        return result, tuple(events), None

    for fail in (False, True):
        expected, expected_events, expected_error = run(
            lambda: [
                value.clone(memory_format=torch.contiguous_format) for value in leaves
            ]
        )
        actual, actual_events, actual_error = run(lambda: clone(tree))
        assert (
            actual_events
            == expected_events
            == tuple(id(value) for value in leaves[: 2 if fail else 3])
        )
        assert actual_error is expected_error is (failure if fail else None)
        if not fail:
            flat = [actual["first"], actual["nested"][0], actual["nested"][1][0]]
            for output, reference in zip(flat, expected, strict=True):
                assert type(output) is type(reference)
                torch.testing.assert_close(output, reference)
