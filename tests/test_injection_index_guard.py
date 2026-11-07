"""Warm intracellular coordinate guards avoid walking unchanged tuples."""

from __future__ import annotations

import builtins
import sys
from collections import Counter

import pytest
import torch

import dendra as dn
from dendra.models.stim.intra import Intra
from tests.test_injection_assembly import _model, _native, _ordered_reference

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
        yield


def _case(count=3):
    model = _model([((slice(None), index), index + 1.0) for index in range(count)])
    intra = _native(model)
    tapes, indices = intra.init(torch.tensor([0.0], dtype=torch.float64))
    return intra, [tape[..., 0] for tape in tapes], indices


def _observe_guard_calls(intra, samples, indices):
    """Observe the original method boundary without replacing its execution."""
    counts = Counter()
    previous = sys.getprofile()
    code = Intra._prepare_assembly.__code__

    def observe(frame, event, arg):
        if event == "c_call" and frame.f_code is code:
            for function in (builtins.any, builtins.all, builtins.len):
                if arg is function:
                    counts[function.__name__] += 1
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        prepared = intra._prepare_assembly(samples, indices)
    finally:
        sys.setprofile(previous)
    return prepared, counts


def test_unchanged_coordinate_tuples_skip_member_guards_but_check_versions():
    lengths = []
    for count in (2, 5):
        intra, samples, indices = _case(count)
        original = intra._prepare_assembly(samples, indices)
        # Replacing only the outer list keeps the exact cached coordinate tuples.
        actual, calls = _observe_guard_calls(intra, samples, list(indices))
        assert actual is original
        assert calls["any"] == 0
        assert calls["all"] == 1
        lengths.append(calls["len"])
    # The remaining lengths concern outer inputs/cache size, not tuple members.
    assert lengths[0] == lengths[1]


def test_equal_rebound_tuples_keep_member_checks_and_reuse_prepared_indices():
    intra, samples, indices = _case()
    original = intra._prepare_assembly(samples, indices)
    _, unchanged = _observe_guard_calls(intra, samples, indices)
    replacement = [tuple(list(index)) for index in indices]
    assert all(left is not right for left, right in zip(replacement, indices))
    actual, rebound = _observe_guard_calls(intra, samples, replacement)
    assert actual is original
    assert rebound["any"] == len(indices)
    assert rebound["len"] == unchanged["len"] + 2 * len(indices)
    assert rebound["all"] == 1


@pytest.mark.parametrize("replacement", ("clone", "independent_storage_alias"))
def test_outer_index_replacement_refreshes_the_watched_coordinate_identity(
    replacement,
):
    intra, samples, indices = _case()
    original = intra._prepare_assembly(samples, indices)
    coordinate = indices[0][-1]
    if replacement == "clone":
        changed = coordinate.clone()
    else:
        changed = torch.empty(0, dtype=coordinate.dtype).set_(
            coordinate.untyped_storage(),
            coordinate.storage_offset(),
            coordinate.shape,
            coordinate.stride(),
        )
        assert changed._base is None
    # The public outer list stays identical while a coordinate Tensor changes.
    indices[0] = (*indices[0][:-1], changed)
    refreshed = intra._prepare_assembly(samples, indices)
    assert refreshed is not original
    assert refreshed.indices[0] is indices[0]
    torch.testing.assert_close(
        intra(samples, indices),
        _ordered_reference(intra, samples, indices),
        rtol=0,
        atol=0,
    )

    version = coordinate._version
    changed.fill_(1)
    # Independent storage aliases change values without the old Tensor's counter.
    assert coordinate._version == version
    after_mutation = intra._prepare_assembly(samples, indices)
    assert after_mutation is not refreshed
    torch.testing.assert_close(
        intra(samples, indices),
        _ordered_reference(intra, samples, indices),
        rtol=0,
        atol=0,
    )

    dispatches = []

    class AuthoredCoordinate(torch.Tensor):
        @property
        def _version(self):
            raise AssertionError("coordinate guards must not read authored versions")

        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            dispatches.append(function)
            return super().__torch_function__(function, types, args, kwargs or {})

    gradient_samples = [sample.detach().clone().requires_grad_() for sample in samples]
    frozen_samples = [sample.detach().clone() for sample in gradient_samples]
    frozen_indices = [
        tuple(item.detach().clone() for item in index) for index in indices
    ]
    authored = changed.as_subclass(AuthoredCoordinate)
    indices[0] = (*indices[0][:-1], authored)
    dispatches.clear()
    # A warm plain-coordinate cache cannot certify replacement authored indices.
    # Fall back before consulting their metadata, mutation counters or dispatch.
    assert intra._prepare_assembly(gradient_samples, indices) is indices
    assert dispatches == []

    actual = intra(gradient_samples, indices)
    assert torch.Tensor.index_put_ in dispatches
    expected = _ordered_reference(intra, gradient_samples, frozen_indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.arange(1, actual.numel() + 1, dtype=actual.dtype).reshape_as(actual)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), gradient_samples)
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), gradient_samples
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert torch.isfinite(actual_gradient).all() and (actual_gradient != 0).all()
    for sample, frozen in zip(gradient_samples, frozen_samples, strict=True):
        torch.testing.assert_close(sample, frozen, rtol=0, atol=0)
    for index, frozen in zip(indices, frozen_indices, strict=True):
        for item, original_item in zip(index, frozen, strict=True):
            torch.testing.assert_close(
                item.as_subclass(torch.Tensor), original_item, rtol=0, atol=0
            )
