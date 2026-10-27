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
