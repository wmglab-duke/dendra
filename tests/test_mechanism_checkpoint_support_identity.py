"""Runtime-checkpoint contracts for structural mechanism support identity."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn  # noqa: F401 - configure Dendra before defining mechanisms
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import (
    _SUPPORT_IDENTITY_STATE_KEY,
    MechanismHandler,
)
from dendra.models.mechanisms._support import SupportKind, SupportSpec

DTYPE = torch.float64
CORE_SHAPE = (2, 4)


class _SupportState(Mechanism):
    Mechanism.BUFFER("scratch")


class _StatelessSupport(Mechanism):
    pass


def _flat_columns(columns):
    return torch.tensor(
        [
            row * CORE_SHAPE[1] + column
            for row in range(CORE_SHAPE[0])
            for column in columns
        ],
        dtype=torch.long,
    )


def _handler(columns, *, mechanism_type=_SupportState, key=None):
    field = torch.ones(CORE_SHAPE, dtype=DTYPE)
    key = (
        _flat_columns(columns)
        if key is None
        else torch.as_tensor(key, dtype=torch.long)
    )
    local_shape = (int(key.numel()),)
    support_spec = SupportSpec.from_compiled(
        core_shape=CORE_SHAPE,
        key=key,
        is_composable=False,
        local_shape=local_shape,
    )
    mechanism = mechanism_type(
        "probe",
        torch.full_like(field, 34.0),
        field,
        local_shape,
        local_shape,
        key=key,
        support_spec=support_spec,
    )
    return MechanismHandler(
        torch.full_like(field, 34.0),
        field,
        {"probe": mechanism},
    )


def test_same_shape_different_support_is_rejected_before_state_mutation():
    source = _handler((0, 2))
    target = _handler((1, 3))
    source.probe.scratch = torch.arange(4, dtype=DTYPE)
    target.probe.scratch = torch.full((4,), 17.0, dtype=DTYPE)
    checkpoint = source.mutable_state_dict()
    live_tensor = target.probe.scratch
    expected = live_tensor.clone()

    with pytest.raises(ValueError, match="support identity mismatch.*probe"):
        target.restore_mutable_state_dict(checkpoint)

    assert target.probe.scratch is live_tensor
    torch.testing.assert_close(target.probe.scratch, expected)


def test_support_identity_covers_stateless_mechanisms():
    source = _handler((0, 2), mechanism_type=_StatelessSupport)
    target = _handler((1, 3), mechanism_type=_StatelessSupport)

    with pytest.raises(ValueError, match="support identity mismatch.*probe"):
        target.restore_mutable_state_dict(source.mutable_state_dict())


def test_compact_support_identity_does_not_duplicate_the_flat_runtime_key():
    handler = _handler((0, 2))
    checkpoint = handler.mutable_state_dict()
    entry = checkpoint[_SUPPORT_IDENTITY_STATE_KEY]["mechanisms"]["probe"]

    assert handler.probe.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert entry["flat_key"] is None


def test_compact_checkpoint_rejects_in_place_structural_key_mutation():
    handler = _handler((0, 2))
    handler.probe.key.copy_(_flat_columns((1, 3)))

    with pytest.raises(ValueError, match="Structural selector keys are immutable"):
        handler.mutable_state_dict()


def test_packed_support_identity_validates_the_exact_key_not_only_its_signature():
    packed_key = torch.tensor([0, 3, 5, 6], dtype=torch.long)
    handler = _handler((), key=packed_key)
    handler.probe.scratch = torch.full((4,), 23.0, dtype=DTYPE)
    checkpoint = handler.mutable_state_dict()
    # entry = checkpoint[_SUPPORT_IDENTITY_STATE_KEY]["mechanisms"]["probe"]
    assert handler.probe.support_spec.kind is SupportKind.PACKED_FLAT

    corrupt = copy.deepcopy(checkpoint)
    corrupt_entry = corrupt[_SUPPORT_IDENTITY_STATE_KEY]["mechanisms"]["probe"]
    corrupt_entry["flat_key"] = corrupt_entry["flat_key"].roll(1)
    live_tensor = handler.probe.scratch
    expected = live_tensor.clone()

    with pytest.raises(ValueError, match="flat key addresses different"):
        handler.restore_mutable_state_dict(corrupt)

    assert handler.probe.scratch is live_tensor
    torch.testing.assert_close(handler.probe.scratch, expected)


def test_checkpoint_without_support_identity_uses_legacy_compatibility_rule():
    source = _handler((0, 2))
    target = _handler((0, 2))
    source.probe.scratch = torch.arange(4, dtype=DTYPE)
    checkpoint = source.mutable_state_dict()
    checkpoint.pop(_SUPPORT_IDENTITY_STATE_KEY)
    target.probe.scratch = torch.full((4,), -1.0, dtype=DTYPE)

    target.restore_mutable_state_dict(checkpoint)

    torch.testing.assert_close(target.probe.scratch, source.probe.scratch)
