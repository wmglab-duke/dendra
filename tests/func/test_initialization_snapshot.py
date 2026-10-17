"""Purity snapshots compare exact bytes independently of floating semantics."""

import pytest
import torch

from tests.func.test_functional_initialization import (
    _assert_tensor_snapshot,
    _tensor_snapshot,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("layout", ["scalar", "contiguous", "noncontiguous", "empty"])
def test_unchanged_snapshot_accepts_exact_nan_payloads_and_all_layouts(layout):
    values = torch.tensor([0x7FC00001, 0x7FC00002, 0, -2147483648], dtype=torch.int32)
    value = values.view(torch.float32)
    if layout == "scalar":
        value = value[0]
    elif layout == "noncontiguous":
        value = value.repeat(2).reshape(2, 4).t()
        assert not value.is_contiguous()
    elif layout == "empty":
        value = value[:0]
    _assert_tensor_snapshot(value, _tensor_snapshot(value))


@pytest.mark.parametrize("change", ["nan_payload", "signed_zero", "finite"])
def test_snapshot_rejects_bit_changes_even_without_version_updates(change):
    bits = {"nan_payload": 0x7FC00001, "signed_zero": 0, "finite": 0x3F800000}
    replacements = {
        "nan_payload": 0x7FC00002,
        "signed_zero": -2147483648,
        "finite": 0x40000000,
    }
    value = torch.tensor([bits[change]], dtype=torch.int32).view(torch.float32)
    snapshot = _tensor_snapshot(value)
    # Deliberately bypass the source version counter to prove the byte check
    # catches mutations independently of the identity/storage/version checks.
    value.data.view(torch.int32)[0] = replacements[change]
    assert value._version == snapshot[2]
    with pytest.raises(AssertionError):
        _assert_tensor_snapshot(value, snapshot)


@pytest.mark.parametrize("change", ["dtype", "shape"])
def test_snapshot_preserves_metadata_checks_when_bytes_match(change):
    value = torch.zeros(2, dtype=torch.float32)
    snapshot = _tensor_snapshot(value)
    if change == "dtype":
        value.data = value.data.view(torch.int32)
    else:
        value.data = value.data.reshape(1, 2)
    assert value._version == snapshot[2]
    with pytest.raises(AssertionError):
        _assert_tensor_snapshot(value, snapshot)
