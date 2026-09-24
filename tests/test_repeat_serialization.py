"""Repeat's new outer edge parameter preserves versioned saved waveforms."""

from __future__ import annotations

import copy
import pickle

import pytest
import torch

import dendra as dn
from dendra.models.stim.waveform.core import _repeat as RepeatWaveform

pytestmark = pytest.mark.cpu


def _repeat(*, dtype=torch.float64, trainable=True, tau=0.01):
    with dn.ctx(REQUIRE_GRAD=int(trainable), DTYPE=dtype):
        return dn.constant(value=2.0).repeat(freq=1.25, delay=0.1, off=0.8, tau=tau)


def _legacy_state_dict(module):
    state = copy.deepcopy(module.state_dict())
    for name, child in module.named_modules():
        if isinstance(child, RepeatWaveform):
            state.pop(f"{name}.tau" if name else "tau")
            state._metadata[name]["version"] = 1
    return state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("assign", [False, True])
def test_version_one_state_dict_restores_default_tau(dtype, assign):
    original = _repeat(dtype=dtype)
    state = _legacy_state_dict(original)
    target = _repeat(dtype=dtype, tau=0.3)
    restored = target.load_state_dict(state, strict=True, assign=assign)

    assert not restored.missing_keys and not restored.unexpected_keys
    assert "tau" not in state  # Loading does not modify the caller's checkpoint.
    assert target.tau.dtype == target.freq.dtype == dtype
    assert target.tau.device == target.freq.device
    torch.testing.assert_close(target.tau, target.freq.new_tensor(0.01))
    times = torch.tensor([0.0, 0.1, 0.4, 0.8], dtype=dtype)
    torch.testing.assert_close(target(times), original(times), rtol=0, atol=0)
    assert target.state_dict()._metadata[""]["version"] == 2


def test_nested_legacy_repeats_restore_each_prefixed_tau():
    original = torch.nn.ModuleDict({"stimulus": _repeat().repeat(0.75, tau=0.06)})
    state = _legacy_state_dict(original)
    target = torch.nn.ModuleDict({"stimulus": _repeat(tau=0.2).repeat(0.75, tau=0.3)})
    target.load_state_dict(state, strict=True)

    for repeat in (target["stimulus"], target["stimulus"].waveform):
        torch.testing.assert_close(repeat.tau, repeat.freq.new_tensor(0.01))
    assert "stimulus.tau" not in state
    assert "stimulus.waveform.tau" not in state


def test_legacy_assign_restore_uses_checkpoint_frequency_dtype():
    state = _legacy_state_dict(_repeat(dtype=torch.float64))
    target = _repeat(dtype=torch.float32)
    target.load_state_dict(state, strict=True, assign=True)
    assert target.tau.dtype == target.freq.dtype == torch.float64
    torch.testing.assert_close(target.tau, target.freq.new_tensor(0.01))


@pytest.mark.parametrize("metadata", ["current", "absent"])
def test_missing_tau_is_not_silently_repaired_without_legacy_metadata(metadata):
    original = _repeat()
    state = copy.deepcopy(original.state_dict())
    state.pop("tau")
    if metadata == "absent":
        state = dict(state)

    with pytest.raises(RuntimeError, match='Missing key.*"tau"'):
        _repeat().load_state_dict(state, strict=True)


def test_legacy_migration_does_not_hide_other_missing_parameters():
    state = _legacy_state_dict(_repeat())
    state.pop("delay")
    with pytest.raises(RuntimeError, match='Missing key.*"delay"'):
        _repeat().load_state_dict(state, strict=True)


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_legacy_pickle_and_deepcopy_register_tau_with_matching_tensor_policy(
    trainable, dtype, restore
):
    original = _repeat(dtype=dtype, trainable=trainable)
    del original.tau
    original.params.pop("tau")
    restored = restore(original)

    assert "tau" in dict(restored.named_parameters())
    assert restored.tau.is_leaf
    assert restored.tau.requires_grad is trainable
    assert restored.tau.dtype == restored.freq.dtype == dtype
    assert restored.tau.device == restored.freq.device
    torch.testing.assert_close(restored.tau, restored.freq.new_tensor(0.01))
    times = torch.tensor([0.0, 0.1, 0.4, 0.8], dtype=dtype)
    torch.testing.assert_close(
        restored(times), torch.tensor([0.0, 2.0, 2.0, 0.0], dtype=dtype)
    )
    assert not hasattr(original, "tau")
    if trainable:
        restored(times).sum().backward()
        assert restored.tau.grad is not None
        assert torch.isfinite(restored.tau.grad)


def test_current_checkpoints_and_pickles_preserve_explicit_tau():
    original = _repeat(tau=0.07)
    target = _repeat(tau=0.3)
    target.load_state_dict(original.state_dict(), strict=True)
    for restored in (
        target,
        copy.deepcopy(original),
        pickle.loads(pickle.dumps(original)),
    ):
        torch.testing.assert_close(restored.tau, original.tau, rtol=0, atol=0)
        assert restored.tau.requires_grad == original.tau.requires_grad
