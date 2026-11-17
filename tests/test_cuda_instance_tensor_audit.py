"""The direct CUDA attribute audit retains bitwise inference-mode semantics."""

import pytest
import torch

from dendra.models.mechanisms import _state


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(("index", "replacement"), [(0, -2147483648), (1, 0x7FC00002)])
def test_cuda_instance_tensor_audit_rejects_bit_changes_without_host_fingerprint(
    index, replacement, monkeypatch
):
    original = _state._logical_tensor_bytes

    def no_cuda_fingerprint(tensor, **kwargs):
        assert not tensor.is_cuda
        return original(tensor, **kwargs)

    monkeypatch.setattr(_state, "_logical_tensor_bytes", no_cuda_fingerprint)
    with torch.inference_mode():
        owner = torch.nn.Module()
        owner.unregistered = torch.tensor(
            [0, 0x7FC00001], dtype=torch.int32, device="cuda"
        ).view(torch.float32)

        before = _state._python_instance_state_snapshot(owner)
        after = _state._python_instance_state_snapshot(
            owner, capture_tensor_values=False
        )
        assert before == after
        key = "<root>.unregistered"
        assert before[key].bits.data_ptr() != owner.unregistered.data_ptr()
        assert after[key].bits.data_ptr() == owner.unregistered.data_ptr()

        owner.unregistered.data.view(torch.int32)[index] = replacement
        assert before != _state._python_instance_state_snapshot(
            owner, capture_tensor_values=False
        )
