"""Checkpoint boundaries and concatenated physical regions retain their contracts."""

from __future__ import annotations

import io

import pytest
import torch

import dendra as dn
from tests.func.test_callback_checkpoints import _case
from tests.func.test_callback_checkpoints import _environment as _environment
from tests.func.test_network_recorder_regions import DT, _network

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("operation", ("export", "restore"))
@pytest.mark.parametrize("context", ("grad", "forward_ad"))
def test_portable_callback_operations_reject_transform_contexts(operation, context):
    _, functional, tensors = _case()
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v",))})
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    carry = auxiliary["callbacks"].state
    packet = callbacks.export_state(carry)

    def work(value):
        if operation == "export":
            callbacks.export_state(carry)
        else:
            callbacks.restore_state(packet)
        return value.square()

    value = torch.tensor(0.5)
    with pytest.raises(dn.func.FunctionalizationError, match="outside torch.compile"):
        if context == "grad":
            torch.func.grad(work)(value)
        else:
            with torch.autograd.forward_ad.dual_level():
                work(torch.autograd.forward_ad.make_dual(value, torch.ones_like(value)))
    # Rejected cold operations do not invalidate ordinary continuation.
    callbacks.restore_state(packet)


@pytest.mark.parametrize("packed", (False, True))
def test_named_network_region_checkpoint_restores_into_reconstructed_geometry(packed):
    model, receiver = _network(packed=packed), _network(packed=packed)
    source, first_tensors = dn.func.make_functional_network(model)
    target, second_tensors = dn.func.make_functional_network(receiver)
    name = "packed" if packed else "left"

    def specifications(network):
        region = network.packed.left.soma if packed else network.left.soma
        return {"trace": dn.func.Recorder((f"{name}.v", "t"), region=region)}

    callbacks = source.make_callbacks(specifications(model))
    receiving_callbacks = target.make_callbacks(specifications(receiver))
    prefix, first = source.prepare_and_rollout(
        first_tensors.parameters,
        first_tensors.constants,
        first_tensors.state,
        steps=2,
        callbacks=callbacks,
    )
    buffer = io.BytesIO()
    torch.save(callbacks.export_state(first["callbacks"].state), buffer)
    buffer.seek(0)
    restored = receiving_callbacks.restore_state(torch.load(buffer, weights_only=True))
    final, second = target.prepare_and_rollout(
        second_tensors.parameters,
        second_tensors.constants,
        prefix,
        steps=2,
        callbacks=receiving_callbacks,
        callback_state=restored,
    )
    expected, whole = source.prepare_and_rollout(
        first_tensors.parameters,
        first_tensors.constants,
        first_tensors.state,
        steps=4,
        callbacks=callbacks,
    )
    torch.testing.assert_close(final, expected)
    for field in (f"{name}.v", "t"):
        observed = torch.cat(
            (first["callbacks"]["trace"][field], second["callbacks"]["trace"][field])
        )
        assert observed.shape[0] == 5
        torch.testing.assert_close(observed, whole["callbacks"]["trace"][field])
    assert final["clock"]["t"].item() == pytest.approx(4 * DT)
