"""Canonical runtime packet edits invalidate checkpoint replay."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from tests.func.test_poisson_stimulation import STEPS, _clone, _close
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import _lower
from tests.func.test_stimulation_rng_sharing import _network as _borrowed_network
from tests.func.test_stimulation_rng_sharing import _population

pytestmark = pytest.mark.cpu


def _rng_slot(tree):
    """Select an authored RNG tensor without depending on stream names."""
    for name, value in tree.items():
        if torch.is_tensor(value) and value.dtype == torch.uint8:
            return tree, name
        if isinstance(value, Mapping):
            selected = _rng_slot(value)
            if selected is not None:
                return selected
    return None


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("mutation", ("value", "rebind"))
def test_runtime_draw_packet_mutation_rejects_checkpoint_replay(network, mutation):
    source = _borrowed_network()[0] if network else _population()[0]
    functional, tensors = _lower(source)
    original = _clone(functional.extract())
    global_rng = torch.get_rng_state().clone()
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    if network:
        bound = functional.bind(tensors.parameters, tensors.constants)
        amplitude = next(
            value
            for name, value in tensors.named_parameters.items()
            if name.endswith("waveform.amp")
        )
    else:
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        bound = functional.bind(tensors.parameters, prepared)
        amplitude = next(
            value
            for name, value in tensors.parameters.items()
            if name.endswith("waveform.amp")
        )
    final, _ = dn.func.run(
        bound, tensors.state, steps=STEPS, draws=packet, checkpoint_every=2
    )
    loss = (
        sum(
            values["integrator"]["v"].square().mean()
            for values in final["populations"].values()
        )
        if network
        else final["integrator"]["v"].square().mean()
    )
    assert loss.requires_grad
    slot = _rng_slot(packet.rng)
    assert slot is not None
    owner, name = slot
    if mutation == "value":
        with torch.no_grad():
            owner[name].reshape(-1)[0] ^= 1
        # PyTorch can detect mutation while unpacking the saved checkpoint
        # operands, before Dendra's explicit replay freshness check executes.
        error = RuntimeError
        message = "runtime draws changed after checkpoint forward|modified by an inplace operation"
    else:
        owner[name] = owner[name].clone()
        error = dn.func.FunctionalizationError
        message = (
            "Network parameters, constants, or input tapes changed after checkpoint forward"
            if network
            else "runtime draws changed after checkpoint forward"
        )
    with pytest.raises(error, match=message):
        torch.autograd.grad(loss, amplitude)
    _close(functional.extract(), original)
    _close(torch.get_rng_state(), global_rng)
