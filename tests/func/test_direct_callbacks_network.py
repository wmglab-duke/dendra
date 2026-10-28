"""Network bindings forward callbacks without changing frame or tape inputs."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_inputs_network import _drives
from tests.func.test_inputs_network import _environment as _environment
from tests.func.test_inputs_network import _inputs
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_rng_sharing import _network as _noise_network
from tests.func.test_network_rollout import _driven_network

pytestmark = pytest.mark.cpu


class _NetworkEnergy(dn.func.FunctionalCallback):
    @staticmethod
    def _energy(state):
        return sum(
            value["integrator"]["v"].square().mean()
            for value in state["populations"].values()
        )

    def initialize(self, state, auxiliary):
        del auxiliary
        return self._energy(state), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return carry + self._energy(state), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


def _callbacks(functional):
    return functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("left.v", "right.v", "t")),
            "energy": _NetworkEnergy(),
        }
    )


def _loss(auxiliary):
    results = auxiliary["callbacks"]
    return results["energy"] + results["trace"]["left.v"].square().mean()


@pytest.mark.parametrize("compiled", (False, True))
def test_direct_network_binding_callbacks_keep_inputs_and_nonzero_gradients(compiled):
    functional, tensors = dn.func.make_functional_network(_driven_network())
    parameters = tensors.independent_parameters(trainable="*g_param")
    state = _clone(tensors.state, detached=True)
    steps = 3 if compiled else 1
    ve, intra = _drives(functional, steps)
    inputs = _inputs(ve, intra)
    callbacks = _callbacks(functional)
    local = parameters["populations"]["left"]
    conductance = local[next(name for name in local if name.endswith("g_param"))]
    targets = (conductance, ve, intra)
    expected, expected_auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        state,
        _inputs(ve, intra),
        steps=steps,
        callbacks=callbacks,
    )
    reference = torch.autograd.grad(_loss(expected_auxiliary), targets)
    owner = (
        functional.compile_rollout_chunk(steps, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(parameters, constants=tensors.constants)
    selected = (
        inputs
        if compiled
        else pytree.tree_map_only(torch.Tensor, lambda value: value[0], inputs)
    )
    actual, auxiliary = bound(state, selected, callbacks=callbacks)
    _close(actual, expected)
    _close(dict(auxiliary["callbacks"]), dict(expected_auxiliary["callbacks"]))
    gradients = torch.autograd.grad(_loss(auxiliary), targets)
    _close(gradients, reference)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    assert not hasattr(bound, "_inference_prepared")


@pytest.mark.parametrize("compiled", (False, True))
def test_direct_network_callback_resume_preserves_reducer_and_sample_boundaries(
    compiled,
):
    functional, tensors = dn.func.make_functional_network(_driven_network())
    parameters = tensors.independent_parameters(trainable="*g_param")
    width = 2 if compiled else 1
    ve, intra = _drives(functional, 2 * width)
    callbacks = _callbacks(functional)
    expected, expected_auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        tensors.state,
        _inputs(ve, intra),
        steps=2 * width,
        callbacks=callbacks,
    )
    owner = (
        functional.compile_rollout_chunk(width, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(parameters, constants=tensors.constants)

    def segment(start, stop):
        values = _inputs(ve[start:stop], intra[start:stop])
        return (
            values
            if compiled
            else pytree.tree_map_only(torch.Tensor, lambda value: value[0], values)
        )

    prefix, prefix_auxiliary = bound(
        tensors.state, segment(0, width), callbacks=callbacks
    )
    final, auxiliary = bound(
        prefix,
        segment(width, 2 * width),
        callbacks=callbacks,
        callback_state=prefix_auxiliary["callbacks"].state,
    )
    _close(final, expected)
    for name in ("left.v", "right.v", "t"):
        combined = torch.cat(
            (
                prefix_auxiliary["callbacks"]["trace"][name],
                auxiliary["callbacks"]["trace"][name],
            )
        )
        _close(combined, expected_auxiliary["callbacks"]["trace"][name])
    _close(
        auxiliary["callbacks"]["energy"],
        expected_auxiliary["callbacks"]["energy"],
    )
    assert auxiliary["callbacks"]["trace"]["t"].shape[0] == width


@pytest.mark.parametrize("compiled", (False, True))
def test_direct_network_callbacks_keep_canonical_draws_and_eager_legacy_tapes(compiled):
    functional, tensors = dn.func.make_functional_network(_noise_network())
    parameters = tensors.independent_parameters()
    state = _clone(tensors.state, detached=True)
    shape = functional.populations["left"].shape
    current = torch.full(shape, 0.02, dtype=DTYPE)
    inputs = {name: dn.func.Inputs(intra=current) for name in functional.populations}
    packet = functional.sample_runtime_draws(state)
    tape = pytree.tree_map_only(torch.Tensor, lambda value: value.unsqueeze(0), packet)
    input_tape = pytree.tree_map_only(
        torch.Tensor, lambda value: value.unsqueeze(0), inputs
    )
    callbacks = _callbacks(functional)
    expected, expected_auxiliary = functional.prepare_and_rollout(
        parameters,
        tensors.constants,
        state,
        input_tape,
        steps=1,
        callbacks=callbacks,
        draws=tape,
    )
    owner = (
        functional.compile_rollout_chunk(1, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(parameters, constants=tensors.constants)
    actual, auxiliary = bound(
        state,
        input_tape if compiled else inputs,
        callbacks=callbacks,
        draws=tape if compiled else packet,
    )
    _close(actual, expected)
    _close(dict(auxiliary["callbacks"]), dict(expected_auxiliary["callbacks"]))
    legacy = dn.func.NetworkRolloutInput(
        {
            name: dn.func.RolloutInput(intra=current.unsqueeze(0))
            for name in functional.populations
        }
    )
    compatible, compatible_auxiliary = bound(
        state, legacy, callbacks=callbacks, draws=tape
    )
    _close(compatible, expected)
    _close(
        dict(compatible_auxiliary["callbacks"]), dict(expected_auxiliary["callbacks"])
    )


def test_eager_network_callback_binding_keeps_captured_rollout_and_error_checks():
    model = _driven_network()
    functional, tensors = dn.func.make_functional_network(model)
    bound = functional.bind(tensors.parameters, constants=tensors.constants)
    callbacks = _callbacks(functional)

    def replaced(*args, **kwargs):
        raise AssertionError("binding must retain its original rollout")

    functional.prepare_and_rollout = replaced
    actual, auxiliary = bound(tensors.state, callbacks=callbacks)
    assert actual["clock"]["t"].item() == pytest.approx(DT)
    assert auxiliary["callbacks"]["trace"]["t"].shape[0] == 2
    with pytest.raises(ValueError, match="callback_state requires callbacks"):
        bound(tensors.state, callback_state={})
    with pytest.raises(TypeError, match="callbacks must be a FunctionalCallbacks"):
        bound(tensors.state, callbacks={})
    model.populations["left"]._flag_rebuild = True
    with pytest.raises(dn.func.FunctionalizationError, match="pending structural"):
        bound(tensors.state, callbacks=callbacks)
