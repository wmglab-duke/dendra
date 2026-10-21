"""Host Network scheduling accepts public mappings and inference-owned tapes."""

from contextlib import nullcontext
from types import MappingProxyType

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import DTYPE, _close, _stim_network
from tests.func.test_network_rollout import _driven_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _known_compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _voltage_loss(state):
    return sum(
        value["integrator"]["v"].square().sum()
        for value in state["populations"].values()
    )


def _ordinary(tree):
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


@pytest.mark.parametrize("checkpoint_every", (None, 3))
def test_public_mapping_proxy_input_tapes_preserve_values_and_gradients(
    checkpoint_every,
):
    model = _driven_network()
    functional, tensors = dn.func.make_functional_network(model)
    steps = 5
    shape = (steps, *model.left.shape)
    current = (
        torch.linspace(-0.7, 1.1, steps * model.left.v.numel(), dtype=DTYPE)
        .reshape(shape)
        .mul(1e-8)
        .requires_grad_()
    )
    expected_current = current.detach().clone().requires_grad_()
    inputs = dn.func.NetworkRolloutInput(
        populations=MappingProxyType({"left": dn.func.RolloutInput(intra=current)})
    )
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.NetworkRolloutInput(
            populations={"left": dn.func.RolloutInput(intra=expected_current)}
        ),
        steps=steps,
    )
    actual, _ = functional.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        steps=steps,
        chunklength=2,
        checkpoint_every=checkpoint_every,
    )
    _close(actual, expected)
    key = "integrator.mech.mechanisms.pas.g_param"
    conductance = tensors.parameters["populations"]["left"][key]
    observed = torch.autograd.grad(_voltage_loss(actual), (current, conductance))
    oracle = torch.autograd.grad(
        _voltage_loss(expected), (expected_current, conductance)
    )
    _close(observed, oracle)
    assert all(value.abs().sum() > 0 for value in observed)


def _stim_inputs(functional, model, state, steps):
    with torch.inference_mode():
        draws = functional.sample_draw_tape(state, steps)
        current = (
            torch.linspace(
                -0.1, 0.4, steps * model.cell.v.numel(), dtype=DTYPE
            ).reshape(steps, *model.cell.shape)
            * 1e-5
        )
    assert current.is_inference() and draws.exp_rand.is_inference()
    assert draws.rng_state.is_inference()
    return dn.func.NetworkRolloutInput(
        populations={"cell": dn.func.RolloutInput(intra=current)}, netstim=draws
    )


@pytest.mark.parametrize("compiled", (False, True))
def test_checkpoint_snapshots_inference_owned_draws_and_drives_for_parameter_gradients(
    compiled,
):
    model = _stim_network()
    model.netstim.noise.requires_grad_()
    functional, tensors = dn.func.make_functional_network(model)
    inputs = _stim_inputs(functional, model, tensors.state, 9)
    reference_inputs = _ordinary(inputs)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder("cell.v")})
    expected, expected_aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        reference_inputs,
        steps=9,
        callbacks=callbacks,
    )
    chunk = (
        functional.compile_rollout_chunk(2, execution="scan", backend="aot_eager")
        if compiled
        else None
    )
    actual, actual_aux = functional.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        steps=9,
        chunklength=2,
        checkpoint_every=3,
        callbacks=callbacks,
        compiled_chunk=chunk,
    )
    _close(actual, expected)
    _close(actual_aux["callbacks"]["trace"], expected_aux["callbacks"]["trace"])
    # Inference tensors have no version counter. Replay uses independently
    # materialized checkpoint operands, even if their public owners are edited.
    with torch.inference_mode():
        inputs.populations["cell"].intra.add_(1.0)
        inputs.netstim.exp_rand.add_(0.5)
    noise = tensors.parameters["netstim"]["parameters"]["noise"]
    weights = next(iter(tensors.parameters["synapses"].values()))["weights"]
    actual_loss = actual_aux["callbacks"]["trace"]["cell.v"].square().sum()
    expected_loss = expected_aux["callbacks"]["trace"]["cell.v"].square().sum()
    observed = torch.autograd.grad(actual_loss, (noise, weights))
    oracle = torch.autograd.grad(expected_loss, (noise, weights))
    _close(observed, oracle)
    assert all(value.abs().sum() > 0 for value in observed)


@pytest.mark.parametrize("mode", ("no_grad", "eval"))
@pytest.mark.parametrize("compiled", (False, True))
def test_disabled_gradients_bypass_checkpoint_with_inference_owned_tapes(
    monkeypatch, mode, compiled
):
    model = _stim_network()
    if mode == "eval":
        model.eval()
    functional, tensors = dn.func.make_functional_network(model)
    inputs = _stim_inputs(functional, model, tensors.state, 5)
    chunk = (
        functional.compile_rollout_chunk(2, execution="scan", backend="aot_eager")
        if compiled
        else None
    )

    def unexpected_checkpoint(*_args, **_kwargs):
        raise AssertionError("checkpoint must be bypassed when gradients are disabled")

    monkeypatch.setattr("dendra.func._runners.checkpoint", unexpected_checkpoint)
    with torch.no_grad() if mode == "no_grad" else nullcontext():
        expected, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            _ordinary(inputs),
            steps=5,
        )
        actual, _ = functional.run(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs,
            steps=5,
            chunklength=2,
            checkpoint_every=3,
            compiled_chunk=chunk,
        )
    _close(actual, expected)
    assert not actual["populations"]["cell"]["integrator"]["v"].requires_grad
