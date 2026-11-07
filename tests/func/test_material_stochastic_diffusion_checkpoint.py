"""Native stochastic diffusion checkpoints retain timestep-owned coefficients."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra.models.mechanisms import DiffusionProcess
from dendra.models.mechanisms._material_stochasticity import (
    stochastic_material_workspace_slots,
)
from tests.func import test_material_diffusion as diffusion_fixture

pytestmark = pytest.mark.cpu


class _NoisyCheckpointDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(rate=0.13, target=1.4)
    DiffusionProcess.GLOBALNOISE(
        "D", distribution="uniform", low=0.3, high=0.7, seed=79
    )
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("pool", field="amount", D="D")
    DiffusionProcess.DIFFUSE("na", field="nai", D="D")
    DiffusionProcess.RELAX("pool", field="amount", rate="rate", target="target")


def _population(kind, monkeypatch, *, configure=True):
    monkeypatch.setattr(
        diffusion_fixture, "_process", lambda *_args: _NoisyCheckpointDiffusion
    )
    population = diffusion_fixture._model(kind)
    if configure:
        population.integrator._initialize(
            population, population.t.new_tensor(diffusion_fixture.DT), force=True
        )
    return population


def _step(population):
    dt = population.t.new_tensor(diffusion_fixture.DT)
    population._step(population.integrator, population, dt, None, None)
    population.t = population.t + dt


def _clone(tree):
    return _pytree.tree_map(
        lambda value: value.clone() if torch.is_tensor(value) else value, tree
    )


def _workspace_keys(checkpoint):
    return tuple(key for key in checkpoint if ".material_workspace." in key)


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_native_checkpoint_restores_stochastic_diffusion_workspace(kind, monkeypatch):
    source, target = _population(kind, monkeypatch), _population(kind, monkeypatch)
    _step(source)
    # Reconfigure after a runtime draw so the held operator cannot be recovered
    # from the target's original timestep preparation or its seed alone.
    source.integrator._initialize(
        source, source.t.new_tensor(diffusion_fixture.DT), force=True
    )
    checkpoint = _clone(source.state_dict_for_checkpoint())
    keys = _workspace_keys(checkpoint["mech"])
    assert keys
    before = target.mech.mutable_state_dict()
    assert any(not torch.equal(checkpoint["mech"][key], before[key]) for key in keys)
    target.restore_dict_from_checkpoint(checkpoint)
    _step(source)
    _step(target)
    for name in ("pool",):
        torch.testing.assert_close(
            source.mech.materials[name].amount,
            target.mech.materials[name].amount,
            atol=1e-13,
            rtol=1e-13,
        )
    torch.testing.assert_close(source.v, target.v, atol=1e-13, rtol=1e-13)
    left, right = source.mech.mutable_state_dict(), target.mech.mutable_state_dict()
    for key in keys:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


def test_legacy_stochastic_diffusion_checkpoint_retains_configured_workspace(
    monkeypatch,
):
    population = _population("path", monkeypatch)
    checkpoint = _clone(population.mech.mutable_state_dict())
    keys = _workspace_keys(checkpoint)
    assert keys
    process = next(iter(population.mech.material_processes.values()))
    before = tuple(
        (owner, name, owner._buffers[name])
        for _suffix, owner, name in stochastic_material_workspace_slots(process)
    )
    legacy = {key: value for key, value in checkpoint.items() if key not in keys}
    population.mech.restore_mutable_state_dict(legacy)
    assert all(owner._buffers[name] is value for owner, name, value in before)


def test_stochastic_diffusion_checkpoint_schema_failure_is_transactional(monkeypatch):
    population = _population("path", monkeypatch)
    checkpoint = _clone(population.mech.mutable_state_dict())
    keys = _workspace_keys(checkpoint)
    assert keys
    checkpoint[keys[-1]] = checkpoint[keys[-1]].unsqueeze(0)
    before = {name: value for name, value in population.named_buffers()}
    with pytest.raises(ValueError, match="has shape"):
        population.mech.restore_mutable_state_dict(checkpoint)
    assert all(population.get_buffer(name) is value for name, value in before.items())


@pytest.mark.parametrize("corruption", ("missing_noise", "unknown_workspace"))
def test_stochastic_diffusion_checkpoint_workspace_namespace_is_complete(
    corruption, monkeypatch
):
    population = _population("path", monkeypatch)
    checkpoint = _clone(population.mech.mutable_state_dict())
    noise_key = next(
        key
        for key in _workspace_keys(checkpoint)
        if ".material_workspace.noise." in key
    )
    if corruption == "missing_noise":
        del checkpoint[noise_key]
        error, message = KeyError, "workspace state is missing"
    else:
        checkpoint[f"{noise_key}_unknown"] = checkpoint[noise_key].clone()
        error, message = ValueError, "workspace state has unexpected"
    before = {name: value for name, value in population.named_buffers()}
    with pytest.raises(error, match=message):
        population.mech.restore_mutable_state_dict(checkpoint)
    assert all(population.get_buffer(name) is value for name, value in before.items())


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_functional_commit_preserves_held_diffusion_noise_for_native_continuation(
    kind, monkeypatch
):
    source, target = _population(kind, monkeypatch), _population(kind, monkeypatch)
    _step(source)
    source.integrator._initialize(
        source, source.t.new_tensor(diffusion_fixture.DT), force=True
    )
    _step(source)
    functional, tensors = dn.func.make_functional(source, dt=diffusion_fixture.DT)
    packet = functional.sample_noise_draw(tensors.state)
    state, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=packet),
    )
    next_packet = functional.sample_noise_draw(state)
    expected, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        state,
        dn.func.StepInput(noise=next_packet),
    )
    functional.commit_state_(target, state)
    _step(target)
    actual = functional.extract(target).state
    left, left_spec = _pytree.tree_flatten(actual)
    right, right_spec = _pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for result, reference in zip(left, right, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_cold_diffusion_source_and_commit_register_complete_workspace(
    kind, monkeypatch
):
    source = _population(kind, monkeypatch, configure=False)
    target = _population(kind, monkeypatch, configure=False)
    assert not next(iter(source.mech.material_processes.values()))._spatial_configured
    functional, tensors = dn.func.make_functional(source, dt=diffusion_fixture.DT)
    packet = functional.sample_noise_draw(tensors.state)
    state, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=packet),
    )
    functional.commit_state_(target, state)
    actual = functional.extract(target).state
    left, left_spec = _pytree.tree_flatten(actual)
    right, right_spec = _pytree.tree_flatten(state)
    assert left_spec == right_spec
    for result, reference in zip(left, right, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)


def test_loaded_state_dict_rebuilds_diffusion_noise_before_snapshot_lowering(
    monkeypatch,
):
    source = _population("path", monkeypatch)
    _step(source)
    saved = _clone(source.state_dict())
    target, native = _population("path", monkeypatch), _population("path", monkeypatch)
    target.load_state_dict(saved)
    native.load_state_dict(saved)
    assert not target.integrator.initialized
    native.integrator._initialize(
        native, native.t.new_tensor(diffusion_fixture.DT), force=True
    )
    functional, tensors = dn.func.make_functional(target, dt=diffusion_fixture.DT)
    packet = functional.sample_noise_draw(tensors.state)
    actual, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=packet),
    )
    _step(native)
    expected = functional.extract(native).state
    left, left_spec = _pytree.tree_flatten(actual)
    right, right_spec = _pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for result, reference in zip(left, right, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)
