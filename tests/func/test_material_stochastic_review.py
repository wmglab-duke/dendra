"""Reviewed stochastic workspaces survive cold native lifecycle boundaries."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import _handler
from tests.func.test_material_diffusion import DT
from tests.func.test_material_stochastic_diffusion_checkpoint import (
    _clone,
    _population,
    _step,
    _workspace_keys,
)

pytestmark = pytest.mark.cpu


def _advanced_source(kind, monkeypatch):
    source = _population(kind, monkeypatch).eval()
    _step(source)
    source.integrator._initialize(source, source.t.new_tensor(DT), force=True)
    _step(source)
    process = next(iter(source.mech.material_processes.values()))
    assert not torch.equal(process.D, process._material_held_noise_D)
    return source


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_cold_native_restore_retains_workspace_through_public_step(kind, monkeypatch):
    source = _advanced_source(kind, monkeypatch)
    target = _population(kind, monkeypatch, configure=False).eval()
    integrator = target.integrator
    target.restore_dict_from_checkpoint(_clone(source.state_dict_for_checkpoint()))
    process = next(iter(target.mech.material_processes.values()))
    operators = process._spatial_operators
    held = process._material_held_noise_D.clone()
    assert target.integrator is integrator
    assert not integrator.initialized
    source.step(dt=DT)
    target.step(dt=DT)
    assert process._spatial_operators is operators
    torch.testing.assert_close(process._material_held_noise_D, held, atol=0, rtol=0)
    torch.testing.assert_close(
        target.mech.materials["pool"].amount,
        source.mech.materials["pool"].amount,
        atol=1e-12,
        rtol=1e-12,
    )
    # A subsequent explicit timestep setup retains normal native semantics.
    process.set_dt(DT)
    assert process._spatial_operators is not operators
    torch.testing.assert_close(process._material_held_noise_D, process.D)


def test_cold_native_restore_changed_timestep_rebuilds_normally(monkeypatch):
    source = _advanced_source("path", monkeypatch)
    target = _population("path", monkeypatch, configure=False).eval()
    target.restore_dict_from_checkpoint(_clone(source.state_dict_for_checkpoint()))
    process = next(iter(target.mech.material_processes.values()))
    operators = process._spatial_operators
    current = process.D.clone()
    target.step(dt=DT / 2)
    assert process._spatial_operators is not operators
    torch.testing.assert_close(process._material_held_noise_D, current)


def test_lowering_cold_restored_source_configures_its_isolated_clone(monkeypatch):
    source = _advanced_source("path", monkeypatch)
    target = _population("path", monkeypatch, configure=False).eval()
    target.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    process = next(iter(target.mech.material_processes.values()))
    operators = process._spatial_operators
    functional, tensors = dn.func.make_functional(target, dt=DT)
    packet = functional.sample_noise_draw(tensors.state)
    expected, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=packet),
    )
    assert process._spatial_operators is operators
    target.step(dt=DT)
    torch.testing.assert_close(
        target.mech.materials["pool"].amount,
        expected["materials"]["pool"]["amount"],
        atol=1e-12,
        rtol=1e-12,
    )


def test_regular_state_dict_load_invalidates_pending_workspace_handoff(monkeypatch):
    source = _advanced_source("path", monkeypatch)
    target = _population("path", monkeypatch, configure=False).eval()
    target.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    process = next(iter(target.mech.material_processes.values()))
    assert "_material_restored_workspace_dt" in process.__dict__
    target.load_state_dict(_clone(source.state_dict()))
    assert not process._spatial_configured
    current = process.D.clone()
    target.step(dt=DT)
    assert process._spatial_configured
    assert "_material_restored_workspace_dt" not in process.__dict__
    torch.testing.assert_close(process._material_held_noise_D, current)


def test_successive_restore_retains_latest_workspace_timestep(monkeypatch):
    first = _advanced_source("path", monkeypatch)
    latest = _advanced_source("path", monkeypatch)
    latest.integrator._initialize(latest, latest.t.new_tensor(DT / 2), force=True)
    latest.step(dt=DT / 2)
    target = _population("path", monkeypatch, configure=False).eval()
    target.restore_dict_from_checkpoint(first.state_dict_for_checkpoint())
    target.restore_dict_from_checkpoint(latest.state_dict_for_checkpoint())
    process = next(iter(target.mech.material_processes.values()))
    operators = process._spatial_operators
    held = process._material_held_noise_D.clone()
    latest.step(dt=DT / 2)
    target.step(dt=DT / 2)
    assert process._spatial_operators is operators
    torch.testing.assert_close(process._material_held_noise_D, held, atol=0, rtol=0)
    torch.testing.assert_close(
        target.mech.materials["pool"].amount,
        latest.mech.materials["pool"].amount,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("corruption", ("missing", "unknown", "shape"))
def test_cold_native_restore_preflight_is_transactional(corruption, monkeypatch):
    source = _advanced_source("path", monkeypatch)
    target = _population("path", monkeypatch, configure=False).eval()
    checkpoint = _clone(source.state_dict_for_checkpoint())
    keys = _workspace_keys(checkpoint["mech"])
    key = next(key for key in keys if key.endswith(".main"))
    if corruption == "missing":
        del checkpoint["mech"][key]
    elif corruption == "unknown":
        checkpoint["mech"][f"{key}_unknown"] = checkpoint["mech"][key]
    else:
        checkpoint["mech"][key] = checkpoint["mech"][key].unsqueeze(0)
    process = next(iter(target.mech.material_processes.values()))
    operators = process._spatial_operators
    before = dict(target.named_buffers())
    with pytest.raises((KeyError, ValueError)):
        target.mech.restore_mutable_state_dict(checkpoint["mech"])
    assert not process._spatial_configured
    assert process._spatial_operators is operators
    assert all(target.get_buffer(name) is value for name, value in before.items())
    assert "_material_held_noise_D" not in process._buffers


def test_cold_native_restore_publication_failure_rolls_back_owners(monkeypatch):
    source = _advanced_source("path", monkeypatch)
    target = _population("path", monkeypatch, configure=False).eval()
    process = next(iter(target.mech.material_processes.values()))
    operators = process._spatial_operators
    before = dict(target.named_buffers())
    stochastic = _clone(target.mech.mutable_state_dict())

    def fail_publication(_process):
        raise RuntimeError("staged publication failed")

    monkeypatch.setattr(
        _handler, "retain_stochastic_material_workspace_once", fail_publication
    )
    with pytest.raises(RuntimeError, match="staged publication failed"):
        target.mech.restore_mutable_state_dict(source.mech.mutable_state_dict())
    assert not process._spatial_configured
    assert process._spatial_operators is operators
    assert all(target.get_buffer(name) is value for name, value in before.items())
    assert "_material_held_noise_D" not in process._buffers
    after = target.mech.mutable_state_dict()
    for key, value in stochastic.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(after[key], value, atol=0, rtol=0)
        elif ".rng." in key:
            assert after[key]["base_seed"] == value["base_seed"]
            for state_name, state in value["rng_state"].items():
                torch.testing.assert_close(after[key]["rng_state"][state_name], state)


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_cold_functional_commit_retains_workspace_through_public_step(
    kind, monkeypatch
):
    # Source lowering and target mode match; inference executes after commit.
    source = _advanced_source(kind, monkeypatch).train()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    target = _population(kind, monkeypatch, configure=False)
    functional.commit_state_(target, tensors.state)
    target.eval()
    packet = functional.sample_noise_draw(tensors.state)
    expected, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=packet),
    )
    target.step(dt=DT)
    torch.testing.assert_close(
        target.mech.materials["pool"].amount,
        expected["materials"]["pool"]["amount"],
        atol=1e-12,
        rtol=1e-12,
    )
