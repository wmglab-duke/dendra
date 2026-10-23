"""Declared workspaces retain the NOISE frame from their native phase."""

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._noise_workspaces import (
    capture_noise_workspace,
    complete_legacy_noise_workspace_state,
    noise_workspace_noise_values,
)


class _Workspace(Mechanism):
    Mechanism.GLOBAL(scale=0.5)
    Mechanism.RANGENOISE("eta", distribution="normal", mu=0.2, sigma=0.1, seed=31)
    Mechanism.DERIVED_BUFFER("static")
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_buffers(self):
        return {"static": self.scale * (1.0 + self.eta)}

    def derive_timestep_buffers(self, dt):
        return {"coefficient": dt * self.scale * (1.0 + self.eta)}


class _NoiseState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0")
    State.RANGENOISE("eta", distribution="normal", mu=0.2, sigma=0.1, seed=32)


class _ChildNoiseWorkspace(Mechanism):
    Mechanism.STATE_BUNDLE(_NoiseState)
    Mechanism.GLOBAL(scale=0.5)
    Mechanism.DERIVED_BUFFER("static")
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_buffers(self):
        return {"static": self.scale * (1.0 + self.DE["_NoiseState"].eta)}

    def derive_timestep_buffers(self, dt):
        return {"coefficient": dt * self.static}


class _DerivedNoiseState(_NoiseState):
    State.GLOBAL(scale=0.5)
    State.DERIVED_BUFFER("static")
    State.TIMESTEP_BUFFER("coefficient")

    derive_buffers = _Workspace.derive_buffers
    derive_timestep_buffers = _Workspace.derive_timestep_buffers


class _StateNoiseWorkspace(Mechanism):
    Mechanism.STATE_BUNDLE(_DerivedNoiseState)


class _DeferredWorkspace(_Workspace):
    Mechanism.DERIVED_BUFFER("indices", dtype=torch.long, shape="deferred")

    def derive_buffers(self):
        values = super().derive_buffers()
        values["indices"] = torch.arange(self.eta.numel(), device=self.eta.device)
        return values


class _DirectNoiseWorkspace(_Workspace):
    def derive_buffers(self):
        return {"static": self.scale * self.eta}

    def derive_timestep_buffers(self, dt):
        return {"coefficient": dt * self.scale * self.eta}


class _GlobalNoiseWorkspace(Mechanism):
    Mechanism.GLOBAL(scale=0.5)
    Mechanism.GLOBALNOISE("eta", distribution="normal", mu=0.2, sigma=0.1, seed=33)
    Mechanism.DERIVED_BUFFER("static")
    Mechanism.TIMESTEP_BUFFER("coefficient")

    derive_buffers = _Workspace.derive_buffers
    derive_timestep_buffers = _Workspace.derive_timestep_buffers


class _BatchNoiseWorkspace(Mechanism):
    Mechanism.GLOBAL(scale=0.5)
    Mechanism.BATCHNOISE("eta", distribution="normal", mu=0.2, sigma=0.1, seed=34)
    Mechanism.DERIVED_BUFFER("static")
    Mechanism.TIMESTEP_BUFFER("coefficient")

    derive_buffers = _Workspace.derive_buffers
    derive_timestep_buffers = _Workspace.derive_timestep_buffers


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        yield


def _population(mechanism=_Workspace, batch=None):
    population = dn.SingleCompartment(N=2, C=2, v_init=-60.0)
    population.insert(mechanism, alias="workspace")
    if batch is not None:
        population.batch(batch)
    population.initialize()
    return population, next(iter(population.mech.mechanisms.values()))


@pytest.mark.parametrize("mechanism", (_Workspace, _ChildNoiseWorkspace))
def test_native_workspace_noise_capture_precedes_runtime_refresh(mechanism):
    population, owner = _population(mechanism)
    initial_noise = noise_workspace_noise_values(owner, "derived")
    population.mech.set_dt(0.1)
    configured_noise = noise_workspace_noise_values(owner, "timestep")
    static, coefficient = owner.static.clone(), owner.coefficient.clone()
    population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    torch.testing.assert_close(owner.static, static)
    torch.testing.assert_close(owner.coefficient, coefficient)
    for name, value in initial_noise.items():
        torch.testing.assert_close(
            noise_workspace_noise_values(owner, "derived")[name], value
        )
        torch.testing.assert_close(configured_noise[name], value)
        assert not value.requires_grad


def test_native_workspace_noise_does_not_share_runtime_storage_or_block_backward():
    population, owner = _population()
    population.mech.set_dt(0.1)
    noise = noise_workspace_noise_values(owner, "derived")["eta"]
    assert noise.untyped_storage().data_ptr() != owner.eta.untyped_storage().data_ptr()
    expected = ((1.0 + noise) * 1.1).sum()
    population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    gradient = torch.autograd.grad(
        (owner.static + owner.coefficient).sum(), owner.scale_param
    )[0]
    torch.testing.assert_close(gradient.sum(), expected)


def test_native_direct_noise_workspace_backward_survives_several_runtime_draws():
    population, owner = _population(_DirectNoiseWorkspace)
    population.mech.set_dt(0.1)
    held = noise_workspace_noise_values(owner, "derived")["eta"].clone()
    runtime_noise = owner.eta
    for _ in range(4):
        population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    assert owner.eta is runtime_noise
    static_gradient, coefficient_gradient = (
        torch.autograd.grad(
            value.sum(),
            (owner.scale_param, owner.eta_mu_param),
            allow_unused=True,
        )
        for value in (owner.static, owner.coefficient)
    )
    torch.testing.assert_close(static_gradient[0].sum(), held.sum())
    torch.testing.assert_close(coefficient_gradient[0].sum(), 0.1 * held.sum())
    assert static_gradient[1] is None
    assert coefficient_gradient[1] is None


@pytest.mark.parametrize(
    "mechanism", (_GlobalNoiseWorkspace, _BatchNoiseWorkspace, _Workspace)
)
@pytest.mark.parametrize("batch", (None, 2))
def test_native_held_noise_preserves_declaration_shape_with_padded_inputs(
    mechanism, batch
):
    population, owner = _population(mechanism, batch=batch)
    canonical_shape = owner.eta.shape
    population.mech.set_dt(0.1)
    runtime_noise = owner.eta
    for phase in ("derived", "timestep"):
        owner._buffers["eta"] = runtime_noise.reshape((1, 1, 1) + canonical_shape)
        try:
            capture_noise_workspace(owner, phase, dt=0.1)
        finally:
            owner._buffers["eta"] = runtime_noise
        held = noise_workspace_noise_values(owner, phase)["eta"]
        assert held.shape == canonical_shape
        torch.testing.assert_close(held, runtime_noise)


@pytest.mark.parametrize("same_dt", (False, True))
def test_native_checkpoint_restores_workspace_before_first_solver_setup(same_dt):
    population, owner = _population()
    population.mech.set_dt(0.1)
    coefficient = owner.coefficient.clone()
    population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    runtime_noise = owner.eta.clone()
    snapshot = population.mech.mutable_state_dict()
    restored, target = _population()
    restored.mech.restore_mutable_state_dict(snapshot)
    dt = 0.1 if same_dt else 0.2
    restored.mech.set_dt(dt)
    if same_dt:
        torch.testing.assert_close(target.coefficient, coefficient)
        # The handoff is consumed exactly once: a further explicit timestep
        # configuration uses the current runtime realization.
        restored.mech.set_dt(dt)
    torch.testing.assert_close(
        target.coefficient, dt * target.scale * (1.0 + runtime_noise)
    )


def test_native_checkpoint_rejects_partial_noise_workspace_atomically():
    population, owner = _population()
    population.mech.set_dt(0.1)
    original = population.mech.mutable_state_dict()
    malformed = copy.copy(original)
    key = next(name for name in malformed if "noise_workspace.derived.noise." in name)
    del malformed[key]
    noise, coefficient = owner.eta, owner.coefficient
    with pytest.raises(KeyError, match="NOISE workspace state is missing"):
        population.mech.restore_mutable_state_dict(malformed)
    assert owner.eta is noise
    assert owner.coefficient is coefficient


def test_native_legacy_checkpoint_retains_target_workspace():
    population, owner = _population()
    population.mech.set_dt(0.1)
    snapshot = {
        key: value
        for key, value in population.mech.mutable_state_dict().items()
        if "noise_workspace." not in key
    }
    population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    population.mech.set_dt(0.2)
    coefficient = owner.coefficient
    population.mech.restore_mutable_state_dict(snapshot)
    assert owner.coefficient is coefficient


def test_native_reinitialize_resets_timestep_workspace_provenance():
    population, owner = _population()
    population.mech.set_dt(0.1)
    population.initialize()
    assert not owner._buffers["_noise_workspace_timestep_active"].item()
    assert owner._buffers["_noise_workspace_timestep_dt"].item() == 0.0
    torch.testing.assert_close(owner.coefficient, torch.zeros_like(owner.coefficient))


def test_native_checkpoint_resolves_cold_deferred_workspace():
    population, owner = _population(_DeferredWorkspace)
    population.mech.set_dt(0.1)
    snapshot = population.mech.mutable_state_dict()
    target = dn.SingleCompartment(N=2, C=2, v_init=-60.0)
    target.insert(_DeferredWorkspace)
    target.build()
    restored = next(iter(target.mech.mechanisms.values()))
    assert restored._derived_resolved_shapes["indices"] is None
    target.mech.restore_mutable_state_dict(snapshot)
    torch.testing.assert_close(restored.indices, owner.indices)
    assert restored._derived_resolved_shapes["indices"] == tuple(owner.indices.shape)


def test_native_state_dict_load_discards_previous_checkpoint_handoff():
    population, owner = _population()
    population.mech.set_dt(0.1)
    population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    snapshot = population.mech.mutable_state_dict()
    target, restored = _population()
    target.mech.restore_mutable_state_dict(snapshot)
    assert "_noise_workspace_restored_dt" in restored.__dict__
    target.load_state_dict(population.state_dict())
    assert "_noise_workspace_restored_dt" not in restored.__dict__
    assert not restored._buffers["_noise_workspace_timestep_active"].item()
    torch.testing.assert_close(restored.coefficient, owner.coefficient)
    torch.testing.assert_close(
        restored._buffers["_noise_workspace_timestep_dt"],
        owner._buffers["_noise_workspace_timestep_dt"],
    )
    target.mech.set_dt(0.1)
    torch.testing.assert_close(
        restored.coefficient, 0.1 * restored.scale * (1.0 + owner.eta)
    )


@pytest.mark.parametrize("invalid", ("extra", "active", "dt", "shape"))
def test_native_checkpoint_rejects_invalid_workspace_schema_atomically(invalid):
    population, owner = _population()
    population.mech.set_dt(0.1)
    snapshot = population.mech.mutable_state_dict()
    prefix = next(
        key.removesuffix("timestep.dt")
        for key in snapshot
        if key.endswith("noise_workspace.timestep.dt")
    )
    if invalid == "extra":
        snapshot[f"{prefix}undeclared"] = owner.eta.clone()
    elif invalid == "active":
        snapshot[f"{prefix}derived.active"] = owner.eta.new_tensor(1.0)
    elif invalid == "dt":
        snapshot[f"{prefix}timestep.dt"] = owner.eta.new_tensor(float("nan"))
    else:
        snapshot[f"{prefix}derived.buffer.static"] = owner.eta.new_zeros((99,))
    identities = tuple(owner._buffers.items())
    rng = owner.eta_rng.rng_state()
    with pytest.raises(ValueError):
        population.mech.restore_mutable_state_dict(snapshot)
    assert all(owner._buffers[name] is value for name, value in identities)
    after = owner.eta_rng.rng_state()
    torch.testing.assert_close(after["cpu"], rng["cpu"])


@pytest.mark.parametrize(
    "mechanism", (_Workspace, _ChildNoiseWorkspace, _StateNoiseWorkspace)
)
def test_native_state_dict_preserves_initialization_noise_after_runtime_draws(
    mechanism,
):
    population, _owner = _population(mechanism)
    population.initialize()
    population.mech.set_dt(0.1)
    for _ in range(3):
        population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    snapshot = copy.deepcopy(population.state_dict())
    derived_keys = [key for key in snapshot if "_noise_workspace_derived_" in key]
    assert derived_keys
    assert not any("_noise_workspace_timestep_" in key for key in snapshot)
    target, _target_owner = _population(mechanism)
    before = target.state_dict()
    assert any(not torch.equal(snapshot[key], before[key]) for key in derived_keys)
    target.load_state_dict(snapshot, strict=True)
    restored = target.state_dict()
    for key in derived_keys:
        torch.testing.assert_close(restored[key], snapshot[key])
    for name, owner in population.named_modules():
        if "_noise_workspace_derived_active" in owner._buffers:
            loaded_owner = target.get_submodule(name)
            for buffer_name in owner._derived_buffers:
                torch.testing.assert_close(
                    loaded_owner._buffers[buffer_name], owner._buffers[buffer_name]
                )
            assert not loaded_owner._buffers["_noise_workspace_timestep_active"].item()


@pytest.mark.parametrize(
    "mechanism", (_Workspace, _ChildNoiseWorkspace, _StateNoiseWorkspace)
)
def test_native_legacy_state_dict_load_preserves_destination_initialization_frame(
    mechanism,
):
    population, _owner = _population(mechanism)
    population.initialize()
    population.mech.set_dt(0.1)
    population.mech.sample_runtime_noises_(dt=0.1, phase="pre_state")
    legacy = {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in population.state_dict().items()
        if "_noise_workspace_derived_" not in key
    }
    keys = set(legacy)
    target, _target_owner = _population(mechanism)
    held = {
        key: value.detach().clone()
        for key, value in target.state_dict().items()
        if "_noise_workspace_derived_" in key
    }
    result = target.load_state_dict(legacy, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert set(legacy) == keys
    for key, value in held.items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_legacy_workspace_completion_does_not_mutate_inputs_or_sample():
    _population_value, owner = _population()
    snapshot = {"other": owner.scale}
    frame = owner._buffers["_noise_workspace_derived_noise_0"]
    rng = owner.eta_rng.rng_state()
    completed = complete_legacy_noise_workspace_state(owner, snapshot, "probe.")
    assert set(snapshot) == {"other"}
    assert snapshot["other"] is owner.scale
    torch.testing.assert_close(
        completed["probe._noise_workspace_derived_noise_0"], frame
    )
    assert completed["probe._noise_workspace_derived_noise_0"] is not frame
    assert not completed["probe._noise_workspace_derived_noise_0"].requires_grad
    assert owner._buffers["_noise_workspace_derived_noise_0"] is frame
    torch.testing.assert_close(owner.eta_rng.rng_state()["cpu"], rng["cpu"])


@pytest.mark.parametrize("missing", ("active", "noise"))
def test_native_state_dict_rejects_partial_initialization_noise_frame(missing):
    population, _owner = _population()
    snapshot = population.state_dict()
    suffix = (
        "_noise_workspace_derived_active"
        if missing == "active"
        else "_noise_workspace_derived_noise_0"
    )
    key = next(key for key in snapshot if key.endswith(suffix))
    del snapshot[key]
    keys = set(snapshot)
    target, _target_owner = _population()
    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_state_dict(snapshot, strict=True)
    assert set(snapshot) == keys
