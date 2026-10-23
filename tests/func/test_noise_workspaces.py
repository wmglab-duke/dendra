"""NOISE-derived workspaces retain their native initialization/configuration frame."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import pas
from tests.func.test_network import _clone, _close

pytestmark = pytest.mark.cpu
DT = 0.125
ALT_DT = 0.25
PREFIX = "integrator.mech.mechanisms.probe."
GATE = "DE._WorkspaceGate."
SCALE = PREFIX + "scale_param"


class _WorkspaceGate(State):
    State.STATE("x")
    State.GLOBAL(scale=0.3)
    State.RANGENOISE("eta", mu=0.4, sigma=0.05, seed=31, rng_name="stream")
    State.DERIVED_BUFFER("static")
    State.TIMESTEP_BUFFER("coefficient")
    State.CARRY("seed")
    State.DERIVATIVE("x' = coefficient")

    def derive_buffers(self):
        return {"static": self.scale * self.eta.clone()}

    def derive_timestep_buffers(self, dt):
        # Timestep-local workspaces use shape_p. Collapse explicit trial lanes
        # while preserving each compartment; static buffers keep shape_f.
        static = self.static.reshape(
            math.prod(self.static.shape[:-1]), self.static.shape[-1]
        ).mean(0, keepdim=True)
        noise = (
            self.eta.clone()
            .reshape(math.prod(self.eta.shape[:-1]), self.eta.shape[-1])
            .mean(0, keepdim=True)
        )
        return {
            "coefficient": dt.reshape((1,) * static.ndim)
            * (static + self.scale * noise)
        }

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}

    def initial_values(self, v, values):
        del values
        return {"seed": torch.zeros_like(v) + self.static + self.coefficient}


class _WorkspaceLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_WorkspaceGate)
    Mechanism.GLOBAL(scale=0.3, g=0.001, e=-65.0)
    Mechanism.GLOBALRAND(
        "offset",
        mu=0.2,
        sigma=0.01,
        seed=17,
        rng_name="stream",
        resample_on_initialize=False,
    )
    Mechanism.RANGENOISE("eta", mu=0.4, sigma=0.05, seed=19, rng_name="stream")
    Mechanism.DERIVED_BUFFER("static")
    Mechanism.TIMESTEP_BUFFER("coefficient")
    Mechanism.CARRY("seed", "accumulator")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {"static": self.scale * self.eta.clone() + self.offset}

    def derive_timestep_buffers(self, dt):
        static = self.static.reshape(
            math.prod(self.static.shape[:-1]), self.static.shape[-1]
        ).mean(0, keepdim=True)
        noise = (
            self.eta.clone()
            .reshape(math.prod(self.eta.shape[:-1]), self.eta.shape[-1])
            .mean(0, keepdim=True)
        )
        return {
            "coefficient": dt.reshape((1,) * static.ndim)
            * (static + self.scale * noise)
        }

    def initial_values(self, v, values):
        del values
        return {
            "seed": torch.zeros_like(v) + self.static + self.coefficient,
            "accumulator": torch.zeros_like(v),
        }

    def advance(self, v, dt, values):
        del v, dt
        return {
            "accumulator": values["accumulator"]
            + self.coefficient
            + 0.1 * self.eta.clone()
        }

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + 1.0e-4 * self.coefficient


class _ScopedNoiseWorkspace(Mechanism):
    Mechanism.GLOBAL(scale=0.3)
    Mechanism.DERIVED_BUFFER("static")
    Mechanism.TIMESTEP_BUFFER("coefficient")
    Mechanism.CARRY("accumulator")

    def derive_buffers(self):
        return {"static": self.scale * self.eta}

    def derive_timestep_buffers(self, dt):
        static = self.static.mean(0, keepdim=True)
        return {"coefficient": dt.reshape((1,) * static.ndim) * static}

    def initial_values(self, v, values):
        del values
        return {"accumulator": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v, dt
        return {"accumulator": values["accumulator"] + self.coefficient}


class _GlobalNoiseWorkspace(_ScopedNoiseWorkspace):
    Mechanism.GLOBALNOISE("eta", mu=0.4, sigma=0.05, seed=53)


class _BatchedNoiseWorkspace(_ScopedNoiseWorkspace):
    Mechanism.BATCHNOISE("eta", mu=0.4, sigma=0.05, seed=59)


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _population(mechanism_type=_WorkspaceLeak, *, batch=None, configure=True):
    population = dn.SingleCompartment(
        N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism_type.rename("probe"))
    if batch is not None:
        population.batch(batch)
    population.train()
    population.initialize()
    mechanism = population.mech.probe
    mechanism.DE._WorkspaceGate.scale_param = mechanism.scale_param
    mechanism.DE._WorkspaceGate.stream = mechanism.stream
    population.initialize()
    if configure:
        _configure(population, DT)
    return population


def _configure(population, dt):
    population.integrator._initialize(
        population, population.t.new_tensor(dt), force=True
    )


def _step(population, dt=DT):
    population._step(
        population.integrator, population, population.t.new_tensor(dt), None, None
    )
    population.t = population.t + dt


def _fresh(functional, tensors, *, packet=None):
    if packet is None:
        packet = functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        )
    return functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(random_draw=packet),
        state=tensors.state,
    )


def _assert_live_state(actual, expected):
    for name in (
        "integrator",
        "mechanisms",
        "mechanism_buffers",
        "state_buffers",
        "runtime_noise",
        "noise_workspaces",
    ):
        if name in actual or name in expected:
            _close(actual[name], expected[name])
    _close(actual["random"]["rng"], expected["random"]["rng"])


class _MutatingAdvance(_WorkspaceLeak):
    def advance(self, v, dt, values):
        self.static.add_(1.0)
        return _WorkspaceLeak.advance(self, v, dt, values)


class _MutatingInitializer(_WorkspaceLeak):
    mutate_workspace = False

    def initial_values(self, v, values):
        if self.mutate_workspace:
            self.static.add_(1.0)
        return _WorkspaceLeak.initial_values(self, v, values)


@pytest.mark.parametrize("mechanism", (_MutatingAdvance, _MutatingInitializer))
def test_diagnostics_reject_authored_noise_workspace_mutation(mechanism):
    source = _population(mechanism)
    if mechanism is _MutatingInitializer:
        source.mech.probe.mutate_workspace = True
    before = _clone(source.state_dict_for_checkpoint(), detached=True)
    with (
        dn.ctx(CHECK_PURITY=True),
        pytest.raises(dn.func.FunctionalizationError, match="read-only"),
    ):
        functional, tensors = dn.func.make_functional(source, dt=DT)
        _fresh(functional, tensors)
    _close(source.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("diagnostics", (False, True))
def test_fresh_workspaces_use_initial_noise_and_zero_initialization_timestep(
    batch, diagnostics
):
    source, native = _population(batch=batch), _population(batch=batch)
    with dn.ctx(CHECK_PURITY=diagnostics):
        functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone(tensors, detached=True)
    for _initialization in range(2):
        initial = _fresh(functional, tensors)
        native.initialize()
        _assert_live_state(initial.state, functional.extract(native).state)
        mechanism = native.mech.probe
        torch.testing.assert_close(
            initial.state["mechanism_buffers"]["probe"]["seed"],
            mechanism.static.expand_as(mechanism.seed),
        )
        assert torch.count_nonzero(mechanism.coefficient) == 0
        assert torch.count_nonzero(mechanism.DE._WorkspaceGate.coefficient) == 0
        _configure(native, DT)
        prepared = functional.prepare(initial.parameters, initial.constants)
        for suffix, owner in (("", mechanism), (GATE, mechanism.DE._WorkspaceGate)):
            for name in ("static", "coefficient"):
                torch.testing.assert_close(
                    prepared.values["mechanisms"][PREFIX + suffix + name],
                    getattr(owner, name),
                )
        tape = functional.sample_noise_draw_tape(initial.state, 3)
        final, _ = functional.prepare_and_rollout(
            initial.parameters,
            initial.constants,
            initial.state,
            dn.func.RolloutInput(noise=tape),
        )
        held = (mechanism.static.clone(), mechanism.coefficient.clone())
        for _ in range(3):
            _step(native)
        _assert_live_state(final, functional.extract(native).state)
        torch.testing.assert_close(mechanism.static, held[0])
        torch.testing.assert_close(mechanism.coefficient, held[1])
        tensors = initial._replace(state=final)
    _close(functional.extract(source), before)


@pytest.mark.parametrize("mechanism", (_GlobalNoiseWorkspace, _BatchedNoiseWorkspace))
@pytest.mark.parametrize("diagnostics", (False, True))
def test_scoped_noise_workspaces_preserve_batched_initialization_schema_and_gradients(
    mechanism, diagnostics
):
    def build():
        population = dn.SingleCompartment(
            N=1, C=2, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(mechanism.rename("probe"))
        population.batch(2)
        population.train()
        population.initialize()
        _configure(population, DT)
        return population

    source, native = build(), build()
    with dn.ctx(CHECK_PURITY=diagnostics):
        functional, tensors = dn.func.make_functional(source, dt=DT)
    initial = _fresh(functional, tensors)
    native.initialize()
    _assert_live_state(initial.state, functional.extract(native).state)
    _configure(native, DT)
    tape = functional.sample_noise_draw_tape(initial.state, 3)
    final, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _ in range(3):
        _step(native)
    _assert_live_state(final, functional.extract(native).state)

    def coefficient(dt):
        prepared = functional.prepare(
            initial.parameters, {**initial.constants, "dt": dt}
        )
        return prepared.values["mechanisms"][PREFIX + "coefficient"]

    dt = initial.constants["dt"]
    primal, tangent = torch.func.jvp(coefficient, (dt,), (torch.ones_like(dt),))
    torch.testing.assert_close(tangent, primal / dt)
    lanes = torch.stack((dt, 2 * dt))
    torch.testing.assert_close(
        torch.vmap(coefficient)(lanes),
        torch.stack([coefficient(value) for value in lanes]),
    )
    assert torch.vmap(coefficient)(lanes[:0]).shape == (0, *primal.shape)
    derivative = torch.autograd.grad(
        final["mechanism_buffers"]["probe"]["accumulator"].sum(),
        tensors.parameters[SCALE],
    )[0]
    native_derivative = torch.autograd.grad(
        native.mech.probe.accumulator.sum(), native.mech.probe.scale_param
    )[0]
    torch.testing.assert_close(derivative, native_derivative)
    assert derivative.isfinite().all() and derivative.abs().sum() > 0


@pytest.mark.parametrize("dt", (DT, ALT_DT))
def test_snapshot_after_runtime_draws_preserves_held_workspaces_and_dt_reconfiguration(
    dt,
):
    source, native = _population(), _population()
    for _ in range(3):
        _step(source)
        _step(native)
    frozen = _clone(source.state_dict_for_checkpoint(), detached=True)
    old_noise = native.mech.probe.eta.clone()
    if dt != DT:
        _configure(native, dt)
    functional, tensors = dn.func.make_functional(source, dt=dt)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    for suffix, owner in (
        ("", native.mech.probe),
        (GATE, native.mech.probe.DE._WorkspaceGate),
    ):
        for name in ("static", "coefficient"):
            torch.testing.assert_close(
                prepared.values["mechanisms"][PREFIX + suffix + name],
                getattr(owner, name),
            )
    if dt != DT:
        torch.testing.assert_close(
            native.mech.probe.coefficient,
            dt * (native.mech.probe.static + native.mech.probe.scale * old_noise),
        )
    tape = functional.sample_noise_draw_tape(tensors.state, 2)
    actual, _ = functional.prepare_and_rollout(
        *tensors[:3], dn.func.RolloutInput(noise=tape)
    )
    for _ in range(2):
        _step(native, dt)
    _assert_live_state(actual, functional.extract(native).state)
    _close(source.state_dict_for_checkpoint(), frozen)


def test_continuation_dt_change_uses_previous_noise_once_then_holds_across_chunks():
    source, native = _population(), _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 2)
    state, _ = functional.prepare_and_rollout(
        *tensors[:3], dn.func.RolloutInput(noise=tape)
    )
    for _ in range(2):
        _step(native)
    _configure(native, ALT_DT)
    configured = native.mech.probe.coefficient.clone()
    constants = {**tensors.constants, "dt": tensors.constants["dt"].new_tensor(ALT_DT)}
    for _chunk in range(2):
        tape = functional.sample_noise_draw_tape(state, 2)
        state, _ = functional.prepare_and_rollout(
            tensors.parameters, constants, state, dn.func.RolloutInput(noise=tape)
        )
        for _ in range(2):
            _step(native, ALT_DT)
        _assert_live_state(state, functional.extract(native).state)
        torch.testing.assert_close(native.mech.probe.coefficient, configured)


def test_packed_population_preserves_noise_workspace_frames_through_fresh_restart():
    def build():
        population = dn.concat_models(
            {name: _population() for name in ("left", "right")}, threads=1
        )
        population.train()
        population.initialize()
        _configure(population, DT)
        return population

    source, native = build(), build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    for restart in (False, True):
        if restart:
            tensors = _fresh(functional, tensors)
            native.initialize()
            _close(tensors.state, functional.extract(native).state)
            _configure(native, DT)
        tape = functional.sample_noise_draw_tape(tensors.state, 3)
        final, _ = functional.prepare_and_rollout(
            *tensors[:3], dn.func.RolloutInput(noise=tape)
        )
        for _ in range(3):
            _step(native)
        _close(final, functional.extract(native).state)
        tensors = tensors._replace(state=final)
    functional.commit_state_(source, tensors.state)
    _close(functional.extract(source).state, tensors.state)
    _step(source)
    _step(native)
    _close(functional.extract(source).state, functional.extract(native).state)


def test_workspace_gradients_use_held_noise_but_noise_inputs_remain_detached():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    packet = packet._replace(
        draws={
            name: value.clone().requires_grad_() for name, value in packet.draws.items()
        }
    )
    initial = _fresh(functional, tensors, packet=packet)
    dt = initial.constants["dt"].detach().clone().requires_grad_()
    prepared = functional.prepare(initial.parameters, {**initial.constants, "dt": dt})
    workspace = prepared.values["mechanisms"][PREFIX + "coefficient"]
    scale = tensors.parameters[SCALE]
    noise_mu = tensors.get_parameter("*probe.eta_mu_param")
    derivatives = torch.autograd.grad(
        workspace.sum(),
        (scale, dt, noise_mu, *packet.draws.values()),
        allow_unused=True,
    )
    noise = initial.state["runtime_noise"][PREFIX + "eta"]
    torch.testing.assert_close(derivatives[0], (2 * dt * noise).sum().reshape_as(scale))
    torch.testing.assert_close(
        derivatives[1],
        (
            prepared.values["mechanisms"][PREFIX + "static"]
            + initial.parameters[SCALE] * noise
        ).sum(),
    )
    assert derivatives[2] is None or torch.count_nonzero(derivatives[2]) == 0
    for name, derivative in zip(packet.draws, derivatives[3:], strict=True):
        if name.startswith("noise."):
            assert derivative is None or torch.count_nonzero(derivative) == 0


def test_workspace_preparation_supports_jvp_and_empty_vmap_lanes():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    initial = _fresh(functional, tensors, packet=packet)

    def coefficient(dt):
        return functional.prepare(
            initial.parameters, {**initial.constants, "dt": dt}
        ).values["mechanisms"][PREFIX + "coefficient"]

    dt = initial.constants["dt"]
    primal, tangent = torch.func.jvp(coefficient, (dt,), (torch.ones_like(dt),))
    torch.testing.assert_close(tangent, primal / dt)
    lanes = torch.stack((dt, 2 * dt))
    torch.testing.assert_close(
        torch.vmap(coefficient)(lanes),
        torch.stack([coefficient(value) for value in lanes]),
    )
    assert torch.vmap(coefficient)(lanes[:0]).shape == (0, *primal.shape)


@pytest.mark.parametrize("edited", (False, True))
def test_population_owned_cache_restores_held_workspaces_and_rng_without_resampling(
    edited,
):
    source, native = _population(), _population()
    for population in (source, native):
        for _ in range(2):
            _step(population)
        population.cache("_steady_state")
        if edited:
            # Saved numerical workspaces are authoritative cache contents;
            # rebuilding from retained NOISE would erase this supplied value.
            population._caches["_steady_state"][PREFIX + "static"].add_(0.75)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initial = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    _assert_live_state(initial.state, functional.extract(native).state)
    _configure(native, DT)
    tape = functional.sample_noise_draw_tape(initial.state, 3)
    final, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _ in range(3):
        _step(native)
    _assert_live_state(final, functional.extract(native).state)


def test_state_dict_round_trip_preserves_derived_frame_for_lowering_and_continuation():
    source, restored = _population(), _population()
    for _ in range(3):
        _step(source)
    saved = _clone(source.state_dict(), detached=True)
    restored.initialize()
    assert not torch.equal(restored.mech.probe.static, source.mech.probe.static)
    restored.load_state_dict(saved)
    functional, tensors = dn.func.make_functional(restored, dt=DT)
    _configure(source, DT)
    _configure(restored, DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    for suffix, owner in (
        ("", source.mech.probe),
        (GATE, source.mech.probe.DE._WorkspaceGate),
    ):
        for name in ("static", "coefficient"):
            torch.testing.assert_close(
                prepared.values["mechanisms"][PREFIX + suffix + name],
                getattr(owner, name),
            )
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    final, _ = functional.prepare_and_rollout(
        *tensors[:3], dn.func.RolloutInput(noise=tape)
    )
    for _ in range(3):
        _step(source)
        _step(restored)
    _assert_live_state(final, functional.extract(source).state)
    _assert_live_state(final, functional.extract(restored).state)


@pytest.mark.parametrize("legacy", (False, True))
def test_population_owned_cache_restores_workspace_frames_across_initializations(
    legacy,
):
    source, native = _population(), _population()
    for population in (source, native):
        for _ in range(2):
            _step(population)
        population.cache("_steady_state")
        saved = population._caches.pop("_steady_state")
        population.initialize()
        if legacy:
            saved = {
                name: value
                for name, value in saved.items()
                if "._noise_workspace_derived_" not in name
            }
        population._caches["_steady_state"] = saved
    functional, tensors = dn.func.make_functional(source, dt=DT)
    if legacy:
        original_cache = source._caches["_steady_state"]
        normalized = functional._owned_cache_plan.normalize(original_cache)
        metadata = [name for name in normalized if "._noise_workspace_derived_" in name]
        assert metadata
        assert all(name not in original_cache for name in metadata)
        partial = {**original_cache, metadata[0]: normalized[metadata[0]]}
        with pytest.raises(dn.func.FunctionalizationError, match="schema changed"):
            functional._owned_cache_plan.normalize(partial)
    initial = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    _assert_live_state(initial.state, functional.extract(native).state)
    _configure(native, DT)
    tape = functional.sample_noise_draw_tape(initial.state, 3)
    final, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _ in range(3):
        _step(native)
    _assert_live_state(final, functional.extract(native).state)


@pytest.mark.filterwarnings(
    "ignore:Converting a tensor with requires_grad=True to a scalar:UserWarning"
)
def test_cache_workspace_boundary_is_detached_but_changed_timestep_keeps_gradients():
    source, native = _population(), _population()
    for population in (source, native):
        for _ in range(2):
            _step(population)
        population.cache("_steady_state")
        population._caches["_steady_state"][PREFIX + "static"].add_(0.75)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initial = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    dt = initial.constants["dt"].new_tensor(ALT_DT, requires_grad=True)
    native_dt = native.t.new_tensor(ALT_DT, requires_grad=True)
    native.integrator._initialize(native, native_dt, force=True)
    constants = {**initial.constants, "dt": dt}
    tape = functional.sample_noise_draw_tape(initial.state, 3)
    final, _ = functional.prepare_and_rollout(
        initial.parameters, constants, initial.state, dn.func.RolloutInput(noise=tape)
    )
    for _ in range(3):
        _step(native, ALT_DT)
    _assert_live_state(final, functional.extract(native).state)
    derivative, parameter_derivative = torch.autograd.grad(
        final["mechanism_buffers"]["probe"]["accumulator"].sum(),
        (dt, tensors.parameters[SCALE]),
        allow_unused=True,
    )
    expected, native_parameter_derivative = torch.autograd.grad(
        native.mech.probe.accumulator.sum(),
        (native_dt, native.mech.probe.scale_param),
        allow_unused=True,
    )
    torch.testing.assert_close(derivative, expected)
    assert derivative.isfinite().all() and derivative.abs().sum() > 0
    assert (
        parameter_derivative is None or torch.count_nonzero(parameter_derivative) == 0
    )
    assert native_parameter_derivative is None


@pytest.mark.parametrize("cold", (False, True))
def test_functional_commit_and_native_checkpoint_continue_held_workspaces(cold):
    source, native = _population(), _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    final, _ = functional.prepare_and_rollout(
        *tensors[:3], dn.func.RolloutInput(noise=tape)
    )
    for _ in range(3):
        _step(native)
    target = _population(configure=not cold)
    functional.commit_state_(target, final)
    if cold:
        # The ordinary host configures a cold integrator before its first step.
        # The checkpoint handoff retains the committed workspace realization.
        _configure(target, DT)
    checkpoint = _clone(target.state_dict_for_checkpoint(), detached=True)
    restored = _population(configure=not cold)
    restored.restore_dict_from_checkpoint(checkpoint)
    if cold:
        _configure(restored, DT)
    tape = functional.sample_noise_draw_tape(final, 2)
    continued, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, final, dn.func.RolloutInput(noise=tape)
    )
    for _ in range(2):
        _step(native)
        _step(target)
        _step(restored)
    expected = functional.extract(native).state
    for actual in (
        continued,
        functional.extract(target).state,
        functional.extract(restored).state,
    ):
        _assert_live_state(actual, expected)


class _ShiftScale(torch.nn.Module):
    def forward(self, value, shift):
        return (value + shift,)


class _RootParameter(torch.nn.Module):
    def forward(self, value):
        return value.sqrt()


class _GetterNoiseWorkspace(Mechanism):
    Mechanism.GLOBAL(scale=4.0)
    Mechanism.RANGENOISE("eta", mu=0.4, sigma=0.05, seed=67)
    Mechanism.TIMESTEP_BUFFER("coefficient")
    Mechanism.CARRY("accumulator")

    def derive_timestep_buffers(self, dt):
        # This authored getter evaluates its registered Torch parametrization.
        return {"coefficient": dt * self.scale_param * self.eta.clone()}

    def initial_values(self, v, values):
        del values
        return {"accumulator": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v, dt
        return {"accumulator": values["accumulator"] + self.coefficient}


class _ReplaceScale(torch.nn.Module):
    def forward(self, value, replacement):
        return (value * 0 + replacement,)


def test_post_initialize_raw_write_refreshes_static_workspace_and_keeps_timestep_zero():
    source, native = _population(), _population()
    for population in (source, native):
        population.register_post_initialize_transform(
            "replace",
            _ReplaceScale(),
            reads=(f"parameters.{SCALE}",),
            writes=(f"parameters.{SCALE}",),
            inputs={"replacement": population.t.new_tensor(0.6)},
        )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initial = _fresh(functional, tensors)
    native.initialize()
    _assert_live_state(initial.state, functional.extract(native).state)
    assert torch.count_nonzero(native.mech.probe.coefficient) == 0
    assert torch.count_nonzero(native.mech.probe.DE._WorkspaceGate.coefficient) == 0
    _configure(native, DT)
    tape = functional.sample_noise_draw_tape(initial.state, 2)
    actual, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _ in range(2):
        _step(native)
    _assert_live_state(actual, functional.extract(native).state)


def _singular_network():
    left = dn.SingleCompartment(N=1, C=2, integrator=dn.bwd_euler_sc(imem=False))
    left.insert(_GetterNoiseWorkspace.rename("probe"))
    left.initialize()
    register_parametrization(left.mech.probe, "scale_param", _RootParameter())
    right = dn.SingleCompartment(N=1, C=2, integrator=dn.bwd_euler_sc(imem=False))
    right.insert(pas)
    right.initialize()
    right.register_parameter(
        "shared_source", left.mech.probe.parametrizations.scale_param.original
    )
    network = dn.Network({"left": left, "right": right})
    network.train()
    network.initialize(DT)
    # Left's valid preparation precedes the later shared write. Re-evaluating
    # its getter after that write would take sqrt(-1) along a discarded route.
    right.register_pre_initialize_transform(
        "replace",
        _ReplaceScale(),
        reads=("parameters.shared_source",),
        writes=("parameters.shared_source",),
        inputs={"replacement": right.t.new_tensor(-1.0)},
    )
    return network


def test_late_shared_raw_write_cannot_poison_retained_getter_workspace_gradients():
    source, native = _singular_network(), _singular_network()
    functional, tensors = dn.func.make_functional_network(source)
    raw_name = PREFIX + "parametrizations.scale_param.original"
    incoming = tensors.parameters["populations"]["left"][raw_name]
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    defaults = functional.initialization_inputs
    initial = functional.initialize(
        *tensors,
        defaults._replace(
            populations={
                name: inputs._replace(random_draw=packets[name])
                for name, inputs in defaults.populations.items()
            }
        ),
    )
    native.initialize(DT)
    assert initial.parameters["populations"]["left"][raw_name].item() == -1.0
    tapes = functional.sample_population_noise_tapes(initial.state, 2)
    final, _ = functional.prepare_and_rollout(
        *initial,
        dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
            }
        ),
    )
    for _ in range(2):
        native.step()
    actual = final["populations"]["left"]["mechanism_buffers"]["probe"]["accumulator"]
    torch.testing.assert_close(actual, native.left.mech.probe.accumulator)
    derivative = torch.autograd.grad(actual.sum(), incoming)[0]
    noise = initial.state["populations"]["left"]["runtime_noise"][PREFIX + "eta"]
    # The native later write changes the raw Parameter's version; the pure
    # path retains the earlier sqrt(4) graph, whose derivative is explicit.
    expected = (2 * DT * noise / (2 * incoming.sqrt())).sum()
    torch.testing.assert_close(derivative, expected)
    assert derivative.isfinite().all() and derivative.abs().sum() > 0


def _network(cache_kind=None, *, transforms=False):
    cells = {name: _population() for name in ("left", "right")}
    network = dn.Network(cells)
    shared = cells["left"].mech.probe.scale_param
    for cell in cells.values():
        cell.mech.probe.scale_param = shared
        cell.mech.probe.DE._WorkspaceGate.scale_param = shared
        if transforms:
            cell.register_pre_initialize_transform(
                "shift",
                _ShiftScale(),
                reads=(f"parameters.{SCALE}",),
                writes=(f"parameters.{SCALE}",),
                inputs={"shift": shared.new_tensor(0.1)},
            )
    network.train()
    network.initialize(DT)
    if cache_kind:
        for _ in range(2):
            network.step()
        if cache_kind == "population":
            cells["right"].cache("_steady_state")
        else:
            network.cache_state()
    return network


@pytest.mark.parametrize("cache_kind", (None, "population", "network", "transforms"))
def test_network_workspaces_preserve_shared_parameters_rng_and_cache_ownership(
    cache_kind,
):
    kwargs = {"transforms": cache_kind == "transforms"}
    cache = None if cache_kind == "transforms" else cache_kind
    source, native = _network(cache, **kwargs), _network(cache, **kwargs)
    functional, tensors = dn.func.make_functional_network(source)
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    defaults = functional.initialization_inputs
    initial = functional.initialize(
        *tensors,
        defaults._replace(
            populations={
                name: inputs._replace(random_draw=packets[name])
                for name, inputs in defaults.populations.items()
            }
        ),
    )
    native.initialize(DT)
    assert (
        initial.parameters["populations"]["left"][SCALE]
        is initial.parameters["populations"]["right"][SCALE]
    )
    tapes = functional.sample_population_noise_tapes(initial.state, 3)
    final, _ = functional.prepare_and_rollout(
        *initial,
        dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
            }
        ),
    )
    for _ in range(3):
        native.step()
    expected = functional.extract(native).state
    for name in ("left", "right"):
        _assert_live_state(final["populations"][name], expected["populations"][name])
    functional.commit_state_(native, final)
    restored = copy.deepcopy(native)
    restored.restore_dict_from_checkpoint(
        _clone(native.state_dict_for_checkpoint(), detached=True)
    )
    native.step()
    restored.step()
    for name in ("left", "right"):
        _assert_live_state(
            functional.extract(restored).state["populations"][name],
            functional.extract(native).state["populations"][name],
        )


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_noise_workspace_inductor_rollout_matches_native_and_keeps_training_gradients(
    execution,
):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        backend="inductor",
        fullgraph=True,
        trainable=(f"populations.left.{SCALE}",) if execution == "scan" else (),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v"))},
    )
    actual, auxiliary = runner()
    native.initialize(DT)
    for _ in range(4):
        native.step()
    expected = runner.functional.extract(native).state
    for name in ("left", "right"):
        _assert_live_state(actual["populations"][name], expected["populations"][name])
    continued, _ = runner(
        actual,
        parameters=auxiliary.get("parameters", runner.parameters),
        constants=auxiliary.get("constants", runner.constants),
    )
    for _ in range(4):
        native.step()
    for name in ("left", "right"):
        torch.testing.assert_close(
            continued["populations"][name]["integrator"]["v"],
            native.populations[name].v,
        )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    if execution == "scan":
        derivative = torch.autograd.grad(
            continued["populations"]["left"]["mechanism_buffers"]["probe"][
                "accumulator"
            ].sum(),
            runner.get_parameter(f"populations.left.{SCALE}"),
        )[0]
        native_derivative = torch.autograd.grad(
            native.left.mech.probe.accumulator.sum(),
            native.left.mech.probe.scale_param,
        )[0]
        torch.testing.assert_close(derivative, native_derivative)
        assert derivative.isfinite().all() and derivative.abs().sum() > 0
