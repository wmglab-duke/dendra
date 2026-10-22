"""Population runtime noise is explicit, detached, and checkpointable."""

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.random_parameters import register_random_distribution
from tests.func.test_network import _clone, _close
from tests.func.test_network_runtime_noise import _NoisyLeak, _SharedRandomNoiseLeak

pytestmark = pytest.mark.cpu
DT = 0.125


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


def _population(mechanism=_NoisyLeak, batch=None):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism)
    if batch is not None:
        population.batch(batch)
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def _native_step(population):
    population._step(
        population.integrator, population, population.t.new_tensor(DT), None, None
    )
    population.t = population.t + DT


@pytest.mark.parametrize("mechanism", (_NoisyLeak, _SharedRandomNoiseLeak))
@pytest.mark.parametrize("batch", (None, 2))
def test_population_explicit_noise_tape_matches_native_and_does_not_mutate_inputs(
    mechanism, batch
):
    source, native = _population(mechanism, batch), _population(mechanism, batch)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone(tensors.state, detached=True)
    tape = functional.sample_noise_draw_tape(tensors.state, 4)
    _close(tensors.state, before)
    actual, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _index in range(4):
        _native_step(native)
    _close(actual, functional.extract(native).state)
    _close(tensors.state, before)
    functional.commit_state_(source, actual)
    _close(functional.extract().state, actual)
    _native_step(source)
    _native_step(native)
    _close(functional.extract().state, functional.extract(native).state)


def test_population_noise_initialization_continues_shared_rand_stream():
    source, native = (
        _population(_SharedRandomNoiseLeak),
        _population(_SharedRandomNoiseLeak),
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    state, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _index in range(3):
        _native_step(native)
    packet = functional.sample_initialization_draw(tensors.parameters, state)
    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(random_draw=packet),
        state=state,
    )
    native.initialize()
    _close(actual.state, functional.extract(native).state)


def test_population_noise_step_requires_explicit_packet_and_validates_schema():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    with pytest.raises(ValueError, match="explicit RuntimeNoiseDraw"):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    packet = functional.sample_noise_draw(tensors.state)
    malformed = packet._replace(draws={})
    with pytest.raises(ValueError, match="names"):
        functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(noise=malformed),
        )
    state, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=0
    )
    _close(state, tensors.state)


def test_noise_distribution_parameters_and_explicit_draws_are_detached():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = functional.sample_noise_draw(tensors.state)
    packet = packet._replace(
        draws={
            name: value.clone().requires_grad_() for name, value in packet.draws.items()
        }
    )
    state, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=packet),
    )
    parameter = tensors.get_parameter("*eta_mu_param")
    gradient = torch.autograd.grad(
        state["integrator"]["v"].sum(),
        (parameter, *packet.draws.values()),
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradient)
    g_gradient = torch.autograd.grad(
        state["integrator"]["v"].sum(),
        tensors.get_parameter("*g_param"),
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(g_gradient)


@pytest.mark.parametrize("lanes", (0, 2))
def test_noise_step_vmap_accepts_explicit_draw_lanes(lanes):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = functional.sample_noise_draw(tensors.state)
    expanded = _pytree.tree_map(
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape).clone(), packet
    )
    result = torch.vmap(
        lambda draw: functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(noise=draw),
        )[0]
    )(expanded)
    assert result["integrator"]["v"].shape == (
        lanes,
        *tensors.state["integrator"]["v"].shape,
    )


def test_noise_transform_cannot_mutate_draws():
    def sampler(rng, shape, *, device, dtype):
        return rng.rand(shape, device=device, dtype=dtype)

    def transform(parameters, draw):
        return draw.add_(parameters["amplitude"])

    register_random_distribution(
        "functional_noise_mutator",
        {"amplitude": 1.0},
        draw_sampler=sampler,
        transform=transform,
        overwrite=True,
    )

    class MutatingNoise(Mechanism):
        Mechanism.RANGENOISE(
            "eta", distribution="functional_noise_mutator", amplitude=1.0
        )

    population = _population(MutatingNoise)
    with (
        dn.ctx(CHECK_PURITY=True),
        pytest.raises(dn.func.FunctionalizationError, match="NOISE.*transform"),
    ):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("enabled", (False, True))
def test_noise_distribution_diagnostics_retain_the_lowering_policy(enabled):
    from dendra.func._runtime_noise import RuntimeNoisePlan

    source = _population()
    with dn.ctx(CHECK_PURITY=enabled):
        plan = RuntimeNoisePlan(source)
    with dn.ctx(CHECK_PURITY=not enabled):
        plan.validate_source()
    assert plan.check_purity is enabled


@pytest.mark.parametrize("builder", ("derived", "timestep"))
def test_noise_dependent_builders_fail_closed_for_fresh_initialization(builder):
    class DerivedNoise(Mechanism):
        Mechanism.RANGENOISE("eta", mu=0.0, sigma=1.0, seed=17)
        Mechanism.DERIVED_BUFFER("static")

        def derive_buffers(self):
            return {"static": self.eta.square()}

    class TimestepNoise(Mechanism):
        Mechanism.RANGENOISE("eta", mu=0.0, sigma=1.0, seed=17)
        Mechanism.TIMESTEP_BUFFER("coefficient")

        def derive_timestep_buffers(self, dt):
            return {"coefficient": self.eta * dt}

    with dn.ctx(CHECK_PURITY=True):
        functional, tensors = dn.func.make_functional(
            _population(DerivedNoise if builder == "derived" else TimestepNoise), dt=DT
        )
    functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=functional.sample_noise_draw(tensors.state)),
    )
    with pytest.raises(dn.func.FunctionalizationError, match="NOISE-dependent"):
        functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )


def test_custom_noise_adapter_and_late_rng_override_fail_closed():
    class CustomAdapter(Mechanism):
        Mechanism.RANGENOISE("eta", seed=17)

        def sample_runtime_noises_(self, *names, **kwargs):
            super().sample_runtime_noises_(*names, **kwargs)
            self.eta.add_(100.0)
            return self

    with pytest.raises(
        dn.func.FunctionalizationError, match="canonical sample_runtime_noises_"
    ):
        dn.func.make_functional(_population(CustomAdapter), dt=DT)
    population = _population()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    rng = getattr(next(iter(population.mech.mechanisms.values())), "e_rng")
    rng.rand = lambda *args, **kwargs: None
    with pytest.raises(dn.func.FunctionalizationError, match="RNG"):
        functional.sample_noise_draw(tensors.state)


@pytest.mark.parametrize("kind", ("noise", "mixed", "random"))
def test_noise_stream_alias_rebinding_fails_closed(kind):
    class AliasedNoise(Mechanism):
        if kind in {"mixed", "random"}:
            Mechanism.GLOBALRAND("x", seed=10)
        else:
            Mechanism.GLOBALNOISE("x", seed=10)
        if kind == "random":
            Mechanism.GLOBALRAND("z", seed=20)
            Mechanism.GLOBALRAND("y", seed=30)
        else:
            Mechanism.GLOBALNOISE("z", seed=20)
            Mechanism.GLOBALNOISE("y", seed=30)

    population = _population(AliasedNoise)
    mechanism = next(iter(population.mech.mechanisms.values()))
    mechanism.y_rng = mechanism.x_rng
    functional, tensors = dn.func.make_functional(population, dt=DT)
    # Both canonical owners retain their IDs and first registration paths.
    # Rebinding this later alias therefore needs its own stream-topology guard.
    mechanism.y_rng = mechanism.z_rng
    with pytest.raises(dn.func.FunctionalizationError, match="RNG stream sharing"):
        functional.sample_noise_draw(tensors.state)


@pytest.mark.parametrize("kind", ("noise", "mixed", "random"))
@pytest.mark.parametrize("operation", ("extract", "commit"))
def test_compatible_population_requires_same_noise_stream_aliases(kind, operation):
    class AliasedNoise(Mechanism):
        if kind in {"mixed", "random"}:
            Mechanism.GLOBALRAND("x", seed=10)
        else:
            Mechanism.GLOBALNOISE("x", seed=10)
        if kind == "random":
            Mechanism.GLOBALRAND("z", seed=20)
            Mechanism.GLOBALRAND("y", seed=30)
        else:
            Mechanism.GLOBALNOISE("z", seed=20)
            Mechanism.GLOBALNOISE("y", seed=30)

    source, compatible = _population(AliasedNoise), _population(AliasedNoise)
    for population in (source, compatible):
        mechanism = next(iter(population.mech.mechanisms.values()))
        mechanism.y_rng = mechanism.x_rng
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def use_target():
        if operation == "extract":
            return functional.extract(compatible)
        return functional.commit_state_(compatible, tensors.state)

    use_target()
    mechanism = next(iter(compatible.mech.mechanisms.values()))
    mechanism.y_rng = mechanism.z_rng
    before = _clone(compatible.state_dict(), detached=True)
    with pytest.raises(dn.func.FunctionalizationError, match="RNG stream sharing"):
        use_target()
    _close(compatible.state_dict(), before)


def test_root_population_noise_is_initialization_only_and_pre_transform_order_fails_closed():
    from tests.func.test_population_initialization_transforms import _Add

    class NoisyPopulation(dn.SingleCompartment):
        dn.SingleCompartment.GLOBALNOISE("trial", mu=0.2, sigma=0.1, seed=19)

    population = NoisyPopulation(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    from dendra.models.mod import pas

    population.insert(pas)
    population.register_pre_initialize_transform(
        "shift",
        _Add(),
        reads=("parameters.trial_mu_param",),
        writes=("parameters.trial_mu_param",),
        inputs={"increment": torch.tensor(0.1)},
    )
    population.initialize()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    assert functional.sample_noise_draw(tensors.state) is None
    functional.prepare_and_step(tensors.parameters, tensors.constants, tensors.state)
    with pytest.raises(
        dn.func.FunctionalizationError, match="parameter-writing pre-initialize"
    ):
        functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )


def test_host_checkpoint_sampling_is_bounded_and_replay_reuses_packets(monkeypatch):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 5)
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.RolloutInput(noise=tape),
    )
    sample = functional.sample_noise_draw_tape
    spans = []

    def bounded(state, steps):
        spans.append(steps)
        return sample(state, steps)

    monkeypatch.setattr(functional, "sample_noise_draw_tape", bounded)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    actual, _ = dn.func.run(
        functional.bind(tensors.parameters, prepared),
        tensors.state,
        steps=5,
        checkpoint_every=2,
    )
    _close(actual, expected)
    assert spans == [2, 2, 1]
    actual["integrator"]["v"].sum().backward()
    assert spans == [2, 2, 1]


def test_custom_noise_tensor_tree_and_broadcast_result_use_shared_native_core():
    def sampler(rng, shape, *, device, dtype):
        return (
            rng.randn(shape, device=device, dtype=dtype),
            rng.rand(shape, device=device, dtype=dtype),
        )

    def transform(parameters, draw):
        return (parameters["offset"] + draw[0].mean() + draw[1].mean()).to(
            torch.float32
        )

    register_random_distribution(
        "functional_noise_tree",
        {"offset": 0.2},
        draw_sampler=sampler,
        transform=transform,
        overwrite=True,
    )

    class CustomNoise(Mechanism):
        Mechanism.RANGENOISE(
            "eta", distribution="functional_noise_tree", offset=0.3, seed=31
        )
        Mechanism.RANGE(g=0.001)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.AFFINE("i")

        def i_affine_terms(self, v):
            del v
            return self.g, self.g * -65.0 + self.eta

    source, native = _population(CustomNoise), _population(CustomNoise)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.StepInput(noise=functional.sample_noise_draw(initialized.state)),
    )
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    _native_step(native)
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("packed", (False, True))
def test_regional_and_packed_noise_scopes_match_native(packed):
    def build():
        if packed:
            population = dn.concat_models(
                {name: _population() for name in ("left", "right")}, threads=1
            )
        else:
            from dendra.models.mod import pas

            population = dn.SingleCompartment(
                N=2, C=3, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
            )
            population.insert(pas)
            population[:, :2].insert(_NoisyLeak)
        population.initialize()
        population.integrator._initialize(
            population, population.t.new_tensor(DT), force=True
        )
        return population

    source, native = build(), build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.StepInput(noise=functional.sample_noise_draw(initialized.state)),
    )
    _native_step(native)
    _close(actual, functional.extract(native).state)


def test_fresh_state_defaults_receive_final_nested_noise_sample():
    from dendra.models.mechanisms import State

    class NoiseInitialState(State):
        State.STATE("m")
        State.RANGENOISE("drive", mu=0.2, sigma=0.1, seed=37)
        State.DERIVATIVE("m' = -m")

        def state_defaults(self, v, values):
            del v, values
            return {"m": self.drive * 2.0}

    class NoiseInitialLeak(Mechanism):
        Mechanism.STATE_BUNDLE(NoiseInitialState)
        Mechanism.RANGE(g=0.001)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.AFFINE("i")

        def i_affine_terms(self, v):
            del v
            return self.g, self.g * -65.0

    source, native = _population(NoiseInitialLeak), _population(NoiseInitialLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
