"""Population RAND sampling belongs to explicit initialization boundaries."""

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.parametric import Parameterized
from dendra.models.random_parameters import register_random_distribution
from tests.func.test_network import _clone, _close
from tests.func.test_network_random_parameters import (
    _RandomLeak,
    _ReseededLeak,
    _StickyLeak,
)

pytestmark = pytest.mark.cpu
DT = 0.125


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


def _population(mechanism=_RandomLeak, *, batch=None):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism)
    if batch is not None:
        population.batch(batch)
    population.initialize()
    population.integrator._initialize(population, torch.tensor(DT), force=True)
    return population


def _step(population):
    population._step(
        population.integrator, population, population.t.new_tensor(DT), None, None
    )
    population.t = population.t + DT


def test_random_snapshot_step_uses_retained_samples_and_has_native_gradients():
    source, native = _population(), _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    actual, _ = functional.step(tensors.parameters, prepared, tensors.state)
    _step(native)
    _close(actual, functional.extract(native).state)
    g_name = next(name for name in tensors.parameters if name.endswith("g_mu_param"))
    actual_gradient = torch.autograd.grad(
        actual["integrator"]["v"].sum(), tensors.parameters[g_name]
    )[0]
    native_gradient = torch.autograd.grad(native.v.sum(), native.get_parameter(g_name))[
        0
    ]
    torch.testing.assert_close(actual_gradient, native_gradient)


def test_random_fresh_draws_are_explicit_and_rng_continuation_matches_native():
    source, native = _population(), _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    saved = _clone(tensors, detached=True)
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        random_draw=draw,
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
    continuation = functional.sample_initialization_draw(
        initialized.parameters, initialized.state
    )
    second = functional.initialize(
        initialized.parameters,
        initialized.constants,
        tensors.initialization,
        state=initialized.state,
        random_draw=continuation,
    )
    native.initialize()
    _close(second.state, functional.extract(native).state)
    _close(tensors, saved)
    with pytest.raises(ValueError, match="draw packet"):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )


def test_sticky_sample_preserves_forward_values_when_distribution_parameters_change():
    source = _population(_StickyLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    changed = {
        name: value + 0.7 if "g_mu_param" in name else value
        for name, value in tensors.parameters.items()
    }
    before = functional.prepare(tensors.parameters, tensors.constants).values
    after = functional.prepare(changed, tensors.constants).values
    key = next(key for key in before["mechanisms"] if key.endswith(".g"))
    torch.testing.assert_close(before["mechanisms"][key], after["mechanisms"][key])
    draw = functional.sample_initialization_draw(changed, tensors.state)
    initialized = functional.initialize(
        changed,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        random_draw=draw,
    )
    _close(initialized.state["random"], tensors.state["random"])


@pytest.mark.parametrize("lanes", (0, 2))
def test_random_pure_initialization_compiles_and_maps(lanes):
    source = _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    inputs = tensors.initialization._replace(random_draw=draw)
    compiled = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    actual = compiled(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )
    expected = functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )
    _close(actual, expected)
    name = next(name for name in tensors.parameters if name.endswith("g_mu_param"))
    values = (
        tensors.parameters[name].expand(lanes, *tensors.parameters[name].shape).clone()
    )
    mapped = torch.vmap(
        lambda value: functional.initialize(
            {**tensors.parameters, name: value},
            tensors.constants,
            inputs,
            state=tensors.state,
        ).state
    )(values)
    for leaf in _pytree.tree_leaves(mapped):
        assert leaf.shape[0] == lanes


def _draw(rng, shape, *, device, dtype):
    return rng.rand(shape, device=device, dtype=dtype)


def _transform(parameters, draw):
    return parameters["center"] + parameters["width"] * draw.square()


register_random_distribution(
    "func_test_pure",
    {"center": 0.1, "width": 0.2},
    sampler=lambda parameters, rng, shape, *, device, dtype: _transform(
        parameters, _draw(rng, shape, device=device, dtype=dtype)
    ),
    draw_sampler=_draw,
    transform=_transform,
    overwrite=True,
)


class _CustomLeak(Mechanism):
    Parameterized.GLOBALRAND(
        "g", distribution="func_test_pure", center=0.01, width=0.02, seed=17
    )
    Mechanism.GLOBAL(e=-65.0)

    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e


def test_custom_pure_distribution_uses_the_same_functional_path():
    source = _population(_CustomLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native = _population(_CustomLeak)
    native.initialize()
    _close(initialized.state, functional.extract(native).state)


def _float32_transform(parameters, draw):
    return _transform(parameters, draw).to(torch.float32)


register_random_distribution(
    "func_test_float32",
    {"center": 0.1, "width": 0.2},
    draw_sampler=_draw,
    transform=_float32_transform,
    overwrite=True,
)


class _ConvertedLeak(Mechanism):
    Mechanism.GLOBALRAND(
        "g", distribution="func_test_float32", center=0.01, width=0.02, seed=17
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        conductance = self.g + 1e-8
        return conductance, conductance * -65


@pytest.mark.parametrize("fresh", (False, True))
def test_custom_distribution_preserves_native_sample_conversion_and_commit(fresh):
    source, native = _population(_ConvertedLeak), _population(_ConvertedLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    if fresh:
        tensors = functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
        native.initialize()
        native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    sample = next(
        value
        for name, value in prepared.values["mechanisms"].items()
        if name.endswith(".g")
    )
    assert sample.dtype == native.mech._ConvertedLeak.g.dtype == torch.float64
    assert sample.device == native.mech._ConvertedLeak.g.device
    with torch.no_grad():
        actual, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
        _step(native)
    _close(actual, functional.extract(native).state)

    functional.commit_state_(source, actual)
    assert source.mech._ConvertedLeak.g.dtype == torch.float64
    assert source.mech._ConvertedLeak.g.requires_grad
    _step(source)
    _step(native)
    torch.testing.assert_close(source.v, native.v)
    gradients = tuple(
        torch.autograd.grad(
            population.v.square().sum(),
            population.mech._ConvertedLeak.g_center_param,
        )[0]
        for population in (source, native)
    )
    torch.testing.assert_close(*gradients)


class _StickyCustomLeak(_CustomLeak):
    Mechanism.GLOBALRAND(
        "g",
        distribution="func_test_pure",
        center=0.01,
        width=0.02,
        seed=17,
        resample_on_initialize=False,
    )


@pytest.mark.parametrize("fresh", (False, True))
@pytest.mark.parametrize("cached", (False, True))
def test_loaded_random_provenance_uses_the_target_draw_schema(fresh, cached):
    mechanism = _StickyCustomLeak if cached else _CustomLeak
    with dn.ctx(DTYPE=torch.float32):
        previous = _population(mechanism)
    source, native = _population(mechanism), _population(mechanism)
    for population in (source, native):
        getattr(population.mech, mechanism.__name__).load_state_dict(
            getattr(previous.mech, mechanism.__name__).state_dict()
        )
        if cached:
            population.cache("_steady_state")
    rng = getattr(source.mech, mechanism.__name__).g_rng
    checkpoint = rng.get_extra_state()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert all(
        value.dtype == torch.float64
        for name, value in tensors.constants.items()
        if name.startswith(("random.draws.", "random.parameters."))
    )
    torch.testing.assert_close(
        checkpoint["rng_state"]["cpu"],
        rng.get_extra_state()["rng_state"]["cpu"],
    )
    if fresh:
        tensors = functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
        native.initialize()
        native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    actual, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    _step(native)
    _close(actual, functional.extract(native).state)


def _mixed_draw(rng, shape, *, device, dtype):
    del dtype
    value = rng.rand(shape, device=device, dtype=torch.float32)
    return {"value": value, "index": (value > 0.5).to(torch.int64)}


def _mixed_transform(parameters, draw):
    return parameters["center"] + parameters["width"] * (draw["value"] + draw["index"])


register_random_distribution(
    "func_test_mixed_draw",
    {"center": 0.1, "width": 0.2},
    draw_sampler=_mixed_draw,
    transform=_mixed_transform,
    overwrite=True,
)


class _MixedDrawLeak(Mechanism):
    Mechanism.GLOBALRAND(
        "g", distribution="func_test_mixed_draw", center=0.01, width=0.02, seed=17
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * -65


@pytest.mark.parametrize("fresh", (False, True))
def test_dtype_conversion_preserves_custom_mixed_draw_contract(fresh):
    source, native = _population(_MixedDrawLeak), _population(_MixedDrawLeak)
    for population in (source, native):
        population.mech._MixedDrawLeak.to(dtype=torch.float32).to(dtype=torch.float64)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    draw_dtypes = tuple(
        value.dtype
        for name, value in tensors.constants.items()
        if name.startswith("random.draws.")
    )
    assert set(draw_dtypes) == {torch.float32, torch.int64}
    if fresh:
        tensors = functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
        native.initialize()
        native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    actual, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    _step(native)
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("seed", (-1, (1 << 64) - 1))
def test_random_rng_seed_metadata_round_trips_signed_and_unsigned_limits(seed):
    source = _population(_CustomLeak)
    source.mech._CustomLeak.g_rng.reseed(seed)
    source.initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(random_draw=draw),
        state=tensors.state,
    )
    functional.commit_state_(source, initialized.state)
    assert source.mech._CustomLeak.g_rng.get_extra_state()["base_seed"] == seed


class _RandomPopulation(dn.SingleCompartment):
    dn.SingleCompartment.GLOBALRAND("factor", mu=0.2, sigma=0.01, seed=91)


def test_population_rand_declaration_is_admitted_and_uses_native_scope():
    source = _RandomPopulation(N=2, C=2, integrator=dn.bwd_euler_sc(imem=False))
    source.insert(_CustomLeak)
    source.initialize()
    source.integrator._initialize(source, source.t.new_tensor(DT), force=True)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert tensors.constants["random.values.factor"].shape == ()
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    source.initialize()
    torch.testing.assert_close(
        initialized.constants["random.values.factor"], source.factor
    )


def test_reseeded_initialization_repeats_from_explicit_advanced_rng_carry():
    source = _population(_ReseededLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    first = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    second = functional.initialize_host(
        first.parameters, first.constants, tensors.initialization, state=first.state
    )
    _close(first.state["random"], second.state["random"])


class _DetachedLeak(Mechanism):
    Mechanism.GLOBALRAND("g", mu=0.01, sigma=0.001, seed=67, reparameterized=False)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * -65


def test_nonreparameterized_rand_cuts_distribution_and_explicit_draw_gradients():
    source = _population(_DetachedLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    draw = draw._replace(
        draws={name: value.requires_grad_() for name, value in draw.draws.items()}
    )
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(random_draw=draw),
        state=tensors.state,
    )
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    parameter = next(
        value
        for name, value in tensors.parameters.items()
        if name.endswith("g_mu_param")
    )
    gradients = torch.autograd.grad(
        actual["integrator"]["v"].sum(),
        (parameter, *draw.draws.values()),
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)


def _bad_transform(parameters, draw):
    draw.add_(0.5)
    return parameters["center"] + parameters["width"] * draw


register_random_distribution(
    "func_test_mutating",
    {"center": 0.01, "width": 0.001},
    sampler=lambda parameters, rng, shape, *, device, dtype: _bad_transform(
        parameters, _draw(rng, shape, device=device, dtype=dtype)
    ),
    draw_sampler=_draw,
    transform=_bad_transform,
    overwrite=True,
)


class _MutatingLeak(Mechanism):
    Mechanism.GLOBALRAND("g", distribution="func_test_mutating", seed=62)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * -65


def test_custom_distribution_cannot_mutate_its_explicit_draw_inputs():
    source = _population(_MutatingLeak)
    saved = source.mech._MutatingLeak.g.clone()
    with pytest.raises(dn.func.FunctionalizationError, match="in-place|mutated"):
        dn.func.make_functional(source, dt=DT)
    torch.testing.assert_close(source.mech._MutatingLeak.g, saved)


def test_sticky_initialization_takes_samples_from_explicit_state_not_old_constants():
    source = _population(_StickyLeak)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    state = _clone(tensors.state, detached=True)
    key = next(
        name for name in state["random"]["values"] if name.startswith("random.values.")
    )
    state["random"]["values"][key] = state["random"]["values"][key] * 1.25
    initialized = functional.initialize_host(
        tensors.parameters, tensors.constants, tensors.initialization, state=state
    )
    torch.testing.assert_close(
        initialized.constants[key], state["random"]["values"][key]
    )


@pytest.mark.parametrize("packed", (False, True))
def test_regional_and_packed_rand_scopes_match_native(packed):
    def build():
        if packed:
            source = dn.concat_models(
                {name: _population() for name in ("left", "right")}, threads=1
            )
        else:
            from dendra.models.mod import pas

            source = dn.SingleCompartment(
                N=2, C=3, v_init=-45, integrator=dn.bwd_euler_sc(imem=False)
            )
            source.insert(pas)
            source[:, :2].insert(_RandomLeak)
        source.initialize()
        source.integrator._initialize(source, source.t.new_tensor(DT), force=True)
        return source

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
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    _step(native)
    _close(actual, functional.extract(native).state)


def test_population_rand_parameter_writing_pre_transform_fails_closed_for_fresh_only():
    from tests.func.test_population_initialization_transforms import _Add

    source = _RandomPopulation(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    source.insert(_CustomLeak)
    source.register_pre_initialize_transform(
        "shift",
        _Add(),
        reads=("parameters.factor_mu_param",),
        writes=("parameters.factor_mu_param",),
        inputs={"increment": torch.tensor(0.1)},
    )
    source.initialize()
    source.integrator._initialize(source, source.t.new_tensor(DT), force=True)
    functional, tensors = dn.func.make_functional(source, dt=DT)
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


@pytest.mark.parametrize("field", ("version", "initialized"))
def test_rand_cache_rejects_unsupported_provenance_flags(field):
    source = _population(_StickyLeak)
    source.cache("_steady_state")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    cache = _clone(tensors.initialization.cache, detached=True)
    metadata = next(
        value
        for name, value in cache.items()
        if name.endswith("_random_parameter_state")
    )
    if field == "version":
        metadata["version"].fill_(2)
    else:
        metadata["initialized"]["g"].fill_(False)
    with pytest.raises(
        dn.func.FunctionalizationError, match="version 1 and initialized"
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization._replace(cache=cache),
        )


def test_plain_cache_tensors_keep_their_native_device():
    from dendra.func._random_parameters import cache_metadata_tensors

    value = torch.ones(2, device="cpu")
    assert cache_metadata_tensors(value, torch.device("meta")).device.type == "cpu"
    assert (
        cache_metadata_tensors({"rng": value}, torch.device("meta"))["rng"].device.type
        == "meta"
    )


def test_raw_native_cache_rejects_integer_initialized_flags_before_normalization():
    source = _population(_StickyLeak)
    source.cache("_steady_state")
    cache = source._caches["_steady_state"]
    key = next(name for name in cache if name.endswith("_random_parameter_state"))
    cache[key]["initialized"]["g"] = 1
    functional, tensors = dn.func.make_functional(source, dt=DT)
    functional.prepare_and_step(tensors.parameters, tensors.constants, tensors.state)
    with pytest.raises(dn.func.FunctionalizationError, match="initialized flags"):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )


@pytest.mark.parametrize("field", ("version", "initialized"))
def test_normalized_cache_provenance_dtypes_reject_before_reference_cast(field):
    source = _population(_StickyLeak)
    source.cache("_steady_state")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    cache = _clone(tensors.initialization.cache, detached=True)
    metadata = next(
        value
        for name, value in cache.items()
        if name.endswith("_random_parameter_state")
    )
    if field == "version":
        metadata["version"] = metadata["version"].to(torch.bool)
    else:
        metadata["initialized"]["g"] = metadata["initialized"]["g"].to(torch.int64)
    with pytest.raises(
        dn.func.FunctionalizationError, match="Boolean initialized flags"
    ):
        functional._owned_cache_plan.normalize(cache)


def test_explicit_population_batch_rand_shapes_match_native():
    source, native = _population(batch=3), _population(batch=3)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    _step(native)
    _close(actual, functional.extract(native).state)


def test_rand_declaration_does_not_admit_private_population_execution_overrides():
    class _AuthoredPoint(_RandomPopulation):
        def _step(self, *args, **kwargs):
            raise RuntimeError("custom numerical execution")

    source = _AuthoredPoint(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    source.insert(_CustomLeak)
    source.initialize()
    with pytest.raises(dn.func.FunctionalizationError, match="topologies"):
        dn.func.make_functional(source, dt=DT)


def test_population_rand_and_noise_share_native_initialization_stream():
    class _RandomNoisyPopulation(dn.SingleCompartment):
        dn.SingleCompartment.GLOBALRAND("trial_value", seed=17, rng_name="shared_rng")
        dn.SingleCompartment.GLOBALNOISE(
            "runtime_value", seed=17, rng_name="shared_rng"
        )

    source = _RandomNoisyPopulation(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    source.insert(_CustomLeak)
    source.initialize()
    rng = source.shared_rng.get_extra_state()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    _close(source.shared_rng.get_extra_state(), rng)
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    source.initialize()
    _close(initialized.state, functional.extract().state)
    assert functional.sample_noise_draw(tensors.state) is None
