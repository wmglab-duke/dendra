"""Host draw plans can share isolated streams across Population boundaries."""

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population_cache import PopulationCachePlan
from dendra.func._random_parameters import _rng_extra, _rng_values
from dendra.models.random_parameters import (
    RandomParameterDrawRequest,
    make_random_parameter_spec,
    sample_random_parameter_draws,
)
from dendra.models.rng import RNGModule
from tests.func.test_network import _clone, _close
from tests.func.test_network_runtime_noise import _SharedRandomNoiseLeak
from tests.func.test_runtime_noise import DT, _population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


def test_shared_host_sampler_reseeds_once_per_population_initialization_boundary():
    source = RNGModule(481, shape_p=(), shape_f=())
    source.randn((3,), dtype=torch.float64)
    checkpoint = source.get_extra_state()
    frozen = _clone(checkpoint)
    shared = RNGModule.from_checkpoint(checkpoint)
    ordinary = make_random_parameter_spec("before", scope="range")
    reset = make_random_parameter_spec(
        "reset", scope="range", reseed_on_initialize=True
    )
    requests = tuple(
        RandomParameterDrawRequest("local", spec, (2,), "cpu", torch.float64)
        for spec in (ordinary, reset, reset)
    )
    first, first_checkpoint = sample_random_parameter_draws(
        requests, {}, samplers={"local": shared}
    )
    oracle = RNGModule.from_checkpoint(checkpoint)
    torch.testing.assert_close(first[0], oracle.randn((2,), dtype=torch.float64))
    oracle.reset()
    torch.testing.assert_close(first[1], oracle.randn((2,), dtype=torch.float64))
    torch.testing.assert_close(first[2], oracle.randn((2,), dtype=torch.float64))
    _close(first_checkpoint["local"], oracle.get_extra_state())

    second, second_checkpoint = sample_random_parameter_draws(
        requests, {}, samplers={"local": shared}
    )
    torch.testing.assert_close(second[0], oracle.randn((2,), dtype=torch.float64))
    oracle.reset()
    torch.testing.assert_close(second[1], oracle.randn((2,), dtype=torch.float64))
    torch.testing.assert_close(second[2], oracle.randn((2,), dtype=torch.float64))
    _close(second_checkpoint["local"], oracle.get_extra_state())
    _close(checkpoint, frozen)
    _close(source.get_extra_state(), frozen)


def test_initialization_plan_consumes_supplied_stream_without_nested_rng_carry():
    population = _population(_SharedRandomNoiseLeak)
    plan, tensors = dn.func.make_functional(population, dt=DT)
    initial = tensors.state["random"]["rng"]
    assert len(initial) == 1
    stream = next(iter(initial))
    shared = RNGModule.from_checkpoint(_rng_extra(initial[stream]))
    pool = {stream: shared}
    state = _clone(tensors.state)
    state["random"].pop("rng")
    before = _clone(state)
    expected = plan._random_plan.sample(tensors.state)
    first = plan._random_plan.sample(state, samplers=pool)
    _close(first, expected)
    expected = plan._random_plan.sample(
        {**tensors.state, "random": {**tensors.state["random"], "rng": first.rng}}
    )
    second = plan._random_plan.sample(state, samplers=pool)
    _close(second, expected)
    _close(second.rng[stream], _rng_values(shared, population.device()))
    _close(state, before)
    _close(plan.extract().state, tensors.state)


@pytest.mark.parametrize("steps", (0, 3))
def test_runtime_plan_tape_advances_supplied_stream_and_preserves_zero_boundary(steps):
    population = _population(_SharedRandomNoiseLeak)
    plan, tensors = dn.func.make_functional(population, dt=DT)
    pool = {
        stream: RNGModule.from_checkpoint(_rng_extra(checkpoint))
        for stream, checkpoint in tensors.state["random"]["rng"].items()
    }
    state = _clone(tensors.state)
    state["random"].pop("rng")
    before = _clone(state)
    expected = plan._noise_plan.sample(tensors.state, steps=steps)
    actual = plan._noise_plan.sample(state, steps=steps, samplers=pool)
    _close(actual, expected)
    final_rng = (
        _pytree.tree_map(lambda value: value[-1], actual.rng)
        if steps
        else tensors.state["random"]["rng"]
    )
    for stream, sampler in pool.items():
        _close(final_rng[stream], _rng_values(sampler, population.device()))
    _close(state, before)
    _close(plan.extract().state, tensors.state)


def test_cache_normalization_validates_late_registered_rng_aliases():
    population = _population(_SharedRandomNoiseLeak)
    plan, _tensors = dn.func.make_functional(population, dt=DT)
    random = plan._random_plan
    stream = next(iter(random._rng_bindings.values()))
    aliases = [
        f"{path}._extra_state"
        for path, module in population.named_modules(remove_duplicate=False)
        if module is stream
    ]
    assert len(aliases) > 1
    cache = _clone(population.state_dict())
    different = RNGModule(604, shape_p=(), shape_f=()).get_extra_state()
    cache[aliases[-1]] = different
    cache_plan = PopulationCachePlan(plan, cache)
    normalized = cache_plan.normalize(cache)
    assert not torch.equal(
        normalized[aliases[0]]["base_seed"]["seed"],
        normalized[aliases[-1]]["base_seed"]["seed"],
    )
    bad = _clone(cache)
    bad[aliases[-1]]["rng_state"]["cpu"].zero_()
    with pytest.raises((ValueError, RuntimeError), match="[Ss]tate"):
        cache_plan.normalize(bad)
