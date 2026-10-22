"""Native RAND sampling shares explicit draws, gradients and RNG checkpoints."""

import copy
import math
from dataclasses import replace

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.random_parameters import (
    RandomParameterDrawRequest,
    evaluate_random_parameter,
    get_random_distribution,
    make_random_parameter_spec,
    register_random_distribution,
    sample_random_parameter,
    sample_random_parameter_draw,
    sample_random_parameter_draws,
    sample_random_parameter_with_draw,
)
from dendra.models.rng import RNGModule

pytestmark = pytest.mark.cpu


def _equal(left, right):
    values, structure = pytree.tree_flatten(left)
    references, other = pytree.tree_flatten(right)
    assert structure == other
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            assert value == reference


def _legacy_distribution(name, parameters, generator, shape, dtype):
    if name in {"normal", "lognormal"}:
        draw = torch.randn(shape, generator=generator, dtype=dtype)
        value = parameters["mu"] + parameters["sigma"] * draw
        return (value.exp() if name == "lognormal" else value), draw
    if name == "uniform":
        draw = torch.rand(shape, generator=generator, dtype=dtype)
        return parameters["low"] + (parameters["high"] - parameters["low"]) * draw, draw
    mu, sigma, low, high = (parameters[key] for key in ("mu", "sigma", "low", "high"))
    cdf_low = 0.5 * (1.0 + torch.erf(((low - mu) / sigma) / math.sqrt(2.0)))
    cdf_high = 0.5 * (1.0 + torch.erf(((high - mu) / sigma) / math.sqrt(2.0)))
    epsilon = torch.finfo(dtype).eps
    cdf_high = torch.maximum(cdf_high, cdf_low + epsilon)
    draw = torch.rand(shape, generator=generator, dtype=dtype)
    probability = torch.clamp(
        cdf_low + (cdf_high - cdf_low) * draw, epsilon, 1 - epsilon
    )
    value = mu + sigma * math.sqrt(2.0) * torch.erfinv(2 * probability - 1)
    return torch.minimum(torch.maximum(value, low), high), draw


@pytest.mark.parametrize("name", ("normal", "lognormal", "uniform", "truncated_normal"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((), (2, 3)))
def test_split_builtin_preserves_exact_native_values_draws_and_rng_bytes(
    name, dtype, shape
):
    declaration = make_random_parameter_spec("value", scope="range", distribution=name)
    parameters = {
        name: torch.tensor(value, dtype=dtype)
        for name, value in declaration.params.items()
    }
    rng = RNGModule(123, shape_p=shape, shape_f=shape)
    oracle = torch.Generator(device="cpu").manual_seed(123)
    expected, expected_draw = _legacy_distribution(
        name, parameters, oracle, shape, dtype
    )
    actual, draw = sample_random_parameter_with_draw(
        declaration, parameters, rng, shape, device="cpu", dtype=dtype
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(draw, expected_draw, rtol=0, atol=0)
    torch.testing.assert_close(
        rng.rng_state()["cpu"], oracle.get_state(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        evaluate_random_parameter(declaration, parameters, draw),
        expected,
        rtol=0,
        atol=0,
    )


def test_explicit_distribution_values_compile_map_and_preserve_reparameterization_policy():
    declaration = make_random_parameter_spec("value", scope="range")
    mu = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([0.5, 1.0], dtype=torch.float64, requires_grad=True)
    draw = torch.tensor([-0.25, 1.25], dtype=torch.float64)

    def evaluate(mu, sigma, draw):
        return evaluate_random_parameter(declaration, {"mu": mu, "sigma": sigma}, draw)

    expected = evaluate(mu, sigma, draw)
    actual = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(
        mu, sigma, draw
    )
    _equal(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), (mu, sigma))
    _equal(gradients, (torch.ones_like(mu), draw))
    _equal(torch.vmap(evaluate)(mu, sigma, draw), expected)
    detached = evaluate_random_parameter(
        replace(declaration, reparameterized=False), {"mu": mu, "sigma": sigma}, draw
    )
    assert not detached.requires_grad


def test_custom_split_distribution_and_legacy_native_sampler_contracts():
    def draw_sampler(rng, shape, *, device, dtype):
        return {
            "uniform": rng.rand(shape, device=device, dtype=dtype),
            "normal": rng.randn(shape, device=device, dtype=dtype),
        }

    def transform(parameters, draw):
        return parameters["offset"] + draw["uniform"] * draw["normal"]

    register_random_distribution(
        "split_native_test",
        {"offset": 2.0},
        draw_sampler=draw_sampler,
        transform=transform,
        overwrite=True,
    )
    declaration = make_random_parameter_spec(
        "value", scope="range", distribution="split_native_test"
    )
    rng = RNGModule(17, shape_p=(2,), shape_f=(2,))
    oracle = torch.Generator(device="cpu").manual_seed(17)
    uniform, normal = (
        torch.rand((2,), generator=oracle),
        torch.randn((2,), generator=oracle),
    )
    offset = torch.tensor(3.0, requires_grad=True)
    actual, draw = sample_random_parameter_with_draw(
        declaration, {"offset": offset}, rng, (2,), device="cpu", dtype=torch.float32
    )
    _equal(actual, offset + uniform * normal)
    _equal(draw, {"uniform": uniform, "normal": normal})
    _equal(rng.rng_state()["cpu"], oracle.get_state())
    _equal(torch.autograd.grad(actual.sum(), offset)[0], offset.new_tensor(2.0))

    def legacy_sampler(parameters, rng, shape, *, device, dtype):
        return parameters["offset"] + rng.rand(shape, device=device, dtype=dtype)

    register_random_distribution(
        "legacy_native_test", {"offset": 2.0}, sampler=legacy_sampler, overwrite=True
    )
    legacy = make_random_parameter_spec(
        "value", scope="range", distribution="legacy_native_test"
    )
    assert sample_random_parameter(
        legacy, {"offset": offset}, rng, (2,), device="cpu", dtype=torch.float32
    ).shape == (2,)
    with pytest.raises(ValueError, match="pure draw_sampler/transform"):
        sample_random_parameter_draw(
            legacy, rng, (2,), device="cpu", dtype=torch.float32
        )
    with pytest.raises(ValueError, match="supplied together"):
        register_random_distribution(
            "incomplete_native_test", {}, draw_sampler=draw_sampler, overwrite=True
        )


def test_host_draws_group_shared_streams_without_mutating_source_or_checkpoints():
    source = RNGModule(None, shape_p=(2,), shape_f=(2,))
    checkpoint = source.get_extra_state()
    frozen = copy.deepcopy(checkpoint)
    normal = get_random_distribution("normal")
    uniform = get_random_distribution("uniform")
    requests = (
        RandomParameterDrawRequest("shared", normal, (2,), "cpu", torch.float64),
        RandomParameterDrawRequest("shared", uniform, (3,), "cpu", torch.float64),
        RandomParameterDrawRequest("separate", normal, (2,), "cpu", torch.float64),
    )
    draws, states = sample_random_parameter_draws(
        requests, {"shared": checkpoint, "separate": checkpoint}
    )
    oracle = RNGModule.from_checkpoint(checkpoint)
    _equal(draws[0], oracle.randn((2,), device="cpu", dtype=torch.float64))
    _equal(draws[1], oracle.rand((3,), device="cpu", dtype=torch.float64))
    _equal(draws[2], draws[0])
    _equal(states["shared"], oracle.get_extra_state())
    assert states["shared"]["base_seed"] == checkpoint["base_seed"]
    _equal(source.get_extra_state(), frozen)
    _equal(checkpoint, frozen)
    continued, next_states = sample_random_parameter_draws(requests[:1], states)
    _equal(continued[0], oracle.randn((2,), device="cpu", dtype=torch.float64))
    _equal(next_states["shared"], oracle.get_extra_state())


class _SharedRandom(Mechanism):
    Mechanism.RANGERAND(
        "normal", distribution="normal", seed=123, rng_name="shared_rng"
    )
    Mechanism.RANGERAND(
        "uniform",
        distribution="uniform",
        seed=999,
        rng_name="shared_rng",
        resample_on_initialize=False,
    )

    def i(self, v):
        return torch.zeros_like(v)


def _mechanism():
    return _SharedRandom(None, torch.tensor(36.0), torch.ones(1, 3), (1, 3), (1, 3))


def test_native_provenance_retains_generation_policy_and_serializes_checkpoint_continuation():
    mechanism = _mechanism()
    first_uniform = mechanism.uniform.clone()
    checkpoint = copy.deepcopy(mechanism.state_dict())
    metadata = checkpoint["_random_parameter_state"]
    assert metadata["initialized"] == {"normal": True, "uniform": True}
    assert (
        set(metadata["draws"]) == set(metadata["parameters"]) == {"normal", "uniform"}
    )
    token = object()
    mechanism.populate_parameter_buffers(random_generation=token)
    _equal(mechanism.uniform, first_uniform)
    state_after_once = mechanism.shared_rng.get_extra_state()
    draw_after_once = mechanism._random_parameter_draws["normal"].clone()
    mechanism.populate_parameter_buffers(random_generation=token)
    _equal(mechanism.shared_rng.get_extra_state(), state_after_once)
    _equal(mechanism._random_parameter_draws["normal"], draw_after_once)

    restored = _mechanism()
    restored.load_state_dict(checkpoint)
    _equal(restored._random_parameter_draws, metadata["draws"])
    _equal(restored._random_parameter_parameters, metadata["parameters"])
    restored.resample_random_parameters()
    mechanism.load_state_dict(checkpoint)
    mechanism.resample_random_parameters()
    _equal(restored.normal, mechanism.normal)
    _equal(restored.uniform, mechanism.uniform)
    _equal(
        restored.shared_rng.get_extra_state(), mechanism.shared_rng.get_extra_state()
    )
    assert all(
        not value.requires_grad for value in pytree.tree_leaves(metadata["draws"])
    )


def test_legacy_random_checkpoint_invalidates_unrelated_draw_provenance():
    mechanism = _mechanism()
    legacy = copy.deepcopy(mechanism.state_dict())
    legacy.pop("_random_parameter_state")
    mechanism.resample_random_parameters()
    mechanism.load_state_dict(legacy, strict=True)
    assert mechanism._random_parameter_draws == {}
    assert mechanism._random_parameter_parameters == {}
    before = mechanism.uniform.clone()
    mechanism.populate_parameter_buffers(random_generation=object())
    _equal(mechanism.uniform, before)


def test_deepcopy_immutable_random_specs_preserves_tensor_aliases_without_graphs():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    declaration = make_random_parameter_spec(
        "value", scope="range", mu=parameter, sigma=parameter
    )
    cloned = copy.deepcopy(declaration)
    assert cloned is not declaration
    assert cloned.params["mu"] is cloned.params["sigma"]
    assert cloned.params["mu"] is not parameter
    assert cloned.params["mu"].requires_grad
    assert cloned.params["mu"].is_leaf
    with pytest.raises(TypeError):
        cloned.params["mu"] = 3
    replacement = torch.nn.Parameter(torch.tensor(4.0))
    assert (
        copy.deepcopy(declaration, {id(parameter): replacement}).params["mu"]
        is replacement
    )


class _MixedReseed(Mechanism):
    Mechanism.RANGERAND("before", seed=23, rng_name="shared_rng")
    Mechanism.RANGERAND(
        "reset", seed=23, rng_name="shared_rng", reseed_on_initialize=True
    )
    Mechanism.RANGERAND(
        "after", seed=23, rng_name="shared_rng", reseed_on_initialize=True
    )
    Mechanism.RANGERAND(
        "sticky",
        seed=23,
        rng_name="shared_rng",
        reseed_on_initialize=True,
        resample_on_initialize=False,
    )


def test_reseed_policy_matches_ordered_host_draws_and_keeps_sticky_and_manual_policy():
    mechanism = _MixedReseed(None, torch.tensor(36.0), torch.ones(1, 3), (1, 3), (1, 3))
    initial = mechanism.shared_rng.get_extra_state()
    sticky = mechanism.sticky.clone()
    requests = tuple(
        RandomParameterDrawRequest(
            "shared",
            mechanism.random_parameters[name],
            (1, 3),
            "cpu",
            mechanism.before.dtype,
        )
        for name in ("before", "reset", "after")
    )
    draws, checkpoints = sample_random_parameter_draws(requests, {"shared": initial})
    token = object()
    mechanism.populate_parameter_buffers(random_generation=token)
    for name, expected in zip(("before", "reset", "after"), draws, strict=True):
        _equal(mechanism._random_parameter_draws[name], expected)
    _equal(mechanism.shared_rng.get_extra_state(), checkpoints["shared"])
    _equal(mechanism.sticky, sticky)
    mechanism.populate_parameter_buffers(random_generation=token)
    _equal(mechanism.shared_rng.get_extra_state(), checkpoints["shared"])
    checkpoint = copy.deepcopy(mechanism.state_dict())
    restored = _MixedReseed(None, torch.tensor(36.0), torch.ones(1, 3), (1, 3), (1, 3))
    restored.load_state_dict(checkpoint)
    next_token = object()
    mechanism.populate_parameter_buffers(random_generation=next_token)
    restored.populate_parameter_buffers(random_generation=next_token)
    _equal(mechanism.reset, restored.reset)
    _equal(mechanism.after, restored.after)
    _equal(
        mechanism.shared_rng.get_extra_state(), restored.shared_rng.get_extra_state()
    )
    # Initialization resets to the current checkpointed base seed, including
    # after an explicit reseed, rather than to the declaration's original seed.
    mechanism.shared_rng.reseed(991)
    current = mechanism.shared_rng.get_extra_state()
    manual_requests = tuple(
        RandomParameterDrawRequest(
            "shared", spec, (1, 3), "cpu", mechanism.before.dtype
        )
        for spec in mechanism.random_parameters.values()
    )
    manual, states = sample_random_parameter_draws(
        manual_requests, {"shared": current}, initialization=False
    )
    mechanism.resample_random_parameters()
    for name, expected in zip(mechanism.random_parameters, manual, strict=True):
        _equal(mechanism._random_parameter_draws[name], expected)
    _equal(mechanism.shared_rng.get_extra_state(), states["shared"])
    mechanism.populate_parameter_buffers(random_generation=object())
    oracle = torch.Generator(device="cpu").manual_seed(991)
    _equal(
        mechanism._random_parameter_draws["reset"],
        torch.randn((1, 3), generator=oracle, dtype=mechanism.reset.dtype),
    )
    _equal(
        mechanism._random_parameter_draws["after"],
        torch.randn((1, 3), generator=oracle, dtype=mechanism.reset.dtype),
    )


def test_reseed_requests_coalesce_across_owners_sharing_one_rng_generation():
    class Reset(Mechanism):
        Mechanism.RANGERAND("value", seed=31, reseed_on_initialize=True)

    first, second = (
        Reset(None, torch.tensor(36.0), torch.ones(1, 3), (1, 3), (1, 3))
        for _ in range(2)
    )
    second.value_rng = first.value_rng
    token = object()
    first.populate_parameter_buffers(random_generation=token)
    second.populate_parameter_buffers(random_generation=token)
    oracle = torch.Generator(device="cpu").manual_seed(31)
    _equal(first.value, torch.randn((1, 3), generator=oracle, dtype=first.value.dtype))
    _equal(
        second.value, torch.randn((1, 3), generator=oracle, dtype=second.value.dtype)
    )
    _equal(first.value_rng.rng_state()["cpu"], oracle.get_state())


@pytest.mark.parametrize("kind", ("GLOBALRAND", "RANGERAND", "BATCHRAND"))
def test_all_random_declarations_expose_reseed_on_initialize_with_false_default(kind):
    class Configured(Mechanism):
        getattr(Mechanism, kind)("reset", seed=7, reseed_on_initialize=True)
        getattr(Mechanism, kind)("advance", seed=7)

    assert Configured._random_parameters["reset"].reseed_on_initialize
    assert not Configured._random_parameters["advance"].reseed_on_initialize


class _RandomPopulation(dn.SingleCompartment):
    dn.SingleCompartment.RANGERAND("population_random", seed=41)


def _random_population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE=torch.float64, DEVICE="cpu"):
        model = _RandomPopulation(N=1, C=3, integrator=dn.bwd_euler_sc(imem=False))
        model.insert(_SharedRandom)
        model.initialize()
        return model


def test_mutable_checkpoints_restore_population_and_mechanism_sample_provenance():
    model = _random_population()
    checkpoint = model.state_dict_for_checkpoint()
    mechanism = next(iter(model.mech.mechanisms.values()))
    metadata = checkpoint["stochastic"]["random_parameter_state"]
    mechanism_key = next(
        name for name in checkpoint["mech"] if name.endswith("random_parameter_state")
    )
    model.resample_random_parameters()
    model.restore_dict_from_checkpoint(checkpoint)
    _equal(model._random_parameter_state_for_checkpoint(), metadata)
    _equal(
        mechanism._random_parameter_state_for_checkpoint(),
        checkpoint["mech"][mechanism_key],
    )
    _equal(model.state_dict_for_checkpoint(), checkpoint)


def test_legacy_mutable_checkpoints_clear_constructor_draw_provenance():
    model = _random_population()
    checkpoint = model.state_dict_for_checkpoint()
    checkpoint["stochastic"].pop("random_parameter_state")
    for key in tuple(checkpoint["mech"]):
        if key.endswith("random_parameter_state"):
            checkpoint["mech"].pop(key)
    model.resample_random_parameters()
    model.restore_dict_from_checkpoint(checkpoint)
    assert model._random_parameter_draws == {}
    assert model._random_parameter_parameters == {}
    mechanism = next(iter(model.mech.mechanisms.values()))
    assert mechanism._random_parameter_draws == {}
    assert mechanism._random_parameter_parameters == {}


@pytest.mark.parametrize("owner", ("population", "mechanism"))
def test_malformed_mutable_provenance_rejects_without_publishing_state_or_rng(owner):
    model = _random_population()
    before = model.state_dict_for_checkpoint()
    bad = copy.deepcopy(before)
    if owner == "population":
        metadata = bad["stochastic"]["random_parameter_state"]
    else:
        key = next(
            name for name in bad["mech"] if name.endswith("random_parameter_state")
        )
        metadata = bad["mech"][key]
    first = next(iter(metadata["initialized"]))
    metadata["initialized"][first] = "invalid"
    bad["t"].add_(1)
    with pytest.raises(ValueError, match="initialized flags"):
        model.restore_dict_from_checkpoint(bad)
    _equal(model.state_dict_for_checkpoint(), before)
