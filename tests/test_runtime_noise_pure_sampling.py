"""Detached NOISE shares explicit draws and native RNG ordering."""

import copy
import math

import pytest
import torch
from torch.utils import _pytree

from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.parametric import Parameterized
from dendra.models.random_parameters import (
    RandomParameterDrawRequest,
    RuntimeNoiseDrawRequest,
    evaluate_runtime_noise,
    iter_initialization_random_declarations,
    make_random_parameter_spec,
    make_runtime_noise_spec,
    register_random_distribution,
    sample_random_parameter_draws,
    sample_runtime_noise,
    sample_runtime_noise_draw,
    sample_runtime_noise_draws,
    sample_runtime_noise_with_draw,
)
from dendra.models.rng import RNGModule

pytestmark = pytest.mark.cpu


def _equal(left, right):
    leaves, structure = _pytree.tree_flatten(left)
    other, other_structure = _pytree.tree_flatten(right)
    assert structure == other_structure
    for value, reference in zip(leaves, other, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            assert value == reference


def _oracle(name, parameters, generator, shape, dtype):
    if name in {"normal", "lognormal"}:
        draw = torch.randn(shape, generator=generator, dtype=dtype)
        value = parameters["mu"] + parameters["sigma"] * draw
        return (value.exp() if name == "lognormal" else value), draw
    draw = torch.rand(shape, generator=generator, dtype=dtype)
    if name == "uniform":
        return parameters["low"] + (parameters["high"] - parameters["low"]) * draw, draw
    mu, sigma, low, high = (parameters[key] for key in ("mu", "sigma", "low", "high"))
    cdf_low = 0.5 * (1 + torch.erf(((low - mu) / sigma) / math.sqrt(2)))
    cdf_high = 0.5 * (1 + torch.erf(((high - mu) / sigma) / math.sqrt(2)))
    epsilon = torch.finfo(dtype).eps
    cdf_high = torch.maximum(cdf_high, cdf_low + epsilon)
    probability = torch.clamp(
        cdf_low + (cdf_high - cdf_low) * draw, epsilon, 1 - epsilon
    )
    value = mu + sigma * math.sqrt(2) * torch.erfinv(2 * probability - 1)
    return torch.minimum(torch.maximum(value, low), high), draw


@pytest.mark.parametrize("name", ("normal", "lognormal", "uniform", "truncated_normal"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((), (2, 3)))
@pytest.mark.parametrize("scale", ("standard", "dW", "white"))
def test_split_noise_preserves_native_values_scaling_and_rng(name, dtype, shape, scale):
    spec = make_runtime_noise_spec("eta", scope="range", distribution=name, scale=scale)
    parameters = {
        key: torch.tensor(value, dtype=dtype, requires_grad=True)
        for key, value in spec.params.items()
    }
    rng = RNGModule(123, shape_p=shape, shape_f=shape)
    generator = torch.Generator().manual_seed(123)
    expected, expected_draw = _oracle(name, parameters, generator, shape, dtype)
    dt = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    if scale == "dW":
        expected = expected * torch.sqrt(dt)
    elif scale == "white":
        expected = expected / torch.sqrt(dt)
    actual, draw = sample_runtime_noise_with_draw(
        spec, parameters, rng, shape, device="cpu", dtype=dtype, dt=dt
    )
    _equal(actual, expected)
    _equal(draw, expected_draw)
    _equal(rng.rng_state()["cpu"], generator.get_state())
    assert not actual.requires_grad
    actual = evaluate_runtime_noise(spec, parameters, draw.requires_grad_(), dt=dt)
    _equal(actual, expected)
    assert not actual.requires_grad


@pytest.mark.parametrize("scale", ("standard", "dW", "white"))
def test_detached_noise_evaluation_compiles_and_maps(scale):
    spec = make_runtime_noise_spec("eta", scope="range", scale=scale)
    mu = torch.tensor([1.0, 2.0], requires_grad=True)
    sigma = torch.tensor([0.5, 0.25], requires_grad=True)
    draw = torch.tensor([-0.25, 1.25], requires_grad=True)
    dt = torch.tensor(0.25, requires_grad=True)

    def evaluate(mu, sigma, draw, dt):
        return evaluate_runtime_noise(spec, {"mu": mu, "sigma": sigma}, draw, dt=dt)

    expected = evaluate(mu, sigma, draw, dt)
    compiled = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    _equal(compiled(mu, sigma, draw, dt), expected)
    mapped = torch.vmap(evaluate, in_dims=(0, 0, 0, None))(mu, sigma, draw, dt)
    _equal(mapped, expected)
    assert not expected.requires_grad
    assert not mapped.requires_grad
    assert not compiled(mu, sigma, draw, dt).requires_grad
    if scale != "standard":
        with pytest.raises(ValueError, match="requires dt"):
            evaluate_runtime_noise(spec, {"mu": mu, "sigma": sigma}, draw)


def test_custom_split_tree_and_legacy_noise_sampler_remain_detached():
    def draw_sampler(rng, shape, *, device, dtype):
        return {
            "uniform": rng.rand(shape, device=device, dtype=dtype),
            "normal": rng.randn(shape, device=device, dtype=dtype),
        }

    def transform(parameters, draw):
        return parameters["offset"] + draw["uniform"] * draw["normal"]

    register_random_distribution(
        "split_runtime_noise_test",
        {"offset": 2.0},
        draw_sampler=draw_sampler,
        transform=transform,
        overwrite=True,
    )
    spec = make_runtime_noise_spec(
        "eta", scope="range", distribution="split_runtime_noise_test", scale="dW"
    )
    rng = RNGModule(17, shape_p=(2,), shape_f=(2,))
    oracle = torch.Generator().manual_seed(17)
    uniform = torch.rand((2,), generator=oracle)
    normal = torch.randn((2,), generator=oracle)
    offset = torch.tensor(3.0, requires_grad=True)
    actual, draw = sample_runtime_noise_with_draw(
        spec, {"offset": offset}, rng, (2,), device="cpu", dtype=torch.float32, dt=0.25
    )
    _equal(actual, (offset + uniform * normal) * 0.5)
    _equal(draw, {"uniform": uniform, "normal": normal})
    assert not actual.requires_grad

    def legacy(parameters, rng, shape, *, device, dtype):
        return parameters["offset"] + rng.rand(shape, device=device, dtype=dtype)

    register_random_distribution(
        "legacy_runtime_noise_test", {"offset": 2.0}, sampler=legacy, overwrite=True
    )
    legacy_spec = make_runtime_noise_spec(
        "eta", scope="range", distribution="legacy_runtime_noise_test"
    )
    actual, draw = sample_runtime_noise_with_draw(
        legacy_spec, {"offset": offset}, rng, (2,), device="cpu", dtype=torch.float32
    )
    assert not actual.requires_grad
    assert draw is None
    with pytest.raises(ValueError, match="pure draw_sampler/transform"):
        sample_runtime_noise_draw(
            legacy_spec, rng, (2,), device="cpu", dtype=torch.float32
        )


def test_mixed_initialization_draws_and_runtime_suffix_preserve_shared_stream_order():
    rng = RNGModule(123, shape_p=(2,), shape_f=(2,))
    rng.randn((2,))
    checkpoint = rng.get_extra_state()
    frozen = copy.deepcopy(checkpoint)
    noise = make_runtime_noise_spec("eta", scope="range", distribution="uniform")
    random = make_random_parameter_spec(
        "quenched", scope="range", reseed_on_initialize=True
    )
    requests = (
        RuntimeNoiseDrawRequest("shared", noise, (2,), "cpu", torch.float64),
        RandomParameterDrawRequest("shared", random, (3,), "cpu", torch.float64),
        RuntimeNoiseDrawRequest("shared", noise, (4,), "cpu", torch.float64),
    )
    draws, states = sample_random_parameter_draws(requests, {"shared": checkpoint})
    oracle = RNGModule.from_checkpoint(checkpoint)
    _equal(draws[0], oracle.rand((2,), dtype=torch.float64))
    oracle.reseed(123)
    _equal(draws[1], oracle.randn((3,), dtype=torch.float64))
    _equal(draws[2], oracle.rand((4,), dtype=torch.float64))
    _equal(states["shared"], oracle.get_extra_state())
    suffix, suffix_states = sample_runtime_noise_draws(requests[-1:], states)
    _equal(suffix[0], oracle.rand((4,), dtype=torch.float64))
    _equal(suffix_states["shared"], oracle.get_extra_state())
    _equal(checkpoint, frozen)
    _equal(rng.get_extra_state(), frozen)
    assert all(not value.requires_grad for value in draws)


@pytest.mark.parametrize("scale", ("standard", "dW", "white"))
def test_native_noise_buffers_retain_identity_and_initialization_uses_unit_dt(scale):
    class Noise(Parameterized):
        Parameterized.GLOBALNOISE(
            "scalar", distribution="uniform", seed=17, scale=scale
        )
        Parameterized.RANGENOISE("eta", seed=123, scale=scale)
        Parameterized.BATCHNOISE("batch", seed=19, scale=scale)

    model = Noise(shape=(2, 3), shape_f=(2, 3), dtype=torch.float64)
    buffers = {name: getattr(model, name) for name in model.runtime_noises}
    checkpoints = {
        name: getattr(model, spec.effective_rng_name).get_extra_state()
        for name, spec in model.runtime_noises.items()
    }
    model.populate_parameter_buffers()
    for name, spec in model.runtime_noises.items():
        value = getattr(model, name)
        parameters = {
            key: getattr(model, spec.full_parameter_name(key)) for key in spec.params
        }
        oracle = RNGModule.from_checkpoint(checkpoints[name])
        expected = sample_runtime_noise(
            spec,
            parameters,
            oracle,
            tuple(value.shape),
            device=value.device,
            dtype=value.dtype,
            dt=1.0,
        )
        _equal(value, expected)
        assert value is buffers[name]
        assert not value.requires_grad
    assert model.scalar.shape == ()
    assert model.eta.shape == (2, 3)
    assert model.batch.shape == (2, 1)
    before = {name: getattr(model, name).clone() for name in buffers}
    assert not model.sample_runtime_noises_(dt=0.25, phase="post_state")
    _equal(before, {name: getattr(model, name) for name in buffers})
    assert model.sample_runtime_noises_(dt=0.25)
    assert all(getattr(model, name) is value for name, value in buffers.items())


def test_native_nested_initialization_and_runtime_order_replay_from_checkpoints():
    class NoisyState(State):
        State.STATE("x")
        State.DERIVATIVE("x' = 0.0")
        State.RANGERAND("fixed", seed=3, rng_name="shared_rng")
        State.RANGENOISE("state_eta", seed=3, rng_name="shared_rng")

    class NoisyMechanism(Mechanism):
        Mechanism.RANGERAND("fixed", seed=3, rng_name="shared_rng")
        Mechanism.RANGENOISE("eta", seed=3, rng_name="shared_rng")
        Mechanism.STATE_BUNDLE(NoisyState)

    shape = (1, 3)
    mechanisms = {
        name: NoisyMechanism(name, torch.tensor(36.0), torch.ones(shape), shape, shape)
        for name in ("first", "second")
    }
    handler = MechanismHandler(torch.tensor(36.0), torch.ones(shape), mechanisms)
    handler.populate(random_generation=object())
    for mechanism in mechanisms.values():
        state = next(iter(mechanism.DE.values()))
        state.shared_rng = mechanism.shared_rng
        assert [
            (source is mechanism, kind, name)
            for source, kind, name, _ in iter_initialization_random_declarations(
                mechanism
            )
        ] == [
            (True, "random", "fixed"),
            (True, "noise", "eta"),
            (False, "noise", "state_eta"),
            (False, "random", "fixed"),
            (False, "noise", "state_eta"),
        ]
    checkpoints = {
        name: mech.shared_rng.get_extra_state() for name, mech in mechanisms.items()
    }
    handler.populate(random_generation=object())
    for name, mechanism in mechanisms.items():
        oracle = RNGModule.from_checkpoint(checkpoints[name])
        state = next(iter(mechanism.DE.values()))
        # Mechanism.populate samples nested NOISE through its virtual sampler,
        # then State.populate samples RAND followed by another NOISE draw.
        _equal(mechanism.fixed, oracle.randn(shape))
        _equal(mechanism.eta, oracle.randn(shape))
        oracle.randn(shape)
        _equal(state.fixed, oracle.randn(shape))
        _equal(state.state_eta, oracle.randn(shape))
        _equal(mechanism.shared_rng.get_extra_state(), oracle.get_extra_state())
    checkpoint = copy.deepcopy(handler.mutable_state_dict())
    handler.sample_runtime_noises_(dt=0.1)
    expected = copy.deepcopy(handler.mutable_state_dict())
    handler.restore_mutable_state_dict(checkpoint)
    handler.sample_runtime_noises_(dt=0.1)
    _equal(handler.mutable_state_dict(), expected)
