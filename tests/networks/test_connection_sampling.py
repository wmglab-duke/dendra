"""Connection sampling shares native draw transforms without changing RNG order."""

import copy
import math
from types import MethodType

import pytest
import torch
from torch.nn.utils import parametrize

from dendra.models.distributions import Normal, TruncatedNormal
from dendra.models.networks.net import (
    connection_distribution_is_supported,
    make_weight,
)

pytestmark = pytest.mark.cpu


def _cdf(value):
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def _normal(dtype=torch.float64):
    return Normal(
        torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=dtype)),
        torch.nn.Parameter(torch.tensor(0.3, dtype=dtype)),
    )


def _truncated(dtype=torch.float64):
    return TruncatedNormal(
        torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=dtype)),
        torch.tensor(0.3, dtype=dtype),
        low=-0.3,
        high=1.0,
    )


def _old_truncated(source, n):
    uniform = torch.rand((n,) + source.mean.shape, device=source.mean.device)
    probability = _cdf(source._a) + uniform * (_cdf(source._b) - _cdf(source._a))
    return source.mean + source.std * (
        math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("factory", [_normal, _truncated])
def test_shared_draw_transform_matches_native_formula_values_gradients_and_rng(
    factory, dtype
):
    source = factory(dtype)
    torch.manual_seed(901)
    expected = (
        torch.distributions.Normal(source.mean, source.std.log().exp()).rsample((5,))
        if factory is _normal
        else _old_truncated(source, 5)
    )
    expected_rng = torch.get_rng_state()
    expected_grad = torch.autograd.grad(expected.sum(), source.mean)[0]
    torch.manual_seed(901)
    actual = source.sample(5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), source.mean)[0],
        expected_grad,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("factory", [_normal, _truncated])
def test_explicit_draw_generator_leaves_global_rng_and_transform_is_pure(factory):
    source = factory()
    generator = torch.Generator().manual_seed(191)
    reference = torch.Generator().manual_seed(191)
    global_rng = torch.get_rng_state()
    shape, dtype, device = source.sample_draw_schema((4,))
    actual = source.sample_draw((4,), generator=generator)
    expected = (
        torch.empty(shape, dtype=dtype, device=device).normal_(generator=reference)
        if factory is _normal
        else torch.rand(shape, dtype=dtype, device=device, generator=reference)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    source.sample_from_draw(actual).sum().backward()
    assert source.mean.grad is not None
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_normal_draw_broadcast_shape_and_invalid_scale_validation_preserved():
    source = Normal(torch.tensor(0.2), torch.tensor([0.3, 0.4]))
    assert source.sample_draw_schema((3,))[0] == torch.Size((3, 2))
    assert source.sample(3).shape == (3, 2)
    invalid = Normal(torch.tensor(0.0), torch.tensor(-1.0))
    rng = torch.get_rng_state()
    # Dynamo disables Torch distribution validation process-wide. Check its
    # enabled policy explicitly so this oracle is independent of suite order.
    previous_validation = torch.distributions.Distribution._validate_args
    try:
        torch.distributions.Distribution.set_default_validate_args(True)
        with pytest.raises(ValueError, match="scale"):
            invalid.sample(2)
    finally:
        torch.distributions.Distribution.set_default_validate_args(previous_validation)
    assert torch.equal(torch.get_rng_state(), rng)


def test_truncated_draw_dtype_and_unclamped_tail_formula_preserved():
    source = TruncatedNormal(torch.tensor(0.0, dtype=torch.float64), 1.0)
    assert source.sample_draw_schema((3,))[1] == torch.get_default_dtype()
    result = source.sample_from_draw(torch.tensor([0.0, 1.0]))
    assert result[0] == 0
    assert torch.isposinf(result[1])


@pytest.mark.parametrize(
    "low,high", [(-math.inf, math.inf), (0.0, math.inf), (-math.inf, 1.0)]
)
def test_infinite_truncation_bounds_have_finite_pathwise_scale_gradients(low, high):
    source = TruncatedNormal(
        torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64)),
        torch.tensor(0.3, dtype=torch.float64),
        low=low,
        high=high,
    )
    source._log_std.requires_grad_(True)
    draw = torch.tensor([0.15, 0.4, 0.8], dtype=torch.float64)
    values = source.sample_from_draw(draw)
    actual = torch.autograd.grad(values.sum(), (source.mean, source._log_std))
    for parameter, derivative in zip(
        (source.mean, source._log_std), actual, strict=True
    ):
        assert torch.isfinite(derivative)
        delta = 1e-5
        with torch.no_grad():
            parameter.add_(delta)
            above = source.sample_from_draw(draw).sum()
            parameter.sub_(2 * delta)
            below = source.sample_from_draw(draw).sum()
            parameter.add_(delta)
        torch.testing.assert_close(derivative, (above - below) / (2 * delta))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_infinite_truncation_bounds_preserve_forward_rng_and_log_prob(dtype):
    source = TruncatedNormal(
        torch.nn.Parameter(torch.tensor([0.2, 0.4, 0.1], dtype=dtype)),
        torch.tensor(0.3, dtype=dtype),
        low=torch.tensor([-math.inf, 0.0, -math.inf], dtype=dtype),
        high=torch.tensor([math.inf, math.inf, 1.0], dtype=dtype),
    )
    source._log_std.requires_grad_(True)
    torch.manual_seed(93)
    expected = _old_truncated(source, 4)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(93)
    actual = source.sample(4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    sample_gradients = torch.autograd.grad(actual.sum(), (source.mean, source._log_std))
    assert all(torch.isfinite(value).all() for value in sample_gradients)
    values = source.mean.detach() + 0.1
    normalizer = _cdf(source._b) - _cdf(source._a)
    expected_log_prob = (
        -0.5 * ((values - source.mean) / source.std) ** 2
        - torch.log(source.std * math.sqrt(2 * math.pi))
        - normalizer.log()
    )
    log_prob = source.log_prob(values)
    torch.testing.assert_close(log_prob, expected_log_prob, rtol=0, atol=0)
    log_gradients = torch.autograd.grad(log_prob.sum(), (source.mean, source._log_std))
    assert all(torch.isfinite(value).all() for value in log_gradients)


def test_infinite_bound_transform_supports_compile_jvp_and_vmap():
    source = TruncatedNormal(
        torch.nn.Parameter(torch.tensor([0.2, 0.4, 0.1], dtype=torch.float64)),
        torch.tensor(0.3, dtype=torch.float64),
        low=torch.tensor([-math.inf, 0.0, -math.inf], dtype=torch.float64),
        high=torch.tensor([math.inf, math.inf, 1.0], dtype=torch.float64),
    )
    source._log_std.requires_grad_(True)
    draw = torch.tensor([[0.15, 0.4, 0.8], [0.2, 0.5, 0.6]], dtype=torch.float64)

    class Evaluate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source = source

        def forward(self, draw):
            return self.source.sample_from_draw(draw)

    execution = Evaluate()

    def evaluate(mean, log_std):
        return torch.func.functional_call(
            execution,
            {"source.mean": mean, "source._log_std": log_std},
            (draw,),
        )

    args = (source.mean, source._log_std)
    expected = evaluate(*args)
    actual = torch.compile(evaluate, backend="eager", fullgraph=True)(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    primal, tangent = torch.func.jvp(
        evaluate, args, tuple(torch.ones_like(value) for value in args)
    )
    torch.testing.assert_close(primal, expected, rtol=0, atol=0)
    assert torch.isfinite(tangent).all()
    means = source.mean.expand(2, -1) + source.mean.new_tensor([[0.0], [0.05]])
    log_stds = source._log_std.expand(2)
    batched = torch.vmap(evaluate)(means, log_stds)
    reference = torch.stack(
        [evaluate(mean, scale) for mean, scale in zip(means, log_stds, strict=True)]
    )
    torch.testing.assert_close(batched, reference, rtol=0, atol=0)


@pytest.mark.parametrize("low,high", [(-0.3, 1.0), (-math.inf, math.inf)])
def test_native_truncated_subclass_retains_authored_standardized_bounds(low, high):
    class ShiftedTruncatedNormal(TruncatedNormal):
        @property
        def _a(self):
            return super()._a + 0.2

        @property
        def _b(self):
            return super()._b + 0.2

    source = ShiftedTruncatedNormal(
        torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64)),
        torch.tensor(0.3, dtype=torch.float64),
        low=low,
        high=high,
    )
    source._log_std.requires_grad_(True)
    assert not connection_distribution_is_supported(source)
    torch.manual_seed(31)
    expected = _old_truncated(source, 4)
    expected_rng = torch.get_rng_state()
    targets = (source.mean, source._log_std)
    expected_gradients = torch.autograd.grad(expected.sum(), targets)
    torch.manual_seed(31)
    weights = make_weight([source], [4]).init()
    torch.testing.assert_close(weights.w, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert weights.sampling_draws is None
    gradients = torch.autograd.grad(weights.w.sum(), targets)
    for observed, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        assert observed.isfinite().all()
    values = source.mean.detach() + 0.1
    normalizer = _cdf(source._b) - _cdf(source._a)
    expected_log_prob = (
        -0.5 * ((values - source.mean) / source.std) ** 2
        - torch.log(source.std * math.sqrt(2 * math.pi))
        - normalizer.log()
    )
    observed_log_prob = source.log_prob(values)
    torch.testing.assert_close(observed_log_prob, expected_log_prob, rtol=0, atol=0)
    expected_gradients = torch.autograd.grad(expected_log_prob.sum(), targets)
    gradients = torch.autograd.grad(observed_log_prob.sum(), targets)
    for observed, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        assert observed.isfinite().all()


def test_mixed_blocks_preserve_independent_draw_order_raw_sharing_and_retention():
    source = _normal(torch.float32)
    raw = torch.nn.Parameter(torch.tensor(0.7))
    weights = make_weight(
        [source, raw, source, _truncated(torch.float32)], [3, 2, 3, 4]
    )
    # Use scalar distributions for expander blocks: native concatenation is flat.
    source.mean = torch.nn.Parameter(source.mean[0].detach())
    truncated = weights.weights[3].param
    truncated.mean = torch.nn.Parameter(truncated.mean[0].detach())
    torch.manual_seed(52)
    expected = torch.cat(
        (
            source.sample(3),
            raw.repeat(2),
            source.sample(3),
            _old_truncated(truncated, 4),
        )
    )
    expected_rng = torch.get_rng_state()
    torch.manual_seed(52)
    weights.init()
    torch.testing.assert_close(weights.w, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert weights.weights[0].param is weights.weights[2].param
    assert set(weights.sampling_draws) == {"0", "2", "3"}
    assert not torch.equal(weights.sampling_draws["0"], weights.sampling_draws["2"])
    assert all(not draw.requires_grad for draw in weights.sampling_draws.values())
    values = weights.derive_values(weights.sampling_draws)
    torch.testing.assert_close(values, weights.w, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.autograd.grad(values.sum(), (source.mean, source.std, raw))
    retained_values, retained_draws = weights.w, weights.sampling_draws
    weights.init(reinit=False)
    assert weights.w is retained_values
    assert weights.sampling_draws is retained_draws
    assert torch.equal(torch.get_rng_state(), expected_rng)
    weights.init(reinit=True)
    assert not torch.equal(weights.w, retained_values)


def _scalar_weights():
    shared = Normal(torch.nn.Parameter(torch.tensor(0.2)), torch.tensor(0.3))
    return make_weight([shared, shared, torch.tensor(0.6)], [3, 2, 1])


def test_sampling_provenance_checkpoint_legacy_and_deepcopy():
    weights = _scalar_weights()
    with torch.no_grad():
        weights.init()
    state = weights.state_dict()
    assert "_extra_state" in state
    assert all(torch.is_tensor(value) for value in state["_extra_state"].values())
    restored = _scalar_weights().init()
    before_rng = torch.get_rng_state()
    restored.load_state_dict(state)
    assert torch.equal(torch.get_rng_state(), before_rng)
    torch.testing.assert_close(restored.w, weights.w, rtol=0, atol=0)
    for key, value in weights.sampling_draws.items():
        torch.testing.assert_close(restored.sampling_draws[key], value, rtol=0, atol=0)
    copied = copy.deepcopy(weights)
    assert copied.weights[0].param is copied.weights[1].param
    assert copied.weights[0].param is not weights.weights[0].param
    torch.testing.assert_close(copied.derive_values(copied.sampling_draws), copied.w)
    legacy = {key: value for key, value in state.items() if key != "_extra_state"}
    restored.load_state_dict(legacy)
    assert restored.sampling_draws is None
    restored.init(reinit=False)
    assert restored.sampling_draws is None
    assert torch.equal(torch.get_rng_state(), before_rng)
    restored.init()
    assert set(restored.sampling_draws) == {"0", "1"}


def test_deterministic_checkpoint_keys_unchanged_and_draw_operand_validation():
    weights = make_weight([torch.tensor(0.5)], [2]).init()
    assert set(weights.state_dict()) == {"w"}
    assert weights.sampling_draws == {}
    weights.load_state_dict({"w": weights.w.clone()})
    with pytest.raises(ValueError, match="do not take draws"):
        weights.derive_values({"0": torch.ones(2)})
    stochastic = _scalar_weights().init()
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match="match stochastic blocks"):
        stochastic.derive_values({"0": torch.ones(3)})
    with pytest.raises(ValueError, match="provenance"):
        stochastic.set_extra_state({"0": torch.ones(3)})
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("factory", [_normal, _truncated])
def test_sampling_checkpoint_load_preserves_torch_dtype_casting_and_hooks(factory):
    source = factory(torch.float32)
    source.mean = torch.nn.Parameter(source.mean[0].detach())
    original = make_weight([source], [4], dtype=torch.float32).init()
    target_source = factory(torch.float64)
    target_source.mean = torch.nn.Parameter(target_source.mean[0].detach())
    restored = make_weight([target_source], [4], dtype=torch.float64).init()
    handle = target_source.register_forward_hook(lambda _module, _args, value: value)
    try:
        restored.load_state_dict(original.state_dict())
        assert restored.w.dtype == torch.float64
        expected_dtype = (
            torch.float64 if factory is _normal else original.sampling_draws["0"].dtype
        )
        assert restored.sampling_draws["0"].dtype == expected_dtype
        restored.float()
        assert restored.sampling_draws["0"].dtype == torch.float32
    finally:
        handle.remove()


@pytest.mark.parametrize("factory", [_normal, _truncated])
def test_provenance_follows_module_dtype_and_recomputes_samples(factory):
    source = factory(torch.float32)
    source.mean = torch.nn.Parameter(source.mean[0].detach())
    weights = make_weight([source], [4]).init().double()
    expected_dtype = torch.float64 if factory is _normal else torch.get_default_dtype()
    assert weights.sampling_draws["0"].dtype == expected_dtype
    values = weights.derive_values(weights.sampling_draws)
    # TruncatedNormal retains Torch's scalar/array promotion for its uniforms.
    assert values.dtype == source.sample_from_draw(weights.sampling_draws["0"]).dtype
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_truncated_uniform_provenance_preserves_values_across_module_dtype_changes(
    device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        weights = make_weight(
            [TruncatedNormal(0.2, 0.3, low=-0.3, high=1.0)], [8]
        ).init()
        before = weights.sampling_draws["0"].clone()
        weights.to(device=device, dtype=torch.float32)
        draw = weights.sampling_draws["0"]
        assert draw.dtype == before.dtype
        assert draw.device.type == device
        torch.testing.assert_close(draw.cpu(), before, rtol=0, atol=0)
        weights.double().cpu()
        torch.testing.assert_close(weights.sampling_draws["0"], before, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous_dtype)


class _Double(torch.nn.Module):
    def forward(self, value):
        return value * 2


def test_parametrized_builtin_distribution_uses_same_pure_recipe_and_gradients():
    source = Normal(torch.nn.Parameter(torch.tensor(0.3)), torch.tensor(0.2))
    parametrize.register_parametrization(source, "mean", _Double())
    assert connection_distribution_is_supported(source)
    weights = make_weight([source], [4]).init()
    raw = source.parametrizations.mean.original
    gradient = torch.autograd.grad(weights.w.sum(), raw)[0]
    torch.testing.assert_close(gradient, raw.new_tensor(8.0))
    torch.testing.assert_close(weights.derive_values(weights.sampling_draws), weights.w)


def test_normal_sampling_resolves_parametrization_getters_once_with_provenance():
    source = Normal(torch.nn.Parameter(torch.tensor(0.3)), torch.tensor(0.2))
    parametrize.register_parametrization(source, "mean", _Double())
    calls = []
    handle = source.parametrizations.mean[0].register_forward_hook(
        lambda _module, _args, _value: calls.append(1)
    )
    try:
        source.sample(3)
        assert calls == [1]
        calls.clear()
        weights = make_weight([source], [3]).init()
        assert calls == [1]
        assert set(weights.sampling_draws) == {"0"}
    finally:
        handle.remove()


@pytest.mark.parametrize("execution", ["direct", "connection"])
def test_native_normal_subclass_retains_authored_distribution_sampling(execution):
    class ShiftedTorchNormal(torch.distributions.Normal):
        def rsample(self, sample_shape=torch.Size()):
            return super().rsample(sample_shape) + 2 * self.loc + 1

    class AuthoredNormal(Normal):
        @property
        def _dist(self):
            return ShiftedTorchNormal(self.mean, self._std)

    source = AuthoredNormal(
        torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64)),
    )
    assert not connection_distribution_is_supported(source)
    torch.manual_seed(31)
    expected = source._dist.rsample((4,))
    expected_rng = torch.get_rng_state()
    targets = (source.mean, source.std)
    expected_gradients = torch.autograd.grad(expected.sum(), targets)
    torch.manual_seed(31)
    if execution == "direct":
        observed = source.rsample((4,))
    else:
        weights = make_weight([source], [4]).init()
        observed = weights.w
        assert weights.sampling_draws is None
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    gradients = torch.autograd.grad(observed.sum(), targets)
    for actual, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("override", ["sample", "rsample", "hook", "subclass"])
def test_authored_distribution_dispatch_retains_native_fallback(override):
    class CustomNormal(Normal):
        def sample(self, n):
            return self.mean.expand(n) * 4

    source = (
        CustomNormal(torch.tensor(0.3), torch.tensor(0.2))
        if override == "subclass"
        else Normal(torch.tensor(0.3), torch.tensor(0.2))
    )
    handle = None
    if override == "sample":
        source.sample = MethodType(lambda self, n: self.mean.expand(n) * 4, source)
    elif override == "rsample":
        source.rsample = MethodType(
            lambda self, shape=torch.Size(): self.mean.expand(shape) * 4, source
        )
    elif override == "hook":
        handle = source.register_forward_hook(lambda _module, _args, value: value)
    try:
        assert not connection_distribution_is_supported(source)
        torch.manual_seed(144)
        expected = source.sample(3)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(144)
        weights = make_weight([source], [3]).init()
        torch.testing.assert_close(weights.w, expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert weights.sampling_draws is None
    finally:
        if handle is not None:
            handle.remove()


def test_expander_explicit_generators_preserve_global_rng():
    weights = _scalar_weights()
    rng = torch.get_rng_state()
    draws = weights.sample_draws(generator={"cpu": torch.Generator().manual_seed(319)})
    assert set(draws) == {"0", "1"}
    assert torch.equal(torch.get_rng_state(), rng)
    assert weights.w.numel() == 0
    assert weights.sampling_draws is None
    with pytest.raises(ValueError, match="No connection sampling generator"):
        weights.sample_draws(generator={})
