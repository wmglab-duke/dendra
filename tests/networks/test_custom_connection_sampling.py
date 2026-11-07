"""Custom connection recipes share explicit draws with native materialization."""

import copy
from types import MethodType

import pytest
import torch
from torch.nn.utils import parametrize

from dendra.models.distributions import (
    Distribution,
    Normal,
    connection_distribution_draw_schema,
    connection_distribution_is_supported,
    connection_distribution_placeholder,
)
from dendra.models.networks.net import make_weight

pytestmark = pytest.mark.cpu


class Triangular(Distribution):
    """Two independent uniforms transformed into a scalar connection value."""

    def __init__(self, location=None, scale=None):
        super().__init__()
        self.location = (
            torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
            if location is None
            else location
        )
        self.scale = (
            torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
            if scale is None
            else scale
        )

    def sample_draw_schema(self, sample_shape=torch.Size()):
        return (
            torch.Size(sample_shape) + (2,),
            self.location.dtype,
            self.location.device,
        )

    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def sample_from_draw(self, draw):
        return self.location + self.scale * (draw.sum(dim=-1) - 1)


def test_custom_draw_adapters_preserve_values_gradients_and_rng_order():
    source = Triangular()
    assert connection_distribution_is_supported(source)
    torch.manual_seed(17)
    draw = torch.rand((5, 2), dtype=source.location.dtype)
    expected = source.location + source.scale * (draw.sum(-1) - 1)
    expected_rng = torch.get_rng_state()
    gradients = torch.autograd.grad(expected.sum(), tuple(source.parameters()))
    torch.manual_seed(17)
    actual, observed_draw = source.sample_with_draw((5,))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(observed_draw, draw, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), tuple(source.parameters())),
        gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_custom_sampling_adapters_keep_base_noise_outside_autograd():
    class ExplicitTrainableDraw(Triangular):
        def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
            shape, dtype, device = self.sample_draw_schema(sample_shape)
            # Even an explicitly grad-enabled authored result is base noise,
            # rather than a second path through distribution parameters.
            return torch.ones(shape, dtype=dtype, device=device, requires_grad=True)

    source = ExplicitTrainableDraw()
    value, draw = source.sample_with_draw((3,))
    assert not draw.requires_grad
    gradient = torch.autograd.grad(value.sum(), source.scale)[0]
    torch.testing.assert_close(gradient, source.scale.new_tensor(3.0))
    sampled = make_weight([source], [3]).sample_draws()
    assert not sampled["0"].requires_grad


def test_custom_expander_retains_raw_aliases_independent_draws_and_checkpoints():
    source = Triangular()
    weights = make_weight([source, source], [4, 3]).init()
    assert weights.weights[0].param is weights.weights[1].param
    assert weights.weights[0].param.location is source.location
    assert weights.sampling_draws["0"].shape == (4, 2)
    assert weights.sampling_draws["1"].shape == (3, 2)
    state = copy.deepcopy(weights.state_dict())
    rng = torch.get_rng_state()
    values = weights.derive_values(weights.sampling_draws)
    torch.testing.assert_close(values, weights.w, rtol=0, atol=0)
    gradient = torch.autograd.grad(values.sum(), source.location)[0]
    torch.testing.assert_close(gradient, source.location.new_tensor(7.0))
    retained, retained_draws = weights.w, weights.sampling_draws
    weights.init(reinit=False)
    assert weights.w is retained
    assert weights.sampling_draws is retained_draws
    assert torch.equal(torch.get_rng_state(), rng)
    weights.w = weights.w.detach()
    restored = copy.deepcopy(weights)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.w, values, rtol=0, atol=0)
    torch.testing.assert_close(restored.derive_values(restored.sampling_draws), values)
    assert torch.equal(torch.get_rng_state(), rng)
    legacy = {key: value for key, value in state.items() if key != "_extra_state"}
    restored.load_state_dict(legacy)
    assert restored.sampling_draws is None
    restored.init(reinit=False)
    assert restored.sampling_draws is None
    restored.init()
    assert restored.sampling_draws["0"].shape == (4, 2)


def test_explicit_generator_sampling_leaves_global_rng_and_model_unmodified():
    source = Triangular()
    weights = make_weight([source, Normal(0.1, 0.2), source], [4, 2, 3])
    generator = torch.Generator().manual_seed(31)
    reference = torch.Generator().manual_seed(31)
    global_rng = torch.get_rng_state()
    draws = weights.sample_draws(generator={"cpu": generator})
    expected = (
        source.sample_draw((4,), generator=reference),
        weights.weights[1].param.sample_draw((2,), generator=reference),
        source.sample_draw((3,), generator=reference),
    )
    for draw, target in zip(draws.values(), expected, strict=True):
        torch.testing.assert_close(draw, target, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert weights.w.numel() == 0
    assert weights.sampling_draws is None
    result = weights.derive_values(draws)
    assert result.shape == (9,)
    assert torch.equal(torch.get_rng_state(), global_rng)


class _LearnedScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, value):
        return self.factor * value


def test_parametrized_custom_distribution_preserves_trainable_raw_sources():
    source = Triangular()
    raw = source.location
    transform = _LearnedScale()
    parametrize.register_parametrization(source, "location", transform)
    assert connection_distribution_is_supported(source)
    weights = make_weight([source, source], [4, 3]).init()
    assert source.parametrizations.location.original is raw
    gradients = torch.autograd.grad(weights.w.sum(), (raw, transform.factor))
    torch.testing.assert_close(gradients[0], raw.new_tensor(14.0))
    torch.testing.assert_close(gradients[1], 7 * raw)
    torch.testing.assert_close(weights.derive_values(weights.sampling_draws), weights.w)


def test_custom_draw_dtype_is_preserved_on_to_and_checkpoint_load():
    with torch.no_grad():
        original = make_weight([Triangular()], [4]).init()
    initial_draw = original.sampling_draws["0"].clone()
    copied = copy.deepcopy(original).float()
    assert copied.sampling_draws["0"].dtype == torch.float64
    torch.testing.assert_close(copied.sampling_draws["0"], initial_draw, rtol=0, atol=0)
    restored = make_weight([Triangular().float()], [4], dtype=torch.float32).init()
    rng = torch.get_rng_state()
    restored.load_state_dict(original.state_dict())
    assert restored.sampling_draws["0"].dtype == torch.float64
    torch.testing.assert_close(
        restored.sampling_draws["0"], initial_draw, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.derive_values(restored.sampling_draws),
        restored.w,
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.equal(torch.get_rng_state(), rng)


def test_custom_schema_fixed_dtype_keeps_provenance_precision_on_to():
    class FixedDrawDtype(Triangular):
        def sample_draw_schema(self, sample_shape=torch.Size()):
            shape, _, device = super().sample_draw_schema(sample_shape)
            return shape, torch.float64, device

    weights = make_weight([FixedDrawDtype()], [4]).init()
    initial_draw = weights.sampling_draws["0"].clone()
    weights.float()
    assert weights.sampling_draws["0"].dtype == torch.float64
    torch.testing.assert_close(
        weights.sampling_draws["0"], initial_draw, rtol=0, atol=0
    )
    assert torch.isfinite(weights.derive_values(weights.sampling_draws)).all()


def test_custom_retained_draw_dtype_survives_changed_sampling_default_dtype():
    class DefaultDrawDtype(Triangular):
        def sample_draw_schema(self, sample_shape=torch.Size()):
            shape, _, device = super().sample_draw_schema(sample_shape)
            return shape, torch.get_default_dtype(), device

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        weights = make_weight([DefaultDrawDtype()], [4]).init()
        retained = weights.w
        draw = weights.sampling_draws["0"].clone()
        torch.set_default_dtype(torch.float32)
        rng = torch.get_rng_state()
        actual = weights.derive_values(weights.sampling_draws)
        torch.testing.assert_close(actual, retained, rtol=0, atol=0)
        torch.testing.assert_close(weights.sampling_draws["0"], draw, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), rng)
        assert weights.sample_draws()["0"].dtype == torch.float32
    finally:
        torch.set_default_dtype(previous_dtype)


def test_custom_checkpoint_retains_historic_dtype_with_unchanged_model_dtype():
    class DefaultDrawDtype(Triangular):
        def sample_draw_schema(self, sample_shape=torch.Size()):
            shape, _, device = super().sample_draw_schema(sample_shape)
            return shape, torch.get_default_dtype(), device

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        source = DefaultDrawDtype()
        original = make_weight([source], [8], dtype=torch.float64).init()
        checkpoint = copy.deepcopy(original.state_dict())
        expected = original.derive_values(original.sampling_draws)
        gradients = torch.autograd.grad(
            expected.square().sum(), tuple(source.parameters())
        )
        torch.set_default_dtype(torch.float32)
        restored_source = DefaultDrawDtype()
        restored = make_weight([restored_source], [8], dtype=torch.float64).init()
        rng = torch.get_rng_state()
        restored.load_state_dict(checkpoint)
        actual = restored.derive_values(restored.sampling_draws)
        assert restored.sampling_draws["0"].dtype == torch.float64
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(
                actual.square().sum(), tuple(restored_source.parameters())
            ),
            gradients,
            strict=True,
        ):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize(
    "invalid", [torch.ones(4), torch.ones((4, 2), dtype=torch.int64), [0.5]]
)
def test_custom_checkpoint_provenance_validates_multidimensional_schema(invalid):
    weights = make_weight([Triangular()], [4]).init()
    previous = weights.sampling_draws
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match="Invalid connection sampling provenance"):
        weights.set_extra_state({"0": invalid})
    assert weights.sampling_draws is previous
    assert torch.equal(torch.get_rng_state(), rng)


def test_custom_inverse_cdf_placeholder_is_safe_without_sampling():
    class Exponential(Triangular):
        def sample_draw_placeholder(self, sample_shape=torch.Size()):
            shape, dtype, device = self.sample_draw_schema(sample_shape)
            return torch.full(shape, 0.5, dtype=dtype, device=device)

        def sample_from_draw(self, draw):
            return self.location - self.scale * torch.log(draw[..., 0])

    source = Exponential()
    rng = torch.get_rng_state()
    draw = connection_distribution_placeholder(source, (4,))
    assert torch.isfinite(source.sample_from_draw(draw)).all()
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(draw, torch.full((4, 2), 0.5, dtype=torch.float64))


@pytest.mark.parametrize("override", ["sample", "rsample", "sample_with_draw"])
def test_authored_legacy_sampling_adapters_keep_native_fallback(override):
    source = Triangular()
    if override == "sample":
        source.sample = MethodType(lambda self, n: self.location.expand(n) * 3, source)
    elif override == "rsample":
        source.rsample = MethodType(
            lambda self, shape=torch.Size(): self.location.expand(shape) * 3,
            source,
        )
    else:
        source.sample_with_draw = MethodType(
            lambda self, shape=torch.Size(), generator=None: (
                self.location.expand(shape) * 3,
                torch.zeros(tuple(shape) + (2,), dtype=self.location.dtype),
            ),
            source,
        )
    assert not connection_distribution_is_supported(source)
    weights = make_weight([source], [4]).init()
    torch.testing.assert_close(weights.w, source.location.expand(4) * 3)
    assert weights.sampling_draws is None


def test_normal_subclass_can_explicitly_adopt_generic_sampling_adapters():
    class LogNormal(Normal):
        sample = Distribution.sample
        rsample = Distribution.rsample
        sample_with_draw = Distribution.sample_with_draw

        def sample_from_draw(self, draw):
            return super().sample_from_draw(draw).exp()

    source = LogNormal(torch.tensor(0.2), torch.tensor(0.3))
    assert connection_distribution_is_supported(source)
    weights = make_weight([source], [4]).init()
    assert (weights.w > 0).all()
    torch.testing.assert_close(
        weights.w, source.sample_from_draw(weights.sampling_draws["0"])
    )


@pytest.mark.parametrize(
    "schema",
    [
        ((3,), torch.int64, "cpu"),
        ((-1,), torch.float32, "cpu"),
        ((3,), torch.float32),
        ((3.5,), torch.float32, "cpu"),
        ((3,), "float32", "cpu"),
        ((3,), torch.float32, "not_a_device"),
    ],
)
def test_custom_invalid_metadata_rejected_without_consuming_rng(schema):
    class Invalid(Triangular):
        def sample_draw_schema(self, sample_shape=torch.Size()):
            return schema

    source = Invalid()
    generator = torch.Generator().manual_seed(13)
    before = generator.get_state()
    with pytest.raises(ValueError, match="schema|floating dtype"):
        source.sample_with_draw((3,), generator=generator)
    assert torch.equal(generator.get_state(), before)
    with pytest.raises(ValueError, match="schema|floating dtype"):
        connection_distribution_draw_schema(source, (3,))


@pytest.mark.parametrize("kind", ["shape", "dtype", "object", "placeholder"])
def test_custom_draws_and_placeholders_must_match_metadata(kind):
    class Invalid(Triangular):
        def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
            if kind == "object":
                return [0.5]
            if kind == "shape":
                return torch.zeros((3,), dtype=self.location.dtype)
            return torch.zeros((3, 2), dtype=torch.float32)

        def sample_draw_placeholder(self, sample_shape=torch.Size()):
            return torch.zeros((3, 2), dtype=torch.float32)

    source = Invalid()
    with pytest.raises(ValueError, match="shape, dtype, and device schema"):
        if kind == "placeholder":
            connection_distribution_placeholder(source, (3,))
        else:
            source.sample_with_draw((3,))


def test_custom_transform_requires_tensor_output():
    class Invalid(Triangular):
        def sample_from_draw(self, draw):
            return 0.5

    with pytest.raises(TypeError, match="must return a Tensor"):
        Invalid().sample(3)
