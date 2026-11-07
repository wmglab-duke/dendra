"""Custom connection recipes retain raw sources and explicit draw semantics."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.func._connection_parameters import ConnectionParameterPlan
from dendra.models.distributions import Distribution, Normal
from dendra.models.networks.net import make_weight

DTYPE = torch.float64
pytestmark = pytest.mark.cpu


class _ShiftedExponential(Distribution):
    def __init__(self):
        super().__init__()
        self.location = torch.nn.Parameter(torch.tensor(0.2, dtype=DTYPE))
        self.log_rate = torch.nn.Parameter(torch.tensor(0.1, dtype=DTYPE))

    def sample_draw_schema(self, sample_shape=torch.Size()):
        return torch.Size(sample_shape), torch.get_default_dtype(), self.location.device

    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def sample_draw_placeholder(self, sample_shape=torch.Size()):
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.full(shape, 0.5, dtype=dtype, device=device)

    def sample_from_draw(self, draw):
        return self.location - draw.log() / self.log_rate.exp()


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.2, dtype=DTYPE))

    def forward(self, value):
        return value * self.factor


@pytest.fixture(autouse=True)
def _environment():
    rng = torch.get_rng_state().clone()
    try:
        with dn.ctx(
            JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=DTYPE, CHECK_PURITY=False
        ):
            yield
    finally:
        torch.set_rng_state(rng)


def _connection(source=None, *, parametrized=False):
    source = _ShiftedExponential() if source is None else source
    if parametrized:
        register_parametrization(source, "location", _Scale())
    builtin = Normal(0.3, 0.1).to(dtype=DTYPE)
    return SimpleNamespace(
        weight=make_weight([source, builtin], [3, 2], dtype=DTYPE).init(),
        delay_ms=make_weight([source], [5], dtype=DTYPE).init(),
    )


@pytest.mark.parametrize("parametrized", (False, True))
def test_custom_raw_sources_shared_with_delays_match_native_values_and_gradients(
    parametrized,
):
    connection = _connection(parametrized=parametrized)
    with dn.ctx(CHECK_PURITY=True):
        plan = ConnectionParameterPlan(connection)
    assert "weight.w" not in plan.parameter_sources()
    assert not any(name.startswith("delay_ms.") for name in plan.parameter_sources())
    targets = tuple(plan.parameter_sources().values())
    expected_gradients = torch.autograd.grad(
        connection.weight.w.square().sum() + connection.delay_ms.w.square().sum(),
        targets,
    )
    parameters, constants = plan.extract()
    values = plan.materialize(parameters, constants)
    torch.testing.assert_close(values["weights"], connection.weight.w, rtol=0, atol=0)
    torch.testing.assert_close(values["delays"], connection.delay_ms.w, rtol=0, atol=0)
    gradients = torch.autograd.grad(
        values["weights"].square().sum() + values["delays"].square().sum(),
        tuple(parameters.values()),
    )
    torch.testing.assert_close(gradients, expected_gradients)


def test_legacy_missing_draws_use_valid_domain_placeholder_and_zero_raw_gradients():
    connection = _connection()
    connection.weight._sampling_draws = None
    connection.delay_ms._sampling_draws = None
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    assert torch.equal(constants["draw.weights.0"], torch.full((3,), 0.5, dtype=DTYPE))
    before = torch.get_rng_state().clone()
    values = plan.materialize(parameters, constants)
    torch.testing.assert_close(values["weights"], connection.weight.w, rtol=0, atol=0)
    torch.testing.assert_close(values["delays"], connection.delay_ms.w, rtol=0, atol=0)
    gradients = torch.autograd.grad(
        values["weights"].sum() + values["delays"].sum(), tuple(parameters.values())
    )
    for value in gradients:
        assert torch.isfinite(value).all() and torch.count_nonzero(value) == 0
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    with pytest.raises(dn.func.FunctionalizationError, match="explicit base draws"):
        plan.initialize_constants(parameters, constants)
    draws = plan.sample_initialization_draw(parameters, constants)
    fresh = plan.initialize_constants(parameters, constants, draws=draws)
    gradients = torch.autograd.grad(
        plan.materialize(parameters, fresh)["weights"].sum(), tuple(parameters.values())
    )
    assert any(value.abs().sum() > 0 for value in gradients)


@pytest.mark.parametrize(
    "captured,fresh", ((torch.float32, torch.float64), (torch.float64, torch.float32))
)
def test_snapshot_and_fresh_custom_draw_dtypes_preserve_values_gradients_and_rng(
    captured, fresh
):
    with dn.ctx(DTYPE=captured):
        connection = _connection()
    with dn.ctx(DTYPE=fresh):
        plan = ConnectionParameterPlan(connection)
        targets = tuple(plan.parameter_sources().values())
        expected_gradients = torch.autograd.grad(
            connection.weight.w.square().sum(), targets
        )
        parameters, constants = plan.extract()
        assert constants["draw.weights.0"].dtype is captured
        assert constants["fresh_draw.weights.0"].dtype is fresh
        snapshot = plan.materialize(parameters, constants)["weights"]
        torch.testing.assert_close(snapshot, connection.weight.w, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(snapshot.square().sum(), tuple(parameters.values())),
            expected_gradients,
        )
        before = torch.get_rng_state().clone()
        generator = torch.Generator().set_state(before)
        draws = plan.sample_initialization_draw(
            parameters, constants, generators={"cpu": generator}
        )
        initialized = plan.initialize_constants(parameters, constants, draws=draws)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        connection.weight.init()
        connection.delay_ms.init()
        actual = plan.materialize(parameters, initialized)
        torch.testing.assert_close(
            actual["weights"], connection.weight.w, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual["delays"], connection.delay_ms.w, rtol=0, atol=0
        )
        torch.testing.assert_close(
            generator.get_state(), torch.get_rng_state(), rtol=0, atol=0
        )


@pytest.mark.parametrize("parametrized", (False, True))
def test_custom_explicit_initialization_supports_compile_jvp_vmap_and_detached_draws(
    parametrized,
):
    plan = ConnectionParameterPlan(_connection(parametrized=parametrized))
    parameters, constants = plan.extract()
    draws = plan.sample_initialization_draw(parameters, constants)

    def objective(values):
        config = plan.initialize_constants(values, constants, draws=draws)
        result = plan.materialize(values, config)
        return result["weights"].square().sum() + result["delays"].square().sum()

    before = torch.get_rng_state().clone()
    expected, tangent = torch.func.jvp(
        objective,
        (parameters,),
        ({key: torch.ones_like(value) for key, value in parameters.items()},),
    )
    gradients = torch.autograd.grad(objective(parameters), tuple(parameters.values()))
    torch.testing.assert_close(tangent, sum(value.sum() for value in gradients))
    compiled = torch.compile(objective, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(parameters), expected)
    batched = {
        key: torch.stack((value, value + 0.01)) for key, value in parameters.items()
    }
    torch.testing.assert_close(
        torch.vmap(objective)(batched),
        torch.stack(
            (
                expected,
                objective({key: value + 0.01 for key, value in parameters.items()}),
            )
        ),
    )
    live = {
        role: {key: value.requires_grad_() for key, value in mapping.items()}
        for role, mapping in draws.items()
    }
    config = plan.initialize_constants(parameters, constants, draws=live)
    result = plan.materialize(parameters, config)
    draw_gradients = torch.autograd.grad(
        result["weights"].sum() + result["delays"].sum(),
        tuple(value for mapping in live.values() for value in mapping.values()),
        allow_unused=True,
    )
    assert all(value is None for value in draw_gradients)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


class _HiddenNoise(_ShiftedExponential):
    def sample_from_draw(self, draw):
        return self.location - draw.log() / self.log_rate.exp() + torch.rand_like(draw)


class _SuperExponential(_ShiftedExponential):
    def sample_from_draw(self, draw):
        return super().sample_from_draw(draw).square()


class _SuperNormal(Normal):
    sample = Distribution.sample
    rsample = Distribution.rsample
    sample_with_draw = Distribution.sample_with_draw

    def sample_from_draw(self, draw):
        return super().sample_from_draw(draw).exp()


@pytest.mark.parametrize("source_type", (_SuperExponential, _SuperNormal))
def test_optional_diagnostics_audit_pure_inherited_transform_dependencies(source_type):
    connection = _connection(source_type().to(dtype=DTYPE))
    with dn.ctx(CHECK_PURITY=True):
        plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    targets = tuple(plan.parameter_sources().values())
    expected = connection.weight.w.square().sum() + connection.delay_ms.w.sum()
    actual = plan.materialize(parameters, constants)
    objective = actual["weights"].square().sum() + actual["delays"].sum()
    torch.testing.assert_close(objective, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(objective, tuple(parameters.values())),
        torch.autograd.grad(expected, targets),
    )


_HIDDEN_LOCATION = torch.tensor(0.25, dtype=DTYPE)


class _HiddenParentTensor(_ShiftedExponential):
    def sample_from_draw(self, draw):
        return self.location - draw.log() / self.log_rate.exp() + _HIDDEN_LOCATION


class _SuperHiddenTensor(_HiddenParentTensor):
    def sample_from_draw(self, draw):
        return super().sample_from_draw(draw).square()


class _SuperHiddenNoise(_HiddenNoise):
    def sample_from_draw(self, draw):
        return super().sample_from_draw(draw).square()


class _SuperExplicitClass(_ShiftedExponential):
    offset = 0.25

    def sample_from_draw(self, draw):
        return super().sample_from_draw(draw) + __class__.offset


class _SuperEscape(_ShiftedExponential):
    def sample_from_draw(self, draw):
        parent = super()
        return parent.sample_from_draw(draw)


@pytest.mark.parametrize(
    "source_type",
    (_SuperHiddenTensor, _SuperHiddenNoise, _SuperExplicitClass, _SuperEscape),
)
def test_optional_diagnostics_reject_hidden_or_unresolved_inherited_dependencies(
    source_type,
):
    connection = _connection(source_type())
    before = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(
            dn.func.FunctionalizationError, match="purity|hidden|authored class"
        ):
            ConnectionParameterPlan(connection)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


class _DrawMutation(_ShiftedExponential):
    def sample_from_draw(self, draw):
        draw.add_(1)
        return self.location.expand_as(draw)


class _MetadataMutation(_ShiftedExponential):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros((), dtype=DTYPE))

    def sample_draw_schema(self, sample_shape=torch.Size()):
        self.calls.add_(1)
        return torch.Size(sample_shape), torch.get_default_dtype(), self.location.device


@pytest.mark.parametrize(
    "source_type", (_HiddenNoise, _DrawMutation, _MetadataMutation)
)
def test_optional_diagnostics_reject_impure_custom_hooks_without_source_mutation(
    source_type,
):
    connection = _connection(source_type())
    saved = {name: value.clone() for name, value in connection.weight.named_buffers()}
    before = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(dn.func.FunctionalizationError, match="purity|in-place"):
            ConnectionParameterPlan(connection)
    for name, value in connection.weight.named_buffers():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


class _IgnoresGenerator(_ShiftedExponential):
    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.rand(shape, dtype=dtype, device=device)


@pytest.mark.parametrize("role", ("weights", "delays"))
def test_sampler_diagnostics_reject_implicit_rng_with_explicit_generators(role):
    connection = _connection(_IgnoresGenerator() if role == "weights" else None)
    if role == "delays":
        connection.delay_ms = make_weight(
            [_IgnoresGenerator()], [5], dtype=DTYPE
        ).init()
    with dn.ctx(CHECK_PURITY=True):
        plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(32)
    checkpoint = generator.get_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="supplied generator"):
        plan.sample_initialization_draw(
            parameters, constants, generators={"cpu": generator}
        )
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    torch.testing.assert_close(checkpoint, generator.get_state(), rtol=0, atol=0)


def test_purity_diagnostics_are_opt_in_for_custom_transforms():
    # An impure author implementation violates the declared contract, but
    # default lowering does not run the optional diagnostic probe.
    ConnectionParameterPlan(_connection(_HiddenNoise()))


class _SamplerMutation(_ShiftedExponential):
    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        self.log_rate.add_(0.1)
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)


def test_sampler_diagnostics_preserve_parameters_and_rng_after_in_place_write():
    connection = _connection(_SamplerMutation())
    with dn.ctx(CHECK_PURITY=True):
        plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    saved = {name: value.clone() for name, value in parameters.items()}
    before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(99)
    checkpoint = generator.get_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="in-place tensor write"):
        plan.sample_initialization_draw(
            parameters, constants, generators={"cpu": generator}
        )
    torch.testing.assert_close(parameters, saved, rtol=0, atol=0)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    torch.testing.assert_close(checkpoint, generator.get_state(), rtol=0, atol=0)


@pytest.mark.parametrize("fault", ("missing", "shape", "dtype"))
def test_malformed_custom_draws_reject_without_mutating_sources_or_rng(fault):
    plan = ConnectionParameterPlan(_connection())
    parameters, constants = plan.extract()
    draws = plan.sample_initialization_draw(parameters, constants)
    if fault == "missing":
        del draws["weights"]["0"]
    elif fault == "shape":
        draws["weights"]["0"] = draws["weights"]["0"][:1]
    else:
        draws["weights"]["0"] = draws["weights"]["0"].float()
    before = torch.get_rng_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="draw"):
        plan.initialize_constants(parameters, constants, draws=draws)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
