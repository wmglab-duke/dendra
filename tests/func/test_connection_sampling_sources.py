"""Raw connection distributions share native draws and pure materialization."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.func._connection_parameters import ConnectionParameterPlan
from dendra.models.distributions import Normal, TruncatedNormal
from dendra.models.networks.net import make_weight

DTYPE = torch.float64
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _context():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=DTYPE, CHECK_PURITY=False):
        yield


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.2, dtype=DTYPE))

    def forward(self, value):
        return value * self.factor


class _CustomDistribution(torch.nn.Module):
    def sample(self, n):
        return torch.rand(n, dtype=DTYPE)


def _distribution(kind="normal"):
    if kind == "normal":
        return Normal(mean=0.7, std=0.2).to(dtype=DTYPE)
    return TruncatedNormal(mean=0.7, std=0.2, low=0.2, high=1.3).to(dtype=DTYPE)


def _connection(kind="normal", *, shared=True, mixed=True, parametrized=False):
    distribution = _distribution(kind)
    if parametrized:
        register_parametrization(distribution, "mean", _Scale())
    weights, counts = [distribution], [3]
    if mixed:
        weights.append(torch.nn.Parameter(torch.tensor(0.4, dtype=DTYPE)))
        counts.append(2)
    weight = make_weight(weights, counts, dtype=DTYPE).init()
    delay = make_weight(
        [distribution if shared else _distribution(kind)],
        [sum(counts)],
        dtype=DTYPE,
    ).init()
    return SimpleNamespace(weight=weight, delay_ms=delay)


@pytest.mark.parametrize("kind", ["normal", "truncated"])
@pytest.mark.parametrize("parametrized", [False, True])
def test_snapshot_preserves_raw_sources_aliases_exact_values_and_native_gradients(
    kind,
    parametrized,
):
    connection = _connection(kind, parametrized=parametrized)
    plan = ConnectionParameterPlan(connection)
    native_sources = plan.parameter_sources()
    assert "weight.w" not in native_sources
    assert not any(key.startswith("delay_ms.") for key in native_sources)
    expected_gradients = torch.autograd.grad(
        connection.weight.w.square().sum() + connection.delay_ms.w.square().sum(),
        tuple(native_sources.values()),
    )
    parameters, constants = plan.extract()
    assert constants["draw_valid.weights"]
    assert constants["draw_valid.delays"]
    assert constants["draw.weights.0"].shape == (3,)
    assert constants["draw.delays.0"].shape == (5,)
    values = plan.materialize(parameters, constants)
    torch.testing.assert_close(values["weights"], connection.weight.w, rtol=0, atol=0)
    torch.testing.assert_close(values["delays"], connection.delay_ms.w, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(
        values["weights"].square().sum() + values["delays"].square().sum(),
        tuple(parameters.values()),
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["normal", "truncated"])
def test_explicit_sampling_matches_native_rng_and_reuses_retained_draws(kind):
    connection = _connection(kind)
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    assert plan.sampling_devices == (torch.device("cpu"),)
    assert plan.has_sampling
    assert plan.requires_initialization_draw()
    assert not plan.requires_initialization_draw(
        reinit_weights=False, reinit_delays=False
    )
    seed_state = torch.random.get_rng_state().clone()
    generator = torch.Generator().set_state(seed_state)
    draws = plan.sample_initialization_draw(
        parameters, constants, generators={"cpu": generator}
    )
    torch.testing.assert_close(torch.random.get_rng_state(), seed_state)
    initialized = plan.initialize_constants(parameters, constants, draws=draws)
    connection.weight.init()
    connection.delay_ms.init()
    torch.testing.assert_close(generator.get_state(), torch.random.get_rng_state())
    actual = plan.materialize(parameters, initialized)
    torch.testing.assert_close(actual["weights"], connection.weight.w, rtol=0, atol=0)
    torch.testing.assert_close(actual["delays"], connection.delay_ms.w, rtol=0, atol=0)
    # Native refresh changes only effective storage and provenance, so an
    # existing plan remains structurally valid after a fresh sample.
    plan.validate_source()
    before = generator.get_state().clone()
    retained_draws = plan.sample_initialization_draw(
        parameters,
        initialized,
        reinit_weights=False,
        reinit_delays=False,
        generators={"cpu": generator},
    )
    assert retained_draws == {"weights": {}, "delays": {}}
    torch.testing.assert_close(before, generator.get_state())
    retained = plan.initialize_constants(
        parameters,
        initialized,
        reinit_weights=False,
        reinit_delays=False,
    )
    changed = {key: value + 0.1 for key, value in parameters.items()}
    values = plan.materialize(changed, retained)
    torch.testing.assert_close(values["weights"], actual["weights"], rtol=0, atol=0)
    torch.testing.assert_close(values["delays"], actual["delays"], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["normal", "truncated"])
def test_legacy_missing_provenance_preserves_storage_until_explicit_refresh(kind):
    connection = _connection(kind)
    connection.weight._sampling_draws = None
    connection.delay_ms._sampling_draws = None
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    assert not constants["draw_valid.weights"]
    assert not constants["draw_valid.delays"]
    values = plan.materialize(parameters, constants)
    torch.testing.assert_close(values["weights"], connection.weight.w, rtol=0, atol=0)
    gradients = torch.autograd.grad(
        values["weights"].sum() + values["delays"].sum(), tuple(parameters.values())
    )
    assert all(torch.count_nonzero(value) == 0 for value in gradients)
    with pytest.raises(dn.func.FunctionalizationError, match="explicit base draws"):
        plan.initialize_constants(parameters, constants)
    draws = plan.sample_initialization_draw(parameters, constants)
    refreshed = plan.initialize_constants(parameters, constants, draws=draws)
    assert refreshed["draw_valid.weights"]
    values = plan.materialize(parameters, refreshed)
    gradients = torch.autograd.grad(
        values["weights"].sum() + values["delays"].sum(), tuple(parameters.values())
    )
    assert any(torch.count_nonzero(value) > 0 for value in gradients)


@pytest.mark.parametrize("kind", ["normal", "truncated"])
def test_pure_materialization_compiles_vmapps_and_preserves_forward_gradients(kind):
    connection = _connection(kind, parametrized=True)
    with dn.ctx(CHECK_PURITY=True):
        plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()

    def function(values):
        result = plan.materialize(values, constants)
        return result["weights"].sum() + result["delays"].sum()

    expected, tangent = torch.func.jvp(
        function,
        (parameters,),
        ({key: torch.ones_like(value) for key, value in parameters.items()},),
    )
    gradients = torch.autograd.grad(function(parameters), tuple(parameters.values()))
    torch.testing.assert_close(tangent, sum(value.sum() for value in gradients))
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(parameters), expected)
    batched = {
        key: torch.stack((value, value + 0.1)) for key, value in parameters.items()
    }
    actual = torch.vmap(function)(batched)
    torch.testing.assert_close(
        actual,
        torch.stack(
            (
                function(parameters),
                function({key: value + 0.1 for key, value in parameters.items()}),
            )
        ),
    )


def test_custom_distributions_keep_initialized_effective_values_and_reject_refresh():
    connection = SimpleNamespace(
        weight=make_weight([_CustomDistribution()], [3], dtype=DTYPE).init(),
        delay_ms=make_weight([torch.tensor(0.1, dtype=DTYPE)], [3], dtype=DTYPE).init(),
    )
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    assert set(parameters) == {"weight.w"}
    assert not plan.has_sampling
    torch.testing.assert_close(
        plan.materialize(parameters, constants)["weights"], connection.weight.w
    )
    with pytest.raises(dn.func.FunctionalizationError, match="custom-distribution"):
        plan.initialize_constants(parameters, constants)
    plan.initialize_constants(parameters, constants, reinit_weights=False)


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype"])
def test_explicit_draw_validation_has_no_rng_side_effect(fault):
    plan = ConnectionParameterPlan(_connection())
    parameters, constants = plan.extract()
    draws = plan.sample_initialization_draw(parameters, constants)
    if fault == "missing":
        draws["weights"] = {}
    elif fault == "shape":
        draws["weights"]["0"] = draws["weights"]["0"][:1]
    else:
        draws["weights"]["0"] = draws["weights"]["0"].float()
    before = torch.random.get_rng_state().clone()
    with pytest.raises(dn.func.FunctionalizationError, match="draw"):
        plan.initialize_constants(parameters, constants, draws=draws)
    torch.testing.assert_close(before, torch.random.get_rng_state())


@pytest.mark.parametrize("kind", ["normal", "truncated"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_source_and_output_dtype_follow_native_sampling(kind, dtype):
    distribution = _distribution(kind).to(dtype=dtype)
    connection = SimpleNamespace(
        weight=make_weight([distribution], [4], dtype=dtype).init(),
        delay_ms=make_weight([torch.tensor(0.1, dtype=dtype)], [4], dtype=dtype).init(),
    )
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    actual = plan.materialize(parameters, constants)["weights"]
    assert actual.dtype is dtype
    assert constants["draw.weights.0"].dtype is (
        dtype if kind == "normal" else torch.get_default_dtype()
    )
    torch.testing.assert_close(actual, connection.weight.w, rtol=0, atol=0)


def test_refresh_one_role_keeps_shared_raw_gradients_and_other_draws():
    plan = ConnectionParameterPlan(_connection(mixed=False))
    parameters, constants = plan.extract()
    mean_key = "weight.weights.0.param.mean"
    scale_key = "weight.weights.0.param.std"
    changed = {**parameters, mean_key: parameters[mean_key] + 0.4}
    generator = torch.Generator().manual_seed(123)
    draws = plan.sample_initialization_draw(
        changed, constants, reinit_delays=False, generators={"cpu": generator}
    )
    assert draws["delays"] == {}
    initialized = plan.initialize_constants(
        changed, constants, reinit_delays=False, draws=draws
    )
    values = plan.materialize(changed, initialized)
    torch.testing.assert_close(
        values["delays"], constants["stored.delays"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        initialized["draw.delays.0"], constants["draw.delays.0"], rtol=0, atol=0
    )
    mean_gradient, scale_gradient = torch.autograd.grad(
        values["weights"].sum() + values["delays"].sum(),
        (parameters[mean_key], parameters[scale_key]),
    )
    torch.testing.assert_close(mean_gradient, torch.tensor(6.0, dtype=DTYPE))
    torch.testing.assert_close(
        scale_gradient, draws["weights"]["0"].sum() + constants["draw.delays.0"].sum()
    )


@pytest.mark.parametrize("kind", ["normal", "truncated"])
def test_one_prepared_stochastic_packet_supports_independent_backward_passes(kind):
    plan = ConnectionParameterPlan(_connection(kind))
    parameters, constants = plan.extract()
    prepared = plan.prepare(parameters, constants)
    baseline = torch.autograd.grad(
        plan.materialize(parameters, constants)["weights"].square().sum(),
        tuple(parameters.values()),
    )
    for multiplier in (1.0, 2.0):
        actual = torch.autograd.grad(
            multiplier
            * plan.validate_prepared(prepared, parameters)["weights"].square().sum(),
            tuple(parameters.values()),
        )
        for value, expected in zip(actual, baseline, strict=True):
            torch.testing.assert_close(value, expected * multiplier)


def test_legacy_unbounded_truncated_distribution_has_finite_zero_raw_gradients():
    distribution = TruncatedNormal(0.7, 0.2, low=-torch.inf, high=torch.inf).to(
        dtype=DTYPE
    )
    connection = SimpleNamespace(
        weight=make_weight([distribution], [3], dtype=DTYPE).init(),
        delay_ms=make_weight([torch.tensor(0.1, dtype=DTYPE)], [3], dtype=DTYPE).init(),
    )
    connection.weight._sampling_draws = None
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    values = plan.materialize(parameters, constants)["weights"]
    torch.testing.assert_close(values, connection.weight.w, rtol=0, atol=0)
    gradients = torch.autograd.grad(values.sum(), tuple(parameters.values()))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize(
    "captured_dtype,fresh_dtype",
    [(torch.float64, torch.float32), (torch.float32, torch.float64)],
)
def test_snapshot_and_fresh_draw_recipes_preserve_changed_default_dtype(
    captured_dtype,
    fresh_dtype,
):
    with dn.ctx(DTYPE=captured_dtype):
        connection = _connection("truncated", parametrized=True)
    with dn.ctx(DTYPE=fresh_dtype):
        plan = ConnectionParameterPlan(connection)
        native_sources = tuple(plan.parameter_sources().values())
        expected_gradients = torch.autograd.grad(
            connection.weight.w.square().sum() + connection.delay_ms.w.square().sum(),
            native_sources,
        )
        parameters, constants = plan.extract()
        assert constants["draw.weights.0"].dtype is captured_dtype
        assert constants["fresh_draw.weights.0"].dtype is fresh_dtype
        actual = plan.materialize(parameters, constants)
        torch.testing.assert_close(
            actual["weights"], connection.weight.w, rtol=0, atol=0
        )
        gradients = torch.autograd.grad(
            actual["weights"].square().sum() + actual["delays"].square().sum(),
            tuple(parameters.values()),
        )
        for value, expected in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(value, expected)
        generator = torch.Generator().set_state(torch.random.get_rng_state())
        draws = plan.sample_initialization_draw(
            parameters, constants, generators={"cpu": generator}
        )
        refreshed = plan.initialize_constants(parameters, constants, draws=draws)
        assert refreshed["use_fresh_draw.weights"]
        connection.weight.init()
        connection.delay_ms.init()
        expected_gradients = torch.autograd.grad(
            connection.weight.w.square().sum() + connection.delay_ms.w.square().sum(),
            native_sources,
        )
        actual = plan.materialize(parameters, refreshed)
        torch.testing.assert_close(
            actual["weights"], connection.weight.w, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual["delays"], connection.delay_ms.w, rtol=0, atol=0
        )
        gradients = torch.autograd.grad(
            actual["weights"].square().sum() + actual["delays"].square().sum(),
            tuple(parameters.values()),
        )
        for value, expected in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(value, expected)
        torch.testing.assert_close(generator.get_state(), torch.random.get_rng_state())
        # Native sampling changes the active dtype lane without changing raw
        # source registration or the lowered packet's fixed tensor schemas.
        extracted_parameters, extracted_constants = plan.extract()
        assert extracted_constants["use_fresh_draw.weights"]
        torch.testing.assert_close(
            plan.materialize(extracted_parameters, extracted_constants)["weights"],
            connection.weight.w,
            rtol=0,
            atol=0,
        )

        def function(values, config):
            return plan.materialize(values, config)["weights"]

        compiled = torch.compile(function, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(parameters, constants), constants["stored.weights"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            compiled(parameters, refreshed), connection.weight.w, rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", ["normal", "truncated"])
def test_explicit_initialization_is_pure_under_compile_jvp_and_vmap(kind):
    plan = ConnectionParameterPlan(_connection(kind, parametrized=True))
    parameters, constants = plan.extract()
    draws = plan.sample_initialization_draw(parameters, constants)

    def function(values):
        config = plan.initialize_constants(values, constants, draws=draws)
        result = plan.materialize(values, config)
        return result["weights"].square().sum() + result["delays"].square().sum()

    before = torch.random.get_rng_state().clone()
    expected, tangent = torch.func.jvp(
        function,
        (parameters,),
        ({key: torch.ones_like(value) for key, value in parameters.items()},),
    )
    gradients = torch.autograd.grad(function(parameters), tuple(parameters.values()))
    torch.testing.assert_close(tangent, sum(value.sum() for value in gradients))
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(parameters), expected)
    batched = {
        key: torch.stack((value, value + 0.1)) for key, value in parameters.items()
    }
    torch.testing.assert_close(
        torch.vmap(function)(batched),
        torch.stack(
            (
                expected,
                function({key: value + 0.1 for key, value in parameters.items()}),
            )
        ),
    )
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
