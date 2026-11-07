"""Raw connection sources share native evaluation and explicit retention."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.func._connection_parameters import ConnectionParameterPlan
from dendra.models.networks.net import make_weight
from dendra.models.parametric import PositiveParam

DTYPE = torch.float64
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _context():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=DTYPE, CHECK_PURITY=False):
        yield


class _Source(torch.nn.Module):
    def __init__(self, value=0.4):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor(value, dtype=DTYPE))
        self.register_buffer("scale", torch.tensor(1.2, dtype=DTYPE))
        self.factor = 1.0

    def forward(self):
        return self.raw * self.scale * self.factor


class _RandomSource(_Source):
    def forward(self):
        return super().forward() + torch.rand_like(self.raw)


class _CountingSource(_Source):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self):
        self.calls += 1
        return super().forward()


class _SquareRootSource(_Source):
    def forward(self):
        return self.raw.sqrt() * self.scale


class _CustomRepeatPositiveParam(PositiveParam):
    def repeat(self, n):
        return self._compute().repeat(n) * 2


class _CustomForwardPositiveParam(PositiveParam):
    def forward(self):
        return self._compute() * 2


class _TransientRepeatPositiveParam(PositiveParam):
    def __init__(self, value):
        super().__init__(value)
        self.calls = 0

    def repeat(self, n):
        previous = self.calls
        self.calls += 1
        value = self._compute().repeat(n)
        self.calls = previous
        return value


class _TransientForwardPositiveParam(PositiveParam):
    def __init__(self, value):
        super().__init__(value)
        self.calls = 0

    def forward(self):
        previous = self.calls
        self.calls += 1
        value = self._compute()
        self.calls = previous
        return value


def _connection(source=None, *, shared=False):
    source = _Source() if source is None else source
    weight = make_weight([source], [2], dtype=DTYPE)
    delay = make_weight(
        [source if shared else PositiveParam(torch.tensor(0.03, dtype=DTYPE))],
        [2],
        dtype=DTYPE,
    )
    weight.init()
    delay.init()
    return SimpleNamespace(weight=weight, delay_ms=delay)


@pytest.mark.parametrize("training", [False, True])
def test_native_pure_derivation_repeats_and_bypasses_parameter_inference_cache(
    training,
):
    bounded = PositiveParam(torch.tensor([0.2, 0.5], dtype=DTYPE), requires_grad=True)
    bounded.train(training)
    bounded()
    with torch.no_grad():
        bounded.rho.add_(0.1)
    weights = make_weight(
        [bounded, torch.nn.Parameter(torch.tensor(0.1, dtype=DTYPE))],
        [2, 3],
        dtype=torch.float32,
    )
    weights.init()
    expected = torch.cat(
        (bounded._compute().repeat(2), bounded.rho.new_tensor(0.1).repeat(3))
    ).float()
    torch.testing.assert_close(weights.w, expected)
    torch.testing.assert_close(weights.derive_values(), expected)
    assert set(weights.state_dict()) == {"w", "weights.0.param.rho", "weights.1.param"}
    previous = weights.w
    with torch.no_grad():
        bounded.rho.add_(0.1)
    weights.init(reinit=False)
    assert weights.w is previous
    weights.init(reinit=True)
    assert not torch.equal(weights.w, previous)
    ordinary = make_weight([torch.tensor(0.2, dtype=DTYPE)], [2], dtype=DTYPE)
    ordinary.init()
    assert set(ordinary.state_dict()) == {"w"}
    ordinary.load_state_dict({"w": ordinary.w.clone()})


def test_prepared_connection_sources_support_two_independent_backward_passes():
    connection = _connection()
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    prepared = plan.prepare(parameters, constants)
    raw = parameters["weight.weights.0.param.raw"]
    assert prepared.values["weights"].grad_fn is None
    for multiplier in (1.0, 2.0):
        values = plan.validate_prepared(prepared, parameters)
        actual = torch.autograd.grad(multiplier * values["weights"].sum(), raw)[0]
        torch.testing.assert_close(actual, raw.new_tensor(2 * 1.2 * multiplier))
    prepared.values["weights"].add_(1.0)
    with pytest.raises(dn.func.FunctionalizationError, match="changed"):
        plan.validate_prepared(prepared, parameters)


@pytest.mark.parametrize(
    "source_type", [_CustomRepeatPositiveParam, _CustomForwardPositiveParam]
)
def test_custom_cacheable_repeat_and_forward_remain_authoritative(source_type):
    source = source_type(torch.tensor(0.3, dtype=DTYPE))
    connection = _connection(source)
    expected = source._compute().expand(2) * 2
    torch.testing.assert_close(connection.weight.w, expected)
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    torch.testing.assert_close(
        plan.materialize(parameters, constants)["weights"], expected
    )


@pytest.mark.parametrize("retain", [False, True])
def test_inactive_connection_source_derivatives_cannot_poison_selected_values(retain):
    connection = _connection(_SquareRootSource())
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    key = "weight.weights.0.param.raw"
    constants = plan.initialize_constants(
        parameters, constants, reinit_weights=not retain, reinit_delays=True
    )
    if retain:
        parameters = {
            **parameters,
            key: torch.tensor(-1.0, dtype=DTYPE, requires_grad=True),
        }
    else:
        constants = {
            **constants,
            f"baseline.weights.parameters.{key}": torch.tensor(-1.0, dtype=DTYPE),
        }
    actual = plan.materialize(parameters, constants)["weights"]
    torch.testing.assert_close(actual, connection.weight.w)
    gradient = torch.autograd.grad(actual.sum(), parameters[key])[0]
    torch.testing.assert_close(gradient, torch.tensor(1.2 / 0.4**0.5, dtype=DTYPE))


def test_compatible_extraction_rebases_stale_native_effective_storage():
    source, native = _connection(), _connection()
    plan = ConnectionParameterPlan(source)
    with torch.no_grad():
        native.weight.weights[0].param.raw.add_(0.3)
    parameters, constants = plan.extract(connection=native)
    torch.testing.assert_close(
        plan.materialize(parameters, constants)["weights"], native.weight.w
    )
    retained = plan.initialize_constants(
        parameters, constants, reinit_weights=False, reinit_delays=False
    )
    torch.testing.assert_close(
        plan.materialize(parameters, retained)["weights"], native.weight.w
    )
    native.weight.init()
    parameters, constants = plan.extract(connection=native)
    torch.testing.assert_close(
        plan.materialize(parameters, constants)["weights"], native.weight.w
    )
    assert all(
        not value.requires_grad
        for key, value in constants.items()
        if key not in plan.constant_sources(native)
    )


@pytest.mark.parametrize("raw_value", [1e10, float("inf")])
def test_snapshot_exactly_anchors_small_stored_values_with_large_sources(raw_value):
    raw = torch.nn.Parameter(torch.tensor(raw_value, dtype=torch.float32))
    weights = make_weight([raw], [1], dtype=torch.float32).init()
    delay = make_weight(
        [torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))],
        [1],
        dtype=torch.float32,
    ).init()
    weights.w = torch.tensor([1.0], dtype=torch.float32)
    plan = ConnectionParameterPlan(SimpleNamespace(weight=weights, delay_ms=delay))
    parameters, constants = plan.extract()
    key = "weight.weights.0.param"
    actual = plan.materialize(parameters, constants)["weights"]
    torch.testing.assert_close(actual, weights.w, rtol=0, atol=0)
    gradient = torch.autograd.grad(actual.sum(), parameters[key])[0]
    torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=0, atol=0)
    if raw_value != float("inf"):
        updated = {**parameters, key: parameters[key] + 1024.0}
        actual = plan.materialize(updated, constants)["weights"]
        torch.testing.assert_close(actual, actual.new_tensor([1025.0]), rtol=0, atol=0)
        gradient = torch.autograd.grad(actual.sum(), parameters[key])[0]
        torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=0, atol=0)
    fresh_parameters = {
        **parameters,
        key: torch.tensor(0.4, dtype=torch.float32, requires_grad=True),
    }
    fresh_constants = plan.initialize_constants(fresh_parameters, constants)
    actual = plan.materialize(fresh_parameters, fresh_constants)["weights"]
    torch.testing.assert_close(actual, actual.new_tensor([0.4]), rtol=0, atol=0)
    gradient = torch.autograd.grad(actual.sum(), fresh_parameters[key])[0]
    torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=0, atol=0)


def test_shared_weight_delay_sources_keep_independent_retention_baselines():
    connection = _connection(shared=True)
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()
    assert len(parameters) == 1
    key = next(iter(parameters))
    changed = {key: parameters[key] + 0.2}
    initialized = plan.initialize_constants(
        changed, constants, reinit_weights=True, reinit_delays=False
    )
    values = plan.materialize(changed, initialized)
    torch.testing.assert_close(values["weights"], changed[key].expand(2) * 1.2)
    torch.testing.assert_close(values["delays"], connection.delay_ms.w)
    retained = plan.initialize_constants(
        changed, initialized, reinit_weights=False, reinit_delays=False
    )
    values_again = plan.materialize({key: changed[key] + 0.3}, retained)
    torch.testing.assert_close(values_again["weights"], values["weights"])
    torch.testing.assert_close(values_again["delays"], connection.delay_ms.w)


def test_default_trust_skips_optional_rng_diagnostics(monkeypatch):
    from dendra.func import _population

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled connection purity diagnostics ran")

    monkeypatch.setattr(_population, "_global_rng_snapshot", unexpected)
    connection = _connection()
    plan = ConnectionParameterPlan(connection)
    plan.materialize(*plan.extract())


@pytest.mark.parametrize("source_type", [_RandomSource, _CountingSource])
def test_connection_purity_diagnostics_reject_implicit_rng_and_python_state(
    source_type,
):
    connection = _connection(source_type())
    random_state = torch.random.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(dn.func.FunctionalizationError):
            ConnectionParameterPlan(connection)
    torch.testing.assert_close(
        torch.random.get_rng_state(), random_state, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "source_type", [_TransientRepeatPositiveParam, _TransientForwardPositiveParam]
)
def test_connection_purity_audits_authored_cacheable_hooks(source_type):
    source = source_type(torch.tensor(0.3, dtype=DTYPE))
    connection = _connection(source)
    assert source.calls == 0
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(dn.func.FunctionalizationError):
            ConnectionParameterPlan(connection)
    assert source.calls == 0


@pytest.mark.parametrize("view", [False, True])
def test_connection_config_cannot_alias_population_runtime_storage(view):
    model = dn.SingleCompartment(N=1, C=2, dtype=DTYPE)
    source = _Source()
    connection = _connection(source)
    source.scale = model.v.reshape(-1) if view else model.v
    connection.pre = model
    with pytest.raises(dn.func.FunctionalizationError, match="alias runtime"):
        ConnectionParameterPlan(connection)


def test_connection_config_cannot_alias_an_unrelated_population_runtime_view():
    third = dn.SingleCompartment(N=1, C=2, dtype=DTYPE)
    source = _Source()
    source.scale = third.v.reshape(-1)
    connection = _connection(source)
    with pytest.raises(dn.func.FunctionalizationError, match="alias runtime"):
        ConnectionParameterPlan(connection)._validate_runtime_ownership(
            connection, owners=(third,)
        )


@pytest.mark.parametrize("change", ["instance", "class"])
def test_warmed_connection_source_guards_reject_config_changes(change, monkeypatch):
    connection = _connection()
    plan = ConnectionParameterPlan(connection)
    parameters, constants = plan.extract()

    def values(parameters, constants):
        return plan.materialize(parameters, constants)["weights"]

    compiled = torch.compile(values, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(parameters, constants), connection.weight.w)
    if change == "instance":
        connection.weight.weights[0].param.factor = 1.5
    else:
        monkeypatch.setattr(_Source, "forward", lambda self: self.raw * self.scale * 2)
    with pytest.raises(dn.func.FunctionalizationError, match="source"):
        values(parameters, constants)
    with pytest.raises(
        (dn.func.FunctionalizationError, RuntimeError, torch._dynamo.exc.Unsupported)
    ):
        compiled(parameters, constants)
