"""Raw source inventories preserve packet coordinates and native tensor ties."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.func._connection_parameters import ConnectionParameterPlan
from dendra.func._netstim_parameters import NetStimParameterPlan
from dendra.models.networks.net import make_weight
from tests.func.test_initialization_execution import _assert_snapshot, _snapshot
from tests.func.test_netstim_parameter_sources import Affine, _source

DTYPE = torch.float64
pytestmark = pytest.mark.cpu
_KINDS = ("connection", "netstim")


class _Recipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.4, dtype=DTYPE))
        self.b = torch.nn.Parameter(torch.tensor(0.7, dtype=DTYPE))
        self.c = self.a
        self.register_buffer("cfg_a", torch.tensor(1.2, dtype=DTYPE))
        self.register_buffer("cfg_b", torch.tensor(0.03, dtype=DTYPE))
        self.register_buffer("cfg_c", self.cfg_a)
        self.register_buffer("cfg_view", self.cfg_a.view(()))
        self.register_buffer("cross", self.a)

    def forward(self):
        return (
            self.a
            + 2 * self.b
            + 3 * self.c
            + 0.1 * self.cfg_a
            + 0.2 * self.cfg_b
            + 0.3 * self.cfg_c
            + 0.4 * self.cfg_view
            + 0.5 * self.cross
        )


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(
        JIT=0, REQUIRE_GRAD=True, DEVICE="cpu", DTYPE=DTYPE, CHECK_PURITY=False
    ):
        yield


def _case(kind):
    recipe = _Recipe()
    if kind == "connection":
        source = SimpleNamespace(
            weight=make_weight([recipe], [2], dtype=DTYPE),
            delay_ms=make_weight([recipe], [2], dtype=DTYPE),
        )
        with torch.no_grad():
            source.weight.init()
            source.delay_ms.init()
        plan = ConnectionParameterPlan(source)
        prefix = "weight.weights.0.param."
    else:
        source = _source()
        source.interval = recipe
        source.initialize()
        plan = NetStimParameterPlan(source)
        prefix = "interval."
    return SimpleNamespace(
        kind=kind, source=source, recipe=recipe, plan=plan, prefix=prefix
    )


def _extract(case, source=None):
    if case.kind == "connection":
        return case.plan.extract(connection=source)
    return case.plan.extract(source=source)


def _native(case, source=None):
    source = case.source if source is None else source
    if case.kind == "connection":
        return {
            "weights": source.weight.derive_values(),
            "delays": source.delay_ms.derive_values(),
        }
    return source.renewal_parameters()


def _assert_values(actual, expected):
    assert tuple(actual) == tuple(expected)
    for key, value in actual.items():
        torch.testing.assert_close(value, expected[key])


def _loss(values):
    return sum(
        (index + 1) * value.square().sum()
        for index, value in enumerate(values.values())
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_interleaved_routes_keep_packet_order_and_identity_distinctions(kind):
    case = _case(kind)
    plan, prefix = case.plan, case.prefix
    parameters, constants = _extract(case)
    raw_names = tuple(
        name.removeprefix(prefix) for name in parameters if name.startswith(prefix)
    )
    config_names = tuple(
        name.removeprefix(prefix)
        for name in plan.constant_sources()
        if name.startswith(prefix)
    )
    if kind == "connection":
        assert raw_names == ("a", "b")
        assert config_names == ("cfg_a", "cfg_b", "cfg_view")
    else:
        assert raw_names == ("a", "b", "c", "cross")
        assert config_names == ("cfg_a", "cfg_b", "cfg_c", "cfg_view")
        assert parameters[prefix + "a"] is parameters[prefix + "c"]
        assert parameters[prefix + "a"] is parameters[prefix + "cross"]
        assert constants[prefix + "cfg_a"] is constants[prefix + "cfg_c"]
    assert prefix + "cross" not in plan.constant_sources()
    assert case.recipe.cfg_a is not case.recipe.cfg_view
    assert torch._C._is_alias_of(case.recipe.cfg_a, case.recipe.cfg_view)
    assert constants[prefix + "cfg_a"] is not constants[prefix + "cfg_view"]
    assert not torch._C._is_alias_of(
        constants[prefix + "cfg_a"], constants[prefix + "cfg_view"]
    )
    torch.testing.assert_close(
        constants[prefix + "cfg_a"], constants[prefix + "cfg_view"]
    )
    supplied = _snapshot(
        (
            parameters,
            constants,
            tuple(case.recipe.parameters()),
            tuple(case.recipe.buffers()),
        )
    )
    _assert_values(plan.materialize(parameters, constants), _native(case))
    _assert_snapshot(
        (
            parameters,
            constants,
            tuple(case.recipe.parameters()),
            tuple(case.recipe.buffers()),
        ),
        supplied,
    )


def test_connection_cross_role_ties_export_one_canonical_parameter_and_config():
    case = _case("connection")
    parameters, constants = _extract(case)
    assert tuple(parameters) == ("weight.weights.0.param.a", "weight.weights.0.param.b")
    assert tuple(case.plan.constant_sources()) == (
        "weight.weights.0.param.cfg_a",
        "weight.weights.0.param.cfg_b",
        "weight.weights.0.param.cfg_view",
    )
    changed = {**parameters, case.prefix + "a": parameters[case.prefix + "a"] + 0.2}
    values = case.plan.materialize(changed, constants)
    for name in ("weights", "delays"):
        torch.testing.assert_close(values[name], _native(case)[name] + 4.5 * 0.2)


@pytest.mark.parametrize("parametrized", (False, True))
def test_netstim_primary_original_aliases_and_synthetic_tau_stay_selectable(
    parametrized,
):
    source = _source()
    if parametrized:
        register_parametrization(source.interval, "rho", Affine())
        original = source.interval.parametrizations.rho.original
        interval_key = "interval.parametrizations.rho.original"
    else:
        original = source.interval.rho
        interval_key = "interval.rho"
    source.start = original
    source.noise = original
    source.initialize()
    plan = NetStimParameterPlan(source)
    parameters, constants = plan.extract()
    assert parameters["start"] is parameters["noise"] is parameters[interval_key]
    assert tuple(parameters)[-1] == "tau"
    assert parameters["tau"].shape == ()
    assert "interval" not in parameters
    if parametrized:
        assert "interval.rho" not in parameters
    _assert_values(plan.materialize(parameters, constants), source.renewal_parameters())
    parameters["tau"].requires_grad_()
    derivative = torch.autograd.grad(
        plan.materialize(parameters, constants)["tau"].sum(), parameters["tau"]
    )[0]
    torch.testing.assert_close(derivative, parameters["tau"].new_tensor(2.0))
    with pytest.raises(dn.func.FunctionalizationError, match="shared tensor aliases"):
        plan.materialize(
            {**parameters, "noise": parameters["noise"].clone()}, constants
        )


@pytest.mark.parametrize("kind", _KINDS)
def test_compatible_sources_keep_order_and_ties_while_using_alternate_values(kind):
    case = _case(kind)
    original = _extract(case)
    alternate = copy.deepcopy(case.source)
    recipe = (
        alternate.weight.weights[0].param
        if kind == "connection"
        else alternate.interval
    )
    with torch.no_grad():
        recipe.a.add_(0.13)
        recipe.cfg_a.add_(0.07)
        if kind == "connection":
            alternate.weight.init()
            alternate.delay_ms.init()
    parameters, constants = _extract(case, alternate)
    assert tuple(parameters) == tuple(original[0])
    assert tuple(constants) == tuple(original[1])
    torch.testing.assert_close(parameters[case.prefix + "a"], recipe.a)
    assert not torch.equal(
        parameters[case.prefix + "a"], original[0][case.prefix + "a"]
    )
    _assert_values(
        case.plan.materialize(parameters, constants), _native(case, alternate)
    )
    recipe.c = torch.nn.Parameter(recipe.a.detach().clone())
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=f"Compatible {kind if kind == 'connection' else 'NetStim'} parameter aliases changed",
    ):
        _extract(case, alternate)


def test_connection_raw_schema_error_keeps_its_parameter_coordinate():
    case = _case("connection")
    parameters, constants = _extract(case)
    key = case.prefix + "a"
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="Connection parameter.*weight.weights.0.param.a.*changed tensor schema",
    ):
        case.plan.materialize(
            {**parameters, key: parameters[key].reshape(1)}, constants
        )


def test_netstim_interleaved_config_alias_error_keeps_existing_message():
    case = _case("netstim")
    parameters, constants = _extract(case)
    key = case.prefix + "cfg_c"
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="Explicit NetStim sources must preserve their shared tensor aliases",
    ):
        case.plan.materialize(parameters, {**constants, key: constants[key].clone()})


@pytest.mark.parametrize("kind", _KINDS)
def test_tied_sources_recompute_fresh_backward_graphs_from_detached_inspection(kind):
    case = _case(kind)
    parameters, constants = _extract(case)
    prepared = case.plan.prepare(parameters, constants)
    assert all(value.grad_fn is None for value in prepared.values.values())
    targets = tuple(parameters[case.prefix + name] for name in ("a", "b"))
    native_targets = (case.recipe.a, case.recipe.b)
    for multiplier in (1.0, 2.0):
        actual = case.plan.validate_prepared(prepared, parameters)
        expected = _native(case)
        _assert_values(actual, expected)
        gradients = torch.autograd.grad(multiplier * _loss(actual), targets)
        native_gradients = torch.autograd.grad(
            multiplier * _loss(expected), native_targets
        )
        for value, native in zip(gradients, native_gradients, strict=True):
            assert torch.isfinite(value).all() and torch.count_nonzero(value)
            torch.testing.assert_close(value, native)
    with torch.no_grad():
        prepared.values[next(iter(prepared.values))].add_(1)
    with pytest.raises(dn.func.FunctionalizationError, match="changed"):
        case.plan.validate_prepared(prepared, parameters)
