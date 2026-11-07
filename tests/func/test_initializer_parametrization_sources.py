"""Parametrized initializer recipes retain their native raw-source contract."""

from __future__ import annotations

import pytest
import torch
from torch.func import functional_call
from torch.nn.utils import parametrize

import dendra as dn
from dendra.func._initialization_sources import (
    InitializationSourcePlan,
    _SourceCall,
    configure_initialization_source_execution,
    initialization_source_roots,
)
from dendra.func._population import _deepcopy_execution_module
from dendra.func._torch_parametrizations import install_output_schemas
from dendra.func._types import FunctionalizationError
from dendra.models.mechanisms._materials import _resolve_initial_source
from dendra.models.parametric import PositiveParam
from tests.func.test_initialization_sources_plan import _AffineSource, _population

pytestmark = pytest.mark.cpu


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
        self.register_buffer("offset", torch.tensor(0.125, dtype=torch.float64))
        self.gain = 2.0

    def forward(self, value):
        return value * self.factor * self.gain + self.offset


class _SumOriginals(torch.nn.Module):
    def forward(self, left, right):
        return left + right

    def right_inverse(self, value):
        return value - 0.25, torch.full_like(value, 0.25)


def _source(kind="custom", *, multiple=False):
    source = (
        _AffineSource()
        if kind in ("custom", "buffer")
        else PositiveParam(torch.tensor(1.5, dtype=torch.float64), requires_grad=True)
    )
    field = "rho" if kind == "positive" else "value"
    if kind == "buffer":
        value = source.value.detach()
        del source.value
        source.register_buffer("value", value)
    if multiple:
        parametrize.register_parametrization(source, field, _SumOriginals())
    parametrize.register_parametrization(source, field, _Scale())
    return source, field


@pytest.mark.parametrize("diagnostic", (False, True))
@pytest.mark.parametrize("name", ("v", "t"))
def test_generic_source_fields_do_not_inherit_population_state_restrictions(
    diagnostic, name
):
    class _NamedSource(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                name, torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
            )

        if name == "v":

            def forward(self):
                return self.v
        else:

            def forward(self):
                return self.t

    source = _NamedSource()
    parametrize.register_parametrization(source, name, _Scale())
    population = _population(source)
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    plan.validate_initialization()


@pytest.mark.parametrize("diagnostic", (False, True))
@pytest.mark.parametrize("kind", ("custom", "positive", "buffer"))
@pytest.mark.parametrize("multiple", (False, True))
def test_parametrized_initializer_recipes_admit_raw_chains_and_native_gradients(
    diagnostic, kind, multiple
):
    source, _field = _source(kind, multiple=multiple)
    population = _population(source, ions=True)
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    plan.validate_initialization()
    private = _deepcopy_execution_module(population)
    install_output_schemas(private)
    configure_initialization_source_execution(private)
    plan.validate_population(private)
    with dn.ctx(CHECK_PURITY=diagnostic):
        private_plan = InitializationSourcePlan(private)
    assert not private_plan.failures, private_plan.failures
    private_plan.validate_population(population)
    cloned = initialization_source_roots(private)[0][1]
    call = _SourceCall(cloned)
    call.audit = False
    originals = dict(source.named_parameters())
    explicit = {f"source.{name}": value for name, value in originals.items()}
    explicit.update({f"source.{name}": value for name, value in source.named_buffers()})
    actual = functional_call(call, explicit, (), strict=True, tie_weights=True)
    expected = _resolve_initial_source(source)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), tuple(originals.values()))
    expected_grad = torch.autograd.grad(expected.sum(), tuple(originals.values()))
    for result, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(result, reference)
    assert (
        initialization_source_roots(private)[0][1]
        is initialization_source_roots(private)[1][1]
    )


@pytest.mark.parametrize("diagnostic", (False, True))
def test_parametrized_initializer_original_and_transform_buffer_aliases_survive(
    diagnostic,
):
    source, field = _source()
    transforms = source.parametrizations[field]
    transforms[0]._buffers["offset"] = transforms.original
    population = _population(source, ions=True)
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    private = _deepcopy_execution_module(population)
    install_output_schemas(private)
    configure_initialization_source_execution(private)
    plan.validate_population(private)
    cloned = initialization_source_roots(private)[0][1]
    assert (
        cloned.parametrizations[field][0].offset
        is cloned.parametrizations[field].original
    )


@pytest.mark.parametrize("diagnostic", (False, True))
def test_parametrized_initializer_getters_ignore_global_torch_cache(diagnostic):
    source, field = _source()
    population = _population(source)
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = InitializationSourcePlan(population)
    assert not plan.failures, plan.failures
    private = _deepcopy_execution_module(population)
    install_output_schemas(private)
    configure_initialization_source_execution(private)
    cloned = initialization_source_roots(private)[0][1]
    raw = cloned.parametrizations[field].original
    with parametrize.cached():
        first = _resolve_initial_source(cloned).detach().clone()
        with torch.no_grad():
            raw.add_(1.0)
        second = _resolve_initial_source(cloned)
    assert not torch.equal(first, second)
    plan.validate_source()


@pytest.mark.parametrize("diagnostic", (False, True))
@pytest.mark.parametrize("hook", ("__getstate__", "__deepcopy__"))
def test_parametrization_cannot_hide_authored_copy_or_serialization_hooks(
    diagnostic, hook
):
    class _Authored(_AffineSource):
        pass

    called = []

    def authored(self, *args):
        called.append(args)
        raise AssertionError("authored copy hook was executed")

    setattr(_Authored, hook, authored)
    source = _Authored()
    parametrize.register_parametrization(source, "value", _Scale())
    population = _population(source)
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = InitializationSourcePlan(population)
    assert plan.failures
    assert "deepcopy/serialization" in " ".join(plan.failures)
    plan.validate_source()
    assert not called


@pytest.mark.parametrize("diagnostic", (False, True))
def test_authored_parametrized_getter_is_rejected_without_invocation(diagnostic):
    source, field = _source()
    population = _population(source)
    called = []

    def authored(self):
        called.append(self)
        raise AssertionError("authored getter was executed")

    setattr(type(source), field, property(authored))
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = InitializationSourcePlan(population)
    assert plan.failures
    assert "canonical generated" in " ".join(plan.failures)
    plan.validate_source()
    assert not called


def test_optional_parametrization_audit_rejects_rng_and_restores_implicit_state():
    class _Random(_Scale):
        def forward(self, value):
            return value + torch.rand_like(value)

    source = _AffineSource()
    parametrize.register_parametrization(source, "value", _Random())
    population = _population(source)
    before = torch.random.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=True):
        plan = InitializationSourcePlan(population)
    assert plan.failures
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    with dn.ctx(CHECK_PURITY=False):
        trusted = InitializationSourcePlan(population)
    assert not trusted.failures, trusted.failures
    torch.testing.assert_close(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("diagnostic", (False, True))
@pytest.mark.parametrize(
    "change", ("getter", "forward", "config", "layout", "copy", "dispatch", "hook")
)
def test_warmed_parametrized_initializer_bindings_guard_source_changes(
    diagnostic, change, monkeypatch
):
    from dendra._bootstrap import torch_compiler_warning_context

    source, field = _source()
    with dn.ctx(CHECK_PURITY=diagnostic):
        plan = InitializationSourcePlan(_population(source))
    assert not plan.failures, plan.failures
    expected = plan.source_binding_signature()

    def evaluate(value):
        if plan.source_binding_signature() != expected:
            raise FunctionalizationError("Initial-value source changed; lower again.")
        return value + 1

    with torch_compiler_warning_context():
        compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(torch.tensor(1.0)), torch.tensor(2.0))
        transform = source.parametrizations[field][0]
        if change == "getter":
            old = vars(type(source))[field]
            monkeypatch.setattr(type(source), field, property(lambda self: self.offset))
            assert old is not vars(type(source))[field]
        elif change == "forward":
            monkeypatch.setattr(_Scale, "forward", lambda self, value: value * 3)
        elif change == "config":
            transform.gain = 3.0
        elif change == "layout":
            transform.offset = torch.zeros(2, dtype=torch.float64)
        elif change == "copy":
            monkeypatch.setattr(type(source), "__deepcopy__", lambda self, memo: self)
        elif change == "dispatch":
            monkeypatch.setattr(type(source), "__call__", lambda self: self.offset)
        else:
            transform.register_forward_hook(lambda *_args: None)
        with pytest.raises(
            (FunctionalizationError, torch._dynamo.exc.Unsupported),
            match="changed|[Ee]xception|raise",
        ):
            compiled(torch.tensor(1.0))
