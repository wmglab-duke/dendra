"""Regression tests for mechanism current and declaration correctness."""

from __future__ import annotations

import pytest
import torch

from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms import _symbolic as symbolic
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._material_process import (
    ClampProcess,
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
)
from dendra.models.mechanisms.compilers.ast import factorize_linear_in_v
from dendra.models.mechanisms.ode import differentiate_rhs_2torch_checked
from dendra.models.mod import expsyn, hh, pas


class _LinearThenExplicit(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("linear", "nonlinear")
    Mechanism.EXPLICIT("nonlinear")

    def linear(self, v):
        return v

    def nonlinear(self, v):
        return v**2


class _ExplicitThenLinear(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("nonlinear", "linear")
    Mechanism.EXPLICIT("nonlinear")

    def nonlinear(self, v):
        return v**2

    def linear(self, v):
        return v


class _QuadraticNumerical(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return v**2


class _ExplicitAnalyticPair(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return 3.0 * (v + 2.0)

    def i_with_conductance(self, v):
        conductance = torch.full_like(v, 3.0)
        return self.i(v), conductance


class _SavedAnalyticPair(_ExplicitAnalyticPair):
    Mechanism.SAVE("i")


class _ReassignedLocalCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        conductance = 2.0
        current = conductance * v
        conductance = 3.0
        return current


class _InheritedPas(pas):
    pass


class _DocumentedCurrent(Mechanism):
    Mechanism.RANGE(g=2.0, e=-5.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        """A documented current remains eligible for symbolic factorization."""
        driving_force = v - self.e
        return self.g * driving_force


SYMBOLIC_TEST_SCALE = 2.0


class _GlobalConstantCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return SYMBOLIC_TEST_SCALE * v


class _ExplicitAffineWithConflictingDerivativeDeclarations(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.EXPLICIT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return 4.0 * (v + 3.0)

    def i_with_conductance(self, v):
        return self.i(v), torch.full_like(v, 4.0)


def _double_current(function):
    def wrapped(self, v):
        return 2.0 * function(self, v)

    return wrapped


class _DecoratedCurrent(Mechanism):
    Mechanism.RANGE(g=0.5, e=-10.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    @_double_current
    def i(self, v):
        return self.g * (v - self.e)


class _VoltageIndependentCurrent(Mechanism):
    Mechanism.RANGE(offset=7.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.offset


class _AlgebraicallyZeroConductance(Mechanism):
    Mechanism.RANGE(g=2.0, offset=7.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * v - self.g * v + self.offset


class _ReassignedVoltageCurrent(Mechanism):
    Mechanism.RANGE(g=2.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        original_voltage = v
        v = 3.0
        return self.g * original_voltage + 0.0 * v


def _mechanism(cls, *, dtype=torch.float64):
    shape = (1, 3)
    return cls(
        cls.__name__,
        torch.tensor(34.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
        shape,
        shape,
    )


@pytest.mark.parametrize(
    "cls,current_order",
    [
        (_LinearThenExplicit, ["linear", "nonlinear"]),
        (_ExplicitThenLinear, ["nonlinear", "linear"]),
    ],
)
def test_dufort_frankel_factorability_is_per_current_and_order_independent(
    cls, current_order
):
    mechanism = _mechanism(cls)
    shape = (1, 3)
    celsius = torch.full(shape, 34.0, dtype=torch.float64)
    area = torch.ones(shape, dtype=torch.float64)
    handler = MechanismHandler(
        celsius,
        area,
        {"mixed": mechanism},
        currents={"nonspecific": {"mixed": current_order}},
    )
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=torch.float64))

    voltage = torch.tensor([[4.0, -3.0, 2.0]], dtype=torch.float64)
    previous_voltage = torch.tensor([[2.0, 6.0, -4.0]], dtype=torch.float64)
    current, conductance = handler.idf(voltage, previous_voltage)

    half_previous = 0.5 * previous_voltage
    torch.testing.assert_close(current, half_previous + voltage**2)
    torch.testing.assert_close(conductance, torch.ones_like(voltage))
    assert mechanism._current_factorable == {"linear": True, "nonlinear": False}
    assert mechanism.factorable is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numerical_current_derivative_is_scale_and_dtype_aware(dtype):
    mechanism = _mechanism(_QuadraticNumerical, dtype=dtype)
    assert mechanism._current_conductance_mode == {"i": "numerical-declared"}
    assert mechanism._current_conductance_fallback_reason == {"i": None}
    voltage = torch.tensor([-1.0e4, -70.0, 0.0], dtype=dtype)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, voltage**2)
    tolerance = 4e-5 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(
        conductance,
        2.0 * voltage,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_numerical_current_derivative_rejects_unsupported_low_precision(dtype):
    mechanism = _mechanism(_QuadraticNumerical, dtype=dtype)
    with pytest.raises(
        TypeError, match="supports only torch.float32 and torch.float64"
    ):
        mechanism.i_with_g(torch.tensor([-70.0], dtype=dtype))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_explicit_analytic_current_pair_is_bound_once_and_supports_all_dtypes(dtype):
    mechanism = _mechanism(_ExplicitAnalyticPair, dtype=dtype)
    assert mechanism._current_conductance_mode == {"i": "analytic"}
    assert mechanism.i_with_g.__func__._dendra_conductance_mode == "analytic"
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=dtype)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, 3.0 * (voltage + 2.0))
    torch.testing.assert_close(conductance, torch.full_like(voltage, 3.0))


def test_saved_analytic_current_pair_updates_current_mirror():
    mechanism = _mechanism(_SavedAnalyticPair)
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=torch.float64)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(mechanism.i_, current)
    torch.testing.assert_close(conductance, torch.full_like(voltage, 3.0))


def test_symbolic_factorization_honors_statement_time_values_on_reassignment():
    mechanism = _mechanism(_ReassignedLocalCurrent)
    assert mechanism._current_conductance_mode == {"i": "symbolic"}
    assert mechanism._current_conductance_fallback_reason == {"i": None}
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=torch.float64)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, 2.0 * voltage)
    assert conductance == 2.0


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_inherited_current_is_symbolically_resolved_at_supported_precisions(dtype):
    mechanism = _mechanism(_InheritedPas, dtype=dtype)
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=dtype)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, mechanism.g * (voltage - mechanism.e))
    torch.testing.assert_close(conductance, mechanism.g.to(conductance))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_current_method_docstring_does_not_force_numerical_fallback(dtype):
    mechanism = _mechanism(_DocumentedCurrent, dtype=dtype)
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=dtype)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, mechanism.g * (voltage - mechanism.e))
    torch.testing.assert_close(conductance, mechanism.g)


def test_unresolved_global_symbol_is_rejected_before_code_generation():
    with pytest.raises(ValueError, match="Unresolved bare symbol.*SYMBOLIC_TEST_SCALE"):
        factorize_linear_in_v(_GlobalConstantCurrent)

    mechanism = _mechanism(_GlobalConstantCurrent)
    assert mechanism._current_conductance_mode == {"i": "numerical-fallback"}
    assert (
        "Unresolved bare symbol"
        in (mechanism._current_conductance_fallback_reason["i"])
    )
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=torch.float64)
    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, SYMBOLIC_TEST_SCALE * voltage)
    torch.testing.assert_close(
        conductance,
        torch.full_like(voltage, SYMBOLIC_TEST_SCALE),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_explicit_declaration_overrides_all_conductance_inference_paths(dtype):
    mechanism = _mechanism(
        _ExplicitAffineWithConflictingDerivativeDeclarations, dtype=dtype
    )
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=dtype)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, mechanism.i(voltage))
    assert conductance == 0.0
    assert mechanism._current_conductance_mode == {"i": "explicit"}
    assert mechanism._current_conductance_fallback_reason == {"i": None}
    assert mechanism._current_factorable == {"i": False}
    assert mechanism.factorable is False


def test_decorated_current_is_not_misclassified_as_its_undecorated_body():
    mechanism = _mechanism(_DecoratedCurrent)
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=torch.float64)

    current, conductance = mechanism.i_with_g(voltage)
    probe = voltage.detach().clone().requires_grad_(True)
    expected_conductance = torch.autograd.grad(mechanism.i(probe).sum(), probe)[0]

    torch.testing.assert_close(current, mechanism.i(voltage))
    torch.testing.assert_close(conductance, expected_conductance.expand_as(conductance))
    assert mechanism._current_conductance_mode == {"i": "numerical-fallback"}
    assert (
        "Decorated current methods"
        in (mechanism._current_conductance_fallback_reason["i"])
    )


@pytest.mark.parametrize(
    "mechanism_cls", [_VoltageIndependentCurrent, _AlgebraicallyZeroConductance]
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_exact_zero_conductance_currents_remain_symbolic_at_all_dtypes(
    mechanism_cls, dtype
):
    mechanism = _mechanism(mechanism_cls, dtype=dtype)
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=dtype)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, mechanism.i(voltage))
    assert conductance == 0
    assert mechanism._current_conductance_mode == {"i": "symbolic"}
    assert mechanism._current_conductance_fallback_reason == {"i": None}


def test_voltage_parameter_reassignment_uses_safe_numerical_fallback():
    with pytest.raises(ValueError, match="reserved name 'v'"):
        factorize_linear_in_v(_ReassignedVoltageCurrent)

    mechanism = _mechanism(_ReassignedVoltageCurrent)
    voltage = torch.tensor([-70.0, -2.0, 5.0], dtype=torch.float64)

    current, conductance = mechanism.i_with_g(voltage)

    torch.testing.assert_close(current, mechanism.i(voltage))
    torch.testing.assert_close(conductance, mechanism.g.to(conductance))
    assert mechanism._current_conductance_mode == {"i": "numerical-fallback"}


def test_unexpected_symbolic_compiler_failures_are_not_silently_downgraded(
    monkeypatch,
):
    def internal_bug(*args, **kwargs):
        raise AssertionError("compiler invariant failed")

    monkeypatch.setattr(symbolic, "linear_conductance_in_v", internal_bug)

    with pytest.raises(AssertionError, match="compiler invariant failed"):
        _mechanism(_DocumentedCurrent)


@pytest.mark.parametrize("scheme", ["central", "forward", "backward"])
def test_finite_difference_fallback_matches_analytic_oracle_for_every_scheme(scheme):
    expression, ok, depends = differentiate_rhs_2torch_checked(
        "x' = custom(x)",
        ["x"],
        "x",
        state_vars=["x"],
        extra_user_functions={"custom": "custom"},
        fd_scheme=scheme,
        fd_eps=1e-6,
    )

    assert ok is True
    assert depends is True
    x = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64)
    actual = eval(expression, {"torch": torch, "custom": torch.sin}, {"x": x})
    torch.testing.assert_close(actual, torch.cos(x), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "fd_eps", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_finite_difference_rejects_nonpositive_or_nonfinite_epsilon(fd_eps):
    with pytest.raises(ValueError, match="fd_eps must be positive and finite"):
        differentiate_rhs_2torch_checked("x' = x**2", ["x"], "x", fd_eps=fd_eps)


def test_finite_difference_rejects_boolean_epsilon_and_unknown_scheme():
    with pytest.raises(TypeError, match="positive, finite real"):
        differentiate_rhs_2torch_checked("x' = x**2", ["x"], "x", fd_eps=True)
    with pytest.raises(ValueError, match="Unsupported fd_scheme"):
        differentiate_rhs_2torch_checked("x' = x**2", ["x"], "x", fd_scheme="mystery")


def test_aborted_mechanism_and_state_declarations_do_not_leak():
    with pytest.raises(RuntimeError, match="abort mechanism"):

        class _AbortedMechanism(Mechanism):
            Mechanism.PARAMETER(flat_ghost=1.0)
            Mechanism.RANGE(range_ghost=2.0)
            Mechanism.BUFFER("buffer_ghost")
            Mechanism.NONSPECIFIC_CURRENT("current_ghost")
            raise RuntimeError("abort mechanism")

    class _CleanMechanism(Mechanism):
        pass

    assert "flat_ghost" not in _CleanMechanism._params
    assert "range_ghost" not in _CleanMechanism._range
    assert "buffer_ghost" not in _CleanMechanism._assigned
    assert "current_ghost" not in _CleanMechanism._currents.get("nonspecific", ())

    with pytest.raises(RuntimeError, match="abort state"):

        class _AbortedState(State):
            State.RANGE(rate_ghost=3.0)
            State.STATE("state_ghost")
            State.DERIVATIVE("state_ghost' = -rate_ghost * state_ghost")
            State.METHOD("derivimplicit")
            raise RuntimeError("abort state")

    class _CleanState(State):
        pass

    assert "rate_ghost" not in _CleanState._range
    assert "state_ghost" not in _CleanState._state
    assert not _CleanState._derivative
    assert _CleanState.method == "cnexp"


def test_class_body_declarations_override_legacy_preclass_queue_order():
    Mechanism.RANGE(precedence_probe=1.0)

    class _ClassBodyWins(Mechanism):
        Mechanism.RANGE(precedence_probe=2.0)

    assert _ClassBodyWins._range["precedence_probe"] == 2.0

    State.METHOD("derivimplicit")

    class _StateClassBodyWins(State):
        State.METHOD("cnexp")

    assert _StateClassBodyWins.method == "cnexp"


@pytest.mark.parametrize(
    "mechanism,alias_name,expected_currents",
    [
        (expsyn, "rename_current_oracle_expsyn", ["i"]),
        (hh, "rename_current_oracle_hh", ["il", "ina", "ik"]),
    ],
)
def test_rename_preserves_independent_nonduplicated_current_declarations(
    mechanism, alias_name, expected_currents
):
    assert mechanism._currents == {"nonspecific": expected_currents}

    renamed = mechanism.rename(alias_name)

    assert mechanism._currents == {"nonspecific": expected_currents}
    assert renamed._currents == {"nonspecific": expected_currents}
    assert renamed._currents["nonspecific"] is not mechanism._currents["nonspecific"]


def test_rename_alias_cache_is_scoped_to_source_mechanism():
    alias_name = "shared_rename_cache_oracle"

    renamed_expsyn = expsyn.rename(alias_name)
    renamed_hh = hh.rename(alias_name)

    assert renamed_expsyn is expsyn.rename(alias_name)
    assert renamed_hh is hh.rename(alias_name)
    assert renamed_expsyn is not renamed_hh
    assert renamed_expsyn.i is expsyn.i
    assert renamed_hh.il is hh.il


def test_rename_preserves_declaration_ownership_for_future_subclass_precedence():
    class GlobalOwner(Mechanism):
        Mechanism.GLOBAL(precedence_probe=1.0)

    class RangeOwner(Mechanism):
        Mechanism.RANGE(precedence_probe=2.0)

    renamed = GlobalOwner.rename("renamed_global_owner")

    ownership_names = tuple(
        name for name in GlobalOwner.__dict__ if name.endswith("_defined_here")
    )
    assert ownership_names
    for name in ownership_names:
        source_registry = getattr(GlobalOwner, name)
        alias_registry = getattr(renamed, name)
        assert alias_registry == source_registry
        assert alias_registry is not source_registry

    class SourceComposition(GlobalOwner, RangeOwner):
        pass

    class AliasComposition(renamed, RangeOwner):
        pass

    assert SourceComposition._global == {"precedence_probe": 1.0}
    assert SourceComposition._range == {}
    assert AliasComposition._global == SourceComposition._global
    assert AliasComposition._range == SourceComposition._range


def test_aborted_material_process_declarations_do_not_leak():
    with pytest.raises(RuntimeError, match="abort clearance"):

        class _AbortedClearance(ClearanceProcess):
            ClearanceProcess.METHOD("explicit")
            ClearanceProcess.PHASE("transport")
            ClearanceProcess.CLEAR("ghost.c", rate=1.0)
            raise RuntimeError("abort clearance")

    class _CleanClearance(ClearanceProcess):
        pass

    assert _CleanClearance._clearance_specs == ()
    assert _CleanClearance._material_process_method == "none"
    assert _CleanClearance._material_process_phase == "post_local"

    with pytest.raises(RuntimeError, match="abort clamp"):

        class _AbortedClamp(ClampProcess):
            ClampProcess.CLAMP("ghost.c", value=1.0)
            raise RuntimeError("abort clamp")

    class _CleanClamp(ClampProcess):
        pass

    assert _CleanClamp._clamp_specs == ()

    with pytest.raises(RuntimeError, match="abort exchange"):

        class _AbortedExchange(ExchangeProcess):
            ExchangeProcess.EXCHANGE("ghost_a.c", "ghost_b.c", rate=1.0)
            raise RuntimeError("abort exchange")

    class _CleanExchange(ExchangeProcess):
        pass

    assert _CleanExchange._exchange_specs == ()

    with pytest.raises(RuntimeError, match="abort diffusion"):

        class _AbortedDiffusion(DiffusionProcess):
            DiffusionProcess.DIFFUSE("ghost.c", D=1.0)
            raise RuntimeError("abort diffusion")

    class _CleanDiffusion(DiffusionProcess):
        pass

    assert _CleanDiffusion._diffusion_specs == ()
