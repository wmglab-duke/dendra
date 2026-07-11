"""Regression tests for mechanism current and declaration correctness."""

from __future__ import annotations

import pytest
import torch

from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._material_process import (
    ClampProcess,
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
)
from dendra.models.mechanisms.ode import differentiate_rhs_2torch_checked


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
