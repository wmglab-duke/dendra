import importlib
from types import SimpleNamespace

import pytest
import torch

from dendra._solver_failures import _SolverFailureCapture
from dendra.func._scan_tree import _capture_tensor_step
from dendra.models.mechanisms._derivimplicit import (
    DerivimplicitStatus,
    build_derivimplicit,
    derivimplicit_step,
)

DTYPE = torch.float64


@pytest.mark.parametrize("options", [{}, {"max_iter": 25}, {"damping": 1.0}])
def test_generated_affine_step_has_backward_euler_gradients_at_equilibrium(options):
    solve = build_derivimplicit(["x"], set(), ["x' = -rate*x + source"], **options)
    state = torch.tensor(0.5, dtype=DTYPE, requires_grad=True)
    rate = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    source = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)

    actual = solve(
        SimpleNamespace(rate=rate, source=source, training=True),
        state.new_tensor(0.1),
        state,
    )["x"]
    gradients = torch.autograd.grad(
        actual, (state, rate, source), materialize_grads=True
    )

    torch.testing.assert_close(actual, state.detach())
    expected = (1.0 / 1.2, -0.05 / 1.2, 0.1 / 1.2)
    torch.testing.assert_close(gradients, tuple(state.new_tensor(g) for g in expected))


def test_generated_affine_step_solves_independent_states_with_batched_inputs():
    solve = build_derivimplicit(
        ["x", "y"],
        {"x_source", "y_source"},
        ["x' = x_source - x_rate*x", "y' = y_source - y_rate*y"],
    )
    x = torch.tensor([[0.25, 1.0, 0.75], [2.0, 0.5, 0.125]], dtype=DTYPE)
    y = torch.tensor([[1.0, 0.5, 3.0], [0.25, 0.75, 2.0]], dtype=DTYPE)
    dt = x.new_tensor([0.1, 0.2, 0.05])
    x_rate = x.new_tensor([[1.0], [2.0]])
    y_rate = y.new_tensor(0.5)
    x_source = x.new_tensor([0.5, -0.1, 1.0])
    y_source = y.new_tensor([[0.25], [-0.5]])

    actual = solve(
        SimpleNamespace(x_rate=x_rate, y_rate=y_rate, training=False),
        dt,
        x,
        y,
        x_source=x_source,
        y_source=y_source,
    )

    torch.testing.assert_close(actual["x"], (x + dt * x_source) / (1 + dt * x_rate))
    torch.testing.assert_close(actual["y"], (y + dt * y_source) / (1 + dt * y_rate))


def test_generated_affine_step_solves_coupled_system_with_zero_diagonal():
    solve = build_derivimplicit(
        ["x", "y", "z"], set(), ["x' = x-z", "y' = y-x", "z' = z-y"]
    )
    x, y, z = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE).unbind()

    actual = solve(SimpleNamespace(training=False), x.new_tensor(1.0), x, y, z)

    torch.testing.assert_close(actual, {"x": y, "y": z, "z": x})


@pytest.mark.parametrize(
    "options", [{}, {"tol": 1e-8}, {"damping": 1.0}, {"allow_inexact": True}]
)
def test_generated_affine_step_rejects_singular_system_even_at_equilibrium(options):
    solve = build_derivimplicit(["x"], set(), ["x' = x-1"], **options)
    state = torch.tensor(1.0, dtype=DTYPE)

    with pytest.raises(RuntimeError, match="(?i)derivimplicit"):
        solve(SimpleNamespace(training=False), state.new_tensor(1.0), state)


def test_generated_affine_step_rejects_undefined_original_equation():
    solve = build_derivimplicit(["x"], set(), ["x' = x/x"])
    state = torch.tensor(0.0, dtype=DTYPE)

    with pytest.raises(RuntimeError, match="(?i)derivimplicit"):
        solve(SimpleNamespace(training=False), state.new_tensor(1.0), state)


def test_generated_affine_step_avoids_cancellation_on_stiff_float32_decay():
    solve = build_derivimplicit(["x"], set(), ["x' = -rate*x"])
    state = torch.tensor(1e8, dtype=torch.float32)
    parameters = SimpleNamespace(rate=state.new_tensor(1e8), training=False)

    actual = solve(parameters, state.new_tensor(1.0), state)["x"]

    torch.testing.assert_close(actual, state.new_tensor(1.0), rtol=0, atol=0)


def test_generated_affine_step_preserves_tiny_float32_decay():
    solve = build_derivimplicit(["x"], set(), ["x' = -rate*x"])
    state = torch.tensor(1.0, dtype=torch.float32)
    parameters = SimpleNamespace(rate=state.new_tensor(5e-8), training=False)

    actual = solve(parameters, state.new_tensor(1.0), state)["x"]

    expected = torch.nextafter(state, state.new_tensor(0.0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_generated_affine_step_has_finite_result_when_float32_numerator_overflows():
    solve = build_derivimplicit(["x"], set(), ["x' = source-x"])
    state = torch.tensor(1.5e38, dtype=torch.float32)
    parameters = SimpleNamespace(source=state.new_tensor(1.35e38), training=False)

    actual = solve(parameters, state.new_tensor(3.0), state)["x"]

    torch.testing.assert_close(actual, state.new_tensor(1.3875e38), rtol=2e-7, atol=0)


def test_generated_affine_step_respects_explicit_newton_tolerance():
    equation = ["x' = source"]
    direct = build_derivimplicit(["x"], set(), equation)
    newton = build_derivimplicit(["x"], set(), equation, tol=1e-3)
    state = torch.tensor(0.5, dtype=DTYPE)
    dt = state.new_tensor(0.1)
    parameters = SimpleNamespace(source=state.new_tensor(1e-8), training=False)

    torch.testing.assert_close(
        direct(parameters, dt, state)["x"], state + 1e-9, rtol=0, atol=0
    )
    torch.testing.assert_close(
        newton(parameters, dt, state)["x"], state, rtol=0, atol=0
    )


def _strategy_kwargs(strategy):
    if strategy == "eager":
        return {"detach_newton": False}
    return {"detach_newton": True, "create_graph": False}


def _python_while_loop(cond_fn, body_fn, carried):
    """Traceable reference for the control flow hidden inside the Torch HOP."""
    while bool(cond_fn(*carried).item()):
        carried = body_fn(*carried)
    return carried


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
@pytest.mark.parametrize("jacobian_source", ["constant", "callable"])
@pytest.mark.parametrize("time_kind", ["scalar", "batched"])
def test_affine_solver_strategies_match_dense_backward_euler(
    strategy, jacobian_source, time_kind
):
    matrix = torch.tensor([[-2.0, 0.5], [1.0, -3.0]], dtype=DTYPE)
    # Keep the RHS operator independent from the tensor returned by the
    # callable Jacobian: torch.while_loop intentionally rejects aliased
    # captured inputs.
    rhs_operator = matrix.mT.clone()
    bias = torch.tensor([0.25, -0.75], dtype=DTYPE)
    forcing = torch.tensor([0.4, -0.1], dtype=DTYPE)
    state = torch.tensor([[1.0, 2.0], [-0.5, 0.75]], dtype=DTYPE)
    dt = torch.tensor([0.1, 0.25], dtype=DTYPE)
    time = (
        torch.tensor(0.5, dtype=DTYPE)
        if time_kind == "scalar"
        else torch.tensor([0.5, 1.25], dtype=DTYPE)
    )
    t_next = time + dt

    kwargs = (
        {"jac": matrix}
        if jacobian_source == "constant"
        else {"jac_fn": lambda _x, _t: matrix}
    )
    actual = derivimplicit_step(
        state,
        dt,
        lambda x, t: x @ rhs_operator + bias + t[..., None] * forcing,
        t_ms=time,
        tol=1e-13,
        **kwargs,
        **_strategy_kwargs(strategy),
    )

    system = torch.eye(2, dtype=DTYPE) - dt[:, None, None] * matrix
    rhs = state + dt[:, None] * (bias + t_next[:, None] * forcing)
    expected = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
@pytest.mark.parametrize("n", [3, 4])
def test_affine_step_solves_invertible_system_with_zero_diagonal(n, strategy):
    state = torch.arange(1, n + 1, dtype=DTYPE)
    identity = torch.eye(n, dtype=DTYPE)
    # At dt=1, the residual system is a cyclic permutation of the identity.
    jacobian = identity - identity.roll(1, dims=0)
    rhs_operator = jacobian.mT.clone()

    actual = derivimplicit_step(
        state,
        torch.tensor(1.0, dtype=DTYPE),
        lambda x, _t: x @ rhs_operator,
        jac=jacobian,
        **_strategy_kwargs(strategy),
    )

    torch.testing.assert_close(actual, state.roll(-1))


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
def test_step_raises_if_backward_euler_has_no_real_root(strategy):
    # The backward-Euler residual x - x**2 - 1 is strictly negative.
    with pytest.raises(RuntimeError, match="(?i)derivimplicit"):
        derivimplicit_step(
            torch.tensor([0.0], dtype=DTYPE),
            torch.tensor(1.0, dtype=DTYPE),
            lambda x, _t: x**2 + 1.0,
            jac_fn=lambda x, _t: (2.0 * x).unsqueeze(-1),
            max_iter=5,
            **_strategy_kwargs(strategy),
        )


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
def test_step_status_isolates_an_unsolvable_batch_member(strategy):
    state = torch.tensor([[2.0], [0.0]], dtype=DTYPE)
    solvable = torch.tensor([[True], [False]])
    actual, status = derivimplicit_step(
        state,
        torch.tensor(1.0, dtype=DTYPE),
        lambda x, _t: torch.where(solvable, -x, x**2 + 1.0),
        jac_fn=lambda x, _t: torch.where(solvable, -1.0, 2.0 * x).unsqueeze(-1),
        max_iter=5,
        return_status=True,
        **_strategy_kwargs(strategy),
    )

    torch.testing.assert_close(actual[0], state.new_tensor([1.0]))
    torch.testing.assert_close(
        status,
        torch.tensor(
            [DerivimplicitStatus.CONVERGED, DerivimplicitStatus.ITERATION_LIMIT]
        ),
    )
    assert actual[1].isfinite().all()


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
def test_singular_backward_euler_system_leaves_other_batch_members_solvable(strategy):
    state = torch.tensor([[2.0], [1.0]], dtype=DTYPE)
    rate = state.new_tensor([[-1.0], [1.0]])
    jacobian = rate.unsqueeze(-1).clone()
    actual, status = derivimplicit_step(
        state,
        state.new_tensor(1.0),
        lambda x, _t: rate * x,
        jac=jacobian,
        return_status=True,
        **_strategy_kwargs(strategy),
    )

    torch.testing.assert_close(actual, state.new_tensor([[1.0], [1.0]]))
    torch.testing.assert_close(
        status,
        torch.tensor(
            [DerivimplicitStatus.CONVERGED, DerivimplicitStatus.LINEAR_SOLVE_FAILED]
        ),
    )


def test_eager_status_backward_rejects_nonfinite_lane():
    state = torch.tensor([[0.0], [1000.0]], dtype=DTYPE, requires_grad=True)
    rate = state.new_tensor(1.0, requires_grad=True)
    dt = state.new_tensor(0.1)

    def solve(incoming):
        return derivimplicit_step(
            incoming,
            dt,
            lambda x, _t: -rate * x.exp(),
            jac_fn=lambda x, _t: (-rate * x.exp()).unsqueeze(-1),
            detach_newton=False,
            return_status=True,
        )

    actual, status = solve(state)
    torch.testing.assert_close(
        status,
        torch.tensor(
            [DerivimplicitStatus.CONVERGED, DerivimplicitStatus.NONFINITE_RESIDUAL]
        ),
    )
    with pytest.raises(
        RuntimeError, match="derivimplicit backward requires finite residuals"
    ):
        torch.autograd.grad(actual[0].sum(), rate)

    retry, retry_status = solve(state[:1])
    assert (retry_status == DerivimplicitStatus.CONVERGED).all()
    torch.testing.assert_close(actual[:1], retry)
    (gradient,) = torch.autograd.grad(retry.sum(), rate)
    expected = (
        -dt * retry.detach().exp() / (1 + dt * rate.detach() * retry.detach().exp())
    )
    torch.testing.assert_close(gradient, expected.sum())


def test_eager_status_backward_rejects_failed_linear_solve():
    state = torch.tensor([[2.0], [1.0]], dtype=DTYPE, requires_grad=True)
    scale = state.new_tensor(1.0, requires_grad=True)
    rate = scale * state.new_tensor([[-1.0], [1.0]])
    actual, status = derivimplicit_step(
        state,
        state.new_tensor(1.0),
        lambda x, _t: rate * x,
        jac=rate.unsqueeze(-1),
        detach_newton=False,
        return_status=True,
    )

    assert status[1] == DerivimplicitStatus.LINEAR_SOLVE_FAILED
    with pytest.raises(
        RuntimeError, match="derivimplicit backward requires finite residuals"
    ):
        torch.autograd.grad(actual[0].sum(), scale)


def test_eager_status_backward_preserves_finite_inexact_gradient():
    state = torch.tensor([1.0], dtype=DTYPE)
    rate = state.new_tensor(2.0, requires_grad=True)
    dt = state.new_tensor(0.1)
    actual, status = derivimplicit_step(
        state,
        dt,
        lambda x, _t: -rate * x.square(),
        jac_fn=lambda x, _t: (-2 * rate * x).unsqueeze(-1),
        max_iter=1,
        detach_newton=False,
        return_status=True,
    )

    assert status == DerivimplicitStatus.ITERATION_LIMIT
    expected = state - dt * rate * state.square() / (1 + 2 * dt * rate * state)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), rate),
        torch.autograd.grad(expected.sum(), rate),
    )


def test_default_tolerance_accepts_float32_roundoff_at_different_scales():
    state = torch.tensor([[1.0], [10_000.0]], dtype=torch.float32)
    dt = state.new_tensor(0.1)
    rate = 0.1
    jacobian = state.new_tensor([[-rate]])
    kwargs = {
        "x_t": state,
        "dt_ms": dt,
        "f": lambda x, _t: -rate * x,
        "jac": jacobian,
        "max_iter": 1,
        "return_status": True,
    }

    actual, default_status = derivimplicit_step(**kwargs)
    _, absolute_status = derivimplicit_step(**kwargs, tol=1e-8)

    torch.testing.assert_close(actual, state / (1.0 + rate * dt))
    assert (default_status == DerivimplicitStatus.CONVERGED).all()
    assert absolute_status[0] == DerivimplicitStatus.CONVERGED
    assert absolute_status[1] != DerivimplicitStatus.CONVERGED


def test_default_tolerance_keeps_representable_float32_drift():
    state = torch.tensor([10_000.0], dtype=torch.float32)

    actual = derivimplicit_step(
        state,
        state.new_tensor(0.1),
        lambda x, _t: torch.full_like(x, 0.02),
        jac=state.new_zeros((1, 1)),
    )

    torch.testing.assert_close(actual, state.new_tensor([10_000.002]), rtol=0, atol=0)
    assert (actual > state).all()


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
@pytest.mark.parametrize("jacobian_source", ["constant", "callable"])
def test_regularized_newton_converges_to_unregularized_equation(
    strategy, jacobian_source
):
    rate = torch.tensor([[0.5], [2.0], [4.0]], dtype=DTYPE)
    source = torch.tensor([[1.0], [-0.5], [2.0]], dtype=DTYPE)
    state = torch.tensor([[0.25], [1.0], [-2.0]], dtype=DTYPE)
    dt = torch.tensor([0.1, 0.2, 0.05], dtype=DTYPE)
    jacobian = -rate.unsqueeze(-1)
    kwargs = (
        {"jac": jacobian}
        if jacobian_source == "constant"
        else {"jac_fn": lambda _x, _t: jacobian}
    )

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, _t: -rate * x + source,
        jacobian_regularization=0.75,
        max_iter=80,
        tol=1e-13,
        **kwargs,
        **_strategy_kwargs(strategy),
    )
    expected = (state + dt[:, None] * source) / (1.0 + dt[:, None] * rate)

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    residual = actual - state - dt[:, None] * (-rate * actual + source)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
def test_damped_affine_newton_iterates_instead_of_returning_partial_step(strategy):
    state = torch.tensor([[1.0], [-2.0]], dtype=DTYPE)
    rate = torch.tensor([[2.0], [0.5]], dtype=DTYPE)
    dt = torch.tensor([0.1, 0.25], dtype=DTYPE)
    jacobian = -rate.unsqueeze(-1)

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, _t: -rate * x,
        jac=jacobian,
        damping=0.5,
        max_iter=60,
        tol=1e-13,
        **_strategy_kwargs(strategy),
    )
    expected = state / (1.0 + dt[:, None] * rate)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
def test_constant_jacobian_honors_zero_iterations_and_initial_tolerance(strategy):
    state = torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    jacobian = -torch.ones(1, 1, dtype=DTYPE)

    zero_iteration = derivimplicit_step(
        state,
        torch.tensor(0.1, dtype=DTYPE),
        lambda x, _t: -x,
        jac=jacobian,
        max_iter=0,
        **_strategy_kwargs(strategy),
    )
    below_tolerance = derivimplicit_step(
        state,
        torch.tensor(0.1, dtype=DTYPE),
        lambda x, _t: -x,
        jac=jacobian,
        tol=1.0,
        **_strategy_kwargs(strategy),
    )

    torch.testing.assert_close(zero_iteration, state)
    torch.testing.assert_close(below_tolerance, state)


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
def test_affine_step_applies_initial_tolerance_independently(strategy):
    state = torch.tensor([[0.5], [2.0]], dtype=DTYPE)
    jacobian = -torch.ones(1, 1, dtype=DTYPE)
    kwargs = {
        "dt_ms": torch.tensor(0.1, dtype=DTYPE),
        "f": lambda x, _t: -x,
        "jac": jacobian,
        "tol": 0.1,
        **_strategy_kwargs(strategy),
    }

    actual = derivimplicit_step(state, **kwargs)
    alone = derivimplicit_step(state[:1], **kwargs)

    expected = state.new_tensor([[0.5], [2.0 / 1.1]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:1], alone)


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
def test_heterogeneous_nonlinear_batch_converges_and_preserves_fixed_point(strategy):
    state = torch.tensor([[0.0], [0.25], [1.5], [3.0]], dtype=DTYPE)
    dt = torch.tensor([0.5, 0.2, 0.1, 0.05], dtype=DTYPE)

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, _t: -(x**2),
        jac_fn=lambda x, _t: (-2.0 * x).unsqueeze(-1),
        max_iter=40,
        tol=1e-13,
        **_strategy_kwargs(strategy),
    )
    expected = (-1.0 + torch.sqrt(1.0 + 4.0 * dt[:, None] * state)) / (
        2.0 * dt[:, None]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert actual[0].item() == 0.0


@pytest.mark.parametrize("strategy", ["eager", "while_loop"])
@pytest.mark.parametrize(
    "approximate_jacobian, expected",
    [
        (2.5, 0.4),  # full step is rejected; half step improves the residual
        (10.0, 2.4),  # neither candidate improves; use the fully decayed step
    ],
)
def test_line_search_acceptance_and_exhaustion_are_deterministic(
    strategy, approximate_jacobian, expected
):
    state = torch.tensor([2.0], dtype=DTYPE)
    actual = derivimplicit_step(
        state,
        torch.tensor(0.2, dtype=DTYPE),
        lambda x, _t: -(x**3),
        jac=torch.tensor([[approximate_jacobian]], dtype=DTYPE),
        line_search=True,
        max_iter=1,
        max_ls_steps=2,
        ls_decay=0.5,
        allow_inexact=True,
        **_strategy_kwargs(strategy),
    )

    torch.testing.assert_close(actual, torch.tensor([expected], dtype=DTYPE))


@pytest.mark.parametrize(
    "approximate_jacobian, expected",
    [(2.5, 0.4), (10.0, 2.4)],
)
def test_while_loop_line_search_control_flow_matches_python_reference(
    monkeypatch, approximate_jacobian, expected
):
    derivimplicit_module = importlib.import_module(
        "dendra.models.mechanisms._derivimplicit"
    )
    monkeypatch.setattr(
        derivimplicit_module, "_get_while_loop", lambda: _python_while_loop
    )

    actual = derivimplicit_module.derivimplicit_step(
        torch.tensor([2.0], dtype=DTYPE),
        torch.tensor(0.2, dtype=DTYPE),
        lambda x, _t: -(x**3),
        jac=torch.tensor([[approximate_jacobian]], dtype=DTYPE),
        line_search=True,
        max_iter=1,
        max_ls_steps=2,
        ls_decay=0.5,
        allow_inexact=True,
    )
    torch.testing.assert_close(actual, torch.tensor([expected], dtype=DTYPE))


def test_while_loop_callable_regularization_control_flow_matches_closed_form(
    monkeypatch,
):
    derivimplicit_module = importlib.import_module(
        "dendra.models.mechanisms._derivimplicit"
    )
    monkeypatch.setattr(
        derivimplicit_module, "_get_while_loop", lambda: _python_while_loop
    )
    state = torch.tensor([[0.0], [0.25], [1.5], [3.0]], dtype=DTYPE)
    dt = torch.tensor([0.5, 0.2, 0.1, 0.05], dtype=DTYPE)

    actual = derivimplicit_module.derivimplicit_step(
        state,
        dt,
        lambda x, _t: -(x**2),
        jac_fn=lambda x, _t: (-2.0 * x).unsqueeze(-1),
        jacobian_regularization=0.25,
        max_iter=60,
        tol=1e-13,
    )
    expected = (-1.0 + torch.sqrt(1.0 + 4.0 * dt[:, None] * state)) / (
        2.0 * dt[:, None]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_batched_autograd_jacobian_respects_batched_time_parameters_and_gradients():
    state = torch.tensor([[0.4], [1.2], [-0.5]], dtype=DTYPE, requires_grad=True)
    rate = torch.tensor([0.5, 1.7, 3.0], dtype=DTYPE, requires_grad=True)
    time = torch.tensor([0.1, 0.5, 1.0], dtype=DTYPE)
    dt = torch.tensor([0.05, 0.15, 0.2], dtype=DTYPE)
    t_next = time + dt

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, t: -rate[:, None] * x + t[:, None],
        t_ms=time,
        create_graph=True,
        detach_newton=False,
        tol=1e-13,
    )
    expected = (state + dt[:, None] * t_next[:, None]) / (
        1.0 + dt[:, None] * rate[:, None]
    )
    expected_gradients = torch.autograd.grad(expected.sum(), (state, rate))

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    actual.sum().backward()
    torch.testing.assert_close(state.grad, expected_gradients[0])
    torch.testing.assert_close(rate.grad, expected_gradients[1])


def test_batched_autograd_jacobian_matches_nonlinear_analytic_solution():
    derivimplicit_module = importlib.import_module(
        "dendra.models.mechanisms._derivimplicit"
    )
    state = torch.tensor([[0.25], [1.5], [3.0]], dtype=DTYPE)
    dt = torch.tensor([0.2, 0.1, 0.05], dtype=DTYPE)

    actual = derivimplicit_module.derivimplicit_step(
        state,
        dt,
        lambda x, _t: -(x**2),
        detach_newton=True,
        tol=1e-13,
    )
    expected = (-1.0 + torch.sqrt(1.0 + 4.0 * dt[:, None] * state)) / (
        2.0 * dt[:, None]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_multistate_batched_autograd_jacobian_matches_dense_systems():
    matrices = torch.tensor(
        [
            [[-2.0, 0.5], [1.0, -3.0]],
            [[-0.5, -0.25], [0.75, -1.5]],
            [[-4.0, 1.0], [-0.5, -2.0]],
        ],
        dtype=DTYPE,
    )
    bias = torch.tensor([[0.25, -0.75], [1.0, 0.5], [-0.5, 2.0]], dtype=DTYPE)
    state = torch.tensor([[1.0, 2.0], [-0.5, 0.75], [2.0, -1.0]], dtype=DTYPE)
    dt = torch.tensor([0.1, 0.25, 0.05], dtype=DTYPE)

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, _t: torch.einsum("bij,bj->bi", matrices, x) + bias,
        detach_newton=True,
        tol=1e-13,
    )
    systems = torch.eye(2, dtype=DTYPE) - dt[:, None, None] * matrices
    rhs = state + dt[:, None] * bias
    expected = torch.linalg.solve(systems, rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_autograd_jacobian_accepts_rhs_constant_in_state():
    state = torch.tensor([[0.25], [1.5], [3.0]], dtype=DTYPE)
    source = torch.tensor([[1.0], [-0.5], [2.0]], dtype=DTYPE)
    dt = torch.tensor([0.2, 0.1, 0.05], dtype=DTYPE)

    actual = derivimplicit_step(
        state,
        dt,
        lambda _x, _t: source,
        detach_newton=True,
        tol=1e-13,
    )
    torch.testing.assert_close(actual, state + dt[:, None] * source)


def test_generated_solver_autograd_fallback_handles_batched_mechanism_parameters(
    monkeypatch,
):
    derivimplicit_module = importlib.import_module(
        "dendra.models.mechanisms._derivimplicit"
    )
    monkeypatch.setattr(
        derivimplicit_module,
        "differentiate_rhs_2torch_checked",
        lambda *_args, **_kwargs: ("0", False, False),
    )
    solve = build_derivimplicit(["x"], set(), ["x' = -rate*x*x"])
    state = torch.tensor([0.25, 1.5, 3.0], dtype=DTYPE)
    rate = torch.tensor([0.5, 2.0, 4.0], dtype=DTYPE)
    dt = torch.tensor([0.2, 0.1, 0.05], dtype=DTYPE)

    actual = solve(
        SimpleNamespace(rate=rate, training=False),
        dt,
        state,
    )["x"]
    expected = 2.0 * state / (1.0 + torch.sqrt(1.0 + 4.0 * dt * rate * state))
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)


def test_nonlinear_callable_jacobian_passes_gradcheck():
    state = torch.tensor([[0.4], [1.2]], dtype=DTYPE, requires_grad=True)
    rate = torch.tensor([0.7, 1.5], dtype=DTYPE, requires_grad=True)
    dt = torch.tensor([0.1, 0.2], dtype=DTYPE)

    def solve(state_arg, rate_arg):
        return derivimplicit_step(
            state_arg,
            dt,
            lambda x, _t: -rate_arg[:, None] * x**2,
            jac_fn=lambda x, _t: (-2.0 * rate_arg[:, None] * x).unsqueeze(-1),
            create_graph=True,
            detach_newton=False,
            tol=1e-13,
        )

    assert torch.autograd.gradcheck(
        solve,
        (state, rate),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


def test_captured_nonlinear_step_has_backward_euler_gradients():
    state = torch.tensor([[0.5], [1.2]], dtype=DTYPE, requires_grad=True)
    rate = torch.tensor([[2.0], [1.5]], dtype=DTYPE, requires_grad=True)
    source = torch.tensor([[0.5], [0.2]], dtype=DTYPE, requires_grad=True)
    dt = torch.tensor([0.125, 0.2], dtype=DTYPE, requires_grad=True)

    def solve(x_old, rate, source, dt):
        solved = derivimplicit_step(
            x_old,
            dt,
            lambda x, _t: source - rate * x.square(),
            jac_fn=lambda x, _t: (-2.0 * rate * x).unsqueeze(-1),
            create_graph=True,
            detach_newton=False,
            tol=1e-13,
        )
        return solved, {}

    inputs = (state, rate, source, dt)
    capture = _SolverFailureCapture()
    step = _capture_tensor_step(capture.wrap(solve), inputs, grad_enabled=True)
    compiled = torch.compile(step, backend="eager", fullgraph=True)
    actual, (_, failure) = compiled(*inputs)
    forcing = state + dt[:, None] * source
    expected = 2 * forcing / (1 + (1 + 4 * dt[:, None] * rate * forcing).sqrt())

    torch.testing.assert_close(failure, state.new_tensor([-1, -1], dtype=torch.int64))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), inputs),
        torch.autograd.grad(expected.sum(), inputs),
        rtol=1e-10,
        atol=1e-12,
    )

    # The first member is still at equilibrium, but its residual Jacobian is
    # zero. A root that cannot be differentiated must report a failed solve.
    _, (_, failure) = compiled(
        state,
        rate.new_tensor([[-1.0], [1.5]], requires_grad=True),
        source.new_tensor([[-0.25], [0.2]], requires_grad=True),
        dt.new_tensor([1.0, 0.2], requires_grad=True),
    )
    torch.testing.assert_close(failure, state.new_tensor([0, 0], dtype=torch.int64))
