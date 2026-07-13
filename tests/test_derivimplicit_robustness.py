import importlib
from types import SimpleNamespace

import pytest
import torch

from dendra.models.mechanisms._derivimplicit import (
    build_derivimplicit,
    derivimplicit_step,
)

DTYPE = torch.float64


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


def test_constant_jacobian_fast_paths_share_global_batch_tolerance_semantics():
    state = torch.tensor([[0.5], [2.0]], dtype=DTYPE)
    jacobian = -torch.ones(1, 1, dtype=DTYPE)
    kwargs = {
        "x_t": state,
        "dt_ms": torch.tensor(0.1, dtype=DTYPE),
        "f": lambda x, _t: -x,
        "jac": jacobian,
        "tol": 0.1,
    }

    eager = derivimplicit_step(**kwargs, detach_newton=False)
    while_loop = derivimplicit_step(**kwargs, detach_newton=True)
    expected = state / 1.1

    torch.testing.assert_close(eager, expected)
    torch.testing.assert_close(while_loop, expected)


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
    solve = build_derivimplicit(["x"], set(), ["x' = -rate*x"])
    state = torch.tensor([0.25, 1.5, 3.0], dtype=DTYPE)
    rate = torch.tensor([0.5, 2.0, 4.0], dtype=DTYPE)
    dt = torch.tensor([0.2, 0.1, 0.05], dtype=DTYPE)

    actual = solve(
        SimpleNamespace(rate=rate, training=False),
        dt,
        state,
    )["x"]
    expected = state / (1.0 + dt * rate)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


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
