"""Tensor-level portability contracts for symbolic ODE differentiation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sympy as sp
import torch

from dendra.models.mechanisms import ode
from dendra.models.mechanisms.ode import (
    TorchCodePrinter,
    differentiate_rhs_2torch_checked,
)


def _evaluate(expression, values, **globals_):
    return eval(expression, {"torch": torch, **globals_}, values)


@pytest.mark.parametrize("simplify", [True, False])
def test_piecewise_relational_derivative_matches_autograd(simplify):
    expression, ok, depends = differentiate_rhs_2torch_checked(
        "x' = Piecewise((x**3 + a*x, And(x > -1, x < 1)), "
        "(sin(x), Or(x <= -1, x >= 1)))",
        ["x", "a"],
        "x",
        state_vars=["x"],
        simplify=simplify,
    )

    assert ok is True
    assert depends is True
    assert "torch.where" in expression
    x = torch.tensor([-2.0, -0.5, 0.25, 1.5], dtype=torch.float64).requires_grad_()
    a = torch.tensor(0.7, dtype=torch.float64)
    rhs = torch.where(
        (x > -1) & (x < 1),
        x**3 + a * x,
        torch.sin(x),
    )
    autograd_derivative = torch.autograd.grad(rhs.sum(), x)[0]

    symbolic_derivative = _evaluate(expression, {"x": x.detach(), "a": a})
    torch.testing.assert_close(symbolic_derivative, autograd_derivative)


def test_piecewise_without_default_is_nan_outside_its_domain():
    expression, ok, depends = differentiate_rhs_2torch_checked(
        "x' = Piecewise((x**2, x > 0))",
        ["x"],
        "x",
        state_vars=["x"],
    )

    assert ok is True
    assert depends is True
    actual = _evaluate(
        expression,
        {"x": torch.tensor([-1.0, 2.0], dtype=torch.float64)},
    )
    assert torch.isnan(actual[0])
    torch.testing.assert_close(actual[1], torch.tensor(4.0, dtype=torch.float64))


def test_boolean_printer_uses_elementwise_tensor_logic():
    left, right = sp.symbols("left right", boolean=True)
    symbolic = sp.And(
        left,
        sp.Or(right, sp.Not(left, evaluate=False), evaluate=False),
        evaluate=False,
    )
    expression = TorchCodePrinter().doprint(symbolic)
    left_value = torch.tensor([True, True, False, False])
    right_value = torch.tensor([True, False, True, False])

    actual = _evaluate(
        expression,
        {"left": left_value, "right": right_value},
    )
    expected = left_value & (right_value | ~left_value)
    torch.testing.assert_close(actual, expected)


def test_indexed_state_derivative_matches_autograd_and_detects_array_declaration():
    expression, ok, depends = differentiate_rhs_2torch_checked(
        "x[0]' = x[0]**2*x[1] + sin(x[2])*x[0]",
        ["x[3]"],
        "x[0]",
        state_vars=["x[3]"],
    )

    assert ok is True
    # ``x[3]`` is the declaration of a three-element state array, not element 3.
    assert depends is True
    x = torch.tensor([0.4, -1.2, 0.7], dtype=torch.float64).requires_grad_()
    rhs = x[0] ** 2 * x[1] + torch.sin(x[2]) * x[0]
    autograd_derivative = torch.autograd.grad(rhs, x)[0][0]

    symbolic_derivative = _evaluate(expression, {"x": x.detach()})
    torch.testing.assert_close(symbolic_derivative, autograd_derivative)


def test_rhs_independent_of_target_has_exact_state_independent_zero_derivative():
    result = differentiate_rhs_2torch_checked(
        "x' = a*y",
        ["x", "y", "a"],
        "x",
        state_vars=["y"],
    )

    # Dependency describes the derivative, not the undifferentiated RHS.
    assert result == ("0", True, False)


@pytest.mark.parametrize(
    "state_var, expected",
    [("x[0]", True), ("x[1]", True), ("x[2]", True), ("x[7]", False)],
)
def test_indexed_state_dependency_can_be_requested_for_an_exact_member(
    state_var, expected
):
    _, ok, depends = differentiate_rhs_2torch_checked(
        "x[0]' = x[0]**2*x[1] + sin(x[2])*x[0]",
        ["x[3]"],
        "x[0]",
        state_vars=[state_var],
    )

    assert ok is True
    assert depends is expected


def test_allowlisted_dotted_callable_is_emitted_and_numerically_accurate():
    expression, ok, depends = differentiate_rhs_2torch_checked(
        "x' = vtrap(x)",
        ["x"],
        "x",
        state_vars=["x"],
        extra_user_functions={"vtrap": "helpers.vtrap"},
        fd_eps=1e-5,
    )

    assert ok is True
    assert depends is True
    helpers = SimpleNamespace(vtrap=torch.sin)
    x = torch.tensor([-0.8, 0.1, 1.3], dtype=torch.float64)
    actual = _evaluate(expression, {"x": x}, helpers=helpers)
    torch.testing.assert_close(actual, torch.cos(x), rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize(
    "extra_user_functions",
    [
        None,
        {"custom": "math.sin"},
        {"custom": "numpy.sin"},
        {"custom": "sympy.sin"},
    ],
)
def test_unknown_or_non_torch_callables_are_rejected(extra_user_functions):
    expression, ok, depends = differentiate_rhs_2torch_checked(
        "x' = custom(x)",
        ["x"],
        "x",
        state_vars=["x"],
        extra_user_functions=extra_user_functions,
    )

    assert expression == ""
    assert ok is False
    assert depends is False


@pytest.mark.parametrize("scheme", ["central", "forward", "backward"])
def test_forced_symbolic_failure_uses_requested_finite_difference_scheme(
    monkeypatch, scheme
):
    def unsupported_symbolic_derivative(*args, **kwargs):
        raise NotImplementedError("forced symbolic failure")

    monkeypatch.setattr(ode.sp, "diff", unsupported_symbolic_derivative)
    expression, ok, depends = differentiate_rhs_2torch_checked(
        "x' = x**3 + sin(x)",
        ["x"],
        "x",
        state_vars=["x"],
        fd_scheme=scheme,
        fd_eps=1e-5,
    )

    assert ok is True
    assert depends is True
    x = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64)
    actual = _evaluate(expression, {"x": x})
    expected = 3 * x**2 + torch.cos(x)
    tolerance = 3e-5 if scheme != "central" else 2e-9
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "diff_string, vars_, wrt",
    [
        ("not an equation", ["x"], "x"),
        ("x' = x", None, "x"),
        ("x' = x", ["x[not-an-integer]"], "x"),
    ],
)
def test_malformed_symbolic_inputs_fail_closed(diff_string, vars_, wrt):
    assert differentiate_rhs_2torch_checked(diff_string, vars_, wrt) == (
        "",
        False,
        False,
    )
