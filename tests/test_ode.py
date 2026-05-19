from importlib import import_module

import pytest
import sympy as sp

INTEGRATE = import_module("dendra.models.mechanisms.ode")
integrate2c = INTEGRATE.integrate2c


def _rhs(expr_str):
    """Utility: return RHS sympy expr of ``'lhs = rhs'`` string."""
    return sp.sympify(expr_str.split("=", 1)[1])


def _syms(expr, *names):
    """Return a tuple of symbols with the requested names that already
    appear in `expr` (helper for the tests)."""
    pool = {s.name: s for s in expr.free_symbols}
    return tuple(pool[n] for n in names)


# ---- 1. Constant ODE ---------------------------------------------------------


def test_integrate2c_constant():
    out = integrate2c("x' = c0", "dt", {"c0"})
    rhs = _rhs(out)
    x, dt, c0 = _syms(rhs, "x", "dt", "c0")
    expected = x + c0 * dt
    diff = sp.simplify(rhs - expected)
    # structural equality is brittle; use .equals or numeric subst
    assert diff.equals(0)


# ---- 2. Homogeneous linear ODE x' = a*x --------------------------------------


def test_integrate2c_linear_homogeneous():
    out = integrate2c("x' = a*x", "dt", {"a"})
    rhs = _rhs(out)
    x, dt, a = _syms(rhs, "x", "dt", "a")
    expected = x * sp.exp(a * dt)
    assert sp.simplify(rhs - expected).equals(0)


# ---- 3. Inhomogeneous linear ODE x' = a + b*x --------------------------------


def test_integrate2c_linear_inhomogeneous():
    out = integrate2c("x' = a + b*x", "dt", {"a", "b"})
    rhs = _rhs(out)
    x, dt, a, b = _syms(rhs, "x", "dt", "a", "b")
    expected = (-a / b) + (a + b * x) * sp.exp(b * dt) / b
    assert sp.simplify(rhs - expected).equals(0)


# ---- 4. Pade approximant correctness to O(dt^2) ------------------------------


def test_integrate2c_pade_matches_series():
    out = integrate2c("x' = a*x", "dt", {"a"}, use_pade_approx=True)
    rhs = _rhs(out)
    x, dt, a = _syms(rhs, "x", "dt", "a")
    # series of exact solution to O(dt^2)
    exact = x * (1 + a * dt + a**2 * dt**2 / 2)
    series = sp.series(sp.sympify(rhs), dt, 0, 3).removeO()
    assert sp.expand(series - exact).equals(0)


@pytest.mark.parametrize(
    "diff_eq, params, expected_fn",
    [
        ("x' = a*x**2", ["a"], lambda x, dt, a: x / (1 - a * x * dt)),
        # keep r before K so the lambda’s signature matches
        (
            "x' = r*x*(1 - x/K)",
            ["r", "K"],
            lambda x, dt, r, K: K * x * sp.exp(r * dt) / (K + x * (sp.exp(r * dt) - 1)),
        ),
        (
            "x' = b*x + a*x**2",
            ["a", "b"],
            lambda x, dt, a, b: x
            * sp.exp(b * dt)
            / (1 + (a * x / b) * (1 - sp.exp(b * dt))),
        ),
    ],
)
def test_integrate2c_extra(diff_eq, params, expected_fn):
    code = integrate2c(diff_eq, "dt", params)
    rhs = _rhs(code)

    # pull the exact symbol objects in the SAME order as in `params`
    x, dt, *syms = _syms(rhs, "x", "dt", *params)
    expected = expected_fn(x, dt, *syms)

    assert sp.simplify(rhs - expected).equals(0)


# ---- 5. Hard ODE should raise -------------------------------------------------


def test_integrate2c_unsupported_raises():
    # SymPy 1.12 returns a list from dsolve for this ODE, which our wrapper
    # does not handle → AttributeError.  Either way, we expect an *exception*.
    with pytest.raises(Exception):
        integrate2c("x' = sin(x) + t", "dt", {"t"})
