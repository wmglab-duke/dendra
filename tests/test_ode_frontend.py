"""Mathematical contracts for the symbolic ODE front end and Torch emission."""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import sympy as sp
import torch

from dendra.models.mechanisms import ode
from dendra.models.mechanisms._cnexp import _linear_exact_step, build_cnexp


def _torch_step(source, **values):
    namespace = dict(values)
    exec(
        source,
        {"torch": torch, "_linear_exact_step": _linear_exact_step},
        namespace,
    )
    return namespace["x"]


@pytest.mark.parametrize(
    "equation, variables",
    [
        ("x' = drive", ["drive"]),
        ("x' = rate*x", ["rate"]),
        ("x' = drive-rate*x", ["drive", "rate"]),
        ("x' = rate*x**2", ["rate"]),
        ("x' = rate*x*(1-x/capacity)", ["rate", "capacity"]),
        ("x' = linear*x+quadratic*x**2", ["linear", "quadratic"]),
    ],
    ids=("constant", "homogeneous", "affine", "quadratic", "logistic", "riccati"),
)
def test_symbolic_update_satisfies_initial_value_and_ode(equation, variables):
    state, rhs = ode._parse_equation(equation, variables)
    elapsed = sp.Symbol("elapsed", real=True, positive=True)

    update = ode._symbolic_update(state, rhs, elapsed)

    # These identities establish the flow directly, without comparing its
    # spelling or implementation to the former integration helper.
    assert sp.simplify(update.subs(elapsed, 0) - state) == 0
    residual = sp.diff(update, elapsed) - rhs.subs(state, update)
    assert sp.simplify(residual) == 0


def test_array_declaration_has_length_semantics_and_preserves_indexed_rhs():
    state, rhs = ode._parse_equation("x[0]' = rate*(x[1]-x[0])", ["x[3]", "rate"])

    assert isinstance(state, sp.Indexed)
    assert state.indices == (0,)
    assert state.base.shape == (3,)
    assert {value.indices for value in rhs.atoms(sp.Indexed)} == {(0,), (1,)}
    elapsed = sp.Symbol("elapsed", real=True, positive=True)
    update = ode._symbolic_update(state, rhs, elapsed)
    assert sp.simplify(update.subs(elapsed, 0) - state) == 0
    assert sp.simplify(sp.diff(update, elapsed) - rhs.subs(state, update)) == 0


@pytest.mark.parametrize("equation", ["x' = 0", "x' = a+b*x", "x' = a*x+b*x**2"])
def test_pade_update_has_the_autonomous_ode_taylor_coefficients(equation):
    state, rhs = ode._parse_equation(equation, ["a", "b"])
    elapsed = sp.Symbol("elapsed", real=True, positive=True)

    update = ode._symbolic_update(state, rhs, elapsed, use_pade_approx=True)

    assert sp.simplify(update.subs(elapsed, 0) - state) == 0
    assert sp.simplify(sp.diff(update, elapsed).subs(elapsed, 0) - rhs) == 0
    second_derivative = sp.diff(update, elapsed, 2).subs(elapsed, 0)
    assert sp.simplify(second_derivative - sp.diff(rhs, state) * rhs) == 0


@pytest.mark.parametrize("coefficient", [-1000.0, -0.2, -1e-12, 0.0, 1e-12, 0.2])
def test_direct_torch_affine_update_matches_high_precision_flow(coefficient):
    source = ode.integrate2torch(
        "x' = drive+coefficient*x",
        "dt",
        ["drive", "coefficient"],
        stable_linear=True,
    )
    values = {
        "x": torch.tensor([0.1, 0.9], dtype=torch.float64),
        "drive": torch.tensor(0.7, dtype=torch.float64),
        "coefficient": torch.tensor(coefficient, dtype=torch.float64),
        "dt": torch.tensor(0.3, dtype=torch.float64),
    }

    actual = _torch_step(source, **values)

    # Evaluate exact decimal inputs at 80 digits so the zero-rate limit and
    # cancellation near zero are checked independently of the runtime helper.
    rate = sp.Float(str(coefficient), 80)
    drive, elapsed = sp.Float("0.7", 80), sp.Float("0.3", 80)
    expected = []
    for initial in ("0.1", "0.9"):
        initial = sp.Float(initial, 80)
        if coefficient == 0:
            exact = initial + drive * elapsed
        else:
            exact = (
                initial * sp.exp(rate * elapsed)
                + drive * (sp.exp(rate * elapsed) - 1) / rate
            )
        expected.append(float(exact.evalf(70)))
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64), rtol=3e-15, atol=1e-16
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_nonlinear_torch_emission_preserves_values_and_parameter_gradients(
    dtype,
):
    source = ode.integrate2torch("x' = rate*x**2", "dt", ["rate"])
    inputs = (
        torch.tensor([0.2, 0.7], dtype=dtype, requires_grad=True),
        torch.tensor(0.3, dtype=dtype, requires_grad=True),
        torch.tensor(0.1, dtype=dtype, requires_grad=True),
    )
    state, rate, elapsed = inputs

    actual = _torch_step(source, x=state, rate=rate, dt=elapsed)

    reference_inputs = tuple(
        value.detach().double().requires_grad_() for value in inputs
    )
    initial, coefficient, duration = reference_inputs
    reference = initial / (1 - coefficient * initial * duration)
    torch.testing.assert_close(actual.double(), reference, rtol=2e-6, atol=1e-8)
    actual_gradients = torch.autograd.grad(actual.sum(), inputs)
    reference_gradients = torch.autograd.grad(reference.sum(), reference_inputs)
    for gradient, expected in zip(actual_gradients, reference_gradients, strict=True):
        assert gradient.dtype == dtype
        torch.testing.assert_close(gradient.double(), expected, rtol=3e-6, atol=1e-8)


def test_direct_torch_emission_preserves_authored_float32_equilibrium():
    source = ode.integrate2torch(
        "x' = (equilibrium-x)/tau", "dt", ["equilibrium", "tau"], stable_linear=True
    )
    equilibrium = torch.tensor(
        [0.9702991247177124, 0.7865960597991943], dtype=torch.float32
    )
    state = equilibrium.clone()
    tau = torch.tensor([0.04862092807888985, 0.08306379616260529], dtype=torch.float32)
    elapsed = torch.tensor(
        [0.023824255913496017, 0.04070126265287399], dtype=torch.float32
    )

    actual = _torch_step(source, x=state, equilibrium=equilibrium, tau=tau, dt=elapsed)

    torch.testing.assert_close(actual, equilibrium, rtol=0, atol=0)


def test_cnexp_torch_path_does_not_call_c_printer(monkeypatch):
    def c_printer_is_unavailable(*args, **kwargs):
        raise AssertionError("CNEXP must emit Torch directly")

    monkeypatch.setattr(ode.sp, "ccode", c_printer_is_unavailable)
    solve = build_cnexp(["x"], {"equilibrium", "tau"}, ["x' = (equilibrium-x)/tau"])
    state = torch.tensor([0.2, 0.7], dtype=torch.float64)
    equilibrium = torch.tensor(0.8, dtype=torch.float64)
    tau = torch.tensor(0.4, dtype=torch.float64)
    elapsed = torch.tensor(0.03, dtype=torch.float64)

    actual = solve(None, elapsed, x=state, equilibrium=equilibrium, tau=tau)["x"]

    expected = equilibrium + (state - equilibrium) * torch.exp(-elapsed / tau)
    torch.testing.assert_close(actual, expected)


def test_import_does_not_mutate_sympy_c99_function_registry():
    script = textwrap.dedent(
        """
        from importlib import import_module

        c_printer = import_module("sympy.printing.c")
        original = dict(c_printer.known_functions_C99)
        import_module("dendra.models.mechanisms.ode")
        assert c_printer.known_functions_C99 == original
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cnexp_binds_fields_without_rewriting_elementary_function_names():
    solve = build_cnexp(
        ["x"],
        {"voltage"},
        ["x' = drive*exp(voltage)+sin(voltage)+sqrt(scale)+Abs(offset)-decay*x"],
    )
    fields = SimpleNamespace(
        drive=torch.tensor(0.4, dtype=torch.float64, requires_grad=True),
        scale=torch.tensor(1.3, dtype=torch.float64, requires_grad=True),
        offset=torch.tensor(-0.2, dtype=torch.float64),
        decay=torch.tensor(0.7, dtype=torch.float64),
    )
    voltage = torch.tensor([-0.3, 0.6], dtype=torch.float64)
    state = torch.tensor([0.2, 0.9], dtype=torch.float64)
    elapsed = torch.tensor(0.03, dtype=torch.float64)

    actual = solve(fields, elapsed, x=state, voltage=voltage)["x"]

    drive = (
        fields.drive * torch.exp(voltage)
        + torch.sin(voltage)
        + torch.sqrt(fields.scale)
        + torch.abs(fields.offset)
    )
    equilibrium = drive / fields.decay
    expected = equilibrium + (state - equilibrium) * torch.exp(-fields.decay * elapsed)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(
        actual.sum(), (fields.drive, fields.scale), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        expected.sum(), (fields.drive, fields.scale)
    )
    for gradient, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize("use_scalar_parameter", [True, False])
def test_cnexp_distinguishes_parameter_named_exp_from_exponential_function(
    use_scalar_parameter,
):
    equation = "x' = exp-x" if use_scalar_parameter else "x' = exp(voltage)-x"
    solve = build_cnexp(["x"], {"voltage"}, [equation])
    fields = SimpleNamespace(exp=torch.tensor(0.6, dtype=torch.float64))
    state = torch.tensor([0.2, 0.4], dtype=torch.float64)
    voltage = torch.tensor([-0.5, 0.8], dtype=torch.float64)
    elapsed = torch.tensor(0.1, dtype=torch.float64)

    actual = solve(fields, elapsed, x=state, voltage=voltage)["x"]

    target = fields.exp if use_scalar_parameter else torch.exp(voltage)
    expected = target + (state - target) * torch.exp(-elapsed)
    torch.testing.assert_close(actual, expected)


def test_cnexp_freezes_other_states_during_each_scalar_update():
    solve = build_cnexp(["x", "y"], set(), ["x' = y-x", "y' = x-y"])
    x = torch.tensor([-0.4, 0.1], dtype=torch.float64)
    y = torch.tensor([0.3, 0.9], dtype=torch.float64)
    elapsed = torch.tensor(0.4, dtype=torch.float64)

    actual = solve(None, elapsed, x=x, y=y)

    decay = torch.exp(-elapsed)
    torch.testing.assert_close(actual["x"], y + (x - y) * decay)
    torch.testing.assert_close(actual["y"], x + (y - x) * decay)


def test_direct_torch_emission_binds_indexed_parameter_base_and_gradients():
    source = ode.integrate2torch(
        "x' = weights[0]-weights[1]*x",
        "dt",
        ["weights[2]"],
        stable_linear=True,
        parameter_names=["weights"],
    )
    weights = torch.tensor([0.7, 0.4], dtype=torch.float64, requires_grad=True)
    fields = SimpleNamespace(weights=weights)
    state = torch.tensor([0.2, 0.9], dtype=torch.float64)
    elapsed = torch.tensor(0.03, dtype=torch.float64)

    actual = _torch_step(source, self=fields, x=state, dt=elapsed)

    equilibrium = weights[0] / weights[1]
    expected = equilibrium + (state - equilibrium) * torch.exp(-weights[1] * elapsed)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), weights, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(expected.sum(), weights)[0]
    torch.testing.assert_close(gradient, reference_gradient)


@pytest.mark.parametrize("kind", ["constant", "homogeneous", "affine"])
def test_cnexp_evaluates_numeric_only_elementary_expressions(kind):
    coefficient = "sin(1)+exp(-2)+sqrt(2)+Abs(-3)"
    equations = {
        "constant": f"x' = {coefficient}",
        "homogeneous": f"x' = ({coefficient})*x",
        "affine": f"x' = {coefficient}-x",
    }
    solve = build_cnexp(["x"], set(), [equations[kind]])
    state = torch.tensor([0.2, 0.9], dtype=torch.float64)
    elapsed = torch.tensor(0.03, dtype=torch.float64)

    actual = solve(None, elapsed, x=state)["x"]

    value = float((sp.sin(1) + sp.exp(-2) + sp.sqrt(2) + 3).evalf(80))
    if kind == "constant":
        expected = state + value * elapsed
    elif kind == "homogeneous":
        expected = state * torch.exp(value * elapsed)
    else:
        expected = value + (state - value) * torch.exp(-elapsed)
    torch.testing.assert_close(actual, expected, rtol=3e-15, atol=1e-16)


@pytest.mark.parametrize(
    "solution_form", ["multiple", "implicit", "integral", "dependent_rhs"]
)
def test_nonlinear_fallback_rejects_nonexplicit_or_unresolved_solutions(
    monkeypatch, solution_form
):
    def symbolic_solver(equation, dependent, **kwargs):
        elapsed = dependent.args[0]
        if solution_form == "multiple":
            return [sp.Eq(dependent, elapsed), sp.Eq(dependent, -elapsed)]
        if solution_form == "implicit":
            return sp.Eq(dependent**2, elapsed)
        if solution_form == "dependent_rhs":
            return sp.Eq(dependent, dependent**2 + elapsed, evaluate=False)
        return sp.Eq(dependent, sp.Integral(sp.sin(elapsed) / elapsed, elapsed))

    monkeypatch.setattr(ode.sp, "classify_ode", lambda *args, **kwargs: ("separable",))
    monkeypatch.setattr(ode.sp, "dsolve", symbolic_solver)
    state, rhs = ode._parse_equation("x' = x+x**2", [])
    elapsed = sp.Symbol("elapsed", real=True, positive=True)

    with pytest.raises(NotImplementedError):
        ode._symbolic_update(state, rhs, elapsed)


@pytest.mark.parametrize("reciprocal", [False, True])
def test_torch_printer_numeric_square_roots_are_directly_evaluable(reciprocal):
    expression = 1 / sp.sqrt(2) if reciprocal else sp.sqrt(2)

    source = ode.TorchCodePrinter().doprint(expression)
    actual = eval(source, {"torch": torch})

    expected = 1 / math.sqrt(2) if reciprocal else math.sqrt(2)
    assert float(actual) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    "coefficient, value", [("sqrt(2)", math.sqrt(2)), ("sin(1)", math.sin(1))]
)
def test_symbolic_derivative_with_numeric_function_is_torch_evaluable(
    coefficient, value
):
    expression, ok, depends = ode.differentiate_rhs_2torch_checked(
        f"x' = {coefficient}*x", ["x"], "x", state_vars=["x"]
    )

    assert ok is True
    assert depends is False
    state = torch.tensor([-0.2, 0.7], dtype=torch.float64, requires_grad=True)
    actual = eval(expression, {"torch": torch}, {"x": state})
    expected = torch.autograd.grad((value * state).sum(), state)[0]
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=state.dtype).expand_as(state), expected
    )


@pytest.mark.parametrize("kind", ["cubic", "reciprocal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonlinear_flows_preserve_initial_sign_and_parameter_gradients(kind, dtype):
    equation = "x' = a*x**3" if kind == "cubic" else "x' = a/x"
    source = ode.integrate2torch(equation, "dt", ["a"])
    initial_values = [-0.9, -0.3, 0.0, 0.3, 0.9]
    if kind == "reciprocal":
        # The reciprocal ODE is undefined at zero; its nonzero branches are
        # checked on both sides of that singularity.
        initial_values.remove(0.0)
    inputs = (
        torch.tensor(initial_values, dtype=dtype, requires_grad=True),
        torch.tensor(0.2, dtype=dtype, requires_grad=True),
        torch.tensor(0.03, dtype=dtype, requires_grad=True),
    )
    state, coefficient, elapsed = inputs

    actual = _torch_step(source, x=state, a=coefficient, dt=elapsed)

    reference_inputs = tuple(
        value.detach().double().requires_grad_() for value in inputs
    )
    initial, rate, duration = reference_inputs
    if kind == "cubic":
        # Integrating d(1/x**2)/dt=-2a and continuing smoothly at x=0
        # selects the branch connected to the supplied initial value.
        expected = initial / torch.sqrt(1 - 2 * rate * initial**2 * duration)
    else:
        # Integrating d(x**2)/dt=2a retains the initial sign, rather than
        # selecting the positive root for both initial-value branches.
        expected = initial * torch.sqrt(1 + 2 * rate * duration / initial**2)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-8)
    assert bool(torch.equal(torch.sign(actual), torch.sign(state)))
    # Unequal weights keep the shared coefficient/time sensitivities from
    # cancelling between equal-magnitude positive and negative initial states.
    weights = torch.arange(1, len(initial_values) + 1, dtype=dtype)
    gradients = torch.autograd.grad((actual * weights).sum(), inputs)
    reference_gradients = torch.autograd.grad(
        (expected * weights.double()).sum(), reference_inputs
    )
    for gradient, reference in zip(gradients, reference_gradients, strict=True):
        assert gradient.dtype == dtype
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient.double(), reference, rtol=3e-6, atol=1e-8)


@pytest.mark.parametrize("power", [-3, -2, -1, 2, 3, 4, 5])
def test_integer_power_flow_has_initial_value_and_ode_tangent(power):
    source = ode.integrate2torch(f"x' = a*x**({power})", "dt", ["a"])
    initial_values = [-1.2, -0.4, 0.4, 1.2]
    if power > 0:
        initial_values.insert(2, 0.0)
    initial = torch.tensor(initial_values, dtype=torch.float64)
    coefficient = torch.tensor(0.02, dtype=torch.float64)

    def flow(duration):
        return _torch_step(source, x=initial, a=coefficient, dt=duration)

    torch.testing.assert_close(
        flow(torch.zeros((), dtype=torch.float64)), initial, rtol=0, atol=0
    )
    duration = torch.tensor(0.03, dtype=torch.float64)
    value, tangent = torch.func.jvp(flow, (duration,), (torch.ones_like(duration),))

    # Differentiate the generated flow with respect to elapsed time and
    # compare directly with the defining ODE at the advanced state.
    assert bool(torch.isfinite(value).all())
    assert bool(torch.isfinite(tangent).all())
    torch.testing.assert_close(
        tangent, coefficient * value**power, rtol=2e-13, atol=1e-15
    )
