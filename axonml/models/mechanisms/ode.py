# -- adapted from now defunct bluebrain/nmodl repository --
from importlib import import_module

import sympy as sp

# import known_functions through low-level mechanism because the ccode
# module is overwritten in sympy and contents of that submodule cannot be
# accessed through regular imports
major, minor = (int(v) for v in sp.__version__.split(".")[:2])
if major >= 1 and minor >= 7:
    known_functions = import_module("sympy.printing.c").known_functions_C99
else:
    known_functions = import_module("sympy.printing.ccode").known_functions_C99

if "Abs" in known_functions:
    known_functions.pop("Abs")
    known_functions["abs"] = "abs"


if not ((major >= 1) and (minor >= 2)):
    raise ImportError(f"Requires SympPy version >= 1.2, found {major}.{minor}")


def _var_to_sympy(var_str):
    """Return sympy variable from string representing variable

    If string contains "[" it is assumed to be an array variable
    of the form "variable_name[N]", where N is the size of the array

    Args:
        var_str: variable as string

    Returns:
        (variable_name_as_string, variable_as_sympy_object)
    """
    if "[" in var_str:
        # var is an array variable with defined size, e.g. X[5]
        var_name = var_str.split("[", 1)[0].strip()
        var_len = int(var_str.split("[", 1)[1].split("]", 1)[0].strip())
        # SymPy equivalent of an array is IndexedBase:
        return var_name, sp.IndexedBase(var_name, shape=(var_len,), real=True)
    else:
        # otherwise can use a standard SymPy symbol:
        return var_str, sp.symbols(var_str, real=True)


def _sympify_diff_eq(diff_string, vars):
    """Parse differential equation into sympy expression

    Given eq_string of the form "x' = df/dx", return sympy
    objects representing x and df/dx

    If x is an array, then it should be declared in vars
    as "x[N]", where N is the size of the array, and an
    IndexedBase sympy object will be returned

    Args:
        eq_string: string containing differential equation
        vars: list of strings containing vars used in equation

    Returns:
        x: sympy object representing x
        dxdt: sympy expression representing df/dx
    """
    sympy_vars = {}
    for var in vars:
        var_name, sympy_object = _var_to_sympy(var)
        sympy_vars[var_name] = sympy_object

    diff_string_lhs, diff_string_rhs = diff_string.split("=", 1)

    # parse dependent variable from LHS of equation:
    x = sp.sympify(diff_string_lhs.replace("'", ""), locals=sympy_vars)

    # parse RHS of equation into SymPy expression
    dxdt = sp.sympify(diff_string_rhs, locals=sympy_vars)

    return x, dxdt


def integrate2c(diff_string, dt_var, vars, use_pade_approx=False):
    """Analytically integrate supplied derivative, return solution as C code.

    Given a differential equation of the form x' = f(x), the value of
    x at time t+dt is found in terms of the value of x at time t:
    x(t + dt) = g( x(t), dt )
    and this equation is returned in the format NEURON expects:
    x = g( x, dt ),
    where the x on the right is the current value of x at time t,
    and the x on the left is the new value of x at time t+dt

    The derivative should be of the form "x' = f(x)",
    and vars should contain the set of all the variables
    referenced by f(x), for example:

    -``integrate2c("x' = a*x", "dt", {"a"})``
    -``integrate2c("x' = a + b*x - sin(3.2)", "dt", {"a","b"})``

    Optionally, the analytic result can be expanded in powers of dt,
    and the (1,1) Pade approximant to the solution returned.
    This approximate solution is correct to second order in dt.

    Args:
        diff_string: Derivative to be integrated e.g. "x' = a*x + b"
        t_var: name of time variable t in NEURON
        dt_var: name of timestep variable dt in NEURON
        vars: set of variables used in expression, e.g. {"a", "b"}
        use_pade_approx: if False, return exact solution
                         if True, return (1,1) Pade approx to solution
                         correct to second order in dt_var

    Returns:
        string containing analytic integral of derivative as C code
    Raises:
        NotImplementedError: if the ODE is too hard, or if it fails to solve it.
    """

    # only try to solve ODEs that are not too hard
    ode_properties_require_all = {"separable"}
    ode_properties_require_one_of = {
        "1st_exact",
        "1st_linear",
        "almost_linear",
        "nth_linear_constant_coeff_homogeneous",
        "1st_exact_Integral",
        "1st_linear_Integral",
    }

    x, dxdt = _sympify_diff_eq(diff_string, vars)
    # set up differential equation d(x(t))/dt = ...
    # where the function x_t = x(t) is substituted for the symbol x
    # the dependent variable is a function of t
    t = sp.Dummy("t", real=True, positive=True)
    x_t = sp.Function("x(t)", real=True)(t)
    diffeq = sp.Eq(x_t.diff(t), dxdt.subs({x: x_t}))

    # for simple linear case write down solution in preferred form:
    dt = sp.symbols(dt_var, real=True, positive=True)
    solution = None
    c1 = dxdt.diff(x).simplify()
    if c1 == 0:
        # constant equation:
        # x' = c0
        # x(t+dt) = x(t) + c0 * dt
        solution = (x + dt * dxdt).simplify()
    elif c1.diff(x) == 0:
        # linear equation:
        # x' = c0 + c1*x
        # x(t+dt) = (-c0 + (c0 + c1*x(t))*exp(c1*dt))/c1
        c0 = (dxdt - c1 * x).simplify()
        solution = (-c0 / c1).simplify() + (c0 + c1 * x).simplify() * sp.exp(
            c1 * dt
        ) / c1
    else:
        # otherwise try to solve ODE with sympy:
        # first classify ODE, if it is too hard then exit
        ode_properties = set(sp.classify_ode(diffeq))
        if not ode_properties_require_all <= ode_properties:
            raise NotImplementedError("ODE too hard")
        if len(ode_properties_require_one_of & ode_properties) == 0:
            raise NotImplementedError("ODE too hard")
        # try to find analytic solution, with initial condition x_t(t=0) = x
        # (note dsolve can return a list of solutions, in which case this currently fails)
        solution = sp.dsolve(diffeq, x_t, ics={x_t.subs({t: 0}): x})
        # evaluate solution at x(dt), extract rhs of expression
        solution = solution.subs({t: dt}).rhs.simplify()

    if use_pade_approx:
        # (1,1) order Pade approximant, correct to 2nd order in dt,
        # constructed from the coefficients of 2nd order Taylor expansion
        taylor_series = sp.Poly(sp.series(solution, dt, 0, 3).removeO(), dt)
        _a0 = taylor_series.nth(0)
        _a1 = taylor_series.nth(1)
        _a2 = taylor_series.nth(2)
        solution = (
            (_a0 * _a1 + (_a1 * _a1 - _a0 * _a2) * dt) / (_a1 - _a2 * dt)
        ).simplify()
        # special case where above form gives 0/0 = NaN
        if _a1 == 0 and _a2 == 0:
            solution = _a0

    custom_fcts = {str(f.func): str(f.func) for f in solution.atoms(sp.Function)}

    # return result as C code in NEURON format:
    #   - in the lhs x_0 refers to the state var at time (t+dt)
    #   - in the rhs x_0 refers to the state var at time t
    return f"{sp.ccode(x)} = {sp.ccode(solution.evalf(), user_functions=custom_fcts)}"
