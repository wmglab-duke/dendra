# -- adapted from now defunct bluebrain/nmodl repository --
import math
import re
from importlib import import_module
from numbers import Real
from typing import Optional

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

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


# -- differentiate --

_where = sp.Function("where")


def _nmodl_preprocess(s: str) -> str:
    # NMODL uses ^ for power; convert for SymPy/Python.
    return s.replace("^", "**")


def _build_locals(vars_):
    locals_map = {}
    for v in vars_:
        name, obj = _var_to_sympy(v)
        locals_map[name] = obj
    return locals_map


def _piecewise_to_where(expr: sp.Expr) -> sp.Expr:
    """Convert SymPy Piecewise to nested where(cond, a, b) to support torch.where."""
    if isinstance(expr, sp.Piecewise):
        pairs = expr.args
        e_last, c_last = pairs[-1]
        else_expr = (
            _piecewise_to_where(e_last)
            if (c_last is True or c_last == sp.true)
            else _where(c_last, _piecewise_to_where(e_last), sp.nan)
        )
        for e, c in reversed(pairs[:-1]):
            else_expr = _where(c, _piecewise_to_where(e), else_expr)
        return else_expr

    if expr.args:
        return expr.func(*(_piecewise_to_where(a) for a in expr.args))
    return expr


class TorchCodePrinter(PythonCodePrinter):
    """Printer that emits torch.* calls and tensor-safe boolean logic."""

    def __init__(self, settings=None, extra_user_functions=None):
        settings = dict(settings or {})
        settings.setdefault("fully_qualified_modules", True)

        uf = dict(settings.get("user_functions", {}))
        uf.update(
            {
                # elementary
                "exp": "torch.exp",
                "log": "torch.log",
                "sqrt": "torch.sqrt",
                "sin": "torch.sin",
                "cos": "torch.cos",
                "tan": "torch.tan",
                "asin": "torch.asin",
                "acos": "torch.acos",
                "atan": "torch.atan",
                "atan2": "torch.atan2",
                "sinh": "torch.sinh",
                "cosh": "torch.cosh",
                "tanh": "torch.tanh",
                "asinh": "torch.asinh",
                "acosh": "torch.acosh",
                "atanh": "torch.atanh",
                "Abs": "torch.abs",
                "floor": "torch.floor",
                "ceiling": "torch.ceil",
                "erf": "torch.erf",
                # Piecewise rewrite target:
                "where": "torch.where",
            }
        )
        # Optional: allow mapping custom NMODL helper fns to your own torch implementations.
        # Example: {"vtrap": "vtrap"} where vtrap is a Python function in your runtime.
        if extra_user_functions:
            uf.update(extra_user_functions)

        settings["user_functions"] = uf
        super().__init__(settings)

    # tensor-safe boolean ops
    def _print_And(self, expr):
        return " & ".join(f"({self._print(a)})" for a in expr.args)

    def _print_Or(self, expr):
        return " | ".join(f"({self._print(a)})" for a in expr.args)

    def _print_Not(self, expr):
        return f"~({self._print(expr.args[0])})"

    def _print_sign(self, expr):
        return f"torch.sign({self._print(expr.args[0])})"

    # tensor-safe constants
    def _print_NaN(self, expr):
        return "torch.nan"

    def _print_Infinity(self, expr):
        return "torch.inf"

    def _print_NegativeInfinity(self, expr):
        return "-torch.inf"

    def _print_Pi(self, expr):
        return "torch.pi"

    def _print_Exp1(self, expr):
        return "2.718281828459045"


_UNEVALUATED_TOKENS = ("Derivative(", "Integral(", "Subs(", "Lambda(")
_DISALLOWED_PREFIXES = ("math.", "numpy.", "sympy.")


def _validate_torch_expression(expr_str: str, allowed_callables=None) -> bool:
    """
    Heuristic validation:
      - rejects unevaluated SymPy constructs
      - rejects math/numpy/sympy prefixes
      - ensures any function calls are torch.* or explicitly allowed
    """
    if not expr_str or not isinstance(expr_str, str):
        return False

    for tok in _UNEVALUATED_TOKENS:
        if tok in expr_str:
            return False
    for pref in _DISALLOWED_PREFIXES:
        if pref in expr_str:
            return False

    # Find all call sites like name(...) or dotted.name(...)
    # We allow torch.xxx(...), and optionally allowlisted names (e.g., vtrap(...)).
    allowed = set(allowed_callables or [])
    call_pat = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
    for m in call_pat.finditer(expr_str):
        fname = m.group(1)
        if fname.startswith("torch."):
            continue
        if fname in allowed:
            continue
        # Disallow any other callable like exp(...), v(...), etc.
        return False

    return True


def _expr_depends_on_state(expr: sp.Expr, state_obj) -> bool:
    """
    Robust dependency check that handles Symbol, IndexedBase, and Indexed.
    """
    if isinstance(state_obj, sp.Symbol):
        return expr.has(state_obj)

    if isinstance(state_obj, sp.Indexed):
        return expr.has(state_obj)

    if isinstance(state_obj, sp.IndexedBase):
        # Expressions involving x[i] typically contain Indexed(x, i), not the base itself.
        if expr.has(state_obj):
            return True
        for idx in expr.atoms(sp.Indexed):
            # idx.base is the IndexedBase
            if getattr(idx, "base", None) == state_obj:
                return True
        # Sometimes IndexedBase may appear directly
        if state_obj in expr.atoms(sp.IndexedBase):
            return True
        return False

    # Fallback
    return expr.has(state_obj)


def differentiate_rhs_2torch_checked(
    diff_string: str,
    vars,
    wrt: str,
    *,
    state_vars: Optional[list[str]] = None,
    simplify: bool = True,
    extra_user_functions: dict | None = None,
    # Finite-difference fallback controls
    fd_scheme: str = "central",  # "central" | "forward" | "backward"
    fd_eps: float = 1e-6,  # constant epsilon (keeps dependency detection meaningful)
) -> tuple[str, bool, bool]:
    """
    Input:
      diff_string: "x' = f(...)"
      vars: iterable of variable declarations used in RHS (prefer list/tuple, not set)
      wrt: variable name/index to differentiate against, e.g. "x", "m", "x[0]"
      state_vars: list of state variable names (subset of vars) to test dependency against

    Output:
      (df_dy_str, ok, depends_on_any_state)
        df_dy_str: PyTorch-valid expression string for ∂f/∂wrt if ok=True else ""
        ok: True if expression is torch-safe per _validate_torch_expression
        depends_on_any_state: True if the returned derivative expr depends on any of state_vars
    """
    if isinstance(fd_eps, bool) or not isinstance(fd_eps, Real):
        raise TypeError("fd_eps must be a positive, finite real number")
    fd_eps = float(fd_eps)
    if not math.isfinite(fd_eps) or fd_eps <= 0.0:
        raise ValueError("fd_eps must be positive and finite")

    scheme = str(fd_scheme).lower().strip()
    if scheme not in {"central", "forward", "backward"}:
        raise ValueError(
            f"Unsupported fd_scheme={fd_scheme!r}. Use 'central', 'forward', or 'backward'."
        )

    # ---- parse RHS (required for both symbolic and FD routes) ----
    try:
        diff_string = _nmodl_preprocess(diff_string)
        wrt = _nmodl_preprocess(wrt)

        vars_list = list(vars)
        locals_map = _build_locals(vars_list)

        _lhs, rhs = diff_string.split("=", 1)
        f_expr = sp.sympify(rhs.strip(), locals=locals_map)

        wrt_sym = sp.sympify(wrt, locals=locals_map)

    except Exception:
        return "", False, False

    # If RHS does not depend on wrt, derivative is exactly zero (fast path).
    try:
        if not f_expr.has(wrt_sym):
            df_str = "0"
            depends = False
            if state_vars:
                # "0" depends on nothing
                depends = False
            return df_str, True, depends
    except Exception:
        # If .has() is unhappy for some exotic sympy object, just continue.
        pass

    # Helper: compute depends flag + torch-string + validate
    def _emit(df_expr: sp.Expr) -> tuple[str, bool, bool]:
        # Determine whether df depends on any state var (on the symbolic df expression)
        depends_local = False
        if state_vars:
            for s in state_vars:
                # ``state_vars`` contains declarations drawn from ``vars``.  For an
                # array declaration such as ``x[3]``, dependency means dependency
                # on any indexed member of ``x``; sympifying the declaration would
                # instead (and incorrectly) ask only about the out-of-range element
                # ``x[3]``.  Expressions not present in ``vars`` remain useful for
                # callers that intentionally request an exact indexed member.
                if s in vars_list:
                    _, s_sym = _var_to_sympy(s)
                else:
                    s_sym = sp.sympify(_nmodl_preprocess(s), locals=locals_map)
                if _expr_depends_on_state(df_expr, s_sym):
                    depends_local = True
                    break

        # Rewrite Piecewise -> where(...) and print to torch code
        df_expr_pw = _piecewise_to_where(df_expr)

        printer = TorchCodePrinter(extra_user_functions=extra_user_functions)
        df_str_local = printer.doprint(df_expr_pw)

        allowed_calls = set()
        if extra_user_functions:
            allowed_calls.update(extra_user_functions.values())

        ok_local = _validate_torch_expression(
            df_str_local, allowed_callables=allowed_calls
        )
        return (df_str_local if ok_local else ""), ok_local, depends_local

    # ---- 1) Try symbolic differentiation ----
    try:
        df = sp.diff(f_expr, wrt_sym)
        if simplify:
            # Can be expensive, but gives cleaner/faster printed code.
            df = sp.simplify(df)

        df_str, ok, depends = _emit(df)
        if ok:
            return df_str, True, depends

    except Exception:
        # fall through to finite-difference
        pass

    # ---- 2) Finite-difference fallback (still returns a torch-valid string) ----
    try:
        eps = sp.Float(fd_eps)

        if scheme == "central":
            f_plus = f_expr.subs({wrt_sym: wrt_sym + eps})
            f_minus = f_expr.subs({wrt_sym: wrt_sym - eps})
            df_fd = (f_plus - f_minus) / (2 * eps)

        elif scheme == "forward":
            f_plus = f_expr.subs({wrt_sym: wrt_sym + eps})
            df_fd = (f_plus - f_expr) / eps

        elif scheme == "backward":
            f_minus = f_expr.subs({wrt_sym: wrt_sym - eps})
            df_fd = (f_expr - f_minus) / eps

        if simplify:
            # Often cancels out the variable and reduces the FD expression dramatically
            # (e.g., for affine-in-wrt RHS, FD reduces to an exact constant derivative).
            df_fd = sp.simplify(df_fd)

        df_str, ok, depends = _emit(df_fd)
        if ok:
            return df_str, True, depends

    except Exception:
        pass

    # ---- 3) Both symbolic and FD failed ----
    return "", False, False
