"""Symbolic scalar ODE updates and tensor-safe code generation."""

import math
import re
from numbers import Real
from typing import Optional

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.pycode import PythonCodePrinter

_DECLARATION = re.compile(r"([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?\Z")
_CALL_NAME = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _var_to_sympy(declaration):
    """Resolve a scalar name or a one-dimensional array declaration.

    ``x[3]`` declares an array of length three; it does not name element three.
    Element references are resolved separately when parsing an expression.
    """
    match = _DECLARATION.fullmatch(declaration.strip())
    if match is None:
        raise ValueError(f"Invalid variable declaration: {declaration!r}")
    name, length = match.groups()
    if length is None:
        value = sp.Symbol(name, real=True)
    else:
        value = sp.IndexedBase(name, shape=(int(length),), real=True)
    return name, value


def _parse_equation(diff_string, variables):
    """Parse ``state' = rhs`` using an equation-local symbol environment."""
    lhs, separator, rhs = diff_string.partition("=")
    target = lhs.strip()
    if not separator or not target.endswith("'") or not rhs.strip():
        raise ValueError(f"Expected a derivative equation, got {diff_string!r}")
    environment = _build_locals(variables)
    # Token inventories used by mechanism builders also include function names.
    # Keep those names available to SymPy's function parser, rather than treating
    # exp(...) (for example) as a call to a scalar parameter named exp.
    for name in _CALL_NAME.findall(rhs):
        environment.pop(name, None)
    state = sp.sympify(target[:-1].strip(), locals=environment)
    if not isinstance(state, (sp.Symbol, sp.Indexed)):
        raise ValueError(f"Invalid derivative target: {target!r}")
    return state, sp.sympify(_nmodl_preprocess(rhs), locals=environment)


def _symbolic_update(state, rhs, dt, *, use_pade_approx=False, stable_linear=False):
    """Construct a frozen-coefficient scalar flow as a SymPy expression.

    Other states and parameters are held at their start-of-step values. Affine
    and integer-power equations use closed forms; other nonlinear equations
    use SymPy's explicit initial-value solutions. The stable affine form
    retains the authored RHS for the numerical runtime helper instead of
    reconstructing it from rounded coefficients.
    """
    slope = sp.simplify(sp.diff(rhs, state))
    if use_pade_approx:
        # For an autonomous scalar flow, y'(0)=f(y) and y''(0)=f'(y)f(y).
        # Matching these coefficients gives a (1,1) rational step directly,
        # without first solving the ODE or expanding its exact solution.
        return sp.simplify(state + dt * rhs / (1 - dt * slope / 2))
    if slope == 0:
        return sp.simplify(state + dt * rhs)
    if not slope.has(state):
        source = sp.simplify(rhs - slope * state)
        if stable_linear and source != 0:
            return sp.Function("_linear_exact_step")(state, rhs, source, slope, dt)
        equilibrium = sp.simplify(-source / slope)
        return equilibrium + sp.simplify(state - equilibrium) * sp.exp(slope * dt)

    coefficient, monomial = rhs.as_independent(state, as_Add=False)
    if monomial.is_Pow and monomial.base == state and monomial.exp.is_Integer:
        # For y'=a*y**p, keep the initial state outside the root. Solving for
        # y**(1-p) and taking a principal root can discard its initial sign.
        order = 1 - monomial.exp
        return state * (1 + order * coefficient * state ** (-order) * dt) ** (1 / order)

    elapsed = sp.Dummy("elapsed", real=True, positive=True)
    dependent = sp.Function("_ode_state")(elapsed)
    equation = sp.Eq(sp.diff(dependent, elapsed), rhs.subs(state, dependent))
    # Limit the fallback to first-order separable equations, avoiding unrelated
    # general solver strategies and preserving the scalar analytic contract.
    if "separable" not in sp.classify_ode(equation, dependent):
        raise NotImplementedError("ODE too hard")
    solved = sp.dsolve(
        equation, dependent, hint="separable", ics={dependent.subs(elapsed, 0): state}
    )
    if not isinstance(solved, sp.Equality) or solved.lhs != dependent:
        raise NotImplementedError("ODE has no unique explicit scalar solution")
    if any(call.func == dependent.func for call in solved.rhs.atoms(AppliedUndef)):
        raise NotImplementedError("ODE has no explicit scalar solution")
    result = sp.simplify(solved.rhs.subs(elapsed, dt))
    if result.has(sp.Integral, sp.Derivative):
        raise NotImplementedError("ODE has no explicit scalar solution")
    return result


def integrate2c(
    diff_string, dt_var, vars, use_pade_approx=False, *, stable_linear=False
):
    """Return a scalar ODE update as a C assignment.

    Parameters
    ----------
    diff_string : str
        Equation such as ``"x' = drive-rate*x"``.
    dt_var : str
        Name of the timestep variable in the generated expression.
    vars : iterable of str
        Scalar names and array declarations, for example ``["drive", "x[3]"]``.
    use_pade_approx : bool, optional
        Emit a second-order (1,1) Padé step instead of an exact scalar flow.
    stable_linear : bool, optional
        Emit ``_linear_exact_step`` for inhomogeneous affine equations. The
        generated namespace must supply that helper.

    Returns
    -------
    str
        An assignment whose right-hand side uses the start-of-step state.

    Raises
    ------
    NotImplementedError
        An exact explicit scalar solution could not be constructed.

    Notes
    -----
    This adapter retains C output for callers that need it. Mechanism CNEXP
    solvers emit Torch code directly through :func:`integrate2torch`.
    """
    state, rhs = _parse_equation(diff_string, vars)
    dt = sp.Symbol(dt_var, real=True, positive=True)
    update = _symbolic_update(
        state, rhs, dt, use_pade_approx=use_pade_approx, stable_linear=stable_linear
    )
    functions = {str(call.func): str(call.func) for call in update.atoms(AppliedUndef)}
    return f"{sp.ccode(state)} = {sp.ccode(update.evalf(), user_functions=functions)}"


def integrate2torch(
    diff_string,
    dt_var,
    vars,
    use_pade_approx=False,
    *,
    stable_linear=False,
    parameter_names=(),
):
    """Return a Torch assignment, optionally reading named parameters from self.

    The symbolic update shares the C adapter's mathematical contract. Parameter
    binding applies only to symbol and array nodes, so function names and Torch
    namespaces remain independent of mechanism field names.
    """
    state, rhs = _parse_equation(diff_string, vars)
    dt = sp.Symbol(dt_var, real=True, positive=True)
    update = _symbolic_update(
        state, rhs, dt, use_pade_approx=use_pade_approx, stable_linear=stable_linear
    )
    functions = {str(call.func): str(call.func) for call in update.atoms(AppliedUndef)}
    bindings = {name: f"self.{name}" for name in parameter_names if name != dt_var}
    printer = TorchCodePrinter(extra_user_functions=functions, symbol_names=bindings)
    expression = printer.doprint(_piecewise_to_where(update.evalf()))
    if not _validate_torch_expression(expression, allowed_callables=functions):
        raise NotImplementedError(f"Cannot emit a Torch ODE update: {expression}")
    return f"{TorchCodePrinter().doprint(state)} = {expression}"


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

    def __init__(self, settings=None, extra_user_functions=None, *, symbol_names=None):
        self._symbol_names = dict(symbol_names or {})
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

    def doprint(self, expr, assign_to=None):
        # Fold constants even inside undefined runtime helpers: evalf alone
        # can leave their numeric arguments unchanged. Tensor functions such
        # as torch.sin and torch.sqrt cannot consume Python numeric arguments.
        expr = sp.sympify(expr).replace(
            lambda term: isinstance(term, sp.Expr)
            and term.is_number
            and not term.is_Atom,
            lambda term: term.evalf(),
        )
        return super().doprint(expr, assign_to)

    def _print_Symbol(self, expr):
        if expr.name in self._symbol_names:
            return self._symbol_names[expr.name]
        return super()._print_Symbol(expr)

    def _print_IndexedBase(self, expr):
        return self._symbol_names.get(expr.name, expr.name)

    def _print_Indexed(self, expr):
        indices = ", ".join(self._print(index) for index in expr.indices)
        return f"{self._print(expr.base)}[{indices}]"

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt="torch.sqrt")

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
