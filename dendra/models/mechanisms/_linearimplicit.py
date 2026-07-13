# ruff: noqa: F401

"""Generated linear implicit solvers for affine-linear coupled ODE systems.

This builder targets systems whose right-hand side is affine-linear in the
state vector,

    x' = A(theta, v, t) x + b(theta, v, t)

with coefficients frozen over the timestep.  It emits the backward-Euler update

    (I - dt * A) x_{n+1} = x_n + dt * b

without Newton iteration.  The public method name is ``linearimplicit``;
``sparse`` is registered as a compatibility alias because this occupies the same
slot as NMODL METHOD sparse for small coupled affine systems.
"""

from __future__ import annotations

from typing import Iterable

import sympy as sp
import torch

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._solve_utils import (
    add_underscore_to_states,
    extract_vars,
    match_derivative_to_states,
)
from ._solvers import _solve_linear_small
from .ode import (
    TorchCodePrinter,
    _build_locals,
    _nmodl_preprocess,
    _piecewise_to_where,
    _validate_torch_expression,
)

# Keep custom mechanism ops available to generated coefficient expressions.
try:  # pragma: no cover - defensive for stripped-down test environments
    from .ops import *  # noqa: F401,F403
    from .ops import all_ops
except Exception:  # pragma: no cover

    def all_ops():
        return set()


_KNOWN_FUNCTION_NAMES = {
    "torch",
    "exp",
    "expm1",
    "expit",
    "sigmoid",
    "log",
    "log10",
    "log2",
    "log1p",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "abs",
    "Abs",
    "floor",
    "ceil",
    "ceiling",
    "erf",
    "where",
    "minimum",
    "maximum",
    "Piecewise",
}


# -- helpers -----------------------------------------------------------------


def _unique_preserve_order(seq: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _normalize_eliminate(eliminate):
    if eliminate is None:
        return {}
    if isinstance(eliminate, dict):
        return dict(eliminate)
    return dict(eliminate)


def _rhs_string(diff_string: str) -> str:
    try:
        return diff_string.split("=", 1)[1].strip()
    except Exception as exc:
        raise ValueError(f"Invalid derivative expression {diff_string!r}.") from exc


def _assignment_rhs(assign_string: str) -> str:
    try:
        return assign_string.split("=", 1)[1].strip()
    except Exception as exc:
        raise ValueError(
            f"Invalid algebraic eliminate assignment {assign_string!r}."
        ) from exc


def _collect_vars(
    states,
    assigned,
    derivative_by_state: dict[str, str],
    eliminate: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Return ``local_vars`` and SymPy ``vars_list`` in deterministic order."""
    state_names = set(states)
    assigned_sorted = sorted(list(assigned))
    assigned_names = set(assigned_sorted)
    function_names = set(all_ops()) | _KNOWN_FUNCTION_NAMES

    vars_set = state_names | assigned_names
    exclude = vars_set | function_names

    for diff in derivative_by_state.values():
        vars_set.update(extract_vars(_rhs_string(diff), exclude))
        exclude = vars_set | function_names

    for assign in eliminate.values():
        vars_set.update(extract_vars(_assignment_rhs(assign), exclude))
        exclude = vars_set | function_names

    local_vars = sorted(vars_set - state_names - assigned_names)
    vars_list = _unique_preserve_order(list(states) + assigned_sorted + local_vars)
    return local_vars, vars_list


def _parse_expr(expr_string: str, locals_map):
    return sp.sympify(_nmodl_preprocess(expr_string), locals=locals_map)


def _parse_rhs(diff_string: str, locals_map):
    return _parse_expr(_rhs_string(diff_string), locals_map)


def _parse_eliminate_assignment(assign_string: str, locals_map):
    lhs, rhs = assign_string.split("=", 1)
    lhs_sym = sp.sympify(_nmodl_preprocess(lhs.strip()), locals=locals_map)
    rhs_expr = sp.sympify(_nmodl_preprocess(rhs.strip()), locals=locals_map)
    return lhs_sym, rhs_expr


def _is_zero(expr) -> bool:
    try:
        return bool(sp.simplify(expr) == 0)
    except Exception:
        return False


def _depends_on_any(expr, symbols) -> bool:
    try:
        return any(expr.has(sym) for sym in symbols)
    except Exception:
        return True


def _emit_expr(expr, *, name: str) -> str:
    """Emit a torch-valid expression string for a SymPy expression."""
    expr = _piecewise_to_where(sp.simplify(expr))
    ops = set(all_ops())
    printer = TorchCodePrinter(extra_user_functions={op: op for op in ops})
    code = printer.doprint(expr)
    if not _validate_torch_expression(code, allowed_callables=ops):
        raise ValueError(
            f"linearimplicit could not emit a torch-safe expression for {name}: {code!r}"
        )
    return code


def _normalize_linearimplicit_options(
    *,
    solver="auto",
    jacobian_regularization=0.0,
    strict=True,
    fallback=None,
    pade=False,
    **method_kwargs,
):
    aliases = {
        "regularization": "jacobian_regularization",
        "jac_regularization": "jacobian_regularization",
        "jac_reg": "jacobian_regularization",
        "reg": "jacobian_regularization",
    }
    unknown = []
    for key, value in dict(method_kwargs).items():
        key = aliases.get(key, key)
        if key == "jacobian_regularization":
            jacobian_regularization = value
        else:
            unknown.append(key)

    if unknown:
        valid = "solver, jacobian_regularization, strict, fallback"
        unknown_s = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown linearimplicit option(s): {unknown_s}. Valid options are: {valid}."
        )
    if pade:
        raise ValueError("linearimplicit/sparse does not support pade=True.")

    solver = str(solver).strip().lower().replace("-", "_")
    solver_aliases = {
        "direct": "auto",
        "small_dense": "small",
        "dense_small": "small",
        "torch": "dense",
        "linalg": "dense",
    }
    solver = solver_aliases.get(solver, solver)
    if solver not in {"auto", "small", "dense"}:
        raise ValueError(
            "linearimplicit solver must be 'auto', 'small', or 'dense'; "
            f"got {solver!r}."
        )

    fallback = (
        None if fallback is None else str(fallback).strip().lower().replace("-", "_")
    )
    fallback_aliases = {
        "implicit": "derivimplicit",
        "deriv_implicit": "derivimplicit",
        "backward_euler": "derivimplicit",
        "be": "derivimplicit",
        "rosenbrock1": "rosenbrock",
        "rosenbrock_euler": "rosenbrock",
        "linearlyimplicit": "rosenbrock",
        "linearly_implicit": "rosenbrock",
        "semiimplicit": "rosenbrock",
        "semi_implicit": "rosenbrock",
    }
    fallback = fallback_aliases.get(fallback, fallback)
    if fallback not in {None, "rosenbrock", "derivimplicit"}:
        raise ValueError(
            "linearimplicit fallback must be None, 'rosenbrock', or 'derivimplicit'; "
            f"got {fallback!r}."
        )

    jacobian_regularization = float(jacobian_regularization)
    if jacobian_regularization < 0.0:
        raise ValueError("linearimplicit jacobian_regularization must be non-negative.")

    return {
        "solver": solver,
        "jacobian_regularization": jacobian_regularization,
        "strict": bool(strict),
        "fallback": fallback,
    }


def _fallback_builder(states, assigned, derivative, eliminate, *, fallback):
    if fallback == "derivimplicit":
        from ._derivimplicit import build_derivimplicit

        return build_derivimplicit(
            states,
            assigned,
            derivative,
            eliminate=eliminate,
            pade=False,
        )

    # Default non-strict fallback is the fast general linearly implicit method.
    from ._rosenbrock import build_rosenbrock1

    return build_rosenbrock1(
        states,
        assigned,
        derivative,
        eliminate=eliminate,
        fallback="derivimplicit",
    )


class _AffineAnalysisError(ValueError):
    pass


def _analyze_affine_system(states, assigned, derivative, eliminate):
    """Infer A and b for x' = A x + b, or raise _AffineAnalysisError."""
    states = list(states)
    eliminate = _normalize_eliminate(eliminate)
    states_to_solve = [s for s in states if s not in eliminate]
    if not states_to_solve:
        raise _AffineAnalysisError(
            "linearimplicit requires at least one non-eliminated state."
        )

    derivative_by_state = match_derivative_to_states(derivative, states)
    missing = [s for s in states_to_solve if s not in derivative_by_state]
    if missing:
        raise _AffineAnalysisError(
            f"linearimplicit missing derivative(s) for state(s): {missing}."
        )

    local_vars, vars_list = _collect_vars(
        states, assigned, derivative_by_state, eliminate
    )
    locals_map = _build_locals(vars_list)

    try:
        state_syms = {
            s: sp.sympify(_nmodl_preprocess(s), locals=locals_map) for s in states
        }
        solve_syms = [state_syms[s] for s in states_to_solve]

        eliminate_subs = {}
        for _elim_name, assign in eliminate.items():
            lhs_sym, rhs_expr = _parse_eliminate_assignment(assign, locals_map)
            eliminate_subs[lhs_sym] = rhs_expr

        rhs_exprs = {}
        for s in states_to_solve:
            rhs = _parse_rhs(derivative_by_state[s], locals_map)
            if eliminate_subs:
                rhs = rhs.subs(eliminate_subs)
            rhs_exprs[s] = sp.simplify(rhs)
    except Exception as exc:
        raise _AffineAnalysisError(
            f"linearimplicit could not parse affine system: {exc}"
        ) from exc

    all_state_symbols = [state_syms[s] for s in states]
    A = []
    b = []

    try:
        for s in states_to_solve:
            f = rhs_exprs[s]

            # Explicit second-derivative test gives better diagnostics for nonlinear systems.
            for xj in solve_syms:
                for xk in solve_syms:
                    if not _is_zero(sp.diff(sp.diff(f, xj), xk)):
                        raise _AffineAnalysisError(
                            f"RHS for state {s!r} is not affine-linear in solved states."
                        )

            row = []
            for xj in solve_syms:
                aij = sp.simplify(sp.diff(f, xj))
                if _depends_on_any(aij, all_state_symbols):
                    raise _AffineAnalysisError(
                        f"linear coefficient d({s}')/d({xj}) depends on a state variable."
                    )
                row.append(aij)

            bi = sp.simplify(
                f - sum(row[j] * solve_syms[j] for j in range(len(solve_syms)))
            )
            if _depends_on_any(bi, all_state_symbols):
                raise _AffineAnalysisError(
                    f"affine offset for state {s!r} depends on a state variable."
                )

            A.append(row)
            b.append(bi)
    except _AffineAnalysisError:
        raise
    except Exception as exc:
        raise _AffineAnalysisError(
            f"linearimplicit affine analysis failed: {exc}"
        ) from exc

    try:
        A_code = [
            [
                _emit_expr(
                    A[i][j], name=f"A[{states_to_solve[i]},{states_to_solve[j]}]"
                )
                for j in range(len(states_to_solve))
            ]
            for i in range(len(states_to_solve))
        ]
        b_code = [
            _emit_expr(bi, name=f"b[{states_to_solve[i]}]") for i, bi in enumerate(b)
        ]
    except Exception as exc:
        raise _AffineAnalysisError(str(exc)) from exc

    return {
        "states_to_solve": states_to_solve,
        "local_vars": local_vars,
        "A": A_code,
        "b": b_code,
    }


# -- source templates ---------------------------------------------------------


linearimplicit_template = """
def solve(self, dt, {states_and_assigned}, **kwargs):
    {locals}
    __ref = {reference_state}
    dt = torch.as_tensor(dt, dtype=__ref.dtype, device=__ref.device)
    {body}
    {eliminate}
    return {returns}
"""


def _reg_expr(options):
    reg = options["jacobian_regularization"]
    return f" + {reg!r}" if reg != 0.0 else ""


def _render_n1(states_to_solve, A, b, options):
    s = states_to_solve[0]
    reg = _reg_expr(options)
    return "\n    ".join(
        [
            f"__a00 = ({A[0][0]})",
            f"__b0 = ({b[0]})",
            f"__m00 = 1.0 - dt * __a00{reg}",
            f"__rhs0 = {s} + dt * __b0",
            f"_{s} = __rhs0 / __m00",
        ]
    )


def _render_n2(states_to_solve, A, b, options):
    s0, s1 = states_to_solve
    reg = _reg_expr(options)
    return "\n    ".join(
        [
            f"__a00 = ({A[0][0]})",
            f"__a01 = ({A[0][1]})",
            f"__a10 = ({A[1][0]})",
            f"__a11 = ({A[1][1]})",
            f"__b0 = ({b[0]})",
            f"__b1 = ({b[1]})",
            f"__m00 = 1.0 - dt * __a00{reg}",
            "__m01 =     - dt * __a01",
            "__m10 =     - dt * __a10",
            f"__m11 = 1.0 - dt * __a11{reg}",
            f"__rhs0 = {s0} + dt * __b0",
            f"__rhs1 = {s1} + dt * __b1",
            "__det = __m00 * __m11 - __m01 * __m10",
            f"_{s0} = (__m11 * __rhs0 - __m01 * __rhs1) / __det",
            f"_{s1} = (-__m10 * __rhs0 + __m00 * __rhs1) / __det",
        ]
    )


def _render_ngeneric(states_to_solve, A, b, options):
    n = len(states_to_solve)
    states_csv = ", ".join(states_to_solve)
    lines = [
        f"__x = torch.stack([{states_csv}], dim=-1)",
        f"__zero = torch.zeros_like({states_to_solve[0]})",
        "__A = __x.new_zeros((*__x.shape, __x.shape[-1]))",
    ]
    for i in range(n):
        for j in range(n):
            lines.append(f"__A[..., {i}, {j}] = __zero + ({A[i][j]})")

    b_terms = [f"(__zero + ({expr}))" for expr in b]
    lines.extend(
        [
            f"__b = torch.stack([{', '.join(b_terms)}], dim=-1)",
            f"__eye = torch.eye({n}, device=__x.device, dtype=__x.dtype).expand((*__x.shape[:-1], {n}, {n}))",
            "__dt_vec = dt[..., None] if dt.ndim > 0 else dt",
            "__dt_mat = dt[..., None, None] if dt.ndim > 0 else dt",
            "__M = __eye - __dt_mat * __A",
        ]
    )
    if options["jacobian_regularization"] != 0.0:
        lines.append(f"__M = __M + ({options['jacobian_regularization']!r}) * __eye")
    lines.extend(
        [
            "__rhs = __x + __dt_vec * __b",
            "__x_new = _solve_linear_small(__M, __rhs)",
            f"{', '.join('_' + s for s in states_to_solve)}{',' if n == 1 else ''} = torch.unbind(__x_new, dim=-1)",
        ]
    )
    return "\n    ".join(lines)


def _render_body(states_to_solve, A, b, options):
    n = len(states_to_solve)
    if options["solver"] != "dense":
        if n == 1:
            return _render_n1(states_to_solve, A, b, options)
        if n == 2:
            return _render_n2(states_to_solve, A, b, options)
    return _render_ngeneric(states_to_solve, A, b, options)


# -- public builder -----------------------------------------------------------


def build_linearimplicit(
    states,
    assigned,
    derivative,
    eliminate=None,
    **method_kwargs,
):
    """Build a generated backward-Euler solver for affine-linear ODE systems.

    The RHS must be affine-linear in the non-eliminated state variables.  If it
    is not, pass ``fallback='rosenbrock'`` or ``strict=False`` to route to the
    fast nonlinear linearly implicit method, or ``fallback='derivimplicit'`` for
    the fully nonlinear implicit solver.
    """
    options = _normalize_linearimplicit_options(**method_kwargs)

    for state in states:
        if state in assigned:
            raise ValueError(
                f"State {state} cannot be assigned and used as a state variable."
            )

    eliminate = _normalize_eliminate(eliminate)

    try:
        analysis = _analyze_affine_system(states, assigned, derivative, eliminate)
    except _AffineAnalysisError as exc:
        if options["fallback"] or not options["strict"]:
            fallback = options["fallback"] or "rosenbrock"
            return _fallback_builder(
                states,
                assigned,
                derivative,
                eliminate,
                fallback=fallback,
            )
        raise ValueError(str(exc)) from exc

    states_to_solve = analysis["states_to_solve"]
    local_vars = analysis["local_vars"]
    A = analysis["A"]
    b = analysis["b"]

    assigned_list_sorted = sorted(list(assigned))
    states_and_assigned = ", ".join(
        _unique_preserve_order(list(states) + assigned_list_sorted)
    )
    locals_str = "\n    ".join([f"{v} = self.{v}" for v in local_vars])

    body = _render_body(states_to_solve, A, b, options)

    eliminate_solves = []
    returns = []
    for s in states:
        if s in eliminate:
            eliminate_solves.append(add_underscore_to_states(eliminate[s], states))
        returns.append(f"'{s}' : _{s}")

    solve_src = linearimplicit_template.format(
        states_and_assigned=states_and_assigned,
        locals=locals_str,
        reference_state=states_to_solve[0],
        body=body,
        eliminate="\n    ".join(eliminate_solves),
        returns="{" + ", ".join(returns) + "}",
    )

    if DEBUG:
        logger.debug(f"Function:\n{solve_src}")

    return compile_generated_function(
        solve_src,
        func_name="solve",
        filename_prefix="dendra.linearimplicit.solve",
        global_ns=globals(),
    )
