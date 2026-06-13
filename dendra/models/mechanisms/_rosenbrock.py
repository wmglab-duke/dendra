"""Generated linearly implicit Rosenbrock-Euler solvers.

This builder implements the one-stage Rosenbrock / linearly implicit Euler update

    (I - gamma * dt * J_f(x_n)) k = dt * f(x_n)
    x_{n+1} = x_n + damping * k

where the Jacobian is computed symbolically at build time and evaluated at the
start-of-step state.  With ``gamma=1`` and ``damping=1``, this method is exact
backward Euler for affine-linear systems and is L-stable on the scalar test
equation.
"""

from __future__ import annotations

from typing import Iterable

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._solve_utils import (
    add_underscore_to_states,
    extract_vars,
    match_derivative_to_states,
    modify_operations,
)
from .ode import differentiate_rhs_2torch_checked

# Keep custom mechanism ops available to generated RHS/Jacobian expressions.
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


def _normalize_rosenbrock_options(
    *,
    gamma=1.0,
    damping=1.0,
    jacobian_regularization=0.0,
    fallback=None,
    strict=True,
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
        valid = "gamma, damping, jacobian_regularization, fallback, strict"
        unknown_s = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown rosenbrock option(s): {unknown_s}. Valid options are: {valid}."
        )

    if pade:
        raise ValueError("rosenbrock does not support pade=True.")

    gamma = float(gamma)
    damping = float(damping)
    jacobian_regularization = float(jacobian_regularization)

    if gamma <= 0.0:
        raise ValueError("rosenbrock gamma must be positive.")
    if damping <= 0.0:
        raise ValueError("rosenbrock damping must be positive.")
    if jacobian_regularization < 0.0:
        raise ValueError("rosenbrock jacobian_regularization must be non-negative.")

    fallback = None if fallback is None else str(fallback).lower().replace("-", "_")
    if fallback not in {None, "derivimplicit", "implicit", "backward_euler", "be"}:
        raise ValueError(
            "rosenbrock fallback must be None or a derivimplicit alias; "
            f"got {fallback!r}."
        )

    return {
        "gamma": gamma,
        "damping": damping,
        "jacobian_regularization": jacobian_regularization,
        "fallback": fallback,
        "strict": bool(strict),
    }


def _fallback_derivimplicit(states, assigned, derivative, eliminate, **kwargs):
    from ._derivimplicit import build_derivimplicit

    return build_derivimplicit(
        states,
        assigned,
        derivative,
        eliminate=eliminate,
        pade=False,
        **kwargs,
    )


def _rhs_expr(diff_string: str) -> str:
    try:
        rhs = diff_string.split("=", 1)[1].strip()
    except Exception as exc:
        raise ValueError(f"Invalid derivative expression {diff_string!r}.") from exc
    return modify_operations(rhs)


def _collect_vars(
    states, assigned, derivative_by_state, states_to_solve
) -> tuple[list[str], list[str]]:
    """Return ``local_vars`` and differentiator ``vars_list`` in deterministic order."""
    states = list(states)
    assigned_sorted = sorted(list(assigned))

    state_names = set(states)
    assigned_names = set(assigned_sorted)
    function_names = set(all_ops()) | _KNOWN_FUNCTION_NAMES

    vars_set = state_names | assigned_names
    exclude = vars_set | function_names
    for s in states_to_solve:
        vars_set.update(extract_vars(derivative_by_state[s], exclude))
        exclude = vars_set | function_names

    local_vars = sorted(vars_set - state_names - assigned_names)
    vars_list = _unique_preserve_order(states + assigned_sorted + local_vars)
    return local_vars, vars_list


def _differentiate_all(derivative_by_state, states_to_solve, vars_list):
    rhs_derivatives = {}
    jac_ok = True
    diff_cache = {}

    extra_user_functions = {name: name for name in all_ops()}

    for s in states_to_solve:
        rhs = derivative_by_state[s]
        for swrt in states_to_solve:
            key = (rhs, swrt, tuple(states_to_solve), tuple(vars_list))
            if key in diff_cache:
                df, ok, _depends = diff_cache[key]
            else:
                df, ok, _depends = differentiate_rhs_2torch_checked(
                    rhs,
                    vars_list,
                    swrt,
                    state_vars=states_to_solve,
                    extra_user_functions=extra_user_functions,
                )
                diff_cache[key] = (df, ok, _depends)
            rhs_derivatives[(s, swrt)] = df
            jac_ok = jac_ok and ok
            if not jac_ok:
                return rhs_derivatives, False
    return rhs_derivatives, True


# -- source templates ---------------------------------------------------------


rosenbrock1_template = """
def solve(self, dt, {states_and_assigned}, **kwargs):
    {locals}
    __ref = {reference_state}
    dt = torch.as_tensor(dt, dtype=__ref.dtype, device=__ref.device)
    {body}
    {eliminate}
    return {returns}
"""


def _render_n1(states_to_solve, derivative_by_state, rhs_derivatives, options):
    s = states_to_solve[0]
    f0 = _rhs_expr(derivative_by_state[s])
    j00 = rhs_derivatives[(s, s)]
    reg = options["jacobian_regularization"]
    reg_expr = f" + {reg!r}" if reg != 0.0 else ""
    return "\n    ".join(
        [
            f"__f0 = ({f0})",
            f"__j00 = ({j00})",
            f"__m00 = 1.0 - ({options['gamma']!r}) * dt * __j00{reg_expr}",
            "__k0 = (dt * __f0) / __m00",
            f"_{s} = {s} + ({options['damping']!r}) * __k0",
        ]
    )


def _render_n2(states_to_solve, derivative_by_state, rhs_derivatives, options):
    s0, s1 = states_to_solve
    f0 = _rhs_expr(derivative_by_state[s0])
    f1 = _rhs_expr(derivative_by_state[s1])
    j00 = rhs_derivatives[(s0, s0)]
    j01 = rhs_derivatives[(s0, s1)]
    j10 = rhs_derivatives[(s1, s0)]
    j11 = rhs_derivatives[(s1, s1)]
    reg = options["jacobian_regularization"]
    reg_expr = f" + {reg!r}" if reg != 0.0 else ""
    gamma = options["gamma"]
    damping = options["damping"]
    return "\n    ".join(
        [
            f"__f0 = ({f0})",
            f"__f1 = ({f1})",
            f"__j00 = ({j00})",
            f"__j01 = ({j01})",
            f"__j10 = ({j10})",
            f"__j11 = ({j11})",
            f"__gdt = ({gamma!r}) * dt",
            f"__m00 = 1.0 - __gdt * __j00{reg_expr}",
            "__m01 =     - __gdt * __j01",
            "__m10 =     - __gdt * __j10",
            f"__m11 = 1.0 - __gdt * __j11{reg_expr}",
            "__b0 = dt * __f0",
            "__b1 = dt * __f1",
            "__det = __m00 * __m11 - __m01 * __m10",
            "__k0 = (__m11 * __b0 - __m01 * __b1) / __det",
            "__k1 = (-__m10 * __b0 + __m00 * __b1) / __det",
            f"_{s0} = {s0} + ({damping!r}) * __k0",
            f"_{s1} = {s1} + ({damping!r}) * __k1",
        ]
    )


def _render_ngeneric(states_to_solve, derivative_by_state, rhs_derivatives, options):
    n = len(states_to_solve)
    states_csv = ", ".join(states_to_solve)
    f_exprs = [
        f"(__zero + ({_rhs_expr(derivative_by_state[s])}))" for s in states_to_solve
    ]

    lines = [
        f"__x = torch.stack([{states_csv}], dim=-1)",
        f"__zero = torch.zeros_like({states_to_solve[0]})",
        f"__f = torch.stack([{', '.join(f_exprs)}], dim=-1)",
        "__J = __x.new_zeros((*__x.shape, __x.shape[-1]))",
    ]
    for i, s in enumerate(states_to_solve):
        for j, swrt in enumerate(states_to_solve):
            lines.append(f"__J[..., {i}, {j}] = ({rhs_derivatives[(s, swrt)]})")

    lines.extend(
        [
            f"__eye = torch.eye({n}, device=__x.device, dtype=__x.dtype).expand((*__x.shape[:-1], {n}, {n}))",
            "__dt_vec = dt[..., None] if dt.ndim > 0 else dt",
            "__dt_mat = dt[..., None, None] if dt.ndim > 0 else dt",
            f"__M = __eye - ({options['gamma']!r}) * __dt_mat * __J",
        ]
    )
    if options["jacobian_regularization"] != 0.0:
        lines.append(f"__M = __M + ({options['jacobian_regularization']!r}) * __eye")
    lines.extend(
        [
            "__b = __dt_vec * __f",
            "__k = _solve_linear_small(__M, __b)",
            f"__x_new = __x + ({options['damping']!r}) * __k",
            f"{', '.join('_' + s for s in states_to_solve)} = torch.unbind(__x_new, dim=-1)",
        ]
    )
    return "\n    ".join(lines)


def _render_body(states_to_solve, derivative_by_state, rhs_derivatives, options):
    n = len(states_to_solve)
    if n == 1:
        return _render_n1(
            states_to_solve, derivative_by_state, rhs_derivatives, options
        )
    if n == 2:
        return _render_n2(
            states_to_solve, derivative_by_state, rhs_derivatives, options
        )
    return _render_ngeneric(
        states_to_solve, derivative_by_state, rhs_derivatives, options
    )


# -- public builder -----------------------------------------------------------


def build_rosenbrock1(
    states,
    assigned,
    derivative,
    eliminate=None,
    **method_kwargs,
):
    """Build a generated one-stage Rosenbrock / linearly implicit Euler solver.

    The generated update is::

        (I - gamma * dt * J_f(x_n)) k = dt * f(x_n)
        x_{n+1} = x_n + damping * k

    It requires an analytic or finite-difference-emitted symbolic Jacobian from
    ``differentiate_rhs_2torch_checked``.  If Jacobian emission fails, pass
    ``fallback='derivimplicit'`` or ``strict=False`` to route to the existing
    nonlinear backward-Euler builder.
    """
    options = _normalize_rosenbrock_options(**method_kwargs)

    for state in states:
        if state in assigned:
            raise ValueError(
                f"State {state} cannot be assigned and used as a state variable."
            )

    eliminate = _normalize_eliminate(eliminate)
    states_to_solve = [s for s in states if s not in eliminate]
    if not states_to_solve:
        raise ValueError("rosenbrock requires at least one non-eliminated state.")

    assigned_list_sorted = sorted(list(assigned))
    states_and_assigned = ", ".join(
        _unique_preserve_order(list(states) + assigned_list_sorted)
    )

    derivative_by_state = match_derivative_to_states(derivative, states)
    missing = [s for s in states_to_solve if s not in derivative_by_state]
    if missing:
        raise ValueError(f"rosenbrock missing derivative(s) for state(s): {missing}.")

    local_vars, vars_list = _collect_vars(
        states,
        assigned,
        derivative_by_state,
        states_to_solve,
    )
    locals_str = "\n    ".join([f"{v} = self.{v}" for v in local_vars])

    rhs_derivatives, jac_ok = _differentiate_all(
        derivative_by_state,
        states_to_solve,
        vars_list,
    )
    if not jac_ok:
        if options["fallback"] or not options["strict"]:
            return _fallback_derivimplicit(
                states,
                assigned,
                derivative,
                eliminate,
            )
        raise ValueError(
            "rosenbrock requires a torch-safe analytic/symbolic Jacobian. "
            "Use fallback='derivimplicit' or strict=False to fall back."
        )

    body = _render_body(
        states_to_solve,
        derivative_by_state,
        rhs_derivatives,
        options,
    )

    eliminate_solves = []
    returns = []
    for s in states:
        if s in eliminate:
            eliminate_solves.append(add_underscore_to_states(eliminate[s], states))
        returns.append(f"'{s}' : _{s}")

    solve_src = rosenbrock1_template.format(
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
        filename_prefix="dendra.rosenbrock.solve",
        global_ns=globals(),
    )
