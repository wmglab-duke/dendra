# ruff: noqa: F401

import re

import torch

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._euler_maruyama import (
    _match_diffusion_to_states,
    _normalize_eliminate,
    _rhs_from_derivative,
    _unique_preserve_order,
)
from ._solve_utils import (
    TORCH_OPS,
    extract_vars,
    match_derivative_to_states,
    modify_operations,
    replace,
)


def _replace_names(expr: str, replacements):
    out = expr
    for name, repl in replacements.items():
        out = re.sub(rf"\b{name}\b", repl, out)
    return out


def _prepare_expr_for_prefixes(
    expr: str, states, assigned, *, state_prefix, assigned_prefix
):
    expr = modify_operations(expr)
    replacements = {state: f"{state_prefix}{state}" for state in states}
    replacements.update({name: f"{assigned_prefix}{name}" for name in assigned})
    expr = _replace_names(expr, replacements)
    exclude = set(replacements.values())
    vars_to_self = extract_vars(expr, exclude)
    vars_to_self = [
        v for v in vars_to_self if v not in {"torch", "dt"} and v not in TORCH_OPS
    ]
    return replace(expr, vars_to_self)


def build_euler_heun(
    states,
    assigned,
    derivative,
    eliminate=None,
    diffusion=None,
    **method_kwargs,
):
    """Build an explicit Euler-Heun State solver for Stratonovich SDEs.

    ``DERIVATIVE`` declarations provide the drift term and ``DIFFUSION``
    declarations provide the multiplicative noise coefficient per state.  The
    default method is the Stratonovich Euler-Heun scheme: it uses an Euler drift
    step and averages only the diffusion coefficient between the old and
    predicted states.  Passing ``average_drift=True`` upgrades the deterministic
    component to a Heun/trapezoidal corrector as well.
    """

    method_kwargs = dict(method_kwargs)
    method_kwargs.pop("pade", None)
    average_drift = bool(method_kwargs.pop("average_drift", False))
    if method_kwargs:
        unknown = ", ".join(sorted(method_kwargs))
        raise ValueError(
            f"Unknown euler_heun option(s): {unknown}. "
            "Supported options are: average_drift."
        )

    states = list(states)
    assigned = list(assigned)
    eliminate = _normalize_eliminate(eliminate)
    derivative_map = match_derivative_to_states(derivative, states)
    diffusion_map = _match_diffusion_to_states(diffusion, states)
    if not diffusion_map:
        raise ValueError(
            "State.METHOD('euler_heun') requires State.DIFFUSION(...). "
            "Use a deterministic method such as 'cnexp' or 'linearimplicit' for ODE-only states."
        )

    assigned_list_sorted = sorted(assigned)
    lines = ["def solve(self, v, dt, values):"]
    lines.append("    __dt_ref = None")
    for state in states:
        lines.append(f"    __old_{state} = values[{state!r}]")
        lines.append(f"    if __dt_ref is None: __dt_ref = __old_{state}")
    lines.append(
        "    __dt = torch.as_tensor(dt, dtype=__dt_ref.dtype, device=__dt_ref.device)"
    )
    lines.append("    __sqrt_dt = torch.sqrt(__dt)")
    lines.append("    __assigned_old = self._derive_assigned_values(v, values)")
    for name in assigned_list_sorted:
        lines.append(f"    __old_assigned_{name} = __assigned_old[{name!r}]")

    returns = []
    for state in states:
        if state in eliminate:
            rhs_line = _prepare_expr_for_prefixes(
                str(eliminate[state]),
                states,
                assigned,
                state_prefix="__old_",
                assigned_prefix="__old_assigned_",
            )
            lines.append(f"    _{state} = {rhs_line}")
            returns.append(f"'{state}': _{state}")
            continue

        drift = (
            _rhs_from_derivative(derivative_map[state])
            if state in derivative_map
            else "0.0"
        )
        diff = diffusion_map.get(state, "0.0")
        drift_old = _prepare_expr_for_prefixes(
            drift,
            states,
            assigned,
            state_prefix="__old_",
            assigned_prefix="__old_assigned_",
        )
        diff_old = _prepare_expr_for_prefixes(
            diff,
            states,
            assigned,
            state_prefix="__old_",
            assigned_prefix="__old_assigned_",
        )
        lines.append(f"    __drift_old_{state} = {drift_old}")
        lines.append(f"    __diff_old_{state} = {diff_old}")
        lines.append(
            f"    __dW_{state} = __sqrt_dt * self._sde_randn_like({state!r}, __old_{state})"
        )
        lines.append(
            f"    __pred_{state} = __old_{state} + __drift_old_{state} * __dt + "
            f"__diff_old_{state} * __dW_{state}"
        )

    lines.append("    __pred_values = {**values,")
    for state in states:
        if state in eliminate:
            lines.append(f"        {state!r}: _{state},")
        else:
            lines.append(f"        {state!r}: __pred_{state},")
    lines.append("    }")
    lines.append("    __assigned_pred = self._derive_assigned_values(v, __pred_values)")
    for name in assigned_list_sorted:
        lines.append(f"    __pred_assigned_{name} = __assigned_pred[{name!r}]")

    for state in states:
        if state in eliminate:
            continue
        drift = (
            _rhs_from_derivative(derivative_map[state])
            if state in derivative_map
            else "0.0"
        )
        diff = diffusion_map.get(state, "0.0")
        drift_pred = _prepare_expr_for_prefixes(
            drift,
            states,
            assigned,
            state_prefix="__pred_",
            assigned_prefix="__pred_assigned_",
        )
        diff_pred = _prepare_expr_for_prefixes(
            diff,
            states,
            assigned,
            state_prefix="__pred_",
            assigned_prefix="__pred_assigned_",
        )
        lines.append(f"    __drift_pred_{state} = {drift_pred}")
        lines.append(f"    __diff_pred_{state} = {diff_pred}")
        if average_drift:
            drift_term = f"0.5 * (__drift_old_{state} + __drift_pred_{state}) * __dt"
        else:
            drift_term = f"__drift_old_{state} * __dt"
        lines.append(
            f"    _{state} = __old_{state} + {drift_term} + "
            f"0.5 * (__diff_old_{state} + __diff_pred_{state}) * __dW_{state}"
        )
        returns.append(f"'{state}': _{state}")

    lines.append(f"    return {{{', '.join(returns)}}}")
    source = "\n".join(lines) + "\n"
    if DEBUG:
        logger.debug(f"Euler-Heun solve function:\n{source}")
    return compile_generated_function(
        source,
        func_name="solve",
        filename_prefix="dendra.euler_heun.solve",
        global_ns=globals(),
    )
