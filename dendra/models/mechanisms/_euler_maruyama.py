import re

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._solve_utils import (
    TORCH_OPS,
    extract_vars,
    match_derivative_to_states,
    modify_operations,
    replace,
)


def _unique_preserve_order(seq):
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


def _match_diffusion_to_states(diffusion, states):
    matched = {}
    for entry in diffusion or ():
        left, right = str(entry).split("=", 1)
        state = left.strip()
        if state in states:
            matched[state] = right.strip()
    return matched


def _replace_state_names(expr: str, states):
    out = expr
    for state in states:
        out = re.sub(rf"\b{state}\b", f"__old_{state}", out)
    return out


def _rhs_from_derivative(derivative_expr: str) -> str:
    _lhs, rhs = derivative_expr.split("=", 1)
    return rhs.strip()


def _prepare_expr(expr: str, states, assigned):
    expr = modify_operations(expr)
    expr = _replace_state_names(expr, states)
    exclude = set(assigned) | {f"__old_{s}" for s in states}
    vars_to_self = extract_vars(expr, exclude)
    # Avoid accidentally converting torch.<op> pieces or Python constants.
    vars_to_self = [
        v for v in vars_to_self if v not in {"torch", "dt"} and v not in TORCH_OPS
    ]
    return replace(expr, vars_to_self)


def build_euler_maruyama(
    states,
    assigned,
    derivative,
    eliminate=None,
    diffusion=None,
    **method_kwargs,
):
    """Build an explicit Euler-Maruyama State solver.

    ``DERIVATIVE`` declarations provide the drift term and optional ``DIFFUSION``
    declarations provide the multiplicative noise coefficient per state.  Noise is
    generated as a detached standard normal increment by ``State._sde_randn_like``.
    """

    # State.__init__ injects the legacy ``pade`` option for historical CNEXP
    # compatibility. It has no meaning for Euler-Maruyama, so ignore it here.
    method_kwargs = dict(method_kwargs)
    method_kwargs.pop("pade", None)
    if method_kwargs:
        unknown = ", ".join(sorted(method_kwargs))
        raise ValueError(
            f"Unknown euler_maruyama option(s): {unknown}. "
            "The v1 Euler-Maruyama builder takes no method options."
        )

    states = list(states)
    assigned = list(assigned)
    eliminate = _normalize_eliminate(eliminate)
    derivative_map = match_derivative_to_states(derivative, states)
    diffusion_map = _match_diffusion_to_states(diffusion, states)

    assigned_list_sorted = sorted(assigned)
    states_and_assigned = ", ".join(
        _unique_preserve_order(states + assigned_list_sorted)
    )

    lines = [f"def solve(self, dt, {states_and_assigned}, **kwargs):"]
    if not states_and_assigned:
        lines = ["def solve(self, dt, **kwargs):"]
    lines.append("    __dt_ref = None")
    for state in states:
        lines.append(f"    __old_{state} = {state}")
        lines.append(f"    if __dt_ref is None: __dt_ref = __old_{state}")
    lines.append(
        "    __dt = torch.as_tensor(dt, dtype=__dt_ref.dtype, device=__dt_ref.device)"
    )
    lines.append("    __sqrt_dt = torch.sqrt(__dt)")

    returns = []
    for state in states:
        if state in eliminate:
            rhs_line = _replace_state_names(str(eliminate[state]), states)
            rhs_line = _prepare_expr(rhs_line, states, assigned)
            lines.append(f"    _{state} = {rhs_line}")
        else:
            drift = (
                _rhs_from_derivative(derivative_map[state])
                if state in derivative_map
                else "0.0"
            )
            drift = _prepare_expr(drift, states, assigned)
            diffusion_expr = diffusion_map.get(state, "0.0")
            diffusion_expr = _prepare_expr(diffusion_expr, states, assigned)
            lines.append(f"    __drift_{state} = {drift}")
            lines.append(f"    __diff_{state} = {diffusion_expr}")
            lines.append(
                f"    _{state} = __old_{state} + __drift_{state} * __dt + "
                f"__diff_{state} * __sqrt_dt * self._sde_randn_like({state!r}, __old_{state})"
            )
        returns.append(f"'{state}': _{state}")

    lines.append(f"    return {{{', '.join(returns)}}}")
    source = "\n".join(lines) + "\n"
    if DEBUG:
        logger.debug(f"Euler-Maruyama solve function:\n{source}")
    return compile_generated_function(
        source,
        func_name="solve",
        filename_prefix="dendra.euler_maruyama.solve",
        global_ns=globals(),
    )
