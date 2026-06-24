# ruff: noqa: F401

"""Generated implicit solvers for two-state free/bound buffer systems.

This builder targets systems of the form

    bound' = ku * free * (1 - bound) - kr * bound
    free'  = source - gamma * bound'

where ``source`` and ``gamma`` may depend on parameters, assigned variables,
voltage-dependent quantities, currents, etc., but not on ``bound`` or ``free``.
The backward-Euler step can be reduced to a scalar quadratic in ``bound``.
"""

from __future__ import annotations

from typing import Iterable

import sympy as sp
import torch

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._solve_utils import extract_vars, match_derivative_to_states
from .ode import (
    TorchCodePrinter,
    _build_locals,
    _nmodl_preprocess,
    _piecewise_to_where,
    _validate_torch_expression,
)

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


def _collect_vars(derivative_by_state: dict[str, str], states, assigned) -> list[str]:
    names = set(states) | set(assigned)
    for diff in derivative_by_state.values():
        names.update(extract_vars(_rhs_string(diff), set()))
    return _unique_preserve_order(
        list(states)
        + sorted(set(assigned))
        + sorted(names - set(states) - set(assigned))
    )


def _parse_rhs(diff_string: str, vars_list: list[str]):
    diff_string = _nmodl_preprocess(diff_string)
    locals_map = _build_locals(vars_list)
    rhs = diff_string.split("=", 1)[1].strip()
    return sp.sympify(rhs, locals=locals_map), locals_map


def _is_zero(expr) -> bool:
    try:
        return bool(sp.simplify(expr) == 0)
    except Exception:
        return False


def _depends_on(expr, *symbols) -> bool:
    try:
        return any(expr.has(sym) for sym in symbols)
    except Exception:
        return True


def _emit_expr(expr, *, name: str) -> str:
    """Emit a torch-valid expression string for a SymPy expression."""
    expr = _piecewise_to_where(sp.simplify(expr))
    printer = TorchCodePrinter()
    code = printer.doprint(expr)
    if not _validate_torch_expression(code):
        raise ValueError(
            f"bufferimplicit could not emit a torch-safe expression for {name}: {code!r}"
        )
    return code


def _normalize_bufferimplicit_options(
    *,
    bound=None,
    free=None,
    clamp_discriminant=True,
    eps=1e-30,
    clamp_bound=False,
    bound_min=0.0,
    bound_max=1.0,
    free_min=None,
    strict=True,
    fallback=None,
    pade=False,
    **method_kwargs,
):
    if method_kwargs:
        valid = (
            "bound, free, clamp_discriminant, eps, clamp_bound, bound_min, "
            "bound_max, free_min, strict, fallback"
        )
        unknown = ", ".join(sorted(method_kwargs))
        raise ValueError(
            f"Unknown bufferimplicit option(s): {unknown}. Valid options are: {valid}."
        )
    if pade:
        raise ValueError("bufferimplicit does not support pade=True.")

    if bound is not None:
        bound = str(bound)
    if free is not None:
        free = str(free)

    eps = float(eps)
    if eps < 0.0:
        raise ValueError("bufferimplicit eps must be non-negative.")

    if free_min is not None:
        free_min = float(free_min)

    fallback = None if fallback is None else str(fallback).lower().replace("-", "_")
    if fallback not in {None, "derivimplicit", "implicit", "backward_euler", "be"}:
        raise ValueError(
            "bufferimplicit fallback must be None or a derivimplicit alias; "
            f"got {fallback!r}."
        )

    return {
        "bound": bound,
        "free": free,
        "clamp_discriminant": bool(clamp_discriminant),
        "eps": eps,
        "clamp_bound": bool(clamp_bound),
        "bound_min": float(bound_min),
        "bound_max": float(bound_max),
        "free_min": free_min,
        "strict": bool(strict),
        "fallback": fallback,
    }


def _infer_bound_free(states, bound, free) -> tuple[str, str]:
    states = list(states)
    if len(states) != 2:
        raise ValueError(
            "bufferimplicit currently supports exactly two state variables: "
            f"one bound state and one free state; got {states!r}."
        )

    if bound is None and free is None:
        # Common Dendra/NEURON calcium-buffer naming: oc/cai.  Otherwise require
        # explicit names to avoid silently solving the wrong pair.
        if "oc" in states and "cai" in states:
            return "oc", "cai"
        raise ValueError(
            "bufferimplicit requires S.METHOD('bufferimplicit', bound=..., free=...) "
            f"for state variables {states!r}."
        )

    if bound is None or free is None:
        raise ValueError(
            "bufferimplicit requires both bound=... and free=... when either is provided."
        )
    if bound == free:
        raise ValueError("bufferimplicit bound and free state names must be distinct.")
    if bound not in states:
        raise ValueError(
            f"bufferimplicit bound state {bound!r} is not in states {states!r}."
        )
    if free not in states:
        raise ValueError(
            f"bufferimplicit free state {free!r} is not in states {states!r}."
        )
    return bound, free


def _analyze_buffer_system(states, assigned, derivative, *, bound, free):
    derivative_by_state = match_derivative_to_states(derivative, states)
    missing = [s for s in states if s not in derivative_by_state]
    if missing:
        raise ValueError(
            f"bufferimplicit missing derivative(s) for state(s): {missing}."
        )

    vars_list = _collect_vars(derivative_by_state, states, assigned)
    F, locals_map = _parse_rhs(derivative_by_state[bound], vars_list)
    G, locals_map = _parse_rhs(derivative_by_state[free], vars_list)

    b = sp.sympify(_nmodl_preprocess(bound), locals=locals_map)
    c = sp.sympify(_nmodl_preprocess(free), locals=locals_map)

    # F = ku*c*(1-b) - kr*b
    try:
        ku = sp.simplify(sp.diff(F, c).subs({b: 0}))
        kr = sp.simplify(-sp.diff(F, b).subs({c: 0}))
    except Exception as exc:
        raise ValueError(
            "bufferimplicit could not infer ku/kr from the bound equation."
        ) from exc

    if _depends_on(ku, b, c) or _depends_on(kr, b, c):
        raise ValueError(
            "bufferimplicit inferred ku/kr that depend on the solved states; "
            f"ku={ku}, kr={kr}."
        )

    expected_F = sp.simplify(ku * c * (1 - b) - kr * b)
    if not _is_zero(F - expected_F):
        raise ValueError(
            "bufferimplicit bound equation must match "
            "bound' = ku*free*(1-bound) - kr*bound. "
            f"For bound={bound!r}, free={free!r}, inferred expected RHS {expected_F}, "
            f"but got {F}."
        )

    # G = source - gamma*F, so mixed_G = -gamma*mixed_F.
    mixed_F = sp.simplify(sp.diff(F, c, b))
    mixed_G = sp.simplify(sp.diff(G, c, b))
    if _is_zero(mixed_F):
        raise ValueError(
            "bufferimplicit could not infer gamma: bound equation has zero mixed derivative."
        )

    gamma = sp.simplify(-mixed_G / mixed_F)
    if _depends_on(gamma, b, c):
        raise ValueError(
            "bufferimplicit inferred gamma that depends on the solved states; "
            f"gamma={gamma}."
        )

    source = sp.simplify(G + gamma * F)
    if _depends_on(source, b, c):
        raise ValueError(
            "bufferimplicit free equation must have the form source - gamma*bound', "
            "with source independent of the bound/free states. "
            f"Inferred source={source}."
        )

    # Verify the entire free RHS reconstruction, not just source independence.
    expected_G = sp.simplify(source - gamma * F)
    if not _is_zero(G - expected_G):
        raise ValueError(
            "bufferimplicit free equation must match free' = source - gamma*bound'. "
            f"Expected {expected_G}, got {G}."
        )

    return {
        "ku": ku,
        "kr": kr,
        "gamma": gamma,
        "source": source,
        "vars_list": vars_list,
        "derivative_by_state": derivative_by_state,
    }


bufferimplicit_template = """
def solve(self, dt, {states_and_assigned}, **kwargs):
    {locals}
    __gamma = ({gamma_expr})
    __source = ({source_expr})
    __ku = ({ku_expr})
    __kr = ({kr_expr})

    __T = {free} + __gamma * {bound} + dt * __source
    __a = dt * __ku * __gamma
    __D = 1.0 + dt * (__ku * (__T + __gamma) + __kr)
    __E = {bound} + dt * __ku * __T
    __disc = __D * __D - 4.0 * __a * __E
    {disc_clamp}
    __sqrt_disc = torch.sqrt(__disc)
    __quad = (2.0 * __E) / (__D + __sqrt_disc)
    __linear = __E / __D
    _{bound} = torch.where(torch.abs(__a) <= {eps!r}, __linear, __quad)
    _{free} = __T - __gamma * _{bound}
    {bound_clamp}
    {free_clamp}
    return {returns}
"""


def build_bufferimplicit(
    states,
    assigned,
    derivative,
    eliminate=None,
    **method_kwargs,
):
    """Build a generated backward-Euler solver for two-state buffer kinetics.

    The supported system is::

        bound' = ku * free * (1 - bound) - kr * bound
        free'  = source - gamma * bound'

    ``ku``, ``kr``, ``gamma``, and ``source`` are inferred symbolically.  All may
    depend on parameters/assigned variables, but not on the solved states.
    """
    options = _normalize_bufferimplicit_options(**method_kwargs)

    for state in states:
        if state in assigned:
            raise ValueError(
                f"State {state} cannot be assigned and used as a state variable."
            )

    eliminate = _normalize_eliminate(eliminate)
    if eliminate:
        raise ValueError(
            "bufferimplicit does not support eliminated kinetic states yet."
        )

    bound, free = _infer_bound_free(states, options["bound"], options["free"])

    try:
        analysis = _analyze_buffer_system(
            states,
            assigned,
            derivative,
            bound=bound,
            free=free,
        )
    except Exception:
        if options["fallback"] or not options["strict"]:
            from ._derivimplicit import build_derivimplicit

            return build_derivimplicit(
                states,
                assigned,
                derivative,
                eliminate=eliminate,
                pade=False,
            )
        raise

    assigned_list_sorted = sorted(list(assigned))
    states_and_assigned = ", ".join(
        _unique_preserve_order(list(states) + assigned_list_sorted)
    )

    vars_needed = set()
    for expr in (analysis["ku"], analysis["kr"], analysis["gamma"], analysis["source"]):
        for sym in expr.free_symbols:
            vars_needed.add(str(sym))

    local_vars = sorted(vars_needed - set(states) - set(assigned_list_sorted))
    locals_str = "\n    ".join([f"{v} = self.{v}" for v in local_vars])

    ku_expr = _emit_expr(analysis["ku"], name="ku")
    kr_expr = _emit_expr(analysis["kr"], name="kr")
    gamma_expr = _emit_expr(analysis["gamma"], name="gamma")
    source_expr = _emit_expr(analysis["source"], name="source")

    disc_clamp = (
        "__disc = torch.clamp_min(__disc, 0.0)" if options["clamp_discriminant"] else ""
    )
    bound_clamp = (
        f"_{bound} = torch.clamp(_{bound}, min={options['bound_min']!r}, max={options['bound_max']!r})"
        if options["clamp_bound"]
        else ""
    )
    free_clamp = (
        f"_{free} = torch.clamp_min(_{free}, {options['free_min']!r})"
        if options["free_min"] is not None
        else ""
    )

    returns = "{" + ", ".join([f"'{s}' : _{s}" for s in states]) + "}"

    solve_src = bufferimplicit_template.format(
        states_and_assigned=states_and_assigned,
        locals=locals_str,
        gamma_expr=gamma_expr,
        source_expr=source_expr,
        ku_expr=ku_expr,
        kr_expr=kr_expr,
        free=free,
        bound=bound,
        eps=options["eps"],
        disc_clamp=disc_clamp,
        bound_clamp=bound_clamp,
        free_clamp=free_clamp,
        returns=returns,
    )

    if DEBUG:
        logger.debug(f"Function:\n{solve_src}")

    return compile_generated_function(
        solve_src,
        func_name="solve",
        filename_prefix="dendra.bufferimplicit.solve",
        global_ns=globals(),
    )
