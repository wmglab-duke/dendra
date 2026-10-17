# ruff: noqa: F401

import torch

from dendra.helpers import DEBUG, PADE, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._solve_utils import (
    add_underscore_to_lhs,
    add_underscore_to_states,
    extract_vars,
    match_derivative_to_states,
    modify_operations,
    replace,
)
from .ode import integrate2c
from .ops import expinv

# -- helpers --


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
    # iterable of pairs
    return dict(eliminate)


cnexp_template = """
def solve(self, dt, {states_and_assigned}, **kwargs):
    {solves}
    return {returns}
"""


def _linear_exact_step(state, rate, source, coefficient, dt):
    """Evaluate the exact affine update without losing small increments.

    Near equilibrium and for small timesteps, advance by the authored rate.
    For stiff decay, retain the equilibrium and the surviving transient
    directly, rather than cancelling nearly the entire old state.
    """
    z = coefficient * dt
    small = z.abs() <= 0.5
    small_z = torch.where(small, z, torch.zeros_like(z))
    increment = state + dt * rate / expinv(small_z)
    # Both where branches are evaluated. Keep the unused denominator and
    # exponential finite at zero rate and in the stiff-decay limit, including
    # under reverse/forward AD and vmap.
    large_coefficient = torch.where(small, torch.ones_like(z), coefficient)
    equilibrium = -source / large_coefficient
    large_z = torch.where(small, torch.zeros_like(z), z)
    relaxed = equilibrium + (state - equilibrium) * torch.exp(large_z)
    return torch.where(small, increment, relaxed)


def convert(deriv, state, states, assigned, use_pade_approx=False):
    """
    Convert a derivative expression into a form suitable for numerical integration.

    Parameters
    ----------
    deriv : str
        The derivative expression to be converted.
    state : str
        The state variable involved in the derivative.
    assigned : set
        A set of variables that are assigned values.
    use_pade_approx : bool, optional
        Whether to use Pade approximation for integration. Defaults to False.
    diffusion : Tuple[float, str], optional
        The diffusion to be added, if any. Defaults to None.
    model : object, optional
        The model object (subclass of Axon). Defaults to None.

    Returns
    -------
    str
        The modified derivative expression after integration and optional diffusion addition.
    """

    exclude = set([state]) | set(states) | set(assigned)
    v = extract_vars(deriv, exclude)
    f = integrate2c(deriv, "dt", v, use_pade_approx=use_pade_approx, stable_linear=True)
    if DEBUG:
        logger.debug(f"Integrated function:\n{f}")
    return add_underscore_to_lhs(modify_operations(replace(f, v)))


def build_cnexp(
    states, assigned, derivative, eliminate=None, pade=False, **method_kwargs
):
    if method_kwargs:
        valid = "pade"
        unknown = ", ".join(sorted(method_kwargs))
        raise ValueError(
            f"Unknown cnexp option(s): {unknown}. Valid cnexp options are: {valid}."
        )

    for state in states:
        if state in assigned:
            raise ValueError(
                f"State {state} cannot be assigned and used as a state variable."
            )

    eliminate = _normalize_eliminate(eliminate)

    # Solve only non-eliminated states, preserving input order
    # states_to_solve = [s for s in states if s not in eliminate]

    # Deterministic signature: states first (in given order), then assigned (sorted)
    assigned_list = list(assigned)
    assigned_list_sorted = sorted(assigned_list)
    states_and_assigned = ", ".join(
        _unique_preserve_order(list(states) + assigned_list_sorted)
    )

    solves = []
    returns = []

    if PADE.value == -1:
        use_pade_approx = pade
    else:
        use_pade_approx = bool(PADE)

    derivative = match_derivative_to_states(derivative, states)
    for state in states:
        if state not in eliminate:
            solves.append(
                convert(
                    derivative[state],
                    state,
                    states,
                    assigned,
                    use_pade_approx=use_pade_approx,
                )
            )
        else:
            solves.append(add_underscore_to_states(eliminate[state], states))
        returns.append(f"'{state}' : _{state}")
    solves = "\n    ".join(solves)
    returns = f"{{{', '.join(returns)}}}"
    f = cnexp_template.format(
        states_and_assigned=states_and_assigned, solves=solves, returns=returns
    )
    if DEBUG:
        logger.debug(f"Function:\n{f}")
    return compile_generated_function(
        f,
        func_name="solve",
        filename_prefix="dendra.cnexp.solve",
        global_ns=globals(),
    )
