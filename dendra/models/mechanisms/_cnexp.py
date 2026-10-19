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


def _bounded_phi1(z):
    """Evaluate ``expm1(z) / z`` for ``|z| <= 0.5`` without a transcendental.

    The Taylor polynomial is evaluated in Horner form, in the input dtype.
    Its degree is nine for float32 and sixteen for float64. For a polynomial
    of degree N, the truncation error in derivative r is bounded by

        exp(0.5) * 0.5**(N + 1 - r) / ((N + 1 - r)! * (N + 2)).

    This follows by differentiating ``phi1(z) = integral_0^1 exp(t*z) dt``.
    For float32 the value, first- and second-derivative bounds are respectively
    4.1e-11, 8.1e-10 and 1.5e-8, below ordinary rounding error. Float64's
    second-derivative truncation error is below 2.2e-18. Reduced-precision and
    other dtypes retain the previous expinv evaluation rather than acquiring
    an unvalidated approximation or an implicit float32 promotion.

    Callers must mask out-of-range inputs before evaluating this helper.
    Keeping the polynomial's inactive branch bounded also protects higher AD.
    """
    # PyTorch 2.14 func.jvp can promote logical-scalar float32 arithmetic with
    # Python floats to float64. A singleton view keeps its field arithmetic
    # vector-shaped without allocating or adding broadcasts to the HH mechanism.
    scalar = z.ndim == 0
    if scalar:
        z = z.unsqueeze(0)
    if z.dtype == torch.float64:
        value = z * (1 / 355687428096000) + 1 / 20922789888000
        value = value * z + 1 / 1307674368000
        value = value * z + 1 / 87178291200
        value = value * z + 1 / 6227020800
        value = value * z + 1 / 479001600
        value = value * z + 1 / 39916800
        value = value * z + 1 / 3628800
        value = value * z + 1 / 362880
    elif z.dtype == torch.float32:
        value = z * (1 / 3628800) + 1 / 362880
    else:
        value = 1 / expinv(z)
        return value.squeeze(0) if scalar else value
    value = value * z + 1 / 40320
    value = value * z + 1 / 5040
    value = value * z + 1 / 720
    value = value * z + 1 / 120
    value = value * z + 1 / 24
    value = value * z + 1 / 6
    value = value * z + 1 / 2
    value = value * z + 1
    return value.squeeze(0) if scalar else value


def _linear_exact_step(state, rate, source, coefficient, dt):
    """Evaluate the exact affine update without losing small increments.

    Near equilibrium and for small timesteps, advance by the authored rate.
    For stiff decay, retain the equilibrium and the surviving transient
    directly, rather than cancelling nearly the entire old state.
    """
    z = coefficient * dt
    small = z.abs() <= 0.5
    small_z = torch.where(small, z, torch.zeros_like(z))
    if z.dtype == torch.float32 or z.dtype == torch.float64:
        increment = state + dt * rate * _bounded_phi1(small_z)
    else:
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
