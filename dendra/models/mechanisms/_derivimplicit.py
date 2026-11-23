import math
from collections.abc import Callable
from enum import IntEnum

import torch

from dendra._solver_failures import _capturing_solver_checks, _record_solver_check
from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._linearimplicit import (
    _AffineAnalysisError,
    _analyze_affine_system,
    _compile_affine_solver,
    _rhs_string,
)
from ._solve_utils import (
    add_underscore_to_states,
    extract_vars,
    match_derivative_to_states,
    modify_operations,
)
from ._solvers import _solve_linear_checked
from .ode import (
    _nmodl_preprocess,
    _validate_torch_expression,
    differentiate_rhs_2torch_checked,
)
from .ops import all_ops

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


def _normalize_derivimplicit_options(options):
    """Normalize method kwargs accepted by build_derivimplicit.

    These values are embedded as constants in the generated ``solve`` function.
    ``detach_newton=None`` means use the historical policy ``not self.training``;
    ``create_graph=None`` means use the historical policy ``self.training``.
    """
    out = {
        "tol": None,
        "max_iter": 25,
        "damping": 1.0,
        "line_search": False,
        "max_ls_steps": 10,
        "ls_decay": 0.5,
        "jacobian_regularization": 0.0,
        "allow_inexact": False,
        "detach_newton": None,
        "create_graph": None,
    }
    aliases = {
        "iterations": "max_iter",
        "iters": "max_iter",
        "n_iter": "max_iter",
        "newton_iters": "max_iter",
        "newton_iterations": "max_iter",
        "regularization": "jacobian_regularization",
        "jac_regularization": "jacobian_regularization",
        "jac_reg": "jacobian_regularization",
    }
    for key, value in dict(options).items():
        key = aliases.get(key, key)
        if key not in out:
            valid = ", ".join(sorted(out))
            raise ValueError(
                f"Unknown derivimplicit option {key!r}. Valid options are: {valid}."
            )
        out[key] = value

    out["max_iter"] = int(out["max_iter"])
    out["max_ls_steps"] = int(out["max_ls_steps"])
    if out["tol"] is not None:
        out["tol"] = float(out["tol"])
    out["damping"] = float(out["damping"])
    out["ls_decay"] = float(out["ls_decay"])
    out["jacobian_regularization"] = float(out["jacobian_regularization"])
    out["line_search"] = bool(out["line_search"])
    out["allow_inexact"] = bool(out["allow_inexact"])

    for nullable_bool in ("detach_newton", "create_graph"):
        value = out[nullable_bool]
        if value is not None:
            out[nullable_bool] = bool(value)

    if out["max_iter"] < 0:
        raise ValueError("derivimplicit max_iter must be non-negative.")
    if out["max_ls_steps"] < 0:
        raise ValueError("derivimplicit max_ls_steps must be non-negative.")
    if not math.isfinite(out["damping"]) or out["damping"] <= 0.0:
        raise ValueError("derivimplicit damping must be positive.")
    if not (0.0 < out["ls_decay"] < 1.0):
        raise ValueError("derivimplicit ls_decay must be in the open interval (0, 1).")
    if out["tol"] is not None and (not math.isfinite(out["tol"]) or out["tol"] < 0.0):
        raise ValueError("derivimplicit tol must be non-negative.")
    if (
        not math.isfinite(out["jacobian_regularization"])
        or out["jacobian_regularization"] < 0.0
    ):
        raise ValueError("derivimplicit jacobian_regularization must be non-negative.")

    return out


def _derivimplicit_option_call_args(options):
    detach_expr = (
        "(not self.training)"
        if options["detach_newton"] is None
        else repr(options["detach_newton"])
    )
    create_expr = (
        "self.training"
        if options["create_graph"] is None
        else repr(options["create_graph"])
    )
    return (
        f"tol={options['tol']!r}, "
        f"max_iter={options['max_iter']!r}, "
        f"damping={options['damping']!r}, "
        f"line_search={options['line_search']!r}, "
        f"max_ls_steps={options['max_ls_steps']!r}, "
        f"ls_decay={options['ls_decay']!r}, "
        f"jacobian_regularization={options['jacobian_regularization']!r}, "
        f"allow_inexact={options['allow_inexact']!r}, "
        f"detach_newton={detach_expr}, "
        f"create_graph={create_expr}"
    )


def _broadcast_dt(dt_ms: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast dt to the batch shape of x (excluding last state dimension)."""
    dt = torch.as_tensor(dt_ms, dtype=x.dtype, device=x.device)
    if dt.ndim == 0:
        return dt
    return dt.expand(x.shape[:-1])


def _end_time(t_ms, dt, x):
    if t_ms is None:
        return None
    time = torch.as_tensor(t_ms, dtype=x.dtype, device=x.device)
    return (time.expand(x.shape[:-1]) if time.ndim else time) + dt


def _get_while_loop():
    # torch.while_loop exists in newer versions; fall back to the HOP location otherwise.
    if hasattr(torch, "while_loop"):
        return torch.while_loop
    from torch._higher_order_ops.while_loop import while_loop

    return while_loop


def _broadcast_jac(
    Jf: torch.Tensor, batch_shape: tuple, n: int, like: torch.Tensor
) -> torch.Tensor:
    """
    Ensure Jacobian is shaped (*batch_shape, n, n).
    Accepts (n,n) or (*batch,n,n).
    """
    Jf = Jf.to(device=like.device, dtype=like.dtype)
    if Jf.shape == (n, n):
        return Jf.expand((*batch_shape, n, n))
    if Jf.shape == (*batch_shape, n, n):
        return Jf
    # Last resort: let broadcasting try (will error if incompatible)
    return Jf.expand((*batch_shape, n, n))


def _autograd_batch_jacobian(
    f: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    x: torch.Tensor,
    t_next: torch.Tensor | None,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Return the independent-state Jacobian for every batch member.

    ``f`` is defined on the complete batched state.  Calling it on one flattened
    row is not generally valid: time and mechanism parameters captured by the
    callback can themselves be batched.  The solver assumes batch members are
    independent, as required by its block-diagonal Newton systems.  For each
    state component, differentiate the sum across members to produce those
    diagonal blocks without materializing a quadratic full-batch Jacobian.
    """
    *batch_shape, n = x.shape
    x_for_jacobian = x if x.requires_grad else x.detach().requires_grad_(True)
    values = f(x_for_jacobian, t_next)
    # Keep a zero-dependence path for RHS components that are constant in state.
    values = values + 0.0 * x_for_jacobian
    rows = []
    for output_index in range(n):
        (row,) = torch.autograd.grad(
            values[..., output_index].sum(),
            x_for_jacobian,
            create_graph=create_graph,
            retain_graph=True,
        )
        rows.append(row)
    return torch.stack(rows, dim=-2).reshape(*batch_shape, n, n)


def _get_dynamo_disable():
    # fall back to the public torch.compiler.disable.
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        return torch._dynamo.disable
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        return torch.compiler.disable
    raise RuntimeError(
        "No Dynamo disable decorator found (expected torch._dynamo.disable or torch.compiler.disable)."
    )


_disable = _get_dynamo_disable()


class DerivimplicitStatus(IntEnum):
    CONVERGED = 0
    NONFINITE_RESIDUAL = 1
    LINEAR_SOLVE_FAILED = 2
    STAGNATED = 3
    ITERATION_LIMIT = 4


_CONVERGED = DerivimplicitStatus.CONVERGED.value
_NONFINITE = DerivimplicitStatus.NONFINITE_RESIDUAL.value
_LINEAR_FAILURE = DerivimplicitStatus.LINEAR_SOLVE_FAILED.value
_STAGNATED = DerivimplicitStatus.STAGNATED.value
_ITERATION_LIMIT = DerivimplicitStatus.ITERATION_LIMIT.value
_ABSOLUTE_TOLERANCE = 1e-8
_ROUNDOFF_FACTOR = 4.0


def _residual_bound(x, x_old, scaled_rhs, absolute_tol, relative_tol):
    if relative_tol is None:
        return absolute_tol
    scale = torch.maximum(torch.maximum(x.abs(), x_old.abs()), scaled_rhs.abs())
    return absolute_tol + relative_tol * scale


def _residual_status(x, x_old, scaled_rhs, absolute_tol, relative_tol):
    residual = x - x_old - scaled_rhs
    bound = _residual_bound(x, x_old, scaled_rhs, absolute_tol, relative_tol)
    within_tol = (residual.abs() <= bound).all(dim=-1)
    status = torch.where(within_tol, _CONVERGED, _ITERATION_LIMIT)
    status = torch.where(torch.isfinite(residual).all(dim=-1), status, _NONFINITE)
    return residual, status


def _eager_while_loop(cond, body, carried):
    while cond(*carried).item():
        carried = body(*carried)
    return carried


def _iterate_newton(loop, cond, body, carried, limit):
    if loop is not None:
        return loop(cond, body, carried)
    # This recurrence is captured inside a differentiable scan. The body
    # freezes settled members, so the fixed bound needs no scalar tensor read.
    for _ in range(limit):
        carried = body(*carried)
    return carried


def _newton_step_sizes(
    x, dx, residual, active, evaluate_residual, loop, damping, max_ls_steps, ls_decay
):
    """Choose the first improving trial per member, or the final decayed step."""
    initial_norm = residual.abs().amax(dim=-1)
    zero = 0 if loop is None else torch.zeros((), device=x.device, dtype=torch.int64)
    alpha = damping.clone()
    selected = torch.zeros_like(initial_norm)

    def cond(it, alpha, selected, pending):
        return (it < max_ls_steps) & pending.any()

    def body(it, alpha, selected, pending):
        trial_residual = evaluate_residual(x + alpha * dx)
        improvement = trial_residual.abs().amax(dim=-1) < initial_norm
        accept = pending & improvement
        return (
            it + 1,
            alpha * ls_decay,
            torch.where(accept, alpha, selected),
            pending & ~accept,
        )

    _, remaining_alpha, selected, pending = _iterate_newton(
        loop, cond, body, (zero, alpha, selected, active.clone()), max_ls_steps
    )
    return torch.where(pending, remaining_alpha, selected)[..., None]


def _solve_derivimplicit(
    x_t,
    dt_ms,
    f,
    t_ms,
    jac,
    jac_fn,
    *,
    tol,
    max_iter,
    damping,
    line_search,
    max_ls_steps,
    ls_decay,
    jacobian_regularization,
    create_graph,
    detach_newton,
    loop,
):
    *batch_shape, n = x_t.shape
    x_old = x_t.detach() if detach_newton else x_t
    x0 = x_old.clone()
    dt = _broadcast_dt(dt_ms, x_t)
    t_next = _end_time(t_ms, dt, x_t)

    dt_vec = dt[..., None] if dt.ndim else dt
    dt_mat = dt[..., None, None] if dt.ndim else dt
    if loop is not None and loop is not _eager_while_loop:
        # Higher-order control flow rejects aliased captured inputs.
        dt_vec = dt_vec.clone()
        dt_mat = dt_mat.clone()

    eye = torch.eye(n, device=x_t.device, dtype=x_t.dtype)
    affine_step = (
        jac is not None
        and not line_search
        and damping == 1.0
        and jacobian_regularization == 0.0
    )
    # Carry tensor scalars into structured control flow, including under
    # Dynamo's dynamic-shape tracing where Python floats can become SymFloats.
    damping_t = x_t.new_tensor(damping)
    decay_t = x_t.new_tensor(ls_decay) if line_search else None
    regularization = (
        x_t.new_tensor(jacobian_regularization) if jacobian_regularization else None
    )
    absolute_tol = x_t.new_tensor(_ABSOLUTE_TOLERANCE if tol is None else tol)
    relative_tol = (
        x_t.new_tensor(_ROUNDOFF_FACTOR * torch.finfo(x_t.dtype).eps)
        if tol is None
        else None
    )

    def evaluate_residual(x):
        return x - x_old - dt_vec * f(x, t_next)

    def evaluate(x, rel_tol):
        return _residual_status(x, x_old, dt_vec * f(x, t_next), absolute_tol, rel_tol)

    def residual_jacobian(x):
        if jac_fn is not None:
            Jf = jac_fn(x, t_next)
        elif x.ndim == 1:
            Jf = torch.autograd.functional.jacobian(
                lambda z: f(z, t_next), x, create_graph=create_graph
            )
        else:
            Jf = _autograd_batch_jacobian(f, x, t_next, create_graph=create_graph)
        return form_system(Jf)

    def form_system(Jf):
        J = eye - dt_mat * _broadcast_jac(Jf, batch_shape, n, like=x_t)
        if regularization is not None:
            J = J + regularization * eye
        return J

    J_const = form_system(jac) if jac is not None else None

    def update(x, r, status, J):
        active = status == _ITERATION_LIMIT
        # Even an already accepted affine root needs an invertible system.
        # Keep its state when the initial residual is within tolerance.
        check_system = active | (status == _CONVERGED) if affine_step else active
        J_eff = torch.where(check_system[..., None, None], J, eye)
        r_eff = torch.where(active[..., None], r, 0.0)
        dx, solved = _solve_linear_checked(J_eff, -r_eff)
        step = active & solved
        alpha = (
            _newton_step_sizes(
                x,
                dx,
                r,
                step,
                evaluate_residual,
                loop,
                damping_t,
                max_ls_steps,
                decay_t,
            )
            if line_search
            else damping_t
        )
        trial = x + alpha * dx
        r_trial, trial_status = evaluate(trial, relative_tol)
        stalled = (trial_status == _ITERATION_LIMIT) & (trial == x).all(dim=-1)
        trial_status = torch.where(stalled, _STAGNATED, trial_status)
        next_status = torch.where(step, trial_status, status)
        next_status = torch.where(check_system & ~solved, _LINEAR_FAILURE, next_status)
        take = step & (trial_status != _NONFINITE)
        x_next = torch.where(take[..., None], trial, x)
        r_next = torch.where(take[..., None], r_trial, r)
        return x_next, r_next, next_status

    # Roundoff tolerance applies to a computed step. Applying it to the
    # initial guess could erase a small but representable change every step.
    r0, status0 = evaluate(x0, None)
    if affine_step:
        x, _, status = update(x0, r0, status0, J_const)
        return x, status

    def cond(it, x, r, status):
        return (it < max_iter) & (status == _ITERATION_LIMIT).any()

    def body(it, x, r, status):
        J = J_const if J_const is not None else residual_jacobian(x)
        x_next, r_next, next_status = update(x, r, status, J)
        if detach_newton and loop is _eager_while_loop:
            x_next, r_next = x_next.detach(), r_next.detach()
        return it + 1, x_next, r_next, next_status

    it0 = 0 if loop is None else torch.zeros((), device=x_t.device, dtype=torch.int64)
    _, x, _, status = _iterate_newton(
        loop, cond, body, (it0, x0, r0, status0), max_iter
    )
    return x, status


_solve_derivimplicit_eager = _disable(_solve_derivimplicit)


def _attach_implicit_gradient(root, status, x_old, dt_ms, f, t_ms, jac_fn):
    """Attach the first-order backward-Euler sensitivity at the accepted root."""
    root = root.detach()
    dt = _broadcast_dt(dt_ms, root)
    t_next = _end_time(t_ms, dt, root)
    dt_vec = dt[..., None] if dt.ndim else dt
    n = root.shape[-1]
    eye = torch.eye(n, dtype=root.dtype, device=root.device)

    accepted = status == _CONVERGED
    with torch.no_grad():
        jac = _broadcast_jac(jac_fn(root, t_next), root.shape[:-1], n, like=root)
        dt_mat = dt[..., None, None] if dt.ndim else dt
        # Differentiate the equation itself. Damping and regularization in
        # the forward iteration only determine how it finds the root.
        matrix = eye - dt_mat * jac
        matrix = torch.where(accepted[..., None, None], matrix, eye)

    source = x_old - root + dt_vec * f(root, t_next)
    source = torch.where(accepted[..., None], source, 0.0)
    # The primal RHS is zero, but its derivative is the negative parameter
    # derivative of the residual at the fixed root. Autograd differentiates
    # the linear solve by applying the transposed residual Jacobian inverse.
    correction, solved = _solve_linear_checked(
        matrix.detach(), source - source.detach()
    )
    status = torch.where(accepted & ~solved, _LINEAR_FAILURE, status)
    return root + correction, status


_CONVERGENCE_ERROR = (
    "derivimplicit failed to converge: a batch member hit a singular or "
    "nonfinite solve, stagnation, or the iteration limit"
)


@torch.library.custom_op(
    "dendra_derivimplicit::require_converged",
    mutates_args=(),
    tags=torch.Tag.cudagraph_unsafe,
)
def _require_converged(accepted: torch.Tensor) -> torch.Tensor:
    # A CUDA device assertion invalidates the context. Keep this host check
    # capturable as an operator for accelerator rollouts.
    if not accepted.item():
        raise RuntimeError(_CONVERGENCE_ERROR)
    return accepted.clone()


@_require_converged.register_fake
def _(accepted):
    return torch.empty_like(accepted)


def _all_solver_members(accepted):
    if (
        not torch.compiler.is_compiling()
        and torch._C._are_functorch_transforms_active()
    ):
        functorch = torch._C._functorch
        while functorch.is_functorch_wrapped_tensor(accepted):
            accepted = functorch.get_unwrapped(accepted)
    return accepted.all()


def _check_step(accepted, *states, previous, solver):
    if not torch.compiler.is_compiling() and _record_solver_check(accepted, solver):
        # A rejected candidate may be nonfinite. Keep the prior values in
        # failed members while the remaining tensor steps finish the chunk.
        return tuple(
            torch.where(
                accepted.reshape(accepted.shape + (1,) * (state.ndim - accepted.ndim)),
                state,
                old,
            )
            for state, old in zip(states, previous, strict=True)
        )
    accepted = _all_solver_members(accepted)
    if accepted.device.type == "cpu":
        torch._assert_async(accepted, _CONVERGENCE_ERROR)
        return states
    # Make the step's result depend on the check so unused diagnostics cannot
    # be pruned from a compiled graph.
    checked = _require_converged(accepted)
    return tuple(torch.where(checked, state, 0.0) for state in states)


def _check_convergence(x, x_old, status, allow_inexact, solver):
    accepted = status == _CONVERGED
    if allow_inexact:
        accepted = accepted | (status == _ITERATION_LIMIT) | (status == _STAGNATED)
    return _check_step(accepted, x, previous=(x_old,), solver=solver)[0]


def _guard_eager_status_backward(x, status):
    # Masking a lane's output cannot prevent 0 * inf in the backward of its
    # authored RHS or Jacobian, particularly for shared parameters.
    valid = _all_solver_members((status != _NONFINITE) & (status != _LINEAR_FAILURE))

    def check(gradient):
        if not valid.item():
            raise RuntimeError(
                "derivimplicit backward requires finite residuals and successful "
                "linear solves in every lane when return_status=True"
            )
        return gradient

    x.register_hook(check)


def derivimplicit_step(
    x_t: torch.Tensor,
    dt_ms: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    t_ms: torch.Tensor | None = None,
    jac: torch.Tensor | None = None,
    jac_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
    *,
    tol: float | None = None,
    max_iter: int = 25,
    damping: float = 1.0,
    line_search: bool = False,
    max_ls_steps: int = 10,
    ls_decay: float = 0.5,
    jacobian_regularization: float = 0.0,
    create_graph: bool = False,
    detach_newton: bool = True,
    allow_inexact: bool = False,
    return_status: bool = False,
    _analytic_jacobian: bool = True,
    _solver=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Solve one backward-Euler step for independent state vectors.

    With ``tol=None``, use an absolute tolerance of 1e-8 at the initial guess
    and add a dtype- and scale-dependent allowance after each correction. An
    explicit ``tol`` is absolute throughout. An iteration budget of zero
    returns the incoming state as a successful no-op.

    Failed steps raise unless ``return_status=True``, which returns the last
    retained state and an integer ``DerivimplicitStatus`` per batch member.
    Eager reverse-mode differentiation of a status-returning step raises if
    any lane reports a nonfinite residual or a failed linear solve. Retry on
    valid inputs to differentiate that step.
    ``allow_inexact=True`` permits finite iterates on stagnation or exhaustion.
    Population scan captures with an analytic Jacobian and strict convergence
    use first-order implicit gradients at the accepted root.
    """
    if x_t.ndim < 1 or x_t.shape[-1] == 0:
        raise ValueError("x_t must have at least 1 dimension (nonempty state dim).")
    if jac is not None and jac_fn is not None:
        raise ValueError(
            "Provide at most one of `jac` or `jac_fn` (or neither to use autograd Jacobian)."
        )
    if max_iter < 0:
        raise ValueError("derivimplicit max_iter must be non-negative.")

    if max_iter == 0:
        x = x_t.detach().clone() if detach_newton else x_t.clone()
        if return_status:
            return x, x.new_zeros(x.shape[:-1], dtype=torch.int64)
        return x

    options = {
        "tol": tol,
        "max_iter": max_iter,
        "damping": damping,
        "line_search": line_search,
        "max_ls_steps": max_ls_steps,
        "ls_decay": ls_decay,
        "jacobian_regularization": jacobian_regularization,
    }
    compiling = torch.compiler.is_compiling()
    capturing = not compiling and _capturing_solver_checks()
    grad_enabled = torch.is_grad_enabled()
    captured_nonlinear = (
        capturing and grad_enabled and jac is None and not detach_newton
    )
    if captured_nonlinear:
        if jac_fn is None or not _analytic_jacobian or allow_inexact:
            raise RuntimeError(
                "Captured nonlinear derivimplicit training requires an analytic "
                "Jacobian and strict convergence (allow_inexact=False)"
            )
        with torch.no_grad():
            x, status = _solve_derivimplicit(
                x_t,
                dt_ms,
                f,
                t_ms,
                jac,
                jac_fn,
                **options,
                create_graph=False,
                detach_newton=True,
                loop=None,
            )
        x, status = _attach_implicit_gradient(x, status, x_t, dt_ms, f, t_ms, jac_fn)
    else:
        if (
            capturing
            and not grad_enabled
            and (jac is not None or (jac_fn is not None and _analytic_jacobian))
        ):
            # A training-mode State may still execute an inference chunk.
            # Its analytic primal needs structured control flow in the trace.
            create_graph = False
            detach_newton = True
        eager = create_graph or not detach_newton or (jac is None and jac_fn is None)
        solve = _solve_derivimplicit_eager if eager else _solve_derivimplicit
        loop = _eager_while_loop if eager else _get_while_loop()
        x, status = solve(
            x_t,
            dt_ms,
            f,
            t_ms,
            jac,
            jac_fn,
            **options,
            create_graph=create_graph,
            detach_newton=detach_newton,
            loop=loop,
        )
    if return_status:
        if not compiling and not capturing and x.requires_grad:
            _guard_eager_status_backward(x, status)
        return x, status
    return _check_convergence(x, x_t, status, allow_inexact, _solver)


derivimplicit_template = """
def solve(self, dt, {states_and_assigned}, **kwargs):
    {locals}
    {concatenate}
    {f_str}
    {jac_fn}
    {empty_jacobian}
    {assemble_jacobian}
    {solve_system}
    {unbind}
    {eliminate}
    return {returns}
"""

f_template = """
    def f(x, t):
        {unbinded_states}
        {derivatives}
        return torch.stack([{states}], dim=-1)
"""

jac_fn_template = """
    def jac_fn(x, t):
        J = x.new_zeros((*x.shape, x.shape[-1]))
        {unbinded_states}
        {assemble_jacobian}
        return J
"""


concatenate_string = "x = torch.stack([{states}], dim=-1)"
empty_jacobian_template = "jac = x.new_zeros((*x.shape, x.shape[-1]))"
unbind_template = "{states} = torch.unbind(x, dim=-1)"


def _solve_system_jac_call(options):
    return (
        "x = derivimplicit_step(x, dt, f, jac=jac, "
        + _derivimplicit_option_call_args(options)
        + ", _solver=self)"
    )


def _solve_system_jac_fn_call(options, *, analytic):
    return (
        "x = derivimplicit_step(x, dt, f, jac_fn=jac_fn, "
        + _derivimplicit_option_call_args(options)
        + f", _analytic_jacobian={analytic!r}, _solver=self)"
    )


def _solve_system_autograd_call(options):
    return (
        "x = derivimplicit_step(x, dt, f, "
        + _derivimplicit_option_call_args(options)
        + ", _solver=self)"
    )


def _affine_residual_accepted(x, x_old, scaled_initial_rhs, scaled_rhs):
    """Check the authored equation, including its domain at the incoming state."""
    residual = (x - x_old - scaled_rhs).abs()
    bound = _residual_bound(
        x,
        x_old,
        scaled_rhs,
        _ABSOLUTE_TOLERANCE,
        _ROUNDOFF_FACTOR * torch.finfo(x_old.dtype).eps,
    )
    return (
        (residual <= bound)
        & (residual < math.inf)
        & (scaled_initial_rhs.abs() < math.inf)
    )


def _solve_affine_diagonal(x, pivot, scaled_initial_rhs, scaled_bias):
    # Solve for the state directly to avoid cancellation on stiff decay.
    # Solve for an increment when the pivot rounds to one or when assembling
    # the direct RHS would overflow even though the correction may be finite.
    use_increment = pivot == 1.0
    numerator = x
    if scaled_bias is not None:
        numerator = x + scaled_bias
        use_increment = use_increment | torch.isinf(numerator)
    rhs = torch.where(use_increment, scaled_initial_rhs, numerator)
    solution = rhs / pivot
    return torch.where(use_increment, x + solution, solution).to(x)


def _render_checked_affine_body(analysis, rhs, eliminate):
    states = analysis["states_to_solve"]
    matrix, bias = analysis["A"], analysis["b"]
    state_args = ", ".join(states)
    updated = ", ".join(f"_{state}" for state in states)
    n = len(states)
    rhs_tuple = ", ".join(f"({expr})" for expr in rhs)
    lines = [
        "if dt.ndim:",
        "    dt = dt.expand(__ref.shape)",
        f"def __rhs({state_args}):",
    ]
    lines.extend(f"    {assignment}" for assignment in eliminate.values())
    lines.append(f"    return ({rhs_tuple},)")
    numerators = [
        s if b == "0" else f"({s} + dt * ({b}))"
        for s, b in zip(states, bias, strict=True)
    ]
    diagonal = all(
        entry == "0"
        for i, row in enumerate(matrix)
        for j, entry in enumerate(row)
        if i != j
    )
    if diagonal:
        lines.append(f"__initial_rhs = __rhs({state_args})")
        for i, s in enumerate(states):
            scaled_bias = "None" if bias[i] == "0" else f"dt * ({bias[i]})"
            lines.extend(
                [
                    f"__scaled_initial_{i} = dt * __initial_rhs[{i}]",
                    f"__pivot_{i} = 1.0 - dt * ({matrix[i][i]})",
                    f"_{s} = _solve_affine_diagonal({s}, __pivot_{i}, __scaled_initial_{i}, {scaled_bias})",
                ]
            )
        checks = [f"(__pivot_{i}.abs() < torch.inf)" for i in range(n)]
    else:
        lines.extend(
            [
                f"__x = torch.stack([{state_args}], dim=-1)",
                "__A = __x.new_zeros((*__x.shape, __x.shape[-1]))",
            ]
        )
        lines.extend(
            f"__A[..., {i}, {j}] = ({entry})"
            for i, row in enumerate(matrix)
            for j, entry in enumerate(row)
            if entry != "0"
        )
        lines.extend(
            [
                f"__b = torch.stack([{', '.join(f'{v}.to(__ref)' for v in numerators)}], dim=-1)",
                "__system = torch.eye(__x.shape[-1], dtype=__x.dtype, device=__x.device) - dt[..., None, None] * __A",
                "__next, __solved = _solve_linear_checked(__system, __b)",
                f"{updated} = torch.unbind(__next, dim=-1)",
            ]
        )
        checks = ["__solved"]

    lines.append("with torch.no_grad():")
    if not diagonal:
        lines.append(f"    __initial_rhs = __rhs({state_args})")
        lines.extend(
            f"    __scaled_initial_{i} = dt * __initial_rhs[{i}]" for i in range(n)
        )
    lines.append(f"    __final_rhs = __rhs({updated})")
    checks.extend(
        f"_affine_residual_accepted(_{s}, {s}, __scaled_initial_{i}, dt * __final_rhs[{i}])"
        for i, s in enumerate(states)
    )
    lines.append(f"    __accepted = {' & '.join(checks)}")
    targets = updated + ("," if n == 1 else "")
    lines.append(
        f"{targets} = _check_step(__accepted, {updated}, previous=({state_args},), solver=self)"
    )
    return "\n    ".join(lines)


def _try_build_checked_affine(states, assigned, derivative, eliminate):
    try:
        analysis = _analyze_affine_system(states, assigned, derivative, eliminate)
    except _AffineAnalysisError:
        return None
    by_state = match_derivative_to_states(derivative, states)
    rhs = [
        modify_operations(_nmodl_preprocess(_rhs_string(by_state[s])))
        for s in analysis["states_to_solve"]
    ]
    ops = all_ops()
    if not all(_validate_torch_expression(expr, allowed_callables=ops) for expr in rhs):
        return None
    body = _render_checked_affine_body(analysis, rhs, eliminate)
    return _compile_affine_solver(
        states,
        assigned,
        eliminate,
        analysis,
        body,
        filename_prefix="dendra.derivimplicit.affine.solve",
        extra_ns={
            "_affine_residual_accepted": _affine_residual_accepted,
            "_check_step": _check_step,
            "_solve_affine_diagonal": _solve_affine_diagonal,
            "_solve_linear_checked": _solve_linear_checked,
        },
    )


def build_derivimplicit(
    states, assigned, derivative, eliminate=None, pade=False, **method_kwargs
):
    options = _normalize_derivimplicit_options(method_kwargs)

    # ---- validation ----
    for state in states:
        if state in assigned:
            raise ValueError(
                f"State {state} cannot be assigned and used as a state variable."
            )

    eliminate = _normalize_eliminate(eliminate)

    # Solve only non-eliminated states, preserving input order
    states_to_solve = [s for s in states if s not in eliminate]
    if not states_to_solve:
        raise ValueError("derivimplicit requires at least one non-eliminated state.")

    if not pade and options == _normalize_derivimplicit_options({}):
        derivative = tuple(derivative)
        affine = _try_build_checked_affine(states, assigned, derivative, eliminate)
        if affine is not None:
            return affine

    # Deterministic signature: states first (in given order), then assigned (sorted)
    assigned_list = list(assigned)
    assigned_list_sorted = sorted(assigned_list)
    states_and_assigned = ", ".join(
        _unique_preserve_order(list(states) + assigned_list_sorted)
    )

    # ---- build stack/unbind strings ----
    concatenate = concatenate_string.format(states=", ".join(states_to_solve))

    unbind_targets = ", ".join(states_to_solve)
    if len(states_to_solve) == 1:
        unbind_targets += ","

    # Only the assignment targets are updated-state names.  Applying the old
    # whole-expression substitution also renamed the RHS stack variable when a
    # mechanism state itself was named ``x`` (``torch.unbind(_x, ...)`` before
    # ``_x`` existed).
    updated_targets = ", ".join(f"_{state}" for state in states_to_solve)
    if len(states_to_solve) == 1:
        updated_targets += ","
    unbind = f"{updated_targets} = torch.unbind(x, dim=-1)"

    # ---- canonicalize derivative mapping ----
    derivative = match_derivative_to_states(derivative, states)
    missing = [s for s in states_to_solve if s not in derivative]
    if missing:
        raise ValueError(
            f"derivimplicit missing derivative(s) for state(s): {missing}."
        )

    # ---- build f(x,t) ----
    derivatives = [
        f"_{s}_deriv = {derivative[s].split('=')[1].strip()}" for s in states_to_solve
    ]
    derivatives_str = "\n        ".join(derivatives)

    f_str = f_template.format(
        unbinded_states=unbind_template.format(states=unbind_targets),
        derivatives=derivatives_str,
        states=", ".join([f"_{s}_deriv" for s in states_to_solve]),
    )

    # ---- build variable set for differentiation / locals ----
    vars_set = set(states_to_solve)
    vars_set.update(assigned_list_sorted)

    for s in states_to_solve:
        rhs = derivative[s]
        vars_set.update(extract_vars(rhs, vars_set))

    # locals = vars that are referenced but not passed in or states
    local_vars = sorted(vars_set - set(states_to_solve) - set(assigned_list_sorted))
    locals_str = "\n    ".join([f"{v} = self.{v}" for v in local_vars])

    # vars list passed to differentiator: deterministic order
    vars_list = _unique_preserve_order(
        states_to_solve + assigned_list_sorted + local_vars
    )

    # ---- differentiate to build jacobian expressions ----
    rhs_derivatives = {}
    jac_ok = True
    jac_analytic = True
    jac_state_independent = True

    # Small cache to avoid repeated sympy work inside a single build call
    diff_cache = {}

    for s in states_to_solve:
        rhs = derivative[s]
        for swrt in states_to_solve:
            key = (rhs, swrt, tuple(states_to_solve), tuple(vars_list))
            if key not in diff_cache:
                df, ok, depends = differentiate_rhs_2torch_checked(
                    rhs,
                    vars_list,
                    swrt,
                    state_vars=states_to_solve,
                    allow_finite_difference=False,
                )
                analytic = ok
                if not analytic:
                    df, ok, depends = differentiate_rhs_2torch_checked(
                        rhs, vars_list, swrt, state_vars=states_to_solve
                    )
                diff_cache[key] = (df, ok, depends, analytic)
            df, ok, depends, analytic = diff_cache[key]

            rhs_derivatives[(s, swrt)] = df
            jac_ok = jac_ok and ok
            jac_analytic = jac_analytic and analytic
            jac_state_independent = jac_state_independent and (not depends)

            if not jac_ok:
                break
        if not jac_ok:
            break

    # ---- emit jac / jac_fn / solve call ----
    if jac_ok:
        if jac_state_independent:
            # Constant-in-state Jacobian tensor built in solve()
            jac_fn_str = "jac_fn = None"
            jac_init_str = empty_jacobian_template  # now x.new_zeros(...)
            assemble_lines = []
            for i, s in enumerate(states_to_solve):
                for j, swrt in enumerate(states_to_solve):
                    assemble_lines.append(
                        f"jac[..., {i}, {j}] = ({rhs_derivatives[(s, swrt)]})"
                    )
            assemble_jacobian_str = "\n    ".join(assemble_lines)
            solve_call = _solve_system_jac_call(options)
        else:
            # State-dependent Jacobian: jac_fn(x,t)
            jac_fn_str = jac_fn_template.format(
                unbinded_states=unbind_template.format(states=unbind_targets),
                assemble_jacobian="\n        ".join(
                    [
                        f"J[..., {i}, {j}] = ({rhs_derivatives[(s, swrt)]})"
                        for i, s in enumerate(states_to_solve)
                        for j, swrt in enumerate(states_to_solve)
                    ]
                ),
            )
            jac_init_str = "jac = None"
            assemble_jacobian_str = ""
            solve_call = _solve_system_jac_fn_call(options, analytic=jac_analytic)
    else:
        # Fallback: no analytic Jacobian available; let solver use autograd Jacobian.
        jac_fn_str = "jac_fn = None"
        jac_init_str = "jac = None"
        assemble_jacobian_str = ""
        solve_call = _solve_system_autograd_call(options)

    # ---- elimination and returns ----
    eliminate_solves = []
    returns = []

    for s in states:
        if s in eliminate:
            # Ensure eliminated state is assigned to _state
            eliminate_solves.append(add_underscore_to_states(eliminate[s], states))
        returns.append(f"'{s}' : _{s}")

    eliminate_str = "\n    ".join(eliminate_solves)
    returns_str = "{" + ", ".join(returns) + "}"

    # ---- render solve() source ----
    solve_src = derivimplicit_template.format(
        locals=locals_str,
        states_and_assigned=states_and_assigned,
        f_str=f_str,
        concatenate=concatenate,
        jac_fn=jac_fn_str,
        empty_jacobian=jac_init_str,
        assemble_jacobian=assemble_jacobian_str,
        solve_system=solve_call,
        unbind=unbind,
        eliminate=eliminate_str,
        returns=returns_str,
    )

    if DEBUG:
        logger.debug(f"Function:\n{solve_src}")

    return compile_generated_function(
        solve_src,
        func_name="solve",
        filename_prefix="dendra.derivimplicit.solve",
        global_ns=globals(),
    )
