from typing import Callable, Optional

import torch

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._solve_utils import (
    add_underscore_to_states,
    extract_vars,
    match_derivative_to_states,
)
from ._solvers import _solve_linear_small
from .ode import differentiate_rhs_2torch_checked

# Optional: faster Jacobian via torch.func if available
try:
    from torch.func import jacrev, vmap

    _HAS_TORCH_FUNC = True
except Exception:
    _HAS_TORCH_FUNC = False


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


def _broadcast_dt(dt_ms: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast dt to the batch shape of x (excluding last state dimension)."""
    dt = torch.as_tensor(dt_ms, dtype=x.dtype, device=x.device)
    if dt.ndim == 0:
        return dt
    return dt.expand(x.shape[:-1])


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

# core machinery


@_disable
def _derivimplicit_step_eager(
    x_t: torch.Tensor,
    dt_ms: torch.Tensor,
    f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    t_ms: Optional[torch.Tensor],
    jac: Optional[torch.Tensor],
    jac_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]],
    *,
    tol: float,
    max_iter: int,
    damping: float,
    line_search: bool,
    max_ls_steps: int,
    ls_decay: float,
    jacobian_regularization: float,
    create_graph: bool,
    detach_newton: bool,
) -> torch.Tensor:
    if x_t.ndim < 1:
        raise ValueError("x_t must have at least 1 dimension (state dim).")

    if (jac is not None) and (jac_fn is not None):
        raise ValueError(
            "Provide at most one of `jac` or `jac_fn` (or neither to use autograd Jacobian)."
        )

    *batch_shape, n = x_t.shape
    dt = _broadcast_dt(dt_ms, x_t)

    t_next = None
    if t_ms is not None:
        t_ms_t = torch.as_tensor(t_ms, dtype=x_t.dtype, device=x_t.device)
        if t_ms_t.ndim == 0:
            t_next = t_ms_t + dt
        else:
            t_next = t_ms_t.expand(batch_shape) + dt

    eye = torch.eye(n, device=x_t.device, dtype=x_t.dtype).expand((*batch_shape, n, n))
    dt_vec = dt[..., None] if dt.ndim > 0 else dt
    dt_mat = dt[..., None, None] if dt.ndim > 0 else dt

    def eval_f(x: torch.Tensor) -> torch.Tensor:
        return f(x, t_next)

    def residual(x: torch.Tensor) -> torch.Tensor:
        # Do NOT detach x_t here: in the backprop mode, you may want grads wrt x_t/params.
        return x - x_t - dt_vec * eval_f(x)

    def inf_norm_global(v: torch.Tensor) -> torch.Tensor:
        # Match your original behavior: global max over all state-vectors in the batch.
        # Equivalent to torch.amax(torch.abs(v)) for your usage.
        return torch.amax(torch.abs(v))

    # Initial guess
    x = x_t.clone()

    if jac is not None and (not line_search):
        # One Newton step is exact for affine RHS
        r = residual(x)
        Jf0 = _broadcast_jac(jac, batch_shape, n, like=x_t)
        J = eye - dt_mat * Jf0
        if jacobian_regularization != 0.0:
            J = J + jacobian_regularization * eye
        dx = _solve_linear_small(J, -r)
        return x + damping * dx

    for _ in range(max_iter):
        # Detach strategy:
        # - detach_newton=True  -> inference-style: keep Newton from building a big graph
        # - detach_newton=False -> differentiable unrolled Newton
        if detach_newton:
            x_var = x.detach()
            # only need requires_grad for autograd Jacobian
            if (jac is None) and (jac_fn is None):
                x_var.requires_grad_(True)
        else:
            x_var = x  # keep graph

        r = residual(x_var)
        r0 = inf_norm_global(r)

        if r0.detach().item() < tol:
            return x_var.detach() if detach_newton else x_var

        # Build df/dx (Jacobian of f)
        if jac is not None:
            Jf = _broadcast_jac(jac, batch_shape, n, like=x_t)
        elif jac_fn is not None:
            Jf = _broadcast_jac(jac_fn(x_var, t_next), batch_shape, n, like=x_t)
        else:
            # Autograd Jacobian (heavy; prefer analytic jac/jac_fn in practice)
            if x_var.ndim == 1:
                Jf = torch.autograd.functional.jacobian(
                    lambda z: f(z, t_next),
                    x_var,
                    create_graph=create_graph,
                )
            else:
                # Batched jacobian via torch.func if available; fallback to loop
                if _HAS_TORCH_FUNC:
                    flat = x_var.reshape(-1, n)

                    def single_f(z):
                        return f(z, t_next)

                    Jflat = vmap(jacrev(single_f))(flat)  # (B, n, n)
                    Jf = Jflat.reshape(*batch_shape, n, n)
                else:
                    flat = x_var.reshape(-1, n)
                    Js = []
                    for i in range(flat.shape[0]):
                        Ji = torch.autograd.functional.jacobian(
                            lambda z: f(z, t_next),
                            flat[i],
                            create_graph=create_graph,
                        )
                        Js.append(Ji)
                    Jf = torch.stack(Js, dim=0).reshape(*batch_shape, n, n)

        # Residual Jacobian: J = I - dt * df/dx
        J = eye - dt_mat * Jf
        if jacobian_regularization != 0.0:
            J = J + jacobian_regularization * eye

        dx = _solve_linear_small(J, -r)

        if line_search:
            # NOTE: line search introduces non-smooth branching in the forward.
            # This is usually OK in practice, but if you want fully smooth grads,
            # consider disabling line_search in training mode.
            alpha = damping
            accepted = False
            for _ls in range(max_ls_steps):
                x_trial = x_var + alpha * dx
                r_trial = residual(x_trial)
                if inf_norm_global(r_trial).detach().item() < r0.detach().item():
                    x = x_trial.detach() if detach_newton else x_trial
                    accepted = True
                    break
                alpha *= ls_decay

            if not accepted:
                x = (
                    (x_var + alpha * dx).detach()
                    if detach_newton
                    else (x_var + alpha * dx)
                )
        else:
            x = (
                (x_var + damping * dx).detach()
                if detach_newton
                else (x_var + damping * dx)
            )

    return x


def _derivimplicit_step_while_loop(
    x_t: torch.Tensor,
    dt_ms: torch.Tensor,
    f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    t_ms: Optional[torch.Tensor],
    jac: Optional[torch.Tensor],
    jac_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]],
    *,
    tol: float,
    max_iter: int,
    damping: float,
    line_search: bool,
    max_ls_steps: int,
    ls_decay: float,
    jacobian_regularization: float,
) -> torch.Tensor:
    if x_t.ndim < 1:
        raise ValueError("x_t must have at least 1 dimension (state dim).")

    if (jac is None) == (jac_fn is None):
        raise ValueError(
            "Provide exactly one of `jac` or `jac_fn` for while_loop path."
        )

    *batch_shape, n = x_t.shape
    device, dtype = x_t.device, x_t.dtype

    dt = _broadcast_dt(dt_ms, x_t)

    t_next = None
    if t_ms is not None:
        t_ms_t = torch.as_tensor(t_ms, dtype=dtype, device=device)
        if t_ms_t.ndim == 0:
            t_next = t_ms_t + dt
        else:
            t_next = t_ms_t.expand(batch_shape) + dt

    eye = torch.eye(n, device=device, dtype=dtype).expand((*batch_shape, n, n))
    dt_vec = dt[..., None] if dt.ndim > 0 else dt
    dt_mat = dt[..., None, None] if dt.ndim > 0 else dt

    tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
    max_iter_t = torch.as_tensor(max_iter, device=device, dtype=torch.int64)
    damping_t = torch.as_tensor(damping, device=device, dtype=dtype)
    ls_decay_t = torch.as_tensor(ls_decay, device=device, dtype=dtype)
    max_ls_steps_t = torch.as_tensor(max_ls_steps, device=device, dtype=torch.int64)

    # Clone to avoid aliasing restrictions in higher-order op tracing.
    # (HOPs are strict about “outputs cannot alias inputs”).
    x0 = x_t.detach().clone()

    def eval_f(x: torch.Tensor) -> torch.Tensor:
        return f(x, t_next)

    def residual(x: torch.Tensor) -> torch.Tensor:
        return x - x_t.detach() - dt_vec * eval_f(x)

    if jac is not None and (not line_search):
        # One Newton step is exact for affine RHS
        r = residual(x0)
        Jf0 = _broadcast_jac(jac, batch_shape, n, like=x_t)
        J = eye - dt_mat * Jf0
        if jacobian_regularization != 0.0:
            J = J + jacobian_regularization * eye
        dx = _solve_linear_small(J, -r)
        return x0 + damping_t * dx

    # If constant jac, precompute J outside the loop
    use_const_jac = jac is not None
    if use_const_jac:
        Jf0 = _broadcast_jac(jac, batch_shape, n, like=x_t)
        J_const = eye - dt_mat * Jf0
        if jacobian_regularization != 0.0:
            J_const = (
                J_const
                + torch.as_tensor(jacobian_regularization, device=device, dtype=dtype)
                * eye
            )

    r0 = residual(x0)
    r0_per = r0.abs().amax(dim=-1)
    r0_g = torch.amax(r0_per)
    it0 = torch.zeros((), device=device, dtype=torch.int64)

    while_loop = _get_while_loop()

    def cond_fn(it, x, r, r_g, r_per):
        return (it < max_iter_t) & (r_g >= tol_t)

    def body_fn(it, x, r, r_g, r_per):
        active = r_per >= tol_t

        if use_const_jac:
            J = J_const
        else:
            Jf = _broadcast_jac(jac_fn(x, t_next), batch_shape, n, like=x_t)
            J = eye - dt_mat * Jf
            if jacobian_regularization != 0.0:
                J = (
                    J
                    + torch.as_tensor(
                        jacobian_regularization, device=device, dtype=dtype
                    )
                    * eye
                )

        r_eff = torch.where(active[..., None], r, torch.zeros_like(r))
        J_eff = torch.where(active[..., None, None], J, eye)

        dx = _solve_linear_small(J_eff, -r_eff)

        if not line_search:
            x_new = torch.where(active[..., None], x + damping_t * dx, x)
        else:
            # Structured backtracking (first alpha that improves global residual)
            ls_it0 = torch.zeros((), device=device, dtype=torch.int64)
            alpha0 = damping_t
            accepted0 = torch.zeros((), device=device, dtype=torch.bool)
            alpha_acc0 = torch.zeros((), device=device, dtype=dtype)

            def ls_cond(ls_it, alpha, accepted, alpha_acc):
                return (ls_it < max_ls_steps_t) & (~accepted)

            def ls_body(ls_it, alpha, accepted, alpha_acc):
                x_trial = x + alpha * dx
                r_trial = residual(x_trial)
                r_trial_g = torch.amax(r_trial.abs())

                accept = r_trial_g < r_g
                take = accept & (~accepted)

                alpha_acc = torch.where(take, alpha, alpha_acc)
                accepted = accepted | accept
                alpha = torch.where(accept, alpha, alpha * ls_decay_t)
                return ls_it + 1, alpha, accepted, alpha_acc

            ls_it_f, alpha_f, accepted_f, alpha_acc_f = while_loop(
                ls_cond, ls_body, (ls_it0, alpha0, accepted0, alpha_acc0)
            )

            alpha_sel = torch.where(accepted_f, alpha_acc_f, alpha_f)
            x_step = x + alpha_sel * dx
            x_new = torch.where(active[..., None], x_step, x)

        r_new = residual(x_new)
        r_new_per = r_new.abs().amax(dim=-1)
        r_new_g = torch.amax(r_new_per)
        return it + 1, x_new, r_new, r_new_g, r_new_per

    it_f, x_f, r_f, r_g_f, r_per_f = while_loop(
        cond_fn, body_fn, (it0, x0, r0, r0_g, r0_per)
    )
    return x_f


def derivimplicit_step(
    x_t: torch.Tensor,
    dt_ms: torch.Tensor,
    f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    t_ms: Optional[torch.Tensor] = None,
    jac: Optional[torch.Tensor] = None,
    jac_fn: Optional[
        Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    ] = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 25,
    damping: float = 1.0,
    line_search: bool = False,
    max_ls_steps: int = 10,
    ls_decay: float = 0.5,
    jacobian_regularization: float = 0.0,
    create_graph: bool = False,
    detach_newton: bool = True,
) -> torch.Tensor:
    """
    - Inference path (compile-friendly): while_loop if detach_newton=True, create_graph=False, and (jac xor jac_fn) provided.
    - Training/backprop path: eager Newton in torch._dynamo.disable region (supports autograd).
    """
    # Any request for grads through the solver should avoid while_loop/cond (no training support).
    want_backprop = create_graph or (not detach_newton)

    # Also: while_loop path requires an explicit Jacobian source
    has_explicit_jac = (jac is not None) ^ (jac_fn is not None)

    if want_backprop or (not has_explicit_jac):
        return _derivimplicit_step_eager(
            x_t,
            dt_ms,
            f,
            t_ms,
            jac,
            jac_fn,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
            line_search=line_search,
            max_ls_steps=max_ls_steps,
            ls_decay=ls_decay,
            jacobian_regularization=jacobian_regularization,
            create_graph=create_graph,
            detach_newton=detach_newton,
        )

    return _derivimplicit_step_while_loop(
        x_t,
        dt_ms,
        f,
        t_ms,
        jac,
        jac_fn,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
        line_search=line_search,
        max_ls_steps=max_ls_steps,
        ls_decay=ls_decay,
        jacobian_regularization=jacobian_regularization,
    )


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
        {unbinded_states}
        J = x.new_zeros((*x.shape, x.shape[-1]))
        {assemble_jacobian}
        return J
"""


concatenate_string = "x = torch.stack([{states}], dim=-1)"
empty_jacobian_template = "jac = x.new_zeros((*x.shape, x.shape[-1]))"
unbind_template = "{states} = torch.unbind(x, dim=-1)"

solve_system_jac_template = (
    "x = derivimplicit_step(x, dt, f, jac=jac, "
    "detach_newton=(not self.training), create_graph=self.training)"
)

solve_system_jac_fn_template = (
    "x = derivimplicit_step(x, dt, f, jac_fn=jac_fn, "
    "detach_newton=(not self.training), create_graph=self.training)"
)

solve_system_autograd_template = (
    "x = derivimplicit_step(x, dt, f, "
    "detach_newton=(not self.training), create_graph=self.training)"
)


def build_derivimplicit(states, assigned, derivative, eliminate=None, pade=False):
    # ---- validation ----
    for state in states:
        if state in assigned:
            raise ValueError(
                f"State {state} cannot be assigned and used as a state variable."
            )

    eliminate = _normalize_eliminate(eliminate)

    # Solve only non-eliminated states, preserving input order
    states_to_solve = [s for s in states if s not in eliminate]

    # Deterministic signature: states first (in given order), then assigned (sorted)
    assigned_list = list(assigned)
    assigned_list_sorted = sorted(assigned_list)
    states_and_assigned = ", ".join(
        _unique_preserve_order(list(states) + assigned_list_sorted)
    )

    # ---- build stack/unbind strings ----
    concatenate = concatenate_string.format(states=", ".join(states_to_solve))

    unbind = unbind_template.format(states=", ".join(states_to_solve))
    unbind = add_underscore_to_states(unbind, states_to_solve)

    # ---- canonicalize derivative mapping ----
    derivative = match_derivative_to_states(derivative, states)

    # ---- build f(x,t) ----
    derivatives = [
        f"_{s}_deriv = {derivative[s].split('=')[1].strip()}" for s in states_to_solve
    ]
    derivatives_str = "\n        ".join(derivatives)

    f_str = f_template.format(
        unbinded_states=unbind_template.format(states=", ".join(states_to_solve)),
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
    jac_state_independent = True

    # Small cache to avoid repeated sympy work inside a single build call
    diff_cache = {}

    for s in states_to_solve:
        rhs = derivative[s]
        for swrt in states_to_solve:
            key = (rhs, swrt, tuple(states_to_solve), tuple(vars_list))
            if key in diff_cache:
                df, ok, depends = diff_cache[key]
            else:
                df, ok, depends = differentiate_rhs_2torch_checked(
                    rhs,
                    vars_list,
                    swrt,
                    state_vars=states_to_solve,
                )
                diff_cache[key] = (df, ok, depends)

            rhs_derivatives[(s, swrt)] = df
            jac_ok = jac_ok and ok
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
            solve_call = solve_system_jac_template
        else:
            # State-dependent Jacobian: jac_fn(x,t)
            jac_fn_str = jac_fn_template.format(
                unbinded_states=unbind_template.format(
                    states=", ".join(states_to_solve)
                ),
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
            solve_call = solve_system_jac_fn_template
    else:
        # Fallback: no analytic Jacobian available; let solver use autograd Jacobian.
        jac_fn_str = "jac_fn = None"
        jac_init_str = "jac = None"
        assemble_jacobian_str = ""
        solve_call = solve_system_autograd_template

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
