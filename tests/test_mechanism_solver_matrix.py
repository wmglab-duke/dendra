from types import SimpleNamespace

import pytest
import sympy as sp
import torch

from dendra.models.mechanisms._bufferimplicit import build_bufferimplicit
from dendra.models.mechanisms._derivimplicit import (
    _broadcast_dt,
    _broadcast_jac,
    _normalize_derivimplicit_options,
    build_derivimplicit,
    derivimplicit_step,
)
from dendra.models.mechanisms._kinetic import (
    _apply_conserve_to_derivs,
    _last_paren_block,
    _parse_compartment,
    _parse_conserve,
    _parse_side,
    _split_top_level_commas,
    kinetic_to_derivatives,
    make_conserve_enforcer,
)
from dendra.models.mechanisms._linearimplicit import build_linearimplicit
from dendra.models.mechanisms._rosenbrock import build_rosenbrock1
from dendra.models.mechanisms._solvers import _solve_linear_small
from dendra.models.mechanisms.compilers.ast import (
    factor_linear_in_x_from_codeblock,
    factorize_linear_in_v,
    replace_v,
)
from dendra.models.mechanisms.compilers.source import safe_source

DTYPE = torch.float64


def _dummy(**values):
    values.setdefault("training", False)
    return SimpleNamespace(**values)


def _stack_result(result, names):
    return torch.stack([result[name] for name in names], dim=-1)


def _well_conditioned_system(batch_shape, n):
    generator = torch.Generator().manual_seed(1729 + n)
    raw = torch.randn(*batch_shape, n, n, generator=generator, dtype=DTYPE)
    eye = torch.eye(n, dtype=DTYPE).expand(*batch_shape, n, n)
    matrix = raw @ raw.mT + (n + 0.5) * eye
    rhs = torch.randn(*batch_shape, n, generator=generator, dtype=DTYPE)
    return matrix, rhs


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_small_linear_solver_matches_torch(n, batch_shape):
    matrix, rhs = _well_conditioned_system(batch_shape, n)

    actual = _solve_linear_small(matrix, rhs)
    expected = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_small_linear_solver_gradcheck(n):
    matrix, rhs = _well_conditioned_system((2,), n)
    matrix.requires_grad_(True)
    rhs.requires_grad_(True)

    assert torch.autograd.gradcheck(
        _solve_linear_small,
        (matrix, rhs),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


def test_small_linear_solver_empty_batch():
    matrix = torch.empty(0, 3, 3, dtype=DTYPE)
    rhs = torch.empty(0, 3, dtype=DTYPE)

    assert _solve_linear_small(matrix, rhs).shape == (0, 3)


def test_derivimplicit_affine_constant_jacobian_matches_dense_reference():
    matrix = torch.tensor([[-2.0, 0.5], [1.0, -3.0]], dtype=DTYPE)
    bias = torch.tensor([0.25, -0.75], dtype=DTYPE)
    state = torch.tensor([[1.0, 2.0], [-0.5, 0.75]], dtype=DTYPE)
    dt = torch.tensor([0.1, 0.25], dtype=DTYPE)

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, _t: x @ matrix.mT + bias,
        jac=matrix,
    )
    system = torch.eye(2, dtype=DTYPE) - dt[:, None, None] * matrix
    expected_rhs = state + dt[:, None] * bias
    expected = torch.linalg.solve(system, expected_rhs.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_derivimplicit_state_dependent_jacobian_solves_backward_euler():
    state = torch.tensor([[0.25], [1.5], [3.0]], dtype=DTYPE)
    dt = torch.tensor([0.2, 0.1, 0.05], dtype=DTYPE)

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, _t: -(x**2),
        jac_fn=lambda x, _t: (-2.0 * x).unsqueeze(-1),
        tol=1e-13,
    )
    expected = (-1.0 + torch.sqrt(1.0 + 4.0 * dt[:, None] * state)) / (
        2.0 * dt[:, None]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        actual - state + dt[:, None] * actual**2, torch.zeros_like(actual)
    )


def test_derivimplicit_autograd_jacobian_and_gradient():
    state = torch.tensor([0.4, 1.2], dtype=DTYPE, requires_grad=True)
    rate = torch.tensor(1.7, dtype=DTYPE, requires_grad=True)
    dt = torch.tensor(0.15, dtype=DTYPE)

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, _t: -rate * x,
        create_graph=True,
        detach_newton=False,
        tol=1e-13,
    )
    expected = state / (1.0 + dt * rate)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    denom = 1.0 + dt * rate.detach()
    torch.testing.assert_close(state.grad, torch.full_like(state, 1.0 / denom))
    torch.testing.assert_close(
        rate.grad,
        (-dt * state.detach() / denom**2).sum(),
    )


def test_derivimplicit_uses_end_of_step_time():
    state = torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    time = torch.tensor([0.5, 1.0], dtype=DTYPE)
    dt = torch.tensor([0.1, 0.2], dtype=DTYPE)

    actual = derivimplicit_step(
        state,
        dt,
        lambda x, t: torch.zeros_like(x) + t[..., None],
        t_ms=time,
        jac=torch.zeros(1, 1, dtype=DTYPE),
    )

    torch.testing.assert_close(actual, state + dt[:, None] * (time + dt)[:, None])


def test_derivimplicit_line_search_and_zero_iterations():
    state = torch.tensor([2.0], dtype=DTYPE)
    solved = derivimplicit_step(
        state,
        torch.tensor(0.2, dtype=DTYPE),
        lambda x, _t: -(x**3),
        jac_fn=lambda x, _t: (-3.0 * x**2).unsqueeze(-1),
        line_search=True,
        detach_newton=False,
        tol=1e-13,
    )
    torch.testing.assert_close(
        solved - state + 0.2 * solved**3,
        torch.zeros_like(solved),
        atol=1e-11,
        rtol=1e-11,
    )

    unchanged = derivimplicit_step(
        state,
        torch.tensor(0.2, dtype=DTYPE),
        lambda x, _t: -x,
        jac_fn=lambda x, _t: -torch.ones(1, 1, dtype=x.dtype),
        max_iter=0,
        detach_newton=False,
    )
    torch.testing.assert_close(unchanged, state)


def test_derivimplicit_tensor_validation_and_broadcast_helpers():
    with pytest.raises(ValueError, match="at least 1 dimension"):
        derivimplicit_step(torch.tensor(1.0), torch.tensor(0.1), lambda x, t: -x)
    with pytest.raises(ValueError, match="at most one"):
        derivimplicit_step(
            torch.ones(1),
            torch.tensor(0.1),
            lambda x, t: -x,
            jac=torch.ones(1, 1),
            jac_fn=lambda x, t: torch.ones(1, 1),
            detach_newton=False,
        )

    state = torch.ones(2, 3, 4, dtype=DTYPE)
    torch.testing.assert_close(
        _broadcast_dt(torch.tensor(0.2), state), torch.tensor(0.2, dtype=DTYPE)
    )
    assert _broadcast_dt(torch.ones(2, 3), state).shape == (2, 3)
    assert _broadcast_jac(torch.eye(4), (2, 3), 4, state).shape == (2, 3, 4, 4)
    with pytest.raises(RuntimeError):
        _broadcast_jac(torch.eye(3), (2, 3), 4, state)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"unknown": 1}, "Unknown derivimplicit option"),
        ({"max_iter": -1}, "max_iter must be non-negative"),
        ({"max_ls_steps": -1}, "max_ls_steps must be non-negative"),
        ({"damping": 0}, "damping must be positive"),
        ({"ls_decay": 1}, "open interval"),
        ({"tol": -1}, "tol must be non-negative"),
        ({"jacobian_regularization": -1}, "must be non-negative"),
    ],
)
def test_derivimplicit_option_validation(options, message):
    with pytest.raises(ValueError, match=message):
        _normalize_derivimplicit_options(options)


@pytest.mark.parametrize("solver", ["auto", "small", "dense"])
def test_generated_linearimplicit_scalar_batched_and_gradient(solver):
    solve = build_linearimplicit(
        ["x"],
        set(),
        ["x' = -rate*x + source"],
        solver=solver,
    )
    rate = torch.tensor([0.5, 2.0, 4.0], dtype=DTYPE, requires_grad=True)
    source = torch.tensor([1.0, -0.5, 2.0], dtype=DTYPE, requires_grad=True)
    state = torch.tensor([0.25, 1.0, -2.0], dtype=DTYPE, requires_grad=True)
    dt = torch.tensor([0.1, 0.2, 0.05], dtype=DTYPE)

    actual = solve(_dummy(rate=rate, source=source), dt, state)["x"]
    expected = (state + dt * source) / (1.0 + dt * rate)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    expected_grads = torch.autograd.grad(expected.sum(), (state, rate, source))
    for actual_grad, expected_grad in zip(
        (state.grad, rate.grad, source.grad), expected_grads
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("n, solver", [(2, "small"), (2, "dense"), (3, "auto")])
def test_generated_linearimplicit_coupled_matches_dense(n, solver):
    names = [f"x{i}" for i in range(n)]
    matrix = torch.tensor(
        [[-2.0, 0.5, 0.25], [1.0, -3.0, 0.5], [-0.25, 0.75, -1.5]],
        dtype=DTYPE,
    )[:n, :n]
    bias = torch.tensor([0.25, -0.75, 1.25], dtype=DTYPE)[:n]
    derivative = []
    for i, name in enumerate(names):
        terms = [f"({matrix[i, j].item()})*{other}" for j, other in enumerate(names)]
        derivative.append(f"{name}' = {' + '.join(terms)} + ({bias[i].item()})")
    solve = build_linearimplicit(names, set(), derivative, solver=solver)

    state = torch.tensor([[1.0, 2.0, -1.0], [-0.5, 0.75, 2.0]], dtype=DTYPE)[:, :n]
    dt = torch.tensor([0.1, 0.25], dtype=DTYPE)
    result = solve(_dummy(), dt, *torch.unbind(state, dim=-1))
    actual = _stack_result(result, names)
    system = torch.eye(n, dtype=DTYPE) - dt[:, None, None] * matrix
    expected = torch.linalg.solve(
        system, (state + dt[:, None] * bias).unsqueeze(-1)
    ).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_linearimplicit_eliminates_conserved_state():
    solve = build_linearimplicit(
        ["closed", "open"],
        set(),
        ["closed' = -alpha*closed + beta*open", "open' = alpha*closed - beta*open"],
        eliminate={"open": "open = 1 - closed"},
    )
    closed = torch.tensor([0.2, 0.8], dtype=DTYPE)
    opened = 1.0 - closed
    dt = torch.tensor(0.1, dtype=DTYPE)
    result = solve(_dummy(alpha=2.0, beta=0.5), dt, closed, opened)
    expected_closed = (closed + dt * 0.5) / (1.0 + dt * 2.5)

    torch.testing.assert_close(result["closed"], expected_closed)
    torch.testing.assert_close(result["open"], 1.0 - expected_closed)


def test_linearimplicit_nonlinear_strict_and_fallback():
    with pytest.raises(ValueError, match="not affine-linear"):
        build_linearimplicit(["x"], set(), ["x' = -x*x"])

    solve = build_linearimplicit(["x"], set(), ["x' = -x*x"], strict=False)
    state = torch.tensor([0.5, 2.0], dtype=DTYPE)
    dt = torch.tensor(0.1, dtype=DTYPE)
    actual = solve(_dummy(), dt, state)["x"]
    expected = state - dt * state**2 / (1.0 + 2.0 * dt * state)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"solver": "bad"}, "solver must be"),
        ({"fallback": "bad"}, "fallback must be"),
        ({"jacobian_regularization": -1}, "must be non-negative"),
        ({"pade": True}, "does not support"),
        ({"mystery": 1}, "Unknown linearimplicit option"),
    ],
)
def test_linearimplicit_option_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_linearimplicit(["x"], set(), ["x' = -x"], **kwargs)


def test_linearimplicit_builder_validation():
    with pytest.raises(ValueError, match="cannot be assigned"):
        build_linearimplicit(["x"], {"x"}, ["x' = -x"])
    with pytest.raises(ValueError, match="missing derivative"):
        build_linearimplicit(["x", "y"], set(), ["x' = -x"])
    with pytest.raises(ValueError, match="at least one non-eliminated"):
        build_linearimplicit(["x"], set(), ["x' = -x"], eliminate={"x": "x = 1"})


@pytest.mark.parametrize("n", [1, 2, 3])
def test_generated_rosenbrock_matches_dense_linear_reference(n):
    names = [f"x{i}" for i in range(n)]
    matrix = torch.tensor(
        [[-2.0, 0.5, 0.25], [1.0, -3.0, 0.5], [-0.25, 0.75, -1.5]],
        dtype=DTYPE,
    )[:n, :n]
    derivatives = []
    for i, name in enumerate(names):
        terms = [f"({matrix[i, j].item()})*{other}" for j, other in enumerate(names)]
        derivatives.append(f"{name}' = {' + '.join(terms)}")
    solve = build_rosenbrock1(names, set(), derivatives)
    state = torch.tensor([[1.0, 2.0, -1.0], [-0.5, 0.75, 2.0]], dtype=DTYPE)[:, :n]
    dt = torch.tensor([0.1, 0.25], dtype=DTYPE)

    result = solve(_dummy(), dt, *torch.unbind(state, dim=-1))
    actual = _stack_result(result, names)
    system = torch.eye(n, dtype=DTYPE) - dt[:, None, None] * matrix
    expected = torch.linalg.solve(system, state.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_rosenbrock_nonlinear_formula_options_and_gradient():
    solve = build_rosenbrock1(["x"], set(), ["x' = -rate*x*x"], gamma=0.75, damping=0.5)
    rate = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)
    state = torch.tensor([0.25, 1.5], dtype=DTYPE, requires_grad=True)
    dt = torch.tensor(0.2, dtype=DTYPE)

    actual = solve(_dummy(rate=rate), dt, state)["x"]
    expected = state - 0.5 * dt * rate * state**2 / (
        1.0 + 0.75 * dt * 2.0 * rate * state
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(rate.grad)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"gamma": 0}, "gamma must be positive"),
        ({"damping": 0}, "damping must be positive"),
        ({"jacobian_regularization": -1}, "must be non-negative"),
        ({"fallback": "bad"}, "fallback must be"),
        ({"pade": True}, "does not support"),
        ({"mystery": 1}, "Unknown rosenbrock option"),
    ],
)
def test_rosenbrock_option_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_rosenbrock1(["x"], set(), ["x' = -x"], **kwargs)


def test_rosenbrock_builder_validation_and_elimination():
    with pytest.raises(ValueError, match="cannot be assigned"):
        build_rosenbrock1(["x"], {"x"}, ["x' = -x"])
    with pytest.raises(ValueError, match="missing derivative"):
        build_rosenbrock1(["x", "y"], set(), ["x' = -x"])
    with pytest.raises(ValueError, match="at least one non-eliminated"):
        build_rosenbrock1(["x"], set(), ["x' = -x"], eliminate={"x": "x = 1"})

    solve = build_rosenbrock1(
        ["x", "y"],
        set(),
        ["x' = -x", "y' = x"],
        eliminate={"y": "y = 1 - x"},
    )
    result = solve(
        _dummy(),
        torch.tensor(0.2, dtype=DTYPE),
        torch.tensor(0.5, dtype=DTYPE),
        torch.tensor(0.5, dtype=DTYPE),
    )
    torch.testing.assert_close(result["x"], torch.tensor(0.5 / 1.2, dtype=DTYPE))
    torch.testing.assert_close(result["y"], 1.0 - result["x"])


def test_generated_derivimplicit_affine_nonlinear_and_elimination():
    affine = build_derivimplicit(["x"], set(), ["x' = -rate*x + source"])
    state = torch.tensor([0.25, 1.0], dtype=DTYPE)
    result = affine(_dummy(rate=2.0, source=0.5), torch.tensor(0.1, dtype=DTYPE), state)
    torch.testing.assert_close(result["x"], (state + 0.05) / 1.2)

    nonlinear = build_derivimplicit(
        ["x"],
        set(),
        ["x' = -x*x"],
        tol=1e-13,
        detach_newton=False,
        create_graph=True,
    )
    nonlinear_state = torch.tensor([0.25, 1.5], dtype=DTYPE, requires_grad=True)
    result = nonlinear(_dummy(training=True), 0.1, nonlinear_state)["x"]
    expected = (-1.0 + torch.sqrt(1.0 + 0.4 * nonlinear_state)) / 0.2
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert torch.isfinite(nonlinear_state.grad).all()

    conserved = build_derivimplicit(
        ["x", "y"],
        set(),
        ["x' = -x", "y' = x"],
        eliminate={"y": "y = 1 - x"},
    )
    result = conserved(_dummy(), 0.2, torch.tensor(0.5), torch.tensor(0.5))
    torch.testing.assert_close(result["y"], 1.0 - result["x"])


def test_derivimplicit_builder_validation():
    with pytest.raises(ValueError, match="cannot be assigned"):
        build_derivimplicit(["x"], {"x"}, ["x' = -x"])
    with pytest.raises(ValueError, match="missing derivative"):
        build_derivimplicit(["x", "y"], set(), ["x' = -x"])
    with pytest.raises(ValueError, match="at least one non-eliminated"):
        build_derivimplicit(["x"], set(), ["x' = -x"], eliminate={"x": "x = 1"})


def test_bufferimplicit_matches_backward_euler_equations_and_gradient():
    solve = build_bufferimplicit(
        ["bound", "free"],
        set(),
        [
            "bound' = ku*free*(1-bound) - kr*bound",
            "free' = source - gamma*(ku*free*(1-bound) - kr*bound)",
        ],
        bound="bound",
        free="free",
    )
    bound = torch.tensor([0.1, 0.7], dtype=DTYPE, requires_grad=True)
    free = torch.tensor([0.25, 2.0], dtype=DTYPE, requires_grad=True)
    dt = torch.tensor([0.05, 0.2], dtype=DTYPE)
    params = _dummy(
        ku=torch.tensor(1.5, dtype=DTYPE, requires_grad=True),
        kr=torch.tensor(0.4, dtype=DTYPE, requires_grad=True),
        gamma=torch.tensor(2.0, dtype=DTYPE, requires_grad=True),
        source=torch.tensor(0.3, dtype=DTYPE, requires_grad=True),
    )

    result = solve(params, dt, bound, free)
    bound_new, free_new = result["bound"], result["free"]
    rate_new = params.ku * free_new * (1.0 - bound_new) - params.kr * bound_new
    torch.testing.assert_close(bound_new - bound, dt * rate_new)
    torch.testing.assert_close(
        free_new + params.gamma * bound_new,
        free + params.gamma * bound + dt * params.source,
    )

    (bound_new + free_new).sum().backward()
    for value in (bound, free, params.ku, params.kr, params.gamma, params.source):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_bufferimplicit_zero_binding_rate_and_clamps():
    solve = build_bufferimplicit(
        ["bound", "free"],
        set(),
        [
            "bound' = ku*free*(1-bound) - kr*bound",
            "free' = source - gamma*(ku*free*(1-bound) - kr*bound)",
        ],
        bound="bound",
        free="free",
        clamp_bound=True,
        bound_min=0.0,
        bound_max=1.0,
        free_min=0.0,
    )
    result = solve(
        _dummy(ku=0.0, kr=2.0, gamma=1.0, source=-100.0),
        0.1,
        torch.tensor(0.5),
        torch.tensor(0.1),
    )
    torch.testing.assert_close(result["bound"], torch.tensor(0.5 / 1.2))
    torch.testing.assert_close(result["free"], torch.tensor(0.0))


def test_bufferimplicit_infers_common_state_names():
    solve = build_bufferimplicit(
        ["oc", "cai"],
        set(),
        [
            "oc' = kon*cai*(1-oc) - koff*oc",
            "cai' = -scale*(kon*cai*(1-oc) - koff*oc)",
        ],
    )
    result = solve(
        _dummy(kon=1.0, koff=0.5, scale=2.0),
        0.1,
        torch.tensor(0.2),
        torch.tensor(0.4),
    )
    torch.testing.assert_close(result["cai"] + 2 * result["oc"], torch.tensor(0.8))


@pytest.mark.parametrize(
    "states, kwargs, message",
    [
        (["x"], {}, "exactly two"),
        (["x", "y"], {}, "requires S.METHOD"),
        (["x", "y"], {"bound": "x"}, "requires both"),
        (["x", "y"], {"bound": "x", "free": "x"}, "must be distinct"),
        (["x", "y"], {"bound": "z", "free": "y"}, "not in states"),
        (["x", "y"], {"bound": "x", "free": "z"}, "not in states"),
    ],
)
def test_bufferimplicit_state_validation(states, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_bufferimplicit(states, set(), [f"{s}' = -{s}" for s in states], **kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"eps": -1}, "eps must be non-negative"),
        ({"fallback": "bad"}, "fallback must be"),
        ({"pade": True}, "does not support"),
        ({"mystery": 1}, "Unknown bufferimplicit option"),
    ],
)
def test_bufferimplicit_option_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_bufferimplicit(
            ["bound", "free"],
            set(),
            ["bound' = free*(1-bound)-bound", "free' = -bound'"],
            bound="bound",
            free="free",
            **kwargs,
        )


def test_bufferimplicit_equation_validation_and_fallback():
    derivatives = ["bound' = -bound", "free' = bound"]
    with pytest.raises(ValueError, match="could not infer gamma"):
        build_bufferimplicit(
            ["bound", "free"],
            set(),
            derivatives,
            bound="bound",
            free="free",
        )

    fallback = build_bufferimplicit(
        ["bound", "free"],
        set(),
        derivatives,
        bound="bound",
        free="free",
        strict=False,
    )
    result = fallback(_dummy(), 0.1, torch.tensor(1.0), torch.tensor(0.0))
    torch.testing.assert_close(result["bound"], torch.tensor(1.0 / 1.1))

    with pytest.raises(ValueError, match="does not support eliminated"):
        build_bufferimplicit(
            ["bound", "free"],
            set(),
            derivatives,
            bound="bound",
            free="free",
            eliminate={"free": "free = 1 - bound"},
        )


def test_kinetic_reversible_conservation_pipeline():
    derivatives, amount_map, conserve, compartments, fluxes = kinetic_to_derivatives(
        ["closed", "open"],
        ["~ closed <-> open (alpha, beta)", "CONSERVE closed + open = 1"],
    )
    assert compartments == {"closed": "1", "open": "1"}
    assert len(fluxes) == 1
    assert fluxes[0]["kind"] == "reaction"
    assert "alpha" in amount_map["closed"]
    assert conserve == [("open", "open = (1 - (1*(1)*closed)) / (1*(1))")]

    solve = build_linearimplicit(
        ["closed", "open"],
        set(),
        derivatives,
        eliminate=dict(conserve),
    )
    closed = torch.tensor([0.2, 0.8], dtype=DTYPE)
    result = solve(_dummy(alpha=2.0, beta=0.5), 0.1, closed, 1.0 - closed)
    expected = (closed + 0.05) / 1.25
    torch.testing.assert_close(result["closed"], expected)
    torch.testing.assert_close(result["open"], 1.0 - expected)


def test_kinetic_flux_compartment_stoichiometry_and_ignored_diffusion():
    derivatives, amounts, conserve, compartments, fluxes = kinetic_to_derivatives(
        ["a", "b", "c"],
        [
            "COMPARTMENT idx, volume[idx] {a, b}",
            "~ 2a + b -> 3c (rate)",
            "~ c << influx",
            "LONGITUDINAL_DIFFUSION D {a}",
        ],
    )
    assert conserve is None
    assert compartments == {"a": "volume[idx]", "b": "volume[idx]", "c": "1"}
    assert [entry["kind"] for entry in fluxes] == ["reaction", "explicit_flux"]
    assert "-2" in amounts["a"]
    assert "3" in amounts["c"] and "influx" in amounts["c"]
    assert "/(volume[idx])" in derivatives[0]


def test_kinetic_parser_helpers_and_validation():
    assert _parse_side("A + 2B + B", {"A", "B"}) == {"A": 1, "B": 3}
    assert _parse_side("", {"A"}) == {}
    assert _last_paren_block("A -> B (f(x), g(y, z))") == (
        "A -> B",
        "f(x), g(y, z)",
    )
    assert _split_top_level_commas("a, f(b, c), d") == ["a", "f(b, c)", "d"]
    assert _parse_compartment("COMPARTMENT i, vol[i] {A B}", ["A", "B"]) == {
        "A": "vol[i]",
        "B": "vol[i]",
    }
    conserve = _parse_conserve("CONSERVE 2A + B = total", ["A", "B"])
    assert conserve.weights == {"A": 2, "B": 1}
    assert conserve.const == "total"
    assert make_conserve_enforcer(conserve, {"A": "va", "B": "vb"}) == (
        "B",
        "B = (total - (2*(va)*A)) / (1*(vb))",
    )
    mapped, eliminated = _apply_conserve_to_derivs(
        ["A", "B"], {"A": "flux", "B": "other"}, {}, conserve, "A"
    )
    assert eliminated == "A"
    assert mapped["A"] == "-(1*(other))/(2)"

    with pytest.raises(ValueError, match="Unknown state"):
        _parse_side("Z", {"A"})
    with pytest.raises(ValueError, match="Bad species term"):
        _parse_side("1.5A", {"A"})
    with pytest.raises(ValueError, match="Missing closing"):
        _last_paren_block("A -> B")
    with pytest.raises(ValueError, match="Trailing garbage"):
        _last_paren_block("A -> B (k) junk")
    with pytest.raises(ValueError, match="must contain"):
        _parse_conserve("A + B", ["A", "B"])
    with pytest.raises(ValueError, match="Right-hand side"):
        _parse_conserve("A =", ["A"])
    with pytest.raises(ValueError, match="Left-hand side"):
        _parse_conserve("= 1", ["A"])
    with pytest.raises(ValueError, match="inside braces"):
        _parse_compartment("COMPARTMENT volume A", ["A"])
    with pytest.raises(ValueError, match="Unknown state"):
        _parse_compartment("COMPARTMENT volume {Z}", ["A"])


@pytest.mark.parametrize(
    "line, message",
    [
        ("not a reaction", "Expected"),
        ("~ A", "Missing"),
        ("~ A -> B (kf, extra)", "one rate"),
        ("~ A <-> B (kf)", "two rates"),
        ("~ A + B << flux", "exactly one"),
    ],
)
def test_kinetic_invalid_reactions(line, message):
    with pytest.raises(ValueError, match=message):
        kinetic_to_derivatives(["A", "B"], [line])


def test_compiler_factorizes_straight_line_current():
    source = """
    class Leak:
        def i(self, v):
            driving = v - self.erev
            return self.g * driving
    """
    conductance, reversal = factorize_linear_in_v(source)
    assert sp.simplify(conductance.replace("self.g", "g") + " - g") == 0
    assert reversal == "self.erev"


def test_compiler_factorization_helpers_and_source():
    source = """
    class Current:
        def i(self, v):
            return 2 * v - 6
    """
    assert factorize_linear_in_v(source) == ("2", "3")
    assert replace_v("v + vtrap + self.v") == "(v + v_n) / 2 + vtrap + self.v"
    assert factor_linear_in_x_from_codeblock(
        "a = self.g * 2\nreturn a + self.h * v_n", x_var="v_n"
    ) == ("2*self.g", "self.h")

    class Plain:
        def method(self):
            return 1

    assert "class Plain" in safe_source(Plain)


def test_compiler_rejects_unsupported_or_non_linear_current():
    with pytest.raises(ValueError, match="No return"):
        factorize_linear_in_v("class Bad:\n    def i(self, v):\n        x = v")
    with pytest.raises(NotImplementedError, match="straight-line"):
        factorize_linear_in_v(
            "class Bad:\n    def i(self, v):\n        if v > 0:\n            return v"
        )
    with pytest.raises(ZeroDivisionError, match="identically zero"):
        factorize_linear_in_v("class Bad:\n    def i(self, v):\n        return 2")
    with pytest.raises(ValueError, match="linear"):
        factorize_linear_in_v(
            "class Bad:\n    def i(self, v):\n        return v**2 + v"
        )
    with pytest.raises(ValueError, match="not linear"):
        factor_linear_in_x_from_codeblock("return v_n**2", x_var="v_n")
