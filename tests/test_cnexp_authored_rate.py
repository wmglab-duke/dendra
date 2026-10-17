"""Generated CNexp retains authored rates and scalar-coefficient broadcasts."""

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._cnexp import build_cnexp


@pytest.fixture(scope="module")
def relaxation_step():
    solve = build_cnexp(["x"], {"x_inf", "tau"}, ["x' = (x_inf-x)/tau"])

    def step(state, equilibrium, tau, dt):
        return solve(None, dt, x=state, x_inf=equilibrium, tau=tau)["x"]

    return step


def _equilibrium_arguments():
    equilibrium = torch.tensor(
        [0.9702991247177124, 0.7865960597991943], dtype=torch.float32
    )
    tau = torch.tensor([0.04862092807888985, 0.08306379616260529], dtype=torch.float32)
    dt = torch.tensor([0.023824255913496017, 0.04070126265287399], dtype=torch.float32)
    return equilibrium.clone(), equilibrium, tau, dt


def test_cnexp_authored_relaxation_retains_exact_equilibrium(relaxation_step):
    state, equilibrium, tau, dt = _equilibrium_arguments()
    # Expanding the authored rate into x_inf/tau - x*(1/tau) moves each
    # chosen equilibrium by one float32 ULP, in opposite directions.
    for _ in range(32):
        state = relaxation_step(state, equilibrium, tau, dt)
    torch.testing.assert_close(state, equilibrium, rtol=0, atol=0)
    assert state.dtype == torch.float32


def test_cnexp_authored_equilibrium_gradients_and_jvp(relaxation_step):
    arguments = _equilibrium_arguments()

    def reference(state, equilibrium, tau, dt):
        return equilibrium + (state - equilibrium) * torch.exp(-dt / tau)

    actual = torch.func.jacrev(relaxation_step, argnums=(0, 1, 2, 3))(*arguments)
    expected = torch.func.jacrev(reference, argnums=(0, 1, 2, 3))(
        *(value.double() for value in arguments)
    )
    for value, truth in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        torch.testing.assert_close(value.double(), truth, rtol=2e-6, atol=1e-8)
    directions = tuple(torch.full_like(value, 0.125) for value in arguments)
    _, tangent = torch.func.jvp(relaxation_step, arguments, directions)
    _, truth = torch.func.jvp(
        reference,
        tuple(value.double() for value in arguments),
        tuple(value.double() for value in directions),
    )
    torch.testing.assert_close(tangent.double(), truth, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("dt", (0.025, 5.0), ids=("small", "stiff"))
def test_cnexp_numeric_coefficients_broadcast_over_state(dt):
    solve = build_cnexp(["x"], set(), ["x' = 1-2*x"])
    state = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.8, 0.3]], dtype=torch.float32)
    duration = state.new_tensor(dt)
    actual = solve(None, duration, x=state)["x"]
    expected = 0.5 + (state.double() - 0.5) * torch.exp(-2 * duration.double())
    assert actual.dtype == torch.float32 and actual.shape == state.shape
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-8)


def test_cnexp_numeric_coefficients_compose_with_mixed_and_empty_vmap():
    solve = build_cnexp(["x"], set(), ["x' = 1-2*x"])

    def step(state, dt):
        return solve(None, dt, x=state)["x"]

    state = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.8, 0.3]], dtype=torch.float32)
    dt = state.new_tensor([0.025, 5.0])
    actual = torch.vmap(step)(state, dt)
    expected = 0.5 + (state.double() - 0.5) * torch.exp(-2 * dt.double()[:, None])
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-8)
    empty = torch.vmap(lambda value: step(value, dt[0]))(state[:0])
    assert empty.dtype == torch.float32 and empty.shape == (0, 3)


class _AuthoredRelaxationState(State):
    State.STATE("x")
    State.GLOBAL(x_inf=0.9702991247177124, tau=0.04862092807888985)
    State.DERIVATIVE("x' = (x_inf-x)/tau")

    def state_defaults(self, voltage, values):
        del values
        return {"x": self.x_inf.expand_as(voltage).clone()}


class _AuthoredRelaxationMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_AuthoredRelaxationState)


def test_cnexp_private_exact_helper_remains_functionalizable():
    dt = 0.023824255913496017
    with (
        torch.no_grad(),
        dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float32),
    ):
        model = dn.SingleCompartment(N=1, C=2, dtype=torch.float32, device="cpu")
        model.insert(_AuthoredRelaxationMechanism)
        model.initialize()
        functional, tensors = dn.func.make_functional(model, dt=dt)
        parameters = tensors.independent_parameters(trainable=())
        prepared = functional.prepare(parameters, tensors.constants)
        state, _auxiliary = dn.func.run(
            functional.bind(parameters, prepared), tensors.state, steps=3
        )
    name = next(iter(tensors.state["mechanisms"]))
    original = tensors.state["mechanisms"][name]["x"]
    torch.testing.assert_close(state["mechanisms"][name]["x"], original, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.slow
def test_cnexp_hybrid_cpu_inductor_values_and_all_input_gradients():
    from torch._inductor import metrics

    solve = build_cnexp(
        ["x"], {"source", "coefficient"}, ["x' = source + coefficient*x"]
    )

    def step(state, source, coefficient, dt):
        return solve(None, dt, x=state, source=source, coefficient=coefficient)["x"]

    def oracle(state, source, coefficient, dt):
        # The exponential-integral form is independent of the hybrid's
        # equilibrium/increment formulas. Its local series retains the
        # derivative of the removable singularity at coefficient=0.
        z = coefficient * dt
        small = z.abs() < 1e-4
        safe_z = torch.where(small, torch.ones_like(z), z)
        series = 1 + z / 2 + z.square() / 6 + z.pow(3) / 24 + z.pow(4) / 120
        integral = torch.where(small, series, torch.expm1(safe_z) / safe_z)
        return state * torch.exp(z) + source * dt * integral

    arguments = tuple(
        torch.tensor(value, dtype=torch.float32, device="cpu", requires_grad=True)
        for value in (
            [0.48, 0.37, 1.0, 0.8],
            [0.03, 0.025, 1e-6, 0.02],
            [0.0, -0.25, -100.0, 2.0],
            [0.025, 0.2, 1.0, 0.75],
        )
    )
    reference_arguments = tuple(
        value.detach().double().requires_grad_() for value in arguments
    )
    weights = torch.tensor([0.5, -0.75, 1.25, 2.0], dtype=torch.float32)
    compiled = torch.compile(step, backend="inductor", fullgraph=True)
    generated_before = metrics.generated_kernel_count
    actual = compiled(*arguments)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), arguments)
    assert metrics.generated_kernel_count > generated_before

    expected = oracle(*reference_arguments)
    expected_gradients = torch.autograd.grad(
        (expected * weights.double()).sum(), reference_arguments
    )
    for value, truth in zip(
        (actual, *actual_gradients), (expected, *expected_gradients), strict=True
    ):
        assert value.device.type == "cpu" and value.dtype == torch.float32
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(value.double(), truth, rtol=2e-6, atol=1e-12)
