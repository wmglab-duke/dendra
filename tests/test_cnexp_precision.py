"""Float32 exact linear state updates retain small rates and sensitivities."""

import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra.models.mechanisms._cnexp import build_cnexp


@pytest.fixture(scope="module")
def gate_step():
    solve = build_cnexp(["x"], {"source", "decay"}, ["x' = source - decay*x"])

    def step(state, source, decay, dt):
        return solve(None, dt, x=state, source=source, decay=decay)["x"]

    return step


def _analytic_reference(state, source, decay, dt):
    """Independent float64 exponential-integral solution, including decay=0.

    The series resolves the removable singularity; expm1 handles larger rates.
    This form permits differentiation at zero without equilibrium division.
    """
    z = -decay * dt
    term = torch.ones_like(z)
    integral = term
    for order in range(1, 25):
        term = term * z / (order + 1)
        integral = integral + term
    small = z.abs() < 0.25
    safe_z = torch.where(small, torch.ones_like(z), z)
    large_integral = torch.expm1(safe_z) / safe_z
    integral = torch.where(small, integral, large_integral)
    return state * torch.exp(z) + source * dt * integral


def _near_equilibrium(decay):
    equilibrium = torch.tensor(0.48894766, dtype=torch.float32)
    rate = equilibrium.new_tensor(decay)
    source = rate * equilibrium
    state = torch.nextafter(equilibrium, equilibrium.new_tensor(float("inf")))
    return state, source, rate, equilibrium.new_tensor(0.025)


@pytest.mark.parametrize("decay", [0.1233, 0.01, 1e-5, 1e-8, -0.125])
def test_cnexp_near_equilibrium_keeps_correctly_rounded_float32_state(gate_step, decay):
    arguments = _near_equilibrium(decay)
    actual = gate_step(*arguments)
    reference = _analytic_reference(*(value.double() for value in arguments))
    assert actual.dtype == torch.float32 and torch.isfinite(actual)
    # The physical update is below half an ULP. Reconstructing an equilibrium
    # can discard one ULP even though the exact state should round unchanged.
    torch.testing.assert_close(actual, reference.float(), rtol=0, atol=0)


@pytest.mark.parametrize("decay", [0.1233, 0.01, 1e-5, 1e-8, -0.125])
def test_cnexp_near_equilibrium_gradients_match_float64_analytic(gate_step, decay):
    arguments = _near_equilibrium(decay)
    actual = torch.func.jacrev(gate_step, argnums=(0, 1, 2))(*arguments)
    expected = torch.func.jacrev(_analytic_reference, argnums=(0, 1, 2))(
        *(value.double() for value in arguments)
    )
    for derivative, reference in zip(actual, expected, strict=True):
        assert derivative.dtype == torch.float32 and torch.isfinite(derivative)
        torch.testing.assert_close(derivative.double(), reference, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("source", [0.0, 0.03])
def test_cnexp_zero_rate_has_finite_value_and_all_argument_derivatives(
    gate_step, source
):
    arguments = tuple(
        torch.tensor(value, dtype=torch.float32) for value in (0.48, source, 0.0, 0.025)
    )
    actual = gate_step(*arguments)
    torch.testing.assert_close(
        actual, arguments[0] + arguments[1] * arguments[3], rtol=0, atol=0
    )
    actual_derivatives = torch.func.jacrev(gate_step, argnums=(0, 1, 2, 3))(*arguments)
    expected_derivatives = torch.func.jacrev(_analytic_reference, argnums=(0, 1, 2, 3))(
        *(value.double() for value in arguments)
    )
    for derivative, reference in zip(
        actual_derivatives, expected_derivatives, strict=True
    ):
        assert torch.isfinite(derivative)
        torch.testing.assert_close(derivative.double(), reference, rtol=2e-6, atol=1e-9)


def test_cnexp_float32_vmap_and_jvp_match_independent_analytic(gate_step):
    state = torch.tensor([0.48, 0.23, 0.81, 0.5], dtype=torch.float32)
    decay = state.new_tensor([0.0, 1e-8, 0.1233, -0.125])
    source = torch.where(decay == 0, state.new_tensor(0.03), decay * state)
    dt = state.new_tensor(0.025)
    arguments = state, source, decay

    def response(*values):
        return gate_step(*values, dt)

    def reference(*values):
        return _analytic_reference(*values, dt.double())

    actual = torch.vmap(gate_step, in_dims=(0, 0, 0, None))(*arguments, dt)
    expected = reference(*(value.double() for value in arguments))
    torch.testing.assert_close(actual.double(), expected, rtol=2e-7, atol=2e-8)
    tangent = tuple(torch.full_like(value, 0.125) for value in arguments)
    _, actual_jvp = torch.func.jvp(response, arguments, tangent)
    _, expected_jvp = torch.func.jvp(
        reference,
        tuple(value.double() for value in arguments),
        tuple(value.double() for value in tangent),
    )
    assert torch.isfinite(actual_jvp).all()
    torch.testing.assert_close(actual_jvp.double(), expected_jvp, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize(
    "state,source,decay,dt",
    [(1.0, 1e-6, 100.0, 1.0), (1.0, 1e-6, 20.0, 1.0), (0.48, 0.03, 80.0, 0.5)],
)
def test_cnexp_stiff_decay_retains_equilibrium_value_and_gradients(
    gate_step, state, source, decay, dt
):
    arguments = tuple(
        torch.tensor(value, dtype=torch.float32) for value in (state, source, decay, dt)
    )
    reference_arguments = tuple(value.double() for value in arguments)
    actual = gate_step(*arguments)
    expected = _analytic_reference(*reference_arguments)
    assert actual.dtype == torch.float32 and torch.isfinite(actual)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-14)
    gradients = torch.func.jacrev(gate_step, argnums=(0, 1, 2))(*arguments)
    reference_gradients = torch.func.jacrev(_analytic_reference, argnums=(0, 1, 2))(
        *reference_arguments
    )
    for derivative, reference in zip(gradients, reference_gradients, strict=True):
        assert derivative.dtype == torch.float32 and torch.isfinite(derivative)
        torch.testing.assert_close(
            derivative.double(), reference, rtol=2e-6, atol=1e-12
        )


def test_cnexp_mixed_small_stiff_zero_and_growth_lanes_compose(gate_step):
    arguments = tuple(
        torch.tensor(values, dtype=torch.float32)
        for values in (
            [0.48894769, 1.0, 0.48, 0.8],
            [0.06028725, 1e-6, 0.03, 0.02],
            [0.1233, 100.0, 0.0, -2.0],
            [0.025, 1.0, 0.025, 0.75],
        )
    )
    reference_arguments = tuple(value.double() for value in arguments)
    actual = torch.vmap(gate_step)(*arguments)
    expected = _analytic_reference(*reference_arguments)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-12)
    tangent = tuple(torch.full_like(value, 0.125) for value in arguments)
    _, actual_jvp = torch.func.jvp(gate_step, arguments, tangent)
    _, expected_jvp = torch.func.jvp(
        _analytic_reference,
        reference_arguments,
        tuple(value.double() for value in tangent),
    )
    assert actual_jvp.dtype == torch.float32 and torch.isfinite(actual_jvp).all()
    torch.testing.assert_close(actual_jvp.double(), expected_jvp, rtol=2e-6, atol=1e-10)


class _Float32Only(TorchDispatchMode):
    @staticmethod
    def check(values):
        for value in pytree.tree_leaves(values):
            if torch.is_tensor(value) and value.is_floating_point():
                assert value.dtype == torch.float32

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        self.check((args, kwargs))
        result = operation(*args, **kwargs)
        self.check(result)
        return result


def test_cnexp_generated_value_backward_and_jvp_stay_finite_float32(gate_step):
    values = tuple(
        torch.tensor([value], dtype=torch.float32, requires_grad=True)
        for value in (0.48, 0.03, 0.0)
    )
    dt = torch.tensor(0.025, dtype=torch.float32)

    def response(*arguments):
        return gate_step(*arguments, dt)

    # Population gate/rate fields retain a state dimension. Forward AD owns
    # dispatch wrappers; inspect its public primal/tangent after the transform.
    with _Float32Only():
        result = response(*values)
        derivatives = torch.autograd.grad(result, values)
    _, tangent = torch.func.jvp(
        response, values, tuple(torch.ones_like(value) for value in values)
    )
    for value in (result, *derivatives, tangent):
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "compiled",
    [False, pytest.param(True, marks=pytest.mark.slow)],
    ids=["eager", "inductor"],
)
def test_cnexp_cuda_mixed_rates_match_float64_value_and_gradient_oracle(
    gate_step, compiled
):
    # Exact binary equilibrium makes the small-rate lane's zero dt sensitivity
    # meaningful: rounding an authored near-zero float32 rate can dominate it.
    host_arguments = tuple(
        torch.tensor(values, dtype=torch.float32)
        for values in (
            [0.5, 1.0, 0.48, 0.8],
            [0.0625, 1e-6, 0.03, 0.02],
            [0.125, 100.0, 0.0, -2.0],
            [0.03125, 1.0, 0.025, 0.75],
        )
    )
    arguments = tuple(value.to("cuda").requires_grad_() for value in host_arguments)
    reference_arguments = tuple(
        value.double().requires_grad_() for value in host_arguments
    )
    weights = torch.tensor([0.5, -0.75, 1.25, 2.0], dtype=torch.float32)
    response = (
        torch.compile(gate_step, backend="inductor", fullgraph=True)
        if compiled
        else gate_step
    )
    actual = response(*arguments)
    expected = _analytic_reference(*reference_arguments)
    derivatives = torch.autograd.grad((actual * weights.to("cuda")).sum(), arguments)
    reference_derivatives = torch.autograd.grad(
        (expected * weights.double()).sum(), reference_arguments
    )
    for value, reference in zip(
        (actual, *derivatives), (expected, *reference_derivatives), strict=True
    ):
        assert value.device.type == "cuda" and value.dtype == torch.float32
        assert torch.isfinite(value).all()
        torch.testing.assert_close(
            value.detach().cpu().double(), reference.detach(), rtol=2e-6, atol=1e-12
        )
