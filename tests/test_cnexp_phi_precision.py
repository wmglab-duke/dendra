"""Bounded phi1 evaluation preserves the stable CNexp value and AD contract."""

import pytest
import torch

from dendra.models.mechanisms._cnexp import _bounded_phi1, _linear_exact_step
from dendra.models.mechanisms.ops import expinv


def _phi1_reference(z):
    # An independently evaluated integral series in float64. The test inputs
    # satisfy |z| <= 0.5; forty terms make truncation negligible through Hessians.
    term = torch.ones_like(z)
    result = term
    for order in range(1, 41):
        term = term * z / (order + 1)
        result = result + term
    return result


def _step(state, source, coefficient, dt):
    return _linear_exact_step(
        state, source + coefficient * state, source, coefficient, dt
    )


def _step_reference(state, source, coefficient, dt):
    z = coefficient * dt
    small = z.abs() <= 0.5
    small_z = torch.where(small, z, torch.zeros_like(z))
    safe_z = torch.where(small, torch.ones_like(z), z)
    integral = torch.where(
        small, _phi1_reference(small_z), torch.expm1(safe_z) / safe_z
    )
    return state * torch.exp(z) + source * dt * integral


def _bounded_points(dtype, device="cpu"):
    half = torch.tensor(0.5, dtype=dtype, device=device)
    inside = torch.nextafter(half, torch.zeros_like(half))
    points = half.new_tensor([0.0, 1e-12, 1e-8, 1e-6, 0.001, 0.125, 0.25])
    positive = torch.cat((points, inside[None], half[None]))
    return torch.cat((-positive.flip(0), positive))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("derivative_order", [0, 1, 2, 3])
def test_bounded_phi1_values_and_transform_derivatives(dtype, derivative_order):
    points = _bounded_points(dtype)
    actual_function = _bounded_phi1
    reference_function = _phi1_reference
    for _ in range(derivative_order):
        actual_function = torch.func.jacrev(actual_function)
        reference_function = torch.func.jacfwd(reference_function)
    actual = torch.vmap(actual_function)(points)
    reference = torch.vmap(reference_function)(points.double())
    tolerance = 4e-6 if dtype == torch.float32 else 3e-14
    assert actual.dtype == dtype and bool(torch.isfinite(actual).all())
    torch.testing.assert_close(
        actual.double(), reference, rtol=tolerance, atol=tolerance / 10
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_phi1_exact_zero_derivatives_and_jvp(dtype):
    zero = torch.tensor(0.0, dtype=dtype)
    torch.testing.assert_close(_bounded_phi1(zero), torch.ones_like(zero))
    torch.testing.assert_close(
        torch.func.jacrev(_bounded_phi1)(zero), zero.new_tensor(0.5)
    )
    torch.testing.assert_close(
        torch.func.hessian(_bounded_phi1)(zero), zero.new_tensor(1 / 3)
    )
    values = _bounded_points(dtype)
    tangent = torch.linspace(-0.75, 0.75, len(values), dtype=dtype)
    primal, actual = torch.func.jvp(_bounded_phi1, (values,), (tangent,))
    _, reference = torch.func.jvp(
        _phi1_reference, (values.double(),), (tangent.double(),)
    )
    assert primal.dtype == dtype and actual.dtype == dtype
    torch.testing.assert_close(actual.double(), reference, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bounded_phi1_reduced_precision_keeps_original_fallback(dtype):
    z = torch.tensor([-0.5, -0.1, 0.0, 0.1, 0.5], dtype=dtype)
    actual = _bounded_phi1(z)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, 1 / expinv(z), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("point", [-0.5, -0.125, 0.0, 0.125, 0.5])
def test_bounded_phi1_scalar_jvp_preserves_dtype_and_accuracy(dtype, point):
    z = torch.tensor(point, dtype=dtype)
    direction = z.new_tensor(0.125)
    actual, tangent = torch.func.jvp(_bounded_phi1, (z,), (direction,))
    reference, reference_tangent = torch.func.jvp(
        _phi1_reference, (z.double(),), (direction.double(),)
    )
    assert actual.dtype == dtype and tangent.dtype == dtype
    tolerance = 2e-6 if dtype == torch.float32 else 3e-14
    torch.testing.assert_close(actual.double(), reference, rtol=tolerance, atol=1e-15)
    torch.testing.assert_close(
        tangent.double(), reference_tangent, rtol=tolerance, atol=1e-15
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_phi1_tracing_scripting_and_empty_vmap(dtype):
    values = _bounded_points(dtype)
    traced = torch.jit.trace(_bounded_phi1, (values,))
    scripted = torch.jit.script(_bounded_phi1)
    expected = _bounded_phi1(values)
    torch.testing.assert_close(traced(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(scripted(values), expected, rtol=0, atol=0)
    # CNexp fields retain a logical compartment axis even when the outer
    # population-vmap axis is empty. Torch 2.14's logical-scalar batching rule
    # can select lane zero for scalar multiplication; the original expinv
    # formula has that same limitation. Do not move its workaround into math.
    empty = torch.vmap(_bounded_phi1)(values[:0, None])
    assert empty.shape == (0, 1) and empty.dtype == dtype


def _boundary_arguments(dtype, device="cpu"):
    half = torch.tensor(0.5, dtype=dtype, device=device)
    negative = -half
    positive_z = torch.stack(
        (
            torch.nextafter(half, torch.zeros_like(half)),
            half,
            torch.nextafter(half, torch.full_like(half, float("inf"))),
        )
    )
    negative_z = torch.stack(
        (
            torch.nextafter(negative, torch.full_like(half, -float("inf"))),
            negative,
            torch.nextafter(negative, torch.zeros_like(half)),
        )
    )
    z = torch.cat((negative_z, positive_z))
    return torch.full_like(z, 0.48), torch.full_like(z, 0.03), z, torch.ones_like(z)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_step_boundary_values_all_argument_gradients_and_hessians(dtype):
    arguments = _boundary_arguments(dtype)
    reference_arguments = tuple(value.double() for value in arguments)
    actual = _step(*arguments)
    expected = _step_reference(*reference_arguments)
    actual_gradients = torch.vmap(torch.func.jacrev(_step, argnums=(0, 1, 2, 3)))(
        *arguments
    )
    expected_gradients = torch.vmap(
        torch.func.jacfwd(_step_reference, argnums=(0, 1, 2, 3))
    )(*reference_arguments)
    actual_hessians = torch.vmap(torch.func.hessian(_step, argnums=(0, 1, 2, 3)))(
        *arguments
    )
    expected_hessians = torch.vmap(
        torch.func.hessian(_step_reference, argnums=(0, 1, 2, 3))
    )(*reference_arguments)
    tolerance = 8e-6 if dtype == torch.float32 else 8e-14
    actual_leaves = torch.utils._pytree.tree_leaves(
        (actual, actual_gradients, actual_hessians)
    )
    expected_leaves = torch.utils._pytree.tree_leaves(
        (expected, expected_gradients, expected_hessians)
    )
    for value, reference in zip(actual_leaves, expected_leaves, strict=True):
        assert value.dtype == dtype and bool(torch.isfinite(value).all())
        torch.testing.assert_close(
            value.double(), reference, rtol=tolerance, atol=tolerance / 10
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_step_mixed_rates_long_times_and_broadcast_jvp(dtype):
    state = torch.tensor([[0.48, 1.0, 0.7, 0.8]], dtype=dtype)
    source = state.new_tensor([0.03, 1e-6, 0.02, 0.02])
    coefficient = state.new_tensor([0.0, -1e4, -0.25, 2.0])
    dt = state.new_tensor([0.025, 1e4, 0.2, 0.75])
    arguments = state, source, coefficient, dt
    reference_arguments = tuple(value.double() for value in arguments)
    tangents = tuple(torch.full_like(value, 0.125) for value in arguments)
    actual, actual_jvp = torch.func.jvp(_step, arguments, tangents)
    expected, expected_jvp = torch.func.jvp(
        _step_reference,
        reference_arguments,
        tuple(value.double() for value in tangents),
    )
    for value, reference in ((actual, expected), (actual_jvp, expected_jvp)):
        assert value.shape == state.shape and value.dtype == dtype
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(value.double(), reference, rtol=2e-6, atol=1e-12)
    scalar_coefficient = state.new_tensor(-0.2)
    scalar_dt = state.new_tensor(0.025)
    actual = _step(state, source, scalar_coefficient, scalar_dt)
    expected = _step_reference(
        state.double(), source.double(), scalar_coefficient.double(), scalar_dt.double()
    )
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-12)


@pytest.mark.cpu
@pytest.mark.slow
def test_exact_step_cpu_inductor_boundary_values_and_all_input_gradients():
    arguments = tuple(
        value.requires_grad_() for value in _boundary_arguments(torch.float32)
    )
    compiled = torch.compile(_step, backend="inductor", fullgraph=True)
    actual = compiled(*arguments)
    actual_gradients = torch.autograd.grad(actual.sum(), arguments)
    reference_arguments = tuple(
        value.detach().double().requires_grad_() for value in arguments
    )
    expected = _step_reference(*reference_arguments)
    expected_gradients = torch.autograd.grad(expected.sum(), reference_arguments)
    for value, reference in zip(
        (actual, *actual_gradients), (expected, *expected_gradients), strict=True
    ):
        assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        torch.testing.assert_close(value.double(), reference, rtol=4e-6, atol=1e-8)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "compiled", [False, pytest.param(True, marks=pytest.mark.slow)]
)
def test_exact_step_cuda_boundary_values_jvp_and_all_input_gradients(compiled):
    arguments = tuple(
        value.requires_grad_() for value in _boundary_arguments(torch.float32, "cuda")
    )
    response = torch.compile(_step, fullgraph=True) if compiled else _step
    actual = response(*arguments)
    derivatives = torch.autograd.grad(actual.sum(), arguments)
    reference_arguments = tuple(
        value.detach().cpu().double().requires_grad_() for value in arguments
    )
    expected = _step_reference(*reference_arguments)
    expected_derivatives = torch.autograd.grad(expected.sum(), reference_arguments)
    for value, reference in zip(
        (actual, *derivatives), (expected, *expected_derivatives), strict=True
    ):
        assert value.device.type == "cuda" and value.dtype == torch.float32
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(
            value.detach().cpu().double(), reference.detach(), rtol=4e-6, atol=1e-8
        )
    # Forward AD is an eager/functional transform contract independent of the
    # compiled autograd path above; do not ask torch.compile to own dual tensors.
    tangents = tuple(torch.full_like(value, 0.125) for value in arguments)
    _, actual_jvp = torch.func.jvp(_step, arguments, tangents)
    _, expected_jvp = torch.func.jvp(
        _step_reference,
        reference_arguments,
        tuple(value.detach().cpu().double() for value in tangents),
    )
    torch.testing.assert_close(
        actual_jvp.detach().cpu().double(), expected_jvp.detach(), rtol=4e-6, atol=1e-8
    )


_COMPILED_DEVICES = [
    pytest.param("cpu", marks=pytest.mark.cpu),
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ],
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("device", _COMPILED_DEVICES)
def test_inductor_near_equilibrium_keeps_exact_float32_rounding(device):
    equilibrium = torch.tensor(0.48894766, dtype=torch.float32, device=device)
    decay = equilibrium.new_tensor([0.1233, 0.01, 1e-5, 1e-8, -0.125])
    state = torch.nextafter(equilibrium, equilibrium.new_tensor(float("inf")))
    arguments = (
        state.expand_as(decay).clone(),
        decay * equilibrium,
        -decay,
        torch.full_like(decay, 0.025),
    )
    arguments = tuple(value.requires_grad_() for value in arguments)
    compiled = torch.compile(_step, backend="inductor", fullgraph=True)
    actual = compiled(*arguments)
    actual_gradients = torch.autograd.grad(actual.sum(), arguments)
    reference_arguments = tuple(
        value.detach().double().requires_grad_() for value in arguments
    )
    expected = _step_reference(*reference_arguments)
    expected_gradients = torch.autograd.grad(expected.sum(), reference_arguments)
    torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
    for index, (value, reference) in enumerate(
        zip(actual_gradients, expected_gradients, strict=True)
    ):
        assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        if index == 3:
            # The authored float32 rate is intentionally retained. At these
            # near-equilibria, its product-rounding error can exceed the true
            # dt derivative itself (also in the pulled baseline). Bound that
            # quantization analytically rather than demanding a float64 rate.
            product = arguments[2].detach() * arguments[0].detach()
            spacing_up = (
                torch.nextafter(product, torch.full_like(product, float("inf")))
                - product
            )
            spacing_down = product - torch.nextafter(
                product, torch.full_like(product, -float("inf"))
            )
            spacing = torch.maximum(spacing_up, spacing_down)
            quantization = spacing.double().abs() / 2
            quantization *= torch.exp(
                reference_arguments[2] * reference_arguments[3]
            ).detach()
            arithmetic = 32 * torch.finfo(torch.float32).eps * reference.abs()
            assert bool(
                ((value.double() - reference).abs() <= quantization + arithmetic).all()
            )
        else:
            torch.testing.assert_close(value.double(), reference, rtol=2e-6, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("device", _COMPILED_DEVICES)
def test_inductor_authored_relaxation_preserves_equilibrium_and_sensitivities(device):
    from dendra.models.mechanisms._cnexp import build_cnexp

    solve = build_cnexp(["x"], {"x_inf", "tau"}, ["x' = (x_inf-x)/tau"])

    def step(state, equilibrium, tau, dt):
        return solve(None, dt, x=state, x_inf=equilibrium, tau=tau)["x"]

    equilibrium = torch.tensor(
        [0.9702991247177124, 0.7865960597991943], dtype=torch.float32, device=device
    )
    arguments = (
        equilibrium.clone(),
        equilibrium.clone(),
        equilibrium.new_tensor([0.04862092807888985, 0.08306379616260529]),
        equilibrium.new_tensor([0.023824255913496017, 0.04070126265287399]),
    )
    arguments = tuple(value.requires_grad_() for value in arguments)
    compiled = torch.compile(step, backend="inductor", fullgraph=True)
    actual = compiled(*arguments)
    derivatives = torch.autograd.grad(actual.sum(), arguments)
    reference_arguments = tuple(
        value.detach().double().requires_grad_() for value in arguments
    )
    state64, equilibrium64, tau64, dt64 = reference_arguments
    expected = equilibrium64 + (state64 - equilibrium64) * torch.exp(-dt64 / tau64)
    expected_derivatives = torch.autograd.grad(expected.sum(), reference_arguments)
    torch.testing.assert_close(actual, equilibrium, rtol=0, atol=0)
    for value, reference in zip(derivatives, expected_derivatives, strict=True):
        assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        torch.testing.assert_close(value.double(), reference, rtol=2e-6, atol=1e-8)
    state = arguments[0]
    for _ in range(32):
        state = compiled(state, *arguments[1:])
    torch.testing.assert_close(state, equilibrium, rtol=0, atol=0)
