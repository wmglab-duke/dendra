"""Independent physiological-range oracles for reciprocal-exprel.

The implementation retains the ordinary exponential's nonlocal range limits;
these checks cover values and derivatives over the physiological input range.
"""

from decimal import Decimal, localcontext

import pytest
import torch

from dendra.models.mechanisms.ops import _reciprocal_exprel_even, expinv, exprelr, vtrap


def _decimal_reference(x, y):
    """Value and analytic partials at the exact binary inputs, not a series."""
    with localcontext() as context:
        context.prec = 110
        xd = Decimal.from_float(float(x))
        yd = Decimal.from_float(float(y))
        q = xd / yd
        if not q:
            r, first, second, third = (
                Decimal(1),
                Decimal(-1) / 2,
                Decimal(1) / 6,
                Decimal(0),
            )
        else:
            exponential = q.exp()
            denominator = exponential - 1
            r = q / denominator
            first = 1 / denominator - q * exponential / denominator**2
            second = (
                exponential * (q * (exponential + 1) - 2 * denominator) / denominator**3
            )
            third = (
                -(3 + q) * exponential / denominator**2
                + (6 + 6 * q) * exponential**2 / denominator**3
                - 6 * q * exponential**3 / denominator**4
            )
        return tuple(
            float(value)
            for value in (
                yd * r,
                first,
                r - q * first,
                second / yd,
                -q * second / yd,
                q * q * second / yd,
                third / (yd * yd),
            )
        )


def _grid(dtype, device="cpu"):
    cutoff = 1.0 if dtype == torch.float32 else 0.5
    boundary = torch.tensor(cutoff, dtype=dtype, device=device)
    neighbors = torch.stack(
        (
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, float("inf"))),
        )
    )
    tiny = torch.logspace(-14, -1, 29, dtype=dtype, device=device)
    ordinary = torch.linspace(0, 20, 129, dtype=dtype, device=device)
    positive = torch.cat((neighbors, tiny, ordinary))
    return torch.cat((-positive.flip(0), positive))


def _reference_arrays(x, y):
    # The Decimal oracle intentionally runs on host, independent of the tested
    # CPU/CUDA Tensor expressions and their exponential approximations.
    broadcast_x, broadcast_y = torch.broadcast_tensors(
        x.detach().cpu(), y.detach().cpu()
    )
    results = [
        _decimal_reference(xi, yi)
        for xi, yi in zip(broadcast_x.flatten(), broadcast_y.flatten(), strict=True)
    ]
    values = torch.tensor(results, dtype=torch.float64)
    return tuple(values[:, index].reshape(broadcast_x.shape) for index in range(7))


def _decimal_even_reference(q):
    with localcontext() as context:
        context.prec = 110
        value = Decimal.from_float(float(q))
        if not value:
            return float(Decimal(1) / 12), 0.0
        exponential = value.exp()
        denominator = exponential - 1
        reciprocal = value / denominator
        first = 1 / denominator - value * exponential / denominator**2
        numerator = reciprocal - 1 + value / 2
        even = numerator / value**2
        derivative = (first + Decimal(1) / 2) / value**2 - 2 * numerator / value**3
        return float(even), float(derivative)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("point", [-0.5, -0.125, 0.0, 0.125, 0.5])
def test_bernoulli_even_scalar_jvp_keeps_dtype_and_independent_accuracy(dtype, point):
    q = torch.tensor(point, dtype=dtype)
    direction = q.new_tensor(0.125)
    actual, tangent = torch.func.jvp(_reciprocal_exprel_even, (q,), (direction,))
    reference, derivative = _decimal_even_reference(q)
    assert actual.dtype == dtype and tangent.dtype == dtype
    tolerance = 2e-6 if dtype == torch.float32 else 3e-14
    torch.testing.assert_close(
        actual.double(),
        torch.tensor(reference, dtype=torch.float64),
        rtol=tolerance,
        atol=1e-15,
    )
    torch.testing.assert_close(
        tangent.double(),
        torch.tensor(derivative * 0.125, dtype=torch.float64),
        rtol=tolerance,
        atol=1e-15,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("point", [-5.0, -1.125, -0.125, 0.0, 0.125, 1.125, 5.0])
def test_reciprocal_exprel_public_scalar_jvp_keeps_authored_dtype(dtype, point):
    x = torch.tensor(point * 10, dtype=dtype)
    y = x.new_tensor(10)
    actual, tangent = torch.func.jvp(
        exprelr, (x, y), (x.new_tensor(0.125), y.new_tensor(-0.25))
    )
    reference = _decimal_reference(x, y)
    _assert_precision(actual, torch.tensor(reference[0], dtype=torch.float64), dtype)
    _assert_precision(
        tangent,
        torch.tensor(0.125 * reference[1] - 0.25 * reference[2], dtype=torch.float64),
        dtype,
        1,
    )
    q = x.new_tensor(point)
    inverse, inverse_tangent = torch.func.jvp(expinv, (q,), (q.new_tensor(0.125),))
    inverse_reference = _decimal_reference(q, 1)
    _assert_precision(
        inverse, torch.tensor(inverse_reference[0], dtype=torch.float64), dtype
    )
    _assert_precision(
        inverse_tangent,
        torch.tensor(0.125 * inverse_reference[1], dtype=torch.float64),
        dtype,
        1,
    )


def _assert_precision(value, reference, dtype, derivative_order=0):
    # Value and derivative tolerances are independent of runtime measurements.
    if dtype == torch.float32:
        rtol = (2e-6, 6e-6, 2e-5, 8e-5)[derivative_order]
        atol = (2e-6, 2e-6, 2e-7, 2e-7)[derivative_order]
    else:
        rtol = (1e-13, 2e-13, 8e-13, 8e-12)[derivative_order]
        atol = (2e-14, 2e-14, 4e-15, 4e-15)[derivative_order]
    assert value.dtype == dtype and bool(torch.isfinite(value).all())
    torch.testing.assert_close(
        value.detach().cpu().double(), reference, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [10.0, -10.0])
@pytest.mark.parametrize("transform", [torch.func.jacrev, torch.func.jacfwd])
def test_exprelr_grid_values_all_first_and_second_partials(dtype, scale, transform):
    y = torch.full_like(_grid(dtype), scale)
    x = _grid(dtype) * y
    reference = _reference_arrays(x, y)
    actual = exprelr(x, y)
    first = torch.vmap(transform(exprelr, argnums=(0, 1)))(x, y)
    second = torch.vmap(transform(transform(exprelr, argnums=(0, 1)), argnums=(0, 1)))(
        x, y
    )
    _assert_precision(actual, reference[0], dtype)
    _assert_precision(first[0], reference[1], dtype, 1)
    _assert_precision(first[1], reference[2], dtype, 1)
    _assert_precision(second[0][0], reference[3], dtype, 2)
    _assert_precision(second[0][1], reference[4], dtype, 2)
    _assert_precision(second[1][0], reference[4], dtype, 2)
    _assert_precision(second[1][1], reference[5], dtype, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_expinv_grid_values_through_third_derivative(dtype):
    x = _grid(dtype)
    reference = _reference_arrays(x, torch.ones_like(x))
    actual_function = expinv
    for order, expected in enumerate(
        (reference[0], reference[1], reference[3], reference[6])
    ):
        actual = torch.vmap(actual_function)(x)
        _assert_precision(actual, expected, dtype, order)
        actual_function = torch.func.jacrev(actual_function)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [10.0, -10.0])
def test_exprelr_zero_cross_partials_and_higher_derivatives(dtype, scale):
    x, y = torch.tensor(0.0, dtype=dtype), torch.tensor(scale, dtype=dtype)
    first = torch.func.jacrev(exprelr, argnums=(0, 1))(x, y)
    second = torch.func.hessian(exprelr, argnums=(0, 1))(x, y)
    third_mixed = torch.func.jacrev(
        lambda x_, y_: torch.func.hessian(exprelr, argnums=0)(x_, y_), argnums=1
    )(x, y)
    torch.testing.assert_close(exprelr(x, y), y, rtol=0, atol=0)
    torch.testing.assert_close(first[0], x.new_tensor(-0.5), rtol=0, atol=0)
    torch.testing.assert_close(first[1], x.new_tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(second[0][0], x.new_tensor(1 / (6 * scale)))
    for value in (second[0][1], second[1][0], second[1][1]):
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
    torch.testing.assert_close(third_mixed, x.new_tensor(-1 / (6 * scale**2)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("row_scale", [False, True])
def test_exprelr_broadcast_scale_gradient_hessian_and_jvp(dtype, row_scale):
    x = torch.tensor([[-20.0, -5.0, 0.0], [5.0, 12.0, 30.0]], dtype=dtype)
    y = x.new_tensor([[10.0], [-5.0]]) if row_scale else x.new_tensor(-10.0)
    weights = x.new_tensor([[0.25, -0.75, 1.25], [0.5, 0.125, -0.25]])
    arguments = torch.cat((x.flatten(), y.flatten()))

    def response(values):
        return (
            exprelr(values[:6].reshape_as(x), values[6:].reshape_as(y)) * weights
        ).sum()

    reference = _reference_arrays(x, y)
    expected_gradient = torch.zeros_like(arguments, dtype=torch.float64)
    expected_hessian = torch.zeros(
        (len(arguments), len(arguments)), dtype=torch.float64
    )
    expected_gradient[:6] = (weights.double() * reference[1]).flatten()
    expected_hessian[:6, :6] = torch.diag((weights.double() * reference[3]).flatten())
    for index in range(6):
        scale_index = 6 + index // 3 if row_scale else 6
        weight = weights.flatten()[index].double()
        expected_gradient[scale_index] += weight * reference[2].flatten()[index]
        expected_hessian[index, scale_index] = weight * reference[4].flatten()[index]
        expected_hessian[scale_index, index] = expected_hessian[index, scale_index]
        expected_hessian[scale_index, scale_index] += (
            weight * reference[5].flatten()[index]
        )
    actual_gradient = torch.func.jacrev(response)(arguments)
    actual_hessian = torch.func.hessian(response)(arguments)
    _assert_precision(actual_gradient, expected_gradient, dtype, 1)
    _assert_precision(actual_hessian, expected_hessian, dtype, 2)
    direction = torch.linspace(-0.25, 0.75, len(arguments), dtype=dtype)
    _, actual_jvp = torch.func.jvp(response, (arguments,), (direction,))
    _assert_precision(
        actual_jvp, (expected_gradient * direction.double()).sum(), dtype, 1
    )


def _near_unit_large_arguments(dtype, device="cpu"):
    info = torch.finfo(dtype)
    unit = torch.tensor(1.0, dtype=dtype, device=device)
    quotient = (
        torch.nextafter(unit, torch.full_like(unit, float("inf")))
        if dtype == torch.float32
        else unit
    )
    x = torch.tensor(
        [info.max / 2, info.max * 0.75, -info.max / 2, -info.max * 0.75],
        dtype=dtype,
        device=device,
    )
    return x, x / quotient


def _check_near_unit_large_first_partials(x, y, response):
    x = x.detach().requires_grad_()
    y = y.detach().requires_grad_()
    actual = response(x, y)
    derivatives = torch.autograd.grad(actual.sum(), (x, y))
    reference = _reference_arrays(x, y)
    _assert_precision(actual, reference[0], x.dtype)
    _assert_precision(derivatives[0], reference[1], x.dtype, 1)
    _assert_precision(derivatives[1], reference[2], x.dtype, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exprelr_near_unit_large_numerator_avoids_new_first_gradient_overflow(dtype):
    x, y = _near_unit_large_arguments(dtype)
    # This is a regression relative to the original implementation, not a
    # claim to cure every underflow/overflow pathology of quotient AD.
    legacy_x, legacy_y = x.requires_grad_(), y.requires_grad_()
    legacy = _legacy_exprelr(legacy_x, legacy_y)
    legacy_derivatives = torch.autograd.grad(legacy.sum(), (legacy_x, legacy_y))
    assert all(
        bool(torch.isfinite(value).all()) for value in (legacy, *legacy_derivatives)
    )
    _check_near_unit_large_first_partials(x, y, exprelr)


def _legacy_exprelr(x, y):
    q = x / y
    small = q.abs() < 1e-6
    q_safe = torch.where(small, torch.ones_like(q), q)
    val = x / torch.expm1(q_safe)
    approx = torch.addcmul(y - x / 2, x, q, value=1 / 12)
    return torch.where(small, approx, val)


def _legacy_expinv(x):
    small = x.abs() < 1e-6
    x_safe = torch.where(small, torch.ones_like(x), x)
    val = x / torch.expm1(x_safe)
    approx = torch.addcmul(1 - 0.5 * x, x, x, value=1 / 12)
    return torch.where(small, approx, val)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.complex64, torch.complex128]
)
def test_reciprocal_exprel_other_dtypes_keep_legacy_value_path(dtype):
    x = torch.tensor([-0.1, 0.0, 0.1], dtype=dtype)
    if dtype.is_complex:
        x = x + 0.1j
    y = torch.full_like(x, 2.0)
    torch.testing.assert_close(exprelr(x, y), _legacy_exprelr(x, y), rtol=0, atol=0)
    torch.testing.assert_close(expinv(x), _legacy_expinv(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reciprocal_exprel_script_trace_scalar_scale_and_empty_vmap(dtype):
    x = _grid(dtype)
    y = x.new_tensor(10.0)
    traced = torch.jit.trace(exprelr, (x, y))
    scripted = torch.jit.script(exprelr)
    for response in (traced, scripted):
        torch.testing.assert_close(response(x, y), exprelr(x, y), rtol=0, atol=0)
    scripted_inverse = torch.jit.script(expinv)
    torch.testing.assert_close(scripted_inverse(x), expinv(x), rtol=0, atol=0)
    torch.testing.assert_close(exprelr(x, 10), exprelr(x, y), rtol=0, atol=0)
    assert vtrap is exprelr
    # Real mechanism fields retain their logical compartment dimension under
    # empty population vmap; logical-scalar constant arithmetic is a separate
    # pre-existing Torch batching limitation, not this helper's model contract.
    empty = torch.vmap(exprelr)(x[:0, None], x[:0, None])
    assert empty.shape == (0, 1) and empty.dtype == dtype


def test_exprelr_float64_gradcheck_and_gradgradcheck_at_singularities_and_boundaries():
    x = torch.tensor([0.0, -5.0, 5.0, 1e-12], dtype=torch.float64, requires_grad=True)
    y = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(exprelr, (x, y), eps=1e-6, rtol=1e-5, atol=1e-6)
    assert torch.autograd.gradgradcheck(exprelr, (x, y), eps=1e-6, rtol=1e-5, atol=1e-6)


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exprelr_cpu_inductor_grid_values_and_all_argument_gradients(dtype):
    x = (_grid(dtype) * 10).requires_grad_()
    y = torch.full_like(x, 10).requires_grad_()
    response = torch.compile(exprelr, fullgraph=True)
    actual = response(x, y)
    gradients = torch.autograd.grad(actual.sum(), (x, y))
    reference = _reference_arrays(x, y)
    _assert_precision(actual, reference[0], dtype)
    _assert_precision(gradients[0], reference[1], dtype, 1)
    _assert_precision(gradients[1], reference[2], dtype, 1)


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exprelr_cpu_inductor_large_numerator_first_partials_remain_finite(dtype):
    x, y = _near_unit_large_arguments(dtype)
    _check_near_unit_large_first_partials(x, y, torch.compile(exprelr, fullgraph=True))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "compiled", [False, pytest.param(True, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exprelr_cuda_grid_value_all_gradients_and_eager_jvp(dtype, compiled):
    x = (_grid(dtype, "cuda") * 10).requires_grad_()
    y = torch.full_like(x, 10).requires_grad_()
    response = torch.compile(exprelr, fullgraph=True) if compiled else exprelr
    actual = response(x, y)
    gradients = torch.autograd.grad(actual.sum(), (x, y))
    reference = _reference_arrays(x, y)
    _assert_precision(actual, reference[0], dtype)
    _assert_precision(gradients[0], reference[1], dtype, 1)
    _assert_precision(gradients[1], reference[2], dtype, 1)
    dx, dy = torch.full_like(x, 0.125), torch.full_like(y, -0.25)
    _, actual_jvp = torch.func.jvp(exprelr, (x, y), (dx, dy))
    expected_jvp = 0.125 * reference[1] - 0.25 * reference[2]
    _assert_precision(actual_jvp, expected_jvp, dtype, 1)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "compiled", [False, pytest.param(True, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exprelr_cuda_large_numerator_first_partials_remain_finite(dtype, compiled):
    x, y = _near_unit_large_arguments(dtype, "cuda")
    response = torch.compile(exprelr, fullgraph=True) if compiled else exprelr
    _check_near_unit_large_first_partials(x, y, response)
