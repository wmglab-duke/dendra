import pytest
import torch

from dendra.models.mechanisms.ops import expinv, exprelr

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@pytest.mark.parametrize("transform", [torch.func.jacrev, torch.func.jacfwd])
def test_exprelr_exact_singularity_has_correct_transform_derivatives(transform):
    x = torch.tensor(0.0, dtype=torch.float64)
    y = torch.tensor(10.0, dtype=torch.float64)

    value = exprelr(x, y)
    dx, dy = transform(lambda xy: exprelr(xy[0], xy[1]))(torch.stack((x, y)))

    torch.testing.assert_close(value, y)
    torch.testing.assert_close(dx, torch.tensor(-0.5, dtype=x.dtype))
    torch.testing.assert_close(dy, torch.tensor(1.0, dtype=x.dtype))


@pytest.mark.parametrize("transform", [torch.func.jacrev, torch.func.jacfwd])
def test_expinv_exact_singularity_has_correct_second_derivative(transform):
    x = torch.tensor(0.0, dtype=torch.float64)

    value = expinv(x)
    first = transform(expinv)(x)
    second = transform(transform(expinv))(x)

    torch.testing.assert_close(value, torch.tensor(1.0, dtype=x.dtype))
    torch.testing.assert_close(first, torch.tensor(-0.5, dtype=x.dtype))
    torch.testing.assert_close(second, torch.tensor(1.0 / 6.0, dtype=x.dtype))
    assert torch.isfinite(torch.stack((value, first, second))).all()


def test_removable_singularity_helpers_support_vmap_and_compile():
    x = torch.zeros(4, dtype=torch.float64)
    y = torch.full_like(x, 10.0)

    vmapped = torch.vmap(exprelr)(x, y)
    compiled = torch.compile(
        lambda x_, y_: exprelr(x_, y_) + expinv(x_),
        backend="aot_eager",
        fullgraph=True,
    )(x, y)

    torch.testing.assert_close(vmapped, y)
    torch.testing.assert_close(compiled, y + 1)


@pytest.mark.parametrize(
    ("function", "inputs"),
    [
        (exprelr, (torch.tensor(0.0), torch.tensor(10.0))),
        (expinv, (torch.tensor(0.0),)),
    ],
)
def test_removable_singularity_helpers_pass_gradcheck_and_gradgradcheck(
    function, inputs
):
    differentiable_inputs = tuple(
        value.to(dtype=torch.float64).requires_grad_() for value in inputs
    )

    assert torch.autograd.gradcheck(
        function,
        differentiable_inputs,
        eps=1e-7,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.autograd.gradgradcheck(
        function,
        differentiable_inputs,
        eps=1e-7,
        atol=1e-5,
        rtol=1e-4,
    )
