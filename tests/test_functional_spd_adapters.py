"""Native SPD selection and CUDA adapter algebra on a CPU reference launcher.

The injected launcher exercises the opaque CUDA adapter's AD and capture
contracts. Actual CUDA kernel parity is checked separately on CUDA hosts.
"""

from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from dendra.models.integrators import _cuda_functional as cuda
from dendra.models.integrators import implicit
from dendra.models.integrators._symmetry import _symmetric_part
from tests.func.test_cuda_functional_reference import _tridiagonal_matrix
from tests.test_integrator_explicit_implicit import CableModel, LinearMechanism

pytestmark = pytest.mark.cpu


def _inputs(size=4, dtype=torch.float64, device="cpu"):
    band = torch.linspace(-0.09, 0.03, 2 * (size - 1) * 3, dtype=dtype, device=device)
    lower = band.reshape(2, size - 1, 3)
    seed = torch.linspace(-0.1, 0.2, 2 * size * 9, dtype=dtype, device=device)
    seed = seed.reshape(2, size, 3, 3)
    main = seed @ seed.transpose(-1, -2) + 2.5 * torch.eye(
        3, dtype=dtype, device=device
    )
    rhs = torch.linspace(-0.8, 0.9, 2 * size * 3, dtype=dtype, device=device)
    return lower, main, lower.clone(), rhs.reshape(2, size, 3)


def _dense(lower, main, upper, rhs):
    # The symmetric projection is the extension specified by the native SPD
    # main-block gradient. Lower and upper retain independent cotangents.
    main = 0.5 * (main + main.transpose(-1, -2))
    matrix = _tridiagonal_matrix(lower, main, upper, block=True)
    solved = torch.linalg.solve(matrix, rhs.flatten(-2).unsqueeze(-1)).squeeze(-1)
    return solved.reshape_as(rhs)


@pytest.fixture
def cpu_spd_launcher(monkeypatch):
    calls = {"factor": 0, "cached": 0}

    def factor(lower, main, upper, rhs):
        calls["factor"] += 1
        return _dense(lower, main, upper, rhs), main.flatten(-2).clone()

    def cached(lower, upper, factor, rhs):
        calls["cached"] += 1
        return _dense(lower, factor.unflatten(-1, (3, 3)), upper, rhs)

    monkeypatch.setattr(cuda, "_launch_block_spd", factor)
    monkeypatch.setattr(cuda, "_launch_block_spd_chol", cached)
    return calls


def _solve(*values):
    # The production public entrypoint validates CUDA placement. The internal
    # entrypoint is used only to test its AD with a CPU launcher.
    return cuda._apply_block_spd(*values)[0]


@pytest.mark.parametrize("size", [1, 4])
def test_cuda_spd_adapter_preserves_cached_vjp_jvp_and_input_ownership(
    cpu_spd_launcher, size
):
    values = tuple(value.requires_grad_() for value in _inputs(size))
    snapshots = tuple((value.clone(), value._version) for value in values)
    expected_values = tuple(value.detach().clone().requires_grad_() for value in values)
    actual, expected = _solve(*values), _dense(*expected_values)
    weights = torch.linspace(0.2, 0.9, actual.numel(), dtype=actual.dtype).reshape_as(
        actual
    )
    gradients = torch.autograd.grad((actual * weights).sum(), values)
    reference = torch.autograd.grad(
        (expected * weights).sum(), expected_values, materialize_grads=True
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(gradients, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-14, atol=2e-14)
    assert cpu_spd_launcher == {"factor": 1, "cached": 1}

    directions = tuple(torch.full_like(value, 0.013) for value in values)
    # An asymmetric main tangent verifies the forward dual of the symmetric
    # backward rule, rather than only tangents on the SPD manifold.
    directions[1][..., 0, 2] = -0.017
    _, tangent = torch.func.jvp(_solve, values, directions)
    _, target = torch.func.jvp(_dense, values, directions)
    torch.testing.assert_close(tangent, target, rtol=2e-14, atol=2e-14)
    for value, (snapshot, version) in zip(values, snapshots, strict=True):
        assert value._version == version
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("size", [1, 4])
def test_cuda_spd_adapter_vmap_partial_empty_and_cotangents(cpu_spd_launcher, size):
    values = _inputs(size)
    mapped = tuple(torch.stack((value, value * 1.01, value * 0.99)) for value in values)
    expected = torch.stack([_solve(*(value[i] for value in mapped)) for i in range(3)])
    torch.testing.assert_close(torch.func.vmap(_solve)(*mapped), expected)

    def partial(main):
        return _solve(values[0], main, values[2], values[3])

    torch.testing.assert_close(
        torch.func.vmap(partial)(mapped[1]),
        torch.stack([partial(main) for main in mapped[1]]),
    )
    empty = torch.func.vmap(_solve)(*(value[:0] for value in mapped))
    assert empty.shape == (0, *values[-1].shape)

    output, pullback = torch.func.vjp(_solve, *values)
    cotangents = torch.stack((torch.ones_like(output), output, output.square()))
    actual = torch.func.vmap(pullback)(cotangents)
    expected = tuple(
        torch.stack(items) for items in zip(*(pullback(item) for item in cotangents))
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize(
    "capture", ["fx", "aot_eager", pytest.param("inductor", marks=pytest.mark.slow)]
)
def test_cuda_spd_adapter_capture_retains_registered_cached_backward(
    cpu_spd_launcher, capture
):
    values = tuple(value.requires_grad_() for value in _inputs())
    reference_values = tuple(
        value.detach().clone().requires_grad_() for value in values
    )
    if capture == "fx":
        with torch.no_grad():
            run = make_fx(_solve)(*values)
    elif capture == "aot_eager":
        run = torch.compile(_solve, backend="aot_eager", fullgraph=True)
    else:
        run = torch.compile(
            _solve, backend="inductor", fullgraph=True, options={"cpp_wrapper": True}
        )
    actual, expected = run(*values), _dense(*reference_values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual.square().sum(), values)
    target_gradients = torch.autograd.grad(expected.square().sum(), reference_values)
    for got, want in zip(actual_gradients, target_gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_symmetric_projection_preserves_large_finite_and_subnormal_values(dtype):
    largest = torch.finfo(dtype).max
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
    )
    values = torch.tensor([[0.0, 0.75 * largest], [0.75 * largest, 0.0]], dtype=dtype)
    projected = _symmetric_part(values)
    assert torch.isfinite(projected).all()
    torch.testing.assert_close(projected, values, rtol=0, atol=0)
    tiny = smallest.expand(3, 3).clone()
    torch.testing.assert_close(_symmetric_part(tiny), tiny, rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_symmetric_projection_preserves_large_finite_and_subnormal_values(
    dtype,
):
    largest = torch.finfo(dtype).max * 0.75
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
    )
    values = torch.tensor(
        [
            [smallest, largest, largest],
            [largest, smallest, -largest],
            [-largest, -largest, smallest],
        ],
        dtype=dtype,
    )
    expected = torch.tensor(
        [
            [smallest, largest, 0.0],
            [largest, smallest, -largest],
            [0.0, -largest, smallest],
        ],
        dtype=dtype,
    )
    compiled = torch.compile(
        _symmetric_part,
        backend="inductor",
        fullgraph=True,
        options={"cpp_wrapper": True},
    )
    actual = compiled(values)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_spd_adapter_large_finite_gradient_and_tangent(cpu_spd_launcher, dtype):
    largest = torch.finfo(dtype).max
    empty = torch.empty((1, 0, 3), dtype=dtype)
    main = torch.eye(3, dtype=dtype).reshape(1, 1, 3, 3)
    rhs = torch.full((1, 1, 3), 0.75, dtype=dtype)
    values = tuple(
        value.requires_grad_() for value in (empty, main, empty.clone(), rhs)
    )
    output = _solve(*values)
    cotangent = torch.full_like(output, largest * 0.8)
    gradient = torch.autograd.grad(output, values, grad_outputs=cotangent)[1]
    target = -(cotangent.unsqueeze(-1) * output.unsqueeze(-2))
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, target, rtol=0, atol=0)

    rhs = torch.full_like(rhs, 0.05)
    values = (empty, main, empty.clone(), rhs)
    direction = torch.full_like(main, largest * 0.8)
    directions = (
        torch.zeros_like(empty),
        direction,
        torch.zeros_like(empty),
        torch.zeros_like(rhs),
    )
    _, tangent = torch.func.jvp(_solve, values, directions)
    target = -(direction @ rhs.unsqueeze(-1)).squeeze(-1)
    assert torch.isfinite(tangent).all()
    torch.testing.assert_close(tangent, target, rtol=0, atol=0)


@pytest.mark.parametrize("block", [False, True])
def test_spd_solver_binding_preserves_native_selection(block):
    model = CableModel(shape=(2, 4))
    integrator_type = implicit._bwd_euler_bt if block else implicit._bwd_euler_ub
    pure = implicit._FUNCTIONAL_BLOCK_SPD if block else implicit._FUNCTIONAL_SCALAR_SPD
    if pure is None:
        pytest.skip("dendra-solvers pure SPD facade is required")
    integrator = integrator_type(model, LinearMechanism(), method="spd").to(
        dtype=torch.float64
    )
    integrator._select_solver(model.device())
    selected = integrator._solve
    assert integrator._functional_solver() is pure
    assert integrator._solve is selected
    assert pure not in implicit._BWD_EULER_INCREMENT_SOLVERS


@pytest.mark.parametrize("block", [False, True])
def test_spd_solver_binding_rejects_older_facades_without_changing_native_selection(
    monkeypatch, block
):
    native_name = "solve_bt_spd_consume_unchecked" if block else "solve_tri_spd"
    if not callable(getattr(implicit.dendra_solvers, native_name, None)):
        pytest.skip("native dendra-solvers SPD implementation is required")
    model = CableModel(shape=(2, 4))
    integrator_type = implicit._bwd_euler_bt if block else implicit._bwd_euler_ub
    integrator = integrator_type(model, LinearMechanism(), method="spd")
    integrator._select_solver(model.device())
    selected = integrator._solve
    assert selected is getattr(implicit.dendra_solvers, native_name)
    # An older installed wheel has no certified pure facade. Exercise the
    # same fail-closed path with either an old wheel or the current package.
    marker = "_FUNCTIONAL_BLOCK_SPD" if block else "_FUNCTIONAL_SCALAR_SPD"
    monkeypatch.setattr(implicit, marker, None)
    with pytest.raises(RuntimeError, match="pure SPD facades"):
        integrator._functional_solver()
    assert integrator._solve is selected


def test_cuda_spd_solver_binding_preserves_consuming_native_selection(monkeypatch):
    def consuming(*values):
        raise AssertionError("Selection must not launch a solver")

    monkeypatch.setattr(implicit, "solve_bt_spd_cuda_consume_unchecked", consuming)
    model = CableModel(shape=(2, 4))
    integrator = implicit._bwd_euler_bt(model, LinearMechanism(), method="spd")
    integrator._select_solver("cuda")
    assert integrator._solve is consuming
    assert integrator._functional_solver() is cuda.block_spd_solve
    assert integrator._solve is consuming


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [1, 4])
def test_real_cuda_spd_adapter_preserves_native_primal_gradients_and_transforms(
    dtype, size
):
    from dendra.models.integrators.triton import solve_bt_spd_cuda

    values = tuple(value.requires_grad_() for value in _inputs(size, dtype, "cuda"))
    native_values = tuple(value.detach().clone().requires_grad_() for value in values)
    actual, expected = cuda.block_spd_solve(*values), solve_bt_spd_cuda(*native_values)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.square().sum(), values)
    reference = torch.autograd.grad(expected.square().sum(), native_values)
    for got, want in zip(gradients, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)
    directions = tuple(torch.full_like(value, 0.013) for value in values)
    directions[1][..., 0, 2] = -0.017
    _, tangent = torch.func.jvp(cuda.block_spd_solve, values, directions)
    _, target = torch.func.jvp(_dense, values, directions)
    torch.testing.assert_close(tangent, target, rtol=tolerance, atol=tolerance)
    mapped = tuple(torch.stack((value, value * 1.01)) for value in values)
    torch.testing.assert_close(
        torch.func.vmap(cuda.block_spd_solve)(*mapped),
        torch.stack(
            [cuda.block_spd_solve(*(value[i] for value in mapped)) for i in range(2)]
        ),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_cuda_spd_extreme_finite_matrix_and_cotangent(dtype):
    from dendra.models.integrators.triton import solve_bt_spd_cuda

    largest = torch.finfo(dtype).max
    main = (
        torch.tensor(
            [[[[0.8, 0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 0.8]]]],
            device="cuda",
            dtype=dtype,
        )
        * largest
    )
    rhs = torch.tensor([[[0.08, -0.03, 0.04]]], device="cuda", dtype=dtype) * largest
    band = torch.empty((1, 0, 3), device="cuda", dtype=dtype)
    reference = torch.linalg.solve(
        main[:, 0] / largest, (rhs[:, 0] / largest).unsqueeze(-1)
    ).reshape_as(rhs)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    for solve in (solve_bt_spd_cuda, cuda.block_spd_solve):
        actual = solve(band, main, band.clone(), rhs)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)

        identity = torch.eye(3, dtype=dtype, device="cuda").reshape(1, 1, 3, 3)
        rhs_small = torch.full_like(rhs, 0.75)
        values = tuple(
            value.requires_grad_()
            for value in (band, identity, band.clone(), rhs_small)
        )
        output = solve(*values)
        cotangent = torch.full_like(output, largest * 0.8)
        gradient = torch.autograd.grad(output, values, grad_outputs=cotangent)[1]
        target = -(cotangent.unsqueeze(-1) * output.unsqueeze(-2))
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, target, rtol=tolerance, atol=0)
