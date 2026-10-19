from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from dendra.models.mechanisms._spatial import SpatialOperator1D, SpatialOperatorTree


@pytest.fixture(autouse=True)
def _restore_thread_count():
    previous = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("solver", ["dense", "thomas", "pcr"])
@pytest.mark.parametrize("method", ["implicit", "explicit"])
def test_chain_preparation_transforms_and_exact_imperative_parity(
    threads, solver, method
):
    torch.set_num_threads(threads)
    c = torch.tensor([[1.0, 2.0, 3.0, 1.5], [2.0, 1.5, 0.5, 3.0]], dtype=torch.float64)
    values = (
        c,
        torch.tensor(0.02, dtype=c.dtype),
        torch.tensor(0.3, dtype=c.dtype),
        torch.tensor([[1.0, 1.2, 1.4, 1.1]], dtype=c.dtype),
        torch.tensor([[1.2, 1.1, 1.4, 1.3]], dtype=c.dtype),
        torch.tensor(0.08, dtype=c.dtype),
        torch.tensor(0.4, dtype=c.dtype),
    )
    op = SpatialOperator1D(solver=solver)
    op.configure_diffusion(*values[:3], *values[3:5])
    if method == "implicit":
        op.configure_relaxation(*values[5:])
    selected = op._functional_solver()
    frozen = {name: tensor.clone() for name, tensor in op._buffers.items()}

    def run(c, dt, diffusion, diameter, length, rate, target):
        work = op.prepare_diffusion(c, dt, diffusion, diameter, length)
        if method == "implicit":
            work = op.prepare_relaxation(work, rate, target)
        trace = [c]
        for _ in range(3):
            if method == "implicit":
                c = op.diffuse_implicit_prepared(c, work, solve=selected)
            else:
                c = op.diffuse_explicit_prepared(c, work)
            trace.append(c)
        return torch.stack(trace)

    def oracle(c, dt, diffusion, diameter, length, rate, target):
        # The independent PyTorch dense solve supplies a valid forward-AD
        # reference too; the raw imperative native dispatcher is reverse-only.
        other = SpatialOperator1D(solver="dense")
        other.configure_diffusion(c, dt, diffusion, diameter, length)
        if method == "implicit":
            other.configure_relaxation(rate, target)
        trace = [c]
        for _ in range(3):
            c = getattr(other, f"diffuse_{method}_configured")(c)
            trace.append(c)
        return torch.stack(trace)

    _check(run, oracle, values)
    for name, value in frozen.items():
        torch.testing.assert_close(op._buffers[name], value, rtol=0, atol=0)
    _assert_numerical_workspace_is_explicit(op, run, values)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("solver", ["dense", "auto"])
@pytest.mark.parametrize("method", ["implicit", "explicit"])
def test_tree_preparation_transforms_and_independent_dense_oracle(
    threads, solver, method
):
    torch.set_num_threads(threads)
    c = torch.tensor([[1.0, 2.0, 3.0, 1.5], [2.0, 1.5, 0.5, 3.0]], dtype=torch.float64)
    parent = torch.tensor([2, 2, -1, 1])
    volume = torch.tensor(
        [[1.0, 0.8, 0.0 if method == "implicit" else 1.0, 1.1]], dtype=c.dtype
    )
    edge = torch.tensor([[0.3, 0.4, 0.5]], dtype=c.dtype)
    model = SimpleNamespace()
    values = (
        c,
        torch.tensor(0.02, dtype=c.dtype),
        torch.tensor(0.3, dtype=c.dtype),
        volume,
        edge,
        torch.tensor(0.08, dtype=c.dtype),
        torch.tensor(0.4, dtype=c.dtype),
    )
    op = SpatialOperatorTree(solver=solver)
    op.configure_diffusion(
        c,
        values[1],
        values[2],
        model,
        volume=volume,
        edge_factor=edge,
        parent_index=parent,
    )
    if method == "implicit":
        op.configure_relaxation(*values[5:])
    else:
        op.validate_explicit()
    selected = op._functional_solver()
    frozen = {name: tensor.clone() for name, tensor in op._buffers.items()}

    def run(c, dt, diffusion, volume, edge, rate, target):
        work = op.prepare_diffusion(
            c, dt, diffusion, model, volume=volume, edge_factor=edge
        )
        assert not (set(work) & op._FUNCTIONAL_TOPOLOGY_BUFFERS)
        if method == "implicit":
            work = op.prepare_relaxation(work, rate, target)
        trace = [c]
        for _ in range(3):
            if method == "implicit":
                c = op.diffuse_implicit_prepared(c, work, solve=selected)
            else:
                c = op.diffuse_explicit_prepared(c, work)
            trace.append(c)
        return torch.stack(trace)

    # Compact edge order is increasing child storage index: (2,0),(2,1),(1,3).
    incidence = torch.tensor(
        [[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
        dtype=c.dtype,
    )

    def oracle(c, dt, diffusion, volume, edge, rate, target):
        laplace = (
            incidence.T @ torch.diag_embed((diffusion * edge).squeeze(0)) @ incidence
        )
        v = volume.expand_as(c)
        matrix = torch.diag_embed(v / dt + v * rate) + laplace
        trace = [c]
        for _ in range(3):
            if method == "implicit":
                c = torch.linalg.solve(
                    matrix, (v / dt * c + v * rate * target).unsqueeze(-1)
                ).squeeze(-1)
            else:
                c = c - dt * (laplace @ c.unsqueeze(-1)).squeeze(-1) / v
            trace.append(c)
        return torch.stack(trace)

    _check(run, oracle, values)
    for name, value in frozen.items():
        torch.testing.assert_close(op._buffers[name], value, rtol=0, atol=0)
    _assert_numerical_workspace_is_explicit(op, run, values)


def test_spd_functional_solver_fails_closed_without_changing_imperative_selection():
    c = torch.ones((1, 3), dtype=torch.float64)
    op = SpatialOperator1D(solver="spd")
    op.configure_diffusion(c, 0.02, 0.3, c, c)
    assert op.solver_name == "spd_cpu"
    selected = op._solve
    with pytest.raises(
        RuntimeError, match="SPD solver currently remains imperative-only"
    ):
        op._functional_solver()
    assert op._solve is selected
    torch.testing.assert_close(op.diffuse_implicit_configured(c), c)


def test_zero_volume_tree_workspace_has_finite_float32_zero_cotangent_backward():
    c = torch.tensor([[1.0, 2.0, 0.7, 1.5]], dtype=torch.float32)
    volume = torch.tensor([[1.0, 0.8, 0.0, 1.1]], dtype=c.dtype)
    edge = torch.tensor([[0.3, 0.4, 0.5]], dtype=c.dtype)
    op = SpatialOperatorTree(solver="dense")
    op.configure_diffusion(
        c,
        0.02,
        0.3,
        None,
        volume=volume,
        edge_factor=edge,
        parent_index=torch.tensor([2, 2, -1, 1]),
    )

    def loss(volume):
        workspace = op.prepare_diffusion(
            c,
            0.02,
            0.3,
            None,
            volume=volume,
            edge_factor=edge,
        )
        # Full preparation VJPs may supply explicit zero cotangents to
        # coefficients unused by implicit execution. Those paths must remain
        # finite too, rather than differentiating 1 / finfo(float32).tiny.
        unused = workspace["inv_volume"].sum() * 0.0
        return op.diffuse_implicit_prepared(c, workspace, solve=None).sum() + unused

    gradient = torch.func.grad(loss)(volume)
    assert torch.isfinite(gradient).all()
    assert op._buffers["inv_volume"][0, 2] == 0.0


def _assert_numerical_workspace_is_explicit(op, run, values):
    expected = run(*values)
    for name, value in tuple(op._buffers.items()):
        if value.is_floating_point():
            op._buffers[name] = torch.full_like(value, torch.nan)
    # Only immutable integer topology may be resident. None of the old
    # numerical coefficients or geometry snapshots may enter preparation or
    # execution when the caller supplies its complete numerical inputs.
    torch.testing.assert_close(run(*values), expected, rtol=0, atol=0)


def _check(run, oracle, values):
    actual = run(*values)
    expected = oracle(*values)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    weights = torch.linspace(0.2, 1.0, actual.numel(), dtype=actual.dtype).reshape_as(
        actual
    )

    def loss(*args):
        return (run(*args) * weights).sum()

    transformed = torch.func.grad(loss, argnums=tuple(range(len(values))))(*values)
    trainable = tuple(value.clone().requires_grad_() for value in values)
    reference_loss = (oracle(*trainable) * weights).sum()
    reference = torch.autograd.grad(reference_loss, trainable, allow_unused=True)
    for got, want, value in zip(transformed, reference, values):
        want = torch.zeros_like(value) if want is None else want
        torch.testing.assert_close(got, want, rtol=2e-11, atol=2e-11)
    directions = tuple(torch.full_like(value, 0.01) for value in values)
    _, actual_jvp = torch.func.jvp(run, values, directions)
    _, reference_jvp = torch.func.jvp(oracle, values, directions)
    torch.testing.assert_close(actual_jvp, reference_jvp, rtol=2e-11, atol=2e-11)
    lanes = tuple(torch.stack((value, value * 1.01, value * 0.99)) for value in values)
    mapped = torch.func.vmap(run)(*lanes)
    serial = torch.stack(
        [run(*(value[index] for value in lanes)) for index in range(3)]
    )
    torch.testing.assert_close(mapped, serial, rtol=2e-11, atol=2e-11)

    # Vary only diffusion while sharing the state and geometry. In-place
    # assembly into an unbatched RHS-derived allocation cannot support this.
    def change_diffusion(diffusion):
        return run(*values[:2], diffusion, *values[3:])

    partial = torch.func.vmap(change_diffusion)(lanes[2])
    partial_reference = torch.stack([change_diffusion(value) for value in lanes[2]])
    torch.testing.assert_close(partial, partial_reference, rtol=2e-11, atol=2e-11)
    empty = torch.func.vmap(run)(*(value.unsqueeze(0)[:0] for value in values))
    assert empty.shape == (0, *actual.shape)
    graph = make_fx(run)(*values)
    torch.testing.assert_close(graph(*values), actual, rtol=0, atol=0)
