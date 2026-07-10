import pytest
import torch

from dendra.models.integrators.imex import (
    A_mv,
    arnoldi,
    expm_krylov_arnoldi,
    expm_krylov_lanczos,
    lanczos,
    phi1_krylov_arnoldi,
    phi1_krylov_arnoldi_g,
    phi1_krylov_lanczos,
    phi1_krylov_lanczos_g,
)

DTYPE = torch.float64


def _dense_matrix(diag, g_left, g_right):
    batch, size = diag.shape
    matrix = torch.diag_embed(diag)
    rows = torch.arange(size - 1)
    matrix[:, rows, rows + 1] = g_right
    matrix[:, rows + 1, rows] = g_left
    return matrix


def _workspace(batch, size, krylov_dim):
    return (
        torch.zeros(batch, size, krylov_dim, dtype=DTYPE),
        torch.zeros(batch, krylov_dim, krylov_dim, dtype=DTYPE),
    )


def test_tridiagonal_matvec_matches_dense_batched_product():
    diag = torch.tensor([[-2.0, -3.0, -4.0], [1.0, 2.0, 3.0]], dtype=DTYPE)
    left = torch.tensor([[0.2, 0.4], [-0.5, 0.3]], dtype=DTYPE)
    right = torch.tensor([[0.1, 0.3], [0.7, -0.2]], dtype=DTYPE)
    vector = torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.4, 0.8]], dtype=DTYPE)

    expected = torch.bmm(_dense_matrix(diag, left, right), vector.unsqueeze(-1))
    assert torch.allclose(A_mv(vector, diag, left, right), expected.squeeze(-1))


def test_arnoldi_builds_orthonormal_basis_and_hessenberg_projection():
    vector = torch.tensor([[1.0, -0.5, 0.25, 0.75]], dtype=DTYPE)
    diag = torch.tensor([[-1.0, -2.0, -3.0, -4.0]], dtype=DTYPE)
    left = torch.tensor([[0.2, 0.3, 0.4]], dtype=DTYPE)
    right = torch.tensor([[0.5, 0.6, 0.7]], dtype=DTYPE)
    V_buf, H_buf = _workspace(1, 4, 3)

    V, H, beta = arnoldi(vector, 3, diag, left, right, V_buf, H_buf)
    matrix = _dense_matrix(diag, left, right)

    assert beta.item() == pytest.approx(torch.linalg.norm(vector).item())
    assert torch.allclose(V.transpose(1, 2) @ V, torch.eye(3, dtype=DTYPE)[None])
    projected = V.transpose(1, 2) @ matrix @ V
    assert torch.allclose(H[:, :2, :2], projected[:, :2, :2], atol=1e-12)


def test_lanczos_builds_symmetric_tridiagonal_projection():
    vector = torch.tensor([[1.0, 0.5, -0.25, 0.75]], dtype=DTYPE)
    diag = torch.tensor([[-1.0, -2.0, -3.0, -4.0]], dtype=DTYPE)
    edges = torch.tensor([[0.2, 0.3, 0.4]], dtype=DTYPE)
    V_buf, T_buf = _workspace(1, 4, 3)

    V, T, _ = lanczos(vector, 3, diag, edges, edges, V_buf, T_buf)
    matrix = _dense_matrix(diag, edges, edges)

    assert torch.allclose(V.transpose(1, 2) @ V, torch.eye(3, dtype=DTYPE)[None])
    assert torch.allclose(T, T.transpose(1, 2))
    assert torch.allclose(T, V.transpose(1, 2) @ matrix @ V, atol=1e-12)


@pytest.mark.parametrize(
    "solver,symmetric",
    [
        (expm_krylov_arnoldi, False),
        (expm_krylov_lanczos, True),
    ],
)
def test_krylov_matrix_exponential_matches_dense_solution(solver, symmetric):
    vector = torch.tensor([[1.0, 0.5, -0.25]], dtype=DTYPE)
    diag = torch.tensor([[-1.0, -2.0, -3.0]], dtype=DTYPE)
    left = torch.tensor([[0.2, 0.3]], dtype=DTYPE)
    right = left if symmetric else torch.tensor([[0.4, 0.1]], dtype=DTYPE)
    V_buf, H_buf = _workspace(1, 3, 3)
    h = torch.tensor(0.05, dtype=DTYPE)

    actual = solver(vector, h, 3, diag, left, right, V_buf, H_buf)
    matrix = _dense_matrix(diag, left, right)
    expected = (torch.matrix_exp(h * matrix) @ vector.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "solver,symmetric",
    [
        (phi1_krylov_arnoldi, False),
        (phi1_krylov_lanczos, True),
    ],
)
def test_krylov_phi1_matches_dense_solution(solver, symmetric):
    vector = torch.tensor([[1.0, 0.5, -0.25]], dtype=DTYPE)
    diag = torch.tensor([[-1.0, -2.0, -3.0]], dtype=DTYPE)
    left = torch.tensor([[0.2, 0.3]], dtype=DTYPE)
    right = left if symmetric else torch.tensor([[0.4, 0.1]], dtype=DTYPE)
    V_buf, H_buf = _workspace(1, 3, 3)
    eye = torch.eye(3, dtype=DTYPE)
    h = torch.tensor(0.05, dtype=DTYPE)

    actual = solver(vector, h, 3, diag, left, right, V_buf, H_buf, eye)
    scaled = h * _dense_matrix(diag, left, right)
    rhs = (torch.matrix_exp(scaled) - eye) @ vector.unsqueeze(-1)
    expected = torch.linalg.solve(scaled, rhs).squeeze(-1)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("solver", [phi1_krylov_arnoldi_g, phi1_krylov_lanczos_g])
def test_guarded_phi1_returns_identity_action_for_zero_operator(solver):
    vector = torch.tensor([[1.0, -2.0, 0.5]], dtype=DTYPE)
    zeros_diag = torch.zeros((1, 3), dtype=DTYPE)
    zeros_edge = torch.zeros((1, 2), dtype=DTYPE)
    V_buf, H_buf = _workspace(1, 3, 3)

    actual = solver(
        vector,
        torch.tensor(0.1, dtype=DTYPE),
        3,
        zeros_diag,
        zeros_edge,
        zeros_edge,
        V_buf,
        H_buf,
        torch.eye(3, dtype=DTYPE),
    )

    assert torch.allclose(actual, vector)


@pytest.mark.parametrize("solver", [expm_krylov_arnoldi, expm_krylov_lanczos])
def test_krylov_exponential_handles_zero_input_without_nan(solver):
    vector = torch.zeros((1, 3), dtype=DTYPE)
    diag = torch.tensor([[-1.0, -2.0, -3.0]], dtype=DTYPE)
    edges = torch.tensor([[0.2, 0.3]], dtype=DTYPE)
    V_buf, H_buf = _workspace(1, 3, 3)

    actual = solver(
        vector,
        torch.tensor(0.1, dtype=DTYPE),
        3,
        diag,
        edges,
        edges,
        V_buf,
        H_buf,
    )

    assert torch.equal(actual, vector)
