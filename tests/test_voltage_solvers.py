import torch
import pytest

from axonml.models.integrators.triton import (
    dhs_solve_cuda,
    thomas_solve_cuda_t,
    thomas_solve_cuda_bt,
    thomas_solve_cuda_bt_n
)
from axonml.models.integrators.tridiag import (
    pcr_solve_t
)
CUDA_AVAILABLE = torch.cuda.is_available()


# ==============================================================================
# Pytest Test Functions for CUDA
# ==============================================================================

# Mark the entire file to be skipped if CUDA is not available or the function isn't found.
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA not available."
)


def test_dhs_cuda_forward_pass():
    """
    Tests the forward pass of the CUDA DHS solver.
    This is identical to the CPU test, just with tensors moved to the 'cuda' device.
    """
    B, K = 1, 2
    device = 'cuda:0' # Explicitly use a CUDA device

    # Inputs
    d_mem = torch.tensor([[1.0, 2.0]], device=device)
    a_geom = torch.tensor([[0.0, 0.5]], device=device)
    b = torch.tensor([[5.0, 3.0]], device=device)
    
    # Structure
    parent_idx = torch.tensor([-1, 0], device=device, dtype=torch.int64)
    order = torch.tensor([1, 0], device=device, dtype=torch.int64)
    layer_ptr = torch.tensor([0, 1, 2], device=device, dtype=torch.int64)
    
    # Solve using the CUDA op
    x_computed = dhs_solve_cuda(d_mem, a_geom, b, parent_idx, order, layer_ptr)

    # Manual solve for verification (can be done on CPU or GPU)
    A = torch.tensor([[1.5, -0.5], [-0.5, 2.5]], device=device)
    x_expected = torch.linalg.solve(A, b.T).T

    # Assert that the computed solution is close to the expected one
    # Moving to CPU for printing is good practice
    print(f"CUDA Computed solution: {x_computed.cpu()}")
    print(f"CUDA Expected solution: {x_expected.cpu()}")
    assert torch.allclose(x_computed, x_expected, atol=1e-6)


def test_dhs_cuda_backward_pass_gradcheck():
    """
    Uses torch.autograd.gradcheck to verify the CUDA backward implementation.
    """
    B, K = 2, 4
    device = 'cuda:0'
    dtype = torch.float64
    
    # Structure
    parent_idx = torch.tensor([-1, 0, 0, 2], device=device, dtype=torch.int64)
    order = torch.tensor([1, 3, 2, 0], device=device, dtype=torch.int64)
    layer_ptr = torch.tensor([0, 2, 3, 4], device=device, dtype=torch.int64)
    
    # Create random but well-conditioned inputs on the CUDA device
    d_mem = torch.rand(B, K, device=device, dtype=dtype) + 0.1
    a_geom = torch.rand(B, K, device=device, dtype=dtype) + 0.1
    b = torch.randn(B, K, device=device, dtype=dtype)
    a_geom[:, 0] = 0.0 # Clean up unused parameter
    
    d_mem.requires_grad_(True)
    a_geom.requires_grad_(True)
    b.requires_grad_(True)

    # --- The gradcheck call ---
    # `gradcheck` works seamlessly with CUDA tensors. It will automatically
    # handle moving data for the numerical differentiation if needed.
    
    # Define a function that calls the CUDA solver
    def solve_fn_cuda(d_mem_in, a_geom_in, b_in):
        # The `threads` argument has a default, so we don't need to pass it
        return dhs_solve_cuda(
            d_mem_in, a_geom_in, b_in, parent_idx, order, layer_ptr
        )
    
    inputs_to_check = (d_mem, a_geom, b)
    
    # `gradcheck` is computationally intensive. It might be a bit slower on GPU
    # for small problems but will work correctly.
    is_correct = torch.autograd.gradcheck(solve_fn_cuda, inputs_to_check, eps=1e-6, atol=1e-5)
    
    assert is_correct, "Gradient check failed for CUDA DHS solver!"


def test_thomas_cuda_t_forward_pass():
    """
    Tests the forward pass of the CUDA Thomas solver with a simple, known case.
    """
    B, K = 1, 3
    device = 'cuda:0'

    # Inputs for a 3x3 system
    b = torch.tensor([[4.0, 5.0, 6.0]], device=device) # main diagonal
    a = torch.tensor([[1.0, 2.0]], device=device)      # sub-diagonal
    c = torch.tensor([[3.0, 4.0]], device=device)      # super-diagonal
    d = torch.tensor([[1.0, 2.0, 3.0]], device=device) # RHS
    
    # Solve using the CUDA op
    x_computed = thomas_solve_cuda_t(a, b, c, d)

    # --- Manual solve for verification ---
    # Construct the full matrix A
    # A = [[4, 3, 0], [1, 5, 4], [0, 2, 6]]
    A = torch.zeros(K, K, device=device)
    A.diagonal(0).copy_(b[0])
    A.diagonal(-1).copy_(a[0])
    A.diagonal(1).copy_(c[0])
    
    x_expected = torch.linalg.solve(A, d.T).T

    print(f"CUDA Thomas Computed solution: {x_computed.cpu()}")
    print(f"CUDA Thomas Expected solution: {x_expected.cpu()}")
    
    assert torch.allclose(x_computed, x_expected, atol=1e-6)


def test_thomas_cuda_t_backward_pass_gradcheck():
    """
    Uses torch.autograd.gradcheck to verify the CUDA Thomas backward implementation.
    """
    B, K = 2, 5
    device = 'cuda:0'
    dtype = torch.float64
    
    # Create a random, diagonally dominant matrix to ensure stability
    b = (torch.rand(B, K, device=device, dtype=dtype) + 1.0) * 2
    a = torch.rand(B, K - 1, device=device, dtype=dtype)
    c = torch.rand(B, K - 1, device=device, dtype=dtype)
    d = torch.randn(B, K, device=device, dtype=dtype)
    
    # Set requires_grad=True for all inputs
    a.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)
    d.requires_grad_(True)

    # The function to be checked by gradcheck
    # It must be a plain function that calls the autograd-wrapped function.
    def solve_fn(a_in, b_in, c_in, d_in):
        return thomas_solve_cuda_t(a_in, b_in, c_in, d_in)
    
    inputs_to_check = (a, b, c, d)
    
    # Run the gradcheck
    is_correct = torch.autograd.gradcheck(solve_fn, inputs_to_check, eps=1e-6, atol=1e-5)
    
    assert is_correct, "Gradient check failed for CUDA Thomas solver!"


@pytest.mark.parametrize("K", [3, 7, 16, 33]) # Test odd, prime, power-of-2, and power-of-2 + 1 sizes
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pcr_forward_pass(K, device):
    """
    Tests the forward pass of the PCR solver against torch.linalg.solve.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available for testing")

    B = 4 # Use a batch size greater than 1
    
    # Create a random, diagonally dominant matrix to ensure stability
    b = (torch.rand(B, K, device=device) + 1.0) * 2
    a = torch.rand(B, K - 1, device=device)
    c = torch.rand(B, K - 1, device=device)
    d = torch.randn(B, K, device=device)

    # 1. Get the solution from the PCR implementation
    x_computed = pcr_solve_t(a, b, c, d)

    # 2. Get the expected solution using a trusted solver
    x_expected_list = []
    for i in range(B): # torch.linalg.solve doesn't support batching for different matrices
        A = torch.zeros(K, K, device=device)
        A.diagonal(0).copy_(b[i])
        A.diagonal(-1).copy_(a[i])
        A.diagonal(1).copy_(c[i])
        x_expected_list.append(torch.linalg.solve(A, d[i].unsqueeze(1)).squeeze(1))
    
    x_expected = torch.stack(x_expected_list)

    # 3. Compare the results
    # PCR can have slightly different floating point characteristics, so a
    # relative tolerance (rtol) is also good to have.
    assert torch.allclose(x_computed, x_expected, atol=1e-5, rtol=1e-4)


# block tridiagonal


def construct_block_tridiagonal_matrix(lower, main, upper):
    """
    Constructs a full dense matrix from block tridiagonal components.
    
    Args:
        lower (torch.Tensor): (B, K-1, 3) - diagonal elements of lower blocks
        main (torch.Tensor):  (B, K, 3, 3) - main diagonal blocks
        upper (torch.Tensor): (B, K-1, 3) - diagonal elements of upper blocks
    
    Returns:
        torch.Tensor: (B, K*3, K*3) - the full dense matrix
    """
    B, K, D, _ = main.shape
    N = K * D  # Total size of the dense matrix
    
    A_list = []
    for i in range(B):
        A = torch.zeros(N, N, device=main.device, dtype=main.dtype)
        
        # Fill in the main diagonal blocks
        for k in range(K):
            start = k * D
            end = start + D
            A[start:end, start:end] = main[i, k]
            
        # Fill in the lower and upper diagonal blocks
        for k in range(K - 1):
            start = k * D
            end = start + D
            # Lower block
            A[end:end+D, start:end] = torch.diag(lower[i, k])
            # Upper block
            A[start:end, end:end+D] = torch.diag(upper[i, k])
            
        A_list.append(A)
        
    return torch.stack(A_list)


def test_block_thomas_cuda_forward_pass():
    """
    Tests the forward pass of the CUDA block Thomas solver.
    """
    B, K, D = 1, 3, 3
    N = K * D
    device = 'cuda:0'
    
    # --- Create Inputs ---
    # Create well-conditioned main diagonal blocks
    main = torch.randn(B, K, D, D, device=device) * 0.1
    # FIX 1: Use torch.diagonal for a safe and correct way to modify the diagonal
    main_diag_view = torch.diagonal(main, dim1=-2, dim2=-1)
    main_diag_view += torch.eye(D, device=device) * 3

    lower = torch.randn(B, K - 1, D, device=device)
    upper = torch.randn(B, K - 1, D, device=device)
    rhs = torch.randn(B, K, D, device=device)
    
    # 1. Solve using the Triton kernel
    x_computed = thomas_solve_cuda_bt(lower, main, upper, rhs)
    
    # 2. Solve using a trusted dense solver
    A_dense = construct_block_tridiagonal_matrix(lower, main, upper)
    rhs_flat = rhs.reshape(B, N)
    x_expected_flat = torch.linalg.solve(A_dense, rhs_flat.unsqueeze(-1)).squeeze(-1)
    x_expected = x_expected_flat.reshape(B, K, D)
    
    print(f"CUDA Block Thomas Computed solution norm: {x_computed.norm().item()}")
    print(f"CUDA Block Thomas Expected solution norm: {x_expected.norm().item()}")
    
    assert torch.allclose(x_computed, x_expected, atol=1e-4, rtol=1e-3)


def test_block_thomas_cuda_backward_pass_gradcheck():
    B, K, D = 1, 4, 3
    device = 'cuda:0'
    dtype = torch.float64
    
    main = torch.randn(B, K, D, D, device=device, dtype=dtype) * 0.1
    main += torch.eye(D, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0) * 5
    
    lower = torch.randn(B, K - 1, D, device=device, dtype=dtype)
    upper = torch.randn(B, K - 1, D, device=device, dtype=dtype)
    rhs = torch.randn(B, K, D, device=device, dtype=dtype)
    
    lower.requires_grad_(True)
    main.requires_grad_(True)
    upper.requires_grad_(True)
    rhs.requires_grad_(True)

    def solve_fn(l, m, u, r):
        return thomas_solve_cuda_bt(l, m, u, r)
    
    inputs_to_check = (lower, main, upper, rhs)
    is_correct = torch.autograd.gradcheck(solve_fn, inputs_to_check, eps=1e-6, atol=1e-5, rtol=1e-4)
    
    assert is_correct, "Gradient check failed for CUDA Block Thomas solver!"
