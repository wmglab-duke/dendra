import triton
import triton.language as tl
import torch

# ---------------------------------------------------------------------
# N×N generic inverse using Gaussian-Jordan elimination
# This function works on pointers to N*N blocks in memory.
# It's designed to be called from within another kernel.
# ---------------------------------------------------------------------
@triton.jit
def invNxN(mat_ptr, inv_ptr, N: tl.constexpr):
    """
    Inverts an N x N matrix.
    `mat_ptr`: Pointer to the input matrix (N*N elements).
    `inv_ptr`: Pointer to store the output inverse (N*N elements).
    `N`: The size of the matrix, must be a compile-time constant.
    """
    # Create local copies of the matrix and an identity matrix
    # Using python loops with tl.static_range to unroll for the compiler
    A = tl.zeros((N, N), dtype=tl.float32)
    I = tl.zeros((N, N), dtype=tl.float32)

    for i in tl.static_range(N):
        for j in tl.static_range(N):
            A[i, j] = tl.load(mat_ptr + i * N + j)
            if i == j:
                I[i, j] = 1.0

    # Perform Gaussian-Jordan elimination
    for i in tl.static_range(N):
        # Find pivot
        pivot = A[i, i]
        inv_pivot = 1.0 / pivot

        # Normalize the pivot row
        for j in tl.static_range(N):
            A[i, j] = A[i, j] * inv_pivot
            I[i, j] = I[i, j] * inv_pivot

        # Eliminate other rows
        for row in tl.static_range(N):
            if row != i:
                factor = A[row, i]
                for col in tl.static_range(N):
                    A[row, col] = A[row, col] - factor * A[i, col]
                    I[row, col] = I[row, col] - factor * I[i, col]

    # Store the result (the transformed identity matrix)
    for i in tl.static_range(N):
        for j in tl.static_range(N):
            tl.store(inv_ptr + i * N + j, I[i, j])

# ---------------------------------------------------------------------
# one thread / one fibre, N and K arbitrary
# ---------------------------------------------------------------------
@triton.jit
def thomas_btn_kernel(
        L_ptr, M_ptr, U_ptr, D_ptr, X_ptr,
        K: tl.constexpr, N: tl.constexpr):

    bid = tl.program_id(0)          # batch id

    # Define block sizes for convenience
    N_SQ = N * N

    # base addresses for this system
    L = L_ptr + bid * (K-1) * N
    M = M_ptr + bid *  K    * N_SQ
    U = U_ptr + bid * (K-1) * N
    D = D_ptr + bid *  K    * N
    X = X_ptr + bid *  K    * N
    
    # Allocate space for a single N*N inverse matrix in SRAM.
    # This is much faster than re-calculating it multiple times.
    inv_sram_ptr = tl.make_tensor_ptr(
        base=tl.zeros((N_SQ,), dtype=tl.float32), 
        shape=(N,N), 
        stride=(N,1), 
        order=(1,0)
    )

    # ---------------- forward elimination ---------------------------
    # invert first main block
    invNxN(M, inv_sram_ptr, N)

    for k in range(1, K):
        # W_k = L_{k-1} * M_{k-1}^{-1}
        # Since L is diagonal, this is L_i * row_i(M_inv)
        # A_k' = A_k - W_k * U_{k-1}
        # Since U is diagonal, this is A_k - (L_i*M_inv_ij*U_j)
        
        # We perform this update in-place on A_k
        base_M_k = M + k * N_SQ
        
        for i in tl.static_range(N):
            l_i = tl.load(L + (k-1)*N + i)
            for j in tl.static_range(N):
                m_inv_ij = tl.load(inv_sram_ptr + i*N + j)
                u_j = tl.load(U + (k-1)*N + j)
                
                # Update M_k[i, j]
                m_k_ij = tl.load(base_M_k + i*N + j)
                m_k_ij -= l_i * m_inv_ij * u_j
                tl.store(base_M_k + i*N + j, m_k_ij)

        # d_k' = d_k - L_{k-1} * M_{k-1}^{-1} * d_{k-1}
        # We perform this update in-place on d_k
        base_D_k = D + k*N
        base_D_prev = D + (k-1)*N

        for i in tl.static_range(N):
            dot_product = 0.0
            l_i = tl.load(L + (k-1)*N + i)
            for j in tl.static_range(N):
                m_inv_ij = tl.load(inv_sram_ptr + i*N + j)
                d_prev_j = tl.load(base_D_prev + j)
                dot_product += l_i * m_inv_ij * d_prev_j
            
            d_k_i = tl.load(base_D_k + i)
            d_k_i -= dot_product
            tl.store(base_D_k + i, d_k_i)
        
        # Invert the new A_k for the next iteration
        invNxN(base_M_k, inv_sram_ptr, N)

    # ---------------- backward substitution -------------------------
    # last block: x_{K-1} = M_{K-1}^{-1} * d_{K-1}
    base_X_last = X + (K-1)*N
    base_D_last = D + (K-1)*N

    for i in tl.static_range(N):
        x_i = 0.0
        for j in tl.static_range(N):
            m_inv_ij = tl.load(inv_sram_ptr + i*N + j)
            d_j = tl.load(base_D_last + j)
            x_i += m_inv_ij * d_j
        tl.store(base_X_last + i, x_i)

    for k in range(K-2, -1, -1):
        # First, we need the original M_k to get its inverse.
        # The forward pass modified M, so we must re-invert it.
        base_M_k = M + k*N_SQ
        invNxN(base_M_k, inv_sram_ptr, N)
        
        # d_k' = d_k - U_k * x_{k+1}
        # Note: d_k is the one modified from the forward pass.
        base_D_k = D + k*N
        base_X_next = X + (k+1)*N

        for i in tl.static_range(N):
            u_i = tl.load(U + k*N + i)
            x_next_i = tl.load(base_X_next + i)
            
            d_k_i = tl.load(base_D_k + i)
            d_k_i -= u_i * x_next_i
            tl.store(base_D_k + i, d_k_i)
        
        # x_k = M_k^{-1} * d_k'
        base_X_k = X + k*N
        for i in tl.static_range(N):
            x_i = 0.0
            for j in tl.static_range(N):
                m_inv_ij = tl.load(inv_sram_ptr + i*N + j)
                d_k_j = tl.load(base_D_k + j)
                x_i += m_inv_ij * d_k_j
            tl.store(base_X_k + i, x_i)


# ---------------------------------------------------------------------
# python launcher for generalized kernel
# ---------------------------------------------------------------------
def _thomas_triton_n(lower, main, upper, rhs):
    """
    lower : (B, K-1, N)   - diagonal elements
    main  : (B, K,   N,N)
    upper : (B, K-1, N)   - diagonal elements
    rhs   : (B, K,   N)
    """
    B, K, N = rhs.shape
    assert lower.shape == (B, K-1, N)
    assert main.shape == (B, K, N, N)
    assert upper.shape == (B, K-1, N)

    out = torch.empty_like(rhs)
    
    # The kernel modifies `main` and `rhs` in-place, so we must pass copies.
    main_clone = main.clone()
    rhs_clone = rhs.clone()

    # The kernel needs contiguous memory blocks.
    grid = (B,)
    thomas_btn_kernel[grid](
        lower      .contiguous(), 
        main_clone .reshape(B, -1),
        upper      .contiguous(), 
        rhs_clone  .reshape(B, -1),
        out        .reshape(B, -1),
        K=K,
        N=N,
        num_warps=2, # May need adjustment based on N
        num_stages=4
    )

    return out


# ---------------------------------------------------------------------
# autograd wrapper for generalized kernel
# ---------------------------------------------------------------------
class ThomasSolveN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lower, main, upper, rhs):
        """
        lower : (B, K-1, N)
        main  : (B, K,   N,N)
        upper : (B, K-1, N)
        rhs   : (B, K,   N)
        returns x : (B, K, N)
        """
        x = _thomas_triton_n(lower, main, upper, rhs)
        ctx.save_for_backward(lower, main, upper, x)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out = dL/dx  (same shape as x)
        returns gradients w.r.t. (lower, main, upper, rhs)
        
        The logic here is independent of N, so it's a direct generalization.
        """
        lower, main, upper, x = ctx.saved_tensors
        B, K, N = x.shape

        # ------ 1. adjoint solve:  Aᵀ g = grad_out -------------------
        # The transpose of a block tridiagonal matrix swaps the lower and
        # upper diagonals and transposes the main diagonal blocks.
        main_T = main.transpose(-1, -2).contiguous()   # (B,K,N,N)
        
        # The `_thomas_triton_n` solver takes diagonal elements for lower/upper.
        # The original L/U blocks were diagonal, so L^T=L and U^T=U.
        g = _thomas_triton_n(
                upper,          # acts as new lower
                main_T,
                lower,          # acts as new upper
                grad_out
            )

        # ------ 2. compute parameter gradients -----------------------
        # grad_rhs
        grad_rhs = g

        # grad_main blocks (B,K,N,N)
        # ∂L/∂M_k = -g_k * x_kᵀ
        grad_main = -(g.unsqueeze(-1) * x.unsqueeze(-2))

        # grad_upper & grad_lower bands
        # The original `lower` and `upper` were diagonal blocks. We assume
        # we are only differentiating w.r.t. these diagonal elements.
        # ∂L/∂U_k(i,i) = -g_k(i) * x_{k+1}(i)
        grad_upper = -(g[:, :-1] * x[:, 1:])   # (B,K-1,N)
        # ∂L/∂L_k(i,i) = -g_{k+1}(i) * x_k(i)
        grad_lower = -(g[:, 1:] * x[:, :-1])   # (B,K-1,N)

        return grad_lower, grad_main, grad_upper, grad_rhs


# convenience function -------------------------------------------------
def thomas_triton_bt_n(lower, main, upper, rhs):
    """
    Differentiable block tridiagonal solver for systems where the
    off-diagonal blocks (lower, upper) are diagonal matrices.
    
    Args:
        lower (torch.Tensor): (B, K-1, N), diagonal elements of lower blocks.
        main (torch.Tensor): (B, K, N, N), main diagonal blocks.
        upper (torch.Tensor): (B, K-1, N), diagonal elements of upper blocks.
        rhs (torch.Tensor): (B, K, N), right-hand side.
        
    Returns:
        torch.Tensor: The solution `x` of shape (B, K, N).
    """
    return ThomasSolveN.apply(lower, main, upper, rhs)

# ---------------------------------------------------------------------
# Example Usage & Verification
# ---------------------------------------------------------------------
if __name__ == '__main__':
    B, K, N = 2, 8, 4  # Batch size, sequence length, block size

    # --- Setup a test problem ---
    # Ensure main diagonal blocks are diagonally dominant for stability
    main_diag = torch.rand(B, K, N, device='cuda') * 5 + 2
    main = torch.randn(B, K, N, N, device='cuda') * 0.1 + torch.diag_embed(main_diag)
    
    # Off-diagonal blocks are diagonal matrices
    lower = torch.randn(B, K - 1, N, device='cuda') * 0.5
    upper = torch.randn(B, K - 1, N, device='cuda') * 0.5
    
    rhs = torch.randn(B, K, N, device='cuda')
    
    # --- Solve with Triton ---
    print(f"Solving system with B={B}, K={K}, N={N}")
    x_triton = thomas_triton_bt_n(lower, main, upper, rhs)

    # --- Solve with PyTorch for verification (slower) ---
    full_A = torch.zeros(B, K * N, K * N, device='cuda')
    for b in range(B):
        for k in range(K):
            # Main diagonal
            full_A[b, k*N:(k+1)*N, k*N:(k+1)*N] = main[b, k]
            # Lower diagonal
            if k > 0:
                full_A[b, k*N:(k+1)*N, (k-1)*N:k*N] = torch.diag(lower[b, k-1])
            # Upper diagonal
            if k < K - 1:
                full_A[b, (k)*N:(k+1)*N, (k+1)*N:(k+2)*N] = torch.diag(upper[b, k])
    
    x_torch = torch.linalg.solve(full_A, rhs.reshape(B, -1, 1)).reshape(B, K, N)
    
    # --- Compare results ---
    print("Forward pass comparison (max abs diff):", torch.max(torch.abs(x_triton - x_torch)).item())
    assert torch.allclose(x_triton, x_torch, atol=1e-5)
    print("Forward pass successful!")
    
    # --- Test gradients ---
    main.requires_grad_()
    lower.requires_grad_()
    upper.requires_grad_()
    rhs.requires_grad_()
    
    x_autograd = thomas_triton_bt_n(lower, main, upper, rhs)
    # Define a dummy loss
    loss = (x_autograd ** 2).sum()
    loss.backward()

    # Get gradients
    grad_main_triton = main.grad.clone()
    grad_lower_triton = lower.grad.clone()
    
    # Check one gradient with torch.autograd
    main.grad, lower.grad = None, None # Reset grads
    x_torch_grad_test = torch.linalg.solve(full_A.detach(), rhs.reshape(B, -1, 1)).reshape(B, K, N)
    
    # The gradient for the full matrix A can be calculated as:
    # A^T * dL/dx = -grad_out => dL/dx = - (A^T)^-1 * grad_out
    # dL/dA_ij = (dL/dx * dx/dA_ij).sum() = (- (A^T)^-1 * grad_out * (-A^-1 * d(Ax)/dA_ij * x)).sum()
    # It simplifies to dL/dA = - (A^-T * grad_out) * x^T = -g * x^T
    grad_out_test = 2 * x_torch_grad_test
    g_test = torch.linalg.solve(full_A.detach().transpose(-1, -2), grad_out_test.reshape(B, -1, 1)).reshape(B, K, N)
    
    grad_main_torch = -(g_test.unsqueeze(-1) * x_torch_grad_test.unsqueeze(-2))
    
    print("Backward pass comparison (max abs diff):", torch.max(torch.abs(grad_main_triton - grad_main_torch)).item())
    assert torch.allclose(grad_main_triton, grad_main_torch, atol=1e-4)
    print("Backward pass successful!")