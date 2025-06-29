import triton
import triton.language as tl
import torch

# ---------------------------------------------------------------------
# N×N generic inverse using Gaussian-Jordan elimination
# This function works on pointers to N*N blocks in memory.
# It's designed to be called from within another kernel.
# ---------------------------------------------------------------------
@triton.jit
def invNxN_scalar(A_sram, N: tl.constexpr, N_padded: tl.constexpr):
    """
    Inverts an N x N matrix by operating on a Python list-of-lists of scalar
    Triton tensors. This is the most robust method.
    `A_sram`: An (N_padded, N_padded) tl.tensor holding the input matrix.
    """
    # 1. Load the N x N portion of A_sram into a list-of-lists of scalars.
    #    Also create the identity matrix in the same format.
    #    These are Python loops that unroll at compile time.
    A = []
    I = []
    for i in range(N):
        A_row = []
        I_row = []
        for j in range(N):
            A_row.append(A_sram[i, j])
            I_row.append(1.0 if i == j else 0.0)
        A.append(A_row)
        I.append(I_row)

    # 2. Perform Gaussian-Jordan elimination on the lists of scalars.
    #    This code is now guaranteed to work because it only uses
    #    operations on scalar tl.tensors.
    for i in range(N):
        pivot = A[i][i]
        inv_pivot = 1.0 / pivot
        for j in range(N):
            A[i][j] *= inv_pivot
            I[i][j] *= inv_pivot
        for row in range(N):
            if row != i:
                factor = A[row][i]
                for col in range(N):
                    A[row][col] -= factor * A[i][col]
                    I[row][col] -= factor * I[i][col]

    # 3. Convert the result list-of-lists back into a single tl.tensor for output.
    #    We create a new padded tensor and fill it using tl.where.
    inv_matrix = tl.zeros((N_padded, N_padded), dtype=tl.float32)
    for r in range(N):
        for c in range(N):
            # This is a functional-style assignment:
            # "new_matrix = where(condition, value_if_true, old_matrix)"
            is_target_cell = (tl.arange(0, N_padded)[:, None] == r) & (tl.arange(0, N_padded)[None, :] == c)
            inv_matrix = tl.where(is_target_cell, I[r][c], inv_matrix)
            
    return inv_matrix


@triton.jit
def load_block_masked(ptr, N: tl.constexpr, N_padded: tl.constexpr):
    """ Loads an N x N block into a padded N_padded x N_padded SRAM tensor. """
    offs_r = tl.arange(0, N_padded)
    offs_c = tl.arange(0, N_padded)
    block_ptr = ptr + (offs_r[:, None] * N + offs_c[None, :])
    mask = (offs_r[:, None] < N) & (offs_c[None, :] < N)
    return tl.load(block_ptr, mask=mask, other=0.0)


@triton.jit
def store_block_masked(ptr, block, N: tl.constexpr, N_padded: tl.constexpr):
    """ Stores the top-left N x N part of a padded SRAM tensor. """
    offs_r = tl.arange(0, N_padded)
    offs_c = tl.arange(0, N_padded)
    block_ptr = ptr + (offs_r[:, None] * N + offs_c[None, :])
    mask = (offs_r[:, None] < N) & (offs_c[None, :] < N)
    tl.store(block_ptr, block, mask=mask)

# =============================================================================
# MAIN KERNEL
# =============================================================================

@triton.jit
def thomas_btn_kernel(
        L_ptr, M_ptr, U_ptr, D_ptr, X_ptr,
        K: tl.constexpr, N: tl.constexpr, N_padded: tl.constexpr):
    
    bid = tl.program_id(0)
    N_SQ = N * N
    
    L = L_ptr + bid * (K-1) * N
    M = M_ptr + bid *  K    * N_SQ
    U = U_ptr + bid * (K-1) * N
    D = D_ptr + bid *  K    * N
    X = X_ptr + bid *  K    * N
    
    # --- Forward elimination ---
    M_k_sram = load_block_masked(M, N, N_padded)
    inv_sram = invNxN_scalar(M_k_sram, N, N_padded)

    for k in range(1, K):
        base_M_k = M + k * N_SQ
        M_k_sram_next = load_block_masked(base_M_k, N, N_padded)

        # A_k' = A_k - L_{k-1} * M_{k-1}^{-1} * U_{k-1}
        # This is now a list-of-lists update
        M_k_updated = []
        for i in range(N):
            row = []
            l_i = tl.load(L + (k-1)*N + i)
            for j in range(N):
                m_inv_ij = inv_sram[i, j]
                u_j = tl.load(U + (k-1)*N + j)
                val = M_k_sram_next[i, j] - (l_i * m_inv_ij * u_j)
                row.append(val)
            M_k_updated.append(row)
        
        # Convert back to tensor to store
        M_k_sram_next_updated = tl.zeros((N_padded, N_padded), dtype=tl.float32)
        for r in range(N):
            for c in range(N):
                is_target_cell = (tl.arange(0, N_padded)[:, None] == r) & (tl.arange(0, N_padded)[None, :] == c)
                M_k_sram_next_updated = tl.where(is_target_cell, M_k_updated[r][c], M_k_sram_next_updated)

        store_block_masked(base_M_k, M_k_sram_next_updated, N, N_padded)

        # d_k' update
        base_D_k = D + k*N
        base_D_prev = D + (k-1)*N
        for i in range(N):
            dot_product = 0.0
            l_i = tl.load(L + (k-1)*N + i)
            for j in range(N):
                m_inv_ij = inv_sram[i, j]
                d_prev_j = tl.load(base_D_prev + j)
                dot_product += l_i * m_inv_ij * d_prev_j
            d_k_i = tl.load(base_D_k + i)
            tl.store(base_D_k + i, d_k_i - dot_product)
        
        inv_sram = invNxN_scalar(M_k_sram_next_updated, N, N_padded)

    # --- Backward substitution ---
    base_D_last = D + (K-1)*N
    base_X_last = X + (K-1)*N
    for i in range(N):
        x_i = 0.0
        for j in range(N):
            d_j = tl.load(base_D_last + j)
            x_i += inv_sram[i, j] * d_j
        tl.store(base_X_last + i, x_i)

    for k in range(K-2, -1, -1):
        base_M_k = M + k*N_SQ
        M_k_sram = load_block_masked(base_M_k, N, N_padded)
        inv_sram = invNxN_scalar(M_k_sram, N, N_padded)
        
        base_D_k = D + k*N
        base_X_next = X + (k+1)*N
        d_k_updated = tl.zeros((N_padded,), dtype=tl.float32)

        for i in range(N):
            u_i = tl.load(U + k*N + i)
            x_next_i = tl.load(base_X_next + i)
            d_k_i = tl.load(base_D_k + i)
            d_k_updated[i] = d_k_updated[i] + (d_k_i - u_i * x_next_i)
        
        base_X_k = X + k*N
        for i in range(N):
            x_i = 0.0
            for j in range(N):
                d_k_j = d_k_updated[j]
                x_i += inv_sram[i, j] * d_k_j
            tl.store(base_X_k + i, x_i)


# ---------------------------------------------------------------------
# python launcher for generalized kernel
# ---------------------------------------------------------------------
def _thomas_triton_n(lower, main, upper, rhs):
    """
    lower : (B, K-1, N)
    main  : (B, K,   N,N)
    upper : (B, K-1, N)
    rhs   : (B, K,   N)
    """
    B, K, N = rhs.shape
    out = torch.empty_like(rhs)
    main_clone = main.clone()
    rhs_clone = rhs.clone()
    
    # --- Calculate padded dimension and pass to kernel ---
    N_padded = triton.next_power_of_2(N)

    grid = (B,)
    thomas_btn_kernel[grid](
        lower.contiguous(), 
        main_clone.reshape(B, -1),
        upper.contiguous(), 
        rhs_clone.reshape(B, -1),
        out.reshape(B, -1),
        K=K,
        N=N,
        N_padded=N_padded, # Pass the new constexpr
        num_warps=2,
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
def thomas_solve_cuda_bt_n(lower, main, upper, rhs):
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