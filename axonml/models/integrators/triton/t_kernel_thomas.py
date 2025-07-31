import torch
import triton
import triton.language as tl


@triton.jit
def _thomas_solve_kernel(
    a_ptr,
    b_ptr,
    cp_ptr,
    x_ptr,
    a_stride_b,
    a_stride_k,
    b_stride_b,
    b_stride_k,
    cp_stride_b,
    cp_stride_k,
    x_stride_b,
    x_stride_k,
    K: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    a_b_ptr = a_ptr + batch_idx * a_stride_b
    b_b_ptr = b_ptr + batch_idx * b_stride_b
    cp_b_ptr = cp_ptr + batch_idx * cp_stride_b
    x_b_ptr = x_ptr + batch_idx * x_stride_b

    # --- Forward Elimination Sweep ---
    b0 = tl.load(b_b_ptr)
    inv_b0 = 1.0 / b0

    cp0 = tl.load(cp_b_ptr)
    cp0 *= inv_b0
    tl.store(cp_b_ptr, cp0)

    x0 = tl.load(x_b_ptr)
    x0 *= inv_b0
    tl.store(x_b_ptr, x0)

    for i in range(1, K):
        offset_i = i * b_stride_k
        offset_im1 = (i - 1) * b_stride_k
        offset_a_im1 = (i - 1) * a_stride_k

        a_im1 = tl.load(a_b_ptr + offset_a_im1)
        b_i = tl.load(b_b_ptr + offset_i)
        cp_im1 = tl.load(cp_b_ptr + offset_im1)
        x_im1 = tl.load(x_b_ptr + offset_im1)

        denom = b_i - a_im1 * cp_im1
        inv_denom = 1.0 / denom

        cp_i = tl.load(cp_b_ptr + offset_i)
        x_i = tl.load(x_b_ptr + offset_i)

        cp_i *= inv_denom
        tl.store(cp_b_ptr + offset_i, cp_i)

        x_i = (x_i - a_im1 * x_im1) * inv_denom
        tl.store(x_b_ptr + offset_i, x_i)

    # --- Backward Substitution Sweep ---
    for i in range(K - 2, -1, -1):
        offset_i = i * x_stride_k
        offset_ip1 = (i + 1) * x_stride_k

        cp_i = tl.load(cp_b_ptr + offset_i)
        x_ip1 = tl.load(x_b_ptr + offset_ip1)
        x_i = tl.load(x_b_ptr + offset_i)

        x_i -= cp_i * x_ip1
        tl.store(x_b_ptr + offset_i, x_i)


class ThomasSolve(torch.autograd.Function):
    """
    Custom autograd Function for the batched tridiagonal solver.
    """

    @staticmethod
    def _solve(a, b, c, d):
        """
        Internal solver function that launches the Triton kernel.
        This can be called by both forward and backward passes.
        """
        # --- Input Validation ---
        B, K = b.shape
        # Basic checks, more can be added if needed
        assert a.shape == (B, K - 1), "Shape mismatch for 'a'"
        assert c.shape == (B, K - 1), "Shape mismatch for 'c'"
        assert d.shape == (B, K), "Shape mismatch for 'd'"

        # --- Ensure Contiguous Inputs ---
        a = a.contiguous()
        b = b.contiguous()
        c = c.contiguous()
        d = d.contiguous()

        # --- Prepare Output and Temporary Tensors ---
        x = d.clone()
        cp = torch.empty_like(b)
        cp[:, :-1] = c
        cp[:, -1] = 0.0

        # --- Kernel Launch ---
        grid = (B,)
        _thomas_solve_kernel[grid](
            a,
            b,
            cp,
            x,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            cp.stride(0),
            cp.stride(1),
            x.stride(0),
            x.stride(1),
            K=K,
        )
        return x

    @staticmethod
    def forward(ctx, a, b, c, d):
        """
        Forward pass: solves Ax = d.
        """
        # Solve the system
        x = ThomasSolve._solve(a, b, c, d)

        # Save tensors needed for backward pass
        # We need a, b, c for the transpose solve, and the solution x
        ctx.save_for_backward(a, b, c, x)

        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        Backward pass: computes gradients for a, b, c, d.
        """
        a, b, c, x = ctx.saved_tensors

        # Initialize gradients to None
        grad_a = grad_b = grad_c = grad_d = None

        # Check which inputs require gradients
        needs_grad = ctx.needs_input_grad

        # If no inputs need gradients, we can exit early
        if not any(needs_grad):
            return None, None, None, None

        # The core of the backward pass is solving A^T * y = grad_x
        # The matrix A^T has diagonals (c, b, a)
        y = ThomasSolve._solve(a=c, b=b, c=a, d=grad_x)

        # Compute gradients based on the formulas derived
        if needs_grad[3]:  # Gradient for d
            grad_d = y

        if needs_grad[0] or needs_grad[1] or needs_grad[2]:  # Gradients for A
            if needs_grad[1]:  # Gradient for b
                grad_b = -y * x

            if needs_grad[2]:  # Gradient for c
                grad_c = -y[:, :-1] * x[:, 1:]

            if needs_grad[0]:  # Gradient for a
                grad_a = -y[:, 1:] * x[:, :-1]

        # Return gradients in the same order as forward inputs
        return grad_a, grad_b, grad_c, grad_d


def thomas_solve_cuda_t(a, b, c, d):
    """
    Differentiable batched tridiagonal solver using a custom Triton kernel.

    Args:
        a (torch.Tensor): The sub-diagonal of the matrix. Shape: (B, K-1).
        b (torch.Tensor): The main diagonal of the matrix. Shape: (B, K).
        c (torch.Tensor): The super-diagonal of the matrix. Shape: (B, K-1).
        d (torch.Tensor): The right-hand side of the equation. Shape: (B, K).

    Returns:
        torch.Tensor: The solution `x` to the system `Ax=d`. Shape: (B, K).
    """
    # Type and device checks
    common_dtype = b.dtype
    common_device = b.device
    for t, name in zip([a, b, c, d], ["a", "b", "c", "d"]):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Input '{name}' must be a torch.Tensor.")
        if t.device != common_device or t.dtype != common_dtype:
            raise ValueError("All input tensors must have the same dtype and device.")

    return ThomasSolve.apply(a, b, c, d)
