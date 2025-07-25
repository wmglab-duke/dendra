import triton
import triton.language as tl
import torch

# ==============================================================================
# 1. The Triton Kernel (Unchanged)
# ==============================================================================


@triton.jit
def _single_dhs_kernel(
    D_ptr,
    A_ptr,
    B_ptr,
    V_ptr,
    P_ptr,
    ORDER_ptr,
    LAYER_PTR_ptr,
    K: tl.constexpr,
    L: tl.constexpr,
    K_THREADS: tl.constexpr,
):
    """
    Triton kernel for a fused Dendritic Hierarchical Scheduling (DHS) solve.

    This kernel performs a "just-in-time" assembly of the system matrix `A`
    followed immediately by forward elimination and backward substitution. It is
    designed to solve the linear system A*x = b for many independent trees
    in a batch.

    - The diagonal `D_ptr` is initialized with the membrane-only conductances (`d_mem`).
    - The kernel adds the structural/axial conductances from `A_ptr` to finalize `D_ptr`.
    - It then performs elimination and substitution, modifying `D_ptr` and `B_ptr` in-place.
    """
    # Each program in the grid handles one neuron in the batch.
    b = tl.program_id(0)
    # Each thread in a block handles one element in a layer chunk.
    lane = tl.arange(0, K_THREADS)

    # Pointers to the data for the current neuron in the batch.
    D = D_ptr + b * K
    A = A_ptr + b * K
    B = B_ptr + b * K
    V = V_ptr + b * K

    # --- 1. FUSED INITIALIZATION & FORWARD ELIMINATION ---
    for l in range(0, L):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        m = lane < (e - s)

        idx = tl.load(ORDER_ptr + tl.where(m, s + lane, 0), mask=m, other=0)
        parent = tl.load(P_ptr + idx, mask=m, other=-1)

        g_i = tl.load(A + idx, mask=m)

        # --- "Just-in-Time" Diagonal Finalization ---
        # The D array starts as d_mem. We add the structural components from a_geom
        # to finalize the diagonal of the system matrix A.
        tl.atomic_add(D + idx, g_i, mask=m)

        valid_parent_mask = m & (parent >= 0)
        safe_parent = tl.where(valid_parent_mask, parent, 0)

        tl.atomic_add(D + safe_parent, g_i, mask=valid_parent_mask)

        # --- Elimination Step ---
        d_i = tl.load(D + idx, mask=m)
        b_i = tl.load(B + idx, mask=m)
        fac = -g_i / d_i
        tl.atomic_add(D + safe_parent, fac * g_i, mask=valid_parent_mask)
        tl.atomic_add(B + safe_parent, -fac * b_i, mask=valid_parent_mask)

    tl.debug_barrier()

    # --- 2. BACKWARD SUBSTITUTION ---
    for l in range(L - 1, -1, -1):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        m = lane < (e - s)

        idx = tl.load(ORDER_ptr + tl.where(m, s + lane, 0), mask=m, other=0)
        parent = tl.load(P_ptr + idx, mask=m, other=-1)
        valid_parent_mask = m & (parent >= 0)
        safe_parent = tl.where(valid_parent_mask, parent, 0)

        g_i = tl.load(A + idx, mask=m)
        d_i = tl.load(D + idx, mask=m)
        b_i = tl.load(B + idx, mask=m)

        v_parent = tl.load(V + safe_parent, mask=valid_parent_mask, other=0.0)

        v_i = (b_i + g_i * v_parent) / d_i
        tl.store(V + idx, v_i, mask=m)


# ==============================================================================
# 2. The Autograd Function (The PyTorch Bridge)
# ==============================================================================


class DHSSolveStable(torch.autograd.Function):
    """
    Differentiable wrapper for the DHS solver kernel.
    This version is consistent and passes numerical gradient checks.
    """

    @staticmethod
    def forward(ctx, d_mem, a_geom, b, parent_idx, order, layer_ptr, threads):
        B, K = d_mem.shape
        L = layer_ptr.numel() - 1
        x = torch.empty_like(b)

        # The kernel modifies its D and B inputs in-place. We must pass clones
        # to avoid side effects on the original tensors.
        _single_dhs_kernel[(B,)](
            d_mem.clone(),
            a_geom,
            b.clone(),
            x,
            parent_idx,
            order,
            layer_ptr,
            K=K,
            L=L,
            K_THREADS=threads,
            num_warps=1,
            num_stages=4,
        )

        # Save the original inputs for the backward pass.
        ctx.save_for_backward(d_mem, a_geom, x, parent_idx, order, layer_ptr)
        ctx.threads = threads
        return x

    @staticmethod
    def backward(ctx, grad_out):
        """
        Computes the vector-Jacobian product for the DHS solve.
        This implementation is the one that correctly passes gradcheck.
        """
        d_mem, a_geom, x, parent_idx, order, layer_ptr = ctx.saved_tensors
        threads = ctx.threads
        B, K = d_mem.shape
        L = layer_ptr.numel() - 1

        # --- 1. Adjoint Solve ---
        # The matrix for the adjoint system must be identical to the one in the
        # forward pass. We achieve this by calling the same kernel with the
        # same physical inputs (`d_mem`, `a_geom`).
        g = torch.empty_like(grad_out)
        _single_dhs_kernel[(B,)](
            d_mem.clone(),
            a_geom,
            grad_out.clone(),
            g,
            parent_idx,
            order,
            layer_ptr,
            K=K,
            L=L,
            K_THREADS=threads,
        )

        # --- 2. Compute Gradients for Original Inputs ---
        grad_b = g
        grad_d_mem = -(g * x)
        is_root = parent_idx < 0

        # --- Gradient of `a_geom` ---
        # First, compute the general gradient formula, which is correct for all non-root nodes.
        parent_idx_clamped = parent_idx.clamp_min(0).to(torch.int64)
        x_parent = x.gather(1, parent_idx_clamped.expand_as(x))
        g_parent = g.gather(1, parent_idx_clamped.expand_as(g))

        delta_x = x - x_parent
        delta_g = g - g_parent

        grad_a_geom = -(delta_g * delta_x)

        # Second, explicitly compute the correct gradient for the root node(s) and
        # overwrite the value calculated by the general formula. The parameter a_geom[root]
        # only affects the diagonal A[root,root], so its gradient is -g[root]*x[root].
        grad_a_geom[:, is_root] = -(g[:, is_root] * x[:, is_root])

        # The return signature must match the forward inputs in order.
        return grad_d_mem, grad_a_geom, grad_b, None, None, None, None


# ==============================================================================
# 3. Public-Facing Wrapper Function
# ==============================================================================


def dhs_solve_cuda(d_mem, a_geom, b, parent_idx, order, layer_ptr, threads=32):
    """
    Public-facing function for the differentiable DHS solver.

    This is the clean entry point that users and tests should call. It simply
    wraps the `autograd.Function` to handle the solver's execution and gradient
    computation.
    """
    return DHSSolveStable.apply(d_mem, a_geom, b, parent_idx, order, layer_ptr, threads)
