import triton
import triton.language as tl
import torch

@triton.jit
def _single_dhs_kernel(
        D_ptr, A_ptr, B_ptr, V_ptr,
        P_ptr, ORDER_ptr, LAYER_PTR_ptr,
        K: tl.constexpr, L: tl.constexpr, K_THREADS: tl.constexpr):
    
    b = tl.program_id(0)
    lane = tl.arange(0, K_THREADS)

    # Pointers for the current batch item
    D = D_ptr + b * K
    A = A_ptr + b * K
    B = B_ptr + b * K
    V = V_ptr + b * K

    # --- 1. FUSED INITIALIZATION & FORWARD ELIMINATION ---
    # This single loop correctly builds the diagonal `D` just-in-time.
    for l in range(0, L):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        m = lane < (e - s)
        
        idx = tl.load(ORDER_ptr + s + lane, mask=m, other=0)
        parent = tl.load(P_ptr + idx, mask=m, other=-1)
        
        # Load the axial conductance g_i for the current compartment `idx`.
        # This value is used for both initialization and elimination.
        g_i = tl.load(A + idx, mask=m)
        
        # --- "Just-in-Time" Diagonal Finalization ---
        # All children of `idx` have already been processed and added their g_child to D[idx].
        # Now, we add g_i to finalize D[idx] and to contribute to D[parent].
        # Atomics are still required because multiple children of the same parent
        # might be processed in parallel by different threads.
        
        # Add g_i to our own diagonal
        tl.atomic_add(D + idx, g_i, mask=m)
        
        valid_parent_mask = m & (parent >= 0)
        # Add g_i to our parent's diagonal
        tl.atomic_add(D + parent, g_i, mask=valid_parent_mask)

        # --- Elimination Step ---
        # At this point, D[idx] is fully computed and can be safely read.
        d_i = tl.load(D + idx, mask=m)
        b_i = tl.load(B + idx, mask=m)

        # The elimination factor calculation is now safe.
        fac = -g_i / d_i

        # Update parent's D and B values using the computed factor.
        tl.atomic_add(D + parent, fac * g_i, mask=valid_parent_mask)
        tl.atomic_add(B + parent, -fac * b_i, mask=valid_parent_mask)

    # A barrier is still needed here to ensure the forward pass is
    # fully complete across the entire tree before back-substitution begins.
    tl.debug_barrier()

    # --- 2. BACKWARD SUBSTITUTION ---
    for l in range(L - 1, -1, -1):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        m = lane < (e - s)
        
        idx = tl.load(ORDER_ptr + s + lane, mask=m, other=0)
        parent = tl.load(P_ptr + idx, mask=m, other=-1)

        g_i = tl.load(A + idx, mask=m)
        d_i = tl.load(D + idx, mask=m)
        b_i = tl.load(B + idx, mask=m)

        valid_p = parent >= 0
        v_parent = tl.load(V + tl.where(valid_p, parent, 0), mask=m & valid_p, other=0.0)
        
        v_i = (b_i + g_i * v_parent) / d_i
        tl.store(V + idx, v_i, mask=m)


# --- KERNEL 1: INITIALIZATION ---
@triton.jit
def dhs_init_kernel(D_ptr, A_ptr, P_ptr, ORDER_ptr, LAYER_PTR_ptr,
                    B: tl.constexpr, K: tl.constexpr, L: tl.constexpr, K_THREADS: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    D = D_ptr + pid_b * K
    A = A_ptr + pid_b * K
    
    # Each thread block processes a unique range of compartments
    base_idx = pid_k * K_THREADS
    
    for l in range(0, L):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        
        # This block's threads work on a subset of the layer
        offsets = base_idx + tl.arange(0, K_THREADS)
        layer_mask = (offsets >= s) & (offsets < e)

        idx = tl.load(ORDER_ptr + offsets, mask=layer_mask, other=0)
        parent = tl.load(P_ptr + idx, mask=layer_mask, other=-1)
        
        g_i = tl.load(A + idx, mask=layer_mask, other=0.)
        
        tl.atomic_add(D + idx, g_i, mask=layer_mask)
        
        valid_parent_mask = layer_mask & (parent >= 0)
        tl.atomic_add(D + parent, g_i, mask=valid_parent_mask)


# --- KERNEL 2: FORWARD ELIMINATION ---
@triton.jit
def dhs_fwd_kernel(D_ptr, B_ptr, A_ptr, P_ptr, ORDER_ptr, LAYER_PTR_ptr,
                   B: tl.constexpr, K: tl.constexpr, L: tl.constexpr, K_THREADS: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    D = D_ptr + pid_b * K
    B = B_ptr + pid_b * K
    A = A_ptr + pid_b * K
    
    base_idx = pid_k * K_THREADS

    for l in range(0, L):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        
        offsets = base_idx + tl.arange(0, K_THREADS)
        layer_mask = (offsets >= s) & (offsets < e)

        idx = tl.load(ORDER_ptr + offsets, mask=layer_mask, other=0)
        parent = tl.load(P_ptr + idx, mask=layer_mask, other=-1)
        
        g_i = tl.load(A + idx, mask=layer_mask)
        d_i = tl.load(D + idx, mask=layer_mask)
        b_i = tl.load(B + idx, mask=layer_mask)

        fac = -g_i / d_i
        
        update_mask = layer_mask & (parent >= 0)
        tl.atomic_add(D + parent, fac * g_i, mask=update_mask)
        tl.atomic_add(B + parent, -fac * b_i, mask=update_mask)


# --- KERNEL 3: BACKWARD SUBSTITUTION ---
@triton.jit
def dhs_bwd_sub_kernel(V_ptr, D_ptr, B_ptr, A_ptr, P_ptr, ORDER_ptr, LAYER_PTR_ptr,
                       B: tl.constexpr, K: tl.constexpr, L: tl.constexpr, K_THREADS: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    V = V_ptr + pid_b * K
    D = D_ptr + pid_b * K
    B = B_ptr + pid_b * K
    A = A_ptr + pid_b * K
    
    base_idx = pid_k * K_THREADS
    
    for l in range(L - 1, -1, -1):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        
        offsets = base_idx + tl.arange(0, K_THREADS)
        layer_mask = (offsets >= s) & (offsets < e)

        idx = tl.load(ORDER_ptr + offsets, mask=layer_mask, other=0)
        parent = tl.load(P_ptr + idx, mask=layer_mask, other=-1)

        g_i = tl.load(A + idx, mask=layer_mask)
        b_i = tl.load(B + idx, mask=layer_mask)
        d_i = tl.load(D + idx, mask=layer_mask)

        valid_p = parent >= 0
        parent_mask = layer_mask & valid_p
        
        v_parent = tl.load(V + tl.where(valid_p, parent, 0), mask=parent_mask, other=0.0)
        
        v_i = (b_i + g_i * v_parent) / d_i
        tl.store(V + idx, v_i, mask=layer_mask)


class DHSSolveStable(torch.autograd.Function):
    """ Differentiable and numerically stable DHS solver. """

    @staticmethod
    def forward(ctx, d_mem, a_geom, b, parent_idx, order, layer_ptr, threads):
        """
        d_mem: (B,K) membrane-only diagonal components
        a_geom: (B,K) positive axial conductances g_i
        b: (B,K) right-hand side
        """
        B, K = d_mem.shape
        L = layer_ptr.numel() - 1
        d_full = d_mem.clone()
        x = torch.empty_like(b)

        # The kernel modifies d_full and b in-place, so we pass clones
        _single_dhs_kernel[(B,)](
            d_full, a_geom, b.clone(), x,
            parent_idx, order, layer_ptr,
            K=K, L=L, K_THREADS=threads,
            num_warps=1, num_stages=4
        )
        
        ctx.save_for_backward(d_mem, a_geom, x, parent_idx, order, layer_ptr)
        ctx.threads = threads
        return x

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: ∂L/∂x, where L is the loss and x is the output of forward.
        """
        d_mem, a_geom, x, parent_idx, order, layer_ptr = ctx.saved_tensors
        threads = ctx.threads
        B, K = d_mem.shape
        L = layer_ptr.numel() - 1

        # --- 1. Adjoint Solve: Aᵀg = grad_out ---
        # Since our implicit matrix A is symmetric, Aᵀ=A. We can reuse the
        # forward kernel to solve A*g = grad_out.
        d_full_adj = d_mem.clone()
        g = torch.empty_like(grad_out) # g is the adjoint vector

        _single_dhs_kernel[(B,)](
            d_full_adj, a_geom, grad_out.clone(), g,
            parent_idx, order, layer_ptr,
            K=K, L=L, K_THREADS=threads,
            num_warps=1, num_stages=4
        )
        
        # --- 2. Compute Parameter Gradients ---

        # Gradient w.r.t RHS `b` is simply the adjoint `g`
        grad_b = g

        # Gradient w.r.t membrane diagonal `d_mem`
        # d_mem only affects the main diagonal of A, so the gradient is -(g * x)
        grad_d_mem = -(g * x)

        # Gradient w.r.t axial conductance `a_geom` (g_i)
        # This is the most important change. The gradient is the negative
        # product of the voltage difference and the adjoint difference.
        
        # Gather parent values for x and g
        # We need to clamp parent indices to 0 for the root to avoid out-of-bounds
        parent_idx_clamped = parent_idx.clamp_min(0).to(torch.int64)
        x_parent = x.gather(1, parent_idx_clamped.expand_as(x))
        g_parent = g.gather(1, parent_idx_clamped.expand_as(g))

        # Calculate differences across each compartment's axial resistance
        delta_x = x - x_parent
        delta_g = g - g_parent

        grad_a_geom = -(delta_g * delta_x)

        # The root compartment has no parent, so its gradient must be zero.
        # Its parent index is -1.
        is_root = (parent_idx == -1).view(1, -1)
        grad_a_geom = grad_a_geom.masked_fill(is_root, 0.0)

        return grad_d_mem, grad_a_geom, grad_b, None, None, None, None


def dhs_solve(d_mem, a_geom, b, parent_idx, order, layer_ptr, threads=32):
    """
    Differentiable and numerically stable DHS solver for tree structures.

    Args:
        d_mem (torch.Tensor): (B,K) Membrane-only components of the main diagonal.
        a_geom (torch.Tensor): (B,K) Positive axial conductances (g_i) to parent.
        b (torch.Tensor): (B,K) Right-hand side of the system.
        parent_idx (torch.Tensor): (K,) Parent indices.
        order (torch.Tensor): (K,) DHS elimination order.
        layer_ptr (torch.Tensor): (L+1,) Pointers to DHS layers.
        threads (int): Threads per block (warp size).

    Returns:
        torch.Tensor: The solution vector x of shape (B,K).
    """
    return DHSSolveStable.apply(d_mem, a_geom, b, parent_idx, order, layer_ptr, threads)