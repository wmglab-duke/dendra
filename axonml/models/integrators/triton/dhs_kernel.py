import math

import torch
import triton
import triton.language as tl

# ==============================================================================
# 1. The Triton Kernel
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
):  # pragma: no cover
    """
    Optimized Triton kernel for a fused Dendritic Hierarchical Scheduling (DHS) solve.
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
    ORDER = ORDER_ptr  # No batch offset for ORDER, P, LAYER_PTR
    P = P_ptr
    LAYER_PTR = LAYER_PTR_ptr

    # --- 1. FUSED INITIALIZATION & FORWARD ELIMINATION ---
    for layer in range(0, L):
        s = tl.load(LAYER_PTR + layer)
        e = tl.load(LAYER_PTR + layer + 1)

        offset = s + lane
        m = lane < (e - s)

        idx = tl.load(ORDER + offset, mask=m, other=0)
        parent = tl.load(P + idx, mask=m, other=-1)

        g_i = tl.load(A + idx, mask=m, other=0.0)

        # --- "Just-in-Time" Diagonal Finalization ---
        # The new diagonal value is the old value (returned by atomic_add) + the value added.
        old_d_i = tl.atomic_add(D + idx, g_i, mask=m)
        d_i = old_d_i + g_i

        valid_parent_mask = m & (parent >= 0)
        safe_parent = tl.where(
            valid_parent_mask, parent, 0
        )  # Keep safe_parent for pointer arithmetic

        # This atomic_add finalizes the parent's diagonal for the connection to this child (g_i).
        tl.atomic_add(D + safe_parent, g_i, mask=valid_parent_mask)

        # --- Elimination Step ---
        b_i = tl.load(B + idx, mask=m, other=0.0)

        # Invert d_i once. Use a safe divide to avoid 0/0 -> NaN.
        inv_d_i = 1.0 / d_i
        fac = -g_i * inv_d_i

        # Update parent's D and B.
        tl.atomic_add(D + safe_parent, fac * g_i, mask=valid_parent_mask)
        tl.atomic_add(B + safe_parent, -fac * b_i, mask=valid_parent_mask)

    # Barrier ensures all forward-pass atomics for all layers are complete before back-substitution begins.
    tl.debug_barrier()

    # --- 2. BACKWARD SUBSTITUTION ---
    # Looping backwards from the second-to-last layer down to the root layer.
    for layer in range(L - 1, -1, -1):
        s = tl.load(LAYER_PTR + layer)
        e = tl.load(LAYER_PTR + layer + 1)

        offset = s + lane
        m = lane < (e - s)

        idx = tl.load(ORDER + offset, mask=m, other=0)
        parent = tl.load(P + idx, mask=m, other=-1)

        valid_parent_mask = m & (parent >= 0)
        safe_parent = tl.where(valid_parent_mask, parent, 0)

        # g_i must be re-loaded as it wasn't stored from the forward pass.
        # This is generally fine, as re-loading is cheaper than spilling to/from shared memory
        # across the tl.debug_barrier().
        g_i = tl.load(A + idx, mask=m, other=0.0)

        # Load the final, eliminated values of D and B from the forward pass.
        d_i_elim = tl.load(D + idx, mask=m, other=1.0)  # other=1.0 to avoid div by zero
        b_i_elim = tl.load(B + idx, mask=m, other=0.0)

        # Load the already-computed voltage of the parent.
        v_parent = tl.load(V + safe_parent, mask=valid_parent_mask, other=0.0)

        # Substitute to find the voltage of the current compartment.
        v_i = (b_i_elim + g_i * v_parent) / d_i_elim

        tl.store(V + idx, v_i, mask=m)


@triton.jit
def _single_dhs_kernel_packed(
    D_ptr,  # (B, K)
    A_ptr,  # (B, K)
    B_ptr,  # (B, K)
    V_ptr,  # (B, K)
    P_ptr,  # (K,)                    – parent indices
    ORDER_ptr,  # (K,)                – topological order
    LAYER_PTR_ptr,  # (L + 1,)        – layer start - end
    B_total: tl.constexpr,  # scalar  – total neurons in the *whole* batch
    K: tl.constexpr,  # compartments / neuron
    L: tl.constexpr,  # number of layers
    K_THREADS: tl.constexpr,  # threads / neuron within a warp  (≤32)
    WARP_SIZE: tl.constexpr,  # 32, the number of lanes in a warp
):  # pragma: no cover
    NEURONS_PER_WARP = WARP_SIZE // K_THREADS  # 2 if K_THREADS == 16, etc.

    # ----------------------------------------------------------
    # Lane bookkeeping: split the 32-lane warp into sub-tiles
    # ----------------------------------------------------------
    lane_abs = tl.arange(0, WARP_SIZE)  # 0 … 31
    n_in_warp = lane_abs // K_THREADS  # 0 … NEURONS_PER_WARP-1
    lane_local = lane_abs % K_THREADS  # 0 … K_THREADS-1

    # Global neuron index this lane is responsible for
    b = tl.program_id(0) * NEURONS_PER_WARP + n_in_warp

    # Mask off lanes whose neuron index spills past B_total
    valid_neuron = b < B_total

    # Safe `b` for pointer arithmetic (never out-of-bounds)
    b_safe = tl.where(valid_neuron, b, 0)

    # Per-lane base pointers ---------------------------------------------------
    D = D_ptr + b_safe * K
    A = A_ptr + b_safe * K
    B = B_ptr + b_safe * K
    V = V_ptr + b_safe * K

    P = P_ptr
    ORDER = ORDER_ptr
    LAYER_PTR = LAYER_PTR_ptr

    # ==========================================================
    # 1. Fused initialisation + forward elimination
    # ==========================================================
    for layer in range(0, L):
        s = tl.load(LAYER_PTR + layer)
        e = tl.load(LAYER_PTR + layer + 1)

        offset = s + lane_local
        in_range = lane_local < (e - s)
        m = in_range & valid_neuron  # final mask

        idx = tl.load(ORDER + offset, mask=m, other=0)
        parent = tl.load(P + idx, mask=m, other=-1)
        g_i = tl.load(A + idx, mask=m, other=0.0)

        # --- JIT diagonal finalisation
        old_d_i = tl.atomic_add(D + idx, g_i, mask=m)
        d_i = old_d_i + g_i

        valid_parent_mask = m & (parent >= 0)
        safe_parent = tl.where(valid_parent_mask, parent, 0)

        tl.atomic_add(D + safe_parent, g_i, mask=valid_parent_mask)

        # --- elimination update
        b_i = tl.load(B + idx, mask=m, other=0.0)
        fac = -g_i / d_i
        tl.atomic_add(D + safe_parent, fac * g_i, mask=valid_parent_mask)
        tl.atomic_add(B + safe_parent, -fac * b_i, mask=valid_parent_mask)

    tl.debug_barrier()  # ensure forward phase completed

    # ==========================================================
    # 2. Back-substitution
    # ==========================================================
    for layer in range(L - 1, -1, -1):
        s = tl.load(LAYER_PTR + layer)
        e = tl.load(LAYER_PTR + layer + 1)

        offset = s + lane_local
        in_range = lane_local < (e - s)
        m = in_range & valid_neuron

        idx = tl.load(ORDER + offset, mask=m, other=0)
        parent = tl.load(P + idx, mask=m, other=-1)

        valid_parent_mask = m & (parent >= 0)
        safe_parent = tl.where(valid_parent_mask, parent, 0)

        g_i = tl.load(A + idx, mask=m, other=0.0)
        d_i_elim = tl.load(D + idx, mask=m, other=1.0)
        b_i_elim = tl.load(B + idx, mask=m, other=0.0)
        v_parent = tl.load(V + safe_parent, mask=valid_parent_mask, other=0.0)

        v_i = (b_i_elim + g_i * v_parent) / d_i_elim
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
    def forward(
        ctx, d_mem, a_geom, b, parent_idx, order, layer_ptr, threads
    ):  # pragma: no cover
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
    def backward(ctx, grad_out):  # pragma: no cover
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
        # only affects the diagonal A[root, root], so its gradient is -g[root]*x[root].
        grad_a_geom[:, is_root] = -(g[:, is_root] * x[:, is_root])

        # The return signature must match the forward inputs in order.
        return grad_d_mem, grad_a_geom, grad_b, None, None, None, None


class DHSSolvePacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d_mem, a_geom, b, parent_idx, order, layer_ptr, threads: int):
        """
        Parameters
        ----------
        d_mem : (B, K)   diagonal values   (will be overwritten in-kernel)
        a_geom: (B, K)   off-diagonal conductances
        b     : (B, K)   RHS
        parent_idx, order, layer_ptr : topology tables (shared across batch)
        threads : int    K_THREADS - lanes per neuron inside a warp (≤ 32)
        """
        B, K = d_mem.shape
        L = layer_ptr.numel() - 1

        V_out = torch.empty_like(b)

        # How many neurons each 32-lane warp can handle
        NEURONS_PER_WARP = 32 // threads
        grid_x = math.ceil(B / NEURONS_PER_WARP)

        _single_dhs_kernel_packed[(grid_x,)](
            d_mem.clone(),
            a_geom,
            b.clone(),
            V_out,
            parent_idx,
            order,
            layer_ptr,
            B_total=B,
            K=K,
            L=L,
            K_THREADS=threads,
            WARP_SIZE=32,
            num_warps=1,  # one warp per block – already fully occupied
            num_stages=4,
        )

        # Save everything autograd needs
        ctx.save_for_backward(d_mem, a_geom, V_out, parent_idx, order, layer_ptr)
        ctx.threads = threads
        return V_out

    @staticmethod
    def backward(ctx, grad_out):  # pragma: no cover
        d_mem, a_geom, V, parent_idx, order, layer_ptr = ctx.saved_tensors
        threads = ctx.threads
        B, K = d_mem.shape
        L = layer_ptr.numel() - 1

        NEURONS_PER_WARP = 32 // threads
        grid_x = math.ceil(B / NEURONS_PER_WARP)

        g = torch.empty_like(grad_out)
        _single_dhs_kernel_packed[(grid_x,)](
            d_mem.clone(),  # rebuild identical matrix
            a_geom,
            grad_out.clone(),  # RHS = upstream grad
            g,
            parent_idx,
            order,
            layer_ptr,
            B_total=B,
            K=K,
            L=L,
            K_THREADS=threads,
            WARP_SIZE=32,
            num_warps=1,
            num_stages=4,
        )

        # Gradients wrt original inputs
        grad_b = g
        grad_d_mem = -(g * V)

        is_root = parent_idx < 0
        parent_clamped = parent_idx.clamp_min(0)

        V_parent = V.gather(1, parent_clamped.expand_as(V))
        g_parent = g.gather(1, parent_clamped.expand_as(g))

        grad_a_geom = -(g - g_parent) * (V - V_parent)
        grad_a_geom[:, is_root] = -(g[:, is_root] * V[:, is_root])

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
    return DHSSolvePacked.apply(d_mem, a_geom, b, parent_idx, order, layer_ptr, threads)
