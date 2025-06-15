import triton
import triton.language as tl
import torch


@triton.jit
def dhs_fwd_bwd(
        D_ptr, A_ptr, B_ptr, V_ptr,
        P_ptr, ORDER_ptr, LAYER_PTR_ptr,
        K: tl.constexpr,
        L: tl.constexpr,
        K_THREADS: tl.constexpr = 32):

    # -------- one block / one neuron ------------------------
    b    = tl.program_id(0)               # batch index
    lane = tl.arange(0, K_THREADS)        # 0..31

    D = D_ptr + b * K
    A = A_ptr + b * K
    B = B_ptr + b * K
    V = V_ptr + b * K

    # ---------------- forward elimination -------------------
    for l in range(0, L):
        s   = tl.load(LAYER_PTR_ptr + l)
        e   = tl.load(LAYER_PTR_ptr + l + 1)
        w   = e - s                       # ≤ 32
        m   = lane < w                    # mask

        idx    = tl.load(ORDER_ptr + s + lane, mask=m, other=0)
        parent = tl.load(P_ptr + idx,      mask=m, other=-1)

        a_i = tl.load(A + idx, mask=m)
        d_i = tl.load(D + idx, mask=m)
        b_i = tl.load(B + idx, mask=m)

        fac   = a_i / d_i
        valid = parent >= 0

        # atomic updates to parent row
        tl.atomic_add(D + parent, -(fac * a_i) * valid, mask=m & valid)
        tl.atomic_add(B + parent, -(fac * b_i) * valid, mask=m & valid)

        # no extra barrier needed: single warp executes in lock‑step

    # ---------------- back substitution --------------------
    for l in range(L - 1, -1, -1):
        s = tl.load(LAYER_PTR_ptr + l)
        e = tl.load(LAYER_PTR_ptr + l + 1)
        w = e - s
        m = lane < w

        idx    = tl.load(ORDER_ptr + s + lane, mask=m, other=0)
        parent = tl.load(P_ptr + idx,         mask=m, other=-1)

        a_i = tl.load(A + idx, mask=m)
        d_i = tl.load(D + idx, mask=m)
        b_i = tl.load(B + idx, mask=m)

        valid_p   = parent >= 0
        parent_cl = tl.where(valid_p, parent, 0)          # safe addr
        v_parent  = tl.where(
            valid_p,
            tl.load(V + parent_cl, mask=m & valid_p),
            b_i / d_i)

        tl.store(V + idx, (b_i - a_i * v_parent) / d_i, mask=m)


def _dhs_triton(d, a, b, parent, order, layer_ptr, threads=32):
    """
    d, a, b : (B, K) float32  (d & b will be mutated in place)
    """
    B, K = d.shape
    L = layer_ptr.numel() - 1
    V = torch.empty_like(b)

    # exactly one warp (32 threads) per block → num_warps = 1
    dhs_fwd_bwd[(B,)](
        d, a, b, V,
        parent, order, layer_ptr,
        K=K, L=L, K_THREADS=threads,
        num_warps=1,             
        num_stages=4
    )
    return V


# autograd wrapper
class DHSSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, a, b,
                parent_idx, order, layer_ptr,
                threads):
        """
        d : (B,K) diag
        a : (B,K) (-g child→parent, root arbitrary)
        b : (B,K) rhs
        parent_idx, order, layer_ptr : buffers (no grad)
        threads : int, number of threads per block (default 32) (no grad)
        """
        v = _dhs_triton(d.clone(), a, b.clone(),
                        parent_idx, order, layer_ptr, threads)
        ctx.save_for_backward(d, a, v, parent_idx, order, layer_ptr, threads)
        return v

    @staticmethod
    def backward(ctx, g_out):
        d, a, v, parent_idx, order, layer_ptr, threads = ctx.saved_tensors
        #    adjoint solve:  Aᵀ g = g_out
        #    swap child ↔ parent by re‑using the same kernel
        g = _dhs_triton(
            d.clone(), a, g_out.clone(),
            parent_idx, order, layer_ptr, threads
        ) # A is symmetric!

        # grads -------------------------------------------------------
        grad_b = g                                     # (B,K)
        grad_d = -(g * v)                              # (B,K)

        # edge‑wise gradient for a : use parent_idx mask
        child  = torch.arange(a.size(1), device=a.device)
        parent = parent_idx.expand_as(a)               # broadcast to (B,K)

        # Δv  and  Δg  on every edge (child row carries a_e)
        dv  = v - v.gather(1, parent.clamp_min(0))
        dg  = g - g.gather(1, parent.clamp_min(0))
        grad_a = -(dg * dv)                            # (B,K)

        # zero‑out the root (parent == -1) which has no real edge
        grad_a = grad_a.masked_fill(parent < 0, 0.)

        return grad_d, grad_a, grad_b, None, None, None, None


def dhs_solve(
    d: torch.Tensor,  # (B,K) diag
    a: torch.Tensor,  # (B,K) (-g child→parent, root arbitrary)
    b: torch.Tensor,  # (B,K) rhs
    parent_idx: torch.Tensor,  # (K,) parent indices
    order: torch.Tensor,        # (K,) elimination order
    layer_ptr: torch.Tensor,    # (L+1,) layer pointers
    threads: int = 32           # number of threads per block
) -> torch.Tensor:
    """
    Solve the DHS system Ax = b using the triton kernel.
    Returns x of shape (B,K).
    """
    return DHSSolve.apply(d, a, b, parent_idx, order, layer_ptr, threads)