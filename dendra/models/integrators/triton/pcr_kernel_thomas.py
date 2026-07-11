import math

import torch
import triton
import triton.language as tl

from ._contracts import validate_tridiagonal


@triton.jit
def _pcr_solve_kernel(
    a_ptr,  # (B, K)  subdiag with a[:, 0] = 0
    b_ptr,  # (B, K)  main diag (working copy)
    c_ptr,  # (B, K)  superdiag with c[:, -1] = 0
    d_ptr,  # (B, K)  RHS (working buffer)
    x_ptr,  # (B, K)  solution
    a_stride_b,
    a_stride_k,
    b_stride_b,
    b_stride_k,
    c_stride_b,
    c_stride_k,
    d_stride_b,
    d_stride_k,
    x_stride_b,
    x_stride_k,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    N_ITERS: tl.constexpr,
):  # pragma: no cover
    batch_idx = tl.program_id(0)

    a_b_ptr = a_ptr + batch_idx * a_stride_b
    b_b_ptr = b_ptr + batch_idx * b_stride_b
    c_b_ptr = c_ptr + batch_idx * c_stride_b
    d_b_ptr = d_ptr + batch_idx * d_stride_b
    x_b_ptr = x_ptr + batch_idx * x_stride_b

    # tile indices along tridiagonal dimension
    i = tl.arange(0, BLOCK)
    mask_i = i < K

    # load initial coefficients into registers
    a = tl.load(a_b_ptr + i * a_stride_k, mask=mask_i, other=0.0)
    b = tl.load(b_b_ptr + i * b_stride_k, mask=mask_i, other=1.0)
    c = tl.load(c_b_ptr + i * c_stride_k, mask=mask_i, other=0.0)
    d = tl.load(d_b_ptr + i * d_stride_k, mask=mask_i, other=0.0)

    # Parallel Cyclic Reduction
    for m in range(N_ITERS):
        # make current stage visible in memory for neighbor loads
        tl.store(a_b_ptr + i * a_stride_k, a, mask=mask_i)
        tl.store(b_b_ptr + i * b_stride_k, b, mask=mask_i)
        tl.store(c_b_ptr + i * c_stride_k, c, mask=mask_i)
        tl.store(d_b_ptr + i * d_stride_k, d, mask=mask_i)

        stride = 1 << m

        left_idx = i - stride
        right_idx = i + stride

        has_left = (i >= stride) & mask_i
        has_right = (i + stride < K) & mask_i

        # neighbor loads; "other" is chosen to keep divisions safe where masked
        a_l = tl.load(a_b_ptr + left_idx * a_stride_k, mask=has_left, other=0.0)
        b_l = tl.load(b_b_ptr + left_idx * b_stride_k, mask=has_left, other=1.0)
        c_l = tl.load(c_b_ptr + left_idx * c_stride_k, mask=has_left, other=0.0)
        d_l = tl.load(d_b_ptr + left_idx * d_stride_k, mask=has_left, other=0.0)

        a_r = tl.load(a_b_ptr + right_idx * a_stride_k, mask=has_right, other=0.0)
        b_r = tl.load(b_b_ptr + right_idx * b_stride_k, mask=has_right, other=1.0)
        c_r = tl.load(c_b_ptr + right_idx * c_stride_k, mask=has_right, other=0.0)
        d_r = tl.load(d_b_ptr + right_idx * d_stride_k, mask=has_right, other=0.0)

        alpha = tl.where(has_left, -a / b_l, 0.0)
        gamma = tl.where(has_right, -c / b_r, 0.0)

        a_new = alpha * a_l
        c_new = gamma * c_r
        b_new = b + alpha * c_l + gamma * a_r
        d_new = d + alpha * d_l + gamma * d_r

        a = a_new
        b = b_new
        c = c_new
        d = d_new

    # after N_ITERS, each equation is decoupled: b_i x_i = d_i
    x = d / b
    tl.store(x_b_ptr + i * x_stride_k, x, mask=mask_i)


class PCRSolve(torch.autograd.Function):
    @staticmethod
    def _solve(a, b, c, d):
        """
        a : (B, K-1)  subdiag
        b : (B, K)    diag
        c : (B, K-1)  superdiag
        d : (B, K)    RHS
        """
        B, K = b.shape
        assert a.shape == (B, K - 1)
        assert c.shape == (B, K - 1)
        assert d.shape == (B, K)

        a = a.contiguous()
        b = b.contiguous()
        c = c.contiguous()
        d = d.contiguous()

        # build full-length a_full, c_full and working RHS / diag
        a_full = torch.empty_like(b)
        a_full[:, 0] = 0.0
        a_full[:, 1:] = a

        c_full = torch.empty_like(b)
        c_full[:, :-1] = c
        c_full[:, -1] = 0.0

        d_work = d.clone()
        b_work = b.clone()  # don't mutate original b

        x = torch.empty_like(d)

        # PCR parameters
        BLOCK = 1 << (K - 1).bit_length()  # next power-of-two ≥ K
        N_ITERS = math.ceil(math.log2(K))

        grid = (B,)

        _pcr_solve_kernel[grid](
            a_full,
            b_work,
            c_full,
            d_work,
            x,
            a_full.stride(0),
            a_full.stride(1),
            b_work.stride(0),
            b_work.stride(1),
            c_full.stride(0),
            c_full.stride(1),
            d_work.stride(0),
            d_work.stride(1),
            x.stride(0),
            x.stride(1),
            K=K,
            BLOCK=BLOCK,
            N_ITERS=N_ITERS,
            num_warps=1,
            num_stages=2,
        )
        return x

    @staticmethod
    def forward(ctx, a, b, c, d):
        x = PCRSolve._solve(a, b, c, d)
        ctx.save_for_backward(a, b, c, x)
        return x

    @staticmethod
    def backward(ctx, grad_x):  # pragma: no cover
        a, b, c, x = ctx.saved_tensors
        grad_a = grad_b = grad_c = grad_d = None
        needs_grad = ctx.needs_input_grad
        if not any(needs_grad):
            return None, None, None, None

        # solve A^T y = grad_x, where A has (a,b,c) so A^T has (c,b,a)
        y = PCRSolve._solve(a=c, b=b, c=a, d=grad_x)

        if needs_grad[3]:
            grad_d = y
        if needs_grad[1]:
            grad_b = -y * x
        if needs_grad[2]:
            grad_c = -y[:, :-1] * x[:, 1:]
        if needs_grad[0]:
            grad_a = -y[:, 1:] * x[:, :-1]

        return grad_a, grad_b, grad_c, grad_d


def pcr_solve_cuda_t(a, b, c, d):
    validate_tridiagonal(a, b, c, d)
    return PCRSolve.apply(a, b, c, d)
