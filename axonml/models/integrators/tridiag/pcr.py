import torch


def pcr_tridiag_solve(
    a: torch.Tensor,  # (B,K-1) sub-diag
    b: torch.Tensor,  # (B,K)   main diag
    c: torch.Tensor,  # (B,K-1) super-diag
    d: torch.Tensor,  # (B,K)   RHS
) -> torch.Tensor:
    """
    Parallel Cyclic Reduction (PCR) batched tridiagonal solver.
    Handles any length K (no power-of-two restriction) and runs in
    O(logK) sequential stages while using only tensor ops.

    Returns x of shape (B, K).
    """
    B, K = b.shape
    dev, dtype = b.device, b.dtype

    # ───── embed sub / super so that a[:,0] = c[:,-1] = 0 ─────
    a_full = torch.zeros(B, K, device=dev, dtype=dtype)
    c_full = torch.zeros_like(a_full)
    a_full[:, 1:] = a          # a_0 … a_{K-1}
    c_full[:, :-1] = c         # c_1 … c_{K-2}

    b_full, d_full = b.clone(), d.clone()

    idx = torch.arange(K, device=dev)

    stride = 1
    while stride < K:
        # roll(stride) gives the neighbour values (dummy for out-of-range rows)
        a_L, b_L, c_L, d_L = (
            torch.roll(a_full,  stride, dims=1),
            torch.roll(b_full,  stride, dims=1),
            torch.roll(c_full,  stride, dims=1),
            torch.roll(d_full,  stride, dims=1),
        )
        a_R, b_R, c_R, d_R = (
            torch.roll(a_full, -stride, dims=1),
            torch.roll(b_full, -stride, dims=1),
            torch.roll(c_full, -stride, dims=1),
            torch.roll(d_full, -stride, dims=1),
        )

        # rows that *really* have those neighbours
        has_L = idx >= stride
        has_R = idx <  K - stride

        # broadcast to (B,K)
        has_L = has_L.expand(B, -1)
        has_R = has_R.expand(B, -1)

        # coefficients a, g set to 0 where the neighbour is absent
        alpha = torch.where(has_L, a_full / b_L, torch.zeros_like(a_full))
        gamma = torch.where(has_R, c_full / b_R, torch.zeros_like(c_full))

        # core PCR updates (Stone & Schultheiss 1973, Alg. 3)
        b_full = b_full - c_L * alpha - a_R * gamma
        d_full = d_full - d_L * alpha - d_R * gamma
        a_full = -a_L * alpha
        c_full = -c_R * gamma

        stride <<= 1      # next distance (x2)

    # after log2(K) stages  a_full[:,1:], c_full[:,:-1] -> 0 => purely diagonal
    x = d_full / b_full
    return x