import torch


def pcr_solve_t(
    a: torch.Tensor,  # (B, K-1) subdiag
    b: torch.Tensor,  # (B, K)   diag
    c: torch.Tensor,  # (B, K-1) superdiag
    d: torch.Tensor,  # (B, K)   RHS
    switch_to_thomas_at: int | None = None,  # e.g., 64 or 32
) -> torch.Tensor:
    """
    Parallel Cyclic Reduction (PCR) tridiagonal solver, batched over B.
    O(log K) stages; avoids torch.roll, uses slice updates; in-place where safe.
    """
    B, K = b.shape
    dev, dtype = b.device, b.dtype

    # Embed a,c to (B,K) with zeros at boundaries
    a_full = torch.zeros(B, K, device=dev, dtype=dtype)
    c_full = torch.zeros(B, K, device=dev, dtype=dtype)
    a_full[:, 1:] = a
    c_full[:, :-1] = c

    b_full = b.clone()
    d_full = d.clone()

    stride = 1
    while stride < K:
        # Optional hybrid switch: stop PCR when the effective segment size is small
        if switch_to_thomas_at is not None and stride >= switch_to_thomas_at:
            break

        # Interior indices where both neighbors exist
        iL = slice(stride, K - stride)  # current rows
        iLL = slice(0, K - 2 * stride)  # left neighbor rows aligned
        iRR = slice(2 * stride, K)  # right neighbor rows aligned

        # Precompute scalars on interior
        # alpha = a[i] / b[i - stride]; gamma = c[i] / b[i + stride]
        alpha = a_full[:, iL] / b_full[:, iLL]
        gamma = c_full[:, iL] / b_full[:, iRR]

        # Cache neighbor coeffs/RHS
        aL = a_full[:, iLL]
        cL = c_full[:, iLL]
        dL = d_full[:, iLL]

        aR = a_full[:, iRR]
        cR = c_full[:, iRR]
        dR = d_full[:, iRR]

        # Update central row (i): b,d,a,c — all in-place on the interior slice
        # b_i <- b_i - c_L*alpha - a_R*gamma
        b_full[:, iL].addcmul_(cL, -alpha).addcmul_(aR, -gamma)
        # d_i <- d_i - d_L*alpha - d_R*gamma
        d_full[:, iL].addcmul_(dL, -alpha).addcmul_(dR, -gamma)
        # a_i <- -a_L*alpha
        a_full[:, iL].copy_(-aL * alpha)
        # c_i <- -c_R*gamma
        c_full[:, iL].copy_(-cR * gamma)

        # Edges: rows [0:stride) and (K-stride:K) lose one neighbor; their alpha/gamma are zero
        # We only need to zero their a/c to keep the invariant tight.
        if stride > 0:
            a_full[:, :stride].zero_()
            c_full[:, -stride:].zero_()

        stride <<= 1

    # If we broke out early, finish each independent segment with Thomas
    if stride < K:
        # Segment length is at most 2*stride; solve each contiguous block independently.
        seg = 2 * stride
        # Iterate blocks [s : s+seg)
        for s in range(0, K, seg):
            e = min(s + seg, K)
            # Extract views
            bb = b_full[:, s:e]
            dd = d_full[:, s:e]
            aa = a_full[:, s:e]
            cc = c_full[:, s:e]
            # Run batched Thomas on this small band (in-place)
            _batched_thomas_inplace(aa, bb, cc, dd)  # defines x in dd via back-sub
        return d_full / b_full  # dd now contains x*diag; divide by diag

    # Pure PCR path: diagonal system
    return d_full / b_full


def _batched_thomas_inplace(a_full, b_full, c_full, d_full):
    """
    In-place Thomas on blocks: a_full has size (B, m) with a_full[:,0]==0,
    c_full[:,m-1]==0. Writes solution into d_full; keeps b_full as diag.
    """
    B, m = b_full.shape
    # forward sweep
    c_full[:, 0] = c_full[:, 0] / b_full[:, 0]
    d_full[:, 0] = d_full[:, 0] / b_full[:, 0]
    for i in range(1, m):
        denom = b_full[:, i] - a_full[:, i] * c_full[:, i - 1]
        if i < m - 1:
            c_full[:, i] = c_full[:, i] / denom
        d_full[:, i] = (d_full[:, i] - a_full[:, i] * d_full[:, i - 1]) / denom
    # back substitution
    for i in range(m - 2, -1, -1):
        d_full[:, i] = d_full[:, i] - c_full[:, i] * d_full[:, i + 1]
